# %%
import torch
import numpy as np
import mne
import os
from omegaconf import OmegaConf
from hydra.utils import instantiate
from sklearn.preprocessing import LabelEncoder
from spdnets.models import PatternInterpretableModel
from library.utils.moabb import CachedParadigm
import spdnets.modules as modules
import spdnets.functionals as F

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from matplotlib.patches import Rectangle
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

# %%
dataset, class_idx = 'BNCI2015001', 0
# dataset, class_idx = 'Hinss2021', 1

preprocessing = 'bb4-36Hz'
model_name = 'TSMNet+SPDDSMBN'
evaluation = 'inter-subject'
subject = 12
session = -1

mdl_path = os.path.join('outputs',dataset,evaluation,preprocessing,'models',str(subject),str(session), model_name + '.pth')
mdl_metadata_path = os.path.join('outputs',dataset,evaluation,preprocessing,'metadata', 'meta-'+ model_name + '.pth')

# %%
# create new model with stored parameters

if not os.path.exists(mdl_path) or not os.path.exists(mdl_metadata_path):
    raise ValueError('no model found')

md = torch.load(mdl_metadata_path)

model_kwargs = md['model_kwargs']
device = torch.device('cpu')
model_kwargs['device'] = device
load_kwargs = dict(map_location=device)
state_dict = torch.load(mdl_path, **load_kwargs)

obj = md['model_class'](**model_kwargs)
device = obj.device_
obj.load_state_dict(state_dict)
obj.eval()

# %%
# load the dataset configuration
file_dsconf = os.path.join('conf', 'dataset', f'{dataset.lower()}.yaml')
conf_ds = OmegaConf.to_container(OmegaConf.load(file_dsconf))
# load the paradigm configuration
file_prepconf = os.path.join('conf', 'preprocessing', f'{preprocessing}.yaml')
conf_prep = OmegaConf.to_container(OmegaConf.load(file_prepconf))
# join the configurations (so that dependent keys can be interpolated)
cfg = OmegaConf.create(dict(dataset=conf_ds, preprocessing=conf_prep))
# instantiat the objects
obj_ds = instantiate(cfg.dataset.type,_convert_='partial')
obj_prep = instantiate(cfg.preprocessing,_convert_='partial')[preprocessing]

if isinstance(obj_prep, CachedParadigm):
    info = obj_prep.get_info(obj_ds)
else:
    raise NotImplementedError()

# %%
# load the data
if evaluation == 'inter-session':
    X, labels, metadata = obj_prep.get_data(obj_ds,[subject],False)
elif evaluation == 'inter-subject':
    X, labels, metadata = obj_prep.get_data(obj_ds,None,False)

le = LabelEncoder().fit(labels)
y = le.transform(labels)
subjects = metadata.subject.to_numpy()
sessions = metadata.session.to_numpy()

if evaluation == 'inter-session':
    keep = (subjects == subject) & (sessions != session)
    domain_expression = "session"
elif evaluation == 'inter-subject':
    keep = subjects != subject
    domain_expression = "session + subject * 1000"

X = X[keep]
y = y[keep]
domains = metadata.eval(domain_expression).to_numpy(dtype=np.int64)[keep]

Xt = torch.tensor(X, dtype=torch.float32)
yt, dt = torch.LongTensor(y), torch.LongTensor(domains)

# %%
assert(isinstance(obj, PatternInterpretableModel))

from scipy.linalg import eigh

def orderedeigh(A : torch.Tensor, B: torch.Tensor):

    Aflat = A.flatten(end_dim=-3).detach()
    assert(B.ndim == 3 and B.shape[0] == 1)
    
    s = []
    U = []
    for i in range(Aflat.shape[0]):
        s_, U_ = eigh(Aflat[i], B[0].detach())
        s.append(torch.tensor(s_))
        U.append(torch.tensor(U_))
    s = torch.stack(s, dim=0).reshape(A.shape[:-1])
    U = torch.stack(U, dim=0).reshape(A.shape)
    # all EVs should be positive
    e = s.clip(min=1e-4)
    # order according to largest eigenvale or inverse eigenvalue
    e = torch.max(e, 1/e)
    ix = torch.argsort(e, dim=-1, descending=True)
    shapeu = (1,) * (ix.ndim-1) + (U.shape[-1], 1)
    ixu = ix[...,None,:].repeat(shapeu)
    e = torch.take_along_dim(e, ix, dim=-1)
    s = torch.take_along_dim(s, ix, dim=-1)
    U = torch.take_along_dim(U, ixu, dim=-1)
    return s, U, e

def compute_patterns(self, x : torch.Tensor, y, d):

    cov_estimator = modules.CovariancePool(alpha=1e-6,unitvar=False)

    # -----
    # forward pass
    domains, domainobs = d.unique(return_counts=True)
    Ch1h1 = []
    Cm = []
    Cll = []
    n_reigactive = 0
    for du in domains: # pass each domain (otherwise we might have too much data)
        # 1. convolutional layers
        h1 = self.cnn[0](x[d==du,None,...])
        # compute spatial filter input covariance matrix
        h1_flat = h1.flatten(start_dim=1, end_dim=2)
        Ch1h1.append(cov_estimator(h1_flat).mean(dim=0, keepdim=True))
        h2 = self.cnn[1](h1)
        h = self.cnn[2](h2)
        # 2. covariance pooling
        C = self.cov_pooling(h).to(dtype=torch.double)
        # 3. SPDNet+BN
        # compute bimap input covariance matrix
        Cm.append(C.mean(dim=0, keepdim=True))
        S = self.spdnet[0](C)
        s = torch.linalg.eigvalsh(S)
        reig_th = self.spdnet[1].threshold
        n_reigactive += (s < reig_th).any(dim=-1).sum()
        S = self.spdnet[1](S)
        S = self.spdbnorm(S) if hasattr(self, 'spdbnorm') else S
        S = self.spddsbnorm(S,d[d==du]) if hasattr(self, 'spddsbnorm') else S
        # 4. tangent space projection + BN
        L = self.logeig(S)
        L = self.tsbnorm(L) if hasattr(self, 'tsbnorm') else L
        L = self.tsdsbnorm(L,d[d==du]) if hasattr(self, 'tsdsbnorm') else L
        Cll.append(cov_estimator(L[None,...].swapaxes(-1,-2)))
        # 5. linear classification
        # y = self.classifier(L)
    
    print(f'ReEig nonlinearity was active in {n_reigactive/x.shape[0] *100}% of the observations.')

    dom_weights = domainobs[:,None,None] / domainobs.sum()
    Ch1h1 = (torch.cat(Ch1h1, dim=0) * dom_weights).sum(dim=0, keepdim=True)
    Cm = (torch.cat(Cm, dim=0) * dom_weights).sum(dim=0, keepdim=True)
    Cll = (torch.cat(Cll, dim=0) * dom_weights).sum(dim=0, keepdim=False)
    
    # -----
    # compute classifier patterns
    W = self.classifier[-1].weight
    if self.nclasses_ == 2:
        # take only one weight vector for the binary case (due to redundancy)
        W = W[:-1,:]
    # convert the regression coefficients to patterns
    Css = W @ Cll @ W.T
    D = torch.linalg.solve(F.sym_sqrtm.apply(Css),W @ Cll)

    # -----
    # SPDNet back projection of patterns
    # revert logeig
    D = modules.TrilEmbedder().inverse_transform(D)
    # revert SPD batch normalization
    if hasattr(self, 'spdbnorm'):
        Cref = self.spdbnorm.running_mean
    elif hasattr(self, 'spddsbnorm'):
        Cref = [self.spddsbnorm.get_domain_obj(du).running_mean for du in domains]
        Cref = torch.cat(Cref, dim=0)
        if Cref.shape[0] > 1:
            # choose the geometric mean of all domains as the representative reference
            Cref = F.spd_mean_kracher_flow(Cref)
    else:
        refshape = (1,) + D.shape[1:-1]
        Cref = torch.diag_embed(torch.ones(refshape, device=D.device, dtype=D.dtype))
    Crefsq = F.sym_sqrtm.apply(Cref)
    # project tangent space patterns and compute contributing sources
    CA = Crefsq @ F.sym_expm.apply(D) @ Crefsq
    s, U, e = orderedeigh(CA, Cref)
    # convert generalized eigenvectors to patterns
    P = torch.linalg.pinv(U).swapaxes(-1,-2)
    # revert BiMap layer
    W_bimap = self.spdnet[0].W.swapaxes(-1,-2)
    A_bimap = torch.linalg.solve(W_bimap @ Cm @ W_bimap.swapaxes(-1,-2), W_bimap @ Cm).swapaxes(-1,-2)
    P = A_bimap @ P

    # -----
    # revert spatial filter step
    kernel_shape = self.cnn[1].weight.squeeze().shape[1:]
    W_spat = self.cnn[1].weight.flatten(start_dim=1)[None,...]    
    A_spat = torch.linalg.solve(W_spat @ Ch1h1 @ W_spat.swapaxes(-1,-2), W_spat @ Ch1h1).swapaxes(-1,-2).to(dtype=torch.double)
    P = A_spat @ P

    V_spat = torch.linalg.pinv(P).detach()

    new_shape = P.shape[:1] + kernel_shape + P.shape[-1:]
    P = torch.reshape(P, new_shape).detach()

    # -----
    # compute spectral filter patterns
    PSD = []
    PSD_layer1 = []
    for du in domains: # pass each domain (otherwise we have too much data)
        # apply spectral filters
        h1 = self.cnn[0](x[d==du,None,...]).to(dtype=torch.double).detach()
        h1_flat = h1.flatten(start_dim=1, end_dim=2)
        s_flat = torch.einsum("oft,clf->colt", h1_flat, V_spat)
        # Bartlett's method to estimate the power spectral density
        ntaps = s_flat.shape[-1]
        PSD_du = torch.fft.fftshift(torch.fft.fft(s_flat, dim=-1),dim=-1).abs().square().mean(axis=1, keepdim=False) / ntaps
        PSD_layer1_du = torch.fft.fftshift(torch.fft.fft(h1, dim=-1),dim=-1).abs().square().mean(axis=0, keepdim=False) / ntaps
        
        PSD.append(PSD_du)
        PSD_layer1.append(PSD_layer1_du)

    PSD = (torch.stack(PSD, dim=0) * dom_weights[:,None,...]).sum(dim=0, keepdim=False)
    PSD = 20 * torch.log10(PSD)
    PSD_layer1 = (torch.stack(PSD_layer1, dim=0) * dom_weights[:,None,...]).sum(dim=0, keepdim=False)
    PSD_layer1 = 20 * torch.log10(PSD_layer1)
    f_psd = torch.fft.fftshift(torch.fft.fftfreq(ntaps))
    
    W_spectral = self.cnn[0].weight.detach().flatten(start_dim=1)[None,...]
    H_spectral = torch.fft.fftshift(torch.fft.fft(W_spectral, dim=-1),dim=-1)
    H_mag = 20 * torch.log10(H_spectral.abs())
    f_mag = torch.fft.fftshift(torch.fft.fftfreq(W_spectral.shape[-1]))

    return s.detach(), P, e.detach(), f_mag, H_mag, f_psd, PSD, PSD_layer1

s, U, e, f_mag, H_mag, f_psd, PSD, PSD_l1 = compute_patterns(obj, Xt, yt, dt)

f_psd = f_psd * info['sfreq']
f_mag = f_mag * info['sfreq']


# %%


def plot_eeg_bands(ax, ylim):
    r_t = Rectangle([4, ylim[0]], 4, ylim[1] - ylim[0], facecolor='r',alpha=0.3)
    ax.add_patch(r_t)  
    r_a = Rectangle([8, ylim[0]], 7, ylim[1] - ylim[0], facecolor='g',alpha=0.3)
    ax.add_patch(r_a)
    r_b = Rectangle([15, ylim[0]], 15, ylim[1] - ylim[0], facecolor='b',alpha=0.3)
    ax.add_patch(r_b)


Ach = U[class_idx]
ech = e[class_idx]
ncols = Ach.shape[0]
nrows = min(5, Ach.shape[2])

# grid layout
fig = plt.figure(tight_layout=False, figsize=[1.*(ncols+1), 1.*(nrows+1)])
gs = fig.add_gridspec(nrows+1, ncols+1, hspace=1., wspace=1.)

# components plot
ax = fig.add_subplot(gs[0, 0])
ax.plot(ech, marker='.', color='k')
ax.set_title('source\nindex')
ax.xaxis.set_ticklabels([])
ax.set_ylabel('eigenvalue')
ax.patch.set_alpha(0.0)
plt.yticks(rotation=90, va='center')

# source spectra plots
xlim = [0, min(info['sfreq']/2, 50)]
fmask = (f_psd >= xlim[0]) & (f_psd <=xlim[1])
ylim = [PSD[class_idx, 0:nrows,fmask].min()-6, PSD[class_idx, 0:nrows,fmask].max()+6]
gs_sub = gs[1:,0].subgridspec(nrows,1, wspace=0.2)
for row in range(nrows):
    ax = fig.add_subplot(gs_sub[row, 0])
    ax.plot(f_psd, PSD[class_idx,row,:], color='k')
    ax.set_xlim(*xlim)
    ax.set_xlabel('frequency (Hz)')
    ax.set_aspect(0.5)

    plot_eeg_bands(ax, ylim)

    ax.set_ylim(*ylim)
    if row < nrows-1:
        ax.xaxis.set_ticklabels([])
        ax.set_xlabel('')
    ax.set_ylabel('magnitude (dB)')

    plt.yticks(rotation=90, va='center')
    ax.patch.set_alpha(0.0)
    # ax.set_ylabel(f'source {row+1}', size='medium')
    ax.set_title(f'source {row+1}', size='medium')


# temporal filter impulse response plots
ylim = [PSD_l1[:,:,fmask].mean(axis=1).min()-6, PSD_l1[:,:,fmask].mean(axis=1).max()+6]
gs_sub = gs[0,1:].subgridspec(1, ncols, hspace=0.2)
for col in range(ncols):
    ax = fig.add_subplot(gs_sub[0, col])
    psddf = pd.DataFrame(PSD_l1[col].swapaxes(0,1).numpy(), columns=[ f'{ch}' for ch in range(PSD_l1.shape[1])])
    psddf['frequency'] = f_psd.numpy()
    psddf = psddf.set_index('frequency').unstack().reset_index(drop=False).rename(columns={'level_0':'channel', 0:'psd'})    
    sns.lineplot(data=psddf, x='frequency', y='psd', color='k', ci='sd')
    ax.set_xlim(*xlim)
    ax.set_aspect(0.5)
    plot_eeg_bands(ax, ylim)   

    ax.yaxis.set_major_locator(MultipleLocator(30))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.set_xlabel('frequency (Hz)')
    if col == 0:
        ax.set_ylabel('magnitude (dB)')
    else:
        ax.set_ylabel('')
        ax.yaxis.set_ticklabels([])
    ax.set_ylim(*ylim)
    plt.yticks(rotation=90, va='center')
    ax.patch.set_alpha(0.0)
    ax.set_title(f'spectral\nchannnel {col+1}', size='medium')

# spatial patterns for components and temporal filters
lim = np.quantile(np.abs(Ach[:,:,0:ncols]).flatten(), 0.99)
gs_sub = gs[1:,1:].subgridspec(nrows, ncols, hspace=0.2, wspace=0.2)
for row in range(nrows):
    for col in range(ncols):
        ax = fig.add_subplot(gs_sub[row, col])
        mne.viz.plot_topomap(Ach[col,:,row], info, show = False, axes = ax, 
                    sensors=False, extrapolate='local', contours = 4,
                    vmin=-lim, vmax=lim)


for ax in fig.axes:
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize('small')

sns.despine()
fig.tight_layout()
os.makedirs('figures', exist_ok=True)
plt.savefig(f'figures/fig-figure3-patterns-{dataset}-{model_name}-{evaluation}-{subject}-{session}-{le.classes_[class_idx]}.pdf')
