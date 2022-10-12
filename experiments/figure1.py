# %%
import torch
import numpy as np
import os
from omegaconf import OmegaConf
from hydra.utils import instantiate
from sklearn.preprocessing import LabelEncoder
from library.utils.moabb import CachedParadigm

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.manifold import TSNE
from library.utils.pyriemann import tsm

# %%
dataset = 'BNCI2015001'

preprocessing = 'bb4-36Hz'
model_name = 'TSMNet+SPDDSMBN'
evaluation = 'inter-subject'
subject = 7
session = -1

prespdbn = True # False

displaysubjects = range(1,11)
displaysessions = range(1,3)

mdl_path = os.path.join('outputs',dataset,evaluation,preprocessing,'models',str(subject),str(session), model_name + '.pth')
mdl_metadata_path = os.path.join('outputs',dataset,evaluation,preprocessing,'metadata', 'meta-'+ model_name + '.pth')

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
# load the data
if evaluation == 'inter-session':
    X, y, metadata = obj_prep.get_data(obj_ds,[subject],False)
elif evaluation == 'inter-subject':
    X, y, metadata = obj_prep.get_data(obj_ds,displaysubjects,False)

msk = metadata.session.isin(displaysessions)
X = X[msk]
y = y[msk]
metadata = metadata[msk]

le = LabelEncoder()
y = le.fit_transform(y)
subjects = metadata.subject.to_numpy()
sessions = metadata.session.to_numpy()

if evaluation == 'inter-session':
    domain_expression = "session"
    groups = sessions
elif evaluation == 'inter-subject':
    domain_expression = "session + subject * 1000"
    groups = subjects

domains = metadata.eval(domain_expression).to_numpy(dtype=np.int64)

if evaluation == 'inter-session':
    target_domains = np.unique(domains[(metadata.subject == subject) & (metadata.session == session)])
elif evaluation == 'inter-subject':
    target_domains = np.unique(domains[metadata.subject == subject])

Xt = torch.tensor(X, dtype=torch.float32)
yt, dt = torch.LongTensor(y), torch.LongTensor(domains)

# %%
obj.eval()
if prespdbn:
    y_hat,l = obj.forward(Xt, dt, return_latent=False, return_prebn=True)
    l = tsm(l.detach().numpy())
    fig_label = 'prespddsmbn'
else:
    y_hat,l = obj.forward(Xt, dt, return_latent=True, return_prebn=False)
    l = l.detach().numpy()
    fig_label = 'postspddsmbn'


# %%
tsne = TSNE(n_components=2)
Xtsne = tsne.fit_transform(l[y<2])      

# %%
df = pd.DataFrame(dict(dim1=Xtsne[:,0], dim2=Xtsne[:,1],label=y[y<2],domain=dt[y<2]))
df['set'] = 'source'
df.loc[df.domain.isin(target_domains),'set'] = 'target'
# df = df.loc[df.domain.isin(range(10))]
df = df.loc[df.label.isin(range(2))]
fig, axes = plt.subplots(1, 2, figsize=(6, 3), sharey=True, sharex=True)
axes[0].set_ylabel("tsne dim 2 (a.u.)")
axes[0].set_xlabel("tsne dim 1 (a.u.)")
axes[1].set_xlabel("tsne dim 1 (a.u.)")
palette = sns.color_palette('Greys', len(df[df.set == 'source'].domain.unique())+1)
palette = sns.color_palette(palette[1:])
palette = sns.color_palette('tab20')
palette_train = sns.color_palette(palette[:6] + palette[8:])
palette_label_train = sns.color_palette(palette[1:5:2])
# palette = sns.color_palette(palette)
palette_test = sns.color_palette(palette[6:8])
palette_label_test = sns.color_palette(palette[0:4:2])

# markers = {0: "$0$", 1: "$1$"}
black = sns.color_palette(['k']*len(dt.unique()))
sns.scatterplot(data=df, x='dim1',y='dim2', hue = 'label', style='set',
                ax=axes[0], alpha = 0.1, palette=palette_label_train)
sns.scatterplot(data=df[df.set == 'target'], x='dim1',y='dim2', hue = 'label',
                ax=axes[0], alpha = 0.5, palette=palette_label_test, marker='X', legend=None)
sns.kdeplot(data=df[df.set == 'source'], x='dim1',y='dim2', hue = 'domain', levels=1,
                ax=axes[1], alpha = 0.75, palette=palette_train)
sns.scatterplot(data=df[df.set == 'source'].groupby('domain').agg('mean'), x='dim1',y='dim2', hue = 'domain',
                ax=axes[1], alpha = 0.75, palette=palette_train, legend=False)
sns.kdeplot(data=df[df.set == 'target'], x='dim1',y='dim2', hue = 'domain', levels=1,
                ax=axes[1], alpha = 1.0, palette=palette_test, linewidths=2, legend=False)
sns.scatterplot(data=df[df.set == 'target'].groupby('domain').agg('mean'), x='dim1',y='dim2', hue = 'domain', 
                ax=axes[1], alpha = 0.75, palette=palette_test)

plt.suptitle(f'latent space {fig_label} (2D tSNE projection)')
os.makedirs('figures', exist_ok=True)
plt.savefig(f'figures/fig-figure1c-{fig_label}.pdf', transparent=True)


