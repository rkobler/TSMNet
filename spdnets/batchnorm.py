from builtins import NotImplementedError
from enum import Enum
from typing import Tuple
import torch
from torch.functional import Tensor
import torch.nn as nn
from torch.types import Number

from geoopt.tensor import ManifoldParameter, ManifoldTensor
from .manifolds import SymmetricPositiveDefinite
from . import functionals
from skorch.callbacks import Callback
from skorch import NeuralNet

# %% schedulers

class DummyScheduler(Callback):
    pass


class ConstantMomentumBatchNormScheduler(Callback):
    def __init__(self, eta, eta_test) -> None:
        self.eta0 = eta
        self.eta0_test = eta_test
        self.bn_modules_ = []

    def initialize(self):
        super().initialize()
        self.eta_ = self.eta0
        self.eta_test_ = self.eta0_test
        self.bn_modules_ = []
        return self

    def on_train_begin(self, net : NeuralNet, **kwargs):
        model = net.module_
        # extract momentum batch norm parameters
        if model is not None:
            self.bn_modules_ = [m for m in model.modules() 
                if isinstance(m, SchedulableBatchNorm) or isinstance(m, SchedulableDomainBatchNorm)]
        else:
            self.bn_modules_ = []

        for m in self.bn_modules_:
            m.set_eta(eta=self.eta_, eta_test = self.eta_test_)

    def __repr__(self) -> str:
        return f'ConstantMomentumBatchNormScheduler - eta={self.eta_:.3f}, eta_test={self.eta_test_:.3f}'


class MomentumBatchNormScheduler(ConstantMomentumBatchNormScheduler):
    def __init__(self, epochs : Number, bs : Number = 32, bs0 : Number = 64, tau0 : Number = 0.9) -> None:
        assert(bs <= bs0)
        super().__init__(1. - tau0, 1. - tau0 ** (bs/bs0))
        self.epochs = epochs
        self.rho = (bs/bs0) ** (1/self.epochs)
        self.tau0 = tau0
        self.bs = bs
        self.bs0 = bs0
    
    def initialize(self):
        super().initialize()
        self.epoch_ = 1
        return self

    def __repr__(self) -> str:
        return f'MomentumBatchNormScheduler - eta={self.eta_:.3f}, eta_tst={self.eta_test_:.3f}'

    def on_epoch_begin(self, net, **kwargs):
        self.eta_ = 1. - (self.rho ** (self.epochs * max(self.epochs - self.epoch_,0)/(self.epochs-1)) - self.rho ** self.epochs)
        for m in self.bn_modules_:
            m.set_eta(eta = self.eta_)
        
        w = max(self.epochs - self.epoch_,0)/(self.epochs-1)
        tau_test = self.tau0 ** (self.bs/self.bs0 * (1-w) + w * 1)
        self.eta_test_ = 1 - tau_test
        for m in self.bn_modules_:
            m.set_eta(eta_test = 1. - self.eta_test_)

        self.epoch_ += 1


class BatchNormTestStatsMode(Enum):
    BUFFER = 'buffer'
    REFIT = 'refit'
    ADAPT = 'adapt'


class BatchNormTestStatsModeScheduler(Callback):

    def __init__(self, 
                 fit_mode : BatchNormTestStatsMode = BatchNormTestStatsMode.BUFFER, 
                 predict_mode : BatchNormTestStatsMode = BatchNormTestStatsMode.BUFFER) -> None:
        self.fit_mode = fit_mode
        self.predict_mode = predict_mode

    def on_train_begin(self, net : NeuralNet, **kwargs):
        model = net.module_
        for m in model.modules():
            if isinstance(m, BatchNormTestStatsInterface):
                m.set_test_stats_mode(self.fit_mode)

    def on_train_end(self, net : NeuralNet, **kwargs):
        model = net.module_
        for m in model.modules():
            if isinstance(m, BatchNormTestStatsInterface):
                m.set_test_stats_mode(self.predict_mode)


class BatchNormDispersion(Enum):
    NONE = 'mean'
    SCALAR = 'scalar'
    VECTOR = 'vector'


class BatchNormTestStatsInterface:
    def set_test_stats_mode(self, mode : BatchNormTestStatsMode):
        pass

# %% base classes

class BaseBatchNorm(nn.Module, BatchNormTestStatsInterface):
    def __init__(self, eta = 1.0, eta_test = 0.1, test_stats_mode : BatchNormTestStatsMode = BatchNormTestStatsMode.BUFFER):
        super().__init__()
        self.eta = eta
        self.eta_test = eta_test
        self.test_stats_mode = test_stats_mode

    def set_test_stats_mode(self, mode : BatchNormTestStatsMode):
        self.test_stats_mode = mode


class SchedulableBatchNorm(BaseBatchNorm):
    def set_eta(self, eta = None, eta_test = None):
        if eta is not None:
            self.eta = eta
        if eta_test is not None:
            self.eta_test = eta_test


class BaseDomainBatchNorm(nn.Module, BatchNormTestStatsInterface):
    def __init__(self):
        super().__init__()
        self.batchnorm = torch.nn.ModuleDict()

    def set_test_stats_mode(self, mode : BatchNormTestStatsMode):
        for bn in self.batchnorm.values():
            if isinstance(bn, BatchNormTestStatsInterface):
                bn.set_test_stats_mode(mode)

    def add_domain_(self, layer : BaseBatchNorm, domain : Tensor):
        self.batchnorm[f'dom {domain.item()}'] = layer

    def get_domain_obj(self, domain : Tensor):
        return self.batchnorm[f'dom {domain.item()}']

    @torch.no_grad()
    def initrunningstats(self, X, domain):
        self.batchnorm[f'dom {domain.item()}'].initrunningstats(X)

    def forward_domain_(self, X, domain):
        res = self.batchnorm[f'dom {domain.item()}'](X)
        return res

    def forward(self, X, d):
        du = d.unique()

        X_normalized = torch.empty_like(X)
        res = [(self.forward_domain_(X[d==domain], domain),torch.nonzero(d==domain))
                for domain in du]
        X_out, ixs = zip(*res)
        X_out, ixs = torch.cat(X_out), torch.cat(ixs).flatten()
        X_normalized[ixs] = X_out
        
        return X_normalized


class SchedulableDomainBatchNorm(BaseDomainBatchNorm,SchedulableBatchNorm):
    def set_eta(self, eta = None, eta_test = None):
        for bn in self.batchnorm.values():
            if isinstance(bn, SchedulableBatchNorm):
                bn.set_eta(eta, eta_test)

# %% Euclidean vector space implementation

class BatchNormImpl(BaseBatchNorm):
    def __init__(self, shape : Tuple[int,...] or torch.Size, batchdim : int, 
                 eta = 0.1, eta_test = 0.1,
                 dispersion : BatchNormDispersion = BatchNormDispersion.NONE,
                 learn_mean : bool = True, learn_std : bool = True,
                 mean = None, std = None, eps = 1e-5, **kwargs):
        super().__init__(eta=eta, eta_test=eta_test)

        self.dispersion = dispersion
        self.batchdim = batchdim
        self.eps = eps

        init_mean = torch.zeros(shape, **kwargs)
        self.register_buffer('running_mean', init_mean)
        self.register_buffer('running_mean_test', init_mean.clone())
        if mean is not None:
            self.mean = mean
        else:
            if learn_mean:
                self.mean = nn.parameter.Parameter(init_mean.clone())
            else:
                self.mean = init_mean.clone()

        if std is not None:
            self.std = std
            self.register_buffer('running_var', torch.ones(std.shape, **kwargs))
            self.register_buffer('running_var_test', torch.ones(std.shape, **kwargs))
        elif self.dispersion == BatchNormDispersion.SCALAR:
            var_init = torch.ones((*shape[:-1], 1), **kwargs)
            self.register_buffer('running_var', var_init)
            self.register_buffer('running_var_test', var_init.clone())
            if learn_std:
                self.std = nn.parameter.Parameter(var_init.clone().sqrt())
            else:
                self.std = var_init.clone().sqrt()
        elif self.dispersion == BatchNormDispersion.VECTOR:
            var_init = torch.ones(shape, **kwargs)
            self.register_buffer('running_var', var_init)
            self.register_buffer('running_var_test', var_init.clone())
            if learn_std:
                self.std = nn.parameter.Parameter(var_init.clone().sqrt())
            else:
                self.std = var_init.clone().sqrt()

    @torch.no_grad()
    def initrunningstats(self, X):
        self.running_mean = X.mean(dim=self.batchdim,keepdim=True).clone()
        self.running_mean_test = self.running_mean.clone()
        if self.dispersion == BatchNormDispersion.SCALAR:
            self.running_var = (X - self.running_mean).square().mean(keepdim=True)
            self.running_var_test = self.running_var.clone()
        elif self.dispersion == BatchNormDispersion.VECTOR:
            self.running_var = (X - self.running_mean).square().mean(dim=self.batchdim, keepdim=True)
            self.running_var_test = self.running_var.clone()

    def forward(self, X):
        if self.training:
            
            batch_mean = X.mean(dim=self.batchdim, keepdim=True)
            rm = (1. - self.eta) * self.running_mean + self.eta * batch_mean

            # compute the dispersion of the batch to the 
            # updated running mean
            if self.dispersion is not BatchNormDispersion.NONE:
                if self.dispersion == BatchNormDispersion.SCALAR:
                    batch_var = (X - batch_mean).square().mean(keepdim=True)
                elif self.dispersion == BatchNormDispersion.VECTOR:
                    batch_var = (X - batch_mean).square().mean(dim=self.batchdim, keepdim=True)
                rv = (1. - self.eta) * self.running_var + self.eta * batch_var
        else:
            if self.test_stats_mode == BatchNormTestStatsMode.BUFFER:
                pass # nothing to do: use the ones in the buffer
            elif self.test_stats_mode == BatchNormTestStatsMode.REFIT:
                self.initrunningstats(X)
            elif self.test_stats_mode == BatchNormTestStatsMode.ADAPT:
                raise NotImplementedError()

            rm = self.running_mean_test
            if self.dispersion is not BatchNormDispersion.NONE:
                rv = self.running_var_test

        # rescale to desired dispersion
        if self.dispersion is not BatchNormDispersion.NONE:
            Xn =  (X - rm) / (rv + self.eps).sqrt() * self.std + self.mean
        else:
            Xn = X - rm + self.mean

        if self.training:
            with torch.no_grad():
                self.running_mean = rm.clone()
                self.running_mean_test = (1. - self.eta_test) * self.running_mean_test + self.eta_test * batch_mean
                if self.dispersion is not BatchNormDispersion.NONE:
                    self.running_var = rv.clone()
                    self.running_var_test = (1. - self.eta_test) * self.running_var_test + \
                        self.eta_test * batch_var
        return Xn


class BatchNorm(BatchNormImpl):
    """
    Standard batch normalization as presented in [Ioffe and Szegedy 2020, ICML].
    """
    def __init__(self, shape: Tuple[int, ...] or torch.Size, 
                 batchdim: int,
                 eta=0.1, **kwargs):
        if 'dispersion' not in kwargs.keys():
            kwargs['dispersion'] = BatchNormDispersion.VECTOR
        if 'eta_test' in kwargs.keys():
            raise RuntimeError('This parameter is ignored in this subclass. Use another batch normailzation variant.')
        super().__init__(shape=shape, batchdim=batchdim, 
                         eta=1.0, eta_test=eta, **kwargs)


class BatchReNorm(BatchNormImpl):
    """
    Standard batch normalization as presented in [Ioffe 2017, NIPS].
    """
    def __init__(self, shape: Tuple[int, ...] or torch.Size, 
                 batchdim: int,
                 eta=0.1, **kwargs):
        if 'dispersion' not in kwargs.keys():
            kwargs['dispersion'] = BatchNormDispersion.VECTOR
        if 'eta_test' in kwargs.keys():
            raise RuntimeError('This parameter is ignored in this subclass.')
        super().__init__(shape=shape, batchdim=batchdim, 
                         eta=eta, eta_test=eta, **kwargs)


class AdaMomBatchNorm(BatchNormImpl,SchedulableBatchNorm):
    """
    Adaptive Momentum Batch Normalization as presented in [Yong et al. 2020, ECCV].

    The momentum terms can be controlled via a momentum scheduler.
    """
    def __init__(self, shape: Tuple[int, ...] or torch.Size, 
                 batchdim: int,
                 eta=1.0, eta_test=0.1, **kwargs):
        super().__init__(shape=shape, batchdim=batchdim, 
                         eta=eta, eta_test=eta_test, **kwargs)


class DomainBatchNormImpl(BaseDomainBatchNorm):
    def __init__(self, shape : Tuple[int,...] or torch.Size, batchdim :int,
                 learn_mean : bool = True, learn_std : bool = True,
                 dispersion : BatchNormDispersion = BatchNormDispersion.NONE,
                 test_stats_mode : BatchNormTestStatsMode = BatchNormTestStatsMode.BUFFER,
                 eta = 1., eta_test = 0.1, domains : list = [], **kwargs):
        super().__init__()

        self.dispersion = dispersion
        self.learn_mean = learn_mean
        self.learn_std = learn_std

        init_mean = torch.zeros(shape, **kwargs)
        if self.learn_mean:
            self.mean = nn.parameter.Parameter(init_mean)
        else:
            self.mean = init_mean
        
        if self.dispersion is BatchNormDispersion.SCALAR:
            init_var = torch.ones((*shape[:-2], 1), **kwargs)
        elif self.dispersion == BatchNormDispersion.VECTOR:
            init_var = torch.ones(shape, **kwargs)
        else:
            init_var = None

        if self.learn_std:
            self.std = nn.parameter.Parameter(init_var.clone())
        else:
            self.std = init_var

        cls = type(self).domain_bn_cls
        for domain in domains:
            self.add_domain_(cls(shape=shape, batchdim=batchdim, 
                                learn_mean=learn_mean, learn_std=learn_std, dispersion=dispersion,
                                mean=self.mean, std=self.std, eta=eta, eta_test=eta_test, **kwargs),
                            domain)

        self.set_test_stats_mode(test_stats_mode)


class DomainBatchNorm(DomainBatchNormImpl):
    """
    Domain-specific batch normalization as presented in [Chang et al. 2019, CVPR]

    Keeps running stats for each domain. Scaling and bias parameters are shared across domains.
    """
    domain_bn_cls = BatchNormImpl


class AdaMomDomainBatchNorm(DomainBatchNormImpl,SchedulableDomainBatchNorm):
    """
    Combines domain-specific batch normalization [Chang et al. 2019, CVPR]
    with adaptive momentum batch normalization [Yong et al. 2020, ECCV].

    Keeps running stats for each domain. Scaling and bias parameters are shared across domains.
    The momentum terms can be controlled with a momentum scheduler.
    """
    domain_bn_cls = AdaMomBatchNorm

# %% SPD manifold implementation

class SPDBatchNormImpl(BaseBatchNorm):
    def __init__(self, shape : Tuple[int,...] or torch.Size, batchdim : int, 
                 eta = 1., eta_test = 0.1,
                 karcher_steps : int = 1, learn_mean = True, learn_std = True, 
                 dispersion : BatchNormDispersion = BatchNormDispersion.SCALAR, 
                 eps = 1e-5, mean = None, std = None, **kwargs):
        super().__init__(eta, eta_test)
        # the last two dimensions are used for SPD manifold
        assert(shape[-1] == shape[-2])

        if dispersion == BatchNormDispersion.VECTOR:
            raise NotImplementedError()

        self.dispersion = dispersion
        self.learn_mean = learn_mean
        self.learn_std = learn_std
        self.batchdim = batchdim
        self.karcher_steps = karcher_steps
        self.eps = eps
        
        init_mean = torch.diag_embed(torch.ones(shape[:-1], **kwargs))
        init_var = torch.ones((*shape[:-2], 1), **kwargs)

        self.register_buffer('running_mean', ManifoldTensor(init_mean, 
                                           manifold=SymmetricPositiveDefinite()))
        self.register_buffer('running_var', init_var)
        self.register_buffer('running_mean_test', ManifoldTensor(init_mean, 
                                           manifold=SymmetricPositiveDefinite()))
        self.register_buffer('running_var_test', init_var)

        if mean is not None:
            self.mean = mean
        else:
            if self.learn_mean:
                self.mean = ManifoldParameter(init_mean.clone(), manifold=SymmetricPositiveDefinite())
            else:
                self.mean = ManifoldTensor(init_mean.clone(), manifold=SymmetricPositiveDefinite())
        
        if self.dispersion is not BatchNormDispersion.NONE:
            if std is not None:
                self.std = std
            else:
                if self.learn_std:
                    self.std = nn.parameter.Parameter(init_var.clone())
                else:
                    self.std = init_var.clone()

    @torch.no_grad()
    def initrunningstats(self, X):
        self.running_mean.data, geom_dist = functionals.spd_mean_kracher_flow(X, dim=self.batchdim, return_dist=True)
        self.running_mean_test.data = self.running_mean.data.clone()

        if self.dispersion is BatchNormDispersion.SCALAR:
            self.running_var = geom_dist.square().mean(dim=self.batchdim, keepdim=True).clamp(min=functionals.EPS[X.dtype])[...,None]
            self.running_var_test = self.running_var.clone()

    def forward(self, X):
        manifold = self.running_mean.manifold
        if self.training:
            # compute the Karcher flow for the current batch
            batch_mean = X.mean(dim=self.batchdim, keepdim=True)
            for _ in range(self.karcher_steps):
                bm_sq, bm_invsq = functionals.sym_invsqrtm2.apply(batch_mean.detach())
                XT = functionals.sym_logm.apply(bm_invsq @ X @ bm_invsq)
                GT = XT.mean(dim=self.batchdim, keepdim=True)
                batch_mean = bm_sq @ functionals.sym_expm.apply(GT) @ bm_sq
            
            # update the running mean
            rm = functionals.spd_2point_interpolation(self.running_mean, batch_mean, self.eta)

            if self.dispersion is BatchNormDispersion.SCALAR:
                GT = functionals.sym_logm.apply(bm_invsq @ rm @ bm_invsq)
                batch_var = torch.norm(XT - GT, p='fro', dim=(-2,-1), keepdim=True).square().mean(dim=self.batchdim, keepdim=True).squeeze(-1)
                rv = (1. - self.eta) * self.running_var + self.eta * batch_var

        else:
            if self.test_stats_mode == BatchNormTestStatsMode.BUFFER:
                pass # nothing to do: use the ones in the buffer
            elif self.test_stats_mode == BatchNormTestStatsMode.REFIT:
                self.initrunningstats(X)
            elif self.test_stats_mode == BatchNormTestStatsMode.ADAPT:
                raise NotImplementedError()

            rm = self.running_mean_test
            if self.dispersion is BatchNormDispersion.SCALAR:
                rv = self.running_var_test

        # rescale to desired dispersion
        if self.dispersion is BatchNormDispersion.SCALAR:
            Xn = manifold.transp_identity_rescale_transp(X, 
                rm, self.std/(rv + self.eps).sqrt(), self.mean)
        else:
            Xn = manifold.transp_via_identity(X, rm, self.mean)

        if self.training:
            with torch.no_grad():
                self.running_mean.data = rm.clone()
                self.running_mean_test.data = functionals.spd_2point_interpolation(self.running_mean_test, batch_mean, self.eta_test)
                if self.dispersion is not BatchNormDispersion.NONE:
                    self.running_var = rv.clone()
                    GT_test = functionals.sym_logm.apply(bm_invsq @ self.running_mean_test @ bm_invsq)
                    batch_var_test = torch.norm(XT - GT_test, p='fro', dim=(-2,-1), keepdim=True).square().mean(dim=self.batchdim, keepdim=True).squeeze(-1)

                    self.running_var_test = (1. - self.eta_test) * self.running_var_test + self.eta_test * batch_var_test

        return Xn


class SPDBatchNorm(SPDBatchNormImpl):
    """
    Batch normalization on the SPD manifold.
    
    Class implements [Brooks et al. 2019, NIPS] (dispersion= ``BatchNormDispersion.NONE``) 
    and [Kobler et al. 2022, ICASSP] (dispersion= ``BatchNormDispersion.SCALAR``).
    By default dispersion = ``BatchNormDispersion.SCALAR``.
    """
    def __init__(self, shape: Tuple[int, ...] or torch.Size, 
                 batchdim: int,
                 eta=0.1, **kwargs):
        if 'dispersion' not in kwargs.keys():
            kwargs['dispersion'] = BatchNormDispersion.SCALAR
        if 'eta_test' in kwargs.keys():
            raise RuntimeError('This parameter is ignored in this subclass. Use another batch normailzation variant.')
        super().__init__(shape=shape, batchdim=batchdim, 
                         eta=1.0, eta_test=eta, **kwargs)


class SPDBatchReNorm(SPDBatchNormImpl):
    """
    Batch re normalization on the SPD manifold [Kobler et al. 2022, ICASSP].
    """
    def __init__(self, shape: Tuple[int, ...] or torch.Size, 
                 batchdim: int,
                 eta=0.1, **kwargs):
        if 'dispersion' not in kwargs.keys():
            kwargs['dispersion'] = BatchNormDispersion.SCALAR
        if 'eta_test' in kwargs.keys():
            raise RuntimeError('This parameter is ignored in this subclass.')
        super().__init__(shape=shape, batchdim=batchdim, 
                         eta=eta, eta_test=eta, **kwargs)


class AdaMomSPDBatchNorm(SPDBatchNormImpl,SchedulableBatchNorm):
    """
    Adaptive momentum batch normalization on the SPD manifold [proposed].

    The momentum terms can be controlled via a momentum scheduler.
    """
    def __init__(self, shape: Tuple[int, ...] or torch.Size, 
                 batchdim: int,
                 eta=1.0, eta_test=0.1, **kwargs):
        super().__init__(shape=shape, batchdim=batchdim, 
                         eta=eta, eta_test=eta_test, **kwargs)


class DomainSPDBatchNormImpl(BaseDomainBatchNorm):
    """
    Domain-specific batch normalization on the SPD manifold [proposed]

    Keeps running stats for each domain. Scaling and bias parameters are shared across domains.
    """

    domain_bn_cls = None # needs to be overwritten by subclasses

    def __init__(self, shape : Tuple[int,...] or torch.Size, batchdim :int,
                 learn_mean : bool = True, learn_std : bool = True,
                 dispersion : BatchNormDispersion = BatchNormDispersion.NONE,
                 test_stats_mode : BatchNormTestStatsMode = BatchNormTestStatsMode.BUFFER,
                 eta = 1., eta_test = 0.1, domains : Tensor = Tensor([]), **kwargs):
        super().__init__()

        assert(shape[-1] == shape[-2])

        if dispersion == BatchNormDispersion.VECTOR:
            raise NotImplementedError()

        self.dispersion = dispersion
        self.learn_mean = learn_mean
        self.learn_std = learn_std

        init_mean = torch.diag_embed(torch.ones(shape[:-1], **kwargs))
        if self.learn_mean:
            self.mean = ManifoldParameter(init_mean, 
                                        manifold=SymmetricPositiveDefinite())
        else:
            self.mean = ManifoldTensor(init_mean, 
                                       manifold=SymmetricPositiveDefinite())
        
        if self.dispersion is BatchNormDispersion.SCALAR:
            init_var = torch.ones((*shape[:-2], 1), **kwargs)
            if self.learn_std:
                self.std = nn.parameter.Parameter(init_var.clone())
            else:
                self.std = init_var.clone()
        else:
            self.std = None

        cls = type(self).domain_bn_cls
        for domain in domains:
            self.add_domain_(cls(shape=shape, batchdim=batchdim, 
                                learn_mean=learn_mean,learn_std=learn_std, dispersion=dispersion,
                                mean=self.mean, std=self.std, eta=eta, eta_test=eta_test, **kwargs),
                            domain)

        self.set_test_stats_mode(test_stats_mode)

class DomainSPDBatchNorm(DomainSPDBatchNormImpl):
    """
    Combines domain-specific batch normalization on the SPD manifold
    with adaptive momentum batch normalization [Yong et al. 2020, ECCV].

    Keeps running stats for each domain. Scaling and bias parameters are shared across domains.
    The momentum terms can be controlled with a momentum scheduler.
    """
    
    domain_bn_cls = SPDBatchNormImpl


class AdaMomDomainSPDBatchNorm(SchedulableDomainBatchNorm,DomainSPDBatchNormImpl):
    """
    Combines domain-specific batch normalization on the SPD manifold
    with adaptive momentum batch normalization [Yong et al. 2020, ECCV].

    Keeps running stats for each domain. Scaling and bias parameters are shared across domains.
    The momentum terms can be controlled with a momentum scheduler.
    """

    domain_bn_cls = AdaMomSPDBatchNorm
