import math
from typing import Optional, Union
import torch

import spdnets.modules as modules
import spdnets.batchnorm as bn
from .base import DomainAdaptFineTuneableModel, FineTuneableModel, PatternInterpretableModel


class TSMNet(DomainAdaptFineTuneableModel, FineTuneableModel, PatternInterpretableModel):
    def __init__(self, temporal_filters, spatial_filters = 40,
                 subspacedims = 20,
                 temp_cnn_kernel = 25,
                 bnorm : Optional[str] = 'spdbn', 
                 bnorm_dispersion : Union[str, bn.BatchNormDispersion] = bn.BatchNormDispersion.SCALAR,
                 **kwargs):
        super().__init__(**kwargs)

        self.temporal_filters_ = temporal_filters
        self.spatial_filters_ = spatial_filters
        self.subspacedimes = subspacedims
        self.bnorm_ = bnorm
        self.spd_device_ = torch.device('cpu')
        if isinstance(bnorm_dispersion, str):
            self.bnorm_dispersion_ = bn.BatchNormDispersion[bnorm_dispersion]
        else:
            self.bnorm_dispersion_ = bnorm_dispersion
        
        tsdim = int(subspacedims*(subspacedims+1)/2)
        
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(1, self.temporal_filters_, kernel_size=(1,temp_cnn_kernel),
                            padding='same', padding_mode='reflect'),
            torch.nn.Conv2d(self.temporal_filters_, self.spatial_filters_,(self.nchannels_, 1)),
            torch.nn.Flatten(start_dim=2),
        ).to(self.device_)

        self.cov_pooling = torch.nn.Sequential(
            modules.CovariancePool(),
        )

        if self.bnorm_ == 'spdbn':
            self.spdbnorm = bn.AdaMomSPDBatchNorm((1,subspacedims,subspacedims), batchdim=0, 
                                          dispersion=self.bnorm_dispersion_, 
                                          learn_mean=False,learn_std=True,
                                          eta=1., eta_test=.1, dtype=torch.double, device=self.spd_device_)
        elif self.bnorm_ == 'brooks':
            self.spdbnorm = modules.BatchNormSPDBrooks((1,subspacedims,subspacedims), batchdim=0, dtype=torch.double, device=self.spd_device_)
        elif self.bnorm_ == 'tsbn':
            self.tsbnorm = bn.AdaMomBatchNorm((1, tsdim), batchdim=0, dispersion=self.bnorm_dispersion_, 
                                        eta=1., eta_test=.1, dtype=torch.double, device=self.spd_device_).to(self.device_)
        elif self.bnorm_ == 'spddsbn':
            self.spddsbnorm = bn.AdaMomDomainSPDBatchNorm((1,subspacedims,subspacedims), batchdim=0, 
                                domains=self.domains_,
                                learn_mean=False,learn_std=True, 
                                dispersion=self.bnorm_dispersion_, 
                                eta=1., eta_test=.1, dtype=torch.double, device=self.spd_device_)
        elif self.bnorm_ == 'tsdsbn':
            self.tsdsbnorm = bn.AdaMomDomainBatchNorm((1, tsdim), batchdim=0, 
                                domains=self.domains_,
                                dispersion=self.bnorm_dispersion_,
                                eta=1., eta_test=.1, dtype=torch.double).to(self.device_)
        elif self.bnorm_ is not None:
            raise NotImplementedError('requested undefined batch normalization method.')

        self.spdnet = torch.nn.Sequential(
            modules.BiMap((1,self.spatial_filters_,subspacedims), dtype=torch.double, device=self.spd_device_),
            modules.ReEig(threshold=1e-4),
        )
        self.logeig = torch.nn.Sequential(
            modules.LogEig(subspacedims),
            torch.nn.Flatten(start_dim=1),
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(tsdim,self.nclasses_).double(),
        ).to(self.spd_device_)

    def to(self, device: Optional[Union[int, torch.device]] = None, dtype: Optional[Union[int, torch.dtype]] = None, non_blocking: bool = False):
        if device is not None:
            self.device_ = device
            self.cnn.to(self.device_)
        return super().to(device=None, dtype=dtype, non_blocking=non_blocking)

    def forward(self, x, d, return_latent=True, return_prebn=False, return_postbn=False):
        out = ()
        h = self.cnn(x.to(device=self.device_)[:,None,...])
        C = self.cov_pooling(h).to(device=self.spd_device_, dtype=torch.double)
        l = self.spdnet(C)
        out += (l,) if return_prebn else ()
        l = self.spdbnorm(l) if hasattr(self, 'spdbnorm') else l
        l = self.spddsbnorm(l,d.to(device=self.spd_device_)) if hasattr(self, 'spddsbnorm') else l
        out += (l,) if return_postbn else ()
        l = self.logeig(l)
        l = self.tsbnorm(l) if hasattr(self, 'tsbnorm') else l
        l = self.tsdsbnorm(l,d) if hasattr(self, 'tsdsbnorm') else l
        out += (l,) if return_latent else ()
        y = self.classifier(l)
        out = y if len(out) == 0 else (y, *out[::-1])
        return out

    def domainadapt_finetune(self, x, y, d, target_domains):
        if hasattr(self, 'spddsbnorm'):
            self.spddsbnorm.set_test_stats_mode(bn.BatchNormTestStatsMode.REFIT)
        if hasattr(self, 'tsdsbnorm'):
            self.tsdsbnorm.set_test_stats_mode(bn.BatchNormTestStatsMode.REFIT)

        with torch.no_grad():
            for du in d.unique():
                self.forward(x[d==du], d[d==du])

        if hasattr(self, 'spddsbnorm'):
            self.spddsbnorm.set_test_stats_mode(bn.BatchNormTestStatsMode.BUFFER)
        if hasattr(self, 'tsdsbnorm'):
            self.tsdsbnorm.set_test_stats_mode(bn.BatchNormTestStatsMode.BUFFER)  

    def finetune(self, x, y, d):
        if hasattr(self, 'spdbnorm'):
            self.spdbnorm.set_test_stats_mode(bn.BatchNormTestStatsMode.REFIT)
        if hasattr(self, 'tsbnorm'):
            self.tsbnorm.set_test_stats_mode(bn.BatchNormTestStatsMode.REFIT)

        with torch.no_grad():
            self.forward(x, d)

        if hasattr(self, 'spdbnorm'):
            self.spdbnorm.set_test_stats_mode(bn.BatchNormTestStatsMode.BUFFER)
        if hasattr(self, 'tsbnorm'):
            self.tsbnorm.set_test_stats_mode(bn.BatchNormTestStatsMode.BUFFER)

    def compute_patterns(self, x, y, d):
        pass



class CNNNet(DomainAdaptFineTuneableModel, FineTuneableModel):
    def __init__(self, temporal_filters, spatial_filters = 40,
                 temp_cnn_kernel = 25,
                 bnorm : Optional[str] = 'bn', 
                 bnorm_dispersion : Union[str, bn.BatchNormDispersion] = bn.BatchNormDispersion.SCALAR,
                 **kwargs):
        super().__init__(**kwargs)

        self.temporal_filters_ = temporal_filters
        self.spatial_filters_ = spatial_filters
        self.bnorm_ = bnorm

        if isinstance(bnorm_dispersion, str):
            self.bnorm_dispersion_ = bn.BatchNormDispersion[bnorm_dispersion]
        else:
            self.bnorm_dispersion_ = bnorm_dispersion
        
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(1, self.temporal_filters_, kernel_size=(1,temp_cnn_kernel),
                            padding='same', padding_mode='reflect'),
            torch.nn.Conv2d(self.temporal_filters_, self.spatial_filters_,(self.nchannels_, 1)),
            torch.nn.Flatten(start_dim=2),
        ).to(self.device_)

        self.cov_pooling = torch.nn.Sequential(
            modules.CovariancePool(),
        )

        if self.bnorm_ == 'bn':
            self.bnorm = bn.AdaMomBatchNorm((1, self.spatial_filters_), batchdim=0, dispersion=self.bnorm_dispersion_, 
                                        eta=1., eta_test=.1).to(self.device_)
        elif self.bnorm_ == 'dsbn':
            self.dsbnorm = bn.AdaMomDomainBatchNorm((1, self.spatial_filters_), batchdim=0, 
                                domains=self.domains_,
                                dispersion=self.bnorm_dispersion_,
                                eta=1., eta_test=.1).to(self.device_)
        elif self.bnorm_ is not None:
            raise NotImplementedError('requested undefined batch normalization method.')

        self.logarithm = torch.nn.Sequential(
            modules.MyLog(),
            torch.nn.Flatten(start_dim=1),
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.spatial_filters_,self.nclasses_),
        ).to(self.device_)

    def forward(self, x, d, return_latent=True):
        out = ()
        h = self.cnn(x.to(device=self.device_)[:,None,...])
        C = self.cov_pooling(h)
        l = torch.diagonal(C, dim1=-2, dim2=-1)
        l = self.logarithm(l)
        l = self.bnorm(l) if hasattr(self, 'bnorm') else l
        l = self.dsbnorm(l,d) if hasattr(self, 'dsbnorm') else l
        out += (l,) if return_latent else ()
        y = self.classifier(l)
        out = y if len(out) == 0 else (y, *out[::-1])
        return out

    def domainadapt_finetune(self, x, y, d, target_domains):
        if hasattr(self, 'dsbnorm'):
            self.dsbnorm.set_test_stats_mode(bn.BatchNormTestStatsMode.REFIT)

        with torch.no_grad():
            for du in d.unique():
                self.forward(x[d==du], d[d==du])

        if hasattr(self, 'dsbnorm'):
            self.dsbnorm.set_test_stats_mode(bn.BatchNormTestStatsMode.BUFFER)  

    def finetune(self, x, y, d):
        if hasattr(self, 'bnorm'):
            self.bnorm.set_test_stats_mode(bn.BatchNormTestStatsMode.REFIT)

        with torch.no_grad():
            self.forward(x, d)

        if hasattr(self, 'bnorm'):
            self.bnorm.set_test_stats_mode(bn.BatchNormTestStatsMode.BUFFER)
