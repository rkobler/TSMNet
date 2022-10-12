from typing import Union
import torch
from .base import BaseModel, DomainAdaptFineTuneableModel
from .dann import DANNBase
import spdnets.batchnorm as bn
import spdnets.modules as modules


class ShallowConvNet(BaseModel):
    def __init__(self, spatial_filters = 40, temporal_filters = 40, pdrop = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.spatial_filters_ = spatial_filters
        self.temporal_filters_ = temporal_filters

        temp_cnn_kernel = 25
        temp_pool_kernel = 75
        temp_pool_stride = 15
        ntempconvout = int((self.nsamples_ - 1*(temp_cnn_kernel-1) - 1)/1 + 1)
        navgpoolout = int((ntempconvout - temp_pool_kernel)/temp_pool_stride + 1)

        self.bn = torch.nn.BatchNorm2d(self.spatial_filters_)
        drop = torch.nn.Dropout(p=pdrop)

        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(1,self.temporal_filters_,(1, temp_cnn_kernel)),
            torch.nn.Conv2d(self.temporal_filters_, self.spatial_filters_,(self.nchannels_, 1)),
        ).to(self.device_)
        self.pool = torch.nn.Sequential(
            modules.MySquare(),
            torch.nn.AvgPool2d(kernel_size=(1, temp_pool_kernel), stride=(1, temp_pool_stride)),
            modules.MyLog(),
            drop,
            torch.nn.Flatten(start_dim=1),
        ).to(self.device_)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.spatial_filters_ * navgpoolout, self.nclasses_),
        ).to(self.device_)
    
    def forward(self,x, d):
        l = self.cnn(x.to(self.device_)[:,None,...])
        l = self.bn(l)
        l = self.pool(l)
        y = self.classifier(l)
        return y, l


class DANNShallowConvNet(DANNBase, ShallowConvNet):
    """
    Domain adeversarial neural network (DANN) proposed for EEG MI classification
    by Ozdenizci et al. 2020, IEEE Access
    """
    def __init__(self, daloss_scaling = 0.05, dann_mode = 'ganin2016', **kwargs):
        kwargs['daloss_scaling'] = daloss_scaling
        kwargs['dann_mode'] = dann_mode
        super().__init__(**kwargs)

    def _ndim_latent(self):
        return self.classifier[-1].weight.shape[-1]

    def forward(self, x, d):
        y, l = ShallowConvNet.forward(self, x, d)
        y_domain = DANNBase.forward(self, l, d)
        return y, y_domain


class ShConvNetDSBN(ShallowConvNet, DomainAdaptFineTuneableModel):
    def __init__(self,
                 bnorm_dispersion : Union[str, bn.BatchNormDispersion] = bn.BatchNormDispersion.VECTOR, 
                 **kwargs):
        super().__init__(**kwargs)

        if isinstance(bnorm_dispersion, str):
            bnorm_dispersion = bn.BatchNormDispersion[bnorm_dispersion]

        self.bn = bn.AdaMomDomainBatchNorm((1, self.spatial_filters_, 1, 1), 
                                batchdim=[0,2,3], # same as batch norm 2D 
                                domains=self.domains_,
                                dispersion=bnorm_dispersion,
                                eta=1., eta_test=.1).to(self.device_)
    
    def forward(self,x, d):
        l = self.cnn(x.to(self.device_)[:,None,...])
        l = self.bn(l,d.to(device=self.device_))
        l = self.pool(l)
        y = self.classifier(l)
        return y, l

    def domainadapt_finetune(self, x, y, d, target_domains):
        self.bn.set_test_stats_mode(bn.BatchNormTestStatsMode.REFIT)
        for du in d.unique():
            self.forward(x[d==du], d[d==du])
        self.bn.set_test_stats_mode(bn.BatchNormTestStatsMode.BUFFER)
