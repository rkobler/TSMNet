import torch
from .base import BaseModel
from .dann import DANNBase
import spdnets.modules as modules


class EEGNetv4(BaseModel):
    def __init__(self, is_within = False, srate = 128, f1 = 8, d = 2, **kwargs):
        super().__init__(**kwargs)
        self.is_within_ = is_within
        self.srate_ = srate
        self.f1_ = f1
        self.d_ = d
        self.f2_ = self.f1_ * self.d_
        momentum = 0.01

        kernel_length = int(self.srate_ // 2)
        nlatsamples_time = self.nsamples_ // 32

        temp2_kernel_length = int(self.srate_ // 2 // 4)

        if self.is_within_:
            drop_prob = 0.5
        else:
            drop_prob = 0.25

        bntemp = torch.nn.BatchNorm2d(self.f1_, momentum=momentum, affine=True, eps=1e-3)
        bnspat = torch.nn.BatchNorm2d(self.f1_ * self.d_, momentum=momentum, affine=True, eps=1e-3)

        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(1,self.f1_,(1, kernel_length), bias=False, padding='same'),
            bntemp,
            modules.Conv2dWithNormConstraint(self.f1_, self.f1_ * self.d_, (self.nchannels_, 1), max_norm=1,
                stride=1, bias=False, groups=self.f1_, padding=(0, 0)),
            bnspat,
            torch.nn.ELU(),
            torch.nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4)),
            torch.nn.Dropout(p=drop_prob),
            torch.nn.Conv2d(self.f1_ * self.d_, self.f1_ * self.d_, (1, temp2_kernel_length),
                stride=1, bias=False, groups=self.f1_ * self.d_, padding='same'),
            torch.nn.Conv2d(self.f1_ * self.d_, self.f2_, (1, 1),
                stride=1, bias=False, padding=(0, 0)),
            torch.nn.BatchNorm2d(self.f2_, momentum=momentum, affine=True, eps=1e-3),
            torch.nn.ELU(),
            torch.nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8)),
            torch.nn.Dropout(p=drop_prob),
        ).to(self.device_)

        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(start_dim=1),
            modules.LinearWithNormConstraint(self.f2_ * nlatsamples_time, self.nclasses_, max_norm=0.25)
        ).to(self.device_)

    def get_hyperparameters(self):
        kwargs = super().get_hyperparameters()
        kwargs['nsamples'] = self.nsamples_
        kwargs['is_within_subject'] = self.is_within_subject_
        kwargs['srate'] = self.srate_
        kwargs['f1'] = self.f1_
        kwargs['d'] = self.d_
        return kwargs        

    def forward(self, x, d):
        l = self.cnn(x[:,None,...])
        y = self.classifier(l)
        return y, l


class DANNEEGNet(DANNBase, EEGNetv4):
    """
    Domain adeversarial neural network (DANN) proposed for EEG MI classification
    by Ozdenizci et al. 2020, IEEE Access
    """
    def __init__(self, daloss_scaling = 0.03, dann_mode = 'ganin2016', **kwargs):
        kwargs['daloss_scaling'] = daloss_scaling
        kwargs['dann_mode'] = dann_mode
        super().__init__(**kwargs)

    def _ndim_latent(self):
        return self.classifier[-1].weight.shape[-1]

    def forward(self, x, d):
        y, l = EEGNetv4.forward(self, x, d)
        y_domain = DANNBase.forward(self, l, d)
        return y, y_domain