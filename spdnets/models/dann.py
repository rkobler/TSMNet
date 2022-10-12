import torch
from .base import DomainAdaptJointTrainableModel
import spdnets.modules as modules

class DANNBase(DomainAdaptJointTrainableModel):
    """
    Domain adeversarial neural network (DANN) proposed
    by Ganin et al. 2016, JMLR
    """
    def __init__(self, daloss_scaling = 1., dann_mode = 'ganin2016', **kwargs):
        domains = kwargs['domains']
        assert (domains.dtype == torch.long)
        kwargs['domains'] = domains.sort()[0]
        super().__init__(**kwargs)
        self.dann_mode_ = dann_mode

        if self.dann_mode_ == 'ganin2015':
            grad_reversal_scaling = daloss_scaling
            self.daloss_scaling_ = 1.
        elif self.dann_mode_ == 'ganin2016':
            grad_reversal_scaling = 1.
            self.daloss_scaling_ = daloss_scaling
        else:
            raise NotImplementedError()

        ndim_latent = self._ndim_latent()
        self.adversary_loss = torch.nn.CrossEntropyLoss()

        self.adversary = torch.nn.Sequential(
            torch.nn.Flatten(start_dim=1),
            modules.ReverseGradient(scaling=grad_reversal_scaling),
            torch.nn.Linear(ndim_latent, len(self.domains_))
        ).to(self.device_)

    def _ndim_latent(self):
        raise NotImplementedError()

    def forward(self, l, d):
        # super().forward()
        # h = self.cnn(x[:,None,...]).flatten(start_dim=1)
        # y = self.classifier(h)
        y_domain = self.adversary(l)
        return y_domain

    def domainadapt(self, x, y, d, target_domain):
        pass # domain adaptation is done during the training process

    def calculate_objective(self, model_pred, y_true, model_inp):
        loss = super().calculate_objective(model_pred, y_true, model_inp)
        domain = model_inp['d']
        y_dom_hat = model_pred[1]
        # check if all requested domains were declared
        assert ((self.domains_[..., None] == domain[None,...]).any(dim=0).all())
        # assign to the class indices (buckets)
        y_dom = torch.bucketize(domain, self.domains_).to(y_dom_hat.device)
        
        adversarial_loss = self.adversary_loss(y_dom_hat, y_dom)
        loss = loss + self.daloss_scaling_ * adversarial_loss

        return loss