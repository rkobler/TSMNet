import logging
import torch
from skorch.classifier import NeuralNetClassifier
from skorch.callbacks.logging import EpochTimer, PrintLog
from skorch.callbacks.scoring import EpochScoring, PassthroughScoring

from spdnets.models import BaseModel, DomainAdaptFineTuneableModel, FineTuneableModel

from .logging import TrainLog

log = logging.getLogger(__name__)

class DomainAdaptNeuralNetClassifier(NeuralNetClassifier):
    def __init__(self, module, *args, criterion=torch.nn.CrossEntropyLoss, **kwargs):
        super().__init__(module, *args, criterion=criterion, **kwargs)

    @property
    def _default_callbacks(self):
        return [
            ('epoch_timer', EpochTimer()),
            ('train_loss', PassthroughScoring(
                name='train_loss',
                on_train=True,
            )),
            ('valid_loss', PassthroughScoring(
                name='valid_loss',
            )),
            ('print_log', TrainLog()),
        ]

    def get_loss(self, mdl_pred, y_true, X=None, **kwargs):
        if isinstance(self.module_, BaseModel):
            return self.module_.calculate_objective(mdl_pred, y_true, X)
        elif isinstance(mdl_pred, (list, tuple)):
            y_hat = mdl_pred[0]
        else:
            y_hat = mdl_pred
        return self.criterion_(y_hat, y_true.to(y_hat.device))

    def domainadapt_finetune(self, x: torch.Tensor, y: torch.Tensor, d : torch.Tensor, target_domains=None):
        if isinstance(self.module_, DomainAdaptFineTuneableModel):
            self.module_.domainadapt_finetune(x=x.to(self.device), y=y, d=d, target_domains=target_domains)
        else:
            log.info("Model does not support domain adapt fine tuning.")

    def finetune(self, x: torch.Tensor, y: torch.Tensor, d : torch.Tensor):
        if isinstance(self.module_, FineTuneableModel):
            self.module_.finetune(x=x.to(self.device), y=y, d=d)
        else:
            log.info("Model does not support fine-tuning.")
