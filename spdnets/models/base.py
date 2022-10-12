import torch
import torch.nn as nn


class BaseModel(nn.Module):
    def __init__(self, nclasses=None, nchannels=None, nsamples=None, nbands=None, device=None, input_shape=None):
        super().__init__()
        self.device_ = device
        self.lossfn = torch.nn.CrossEntropyLoss()
        self.nclasses_ = nclasses
        self.nchannels_ = nchannels
        self.nsamples_ = nsamples
        self.nbands_ = nbands
        self.input_shape_ = input_shape

    # AUXILIARY METHODS
    def calculate_classification_accuracy(self, Y, Y_lat):
        Y_hat = Y_lat.argmax(1)
        acc = Y_hat.eq(Y).float().mean().item()
        P_hat = torch.softmax(Y_lat, dim=1)
        return acc, P_hat

    def calculate_objective(self, model_pred, y_true, model_inp=None):
        # Y_lat, l = self(X.to(self.device_), B)
        if isinstance(model_pred, (list, tuple)):
            y_class_hat = model_pred[0]
        else:
            y_class_hat = model_pred
        loss = self.lossfn(y_class_hat, y_true.to(y_class_hat.device))
        return loss

    def get_hyperparameters(self):
        return dict(nchannels = self.nchannels_, 
                    nclasses=self.nclasses_, 
                    nsamples=self.nsamples_, 
                    nbands=self.nbands_)


class CPUModel:
    pass


class FineTuneableModel:
    def finetune(self, x, y, d):
        raise NotImplementedError()


class DomainAdaptBaseModel(BaseModel):
    def __init__(self, domains = [], **kwargs):
        super().__init__(**kwargs)
        self.domains_ = domains


class DomainAdaptFineTuneableModel(DomainAdaptBaseModel):
    def domainadapt_finetune(self, x, y, d, target_domains):
        raise NotImplementedError()


class DomainAdaptJointTrainableModel(DomainAdaptBaseModel):
    def calculate_objective(self, model_pred, y_true, model_inp=None):
        # filter out masked observations
        keep = y_true != -1 # special label

        if isinstance(model_pred, (list, tuple)):
            y_class_hat = model_pred[0]
        else:
            y_class_hat = model_pred

        return super().calculate_objective(y_class_hat[keep], y_true[keep], None)


class PatternInterpretableModel:
    def compute_patterns(self, x, y, d):
        raise NotImplementedError()
