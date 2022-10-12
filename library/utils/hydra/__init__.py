import os
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, DictConfig
import torch

from sklearn.pipeline import Pipeline

def hydra_helpers(func):
    def inner(*args, **kwargs):
        # setup helpers

        # omega conf helpers
        OmegaConf.register_new_resolver("len", lambda x:len(x), replace=True)
        OmegaConf.register_new_resolver("add", lambda x,y:x+y, replace=True)
        OmegaConf.register_new_resolver("sub", lambda x,y:x-y, replace=True)
        OmegaConf.register_new_resolver("mul", lambda x,y:x*y, replace=True)
        OmegaConf.register_new_resolver("rdiv", lambda x,y:x/y, replace=True)

        STR2TORCHDTYPE = {
            'float32': torch.float32,
            'float64': torch.float64,
            'double': torch.double,
        }
        OmegaConf.register_new_resolver("torchdtype", lambda x:STR2TORCHDTYPE[x], replace=True)
        if func is not None:
            func(*args, **kwargs)
    return inner


def make_sklearn_pipeline(steps_config) -> Pipeline:

    steps = []
    for step_config in steps_config:

        # retrieve the name and parameter dictionary of the current steps
        step_name, step_transform = next(iter(step_config.items()))
        # instantiate the pipeline step, and append to the list of steps
        if isinstance(step_transform, DictConfig):
            pipeline_step = (step_name, hydra.utils.instantiate(step_transform, _convert_='partial'))
        else:
            pipeline_step = (step_name, step_transform)
        steps.append(pipeline_step)

    return Pipeline(steps)
