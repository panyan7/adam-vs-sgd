from landscape_utils import (AdamParams,
                             SGDParams,
                             NormalizedSGDParams,
                             SignSGDParams,
                             AdafactorParams,
                             LionParams)
from torch_optimizer.adafactor import Adafactor
from torch_optimizer.adam_clip import AdamClip
from torch_optimizer.sgd_clip import SGDClip
from torch_optimizer.lion import Lion
from collections import defaultdict


class AlgoConfig:
    def __init__(self, name, eps=100, optimizer_class=None, lr=None, device='cpu', **kwargs):
        assert type(kwargs) is dict
        self.name = name
        self.eps = eps
        self.kwargs = kwargs
        self.device = device
        self.optimizer_class = optimizer_class
        self.lr = lr

    def __str__(self):
        name = self.name
        for key, val in self.kwargs.items():
            if key == 'coordinate_clip':
                name += f', clip grad {val}'
            if key == 'threshold_scale':
                name += f', scale {val}'
            if key == 'clip_update':
                name += f', clip update {val}'
        return name


optim_param_type = {
    'adam' : AdamParams,
    'adam_clip' : AdamParams,
    'sgd' : SGDParams,
    'sgd_clip' : SGDParams,
    'normalized_sgd' : NormalizedSGDParams,
    'normalized_sgd_clip' : NormalizedSGDParams,
    'sign_sgd' : SignSGDParams,
    'sign_sgd_clip' : SignSGDParams,
    'adafactor' : AdafactorParams,
    'adafactor_clip' : AdafactorParams,
    'lion' : LionParams,
    'lion_clip' : LionParams,
    'lion_no_coup' : LionParams,
}

test_eps_machine_translation = {
    'adam' : 1,
    'adam_clip' : 5,
    'sgd' : 0.1,
    'sgd_clip' : 1,
    'normalized_sgd' : 1,
    'normalized_sgd_clip' : 1,
    'sign_sgd' : 1,
    'sign_sgd_clip' : 1,
    'adafactor' : 20,
    'adafactor_clip' : 150,
    'lion' : 2,
    'lion_clip' : 2,
    'lion_no_coup' : 2,
}

test_eps_masked_language_modeling = {
    'adam' : 0.05,
    'adam_clip' : 0.25,
    'sgd' : 0.01,
    'sgd_clip' : 0.05,
    'normalized_sgd' : 0.01,
    'normalized_sgd_clip' : 0.05,
    'sign_sgd' : 0.05,
    'sign_sgd_clip' : 0.05,
    'adafactor' : 0.25,
    'adafactor_clip' : 1.25,
    'lion' : 0.1,
    'lion_clip' : 0.1,
    'lion_no_coup' : 0.1,
}

update_kwargs = defaultdict(dict)
update_kwargs['lion_no_coup'] = {'beta1' : 0.9, 'beta2' : 0.9}

test_epochs = [2, 5, 10, 20]

thresholds = [40, 400, 4000]
