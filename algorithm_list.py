from train_configs import AlgoConfig
import torch.optim as optim
from torch_optimizer.adafactor import Adafactor
from torch_optimizer.adam_clip import AdamClip
from torch_optimizer.lion import Lion
from torch_optimizer.sgd_clip import SGDClip
from torch_optimizer.normalized_sgd import NormalizedSGD
from torch_optimizer.sign_sgd import SignSGD

# SGD machine translation
algorithm_list_sgd_machine_translation = list(map(lambda x: AlgoConfig(x[0], eps=x[1], device=x[2], **x[3]), [
    ('sgd',            0.1, 'cuda', {}),
    ('sgd',            1,   'cuda', {'coordinate_clip' : 0.1}),
    ('adam',           1,   'cuda', {}),
    ('adam',           10,  'cuda', {'coordinate_clip' : 0.5}),
    ('adam',           10,  'cuda', {'clip_update' : 0.5}),
    ('adafactor',      20,  'cuda', {}),
    ('adafactor',      200, 'cuda', {'coordinate_clip' : 0.5}),
    ('lion',           2,   'cuda', {}),
    ('lion',           2,   'cuda', {'clip_update' : 0.5}),
    ('sign_sgd',       1,   'cuda', {}),
    ('normalized_sgd', 1,   'cuda', {}),
    ('normalized_sgd', 1,   'cuda', {'coordinate_clip' : 0.1}),
]))

# Adam machine translation
algorithm_list_adam_machine_translation = list(map(lambda x: AlgoConfig(x[0], eps=x[1], device=x[2], **x[3]), [
    ('sgd',            0.1, 'cuda', {}),
    ('sgd',            1,   'cuda', {'coordinate_clip' : 0.1}),
    ('adam',           5,   'cuda', {}),
    ('adam',           80,  'cuda', {'coordinate_clip' : 0.5}),
    ('adam',           5,   'cuda', {'clip_update' : 0.5}),
    ('adafactor',      80,  'cuda', {}),
    ('adafactor',      400, 'cuda', {'coordinate_clip' : 0.5}),
    ('lion',           2,   'cuda', {}),
    ('lion',           2,   'cuda', {'clip_update' : 0.5}),
    ('sign_sgd',       1,   'cuda', {}),
    ('normalized_sgd', 1,   'cuda', {}),
    ('normalized_sgd', 1,   'cuda', {'coordinate_clip' : 0.1}),
]))

# Adam autoregressive
algorithm_list_adam_autoregressive = list(map(lambda x: AlgoConfig(x[0], eps=x[1], device=x[2], **x[3]), [
    ('sgd',            0.01, 'cpu', {}),
    ('adam',           0.25, 'cpu', {}),
    ('adam',           1,    'cpu', {'coordinate_clip' : 0.5}),
    ('adafactor',      0.1,  'cuda', {}),
    ('adafactor',      1,    'cuda', {'coordinate_clip' : 0.5}),
    ('lion',           0.1,  'cuda', {}),
]))

# SGD autoregressive
algorithm_list_sgd_autoregressive = list(map(lambda x: AlgoConfig(x[0], eps=x[1], device=x[2], **x[3]), [
    ('sgd',            0.01, 'cpu', {}),
    ('adam',           0.2,  'cpu', {}),
    ('adam',           1,    'cpu', {'coordinate_clip' : 0.5}),
    ('adafactor',      0.2,  'cuda', {}),
    ('adafactor',      1,    'cuda', {'coordinate_clip' : 0.5}),
    ('lion',           0.2,  'cuda', {}),
]))

algorithm_list_convergence_machine_translation = list(map(lambda x: AlgoConfig(x[0], optimizer_class=x[1], lr=x[2], **x[3]), [
    ('sgd',            optim.SGD,     1e-3, {}),
    ('sgd',            SGDClip,       5e-3, {'coordinate_clip' : 0.1}),
    ('adam',           optim.Adam,    2e-3, {}),
    ('adam',           AdamClip,      3e-3, {'coordinate_clip' : 0.5}),
    ('adafactor',      Adafactor,     1e-2, {'relative_step' : True}),
    ('adafactor',      Adafactor,     3e-2, {'coordinate_clip' : 0.5, 'relative_step' : False}),
    ('lion',           Lion,          2e-3, {}),
    ('sign_sgd',       SignSGD,       2e-3, {}),
    ('normalized_sgd', NormalizedSGD, 5e-4, {}),
    ('normalized_sgd', NormalizedSGD, 6e-4, {'coordinate_clip' : 0.1}),
]))

algorithm_list_convergence_autoregressive = list(map(lambda x: AlgoConfig(x[0], optimizer_class=x[1], lr=x[2], **x[3]), [
    ('sgd',            optim.SGD,     3e-5, {}),
    ('adam',           optim.Adam,    1e-4, {}),
    ('adam',           AdamClip,      1.5e-4, {'coordinate_clip' : 0.5}),
    ('adafactor',      Adafactor,     5e-4, {'relative_step' : False}),
    ('adafactor',      Adafactor,     5e-3, {'coordinate_clip' : 0.5, 'relative_step' : False}),
    ('lion',           Lion,          2e-4, {}),
]))

algorithm_lists = [
    algorithm_list_sgd_machine_translation,
    algorithm_list_adam_machine_translation,
    algorithm_list_sgd_autoregressive,
    algorithm_list_adam_autoregressive,
    algorithm_list_convergence_machine_translation,
    algorithm_list_convergence_autoregressive,
]

