# Toward Understanding Why Adam Converges Faster Than SGD for Transformers
Experiments for our paper on Adam vs SGD https://arxiv.org/abs/2306.00204

Still under construction, some key notes:
- Function for computing directional sharpness efficiently using `torch.autograd.functional.vhp`: https://github.com/panyan7/adam-vs-sgd/blob/main/landscape_utils.py#L498
- Implementation of pseudo-update steps: https://github.com/panyan7/adam-vs-sgd/blob/main/landscape_utils.py#L90C3-L319
- Function for visualizing the landscape in the update direction: https://github.com/panyan7/adam-vs-sgd/blob/main/landscape_utils.py#L322
