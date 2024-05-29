# [OPT-2022] Toward Understanding Why Adam Converges Faster Than SGD for Transformers
Experiments for **Toward Understanding Why Adam Converges Faster Than SGD for Transformers [[ArXiv]](https://arxiv.org/abs/2306.00204)[[OPT-2022]](https://openreview.net/forum?id=Sf1NlV2r6PO)**.

## The Experiments
As explained in the paper, there are two main experiments. We explain them in the next two subsections.

### Landscape Visualization and Directional Sharpness
This experiments involves analysis of the update direction of different optimization algorithms.
The definition of directional sharpness in the direction $v$ for some unit vector $v$ is $\langle v, \nabla^2f(x_t)v\rangle$, which is explained in the paper.
In this part, we use `torch.autograd.functional.vhp` to compute the directional sharpness efficiently, as explained in [`landscape_utils.py`](https://github.com/panyan7/adam-vs-sgd/blob/main/landscape_utils.py#L498).

To ensure a fair comparison of different optimization algorithms, we consider the same optimization trajectory and use pseudo-update steps to compute the directions for different optimization algorithms. This is explained in [`landscape_utils.py`](https://github.com/panyan7/adam-vs-sgd/blob/main/landscape_utils.py#L90C3-L319).

We visualize the landscape in the update direction using a grid search, as explained in [`landscape_utils.py`](https://github.com/panyan7/adam-vs-sgd/blob/main/landscape_utils.py#L322).

### Convergence of Optimization Algorithms
There are nothing special to note. The implementation of the clipped version of optimization algorithms can be found in `/torch_optimizer`. The implementation is based on the Python library [pytorch-optimizer](https://pytorch-optimizer.readthedocs.io/en/latest/). We implemented the clipping ourselves.

### Hyperparameter Configurations
The hyperparameter configurations can be found in `algorithm_list.py`.

## Reproducing the Experiments
Run `$ python run_convergence.py` and `$ python run_landscape.py`. You would need a [Huggingface key](https://huggingface.co/docs/hub/en/security-tokens) to access [the stack](https://huggingface.co/datasets/bigcode/the-stack) dataset.
