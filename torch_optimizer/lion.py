# Copyright 2023 Google Research. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""PyTorch implementation of the Lion optimizer."""
import torch
from torch.optim.optimizer import Optimizer


class Lion(Optimizer):
    r"""Implements Lion algorithm."""

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0,
                 coordinate_clip=0.0, threshold_scale=1.0):
        """Initialize the hyperparameters.
        Args:
            params (iterable): iterable of parameters to optimize or dicts defining
                parameter groups
            lr (float, optional): learning rate (default: 1e-4)
            betas (Tuple[float, float], optional): coefficients used for computing
                running averages of gradient and its square (default: (0.9, 0.99))
            weight_decay (float, optional): weight decay coefficient (default: 0)
        """
        print("Initialized Lion")

        if not 0.0 <= lr:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError('Invalid beta parameter at index 0: {}'.format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError('Invalid beta parameter at index 1: {}'.format(betas[1]))
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        self.coordinate_clip = coordinate_clip
        self.threshold_scale = threshold_scale
        super().__init__(params, defaults)

    def _get_top_coordinates_threshold(self, grads, coordinate_clip=0):
        """
        Find the adaptive clipping threshold for top coordinates.
        """
        if type(grads) is dict:
            grads = [p for n, p in grads.items()]
        all_grad = torch.hstack(list(map(lambda x: x.reshape(-1), grads))).detach()
        all_grad_abs = torch.abs(all_grad)
        all_grad_abs, _ = all_grad_abs.sort(descending=True)
        threshold = float(all_grad_abs[int(all_grad_abs.shape[0] * coordinate_clip)])
        del all_grad, all_grad_abs
        return threshold

    def _clip_threshold(self, grads, threshold, threshold_scale=1):
        """
        Clip with respect to a fixed threshold.
        grads: List[torch.Tensor] or Dict
        """
        threshold *= threshold_scale
        if type(grads) is list:
            grads = [p * (p.abs() < threshold) \
                     + torch.sign(p) * (p.abs() >= threshold) * threshold for p in grads]
        elif type(grads) is dict:
            grads = {n : p * (p.abs() < threshold) \
                     + torch.sign(p) * (p.abs() >= threshold) * threshold \
                     for n, p in grads.items()}
        else:
            grads = grads * (grads.abs() < threshold) \
                     + torch.sign(grads) * (grads.abs() >= threshold) * threshold
        return grads

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        Returns:
            the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            grads = [p.grad.data for p in group['params']]
            coordinate_clip_threshold = self._get_top_coordinates_threshold(grads, self.coordinate_clip)
            for p in group['params']:
                if p.grad is None:
                    continue

                # Perform stepweight decay
                p.data.mul_(1 - group['lr'] * group['weight_decay'])

                grad = p.grad
                grad_clip = self._clip_threshold(grad, coordinate_clip_threshold, threshold_scale=self.threshold_scale)
                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p)

                exp_avg = state['exp_avg']
                beta1, beta2 = group['betas']

                # Weight update
                update = exp_avg * beta1 + grad_clip * (1 - beta1)
                p.add_(torch.sign(update), alpha=-group['lr'])
                # Decay the momentum running average coefficient
                exp_avg.mul_(beta2).add_(grad_clip, alpha=1 - beta2)

        return loss
