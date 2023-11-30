import torch
import torch.nn as nn
import torch.optim as optim
from torch.func import functional_call
import numpy as np
import matplotlib.pyplot as plt
from transformers import BertConfig, BertTokenizer, BertForSequenceClassification
from datasets import load_dataset
from itertools import accumulate
from tqdm.notebook import tqdm
import pickle
import collections
from collections import defaultdict
import gc
import resource
import sys
import os


def _get_top_coordinates_threshold(grads, coordinate_clip=0):
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


def clip_threshold(grads, threshold, threshold_scale=1):
    """
    Clip with respect to a fixed threshold.
    grads: List[torch.Tensor] or Dict
    """
    threshold *= threshold_scale
    if type(grads) is list:
        grads_clip = [p * (p.abs() < threshold) \
                      + torch.sign(p) * (p.abs() >= threshold) * threshold \
                      for p in grads]
    elif type(grads) is dict:
        grads_clip = {n : p * (p.abs() < threshold) \
                      + torch.sign(p) * (p.abs() >= threshold) * threshold \
                      for n, p in grads.items()}
    return grads_clip


def clip_top_coordinates(grads, coordinate_clip=0, threshold_scale=1):
    """
    Clip with respect to the top coordinates.
    grads: List[torch.Tensor] or Dict
    """
    if coordinate_clip == 0:
        return grads

    if coordinate_clip == 1:
        threshold = 0
    else:
        threshold = _get_top_coordinates_threshold(grads, coordinate_clip=coordinate_clip)
    return clip_threshold(grads, threshold, threshold_scale=threshold_scale)


class OptimParams:
    def __init__(self, model, device='cpu'):
        self.params = defaultdict(dict)
        self.param_names = []
        self.device = device
        for param_name, _ in model.named_parameters():
            self.params[param_name]['num_iters'] = 1
            self.param_names.append(param_name)

    def update(self, grads, weights, coordinate_clip=0):
        """
        Computes the pseudo momentum update.
        grads: dict[torch.Tensor] that maps param names to gradients
        weights: dict[torch.Tensor] that maps param names to weights
        """
        pass

    def get_update_step(self, device='cuda'):
        """
        Returns the update step of the optimizer.
        """
        pass


class AdamParams(OptimParams):
    def __init__(self, model, device='cpu'):
        super().__init__(model, device=device)
        self.clip_update = 0
        for param_name, param in model.named_parameters():
            self.params[param_name]['exp_avg'] = torch.zeros_like(param).to(self.device)
            self.params[param_name]['exp_avg_sq'] = torch.zeros_like(param).to(self.device)

    def update(self, grads, weights, beta1=0.9, beta2=0.999, coordinate_clip=0,
               threshold_scale=1, clip_update=0):
        """
        Requires: grads on self.device
        """
        grads_clip = clip_top_coordinates(grads, coordinate_clip=coordinate_clip,
                                          threshold_scale=threshold_scale)
        self.clip_update = clip_update
        for param_name in self.param_names:
            self.params[param_name]['exp_avg'] = beta1 * self.params[param_name]['exp_avg'] \
                + (1 - beta1) * grads_clip[param_name]
            self.params[param_name]['exp_avg_sq'] = beta2 * self.params[param_name]['exp_avg_sq'] \
                + (1 - beta2) * (grads[param_name] ** 2)
            self.params[param_name]['num_iters'] += 1
        del grads_clip

    def get_update_step(self, beta1=0.9, beta2=0.999, eps=1e-8, device='cuda'):
        """
        device: str, device of return tensor
        """
        update_step = []
        for param_name in self.param_names:
            num_iters = self.params[param_name]['num_iters']
            exp_avg = self.params[param_name]['exp_avg']
            exp_avg = exp_avg / (1 - beta1 ** num_iters)
            exp_avg_sq = self.params[param_name]['exp_avg_sq']
            exp_avg_sq = exp_avg_sq / (1 - beta2 ** num_iters)
            update = exp_avg / (exp_avg_sq ** 0.5 + eps)
            update_step.append(-update.to(device))
        update_step_clip = clip_top_coordinates(update_step, coordinate_clip=self.clip_update)
        return update_step_clip


class SGDParams(OptimParams):
    def __init__(self, model, device='cpu'):
        super().__init__(model, device=device)
        for param_name, param in model.named_parameters():
            self.params[param_name]['exp_avg'] = torch.zeros_like(param).cuda()

    def update(self, grads, weights, momentum=0.9, coordinate_clip=0,
               threshold_scale=1):
        grads_clip = clip_top_coordinates(grads, coordinate_clip=coordinate_clip,
                                          threshold_scale=threshold_scale)
        for param_name in self.param_names:
            if self.params[param_name]['num_iters'] > 1:
                self.params[param_name]['exp_avg'] = momentum * self.params[param_name]['exp_avg'] \
                    + (1 - momentum) * grads_clip[param_name]
            else:
                self.params[param_name]['exp_avg'] = grads_clip[param_name]
            self.params[param_name]['num_iters'] += 1
        del grads_clip

    def get_update_step(self, device='cuda'):
        update_step = []
        for param_name in self.param_names:
            exp_avg = self.params[param_name]['exp_avg']
            update_step.append(-exp_avg.to(device))
        return update_step


class NormalizedSGDParams(SGDParams):
    def get_update_step(self, device='cuda'):
        sgd_update_step = super().get_update_step(device=device)
        update_step = [p / p.norm() * (torch.numel(p) ** 0.5) for p in sgd_update_step]
        del sgd_update_step
        return update_step


class SignSGDParams(SGDParams):
    def get_update_step(self, device='cuda'):
        sgd_update_step = super().get_update_step(device=device)
        update_step = [torch.sign(p) for p in sgd_update_step]
        del sgd_update_step
        return update_step


class AdafactorParams(OptimParams):
    def __init__(self, model, device='cpu'):
        super().__init__(model, device=device)
        for param_name, param in model.named_parameters():
            if int(param.dim()) == 2:
                self.params[param_name]['r'] = torch.zeros(param.shape[0]).to(self.device)
                self.params[param_name]['c'] = torch.zeros(param.shape[1]).to(self.device)
            else:
                self.params[param_name]['exp_avg_sq'] = torch.zeros_like(param).to(self.device)

    def _RMS(self, x):
        return x.norm(2) / (x.numel() ** 0.5)

    def update(self, grads, weights, eps1=1e-30, eps2=1e-3, d=1,
               coordinate_clip=0, threshold_scale=1):
        """
        Requires: grads on self.device
        """
        self.params_update_step = []
        grads_clip = clip_top_coordinates(grads, coordinate_clip=coordinate_clip,
                                          threshold_scale=threshold_scale)

        for param_name in self.param_names:
            grad = grads[param_name].to(self.device)

            beta2 = 1 - (self.params[param_name]['num_iters'] ** (-0.8))
            rho = min(1e-2, self.params[param_name]['num_iters'] ** (-0.5))
            self.params[param_name]['alpha'] = max(eps2, self._RMS(weights[param_name])) * rho
            if int(weights[param_name].dim()) == 2:
                self.params[param_name]['r'] = beta2 * self.params[param_name]['r'] \
                    + (1 - beta2) * torch.sum(grad ** 2 + eps1, dim=1)
                self.params[param_name]['c'] = beta2 * self.params[param_name]['c'] \
                    + (1 - beta2) * torch.sum(grad ** 2 + eps1, dim=0)
                v = torch.matmul(self.params[param_name]['r'].view((-1, 1)),
                                 self.params[param_name]['c'].view((1, -1)))
                v = v / torch.sum(self.params[param_name]['r'])
            else:
                self.params[param_name]['exp_avg_sq'] = beta2 * self.params[param_name]['exp_avg_sq'] \
                    + (1 - beta2) * (grad ** 2 + eps1)
                v = self.params[param_name]['exp_avg_sq']

            grad = grads_clip[param_name].to(self.device)

            self.params[param_name]['u'] = grad / (v ** 0.5)
            self.params[param_name]['u'] /= max(1, self._RMS(self.params[param_name]['u']) / d)
            self.params[param_name]['u'] *= self.params[param_name]['alpha']
            self.params[param_name]['num_iters'] += 1
        del grads_clip

    def get_update_step(self, device='cuda'):
        update_step = []
        for param_name in self.param_names:
            update = self.params[param_name]['u']
            update_step.append(-update.to(device))
        return update_step

class LionParams(OptimParams):
    def __init__(self, model, device='cpu'):
        super().__init__(model, device=device)
        for param_name, param in model.named_parameters():
            self.params[param_name]['exp_avg'] = torch.zeros_like(param).to(self.device)
            self.params[param_name]['update'] = torch.zeros_like(param).to(self.device)

    def update(self, grads, weights, beta1=0.9, beta2=0.99, coordinate_clip=0,
               threshold_scale=1, clip_update=1):
        """
        Requires: grads on self.device
        """
        grads_clip = clip_top_coordinates(grads, coordinate_clip=coordinate_clip,
                                          threshold_scale=threshold_scale)
        update_step = {}
        for param_name in self.param_names:
            grad = grads_clip[param_name].to(self.device)
            """
            if self.params[param_name]['num_iters'] > 1:
                exp_avg = self.params[param_name]['exp_avg']
                update = beta1 * exp_avg + (1 - beta1) * grads[param_name]
                exp_avg = beta2 * exp_avg + (1 - beta2) * grads[param_name]
            else:
                update = grads[param_name]
                exp_avg = grads[param_name]
            """
            exp_avg = self.params[param_name]['exp_avg']
            update = beta1 * exp_avg + (1 - beta1) * grad
            exp_avg = beta2 * exp_avg + (1 - beta2) * grad
            if clip_update == 1:
                update = torch.sign(update)
            update_step[param_name] = update
            self.params[param_name]['exp_avg'] = exp_avg
            self.params[param_name]['num_iters'] += 1

        if clip_update < 1:
            update_step = clip_top_coordinates(update_step, coordinate_clip=clip_update)

        for param_name in self.param_names:
            self.params[param_name]['update'] = update_step[param_name]
        del grads_clip

    def get_update_step(self, device='cuda'):
        update_step = []
        for param_name in self.param_names:
            update = self.params[param_name]['update']
            update_step.append(-update.to(device))
        return update_step


class LionFullParams(LionParams):
    def update(self, grads, weights, beta1=0.9, beta2=1.11, coordinate_clip=0,
               threshold_scale=1):
        grads = clip_top_coordinates(grads, coordinate_clip=coordinate_clip,
                                     threshold_scale=threshold_scale)
        for param_name in self.param_names:
            grad = grads[param_name]
            grad = torch.arcsin(grad)
            if self.params[param_name]['num_iters'] > 1:
                exp_avg = beta1 * self.params[param_name]['exp_avg2'] + (1 - beta1) * grad
                exp_avg2 = beta2 * exp_avg + (1 - beta2) * grad
            else:
                exp_avg = grad
                exp_avg2 = grad
            update = torch.sign(exp_avg)
            weight_decay = weights[param_name] * 0.4602
            update = update + weight_decay
            exp_avg = torch.cosh(update)
            self.params[param_name]['exp_avg'] = exp_avg
            self.params[param_name]['exp_avg2'] = exp_avg2
            self.params[param_name]['update'] = update

    def get_update_step(self, device='cuda'):
        update_step = []
        for param_name in self.param_names:
            update = self.params[param_name]['update']
            update_step.append(-update.to(device))
        return update_step


def get_params(model, device='cpu', return_dict=False):
    if return_dict:
        return {n : p.data.to(device).clone().detach() for n, p in model.named_parameters()}
    return [p.data.to(device).clone().detach() for p in model.parameters()]


def get_grads(model, device='cpu', return_dict=False):
    if return_dict:
        return {n : p.grad.to(device).clone().detach() for n, p in model.named_parameters()}
    return [p.grad.to(device).clone().detach() for p in model.parameters()]


def search(model,
           grads,
           algo_params,
           batch_data,
           title,
           algorithm,
           eps=1,
           distance=20,
           device='cuda',
           image=False):
    if not os.path.exists('output/' + title):
        os.mkdir('output/' + title)

    weight_names = [n for n, p in model.named_parameters()]
    params = get_params(model, device=device)

    update_step = algo_params.get_update_step(device=device)

    all_grad = torch.hstack(list(map(lambda x: x.reshape(-1), update_step))).detach()
    all_grad_norm = float(all_grad.norm())

    for p, p_old in zip(model.parameters(), params):
        p_data = p.data
        p.data = p_old.clone().detach()
        del p_data

    lr_list = []
    loss_list = []
    for i in range(int(distance)+1):
        batch_loss = 0.0
        lr = i * eps
        for p, p_step in zip(model.parameters(), update_step):
            p.data += lr * p_step / all_grad_norm
        torch.cuda.empty_cache()

        model.eval()
        with torch.no_grad():
            for batch in batch_data:
                input_ids, labels = batch
                if image:
                    out = model(input_ids)
                    criterion = nn.CrossEntropyLoss()
                    loss1 = criterion(out, labels)
                else:
                    out = model(input_ids=input_ids, labels=labels)
                    loss1 = out[0]
                batch_loss += float(loss1) / len(batch_data)
                del input_ids, labels
                torch.cuda.empty_cache()

        for p, p_old in zip(model.parameters(), params):
            p_data = p.data
            p.data = p_old.clone().detach()
            del p_data
        torch.cuda.empty_cache()

        lr_list.append(lr)
        loss_list.append(batch_loss)
        if batch_loss > loss_list[0]:
            break

    torch.cuda.empty_cache()

    if len(loss_list) <= 3:
        del update_step, params, all_grad
        torch.cuda.empty_cache()
        return search(model,
                      grads,
                      algo_params,
                      batch_data,
                      title,
                      algorithm,
                      eps=eps/5,
                      distance=distance,
                      device=device,
                      image=image)

    plt.figure()
    plt.plot(lr_list, loss_list)
    best_index = np.argmin(loss_list)
    # plt.title(title + f', Best Distance {lr_list[best_index]}')
    plt.title(title)
    plt.scatter(lr_list[best_index], loss_list[best_index], c='r')
    plt.savefig('output/' + title + '/' + algorithm + '.pdf')
    plt.close()
    with open('output/' + title + '/' + algorithm + '_loss_list.pkl', 'wb') as f:
        pickle.dump((loss_list, eps), f)

    text_info = open('output/' + title + '/' + algorithm + '_summary.txt', 'w')
    text_info.write(algorithm + '\n')

    grad_shape = int(all_grad.shape[0])
    # calculate l2 norm
    text_info.write(str(float(all_grad.norm(2))) + '\n')
    # calculate l infinity norm
    text_info.write(str(float(all_grad.norm(float('inf')))) + '\n')

    # histogram
    plt.figure(figsize=(10,6))
    all_grad_abs = torch.abs(all_grad)
    all_grad_abs = all_grad_abs + 1e-15
    min_val = 1e-15
    max_val = 10
    logbins = np.logspace(np.log10(min_val), np.log10(max_val), 30)
    logbins = torch.tensor(logbins).float()
    hist, bins = torch.histogram(all_grad_abs.cpu(), bins=logbins)
    plt.hist(logbins[:-1], bins=logbins, weights=hist, histtype='step')
    plt.xscale('log')
    plt.title(title)
    plt.savefig('output/' + title + '/' + algorithm + '_hist.pdf')
    plt.close()

    text_info.write('Mean of update step: ' + str(float(torch.mean(torch.abs(all_grad)))) + '\n')
    text_info.write('Max of update step: ' + str(float(torch.max(torch.abs(all_grad)))) + '\n')

    del hist, logbins

    for p, p_old in zip(model.parameters(), params):
        p_data = p.data
        p.data = p_old.clone().detach()
        del p_data

    text_info.close()
    del update_step, params, all_grad, all_grad_abs
    torch.cuda.empty_cache()

    return loss_list[0] - loss_list[best_index]


color_list = {
    'adam' : 'tab:blue',
    'adam_clip' : 'tab:blue',
    'sgd' : 'tab:orange',
    'sgd_clip' : 'tab:orange',
    'sign_sgd' : 'tab:cyan',
    'sign_sgd_clip' : 'tab:cyan',
    'normalized_sgd' : 'tab:green',
    'normalized_sgd_clip' : 'tab:green',
    'adafactor' : 'tab:red',
    'adafactor_clip' : 'tab:red',
    'lion' : 'tab:purple',
    'lion_clip' : 'tab:purple',
}


def compare_landscape(title, algorithm_list, thresholds=[40,400,4000]):
    for i,threshold in enumerate(thresholds + [float('inf')]):
        plt.figure()
        for algorithm in algorithm_list:
            with open('output/' + title + '/' + str(algorithm) + '_loss_list.pkl', 'rb') as f:
                loss_list, eps = pickle.load(f)
            lr_list = [i * eps for i in range(len(loss_list))]
            if max(lr_list) > threshold:
                continue

            if 'grad' in str(algorithm):
                alpha = 0.6
            elif 'update' in str(algorithm):
                alpha = 0.3
            else:
                alpha = 0.9

            plt.plot(lr_list, loss_list, label=str(algorithm),
                     c=color_list[algorithm.name], alpha=alpha)
            best_index = np.argmin(loss_list)
            plt.scatter(lr_list[best_index], loss_list[best_index],
                        c=color_list[algorithm.name], alpha=alpha)

        plt.title(title)
        plt.xlabel('Step Size')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('output/' + title + f'/compare{i+1}.pdf')
        plt.close()


def compute_sharpness(model, algo_params, batch_data, device='cuda', hessian_shift=0, image=False):
    """
    Computes the sharpness of algorithm
    """
    param_len = len(algo_params.param_names)
    names = list([n for n, _ in model.named_parameters()])[:param_len]

    update_step = algo_params.get_update_step(device=device)

    all_grad = torch.hstack(list(map(lambda x: x.reshape(-1), update_step))).detach()
    v_norm = float(all_grad.norm())
    total_sharpness = 0.0
    model.eval()

    for batch in batch_data:
        input_ids, labels = batch

        if image:
            def f(*params):
                params = dict(zip(names, params))
                out = functional_call(model, params, (input_ids), {})
                criterion = nn.CrossEntropyLoss()
                loss = criterion(out, labels)
                return loss
        else:
            def f(*params):
                params = dict(zip(names, params))
                loss = functional_call(model, params, (),
                                       {'input_ids' : input_ids,
                                        'labels' : labels})[0]
                return loss

        params = tuple(get_params(model, device=device)[:param_len])
        if hessian_shift > 0:
            params = [p + hessian_shift * u / v_norm for p, u in zip(list(params), update_step)]
        output, vhp = torch.autograd.functional.vhp(f, tuple(params),
                                                    tuple(update_step))
        sharpness = float(sum([torch.dot(vh.reshape(-1), v.reshape(-1)) \
                               for vh, v in zip(vhp, update_step)]))
        sharpness /= (v_norm ** 2)
        total_sharpness += sharpness / len(batch_data)
        torch.cuda.empty_cache()
        del output, vhp

    torch.cuda.empty_cache()
    del all_grad, update_step
    return total_sharpness


def check_sharpness(model, optimizer, batch_data):
    """
    Checks whether the sharpness computed is correct for small models
    This function computes the full Hessian, so model should be sufficiently small
    """
    names = list(n for n, _ in model.named_parameters())
    update_step = optimizer.get_update_step()
    v_norm = float(torch.sqrt(sum([g.norm() ** 2 for g in update_step])))
    total_sharpness = 0.0
    total_sharpness2 = 0.0
    lengths = [int(p.view(-1).shape[0]) for p in tuple(model.parameters())[1:]]
    plengths = list(accumulate(lengths))
    model.zero_grad()
    model.eval()
    parameters = [p.data.clone().detach() for p in model.parameters()]
    for p in parameters:
        p.requires_grad = True

    loss = 0.0
    for batch in tqdm(batch_data):
        model.zero_grad()
        input_ids, labels = batch

        def f(*params):
            loss = functional_call(model,
                                   {n: p for n, p in zip(names, list(params))},
                                   (),
                                   {'input_ids' : input_ids, 'labels' : labels})[0]
            return loss

        output, vhp = torch.autograd.functional.vhp(f, tuple(parameters[1:]),
                                                    tuple(update_step)[1:])
        loss += float(output) / len(batch_data)

        hessian = torch.autograd.functional.hessian(f, tuple(parameters[1:]))
        final_hessian = torch.zeros((sum(lengths), sum(lengths))).cuda()

        vhp_test = [torch.zeros_like(vh) for vh in vhp]
        for i in range(len(hessian)):
            for j in range(len(hessian)):
                hessian_matrix = hessian[i][j].view(lengths[i], lengths[j])
                final_hessian[plengths[i]-lengths[i]:plengths[i],
                              plengths[j]-lengths[j]:plengths[j]] = hessian_matrix
                vhp_test[j] += torch.matmul(update_step[i+1].view(-1),
                                            hessian[i][j].view(lengths[i], lengths[j])
                                            ).view(vhp_test[j].shape)

        all_grad = torch.hstack(list(map(lambda x: x.reshape(-1),
                                         update_step[1:]))).cuda().detach()
        sharpness = float(sum([torch.dot(vh.reshape(-1), v.reshape(-1)) \
                               for vh, v in zip(vhp, update_step[1:])]))
        sharpness2 = float(torch.matmul(all_grad, torch.matmul(final_hessian, all_grad).view(-1)))

        sharpness /= (v_norm ** 2)
        sharpness2 /= (v_norm ** 2)

        print(sharpness, sharpness2) # should be close

        total_sharpness += sharpness / len(batch_data)
        total_sharpness2 += sharpness2 / len(batch_data)
        del output, vhp, final_hessian, hessian
    print(total_sharpness, total_sharpness2) # should be close
    print(loss) # should be close to the loss computed in search


def plot_ratio_list(algorithm_list, ratio_list, title):
    plt.figure()
    for (algorithm, algo_ratio_list) in zip(algorithm_list, ratio_list):
        plt.scatter(range(len(algo_ratio_list)), algo_ratio_list, label=str(algorithm))
    plt.legend()
    plt.title(title)
    plt.savefig('output/' + title + '_ratio.pdf')
    plt.close()


def plot_step_list(algorithm_list, best_step_list, title):
    plt.figure()
    for (algorithm, algo_best_step_list) in zip(algorithm_list, best_step_list):
        plt.scatter(range(len(algo_best_step_list)), algo_best_step_list, label=str(algorithm))
    plt.legend()
    plt.title(title)
    plt.savefig('output/' + title + '_best_step.pdf')
    plt.close()


def compare_top_diff(sgd_update_step_old,
                     sgd_update_step_new,
                     model,
                     title,
                     coordinate_clip=0):
    if not os.path.exists('output/' + title):
        os.mkdir('output/' + title)

    names = [n for n, p in model.named_parameters()]

    threshold_old = _get_top_coordinates_threshold(sgd_update_step_old,
                                                   coordinate_clip=coordinate_clip)
    threshold_new = _get_top_coordinates_threshold(sgd_update_step_new,
                                                   coordinate_clip=coordinate_clip)

    text_info = open('output/' + title + '/top_diff.txt', 'w')
    for n, p_old, p_new in zip(names, sgd_update_step_old, sgd_update_step_new):
        p_old_count = int(torch.sum(p_old.view(-1) >= threshold_old))
        p_new_count = int(torch.sum(p_new.view(-1) >= threshold_new))
        p_same_count = int(torch.sum((p_old.view(-1) >= threshold_old) \
                                     * (p_new.view(-1) >= threshold_new)))
        text_info.write(f'{n}: Old {p_old_count}, New {p_new_count},'
                        f'Same {p_same_count} / {int(p_old.view(-1).shape[0])}\n')

    text_info.close()


def compute_sharpness_masked(model, optimizer, batch_data, title, mask):
    if not os.path.exists('output/' + title):
        os.mkdir('output/' + title)

    names = list(n for n, _ in model.named_parameters())
    sgd_update_step = optimizer.get_update_step()
    v_norm = float(torch.sqrt(sum([g.norm() ** 2 for g in sgd_update_step])))
    sgd_update_step_masked = [p * (1 - p_m) for p, p_m in zip(sgd_update_step, mask)]
    v_norm_masked = float(torch.sqrt(sum([g.norm() ** 2 for g in sgd_update_step_masked])))
    total_sharpness = 0.0
    total_sharpness_masked = 0.0
    model.eval()

    for batch in batch_data:
        input_ids, labels = batch

        def f(*params):
            loss = functional_call(model, params, (),
                                   {'input_ids' : input_ids, 'labels' : labels})[0]
            return loss

        output, vhp = torch.autograd.functional.vhp(f, dict(model.named_parameters()),
                                                    tuple(sgd_update_step))
        sharpness = float(sum([torch.dot(vh.reshape(-1), v.reshape(-1)) \
                               for vh, v in zip(vhp, sgd_update_step)]))
        sharpness /= (v_norm ** 2)
        total_sharpness += sharpness / len(batch_data)

        output_masked, vhp_masked = torch.autograd.functional.vhp(f, tuple(model.parameters()),
                                                                  tuple(sgd_update_step_masked))
        sharpness_masked = float(sum([torch.dot(vh.reshape(-1), v.reshape(-1)) \
                                      for vh, v in zip(vhp, sgd_update_step_masked)]))
        sharpness_masked /= (v_norm_masked ** 2)
        total_sharpness_masked += sharpness_masked / len(batch_data)
        del output, vhp
        del output_masked, vhp_masked

    text_info = open('output/' + title + '/sgd_sharpness_masked.txt', 'w')
    text_info.write('SGD\n')
    text_info.write(f'SGD Sharpness: {total_sharpness}\n')
    text_info.write(f'SGD Top Sharpness: {total_sharpness_masked}\n')
    text_info.write(f'Masked Norm: {v_norm_masked}\n')
    text_info.write('Masked Norm / Norm:'
                    f'{(v_norm_masked ** 2) / (v_norm ** 2)}\n')
    text_info.write('Top Sharpness / Sharpness:'
                    f'{total_sharpness_masked / total_sharpness}\n')
    print(f'SGD Sharpness: {total_sharpness}')
    print(f'SGD Masked Sharpness: {total_sharpness_masked}')
    print(f'Masked Norm: {v_norm_masked}')
    print('Masked Norm / Norm:'
          f'{(v_norm_masked ** 2) / (v_norm ** 2)}')
    print('Masked Sharpness / Sharpness:'
          f'{total_sharpness_masked / total_sharpness}')

    for n, p in zip(names, mask):
        text_info.write(f'{n}: {int(torch.sum(p.view(-1)))} / {int(p.view(-1).shape[0])}\n')

    text_info.close()

    del sgd_update_step, sgd_update_step_masked
    return total_sharpness_masked / total_sharpness


def init_mask(model):
    mask = [torch.ones_like(p) for p in model.parameters()]
    return mask


def compute_top_mask(optimizer, mask, coordinate_clip=0):
    sgd_update_step = optimizer.get_update_step()

    all_grad = torch.hstack(list(map(lambda x: x.reshape(-1), sgd_update_step))).detach()
    all_grad_abs = torch.abs(all_grad)
    all_grad_abs, _ = all_grad_abs.sort(descending=True)
    threshold = float(all_grad_abs[int(all_grad_abs.shape[0] * coordinate_clip)])

    mask = [(p.abs() >= threshold) * p_mask for p, p_mask in zip(sgd_update_step, mask)]
    return mask


def plot_loss_list(loss_lists, algorithm_list, title, fmt='pdf'):
    plt.figure(figsize=[9.6, 5.4])
    for loss_list, algorithm in zip(loss_lists, algorithm_list):
        if 'grad' in str(algorithm):
            alpha = 0.6
        elif 'update' in str(algorithm):
            alpha = 0.3
        else:
            alpha = 0.9
        plt.plot(range(len(loss_list)), loss_list, label=str(algorithm),
                 c=color_list[algorithm.name], alpha=alpha)

    plt.title(title)
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('output/loss.' + fmt)
    plt.close()


