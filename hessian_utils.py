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
import gc
import resource



def debug_memory():
    print('maxrss = {}'.format(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    tensors = collections.Counter((str(o.device), o.dtype, tuple(o.shape))
                                  for o in gc.get_objects()
                                  if torch.is_tensor(o))
    for line in tensors.items():
        print('{}\t{}'.format(*line))



def compute_hessian(model, batch, parameter_name, image_name):
    names = list(n for n, _ in model.named_parameters())
    index = names.index(parameter_name)
    params = list(model.parameters())
    shape = int(params[index].view(-1).shape[0])

    def f(param):
        input_ids, labels = batch
        loss = functional_call(model,
                               {n: p for n, p in zip(names, params[:index] + [param] + params[index+1:])},
                               (input_ids, None, None, None, None, None, labels))[0]
        return loss

    hessian = torch.autograd.functional.hessian(f, params[index])

    hessian = hessian.view((shape, shape)).cpu().detach()
    plt.figure(figsize=(100,100))
    plt.imshow(hessian)
    plt.colorbar()
    plt.title(image_name)
    plt.savefig('hessian/' + image_name + '_' + parameter_name + '.png')
    return hessian



def compute_parameter_size(model):
    # Compute parameter size
    parameter_size = {}
    for n, _ in model.named_parameters():
        parameter_size[n] = int(_.view(-1).shape[0])
    return parameter_size



def diagonal_hessian(model, batch, image_name, num_samples=1e-4, max_parameter_size=6e5):
    '''
    Compute the diagonal of the Hessian matrix with randomly sampled rows.
    '''
    parameter_size = compute_parameter_size(model)

    grads_2_all = []
    input_ids, labels = batch
    out = model(input_ids=input_ids, labels=labels)

    for n, p in tqdm(model.named_parameters()):
        if parameter_size[n] > max_parameter_size:
            continue
        grads_1_out = torch.autograd.grad(out[0], p, create_graph=True, retain_graph=True)
        grads_1 = grads_1_out[0].view(-1)
        indices = torch.multinomial(torch.ones_like(grads_1), num_samples=max(int(num_samples * grads_1.shape[0]), 1))
        for i in indices.tolist():
            grads_2_out = torch.autograd.grad(grads_1[i], p, create_graph=True, retain_graph=True)
            grads_2 = grads_2_out[0].view(-1)
            grads_2_all.append(abs(float(grads_2[i])))
            torch.cuda.empty_cache()

            del grads_2, grads_2_out

        torch.cuda.empty_cache()
        del grads_1, grads_1_out, indices

    torch.cuda.empty_cache()
    del out

    grads_2_all.sort()

    # Plot the result
    plt.figure()
    plt.scatter(range(len(grads_2_all)), grads_2_all)
    title = image_name
    mean = float(sum(grads_2_all) / len(grads_2_all))
    plt.axhline(y=mean, color='r', linestyle='-')
    plt.title(title)
    plt.savefig('hessian/' + image_name + '_' + str(num_samples) + '.png')

    with open('hessian/' + image_name + '_' + str(num_samples) + '.pkl', 'wb') as f:
        pickle.dump(grads_2_all, f)

    print(f'Finish computing diagonal hessian of {image_name}')



def sample_hessian(model, epoch, batch_id, batch_data):
    parameter_size = compute_parameter_size(model)

    # First store the gradients
    grads = [p.grad.clone().detach() for p in model.parameters()]

    # Sample random indices
    params = list(model.named_parameters())
    params = [(n, p) for n, p in model.named_parameters() \
              if 100000 <= parameter_size[n] <= 600000 and 'attention' in n and 'weight' in n]

    for n, p in tqdm(params):
        indices = torch.multinomial(torch.ones_like(p.view(-1)),
                                    num_samples=max(int(1e-4 * p.view(-1).shape[0]), 1)).tolist()
        grads_2_all = [0.0 for _ in range(len(indices))]
        for batch in batch_data:
            input_ids, labels = batch
            out = model(input_ids=input_ids, labels=labels)

            grads_1_out = torch.autograd.grad(out[0], p, create_graph=True, retain_graph=True)
            grads_1 = grads_1_out[0].view(-1)
            for i in range(len(indices)):
                j = indices[i]
                grads_2_out = torch.autograd.grad(grads_1[j], p, create_graph=True, retain_graph=True)
                grads_2 = grads_2_out[0].view(-1)
                grads_2_all[i] += float(grads_2[j])
                torch.cuda.empty_cache()

                del grads_2_out
            del grads_1_out

        model.zero_grad()

        grads_2_all = list(map(abs, grads_2_all))

        plt.figure(figsize=(10,10))
        plt.scatter(range(len(grads_2_all)), grads_2_all)
        image_name = f'adam_{n}_{epoch}_{batch_id}'
        title = image_name
        mean = float(sum(grads_2_all) / len(grads_2_all))
        title += '\nmean: ' + str(mean)
        title += '\nsum: ' + str(float(sum(grads_2_all)))
        title += '\nmax: ' + str(float(max(grads_2_all)))
        title += '\nmax/mean: ' + str(float(max(grads_2_all) * len(grads_2_all) / sum(grads_2_all)))
        title += '\nsum <= mean: ' + str(float(sum(map(lambda x : x if x <= mean else 0, grads_2_all))))
        title += '\nsum <= 10*mean: ' + str(float(sum(map(lambda x : x if x <= 10 * mean else 0, grads_2_all))))
        plt.axhline(y=mean, color='r', linestyle='-')
        plt.title(n)
        plt.savefig('hessian/' + image_name + '.png')

        with open('hessian/' + image_name + '.pkl', 'wb') as f:
            pickle.dump(grads_2_all, f)

        model.zero_grad()

    for g, p in zip(grads, list(model.parameters())):
        p.grad = g



def gauss_newton_hessian(model,
                         epoch,
                         batch_id,
                         batch_data,
                         lb_parameter_size=1e5,
                         ub_parameter_size=6e5,
                         keyword='attention',
                         check_sol=False,
                         check_fraction=0.02):
    parameter_size = compute_parameter_size(model)

    softmax = nn.Softmax(dim=1)
    G = []
    params = [p for n, p in model.named_parameters() \
              if lb_parameter_size <= parameter_size[n] <= ub_parameter_size and keyword in n]

    for batch in batch_data:
        model.zero_grad()
        input_ids, labels = batch
        out = model(input_ids=input_ids, labels=labels)
        logits = out[1]
        p = softmax(logits).T.clone().detach().reshape(-1)
        scale = torch.sqrt(p[0] * p[1])

        grad_all = []
        for i in range(2):
            grad = torch.autograd.grad(logits[0][i], params, retain_graph=True)
            grad_flatten = [p.view(-1).clone().detach() for p in grad]
            grad_flatten = torch.cat(grad_flatten, dim=-1)
            model.zero_grad()
            grad_all.append(grad_flatten)
        g = scale * (grad_all[0] - grad_all[1])
        G.append(g.cpu().clone().detach())
        del p, g, grad, grad_flatten, grad_all, out, scale
    G = torch.vstack(G)
    G_c = G.cuda()
    norms = torch.linalg.norm(G_c, ord=2, dim=0)
    image_name = f'GGN_{epoch}_{batch_id}'
    plt.figure()
    plt.hist(norms.cpu(), bins=100)
    plt.title(image_name)
    plt.savefig('hessian/' + image_name + '.png')

    # Spectral norm of Hessian
    sp_norm = torch.linalg.norm(G, ord=2) ** 2

    # Remove coordinates greater than 2 * mean and compute spectral norm of Hessian
    G_r = G[:, norms <= 2 * torch.mean(norms)]
    sp_norm_r = torch.linalg.norm(G_r, ord=2) ** 2

    if check_sol:
        # Sample random index to check solution
        indices = np.random.choice(G.shape[1], size=int((1 - check_fraction) * G.shape[1]), replace=False)
        G_random = G[:, indices]
        sp_norm_random = torch.linalg.norm(G_random, ord=2) ** 2
        del G_random

    print(f'Smoothness: {sp_norm}')
    print(f'Robust smoothness: {sp_norm_r}')
    fraction_removed = 1 - G_r.shape[1] / G.shape[1]
    print(f'Fraction removed: {fraction_removed}')
    if check_sol:
        print(f'Random smoothness: {sp_norm_random}')

    del G, G_r, G_c
    return sp_norm, sp_norm_r, fraction_removed

