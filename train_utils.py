import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from transformers import (BertConfig,
                          BertTokenizer,
                          BertForSequenceClassification)
from datasets import load_dataset
from itertools import accumulate
from tqdm.notebook import tqdm
from torch_optimizer.adafactor import Adafactor
from torch_optimizer.adam_clip import AdamClip
from torch_optimizer.lion import Lion
from torch_optimizer.sgd_clip import SGDClip
from torch_optimizer.normalized_sgd import NormalizedSGD
from torch_optimizer.sign_sgd import SignSGD
import pickle
import collections
import gc
import resource
import sys
import os
import random
from collections import defaultdict
from landscape_utils import (search,
                             compare_landscape,
                             compute_sharpness,
                             check_sharpness,
                             plot_step_list,
                             clip_top_coordinates,
                             plot_loss_list,
                             get_params,
                             get_grads)
from landscape_utils import *
from hessian_utils import sample_hessian, gauss_newton_hessian
from dataset_utils import load_model, load_dataloader
from train_configs import (test_eps_machine_translation,
                           test_eps_masked_language_modeling,
                           optim_param_type,
                           update_kwargs,
                           test_epochs,
                           sample_iterations,
                           AlgoConfig)


def train(model,
          dataloader,
          num_epochs,
          lr,
          grad_accumulation,
          algorithm_list,
          task,
          write_log=True,
          save_model=False,
          momentum=False,
          start_epoch=0,
          train_optimizer='sgd',
          use_test_batch=False,
          test_batch_size=0,
          check_sharpness_correctness=False,
          skip_search=False,
          skip_sharpness=False,
          test_epochs=test_epochs,
          num_sample=1,
          num_sample_sharpness=5,
          sample_batch=False,
          sample_sharpness=False,
          thresholds=[40,400,4000],
          sample_iterations=None,
          print_memory_summary=False,
          device='cuda',
          optimizer_device='cpu'):
    if write_log:
        log_file = open('log.txt', 'w')
        log_file.write(f'lr = {lr}')

    algo_params_dict = {}

    if start_epoch != 0:
        with open(f'model/algo_params_dict.pkl', 'rb') as f:
            algo_params_dict = pickle.load(f)
        model = torch.load('model/model.pt')
        model.train()
    else:
        algo_params_dict = {}
        for algorithm in algorithm_list:
            algorithm_name = algorithm.name
            optim_param_class = optim_param_type[algorithm_name]
            algo_params_dict[str(algorithm)] = optim_param_class(model, device=algorithm.device)

    if train_optimizer == 'sgd':
        if momentum:
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, dampening=0.9)
        else:
            optimizer = optim.SGD(model.parameters(), lr=lr)
    elif train_optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif train_optimizer == 'adafactor':
        optimizer = Adafactor(model.parameters(), lr=lr, relative_step=False)

    if start_epoch != 0:
        optimizer.load_state_dict(torch.load('model/optimizer.pth'))
        # optimizer.state = defaultdict(dict, optimizer.state)

    loss_list = []

    if task == 'dbg':
        task = 'machine_translation'

    if use_test_batch:
        if start_epoch == 0:
            test_batch = []
            test_batch_sample = random.sample(list(range(len(dataloader))), k=test_batch_size)

            for i, batch in enumerate(tqdm(dataloader)):
                if i in test_batch_sample:
                    test_batch.append(batch)
            torch.save(test_batch, f'output/test_batch.pt')
            print(f"Test batch: {len(test_batch)}")
        else:
            test_batch = torch.load(f'output/test_batch.pt')

    for epoch in range(start_epoch, num_epochs):
        loss = 0.0
        num_calc = 0
        torch.cuda.empty_cache()
        model.train()

        if sample_batch:
            batch_sample = random.sample(list(range(len(dataloader) // grad_accumulation)),
                                         k=num_sample)
        else:
            batch_sample = [int(len(dataloader) // grad_accumulation) // 2]

        if sample_sharpness:
            batch_sample_sharpness = random.sample(list(range(len(dataloader) // grad_accumulation)),
                                                   k=num_sample_sharpness)
        else:
            batch_sample_sharpness = [int(len(dataloader) // grad_accumulation) // 2]

        with open(f'output/{epoch+1}_batch_sample_sharpness.pkl', 'wb') as f:
            pickle.dump(batch_sample_sharpness, f)

        best_step_list = [[] for _ in range(len(algorithm_list))]
        sharpness_list = {str(algorithm) : [] for algorithm in algorithm_list}
        for i, batch in enumerate(tqdm(dataloader)):
            input_ids, labels = batch
            if task == 'image_classification':
                out = model(input_ids)
                criterion = nn.CrossEntropyLoss()
                loss1 = criterion(out, labels)
            else:
                out = model(input_ids=input_ids, labels=labels)
                loss1 = out[0]
            loss += float(loss1) / len(dataloader)
            loss1 /= grad_accumulation

            if i % grad_accumulation == 0:
                batch_loss = 0.0
                batch_data = []

            batch_data.append(batch)

            loss1.backward(retain_graph=True)

            batch_loss += float(loss1)

            if (i+1) % grad_accumulation == 0 or i+1 == len(dataloader):
                batch_idx = ((i+1) // grad_accumulation) + epoch * (len(dataloader) // grad_accumulation)

                if sample_iterations is not None:
                    title = f'Iteration {batch_idx}'
                else:
                    title = f'Epoch {epoch+1}, Batch {(i+1) // grad_accumulation}'

                params_list = get_params(model, device=device, return_dict=False)
                params = get_params(model, device=optimizer_device, return_dict=True)
                grads_list = get_grads(model, device=device, return_dict=False)
                grads = get_grads(model, device=optimizer_device, return_dict=True)

                for algorithm in algorithm_list:
                    kwargs = algorithm.kwargs
                    algo_params_dict[str(algorithm)].update(grads, params, **kwargs)

                if sample_iterations is not None:
                    cur_sample = batch_idx in sample_iterations
                else:
                    cur_sample = (epoch+1) in test_epochs and (i // grad_accumulation) in batch_sample_sharpness

                if cur_sample:
                    if skip_sharpness:
                        batch_id = (i+1) // grad_accumulation
                        torch.save(model, f'output/{epoch+1}_{batch_id}_model.pt')
                        torch.save(grads_list, f'output/{epoch+1}_{batch_id}_grads_list.pt')
                        with open(f'output/{epoch+1}_{batch_id}_algo_params_dict.pkl', 'wb') as f:
                            pickle.dump(algo_params_dict, f)
                    else:
                        print("Computing Sharpness")
                        for algorithm in algorithm_list:
                            sharpness = compute_sharpness(model,
                                                          algo_params_dict[str(algorithm)],
                                                          test_batch,
                                                          image=(task=='image_classification'))
                            sharpness_list[str(algorithm)].append(sharpness)
                            for p, g_old in zip(model.parameters(), grads_list):
                                p_data = p.grad
                                p.grad = g_old.clone().detach()
                                del p_data

                            for p, p_old in zip(model.parameters(), params_list):
                                p_data = p.data
                                p.data = p_old.clone().detach()
                                del p_data

                            if print_memory_summary:
                                print(torch.cuda.memory_summary(abbreviated=True))
                            torch.cuda.empty_cache()

                        if sample_iterations is not None:
                            for algorithm in algorithm_list:
                                print(str(algorithm), sharpness_list[str(algorithm)][-1] / abs(sharpness_list['sgd'][-1]))

                if sample_iterations is not None:
                    cur_sample = batch_idx in sample_iterations
                else:
                    cur_sample = (epoch+1) in test_epochs and (i // grad_accumulation) in batch_sample

                if cur_sample:
                    print('\n' + title)

                    if check_sharpness_correctness:
                        check_sharpness(algo_params_dict['sgd'], batch_data, title)

                    test_batch = test_batch if use_test_batch else batch_data

                    if skip_search:
                        batch_id = (i+1) // grad_accumulation
                        torch.save(model, f'output/{epoch+1}_{batch_id}_model.pt')
                        torch.save(grads, f'output/{epoch+1}_{batch_id}_grads.pt')
                        with open(f'output/{epoch+1}_{batch_id}_algo_params_dict.pkl', 'wb') as f:
                            pickle.dump(algo_params_dict, f)
                    else:
                        print("Visualizing Landscape")
                        for j, algorithm in enumerate(algorithm_list):
                            best_step = search(model,
                                               grads,
                                               algo_params_dict[str(algorithm)],
                                               test_batch,
                                               title,
                                               str(algorithm),
                                               eps=algorithm.eps,
                                               image=(task=='image_classification'))
                            for p, p_old in zip(model.parameters(), params_list):
                                p_data = p.data
                                p.data = p_old.clone().detach()
                                del p_data

                            best_step_list[j].append(best_step)

                            if print_memory_summary:
                                print(torch.cuda.memory_summary(abbreviated=True))
                            torch.cuda.empty_cache()

                        for p, p_old in zip(model.parameters(), params_list):
                            p_data = p.data
                            p.data = p_old.clone().detach()
                            del p_data

                        compare_landscape(title, algorithm_list, thresholds=thresholds)

                    model.zero_grad()
                    for p, g_old in zip(model.parameters(), grads_list):
                        p_data = p.grad
                        p.grad = g_old.clone().detach()
                        del p_data
                    for p, p_old in zip(model.parameters(), params_list):
                        p_data = p.data
                        p.data = p_old.clone().detach()
                        del p_data
                    model.train()

                    if print_memory_summary:
                        print(torch.cuda.memory_summary(abbreviated=True))
                    torch.cuda.empty_cache()

                del grads, grads_list, params, params_list

                optimizer.step()
                optimizer.zero_grad()

                del batch_data

                torch.cuda.empty_cache()

            del input_ids, labels, out

            torch.cuda.empty_cache()

        if sample_iterations is None:
            if not skip_sharpness and (epoch+1) in test_epochs and num_sample_sharpness > 0:
                with open(f'output/sharpness_{epoch+1}.pkl', 'wb') as f:
                    pickle.dump(sharpness_list, f)
                for k, v in sharpness_list.items():
                    vs = []
                    for i in range(len(v)):
                        vs.append(v[i] / abs(sharpness_list['sgd'][i]))
                    print(k, sum(vs) / len(vs))

            print(f'Epoch {epoch+1}, loss {loss}')
            if write_log:
                log_file.write(f'Epoch {epoch+1}, loss {loss}\n')

        loss_list.append(loss)

        if (epoch+1) in test_epochs:
            plot_step_list(algorithm_list, best_step_list, f'Epoch {epoch+1}')

        if save_model:
            torch.save(model, f'model/model_{epoch+1}.pt')
            epoch_file = open('model/epoch.txt', 'w')
            epoch_file.write(str(epoch+1) + '\n')
            epoch_file.close()

    with open(f'model/algo_params_dict.pkl', 'wb') as f:
        pickle.dump(algo_params_dict, f)
    torch.save(model, 'model/model.pt')
    torch.save(optimizer.state_dict(), 'model/optimizer.pth')

    del optimizer

    return loss_list


def train_one_optimizer(model,
                        dataloader,
                        num_epochs,
                        grad_accumulation,
                        algorithm,
                        num_epoch_to_save=1):
    """
    algorithm : AlgoConfig
    """
    OptimizerClass = algorithm.optimizer_class
    lr = algorithm.lr
    kwargs = algorithm.kwargs
    optimizer = OptimizerClass(model.parameters(), lr=lr, **kwargs)

    model.train()
    loss_list = []
    for epoch in range(num_epochs):
        loss = 0.0
        for i, batch in enumerate(tqdm(dataloader)):
            input_ids, labels = batch
            out = model(input_ids=input_ids, labels=labels)
            loss1 = out[0]
            loss += float(loss1) / len(dataloader)
            loss1 /= grad_accumulation

            if i % grad_accumulation == 0:
                batch_loss = 0.0

            loss1.backward(retain_graph=True)

            batch_loss += float(loss1)

            if (i+1) % grad_accumulation == 0 or i+1 == len(dataloader):
                optimizer.step()
                optimizer.zero_grad()

                torch.cuda.empty_cache()

            del input_ids, labels, out

            torch.cuda.empty_cache()

        loss_list.append(loss)

        print(f'Epoch {epoch+1}, loss {loss}')

        if (epoch+1) % num_epoch_to_save == 0:
            torch.save(model, f'model/model.pt')

    del optimizer

    return loss_list


def train_optimizers(task,
                     num_epochs,
                     grad_accumulation,
                     algorithm_list,
                     num_epoch_to_save=1):
    loss_lists = []
    for algorithm in algorithm_list:
        print(str(algorithm))
        model = load_model(task)
        dataloader = load_dataloader(task)
        loss_list = train_one_optimizer(model,
                                        dataloader,
                                        num_epochs,
                                        grad_accumulation,
                                        algorithm,
                                        num_epoch_to_save=num_epoch_to_save)
        print(loss_list)
        loss_lists.append(loss_list)
        with open(f'output/loss_lists.pkl', 'wb') as f:
            pickle.dump(loss_lists, f)
    plot_loss_list(loss_lists, algorithm_list, f'Convergence on {task}', fmt='pdf')
    return loss_lists


def compute_saved_model_landscape(algorithm_list,
                                  test_epochs=test_epochs,
                                  batch_sample=[6],
                                  thresholds=[40,400,4000],
                                  device='cuda'):
    test_batch = torch.load(f'output/test_batch.pt')
    for epoch in test_epochs:
        for batch_id in batch_sample:
            model = torch.load(f'output/{epoch}_{batch_id}_model.pt')
            grads = torch.load(f'output/{epoch}_{batch_id}_grads.pt')
            params_list = get_params(model, device=device, return_dict=False)
            with open(f'output/{epoch}_{batch_id}_algo_params_dict.pkl', 'rb') as f:
                algo_params_dict = pickle.load(f)
            title = f'Epoch {epoch}, Batch {batch_id}'
            for j, algorithm in enumerate(algorithm_list):
                best_step = search(model,
                                   grads,
                                   algo_params_dict[str(algorithm)],
                                   test_batch,
                                   title,
                                   str(algorithm),
                                   eps=algorithm.eps)
                for p, p_old in zip(model.parameters(), params_list):
                    p_data = p.data
                    p.data = p_old.clone().detach()
                    del p_data

            for p, p_old in zip(model.parameters(), params_list):
                p_data = p.data
                p.data = p_old.clone().detach()
                del p_data

            compare_landscape(title, algorithm_list, thresholds=thresholds)


def compute_saved_model_sharpness(algorithm_list,
                                  test_epochs=test_epochs,
                                  device='cuda',
                                  hessian_shift=0):
    test_batch = torch.load(f'output/test_batch.pt')
    for epoch in test_epochs:
        with open(f'output/{epoch}_batch_sample_sharpness.pkl', 'rb') as f:
            batch_sample_sharpness = pickle.load(f)
        sharpness_list = {str(algorithm) : [] for algorithm in algorithm_list}
        print(f'Epoch {epoch}')
        for batch_id in batch_sample_sharpness:
            batch_id = batch_id + 1
            title = f'Epoch {epoch}, Batch {batch_id}'
            model = torch.load(f'output/{epoch}_{batch_id}_model.pt')
            grads_list = torch.load(f'output/{epoch}_{batch_id}_grads_list.pt')
            params_list = get_params(model, device=device, return_dict=False)
            with open(f'output/{epoch}_{batch_id}_algo_params_dict.pkl', 'rb') as f:
                algo_params_dict = pickle.load(f)
            for algorithm in algorithm_list:
                sharpness = compute_sharpness(model,
                                              algo_params_dict[str(algorithm)],
                                              test_batch,
                                              hessian_shift=hessian_shift)
                sharpness_list[str(algorithm)].append(sharpness)
                for p, g_old in zip(model.parameters(), grads_list):
                    p_data = p.grad
                    p.grad = g_old.clone().detach()
                    del p_data

                for p, p_old in zip(model.parameters(), params_list):
                    p_data = p.data
                    p.data = p_old.clone().detach()
                    del p_data

        with open(f'output/sharpness_{epoch}.pkl', 'wb') as f:
            pickle.dump(sharpness_list, f)
        for k, v in sharpness_list.items():
            vs = []
            for i in range(len(v)):
                vs.append(v[i] / abs(sharpness_list['sgd'][i]))
            print(k, sum(vs) / len(vs))


def train_robust_smoothness(model, dataloader, num_epochs, lr, grad_accumulation, threshold=2.5, write_log=True, test_batch_size=0):
    if write_log:
        log_file = open('log_adam.txt', 'w')
        log_file.write(f'lr = {lr}')
        log_file.write(f'threshold = {threshold}')

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_list = []
    adam_norm_list = []

    test_batch = []
    test_batch_sample = random.sample(list(range(len(dataloader))), k=test_batch_size)

    for i, batch in enumerate(tqdm(dataloader)):
        if i in test_batch_sample:
            test_batch.append(batch)

    for epoch in range(num_epochs):
        loss = 0.0
        total_norm = 0.0
        norms = []
        num_calc = 0
        for i, batch in enumerate(tqdm(dataloader)):
            input_ids, labels = batch
            out = model(input_ids=input_ids, labels=labels)
            loss1 = out[0]
            loss += float(loss1) / len(dataloader)
            loss1 /= grad_accumulation

            if i % grad_accumulation == 0:
                batch_loss = 0.0

            loss1.backward(retain_graph=True)

            batch_loss += float(loss1)

            if (i+1) % grad_accumulation == 0 or i+1 == len(dataloader):
                if (i // grad_accumulation) == (len(dataloader) // grad_accumulation) / 2:
                    batch_id = (i+1) // grad_accumulation
                    sp_norm, sp_norm_r, fraction_removed = gauss_newton_hessian(model, epoch, (i+1) // grad_accumulation, test_batch)
                    if write_log:
                        log_file.write(f'Epoch {epoch}, Batch {(i+1) // grad_accumulation}, Loss {batch_loss}\n')
                        log_file.write(f'Smoothness {sp_norm}\n')
                        log_file.write(f'Robust Smoothness {sp_norm_r}\n')
                        log_file.write(f'Fraction Removed {fraction_removed}\n')

                    num_calc += 1


                parameters_before_update = [p.clone().detach() for p in model.parameters()]
                optimizer.step()

                # calculate norm
                norm = 0.0
                parameters_after_update = [p for p in model.parameters()]
                with torch.no_grad():
                    for p1, p2 in zip(parameters_before_update, parameters_after_update):
                        param_norm = (p1 - p2).detach().data.norm(2)
                        norm += param_norm.item() ** 2
                    norm = norm ** 0.5
                    norms.append(norm)

                total_norm += norm / (len(dataloader) / grad_accumulation)

                optimizer.zero_grad()

                torch.cuda.empty_cache()

            del input_ids, labels, out

            torch.cuda.empty_cache()


        plt.figure()
        plt.hist(norms, bins=50)
        plt.title(f'Epoch {epoch}')
        plt.savefig(f'norm/adam_{epoch}.png')

        print(f'Epoch {epoch}, loss {loss}, norm {total_norm}')
        if write_log:
            log_file.write(f'Epoch {epoch}, loss {loss}, norm {total_norm}')

        loss_list.append(loss)
        adam_norm_list.append(total_norm)

        torch.save(model, f'model/model_adam_{epoch}.pt')

    del optimizer

    with open('output/adam_norm_list.pkl', 'wb') as f:
        pickle.dump(adam_norm_list, f)

    return loss_list, adam_norm_list


