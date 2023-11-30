import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from transformers import (AutoConfig,
                          AutoTokenizer,
                          AutoModelForSeq2SeqLM,
                          AutoModelForSequenceClassification,
                          AutoModelForMaskedLM,
                          AutoModelForCausalLM)
from tasks.machine_translation import MachineTranslationDataset
from tasks.binary_classification import BinaryClassificationDataset
from tasks.masked_language_modeling import MaskedLanguageModelingDataset
from tasks.autoregressive import AutoRegressiveDataset
from tasks.image_classification import CIFARDataset
from datasets import load_dataset
from itertools import accumulate
from tqdm.notebook import tqdm
import torchvision
import pickle
import os


DBG_DATASET_SIZE = 320 # size of the dataset for sanity check


def load_model(task, device='cuda'):
    if task == 'image_classification':
        model = torchvision.models.resnet152(pretrained=True).to(device)
        model.train()
        return model

    task_to_model_name = {
        'machine_translation' : 't5-small',
        'binary_classification' : 'distilbert-base-uncased',
        'masked_language_modeling' : 'distilroberta-base',
        'autoregressive' : 'flax-community/gpt-neo-125M-code-clippy',
    }
    task_to_type = {
        'machine_translation' : AutoModelForSeq2SeqLM,
        'binary_classification' : AutoModelForSequenceClassification,
        'masked_language_modeling' : AutoModelForMaskedLM,
        'autoregressive' : AutoModelForCausalLM,
    }
    if task == 'dbg':
        task = 'machine_translation'
    model_name = task_to_model_name[task]
    task_type = task_to_type[task]
    config = AutoConfig.from_pretrained(model_name)
    model = task_type.from_pretrained(model_name).to(device)
    model.train()
    return model


def load_dataloader(task, **kwargs):
    task_to_dataset = {
        'machine_translation' : MachineTranslationDataset,
        'binary_classification' : BinaryClassificationDataset,
        'masked_language_modeling' : MaskedLanguageModelingDataset,
        'autoregressive' : AutoRegressiveDataset,
        'image_classification' : CIFARDataset,
    }
    if task == 'dbg':
        task = 'machine_translation'
        kwargs['dataset_size'] = DBG_DATASET_SIZE
        kwargs['dbg'] = True
    dataset = task_to_dataset[task](**kwargs)
    batch_size = 1
    if task == "image_classification":
        batch_size = 8
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def _load_dbg_model():
    """
    Loads an extremely small model for checking the Hessian sharpness computation by brute force.
    Should not be used in actual experiments.
    """
    config = AutoConfig.from_pretrained('t5-small', num_heads=2, num_layers=2, d_ff=16, d_kv=4, d_model=4)
    model = AutoModelForSeq2SeqLM.from_config(config)
    model = model.to('cuda')
    return model

