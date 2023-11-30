import torch
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
from datasets import load_dataset
import os
from tqdm.notebook import tqdm
import random


class CIFARDataset(Dataset):
    def __init__(self, data_path='../data/', dataset_size=10000, device='cuda'):
        super().__init__()
        self.dataset = self._load_cifar(data_path=data_path,
                                        dataset_size=dataset_size)
        self.device = device

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        input_ids, labels = self.dataset[idx]
        return input_ids.to(self.device), torch.tensor(labels).to(self.device)

    def _load_cifar(self, data_path='../data/', max_length=2048, dataset_size=10000):
        dataset_path = data_path + 'cifar_dataset.pth'
        if os.path.exists(dataset_path):
            dataset = torch.load(dataset_path)
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            dataset = torchvision.datasets.CIFAR10('../data', train=True, download=True, transform=transform)
            if dataset_size < len(dataset):
                subset_idx = random.sample(list(range(len(dataset))), k=dataset_size)
                dataset = torch.utils.data.dataset.Subset(dataset, subset_idx)
            torch.save(dataset, dataset_path)
        return dataset

