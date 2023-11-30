import torch
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
)
from datasets import load_dataset
import os
from tqdm.notebook import tqdm


class AutoRegressiveDataset(Dataset):
    def __init__(self, data_path='../data/', max_length=2048, dataset_size=10000, device='cuda'):
        super().__init__()
        self.max_length = max_length
        self.dataset = self._load_autoregressive(data_path=data_path,
                                                 max_length=self.max_length,
                                                 dataset_size=dataset_size)
        self.device = device

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        input_ids = self.dataset[idx]
        input_ids = torch.tensor(input_ids + [0] * (self.max_length - len(input_ids))).to(self.device)
        return input_ids, input_ids.clone().detach()

    def _load_autoregressive(self, data_path='../data/', max_length=2048, dataset_size=10000):
        dataset_path = data_path + 'autoregressive_dataset.pth'
        if os.path.exists(dataset_path):
            dataset = torch.load(dataset_path)
        else:
            # tokenizer = AutoTokenizer.from_pretrained('Salesforce/codegen-350M-mono')
            tokenizer = AutoTokenizer.from_pretrained("flax-community/gpt-neo-125M-code-clippy")
            dataset = load_dataset("bigcode/the-stack-smol", data_dir="data/python", split=f"train[0:{dataset_size}]")
            dataset = dataset.shuffle()
            dataset = [tokenizer.encode(row['content'])[:max_length] for row in tqdm(dataset)]
            torch.save(dataset, dataset_path)
        return dataset

