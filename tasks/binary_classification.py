import torch
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)
from datasets import load_dataset
import os
from tqdm.notebook import tqdm


class BinaryClassificationDataset(Dataset):
    def __init__(self, data_path='../data/', max_length=512, dataset_size=10240, device='cuda'):
        super().__init__()
        self.max_length = max_length
        self.dataset = self._load_binary_classification(data_path=data_path,
                                                        dataset_size=dataset_size,
                                                        max_length=self.max_length)
        self.device = device

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        input_ids, labels = self.dataset[idx]
        input_ids = torch.tensor(input_ids \
                                 + [0] * (self.max_length - len(input_ids))).to(self.device)
        labels = torch.tensor(labels).to(self.device)
        return input_ids, labels

    def _load_binary_classification(self, data_path='../data/', dataset_size=10240, max_length=512):
        dataset_path = data_path + 'binary_classification_dataset.pth'
        if os.path.exists(dataset_path):
            dataset = torch.load(dataset_path)
        else:
            tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
            dataset = load_dataset("imdb")
            dataset = dataset['train'].shuffle()
            dataset = [(tokenizer.encode(row['text'], truncation=True)[:max_length], row['label'])
                       for row in tqdm(dataset)][:dataset_size]
            torch.save(dataset, dataset_path)
        return dataset

