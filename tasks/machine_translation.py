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


class MachineTranslationDataset(Dataset):
    def __init__(self, data_path='../data/', max_length=512, dataset_size=10240, dbg=False, device='cuda'):
        super().__init__()
        self.dbg = dbg
        self.max_length = max_length
        self.dataset = self._load_machine_translation(data_path=data_path,
                                                      max_length=self.max_length,
                                                      dataset_size=dataset_size)
        self.device = device

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        input_ids, labels = self.dataset[idx]
        input_ids = torch.tensor(input_ids + [0] * (self.max_length - len(input_ids))).long().to(self.device)
        labels = torch.tensor(labels).long().to(self.device)
        return input_ids, labels

    def _load_machine_translation(self, data_path='../data/', max_length=512, dataset_size=10240):
        dataset_path = data_path + 'machine_translation_dataset.pth'
        if self.dbg:
            dataset_path = data_path + 'dbg_dataset.pth'

        if os.path.exists(dataset_path):
            dataset = torch.load(dataset_path)
        else:
            tokenizer = AutoTokenizer.from_pretrained('t5-small')
            dataset = load_dataset("opus_books", "en-fr")
            dataset = dataset['train'].shuffle()
            prefix = "translate English to French: "
            input_ids = [tokenizer.encode(prefix + row['en'])[:max_length] for row in tqdm(dataset['translation'])]
            with tokenizer.as_target_tokenizer():
                labels = [tokenizer.encode(row['fr'])[:max_length] for row in tqdm(dataset['translation'])]
            dataset = list(zip(input_ids, labels))[:dataset_size]
            torch.save(dataset, dataset_path)
        return dataset

