import torch
from torch.utils.data import Dataset
from transformers import (AutoConfig,
                          AutoTokenizer,
                          AutoModelForMaskedLM,
                          DataCollatorForLanguageModeling)
from datasets import load_dataset
import os
from tqdm.notebook import tqdm


class MaskedLanguageModelingDataset(Dataset):
    def __init__(self, data_path='../data/', max_length=512, dataset_size=10240, device='cuda', random_collate=False):
        super().__init__()
        self.max_length = max_length
        self.random_collate = random_collate
        self.tokenizer = AutoTokenizer.from_pretrained('distilroberta-base')
        self.data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer,
                                                             mlm_probability=0.15,
                                                             pad_to_multiple_of=max_length)
        self.dataset = self._load_masked_language_modeling(data_path=data_path,
                                                           max_length=self.max_length,
                                                           dataset_size=dataset_size)
        self.device = device

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        input_ids, labels = self.dataset[idx]
        input_ids = input_ids.to(self.device)
        labels = labels.to(self.device)
        return input_ids, labels

    def _load_masked_language_modeling(self, data_path='../data/', max_length=512, dataset_size=10240):
        dataset_path = data_path + 'masked_language_modeling_dataset.pth'
        if os.path.exists(dataset_path):
            dataset = torch.load(dataset_path)
        else:
            dataset = load_dataset("imdb", split=f"train[:{dataset_size}]").shuffle()
            dataset = [self.tokenizer.encode(row['text'], max_length=max_length) for row in dataset]
            if not self.random_collate:
                def preprocess(input_ids):
                    mlm_input = self.data_collator([input_ids])
                    input_ids, labels = mlm_input['input_ids'], mlm_input['labels']
                    input_ids = torch.squeeze(input_ids, 0)
                    labels = torch.squeeze(labels, 0)
                    return input_ids, labels

                dataset = list(map(preprocess, dataset))
            torch.save(dataset, dataset_path)
        return dataset
