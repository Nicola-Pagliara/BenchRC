from abc import ABC

from torch.utils.data import Dataset
from Preprocessing import tokenizer as tkn
import torch


class DatasetCustom(Dataset):
    def __init__(self, dataset, max_len, model_name):
        self.tokenizer = tkn
        self.dataset = dataset
        self.sents = self.dataset['sents head tail'].tolist()
        self.max_len = max_len
        self.labels = self.dataset['Mapped Labels'].tolist()
        self.model_name = model_name

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, index):
        sent = self.sents[index]
        tokens = self.tokenizer.token_to_tensor(sent, self.max_len, self.model_name)

        ids = tokens['input_ids']
        mask = tokens['attention_mask']
        token_type_ids = tokens["token_type_ids"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'labels': torch.tensor(self.labels[index], dtype=torch.long)
        }

class NewDatasetCustom(Dataset):
    def __init__(self, dataset, max_len, model_name):
        self.tokenizer = tkn
        self.dataset = dataset
        self.sents_head = self.dataset['head sent'].tolist()
        self.sents_tail = self.dataset['tail sent'].tolist()
        self.max_len = max_len
        self.labels = self.dataset['Mapped Labels'].tolist()
        self.model_name = model_name

    def __len__(self):
        return len(self.sents_head)

    def __getitem__(self, index):
        sent_head = self.sents_head[index]
        sent_tail = self.sents_tail[index]
        tokens = self.tokenizer.token_to_tensor_pair(sent_head,sent_tail, self.max_len, self.model_name)

        ids = tokens['input_ids']
        mask = tokens['attention_mask']
        token_type_ids = tokens["token_type_ids"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'labels': torch.tensor(self.labels[index], dtype=torch.long)
        }
