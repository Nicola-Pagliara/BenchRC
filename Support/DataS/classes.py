from abc import ABC

from torch.utils.data import Dataset
from Preprocessing import tokenizer as tkn
import torch


class DatasetCustom(Dataset):
    def __init__(self, dataset, max_len):
        self.tokenizer = tkn
        self.dataset = dataset
        self.sents = self.dataset['evidence sent . head ent mentions, tail ent mentions'].tolist()
        self.max_len = max_len
        self.labels = self.dataset['Mapped Labels'].tolist()

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, index):
        sent = self.sents[index]
        tokens = self.tokenizer.token_to_tensor(sent, self.max_len)

        ids = tokens['input_ids']
        mask = tokens['attention_mask']
        token_type_ids = tokens["token_type_ids"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'labels': torch.tensor(self.labels[index], dtype=torch.int)
        }

    # Put here class for test dataset
