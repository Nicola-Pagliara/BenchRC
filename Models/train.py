import torch.optim

from Models import BERT as brt
from Support.DataS.classes import DatasetCustom
from torch.utils.data import DataLoader
from Support import constant as const
from tqdm import tqdm
import pandas as pd


def generate_dataloader(path_csv):
    dataset = pd.read_csv(path_csv)
    dataset_train = DatasetCustom(dataset=dataset, max_len=const.MAX_LEN)
    train_loader = DataLoader(dataset=dataset_train, **const.train_params)
    return train_loader


def train():
    dataloader_train = generate_dataloader(const.TRAIN_CSV_PATH)
    model = brt.BERTClass()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=const.LEARNING_RATE)
    loss_fn = torch.nn.CrossEntropyLoss()
    model.train()
    for epoch in range(const.EPOCHS):
        for step, sample in tqdm(enumerate(dataloader_train, 0)):
            ids = sample['ids']
            att_mask = sample['mask']
            token_type_ids = sample['token_type_ids']
            label = sample['labels']
            outputs = model(ids, att_mask, token_type_ids)
            _, outputs = torch.max(outputs, 1)
            optimizer.zero_grad()
            loss = loss_fn(outputs.float(), label.float())
            if step % 1000 == 0:
                print('Epoch {}, Loss {}'.format(epoch, loss.item()))

            loss.requires_grad = True
            loss.backward()
            optimizer.step()
    print('Train finished for number of tot epochs {}'.format(const.EPOCHS))
    torch.save(model.state_dict(), const.TRAIN_BERT_SAVE_WEIGHTS)
    print('Model weights saved at {}'.format(const.TRAIN_BERT_SAVE_WEIGHTS))
    return
