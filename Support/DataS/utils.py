import pandas as pd
from Support.DataS.classes import DatasetCustom, NewDatasetCustom
from torch.utils.data import DataLoader
from Support import constant as const

def generate_dataloader(path_train_csv, path_test_csv, model_name):
    dataset_tr = pd.read_csv(path_train_csv)
    dataset_tt = pd.read_csv(path_test_csv)
    dataset_train = NewDatasetCustom(dataset=dataset_tr, max_len=const.MAX_LEN, model_name=model_name)
    #dataset_test = NewDatasetCustom(dataset=dataset_tt, max_len=const.MAX_LEN, model_name=model_name)
    train_loader = DataLoader(dataset=dataset_train, **const.train_params)
    #test_loader = DataLoader(dataset=dataset_test, **const.test_params)
    return train_loader
