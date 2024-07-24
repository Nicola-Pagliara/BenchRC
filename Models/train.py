import torch.optim
from Models import BERT as brt
from Support import constant as const
from Support.DataS import utils as utl
from tqdm import tqdm
import pandas as pd



def train():
    train_loss = 0
    nb_train_step = 0
    dataloader_train = generate_dataloader(const.TRAIN_CSV_PATH)
    model = brt.SqueezeBERTClass()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=const.LEARNING_RATE)
    loss_fn = torch.nn.CrossEntropyLoss()
    model.train()
    for epoch in range(const.EPOCHS):
        for _, sample in tqdm(enumerate(dataloader_train, 0)):
            ids = sample['ids']
            att_mask = sample['mask']
            token_type_ids = sample['token_type_ids']
            label = sample['labels']
            outputs = model(ids, att_mask, token_type_ids)
            #outputs_max, outputs_index = torch.max(outputs.data, dim=1)
            optimizer.zero_grad()
            loss = loss_fn(outputs, label)
            train_loss+=loss.item()
            nb_train_step+=1
            #loss.requires_grad = True
            if _ % 100 == 0:
                tr_loss_steps = train_loss/nb_train_step
                print('Epoch {}, Loss {}, for nb step {}.'.format(epoch, tr_loss_steps, nb_train_step))
            loss.backward()
            optimizer.step()
    print('Train finished for number of tot epochs {}'.format(const.EPOCHS))
    torch.save(model.state_dict(), const.SQUEEZEBERT_WEIGHTS)
    print('Model weights saved at {}'.format(const.SQUEEZEBERT_WEIGHTS))
    return


def general_train():
    loss_fn = torch.nn.CrossEntropyLoss()
    for i in range(0, len(const.MODELS_NAME)):
            choice = const.MODELS_NAME[i]
            if choice == 'DistillBert':
                dataloader_train = utl.generate_dataloader(const.TRAIN_CSV_PATH, const.TEST_CSV_PATH, choice)
                WEIGHTS = const.DISTILLBERT_WEIGHTS
                model = brt.DistilBERTClass()
                optimizer = torch.optim.Adam(params=model.parameters(), lr=const.LEARNING_RATE)
                model.train()
            elif choice == 'Bert':
                dataloader_train = utl.generate_dataloader(const.TRAIN_CSV_PATH, const.TEST_CSV_PATH, choice)
                WEIGHTS = const.BERT_WEIGHTS
                model = brt.BERTClass()
                optimizer = torch.optim.Adam(params=model.parameters(), lr=const.LEARNING_RATE)
                model.train()
            elif choice == 'SqueezeBert':
                dataloader_train = utl.generate_dataloader(const.TRAIN_CSV_PATH, const.TEST_CSV_PATH, choice)
                WEIGHTS = const.SQUEEZEBERT_WEIGHTS
                model = brt.SqueezeBERTClass()
                optimizer = torch.optim.Adam(params=model.parameters(), lr=const.LEARNING_RATE)
                model.train()
            elif choice == 'Roberta':
                dataloader_train = utl.generate_dataloader(const.TRAIN_CSV_PATH, const.TEST_CSV_PATH, choice)
                WEIGHTS = const.ROBERTA_WEIGHTS
                model = brt.RobertaClass()
                optimizer = torch.optim.Adam(params=model.parameters(), lr=const.LEARNING_RATE)
                model.train()
            else:
                print('Problem in model choice')
                break
            
            for epoch in range(0, const.EPOCHS):
                for _, sample in tqdm(enumerate(dataloader_train, 0)):
                    ids = sample['ids']
                    att_mask = sample['mask']
                    token_type_ids = sample['token_type_ids']
                    label = sample['labels']
                    outputs = model(ids, att_mask, token_type_ids)
                    optimizer.zero_grad()
                    loss = loss_fn(outputs, label)
                    if _ % 100 == 0:
                        print('Epoch {}, Loss {}, model {}.'.format(epoch, loss.item(), choice))
                    loss.backward()
                    optimizer.step()
            print('Train finished for model {}, and number of tot epochs {}'.format(choice,const.EPOCHS))
            torch.save(model.state_dict(), WEIGHTS)
            print('Model weights saved at {}'.format(WEIGHTS))



                                      
    return
