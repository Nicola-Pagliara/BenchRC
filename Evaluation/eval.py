from Models import BERT as brt
from Support import constant as const
from Support.DataS import utils as utl
import torch
from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_score
from tqdm import tqdm
import pandas as pd
import numpy as np

def validate():
    model = brt.DistilBERTClass()
    model.load_state_dict(torch.load(const.DISTILBERT_WEIGHTS))
    model.eval()
    final_targets=[]
    model_outputs=[]
    testing_loader = trn.generate_dataloader(const.TEST_CSV_PATH)
    with torch.no_grad():
        for _, data in tqdm(enumerate(testing_loader, 0)):
            ids = data['ids']
            mask = data['mask']
            token_type_ids = data['token_type_ids']
            targets = data['labels']
            outputs = model(ids, mask, token_type_ids)
            final_targets.extend(targets.cpu().detach().numpy().tolist())
            model_outputs.extend(torch.nn.functional.softmax(outputs, dim=-1).cpu().detach().numpy().tolist())
    return model_outputs, final_targets

def compute_numeric_metrics():
    for i in range(0, len(const.MODELS_NAME)):
        choice = const.MODELS_NAME[i]
        eval_value = []
        outputs, targets = general_validation(choice)
        outputs_idx = np.argmax(outputs, axis=1, keepdims=True)
        f1_score_micro = f1_score(targets, outputs_idx, average='micro')
        f1_score_macro = f1_score(targets, outputs_idx, average='macro')
        accuracy = accuracy_score(targets, outputs_idx)
        precison = precision_score(targets, outputs_idx, average='weighted', zero_division=np.nan)
        recall = recall_score(targets, outputs_idx, average='weighted')
        eval_value.extend([round(f1_score_micro, 2), round(f1_score_macro, 2), round(accuracy, 2), round(precison, 2), round(recall, 2)])
        df = pd.DataFrame(data=eval_value, columns=['Eval Metrics'], index=['F1_micro', 'F1_macro', 'Accuracy', 'Precision', 'Recall'])
        df.to_csv(const.EVAL_ROOT + '/' + choice +'_eval_value.csv')
    #print(f'Acc {accuracy}, Prec {precison}, Rec {recall}, f1_micro {f1_score_micro}, f1_macro {f1_score_macro}')
    #print(f'shape of preds = {len(outputs)}, shape targets {len(targets)}, {len(outputs_idx)}')
    #print(f'{outputs}, {outputs_idx}, {targets}')
    # try to use sklearn encoding  to resolve this problem
    #roc_auc = roc_auc_score(targets, outputs, multi_class='ovr', average='weighted') # problems with missing classes
    #print(f'ROC {roc_auc}')
    return

def general_validation(model_name):
    final_targets=[]
    model_outputs=[]
    if model_name == 'DistillBert':
        _, testing_loader = utl.generate_dataloader(const.TRAIN_CSV_PATH, const.TEST_CSV_PATH, model_name)
        model = brt.DistilBERTClass()
        model.load_state_dict(torch.load(const.DISTILLBERT_WEIGHTS_INTRA))
        model.eval()
    elif model_name == 'Bert':
        _, testing_loader = utl.generate_dataloader(const.TRAIN_CSV_PATH, const.TEST_CSV_PATH, model_name)
        model = brt.BERTClass()
        model.load_state_dict(torch.load(const.BERT_WEIGHTS_INTRA))
        model.eval()
    elif model_name == 'SqueezeBert':
        _, testing_loader = utl.generate_dataloader(const.TRAIN_CSV_PATH, const.TEST_CSV_PATH, model_name) 
        model = brt.SqueezeBERTClass()
        model.load_state_dict(torch.load(const.SQUEEZEBERT_WEIGHTS_INTRA))
        model.eval()
    elif model_name == 'Roberta':
        _, testing_loader = utl.generate_dataloader(const.TRAIN_CSV_PATH, const.TEST_CSV_PATH, model_name)
        model = brt.RobertaClass()
        model.load_state_dict(torch.load(const.ROBERTA_WEIGHTS_INTRA))
        model.eval()
    else:
        print('Model choice problem')
        exit(0)
    with torch.no_grad():
        for _, data in tqdm(enumerate(testing_loader, 0)):
            ids = data['ids']
            mask = data['mask']
            token_type_ids = data['token_type_ids']
            targets = data['labels']
            outputs = model(ids, mask, token_type_ids)
            final_targets.extend(targets.cpu().detach().numpy().tolist())
            model_outputs.extend(torch.nn.functional.softmax(outputs, dim=-1).cpu().detach().numpy().tolist())
    return model_outputs, final_targets
    