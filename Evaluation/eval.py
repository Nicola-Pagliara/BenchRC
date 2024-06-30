from Models import BERT as brt
from Models import train as trn
from Support import constant as const
import torch
from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_score, roc_auc_score
import numpy as np

def validate():
    model = brt.BERTClass()
    model = torch.load(const.TRAIN_BERT_SAVE_WEIGHTS)
    model.eval()
    final_targets=[]
    model_outputs=[]
    testing_loader = trn.generate_dataloader(const.TEST_CSV_PATH)
    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            ids = data['ids']
            mask = data['mask']
            token_type_ids = data['token_type_ids']
            targets = data['labels']
            outputs = model(ids, mask, token_type_ids)
            final_targets.extend(targets.cpu().detach().numpy().tolist())
            model_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
    return model_outputs, final_targets

def compute_numeric_metrics():
    outputs, targets = validate()
    outputs = np.array(outputs) >= 0.5
    f1_score_micro = f1_score(targets, outputs, average='micro')
    f1_score_macro = f1_score(targets, outputs, average='macro')
    accuracy = accuracy_score(targets, outputs)
    precison = precision_score(targets, outputs)
    recall = recall_score(targets, outputs)
    roc_auc = roc_auc_score(targets, outputs)
    print(f'Acc {accuracy}, Prec {precison}, Rec {recall}, f1_micro {f1_score_micro}, f1_macro {f1_score_macro}')
    print(f'roc_auc {roc_auc}')
    return