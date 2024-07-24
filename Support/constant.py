import os
DATA_ROOT = 'Data'
PREPROCESS_ROOT = 'Preprocessing'
MODEL_ROOT = 'Models'
EVAL_ROOT = 'Evaluation'

# Preprocessing module constants
DoCRED_PREFIX = os.path.join(DATA_ROOT, 'DocRED')
DATASET_TRAIN_HUMAN = os.path.join(DoCRED_PREFIX, 'train_annotated.json')
DATASET_EVAL = os.path.join(DoCRED_PREFIX, 'dev.json')
PREFIX_INFO_REL = os.path.join(DoCRED_PREFIX, 'rel_info.json')
DATASET_HT = os.path.join(PREPROCESS_ROOT, 'dataset_ht.csv')
DATASET_INTRA = os.path.join(PREPROCESS_ROOT, 'dataset_intra.csv')


# Model module constants
MODELS_NAME = ['DistillBert', 'Bert', 'SqueezeBert', 'Roberta']
TRAIN_BATCH_SIZE = 10
EPOCHS = 1
LEARNING_RATE = 1e-05
MAX_LEN = 400
DROPOUT_RATE = 0.25
train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }
TRAIN_CSV_PATH = os.path.join(PREPROCESS_ROOT, 'dataset_train_ht.csv')
BERT_WEIGHTS = os.path.join(MODEL_ROOT, 'BERT_weights.pth')
DISTILLBERT_WEIGHTS = os.path.join(MODEL_ROOT, 'distillBERT_weights.pth')
ROBERTA_WEIGHTS = os.path.join(MODEL_ROOT, 'roberta_weights.pth')
SQUEEZEBERT_WEIGHTS = os.path.join(MODEL_ROOT, 'squeezebert_weights.pth')
BERT_WEIGHTS_INTRA = os.path.join(MODEL_ROOT, 'BERT_weights_intra.pth')
DISTILLBERT_WEIGHTS_INTRA = os.path.join(MODEL_ROOT, 'distillBERT_weights_intra.pth')
SQUEEZEBERT_WEIGHTS_INTRA = os.path.join(MODEL_ROOT, 'squeezebert_weights_intra.pth')
ROBERTA_WEIGHTS_INTRA = os.path.join(MODEL_ROOT, 'roberta_weights_intra.pth')
NUM_REL = 96

# Evaluation and testing module
TEST_CSV_PATH = os.path.join(PREPROCESS_ROOT, 'test_intra_dataset.csv')
TEST_BATCH_SIZE = 12
test_params = { 'batch_size': TEST_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
}



