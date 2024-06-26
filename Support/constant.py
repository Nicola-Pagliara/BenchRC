import os

DATA_ROOT = 'Data'
PREPROCESS_ROOT = 'Preprocessing'
MODEL_ROOT = 'Model'
EVAL_ROOT = 'Evaluation'

# Preprocessing module constants
DoCRED_PREFIX = os.path.join(DATA_ROOT, 'DocRED')
DATASET_TRAIN_HUMAN = os.path.join(DoCRED_PREFIX, 'train_annotated.json')
PREFIX_SAVE_ENT = '/entity.json'
PREFIX_SAVE_REL = '/relations.json'
PREFIX_SAVE_SEN = '/sentences.json'  # change this prefix
PREFIX_INFO_REL = os.path.join(DoCRED_PREFIX, 'rel_info.json')


# Model module constants
TRAIN_BATCH_SIZE = 4
EPOCHS = 1
LEARNING_RATE = 1e-04
MAX_LEN = 500
train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': False,
                'num_workers': 0
                }
TRAIN_CSV_PATH = os.path.join(PREPROCESS_ROOT, 'complete_docred.csv')
# TRAIN_SIZE = 0.8
TRAIN_BERT_SAVE_WEIGHTS = os.path.join(MODEL_ROOT, 'BERT_weights.pth')
NUM_REL = 96

# Evaluation and testing module
TEST_CSV_PATH = os.path.join(EVAL_ROOT, 'test_docred.csv')


