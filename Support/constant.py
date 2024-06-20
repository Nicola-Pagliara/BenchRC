import os

DATA_ROOT = 'Data'
PREPROCESS_ROOT = 'Preprocessing'

# Preprocessing module constants
DoCRED_PREFIX = os.path.join(DATA_ROOT, 'DocRED')
DATASET_TRAIN_HUMAN = os.path.join(DoCRED_PREFIX, 'train_annotated.json')
PREFIX_SAVE_ENT = '/entity.json'
PREFIX_SAVE_REL = '/relations.json'
PREFIX_SAVE_SEN = '/sentences.json'



