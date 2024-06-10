"""
This file contains methods that uses tokenizer for prepare sentences and enrich them with entity information for BERT
embeddings. opz wrapping in classes
"""
from transformers import BertTokenizer
from tokenizers.models import WordPiece


def TokenForBERT(obj_to_encode):
    bert_tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path='google-bert/bert-large-uncased')
    bert_tokenizer.add_special_tokens()
    tokens_id = bert_tokenizer.encode(obj_to_encode)
    token = bert_tokenizer.convert_ids_to_tokens(tokens_id)
    return tokens_id, token


def buit_in_tokenizer():
    return
