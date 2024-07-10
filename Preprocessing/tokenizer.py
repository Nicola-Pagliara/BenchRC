"""
This file contains methods that uses tokenizer for prepare sentences and enrich them with entity information for BERT
embeddings. opz wrapping in classes
"""
from transformers import BertTokenizer


def TokenForBERT(obj_to_encode):
    bert_tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path='google-bert/bert-base-uncased')
    tokens_id = bert_tokenizer.encode(obj_to_encode)
    token = bert_tokenizer.convert_ids_to_tokens(tokens_id)
    return tokens_id, token


def buit_in_tokenizer():
    return


def token_to_tensor(tokens, max_len):
    bert_tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path='google-bert/bert-large-uncased')
# add [CLS] and [SEP] tokens and convert it to tensor + additional information, like max sequence length seq and
    # padding strategy with string.
    ids_plus = bert_tokenizer.encode_plus(tokens, return_token_type_ids=True,
                                          return_attention_mask=True, add_special_tokens=True, max_length=max_len,
                                          padding='max_length', truncation=True)

    return ids_plus
