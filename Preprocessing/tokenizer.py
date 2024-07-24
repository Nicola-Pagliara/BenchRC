"""
This file contains methods that uses tokenizer for prepare sentences and enrich them with entity information for BERT
embeddings. opz wrapping in classes
"""
from transformers import BertTokenizer, DistilBertTokenizer, RobertaTokenizer, SqueezeBertTokenizer


def TokenForBERT(obj_to_encode):
    bert_tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path='google-bert/bert-base-uncased')
    tokens_id = bert_tokenizer.encode(obj_to_encode)
    token = bert_tokenizer.convert_ids_to_tokens(tokens_id)
    return tokens_id, token


def token_to_tensor(tokens, max_len, model_name):

    if model_name == 'DistillBert':
            distilbert_tokenizer = DistilBertTokenizer.from_pretrained(pretrained_model_name_or_path='distilbert-base-uncased')
            # add [CLS] and [SEP] tokens and convert it to tensor + additional information, like max sequence length seq and
            # padding strategy with string.
            ids_plus = distilbert_tokenizer.encode_plus(tokens, return_token_type_ids=True,
                                          return_attention_mask=True, add_special_tokens=True, max_length=max_len,
                                          padding='max_length', truncation=True)
    elif model_name == 'Bert':
            bert_tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path='google-bert/bert-base-uncased')
            ids_plus = bert_tokenizer.encode_plus(tokens, return_token_type_ids=True,
                                          return_attention_mask=True, add_special_tokens=True, max_length=max_len,
                                          padding='max_length', truncation=True)
    elif model_name == 'SqueezeBert':
            squeezebert_tokenizer = SqueezeBertTokenizer.from_pretrained(pretrained_model_name_or_path='squeezebert/squeezebert-uncased')
            ids_plus = squeezebert_tokenizer.encode_plus(tokens, return_token_type_ids=True,
                                          return_attention_mask=True, add_special_tokens=True, max_length=max_len,
                                          padding='max_length', truncation=True)
    elif model_name == 'Roberta':
            roberta_tokenizer = RobertaTokenizer.from_pretrained(pretrained_model_name_or_path='roberta-base')
            ids_plus = roberta_tokenizer.encode_plus(tokens, return_token_type_ids=True,
                                          return_attention_mask=True, add_special_tokens=True, max_length=max_len,
                                          padding='max_length', truncation=True)
    else:
        print('Error choice tokenizer')
        exit(0)

    return ids_plus

def token_to_tensor_pair(token1,token2, max_len, model_name):

    if model_name == 'DistillBert':
            distilbert_tokenizer = DistilBertTokenizer.from_pretrained(pretrained_model_name_or_path='distilbert-base-uncased')
            # add [CLS] and [SEP] tokens and convert it to tensor + additional information, like max sequence length seq and
            # padding strategy with string.
            ids_plus = distilbert_tokenizer.encode_plus([token1, token2], return_token_type_ids=True,
                                          return_attention_mask=True, add_special_tokens=True, max_length=max_len,
                                          padding='max_length', truncation=True)
    elif model_name == 'Bert':
            bert_tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path='google-bert/bert-base-uncased')
            ids_plus = bert_tokenizer.encode_plus([token1, token2], return_token_type_ids=True,
                                          return_attention_mask=True, add_special_tokens=True, max_length=max_len,
                                          padding='max_length', truncation=True)
    elif model_name == 'SqueezeBert':
            squeezebert_tokenizer = SqueezeBertTokenizer.from_pretrained(pretrained_model_name_or_path='squeezebert/squeezebert-uncased')
            ids_plus = squeezebert_tokenizer.encode_plus([token1, token2], return_token_type_ids=True,
                                          return_attention_mask=True, add_special_tokens=True, max_length=max_len,
                                          padding='max_length', truncation=True)
    elif model_name == 'Roberta':
            roberta_tokenizer = RobertaTokenizer.from_pretrained(pretrained_model_name_or_path='roberta-base')
            ids_plus = roberta_tokenizer.encode_plus([token1, token2], return_token_type_ids=True,
                                          return_attention_mask=True, add_special_tokens=True, max_length=max_len,
                                          padding='max_length', truncation=True)
    else:
        print('Error choice tokenizer')
        exit(0)

    return ids_plus

