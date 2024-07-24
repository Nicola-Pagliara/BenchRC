from transformers import BertModel, DistilBertModel, RobertaModel, SqueezeBertModel
from Support import constant as const
import torch


class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.l1 = BertModel.from_pretrained('google-bert/bert-base-uncased')
        self.l2 = torch.nn.Linear(768, 768)
        self.l3 = torch.nn.Dropout(const.DROPOUT_RATE)
        self.l4 = torch.nn.Linear(768, const.NUM_REL)

    def forward(self, ids, mask, token_type_ids):
        _, output_1 = self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False)
        output_2 = self.l2(output_1)
        output_3 = self.l3(output_2)
        output = self.l4(output_3)
        return output


class DistilBERTClass(torch.nn.Module):  # check if introduce tanh or other function between l2 and l3
    def __init__(self):
        super(DistilBERTClass, self).__init__()
        self.l1 = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.l2 = torch.nn.Linear(768, 768)
        self.l3 = torch.nn.Dropout(const.DROPOUT_RATE)
        self.l4 = torch.nn.Linear(768, const.NUM_REL)

    def forward(self, input_ids, attention_mask, token_type_ids):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.l2(pooler)
        pooler = torch.nn.Tanh()(pooler)
        pooler = self.l3(pooler)
        output = self.l4(pooler)
        return output


class RobertaClass(torch.nn.Module):
    def __init__(self):
        super(RobertaClass, self).__init__()
        self.l1 = RobertaModel.from_pretrained("roberta-base")
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(const.DROPOUT_RATE)
        self.classifier = torch.nn.Linear(768, const.NUM_REL)

    def forward(self, input_ids, attention_mask, token_type_ids):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output


class SqueezeBERTClass(torch.nn.Module):
    def __init__(self):
        super(SqueezeBERTClass, self).__init__()
        self.l1 = SqueezeBertModel.from_pretrained("squeezebert/squeezebert-uncased")
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(const.DROPOUT_RATE)
        self.classifier = torch.nn.Linear(768, const.NUM_REL)

    def forward(self, input_ids, attention_mask, token_type_ids):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.GELU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output
