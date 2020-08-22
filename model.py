from config import params
from transformers import AutoModel
from torch import nn


class RobertaBOYL(nn.Module):
    def __init__(self):
        super(BertTripletClassifier, self).__init__()
        self.model = AutoModel.from_pretrained(params["model"])
        self.dropout = nn.Dropout(0.25)
        self.out = nn.Linear(768, 1)

    def forward(self, ids, mask, token_type_ids):
        _, cls = self.model(ids, attention_mask=mask, token_type_ids=token_type_ids)

        bert_out = self.dropout(cls)
        output = self.out(bert_out)
        return output
