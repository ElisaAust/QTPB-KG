import os, sys
import torch
import torch.nn as nn
from transformers import BertModel
from torchcrf import CRF

ROOT = os.path.dirname(__file__)
if ROOT not in sys.path:
    sys.path.append(ROOT)

class BertBiLSTMCRF_BBC(nn.Module):
    """
    BERT + BiLSTM + CRF（不含 Attention）
    """
    def __init__(
        self,
        bert_model_name: str,
        lstm_hidden: int,
        num_tags: int,
        lstm_layers: int = 1,
        dropout: float = 0.2
    ):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        hidden_size = self.bert.config.hidden_size

        self.bilstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=lstm_hidden // 2,
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=True
        )

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(lstm_hidden, num_tags)
        self.crf = CRF(num_tags, batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None):
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state  # (B, L, H)
        lstm_out, _ = self.bilstm(bert_out)  # (B, L, hidden)
        emissions = self.classifier(self.dropout(lstm_out))  # (B, L, num_tags)

        mask = attention_mask.bool()
        if labels is not None:
            log_likelihood = self.crf(emissions, labels, mask=mask, reduction='none')
            token_counts = mask.sum(dim=1).float()
            loss = - (log_likelihood / token_counts).mean()
            return loss
        else:
            return self.crf.decode(emissions, mask=mask)
