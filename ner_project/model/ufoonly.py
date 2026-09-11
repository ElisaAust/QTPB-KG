import os, sys
import torch
import torch.nn as nn
from transformers import BertModel
from torchcrf import CRF

from UFOAttention import UFOAttention

ROOT = os.path.dirname(__file__)
if ROOT not in sys.path:
    sys.path.append(ROOT)

class BertBiLSTMCRF_UFOOnly(nn.Module):
    """
    消融实验: BERT + BiLSTM + UFO + Residual + CRF
    (无 LayerNorm)
    """
    def __init__(
        self,
        bert_model_name: str,
        lstm_hidden: int,
        num_tags: int,
        lstm_layers: int = 1,
        dropout: float = 0.2,
        num_heads: int = 8
    ):
        super().__init__()
        # 1) BERT
        self.bert = BertModel.from_pretrained(bert_model_name)
        hidden_size = self.bert.config.hidden_size

        # 2) BiLSTM
        self.bilstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=lstm_hidden // 2,
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=True
        )

        # 3) UFO Attention
        self.attention = UFOAttention(
            d_model=lstm_hidden,
            d_k=lstm_hidden // num_heads,
            d_v=lstm_hidden // num_heads,
            h=num_heads
        )

        # 4) Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 5) 分类器
        self.classifier = nn.Linear(lstm_hidden, num_tags)
        
        # 6) CRF
        self.crf = CRF(num_tags, batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None):
        # 1) BERT 特征提取
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

        # 2) BiLSTM 处理
        lstm_out, _ = self.bilstm(bert_out)

        # 3) UFO Attention
        attn_out = self.attention(lstm_out, lstm_out, lstm_out)

        # 4) ✅ 残差连接（保留）fused = lstm_out + attn_out
        fused = lstm_out + attn_out
        
        # ❌ 无 LayerNorm（消融实验）

        # 5) CRF 解码
        emissions = self.classifier(self.dropout(fused))
        mask = attention_mask.bool()

        if labels is not None:
            log_likelihood = self.crf(emissions, labels, mask=mask, reduction='none')
            token_counts = mask.sum(dim=1).float()
            return - (log_likelihood / token_counts).mean()
        else:
            return self.crf.decode(emissions, mask=mask)