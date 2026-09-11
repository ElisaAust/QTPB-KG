import os, sys
ROOT = os.path.dirname(__file__)
if ROOT not in sys.path:
    sys.path.append(ROOT)
import torch
import torch.nn as nn
from transformers import BertModel
from torchcrf import CRF

# 引入自定义 UFO Attention
from UFOAttention import UFOAttention

# 确保能导入同级模块
ROOT = os.path.dirname(__file__)
if ROOT not in sys.path:
    sys.path.append(ROOT)

class BertBiLSTMCRF(nn.Module):
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

        # 🔴 关键修复：为了兼容 train.py 的第 92 和 94 行
        self.attention_type = 'ufo' 
        self.self_attn = UFOAttention( # 必须叫 self_attn
            d_model=lstm_hidden,
            d_k=lstm_hidden // num_heads,
            d_v=lstm_hidden // num_heads,
            h=num_heads
        )

        # 4) 稳定性增强
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(lstm_hidden)

        # 5) 解码层
        self.classifier = nn.Linear(lstm_hidden, num_tags)
        self.crf = CRF(num_tags, batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None):
        # 1) BERT 特征提取
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

        # 2) BiLSTM 处理
        lstm_out, _ = self.bilstm(bert_out)

        # 3) UFO Attention 核心计算
        # 对齐属性名 self.self_attn
        ufo_out = self.self_attn(lstm_out, lstm_out, lstm_out)

        # 4) 残差 + LayerNorm
        combined_out = ufo_out + lstm_out
        norm_out = self.norm(combined_out)

        # 5) CRF 解码
        emissions = self.classifier(self.dropout(norm_out))
        mask = attention_mask.bool()

        if labels is not None:
            log_likelihood = self.crf(emissions, labels, mask=mask, reduction='none')
            token_counts = mask.sum(dim=1).float()
            return - (log_likelihood / token_counts).mean()
        else:
            return self.crf.decode(emissions, mask=mask)