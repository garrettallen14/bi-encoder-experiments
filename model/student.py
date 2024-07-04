import torch
import torch.nn as nn
from transformers import AutoModel
from ..config import Config

class PMA(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(dim, num_heads)
        self.ln = nn.LayerNorm(dim)
        self.ff = nn.Linear(dim, dim)

    def forward(self, x):
        attn_output, _ = self.multihead_attn(x, x, x)
        x = self.ln(x + attn_output)
        x = self.ln(x + self.ff(x))
        return x

class IEM(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear1 = nn.Linear(dim * 2, dim)
        self.linear2 = nn.Linear(dim, 1)

    def forward(self, x, y):
        concat = torch.cat([x, y], dim=-1)
        hidden = torch.relu(self.linear1(concat))
        return self.linear2(hidden)

class StudentModel(nn.Module):
    def __init__(self, model_name, num_heads=Config.NUM_HEADS):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.pma = PMA(self.encoder.config.hidden_size, num_heads)
        self.iem = IEM(self.encoder.config.hidden_size)

    def encode(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = self.pma(outputs.last_hidden_state)
        return embeddings.mean(dim=1)

    def forward(self, query_ids, query_mask, doc_ids, doc_mask):
        query_emb = self.encode(query_ids, query_mask)
        doc_emb = self.encode(doc_ids, doc_mask)
        return self.iem(query_emb, doc_emb)