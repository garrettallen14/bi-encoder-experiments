import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

class LinearProbe(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 2)

    def forward(self, x):
        return self.linear(x)

def load_model(device):
    model = AutoModelForCausalLM.from_pretrained('gpt2')
    model = model.to(device)
    return model

def contrastive_loss(query_emb, answer_emb):
    similarity = torch.mm(query_emb, answer_emb.t())
    labels = torch.arange(similarity.size(0)).to(similarity.device)
    loss_row = nn.functional.cross_entropy(similarity, labels)
    loss_col = nn.functional.cross_entropy(similarity.t(), labels)
    return (loss_row + loss_col) / 2