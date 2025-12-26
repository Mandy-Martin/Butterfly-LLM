import torch
import torch.nn as nn
from utils.masks import local_mask

class LocalBlock(nn.Module):
    def __init__(self, d_model, n_heads, ffn_mult=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ffn_mult*d_model),
            nn.GELU(),
            nn.Linear(ffn_mult*d_model, d_model),
            nn.Dropout(dropout),
        )
        self.n1 = nn.LayerNorm(d_model)
        self.n2 = nn.LayerNorm(d_model)

    def forward(self, x):
        mask = local_mask(x.size(1), device=x.device)
        y,_ = self.attn(x, x, x, attn_mask=mask)
        x = self.n1(x + y)
        x = self.n2(x + self.ff(x))
        return x
