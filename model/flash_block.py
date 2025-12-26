import torch
import torch.nn as nn
from flash_attn.flash_attn_interface import flash_attn_func
from model.rope import apply_rope

class FlashBlock(nn.Module):
    def __init__(self, d_model, n_heads, ffn_mult=4, dropout=0.1):
        super().__init__()
        self.h = n_heads
        self.dh = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3*d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

        self.ff = nn.Sequential(
            nn.Linear(d_model, ffn_mult*d_model),
            nn.GELU(),
            nn.Linear(ffn_mult*d_model, d_model),
            nn.Dropout(dropout),
        )
        self.n1 = nn.LayerNorm(d_model)
        self.n2 = nn.LayerNorm(d_model)

    def forward(self, x):
        B, N, D = x.shape
        pos = torch.arange(N, device=x.device)

        # Project QKV
        qkv = self.qkv(x).view(B, N, 3, self.h, self.dh)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        # Rotary embeddings
        q, k = apply_rope(q, k, pos)

        # Flash-Attention
        y = flash_attn_func(q, k, v, causal=False)
        y = y.reshape(B, N, D)

        # Residual + FFN
        x = self.n1(x + self.proj(y))
        x = self.n2(x + self.ff(x))
        return x
