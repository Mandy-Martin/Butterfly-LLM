import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from model.local_block import LocalBlock
from model.butterfly_block import ButterflyBlock


class CharCodeButterfly(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.embed = nn.Embedding(cfg["vocab_size"], cfg["hidden_size"])

        # Local encoder
        self.local_layers = nn.ModuleList([
            LocalBlock(cfg["hidden_size"], cfg["num_heads"],
                       cfg["ffn_mult"], cfg["dropout"])
            for _ in range(cfg["local_layers"])
        ])

        # Butterfly global mixer
        self.butterfly_layers = nn.ModuleList([
            ButterflyBlock(cfg["hidden_size"], cfg["num_heads"],
                           cfg["chunk_size"])
            for _ in range(cfg["butterfly_passes"] * cfg["butterfly_layers"])
        ])

        # Refinement
        self.refine_layers = nn.ModuleList([
            LocalBlock(cfg["hidden_size"], cfg["num_heads"],
                       cfg["ffn_mult"], cfg["dropout"])
            for _ in range(cfg["refine_layers"])
        ])

        self.head = nn.Linear(cfg["hidden_size"], cfg["vocab_size"])

    # ----------------------------------------------------------
    # Full forward (training / non-streaming inference)
    # ----------------------------------------------------------
    def forward(self, x):
        x = self.embed(x)

        # Local encoder
        for layer in self.local_layers:
            x = checkpoint(layer, x)

        # Butterfly mixer
        bits = self.cfg["butterfly_layers"]
        for i, layer in enumerate(self.butterfly_layers):
            bit = i % bits
            x = checkpoint(layer, x, bit)

        # Refinement
        for layer in self.refine_layers:
            x = checkpoint(layer, x)

        return self.head(x)

    # ----------------------------------------------------------
    # Streaming forward (real-time million-token inference)
    # ----------------------------------------------------------
    def forward_stream(self, last_token, cache, base_pos):
        """
        last_token: (B,1)
        cache: ButterflyStreamCache
        base_pos: absolute position index
        """
        B = last_token.size(0)
        device = last_token.device
        chunk = self.cfg["chunk_size"]
        bit_depth = self.cfg["butterfly_layers"]

        # Embed new token
        x = self.embed(last_token)

        # Local layers (cheap)
        for layer in self.local_layers:
            x = checkpoint(layer, x)

        # Chunk index
        chunk_id = base_pos // chunk
        slot = base_pos % chunk

        # Butterfly streaming
        for l, layer in enumerate(self.butterfly_layers):
            bit = l % bit_depth
            partner = chunk_id ^ (1 << bit)

            a = cache.get(l, chunk_id)
            b = cache.get(l, partner)

            if a is None:
                a = torch.zeros(B, chunk, x.size(-1), device=device)
            if b is None:
                b = torch.zeros_like(a)

            a[:, slot:slot+1] = x

            # Only process this chunk pair
            merged = torch.cat([a, b], dim=1)
            merged = checkpoint(layer.block, merged)

            a_new = merged[:, :chunk]
            b_new = merged[:, chunk:]

            cache.set(l, chunk_id, a_new)
            cache.set(l, partner, b_new)

            x = a_new[:, slot:slot+1]

        # Refinement layers
        for layer in self.refine_layers:
            x = checkpoint(layer, x)

        return self.head(x)
