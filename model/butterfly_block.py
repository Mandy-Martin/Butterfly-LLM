import torch
import torch.nn as nn
from model.flash_block import FlashBlock

class ButterflyBlock(nn.Module):
    def __init__(self, d_model, n_heads, chunk_size):
        super().__init__()
        self.chunk = chunk_size
        self.block = FlashBlock(d_model, n_heads)

    def forward(self, x, layer_bit):
        B,N,D = x.shape
        C = N // self.chunk

        ids = torch.arange(C, device=x.device)
        partner = ids ^ (1 << layer_bit)

        # only unique pairs
        mask = ids < partner
        pairs = torch.stack([ids[mask], partner[mask]], dim=1)

        # gather all chunk pairs
        blocks = torch.cat([
            torch.cat([x[:,a*self.chunk:(a+1)*self.chunk],
                       x[:,b*self.chunk:(b+1)*self.chunk]], dim=1)
            for a,b in pairs
        ], dim=0)

        # one big Flash-Attention call
        blocks = self.block(blocks)

        # scatter back
        y = x.clone()
        for i,(a,b) in enumerate(pairs):
            y[:,a*self.chunk:(a+1)*self.chunk] = blocks[i*B:(i+1)*B,:self.chunk]
            y[:,b*self.chunk:(b+1)*self.chunk] = blocks[i*B:(i+1)*B,self.chunk:]
        return y
