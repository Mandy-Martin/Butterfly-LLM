import torch
from torch.utils.data import IterableDataset

class CharDataset(IterableDataset):
    def __init__(self, files, seq_len):
        self.files = files
        self.seq_len = seq_len

    def __iter__(self):
        for fname in self.files:
            with open(fname, "rb") as f:
                data = torch.frombuffer(f.read(), dtype=torch.uint8)
            for i in range(0, len(data) - self.seq_len - 1, self.seq_len):
                x = data[i:i+self.seq_len]
                y = data[i+1:i+self.seq_len+1]
                yield x.long(), y.long()
