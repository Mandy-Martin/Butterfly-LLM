import torch

class ButterflyStreamCache:
    def __init__(self, layers, chunk_size):
        self.chunk = chunk_size
        self.layers = layers
        self.cache = [{} for _ in range(layers)]

    def get(self, layer, chunk_id):
        return self.cache[layer].get(chunk_id, None)

    def set(self, layer, chunk_id, value):
        self.cache[layer][chunk_id] = value
