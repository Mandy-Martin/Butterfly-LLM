import torch

def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat([-x2, x1], dim=-1)

def apply_rope(q, k, positions):
    dim = q.shape[-1]
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=q.device) / dim))
    sinusoid = torch.einsum("i,j->ij", positions, inv_freq)

    sin = sinusoid.sin()[None,:,None,:]
    cos = sinusoid.cos()[None,:,None,:]

    q[...,::2], q[...,1::2] = q[...,::2]*cos - q[...,1::2]*sin, q[...,::2]*sin + q[...,1::2]*cos
    k[...,::2], k[...,1::2] = k[...,::2]*cos - k[...,1::2]*sin, k[...,::2]*sin + k[...,1::2]*cos
    return q, k
