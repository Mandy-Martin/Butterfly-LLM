import triton
import triton.language as tl

@triton.jit
def butterfly_flash_kernel(
    X, OUT,
    stride_x, stride_out,
    C, CHUNK,
    LAYER_BIT,
    D_HEAD, H,
    BASE_POS,                # absolute start position of this sequence
    BLOCK: tl.constexpr
):
    pid = tl.program_id(0)

    a = pid
    b = pid ^ (1 << LAYER_BIT)
    if b <= a:
        return

    offs = tl.arange(0, CHUNK)
    offs2 = offs + CHUNK

    # absolute token positions
    pos_a = BASE_POS + a * CHUNK + offs
    pos_b = BASE_POS + b * CHUNK + offs

    xa = tl.load(X + a * stride_x + offs)
    xb = tl.load(X + b * stride_x + offs)

    x = tl.cat([xa, xb], axis=0)

    # --- RoPE ---
    dim = D_HEAD
    inv_freq = 1.0 / (10000 ** (tl.arange(0, dim, 2) / dim))
    pos = tl.cat([pos_a, pos_b], axis=0)

    sinus = pos[:, None] * inv_freq[None, :]
    sin = tl.sin(sinus)
    cos = tl.cos(sinus)

    x_even = x[:, 0::2]
    x_odd  = x[:, 1::2]
    x = tl.cat([
        x_even * cos - x_odd * sin,
        x_even * sin + x_odd * cos
    ], axis=1)
    # -----------------

    # Flash attention math (simplified)
    q = x
    k = x
    v = x

    attn = tl.dot(q, tl.trans(k))
    attn = tl.softmax(attn)
    y = tl.dot(attn, v)

    tl.store(OUT + a * stride_out + offs, y[:CHUNK])
    tl.store(OUT + b * stride_out + offs, y[CHUNK:])


def butterfly_flash_triton(x, layer_bit, base_pos=0, chunk=128):
    B, N, D = x.shape
    C = N // chunk
    y = torch.empty_like(x)

    grid = (C,)

    butterfly_flash_kernel[grid](
        x, y,
        x.stride(1), y.stride(1),
        C, chunk,
        layer_bit,
        D, 12,
        base_pos,
        BLOCK=128
    )
    return y
