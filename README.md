# Butterfly-LLM

Butterfly-LLM is a new class of long-context language model architecture designed to make **million-token reasoning practical on consumer GPUs**.

This repo contains experimental code to test and explore the architecture. All claims are theoretical until benchmarked—but you have to start with a thesis.

---

## The Core Idea

Dense attention doesn't need to happen in one layer.

Instead of compressing memory, approximating attention, or sacrificing global expressivity, Butterfly factorizes **dense global self-attention across depth rather than width** using structured butterfly (hypercube) routing.

1. Tokens mix within fixed-size chunks locally
2. Global information propagates across layers via structured butterfly routing
3. After O(log N) layers, every token can influence every other token

This achieves **exact global expressivity** without ever performing full N×N attention—analogous to how FFT computes dense transforms using logarithmic-depth structured mixing.

---

## Theoretical Complexity

| Property | Standard Transformer | Butterfly-LLM |
|----------|---------------------|---------------|
| Training compute | O(N²) | O(N log N) |
| Inference compute per token | O(N) | O(log N) |
| KV memory (theoretical) | O(N · d · L) | O(B · d · L) (Constant)* |
| Attention window per layer | Global (N×N) | Constant (2B × 2B) |
| Global connectivity | Single layer | O(log N) hops |

*With sliding-window eviction. See [Memory Scaling](#memory-scaling) for current implementation status.

---

## Why Transformers Break at Long Context

Standard Transformers require every token to attend to every other token in a single layer. As context grows, **memory becomes the bottleneck**—not compute—making million-token reasoning physically impractical on standard hardware.

Butterfly sidesteps this by routing information through depth instead of width: constant-size attention windows per layer, with full global connectivity emerging after logarithmic depth.

---

## How It Works

### 1. Chunking

The sequence is split into fixed-size chunks (default: 128 tokens).
No token pooling or compression—all tokens are preserved.

### 2. Structured Cross-Chunk Attention

At each layer, every chunk performs dense self-attention over:
- Its own tokens (local attention)
- One partner chunk selected by the butterfly routing pattern

This preserves exact token-level interactions while keeping attention windows constant.

### 3. Butterfly Routing Across Depth

Chunk pairings follow a hypercube connectivity pattern, changing each layer.

**Example with 8 chunks (3 layers needed):**

```text
Layer 0: Chunks differ in bit 0 → (0↔1), (2↔3), (4↔5), (6↔7)
Layer 1: Chunks differ in bit 1 → (0↔2), (1↔3), (4↔6), (5↔7)
Layer 2: Chunks differ in bit 2 → (0↔4), (1↔5), (2↔6), (3↔7)
```

After log₂(N/B) layers, information from every chunk has reached every other chunk. Multiple passes provide multi-step reasoning depth.

---

## Memory Scaling

This is where the architecture's potential lies—and where the current implementation falls short.

| Implementation | KV Cache Size | Status |
|----------------|---------------|--------|
| Standard Transformer | O(N × d × L) | — |
| Butterfly (current) | O(N × d × L) | ⚠️ Caches all visited chunks |
| Butterfly (with eviction) | O(B × d × L) | ❌ Not implemented |

**Current state:** The implementation caches all visited chunks for correctness during generation, resulting in the same O(N) memory scaling as standard transformers.

**Theoretical goal:** With a sliding-window eviction policy that discards chunks after they've propagated their information, memory can be bounded to O(B × d × L)—constant with respect to sequence length.

**Example projection (with eviction):** For 1M tokens with B=128, d=768:
- Standard Transformer: ~6GB KV cache
- Butterfly-LLM: ~35MB KV cache

The ~200× memory reduction would enable million-token streaming inference on consumer GPUs. This requires implementing proper cache eviction, which is future work.

---

## Streaming Inference

With proper eviction (not yet implemented), Butterfly would enable **logarithmic-time streaming generation**:

1. Only O(log N) chunk interactions recomputed per token
2. No global KV cache recomputation needed
3. Constant memory per forward pass

---

## Comparison with Other Long-Context Architectures

| Architecture | Memory | Global Expressivity | Streaming | Compression |
|--------------|--------|---------------------|-----------|-------------|
| Standard Transformer | O(N) | Exact | O(N) per token | None |
| Longformer | O(N) | Sparse approximation | O(N) per token | Sparse patterns |
| Mamba | O(1) | Implicit/compressed | O(1) per token | State compression |
| RWKV | O(1) | Implicit | O(1) per token | Recurrent compression |
| **Butterfly-LLM** | **O(1)*** | **Exact** | **O(log N) per token** | **None** |

*Theoretical with eviction; current implementation is O(N).

Butterfly aims to be the only architecture with **exact global attention, constant memory, and logarithmic streaming inference** simultaneously.

---

## What This Could Enable

- **Million-token reasoning** on consumer hardware
- **Long-horizon planning** agents with full context memory
- **Full-document understanding** without summarization
- **Real-time streaming LLMs** with bounded latency

---

## Limitations & Open Questions

- Multi-hop reasoning may degrade signal compared to direct attention
- Not yet benchmarked against established long-context models
- RoPE positions are chunk-local, not globally indexed (potential bug)
- Memory benefits require eviction policy (not implemented)
- Triton kernel is a simplified sketch, not production-ready

---

## Reference Model Specifications

| Parameter | Micro (Testing) | Full (Research) |
|-----------|-----------------|-----------------|
| Max context | 4,096 tokens | 65,536 tokens |
| Chunk size | 64 | 128 |
| Hidden dim | 256 | 768 |
| Attention heads | 4 | 12 |
| Local encoder layers | 2 | 4 |
| Butterfly layers | 6 per pass | 9 per pass |
| Butterfly passes | 2 | 5 |
| Refinement layers | 2 | 4 |
| Total depth | ~18 | ~53 |
| Vocab | 256 (byte-level) | 256 (byte-level) |
| Target hardware | Consumer GPU | A100 (training) / RTX (inference) |

**Theoretical max context:** With 9 butterfly layers per pass, the architecture supports up to 2⁹ × 128 = 65,536 tokens per pass. Extending to 1M+ tokens requires ~13 butterfly layers.

---

## Code Status

| Component | Status |
|-----------|--------|
| Core architecture | ⚠️ Implemented |
| Flash Attention integration | ⚠️ Implemented |
| Streaming inference cache | ⚠️ Basic (no eviction) |
| Triton kernel | ⚠️ Sketch only |
| RoPE positioning | ⚠️ Local (needs global fix) |
| Cache eviction policy | ❌ Not implemented |
| Pretrained weights | ❌ None |
| Benchmarks | ❌ None |

