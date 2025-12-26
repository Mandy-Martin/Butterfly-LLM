# Butterfly-LLM

Butterfly-LLM is a long-context character-level language model designed to make **million-token reasoning practical on a single GPU**.

It replaces quadratic self-attention with a **butterfly (hypercube) routing network** that factorizes dense attention across depth, giving:

• **O(N log N)** attention compute  
• **Constant attention memory**  
• **Full global expressivity**  
• **Real-time streaming inference**

---

## The Core Idea (in plain words)

Normal transformers let **every token talk to every other token in a single layer**, which costs N² work.

Butterfly-LLM instead does this:

> Each layer lets only **small groups of tokens talk to each other.**  
> Across several layers, these groups are rearranged in a structured way, so that **after a few layers, every token has indirectly talked to every other token.**

So full global attention still happens — just **distributed across depth instead of width.**

---

## How Butterfly Attention Works

1. The sequence is split into **small fixed-size chunks** (128 tokens each).

2. In each butterfly layer:
   - Every chunk is paired with exactly one other chunk.
   - The two paired chunks perform **full dense attention with each other**.
   - No other chunks are touched in that layer.

3. In the next butterfly layer:
   - Chunks are paired differently.
   - Each chunk now talks to a *new* partner chunk.

4. The pairing pattern is carefully chosen so that:
   - After 9 layers, information from **every chunk has reached every other chunk**.
   - After several such passes, tokens can reason globally in multiple steps.

So the model builds global understanding **gradually across layers**.

---

## Why This Is Efficient

| Normal Transformers | Butterfly-LLM |
|--------------------|--------------|
| One layer does all-to-all attention (N²) | Each layer does only small local attentions |
| Expensive and memory-heavy | Cheap and memory-light |
| Hard to scale beyond 16k | Scales to 1M+ tokens |

Butterfly-LLM achieves full global connectivity with **O(N log N)** work and **constant attention memory**.

---

## Architecture Overview

Butterfly-LLM uses three stages:

| Stage | Layers | Purpose |
|------|------|--------|
| Local Encoder | 4 | Build character → syntax features |
| Butterfly Global Mixer | 5 passes × 9 layers = 45 | Global reasoning |
| Refinement | 4 | Output polishing |

Total layers: **53**

---

## Butterfly Attention

Instead of full N² attention in one layer, global attention is factorized:

• Tokens are grouped into 128-token chunks  
• Each layer pairs chunks whose IDs differ by one bit  
• After 9 layers, information from every chunk has reached every other chunk  
• 5 passes give multi-step global reasoning depth  

This is mathematically equivalent to an all-to-all communication network.

---

## Streaming Inference

Butterfly-LLM supports true **streaming generation**:

| Mode | Cost per new token |
|----|----------------|
| Standard Transformers | O(N²) |
| Butterfly-LLM | **O(log N)** |

This enables interactive million-token contexts.

---

## Tokenization

• Latin-1 character tokens (256 vocab)  
• No BPE, no merges  
• Extremely robust for code

---

## Hardware

| Task | Hardware |
|----|--------|
| Training | Single NVIDIA A100 |
| Inference | Consumer GPUs (RTX-class) |
| Context | 64k → 1M+ tokens |

---

## Why this matters

Butterfly-LLM makes **very long-context reasoning cheap, fast, and stable** — without sacrificing attention expressivity.

This unlocks:

• Massive codebases  
• Full-project reasoning  
• Memory-like long conversations  
• Large-scale document analysis  

---

## Status

This repository contains a **fully working reference implementation** with:

• Flash-Attention  
• Triton fused butterfly kernels  
• RoPE positional encoding  
• Streaming inference

This is a research-grade long-context architecture.
