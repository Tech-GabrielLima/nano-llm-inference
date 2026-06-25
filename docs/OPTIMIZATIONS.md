# Inference optimizations — what each one does and why

Measured numbers below are from this repo's CPU run (`scripts/benchmark.py`,
`scripts/eval_perplexity.py`, GPT-2 124M). The *mechanisms* are the same on GPU;
the absolute throughput is not.

## 1. KV-cache — the single biggest decode win

Autoregressive decoding generates one token at a time. Without a cache, step *t*
re-runs attention over the whole prefix `0..t`, recomputing the keys and values
for every past token — **O(t²)** total work to emit a sequence.

But past keys/values never change. The **KV-cache** stores `K` and `V` per layer
and per step appends only the new token's row, so each step is **O(t)** and the
per-step cost is roughly constant. The model then does a single forward over a
**length-1** sequence (plus the cached attention).

> Measured: **39.1 tok/s with cache vs 10.0 without → 3.9× faster** for 32 new
> tokens, and the gap widens with sequence length. Implemented in
> [`model.py`](../nanollm/model.py) (`KVCache`) and exercised by the
> `KV-cache == no-cache` correctness test (identical greedy output).

It also splits latency into two regimes worth reporting separately:
- **TTFT** (time-to-first-token) = the prefill over the prompt (≈ 58 ms here);
- **per-token decode** latency (≈ 24 ms here) = what the user feels while streaming.

## 2. FlashAttention-style fused attention

Naive attention materializes the full `T×T` score matrix, softmaxes it, then
multiplies by `V`. That matrix is **O(T²)** memory traffic to/from HBM and is the
attention bottleneck at long context.

**FlashAttention** never materializes it. It tiles over the key/value sequence
and keeps an **online softmax** — a running max `m` and running normalizer `l` —
rescaling the partial output as each new K/V block arrives. The scores live only
in fast on-chip memory (SRAM/registers); HBM traffic drops to O(T). It is the
canonical *fused kernel*: QKᵀ, softmax and ·V are one kernel launch, not three.

This repo has it twice:
- [`attention.py::sdpa_flash`](../nanollm/attention.py) — the exact algorithm in
  NumPy, **validated equal to naive attention** (rel err ~1e-8) by the tests;
- [`triton_kernels.py::_flash_attn_fwd`](../nanollm/triton_kernels.py) — the same
  math as a fused Triton GPU kernel (the production path).

## 3. Weight-only quantization (INT8 / INT4)

LLMs are memory-bound at inference: most time is spent reading weights from
memory, not computing. Storing weights as small integers shrinks that traffic.

We use **symmetric, per-output-channel** quantization: `W ≈ q · scale`, with `q`
in int8 (−127..127) or int4 (−7..7) and one fp32 `scale` per output column.
Activations stay fp32; only the big linear weight matrices are quantized — the
LLM.int8()/GPTQ recipe. INT4 values are **packed two-per-byte** so storage really
halves again.

> Measured (GPT-2 124M), perplexity on a fixed passage vs parameter memory:
>
> | precision | perplexity | memory | vs fp32 |
> |-----------|-----------:|-------:|--------:|
> | fp32      | 18.06      | 498 MB | 1.00×   |
> | **INT8**  | **18.13**  | 243 MB | **2.05×** |
> | INT4      | 31.38      | 201 MB | 2.48×   |
>
> **INT8 is nearly free** (+0.4% perplexity) for half the memory; **INT4** trades
> real quality for more compression. (Total ratio is ~2× rather than ~4× because
> the 38M-parameter tied token embedding is left in fp32 — quantizing it too is a
> documented next step.)

On CPU we dequantize before the matmul (the **memory** win is real; there is no
compute win). The **speed** win comes from the **fused dequant-matmul** Triton
kernel ([`_dequant_matmul_int8`](../nanollm/triton_kernels.py)): the weight is
read as int8 and turned into fp *inside* the kernel, so the GEMM is fed by half/
quarter the memory bandwidth.

## 4. Static batching

A single decode step underutilizes the hardware (one length-1 matmul). **Static
batching** runs `B` sequences together so each step is a `(B × …)` GEMM — far
better arithmetic intensity. Prompts of different lengths are **left-padded** and
the padded keys are masked out of attention (`key_valid`), which keeps results
identical to running each prompt alone (verified by the `static batch == solo`
test).

> Measured total throughput: **38 → 50 → 53 tok/s** at batch 1 → 4 → 8 on CPU.
> The gain is modest on CPU (cores already saturated by one stream); on a GPU,
> where a single stream leaves the device idle, batching is the main throughput
> lever — exactly what *continuous batching* in vLLM/TensorRT-LLM exploits.

## 5. Putting it together

The optimizations compose: KV-cache makes each step cheap, batching fills the
hardware, fused attention removes the O(T²) memory wall at long context, and
quantization shrinks the weight traffic. An engine like vLLM or TensorRT-LLM is
these same ideas plus **continuous batching** (swap finished sequences out of the
batch mid-flight) and **paged KV-cache** (PagedAttention) — the natural next
steps from here.
