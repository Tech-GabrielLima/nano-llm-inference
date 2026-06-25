"""Weight-only quantization (INT8 / INT4) for the linear layers.

Symmetric, per-output-channel quantization:  W ≈ q · scale, where q are small
integers and `scale` is one float per output column. This is the scheme behind
LLM.int8()/GPTQ-style weight-only quant: activations stay FP32, only the big
weight matrices are stored as integers — cutting model memory ~4× (INT8) or ~8×
(INT4) with little quality loss.

INT4 values (-7..7) are packed two-per-byte so the stored size is really halved.
On CPU we dequantize before the matmul (the memory win is real; the *speed* win
comes from a fused dequant-matmul GPU kernel — see nanollm/triton_kernels.py).
"""
from __future__ import annotations

import numpy as np

# the linear weight matrices to quantize (the FLOP/param-heavy ones)
QUANT_SUFFIXES = ("attn.c_attn.weight", "attn.c_proj.weight",
                  "mlp.c_fc.weight", "mlp.c_proj.weight")


class QTensor:
    """A quantized 2D weight of logical shape (in_features, out_features)."""

    def __init__(self, q_packed, scale, bits, shape):
        self.q_packed = q_packed     # int8 (bits=8) or uint8-packed nibbles (bits=4)
        self.scale = scale.astype(np.float32)  # (out,)
        self.bits = bits
        self.shape = tuple(shape)

    @property
    def nbytes(self):
        return self.q_packed.nbytes + self.scale.nbytes

    def dequant(self):
        if self.bits == 8:
            q = self.q_packed.astype(np.float32)
        else:  # unpack two int4 per byte
            in_f, out_f = self.shape
            flat = np.empty(in_f * out_f, dtype=np.float32)
            lo = (self.q_packed & 0x0F).astype(np.int8)
            hi = (self.q_packed >> 4).astype(np.int8)
            flat[0::2] = lo[: flat[0::2].size] - 8
            flat[1::2] = hi[: flat[1::2].size] - 8
            q = flat.reshape(in_f, out_f)
        return q * self.scale  # broadcast (out,) over rows

    def matmul(self, x):
        return x @ self.dequant()


def quantize_weight(W, bits=8):
    """W: (in, out) float32 -> QTensor (symmetric, per-output-column)."""
    in_f, out_f = W.shape
    qmax = (1 << (bits - 1)) - 1                       # int8:127, int4:7
    scale = np.maximum(np.abs(W).max(axis=0), 1e-8) / qmax   # (out,)
    q = np.round(W / scale).clip(-qmax, qmax).astype(np.int8)

    if bits == 8:
        return QTensor(q, scale, 8, W.shape)
    # pack INT4: map -7..7 -> 0..15, two nibbles per byte
    u = (q + 8).astype(np.uint8).reshape(-1)
    if u.size % 2:
        u = np.append(u, 0)
    packed = (u[0::2] | (u[1::2] << 4)).astype(np.uint8)
    return QTensor(packed, scale, 4, W.shape)


def quantize_model(weights, bits=8, suffixes=QUANT_SUFFIXES):
    """Return a new weights dict with the targeted linear weights quantized."""
    out = {}
    for k, v in weights.items():
        if any(k.endswith(s) for s in suffixes):
            out[k] = quantize_weight(v, bits)
        else:
            out[k] = v
    return out


def model_bytes(weights):
    total = 0
    for v in weights.values():
        total += v.nbytes
    return total
