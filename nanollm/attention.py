"""Scaled dot-product attention — two NumPy implementations that must agree:

  * sdpa_naive : the textbook path. Materialises the full (Tq x Tk) score
                 matrix, softmaxes it, then multiplies by V. Memory is O(Tq*Tk).
  * sdpa_flash : the FlashAttention idea (Dao et al.) done on CPU — tile over
                 the key/value sequence and keep a running max + running
                 normaliser (online softmax), so the full score matrix is never
                 materialised. Memory is O(Tq*block). Same result as naive.

`sdpa_flash` mirrors what the Triton kernel (nanollm/triton_kernels.py) does on
the GPU; validating flash == naive here validates the algorithm the kernel runs.
"""
from __future__ import annotations

import numpy as np

NEG_INF = -1e30


def _causal_mask(Tq, Tk, offset):
    # query i (absolute position offset+i) may attend to key j iff j <= offset+i
    i = np.arange(Tq)[:, None]
    j = np.arange(Tk)[None, :]
    return j <= (offset + i)


def sdpa_naive(q, k, v, causal_offset=0, key_valid=None):
    """q,k,v: (B, H, T*, hd). key_valid: (B, Tk) bool mask of real (non-pad)
    keys, or None. Returns (B, H, Tq, hd)."""
    B, H, Tq, hd = q.shape
    Tk = k.shape[2]
    scale = 1.0 / np.sqrt(hd)
    scores = np.matmul(q, k.transpose(0, 1, 3, 2)) * scale          # (B,H,Tq,Tk)
    mask = _causal_mask(Tq, Tk, causal_offset)
    scores = np.where(mask, scores, NEG_INF)
    if key_valid is not None:
        scores = np.where(key_valid[:, None, None, :], scores, NEG_INF)
    scores -= scores.max(axis=-1, keepdims=True)
    e = np.exp(scores)
    attn = e / e.sum(axis=-1, keepdims=True)
    return np.matmul(attn, v)


def sdpa_flash(q, k, v, causal_offset=0, block=128):
    """Online-softmax (FlashAttention-style) attention; same result as naive."""
    B, H, Tq, hd = q.shape
    Tk = k.shape[2]
    scale = 1.0 / np.sqrt(hd)

    out = np.zeros((B, H, Tq, hd), dtype=np.float32)
    m = np.full((B, H, Tq, 1), NEG_INF, dtype=np.float32)   # running max
    l = np.zeros((B, H, Tq, 1), dtype=np.float32)           # running sum of exp
    qi = np.arange(Tq)[:, None]

    for start in range(0, Tk, block):
        end = min(start + block, Tk)
        kj = k[:, :, start:end, :]
        vj = v[:, :, start:end, :]
        s = np.matmul(q, kj.transpose(0, 1, 3, 2)) * scale          # (B,H,Tq,blk)
        keys = np.arange(start, end)[None, :]
        valid = keys <= (causal_offset + qi)                        # (Tq,blk)
        s = np.where(valid, s, NEG_INF)

        m_new = np.maximum(m, s.max(axis=-1, keepdims=True))
        p = np.exp(s - m_new)                                       # (B,H,Tq,blk)
        alpha = np.exp(m - m_new)                                   # rescale prev
        l = alpha * l + p.sum(axis=-1, keepdims=True)
        out = alpha * out + np.matmul(p, vj)
        m = m_new

    return out / np.maximum(l, 1e-20)


def sdpa(q, k, v, causal_offset=0, impl="naive", key_valid=None):
    # padded batches need explicit key masking -> use the naive path
    if impl == "flash" and key_valid is None:
        return sdpa_flash(q, k, v, causal_offset)
    return sdpa_naive(q, k, v, causal_offset, key_valid=key_valid)
