"""Triton GPU kernels — the *fused* kernels of the engine.

Two kernels:
  * flash_attention      — FlashAttention-style fused attention forward: the
                           QKᵀ→softmax→·V pipeline is computed in one kernel with
                           online softmax, tiling over the K/V sequence so the
                           (T×T) score matrix never touches HBM. Same math as the
                           tested NumPy `attention.sdpa_flash`.
  * dequant_matmul_int8  — fused INT8 weight dequant + GEMM: the weight is loaded
                           as int8 and turned into fp inside the kernel (no
                           separate dequant pass / no fp weight in memory). This
                           is where weight-only quantization turns into *speed*.

These run on an NVIDIA GPU (Triton JIT-compiles to PTX). This repo's dev machine
has no GPU and Triton 2.1's CPU interpreter is unavailable, so they are exercised
on real hardware; their algorithms are validated on CPU by the NumPy equivalents
(`tests/test_nanollm.py`: flash == naive; quant round-trip).
"""
from __future__ import annotations

import numpy as np

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except ImportError:  # pragma: no cover
    HAVE_TRITON = False


def triton_available():
    if not HAVE_TRITON:
        return False
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


if HAVE_TRITON:

    @triton.jit
    def _flash_attn_fwd(
        Q, K, V, Out, sm_scale,
        stride_qb, stride_qh, stride_qm, stride_qk,
        stride_kb, stride_kh, stride_kn, stride_kk,
        stride_vb, stride_vh, stride_vn, stride_vk,
        stride_ob, stride_oh, stride_om, stride_ok,
        H, N_CTX,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
        CAUSAL: tl.constexpr,
    ):
        # one program == one (batch*head, block of BLOCK_M queries)
        start_m = tl.program_id(0)
        off_bh = tl.program_id(1)
        off_b = off_bh // H
        off_h = off_bh % H

        q_base = Q + off_b * stride_qb + off_h * stride_qh
        k_base = K + off_b * stride_kb + off_h * stride_kh
        v_base = V + off_b * stride_vb + off_h * stride_vh
        o_base = Out + off_b * stride_ob + off_h * stride_oh

        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, HEAD_DIM)

        q = tl.load(q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk,
                    mask=offs_m[:, None] < N_CTX, other=0.0)

        m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)   # running max
        l_i = tl.zeros([BLOCK_M], tl.float32)                 # running sum
        acc = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)       # running output

        n_end = (start_m + 1) * BLOCK_M if CAUSAL else N_CTX
        for start_n in range(0, n_end, BLOCK_N):
            n = start_n + offs_n
            k = tl.load(k_base + n[:, None] * stride_kn + offs_d[None, :] * stride_kk,
                        mask=n[:, None] < N_CTX, other=0.0)
            v = tl.load(v_base + n[:, None] * stride_vn + offs_d[None, :] * stride_vk,
                        mask=n[:, None] < N_CTX, other=0.0)
            qk = tl.dot(q, tl.trans(k)) * sm_scale            # (BLOCK_M, BLOCK_N)
            if CAUSAL:
                qk = tl.where(offs_m[:, None] >= n[None, :], qk, float("-inf"))
            else:
                qk = tl.where(n[None, :] < N_CTX, qk, float("-inf"))

            m_new = tl.maximum(m_i, tl.max(qk, 1))
            p = tl.exp(qk - m_new[:, None])
            alpha = tl.exp(m_i - m_new)
            l_i = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            m_i = m_new

        acc = acc / l_i[:, None]
        tl.store(o_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok,
                 acc, mask=offs_m[:, None] < N_CTX)

    @triton.jit
    def _dequant_matmul_int8(
        X, Wq, Scale, Out,
        M, N, K,
        stride_xm, stride_xk, stride_wk, stride_wn, stride_om, stride_on,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        # Out(M,N) = X(M,K) @ dequant(Wq(K,N)); Wq is int8, Scale is (N,)
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        acc = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
        for k0 in range(0, K, BLOCK_K):
            x = tl.load(X + offs_m[:, None] * stride_xm + (k0 + offs_k)[None, :] * stride_xk,
                        mask=offs_m[:, None] < M, other=0.0)
            wq = tl.load(Wq + (k0 + offs_k)[:, None] * stride_wk + offs_n[None, :] * stride_wn,
                         mask=offs_n[None, :] < N, other=0)
            acc += tl.dot(x, wq.to(tl.float32))               # int8 weights -> fp in-register
        scale = tl.load(Scale + offs_n, mask=offs_n < N, other=0.0)
        acc = acc * scale[None, :]                            # apply per-column scale
        tl.store(Out + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
                 acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def flash_attention(q, k, v, causal=True):
    """Host wrapper. q,k,v: torch CUDA tensors (B,H,T,hd). Returns (B,H,T,hd)."""
    if not triton_available():
        raise RuntimeError("Triton flash-attention needs an NVIDIA GPU. "
                           "Use the NumPy engine (attn_impl='flash') on CPU.")
    import torch
    B, H, T, hd = q.shape
    out = torch.empty_like(q)
    sm_scale = 1.0 / (hd ** 0.5)
    BLOCK_M = BLOCK_N = 64
    grid = (triton.cdiv(T, BLOCK_M), B * H)
    _flash_attn_fwd[grid](
        q, k, v, out, sm_scale,
        *q.stride(), *k.stride(), *v.stride(), *out.stride(),
        H, T, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=hd, CAUSAL=causal,
    )
    return out


def dequant_matmul_int8(x, wq, scale):
    """x: (M,K) torch CUDA; wq: (K,N) int8; scale: (N,). Returns (M,N) fp32."""
    if not triton_available():
        raise RuntimeError("Triton dequant-matmul needs an NVIDIA GPU.")
    import torch
    M, K = x.shape
    _, N = wq.shape
    out = torch.empty((M, N), device=x.device, dtype=torch.float32)
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _dequant_matmul_int8[grid](
        x, wq, scale, out, M, N, K,
        x.stride(0), x.stride(1), wq.stride(0), wq.stride(1), out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return out
