"""GPT-2 forward pass in NumPy, from scratch, with a KV-cache.

Architecture (Radford et al. 2019), pre-LayerNorm:
    h = wte[ids] + wpe[pos]
    for each block:  h += attn(ln_1(h));  h += mlp(ln_2(h))
    logits = ln_f(h) @ wte.T            # output embedding tied to input

GPT-2's linear layers are Conv1D (weight stored as (in, out)), so every
projection is just `x @ W + b` — no transpose. Weight matrices may be plain
NumPy arrays or quantized `QTensor`s (nanollm/quant.py); `_mm` dispatches.
"""
from __future__ import annotations

import numpy as np

from .attention import sdpa


def gelu(x):
    # GPT-2's "gelu_new" (tanh approximation)
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


def layer_norm(x, g, b, eps):
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * g + b


class KVCache:
    """Per-layer key/value cache, grown along the time axis (B, H, T, hd)."""

    def __init__(self, n_layer):
        self.k = [None] * n_layer
        self.v = [None] * n_layer

    def length(self):
        return 0 if self.k[0] is None else self.k[0].shape[2]

    def append(self, layer, k, v):
        if self.k[layer] is None:
            self.k[layer], self.v[layer] = k, v
        else:
            self.k[layer] = np.concatenate([self.k[layer], k], axis=2)
            self.v[layer] = np.concatenate([self.v[layer], v], axis=2)
        return self.k[layer], self.v[layer]


class GPT2:
    def __init__(self, config, weights, attn_impl="naive"):
        self.cfg = config
        self.w = weights
        self.attn_impl = attn_impl

    # weight matmul that understands quantized tensors
    def _mm(self, x, key):
        w = self.w[key]
        if hasattr(w, "matmul"):      # QTensor
            return w.matmul(x)
        return x @ w

    def make_cache(self):
        return KVCache(self.cfg.n_layer)

    def forward(self, input_ids, cache=None, last_only=False,
                position_ids=None, key_valid=None):
        cfg, w = self.cfg, self.w
        input_ids = np.asarray(input_ids)
        if input_ids.ndim == 1:
            input_ids = input_ids[None, :]
        B, T = input_ids.shape
        past = cache.length() if cache is not None else 0

        if position_ids is None:
            pos = np.arange(past, past + T)
            wpe = w["wpe.weight"][pos]                          # (T,E)
        else:
            wpe = w["wpe.weight"][np.asarray(position_ids)]     # (B,T,E)
        h = w["wte.weight"][input_ids] + wpe                   # (B,T,E)

        for li in range(cfg.n_layer):
            h = self._block(h, li, cache, past, key_valid)

        h = layer_norm(h, w["ln_f.weight"], w["ln_f.bias"], cfg.layer_norm_epsilon)
        if last_only:
            h = h[:, -1:, :]
        logits = h @ w["wte.weight"].T                          # tied embedding
        return logits

    def _block(self, h, li, cache, past, key_valid):
        cfg, w = self.cfg, self.w
        p = f"h.{li}."
        a = self._attn(layer_norm(h, w[p + "ln_1.weight"], w[p + "ln_1.bias"],
                                  cfg.layer_norm_epsilon), li, cache, past, key_valid)
        h = h + a
        m = self._mlp(layer_norm(h, w[p + "ln_2.weight"], w[p + "ln_2.bias"],
                                 cfg.layer_norm_epsilon), li)
        return h + m

    def _attn(self, x, li, cache, past, key_valid=None):
        cfg = self.cfg
        B, T, E = x.shape
        H, hd = cfg.n_head, cfg.head_dim
        p = f"h.{li}.attn."

        qkv = self._mm(x, p + "c_attn.weight") + self.w[p + "c_attn.bias"]  # (B,T,3E)
        q, k, v = np.split(qkv, 3, axis=-1)

        def to_heads(t):
            return t.reshape(B, T, H, hd).transpose(0, 2, 1, 3)            # (B,H,T,hd)

        q, k, v = to_heads(q), to_heads(k), to_heads(v)
        if cache is not None:
            k, v = cache.append(li, k, v)                                  # grow cache

        o = sdpa(q, k, v, causal_offset=past, impl=self.attn_impl,
                 key_valid=key_valid)                                      # (B,H,T,hd)
        o = o.transpose(0, 2, 1, 3).reshape(B, T, E)
        return self._mm(o, p + "c_proj.weight") + self.w[p + "c_proj.bias"]

    def _mlp(self, x, li):
        p = f"h.{li}.mlp."
        h = gelu(self._mm(x, p + "c_fc.weight") + self.w[p + "c_fc.bias"])
        return self._mm(h, p + "c_proj.weight") + self.w[p + "c_proj.bias"]
