"""Evaluation metrics: perplexity and a parameter-memory report."""
from __future__ import annotations

import numpy as np


def _log_softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=-1, keepdims=True))


def perplexity(model, token_ids, max_len=512):
    """Causal-LM perplexity over a single token sequence."""
    ids = np.asarray(token_ids).reshape(-1)[:max_len]
    logits = model.forward(ids[None, :])[0]            # (T, V)
    logp = _log_softmax(logits[:-1].astype(np.float64))
    tgt = ids[1:]
    nll = -logp[np.arange(len(tgt)), tgt]
    return float(np.exp(nll.mean()))


def param_memory_mb(weights):
    return sum(getattr(v, "nbytes", v.nbytes) for v in weights.values()) / 1e6
