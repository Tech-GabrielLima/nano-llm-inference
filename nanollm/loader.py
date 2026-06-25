"""Download and load open-source GPT-2 weights into a dict of NumPy arrays.

No torch / transformers needed: weights are read straight from the HuggingFace
`model.safetensors` with the `safetensors` reader.
"""
from __future__ import annotations

import os
import urllib.request

import numpy as np
from safetensors import safe_open

_FILES = ["model.safetensors", "vocab.json", "merges.txt", "config.json"]
_BASE = "https://huggingface.co/{model}/resolve/main/{f}"


def download_gpt2(model="gpt2", model_dir=None):
    """Ensure the GPT-2 files exist locally; returns the directory."""
    model_dir = model_dir or os.path.join("models", model)
    os.makedirs(model_dir, exist_ok=True)
    for f in _FILES:
        dst = os.path.join(model_dir, f)
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            continue
        url = _BASE.format(model=model, f=f)
        print(f"downloading {url} ...")
        urllib.request.urlretrieve(url, dst)
    return model_dir


def load_weights(model_dir):
    """Load all parameter tensors as float32 NumPy arrays.

    Skips the per-layer causal-mask buffers (`attn.bias`, `attn.masked_bias`),
    which are not learnable parameters.
    """
    path = os.path.join(model_dir, "model.safetensors")
    weights = {}
    with safe_open(path, framework="numpy") as f:
        for k in f.keys():
            # skip the causal-mask buffers (".attn.bias"/".attn.masked_bias"),
            # but NOT the real projection bias "...attn.c_attn.bias"
            if k.endswith(".attn.bias") or k.endswith(".attn.masked_bias"):
                continue
            weights[k] = np.ascontiguousarray(f.get_tensor(k).astype(np.float32))
    return weights


def param_bytes(weights):
    return sum(a.nbytes for a in weights.values())
