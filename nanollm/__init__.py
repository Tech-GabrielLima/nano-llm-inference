"""nanollm — a from-scratch GPT-2 inference engine (NumPy + Triton).

    import nanollm
    engine, model, tok = nanollm.load_engine("models/gpt2")
    print(engine.generate("The capital of France is", max_new_tokens=10, greedy=True)[0])
"""
from __future__ import annotations

import os

from .config import GPT2Config, GPT2_SIZES
from .tokenizer import GPT2Tokenizer
from .loader import download_gpt2, load_weights, param_bytes
from .model import GPT2, KVCache
from .generate import Engine

__all__ = [
    "GPT2Config", "GPT2_SIZES", "GPT2Tokenizer", "GPT2", "KVCache", "Engine",
    "download_gpt2", "load_weights", "param_bytes", "load_engine",
]


def load_engine(model_dir="models/gpt2", attn_impl="naive", download=False):
    if download:
        download_gpt2(model_dir=model_dir)
    cfg = GPT2Config.from_json(os.path.join(model_dir, "config.json"))
    weights = load_weights(model_dir)
    tok = GPT2Tokenizer(os.path.join(model_dir, "vocab.json"),
                        os.path.join(model_dir, "merges.txt"))
    model = GPT2(cfg, weights, attn_impl=attn_impl)
    return Engine(model, tok), model, tok
