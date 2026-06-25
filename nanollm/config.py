"""GPT-2 model configuration."""
from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass
class GPT2Config:
    vocab_size: int = 50257
    n_positions: int = 1024     # max context length
    n_embd: int = 768
    n_layer: int = 12
    n_head: int = 12
    layer_norm_epsilon: float = 1e-5

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    @classmethod
    def from_json(cls, path: str) -> "GPT2Config":
        with open(path) as f:
            c = json.load(f)
        return cls(
            vocab_size=c.get("vocab_size", 50257),
            n_positions=c.get("n_positions", 1024),
            n_embd=c.get("n_embd", 768),
            n_layer=c.get("n_layer", 12),
            n_head=c.get("n_head", 12),
            layer_norm_epsilon=c.get("layer_norm_epsilon", 1e-5),
        )


# Known GPT-2 family sizes (for reference / config-only construction).
GPT2_SIZES = {
    "gpt2":        dict(n_embd=768,  n_layer=12, n_head=12),   # 124M
    "gpt2-medium": dict(n_embd=1024, n_layer=24, n_head=16),   # 355M
    "gpt2-large":  dict(n_embd=1280, n_layer=36, n_head=20),   # 774M
    "gpt2-xl":     dict(n_embd=1600, n_layer=48, n_head=25),   # 1558M
}
