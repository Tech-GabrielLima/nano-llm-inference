"""Quantization quality vs memory: perplexity of fp32 / INT8 / INT4 on a fixed
passage, alongside each model's parameter memory. Writes results/perplexity.csv.

    python scripts/eval_perplexity.py
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import nanollm
from nanollm.metrics import perplexity
from nanollm.quant import quantize_model, model_bytes

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

# A standard short evaluation passage (public-domain text).
TEXT = (
    "Artificial intelligence is the simulation of human intelligence processes "
    "by machines, especially computer systems. These processes include learning, "
    "reasoning, and self-correction. Machine learning is a subset of AI that "
    "enables systems to learn from data without being explicitly programmed. "
    "Deep learning, in turn, uses neural networks with many layers to model "
    "complex patterns in large amounts of data."
)


def main():
    engine, model, tok = nanollm.load_engine(os.path.join("models", "gpt2"))
    ids = tok.encode(TEXT)
    print(f"eval tokens: {len(ids)}\n")

    base_w = model.w
    base_bytes = model_bytes(base_w)
    rows = [["precision", "perplexity", "mem_mb", "compression"]]
    print(f"{'precision':9s} {'perplexity':>11s} {'mem(MB)':>9s} {'vs fp32':>8s}")
    for name, bits in (("fp32", None), ("int8", 8), ("int4", 4)):
        model.w = base_w if bits is None else quantize_model(base_w, bits=bits)
        ppl = perplexity(model, ids)
        mb = model_bytes(model.w) / 1e6
        comp = base_bytes / model_bytes(model.w)
        print(f"{name:9s} {ppl:11.3f} {mb:9.1f} {comp:7.2f}x")
        rows.append([name, f"{ppl:.4f}", f"{mb:.1f}", f"{comp:.3f}"])
    model.w = base_w

    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "perplexity.csv"), "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"\nwrote {os.path.join(RESULTS, 'perplexity.csv')}")


if __name__ == "__main__":
    main()
