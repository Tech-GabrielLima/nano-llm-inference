"""Baseline comparison.

Correctness: if HuggingFace `transformers` is installed, compare nanollm's
logits against the reference GPT-2 implementation (the gold standard) on the
same input — they should match to float32 round-off.

Speed: prints nanollm's tok/s. For a production-engine baseline, compare against
llama.cpp or vLLM on a GPU (commands documented in the README); those require a
GPU and their own setup, so they are not run here.

    python scripts/compare_baseline.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import nanollm


def main():
    engine, model, tok = nanollm.load_engine(os.path.join("models", "gpt2"))
    prompt = "The theory of relativity was developed by"
    ids = tok.encode(prompt)

    try:
        import torch
        from transformers import GPT2LMHeadModel
        hf = GPT2LMHeadModel.from_pretrained("gpt2").eval()
        with torch.no_grad():
            ref = hf(torch.tensor([ids])).logits[0].numpy()
        ours = model.forward(np.array([ids]))[0]
        rel = np.linalg.norm(ours - ref) / np.linalg.norm(ref)
        print(f"[correctness] nanollm vs HuggingFace GPT-2 logits rel_err = {rel:.2e}")
        print(f"              argmax next-token agree: {ours[-1].argmax() == ref[-1].argmax()}")
    except ImportError:
        print("[correctness] transformers not installed — skipping reference check.")
        print("              (pip install transformers torch  to enable it)")

    # nanollm speed
    engine.generate(prompt, max_new_tokens=4, greedy=True, use_cache=True, stop_at_eos=False)
    t0 = time.perf_counter()
    _, gen = engine.generate(prompt, max_new_tokens=32, greedy=True, use_cache=True, stop_at_eos=False)
    dt = time.perf_counter() - t0
    n = sum(len(g) for g in gen)
    print(f"[speed] nanollm: {n/dt:.1f} tok/s (CPU, KV-cache)")
    print("\nGPU baselines (run on hardware, see README):")
    print("  llama.cpp:  ./llama-bench -m gpt2.gguf")
    print("  vLLM:       python -m vllm.entrypoints.openai.api_server --model gpt2")


if __name__ == "__main__":
    main()
