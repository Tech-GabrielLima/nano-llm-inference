"""Correctness tests for the inference engine.

Unit tests (no weights) run first; the model tests load real GPT-2 weights from
models/gpt2 (run scripts/download_model.py first if missing).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import nanollm
from nanollm.attention import sdpa_naive, sdpa_flash
from nanollm.quant import quantize_weight
from nanollm.tokenizer import GPT2Tokenizer

FAIL = 0
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "gpt2")


def check(name, ok, detail=""):
    global FAIL
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")
    if not ok:
        FAIL += 1


def relerr(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    return float(np.linalg.norm(a - b) / (np.linalg.norm(a) + np.linalg.norm(b) + 1e-9))


print("nanollm — correctness tests\n--- units ---")

# 1) tokenizer roundtrip + canonical ids
tok = GPT2Tokenizer(os.path.join(MODEL_DIR, "vocab.json"), os.path.join(MODEL_DIR, "merges.txt"))
s = "The quick brown fox, jumping! 123."
check("tokenizer roundtrip", tok.decode(tok.encode(s)) == s)
check("tokenizer canonical ids", tok.encode(" the") == [262] and tok.encode("Hello") == [15496])

# 2) flash attention == naive attention
rng = np.random.default_rng(0)
q = rng.standard_normal((2, 4, 16, 8)).astype(np.float32)
k = rng.standard_normal((2, 4, 16, 8)).astype(np.float32)
v = rng.standard_normal((2, 4, 16, 8)).astype(np.float32)
check("flash == naive (prefill)", relerr(sdpa_naive(q, k, v), sdpa_flash(q, k, v, block=5)) < 1e-5,
      f"rel_err={relerr(sdpa_naive(q, k, v), sdpa_flash(q, k, v, block=5)):.2e}")
# decode step: 1 query against 16 keys with an offset
q1 = rng.standard_normal((2, 4, 1, 8)).astype(np.float32)
check("flash == naive (decode)", relerr(sdpa_naive(q1, k, v, causal_offset=15),
                                        sdpa_flash(q1, k, v, causal_offset=15, block=5)) < 1e-5)

# 3) quantization round-trip error
W = rng.standard_normal((256, 128)).astype(np.float32)
e8 = relerr(W, quantize_weight(W, 8).dequant())
e4 = relerr(W, quantize_weight(W, 4).dequant())
check("INT8 quant error small", e8 < 5e-3, f"rel_err={e8:.2e}")
check("INT4 quant error bounded", e4 < 8e-2, f"rel_err={e4:.2e}")

# --- model tests (need weights) ---
if not os.path.exists(os.path.join(MODEL_DIR, "model.safetensors")):
    print("\n[SKIP] model tests — run scripts/download_model.py first")
else:
    print("\n--- model ---")
    engine, model, tok = nanollm.load_engine(MODEL_DIR)

    # 4) KV-cache must give identical greedy output to the no-cache path
    _, g_cache = engine.generate("The meaning of life is", max_new_tokens=12, greedy=True, use_cache=True)
    _, g_nocache = engine.generate("The meaning of life is", max_new_tokens=12, greedy=True, use_cache=False)
    check("KV-cache == no-cache (greedy)", g_cache == g_nocache, f"{g_cache[0][:5]}...")

    # 5) flash-attention model path matches naive model path
    model.attn_impl = "flash"
    _, g_flash = engine.generate("The meaning of life is", max_new_tokens=12, greedy=True, use_cache=True)
    model.attn_impl = "naive"
    check("flash-attn model == naive model", g_flash == g_cache)

    # 6) static batching: a prompt in a batch matches it generated alone
    prompts = ["The capital of France is", "Water is made of"]
    outs_batch, _ = engine.generate(prompts, max_new_tokens=10, greedy=True)
    out_solo, _ = engine.generate(prompts[1], max_new_tokens=10, greedy=True)
    check("static batch == solo (per row)", outs_batch[1] == out_solo, repr(outs_batch[1]))

print()
print("ALL TESTS PASSED" if FAIL == 0 else f"{FAIL} TEST(S) FAILED")
sys.exit(0 if FAIL == 0 else 1)
