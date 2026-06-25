"""Throughput / latency / memory benchmark.

Measures, on the current device (CPU here; the same code runs on GPU):
  * KV-cache OFF vs ON   — decode speed (tok/s) and the speedup
  * prefill latency (TTFT) and per-token decode latency
  * static batching      — total tok/s as batch size grows
  * model memory         — fp32 vs INT8 vs INT4 parameter footprint

Writes results/benchmark.csv.
"""
import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import nanollm
from nanollm.quant import quantize_model, model_bytes

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def time_gen(engine, prompts, n_new, use_cache):
    t0 = time.perf_counter()
    _, gen = engine.generate(prompts, max_new_tokens=n_new, greedy=True,
                             use_cache=use_cache, stop_at_eos=False)
    dt = time.perf_counter() - t0
    ntok = sum(len(g) for g in gen)
    return dt, ntok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="models/gpt2")
    ap.add_argument("--new", type=int, default=32, help="tokens to generate")
    ap.add_argument("--prompt", default="In a shocking finding, scientists discovered")
    args = ap.parse_args()

    engine, model, tok = nanollm.load_engine(args.model_dir)
    rows = []
    print(f"device=CPU  model=gpt2  generating {args.new} tokens\n")

    # 1) KV-cache off vs on
    print("== KV-cache ==")
    for use_cache in (False, True):
        # warmup
        engine.generate(args.prompt, max_new_tokens=4, greedy=True, use_cache=use_cache, stop_at_eos=False)
        dt, ntok = time_gen(engine, args.prompt, args.new, use_cache)
        tps = ntok / dt
        tag = "cache" if use_cache else "no-cache"
        print(f"  {tag:9s}  {dt*1000:8.1f} ms total   {tps:7.2f} tok/s")
        rows.append(["kvcache", tag, args.new, dt, tps, ""])

    # speedup
    nc = next(r for r in rows if r[1] == "no-cache")
    c = next(r for r in rows if r[1] == "cache")
    print(f"  -> KV-cache speedup: {c[4]/nc[4]:.1f}x\n")

    # 2) latency split (cached): TTFT (prefill) vs per-token decode
    print("== latency (KV-cache) ==")
    cache = model.make_cache()
    ids = np.array([tok.encode(args.prompt)])
    t0 = time.perf_counter(); model.forward(ids, cache=cache, last_only=True); ttft = time.perf_counter() - t0
    last = ids[:, -1:]
    pos = np.array([[ids.shape[1]]])
    kv = np.ones((1, ids.shape[1] + 1), bool)
    t0 = time.perf_counter()
    for i in range(args.new):
        model.forward(last, cache=cache, last_only=True, position_ids=pos + i,
                      key_valid=np.ones((1, ids.shape[1] + 1 + i), bool))
    dec = (time.perf_counter() - t0) / args.new
    print(f"  TTFT (prefill {ids.shape[1]} tok): {ttft*1000:7.1f} ms")
    print(f"  per-token decode:             {dec*1000:7.1f} ms  ({1/dec:.1f} tok/s)\n")
    rows.append(["latency", "ttft_ms", ids.shape[1], ttft * 1000, "", ""])
    rows.append(["latency", "decode_ms", 1, dec * 1000, 1 / dec, ""])

    # 3) static batching throughput (equal-length prompts)
    print("== static batching (total tok/s) ==")
    for bs in (1, 4, 8):
        prompts = [args.prompt] * bs
        engine.generate(prompts, max_new_tokens=2, greedy=True, use_cache=True, stop_at_eos=False)
        dt, ntok = time_gen(engine, prompts, args.new, True)
        print(f"  batch={bs:2d}  {ntok:4d} tok  {dt*1000:8.1f} ms  {ntok/dt:7.2f} tok/s")
        rows.append(["batching", f"bs{bs}", ntok, dt, ntok / dt, ""])

    # 4) memory: fp32 vs int8 vs int4
    print("\n== model memory (parameters) ==")
    base = model_bytes(model.w)
    for name, bits in (("fp32", None), ("int8", 8), ("int4", 4)):
        w = model.w if bits is None else quantize_model(model.w, bits=bits)
        mb = model_bytes(w) / 1e6
        print(f"  {name:5s}  {mb:7.1f} MB   ({base / model_bytes(w):.2f}x smaller)")
        rows.append(["memory", name, "", "", "", mb])

    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "benchmark.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "label", "n", "seconds", "tok_per_s", "mb"])
        w.writerows(rows)
    print(f"\nwrote {os.path.join(RESULTS, 'benchmark.csv')}")


if __name__ == "__main__":
    main()
