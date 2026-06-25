"""Generate text with the engine.

    python scripts/generate.py "The future of AI is" --max-new 40
    python scripts/generate.py "Once upon a time" --temperature 0.8 --top-k 40
    python scripts/generate.py "A," "B," "C," --max-new 20      # static batch
    python scripts/generate.py "..." --quant int8               # quantized weights
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import nanollm
from nanollm.quant import quantize_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompts", nargs="+")
    ap.add_argument("--model-dir", default="models/gpt2")
    ap.add_argument("--max-new", type=int, default=40)
    ap.add_argument("--temperature", type=float, default=0.0, help="0 = greedy")
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--quant", choices=["none", "int8", "int4"], default="none")
    ap.add_argument("--attn", choices=["naive", "flash"], default="naive")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    engine, model, tok = nanollm.load_engine(args.model_dir, attn_impl=args.attn)
    if args.quant != "none":
        model.w = quantize_model(model.w, bits=8 if args.quant == "int8" else 4)
        print(f"[quantized to {args.quant}]")

    prompts = args.prompts if len(args.prompts) > 1 else args.prompts[0]
    t0 = time.time()
    out, gen = engine.generate(prompts, max_new_tokens=args.max_new,
                               temperature=args.temperature,
                               top_k=args.top_k, greedy=(args.temperature == 0.0),
                               use_cache=not args.no_cache)
    dt = time.time() - t0
    n_tok = sum(len(g) for g in gen)
    outs = out if isinstance(out, list) else [out]
    for o in outs:
        print("─" * 60)
        print(o)
    print("─" * 60)
    print(f"{n_tok} tokens in {dt:.2f}s  ({n_tok / dt:.1f} tok/s)")


if __name__ == "__main__":
    main()
