"""Plot the benchmark / perplexity CSVs into results/*.png."""
import csv
import os
import sys

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit("matplotlib required: pip install matplotlib")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")


def read(name):
    p = os.path.join(RES, name)
    if not os.path.exists(p):
        print(f"  (skip {name})")
        return None
    with open(p) as f:
        return list(csv.DictReader(f))


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(RES, name), dpi=130)
    plt.close(fig)
    print(f"  wrote results/{name}")


def plot_benchmark():
    rows = read("benchmark.csv")
    if not rows:
        return
    kv = {r["label"]: float(r["tok_per_s"]) for r in rows if r["group"] == "kvcache"}
    bt = [(int(r["label"][2:]), float(r["tok_per_s"])) for r in rows if r["group"] == "batching"]
    mem = {r["label"]: float(r["mb"]) for r in rows if r["group"] == "memory"}

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    if kv:
        a = axes[0]
        bars = a.bar(["no-cache", "KV-cache"], [kv.get("no-cache", 0), kv.get("cache", 0)],
                     color=["#bbb", "#4C72B0"])
        a.bar_label(bars, fmt="%.0f")
        a.set_title(f"KV-cache decode speed ({kv.get('cache',0)/max(kv.get('no-cache',1),1e-9):.1f}x)")
        a.set_ylabel("tokens / s"); a.grid(axis="y", alpha=0.3)
    if bt:
        a = axes[1]
        bt.sort()
        a.plot([b for b, _ in bt], [t for _, t in bt], "-o", color="#55A868", lw=2)
        a.set_title("Static batching throughput"); a.set_xlabel("batch size")
        a.set_ylabel("total tokens / s"); a.grid(alpha=0.3)
    if mem:
        a = axes[2]
        order = ["fp32", "int8", "int4"]
        bars = a.bar(order, [mem.get(k, 0) for k in order], color=["#C44E52", "#4C72B0", "#8172B3"])
        a.bar_label(bars, fmt="%.0f MB")
        a.set_title("Model memory (parameters)"); a.set_ylabel("MB"); a.grid(axis="y", alpha=0.3)
    save(fig, "benchmark.png")


def plot_perplexity():
    rows = read("perplexity.csv")
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    names = [r["precision"] for r in rows]
    ppl = [float(r["perplexity"]) for r in rows]
    mem = [float(r["mem_mb"]) for r in rows]
    sc = ax.scatter(mem, ppl, s=140, c=["#C44E52", "#4C72B0", "#8172B3"][:len(rows)], zorder=3)
    for n, m, p in zip(names, mem, ppl):
        ax.annotate(f"{n}\nppl={p:.1f}", (m, p), textcoords="offset points",
                    xytext=(8, 8), fontsize=10)
    ax.set_xlabel("model memory (MB)"); ax.set_ylabel("perplexity (lower=better)")
    ax.set_title("Quantization: quality vs memory")
    ax.grid(alpha=0.3)
    save(fig, "perplexity.png")


if __name__ == "__main__":
    print("plotting into results/ ...")
    plot_benchmark()
    plot_perplexity()
    print("done.")
