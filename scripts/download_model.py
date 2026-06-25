"""Download GPT-2 weights/tokenizer into models/<name>/.

    python scripts/download_model.py            # gpt2 (124M)
    python scripts/download_model.py gpt2-medium
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanollm.loader import download_gpt2

if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "gpt2"
    d = download_gpt2(model=model, model_dir=os.path.join("models", model))
    print(f"ready: {d}")
