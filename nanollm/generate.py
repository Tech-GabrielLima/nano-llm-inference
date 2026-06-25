"""Autoregressive generation: sampling, KV-cache decode loop, and static
batching (left-padding + key masking so prompts of different lengths batch
correctly)."""
from __future__ import annotations

import numpy as np

EOS = 50256  # <|endoftext|>, also used as the pad id


def _softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def sample_logits(logits, temperature=1.0, top_k=None, greedy=False, rng=None):
    """logits: (B, vocab) -> next token ids (B,)."""
    if greedy or temperature <= 0:
        return logits.argmax(axis=-1)
    rng = rng or np.random.default_rng()
    logits = logits / temperature
    if top_k:
        kth = np.sort(logits, axis=-1)[:, -top_k][:, None]
        logits = np.where(logits < kth, -1e30, logits)
    probs = _softmax(logits)
    return np.array([rng.choice(probs.shape[-1], p=probs[i]) for i in range(probs.shape[0])])


class Engine:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tok = tokenizer

    def generate(self, prompts, max_new_tokens=20, temperature=1.0, top_k=None,
                 greedy=False, use_cache=True, seed=0, stop_at_eos=True):
        single = isinstance(prompts, str)
        if single:
            prompts = [prompts]
        rng = np.random.default_rng(seed)

        enc = [self.tok.encode(p) for p in prompts]
        lens = [len(e) for e in enc]
        B, maxlen = len(enc), max(lens)

        # left-pad
        ids = np.full((B, maxlen), EOS, dtype=np.int64)
        valid = np.zeros((B, maxlen), dtype=bool)
        for i, e in enumerate(enc):
            ids[i, maxlen - len(e):] = e
            valid[i, maxlen - len(e):] = True
        pos = np.cumsum(valid, axis=1) - 1
        pos[pos < 0] = 0
        cur_pos = np.array(lens, dtype=np.int64) - 1  # last real position, per row

        generated = [[] for _ in range(B)]
        done = np.zeros(B, dtype=bool)

        if use_cache:
            cache = self.model.make_cache()
            logits = self.model.forward(ids, cache=cache, last_only=True,
                                        position_ids=pos, key_valid=valid)[:, -1, :]
            key_valid = valid
            for step in range(max_new_tokens):
                nxt = sample_logits(logits, temperature, top_k, greedy, rng)
                _record(generated, done, nxt, stop_at_eos)
                if done.all():
                    break
                cur_pos += 1
                key_valid = np.concatenate([key_valid, np.ones((B, 1), bool)], axis=1)
                logits = self.model.forward(nxt[:, None], cache=cache, last_only=True,
                                            position_ids=cur_pos[:, None],
                                            key_valid=key_valid)[:, -1, :]
        else:
            # no KV-cache: re-encode the whole sequence every step (the baseline)
            seq, vmask = ids, valid
            posids = pos
            for step in range(max_new_tokens):
                logits = self.model.forward(seq, last_only=True,
                                            position_ids=posids, key_valid=vmask)[:, -1, :]
                nxt = sample_logits(logits, temperature, top_k, greedy, rng)
                _record(generated, done, nxt, stop_at_eos)
                if done.all():
                    break
                seq = np.concatenate([seq, nxt[:, None]], axis=1)
                vmask = np.concatenate([vmask, np.ones((B, 1), bool)], axis=1)
                cur_pos += 1
                posids = np.concatenate([posids, cur_pos[:, None]], axis=1)

        texts = [self.tok.decode(g) for g in generated]
        full = [p + t for p, t in zip(prompts, texts)]
        return (full[0] if single else full), generated


def _record(generated, done, nxt, stop_at_eos):
    for i, t in enumerate(nxt):
        if done[i]:
            continue
        if stop_at_eos and t == EOS:
            done[i] = True
            continue
        generated[i].append(int(t))
