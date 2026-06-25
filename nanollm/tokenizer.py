"""GPT-2 byte-level BPE tokenizer, implemented from scratch.

Two stages, exactly as in Radford et al. (2019):
  1. byte-level mapping  — every byte 0..255 is mapped to a printable unicode
     char, so any text (incl. arbitrary bytes) becomes a string of "safe" chars.
  2. BPE merges          — greedily merge the most frequent adjacent pair using
     the rank table from merges.txt, until no more merges apply.

Uses the `regex` module's GPT-2 pre-tokenization pattern when available;
otherwise falls back to an ASCII-oriented `re` pattern (fine for English demos).
"""
from __future__ import annotations

import json
from functools import lru_cache

try:
    import regex as _re
    _PAT = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    _HAVE_REGEX = True
except ImportError:  # pragma: no cover
    import re as _re
    _PAT = r"""'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+"""
    _HAVE_REGEX = False


@lru_cache()
def bytes_to_unicode():
    """Reversible map: byte (0..255) -> a printable unicode char."""
    bs = (list(range(ord("!"), ord("~") + 1)) +
          list(range(ord("\xa1"), ord("\xac") + 1)) +
          list(range(ord("\xae"), ord("\xff") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


def get_pairs(word):
    return set(zip(word[:-1], word[1:]))


class GPT2Tokenizer:
    def __init__(self, vocab_path, merges_path):
        with open(vocab_path, encoding="utf-8") as f:
            self.encoder = json.load(f)             # token-string -> id
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        with open(merges_path, encoding="utf-8") as f:
            merges = f.read().split("\n")[1:-1]     # drop header + trailing ""
        merges = [tuple(m.split()) for m in merges]
        self.bpe_ranks = {pair: i for i, pair in enumerate(merges)}
        self.pat = _re.compile(_PAT)
        self._cache = {}

    # --- BPE on a single pre-token --------------------------------------
    def _bpe(self, token):
        if token in self._cache:
            return self._cache[token]
        word = tuple(token)
        pairs = get_pairs(word)
        while pairs:
            bigram = min(pairs, key=lambda p: self.bpe_ranks.get(p, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word, i = [], 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                new_word.extend(word[i:j])
                if word[j] == first and j < len(word) - 1 and word[j + 1] == second:
                    new_word.append(first + second)
                    i = j + 2
                else:
                    new_word.append(word[j])
                    i = j + 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = get_pairs(word)
        out = " ".join(word)
        self._cache[token] = out
        return out

    # --- public API ------------------------------------------------------
    def encode(self, text):
        ids = []
        for tok in self.pat.findall(text):
            tok = "".join(self.byte_encoder[b] for b in tok.encode("utf-8"))
            ids.extend(self.encoder[bpe] for bpe in self._bpe(tok).split(" "))
        return ids

    def decode(self, ids):
        text = "".join(self.decoder[int(i)] for i in ids)
        return bytearray(self.byte_decoder[c] for c in text).decode("utf-8", errors="replace")

    @property
    def vocab_size(self):
        return len(self.encoder)
