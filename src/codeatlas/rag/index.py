"""
rag/index.py

A small search engine over chunks: BM25 (the classic keyword-ranking formula)
with identifier-aware tokenizing, so the question "where is the request context
pushed?" matches `push_request_context`. No API or model is needed for search.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from codeatlas.rag.chunker import Chunk

STOP = set("""a an the is are was were be been being of to in on for and or not with as at by from this that
these those it its how what where when which who why do does did can could should would i you we they there
about into over than then so if me my our your repo repository code file files function functions class
use used using please explain tell show give""".split())
KIND_WEIGHT = {"source": 1.0, "docs": 0.85, "tests": 0.6}
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9]*")
_CAMEL = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+")


def tokenize(text: str) -> list[str]:
    """Words, plus the pieces of snake_case / camelCase identifiers."""
    out = []
    for raw in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text):
        whole = raw.lower().strip("_")
        pieces = [p.lower() for part in raw.split("_") for p in _CAMEL.findall(part)]
        if len(pieces) > 1 and whole:
            out.append(whole)
        out += pieces
    return [t for t in out if len(t) > 1 and t not in STOP]


class Index:
    k1, b = 1.5, 0.75

    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self._tf: list[Counter] = []
        self._len: list[int] = []
        df: Counter = Counter()
        for c in chunks:
            toks = tokenize(c.text) + tokenize(c.path) * 2 + tokenize(c.name) * 3
            tf = Counter(toks)
            self._tf.append(tf)
            self._len.append(max(1, len(toks)))
            df.update(tf.keys())
        n = max(1, len(chunks))
        self._avg = sum(self._len) / n if chunks else 1.0
        self._idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}
        self.by_path: dict[str, list[int]] = {}
        for i, c in enumerate(chunks):
            self.by_path.setdefault(c.path, []).append(i)

    def search(self, question: str, k: int = 8) -> list[tuple[float, int]]:
        q = tokenize(question)
        if not q:
            return []
        scored = []
        for i, tf in enumerate(self._tf):
            s = 0.0
            for t in set(q):
                f = tf.get(t)
                if f:
                    s += self._idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self._len[i] / self._avg))
            if s > 0:
                scored.append((s * KIND_WEIGHT.get(self.chunks[i].kind, 1.0), i))
        scored.sort(key=lambda t: -t[0])
        return scored[:k]
