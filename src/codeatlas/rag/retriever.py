"""
rag/retriever.py

Picks the chunks to show the LLM: the best keyword matches, plus chunks from
files that the best matches IMPORT or are imported by (the dependency graph),
because understanding code usually needs its neighbours. Overview questions
("what does this project do?") also get the README.
"""

from __future__ import annotations

import re

from codeatlas.rag.chunker import Chunk
from codeatlas.rag.index import Index

_OVERVIEW = re.compile(r"\b(what is|what does|overview|purpose|about|architecture|structure|getting started|install|summary|summarize)\b", re.I)


_ONBOARD = re.compile(r"(start reading|where (do|should|to) .*start|new contributor|onboard|get started|getting started|"
                      r"entry ?point|main (file|module)|important (file|module)|how is .*(organi[sz]ed|structured)|"
                      r"architecture|overview|structure|read first)", re.I)


def _central_chunks(index: Index, graph: dict | None, n: int = 4) -> list[int]:
    """First chunk of the most-imported source files: where a newcomer should look."""
    if not graph:
        return []
    nodes = sorted(graph.get("nodes", []), key=lambda x: (-x.get("pagerank", 0), -x.get("in_degree", 0)))
    out = []
    for node in nodes:
        if node["id"].rsplit("/", 1)[-1] == "__init__.py" and len(out) >= 1:
            continue
        for j in index.by_path.get(node["id"], []):
            if index.chunks[j].kind == "source":
                out.append(j)
                break
        if len(out) >= n:
            break
    return out


def retrieve(index: Index, question: str, graph: dict | None, k: int = 8, max_chars: int = 24000) -> list[Chunk]:
    hits = index.search(question, k)
    picked: list[int] = [i for _, i in hits]

    if _OVERVIEW.search(question):
        for i, c in enumerate(index.chunks):
            if c.kind == "docs" and c.path.lower().startswith("readme") and i not in picked:
                picked.append(i)
                if sum(1 for j in picked if index.chunks[j].path.lower().startswith("readme")) >= 3:
                    break

    if _ONBOARD.search(question):
        for j in _central_chunks(index, graph):
            if j not in picked:
                picked.append(j)

    # neighbour expansion through the import graph (top 3 hits only)
    if graph and hits:
        edges = graph.get("edges", [])
        extra: list[int] = []
        for _, i in hits[:3]:
            path = index.chunks[i].path
            nbrs = [e["target"] for e in edges if e["source"] == path] + [e["source"] for e in edges if e["target"] == path]
            for nb in nbrs[:3]:
                cand = [j for j in index.by_path.get(nb, []) if index.chunks[j].kind == "source"]
                if cand:
                    extra.append(cand[0])
        for j in extra[:4]:
            if j not in picked:
                picked.append(j)

    out, total, seen = [], 0, set()
    for i in picked:
        if i in seen:
            continue
        seen.add(i)
        c = index.chunks[i]
        if total + len(c.text) > max_chars and out:
            break
        out.append(c)
        total += len(c.text)
    return out
