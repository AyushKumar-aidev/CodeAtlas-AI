"""
rag/chunker.py

Cuts a repository into small, meaningful pieces ("chunks") that can be searched:
one chunk per function/method, one per class header, one per file's top-level
code, plus documentation files. Small chunks mean the LLM is shown only the
few relevant pieces, not whole files.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from codeatlas.ingestion.repo_walker import IGNORED_DIRS
from codeatlas.parsing.python_parser import PythonParser

MAX_FILE_BYTES = 400_000
MAX_CHUNK_LINES = 80          # longer functions are split into windows
MAX_DOC_FILES = 60
DOC_EXT = {".md", ".rst", ".txt"}
SKIP_DOCS = {"license", "licence", "changes", "changelog", "authors", "notice", "copying"}


@dataclass
class Chunk:
    path: str          # repo-relative, forward slashes
    start_line: int
    end_line: int
    name: str
    kind: str          # source | docs | tests
    text: str


def _is_test(rel: str) -> bool:
    low = rel.lower()
    parts = low.split("/")
    return any(p in ("test", "tests") for p in parts[:-1]) or parts[-1].startswith("test_") or parts[-1].endswith("_test.py")


def _windows(lines, start, end, name, rel, kind):
    out = []
    i = start
    while i <= end:
        j = min(end, i + MAX_CHUNK_LINES - 1)
        text = "\n".join(lines[i - 1:j])
        if text.strip():
            label = name if (i == start and j == end) else f"{name} (part {len(out) + 1})"
            out.append(Chunk(rel, i, j, label, kind, text))
        i = j + 1
    return out


def _safe_read(path: Path, root: Path):
    try:
        if path.is_symlink():
            return None
        resolved = path.resolve()
        if root not in resolved.parents and resolved != root:
            return None
        if path.stat().st_size > MAX_FILE_BYTES:
            return None
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _files(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in IGNORED_DIRS)
        for f in sorted(filenames):
            yield Path(dirpath) / f


def chunk_python(path: Path, rel: str, text: str, parser) -> list[Chunk]:
    lines = text.splitlines()
    if not lines:
        return []
    kind = "tests" if _is_test(rel) else "source"
    try:
        parsed = parser.parse_file(str(path))
    except Exception:  # unreadable/odd file: fall back to plain windows
        return _windows(lines, 1, len(lines), Path(rel).name, rel, kind)

    covered = [False] * (len(lines) + 2)
    chunks: list[Chunk] = []
    funcs = sorted(parsed.functions, key=lambda f: (f.start_line, -f.end_line))
    for f in funcs:
        s, e = f.start_line, min(f.end_line, len(lines))
        # a method nested in a bigger function would be double-counted; keep outermost + inner both is noisy
        if all(covered[s:e + 1]):
            continue
        chunks += _windows(lines, s, e, f.name, rel, kind)
        for n in range(s, e + 1):
            covered[n] = True
    for c in parsed.classes:
        s = c.start_line
        e = s
        while e <= min(c.end_line, len(lines)) and not covered[e] and e - s < 25:
            e += 1
        e -= 1
        if e >= s:
            chunks += _windows(lines, s, e, f"class {c.name}", rel, kind)
            for n in range(s, e + 1):
                covered[n] = True
    # whatever is left: imports, constants, top-level code
    n = 1
    while n <= len(lines):
        if covered[n]:
            n += 1
            continue
        m = n
        while m + 1 <= len(lines) and not covered[m + 1]:
            m += 1
        seg = "\n".join(lines[n - 1:m])
        if seg.strip():
            chunks += _windows(lines, n, m, "module level", rel, kind)
        n = m + 1
    chunks.sort(key=lambda c: c.start_line)
    return chunks


def chunk_docs(rel: str, text: str) -> list[Chunk]:
    lines = text.splitlines()
    return _windows(lines, 1, len(lines), Path(rel).name, rel, "docs") if lines else []


def chunk_repository(root_path: str | Path) -> list[Chunk]:
    root = Path(root_path).resolve()
    parser = PythonParser()
    chunks: list[Chunk] = []
    docs: list[tuple[int, str, str]] = []
    for path in _files(root):
        rel = path.relative_to(root).as_posix()
        suffix = path.suffix.lower()
        if suffix == ".py":
            text = _safe_read(path, root)
            if text is not None:
                chunks += chunk_python(path, rel, text, parser)
        elif (suffix in DOC_EXT and path.stem.lower() not in SKIP_DOCS
              and not any(part.startswith(".") for part in rel.split("/")[:-1])):
            text = _safe_read(path, root)
            if text is not None:
                docs.append((rel.count("/"), rel, text))
    docs.sort()  # shallow docs (README) first
    for _, rel, text in docs[:MAX_DOC_FILES]:
        chunks += chunk_docs(rel, text)
    return chunks
