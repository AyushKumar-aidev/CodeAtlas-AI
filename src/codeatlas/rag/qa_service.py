"""
rag/qa_service.py

Ties it together: build (and cache) the search index for a repo, retrieve the
relevant chunks, ask the LLM, return answer + sources. Also rate-limits, to
protect the free API quota. It never holds the job manager: the router passes
the finished analysis in.
"""

from __future__ import annotations

import re
import threading
import time
from collections import OrderedDict, deque
from datetime import date
from pathlib import Path

from codeatlas.config import Settings
from codeatlas.rag.chunker import chunk_repository
from codeatlas.rag.index import Index
from codeatlas.rag.llm import LLMError
from codeatlas.rag.prompt import build_prompt, repo_facts
from codeatlas.rag.retriever import retrieve


_CITE = re.compile(r"([\w./\-]+\.\w+):(\d+)(?:-(\d+))?")


def cited_sources(answer: str, chunks: list) -> list:
    """Chunks the answer refers to: by path:lines overlap, or by naming the file."""
    picked, seen = [], set()

    def add(c):
        if (c.path, c.start_line) not in seen:
            seen.add((c.path, c.start_line))
            picked.append(c)

    for m in _CITE.finditer(answer):
        path, a = m.group(1), int(m.group(2))
        b = int(m.group(3) or a)
        for c in chunks:
            if c.path == path and c.start_line <= b and a <= c.end_line:
                add(c)
    for c in chunks:  # file named without line numbers: keep its best-ranked chunk
        if c.path in answer and not any(p.path == c.path for p in picked):
            add(c)
    return picked


class QAUnavailable(Exception):
    pass


class RateLimited(Exception):
    pass


class QAService:
    MAX_CACHED = 3

    def __init__(self, settings: Settings, llm=None, clock=time.monotonic) -> None:
        self._s = settings
        self._llm = llm
        self._clock = clock
        self._cache: OrderedDict[tuple, Index] = OrderedDict()
        self._lock = threading.Lock()
        self._build_locks: dict[tuple, threading.Lock] = {}
        self._recent: deque[float] = deque()
        self._day = date.today()
        self._today = 0

    @property
    def enabled(self) -> bool:
        return self._llm is not None

    def _check_rate(self) -> None:
        with self._lock:
            now = self._clock()
            while self._recent and now - self._recent[0] > 60:
                self._recent.popleft()
            if date.today() != self._day:
                self._day, self._today = date.today(), 0
            if len(self._recent) >= self._s.qa_rate_per_minute:
                raise RateLimited("Too many questions in a minute. Please wait a little.")
            if self._today >= self._s.qa_daily_limit:
                raise RateLimited("The daily question limit was reached. Try again tomorrow.")
            self._recent.append(now)
            self._today += 1

    def _index(self, owner: str, name: str, head: str) -> Index:
        key = (owner.lower(), name.lower(), head)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            build_lock = self._build_locks.setdefault(key, threading.Lock())
        with build_lock:
            with self._lock:
                if key in self._cache:
                    return self._cache[key]
            root = Path(self._s.repos_dir) / owner.lower() / name.lower()
            if not root.is_dir():
                raise QAUnavailable("The repository files are no longer on the server. Analyse it again.")
            idx = Index(chunk_repository(root))
            with self._lock:
                self._cache[key] = idx
                while len(self._cache) > self.MAX_CACHED:
                    self._cache.popitem(last=False)
                self._build_locks.pop(key, None)
            return idx

    def ask(self, owner: str, name: str, result: dict, question: str, history: list[dict]) -> dict:
        if not self.enabled:
            raise QAUnavailable("Q&A is turned off: no GEMINI_API_KEY is configured on the server.")
        question = question.strip()
        if not question or len(question) > self._s.qa_max_question_chars:
            raise ValueError(f"Ask a question of 1 to {self._s.qa_max_question_chars} characters.")
        self._check_rate()
        idx = self._index(owner, name, result["summary"]["head_commit"])
        chunks = retrieve(idx, question, result.get("graph"), self._s.qa_top_k, self._s.qa_max_context_chars)
        system, contents = build_prompt(repo_facts(owner, name, result), chunks, question, history)
        text, model = self._llm.generate(system, contents)
        used = cited_sources(text, chunks) or chunks[:3]  # nothing cited: show the top matches
        sources = [{"path": c.path, "start_line": c.start_line, "end_line": c.end_line,
                    "name": c.name, "kind": c.kind} for c in used]
        return {"answer": text, "sources": sources[:8], "model": model}
