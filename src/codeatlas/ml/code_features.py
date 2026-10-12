"""
ml/code_features.py

WHAT THIS FILE DOES, IN PLAIN TERMS
-------------------------------------
Git history tells us how BUSY a file is. This file looks at the CODE itself
and adds three new kinds of information per Python file:

1. loc              -> size: number of non-blank lines
2. complexity_total -> cyclomatic complexity of the whole file
                       (1 + number of decision points: if / elif / for /
                       while / except / and / or / ternary / comprehension
                       filters / match cases). More paths = harder to
                       understand and test.
   complexity_max   -> the most complex single function in the file
3. in_degree        -> how many OTHER files import this one
   pagerank         -> importance in the import graph (a file imported by
                       important files scores higher). Scaled so the
                       average file in a repo scores 1.0, which makes
                       repos of different sizes comparable.

THE IMPORTANT HONESTY RULE (time travel)
-----------------------------------------
Our ML test predicts the FUTURE from the PAST. So we must read the code as
it looked on the cutoff date, NOT as it looks today. Today's code already
contains later bug fixes; using it would leak the answer into the inputs.

We do this safely: `git archive <commit>` streams the .py files as they
were at that commit into a temporary folder. Your real working folder is
never touched, and the temporary folder is deleted afterwards.
"""

from __future__ import annotations

import subprocess
import tarfile
import tempfile
from collections import Counter
from pathlib import Path, PurePosixPath

import tree_sitter_python as tspython
from tree_sitter import Language, Parser

from codeatlas.graph.builder import build_dependency_graph
from codeatlas.ingestion.repo_walker import parse_repository

# tree-sitter node types that add one independent path through the code.
DECISION_NODES = {
    "if_statement", "elif_clause", "for_statement", "while_statement",
    "except_clause", "conditional_expression", "boolean_operator",
    "for_in_clause", "if_clause", "case_clause",
}


def _git(repo: str, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return result.stdout.strip()


def commit_at(repo: str, cutoff: str) -> str | None:
    """The last commit made before the cutoff date (same date logic as git log --before)."""
    return _git(repo, "rev-list", "-1", f"--before={cutoff}", "HEAD") or None


def extract_py_files_at_commit(repo: str, commit: str, dest: str) -> int:
    """Stream the repo as of `commit` and write only its .py files into `dest`."""
    proc = subprocess.Popen(
        ["git", "archive", "--format=tar", commit],
        cwd=repo,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    written = 0
    with tarfile.open(fileobj=proc.stdout, mode="r|") as tar:
        for member in tar:
            if not member.isfile() or not member.name.endswith(".py"):
                continue
            parts = PurePosixPath(member.name).parts
            if member.name.startswith("/") or ".." in parts:  # path-traversal guard
                continue
            try:
                target = Path(dest).joinpath(*parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(tar.extractfile(member).read())
                written += 1
            except OSError:
                continue  # e.g. a filename Windows cannot store; skip, never crash
    proc.stdout.close()
    if proc.wait() != 0:
        raise RuntimeError(f"git archive failed for commit {commit} in {repo}")
    return written


def _count_decisions(node) -> int:
    count = 0
    stack = [node]
    while stack:
        n = stack.pop()
        if n.type in DECISION_NODES:
            count += 1
        stack.extend(n.children)
    return count


def file_metrics(parser: Parser, source: bytes) -> dict:
    root = parser.parse(source).root_node
    total = 1
    functions = []
    stack = [root]
    while stack:
        n = stack.pop()
        if n.type in DECISION_NODES:
            total += 1
        if n.type == "function_definition":
            functions.append(n)
        stack.extend(n.children)
    max_fn = max((1 + _count_decisions(fn) for fn in functions), default=0)
    loc = sum(1 for line in source.splitlines() if line.strip())
    return {"loc": loc, "complexity_total": total, "complexity_max": max_fn}


def pagerank(edges: dict[str, set[str]], nodes: list[str],
             damping: float = 0.85, iterations: int = 50, tol: float = 1e-9) -> dict[str, float]:
    """
    Plain PageRank. Edge A -> B means "A imports B", so B earns rank from A.
    Returned scores are multiplied by the number of files, so the average is 1.0.
    """
    n = len(nodes)
    if n == 0:
        return {}
    node_set = set(nodes)
    out = {v: [t for t in edges.get(v, ()) if t in node_set] for v in nodes}
    rank = {v: 1.0 / n for v in nodes}
    for _ in range(iterations):
        dangling = sum(rank[v] for v in nodes if not out[v])
        new = {v: (1 - damping) / n + damping * dangling / n for v in nodes}
        for v in nodes:
            if out[v]:
                share = damping * rank[v] / len(out[v])
                for t in out[v]:
                    new[t] += share
        diff = sum(abs(new[v] - rank[v]) for v in nodes)
        rank = new
        if diff < tol:
            break
    return {v: rank[v] * n for v in nodes}


def compute_code_features(repo: str, cutoff: str) -> dict[str, dict]:
    """
    Returns {relative_posix_path: {loc, complexity_total, complexity_max,
    in_degree, pagerank}} for every .py file as it existed at the cutoff.
    Paths use forward slashes, matching `git ls-files`.
    """
    commit = commit_at(repo, cutoff)
    if commit is None:
        return {}

    parser = Parser(Language(tspython.language()))
    features: dict[str, dict] = {}

    with tempfile.TemporaryDirectory(prefix="codeatlas_", ignore_cleanup_errors=True) as tmp:
        extract_py_files_at_commit(repo, commit, tmp)

        parsed = parse_repository(tmp)
        graph = build_dependency_graph(parsed.files, tmp)

        paths = [pf.path for pf in parsed.files]
        in_degree: Counter = Counter()
        for targets in graph.edges.values():
            for t in targets:
                in_degree[t] += 1
        ranks = pagerank(graph.edges, paths)

        for p in paths:
            rel = Path(p).relative_to(tmp).as_posix()
            metrics = file_metrics(parser, Path(p).read_bytes())
            metrics["in_degree"] = in_degree.get(p, 0)
            metrics["pagerank"] = round(ranks.get(p, 1.0), 6)
            features[rel] = metrics

    return features


if __name__ == "__main__":
    import json
    import sys

    repo_arg = sys.argv[1] if len(sys.argv) > 1 else "."
    cutoff_arg = sys.argv[2] if len(sys.argv) > 2 else "2025-01-01"
    feats = compute_code_features(repo_arg, cutoff_arg)
    print(f"{len(feats)} files at cutoff {cutoff_arg}\n")
    top = sorted(feats.items(), key=lambda kv: -kv[1]["pagerank"])[:10]
    print(json.dumps(dict(top), indent=2))
