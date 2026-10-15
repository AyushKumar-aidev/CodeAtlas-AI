"""
services/analysis.py

THE ORCHESTRATOR: the one place that knows the pipeline order.

    parse every .py file  ->  build the import graph  ->  centrality + complexity
    ->  git-history features per file  ->  risk score  ->  ranked result

It only CALLS the components that already exist (repo_walker, builder,
code_features, git_history, risk_model). It contains no parsing or ML logic of
its own, so the API and any future command-line tool can share it.

The feature definitions here match the ones the model was trained on in
ml/build_dataset.py:
  - code features: loc, cyclomatic complexity, in-degree and PageRank, computed
    with the very same functions (file_metrics, pagerank) the training used;
  - git features: churn, author count, bug-fix-commit count. Training counted
    commits BEFORE a cutoff date. Here the "cutoff" is today: we score a file
    from its whole history so far, which is exactly the situation the model
    is meant to predict forward from.

Known limitations (stated, not hidden):
  - Python files only (JS/TS and C++ parsers are not written yet).
  - "Bug-fix commit" is a keyword heuristic on commit messages (see git_history.py).
  - risk_score is a ranking score, not a calibrated probability.
"""

from __future__ import annotations

import subprocess
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import tree_sitter_python as tspython
from tree_sitter import Language, Parser

from codeatlas.graph.builder import build_dependency_graph
from codeatlas.graph.git_history import extract_features_for_file
from codeatlas.ingestion.repo_walker import parse_repository
from codeatlas.ml.code_features import file_metrics, pagerank
from codeatlas.ml.risk_model import ModelError, load_model, score_files

GIT_WORKERS = 4  # parallel `git log` calls; each is a separate process


class AnalysisError(Exception):
    """The analysis cannot continue. The message is safe to show to the user."""


def _head_commit(repo: Path) -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True,
        text=True, encoding="utf-8", errors="replace",
    )
    return proc.stdout.strip()


def _round(value: float) -> float:
    return round(float(value), 4)


def analyze_repo(
    repo_root: str | Path,
    model_path: str | Path,
    max_files: int = 1500,
    on_stage: Callable[[str], None] = lambda _stage: None,
) -> dict:
    """
    Analyse an already-cloned repository and return a JSON-friendly dict:
    {"summary": {...}, "files": [ranked file reports], "graph": {"nodes", "edges"}}
    """
    started = time.monotonic()
    root = Path(repo_root).resolve()
    if not (root / ".git").exists():
        raise AnalysisError("This folder is not a git repository.")

    # 1. Parse every supported file, one failure never aborts the rest (NFR3).
    on_stage("parsing files")
    parsed = parse_repository(str(root))
    if not parsed.files:
        raise AnalysisError("No Python files were found in this repository.")
    if len(parsed.files) > max_files:
        raise AnalysisError(
            f"This repository has {len(parsed.files)} Python files; "
            f"the limit on this server is {max_files}."
        )

    # 2. Import graph + centrality + complexity.
    on_stage("building the dependency graph")
    graph = build_dependency_graph(parsed.files, str(root))

    def rel(path: str) -> str:
        return Path(path).relative_to(root).as_posix()

    paths = [pf.path for pf in parsed.files]
    ranks = pagerank(graph.edges, paths)
    in_degree: Counter = Counter()
    edges: list[dict] = []
    for source, targets in graph.edges.items():
        for target in sorted(targets):
            try:
                edges.append({"source": rel(source), "target": rel(target)})
            except ValueError:  # an edge pointing outside the repo; ignore it
                continue
            in_degree[target] += 1

    on_stage("measuring code complexity")
    ts_parser = Parser(Language(tspython.language()))
    reports: dict[str, dict] = {}
    for pf in parsed.files:
        metrics = file_metrics(ts_parser, Path(pf.path).read_bytes())
        reports[rel(pf.path)] = {
            "path": rel(pf.path),
            **metrics,
            "in_degree": in_degree.get(pf.path, 0),
            "pagerank": _round(ranks.get(pf.path, 1.0)),
            "function_count": len(pf.functions),
            "class_count": len(pf.classes),
            "has_parse_errors": bool(pf.parse_errors),
        }

    # 3. Git history features (FR7), several `git log` calls at a time.
    on_stage("reading git history")
    with ThreadPoolExecutor(max_workers=GIT_WORKERS) as pool:
        futures = {
            path: pool.submit(extract_features_for_file, str(root), path)
            for path in reports
        }
        for path, future in futures.items():
            history = future.result()
            reports[path].update(
                churn_count=history.churn_count,
                author_count=history.author_count,
                bug_fix_commit_count=history.bug_fix_commit_count,
            )

    # 4. Risk score (FR8): the trained model if we have one, else a plain baseline.
    on_stage("scoring risk")
    rows = list(reports.values())
    try:
        bundle = load_model(model_path)
    except ModelError as exc:
        raise AnalysisError(str(exc)) from exc

    if bundle is not None:
        try:
            scores = score_files(bundle, rows)
        except ModelError as exc:
            raise AnalysisError(str(exc)) from exc
        scoring_method = "random_forest"
        model_info = bundle["info"]
    else:
        top = max((r["churn_count"] for r in rows), default=0)
        scores = [r["churn_count"] / top if top else 0.0 for r in rows]
        scoring_method = "churn_baseline"
        model_info = None

    for row, score in zip(rows, scores):
        row["risk_score"] = _round(score)

    ranked = sorted(rows, key=lambda r: (-r["risk_score"], -r["churn_count"], r["path"]))
    for position, row in enumerate(ranked, start=1):
        row["rank"] = position

    graph_nodes = [
        {"id": r["path"], "risk_score": r["risk_score"],
         "pagerank": r["pagerank"], "in_degree": r["in_degree"]}
        for r in ranked
    ]

    return {
        "summary": {
            "head_commit": _head_commit(root),
            "analyzed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "elapsed_seconds": round(time.monotonic() - started, 1),
            "n_files": len(ranked),
            "n_edges": len(edges),
            "n_functions": sum(r["function_count"] for r in ranked),
            "n_classes": sum(r["class_count"] for r in ranked),
            "n_files_with_parse_errors": sum(r["has_parse_errors"] for r in ranked),
            "n_unsupported_files_skipped": len(parsed.skipped_unsupported),
            "scoring_method": scoring_method,
            "model": model_info,
        },
        "files": ranked,
        "graph": {"nodes": graph_nodes, "edges": edges},
    }
