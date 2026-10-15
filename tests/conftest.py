"""Shared test helpers: a tiny but REAL git repository built in a temp folder."""

from __future__ import annotations

import csv
import os
import subprocess
from pathlib import Path

import pytest


def _git(repo: Path, *args: str) -> None:
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "Ada", "GIT_AUTHOR_EMAIL": "ada@example.com",
        "GIT_COMMITTER_NAME": "Ada", "GIT_COMMITTER_EMAIL": "ada@example.com",
    }
    subprocess.run(["git", *args], cwd=repo, env=env, check=True,
                   capture_output=True, text=True)


def _commit(repo: Path, message: str, author: str | None = None) -> None:
    args = ["commit", "-q", "-m", message]
    if author:
        args += ["--author", f"{author} <{author.lower()}@example.com>"]
    _git(repo, "commit", *args[1:])


@pytest.fixture(scope="session")
def sample_repo(tmp_path_factory) -> Path:
    """
    app.py  imports pkg.core and pkg.util
    pkg/core.py imports pkg.util
    pkg/util.py is imported by both  -> highest in_degree / PageRank
    util.py is changed in 3 'fix' commits, so it has the most churn.
    """
    repo = tmp_path_factory.mktemp("sample") / "sample"
    repo.mkdir()
    (repo / "pkg").mkdir()
    _git(repo, "init", "-q")

    (repo / "pkg" / "__init__.py").write_text("")
    (repo / "pkg" / "util.py").write_text("def helper(x):\n    return x\n")
    (repo / "pkg" / "core.py").write_text(
        "from pkg.util import helper\n\n"
        "def run(x):\n    if x:\n        return helper(x)\n    return None\n"
    )
    (repo / "app.py").write_text("from pkg.core import run\nfrom pkg.util import helper\n")
    _git(repo, "add", ".")
    _commit(repo, "initial commit")

    for i in range(3):
        (repo / "pkg" / "util.py").write_text(
            f"def helper(x):\n    if x > {i}:\n        return x\n    return {i}\n"
        )
        _git(repo, "add", ".")
        _commit(repo, f"fix bug number {i}", author="Grace")

    (repo / "pkg" / "core.py").write_text(
        (repo / "pkg" / "core.py").read_text() + "\n# tidy\n"
    )
    _git(repo, "add", ".")
    _commit(repo, "tidy up")
    return repo


@pytest.fixture(scope="session")
def dataset_csv(tmp_path_factory) -> Path:
    """A synthetic training CSV in the format build_dataset.py writes."""
    path = tmp_path_factory.mktemp("data") / "dataset.csv"
    fields = ["repo", "file_path", "churn_count", "author_count", "bug_fix_commit_count",
              "loc", "complexity_total", "complexity_max", "in_degree", "pagerank",
              "future_bug_fixes", "risky"]
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for i in range(60):
            risky = 1 if i % 3 == 0 else 0
            writer.writerow({
                "repo": "r1" if i < 30 else "r2", "file_path": f"f{i}.py",
                "churn_count": 20 + i if risky else 2 + i % 4,
                "author_count": 3 + risky, "bug_fix_commit_count": 5 * risky,
                "loc": 200 if risky else 40, "complexity_total": 30 if risky else 4,
                "complexity_max": 9 if risky else 2, "in_degree": 2 * risky,
                "pagerank": 1.5 if risky else 0.8,
                "future_bug_fixes": risky, "risky": risky,
            })
    return path
