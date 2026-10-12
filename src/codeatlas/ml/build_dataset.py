"""
ml/build_dataset.py

WHAT THIS FILE DOES, IN PLAIN TERMS
-------------------------------------
To train a risk model we need examples where we already know the answer.
Git history lets us fake a "time machine":

    PAST (before cutoff)              FUTURE (after cutoff)
    ----------------------            ----------------------
    count churn, authors,     --->    did this file get a
    bug-fix commits                   bug-fix commit AFTER?
    (these are the FEATURES)          (this is the LABEL: risky = 1/0)

The model only ever sees the past. We check whether it can predict the
future. That is an honest test (NFR5): the answer is never leaked into
the inputs.

Usage (from the project root):
    python src/codeatlas/ml/build_dataset.py C:/repo1 --out dataset.csv
    python src/codeatlas/ml/build_dataset.py C:/repo1 C:/repo2 C:/repo3 --out all.csv
    python src/codeatlas/ml/build_dataset.py C:/repo1 --cutoff 2025-01-01

With several repos, each repo gets its OWN cutoff (12 months before ITS latest
commit) unless you pass --cutoff, and every row records which repo it came from.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Make `import codeatlas...` work when this file is run directly as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from codeatlas.graph.git_history import extract_features_for_file  # noqa: E402


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


def default_cutoff(repo: str, months_back: int = 12) -> str:
    """Latest commit date minus ~12 months, as YYYY-MM-DD."""
    latest = _git(repo, "log", "-1", "--format=%cI")
    latest_dt = datetime.fromisoformat(latest)
    return (latest_dt - timedelta(days=30 * months_back)).strftime("%Y-%m-%d")


def list_source_files(repo: str, extensions: tuple[str, ...]) -> list[str]:
    files = _git(repo, "ls-files").split("\n")
    return [f for f in files if f and f.endswith(extensions)]


def build_dataset(repo: str, cutoff: str, extensions: tuple[str, ...]) -> list[dict]:
    rows = []
    repo_name = Path(repo).resolve().name
    files = list_source_files(repo, extensions)
    for i, path in enumerate(files, 1):
        if i % 25 == 0:
            print(f"  [{repo_name}] ...{i}/{len(files)} files", file=sys.stderr)

        past = extract_features_for_file(repo, path, before=cutoff)
        future = extract_features_for_file(repo, path, after=cutoff)

        # File did not exist yet at the cutoff -> nothing to learn from.
        if past.churn_count == 0:
            continue

        rows.append({
            "repo": repo_name,
            "file_path": path,
            "churn_count": past.churn_count,
            "author_count": past.author_count,
            "bug_fix_commit_count": past.bug_fix_commit_count,
            "future_bug_fixes": future.bug_fix_commit_count,
            "risky": 1 if future.bug_fix_commit_count >= 1 else 0,
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the ML dataset from git history.")
    ap.add_argument("repos", nargs="+", help="path(s) to cloned git repo(s)")
    ap.add_argument("--cutoff", default=None,
                    help="YYYY-MM-DD split date for ALL repos "
                         "(default: 12 months before each repo's latest commit)")
    ap.add_argument("--ext", nargs="+", default=[".py"],
                    help="source file extensions to keep (default: .py)")
    ap.add_argument("--out", default="dataset.csv", help="output CSV path")
    args = ap.parse_args()

    all_rows: list[dict] = []
    for repo in args.repos:
        cutoff = args.cutoff or default_cutoff(repo)
        print(f"\n== {Path(repo).resolve().name}  (cutoff {cutoff}) ==", file=sys.stderr)
        rows = build_dataset(repo, cutoff, tuple(args.ext))
        risky = sum(r["risky"] for r in rows)
        print(f"   kept {len(rows)} files | risky {risky} | not risky {len(rows) - risky}",
              file=sys.stderr)
        all_rows.extend(rows)

    if not all_rows:
        print("No usable files found. Try an earlier --cutoff.", file=sys.stderr)
        sys.exit(1)

    fields = ["repo", "file_path", "churn_count", "author_count",
              "bug_fix_commit_count", "future_bug_fixes", "risky"]
    with open(args.out, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)

    risky = sum(r["risky"] for r in all_rows)
    print(f"\nWrote {len(all_rows)} files from {len(args.repos)} repo(s) to {args.out}")
    print(f"Risky (bug-fixed after cutoff): {risky}  |  Not risky: {len(all_rows) - risky}")
    if risky < 0.1 * len(all_rows) or risky > 0.9 * len(all_rows):
        print("WARNING: classes are very unbalanced; try a different --cutoff.")


if __name__ == "__main__":
    main()
