"""
graph/git_history.py

WHAT THIS FILE DOES, IN PLAIN TERMS
-------------------------------------
For every file in a repo, we ask git: "show me every commit that ever
touched this file." From that list we count three things per file:

1. churn_count       -> how many times has this file changed?
2. author_count      -> how many DIFFERENT people have changed it?
3. bug_fix_commit_count -> how many of those changes LOOK like bug fixes?

These three numbers are exactly what FR7 in the SRS asks for, and they
become the input features for the ML risk model (FR8) later — a file
that's been changed 50 times by 8 different people, with 20 of those
being bug fixes, is a very different file from one nobody has touched
since it was written.

HOW WE "GUESS" A COMMIT IS A BUG FIX
--------------------------------------
Git doesn't have a built-in "this was a bug fix" flag — commit messages
are just free text. So we use a simple, honest heuristic: if the commit
message contains a word like "fix" or "bug", we count it as a likely
bug-fix commit. This is a proxy, not ground truth — a commit message
could say "fix" and not really be a bug fix, or fix a real bug without
using that word. We're not hiding this limitation; it's the exact kind
of thing NFR5 (evaluation honesty) asks us to be upfront about.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import subprocess


# Words that suggest a commit message is describing a bug fix.
# Lowercase, since we'll lowercase the commit message before checking.
BUG_FIX_KEYWORDS = ["fix", "bug", "issue", "error", "crash", "broken"]


@dataclass
class GitHistoryFeature:
    file_path: str
    churn_count: int = 0
    author_count: int = 0
    bug_fix_commit_count: int = 0
    authors: set = field(default_factory=set)


def run_git_log_for_file(repo_path: str, file_path: str) -> list[tuple[str, str]]:
    """
    Runs the actual git command that lists every commit touching one file.

    The command we run is:
        git log --follow --format=%an|||%s -- <file_path>

    Breaking that down in plain terms:
    - `git log`            -> show commit history
    - `--follow`           -> keep tracking the file even if it was renamed
    - `--format=%an|||%s`  -> for each commit, print "AuthorName|||Subject"
                              (we use ||| as a separator that's very unlikely
                              to appear in real text, so we can split on it)
    - `-- <file_path>`     -> only show commits that touched THIS file

    Returns a list of (author_name, commit_message) tuples, one per commit.
    """
    result = subprocess.run(
        ["git", "log", "--follow", "--format=%an|||%s", "--", file_path],
        cwd=repo_path,
        capture_output=True,
        text=True,
    )
    lines = [line for line in result.stdout.strip().split("\n") if line]

    commits = []
    for line in lines:
        if "|||" in line:
            author, message = line.split("|||", 1)
            commits.append((author, message))
    return commits


def is_bug_fix_commit(commit_message: str) -> bool:
    lowered = commit_message.lower()
    return any(keyword in lowered for keyword in BUG_FIX_KEYWORDS)


def extract_features_for_file(repo_path: str, file_path: str) -> GitHistoryFeature:
    commits = run_git_log_for_file(repo_path, file_path)

    feature = GitHistoryFeature(file_path=file_path)
    feature.churn_count = len(commits)

    for author, message in commits:
        feature.authors.add(author)
        if is_bug_fix_commit(message):
            feature.bug_fix_commit_count += 1

    feature.author_count = len(feature.authors)
    return feature


if __name__ == "__main__":
    import sys
    import json
    from pathlib import Path

    repo_path = sys.argv[1] if len(sys.argv) > 1 else "."

    # Find every tracked file in the repo using git itself, so we don't
    # need our own file-walking logic here — git already knows this.
    result = subprocess.run(
        ["git", "ls-files"], cwd=repo_path, capture_output=True, text=True
    )
    tracked_files = [f for f in result.stdout.strip().split("\n") if f]

    all_features = []
    for f in tracked_files:
        feature = extract_features_for_file(repo_path, f)
        all_features.append({
            "file_path": feature.file_path,
            "churn_count": feature.churn_count,
            "author_count": feature.author_count,
            "bug_fix_commit_count": feature.bug_fix_commit_count,
            "authors": list(feature.authors),
        })

    print(json.dumps(all_features, indent=2))
