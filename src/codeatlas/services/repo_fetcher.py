"""
services/repo_fetcher.py

Implements FR1: accept a public GitHub URL and clone the repository.

SECURITY (SRS: "must not execute arbitrary code from cloned repositories")
-------------------------------------------------------------------------
The URL comes from a user, and we hand it to `git clone`, so it is validated
with a strict pattern BEFORE anything runs:
  - only https://github.com/<owner>/<repo>  (no ssh, file://, other hosts)
  - owner/repo may only contain characters GitHub itself allows, so a value
    like "--upload-pack=..." or "../../x" can never reach git or the filesystem
  - the command is built as a list (never through a shell), with "--" before
    the URL so it can never be read as an option
We also clone with core.symlinks=false, so a malicious repo cannot plant a
symlink that points at a file on THIS server (e.g. /etc/passwd) and trick the
analyser, or later the Q&A feature, into reading it. And git never prompts
for a password: a private repo simply fails (public repos only in V1).

Nothing in a cloned repo is ever executed - it is only read as text.
"""

from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path

# GitHub owner: letters, digits, single hyphens; up to 39 characters.
# GitHub repo name: letters, digits, '.', '_', '-'.
_URL_RE = re.compile(
    r"^https://github\.com/"
    r"(?P<owner>[A-Za-z0-9](?:[A-Za-z0-9-]{0,38}))/"
    r"(?P<name>[A-Za-z0-9._-]{1,100}?)"
    r"(?:\.git)?/?$"
)


class InvalidRepoUrl(ValueError):
    """The text is not a public GitHub repository URL we accept."""


class CloneError(RuntimeError):
    """Cloning failed. The message is safe to show to the user."""


def parse_github_url(url: str) -> tuple[str, str]:
    """Return (owner, name) for a valid GitHub URL, else raise InvalidRepoUrl."""
    match = _URL_RE.match((url or "").strip())
    if not match:
        raise InvalidRepoUrl(
            "Enter a public GitHub repository URL like https://github.com/pallets/flask"
        )
    owner, name = match.group("owner"), match.group("name")
    validate_repo_id(owner, name)
    return owner, name


def validate_repo_id(owner: str, name: str) -> None:
    """Also used for URL path parameters, because they become folder names."""
    if not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})", owner or ""):
        raise InvalidRepoUrl("Invalid repository owner.")
    if not re.fullmatch(r"[A-Za-z0-9._-]{1,100}", name or "") or set(name) <= {"."}:
        raise InvalidRepoUrl("Invalid repository name.")


def repo_key(owner: str, name: str) -> str:
    """Stable, filesystem-safe identity. GitHub names are case-insensitive."""
    return f"{owner}/{name}".lower()


def _on_rm_error(func, path, _exc) -> None:
    # Windows marks git's pack files read-only; make them writable and retry.
    os.chmod(path, stat.S_IWRITE)
    func(path)


def remove_tree(path: Path) -> None:
    if not path.exists():
        return
    if sys.version_info >= (3, 12):
        shutil.rmtree(path, onexc=_on_rm_error)
    else:
        shutil.rmtree(path, onerror=_on_rm_error)


def clone_repo(owner: str, name: str, dest: Path, timeout: int) -> None:
    """
    Full clone (not shallow): the risk model needs the whole commit history.
    Any previous copy at `dest` is replaced. On failure nothing is left behind.
    """
    validate_repo_id(owner, name)
    remove_tree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://github.com/{owner}/{name}.git"
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    try:
        proc = subprocess.run(
            ["git", "-c", "core.symlinks=false", "clone", "--quiet", "--", url, str(dest)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        remove_tree(dest)
        raise CloneError(f"Cloning took longer than {timeout} seconds, so it was cancelled.")
    except FileNotFoundError:
        raise CloneError("git is not installed on the server.")
    if proc.returncode != 0:
        remove_tree(dest)
        raise CloneError(
            "Could not clone the repository. It may not exist, may be private "
            "(only public repositories are supported), or GitHub may be unreachable."
        )
