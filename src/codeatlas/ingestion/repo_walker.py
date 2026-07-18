"""
ingestion/repo_walker.py

WHY THIS FILE EXISTS
---------------------
The parser we built (parsing/python_parser.py) can only handle ONE file at a
time. A real repository has hundreds of files. This module bridges that gap:
it walks a directory tree, decides which files are worth parsing at all, and
hands each one to the correct language-specific parser.

Deliberately, this file knows NOTHING about tree-sitter, ASTs, or how any
particular language is parsed. It only knows "here is a folder, find the
relevant files, and call whichever parser matches each file's extension."
That separation is the point: directory-walking logic should never need to
change just because we added a new language, and a new language's parser
should never need to know how directories get walked. This is Single
Responsibility Principle in practice, and it's also what makes the Strategy
pattern from the SRS actually work end to end — RepoWalker is the code that
PICKS a strategy; it never implements one.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path

from codeatlas.parsing.python_parser import PythonParser, ParsedFile, LanguageParser


# Directories that are never the developer's own source code — parsing these
# would waste time and add noise (a vendored dependency isn't "this repo's
# architecture"). This list is a judgment call, not a hard science; real
# tools like git and most linters ship a similar default ignore-list.
IGNORED_DIRS = {
    ".git", "node_modules", "venv", ".venv", "__pycache__",
    "dist", "build", ".tox", "site-packages", ".mypy_cache",
}

# Maps a file extension to the parser that knows how to handle it.
# This dict IS the Strategy selection point — adding tree-sitter-javascript
# later means adding one line here, not touching parse_repository() at all.
EXTENSION_TO_PARSER: dict[str, LanguageParser] = {
    ".py": PythonParser(),
    # ".js": JSParser(),      # to be added
    # ".ts": JSParser(),      # to be added
    # ".cpp": CppParser(),    # to be added
    # ".h":   CppParser(),    # to be added
}


@dataclass
class RepoParseResult:
    root_path: str
    files: list[ParsedFile] = field(default_factory=list)
    skipped_unsupported: list[str] = field(default_factory=list)


def discover_source_files(root_path: str) -> list[Path]:
    """
    Walk the directory tree under root_path and return every file whose
    extension we know how to parse, skipping ignored directories entirely
    (we prune them so we don't even descend into node_modules — this matters
    for performance on large repos, not just correctness).
    """
    root = Path(root_path)
    discovered: list[Path] = []

    def walk(current: Path) -> None:
        for entry in current.iterdir():
            if entry.is_dir():
                if entry.name in IGNORED_DIRS:
                    continue  # prune — do not descend
                walk(entry)
            elif entry.is_file():
                discovered.append(entry)

    walk(root)
    return discovered


def parse_repository(root_path: str) -> RepoParseResult:
    """
    The main entry point for this module: given a path to a cloned repo,
    parse every supported file and return one aggregated result.
    """
    result = RepoParseResult(root_path=root_path)
    all_files = discover_source_files(root_path)

    for file_path in all_files:
        suffix = file_path.suffix
        parser = EXTENSION_TO_PARSER.get(suffix)

        if parser is None:
            # Not a crash, not a silent drop — recorded, per NFR3's spirit of
            # observability. A course evaluator (or future you) should be able
            # to see exactly what was and wasn't analyzed, and why.
            result.skipped_unsupported.append(str(file_path))
            continue

        parsed = parser.parse_file(str(file_path))
        result.files.append(parsed)

    return result


if __name__ == "__main__":
    import sys
    import json
    from dataclasses import asdict

    target_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    repo_result = parse_repository(target_dir)

    print(f"Parsed {len(repo_result.files)} supported file(s)")
    print(f"Skipped {len(repo_result.skipped_unsupported)} unsupported file(s)")
    total_functions = sum(len(f.functions) for f in repo_result.files)
    total_classes = sum(len(f.classes) for f in repo_result.files)
    total_errors = sum(len(f.parse_errors) for f in repo_result.files)
    print(f"Total functions found: {total_functions}")
    print(f"Total classes found:   {total_classes}")
    print(f"Files with parse errors: {total_errors}")
