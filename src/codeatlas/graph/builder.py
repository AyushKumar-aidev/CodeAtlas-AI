"""
graph/builder.py

WHY THIS FILE EXISTS
---------------------
ingestion/repo_walker.py gives us a list of ParsedFile objects. Each one
knows its OWN functions/classes/imports, but nothing about how files relate
to EACH OTHER. This file's job: turn a list of independent ParsedFile
objects into an actual graph — nodes are files, edges are "this file
imports that file."

This is genuinely the first non-trivial engineering problem in the whole
pipeline: an import string like "parsing.python_parser" is just TEXT. To
turn it into a real graph edge, we have to figure out which actual file on
disk that text refers to. That's what resolve_import() below does — still
a simplified version (real Python import resolution has many more edge
cases: relative imports, __init__.py re-exports, namespace packages), but
this handles the common case correctly and is honest about what it doesn't
handle yet (see the docstring on resolve_import).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DependencyGraph:
    # Each key is a file path; each value is the set of file paths it imports.
    edges: dict[str, set[str]] = field(default_factory=dict)
    unresolved_imports: dict[str, list[str]] = field(default_factory=dict)

    def add_edge(self, from_file: str, to_file: str) -> None:
        self.edges.setdefault(from_file, set()).add(to_file)


def _find_module(base: Path, parts: list[str]) -> str | None:
    """
    Look for a module under `base`: first `a/b/c.py`, then the package
    `a/b/c/__init__.py`. With no parts, the package `base/__init__.py`.
    """
    if parts:
        target = base.joinpath(*parts)
        as_file = Path(str(target) + ".py")
        if as_file.is_file():
            return str(as_file)
        as_pkg = target / "__init__.py"
    else:
        as_pkg = base / "__init__.py"
    if as_pkg.is_file():
        return str(as_pkg)
    return None


def resolve_import(
    import_module: str, repo_root: str, importer_path: str | None = None
) -> str | None:
    """
    Given an import string and the repo's root folder, figure out which
    actual .py file inside THIS repo it refers to, if any.

    Handles:
      - absolute imports:  "parsing.python_parser" -> parsing/python_parser.py
      - packages:          "flask"                 -> flask/__init__.py
      - src/ layouts:      also tries <repo_root>/src/...
      - relative imports:  ".app", "..utils", "." (needs importer_path, the
                           file that contains the import statement)

    STILL NOT handled (stated honestly): names re-exported through
    __init__.py chains, namespace packages, dynamic imports. Third-party
    libraries correctly return None - they are not part of THIS repo's graph.
    """
    root = Path(repo_root)

    if import_module.startswith("."):
        if importer_path is None:
            return None
        dots = len(import_module) - len(import_module.lstrip("."))
        rest = import_module[dots:]
        base = Path(importer_path).parent
        for _ in range(dots - 1):  # one dot = same package, two dots = parent...
            base = base.parent
        return _find_module(base, rest.split(".") if rest else [])

    parts = import_module.split(".")
    for search_root in (root, root / "src"):
        found = _find_module(search_root, parts)
        if found is not None:
            return found
    return None


def build_dependency_graph(parsed_files, repo_root: str) -> DependencyGraph:
    """
    parsed_files: list[ParsedFile] from ingestion.repo_walker
    """
    graph = DependencyGraph()

    for parsed_file in parsed_files:
        for imp in parsed_file.imports:
            resolved = resolve_import(imp.module, repo_root, parsed_file.path)
            if resolved is not None and resolved != parsed_file.path:
                graph.add_edge(parsed_file.path, resolved)
            else:
                # Not a bug — this correctly happens for every external
                # library import (os, json, etc.), which isn't part of
                # THIS repo's internal structure and shouldn't be an edge.
                graph.unresolved_imports.setdefault(
                    parsed_file.path, []
                ).append(imp.module)

    return graph


if __name__ == "__main__":
    import sys
    from codeatlas.ingestion.repo_walker import parse_repository

    target_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    repo_result = parse_repository(target_dir)
    graph = build_dependency_graph(repo_result.files, target_dir)

    print("=== Internal dependency edges (this repo's own files) ===")
    for source, targets in graph.edges.items():
        for t in targets:
            print(f"  {source}  ->  {t}")

    print(f"\n=== {sum(len(v) for v in graph.unresolved_imports.values())} external/unresolved imports (expected — these are libraries) ===")
