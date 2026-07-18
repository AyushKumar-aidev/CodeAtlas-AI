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


def resolve_import(import_module: str, repo_root: str) -> str | None:
    """
    Given a dotted import string like "parsing.python_parser" and the repo's
    root folder, figure out which actual .py file it refers to, if any.

    LIMITATION (stated honestly, not hidden): this only handles the common
    case of a dotted path mapping directly to a file under the repo root.
    It does NOT yet handle: relative imports (from . import x), imports of
    third-party installed packages (which correctly return None here, since
    they're not part of THIS repo's graph), or __init__.py re-export chains.
    Each of those is a real, separate piece of future work — not silently
    assumed to be "basically the same problem."
    """
    candidate = Path(repo_root) / (import_module.replace(".", "/") + ".py")
    if candidate.exists():
        return str(candidate)
    return None


def build_dependency_graph(parsed_files, repo_root: str) -> DependencyGraph:
    """
    parsed_files: list[ParsedFile] from ingestion.repo_walker
    """
    graph = DependencyGraph()

    for parsed_file in parsed_files:
        for imp in parsed_file.imports:
            resolved = resolve_import(imp.module, repo_root)
            if resolved is not None:
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
