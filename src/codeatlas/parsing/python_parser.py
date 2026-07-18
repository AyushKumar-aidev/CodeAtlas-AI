"""
parsing/python_parser.py

First real component of the CodeAtlas ingestion pipeline: parses a single
Python file using tree-sitter and extracts a structured summary — imports,
function definitions, and class definitions, each with line numbers.

This is deliberately the simplest possible slice of FR2 (parse files into
ASTs) and a precursor to FR3 (build dependency/call graph): before we can
build a graph, we need reliable extraction of "what does this file define"
and "what does this file import" from every file in a repo.

Design note: this module implements the Strategy pattern referenced in the
SRS (Section 6.1) — LanguageParser is the interface; PythonParser is one
concrete strategy. JSParser and CppParser will implement the same interface
later without touching this file or the pipeline that calls it.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path

import tree_sitter_python as tspython
from tree_sitter import Language, Parser


@dataclass
class FunctionDef:
    name: str
    start_line: int
    end_line: int


@dataclass
class ClassDef:
    name: str
    start_line: int
    end_line: int


@dataclass
class ImportStmt:
    module: str
    start_line: int


@dataclass
class ParsedFile:
    path: str
    language: str
    functions: list[FunctionDef] = field(default_factory=list)
    classes: list[ClassDef] = field(default_factory=list)
    imports: list[ImportStmt] = field(default_factory=list)
    parse_errors: list[str] = field(default_factory=list)


class LanguageParser:
    """Interface every language-specific parser implements (Strategy pattern)."""

    def parse_file(self, path: str) -> ParsedFile:
        raise NotImplementedError


class PythonParser(LanguageParser):
    def __init__(self) -> None:
        py_language = Language(tspython.language())
        self._parser = Parser(py_language)

    def parse_file(self, path: str) -> ParsedFile:
        source_bytes = Path(path).read_bytes()
        tree = self._parser.parse(source_bytes)
        root = tree.root_node

        result = ParsedFile(path=path, language="python")

        # tree-sitter reports syntax errors as ERROR nodes rather than raising —
        # NFR3 requires we log per-file failures instead of crashing the pipeline.
        if root.has_error:
            result.parse_errors.append(
                f"Syntax error(s) detected while parsing {path}"
            )

        def node_text(node) -> str:
            return source_bytes[node.start_byte:node.end_byte].decode(
                "utf-8", errors="replace"
            )

        def walk(node) -> None:
            if node.type == "function_definition":
                name_node = node.child_by_field_name("name")
                if name_node is not None:
                    result.functions.append(
                        FunctionDef(
                            name=node_text(name_node),
                            start_line=node.start_point[0] + 1,
                            end_line=node.end_point[0] + 1,
                        )
                    )
            elif node.type == "class_definition":
                name_node = node.child_by_field_name("name")
                if name_node is not None:
                    result.classes.append(
                        ClassDef(
                            name=node_text(name_node),
                            start_line=node.start_point[0] + 1,
                            end_line=node.end_point[0] + 1,
                        )
                    )
            elif node.type == "import_statement":
                for child in node.children:
                    if child.type == "dotted_name":
                        result.imports.append(
                            ImportStmt(
                                module=node_text(child),
                                start_line=node.start_point[0] + 1,
                            )
                        )
            elif node.type == "import_from_statement":
                module_node = node.child_by_field_name("module_name")
                if module_node is not None:
                    result.imports.append(
                        ImportStmt(
                            module=node_text(module_node),
                            start_line=node.start_point[0] + 1,
                        )
                    )

            for child in node.children:
                walk(child)

        walk(root)
        return result


if __name__ == "__main__":
    import sys
    import json
    from dataclasses import asdict

    target = sys.argv[1] if len(sys.argv) > 1 else __file__
    parser = PythonParser()
    parsed = parser.parse_file(target)
    print(json.dumps(asdict(parsed), indent=2))
