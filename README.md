# CodeAtlas AI — early build

This is the in-progress codebase for CodeAtlas AI (see the SRS document for
full requirements/architecture context).

## Project layout (src/ layout — real Python packaging convention)

```
codeatlas-ai/
├── src/
│   └── codeatlas/              # the actual installable package
│       ├── parsing/
│       │   └── python_parser.py   # FR2: tree-sitter based Python parsing
│       ├── ingestion/
│       │   └── repo_walker.py     # walks a repo, dispatches files to parsers
│       └── graph/
│           └── builder.py         # FR3: resolves imports into a dependency graph
├── tests/                      # tests live OUTSIDE src/ — standard convention
│   ├── sample_repo_file.py
│   └── broken_file.py
├── pyproject.toml              # declares this as an installable package
└── README.md
```

## Setup (do this once per machine)

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Editable install — this is what makes `import codeatlas...` work
# anywhere, while still editing the source live.
pip install -e .
```

## Running things

```bash
# Parse one file
python3 -m codeatlas.parsing.python_parser tests/sample_repo_file.py

# Walk and parse an entire folder (try it on src/ itself)
python3 -m codeatlas.ingestion.repo_walker src/

# Build the dependency graph for a folder
python3 -m codeatlas.graph.builder src/
```
