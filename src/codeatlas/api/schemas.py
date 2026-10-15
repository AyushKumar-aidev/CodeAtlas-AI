"""Request and response shapes of the REST API (they also drive the /docs page)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    url: str = Field(description="Public GitHub repository URL, e.g. https://github.com/pallets/flask")


class ModelInfo(BaseModel):
    trained_at: str
    n_files: int
    n_risky: int
    repos: list[str]
    feature_names: list[str]


class Summary(BaseModel):
    owner: str | None = None
    name: str | None = None
    head_commit: str
    analyzed_at: str
    elapsed_seconds: float
    n_files: int
    n_edges: int
    n_functions: int
    n_classes: int
    n_files_with_parse_errors: int
    n_unsupported_files_skipped: int
    scoring_method: Literal["random_forest", "churn_baseline"] = Field(
        description="'random_forest' = the trained model. 'churn_baseline' = no trained "
                    "model was found, so files are ranked by how often they changed."
    )
    model: ModelInfo | None = None


class JobStatus(BaseModel):
    owner: str
    name: str
    status: Literal["queued", "running", "done", "failed"]
    stage: str
    error: str | None = None
    summary: Summary | None = None


class FileRisk(BaseModel):
    rank: int
    path: str
    risk_score: float = Field(
        description="0 to 1, higher = riskier. A RANKING score, not a calibrated probability."
    )
    churn_count: int
    author_count: int
    bug_fix_commit_count: int
    loc: int
    complexity_total: int
    complexity_max: int
    in_degree: int
    pagerank: float
    function_count: int
    class_count: int
    has_parse_errors: bool


class FilesPage(BaseModel):
    scoring_method: str
    total: int
    limit: int
    offset: int
    items: list[FileRisk]


class GraphNode(BaseModel):
    id: str
    risk_score: float
    pagerank: float
    in_degree: int


class GraphEdge(BaseModel):
    source: str = Field(description="The file that contains the import")
    target: str = Field(description="The file being imported")


class Graph(BaseModel):
    nodes: list[GraphNode]
    edges: list[GraphEdge]


class AskTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=4000)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    history: list[AskTurn] = Field(default_factory=list, max_length=12)


class SourceRef(BaseModel):
    path: str
    start_line: int
    end_line: int
    name: str
    kind: str


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceRef]
    model: str


class QAStatus(BaseModel):
    enabled: bool
    model: str
