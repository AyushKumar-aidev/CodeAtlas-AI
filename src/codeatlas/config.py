"""
config.py

All tunable settings live here, read from environment variables (or a local
.env file). Nothing secret is hard-coded: API keys, when we add them for the
RAG step, will be read the same way and never committed (SRS: Security).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


@dataclass
class Settings:
    # Where cloned repos and saved analysis results are kept.
    data_dir: Path = Path(".codeatlas_data")
    # The trained risk model (produced by `python -m codeatlas.ml.risk_model`).
    model_path: Path = Path("models/risk_model.joblib")
    # Refuse repos with more Python files than this (protects a small server).
    max_files: int = 1500
    # Give up on a clone that takes longer than this many seconds.
    clone_timeout: int = 600
    # How many analyses may run at once. 1 is right for a 1 GB cloud VM.
    workers: int = 1
    # Browser origins allowed to call the API (the React dev server, for now).
    cors_origins: list[str] = field(
        default_factory=lambda: ["http://localhost:5173", "http://127.0.0.1:5173"]
    )

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()  # reads a local .env file if present; harmless if absent
        defaults = cls()
        origins = os.environ.get("CODEATLAS_CORS_ORIGINS")
        return cls(
            data_dir=Path(os.environ.get("CODEATLAS_DATA_DIR", defaults.data_dir)),
            model_path=Path(os.environ.get("CODEATLAS_MODEL_PATH", defaults.model_path)),
            max_files=int(os.environ.get("CODEATLAS_MAX_FILES", defaults.max_files)),
            clone_timeout=int(os.environ.get("CODEATLAS_CLONE_TIMEOUT", defaults.clone_timeout)),
            workers=int(os.environ.get("CODEATLAS_WORKERS", defaults.workers)),
            cors_origins=[o.strip() for o in origins.split(",") if o.strip()]
            if origins else defaults.cors_origins,
        )

    @property
    def repos_dir(self) -> Path:
        return self.data_dir / "repos"

    @property
    def results_dir(self) -> Path:
        return self.data_dir / "results"
