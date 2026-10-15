"""
services/jobs.py

Cloning a repo and reading its git history can take minutes, far too long for
one web request. So the API starts a background job and returns immediately;
the browser then asks "is it done yet?" every second or two.

  - A job is identified by the repo (owner/name). Asking for a repo that is
    already being analysed returns the running job instead of starting a twin.
  - Finished results are saved as JSON files, so they survive a server restart.
  - Only `workers` analyses run at once (default 1: a 1 GB cloud VM cannot
    afford more); the rest wait in a queue.

Status values: queued -> running -> done | failed
"""

from __future__ import annotations

import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from codeatlas.config import Settings
from codeatlas.services.analysis import AnalysisError, analyze_repo
from codeatlas.services.repo_fetcher import CloneError, clone_repo, repo_key

log = logging.getLogger(__name__)

Fetcher = Callable[[str, str, Path, int], None]


@dataclass
class Job:
    owner: str
    name: str
    status: str = "queued"      # queued | running | done | failed
    stage: str = "waiting"
    error: str | None = None
    result: dict | None = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )


class JobManager:
    def __init__(self, settings: Settings, fetcher: Fetcher = clone_repo,
                 analyzer: Callable[..., dict] = analyze_repo) -> None:
        self._settings = settings
        self._fetcher = fetcher
        self._analyzer = analyzer
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(
            max_workers=max(1, settings.workers), thread_name_prefix="analysis"
        )

    # ---- public API -------------------------------------------------------

    def submit(self, owner: str, name: str) -> Job:
        """Start analysing a repo, or return the job already in progress."""
        key = repo_key(owner, name)
        with self._lock:
            existing = self._jobs.get(key)
            if existing is not None and existing.status in ("queued", "running"):
                return existing
            job = Job(owner=owner, name=name)
            self._jobs[key] = job
        self._pool.submit(self._run, job, key)
        return job

    def get(self, owner: str, name: str) -> Job | None:
        """The job in memory, or a finished result saved by an earlier run."""
        key = repo_key(owner, name)
        with self._lock:
            job = self._jobs.get(key)
        if job is not None:
            return job
        saved = self._result_path(key)
        if saved.is_file():
            try:
                result = json.loads(saved.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return None
            return Job(owner=owner, name=name, status="done", stage="finished", result=result)
        return None

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)

    # ---- internals ---------------------------------------------------------

    def _result_path(self, key: str) -> Path:
        return self._settings.results_dir / (key.replace("/", "__") + ".json")

    def _run(self, job: Job, key: str) -> None:
        job.status = "running"
        try:
            job.stage = "cloning the repository"
            dest = self._settings.repos_dir / job.owner.lower() / job.name.lower()
            self._fetcher(job.owner, job.name, dest, self._settings.clone_timeout)

            def set_stage(stage: str) -> None:
                job.stage = stage

            result = self._analyzer(
                dest, self._settings.model_path,
                max_files=self._settings.max_files, on_stage=set_stage,
            )
            result["summary"]["owner"] = job.owner
            result["summary"]["name"] = job.name

            path = self._result_path(key)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result), encoding="utf-8")

            job.result = result
            job.stage = "finished"
            job.status = "done"
        except (CloneError, AnalysisError) as exc:
            job.error = str(exc)
            job.status = "failed"
        except Exception:  # a bug: log the details, show the user nothing sensitive
            log.exception("Unexpected failure analysing %s/%s", job.owner, job.name)
            job.error = "Something went wrong while analysing this repository."
            job.status = "failed"
