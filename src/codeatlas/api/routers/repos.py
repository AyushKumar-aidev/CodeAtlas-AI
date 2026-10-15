"""
HTTP layer for repository analysis. This file only translates between HTTP and
the services: it validates input, calls JobManager, and shapes the response.
No analysis logic lives here.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Path, Query, Request

from codeatlas.api.schemas import AnalyzeRequest, FilesPage, Graph, JobStatus
from codeatlas.services.jobs import Job, JobManager
from codeatlas.services.repo_fetcher import InvalidRepoUrl, parse_github_url, validate_repo_id

router = APIRouter(prefix="/repos", tags=["repositories"])

_SEGMENT = r"^[A-Za-z0-9._-]+$"
Owner = Path(pattern=_SEGMENT, max_length=39, description="GitHub owner, e.g. pallets")
Name = Path(pattern=_SEGMENT, max_length=100, description="GitHub repository name, e.g. flask")


def _jobs(request: Request) -> JobManager:
    return request.app.state.jobs


def _status(job: Job) -> JobStatus:
    return JobStatus(
        owner=job.owner, name=job.name, status=job.status, stage=job.stage,
        error=job.error, summary=job.result["summary"] if job.result else None,
    )


def _finished_job(request: Request, owner: str, name: str) -> Job:
    try:
        validate_repo_id(owner, name)
    except InvalidRepoUrl as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    job = _jobs(request).get(owner, name)
    if job is None:
        raise HTTPException(status_code=404, detail="This repository has not been analysed yet.")
    if job.status == "failed":
        raise HTTPException(status_code=409, detail=f"The analysis failed: {job.error}")
    if job.status != "done":
        raise HTTPException(status_code=409, detail=f"The analysis is not finished yet ({job.status}).")
    return job


@router.post("", response_model=JobStatus, status_code=202,
             summary="Start analysing a public GitHub repository")
def start_analysis(body: AnalyzeRequest, request: Request) -> JobStatus:
    try:
        owner, name = parse_github_url(body.url)
    except InvalidRepoUrl as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return _status(_jobs(request).submit(owner, name))


@router.get("/{owner}/{name}", response_model=JobStatus,
            summary="Progress of an analysis (poll this until status is done)")
def get_status(request: Request, owner: str = Owner, name: str = Name) -> JobStatus:
    try:
        validate_repo_id(owner, name)
    except InvalidRepoUrl as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    job = _jobs(request).get(owner, name)
    if job is None:
        raise HTTPException(status_code=404, detail="This repository has not been analysed yet.")
    return _status(job)


@router.get("/{owner}/{name}/files", response_model=FilesPage,
            summary="Files ranked from riskiest to safest")
def list_files(
    request: Request,
    owner: str = Owner,
    name: str = Name,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> FilesPage:
    job = _finished_job(request, owner, name)
    files = job.result["files"]
    return FilesPage(
        scoring_method=job.result["summary"]["scoring_method"],
        total=len(files), limit=limit, offset=offset,
        items=files[offset:offset + limit],
    )


@router.get("/{owner}/{name}/graph", response_model=Graph,
            summary="The import dependency graph (nodes = files, edges = imports)")
def get_graph(request: Request, owner: str = Owner, name: str = Name) -> Graph:
    return Graph(**_finished_job(request, owner, name).result["graph"])
