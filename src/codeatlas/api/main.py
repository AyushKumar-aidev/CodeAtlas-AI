"""CodeAtlas API entry point.

This file only wires things together: it builds the settings, creates the job
manager, and registers the routers. No business logic belongs here. Run with:
    uvicorn codeatlas.api.main:app --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from codeatlas.api.routers import health, repos
from codeatlas.config import Settings
from codeatlas.services.jobs import JobManager

WEB_DIR = Path(__file__).resolve().parent.parent / "web"  # the single-page UI


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    jobs = JobManager(settings)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        _app.state.jobs.shutdown()  # stop background workers on shutdown

    app = FastAPI(
        title="CodeAtlas",
        version="0.1.0",
        description="Repository onboarding and code intelligence platform",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.jobs = jobs

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    app.include_router(health.router)
    app.include_router(repos.router, prefix="/api")
    # Registered LAST so it never hides an API route: serves the UI at "/".
    if WEB_DIR.is_dir():
        app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
    return app


app = create_app()
