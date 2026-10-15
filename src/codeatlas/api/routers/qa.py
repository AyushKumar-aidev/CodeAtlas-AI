"""HTTP layer for the Q&A chat. Thin: validation and error mapping only."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from codeatlas.api.routers.repos import Name, Owner, finished_job
from codeatlas.api.schemas import AskRequest, AskResponse, QAStatus
from codeatlas.rag.llm import LLMError
from codeatlas.rag.qa_service import QAUnavailable, RateLimited

router = APIRouter(tags=["q&a"])


@router.get("/qa/status", response_model=QAStatus, summary="Is the Q&A chat available?")
def qa_status(request: Request) -> QAStatus:
    return QAStatus(enabled=request.app.state.qa.enabled, model=request.app.state.settings.llm_model)


@router.post("/repos/{owner}/{name}/ask", response_model=AskResponse,
             summary="Ask a question about an analysed repository")
def ask(body: AskRequest, request: Request, owner: str = Owner, name: str = Name) -> AskResponse:
    qa = request.app.state.qa
    if not qa.enabled:
        raise HTTPException(503, "Q&A is turned off: no GEMINI_API_KEY is configured on the server.")
    job = finished_job(request, owner, name)
    try:
        out = qa.ask(owner, name, job.result, body.question, [t.model_dump() for t in body.history])
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    except RateLimited as exc:
        raise HTTPException(429, str(exc))
    except QAUnavailable as exc:
        raise HTTPException(409, str(exc))
    except LLMError as exc:
        raise HTTPException(502, str(exc))
    return AskResponse(**out)
