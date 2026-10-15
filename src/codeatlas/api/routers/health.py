"""System endpoints: used by the UI, Docker and the cloud load balancer to check the API is up."""
from fastapi import APIRouter

router = APIRouter(tags=["system"])


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
