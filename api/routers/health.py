"""Health check endpoint."""

from __future__ import annotations

from fastapi import APIRouter

from api.schemas.responses import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness probe."""
    return HealthResponse(status="OK")
