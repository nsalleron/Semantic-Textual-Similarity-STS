"""Models listing endpoint."""

from __future__ import annotations

from fastapi import APIRouter

from api.core.device import get_device
from api.core.model_manager import ModelManager
from api.schemas.responses import ModelInfo, ModelsResponse

router = APIRouter(tags=["models"])


@router.get("/models", response_model=ModelsResponse)
def list_models() -> ModelsResponse:
    """List the models currently loaded, with their thresholds and dimensions."""
    manager = ModelManager.instance()
    return ModelsResponse(
        device=get_device(),
        models=[ModelInfo(**info) for info in manager.model_infos()],
    )
