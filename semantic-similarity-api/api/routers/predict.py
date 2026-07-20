"""Prediction endpoint."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.core.config import resolve_model_id
from api.core.model_manager import ModelManager
from api.schemas.requests import PredictRequest
from api.schemas.responses import PredictResponse
from api.utils.confidence import confidence_level
from api.utils.timing import measure_ms

router = APIRouter(tags=["predict"])


@router.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    manager = ModelManager.instance()

    canonical = resolve_model_id(request.model)
    if canonical is None:
        raise HTTPException(
            status_code=404, detail=f"Unknown model '{request.model}'."
        )
    if not manager.is_loaded(request.model):
        raise HTTPException(
            status_code=503,
            detail=f"Model '{canonical}' is not loaded. Check server artifacts.",
        )

    model = manager.get(request.model)
    threshold = manager.threshold_for(canonical)

    with measure_ms() as elapsed:
        similarity = model.predict(request.text1, request.text2)
    inference_ms = round(elapsed(), 3)

    similarity = round(float(similarity), 6)
    prediction = similarity >= threshold

    return PredictResponse(
        model=request.model,
        similarity=similarity,
        prediction=prediction,
        threshold=threshold,
        confidence=confidence_level(similarity, threshold),
        inference_time_ms=inference_ms,
        embedding_dim=model.embedding_dim,
        device=model.device,
    )
