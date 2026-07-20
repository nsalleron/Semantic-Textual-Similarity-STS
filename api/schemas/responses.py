"""Pydantic response models."""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(..., examples=["OK"])


class ModelInfo(BaseModel):
    id: str = Field(..., examples=["all-mpnet-base-v2"])
    kind: str = Field(..., examples=["sentence_transformer"])
    embedding_dim: int | None = Field(
        None, description="Embedding dimensionality, if known.", examples=[768]
    )
    threshold: float = Field(
        ..., description="Decision threshold used for binary prediction.",
        examples=[0.81],
    )
    aliases: List[str] = Field(default_factory=list, examples=[["mpnet"]])


class ModelsResponse(BaseModel):
    device: str = Field(..., examples=["cpu"])
    models: List[ModelInfo]


class PredictResponse(BaseModel):
    model: str = Field(..., examples=["mpnet"])
    similarity: float = Field(..., examples=[0.87])
    prediction: bool = Field(..., examples=[True])
    threshold: float = Field(..., examples=[0.81])
    confidence: str = Field(..., examples=["High"])
    inference_time_ms: float = Field(..., examples=[23.0])
    embedding_dim: int | None = Field(None, examples=[768])
    device: str = Field(..., examples=["cpu"])
