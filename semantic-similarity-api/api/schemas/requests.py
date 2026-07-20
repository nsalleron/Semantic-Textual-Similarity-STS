"""Pydantic request models."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class PredictRequest(BaseModel):
    model: str = Field(..., examples=["mpnet"], description="Model id or alias.")
    text1: str = Field(..., min_length=1, examples=["I love machine learning"])
    text2: str = Field(..., min_length=1, examples=["I enjoy AI and ML"])

    @field_validator("text1", "text2")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("must not be blank")
        return v
