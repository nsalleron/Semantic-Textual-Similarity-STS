"""Application configuration.

Centralises paths, the model registry (canonical id -> backend spec), alias
resolution, and tunable parameters (binarization label threshold, confidence
bands, default decision threshold).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict


# --- Paths -----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
TFIDF_PATH = ARTIFACTS_DIR / "tfidf" / "vectorizer.joblib"
BERT_FINETUNED_DIR = ARTIFACTS_DIR / "bert_finetuned"
THRESHOLDS_PATH = ARTIFACTS_DIR / "thresholds.json"


# --- Model registry --------------------------------------------------------
# ``kind`` selects the SimilarityModel implementation; ``source`` is the HF id
# (or local path) used to load it.
@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    kind: str  # "tfidf" | "bert_finetuned" | "sentence_transformer"
    source: str = ""
    query_prefix: str = ""
    passage_prefix: str = ""


MODEL_REGISTRY: Dict[str, ModelSpec] = {
    "tfidf": ModelSpec(model_id="tfidf", kind="tfidf"),
    "bert_finetuned": ModelSpec(
        model_id="bert_finetuned",
        kind="bert_finetuned",
        source=str(BERT_FINETUNED_DIR),
    ),
    "all-MiniLM-L6-v2": ModelSpec(
        model_id="all-MiniLM-L6-v2",
        kind="sentence_transformer",
        source="sentence-transformers/all-MiniLM-L6-v2",
    ),
    "all-mpnet-base-v2": ModelSpec(
        model_id="all-mpnet-base-v2",
        kind="sentence_transformer",
        source="sentence-transformers/all-mpnet-base-v2",
    ),
    "e5-base-v2": ModelSpec(
        model_id="e5-base-v2",
        kind="sentence_transformer",
        source="intfloat/e5-base-v2",
        # e5 models require these prefixes to work correctly.
        query_prefix="query: ",
        passage_prefix="passage: ",
    ),
}

# Short aliases accepted by the API in addition to the canonical ids.
MODEL_ALIASES: Dict[str, str] = {
    "minilm": "all-MiniLM-L6-v2",
    "mpnet": "all-mpnet-base-v2",
    "e5": "e5-base-v2",
    "bert": "bert_finetuned",
}


def resolve_model_id(name: str) -> str | None:
    """Resolve an alias or canonical id to a canonical registry id.

    Returns ``None`` if the name is unknown.
    """
    if name in MODEL_REGISTRY:
        return name
    return MODEL_ALIASES.get(name)


@dataclass(frozen=True)
class Settings:
    """Runtime settings for the API."""

    app_name: str = "Semantic Similarity API"
    # STSB gold score (normalized to 0..1) at/above which a pair is considered
    # to have the "same meaning" when building binary labels for F1 tuning.
    label_threshold: float = 0.6
    # Decision threshold used when no tuned threshold is available.
    default_threshold: float = 0.8
    # Grid search bounds for optimal threshold computation.
    grid_min: float = 0.5
    grid_max: float = 0.95
    grid_step: float = 0.01
    # Confidence band half-width around the decision threshold.
    # |similarity - threshold| >= high_margin -> "High";
    # >= medium_margin -> "Medium"; otherwise -> "Low".
    high_margin: float = 0.15
    medium_margin: float = 0.05
    model_registry: Dict[str, ModelSpec] = field(
        default_factory=lambda: dict(MODEL_REGISTRY)
    )


settings = Settings()
