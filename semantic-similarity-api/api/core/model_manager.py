"""ModelManager: loads models once and serves them for the app lifetime.

Implemented as a process-wide singleton. Models are loaded eagerly at startup;
backends whose artifacts are missing (e.g. an untrained TF-IDF vectorizer or
BERT checkpoint) are skipped with a warning instead of crashing the app.
"""

from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional

from api.core.config import (
    MODEL_ALIASES,
    MODEL_REGISTRY,
    THRESHOLDS_PATH,
    ModelSpec,
    resolve_model_id,
    settings,
)
from api.models.base import SimilarityModel

logger = logging.getLogger(__name__)


def _aliases_for(model_id: str) -> List[str]:
    return [alias for alias, target in MODEL_ALIASES.items() if target == model_id]


def _build_model(spec: ModelSpec) -> SimilarityModel:
    """Instantiate a model from its registry spec."""
    if spec.kind == "tfidf":
        from api.models.tfidf_model import TfidfModel

        return TfidfModel.load()
    if spec.kind == "bert_finetuned":
        from api.models.bert_finetuned_model import BertFinetunedModel

        return BertFinetunedModel.load()
    if spec.kind == "sentence_transformer":
        from api.models.sentence_transformer_model import SentenceTransformerModel

        return SentenceTransformerModel(spec)
    raise ValueError(f"Unknown model kind: {spec.kind}")


class ModelManager:
    """Singleton registry of loaded similarity models and their thresholds."""

    _instance: Optional["ModelManager"] = None

    def __init__(self) -> None:
        self._models: Dict[str, SimilarityModel] = {}
        self._thresholds: Dict[str, float] = {}

    # --- singleton access --------------------------------------------------
    @classmethod
    def instance(cls) -> "ModelManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Drop the singleton (used by tests)."""
        cls._instance = None

    # --- loading -----------------------------------------------------------
    def load_thresholds(self, path=THRESHOLDS_PATH) -> None:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                self._thresholds = {k: float(v) for k, v in json.load(fh).items()}
            logger.info("Loaded thresholds for %d models", len(self._thresholds))
        except FileNotFoundError:
            logger.warning(
                "No thresholds file at %s; using default threshold %.2f",
                path,
                settings.default_threshold,
            )
            self._thresholds = {}

    def load(self, model_ids: Optional[List[str]] = None) -> None:
        """Load the requested models (default: all registry entries)."""
        self.load_thresholds()
        ids = model_ids if model_ids is not None else list(MODEL_REGISTRY)
        for model_id in ids:
            spec = MODEL_REGISTRY[model_id]
            try:
                logger.info("Loading model '%s'...", model_id)
                self._models[model_id] = _build_model(spec)
            except Exception as exc:  # pragma: no cover - depends on artifacts
                logger.warning("Skipping model '%s': %s", model_id, exc)

    def register(self, model: SimilarityModel) -> None:
        """Inject a preloaded model (used by tests)."""
        self._models[model.model_id] = model

    # --- access ------------------------------------------------------------
    def is_loaded(self, name: str) -> bool:
        model_id = resolve_model_id(name)
        return model_id is not None and model_id in self._models

    def get(self, name: str) -> SimilarityModel:
        """Return a loaded model by canonical id or alias.

        Raises ``KeyError`` if the name is unknown or the model is not loaded.
        """
        model_id = resolve_model_id(name)
        if model_id is None or model_id not in self._models:
            raise KeyError(name)
        return self._models[model_id]

    def available(self) -> List[str]:
        return list(self._models)

    def threshold_for(self, model_id: str) -> float:
        return self._thresholds.get(model_id, settings.default_threshold)

    def model_infos(self) -> List[dict]:
        infos = []
        for model_id, model in self._models.items():
            infos.append(
                {
                    "id": model_id,
                    "kind": MODEL_REGISTRY[model_id].kind,
                    "embedding_dim": model.embedding_dim,
                    "threshold": self.threshold_for(model_id),
                    "aliases": _aliases_for(model_id),
                }
            )
        return infos
