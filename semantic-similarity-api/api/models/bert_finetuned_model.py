"""Loader for the fine-tuned BERT bi-encoder checkpoint.

The training script saves a ``sentence-transformers`` model, so loading and
inference reuse :class:`SentenceTransformerModel`; the only difference is that
the source is a local directory produced by ``scripts/train_bert.py``.
"""

from __future__ import annotations

from pathlib import Path

from api.core.config import BERT_FINETUNED_DIR, ModelSpec
from api.models.sentence_transformer_model import SentenceTransformerModel


class BertFinetunedModel(SentenceTransformerModel):
    """Fine-tuned BERT model loaded from a local checkpoint directory."""

    @classmethod
    def load(
        cls, path: Path | str = BERT_FINETUNED_DIR, device: str | None = None
    ) -> "BertFinetunedModel":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Fine-tuned BERT checkpoint not found at {path}. "
                "Run `python -m scripts.train_bert` first."
            )
        spec = ModelSpec(
            model_id="bert_finetuned",
            kind="bert_finetuned",
            source=str(path),
        )
        return cls(spec, device=device)
