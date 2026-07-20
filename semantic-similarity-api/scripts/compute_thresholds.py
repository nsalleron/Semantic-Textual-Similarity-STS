"""Compute optimal decision thresholds per model via F1 grid search on STSB.

For each model we:
  1. compute cosine similarities on the STSB test split,
  2. binarize the gold scores (normalized >= label_threshold -> "same meaning"),
  3. grid-search a threshold in [grid_min, grid_max] maximizing F1,
  4. write the per-model thresholds to ``artifacts/thresholds.json``.

Usage:
    python -m scripts.compute_thresholds [--models mpnet e5 tfidf ...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.metrics import f1_score

from api.core.config import THRESHOLDS_PATH, settings


def binarize(gold_normalized: Sequence[float], label_threshold: float) -> List[int]:
    """Turn normalized gold scores into binary same-meaning labels."""
    return [1 if s >= label_threshold else 0 for s in gold_normalized]


def best_threshold(
    similarities: Sequence[float],
    labels: Sequence[int],
    grid_min: float = None,
    grid_max: float = None,
    grid_step: float = None,
) -> Tuple[float, float]:
    """Return ``(threshold, f1)`` maximizing F1 over the similarity grid."""
    grid_min = settings.grid_min if grid_min is None else grid_min
    grid_max = settings.grid_max if grid_max is None else grid_max
    grid_step = settings.grid_step if grid_step is None else grid_step

    sims = np.asarray(similarities, dtype=np.float64)
    y_true = np.asarray(labels, dtype=int)

    best_t = grid_min
    best_f1 = -1.0
    grid = np.arange(grid_min, grid_max + grid_step / 2, grid_step)
    for t in grid:
        y_pred = (sims >= t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_t = float(round(t, 4))
    return best_t, best_f1


def _load_stsb_test():
    from datasets import load_dataset

    ds = load_dataset("PhilipMay/stsb_multi_mt", "en", split="test")
    pairs = [(str(r["sentence1"]), str(r["sentence2"])) for r in ds]
    gold_norm = [r["similarity_score"] / 5.0 for r in ds]
    return pairs, gold_norm


def compute_for_models(model_ids: List[str]) -> Dict[str, float]:
    """Compute optimal thresholds for the given model ids on the STSB test set."""
    from api.core.model_manager import ModelManager

    manager = ModelManager.instance()
    if not manager.available():
        manager.load(model_ids)

    pairs, gold_norm = _load_stsb_test()
    labels = binarize(gold_norm, settings.label_threshold)

    thresholds: Dict[str, float] = {}
    for model_id in model_ids:
        if not manager.is_loaded(model_id):
            print(f"Skipping '{model_id}' (not loaded).")
            continue
        model = manager.get(model_id)
        print(f"Scoring {len(pairs)} pairs with '{model_id}'...")
        sims = [model.predict(a, b) for a, b in pairs]
        threshold, f1 = best_threshold(sims, labels)
        thresholds[model_id] = threshold
        print(f"  -> threshold={threshold:.3f} (F1={f1:.3f})")
    return thresholds


def save_thresholds(thresholds: Dict[str, float], path: Path | str = THRESHOLDS_PATH):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(thresholds, fh, indent=2, sort_keys=True)
    return path


def main() -> None:
    from api.core.config import MODEL_REGISTRY

    parser = argparse.ArgumentParser(description="Compute optimal thresholds on STSB.")
    parser.add_argument(
        "--models",
        nargs="*",
        default=list(MODEL_REGISTRY),
        help="Model ids to evaluate (default: all in the registry).",
    )
    args = parser.parse_args()

    thresholds = compute_for_models(args.models)
    path = save_thresholds(thresholds)
    print(f"Wrote {len(thresholds)} thresholds to {path}")


if __name__ == "__main__":
    main()
