"""Tests for optimal threshold computation (no downloads)."""

from __future__ import annotations

import json

from scripts.compute_thresholds import (
    best_threshold,
    binarize,
    save_thresholds,
)


def test_binarize_uses_label_threshold():
    assert binarize([0.2, 0.6, 0.9, 0.59], label_threshold=0.6) == [0, 1, 1, 0]


def test_best_threshold_separable_data():
    # Similarities clearly separate the two classes at ~0.5.
    sims = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
    labels = [0, 0, 0, 1, 1, 1]
    threshold, f1 = best_threshold(
        sims, labels, grid_min=0.0, grid_max=1.0, grid_step=0.05
    )
    # Perfect separation is achievable -> F1 == 1.0 and threshold separates
    # the negatives (<=0.3) from the positives (>=0.7).
    assert f1 == 1.0
    assert 0.3 <= threshold <= 0.7


def test_best_threshold_maximizes_f1():
    sims = [0.9, 0.85, 0.4, 0.2, 0.55, 0.6]
    labels = [1, 1, 0, 0, 1, 1]
    threshold, f1 = best_threshold(
        sims, labels, grid_min=0.0, grid_max=1.0, grid_step=0.05
    )
    # Brute-force reference over the same grid.
    import numpy as np
    from sklearn.metrics import f1_score

    sims_arr = np.asarray(sims)
    labels_arr = np.asarray(labels)
    ref = max(
        f1_score(labels_arr, (sims_arr >= t).astype(int), zero_division=0)
        for t in np.arange(0.0, 1.0001, 0.05)
    )
    assert abs(f1 - ref) < 1e-9


def test_save_thresholds_roundtrip(tmp_path):
    path = tmp_path / "thresholds.json"
    data = {"all-mpnet-base-v2": 0.81, "tfidf": 0.62}
    save_thresholds(data, path)
    assert json.loads(path.read_text()) == data
