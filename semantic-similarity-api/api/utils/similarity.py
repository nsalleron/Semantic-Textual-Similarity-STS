"""Vector similarity helpers."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity between two 1-D vectors, in ``[-1, 1]``.

    Returns ``0.0`` if either vector has zero norm.
    """
    va = np.asarray(a, dtype=np.float64).ravel()
    vb = np.asarray(b, dtype=np.float64).ravel()
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


def clamp01(x: float) -> float:
    """Clamp a value to the ``[0, 1]`` range."""
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)
