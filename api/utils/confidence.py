"""Confidence level derivation.

Confidence reflects how far the similarity is from the decision threshold:
the further away, the more confident the binary decision.
"""

from __future__ import annotations

from api.core.config import settings


def confidence_level(
    similarity: float,
    threshold: float,
    high_margin: float | None = None,
    medium_margin: float | None = None,
) -> str:
    """Return ``"High"``, ``"Medium"`` or ``"Low"`` based on distance to threshold."""
    high = settings.high_margin if high_margin is None else high_margin
    medium = settings.medium_margin if medium_margin is None else medium_margin

    distance = abs(similarity - threshold)
    if distance >= high:
        return "High"
    if distance >= medium:
        return "Medium"
    return "Low"
