"""Pure helpers for indicator-level leave-one-out alignment metrics.

The primary estimand is the signed change in target similarity:

    delta = similarity(full_output, target)
            - similarity(masked_output, target)

Equivalently, when cosine distance is defined as ``1 - similarity``,
``delta = masked_distance - full_distance``.  The helpers in this module are
deliberately independent of a particular embedding model so that the
estimand can be tested without loading a GPU model.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable


SimilarityScorer = Callable[[str, str], float]


def leave_one_out_metrics_from_similarities(
    similarity_full: float,
    similarity_masked: float,
    *,
    self_similarity: float | None = None,
) -> dict[str, float | None]:
    """Construct the complete LOO metric record from cosine similarities."""

    scores = {
        "similarity_full": float(similarity_full),
        "similarity_masked": float(similarity_masked),
    }
    if self_similarity is not None:
        scores["self_similarity"] = float(self_similarity)

    for name, value in scores.items():
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value!r}")
        if value < -1.000001 or value > 1.000001:
            raise ValueError(f"{name} must be a cosine similarity in [-1, 1], got {value!r}")

    distance_full = 1.0 - scores["similarity_full"]
    distance_masked = 1.0 - scores["similarity_masked"]
    delta = scores["similarity_full"] - scores["similarity_masked"]

    return {
        "similarity_full": scores["similarity_full"],
        "similarity_masked": scores["similarity_masked"],
        "delta": delta,
        "distance_full": distance_full,
        "distance_masked": distance_masked,
        "self_similarity": None if self_similarity is None else scores["self_similarity"],
        "self_distance": None if self_similarity is None else 1.0 - scores["self_similarity"],
    }


def score_leave_one_out(
    *,
    full_output: str,
    masked_output: str,
    target: str,
    scorer: SimilarityScorer,
) -> dict[str, float | None]:
    """Score one paired full/masked output against a fixed target."""

    if not callable(scorer):
        raise TypeError("scorer must be callable")

    texts = {
        "full_output": full_output,
        "masked_output": masked_output,
        "target": target,
    }
    for name, value in texts.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string")

    similarity_full = scorer(target, full_output)
    similarity_masked = scorer(target, masked_output)
    self_similarity = scorer(full_output, masked_output)
    return leave_one_out_metrics_from_similarities(
        similarity_full,
        similarity_masked,
        self_similarity=self_similarity,
    )


def add_metrics_to_similarity_rows(
    rows: Iterable[tuple[float, float, float | None]],
) -> list[dict[str, float | None]]:
    """Convert batched ``(s_full, s_masked, s_self)`` triples to metric rows."""

    return [
        leave_one_out_metrics_from_similarities(
            similarity_full,
            similarity_masked,
            self_similarity=self_similarity,
        )
        for similarity_full, similarity_masked, self_similarity in rows
    ]
