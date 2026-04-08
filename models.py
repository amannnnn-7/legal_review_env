"""Compatibility wrapper for OpenEnv CLI deployment validation."""

try:
    from .src.legal_review_env.models import (
        LegalReviewAction,
        LegalReviewObservation,
        LegalReviewState,
        TaskDifficulty,
    )
except ImportError:
    from src.legal_review_env.models import (
        LegalReviewAction,
        LegalReviewObservation,
        LegalReviewState,
        TaskDifficulty,
    )

__all__ = [
    "LegalReviewAction",
    "LegalReviewObservation",
    "LegalReviewState",
    "TaskDifficulty",
]