"""Compatibility wrapper for OpenEnv CLI deployment validation."""

try:
    from .src.legal_review_env.client import LegalReviewEnvClient
except ImportError:
    from src.legal_review_env.client import LegalReviewEnvClient

__all__ = ["LegalReviewEnvClient"]