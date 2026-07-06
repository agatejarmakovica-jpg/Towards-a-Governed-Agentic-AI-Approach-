"""Backward-compatible import aliases for the optional OpenAI SemanticAdvisor."""
from .semantic_advisor import (
    SemanticAdvisorConfig as OpenAIReviewConfig,
    SemanticAdvisorOutput as SemanticReviewOutput,
    build_advisor_payload as build_semantic_review_payload,
    run_semantic_advisor as run_openai_semantic_review,
)

__all__ = [
    "OpenAIReviewConfig",
    "SemanticReviewOutput",
    "build_semantic_review_payload",
    "run_openai_semantic_review",
]
