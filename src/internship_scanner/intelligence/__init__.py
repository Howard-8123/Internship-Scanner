"""Provider- and model-independent job intelligence components."""

from internship_scanner.intelligence.models import JobAnalysis, Recommendation
from internship_scanner.intelligence.pipeline import (
    HybridIntelligencePipeline,
    JobRecommender,
    KeywordFallbackRecommender,
)

__all__ = [
    "HybridIntelligencePipeline",
    "JobAnalysis",
    "JobRecommender",
    "KeywordFallbackRecommender",
    "Recommendation",
]
