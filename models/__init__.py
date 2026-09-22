"""
Data models package for ResearchLens AI.
"""

from .paper import (
    PaperMetadata,
    DocumentChunk,
    PaperDocument,
    ExtractedParagraph,
    ExtractedSection,
    ExtractedTable,
    ExtractedReference,
    PageContent,
)
from .analysis import PaperAnalysisResult, EvidenceCitation
from .research_gap import ResearchGap, ResearchQuestion, FutureDirection
from .literature_review import (
    LiteratureReview,
    LiteratureReviewTheme,
    LiteratureReviewSection,
    CLAIM_TYPES,
)

__all__ = [
    "PaperMetadata",
    "DocumentChunk",
    "PaperDocument",
    "ExtractedParagraph",
    "ExtractedSection",
    "ExtractedTable",
    "ExtractedReference",
    "PageContent",
    "PaperAnalysisResult",
    "EvidenceCitation",
    "ResearchGap",
    "ResearchQuestion",
    "FutureDirection",
    "LiteratureReview",
    "LiteratureReviewTheme",
    "LiteratureReviewSection",
    "CLAIM_TYPES",
]
