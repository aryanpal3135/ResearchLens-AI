"""
Paper Analyzer Service for ResearchLens AI.
Extracts the 18 structured research parameters with explicit vs inferred indicators.
Designed for Phase 5 implementation.
"""

from typing import Optional
from models.paper import PaperDocument
from models.analysis import PaperAnalysisResult, AnalyzedParameter
from services.foundry_client import MicrosoftFoundryClient


class PaperAnalyzer:
    """Service to perform comprehensive 18-parameter paper analysis."""

    def __init__(self, foundry_client: Optional[MicrosoftFoundryClient] = None):
        self.foundry_client = foundry_client or MicrosoftFoundryClient()

    def analyze_paper(self, paper: PaperDocument) -> PaperAnalysisResult:
        """
        Analyzes a single paper document and returns structured 18 parameters.
        (Full model integration will be hooked up in Phase 5).
        """
        result = PaperAnalysisResult(
            paper_id=paper.id,
            paper_filename=paper.filename,
        )

        # Populate basic extracted info from metadata if available
        title_val = paper.metadata.title if paper.metadata and paper.metadata.title else paper.filename
        result.title = AnalyzedParameter(parameter_name="Title", value=title_val, is_explicit=True)

        if paper.metadata and paper.metadata.authors:
            result.authors = AnalyzedParameter(
                parameter_name="Authors",
                value=", ".join(paper.metadata.authors),
                is_explicit=True
            )

        return result
