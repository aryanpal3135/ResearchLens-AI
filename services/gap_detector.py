"""
Research Gap Detector for ResearchLens AI.
Identifies evidence-grounded research gaps, repeated limitations, and unexplored territories
using Microsoft Foundry (gpt-4.1-mini) and Phase 3 Hybrid RAG retrieval.
"""

from typing import List, Optional
from models.paper import PaperDocument
from models.analysis import PaperAnalysisResult
from models.research_gap import ResearchGap, ResearchGapAnalysisResult
from services.retrieval import HybridRetrievalEngine
from services.foundry_agent import ResearchLensAgent


class ResearchGapDetector:
    """Detects systematic, evidence-grounded research gaps across literature collections."""

    def __init__(self, agent: Optional[ResearchLensAgent] = None):
        self.agent = agent or ResearchLensAgent()

    def detect_gaps_for_papers(
        self,
        papers: List[PaperDocument],
        engine: Optional[HybridRetrievalEngine] = None,
        custom_query: Optional[str] = None,
        top_k_per_topic: int = 4,
    ) -> ResearchGapAnalysisResult:
        """
        Primary entry point for research gap detection on uploaded papers.
        Delegates to ResearchLensAgent with strict per-paper retrieval isolation.
        """
        return self.agent.detect_research_gaps(
            papers=papers,
            engine=engine,
            custom_query=custom_query,
            top_k_per_topic=top_k_per_topic,
        )

    def detect_gaps(self, analyses: List[PaperAnalysisResult]) -> List[ResearchGap]:
        """
        Backwards-compatible legacy method for paper analysis lists.
        Returns an empty list if papers/engine are not directly supplied.
        """
        return []
