"""
Structured paper analysis and evidence citation models.
Strictly adheres to the 18 required research analysis parameters.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class EvidenceCitation(BaseModel):
    """Source evidence to ground every extracted claim."""
    claim: str
    paper_id: str
    paper_filename: str
    page: Optional[int] = None
    section: Optional[str] = None
    exact_quote: Optional[str] = None
    is_explicit: bool = True  # True if directly stated, False if inferred by AI
    confidence: str = "High"  # High, Medium, Low


class AnalyzedParameter(BaseModel):
    """A single research parameter with explicit vs inferred attribution and citations."""
    parameter_name: str
    value: str
    is_explicit: bool = True  # True: Explicitly Stated, False: AI Inferred
    citations: List[EvidenceCitation] = Field(default_factory=list)
    notes: Optional[str] = None


class PaperAnalysisResult(BaseModel):
    """The complete 18-parameter structured analysis for a research paper."""
    paper_id: str
    paper_filename: str

    # 1. Title
    title: AnalyzedParameter = Field(default_factory=lambda: AnalyzedParameter(parameter_name="Title", value=""))
    # 2. Authors
    authors: AnalyzedParameter = Field(default_factory=lambda: AnalyzedParameter(parameter_name="Authors", value=""))
    # 3. Publication Year
    publication_year: AnalyzedParameter = Field(default_factory=lambda: AnalyzedParameter(parameter_name="Publication Year", value=""))
    # 4. Research Domain
    research_domain: AnalyzedParameter = Field(default_factory=lambda: AnalyzedParameter(parameter_name="Research Domain", value=""))
    # 5. Research Problem
    research_problem: AnalyzedParameter = Field(default_factory=lambda: AnalyzedParameter(parameter_name="Research Problem", value=""))
    # 6. Research Objective
    research_objective: AnalyzedParameter = Field(default_factory=lambda: AnalyzedParameter(parameter_name="Research Objective", value=""))
    # 7. Research Questions
    research_questions: AnalyzedParameter = Field(default_factory=lambda: AnalyzedParameter(parameter_name="Research Questions", value=""))
    # 8. Methodology
    methodology: AnalyzedParameter = Field(default_factory=lambda: AnalyzedParameter(parameter_name="Methodology", value=""))
    # 9. Dataset
    dataset: AnalyzedParameter = Field(default_factory=lambda: AnalyzedParameter(parameter_name="Dataset", value=""))
    # 10. Dataset Size
    dataset_size: AnalyzedParameter = Field(default_factory=lambda: AnalyzedParameter(parameter_name="Dataset Size", value=""))
    # 11. Algorithms / Models
    algorithms: AnalyzedParameter = Field(default_factory=lambda: AnalyzedParameter(parameter_name="Algorithms / Models", value=""))
    # 12. Features / Variables
    features: AnalyzedParameter = Field(default_factory=lambda: AnalyzedParameter(parameter_name="Features / Variables", value=""))
    # 13. Evaluation Metrics
    evaluation_metrics: AnalyzedParameter = Field(default_factory=lambda: AnalyzedParameter(parameter_name="Evaluation Metrics", value=""))
    # 14. Main Results
    main_results: AnalyzedParameter = Field(default_factory=lambda: AnalyzedParameter(parameter_name="Main Results", value=""))
    # 15. Key Findings
    key_findings: AnalyzedParameter = Field(default_factory=lambda: AnalyzedParameter(parameter_name="Key Findings", value=""))
    # 16. Limitations
    limitations: AnalyzedParameter = Field(default_factory=lambda: AnalyzedParameter(parameter_name="Limitations", value=""))
    # 17. Future Work
    future_work: AnalyzedParameter = Field(default_factory=lambda: AnalyzedParameter(parameter_name="Future Work", value=""))
    # 18. Keywords
    keywords: AnalyzedParameter = Field(default_factory=lambda: AnalyzedParameter(parameter_name="Keywords", value=""))


class ResearchAnswer(BaseModel):
    """
    Structured response model for RAG-grounded AI research analysis and Q&A.
    Preserves answer text, confidence, evidence items, and explicit provenance.
    """
    query: str
    answer: str
    paper_id: Optional[str] = None
    paper_title: Optional[str] = None
    analysis_type: str = "chat_qa"  # "chat_qa", "section_analysis"
    confidence_status: str = "grounded"  # "grounded", "insufficient_evidence", "error"
    evidence_items: List[Any] = Field(default_factory=list)  # List[EvidenceItem]
    citation_labels: List[str] = Field(default_factory=list)
    execution_latency_ms: float = 0.0
    model_used: Optional[str] = None
    error_message: Optional[str] = None


class AnalysisSectionItem(BaseModel):
    """A single research section analysis item with evidence provenance."""
    section_name: str
    content: str
    is_explicit: bool = True
    evidence_items: List[Any] = Field(default_factory=list)
    citation_labels: List[str] = Field(default_factory=list)


class PaperSectionAnalysis(BaseModel):
    """Complete 7-section academic analysis for a paper, grounded in retrieved evidence."""
    paper_id: str
    paper_title: str
    sections: Dict[str, AnalysisSectionItem] = Field(default_factory=dict)
    status: str = "completed"  # "completed", "error", "insufficient_evidence"
    execution_latency_ms: float = 0.0
    model_used: Optional[str] = None
    error_message: Optional[str] = None


class ComparisonPoint(BaseModel):
    """A grounded comparison point (similarity or difference) with paper attribution."""
    topic: str
    description: str
    paper_a_claim: str = ""
    paper_b_claim: str = ""
    citation_labels: List[str] = Field(default_factory=list)
    evidence_items: List[Any] = Field(default_factory=list)


class DimensionComparisonItem(BaseModel):
    """Comparative analysis for a single academic dimension (e.g. Methodology)."""
    dimension_name: str
    paper_summaries: Dict[str, str] = Field(default_factory=dict)  # paper_id -> summary
    paper_evidence: Dict[str, List[Any]] = Field(default_factory=dict)  # paper_id -> List[EvidenceItem]
    synthesis: str = ""
    is_explicit: bool = True
    citation_labels: List[str] = Field(default_factory=list)
    evidence_items: List[Any] = Field(default_factory=list)


class MultiPaperComparison(BaseModel):
    """Complete multi-paper comparative analysis model grounded in retrieved evidence."""
    comparison_id: str
    paper_ids: List[str] = Field(default_factory=list)
    paper_titles: Dict[str, str] = Field(default_factory=dict)
    comparison_request: str = "Standard 10-Dimension Academic Comparison"
    dimensions: Dict[str, DimensionComparisonItem] = Field(default_factory=dict)
    similarities: List[ComparisonPoint] = Field(default_factory=list)
    differences: List[ComparisonPoint] = Field(default_factory=list)
    custom_query_answer: Optional[str] = None
    all_evidence_items: List[Any] = Field(default_factory=list)
    status: str = "completed"  # "completed", "insufficient_evidence", "error"
    execution_latency_ms: float = 0.0
    model_used: Optional[str] = None
    error_message: Optional[str] = None


