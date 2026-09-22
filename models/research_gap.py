"""
Models for Research Gaps, Research Questions, Future Directions, and Provenance.
Adheres strictly to Phase 6 and Phase 7 requirements:
- Distinguishes explicit, evidence-derived, and cross-paper gaps and questions.
- Transparent categorical evidence support (High, Moderate, Limited).
- Rigorous separation of evidence-grounded gaps/questions from AI-generated future research directions.
- Gap -> Question -> Paper -> Chunk end-to-end traceability.
- Preserves full chunk-level evidence items and provenance.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from models.analysis import EvidenceCitation


# Standard Gap Taxonomy Categories (Phase 6)
GAP_CATEGORIES = [
    "Methodological Gap",
    "Dataset Gap",
    "Evaluation Gap",
    "Generalization Gap",
    "Domain Gap",
    "Theoretical Gap",
    "Empirical Gap",
    "Comparative Gap",
    "Scalability Gap",
    "Reproducibility Gap",
    "Contradictory Evidence",
    "Future-Work Gap",
]

# Standard Research Question Types (Phase 7)
QUESTION_TYPES = [
    "Exploratory",
    "Explanatory",
    "Comparative",
    "Methodological",
    "Empirical",
    "Evaluation",
    "Generalization",
    "Dataset-focused",
    "Scalability",
    "Reproducibility",
    "Theoretical",
    "Cross-domain",
    "Cross-paper",
]

# Standard Future Research Direction Types (Phase 7)
FUTURE_DIRECTION_TYPES = [
    "New experiments",
    "New datasets",
    "New evaluation settings",
    "New model variants",
    "New domains",
    "New languages",
    "New baselines",
    "Scalability studies",
    "Ablation studies",
    "Reproducibility studies",
    "Cross-paper synthesis",
    "Theoretical investigation",
]


class ResearchGap(BaseModel):
    """
    Evidence-grounded research gap identified across one or more research papers.
    Strictly preserves paper attribution, chunk provenance, and explicit vs inferred distinction.
    """
    gap_id: str
    title: str
    category: str = "Methodological Gap"
    description: str
    evidence_type: str = "evidence_derived"  # "explicit", "evidence_derived", "cross_paper", "insufficient_evidence"
    evidence_type_label: str = "Derived from retrieved evidence"  # Human-readable label
    evidence_support: str = "High evidence support"  # "High evidence support", "Moderate evidence support", "Limited evidence support"
    supporting_papers: List[str] = Field(default_factory=list)  # List of paper_id strings
    evidence_items: List[Any] = Field(default_factory=list)  # List[EvidenceItem]
    citation_labels: List[str] = Field(default_factory=list)  # e.g. ["[Evidence A1: paper_001, p. 8]"]
    rationale: str = ""  # Why the retrieved evidence supports calling this a gap
    affected_dimension: Optional[str] = None  # e.g. "Evaluation", "Methodology"
    potential_research_direction: Optional[str] = None  # AI-suggested future research direction
    potential_tension: Optional[str] = None  # Specific tension for contradictory/conflicting evidence
    
    # Backwards compatibility fields
    is_explicit: bool = False
    confidence_level: str = "High evidence support"  # Categorical support level
    status: str = "grounded"  # "grounded", "insufficient_evidence", "error"
    model: Optional[str] = None
    execution_latency_ms: float = 0.0

    def model_post_init(self, __context: Any) -> None:
        # Deterministically derive supporting_papers strictly from evidence_items if present
        if self.evidence_items:
            evidence_paper_ids = []
            for ev in self.evidence_items:
                pid = getattr(ev, "paper_id", None) or (ev.get("paper_id") if isinstance(ev, dict) else None)
                if pid and pid not in evidence_paper_ids:
                    evidence_paper_ids.append(pid)
            if evidence_paper_ids:
                self.supporting_papers = evidence_paper_ids

        # Synchronize evidence_type based on actual supporting_papers
        if len(self.supporting_papers) < 2 and self.evidence_type == "cross_paper":
            self.evidence_type = "explicit" if self.is_explicit else "evidence_derived"
            self.evidence_type_label = "Explicitly stated by the paper" if self.is_explicit else "Derived from retrieved evidence"
        elif len(self.supporting_papers) >= 2 and self.evidence_type != "cross_paper":
            self.evidence_type = "cross_paper"
            self.evidence_type_label = "Cross-paper gap"


class ResearchQuestion(BaseModel):
    """
    Evidence-grounded research question formulated from detected research gaps.
    Strictly preserves gap linkage, supporting papers, and chunk provenance.
    """
    question_id: str
    question: str = ""
    question_type: str = "Methodological"  # One of QUESTION_TYPES
    question_origin: str = "evidence_derived"  # "author_inspired", "evidence_derived", "cross_paper"
    question_origin_label: str = "Evidence-Derived Question"
    research_gap_ids: List[str] = Field(default_factory=list)  # Links to ResearchGap.gap_id
    supporting_papers: List[str] = Field(default_factory=list)
    evidence_items: List[Any] = Field(default_factory=list)  # List[EvidenceItem]
    citation_labels: List[str] = Field(default_factory=list)
    rationale: str = ""  # How this question addresses the linked gap
    novelty_basis: str = "Motivated by the identified gap in the selected papers"
    potential_research_direction: Optional[str] = None  # Actionable future direction
    direction_type: Optional[str] = None  # One of FUTURE_DIRECTION_TYPES
    suggested_evaluation: Optional[str] = None
    suggested_dataset_or_setting: Optional[str] = None
    evidence_support: str = "High evidence support"
    status: str = "grounded"  # "grounded", "insufficient_evidence", "error"
    model: Optional[str] = None
    execution_latency_ms: float = 0.0

    # User-facing contiguous numbering and provenance audit fields
    display_index: int = 1  # 1-based sequential display order for UI presentation
    display_id: str = ""    # Contiguous user-facing display identifier (e.g. "rq_001", "rq_002")
    candidate_id: Optional[str] = None  # Raw LLM candidate identifier for auditability

    # Backwards compatibility fields
    gap_id: str = ""
    question_text: str = ""
    suggested_focus: str = ""
    is_ai_generated: bool = True

    def model_post_init(self, __context: Any) -> None:
        if not self.display_id and self.display_index:
            self.display_id = f"rq_{self.display_index:03d}"
        if not self.candidate_id and self.question_id:
            self.candidate_id = self.question_id
        if not self.question and self.question_text:
            self.question = self.question_text
        elif not self.question_text and self.question:
            self.question_text = self.question
        if not self.research_gap_ids and self.gap_id:
            self.research_gap_ids = [self.gap_id]
        elif not self.gap_id and self.research_gap_ids:
            self.gap_id = self.research_gap_ids[0]
        if not self.suggested_focus and self.question_type:
            self.suggested_focus = self.question_type

        # Strict Provenance Normalization:
        # If evidence_items are provided, supporting_papers MUST be derived from evidence_items.
        # Every supporting paper must have at least one supporting evidence item.
        if self.evidence_items:
            evidence_paper_ids = []
            for ev in self.evidence_items:
                pid = getattr(ev, "paper_id", None) or (ev.get("paper_id") if isinstance(ev, dict) else None)
                if pid and pid not in evidence_paper_ids:
                    evidence_paper_ids.append(pid)
            if evidence_paper_ids:
                self.supporting_papers = evidence_paper_ids

        # Cross-paper validation:
        # If supporting_papers has only 1 paper, it cannot be a cross-paper question.
        if len(self.supporting_papers) == 1 and self.question_origin == "cross_paper":
            self.question_origin = "evidence_derived"
        elif len(self.supporting_papers) >= 2 and self.question_origin not in ["cross_paper"]:
            self.question_origin = "cross_paper"

        # Synchronize origin label
        if self.question_origin == "author_inspired":
            self.question_origin_label = "Author-Inspired Question"
        elif self.question_origin == "cross_paper":
            self.question_origin_label = "Cross-Paper Synthesis Question"
        else:
            self.question_origin_label = "Evidence-Derived Question"

        # Enforce literature-scoped novelty guardrail
        lower_nov = self.novelty_basis.lower()
        if any(w in lower_nov for w in ["in the field", "first ever", "never done before", "entirely novel"]):
            self.novelty_basis = "Motivated by the identified gap in the selected papers"


class FutureDirection(BaseModel):
    """
    Actionable proposed future research pathway clearly partitioned from author claims.
    Visibly flagged with is_suggestion=True.
    """
    direction_id: str
    direction_type: str = "New experiments"  # One of FUTURE_DIRECTION_TYPES
    title: str = ""
    proposed_direction: str = ""
    description: str = ""
    gap_addressed: str = ""
    linked_question_id: Optional[str] = None
    linked_display_id: Optional[str] = None
    display_index: int = 1
    display_id: str = ""
    supporting_papers: List[str] = Field(default_factory=list)
    possible_methodology: str = ""
    suggested_methodology: str = ""
    possible_dataset: str = ""
    suggested_dataset_or_setting: str = ""
    expected_research_contribution: str = ""
    expected_contribution: str = ""
    is_suggestion: bool = True  # Explicit flag that this is AI suggestion, not author statement
    gap_id: str = ""

    def model_post_init(self, __context: Any) -> None:
        if not self.display_id and self.display_index:
            self.display_id = f"fd_{self.display_index:03d}"
        if not self.linked_display_id and self.linked_question_id:
            self.linked_display_id = self.linked_question_id
        if not self.gap_id and self.gap_addressed:
            self.gap_id = self.gap_addressed
        elif not self.gap_addressed and self.gap_id:
            self.gap_addressed = self.gap_id
        if not self.description and self.proposed_direction:
            self.description = self.proposed_direction
        elif not self.proposed_direction and self.description:
            self.proposed_direction = self.description
        if not self.title and self.description:
            self.title = self.description[:60] + "..." if len(self.description) > 60 else self.description
        if not self.suggested_methodology and self.possible_methodology:
            self.suggested_methodology = self.possible_methodology
        elif not self.possible_methodology and self.suggested_methodology:
            self.possible_methodology = self.suggested_methodology
        if not self.suggested_dataset_or_setting and self.possible_dataset:
            self.suggested_dataset_or_setting = self.possible_dataset
        elif not self.possible_dataset and self.suggested_dataset_or_setting:
            self.possible_dataset = self.suggested_dataset_or_setting
        if not self.expected_contribution and self.expected_research_contribution:
            self.expected_contribution = self.expected_research_contribution
        elif not self.expected_research_contribution and self.expected_contribution:
            self.expected_research_contribution = self.expected_contribution


class ResearchGapAnalysisResult(BaseModel):
    """
    Complete analysis result for research gap detection across single or multiple papers.
    Preserves summary metrics, structured gaps, full evidence provenance, and execution metadata.
    """
    analysis_id: str
    query: str
    selected_paper_ids: List[str] = Field(default_factory=list)
    selected_paper_titles: Dict[str, str] = Field(default_factory=dict)
    gaps: List[ResearchGap] = Field(default_factory=list)
    summary_counts: Dict[str, int] = Field(default_factory=lambda: {
        "total_gaps": 0,
        "explicit_gaps": 0,
        "evidence_derived_gaps": 0,
        "cross_paper_gaps": 0,
        "contradictory_evidence": 0,
    })
    all_evidence_items: List[Any] = Field(default_factory=list)
    status: str = "completed"  # "completed", "insufficient_evidence", "error"
    execution_latency_ms: float = 0.0
    model_used: Optional[str] = None
    error_message: Optional[str] = None


class ResearchQuestionAnalysisResult(BaseModel):
    """
    Complete container for generated research questions, future pathways, summary counts,
    and end-to-end provenance traceability.
    """
    analysis_id: str
    query: str
    selected_paper_ids: List[str] = Field(default_factory=list)
    selected_paper_titles: Dict[str, str] = Field(default_factory=dict)
    linked_gap_ids: List[str] = Field(default_factory=list)
    questions: List[ResearchQuestion] = Field(default_factory=list)
    future_directions: List[FutureDirection] = Field(default_factory=list)
    summary_counts: Dict[str, int] = Field(default_factory=lambda: {
        "total_questions": 0,
        "author_inspired_questions": 0,
        "evidence_derived_questions": 0,
        "cross_paper_questions": 0,
        "future_directions": 0,
    })
    all_evidence_items: List[Any] = Field(default_factory=list)
    rejected_candidate_reasons: List[str] = Field(default_factory=list)
    status: str = "completed"  # "completed", "insufficient_evidence", "error"
    execution_latency_ms: float = 0.0
    model_used: Optional[str] = None
    error_message: Optional[str] = None
