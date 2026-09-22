"""
Literature Review Data Models for ResearchLens AI (Phase 8).
Defines typed schemas for academic literature synthesis, emergent themes,
substantive sections, categorical claim classifications, and evidence provenance.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from models.paper import EvidenceItem
from models.research_gap import ResearchGap, FutureDirection


# Permitted Claim Classifications
CLAIM_TYPES = [
    "DOCUMENTED",            # Directly supported by source paper evidence
    "SYNTHESIS",             # Formed by combining evidence across multiple papers
    "INFERENCE",             # Reasonable interpretation grounded in evidence, not explicit author claim
    "INSUFFICIENT_EVIDENCE", # Selected papers do not provide enough support
]

CLAIM_TYPE_LABELS = {
    "DOCUMENTED": "Documented in Literature",
    "SYNTHESIS": "Cross-Paper Synthesis",
    "INFERENCE": "Analytical Inference",
    "INSUFFICIENT_EVIDENCE": "Insufficient Evidence",
}

# Permitted Support Levels
SUPPORT_LEVELS = [
    "High evidence support",
    "Moderate evidence support",
    "Limited evidence support",
]


class LiteratureReviewTheme(BaseModel):
    """
    An emergent research theme synthesized from the actual retrieved paper evidence.
    Must not be hardcoded; emerges dynamically from paper corpus.
    """
    theme_id: str
    title: str
    description: str = ""
    supporting_papers: List[str] = Field(default_factory=list)
    evidence_items: List[EvidenceItem] = Field(default_factory=list)
    synthesis: str = ""
    support_level: str = "High evidence support"
    citation_labels: List[str] = Field(default_factory=list)

    def model_post_init(self, __context: Any) -> None:
        if self.evidence_items and not self.supporting_papers:
            pids = []
            for ev in self.evidence_items:
                pid = getattr(ev, "paper_id", None) or (ev.get("paper_id") if isinstance(ev, dict) else None)
                if pid and pid not in pids:
                    pids.append(pid)
            self.supporting_papers = pids
        if self.evidence_items and not self.citation_labels:
            labels = []
            for ev in self.evidence_items:
                cl = getattr(ev, "citation_label", None) or (ev.get("citation_label") if isinstance(ev, dict) else None)
                if cl and cl not in labels:
                    labels.append(cl)
            self.citation_labels = labels


class LiteratureReviewSection(BaseModel):
    """
    Substantive academic section within the literature review.
    Preserves categorical claim classification and verified evidence provenance.
    """
    section_id: str
    title: str
    content: str = ""
    claim_type: str = "DOCUMENTED"  # One of CLAIM_TYPES
    claim_type_label: str = "Documented in Literature"
    supporting_papers: List[str] = Field(default_factory=list)
    evidence_items: List[EvidenceItem] = Field(default_factory=list)
    citation_labels: List[str] = Field(default_factory=list)
    subsections: List[Dict[str, Any]] = Field(default_factory=list)
    is_insufficient_evidence: bool = False

    def model_post_init(self, __context: Any) -> None:
        if self.claim_type not in CLAIM_TYPES:
            self.claim_type = "SYNTHESIS" if len(self.supporting_papers) > 1 else "DOCUMENTED"
        self.claim_type_label = CLAIM_TYPE_LABELS.get(self.claim_type, "Documented in Literature")

        if "insufficient evidence" in self.content.lower() or self.claim_type == "INSUFFICIENT_EVIDENCE":
            self.is_insufficient_evidence = True
            self.claim_type = "INSUFFICIENT_EVIDENCE"
            self.claim_type_label = CLAIM_TYPE_LABELS["INSUFFICIENT_EVIDENCE"]

        if self.evidence_items and not self.supporting_papers:
            pids = []
            for ev in self.evidence_items:
                pid = getattr(ev, "paper_id", None) or (ev.get("paper_id") if isinstance(ev, dict) else None)
                if pid and pid not in pids:
                    pids.append(pid)
            self.supporting_papers = pids

        if self.evidence_items and not self.citation_labels:
            labels = []
            for ev in self.evidence_items:
                cl = getattr(ev, "citation_label", None) or (ev.get("citation_label") if isinstance(ev, dict) else None)
                if cl and cl not in labels:
                    labels.append(cl)
            self.citation_labels = labels


class LiteratureReview(BaseModel):
    """
    Complete, publication-grade academic Literature Review container.
    Synthesizes single or multiple papers adhering to the formal 11-section structure,
    end-to-end evidence provenance, and verified Phase 6/7 integration.
    """
    review_id: str
    title: str = "Academic Literature Review"
    review_question: str = ""
    selected_paper_ids: List[str] = Field(default_factory=list)
    selected_paper_titles: Dict[str, str] = Field(default_factory=dict)
    scope_style: str = "Comprehensive Scholarly Review"

    # 11 Formal Academic Sections
    introduction: LiteratureReviewSection = Field(default_factory=lambda: LiteratureReviewSection(
        section_id="sec_intro", title="1. Introduction & Research Scope", content=""
    ))
    themes: List[LiteratureReviewTheme] = Field(default_factory=list)  # Section 4: Major Themes
    methodology_synthesis: LiteratureReviewSection = Field(default_factory=lambda: LiteratureReviewSection(
        section_id="sec_methodology", title="2. Methodological Synthesis", content=""
    ))
    findings_synthesis: LiteratureReviewSection = Field(default_factory=lambda: LiteratureReviewSection(
        section_id="sec_findings", title="3. Findings & Evidence Synthesis", content=""
    ))
    agreements_differences: LiteratureReviewSection = Field(default_factory=lambda: LiteratureReviewSection(
        section_id="sec_agreements_differences", title="4. Agreements, Divergences & Comparative Synthesis", content=""
    ))
    limitations: LiteratureReviewSection = Field(default_factory=lambda: LiteratureReviewSection(
        section_id="sec_limitations", title="5. Limitations in the Reviewed Literature", content=""
    ))
    research_gaps: LiteratureReviewSection = Field(default_factory=lambda: LiteratureReviewSection(
        section_id="sec_gaps", title="6. Synthesized Research Gaps", content=""
    ))
    linked_gaps: List[ResearchGap] = Field(default_factory=list)  # Phase 6 gaps
    future_directions: List[FutureDirection] = Field(default_factory=list)  # Phase 7 pathways (AI suggestions)
    conclusion: LiteratureReviewSection = Field(default_factory=lambda: LiteratureReviewSection(
        section_id="sec_conclusion", title="7. Conclusion & Research Horizons", content=""
    ))

    # Evidence, Citations, & Reference Sources
    references: List[Dict[str, Any]] = Field(default_factory=list)
    all_evidence_items: List[EvidenceItem] = Field(default_factory=list)
    rejected_claims: List[Dict[str, Any]] = Field(default_factory=list)

    # Runtime Execution Metadata
    model: Optional[str] = "gpt-4.1-mini"
    retrieval_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0
    execution_latency_ms: float = 0.0
    summary_counts: Dict[str, int] = Field(default_factory=lambda: {
        "total_papers": 0,
        "total_chunks_retrieved": 0,
        "total_themes": 0,
        "total_gaps": 0,
        "total_future_directions": 0,
        "documented_claims": 0,
        "synthesis_claims": 0,
    })
    status: str = "completed"  # "completed", "insufficient_evidence", "error"
    error_message: Optional[str] = None

    @property
    def author_stated_directions(self) -> List[FutureDirection]:
        """Returns directions directly documented by authors in source paper text."""
        return [fd for fd in self.future_directions if not getattr(fd, "is_suggestion", True)]

    @property
    def ai_suggested_directions(self) -> List[FutureDirection]:
        """Returns exploratory research pathways formulated as AI suggestions."""
        return [fd for fd in self.future_directions if getattr(fd, "is_suggestion", False)]
