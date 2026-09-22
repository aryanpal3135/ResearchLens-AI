"""
Citation & Evidence Tracking Service for ResearchLens AI.
Ensures every extracted claim has verifiable evidence (page number, section, quote)
or explicitly states 'Evidence not found'. Never fabricates citations.
"""

from typing import Optional, Dict, Any
from models.analysis import EvidenceCitation


class CitationService:
    """Service to validate and format citations."""

    @staticmethod
    def format_citation(citation: Optional[EvidenceCitation]) -> str:
        """Formats a citation into a traceable human-readable badge/string."""
        if not citation:
            return "⚠️ Evidence not found."

        parts = [f"**Paper:** {citation.paper_filename}"]
        if citation.page:
            parts.append(f"**Page:** {citation.page}")
        if citation.section:
            parts.append(f"**Section:** {citation.section}")

        attribution = "Explicitly Stated" if citation.is_explicit else "AI Inferred"
        parts.append(f"**Attribution:** {attribution}")
        parts.append(f"**Confidence:** {citation.confidence}")

        return " | ".join(parts)
