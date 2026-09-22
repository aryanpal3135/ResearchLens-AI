"""
Evidence Bundler and Grounded RAG Context Builder for ResearchLens AI.
Prepares traceable, citation-ready research evidence bundles for downstream Foundry agents.
Adheres strictly to Phase 3 scope:
- Collects, filters, and formats retrieved evidence with provenance
- Token budget management to prevent context window overflow
- Does NOT execute LLM answer generation (reserved for Phase 4)
"""

from typing import List, Optional
from models.paper import DocumentChunk, RetrievalResult, EvidenceItem, EvidenceBundle
from services.chunking import estimate_tokens


class EvidenceBundler:
    """
    Constructs grounded evidence bundles and citation context blocks
    from hybrid retrieval results.
    """

    def __init__(self, default_token_budget: int = 3500):
        self.default_token_budget = default_token_budget

    def build_bundle(
        self,
        query: str,
        results: List[RetrievalResult],
        max_context_tokens: Optional[int] = None,
        retrieval_mode: str = "hybrid"
    ) -> EvidenceBundle:
        """
        Transforms raw retrieval results into an ordered EvidenceBundle
        with formatted LLM context within the allowed token budget.
        """
        budget = max_context_tokens or self.default_token_budget
        evidence_items: List[EvidenceItem] = []
        context_parts: List[str] = []
        accumulated_tokens = 0

        # Header for the context block
        header = (
            "=======================================================\n"
            "RETRIEVED RESEARCH EVIDENCE (GROUNDED CITATION CONTEXT)\n"
            f"Query: {query}\n"
            f"Retrieval Mode: {retrieval_mode.upper()}\n"
            "=======================================================\n"
        )
        context_parts.append(header)
        accumulated_tokens += estimate_tokens(header)

        for idx, res in enumerate(results, 1):
            chunk = res.chunk
            ev_id = f"ev_{idx:03d}"

            # Format page range label
            if chunk.page_start == chunk.page_end:
                page_str = f"p. {chunk.page_start}"
            else:
                page_str = f"pp. {chunk.page_start}-{chunk.page_end}"

            citation_label = (
                f"[Evidence {idx}: {chunk.paper_id}, {page_str}, {chunk.normalized_section} | Chunk {chunk.chunk_id}]"
            )

            # Gather source IDs
            sources = []
            if chunk.source_paragraph_ids:
                sources.extend(chunk.source_paragraph_ids)
            if chunk.source_table_ids:
                sources.extend(chunk.source_table_ids)
            if chunk.source_reference_ids:
                sources.extend(chunk.source_reference_ids)

            ev_item = EvidenceItem(
                evidence_id=ev_id,
                paper_id=chunk.paper_id,
                chunk_id=chunk.chunk_id,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                normalized_section=chunk.normalized_section,
                original_heading=chunk.original_heading,
                content_type=chunk.content_type,
                text=chunk.text,
                score=res.score,
                citation_label=citation_label,
                source_ids=sources,
            )

            item_block = (
                f"\n--- {citation_label} ---\n"
                f"Section: {chunk.original_heading} (Category: {chunk.normalized_section})\n"
                f"Type: {chunk.content_type} | Retrieval Score: {res.score:.4f}\n"
                f"Content:\n{chunk.text}\n"
            )

            item_tokens = estimate_tokens(item_block)
            if accumulated_tokens + item_tokens > budget and evidence_items:
                # Stop if budget would be exceeded
                break

            evidence_items.append(ev_item)
            context_parts.append(item_block)
            accumulated_tokens += item_tokens

        formatted_context = "\n".join(context_parts)

        return EvidenceBundle(
            query=query,
            retrieval_mode=retrieval_mode,
            total_found=len(results),
            items=evidence_items,
            formatted_context=formatted_context,
            estimated_context_tokens=accumulated_tokens,
        )
