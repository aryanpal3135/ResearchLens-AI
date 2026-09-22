"""
Structure-Aware Document Chunking Service for ResearchLens AI.
Implements token-bounded, section-bounded chunking preserving strict provenance,
document order, sequential linkage, table extraction, and bibliographic references.
"""

import re
from typing import List, Optional, Dict, Any, Tuple
from config.settings import settings

from models.paper import DocumentChunk, PaperDocument, ExtractedParagraph, ExtractedTable, ExtractedReference


def estimate_tokens(text: str) -> int:
    """
    Accurate token estimation for academic text.
    Uses tiktoken if installed; otherwise uses calibrated whitespace/punctuation estimation.
    """
    if not text:
        return 0
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        # Calibrated heuristic: 1 word ≈ 1.33 tokens, plus punctuation
        words = text.split()
        if not words:
            return max(1, len(text) // 4)
        return max(1, int(len(words) * 1.32 + text.count("\n") * 0.5))


class DocumentChunker:
    """
    Structure-aware, provenance-preserving chunker for academic research papers.
    Adheres strictly to Phase 3 requirements:
    - Target: 600 tokens (min 100, max 900, overlap 80)
    - Never crosses major section boundaries
    - Full document_order, chunk_index, previous_chunk_id, and next_chunk_id linkage
    - Dedicated chunks for validated tables and references
    - 100% coverage of extracted text with zero permanent data loss
    """

    def __init__(
        self,
        target_chunk_size: Optional[int] = None,
        overlap: Optional[int] = None,
        min_chunk_size: Optional[int] = None,
        max_chunk_size: Optional[int] = None,
    ):
        self.target_tokens = target_chunk_size or settings.CHUNK_TARGET_TOKENS
        self.overlap_tokens = overlap or settings.CHUNK_OVERLAP_TOKENS
        self.min_tokens = min_chunk_size or settings.CHUNK_MIN_TOKENS
        self.max_tokens = max_chunk_size or settings.CHUNK_MAX_TOKENS

    def chunk_document(self, paper: PaperDocument) -> List[DocumentChunk]:
        """
        Processes a full PaperDocument into a list of linked DocumentChunks.
        Preserves complete document sequence:
        1. Front matter & Abstract
        2. Body sections (grouped by section without crossing boundaries)
        3. Tables
        4. Bibliographic References
        """
        raw_chunks: List[Dict[str, Any]] = []

        # 1. Process Extracted Paragraphs
        if paper.paragraphs:
            raw_chunks.extend(self._chunk_paragraphs(paper.paragraphs, paper.id))
        elif paper.raw_text:
            # Fallback if only raw text is present (e.g. basic tests)
            raw_chunks.extend(self._chunk_raw_text(paper.raw_text, paper.id))

        # 2. Process Tables into Dedicated Chunks
        if paper.tables:
            raw_chunks.extend(self._chunk_tables(paper.tables, paper.id))

        # 3. Process References into Dedicated Chunks
        if paper.references:
            raw_chunks.extend(self._chunk_references(paper.references, paper.id))

        if not raw_chunks:
            return []

        # 4. Finalize IDs, Document Order, and Bidirectional Linkages
        final_chunks: List[DocumentChunk] = []
        total_chunks = len(raw_chunks)

        for i, item in enumerate(raw_chunks):
            chunk_num = i + 1
            chunk_id = f"{paper.id}_c{chunk_num:03d}"
            prev_id = f"{paper.id}_c{(chunk_num - 1):03d}" if chunk_num > 1 else None
            next_id = f"{paper.id}_c{(chunk_num + 1):03d}" if chunk_num < total_chunks else None

            text = item["text"]
            token_count = item.get("token_count") or estimate_tokens(text)

            chunk = DocumentChunk(
                paper_id=paper.id,
                chunk_id=chunk_id,
                document_order=chunk_num,
                chunk_order=chunk_num,
                chunk_index=i,
                previous_chunk_id=prev_id,
                next_chunk_id=next_id,
                page=item.get("page_start", 1),
                page_start=item.get("page_start", 1),
                page_end=item.get("page_end", item.get("page_start", 1)),
                section=item.get("normalized_section", "General"),
                original_section=item.get("original_heading", "General"),
                original_heading=item.get("original_heading", "General"),
                normalized_section=item.get("normalized_section", "General"),
                content_type=item.get("content_type", "body"),
                text=text,
                token_count=token_count,
                token_estimate=token_count,
                char_count=len(text),
                source_paragraph_ids=item.get("source_paragraph_ids", []),
                source_table_ids=item.get("source_table_ids", []),
                source_reference_ids=item.get("source_reference_ids", []),
                metadata=item.get("metadata", {}),
            )
            final_chunks.append(chunk)

        # Update paper document chunks
        paper.chunks = final_chunks
        return final_chunks

    def _chunk_paragraphs(
        self, paragraphs: List[ExtractedParagraph], paper_id: str
    ) -> List[Dict[str, Any]]:
        """
        Groups paragraphs sequentially respecting section boundaries.
        Aggregates consecutive paragraphs within the same normalized section
        up to target_tokens without exceeding max_tokens, with paragraph-aware sliding overlap.
        """
        chunks_data: List[Dict[str, Any]] = []

        current_section = None
        current_heading = None
        current_type = None
        buffer_paras: List[ExtractedParagraph] = []
        buffer_tokens = 0

        def flush_buffer() -> List[ExtractedParagraph]:
            nonlocal buffer_paras, buffer_tokens
            if not buffer_paras:
                return []

            text = "\n\n".join(p.text for p in buffer_paras).strip()
            if text:
                chunks_data.append({
                    "text": text,
                    "token_count": buffer_tokens,
                    "page_start": min(p.page for p in buffer_paras),
                    "page_end": max(p.page for p in buffer_paras),
                    "normalized_section": current_section or "General",
                    "original_heading": buffer_paras[0].original_heading or current_heading or "General",
                    "content_type": current_type or "body",
                    "source_paragraph_ids": [p.paragraph_id for p in buffer_paras],
                    "metadata": {
                        "paragraph_count": len(buffer_paras),
                    }
                })

            flushed = list(buffer_paras)
            buffer_paras = []
            buffer_tokens = 0
            return flushed

        def compute_overlap(flushed_paras: List[ExtractedParagraph]) -> Tuple[List[ExtractedParagraph], int]:
            overlap_list: List[ExtractedParagraph] = []
            overlap_tok = 0
            for p in reversed(flushed_paras):
                pt = estimate_tokens(p.text)
                if overlap_tok + pt <= self.overlap_tokens:
                    overlap_list.insert(0, p)
                    overlap_tok += pt
                else:
                    break
            # If no paragraph fit but last paragraph is reasonably sized (<= 120 tokens), keep it
            if not overlap_list and flushed_paras and estimate_tokens(flushed_paras[-1].text) <= 120:
                overlap_list = [flushed_paras[-1]]
                overlap_tok = estimate_tokens(flushed_paras[-1].text)
            return overlap_list, overlap_tok

        for para in paragraphs:
            para_text = (para.text or "").strip()
            if not para_text:
                continue

            para_tokens = estimate_tokens(para_text)
            sec = para.normalized_section or "General"
            heading = para.original_heading or "General"
            c_type = para.content_type or "body"

            # 1. Section change or content type change (Front Matter vs Body, etc.)
            section_changed = (
                current_section is not None
                and (sec != current_section or c_type != current_type)
            )
            if section_changed:
                flush_buffer()
                buffer_paras = []
                buffer_tokens = 0

            # 2. Heading change within section: keep subsection cohesive if buffer is already substantial
            elif current_heading is not None and heading != current_heading and buffer_tokens >= self.min_tokens:
                flushed = flush_buffer()
                buffer_paras, buffer_tokens = compute_overlap(flushed)

            # 3. Buffer capacity reached: target tokens exceeded
            elif buffer_tokens > 0 and (buffer_tokens + para_tokens > self.target_tokens):
                flushed = flush_buffer()
                buffer_paras, buffer_tokens = compute_overlap(flushed)

            current_section = sec
            current_heading = heading
            current_type = c_type

            # If a single paragraph is larger than max_tokens, split it internally
            if para_tokens > self.max_tokens:
                flushed = flush_buffer()
                sub_chunks = self._split_large_paragraph(para)
                chunks_data.extend(sub_chunks)
                buffer_paras = []
                buffer_tokens = 0
                continue

            buffer_paras.append(para)
            buffer_tokens += para_tokens

        flush_buffer()
        return chunks_data


    def _split_large_paragraph(self, para: ExtractedParagraph) -> List[Dict[str, Any]]:
        """Splits an oversized paragraph into sentence-bounded chunks with overlap."""
        sentences = re.split(r'(?<=[.!?])\s+', para.text.strip())
        sub_chunks: List[Dict[str, Any]] = []

        curr_sentences: List[str] = []
        curr_tokens = 0

        for s in sentences:
            s_tok = estimate_tokens(s)
            if curr_tokens + s_tok > self.target_tokens and curr_sentences:
                chunk_text = " ".join(curr_sentences).strip()
                sub_chunks.append({
                    "text": chunk_text,
                    "token_count": curr_tokens,
                    "page_start": para.page,
                    "page_end": para.page,
                    "normalized_section": para.normalized_section or "General",
                    "original_heading": para.original_heading or "General",
                    "content_type": para.content_type or "body",
                    "source_paragraph_ids": [para.paragraph_id],
                    "metadata": {"oversized_split": True}
                })
                # Overlap: keep last sentence if small
                if s_tok < self.overlap_tokens:
                    curr_sentences = [curr_sentences[-1], s]
                    curr_tokens = estimate_tokens(" ".join(curr_sentences))
                else:
                    curr_sentences = [s]
                    curr_tokens = s_tok
            else:
                curr_sentences.append(s)
                curr_tokens += s_tok

        if curr_sentences:
            chunk_text = " ".join(curr_sentences).strip()
            sub_chunks.append({
                "text": chunk_text,
                "token_count": curr_tokens,
                "page_start": para.page,
                "page_end": para.page,
                "normalized_section": para.normalized_section or "General",
                "original_heading": para.original_heading or "General",
                "content_type": para.content_type or "body",
                "source_paragraph_ids": [para.paragraph_id],
                "metadata": {"oversized_split": True}
            })

        return sub_chunks

    def _chunk_tables(
        self, tables: List[ExtractedTable], paper_id: str
    ) -> List[Dict[str, Any]]:
        """Converts extracted tables into dedicated, searchable table chunks."""
        table_chunks: List[Dict[str, Any]] = []

        for tbl in tables:
            # We index valid or reviewable tables
            if tbl.status == "Rejected":
                continue

            caption_str = f"Caption: {tbl.caption}\n" if tbl.caption else ""
            table_text = (
                f"### [Table {tbl.table_id} - Page {tbl.page}]\n"
                f"{caption_str}"
                f"{tbl.markdown_content}\n"
                f"Dimensions: {tbl.row_count} rows x {tbl.col_count} columns"
            )

            tok_count = estimate_tokens(table_text)
            table_chunks.append({
                "text": table_text,
                "token_count": tok_count,
                "page_start": tbl.page,
                "page_end": tbl.page,
                "normalized_section": "Tables",
                "original_heading": tbl.caption or f"Table {tbl.table_id}",
                "content_type": "table",
                "source_table_ids": [tbl.table_id],
                "metadata": {
                    "table_id": tbl.table_id,
                    "row_count": tbl.row_count,
                    "col_count": tbl.col_count,
                    "quality": tbl.extraction_quality,
                    "status": tbl.status
                }
            })

        return table_chunks

    def _chunk_references(
        self, references: List[ExtractedReference], paper_id: str
    ) -> List[Dict[str, Any]]:
        """Groups extracted references into structured reference chunks."""
        ref_chunks: List[Dict[str, Any]] = []
        batch_size = 8  # ~8 references per chunk fits neatly in ~400-600 tokens

        for i in range(0, len(references), batch_size):
            batch = references[i:i + batch_size]
            ref_texts = [f"[{r.ref_id}] {r.raw_text}" for r in batch if r.raw_text]
            if not ref_texts:
                continue

            chunk_text = "### [References / Bibliography]\n" + "\n\n".join(ref_texts)
            tok_count = estimate_tokens(chunk_text)

            ref_chunks.append({
                "text": chunk_text,
                "token_count": tok_count,
                "page_start": 1,  # Reference pages are typically near end
                "page_end": 1,
                "normalized_section": "References",
                "original_heading": "References",
                "content_type": "reference",
                "source_reference_ids": [r.ref_id for r in batch],
                "metadata": {
                    "reference_count": len(batch),
                    "first_ref": batch[0].ref_id,
                    "last_ref": batch[-1].ref_id
                }
            })

        return ref_chunks

    def _chunk_raw_text(self, text: str, paper_id: str) -> List[Dict[str, Any]]:
        """Fallback chunker for plain text documents without paragraph structures."""
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks_data: List[Dict[str, Any]] = []
        curr_paras: List[str] = []
        curr_tok = 0

        for p in paragraphs:
            tok = estimate_tokens(p)
            if curr_tok + tok > self.target_tokens and curr_paras:
                chunks_data.append({
                    "text": "\n\n".join(curr_paras),
                    "token_count": curr_tok,
                    "page_start": 1,
                    "page_end": 1,
                    "normalized_section": "General",
                    "original_heading": "General",
                    "content_type": "body"
                })
                curr_paras = [p]
                curr_tok = tok
            else:
                curr_paras.append(p)
                curr_tok += tok

        if curr_paras:
            chunks_data.append({
                "text": "\n\n".join(curr_paras),
                "token_count": curr_tok,
                "page_start": 1,
                "page_end": 1,
                "normalized_section": "General",
                "original_heading": "General",
                "content_type": "body"
            })

        return chunks_data
