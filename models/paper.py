"""
Paper, Section, Paragraph, and Document chunking data models.
Provides structured provenance, RAG-ready paragraph units, and metadata tracking.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime, timezone


class ExtractedParagraph(BaseModel):
    """
    Granular, RAG/chunk-ready extracted paragraph with strict provenance and semantic type.
    Ready to be indexed or converted into semantic RAG chunks in Phase 3.
    """
    paper_id: str                          # e.g. "paper_001"
    paragraph_id: str                      # e.g. "paper_001_p002_para003"
    page: int                              # 1-indexed page number
    original_heading: str = "General"      # Raw author heading (e.g. "3. Materials and Methods")
    normalized_section: str = "General"    # Standardized category (e.g. "Methodology", "Front Matter")
    content_type: str = "body"             # "front_matter", "title", "author", "affiliation", "publication_metadata", "abstract", "body", "reference"
    text: str                              # Clean paragraph text
    char_count: int = 0                    # Character length


class ExtractedSection(BaseModel):
    """
    Cohesive academic section spanning one or more pages.
    Preserves both original author heading and normalized taxonomy category.
    """
    paper_id: str                          # e.g. "paper_001"
    section_id: str                        # e.g. "paper_001_sec002"
    original_heading: str                  # e.g. "III. MATERIALS AND METHODS"
    normalized_section: str                # e.g. "Methodology"
    start_page: int
    end_page: int
    paragraph_ids: List[str] = Field(default_factory=list)
    full_text: str = ""


class ExtractedTable(BaseModel):
    """Structured table extracted from research paper with validation status and extraction quality."""
    paper_id: str                          # e.g. "paper_001"
    table_id: str                          # e.g. "paper_001_t001"
    page: int                              # Page number where table appears
    caption: Optional[str] = None          # Optional detected table caption
    markdown_content: str                  # Markdown representation of the table
    row_count: int = 0
    col_count: int = 0
    status: str = "Valid"                  # "Valid", "Needs Review", "Rejected"
    extraction_quality: str = "high"       # "high", "medium", "low"
    validation_reason: Optional[str] = None


class ExtractedReference(BaseModel):
    """Single parsed bibliographic reference entry."""
    paper_id: str                          # e.g. "paper_001"
    ref_id: str                            # e.g. "paper_001_ref001"
    raw_text: str                          # Full reference text string
    authors: Optional[str] = None
    year: Optional[str] = None
    title: Optional[str] = None


class PageContent(BaseModel):
    """Page-level container with text density and scanned detection."""
    paper_id: str
    page_number: int                       # 1-indexed
    char_count: int = 0
    raw_text: str = ""
    is_scanned_page: bool = False
    table_count: int = 0


class PaperMetadata(BaseModel):
    """Extracted or inferred metadata for a single research paper."""
    paper_id: str = "paper_001"
    title: Optional[str] = "Untitled Paper"
    authors: List[str] = Field(default_factory=list)
    publication_year: Optional[str] = None
    venue: Optional[str] = None
    doi: Optional[str] = None
    total_pages: int = 0
    file_size_bytes: int = 0
    is_scanned: bool = False               # Flag indicating little/no digital text
    text_density_chars_per_page: float = 0.0
    abstract: Optional[str] = None
    keywords: List[str] = Field(default_factory=list)


class DocumentChunk(BaseModel):
    """
    Granular chunk of research paper text with end-to-end source traceability,
    document order tracking, and multi-field provenance for Phase 3 RAG.
    """
    paper_id: str
    chunk_id: str                          # e.g. "paper_001_c001"
    document_order: int = 1                # 1-indexed document reading sequence
    chunk_order: int = 1                   # Explicit alias for document reading order
    chunk_index: int = 0                   # 0-indexed index within paper chunks
    previous_chunk_id: Optional[str] = None
    next_chunk_id: Optional[str] = None
    page: int = 1                          # Primary page for backwards compatibility
    page_start: int = 1                    # First page covered by chunk
    page_end: int = 1                      # Last page covered by chunk
    section: str = "General"               # Backwards compatibility alias
    original_section: str = "General"      # Backwards compatibility alias
    original_heading: str = "General"      # Exact author section heading
    normalized_section: str = "General"    # Standardized section taxonomy
    content_type: str = "body"             # "body", "abstract", "table", "reference", "front_matter"
    text: str                              # Clean chunk content text
    token_count: int = 0                   # Accurate token estimation
    token_estimate: Optional[int] = None   # Backwards compatibility alias
    char_count: int = 0
    source_paragraph_ids: List[str] = Field(default_factory=list)
    source_table_ids: List[str] = Field(default_factory=list)
    source_reference_ids: List[str] = Field(default_factory=list)
    embedding: Optional[List[float]] = None
    embedding_metadata: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    """Single retrieval result item containing chunk, fused score, and channel breakdown."""
    chunk: DocumentChunk
    score: float                           # Primary score (RRF or similarity)
    fused_score: Optional[float] = None    # Combined RRF score
    vector_score: Optional[float] = None   # Cosine similarity (if computed)
    dense_score: Optional[float] = None    # Explicit alias for vector_score
    keyword_score: Optional[float] = None  # BM25 score (if computed)
    bm25_score: Optional[float] = None     # Explicit alias for keyword_score
    rank: int = 0                          # 1-indexed rank in result set
    match_type: str = "hybrid"             # "hybrid", "vector", "keyword"



class EvidenceItem(BaseModel):
    """Structured evidence item for RAG context and grounded synthesis."""
    evidence_id: str                       # e.g. "ev_001"
    paper_id: str
    chunk_id: str
    page_start: int
    page_end: int
    normalized_section: str
    original_heading: str
    content_type: str
    text: str
    score: float
    citation_label: str                    # e.g. "[Evidence 1: paper_001, p. 3, Methodology]"
    source_ids: List[str] = Field(default_factory=list)


class EvidenceBundle(BaseModel):
    """Complete bundle of grounded evidence retrieved for a research query."""
    query: str
    retrieval_mode: str
    total_found: int
    items: List[EvidenceItem] = Field(default_factory=list)
    formatted_context: str = ""
    estimated_context_tokens: int = 0



class PaperDocument(BaseModel):
    """Primary document representation for uploaded papers."""
    id: str                                # Sequential ID: "paper_001", "paper_002", etc.
    filename: str
    saved_path: str
    upload_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    is_valid_pdf: bool = True
    is_scanned: bool = False               # Requires OCR alert
    page_count: int = 0
    file_size_kb: float = 0.0
    metadata: PaperMetadata = Field(default_factory=PaperMetadata)
    
    # Deep extracted entities
    pages: List[PageContent] = Field(default_factory=list)
    sections: List[ExtractedSection] = Field(default_factory=list)
    paragraphs: List[ExtractedParagraph] = Field(default_factory=list)
    tables: List[ExtractedTable] = Field(default_factory=list)
    references: List[ExtractedReference] = Field(default_factory=list)
    
    # Chunking & status
    chunks: List[DocumentChunk] = Field(default_factory=list)
    raw_text: Optional[str] = None
    status: str = "uploaded"               # uploaded, processed, scanned_ocr_required, analyzed, failed
    error_message: Optional[str] = None
