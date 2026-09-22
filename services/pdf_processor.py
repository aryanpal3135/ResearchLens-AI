"""
PDF Processing Service for ResearchLens AI.
Handles general-purpose academic PDF extraction:
- Advanced metadata extraction (Title, Authors, Year, Abstract) distinguishing copyright notices
- Pure abstract detection without front-matter contamination
- Robust reading-order academic section detection with parent section inheritance for subsections
- Multi-feature Table Validation (Valid, Needs Review, Rejected) eliminating false positives
- Granular RAG-ready paragraph units with strict provenance
- Structured bibliographic reference parsing
- Scanned / Image-only PDF detection
"""

from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import os
import re

try:
    import pymupdf as fitz
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

from models.paper import (
    PaperDocument,
    PaperMetadata,
    PageContent,
    ExtractedSection,
    ExtractedParagraph,
    ExtractedTable,
    ExtractedReference,
)


class PDFProcessor:
    """Service to safely process, validate, and deeply extract research paper PDFs."""

    # Academic section normalization dictionary
    SECTION_PATTERNS = [
        (r"^(?:abstract|executive summary)\b", "Abstract"),
        (r"^(?:(?:\d+(?:\.\d+)*|[ivxlcdm]+)\.?\s+)?(?:introduction|motivation)\b", "Introduction"),
        (r"^(?:(?:\d+(?:\.\d+)*|[ivxlcdm]+)\.?\s+)?(?:background|preliminaries|overview)\b", "Background"),
        (r"^(?:(?:\d+(?:\.\d+)*|[ivxlcdm]+)\.?\s+)?(?:related work|literature review|prior work|state of the art)\b", "Related Work"),
        (r"^(?:(?:\d+(?:\.\d+)*|[ivxlcdm]+)\.?\s+)?(?:methodology|materials and methods|proposed method|model architecture|system design|approach|proposed architecture|methods|our method|network architecture)\b", "Methodology"),
        (r"^(?:(?:\d+(?:\.\d+)*|[ivxlcdm]+)\.?\s+)?(?:experimental setup|experiments|implementation details|evaluation setup|experimental design|experimental evaluation|training|problem statement)\b", "Experimental Setup"),
        (r"^(?:(?:\d+(?:\.\d+)*|[ivxlcdm]+)\.?\s+)?(?:(?:empirical\s+|experimental\s+|main\s+)?results|empirical evaluation|performance comparison|findings|empirical experiments)\b", "Results"),
        (r"^(?:(?:\d+(?:\.\d+)*|[ivxlcdm]+)\.?\s+)?(?:discussion|analysis|ablation study|ablation analysis|why self-attention)\b", "Discussion"),
        (r"^(?:(?:\d+(?:\.\d+)*|[ivxlcdm]+)\.?\s+)?(?:limitations|threats to validity|constraints)\b", "Limitations"),
        (r"^(?:(?:\d+(?:\.\d+)*|[ivxlcdm]+)\.?\s+)?(?:future work|future directions|future research|next steps)\b", "Future Work"),
        (r"^(?:(?:\d+(?:\.\d+)*|[ivxlcdm]+)\.?\s+)?(?:conclusion|concluding remarks|summary and conclusion|conclusions)\b", "Conclusion"),
        (r"^(?:(?:\d+(?:\.\d+)*|[ivxlcdm]+)\.?\s+)?(?:references|bibliography|works cited)\b", "References"),
    ]

    NOTICE_PATTERNS = [
        r"permission\s+to\s+(?:make|reproduce|copy)",
        r"grants\s+permission",
        r"all\s+rights\s+reserved",
        r"copyright\b",
        r"creative\s+commons",
        r"this\s+paper\s+(?:is|solely)\s+for\s+use",
        r"downloaded\s+from",
        r"open\s+access",
        r"reprinted\s+with\s+permission",
        r"provided\s+proper\s+attribution",
    ]

    AFFILIATION_KEYWORDS = [
        "university", "dept", "department", "institute", "laboratory", "lab", "school",
        "college", "google", "microsoft", "meta", "facebook", "amazon", "research",
        "brain", "toronto", "stanford", "berkeley", "mit", "cmu", "centre", "center",
        "corporation", "inc.", "ltd", "technion", "@", "faculty", "division", "campus"
    ]

    def __init__(self, upload_dir: Path):
        self.upload_dir = upload_dir
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def generate_paper_id(current_count: int) -> str:
        """Generates sequential, user-friendly Paper IDs: paper_001, paper_002, etc."""
        return f"paper_{current_count + 1:03d}"

    def validate_pdf(self, file_bytes: bytes, filename: str) -> Tuple[bool, str]:
        """
        Validate whether the uploaded file is a healthy, non-corrupted PDF.
        Returns: (is_valid, error_or_success_message)
        """
        if not filename.lower().endswith(".pdf"):
            return False, f"File '{filename}' does not have a .pdf extension."

        if len(file_bytes) == 0:
            return False, f"File '{filename}' is completely empty (0 bytes)."

        # Check standard PDF magic header '%PDF-'
        if not file_bytes.startswith(b"%PDF-"):
            return False, f"File '{filename}' does not contain a valid PDF signature."

        if PYMUPDF_AVAILABLE:
            try:
                doc = fitz.open(stream=file_bytes, filetype="pdf")
                page_count = doc.page_count
                doc.close()
                if page_count == 0:
                    return False, f"File '{filename}' has no readable pages."
                return True, f"Valid PDF with {page_count} pages."
            except Exception as e:
                return False, f"Corrupted PDF structure: {str(e)}"

        return True, "Valid PDF header format."

    def save_uploaded_file(self, file_bytes: bytes, filename: str) -> Path:
        """Saves file to upload directory without modifying original binary."""
        safe_filename = os.path.basename(filename)
        destination = self.upload_dir / safe_filename
        with open(destination, "wb") as f:
            f.write(file_bytes)
        return destination

    def is_notice_text(self, text: str) -> bool:
        """Identifies copyright, license, or distribution notice boilerplate."""
        lower = text.lower()
        return any(re.search(pat, lower) for pat in self.NOTICE_PATTERNS)

    def normalize_section_heading(self, heading: str, parent_norm_map: Optional[Dict[str, str]] = None) -> str:
        """Maps an author's raw section heading to a standardized academic category."""
        clean = heading.strip()

        # Check direct patterns
        for pattern, normalized in self.SECTION_PATTERNS:
            if re.search(pattern, clean, re.IGNORECASE):
                return normalized

        # Check if it's a numbered subsection (e.g. "3.1", "5.2") and inherit parent category
        if parent_norm_map:
            sub_m = re.match(r"^(\d+)\.\d+", clean)
            if sub_m:
                parent_prefix = sub_m.group(1)
                if parent_prefix in parent_norm_map:
                    return parent_norm_map[parent_prefix]

        # Academic keyword fallback
        lower = clean.lower()
        if any(w in lower for w in ["background", "preliminar", "overview"]):
            return "Background"
        if any(w in lower for w in ["method", "model", "architecture", "algorithm", "design", "formulation"]):
            return "Methodology"
        if any(w in lower for w in ["experiment", "dataset", "setup", "training", "parameter", "benchmark"]):
            return "Experimental Setup"
        if any(w in lower for w in ["result", "performance", "evaluation", "score", "metric", "accuracy"]):
            return "Results"
        if any(w in lower for w in ["discuss", "ablation", "analysis", "study", "why "]):
            return "Discussion"
        if any(w in lower for w in ["conclud", "summary"]):
            return "Conclusion"

        return "General"

    def is_academic_heading(
        self, text_line: str, parent_norm_map: Optional[Dict[str, str]] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Determines whether a text line qualifies as an academic section heading.
        Avoids false positives from captions, list items, or prose sentences.
        """
        cleaned = text_line.strip()
        if not cleaned or len(cleaned) > 85 or len(cleaned) < 2:
            return False, None

        # Exclude sentences ending with punctuation
        if cleaned.endswith((".", ",", ";", ":", "?", "!")) and not re.match(r"^\d+\.$", cleaned):
            return False, None

        # Numbered headings: "1 Introduction", "3.2 Model Architecture", "IV. EXPERIMENTS"
        if re.match(r"^(?:(?:\d+(?:\.\d+)*|[IVXLCDM]+)\.?\s+)[A-Z]", cleaned):
            return True, self.normalize_section_heading(cleaned, parent_norm_map)

        # Standalone recognized academic headings
        lower = cleaned.lower()
        for pattern, normalized in self.SECTION_PATTERNS:
            if re.match(pattern, lower):
                if len(cleaned.split()) <= 6:
                    return True, normalized

        return False, None

    def reconstruct_table_cells(self, data: List[List[Any]]) -> List[List[str]]:
        """
        Reconstructs table cells by splitting multi-line rows where all populated columns
        have matching internal line breaks (resolving PyMuPDF merged rows), and normalizes cell text.
        """
        if not data:
            return []
        new_rows: List[List[str]] = []
        for row in data:
            lines_per_col = [
                str(c).strip().split("\n") if c is not None and str(c).strip() else [""]
                for c in row
            ]
            max_lines = max(len(l) for l in lines_per_col)
            non_empty_lens = [len(l) for l in lines_per_col if any(x.strip() for x in l)]
            if max_lines > 1 and len(set(non_empty_lens)) == 1 and len(non_empty_lens) >= 2:
                for i in range(max_lines):
                    new_rows.append([
                        lines_per_col[c_idx][i].strip() if i < len(lines_per_col[c_idx]) else ""
                        for c_idx in range(len(row))
                    ])
            else:
                new_rows.append([
                    str(c).strip().replace("\n", " ") if c is not None else ""
                    for c in row
                ])
        return new_rows

    def validate_table_candidate(self, data: List[List[Any]]) -> Tuple[str, str, str]:
        """
        Validates candidate table from find_tables().
        Distinguishes real academic tables from false positives caused by text layouts / visualizers,
        and computes extraction quality.
        Returns: (status, extraction_quality, reason)
        where status in ['Valid', 'Needs Review', 'Rejected']
        and extraction_quality in ['high', 'medium', 'low'].
        """
        if not data or len(data) < 2:
            return "Rejected", "low", "Fewer than 2 rows"

        rows = len(data)
        cols = len(data[0]) if data else 0
        if cols < 2:
            return "Rejected", "low", "Fewer than 2 columns"

        # Calculate cell metrics
        cells = []
        for r in data:
            for c in r:
                val = str(c).strip() if c is not None else ""
                if val:
                    cells.append(val)

        total_grid = rows * cols
        pop_ratio = len(cells) / max(1, total_grid)

        # Rule 1: Sparse grid check
        if pop_ratio < 0.35:
            return "Rejected", "low", f"Sparse grid: only {pop_ratio:.1%} populated"

        avg_cell_len = sum(len(c) for c in cells) / max(1, len(cells))
        one_char_cells = sum(1 for c in cells if len(c) == 1)
        one_char_ratio = one_char_cells / max(1, len(cells))

        # Rule 2: Unrealistic column count for research papers
        if cols > 20:
            return "Rejected", "low", f"Excessive columns ({cols}) with average length {avg_cell_len:.1f}"

        # Rule 3: Fragmented single-character words
        if cols >= 8 and avg_cell_len < 3.0 and one_char_ratio > 0.20:
            return "Rejected", "low", f"Fragmented text: avg length {avg_cell_len:.1f}, {one_char_ratio:.1%} 1-char cells"

        # Rule 4: Consecutive prose words split into columns
        prose_row_count = 0
        for r in data:
            words = [str(c).strip() for c in r if c is not None and str(c).strip()]
            if len(words) >= 4:
                line_str = " ".join(words).lower()
                if any(phrase in line_str for phrase in ["the ", "is ", "in ", "of ", "that ", "with "]):
                    if sum(len(w) for w in words) / len(words) < 5.0 and len(words) >= cols * 0.5:
                        prose_row_count += 1
        if prose_row_count >= 3 and cols >= 10:
            return "Rejected", "low", "Rows resemble prose sentences split across columns"

        # Rule 5: Empty header
        first_row_non_empty = [str(c).strip() for c in data[0] if c is not None and str(c).strip()]
        if len(first_row_non_empty) == 0:
            return "Needs Review", "medium", "Header row is completely empty"

        if cols > 15:
            return "Needs Review", "medium", f"Unusually high column count ({cols})"

        # Determine extraction quality for valid table
        has_merged_nums = any(len(re.findall(r"\b\d+(?:\.\d+)?\b", str(c))) >= 4 for r in data for c in r)
        header_empty_first = (not str(data[0][0]).strip())
        has_multiline_header = any("\n" in str(c) for c in data[0])

        if has_merged_nums or (header_empty_first and has_multiline_header):
            return "Valid", "low", "Valid academic table with merged subcolumns / multi-line headers"
        elif has_multiline_header or pop_ratio < 0.6:
            return "Valid", "medium", "Valid table with minor formatting noise"

        return "Valid", "high", "Clean academic table structure"

    def _extract_tables_from_page(
        self, page: Any, paper_id: str, page_number: int, start_idx: int
    ) -> Tuple[List[ExtractedTable], int]:
        """Extracts, reconstructs, and validates tables from a single page."""
        extracted_tables: List[ExtractedTable] = []
        curr_idx = start_idx

        try:
            tab_finder = page.find_tables()
            for tab in tab_finder.tables:
                raw_data = tab.extract()
                reconstructed_data = self.reconstruct_table_cells(raw_data)
                status, quality, reason = self.validate_table_candidate(reconstructed_data)

                # Discard rejected false-positives
                if status == "Rejected":
                    continue

                curr_idx += 1
                table_id = f"{paper_id}_t{curr_idx:03d}"

                # Format into clean Markdown table
                headers = [str(col).strip() if col is not None else "" for col in reconstructed_data[0]]
                sep = ["---"] * len(headers)
                md_lines = [
                    "| " + " | ".join(headers) + " |",
                    "| " + " | ".join(sep) + " |",
                ]
                for row in reconstructed_data[1:]:
                    row_str = [str(c).strip() if c is not None else "" for c in row]
                    md_lines.append("| " + " | ".join(row_str) + " |")

                extracted_tables.append(
                    ExtractedTable(
                        paper_id=paper_id,
                        table_id=table_id,
                        page=page_number,
                        markdown_content="\n".join(md_lines),
                        row_count=len(reconstructed_data),
                        col_count=len(headers),
                        status=status,
                        extraction_quality=quality,
                        validation_reason=reason,
                    )
                )
        except Exception:
            pass

        return extracted_tables, curr_idx

    def _extract_references_from_text(self, ref_text: str, paper_id: str) -> List[ExtractedReference]:
        """Splits raw references text into structured bibliography entries."""
        references: List[ExtractedReference] = []
        if not ref_text.strip():
            return references

        entries = re.split(r"(?:(?:\n|\s+)\[\d+\]|\n\s*\d+\.\s+)", "\n" + ref_text)
        ref_counter = 0

        for entry in entries:
            clean_entry = entry.strip()
            if len(clean_entry) > 15:
                ref_counter += 1
                ref_id = f"{paper_id}_ref{ref_counter:03d}"
                year_match = re.search(r"\b(19\d{2}|20\d{2})\b", clean_entry)
                year_str = year_match.group(1) if year_match else None

                references.append(
                    ExtractedReference(
                        paper_id=paper_id,
                        ref_id=ref_id,
                        raw_text=clean_entry,
                        year=year_str,
                    )
                )

        return references

    def extract_academic_metadata(
        self, doc: Any, paper_id: str
    ) -> Tuple[PaperMetadata, str]:
        """
        General-purpose academic metadata extraction:
        - Title: largest prominent font in upper half of page 1, excluding margin banners and notices
        - Authors: extracted from region between Title bottom and Abstract top
        - Publication Year: extracted from arXiv pattern, conference/copyright lines, or metadata
        - Abstract: text strictly between Abstract heading and first section
        """
        page1 = doc[0]
        p1_dict = page1.get_text("dict")
        blocks = [b for b in p1_dict.get("blocks", []) if b.get("type") == 0]
        page_width = page1.rect.width
        page_height = page1.rect.height

        # 1. Locate Abstract heading position
        abstract_y = page_height * 0.65
        abstract_block_idx = None
        for idx, b in enumerate(blocks):
            b_text = "".join(s.get("text", "") for l in b.get("lines", []) for s in l.get("spans", [])).strip()
            if re.match(r"^abstract\b", b_text, re.IGNORECASE) and len(b_text) < 40:
                abstract_y = b["bbox"][1]
                abstract_block_idx = idx
                break

        # 2. Extract Title
        candidate_titles = []
        for idx, b in enumerate(blocks):
            bbox = b["bbox"]
            if bbox[1] >= abstract_y:
                continue
            # Exclude margins (e.g. arXiv vertical bar at x0 < 45)
            if bbox[0] < 45 or bbox[2] > page_width - 40:
                continue
            if bbox[1] < 35 or bbox[3] > page_height - 50:
                continue

            b_text = " ".join("".join(s.get("text", "") for s in l.get("spans", [])).strip() for l in b.get("lines", [])).strip()
            if not b_text or len(b_text) < 4:
                continue

            if self.is_notice_text(b_text):
                continue

            if "@" in b_text or any(aff in b_text.lower() for aff in ["university", "department", "google brain", "institute"]):
                continue

            max_sz = max((s.get("size", 0) for l in b.get("lines", []) for s in l.get("spans", [])), default=0)
            is_bold = any(("bold" in s.get("font", "").lower() or s.get("flags", 0) & 2 != 0) for l in b.get("lines", []) for s in l.get("spans", []))

            candidate_titles.append((max_sz, is_bold, bbox[1], b_text, bbox))

        candidate_titles.sort(key=lambda x: (x[0], x[1], -x[2]), reverse=True)
        title = "Untitled Paper"
        title_bbox = (0, 0, page_width, page_height * 0.25)
        if candidate_titles:
            top_cand = candidate_titles[0]
            clean_title = re.sub(r"[\*∗†‡§0-9\^]+$", "", top_cand[3]).strip()
            # Clean hyphenated words across lines (e.g. LAN- GUAGE -> LANGUAGE)
            clean_title = re.sub(r"(\w+)-\s+(\w+)", r"\1\2", clean_title)
            title = clean_title
            title_bbox = top_cand[4]

        # 3. Extract Authors
        authors: List[str] = []
        for b in blocks:
            bbox = b["bbox"]
            if bbox[1] >= (title_bbox[3] - 8) and bbox[3] <= (abstract_y + 8):
                if bbox[0] < 45 or bbox[2] > page_width - 40:
                    continue
                for l in b.get("lines", []):
                    line_str = "".join(s.get("text", "") for s in l.get("spans", [])).strip()
                    if not line_str or "@" in line_str or self.is_notice_text(line_str):
                        continue
                    lower_l = line_str.lower()
                    if any(aff in lower_l for aff in self.AFFILIATION_KEYWORDS):
                        continue

                    # Clean footnote markers
                    cleaned_line = re.sub(r"[\*∗†‡§0-9\^]+", "", line_str).strip()
                    if 2 < len(cleaned_line) < 65:
                        parts = re.split(r",|\band\b|&", cleaned_line)
                        for p in parts:
                            p_name = re.sub(r"[\*∗†‡§0-9\^]+", "", p).strip()
                            words = p_name.split()
                            if 2 <= len(words) <= 4 and all(w[0].isupper() or w.startswith("de") or w.startswith("van") for w in words if w):
                                if p_name not in authors:
                                    authors.append(p_name)

        if not authors and doc.metadata.get("author"):
            meta_auth = doc.metadata.get("author").strip()
            if meta_auth:
                authors = [re.sub(r"[\*∗†‡§0-9\^]+", "", a).strip() for a in re.split(r",|;|\band\b", meta_auth) if len(a.strip()) > 2]

        # 4. Extract Publication Year
        pub_year = None
        full_p1 = page1.get_text("text")

        arxiv_match = re.search(r"arXiv:(\d{2})(\d{2})\.", full_p1)
        if arxiv_match:
            yr_short = int(arxiv_match.group(1))
            pub_year = str(2000 + yr_short) if yr_short <= 40 else str(1900 + yr_short)

        if not pub_year:
            conf_match = re.search(r"(?:NIPS|NeurIPS|ICML|CVPR|ICLR|ICCV|ECCV|ACL|EMNLP|NAACL|IEEE|ACM|Proceedings|Conference|Copyright|©)\s*[\w\s,]*\b(19\d{2}|20\d{2})\b", full_p1, re.IGNORECASE)
            if conf_match:
                pub_year = conf_match.group(1)

        if not pub_year and doc.metadata.get("creationDate"):
            cd = doc.metadata.get("creationDate")
            m = re.search(r"D:(\d{4})", cd)
            if m:
                pub_year = m.group(1)

        if not pub_year:
            m = re.search(r"\b(19\d{2}|20\d{2})\b", full_p1[:1500])
            if m:
                pub_year = m.group(1)

        # 5. Extract Pure Abstract
        abstract_text = ""
        if abstract_block_idx is not None:
            abstract_paras = []
            for idx in range(abstract_block_idx + 1, min(abstract_block_idx + 6, len(blocks))):
                b = blocks[idx]
                b_text = " ".join("".join(s.get("text", "") for s in l.get("spans", [])).strip() for l in b.get("lines", [])).strip()
                if re.match(r"^(?:\d+\.?\s+|[IVXLCDM]+\.?\s+)?(?:introduction|background)\b", b_text, re.IGNORECASE):
                    break
                if b_text.startswith("∗") or b_text.startswith("*") or self.is_notice_text(b_text):
                    break
                if len(b_text) > 35:
                    abstract_paras.append(b_text)
            abstract_text = "\n\n".join(abstract_paras)

        metadata = PaperMetadata(
            paper_id=paper_id,
            title=title.strip(),
            authors=authors,
            publication_year=pub_year,
            total_pages=len(doc),
            file_size_bytes=0,
            abstract=abstract_text[:2500] if abstract_text else None,
        )

        return metadata, abstract_text

    def extract_document(self, file_path: Path, paper_id: str) -> PaperDocument:
        """
        Full deep academic extraction pipeline with true reading order:
        - Layout & metadata extraction (Title, Authors, Year, Abstract)
        - Section boundary tracking with parent category inheritance
        - Table validation (rejection of false positives)
        - Traceable RAG-ready paragraph units
        - References extraction
        - Scanned PDF detection
        """
        file_size_kb = file_path.stat().st_size / 1024.0
        doc_obj = PaperDocument(
            id=paper_id,
            filename=file_path.name,
            saved_path=str(file_path),
            file_size_kb=round(file_size_kb, 2),
            is_valid_pdf=True,
        )

        if not PYMUPDF_AVAILABLE:
            doc_obj.status = "failed"
            doc_obj.error_message = "PyMuPDF (pymupdf) is not available."
            return doc_obj

        try:
            pdf = fitz.open(file_path)
            doc_obj.page_count = len(pdf)

            # 1. Academic Metadata Extraction
            metadata, abstract_text = self.extract_academic_metadata(pdf, paper_id)
            metadata.file_size_bytes = file_path.stat().st_size

            total_characters = 0
            all_page_texts: List[str] = []
            page_contents: List[PageContent] = []
            paragraphs: List[ExtractedParagraph] = []
            sections_map: Dict[str, ExtractedSection] = {}
            tables: List[ExtractedTable] = []
            references_text_buffer: List[str] = []

            # Track parent section numbers to normalized categories: e.g. "3" -> "Methodology"
            parent_norm_map: Dict[str, str] = {}

            # Front Matter section (sec001)
            front_matter_sec_id = f"{paper_id}_sec001"
            sections_map[front_matter_sec_id] = ExtractedSection(
                paper_id=paper_id,
                section_id=front_matter_sec_id,
                original_heading="Front Matter",
                normalized_section="Front Matter",
                start_page=1,
                end_page=1,
                paragraph_ids=[],
                full_text="",
            )

            # Abstract section (sec002)
            abstract_sec_id = f"{paper_id}_sec002"
            sections_map[abstract_sec_id] = ExtractedSection(
                paper_id=paper_id,
                section_id=abstract_sec_id,
                original_heading="Abstract",
                normalized_section="Abstract",
                start_page=1,
                end_page=1,
                paragraph_ids=[],
                full_text=abstract_text or "",
            )

            # State tracking across pages
            active_stage = "front_matter"  # "front_matter" -> "abstract" -> "body"
            active_original_heading = "Front Matter"
            active_normalized_section = "Front Matter"
            current_section_id = front_matter_sec_id
            sec_counter = 2
            para_counter = 0
            table_counter = 0
            parent_section_id_map: Dict[str, str] = {}  # prefix "6" -> section_id

            # --- Page Loop ---
            for page_idx, page in enumerate(pdf):
                page_num = page_idx + 1
                page_text = page.get_text("text") or ""
                page_char_count = len(page_text.strip())
                total_characters += page_char_count
                all_page_texts.append(page_text)

                # Extract and Validate Tables
                page_tables, table_counter = self._extract_tables_from_page(
                    page, paper_id, page_num, table_counter
                )
                tables.extend(page_tables)

                # Page Content
                is_scanned_page = page_char_count < 30
                page_contents.append(
                    PageContent(
                        paper_id=paper_id,
                        page_number=page_num,
                        char_count=page_char_count,
                        raw_text=page_text,
                        is_scanned_page=is_scanned_page,
                        table_count=len(page_tables),
                    )
                )

                # Extract blocks in true reading order
                # PyMuPDF block tuple: (x0, y0, x1, y1, text, block_no, block_type)
                blocks = page.get_text("blocks", sort=True)

                for b in blocks:
                    if b[6] != 0:  # text blocks only
                        continue

                    raw_b_text = b[4].strip()
                    if not raw_b_text:
                        continue

                    lines = [l.strip() for l in raw_b_text.split("\n") if l.strip()]
                    if not lines:
                        continue

                    # Filter lone page numbers and notice blocks
                    if len(lines) == 1 and lines[0].isdigit() and len(lines[0]) <= 3:
                        continue
                    if self.is_notice_text(raw_b_text) and active_stage == "body":
                        continue

                    # Heading Detection: Case 1 (First two lines split: "1" \n "Introduction")
                    detected_heading: Optional[Tuple[str, str]] = None
                    remaining_lines = lines

                    if len(lines) >= 2 and re.match(r"^(?:\d+(?:\.\d+)*|[IVXLCDM]+)$", lines[0]):
                        combined_cand = f"{lines[0]} {lines[1]}"
                        is_h, norm = self.is_academic_heading(combined_cand, parent_norm_map)
                        if is_h and norm:
                            detected_heading = (combined_cand, norm)
                            remaining_lines = lines[2:]

                    # Heading Detection: Case 2 (First line heading: "1 Introduction" or "References")
                    if not detected_heading:
                        first_l = lines[0]
                        is_h, norm = self.is_academic_heading(first_l, parent_norm_map)
                        if is_h and norm:
                            detected_heading = (first_l, norm)
                            remaining_lines = lines[1:]

                    # Check abstract heading explicitly
                    is_abstract_heading = False
                    if detected_heading and detected_heading[1] == "Abstract":
                        is_abstract_heading = True
                    elif re.match(r"^abstract\b", lines[0], re.IGNORECASE) and len(lines[0].split()) <= 4:
                        is_abstract_heading = True
                        detected_heading = ("Abstract", "Abstract")
                        remaining_lines = lines[1:]

                    if is_abstract_heading:
                        active_stage = "abstract"
                        current_section_id = abstract_sec_id
                        active_original_heading = "Abstract"
                        active_normalized_section = "Abstract"
                        sections_map[abstract_sec_id].start_page = min(sections_map[abstract_sec_id].start_page, page_num)
                        sections_map[abstract_sec_id].end_page = max(sections_map[abstract_sec_id].end_page, page_num)
                    elif detected_heading and detected_heading[1] != "Front Matter":
                        orig_h, norm_s = detected_heading
                        active_stage = "body"
                        # Avoid duplicate consecutive sections
                        if orig_h.lower() != active_original_heading.lower():
                            if current_section_id in sections_map:
                                sections_map[current_section_id].end_page = max(
                                    sections_map[current_section_id].end_page, page_num
                                )

                            sec_counter += 1
                            current_section_id = f"{paper_id}_sec{sec_counter:03d}"
                            active_original_heading = orig_h
                            active_normalized_section = norm_s

                            # Record parent number if top-level section: e.g. "6 Results" -> parent_norm_map["6"] = "Results"
                            parent_m = re.match(r"^(\d+)\s+", orig_h)
                            if parent_m:
                                parent_pfx = parent_m.group(1)
                                parent_norm_map[parent_pfx] = norm_s
                                parent_section_id_map[parent_pfx] = current_section_id

                            sections_map[current_section_id] = ExtractedSection(
                                paper_id=paper_id,
                                section_id=current_section_id,
                                original_heading=active_original_heading,
                                normalized_section=active_normalized_section,
                                start_page=page_num,
                                end_page=page_num,
                                paragraph_ids=[],
                                full_text="",
                            )

                    # Build paragraph from remaining lines
                    para_text = " ".join(remaining_lines).strip()
                    if not para_text or len(para_text) < 15:
                        continue

                    clean_para = re.sub(r"\s+", " ", para_text).strip()
                    lower_para = clean_para.lower()

                    # Fine-grained semantic classification
                    if active_stage == "front_matter":
                        para_norm = "Front Matter"
                        if self.is_notice_text(clean_para):
                            content_type = "publication_metadata"
                            para_heading = "Notice / License"
                        elif metadata.title and (
                            re.sub(r"[\s\-_]+", "", metadata.title.lower()) in re.sub(r"[\s\-_]+", "", lower_para)
                            or re.sub(r"[\s\-_]+", "", lower_para) in re.sub(r"[\s\-_]+", "", metadata.title.lower())
                        ):
                            content_type = "title"
                            para_heading = "Title"
                        elif any(re.search(r"\b" + re.escape(a.lower()) + r"\b", lower_para) for a in metadata.authors if len(a) > 3):
                            content_type = "author"
                            para_heading = "Authors"
                        elif "@" in clean_para or any(re.search(r"\b" + re.escape(k) + r"\b", lower_para) for k in self.AFFILIATION_KEYWORDS):
                            content_type = "affiliation"
                            para_heading = "Affiliations"
                        elif re.search(r"arxiv:\d{4}\.\d+", lower_para) or any(k in lower_para for k in ["proceedings", "conference on", "ieee", "acm", "nips", "iclr", "icml"]):
                            content_type = "publication_metadata"
                            para_heading = "Publication Metadata"
                        else:
                            content_type = "front_matter"
                            para_heading = "Front Matter"
                        sec_to_attach = front_matter_sec_id

                    elif active_stage == "abstract":
                        if clean_para.startswith(("∗", "*", "†", "‡", "§")) or any(ph in lower_para for ph in ["equal contribution", "work performed while"]):
                            content_type = "front_matter"
                            para_norm = "Front Matter"
                            para_heading = "Footnotes"
                            sec_to_attach = front_matter_sec_id
                        elif re.search(r"arxiv:\d{4}\.\d+", lower_para) or any(k in lower_para for k in ["proceedings", "conference on", "ieee", "acm", "nips", "iclr", "icml"]):
                            content_type = "publication_metadata"
                            para_norm = "Front Matter"
                            para_heading = "Publication Metadata"
                            sec_to_attach = front_matter_sec_id
                        else:
                            content_type = "abstract"
                            para_norm = "Abstract"
                            para_heading = "Abstract"
                            sec_to_attach = abstract_sec_id

                    else:  # active_stage == "body"
                        if active_normalized_section == "References":
                            content_type = "reference"
                            para_norm = "References"
                            para_heading = active_original_heading
                            references_text_buffer.append(clean_para)
                        else:
                            content_type = "body"
                            para_norm = active_normalized_section
                            para_heading = active_original_heading
                        sec_to_attach = current_section_id

                    # Create RAG-ready paragraph unit
                    para_counter += 1
                    para_id = f"{paper_id}_p{page_num:03d}_para{para_counter:03d}"

                    p_obj = ExtractedParagraph(
                        paper_id=paper_id,
                        paragraph_id=para_id,
                        page=page_num,
                        original_heading=para_heading,
                        normalized_section=para_norm,
                        content_type=content_type,
                        text=clean_para,
                        char_count=len(clean_para),
                    )
                    paragraphs.append(p_obj)

                    if sec_to_attach in sections_map:
                        sec_obj = sections_map[sec_to_attach]
                        sec_obj.paragraph_ids.append(para_id)
                        sec_obj.full_text += ("\n\n" if sec_obj.full_text else "") + clean_para
                        sec_obj.end_page = max(sec_obj.end_page, page_num)

                    # Update parent section end_page if current section is a subsection
                    sub_m = re.match(r"^(\d+)\.\d+", active_original_heading)
                    if sub_m:
                        p_pfx = sub_m.group(1)
                        if p_pfx in parent_section_id_map:
                            p_sec_id = parent_section_id_map[p_pfx]
                            if p_sec_id in sections_map:
                                sections_map[p_sec_id].end_page = max(sections_map[p_sec_id].end_page, page_num)

            # 3. Scanned PDF Detection
            chars_per_page = total_characters / max(1, len(pdf))
            is_scanned = (total_characters < 80) or (chars_per_page < 40)
            metadata.is_scanned = is_scanned
            metadata.text_density_chars_per_page = round(chars_per_page, 2)

            # 4. Bibliographic References
            all_references = self._extract_references_from_text(
                "\n".join(references_text_buffer), paper_id
            )

            # Finalize Document Object
            doc_obj.metadata = metadata
            doc_obj.is_scanned = is_scanned
            doc_obj.page_count = len(pdf)
            doc_obj.pages = page_contents
            # Preserve all populated sections as well as parent numbered headings (e.g. "6 Results")
            doc_obj.sections = [
                s for s in sections_map.values()
                if s.full_text or s.paragraph_ids or re.match(r"^\d+\s+[A-Z]", s.original_heading)
            ]
            doc_obj.paragraphs = paragraphs
            doc_obj.tables = tables
            doc_obj.references = all_references
            doc_obj.raw_text = "\n".join(all_page_texts)
            doc_obj.status = "scanned_ocr_required" if is_scanned else "processed"

            pdf.close()

        except Exception as e:
            doc_obj.status = "failed"
            doc_obj.error_message = f"Error extracting PDF: {str(e)}"

        return doc_obj
