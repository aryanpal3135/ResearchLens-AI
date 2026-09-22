"""
Comprehensive Unit & Integration Tests for Phase 2:
PDF Processor, sequential Paper IDs, deep extraction, section normalization,
scanned PDF detection, multi-feature table validation, and regression tests
on multiple academic paper layouts.
"""

from pathlib import Path
import pytest
import pymupdf as fitz
from services.pdf_processor import PDFProcessor


def test_sequential_paper_id_generation():
    """Verify that paper IDs follow the sequential format paper_001, paper_002, etc."""
    assert PDFProcessor.generate_paper_id(0) == "paper_001"
    assert PDFProcessor.generate_paper_id(1) == "paper_002"
    assert PDFProcessor.generate_paper_id(99) == "paper_100"


def test_pdf_validation_invalid_extension():
    processor = PDFProcessor(upload_dir=Path("data/uploads"))
    is_valid, msg = processor.validate_pdf(b"some content", "test.txt")
    assert not is_valid
    assert "extension" in msg.lower()


def test_pdf_validation_empty_file():
    processor = PDFProcessor(upload_dir=Path("data/uploads"))
    is_valid, msg = processor.validate_pdf(b"", "empty.pdf")
    assert not is_valid
    assert "empty" in msg.lower()


def test_pdf_validation_invalid_header():
    processor = PDFProcessor(upload_dir=Path("data/uploads"))
    is_valid, msg = processor.validate_pdf(b"Not a real PDF file", "fake.pdf")
    assert not is_valid
    assert "signature" in msg.lower() or "corrupted" in msg.lower()


def test_notice_and_copyright_filtering():
    """Test that copyright, permission, and license boilerplate are recognized."""
    processor = PDFProcessor(upload_dir=Path("data/uploads"))
    assert processor.is_notice_text("Provided proper attribution is provided, Google hereby grants permission")
    assert processor.is_notice_text("Permission to make digital or hard copies of all or part of this work")
    assert processor.is_notice_text("All rights reserved. Copyright 2023 IEEE.")
    assert not processor.is_notice_text("Attention Is All You Need")
    assert not processor.is_notice_text("The dominant sequence transduction models are based on complex recurrent")


def test_section_heading_normalization():
    """Test mapping author raw headings to standardized academic categories."""
    processor = PDFProcessor(upload_dir=Path("data/uploads"))
    
    assert processor.normalize_section_heading("Abstract") == "Abstract"
    assert processor.normalize_section_heading("1. Introduction") == "Introduction"
    assert processor.normalize_section_heading("I. BACKGROUND") == "Background"
    assert processor.normalize_section_heading("2. Background and Preliminaries") == "Background"
    assert processor.normalize_section_heading("2. Related Work and Prior Art") == "Related Work"
    assert processor.normalize_section_heading("3. Materials and Methods") == "Methodology"
    assert processor.normalize_section_heading("III. PROPOSED ARCHITECTURE") == "Methodology"
    assert processor.normalize_section_heading("4. Experimental Setup") == "Experimental Setup"
    assert processor.normalize_section_heading("5. Empirical Results") == "Results"
    assert processor.normalize_section_heading("6. Discussion and Ablation Study") == "Discussion"
    assert processor.normalize_section_heading("7. Limitations and Threats to Validity") == "Limitations"
    assert processor.normalize_section_heading("8. Future Research Directions") == "Future Work"
    assert processor.normalize_section_heading("9. Conclusion") == "Conclusion"
    assert processor.normalize_section_heading("References") == "References"
    assert processor.normalize_section_heading("Unknown Custom Heading") == "General"


def test_subsection_parent_inheritance():
    """Verify that subsections like 3.1, 5.2 inherit parent section category."""
    processor = PDFProcessor(upload_dir=Path("data/uploads"))
    parent_map = {"3": "Methodology", "5": "Experimental Setup", "6": "Results"}
    
    assert processor.normalize_section_heading("3.1 Encoder and Decoder Stacks", parent_map) == "Methodology"
    assert processor.normalize_section_heading("3.2 Attention", parent_map) == "Methodology"
    assert processor.normalize_section_heading("5.2 Hardware and Schedule", parent_map) == "Experimental Setup"
    assert processor.normalize_section_heading("6.1 Machine Translation", parent_map) == "Results"


def test_table_validation_rules():
    """Verify multi-feature table validation distinguishes valid tables from layout artifacts and assigns quality."""
    processor = PDFProcessor(upload_dir=Path("data/uploads"))

    # 1. Valid Academic Table (e.g. Model Comparison) -> High Quality
    valid_data = [
        ["Model", "BLEU Score", "Params"],
        ["Transformer (base)", "27.3", "65M"],
        ["Transformer (big)", "28.4", "213M"],
        ["ConvS2S", "25.1", "105M"],
    ]
    status, quality, reason = processor.validate_table_candidate(valid_data)
    assert status == "Valid"
    assert quality == "high"

    # 2. Valid Academic Table with Merged Subcolumns -> Low Quality
    merged_subcol_data = [
        ["", "train\nN d d h d d P ϵ\nmodel ff k v drop ls steps", "PPL BLEU params\n(dev) (dev) ×106"],
        ["base", "6 512 2048 8 64 64 0.1 0.1 100K", "4.92 25.8 65"],
        ["big", "6 1024 4096 16 64 64 0.3 0.1 300K", "4.30 26.4 213"],
    ]
    status, quality, reason = processor.validate_table_candidate(merged_subcol_data)
    assert status == "Valid"
    assert quality == "low"

    # 3. High-column Sparse Artifact (False-Positive) -> Rejected
    sparse_data = [[f"w{c}" if c % 5 == 0 else "" for c in range(35)] for _ in range(8)]
    status, quality, reason = processor.validate_table_candidate(sparse_data)
    assert status == "Rejected"
    assert "columns" in reason.lower() or "sparse" in reason.lower()

    # 4. Fragmented Prose Sentence Artifact (False-Positive: "The | Law | will | never | be | perfect...") -> Rejected
    prose_row = ["The", "Law", "will", "never", "be", "perfect", "in", "this", "country", "under", "these", "rules"]
    prose_data = [prose_row for _ in range(4)]
    status, quality, reason = processor.validate_table_candidate(prose_data)
    assert status == "Rejected"
    assert "prose" in reason.lower() or "fragmented" in reason.lower()

    # 5. Empty Header -> Needs Review
    empty_header_data = [
        ["", "", ""],
        ["Val A", "Val B", "Val C"],
        ["Val D", "Val E", "Val F"],
    ]
    status, quality, reason = processor.validate_table_candidate(empty_header_data)
    assert status == "Needs Review"


def test_synthetic_research_paper_deep_extraction(tmp_path):
    """
    Generate a realistic academic PDF with multiple pages, sections, and references,
    then verify deep extraction, section mapping, and RAG-ready paragraph units.
    """
    pdf_path = tmp_path / "synthetic_paper.pdf"
    doc = fitz.open()

    # Page 1: Title, Authors, Abstract, Introduction
    page1 = doc.new_page()
    page1.insert_text(
        (50, 60),
        "Advancements in Retrieval Augmented Generation\n"
        "Alice Smith and Bob Johnson\n"
        "Published in 2024\n\n"
        "Abstract\n"
        "Retrieval Augmented Generation enhances language models by retrieving grounded knowledge.\n\n"
        "1. Introduction\n"
        "Recent advances in natural language processing emphasize grounding generation in evidence.\n"
        "This paper evaluates section-aware retrieval and citation verification.\n",
        fontsize=11
    )

    # Page 2: Materials and Methods
    page2 = doc.new_page()
    page2.insert_text(
        (50, 60),
        "2. Materials and Methods\n"
        "We construct a dual-encoder retrieval index using dense vector embeddings.\n"
        "The model is evaluated across several academic benchmarks to test precision at K.\n",
        fontsize=11
    )

    # Page 3: Results and Limitations
    page3 = doc.new_page()
    page3.insert_text(
        (50, 60),
        "3. Results\n"
        "Our experimental results show a 28% improvement in citation accuracy.\n\n"
        "4. Limitations\n"
        "The study is currently evaluated primarily on English language papers.\n",
        fontsize=11
    )

    # Page 4: References
    page4 = doc.new_page()
    page4.insert_text(
        (50, 60),
        "References\n"
        "[1] Smith, A. (2023). Foundations of Dense Retrieval. AI Journal.\n"
        "[2] Johnson, B. (2024). Multi-Stage Ranking in Language Models. IEEE.\n",
        fontsize=11
    )

    doc.save(pdf_path)
    doc.close()

    processor = PDFProcessor(upload_dir=tmp_path / "uploads")
    paper_id = "paper_001"
    paper_doc = processor.extract_document(pdf_path, paper_id)

    assert paper_doc.id == "paper_001"
    assert paper_doc.page_count == 4
    assert paper_doc.is_valid_pdf is True
    assert paper_doc.is_scanned is False
    assert paper_doc.status == "processed"

    # Metadata checks
    assert paper_doc.metadata.paper_id == "paper_001"
    assert paper_doc.metadata.publication_year == "2024"
    assert "Retrieval" in paper_doc.metadata.title or "Advancements" in paper_doc.metadata.title
    assert paper_doc.metadata.text_density_chars_per_page > 50

    # Section Normalization checks
    normalized_titles = [s.normalized_section for s in paper_doc.sections]
    assert "Methodology" in normalized_titles
    assert "Results" in normalized_titles
    assert "Limitations" in normalized_titles
    assert "References" in normalized_titles

    methodology_sec = next(s for s in paper_doc.sections if s.normalized_section == "Methodology")
    assert "Materials and Methods" in methodology_sec.original_heading

    # RAG-ready paragraph checks
    assert len(paper_doc.paragraphs) >= 4
    for para in paper_doc.paragraphs:
        assert para.paper_id == "paper_001"
        assert para.paragraph_id.startswith("paper_001_p")
        assert para.page in [1, 2, 3, 4]
        assert len(para.text) > 0

    # References checks
    assert len(paper_doc.references) >= 2


def test_scanned_pdf_detection(tmp_path):
    """Verify that empty or image-only PDFs trigger the scanned/OCR warning."""
    scanned_pdf_path = tmp_path / "scanned_doc.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.save(scanned_pdf_path)
    doc.close()

    processor = PDFProcessor(upload_dir=tmp_path / "uploads")
    paper_doc = processor.extract_document(scanned_pdf_path, "paper_002")

    assert paper_doc.id == "paper_002"
    assert paper_doc.is_scanned is True
    assert paper_doc.status == "scanned_ocr_required"
    assert paper_doc.metadata.is_scanned is True


def test_regression_attention_is_all_you_need():
    """Regression test on real-world Transformer paper ('Attention Is All You Need')."""
    paper_path = Path("data/uploads/Res.pdf")
    if not paper_path.exists():
        pytest.skip("Res.pdf not found in data/uploads")

    processor = PDFProcessor(upload_dir=Path("data/uploads"))
    doc = processor.extract_document(paper_path, "paper_001")

    # 1. Metadata: Title must NOT be copyright notice
    assert doc.metadata.title == "Attention Is All You Need"
    assert not any(w in doc.metadata.title.lower() for w in ["permission", "google", "attribution", "grants"])

    # 2. Authors
    assert len(doc.metadata.authors) >= 6
    assert any("Vaswani" in a for a in doc.metadata.authors)
    assert any("Shazeer" in a for a in doc.metadata.authors)

    # 3. Year
    assert doc.metadata.publication_year == "2017"

    # 4. Abstract: Pure abstract without front matter
    assert doc.metadata.abstract is not None
    assert doc.metadata.abstract.startswith("The dominant sequence transduction models")
    assert not any(w in doc.metadata.abstract.lower() for w in ["attribution is provided", "google hereby grants"])

    # 5. Semantic Paragraph Classification: Front Matter vs Abstract distinction
    title_paras = [p for p in doc.paragraphs if p.content_type == "title"]
    assert len(title_paras) >= 1
    assert title_paras[0].normalized_section == "Front Matter"
    assert "Attention Is All You Need" in title_paras[0].text

    author_paras = [p for p in doc.paragraphs if p.content_type in ["author", "affiliation"]]
    assert len(author_paras) >= 4
    for ap in author_paras:
        assert ap.normalized_section == "Front Matter"

    # Crucial QA Check: Only actual abstract text receives content_type == 'abstract' and normalized_section == 'Abstract'
    abstract_paras = [p for p in doc.paragraphs if p.content_type == "abstract"]
    assert len(abstract_paras) == 1
    assert abstract_paras[0].normalized_section == "Abstract"
    assert abstract_paras[0].text.startswith("The dominant sequence transduction models")

    # Body paragraphs are labeled 'body'
    body_paras = [p for p in doc.paragraphs if p.content_type == "body"]
    assert len(body_paras) >= 40
    assert any("Recurrent neural networks" in p.text for p in body_paras)

    # 6. Section Taxonomy: Background preserved & Parent Sections not skipped
    section_norms = [s.normalized_section for s in doc.sections]
    assert "Background" in section_norms
    assert "Introduction" in section_norms
    assert "Methodology" in section_norms
    assert "Discussion" in section_norms
    assert "Experimental Setup" in section_norms
    assert "Results" in section_norms
    assert "Conclusion" in section_norms
    assert "References" in section_norms

    # "2 Background" is preserved as Background
    bg_sec = next(s for s in doc.sections if s.original_heading.strip() == "2 Background")
    assert bg_sec.normalized_section == "Background"

    # Parent section "6 Results" is retained and not skipped
    results_sec = next(s for s in doc.sections if s.original_heading.strip() == "6 Results")
    assert results_sec.normalized_section == "Results"
    assert results_sec.start_page == 8
    assert results_sec.end_page >= 8

    # Subsections under Section 6 are also retained
    sec_headings = [s.original_heading for s in doc.sections]
    assert "6.1 Machine Translation" in sec_headings
    assert "6.2 Model Variations" in sec_headings

    # 7. Tables: Quality differentiation and row reconstruction
    assert len(doc.tables) == 2
    for t in doc.tables:
        assert t.status == "Valid"
        assert t.col_count <= 15
        assert t.row_count >= 3

    # Table 1 (Page 9): Valid academic table with merged subcolumns -> Low Quality
    t1 = next(t for t in doc.tables if t.page == 9)
    assert t1.status == "Valid"
    assert t1.extraction_quality == "low"

    # Table 2 (Page 10): Clean table with reconstructed rows -> High Quality
    t2 = next(t for t in doc.tables if t.page == 10)
    assert t2.status == "Valid"
    assert t2.extraction_quality == "high"
    assert t2.row_count == 13  # Reconstructed from 6 merged rows

    # 8. References
    assert len(doc.references) >= 30
    assert doc.references[0].ref_id.startswith("paper_001_ref")


def test_regression_second_paper_lora():
    """Regression test on second academic paper with different layout ('LoRA')."""
    lora_path = Path("data/uploads/paper_lora.pdf")
    if not lora_path.exists():
        pytest.skip("paper_lora.pdf not found in data/uploads")

    processor = PDFProcessor(upload_dir=Path("data/uploads"))
    doc = processor.extract_document(lora_path, "paper_002")

    # 1. Title
    assert "LORA" in doc.metadata.title
    assert "LOW-RANK ADAPTATION" in doc.metadata.title

    # 2. Authors
    assert len(doc.metadata.authors) >= 4
    assert any("Edward Hu" in a for a in doc.metadata.authors)

    # 3. Year
    assert doc.metadata.publication_year == "2021"

    # 4. Abstract
    assert doc.metadata.abstract is not None
    assert "paradigm of natural language processing" in doc.metadata.abstract.lower()

    # 5. Semantic Paragraphs
    title_p = next(p for p in doc.paragraphs if p.content_type == "title")
    assert title_p.normalized_section == "Front Matter"
    assert "LORA" in title_p.text

    abs_p = next(p for p in doc.paragraphs if p.content_type == "abstract")
    assert abs_p.normalized_section == "Abstract"

    # 6. Sections
    section_norms = [s.normalized_section for s in doc.sections]
    assert "Front Matter" in section_norms
    assert "Abstract" in section_norms
    assert "Introduction" in section_norms
    assert "Experimental Setup" in section_norms
    assert "Methodology" in section_norms
    assert "Results" in section_norms

    # 7. Tables quality assessment
    assert len(doc.tables) >= 1
    for t in doc.tables:
        assert t.status in ["Valid", "Needs Review"]
        assert t.extraction_quality in ["high", "medium", "low"]


def test_full_data_accessibility_and_consistency():
    """
    Acceptance Test: Verifies that NO extracted data is truncated or discarded.
    Validates complete 40 references, 206 RAG paragraphs, all sections, and table exports
    for 'Attention Is All You Need' (Res.pdf).
    """
    from frontend.upload import table_markdown_to_df
    import pandas as pd
    import json

    paper_path = Path("data/uploads/Res.pdf")
    if not paper_path.exists():
        pytest.skip("Res.pdf not found in data/uploads")

    processor = PDFProcessor(upload_dir=Path("data/uploads"))
    doc = processor.extract_document(paper_path, "paper_001")

    # 1. Exact counts in data model
    assert doc.id == "paper_001"
    assert doc.page_count == 15
    assert len(doc.references) == 40
    assert len(doc.paragraphs) == 206
    assert len(doc.tables) == 2
    assert len(doc.sections) == 25

    # 2. Reference stability & complete access
    assert doc.references[0].ref_id == "paper_001_ref001"
    assert "Jimmy Lei Ba" in doc.references[0].raw_text
    assert doc.references[39].ref_id == "paper_001_ref040"
    assert "Muhua Zhu" in doc.references[39].raw_text

    # 3. Global search on ALL 40 references (not just the first 20)
    query = "zhu"
    matching_refs = [r for r in doc.references if query in r.raw_text.lower() or query in r.ref_id.lower()]
    assert len(matching_refs) >= 1
    assert any(r.ref_id == "paper_001_ref040" for r in matching_refs)

    # 4. Reference Export contains all 40 entries
    ref_records = [
        {"paper_id": r.paper_id, "ref_id": r.ref_id, "year": r.year or "", "text": r.raw_text}
        for r in doc.references
    ]
    assert len(ref_records) == 40
    df_refs = pd.DataFrame(ref_records)
    assert len(df_refs) == 40

    # 5. Global search on ALL 206 paragraphs
    # Search for a term that appears late in the document (e.g. "FLOPs" or "checkpoint")
    p_matches = [p for p in doc.paragraphs if "checkpoint" in p.text.lower()]
    assert len(p_matches) >= 1
    # Verify paragraph ID format is stable
    for p in doc.paragraphs:
        assert p.paper_id == "paper_001"
        assert p.paragraph_id.startswith("paper_001_p")
        assert p.content_type in ["front_matter", "title", "author", "affiliation", "publication_metadata", "abstract", "body", "reference"]

    # 6. Paragraph filtering preserves underlying dataset
    results_paras = [p for p in doc.paragraphs if p.normalized_section == "Results"]
    assert len(results_paras) >= 20

    abstract_paras = [p for p in doc.paragraphs if p.content_type == "abstract"]
    assert len(abstract_paras) == 1
    assert abstract_paras[0].normalized_section == "Abstract"

    # 7. Table DataFrame reconstruction and export
    t1 = doc.tables[0]
    df1 = table_markdown_to_df(t1.markdown_content)
    assert not df1.empty
    assert df1.shape[0] == 7
    assert df1.shape[1] == 3

    t2 = doc.tables[1]
    df2 = table_markdown_to_df(t2.markdown_content)
    assert not df2.empty
    assert df2.shape[0] == 12
    assert df2.shape[1] == 3

    # 8. Section hierarchy completeness
    section_ids = [s.section_id for s in doc.sections]
    assert len(section_ids) == 25
    assert len(set(section_ids)) == 25


def test_table_dataframe_unique_columns_and_json_export():
    """
    Regression test for: ValueError: DataFrame columns must be unique for orient='records'.
    Verifies that tables with duplicate, blank, or repeated header columns produce DataFrames
    with guaranteed unique column names that export seamlessly to JSON (orient='records').
    """
    from frontend.upload import table_markdown_to_df, make_unique_column_names

    # 1. Synthetic test cases with duplicate, blank, and overlapping headers
    test_markdowns = [
        # Table with blank duplicate headers
        "| BERT | | | | | | | |\n|---|---|---|---|---|---|---|---|\n| E[CLS] | E1 | ... | EN | E[SEP] | E1 | ... | EM |",
        # Table with identical headers
        "| E | E | E |\n|---|---|---|\n| is | cute | [SEP] |",
        # Table with repeated named headers
        "| Model | Dev | Test | Dev | Test |\n|---|---|---|---|---|\n| BERT | 88.2 | 87.5 | 89.1 | 88.4 |",
        # Table with all empty headers
        "| | | |\n|---|---|---|\n| 1 | 2 | 3 |",
        # Table with overlapping candidate names
        "| Score | Score_2 | Score |\n|---|---|---|\n| 10 | 20 | 30 |",
    ]

    for md in test_markdowns:
        df = table_markdown_to_df(md)
        assert df.columns.is_unique, f"Columns not unique for markdown:\n{md}\nGot: {list(df.columns)}"
        # Must not raise ValueError: DataFrame columns must be unique for orient='records'
        json_output = df.to_json(orient="records", indent=2)
        assert json_output is not None
        assert len(json_output) > 0

    # 2. Test against real extracted tables from res2.pdf (BERT paper)
    res2_path = Path("data/uploads/res2.pdf")
    if res2_path.exists():
        processor = PDFProcessor(upload_dir=Path("data/uploads"))
        paper_res2 = processor.extract_document(res2_path, "paper_res2")
        assert len(paper_res2.tables) > 0

        for tbl in paper_res2.tables:
            df_tab = table_markdown_to_df(tbl.markdown_content)
            assert df_tab.columns.is_unique, f"Table {tbl.table_id} columns are not unique: {list(df_tab.columns)}"
            # Verify to_json(orient='records') executes cleanly
            exported_json = df_tab.to_json(orient="records", indent=2)
            assert exported_json is not None
            # Verify CSV export executes cleanly
            exported_csv = df_tab.to_csv(index=False)
            assert exported_csv is not None

