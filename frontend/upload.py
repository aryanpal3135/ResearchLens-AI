"""
Upload Papers Page for ResearchLens AI.
Supports single/multi-PDF upload, format validation, corruption handling,
sequential Paper ID assignment, scanned PDF alerts, and a Deep Document Inspector.
"""

from pathlib import Path
import streamlit as st
import pandas as pd
from config.settings import settings
from services.pdf_processor import PDFProcessor
from frontend.components import render_header, render_info_card


import json
from typing import List, Dict


def make_unique_column_names(columns: List[str]) -> List[str]:
    """
    Ensures that all column names in a parsed table are non-empty and strictly unique.
    - Replaces empty strings or whitespace with informative column labels (e.g. 'Column_1', 'Column_2').
    - Disambiguates duplicate names by appending a 1-indexed suffix (e.g. 'E', 'E_2', 'E_3').
    """
    unique_cols: List[str] = []
    seen: Dict[str, int] = {}

    for i, col in enumerate(columns):
        clean_col = col.strip()
        if not clean_col:
            clean_col = f"Column_{i + 1}"

        if clean_col in seen:
            seen[clean_col] += 1
            candidate = f"{clean_col}_{seen[clean_col]}"
            while candidate in seen or candidate in unique_cols:
                seen[clean_col] += 1
                candidate = f"{clean_col}_{seen[clean_col]}"
            unique_cols.append(candidate)
        else:
            seen[clean_col] = 1
            unique_cols.append(clean_col)

    return unique_cols


def table_markdown_to_df(markdown_content: str) -> pd.DataFrame:
    """
    Parses a markdown table into a clean pandas DataFrame with guaranteed unique columns.
    Prevents ValueError: DataFrame columns must be unique for orient='records' in .to_json().
    """
    if not markdown_content or not markdown_content.strip():
        return pd.DataFrame()

    lines = [l.strip() for l in markdown_content.strip().split("\n") if l.strip()]
    if len(lines) < 2:
        return pd.DataFrame()

    raw_header_cols = [c.strip() for c in lines[0].strip("|").split("|")]

    parsed_rows: List[List[str]] = []
    max_cols = len(raw_header_cols)

    # Skip header (lines[0]) and separator line (lines[1])
    for l in lines[2:]:
        cols = [c.strip() for c in l.strip("|").split("|")]
        if len(cols) > max_cols:
            max_cols = len(cols)
        parsed_rows.append(cols)

    # Extend header if data rows contain more columns than header
    while len(raw_header_cols) < max_cols:
        raw_header_cols.append(f"Column_{len(raw_header_cols) + 1}")

    unique_headers = make_unique_column_names(raw_header_cols)

    normalized_rows: List[List[str]] = []
    for cols in parsed_rows:
        if len(cols) < len(unique_headers):
            cols = cols + [""] * (len(unique_headers) - len(cols))
        else:
            cols = cols[:len(unique_headers)]
        normalized_rows.append(cols)

    return pd.DataFrame(normalized_rows, columns=unique_headers)



def render_upload_page():
    """Renders the PDF upload, deep extraction, and inspector interface."""
    render_header(
        title="📤 Upload & Deep Document Extraction",
        subtitle="Upload research papers (PDF) to extract structured text, sections, tables, and bibliography."
    )

    # Initialize processor
    processor = PDFProcessor(upload_dir=settings.UPLOAD_DIR)

    # Session State initialization
    if "uploaded_papers" not in st.session_state:
        st.session_state["uploaded_papers"] = []

    render_info_card(
        title="Phase 2 Deep Extraction Pipeline",
        description=(
            "• **Sequential Paper IDs:** Automatically assigns `paper_001`, `paper_002`, etc., persisted across all entities.\n"
            "• **Scanned PDF Detection:** Evaluates text density to detect image-only PDFs and flags OCR requirements.\n"
            "• **Dual Section Tracking:** Captures the author's `original_heading` while mapping to a `normalized_section`.\n"
            "• **RAG-Ready Paragraphs:** Generates granular paragraph units with strict provenance ready for Phase 3 chunking.\n"
            "• **Table & Reference Extraction:** Parses embedded tables into Markdown and isolates bibliographic citations."
        )
    )

    # File uploader widget
    uploaded_files = st.file_uploader(
        label="Select Research Papers (PDF only)",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more academic research papers in PDF format."
    )

    if uploaded_files:
        st.subheader("📋 Ingestion Queue")

        # Action button to process files
        if st.button("🚀 Process & Ingest Papers", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()

            for idx, uploaded_file in enumerate(uploaded_files):
                filename = uploaded_file.name
                file_bytes = uploaded_file.getvalue()

                status_text.text(f"Validating & extracting: {filename}...")

                # 1. Validation
                is_valid, msg = processor.validate_pdf(file_bytes, filename)

                if not is_valid:
                    st.error(f"❌ Ingestion failed for '{filename}': {msg}")
                    continue

                # Check if paper with this filename is already uploaded in session
                existing = next((p for p in st.session_state["uploaded_papers"] if p.filename == filename), None)
                if existing:
                    st.info(f"ℹ️ '{filename}' is already in your active session (ID: `{existing.id}`).")
                    progress_bar.progress((idx + 1) / len(uploaded_files))
                    continue

                # 2. Save binary
                saved_path = processor.save_uploaded_file(file_bytes, filename)

                # 3. Assign sequential Paper ID
                paper_id = processor.generate_paper_id(len(st.session_state["uploaded_papers"]))

                # 4. Deep Extraction
                paper_doc = processor.extract_document(saved_path, paper_id)

                st.session_state["uploaded_papers"].append(paper_doc)
                progress_bar.progress((idx + 1) / len(uploaded_files))

            status_text.success("Deep extraction complete! All papers are cataloged with RAG-ready units.")

    # Display Current Ingested Papers
    st.divider()
    paper_count = len(st.session_state["uploaded_papers"])
    st.subheader(f"📚 Cataloged Research Papers ({paper_count})")

    if paper_count == 0:
        st.write("No papers currently in session memory. Upload files above to get started.")
    else:
        for idx, paper in enumerate(st.session_state["uploaded_papers"]):
            status_badge = "🟢 Processed" if not paper.is_scanned else "🟠 Scanned (OCR Required)"
            expander_title = f"📄 [{paper.id}] {paper.filename} — {paper.page_count} Pages ({status_badge})"

            with st.expander(expander_title, expanded=(idx == 0)):
                # Scanned PDF Warning
                if paper.is_scanned:
                    st.warning(
                        "⚠️ **Scanned Document Detected:** This PDF contains little to no extractable digital text layer "
                        f"({paper.metadata.text_density_chars_per_page} chars/page) and appears to be a scanned image. "
                        "Optical Character Recognition (OCR) may be required for full textual synthesis."
                    )

                # Top info bar
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Paper ID", paper.id)
                c2.metric("Total Pages", paper.page_count)
                c3.metric("Extracted Paragraphs", len(paper.paragraphs))
                c4.metric("Extracted Tables", len(paper.tables))

                # Deep Document Inspector Tabs
                tab_overview, tab_sections, tab_paras, tab_tables, tab_refs = st.tabs([
                    "📑 Overview & Metadata",
                    f"🧭 Section Hierarchy ({len(paper.sections)})",
                    f"🧩 RAG-Ready Paragraphs ({len(paper.paragraphs)})",
                    f"📊 Extracted Tables ({len(paper.tables)})",
                    f"📖 References ({len(paper.references)})"
                ])

                # --- Tab 1: Overview & Metadata ---
                with tab_overview:
                    m_col1, m_col2 = st.columns(2)
                    with m_col1:
                        st.markdown(f"**Paper ID:** `{paper.id}`")
                        st.markdown(f"**Title:** {paper.metadata.title or 'Not available'}")
                        authors_str = ", ".join(paper.metadata.authors) if paper.metadata.authors else "Not available"
                        st.markdown(f"**Authors:** {authors_str}")
                        st.markdown(f"**Publication Year:** {paper.metadata.publication_year or 'Not available'}")
                        st.markdown(f"**Total Pages:** {paper.page_count}")
                        st.markdown(f"**File Size:** `{paper.file_size_kb} KB`")
                    with m_col2:
                        st.markdown(f"**File Name:** `{paper.filename}`")
                        st.markdown(f"**Stored Path:** `{paper.saved_path}`")
                        st.markdown(f"**Text Density:** `{paper.metadata.text_density_chars_per_page} characters/page`")
                        status_label = "Scanned (OCR Required)" if paper.is_scanned else paper.status.capitalize()
                        st.markdown(f"**Extraction Status:** `{status_label}`")
                        st.markdown(f"**Extracted Paragraphs:** `{len(paper.paragraphs)}` | **Sections:** `{len(paper.sections)}`")
                        st.markdown(f"**Extracted Tables:** `{len(paper.tables)}` | **References:** `{len(paper.references)}`")

                    st.markdown("---")
                    st.markdown("##### 📝 Complete Abstract")
                    if paper.metadata.abstract:
                        st.info(paper.metadata.abstract)
                        st.caption(f"Abstract Length: {len(paper.metadata.abstract)} characters | {len(paper.metadata.abstract.split())} words")
                    else:
                        st.caption("No abstract section detected in this document.")

                # --- Tab 2: Section Hierarchy & Normalization ---
                with tab_sections:
                    st.markdown(f"##### 🧭 Academic Section Hierarchy ({len(paper.sections)} sections)")
                    st.caption("Complete mapping of author headings, normalized categories, and page boundaries.")

                    if not paper.sections:
                        st.caption("No clear section headings detected.")
                    else:
                        sec_s_col, sec_f_col = st.columns([3, 2])
                        with sec_s_col:
                            sec_search = st.text_input(
                                "🔍 Search sections (heading, category, ID)...",
                                key=f"search_sec_{paper.id}"
                            )
                        with sec_f_col:
                            sec_categories = ["All"] + sorted(list(set(s.normalized_section for s in paper.sections)))
                            selected_sec_cat = st.selectbox(
                                "Filter by Category:",
                                options=sec_categories,
                                key=f"filter_sec_cat_{paper.id}"
                            )

                        filtered_sections = paper.sections
                        if sec_search.strip():
                            sq = sec_search.strip().lower()
                            filtered_sections = [
                                s for s in filtered_sections
                                if sq in s.original_heading.lower() or sq in s.normalized_section.lower() or sq in s.section_id.lower()
                            ]
                        if selected_sec_cat != "All":
                            filtered_sections = [s for s in filtered_sections if s.normalized_section == selected_sec_cat]

                        # Export buttons
                        exp_col1, exp_col2, _ = st.columns([1.5, 1.5, 3])
                        sec_export_data = [
                            {
                                "paper_id": s.paper_id,
                                "section_id": s.section_id,
                                "original_heading": s.original_heading,
                                "normalized_section": s.normalized_section,
                                "start_page": s.start_page,
                                "end_page": s.end_page,
                                "paragraph_count": len(s.paragraph_ids),
                                "full_text": s.full_text,
                            }
                            for s in filtered_sections
                        ]
                        with exp_col1:
                            st.download_button(
                                label="📥 Export Sections (CSV)",
                                data=pd.DataFrame([
                                    {
                                        "Section ID": s["section_id"],
                                        "Original Heading": s["original_heading"],
                                        "Normalized Category": s["normalized_section"],
                                        "Pages": f"{s['start_page']} – {s['end_page']}",
                                        "Paragraph Count": s["paragraph_count"],
                                    }
                                    for s in sec_export_data
                                ]).to_csv(index=False),
                                file_name=f"{paper.id}_sections.csv",
                                mime="text/csv",
                                key=f"dl_csv_sec_{paper.id}"
                            )
                        with exp_col2:
                            st.download_button(
                                label="📥 Export Sections (JSON)",
                                data=json.dumps(sec_export_data, indent=2),
                                file_name=f"{paper.id}_sections.json",
                                mime="application/json",
                                key=f"dl_json_sec_{paper.id}"
                            )

                        # Expand/Collapse logic
                        expand_sec_key = f"expand_sec_{paper.id}"
                        if expand_sec_key not in st.session_state:
                            st.session_state[expand_sec_key] = False

                        is_sec_expanded = st.session_state[expand_sec_key]
                        preview_sec_limit = 10

                        if is_sec_expanded or len(filtered_sections) <= preview_sec_limit:
                            display_sections = filtered_sections
                        else:
                            display_sections = filtered_sections[:preview_sec_limit]

                        sec_records = []
                        for s in display_sections:
                            sec_records.append({
                                "Section ID": s.section_id,
                                "Original Author Heading": s.original_heading,
                                "Normalized Category": s.normalized_section,
                                "Pages": f"{s.start_page} – {s.end_page}",
                                "Paragraphs": len(s.paragraph_ids),
                            })
                        st.dataframe(pd.DataFrame(sec_records), use_container_width=True)

                        if len(filtered_sections) > preview_sec_limit:
                            if not is_sec_expanded:
                                if st.button(f"▼ Show all {len(filtered_sections)} sections", key=f"btn_exp_sec_{paper.id}"):
                                    st.session_state[expand_sec_key] = True
                                    st.rerun()
                            else:
                                if st.button(f"▲ Show fewer ({preview_sec_limit})", key=f"btn_col_sec_{paper.id}"):
                                    st.session_state[expand_sec_key] = False
                                    st.rerun()

                        # Section Detail Inspector
                        with st.expander("🔍 Inspect Full Section Text"):
                            sec_choices = {s.section_id: f"[{s.section_id}] {s.original_heading} ({s.normalized_section})" for s in paper.sections}
                            chosen_sec_id = st.selectbox(
                                "Select Section to View Full Text:",
                                options=list(sec_choices.keys()),
                                format_func=lambda x: sec_choices[x],
                                key=f"inspect_sec_sel_{paper.id}"
                            )
                            chosen_sec = next((s for s in paper.sections if s.section_id == chosen_sec_id), None)
                            if chosen_sec:
                                st.markdown(f"**Section ID:** `{chosen_sec.section_id}` | **Category:** `{chosen_sec.normalized_section}` | **Pages:** {chosen_sec.start_page}–{chosen_sec.end_page}")
                                st.markdown(f"**Paragraph IDs ({len(chosen_sec.paragraph_ids)}):** {', '.join([f'`{pid}`' for pid in chosen_sec.paragraph_ids]) if chosen_sec.paragraph_ids else 'None'}")
                                if chosen_sec.full_text:
                                    st.text_area("Section Full Text", value=chosen_sec.full_text, height=200, key=f"sec_txt_area_{paper.id}_{chosen_sec.section_id}")
                                else:
                                    st.caption("No direct prose text in this section header.")

                # --- Tab 3: RAG-Ready Paragraphs ---
                with tab_paras:
                    st.markdown(f"##### 🧩 RAG-Ready Paragraphs ({len(paper.paragraphs)})")
                    st.caption("Complete collection of chunk-ready paragraphs with strict provenance, semantic classification, and section traceability.")

                    if not paper.paragraphs:
                        st.write("No paragraphs extracted.")
                    else:
                        # Search & 3 Filters
                        para_search = st.text_input(
                            "🔍 Search paragraphs across entire paper (text, ID, heading)...",
                            key=f"search_para_{paper.id}"
                        )

                        pf_col1, pf_col2, pf_col3 = st.columns(3)
                        with pf_col1:
                            all_sections = ["All"] + sorted(list(set(p.normalized_section for p in paper.paragraphs)))
                            sel_sec = st.selectbox("Filter by Section:", options=all_sections, key=f"p_sec_{paper.id}")
                        with pf_col2:
                            all_pages = ["All"] + [str(pg) for pg in sorted(list(set(p.page for p in paper.paragraphs)))]
                            sel_page = st.selectbox("Filter by Page:", options=all_pages, key=f"p_page_{paper.id}")
                        with pf_col3:
                            all_types = ["All"] + sorted(list(set(p.content_type for p in paper.paragraphs)))
                            sel_type = st.selectbox("Filter by Semantic Type:", options=all_types, key=f"p_type_{paper.id}")

                        # Execute filter over ALL paragraphs
                        filtered_paragraphs = paper.paragraphs
                        if para_search.strip():
                            pq = para_search.strip().lower()
                            filtered_paragraphs = [
                                p for p in filtered_paragraphs
                                if pq in p.text.lower() or pq in p.paragraph_id.lower() or pq in p.original_heading.lower() or pq in p.normalized_section.lower()
                            ]
                        if sel_sec != "All":
                            filtered_paragraphs = [p for p in filtered_paragraphs if p.normalized_section == sel_sec]
                        if sel_page != "All":
                            filtered_paragraphs = [p for p in filtered_paragraphs if p.page == int(sel_page)]
                        if sel_type != "All":
                            filtered_paragraphs = [p for p in filtered_paragraphs if p.content_type == sel_type]

                        # Export buttons
                        exp_p_col1, exp_p_col2, _ = st.columns([1.5, 1.5, 3])
                        para_export_data = [
                            {
                                "paper_id": p.paper_id,
                                "paragraph_id": p.paragraph_id,
                                "page": p.page,
                                "content_type": p.content_type,
                                "normalized_section": p.normalized_section,
                                "original_heading": p.original_heading,
                                "char_count": p.char_count,
                                "text": p.text,
                            }
                            for p in filtered_paragraphs
                        ]
                        with exp_p_col1:
                            st.download_button(
                                label="📥 Export Paragraphs (JSON)",
                                data=json.dumps(para_export_data, indent=2),
                                file_name=f"{paper.id}_paragraphs.json",
                                mime="application/json",
                                key=f"dl_json_para_{paper.id}"
                            )
                        with exp_p_col2:
                            st.download_button(
                                label="📥 Export Paragraphs (TXT)",
                                data="\n\n".join(
                                    f"[{p.paragraph_id}] Page {p.page} | Section: {p.normalized_section} | Type: {p.content_type}\n{p.text}"
                                    for p in filtered_paragraphs
                                ),
                                file_name=f"{paper.id}_paragraphs.txt",
                                mime="text/plain",
                                key=f"dl_txt_para_{paper.id}"
                            )

                        # Expand / Collapse toggle
                        expand_para_key = f"expand_paras_{paper.id}"
                        if expand_para_key not in st.session_state:
                            st.session_state[expand_para_key] = False

                        is_para_expanded = st.session_state[expand_para_key]
                        preview_para_limit = 20

                        # Status text showing full visibility
                        current_shown = min(len(filtered_paragraphs), preview_para_limit) if not is_para_expanded else len(filtered_paragraphs)
                        st.caption(f"Showing {current_shown} of {len(filtered_paragraphs)} matching paragraphs (Total Extracted: {len(paper.paragraphs)})")

                        if is_para_expanded or len(filtered_paragraphs) <= preview_para_limit:
                            display_paras = filtered_paragraphs
                        else:
                            display_paras = filtered_paragraphs[:preview_para_limit]

                        # Render each paragraph
                        for p in display_paras:
                            with st.container():
                                type_badge = f"`[{p.content_type}]`"
                                st.markdown(
                                    f"**`[{p.paragraph_id}]`** {type_badge} | **Page:** {p.page} | "
                                    f"**Section:** `{p.normalized_section}` *(Original: '{p.original_heading}')*"
                                )
                                st.text(p.text)
                                st.divider()

                        # Dynamic Button
                        if len(filtered_paragraphs) > preview_para_limit:
                            if not is_para_expanded:
                                if st.button(f"▼ Show all {len(filtered_paragraphs)} paragraphs", key=f"btn_exp_para_{paper.id}", type="secondary"):
                                    st.session_state[expand_para_key] = True
                                    st.rerun()
                            else:
                                if st.button(f"▲ Show fewer ({preview_para_limit})", key=f"btn_col_para_{paper.id}", type="secondary"):
                                    st.session_state[expand_para_key] = False
                                    st.rerun()

                # --- Tab 4: Tables ---
                with tab_tables:
                    st.markdown(f"##### 📊 Extracted Academic Tables ({len(paper.tables)})")
                    st.caption("Preserves cell structures, validation status, extraction quality, and raw cell exports.")

                    if not paper.tables:
                        st.info("No structured academic tables detected in this document (or non-tabular text artifacts filtered).")
                    else:
                        expand_tab_key = f"expand_tables_{paper.id}"
                        if expand_tab_key not in st.session_state:
                            st.session_state[expand_tab_key] = False

                        is_tab_expanded = st.session_state[expand_tab_key]
                        preview_tab_limit = 5

                        if is_tab_expanded or len(paper.tables) <= preview_tab_limit:
                            display_tables = paper.tables
                        else:
                            display_tables = paper.tables[:preview_tab_limit]

                        for t in display_tables:
                            status_badge = "🟢 Valid" if t.status == "Valid" else "🟡 Needs Review"
                            quality_color = "🟢" if t.extraction_quality == "high" else ("🟡" if t.extraction_quality == "medium" else "🟠")
                            quality_badge = f"{quality_color} {t.extraction_quality.upper()} Quality"
                            st.markdown(
                                f"##### 📊 Table `{t.table_id}` (Page {t.page}) — {t.row_count} Rows × {t.col_count} Columns [{status_badge} | {quality_badge}]"
                            )
                            if t.status == "Needs Review":
                                st.warning(f"⚠️ Table Validation Note: {t.validation_reason}")
                            elif t.extraction_quality == "low":
                                st.info(f"ℹ️ Extraction Note: {t.validation_reason}")
                            else:
                                st.caption(f"✓ Validation: {t.validation_reason}")

                            df_tab = table_markdown_to_df(t.markdown_content)

                            tb_col1, tb_col2, _ = st.columns([1.5, 1.5, 3])
                            with tb_col1:
                                st.download_button(
                                    label="📥 Export Table (CSV)",
                                    data=df_tab.to_csv(index=False),
                                    file_name=f"{t.table_id}.csv",
                                    mime="text/csv",
                                    key=f"dl_csv_t_{paper.id}_{t.table_id}"
                                )
                            with tb_col2:
                                st.download_button(
                                    label="📥 Export Table (JSON)",
                                    data=df_tab.to_json(orient="records", indent=2),
                                    file_name=f"{t.table_id}.json",
                                    mime="application/json",
                                    key=f"dl_json_t_{paper.id}_{t.table_id}"
                                )

                            st.dataframe(df_tab, use_container_width=True)

                            with st.expander("🔍 View Raw Extracted Markdown"):
                                st.code(t.markdown_content, language="markdown")

                            st.divider()

                        if len(paper.tables) > preview_tab_limit:
                            if not is_tab_expanded:
                                if st.button(f"▼ Show all {len(paper.tables)} tables", key=f"btn_exp_tab_{paper.id}", type="secondary"):
                                    st.session_state[expand_tab_key] = True
                                    st.rerun()
                            else:
                                if st.button(f"▲ Show fewer ({preview_tab_limit})", key=f"btn_col_tab_{paper.id}", type="secondary"):
                                    st.session_state[expand_tab_key] = False
                                    st.rerun()

                # --- Tab 5: References ---
                with tab_refs:
                    st.markdown(f"##### 📖 Bibliography ({len(paper.references)} entries)")
                    st.caption("Complete extracted bibliographic entries with year extraction and searchable citation indices.")

                    if not paper.references:
                        st.info("No bibliography references extracted.")
                    else:
                        ref_s_col, ref_f_col = st.columns([3, 2])
                        with ref_s_col:
                            ref_search = st.text_input(
                                "🔍 Search references (author, title, year, ID, citation text)...",
                                key=f"search_ref_{paper.id}"
                            )
                        with ref_f_col:
                            ref_filter_mode = st.selectbox(
                                "Filter References:",
                                options=["All", "With Publication Year", "Without Publication Year"],
                                key=f"filter_ref_mode_{paper.id}"
                            )

                        filtered_refs = paper.references
                        if ref_search.strip():
                            rq = ref_search.strip().lower()
                            filtered_refs = [
                                r for r in filtered_refs
                                if rq in r.raw_text.lower() or rq in r.ref_id.lower() or (r.year and rq in r.year.lower())
                            ]
                        if ref_filter_mode == "With Publication Year":
                            filtered_refs = [r for r in filtered_refs if r.year is not None]
                        elif ref_filter_mode == "Without Publication Year":
                            filtered_refs = [r for r in filtered_refs if r.year is None]

                        ref_exp1, ref_exp2, _ = st.columns([1.5, 1.5, 3])
                        ref_records = [
                            {
                                "paper_id": r.paper_id,
                                "ref_id": r.ref_id,
                                "year": r.year or "",
                                "citation_text": r.raw_text,
                            }
                            for r in filtered_refs
                        ]
                        with ref_exp1:
                            st.download_button(
                                label="📥 Export References (CSV)",
                                data=pd.DataFrame(ref_records).to_csv(index=False),
                                file_name=f"{paper.id}_references.csv",
                                mime="text/csv",
                                key=f"dl_csv_ref_{paper.id}"
                            )
                        with ref_exp2:
                            st.download_button(
                                label="📥 Export References (JSON)",
                                data=json.dumps(ref_records, indent=2),
                                file_name=f"{paper.id}_references.json",
                                mime="application/json",
                                key=f"dl_json_ref_{paper.id}"
                            )

                        preview_ref_limit = 20
                        expand_ref_key = f"expand_refs_{paper.id}"
                        if expand_ref_key not in st.session_state:
                            st.session_state[expand_ref_key] = False

                        is_ref_expanded = st.session_state[expand_ref_key]

                        current_ref_shown = min(len(filtered_refs), preview_ref_limit) if not is_ref_expanded else len(filtered_refs)
                        st.caption(f"Showing {current_ref_shown} of {len(filtered_refs)} references (Total Extracted: {len(paper.references)})")

                        if is_ref_expanded or len(filtered_refs) <= preview_ref_limit:
                            display_refs = filtered_refs
                        else:
                            display_refs = filtered_refs[:preview_ref_limit]

                        for idx, r in enumerate(display_refs):
                            year_badge = f"`[{r.year}]`" if r.year else ""
                            st.markdown(f"**{idx + 1:03d}** | **`[{r.ref_id}]`** {year_badge} {r.raw_text}")

                        if len(filtered_refs) > preview_ref_limit:
                            if not is_ref_expanded:
                                if st.button(f"▼ Show all {len(filtered_refs)} references", key=f"btn_exp_ref_{paper.id}", type="secondary"):
                                    st.session_state[expand_ref_key] = True
                                    st.rerun()
                            else:
                                if st.button(f"▲ Show fewer ({preview_ref_limit})", key=f"btn_col_ref_{paper.id}", type="secondary"):
                                    st.session_state[expand_ref_key] = False
                                    st.rerun()

                # Remove Button
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("🗑️ Remove Paper", key=f"del_{paper.id}"):
                    st.session_state["uploaded_papers"] = [
                        p for p in st.session_state["uploaded_papers"] if p.id != paper.id
                    ]
                    st.rerun()

        # Clear All Button
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Clear All Uploaded Papers", type="secondary"):
            st.session_state["uploaded_papers"] = []
            st.rerun()
