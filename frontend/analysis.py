"""
Paper Analysis Page for ResearchLens AI.
Displays the structured 7-section deep academic analysis powered by Microsoft Foundry (gpt-4.1-mini)
with explicit vs. inferred attribution, verifiable evidence citations, and zero-hallucination guarantees.
"""

import json
from typing import List, Optional
import streamlit as st

from frontend.components import render_header, render_info_card, render_evidence_badge
from frontend.retrieval_ui import get_or_create_retrieval_engine, sync_papers_to_retrieval_engine
from models.analysis import PaperSectionAnalysis, AnalysisSectionItem
from models.paper import PaperDocument
from services.foundry_agent import ResearchLensAgent
from services.i18n import t


def render_analysis_page():
    """Renders single paper 7-section structured academic analysis."""
    render_header(
        title=t("analysis_title", "📄 Deep Paper Analysis"),
        subtitle=t("analysis_subtitle", "Exhaustive 7-dimension academic breakdown grounded in paper evidence."),
        badge="Phase 4: Structured Analysis",
    )

    papers: List[PaperDocument] = st.session_state.get("uploaded_papers", [])

    if not papers:
        render_info_card(
            title="No Papers Uploaded",
            description="Please upload research papers on the **Upload Papers** page first to generate analysis.",
            icon="ℹ️",
        )
        return

    # Select target paper
    paper_options = {p.id: f"[{p.id}] {p.filename} ({p.page_count} pages)" for p in papers}
    selected_paper_id = st.selectbox(
        "Select Paper to Analyze:",
        options=list(paper_options.keys()),
        format_func=lambda pid: paper_options[pid],
        key="analysis_target_paper_selector",
    )

    selected_paper = next((p for p in papers if p.id == selected_paper_id), None)
    if not selected_paper:
        return

    st.markdown(f"### 📑 Analysis Workspace: `{selected_paper.filename}`")

    agent = ResearchLensAgent()
    conn_status = agent.get_connection_status()

    # Agent Status Bar
    res_agent_name = conn_status.get("research_agent", "researchmate-gpt4-1-mini")
    res_agent_ver = conn_status.get("research_agent_version", "1")
    status_color = "#34D399" if conn_status.get("connected") else "#F87171"
    status_text = "🟢 Active" if conn_status.get("connected") else "🔴 Disconnected"
    st.markdown(
        f"""
        <div style="background-color: #0F172A; border: 1px solid #1E293B; border-radius: 8px; padding: 10px 16px; margin-bottom: 16px;">
            <span style="font-weight: 600; color: #38BDF8;">🧠 Microsoft Foundry Agent:</span>
            <span style="color: #F8FAFC; margin-left: 6px;"><code>{res_agent_name}</code> (v{res_agent_ver})</span>
            &nbsp;|&nbsp; Status: <span style="color: {status_color}; font-weight: 600;">{status_text}</span>
            &nbsp;|&nbsp; Target Scope: <code style="color: #38BDF8;">paper_id: {selected_paper.id}</code>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # State storage for analysis results
    analysis_cache_key = f"foundry_analysis_{selected_paper.id}"
    cached_analysis: Optional[PaperSectionAnalysis] = st.session_state.get(analysis_cache_key)

    # Action Controls
    col_btn, col_info = st.columns([1, 2])
    with col_btn:
        run_analysis = st.button(
            t("analysis_btn", "🔬 Run Deep Analysis"),
            type="primary",
            use_container_width=True,
            key="run_7_section_analysis_btn",
        )
    with col_info:
        if cached_analysis:
            st.caption(
                f"✅ Analysis cached for `{selected_paper.filename}` "
                f"(Latency: {cached_analysis.execution_latency_ms:.1f} ms | Model: `{cached_analysis.model_used or 'gpt-4.1-mini'}`)"
            )
        else:
            st.caption("Click to synthesize all 7 academic dimensions directly from grounded paper chunks.")

    if run_analysis:
        sync_papers_to_retrieval_engine()
        engine = get_or_create_retrieval_engine()

        progress_bar = st.progress(0, text="Initializing Microsoft Foundry agent...")
        with st.spinner(f"Analyzing {selected_paper.filename} across 7 academic dimensions..."):
            progress_bar.progress(20, text="Retrieving section-specific chunk evidence...")
            analysis_result = agent.analyze_paper_sections(selected_paper, engine=engine)
            st.session_state[analysis_cache_key] = analysis_result
            cached_analysis = analysis_result
            progress_bar.progress(100, text="Analysis complete!")

    if not cached_analysis:
        st.info("ℹ️ No analysis generated yet for this paper. Click **'Run 7-Section AI Analysis'** above to generate.")
        return

    # Render 7 Sections in Tabs or Cards
    section_tabs = st.tabs([
        "1. Executive Summary",
        "2. Research Objective",
        "3. Methodology",
        "4. Dataset & Setup",
        "5. Key Findings",
        "6. Limitations",
        "7. Conclusion",
    ])

    sec_items = list(cached_analysis.sections.values())
    for i, tab in enumerate(section_tabs):
        if i < len(sec_items):
            sec: AnalysisSectionItem = sec_items[i]
            with tab:
                # Header with badges
                c_title, c_badge = st.columns([3, 1])
                with c_title:
                    st.markdown(f"#### {sec.section_name}")
                with c_badge:
                    badge_html = render_evidence_badge(sec.is_explicit)
                    st.markdown(badge_html, unsafe_allow_html=True)
                    if sec.evidence_items:
                        st.markdown('<span style="background-color:#064E3B;color:#34D399;padding:2px 8px;border-radius:4px;font-size:0.75rem;">🟢 Grounded</span>', unsafe_allow_html=True)
                    else:
                        st.markdown('<span style="background-color:#78350F;color:#FDE047;padding:2px 8px;border-radius:4px;font-size:0.75rem;">🟡 Insufficient Evidence</span>', unsafe_allow_html=True)

                # Synthesized Content
                st.markdown(sec.content)

                # Verifiable Evidence Provenance
                if sec.evidence_items:
                    with st.expander(f"📍 Verifiable Evidence & Provenance ({len(sec.evidence_items)} Chunks)", expanded=False):
                        for c_idx, ev in enumerate(sec.evidence_items, start=1):
                            st.markdown(
                                f"**[{c_idx}] Section:** `{ev.normalized_section}` (`{ev.original_heading}`) | "
                                f"**Pages:** {ev.page_start}–{ev.page_end} | "
                                f"**RRF Score:** `{ev.score:.4f}` | "
                                f"**Chunk ID:** `[{ev.chunk_id}]`"
                            )
                            st.markdown(
                                f"<div style='background-color: #0F172A; border-left: 3px solid #38BDF8; padding: 8px 12px; margin-bottom: 8px; font-size: 0.85rem; max-height: 200px; overflow-y: auto;'>"
                                f"{ev.text}"
                                f"</div>",
                                unsafe_allow_html=True,
                            )
                else:
                    st.caption("No specific chunk citations available for this section.")

    # Export Section
    st.divider()
    col_exp1, col_exp2 = st.columns([1, 1])
    with col_exp1:
        # Markdown export
        md_export = f"# Research Analysis: {cached_analysis.paper_title}\n\n"
        md_export += f"- Paper ID: {cached_analysis.paper_id}\n"
        md_export += f"- Model: {cached_analysis.model_used or 'gpt-4.1-mini'}\n"
        md_export += f"- Total Latency: {cached_analysis.execution_latency_ms:.1f} ms\n\n"
        for s in cached_analysis.sections.values():
            md_export += f"## {s.section_name}\n"
            md_export += f"Attribution: {'Explicitly Stated' if s.is_explicit else 'AI Inferred'}\n\n"
            md_export += f"{s.content}\n\n"
            if s.evidence_items:
                md_export += "### Provenance\n"
                for ev in s.evidence_items:
                    md_export += f"- Chunk `{ev.chunk_id}` (Page {ev.page_start}-{ev.page_end}, Section: {ev.normalized_section}, Score: {ev.score:.4f})\n"
            md_export += "\n---\n\n"

        st.download_button(
            "📥 Download Analysis as Markdown",
            data=md_export,
            file_name=f"analysis_{cached_analysis.paper_id}.md",
            mime="text/markdown",
            use_container_width=True,
        )
    with col_exp2:
        # JSON export
        json_export = json.dumps(cached_analysis.model_dump(), indent=2)
        st.download_button(
            "📥 Download Analysis as JSON",
            data=json_export,
            file_name=f"analysis_{cached_analysis.paper_id}.json",
            mime="application/json",
            use_container_width=True,
        )

