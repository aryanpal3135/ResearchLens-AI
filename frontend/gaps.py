"""
Research Gap Detection Page for ResearchLens AI.
Identifies evidence-grounded research gaps, repeated limitations, and unexplored territories
using Microsoft Foundry (gpt-4.1-mini) and Phase 3 Hybrid RAG retrieval.
Strictly enforces:
- Evidence-first retrieval with isolated per-paper querying.
- Categorical evidence support (High, Moderate, Limited).
- Distinct separation of evidence-grounded gaps from AI-generated future research directions.
- Verifiable chunk provenance without truncation.
"""

import json
from typing import List, Optional
import streamlit as st

from config.settings import settings
from services.i18n import t
from frontend.components import render_header, render_info_card
from frontend.retrieval_ui import get_or_create_retrieval_engine, sync_papers_to_retrieval_engine
from models.paper import PaperDocument, EvidenceItem
from models.research_gap import ResearchGap, ResearchGapAnalysisResult, GAP_CATEGORIES
from services.foundry_agent import ResearchLensAgent


GAP_FOCUS_OPTIONS = {
    "Comprehensive Research Gap Detection": "Comprehensive research gap detection across limitations, evaluation bounds, and future directions",
    "What limitations remain in the evaluation of these papers?": "What limitations, constraints, and assumptions remain in the evaluation of these papers?",
    "What research areas are insufficiently explored?": "What research areas, domains, and application contexts are insufficiently explored?",
    "What experiments are missing from these studies?": "What experiments, baseline comparisons, and ablation studies are missing from these studies?",
    "What future research directions are explicitly mentioned by the authors?": "What explicit future research directions and extensions are mentioned by the authors?",
    "✏️ Custom Focus Question / Inquiry": "",
}
PRESET_QUERIES = list(GAP_FOCUS_OPTIONS.keys())


def render_gaps_page():
    """Renders research gap detection view."""
    # Ensure dropdown styling wraps text properly
    st.markdown(
        """
        <style>
        div[data-baseweb="select"] { min-height: 48px !important; height: auto !important; }
        div[data-baseweb="select"] > div { min-height: 48px !important; height: auto !important; white-space: normal !important; padding: 4px 8px !important; }
        div[data-baseweb="select"] span, div[data-baseweb="select"] div { white-space: normal !important; word-break: break-word !important; overflow: visible !important; }
        div[data-baseweb="popover"] { min-width: 100% !important; max-width: 95vw !important; width: auto !important; }
        ul[role="listbox"] { max-height: 480px !important; width: 100% !important; }
        ul[role="listbox"] li, ul[role="listbox"] li[role="option"] {
            white-space: normal !important; word-break: break-word !important; overflow: visible !important;
            height: auto !important; min-height: 44px !important; line-height: 1.45 !important; padding: 10px 14px !important;
            border-bottom: 1px solid rgba(51, 65, 85, 0.4) !important;
        }
        ul[role="listbox"] li > div, ul[role="listbox"] li[role="option"] > div { white-space: normal !important; word-break: break-word !important; }
        .stTextArea textarea { font-size: 0.95rem !important; line-height: 1.5 !important; white-space: normal !important; word-break: break-word !important; border-radius: 8px !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    render_header(
        title=t("gaps_title"),
        subtitle=t("gaps_subtitle"),
        badge="Phase 6: Research Gap Detection",
    )

    papers: List[PaperDocument] = st.session_state.get("uploaded_papers", [])

    if not papers:
        render_info_card(
            title="No Research Papers Uploaded",
            description=(
                "Please upload at least **1 research paper** on the **Upload Papers** page "
                "to enable evidence-grounded research gap detection."
            ),
            icon="ℹ️",
        )
        return

    # Synchronize papers with retrieval engine
    sync_papers_to_retrieval_engine()
    engine = get_or_create_retrieval_engine()
    agent = ResearchLensAgent()
    status = agent.get_connection_status()

    # Agent Status Banner
    st.markdown(
        f"""
        <div style="background-color: #0F172A; border: 1px solid #1E293B; border-radius: 8px; padding: 10px 16px; margin-bottom: 16px;">
            <span style="font-weight: 600; color: #38BDF8;">🧠 Microsoft Foundry Agent:</span>
            <span style="color: #F8FAFC; margin-left: 6px;"><code>{status.get('research_agent', 'researchmate-gpt4-1-mini')}</code> (v{status.get('research_agent_version', '1')})</span>
            &nbsp;|&nbsp; Status: <span style="color: {'#34D399' if status.get('configured') else '#F87171'}; font-weight: 600;">
                {'🟢 Active (Foundry Agent)' if status.get('configured') else '🔴 Unconfigured'}
            </span>
            &nbsp;|&nbsp; Retrieval Engine: <span style="color: #38BDF8; font-weight: 600;">Phase 3 Hybrid RAG</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 1. Paper Selection (Single or Multi-Paper)
    st.markdown(f"#### {t('gaps_select_papers')}")
    paper_options = {p.id: f"[{p.id}] {p.metadata.title or p.filename} ({p.page_count} pp)" for p in papers}
    
    selected_paper_ids = st.multiselect(
        "Choose 1 or more papers to detect research gaps:",
        options=list(paper_options.keys()),
        default=list(paper_options.keys())[:min(2, len(paper_options))],
        format_func=lambda pid: paper_options[pid],
        key="gap_selected_paper_ids",
    )

    if not selected_paper_ids:
        st.warning("⚠️ Please select at least 1 paper to run research gap detection.")
        return

    selected_paper_objs = [p for p in papers if p.id in selected_paper_ids]
    st.caption(
        f"Selected **{len(selected_paper_objs)} paper(s)**: "
        + ", ".join([f"`{p.id}` ({p.metadata.title or p.filename})" for p in selected_paper_objs])
    )

    # 2. Focus Query / Research Question
    st.markdown(f"#### {t('gaps_focus_heading')}")
    preset_choice = st.selectbox(
        "Choose an analysis angle or custom query:",
        options=list(GAP_FOCUS_OPTIONS.keys()),
        index=0,
        key="gap_preset_choice",
        help="Select a focused research angle or type your own question.",
    )
    if preset_choice == "✏️ Custom Focus Question / Inquiry":
        custom_query_val = st.text_area(
            "Enter custom research question or focus topic (full question shown):",
            value=st.session_state.get("gap_custom_query_input", ""),
            placeholder="Type your complete research question here... e.g. What limitations remain in the evaluation and experimental methodologies of these papers?",
            height=90,
            key="gap_custom_query_input",
            help="Write your complete research question. The full question remains visible without scrolling out of view.",
        )
        active_query_str = custom_query_val.strip() or "Custom research gap inquiry"
    else:
        active_query_str = GAP_FOCUS_OPTIONS.get(preset_choice, "") or preset_choice

    # Display the full question so it is 100% visible and never cut off
    st.markdown(
        f"""
        <div style="background-color: #0F172A; border: 1px solid #1E293B; border-left: 4px solid #38BDF8; border-radius: 8px; padding: 12px 16px; margin: 8px 0 16px 0;">
            <div style="font-size: 0.75rem; font-weight: 700; color: #38BDF8; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px;">
                📌 Selected Full Research Question / Focus:
            </div>
            <div style="color: #F8FAFC; font-size: 0.95rem; font-weight: 500; line-height: 1.5;">
                {active_query_str}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 3. Run Gap Detection Button
    if st.button(t("gaps_btn"), type="primary", use_container_width=True):
        with st.spinner("Retrieving evidence per paper and detecting research gaps with Microsoft Foundry (gpt-4.1-mini)..."):
            result = agent.detect_research_gaps(
                papers=selected_paper_objs,
                engine=engine,
                custom_query=active_query_str,
            )
            st.session_state["gap_detection_result"] = result

    # 4. Render Gap Detection Results
    result: Optional[ResearchGapAnalysisResult] = st.session_state.get("gap_detection_result")

    if result is None:
        st.info("💡 Click **Detect Research Gaps** above to analyze limitations, future directions, and unexplored areas.")
        return

    if result.status == "error":
        st.error(f"❌ Research Gap Detection Error: {result.error_message}")
        return

    if result.status == "insufficient_evidence":
        st.warning(f"⚠️ {result.error_message or 'Insufficient evidence to identify grounded research gaps.'}")
        return

    st.markdown("---")
    st.markdown(f"### 🎯 Detected Research Gaps ({len(result.gaps)})")
    st.caption(
        f"Execution Latency: **{result.execution_latency_ms:.1f} ms** | "
        f"Model: **{result.model_used or 'Microsoft Foundry (gpt-4.1-mini)'}** | "
        f"Retrieved Evidence Chunks: **{len(result.all_evidence_items)}**"
    )

    # Summary Overview Metrics Bar
    counts = result.summary_counts
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("Total Gaps", counts.get("total_gaps", len(result.gaps)))
    with m2:
        st.metric("Explicit Gaps", counts.get("explicit_gaps", 0))
    with m3:
        st.metric("Derived Gaps", counts.get("evidence_derived_gaps", 0))
    with m4:
        st.metric("Cross-Paper Gaps", counts.get("cross_paper_gaps", 0))
    with m5:
        st.metric("Potential Conflicts", counts.get("contradictory_evidence", 0))

    if not result.gaps:
        st.info("No grounded research gaps could be verified from the retrieved evidence.")
        return

    # Render Individual Research Gaps
    for idx, gap in enumerate(result.gaps):
        # Format badges
        type_color = "#10B981" if gap.evidence_type == "explicit" else ("#8B5CF6" if gap.evidence_type == "cross_paper" else "#0EA5E9")
        support_color = "#10B981" if "high" in gap.evidence_support.lower() else ("#F59E0B" if "moderate" in gap.evidence_support.lower() else "#64748B")

        expander_title = f"📍 [{gap.gap_id}] {gap.title} — ({gap.category})"
        with st.expander(expander_title, expanded=(idx < 2)):
            # Badges row
            st.markdown(
                f"""
                <div style="display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 12px;">
                    <span style="background: #1E293B; color: #94A3B8; border: 1px solid #334155; padding: 3px 8px; border-radius: 4px; font-size: 12px; font-weight: 600;">
                        🏷️ {gap.category}
                    </span>
                    <span style="background: rgba({ '16,185,129,0.15' if gap.evidence_type == 'explicit' else ('139,92,246,0.15' if gap.evidence_type == 'cross_paper' else '14,165,233,0.15') }); color: {type_color}; border: 1px solid {type_color}; padding: 3px 8px; border-radius: 4px; font-size: 12px; font-weight: 600;">
                        🔍 {gap.evidence_type_label}
                    </span>
                    <span style="background: rgba({ '16,185,129,0.15' if 'high' in gap.evidence_support.lower() else ('245,158,11,0.15' if 'moderate' in gap.evidence_support.lower() else '100,116,139,0.15') }); color: {support_color}; border: 1px solid {support_color}; padding: 3px 8px; border-radius: 4px; font-size: 12px; font-weight: 600;">
                        📊 {gap.evidence_support}
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Description
            st.markdown(f"**Description:** {gap.description}")

            # Rationale
            if gap.rationale:
                st.markdown(f"**Why this is an evidence-grounded gap:** {gap.rationale}")

            # Supporting Papers
            st.markdown("**Supporting Papers:** " + " ".join([f"`{pid}`" for pid in gap.supporting_papers]))

            # Contradiction / Tension Callout (if present)
            if gap.potential_tension:
                st.warning(f"⚠️ **Potentially Conflicting Evidence:** {gap.potential_tension}")

            # Potential Research Direction (Separated AI Suggestion)
            if gap.potential_research_direction:
                st.markdown(
                    f"""
                    <div style="background-color: #1E1B4B; border: 1px solid #4338CA; border-radius: 6px; padding: 12px; margin-top: 10px; margin-bottom: 12px;">
                        <span style="color: #A5B4FC; font-weight: 600;">💡 Potential Future Research Direction (AI-Generated Suggestion):</span>
                        <p style="color: #E0E7FF; margin: 4px 0 0 0; font-size: 14px;">{gap.potential_research_direction}</p>
                        <span style="color: #818CF8; font-size: 11px; font-style: italic;">*Note: This is an AI-formulated hypothesis for future work, not an author-stated finding.</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Verifiable Supporting Evidence & Provenance
            if gap.evidence_items:
                with st.expander(f"📑 Verifiable Supporting Evidence & Provenance ({len(gap.evidence_items)} chunks)", expanded=False):
                    for ev_idx, ev in enumerate(gap.evidence_items):
                        p_title = result.selected_paper_titles.get(ev.paper_id, ev.paper_id)
                        st.markdown(
                            f"""
                            <div style="background-color: #0F172A; border-left: 3px solid #38BDF8; padding: 8px 12px; margin-bottom: 8px; border-radius: 0 4px 4px 0;">
                                <div style="display: flex; justify-content: space-between; font-size: 12px; color: #94A3B8; margin-bottom: 4px;">
                                    <span><strong>{ev.citation_label}</strong> &bull; Paper: <code>{ev.paper_id}</code> ({p_title})</span>
                                    <span>Page {ev.page_start} | Score: {ev.score:.4f}</span>
                                </div>
                                <div style="color: #CBD5E1; font-size: 13px; line-height: 1.5; font-family: monospace; white-space: pre-wrap;">
{ev.text}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

    # 5. Export Section
    st.markdown("---")
    st.markdown("#### 📥 Export Gap Detection Results")
    exp_col1, exp_col2 = st.columns(2)

    with exp_col1:
        # Markdown Export
        md_lines = [
            f"# ResearchLens AI — Research Gap Detection Report",
            f"**Query:** {result.query}",
            f"**Analyzed Papers:** {', '.join(result.selected_paper_ids)}",
            f"**Model Deployment:** {result.model_used or 'gpt-4.1-mini'}",
            f"**Execution Latency:** {result.execution_latency_ms:.1f} ms",
            f"**Total Gaps:** {len(result.gaps)}",
            f"\n## Summary Metrics",
            f"- Explicit Gaps: {counts.get('explicit_gaps', 0)}",
            f"- Evidence-Derived Gaps: {counts.get('evidence_derived_gaps', 0)}",
            f"- Cross-Paper Gaps: {counts.get('cross_paper_gaps', 0)}",
            f"- Potential Conflicts: {counts.get('contradictory_evidence', 0)}",
            f"\n## Identified Research Gaps\n",
        ]
        for g in result.gaps:
            md_lines.append(f"### [{g.gap_id}] {g.title}")
            md_lines.append(f"- **Category:** {g.category}")
            md_lines.append(f"- **Evidence Type:** {g.evidence_type_label} (`{g.evidence_type}`)")
            md_lines.append(f"- **Evidence Support:** {g.evidence_support}")
            md_lines.append(f"- **Supporting Papers:** {', '.join(g.supporting_papers)}")
            md_lines.append(f"- **Description:** {g.description}")
            if g.rationale:
                md_lines.append(f"- **Rationale:** {g.rationale}")
            if g.potential_tension:
                md_lines.append(f"- **Potential Tension:** {g.potential_tension}")
            if g.potential_research_direction:
                md_lines.append(f"- **Potential Research Direction (AI Suggestion):** {g.potential_research_direction}")
            if g.evidence_items:
                md_lines.append(f"\n#### Supporting Evidence Chunks ({len(g.evidence_items)}):")
                for it in g.evidence_items:
                    md_lines.append(f"> **{it.citation_label}** (Page {it.page_start}, Score: {it.score:.4f}):\n> {it.text}\n")
            md_lines.append("\n---\n")

        st.download_button(
            label="📄 Download Markdown Report",
            data="\n".join(md_lines),
            file_name="research_gaps_report.md",
            mime="text/markdown",
            use_container_width=True,
        )

    with exp_col2:
        # JSON Export
        json_export_data = {
            "analysis_id": result.analysis_id,
            "query": result.query,
            "selected_paper_ids": result.selected_paper_ids,
            "selected_paper_titles": result.selected_paper_titles,
            "summary_counts": result.summary_counts,
            "model_used": result.model_used,
            "execution_latency_ms": result.execution_latency_ms,
            "gaps": [
                {
                    "gap_id": g.gap_id,
                    "title": g.title,
                    "category": g.category,
                    "description": g.description,
                    "evidence_type": g.evidence_type,
                    "evidence_type_label": g.evidence_type_label,
                    "evidence_support": g.evidence_support,
                    "supporting_papers": g.supporting_papers,
                    "rationale": g.rationale,
                    "affected_dimension": g.affected_dimension,
                    "potential_research_direction": g.potential_research_direction,
                    "potential_tension": g.potential_tension,
                    "citation_labels": g.citation_labels,
                    "evidence_items": [
                        {
                            "citation_label": ev.citation_label,
                            "paper_id": ev.paper_id,
                            "chunk_id": ev.chunk_id,
                            "page_start": ev.page_start,
                            "page_end": ev.page_end,
                            "normalized_section": ev.normalized_section,
                            "score": ev.score,
                            "text": ev.text,
                        }
                        for ev in g.evidence_items
                    ],
                }
                for g in result.gaps
            ],
        }
        st.download_button(
            label="💾 Download JSON Data",
            data=json.dumps(json_export_data, indent=2),
            file_name="research_gaps_data.json",
            mime="application/json",
            use_container_width=True,
        )
