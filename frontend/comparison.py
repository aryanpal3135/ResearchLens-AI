"""
Multi-Paper Comparison Page for ResearchLens AI.
Provides evidence-grounded comparative analysis across 10 academic dimensions,
verifiable similarities, and differences using Microsoft Foundry (gpt-4.1-mini)
with strict per-paper retrieval isolation and full provenance.
"""

import json
from typing import List, Optional
import streamlit as st
import pandas as pd

from frontend.components import render_header, render_info_card
from frontend.retrieval_ui import get_or_create_retrieval_engine, sync_papers_to_retrieval_engine
from models.analysis import MultiPaperComparison, DimensionComparisonItem, ComparisonPoint
from models.paper import PaperDocument, EvidenceItem
from services.foundry_agent import ResearchLensAgent
from services.i18n import t


def render_comparison_page():
    """Renders cross-paper evidence-grounded comparative analysis."""
    render_header(
        title=t("comp_title", "⚖️ Multi-Paper Comparison"),
        subtitle=t("comp_subtitle", "Compare methodologies, datasets, findings, and limitations using strictly retrieved evidence."),
        badge="Phase 5: Multi-Paper Comparison",
    )

    papers: List[PaperDocument] = st.session_state.get("uploaded_papers", [])

    if len(papers) < 2:
        render_info_card(
            title="At Least Two Papers Required",
            description=(
                f"Currently, {len(papers)} paper(s) are uploaded in the session. "
                "Please upload at least **2 research papers** on the **Upload Papers** page to enable comparative analysis."
            ),
            icon="ℹ️",
        )
        return

    # Synchronize papers to retrieval engine
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
            &nbsp;|&nbsp; Evidence Mode: <span style="color: #38BDF8; font-weight: 600;">Strict Per-Paper Isolation</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Paper Selection
    st.markdown("#### 1. Select Papers to Compare")
    paper_options = {p.id: f"[{p.id}] {p.metadata.title or p.filename} ({p.page_count} pp)" for p in papers}
    selected_paper_ids = st.multiselect(
        "Choose at least 2 papers for comparative analysis:",
        options=list(paper_options.keys()),
        default=list(paper_options.keys())[:min(2, len(paper_options))],
        format_func=lambda pid: paper_options[pid],
        key="comparison_selected_paper_ids",
    )

    if len(selected_paper_ids) < 2:
        st.warning("⚠️ Please select at least 2 papers from the dropdown above to run comparison.")
        return

    selected_paper_objs = [p for p in papers if p.id in selected_paper_ids]
    paper_a = selected_paper_objs[0]
    paper_b = selected_paper_objs[1]
    title_a = paper_a.metadata.title or paper_a.filename
    title_b = paper_b.metadata.title or paper_b.filename

    st.caption(
        f"Selected {len(selected_paper_objs)} papers: "
        f"**Paper A:** `[{paper_a.id}] {title_a}` vs. **Paper B:** `[{paper_b.id}] {title_b}`"
    )

    # Optional Custom Comparison Query
    st.markdown("#### 2. Comparison Focus / Query (Optional)")
    default_test_query = (
        "Compare the research objectives, methodologies, architectures, "
        "training approaches, and evaluation approaches of these two papers."
    )
    custom_query = st.text_area(
        "Enter an optional specific comparison question (or leave default for complete 10-dimension comparison):",
        value=st.session_state.get("comparison_query_input", default_test_query),
        height=75,
        key="comparison_custom_query_field",
        help="Write your complete comparison question. The full question remains visible without scrolling.",
    )

    # Run Comparison Button
    comp_cache_key = f"comparison_{'_'.join(sorted(selected_paper_ids))}_{hash(custom_query)}"
    cached_comparison: Optional[MultiPaperComparison] = st.session_state.get(comp_cache_key)

    col_btn, col_info = st.columns([1, 2])
    with col_btn:
        run_comp = st.button(
            t("comp_btn", "⚖️ Compare Papers"),
            type="primary",
            use_container_width=True,
            key="run_comparison_btn",
        )
    with col_info:
        if cached_comparison and cached_comparison.status == "completed":
            st.caption(
                f"✅ Comparison cached (Latency: `{cached_comparison.execution_latency_ms:.1f} ms` | "
                f"Model: `{cached_comparison.model_used or 'gpt-4.1-mini'}` | "
                f"Evidence Chunks: `{len(cached_comparison.all_evidence_items)}`)"
            )
        else:
            st.caption("Executes strict per-paper hybrid retrieval, bundles evidence, and synthesizes comparison via Foundry.")

    if run_comp:
        with st.spinner("Retrieving evidence per-paper and synthesizing comparison via Foundry gpt-4.1-mini..."):
            comparison_res = agent.compare_papers(
                papers=selected_paper_objs,
                engine=engine,
                custom_query=custom_query,
            )
            st.session_state[comp_cache_key] = comparison_res
            cached_comparison = comparison_res

    if not cached_comparison:
        st.info("ℹ️ Click **'⚖️ Compare Papers'** above to generate the grounded comparative analysis.")
        return

    if cached_comparison.status == "error":
        st.error(f"❌ Comparison synthesis error: {cached_comparison.error_message}")
        return

    st.divider()

    st.divider()

    # Helper function to reliably resolve paper summary from dimension item
    def _resolve_paper_summary(item: DimensionComparisonItem, target_paper: PaperDocument, other_paper: PaperDocument, idx: int) -> str:
        if not item or not item.paper_summaries:
            return "Not clearly identified in the retrieved evidence."
        # 1. Direct ID match
        if target_paper.id in item.paper_summaries:
            val = item.paper_summaries[target_paper.id]
            if val and "Not clearly identified" not in val:
                return val
        # 2. Check title or alias keys
        target_name_lower = (target_paper.metadata.title or target_paper.filename).lower()
        for k, v in item.paper_summaries.items():
            k_lower = k.lower()
            if (target_paper.id.lower() in k_lower or f"paper_{idx+1}" in k_lower or f"paper_{'a' if idx==0 else 'b'}" in k_lower) and v:
                if "Not clearly identified" not in v:
                    return v
            if any(word in k_lower for word in target_name_lower.split()[:3]) and v:
                return v
        # 3. Positional fallback if exactly two entries exist
        vals = list(item.paper_summaries.values())
        if len(vals) > idx and vals[idx] and "Not clearly identified" not in vals[idx]:
            return vals[idx]
        return item.paper_summaries.get(target_paper.id, "Not clearly identified in the retrieved evidence.")

    # 1. Executive / Focused Synthesis Callout (Prominently displayed)
    if cached_comparison.custom_query_answer:
        st.markdown("### 🎯 Executive Comparative Synthesis")
        st.markdown(
            f"""
            <div style="background-color: #0F172A; border-left: 4px solid #38BDF8; padding: 14px 18px; border-radius: 6px; margin-bottom: 20px; line-height: 1.6; font-size: 1rem; color: #F1F5F9;">
                {cached_comparison.custom_query_answer}
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 2. Cross-Paper Similarities & Differences (Immediately visible)
    st.markdown("### 🧩 Key Similarities & Core Differences")
    col_sim, col_diff = st.columns(2)

    with col_sim:
        st.markdown("#### 🔄 Grounded Similarities")
        if cached_comparison.similarities:
            for s in cached_comparison.similarities:
                st.markdown(
                    f"""
                    <div style="background: rgba(30, 41, 59, 0.6); border: 1px solid rgba(56, 189, 248, 0.3); border-radius: 8px; padding: 12px; margin-bottom: 10px;">
                        <div style="font-weight: 700; color: #38BDF8; font-size: 0.95rem; margin-bottom: 4px;">• {s.topic}</div>
                        <div style="font-size: 0.88rem; color: #E2E8F0; line-height: 1.5;">{s.description}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("No direct methodological or experimental similarities identified in retrieved evidence.")

    with col_diff:
        st.markdown("#### ⚡ Core Differences")
        if cached_comparison.differences:
            for d in cached_comparison.differences:
                st.markdown(
                    f"""
                    <div style="background: rgba(30, 41, 59, 0.6); border: 1px solid rgba(248, 113, 113, 0.3); border-radius: 8px; padding: 12px; margin-bottom: 10px;">
                        <div style="font-weight: 700; color: #F87171; font-size: 0.95rem; margin-bottom: 4px;">• {d.topic}</div>
                        <div style="font-size: 0.88rem; color: #E2E8F0; line-height: 1.5;">{d.description}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("No distinct differences identified in retrieved evidence.")

    st.divider()

    # 3. Detailed Dimension Inspection
    st.markdown("### 📑 10-Dimension Academic Breakdown & Evidence Provenance")

    dim_names = list(cached_comparison.dimensions.keys())
    dim_tabs = st.tabs([f"{i+1}. {name}" for i, name in enumerate(dim_names)])

    for i, tab in enumerate(dim_tabs):
        d_name = dim_names[i]
        d_item: DimensionComparisonItem = cached_comparison.dimensions[d_name]
        sum_a = _resolve_paper_summary(d_item, paper_a, paper_b, 0)
        sum_b = _resolve_paper_summary(d_item, paper_b, paper_a, 1)

        with tab:
            st.markdown(f"#### Dimension: {d_name}")

            # Synthesis block
            if d_item.synthesis:
                st.markdown(
                    f"""
                    <div style='background-color: #1E293B; border-radius: 8px; border-left: 4px solid #818CF8; padding: 12px 16px; margin-bottom: 16px; font-size: 0.95rem; color: #F8FAFC;'>
                        <strong>⚖️ Comparative Contrast:</strong><br/>
                        <span style="color: #CBD5E1;">{d_item.synthesis}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"**📄 Paper A:** `{paper_a.id}` — *{title_a}*")
                st.markdown(
                    f"""
                    <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 8px; padding: 12px; margin-bottom: 12px; font-size: 0.9rem; line-height: 1.6; color: #F1F5F9;">
                        {sum_a}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                ev_a_list = d_item.paper_evidence.get(paper_a.id, [])
                if ev_a_list:
                    with st.expander(f"📍 Paper A Evidence ({len(ev_a_list)} chunks)", expanded=False):
                        for ev in ev_a_list:
                            st.markdown(
                                f"**Section:** `{ev.normalized_section}` (`{ev.original_heading}`) | "
                                f"**Page:** {ev.page_start}–{ev.page_end} | "
                                f"**Score:** `{ev.score:.4f}` | **Chunk:** `[{ev.chunk_id}]`"
                            )
                            st.markdown(
                                f"<div style='background-color: #0F172A; border-left: 3px solid #38BDF8; padding: 8px 12px; margin-bottom: 8px; font-size: 0.85rem; max-height: 200px; overflow-y: auto;'>"
                                f"{ev.text}"
                                f"</div>",
                                unsafe_allow_html=True,
                            )
                else:
                    st.caption("No direct chunks retrieved for Paper A in this dimension.")

            with col_b:
                st.markdown(f"**📄 Paper B:** `{paper_b.id}` — *{title_b}*")
                st.markdown(
                    f"""
                    <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(129, 140, 248, 0.25); border-radius: 8px; padding: 12px; margin-bottom: 12px; font-size: 0.9rem; line-height: 1.6; color: #F1F5F9;">
                        {sum_b}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                ev_b_list = d_item.paper_evidence.get(paper_b.id, [])
                if ev_b_list:
                    with st.expander(f"📍 Paper B Evidence ({len(ev_b_list)} chunks)", expanded=False):
                        for ev in ev_b_list:
                            st.markdown(
                                f"**Section:** `{ev.normalized_section}` (`{ev.original_heading}`) | "
                                f"**Page:** {ev.page_start}–{ev.page_end} | "
                                f"**Score:** `{ev.score:.4f}` | **Chunk:** `[{ev.chunk_id}]`"
                            )
                            st.markdown(
                                f"<div style='background-color: #0F172A; border-left: 3px solid #818CF8; padding: 8px 12px; margin-bottom: 8px; font-size: 0.85rem; max-height: 200px; overflow-y: auto;'>"
                                f"{ev.text}"
                                f"</div>",
                                unsafe_allow_html=True,
                            )
                else:
                    st.caption("No direct chunks retrieved for Paper B in this dimension.")

    st.divider()

    # 4. Summary Matrix Table (Collapsible)
    with st.expander("📊 View Complete 10-Dimension Summary Table", expanded=True):
        table_data = []
        for dim_name, item in cached_comparison.dimensions.items():
            summary_a = _resolve_paper_summary(item, paper_a, paper_b, 0)
            summary_b = _resolve_paper_summary(item, paper_b, paper_a, 1)
            table_data.append({
                "Dimension": dim_name,
                f"Paper A ({paper_a.id})": summary_a,
                f"Paper B ({paper_b.id})": summary_b,
                "Synthesis / Contrast": item.synthesis or "Grounded comparison provided in tabs above.",
            })

        df_matrix = pd.DataFrame(table_data)
        st.dataframe(df_matrix, use_container_width=True, hide_index=True)

    st.caption("⚠️ Note: ResearchLens AI never ranks papers as 'better', 'worse', or 'superior'. It reports documented similarities and differences objectively.")

    # 6. Complete Data Export Suite
    st.divider()
    col_exp1, col_exp2 = st.columns(2)

    with col_exp1:
        # Markdown export
        md_export = f"# Multi-Paper Comparison: {title_a} vs. {title_b}\n\n"
        md_export += f"- Paper A: `{paper_a.id}` ({title_a})\n"
        md_export += f"- Paper B: `{paper_b.id}` ({title_b})\n"
        md_export += f"- Model: `{cached_comparison.model_used or 'gpt-4.1-mini'}`\n"
        md_export += f"- Latency: `{cached_comparison.execution_latency_ms:.1f} ms`\n\n"

        if cached_comparison.custom_query_answer:
            md_export += f"## Custom Focus Analysis\n**Query:** {custom_query}\n\n{cached_comparison.custom_query_answer}\n\n---\n\n"

        md_export += "## 10-Dimension Comparative Analysis\n\n"
        for d_name, d_item in cached_comparison.dimensions.items():
            md_export += f"### {d_name}\n"
            md_export += f"- **{paper_a.id}:** {d_item.paper_summaries.get(paper_a.id, 'N/A')}\n"
            md_export += f"- **{paper_b.id}:** {d_item.paper_summaries.get(paper_b.id, 'N/A')}\n"
            md_export += f"- **Synthesis:** {d_item.synthesis}\n\n"

        md_export += "## Similarities\n\n"
        for s in cached_comparison.similarities:
            md_export += f"- **{s.topic}:** {s.description}\n"
            if s.paper_a_claim:
                md_export += f"  - {paper_a.id}: {s.paper_a_claim}\n"
            if s.paper_b_claim:
                md_export += f"  - {paper_b.id}: {s.paper_b_claim}\n"

        md_export += "\n## Differences\n\n"
        for d in cached_comparison.differences:
            md_export += f"- **{d.topic}:** {d.description}\n"
            if d.paper_a_claim:
                md_export += f"  - {paper_a.id}: {d.paper_a_claim}\n"
            if d.paper_b_claim:
                md_export += f"  - {paper_b.id}: {d.paper_b_claim}\n"

        st.download_button(
            "📥 Download Comparison as Markdown",
            data=md_export,
            file_name=f"comparison_{paper_a.id}_{paper_b.id}.md",
            mime="text/markdown",
            use_container_width=True,
        )

    with col_exp2:
        # JSON export
        json_export = json.dumps(cached_comparison.model_dump(), indent=2)
        st.download_button(
            "📥 Download Comparison as JSON",
            data=json_export,
            file_name=f"comparison_{paper_a.id}_{paper_b.id}.json",
            mime="application/json",
            use_container_width=True,
        )
