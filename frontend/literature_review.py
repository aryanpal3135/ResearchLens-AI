"""
Literature Review Generator Page for ResearchLens AI (Phase 8).
Provides an evidence-grounded academic literature review interface adhering to
the formal 11-section academic synthesis framework, dynamic theme emergence,
categorical claim classifications, and verifiable chunk-level provenance.
"""

import json
import streamlit as st

from frontend.components import render_header, render_info_card
from services.foundry_agent import ResearchLensAgent
from services.retrieval import HybridRetrievalEngine
from services.embeddings import get_embedding_service
from services.vector_store import get_vector_store
from models.literature_review import LiteratureReview, LiteratureReviewSection, LiteratureReviewTheme, CLAIM_TYPE_LABELS
from services.i18n import t


def render_claim_type_badge(claim_type: str) -> str:
    """Returns styled HTML badge for categorical claim classification."""
    colors = {
        "DOCUMENTED": ("#10B981", "rgba(16, 185, 129, 0.15)"),
        "SYNTHESIS": ("#8B5CF6", "rgba(139, 92, 246, 0.15)"),
        "INFERENCE": ("#0EA5E9", "rgba(14, 165, 233, 0.15)"),
        "INSUFFICIENT_EVIDENCE": ("#EF4444", "rgba(239, 68, 68, 0.15)"),
    }
    color, bg = colors.get(claim_type, ("#94A3B8", "rgba(148, 163, 184, 0.15)"))
    label = CLAIM_TYPE_LABELS.get(claim_type, claim_type)
    return (
        f'<span style="background: {bg}; color: {color}; border: 1px solid {color}; '
        f'padding: 3px 8px; border-radius: 4px; font-size: 11px; font-weight: 600;">'
        f'🏷️ {label}</span>'
    )


def render_evidence_expander(sec_title: str, evidence_items: list, selected_titles: dict):
    """Renders verifiable evidence chunks with chunk_id, page, section, and text."""
    if not evidence_items:
        return

    with st.expander(f"🔎 Supporting Evidence & Provenance ({len(evidence_items)} chunks) — {sec_title}", expanded=False):
        for ev in evidence_items:
            pid = getattr(ev, "paper_id", "")
            p_title = selected_titles.get(pid, pid)
            citation = getattr(ev, "citation_label", f"[{pid}]")
            page = getattr(ev, "page_start", "?")
            sec = getattr(ev, "normalized_section", "Section")
            score = getattr(ev, "score", 0.0)
            chunk_id = getattr(ev, "chunk_id", "")
            text = getattr(ev, "text", "")

            st.markdown(
                f"""
                <div style="background-color: #0F172A; border-left: 3px solid #38BDF8; padding: 8px 12px; margin-bottom: 8px; border-radius: 0 4px 4px 0;">
                    <div style="display: flex; justify-content: space-between; font-size: 12px; color: #94A3B8; margin-bottom: 4px;">
                        <span><strong>{citation}</strong> &bull; Paper: <code>{pid}</code> ({p_title})</span>
                        <span>Page {page} &bull; §{sec} &bull; Score: {score:.4f}</span>
                    </div>
                    <div style="color: #64748B; font-size: 11px; margin-bottom: 4px;">Chunk ID: <code>{chunk_id}</code></div>
                    <div style="color: #CBD5E1; font-size: 13px; line-height: 1.5; font-family: monospace; white-space: pre-wrap;">
{text}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_literature_review_page():
    """Renders the comprehensive Phase 8 Literature Review Generator page."""
    render_header(
        title=t("review_title", "📚 Evidence-Grounded Literature Review"),
        subtitle=t("review_subtitle", "Scholarly cross-paper synthesis with dynamic theme emergence, categorical claim classification, and verifiable provenance."),
        badge="Phase 8: Literature Review",
    )

    papers = st.session_state.get("uploaded_papers", [])

    if not papers:
        st.warning("⚠️ No research papers uploaded yet. Please upload papers on the **Upload Papers** page first.")
        render_info_card(
            title="Scholarly 11-Section Framework",
            description=(
                "ResearchLens AI automatically structures reviews into 11 academic sections:\n"
                "1. Introduction & Scope | 2. Methodological Synthesis | 3. Findings & Evidence | 4. Major Themes\n"
                "5. Agreements, Divergences & Contradictions | 6. Limitations | 7. Research Gaps | 8. Future Directions\n"
                "9. Conclusion | 10. Evidence Provenance | 11. Reference Sources\n\n"
                "Every claim is grounded in retrieved evidence with exact citation tags: `[Paper, Page, Section]`."
            )
        )
        return

    # ----------------------------------------------------------------------
    # 1. Paper Selection & Configuration
    # ----------------------------------------------------------------------
    st.markdown("### 1. Select Target Papers & Review Focus")
    paper_options = {p.id: f"[{p.id}] {p.metadata.title or p.filename} ({len(p.chunks)} chunks)" for p in papers}

    col_sel1, col_sel2 = st.columns([4, 1])
    with col_sel2:
        st.write("")
        st.write("")
        if st.button("Select All", use_container_width=True, key="lit_sel_all"):
            st.session_state["lit_selected_paper_ids"] = list(paper_options.keys())
        if st.button("Clear", use_container_width=True, key="lit_sel_clear"):
            st.session_state["lit_selected_paper_ids"] = []

    default_selected = st.session_state.get("lit_selected_paper_ids", list(paper_options.keys()))

    with col_sel1:
        selected_paper_ids = st.multiselect(
            "Select Papers for Literature Synthesis:",
            options=list(paper_options.keys()),
            format_func=lambda pid: paper_options[pid],
            default=default_selected,
            help="Select one paper for single-paper literature synthesis or multiple papers for comparative cross-synthesis.",
            key="lit_multiselect"
        )
        st.session_state["lit_selected_paper_ids"] = selected_paper_ids

    # Review Focus / Custom Question
    col_q, col_style = st.columns([3, 2])
    with col_q:
        custom_query = st.text_area(
            "Review Focus / Research Question (Optional):",
            value=st.session_state.get("lit_custom_query", ""),
            placeholder="e.g., How do these papers address computational efficiency in Transformer architectures?",
            height=75,
            help="Directs evidence retrieval and thematic synthesis toward a specific academic research question.",
            key="lit_query_input"
        )

    with col_style:
        review_style = st.selectbox(
            "Review Scope & Style:",
            options=[
                "Comprehensive Scholarly Review",
                "Methodological & Architectural Synthesis",
                "Empirical Findings & Benchmark Comparison",
                "Thematic Research Gap Synthesis",
            ],
            index=0,
            help="Directs the depth and synthesis style of the generated review."
        )

    # Quick Suggestion Chips
    st.caption("💡 Quick Suggestions for Review Focus:")
    chip_col1, chip_col2, chip_col3 = st.columns(3)
    with chip_col1:
        if st.button("Transformer Language Models & Differences", use_container_width=True, key="chip_1"):
            st.session_state["lit_custom_query"] = "How do these two papers contribute to the development of Transformer-based language models, and what methodological and evaluation differences are evident between them?"
            st.rerun()
    with chip_col2:
        if st.button("Limitations & Unresolved Problems", use_container_width=True, key="chip_2"):
            st.session_state["lit_custom_query"] = "What limitations and unresolved research problems emerge across these two papers?"
            st.rerun()
    with chip_col3:
        if st.button("Computational Efficiency & Scalability", use_container_width=True, key="chip_3"):
            st.session_state["lit_custom_query"] = "Focus specifically on computational efficiency and scalability in these papers."
            st.rerun()

    st.divider()

    # ----------------------------------------------------------------------
    # 2. Execution Action
    # ----------------------------------------------------------------------
    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        generate_btn = st.button(t("review_btn", "📚 Generate Literature Review"), type="primary", use_container_width=True)

    with col_info:
        st.markdown(
            f"<div style='padding-top: 8px; color: #94A3B8; font-size: 13px;'>"
            f"Mode: <strong>{'Multi-Paper Synthesis' if len(selected_paper_ids) > 1 else 'Single-Paper Literature Synthesis'}</strong> &bull; "
            f"Selected: <strong>{len(selected_paper_ids)}</strong> papers &bull; "
            f"Engine: <code>text-embedding-3-large</code> + <code>Foundry Agent: researchmate-gpt4-1-mini (v1)</code>"
            f"</div>",
            unsafe_allow_html=True
        )

    if generate_btn:
        if not selected_paper_ids:
            st.error("Please select at least one paper for literature review generation.")
            return

        target_papers = [p for p in papers if p.id in selected_paper_ids]

        with st.spinner("Synthesizing literature with Microsoft Foundry Agent (researchmate-gpt4-1-mini)..."):
            try:
                emb_service = get_embedding_service()
                vector_store = get_vector_store()
                engine = HybridRetrievalEngine(embedding_service=emb_service, vector_store=vector_store)

                for p in target_papers:
                    if p.id not in [getattr(c, "paper_id", "") for c in engine.indexed_chunks]:
                        engine.index_paper(p)

                agent = ResearchLensAgent()

                # Reuse Phase 6 cached gaps if available in session state
                cached_gap_res = st.session_state.get("research_gap_result")
                cached_gaps = None
                if cached_gap_res and getattr(cached_gap_res, "gaps", None):
                    cached_gaps = [g for g in cached_gap_res.gaps if any(pid in selected_paper_ids for pid in g.supporting_papers)]

                # Reuse Phase 7 cached future directions if available
                cached_q_res = st.session_state.get("research_questions_result")
                cached_dirs = None
                if cached_q_res and getattr(cached_q_res, "future_directions", None):
                    cached_dirs = cached_q_res.future_directions

                result = agent.generate_literature_review(
                    papers=target_papers,
                    engine=engine,
                    custom_query=custom_query,
                    review_style=review_style,
                    gaps=cached_gaps,
                    future_directions=cached_dirs,
                )

                st.session_state["literature_review_result"] = result
                st.session_state["lit_active_query"] = custom_query
                st.success(f"✅ Literature Review successfully generated in {result.execution_latency_ms:.1f} ms!")
            except Exception as e:
                st.error(f"Literature review generation failed: {str(e)}")
                return

    # ----------------------------------------------------------------------
    # 3. Render Review Results
    # ----------------------------------------------------------------------
    result: LiteratureReview = st.session_state.get("literature_review_result")

    if not result:
        st.info("Select papers and click **Generate Literature Review** above to produce a publication-grade academic synthesis.")
        return

    st.markdown("---")

    # Overview & Metrics Card
    st.markdown(f"## 📖 {result.title}")
    if result.review_question:
        st.markdown(f"**Review Focus:** *{result.review_question}* &bull; **Scope:** `{result.scope_style}`")

    # Metrics row
    m_col1, m_col2, m_col3, m_col4, m_col5, m_col6 = st.columns(6)
    with m_col1:
        st.metric("Analyzed Papers", len(result.selected_paper_ids))
    with m_col2:
        st.metric("Retrieved Chunks", len(result.all_evidence_items))
    with m_col3:
        st.metric("Major Themes", len(result.themes))
    with m_col4:
        st.metric("Gaps Connected", len(result.linked_gaps))
    with m_col5:
        st.metric("Future Directions", len(result.future_directions))
    with m_col6:
        st.metric("Total Latency", f"{result.execution_latency_ms:.0f} ms")

    st.markdown(
        f"""
        <div style="display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 16px; align-items: center;">
            <span style="background: #1E293B; color: #38BDF8; border: 1px solid #0284C7; padding: 3px 8px; border-radius: 4px; font-size: 12px; font-weight: 600;">
                🧠 Agent: Microsoft Foundry — {result.model or 'researchmate-gpt4-1-mini'}
            </span>
            <span style="background: #1E293B; color: #34D399; border: 1px solid #059669; padding: 3px 8px; border-radius: 4px; font-size: 12px; font-weight: 600;">
                ⚡ Retrieval Latency: {result.retrieval_latency_ms:.1f} ms
            </span>
            <span style="background: #1E293B; color: #A78BFA; border: 1px solid #8B5CF6; padding: 3px 8px; border-radius: 4px; font-size: 12px; font-weight: 600;">
                🧠 Generation Latency: {result.generation_latency_ms:.1f} ms
            </span>
            <span style="background: #1E293B; color: #F59E0B; border: 1px solid #D97706; padding: 3px 8px; border-radius: 4px; font-size: 12px; font-weight: 600;">
                📑 Embedded Citations: {len(result.references)} unique chunks
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ----------------------------------------------------------------------
    # 4. Formal Academic Sections Tabs
    # ----------------------------------------------------------------------
    (
        tab_intro,
        tab_themes,
        tab_method,
        tab_findings,
        tab_agreements,
        tab_limitations,
        tab_gaps,
        tab_future,
        tab_conclusion,
        tab_evidence,
    ) = st.tabs([
        "📖 1. Intro & Scope",
        f"🏷️ 2. Themes ({len(result.themes)})",
        "🔬 3. Methodology",
        "📊 4. Findings",
        "⚖️ 5. Agreements & Divergences",
        "⚠️ 6. Limitations",
        f"🔍 7. Research Gaps ({len(result.linked_gaps)})",
        f"🔭 8. Future Directions ({len(result.future_directions)})",
        "🎯 9. Conclusion",
        f"📑 10. References & Provenance ({len(result.references)})",
    ])

    # 1. Introduction
    with tab_intro:
        st.markdown(f"### {result.introduction.title}")
        st.markdown(
            f"<div style='margin-bottom: 10px;'>"
            f"{render_claim_type_badge(result.introduction.claim_type)} &nbsp; "
            f"<span style='color: #94A3B8; font-size: 12px;'>Supporting Papers: {' '.join([f'<code>{pid}</code>' for pid in result.introduction.supporting_papers])}</span>"
            f"</div>",
            unsafe_allow_html=True
        )
        st.markdown(result.introduction.content)
        render_evidence_expander(result.introduction.title, result.introduction.evidence_items, result.selected_paper_titles)

    # 2. Major Themes
    with tab_themes:
        st.markdown("### 🏷️ Emergent Research Themes")
        st.caption("Themes derived directly from the retrieved literature passages without hardcoded preconceptions.")

        if not result.themes:
            st.info("No emergent themes identified from the retrieved passages.")
        else:
            for idx, th in enumerate(result.themes):
                support_color = "#10B981" if "high" in th.support_level.lower() else ("#F59E0B" if "moderate" in th.support_level.lower() else "#64748B")
                with st.expander(f"✨ [{th.theme_id}] {th.title} — ({th.support_level})", expanded=(idx < 2)):
                    st.markdown(
                        f"""
                        <div style="display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 8px;">
                            <span style="background: rgba(16,185,129,0.15); color: {support_color}; border: 1px solid {support_color}; padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: 600;">
                                📊 {th.support_level}
                            </span>
                            <span style="background: #1E293B; color: #38BDF8; border: 1px solid #0284C7; padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: 600;">
                                📚 Supporting Papers: {', '.join([f'`{pid}`' for pid in th.supporting_papers])}
                            </span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    if th.description:
                        st.markdown(f"**Thematic Scope:** {th.description}")
                    st.markdown(f"**Cross-Paper Synthesis:**\n{th.synthesis}")
                    render_evidence_expander(th.title, th.evidence_items, result.selected_paper_titles)

    # 3. Methodological Synthesis
    with tab_method:
        st.markdown(f"### {result.methodology_synthesis.title}")
        st.markdown(
            f"<div style='margin-bottom: 10px;'>"
            f"{render_claim_type_badge(result.methodology_synthesis.claim_type)} &nbsp; "
            f"<span style='color: #94A3B8; font-size: 12px;'>Supporting Papers: {' '.join([f'<code>{pid}</code>' for pid in result.methodology_synthesis.supporting_papers])}</span>"
            f"</div>",
            unsafe_allow_html=True
        )
        st.markdown(result.methodology_synthesis.content)
        render_evidence_expander(result.methodology_synthesis.title, result.methodology_synthesis.evidence_items, result.selected_paper_titles)

    # 4. Findings & Evidence
    with tab_findings:
        st.markdown(f"### {result.findings_synthesis.title}")
        st.markdown(
            f"<div style='margin-bottom: 10px;'>"
            f"{render_claim_type_badge(result.findings_synthesis.claim_type)} &nbsp; "
            f"<span style='color: #94A3B8; font-size: 12px;'>Supporting Papers: {' '.join([f'<code>{pid}</code>' for pid in result.findings_synthesis.supporting_papers])}</span>"
            f"</div>",
            unsafe_allow_html=True
        )
        st.markdown(result.findings_synthesis.content)
        render_evidence_expander(result.findings_synthesis.title, result.findings_synthesis.evidence_items, result.selected_paper_titles)

    # 5. Agreements & Divergences
    with tab_agreements:
        st.markdown(f"### {result.agreements_differences.title}")
        st.markdown(
            f"<div style='margin-bottom: 10px;'>"
            f"{render_claim_type_badge(result.agreements_differences.claim_type)} &nbsp; "
            f"<span style='color: #94A3B8; font-size: 12px;'>Supporting Papers: {' '.join([f'<code>{pid}</code>' for pid in result.agreements_differences.supporting_papers])}</span>"
            f"</div>",
            unsafe_allow_html=True
        )
        st.markdown(result.agreements_differences.content)
        render_evidence_expander(result.agreements_differences.title, result.agreements_differences.evidence_items, result.selected_paper_titles)

    # 6. Limitations
    with tab_limitations:
        st.markdown(f"### {result.limitations.title}")
        st.markdown(
            f"<div style='margin-bottom: 10px;'>"
            f"{render_claim_type_badge(result.limitations.claim_type)} &nbsp; "
            f"<span style='color: #94A3B8; font-size: 12px;'>Supporting Papers: {' '.join([f'<code>{pid}</code>' for pid in result.limitations.supporting_papers])}</span>"
            f"</div>",
            unsafe_allow_html=True
        )
        st.markdown(result.limitations.content)
        render_evidence_expander(result.limitations.title, result.limitations.evidence_items, result.selected_paper_titles)

    # 7. Research Gaps
    with tab_gaps:
        st.markdown(f"### {result.research_gaps.title}")
        st.markdown(
            f"<div style='margin-bottom: 10px;'>"
            f"{render_claim_type_badge(result.research_gaps.claim_type)} &nbsp; "
            f"<span style='color: #94A3B8; font-size: 12px;'>Supporting Papers: {' '.join([f'<code>{pid}</code>' for pid in result.research_gaps.supporting_papers])}</span>"
            f"</div>",
            unsafe_allow_html=True
        )
        st.markdown(result.research_gaps.content)

        if result.linked_gaps:
            st.markdown("#### 🔗 Linked Phase 6 Research Gaps")
            for g in result.linked_gaps:
                origin_color = "#10B981" if g.evidence_type == "explicit" else ("#8B5CF6" if g.evidence_type == "cross_paper" else "#0EA5E9")
                with st.expander(f"📌 [{g.gap_id}] {g.title} ({g.category})", expanded=False):
                    st.markdown(
                        f"""
                        <div style="display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 8px;">
                            <span style="background: rgba({ '16,185,129,0.15' if g.evidence_type == 'explicit' else ('139,92,246,0.15' if g.evidence_type == 'cross_paper' else '14,165,233,0.15') }); color: {origin_color}; border: 1px solid {origin_color}; padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: 600;">
                                {g.evidence_type_label}
                            </span>
                            <span style="background: #1E293B; color: #38BDF8; border: 1px solid #0284C7; padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: 600;">
                                🏷️ {g.category}
                            </span>
                            <span style="background: #1E293B; color: #94A3B8; border: 1px solid #334155; padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: 600;">
                                📚 Papers: {', '.join([f'`{pid}`' for pid in g.supporting_papers])}
                            </span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    st.markdown(f"**Description:** {g.description}")
                    render_evidence_expander(g.title, g.evidence_items, result.selected_paper_titles)

    # 8. Future Directions
    with tab_future:
        st.markdown("### 🔭 Future Research Directions")
        st.caption(
            "Strictly partitioned by epistemic source: "
            "**Author-Documented Future Work** (`is_suggestion=False`) vs **AI-Formulated Pathways** (`is_suggestion=True`) grounded in verified gaps."
        )

        author_dirs = [fd for fd in result.future_directions if not fd.is_suggestion]
        ai_dirs = [fd for fd in result.future_directions if fd.is_suggestion]

        st.markdown(f"#### 📌 Author-Documented Future Work ({len(author_dirs)})")
        if not author_dirs:
            st.info("No explicit future work statements were extracted from the authors' concluding sections.")
        else:
            for f_idx, fd in enumerate(author_dirs):
                with st.expander(f"📌 [{fd.direction_id}] {fd.title} — ({fd.direction_type})", expanded=(f_idx < 2)):
                    st.markdown(
                        f"""
                        <div style="display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 8px;">
                            <span style="background: rgba(16,185,129,0.15); color: #10B981; border: 1px solid #10B981; padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: 600;">
                                📖 Author-Stated
                            </span>
                            <span style="background: #1E293B; color: #94A3B8; border: 1px solid #334155; padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: 600;">
                                🏷️ {fd.direction_type}
                            </span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    st.markdown(f"**Author-Stated Direction:** {fd.description}")
                    if fd.expected_contribution:
                        st.markdown(f"**Documented Context:** {fd.expected_contribution}")

        st.markdown("---")
        st.markdown(f"#### 💡 AI-Formulated Exploratory Pathways ({len(ai_dirs)})")
        if not ai_dirs:
            st.info("No separate exploratory pathways generated.")
        else:
            for f_idx, fd in enumerate(ai_dirs):
                with st.expander(f"✨ [{fd.direction_id}] {fd.title} — ({fd.direction_type})", expanded=(f_idx < 2)):
                    st.markdown(
                        f"""
                        <div style="display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 8px;">
                            <span style="background: rgba(139,92,246,0.15); color: #A78BFA; border: 1px solid #8B5CF6; padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: 600;">
                                🤖 AI-Generated Suggestion
                            </span>
                            <span style="background: #1E293B; color: #94A3B8; border: 1px solid #334155; padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: 600;">
                                🏷️ {fd.direction_type}
                            </span>
                            <span style="background: #1E293B; color: #38BDF8; border: 1px solid #0284C7; padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: 600;">
                                🔗 Addressed Gap: <code>{fd.gap_addressed}</code>
                            </span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    st.markdown(f"**Proposed Pathway:** {fd.description}")
                    if fd.suggested_methodology:
                        st.markdown(f"**Suggested Methodology:** {fd.suggested_methodology}")
                    if fd.suggested_dataset_or_setting:
                        st.markdown(f"**Suggested Dataset or Benchmark:** {fd.suggested_dataset_or_setting}")
                    if fd.expected_contribution:
                        st.markdown(f"**Expected Academic Contribution:** {fd.expected_contribution}")

    # 9. Conclusion
    with tab_conclusion:
        st.markdown(f"### {result.conclusion.title}")
        st.markdown(
            f"<div style='margin-bottom: 10px;'>"
            f"{render_claim_type_badge(result.conclusion.claim_type)} &nbsp; "
            f"<span style='color: #94A3B8; font-size: 12px;'>Supporting Papers: {' '.join([f'<code>{pid}</code>' for pid in result.conclusion.supporting_papers])}</span>"
            f"</div>",
            unsafe_allow_html=True
        )
        st.markdown(result.conclusion.content)
        render_evidence_expander(result.conclusion.title, result.conclusion.evidence_items, result.selected_paper_titles)

    # 10. References & Provenance
    with tab_evidence:
        st.markdown("### 📑 Complete Evidence Provenance & References")
        st.caption(
            "Every cited chunk is drawn from the uploaded PDFs via Phase 3 Hybrid RAG. "
            "No citations are fabricated."
        )

        ref_search = st.text_input("Filter References by Paper ID, Chunk ID, or Keyword:", value="", key="ref_search_filter")

        filtered_refs = result.references
        if ref_search:
            s_low = ref_search.lower()
            filtered_refs = [
                r for r in result.references
                if s_low in r["paper_id"].lower()
                or s_low in r["chunk_id"].lower()
                or s_low in r["section"].lower()
                or s_low in r["snippet"].lower()
            ]

        # Complete-Data rule compliance: Show All / Show Fewer
        show_all = st.checkbox("Show All References (Unlimited)", value=True, key="lit_show_all_refs")
        display_refs = filtered_refs if show_all else filtered_refs[:10]

        st.markdown(f"**Showing {len(display_refs)} of {len(filtered_refs)} Reference Passages**")

        for r in display_refs:
            st.markdown(
                f"""
                <div style="background-color: #0F172A; border-left: 3px solid #6366F1; padding: 10px 14px; margin-bottom: 10px; border-radius: 0 4px 4px 0;">
                    <div style="display: flex; justify-content: space-between; font-size: 12px; color: #94A3B8; margin-bottom: 4px;">
                        <span><strong>{r['citation_label']}</strong> &bull; <code>{r['paper_id']}</code> ({r['paper_title']})</span>
                        <span>Page {r['page']} &bull; §{r['section']}</span>
                    </div>
                    <div style="color: #64748B; font-size: 11px; margin-bottom: 4px;">Chunk ID: <code>{r['chunk_id']}</code></div>
                    <div style="color: #E2E8F0; font-size: 13px; line-height: 1.5; font-family: monospace;">
{r['snippet']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    # ----------------------------------------------------------------------
    # 4.5 Guardrail Audit Trail (if any claims rejected)
    # ----------------------------------------------------------------------
    if getattr(result, "rejected_claims", None):
        with st.expander(f"🛡️ Guardrail Audit Trail ({len(result.rejected_claims)} Filtered Items)", expanded=False):
            st.caption(
                "ResearchLens AI applies deterministic grounding and leakage filters to prevent "
                "host architecture leakage, fabricated percentages, unverified citations, or cross-paper attribution mismatches."
            )
            for r_idx, rc in enumerate(result.rejected_claims):
                st.markdown(
                    f"""
                    <div style="background-color: #1E1B4B; border-left: 3px solid #EF4444; padding: 8px 12px; margin-bottom: 8px; border-radius: 0 4px 4px 0; font-size: 13px;">
                        <span style="color: #F87171; font-weight: 600;">⚠️ [{rc.get('reason', 'FILTERED')}] in {rc.get('section', 'General')}:</span>
                        <div style="color: #CBD5E1; margin-top: 4px; font-style: italic;">"{rc.get('sentence', '')}"</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    st.markdown("---")

    # ----------------------------------------------------------------------
    # 5. Export Section
    # ----------------------------------------------------------------------
    st.markdown("### 📥 Export Complete Literature Review")
    exp_col1, exp_col2 = st.columns(2)

    with exp_col1:
        # Markdown Export
        md_lines = [
            f"# {result.title}\n",
            f"**Review Question:** {result.review_question or 'Comprehensive Scholarly Review'}",
            f"**Analyzed Papers:** {', '.join(result.selected_paper_ids)}",
            f"**Agent Deployment:** Microsoft Foundry ({result.model or 'researchmate-gpt4-1-mini'})",
            f"**Embedding Model:** Azure OpenAI text-embedding-3-large (3072d)",
            f"**Execution Latency:** {result.execution_latency_ms:.1f} ms\n",
            f"## Summary Metrics",
            f"- Total Papers: {len(result.selected_paper_ids)}",
            f"- Retrieved Chunks: {len(result.all_evidence_items)}",
            f"- Major Themes: {len(result.themes)}",
            f"- Connected Gaps: {len(result.linked_gaps)}",
            f"- Future Directions: {len(result.future_directions)}\n",
            f"## 1. Introduction & Research Scope",
            f"*{render_claim_type_badge(result.introduction.claim_type)}*",
            f"{result.introduction.content}\n",
            f"## 2. Major Themes\n",
        ]

        for th in result.themes:
            md_lines.append(f"### [{th.theme_id}] {th.title} ({th.support_level})")
            md_lines.append(f"- **Supporting Papers:** {', '.join(th.supporting_papers)}")
            if th.description:
                md_lines.append(f"- **Thematic Scope:** {th.description}")
            md_lines.append(f"{th.synthesis}\n")

        md_lines.extend([
            f"## 3. Methodological Synthesis",
            f"{result.methodology_synthesis.content}\n",
            f"## 4. Findings & Evidence Synthesis",
            f"{result.findings_synthesis.content}\n",
            f"## 5. Agreements, Divergences & Contradictions",
            f"{result.agreements_differences.content}\n",
            f"## 6. Limitations in the Reviewed Literature",
            f"{result.limitations.content}\n",
            f"## 7. Synthesized Research Gaps",
            f"{result.research_gaps.content}\n",
        ])

        if result.future_directions:
            md_lines.append("## 8. Proposed Future Research Directions (AI Suggestions)\n")
            for fd in result.future_directions:
                md_lines.append(f"### [{fd.direction_id}] {fd.title} ({fd.direction_type})")
                md_lines.append(f"- **Addressed Gap:** {fd.gap_addressed}")
                md_lines.append(f"- **Proposed Direction:** {fd.description}")
                if fd.suggested_methodology:
                    md_lines.append(f"- **Suggested Methodology:** {fd.suggested_methodology}")
                md_lines.append("\n")

        md_lines.extend([
            f"## 9. Conclusion & Research Horizons",
            f"{result.conclusion.content}\n",
            f"## 10. References & Evidence Sources\n",
        ])

        for r in result.references:
            md_lines.append(f"- **{r['citation_label']}**: `{r['paper_id']}` ({r['paper_title']}), Page {r['page']}, §{r['section']} (Chunk: `{r['chunk_id']}`)")
            md_lines.append(f"  > {r['snippet']}\n")

        st.download_button(
            label="📄 Download Markdown Report",
            data="\n".join(md_lines),
            file_name="literature_review_report.md",
            mime="text/markdown",
            use_container_width=True,
        )

    with exp_col2:
        # JSON Export: Complete Structured Object
        json_export_data = {
            "review_id": result.review_id,
            "title": result.title,
            "review_question": result.review_question,
            "selected_paper_ids": result.selected_paper_ids,
            "selected_paper_titles": result.selected_paper_titles,
            "scope_style": result.scope_style,
            "model": result.model,
            "retrieval_latency_ms": result.retrieval_latency_ms,
            "generation_latency_ms": result.generation_latency_ms,
            "execution_latency_ms": result.execution_latency_ms,
            "summary_counts": result.summary_counts,
            "introduction": result.introduction.model_dump(),
            "themes": [t.model_dump() for t in result.themes],
            "methodology_synthesis": result.methodology_synthesis.model_dump(),
            "findings_synthesis": result.findings_synthesis.model_dump(),
            "agreements_differences": result.agreements_differences.model_dump(),
            "limitations": result.limitations.model_dump(),
            "research_gaps": result.research_gaps.model_dump(),
            "future_directions": [fd.model_dump() for fd in result.future_directions],
            "conclusion": result.conclusion.model_dump(),
            "references": result.references,
            "status": result.status,
        }

        st.download_button(
            label="💾 Download Complete JSON Data",
            data=json.dumps(json_export_data, indent=2),
            file_name="literature_review_data.json",
            mime="application/json",
            use_container_width=True,
        )
