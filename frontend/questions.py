"""
Research Question & Future Research Direction Generator Page for ResearchLens AI.
Formulates rigorous, evidence-grounded academic research questions and concrete
future research pathways directly mapped to verified research gaps.
Powered by Microsoft Foundry (gpt-4.1-mini) and Phase 3 Hybrid RAG retrieval.
Strictly enforces:
- Gap -> Question -> Evidence traceability.
- Clear distinction between Author-Inspired and Evidence-Derived questions.
- Separation of evidence-grounded questions from AI-suggested future directions (is_suggestion=True).
- Literature-scoped novelty framing (no field-wide overclaims).
- Active query synchronization.
"""

import json
from typing import List, Optional
import streamlit as st

from frontend.components import render_header, render_info_card
from frontend.retrieval_ui import get_or_create_retrieval_engine, sync_papers_to_retrieval_engine
from models.paper import PaperDocument, EvidenceItem
from models.research_gap import (
    ResearchQuestion,
    FutureDirection,
    ResearchQuestionAnalysisResult,
    ResearchGapAnalysisResult,
    QUESTION_TYPES,
    FUTURE_DIRECTION_TYPES,
)
from services.foundry_agent import ResearchLensAgent
from services.i18n import t

QUESTION_FOCUS_OPTIONS = [
    "Comprehensive Research Question Generation",
    "What methodological and architectural inquiry questions arise from these papers?",
    "What empirical evaluation and benchmarking questions remain open?",
    "What cross-paper synthesis and comparative questions can be formulated?",
    "What scalability, efficiency, and resource constraint questions are indicated?",
    "What domain transfer and generalization questions emerge from the findings?",
    "✏️ Custom Research Goal / Inquiry",
]

QUESTION_FOCUS_QUERY_MAP = {
    "Comprehensive Research Question Generation": "Formulate comprehensive, gap-grounded academic research questions across these papers.",
    "What methodological and architectural inquiry questions arise from these papers?": "What methodological and architectural inquiry questions arise from these papers?",
    "What empirical evaluation and benchmarking questions remain open?": "What empirical evaluation and benchmarking questions remain open?",
    "What cross-paper synthesis and comparative questions can be formulated?": "What cross-paper synthesis and comparative questions can be formulated?",
    "What scalability, efficiency, and resource constraint questions are indicated?": "What scalability, efficiency, and resource constraint questions are indicated?",
    "What domain transfer and generalization questions emerge from the findings?": "What domain transfer and generalization questions emerge from the findings?",
    "✏️ Custom Research Goal / Inquiry": "",
}


def render_questions_page():
    """Renders research questions and future research directions view."""
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
        title=t("questions_title", "❓ Evidence-Grounded Research Questions"),
        subtitle=t("questions_subtitle", "Formulate novel, actionable research questions directly addressing verified gaps in the literature."),
        badge="Phase 7: Research Questions & Future Directions",
    )

    render_info_card(
        title="Evidence Grounding & AI Attribution Notice",
        description=(
            "⚠️ **Academic Integrity Notice:** All research questions are formulated directly from "
            "the empirical gaps and limitations retrieved from the uploaded literature. "
            "**Future Research Directions** are designated as exploratory AI-generated suggestions "
            "(`is_suggestion=True`) to guide new academic projects and theses—they are not claims "
            "authored by the original paper writers."
        ),
        icon="💡",
    )

    papers: List[PaperDocument] = st.session_state.get("uploaded_papers", [])

    if not papers:
        render_info_card(
            title="No Research Papers Uploaded",
            description=(
                "Please upload at least **1 research paper** on the **Upload Papers** page "
                "to enable gap-grounded research question generation."
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
            &nbsp;|&nbsp; Grounding: <span style="color: #38BDF8; font-weight: 600;">Phase 6 Gaps &bull; Verifiable Chunks</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 1. Paper Selection (Single or Multi-Paper)
    st.markdown("#### 1. Select Papers to Analyze")
    paper_options = {p.id: f"[{p.id}] {p.metadata.title or p.filename} ({p.page_count} pp)" for p in papers}

    selected_paper_ids = st.multiselect(
        "Choose 1 or more papers to generate research questions from:",
        options=list(paper_options.keys()),
        default=list(paper_options.keys())[:min(2, len(paper_options))],
        format_func=lambda pid: paper_options[pid],
        key="question_selected_paper_ids",
    )

    if not selected_paper_ids:
        st.warning("⚠️ Please select at least 1 paper to run research question generation.")
        return

    selected_paper_objs = [p for p in papers if p.id in selected_paper_ids]
    st.caption(
        f"Selected **{len(selected_paper_objs)} paper(s)**: "
        + ", ".join([f"`{p.id}` ({p.metadata.title or p.filename})" for p in selected_paper_objs])
    )

    # 2. Focus Query / Research Question
    st.markdown(f"#### {t('questions_focus_heading', '2. Question Generation Focus')}")
    focus_selection = st.selectbox(
        "Select question focus:",
        options=QUESTION_FOCUS_OPTIONS,
        index=0,
        key="question_preset_choice",
        help="Select a research inquiry or write a custom goal.",
    )
    if focus_selection == "✏️ Custom Research Goal / Inquiry":
        custom_query_val = st.text_area(
            "Enter custom research goal or question (full question shown):",
            value=st.session_state.get("questions_custom_input", ""),
            placeholder="Type your complete research question or thesis inquiry here... e.g. What novel architectural improvements can alleviate memory bottlenecks in multi-head attention?",
            height=90,
            key="questions_custom_input",
            help="Write your complete research question. The full question remains visible without scrolling out of view.",
        )
        active_query_str = custom_query_val.strip() or "Custom research question focus"
    else:
        active_query_str = QUESTION_FOCUS_QUERY_MAP.get(focus_selection, "") or focus_selection

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

    st.session_state["questions_active_query"] = active_query_str

    # 3. Optional: Existing Gaps or Fresh Detection
    existing_gap_result: Optional[ResearchGapAnalysisResult] = st.session_state.get("gap_detection_result")
    use_existing_gaps = False
    if existing_gap_result and existing_gap_result.gaps:
        # Check if analyzed papers match
        matching = set(existing_gap_result.selected_paper_ids) == set(selected_paper_ids)
        if matching:
            is_same_query = (
                existing_gap_result.query.strip().lower() == active_query_str.strip().lower()
            )
            use_existing_gaps = st.checkbox(
                f"🔗 Use existing detected gaps ({len(existing_gap_result.gaps)} gaps from Phase 6 cache)",
                value=is_same_query,
                key="use_cached_gaps_checkbox",
                help="Checked by default when reusing gaps from an identical query. Uncheck to detect fresh query-tailored gaps.",
            )

    # 4. Generate Research Questions Button
    if st.button(t("questions_btn", "❓ Generate Research Questions"), type="primary", use_container_width=True):
        with st.spinner("Analyzing literature gaps and generating grounded research questions with Microsoft Foundry (gpt-4.1-mini)..."):
            gaps_to_pass = existing_gap_result.gaps if (use_existing_gaps and existing_gap_result) else None
            result = agent.generate_research_questions(
                papers=selected_paper_objs,
                engine=engine,
                gaps=gaps_to_pass,
                custom_query=active_query_str,
            )
            st.session_state["research_questions_result"] = result

    # 5. Render Results
    result: Optional[ResearchQuestionAnalysisResult] = st.session_state.get("research_questions_result")

    if result is None:
        st.info("💡 Click **Generate Research Questions** above to formulate academic questions and future pathways.")
        return

    if result.status == "error":
        st.error(f"❌ Research Question Generation Error: {result.error_message}")
        return

    if result.status == "insufficient_evidence":
        st.warning(f"⚠️ {result.error_message or 'Insufficient evidence to formulate grounded research questions.'}")
        return

    st.markdown("---")
    st.markdown(f"### 🎯 Formulated Research Questions ({len(result.questions)})")
    st.markdown(f"**Active Query:** `{result.query}`")
    st.caption(
        f"Execution Latency: **{result.execution_latency_ms:.1f} ms** | "
        f"Model: **{result.model_used or 'Microsoft Foundry (gpt-4.1-mini)'}** | "
        f"Underlying Gaps Linked: **{len(result.linked_gap_ids)}** | "
        f"Retrieved Evidence Chunks: **{len(result.all_evidence_items)}**"
    )

    # Summary Overview Metrics Bar
    counts = result.summary_counts
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("Total Questions", counts.get("total_questions", len(result.questions)))
    with m2:
        st.metric("Author-Inspired", counts.get("author_inspired", 0))
    with m3:
        st.metric("Evidence-Derived", counts.get("evidence_derived", 0))
    with m4:
        st.metric("Cross-Paper", counts.get("cross_paper", 0))
    with m5:
        st.metric("Future Directions", counts.get("future_directions", len(result.future_directions)))

    if not result.questions:
        st.info("No grounded research questions could be verified from the identified gaps.")
        return

    # Tabs for Questions vs Future Directions
    tab_questions, tab_future = st.tabs([
        f"🎯 Grounded Research Questions ({len(result.questions)})",
        f"🔭 Future Research Pathways ({len(result.future_directions)})",
    ])

    with tab_questions:
        for idx, q in enumerate(result.questions):
            disp_idx = q.display_index or (idx + 1)
            disp_id = q.display_id or f"rq_{disp_idx:03d}"

            # High-visibility prominent question card
            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%); border: 1px solid #38BDF8; border-left: 5px solid #38BDF8; border-radius: 8px; padding: 18px 20px; margin-bottom: 12px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);">
                    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
                        <span style="color: #38BDF8; font-weight: 700; font-size: 13px; text-transform: uppercase; letter-spacing: 0.5px;">Question {disp_idx} ({disp_id})</span>
                        <span style="background: rgba(56, 189, 248, 0.15); color: #38BDF8; border: 1px solid #38BDF8; padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: 600;">{q.question_type}</span>
                    </div>
                    <div style="font-size: 1.2rem; font-weight: 700; color: #FFFFFF; line-height: 1.5; margin-bottom: 10px;">
                        {q.question}
                    </div>
                    <div style="color: #CBD5E1; font-size: 13.5px; line-height: 1.5; margin-bottom: 6px;">
                        <strong style="color: #38BDF8;">Rationale:</strong> {q.rationale}
                    </div>
                    {f'<div style="color: #A5B4FC; font-size: 13px; margin-top: 4px;"><strong style="color: #818CF8;">Suggested Methodology:</strong> {q.suggested_evaluation}</div>' if q.suggested_evaluation else ''}
                    {f'<div style="color: #94A3B8; font-size: 12px; margin-top: 6px;"><strong>Supporting Papers:</strong> {" ".join([f"`{pid}`" for pid in q.supporting_papers])}</div>' if q.supporting_papers else ''}
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Optional supporting chunks only if user wants evidence verification
            if q.evidence_items:
                with st.expander(f"📑 View Supporting Evidence ({len(q.evidence_items)} chunks) for Question {disp_idx}", expanded=False):
                    for ev in q.evidence_items:
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

    with tab_future:
        st.markdown("#### 🔭 Proposed Future Research Pathways")
        st.caption(
            "These directions are generated as academic research suggestions directly addressing the verified research gaps."
        )

        if not result.future_directions:
            st.info("No separate future research directions generated.")
        else:
            for f_idx, fd in enumerate(result.future_directions):
                fd_disp_idx = fd.display_index or (f_idx + 1)
                fd_disp_id = fd.display_id or f"fd_{fd_disp_idx:03d}"
                st.markdown(
                    f"""
                    <div style="background: linear-gradient(135deg, #1E1B4B 0%, #0F172A 100%); border: 1px solid #6366F1; border-left: 5px solid #818CF8; border-radius: 8px; padding: 18px 20px; margin-bottom: 12px;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                            <span style="color: #A5B4FC; font-weight: 700; font-size: 13px; text-transform: uppercase;">Pathway {fd_disp_idx} ({fd_disp_id})</span>
                            <span style="background: rgba(139,92,246,0.25); color: #C4B5FD; border: 1px solid #8B5CF6; padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: 600;">{fd.direction_type}</span>
                        </div>
                        <div style="font-size: 1.15rem; font-weight: 700; color: #FFFFFF; line-height: 1.4; margin-bottom: 8px;">
                            {fd.title}
                        </div>
                        <div style="color: #CBD5E1; font-size: 13.5px; line-height: 1.5; margin-bottom: 8px;">
                            {fd.description}
                        </div>
                        {f'<div style="color: #38BDF8; font-size: 13px; margin-bottom: 4px;"><strong>Suggested Methodology:</strong> {fd.suggested_methodology}</div>' if fd.suggested_methodology else ''}
                        {f'<div style="color: #34D399; font-size: 13px; margin-bottom: 4px;"><strong>Expected Contribution:</strong> {fd.expected_contribution}</div>' if fd.expected_contribution else ''}
                        <div style="color: #94A3B8; font-size: 12px; margin-top: 6px;"><strong>Addressed Gap:</strong> <code>{fd.gap_addressed}</code></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # 6. Export Section
    st.markdown("---")
    st.markdown("#### 📥 Export Research Questions & Directions")
    exp_col1, exp_col2 = st.columns(2)

    with exp_col1:
        # Markdown Export
        md_lines = [
            f"# ResearchLens AI — Research Question & Future Directions Report",
            f"**Active Query:** {result.query}",
            f"**Analyzed Papers:** {', '.join(result.selected_paper_ids)}",
            f"**Model Deployment:** {result.model_used or 'gpt-4.1-mini'}",
            f"**Execution Latency:** {result.execution_latency_ms:.1f} ms",
            f"**Total Questions:** {len(result.questions)}",
            f"**Total Future Directions:** {len(result.future_directions)}",
            f"\n## Summary Metrics",
            f"- Author-Inspired Questions: {counts.get('author_inspired', 0)}",
            f"- Evidence-Derived Questions: {counts.get('evidence_derived', 0)}",
            f"- Cross-Paper Questions: {counts.get('cross_paper', 0)}",
            f"- Future Directions: {counts.get('future_directions', len(result.future_directions))}",
            f"\n## Grounded Research Questions\n",
        ]
        for q in result.questions:
            disp_id = q.display_id or q.question_id
            id_header = f"[{disp_id}]" if disp_id == q.question_id else f"[{disp_id}] (Internal ID: {q.question_id})"
            md_lines.append(f"### Question {q.display_index} {id_header}: {q.question}")
            md_lines.append(f"- **Type:** {q.question_type}")
            md_lines.append(f"- **Origin:** {q.question_origin_label} (`{q.question_origin}`)")
            md_lines.append(f"- **Evidence Support:** {q.evidence_support}")
            md_lines.append(f"- **Linked Gaps:** {', '.join(q.research_gap_ids)}")
            md_lines.append(f"- **Supporting Papers:** {', '.join(q.supporting_papers)}")
            md_lines.append(f"- **Rationale:** {q.rationale}")
            if q.novelty_basis:
                md_lines.append(f"- **Novelty Basis:** {q.novelty_basis}")
            if q.suggested_evaluation:
                md_lines.append(f"- **Suggested Evaluation:** {q.suggested_evaluation}")
            if q.suggested_dataset_or_setting:
                md_lines.append(f"- **Suggested Dataset/Setting:** {q.suggested_dataset_or_setting}")
            if q.potential_research_direction:
                md_lines.append(f"- **Future Direction (AI Suggestion):** {q.potential_research_direction} ({q.direction_type})")
            if q.evidence_items:
                md_lines.append(f"\n#### Supporting Evidence Chunks ({len(q.evidence_items)}):")
                for it in q.evidence_items:
                    md_lines.append(f"> **{it.citation_label}** (Page {it.page_start}, Score: {it.score:.4f}):\n> {it.text}\n")
            md_lines.append("\n---\n")

        if result.future_directions:
            md_lines.append(f"\n## Proposed Future Research Directions (AI Suggestions)\n")
            for fd in result.future_directions:
                fd_disp_id = fd.display_id or fd.direction_id
                md_lines.append(f"### [{fd_disp_id}] {fd.title}")
                md_lines.append(f"- **Type:** {fd.direction_type}")
                md_lines.append(f"- **Addressed Gap:** {fd.gap_addressed}")
                if fd.linked_display_id or fd.linked_question_id:
                    link_info = fd.linked_display_id or fd.linked_question_id
                    if fd.linked_question_id and fd.linked_question_id != fd.linked_display_id:
                        link_info += f" (Internal: {fd.linked_question_id})"
                    md_lines.append(f"- **Linked Question:** {link_info}")
                md_lines.append(f"- **Description:** {fd.description}")
                if fd.suggested_methodology:
                    md_lines.append(f"- **Suggested Methodology:** {fd.suggested_methodology}")
                if fd.suggested_dataset_or_setting:
                    md_lines.append(f"- **Suggested Dataset/Setting:** {fd.suggested_dataset_or_setting}")
                if fd.expected_contribution:
                    md_lines.append(f"- **Expected Contribution:** {fd.expected_contribution}")
                md_lines.append("\n---\n")

        st.download_button(
            label="📄 Download Markdown Report",
            data="\n".join(md_lines),
            file_name="research_questions_report.md",
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
            "linked_gap_ids": result.linked_gap_ids,
            "rejected_candidate_reasons": result.rejected_candidate_reasons,
            "questions": [
                {
                    "display_index": q.display_index,
                    "display_id": q.display_id,
                    "question_id": q.question_id,
                    "candidate_id": q.candidate_id,
                    "question": q.question,
                    "question_type": q.question_type,
                    "question_origin": q.question_origin,
                    "question_origin_label": q.question_origin_label,
                    "evidence_support": q.evidence_support,
                    "research_gap_ids": q.research_gap_ids,
                    "supporting_papers": q.supporting_papers,
                    "rationale": q.rationale,
                    "novelty_basis": q.novelty_basis,
                    "suggested_evaluation": q.suggested_evaluation,
                    "suggested_dataset_or_setting": q.suggested_dataset_or_setting,
                    "potential_research_direction": q.potential_research_direction,
                    "direction_type": q.direction_type,
                    "citation_labels": q.citation_labels,
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
                        for ev in q.evidence_items
                    ],
                }
                for q in result.questions
            ],
            "future_directions": [
                {
                    "display_index": fd.display_index,
                    "display_id": fd.display_id,
                    "direction_id": fd.direction_id,
                    "title": fd.title,
                    "direction_type": fd.direction_type,
                    "gap_addressed": fd.gap_addressed,
                    "linked_display_id": fd.linked_display_id,
                    "linked_question_id": fd.linked_question_id,
                    "description": fd.description,
                    "suggested_methodology": fd.suggested_methodology,
                    "suggested_dataset_or_setting": fd.suggested_dataset_or_setting,
                    "expected_contribution": fd.expected_contribution,
                    "is_suggestion": fd.is_suggestion,
                }
                for fd in result.future_directions
            ],
        }
        st.download_button(
            label="💾 Download JSON Data",
            data=json.dumps(json_export_data, indent=2),
            file_name="research_questions_data.json",
            mime="application/json",
            use_container_width=True,
        )
