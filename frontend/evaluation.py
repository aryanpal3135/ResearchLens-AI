"""
Evaluation Page for ResearchLens AI.
Displays formal scientific evaluation metrics across Retrieval, Generation, and Extraction.
Strictly adheres to scientific rigor: NO FAKE SCORES; scores are only displayed upon actual benchmark execution.
"""

import streamlit as st
from frontend.components import render_header, render_info_card


def render_evaluation_page():
    """Renders the AI system evaluation suite."""
    render_header(
        title="📊 System Evaluation & Benchmarks",
        subtitle="Rigorously evaluate Retrieval, Grounded Generation, Information Extraction, and Translation."
    )

    render_info_card(
        title="Scientific Evaluation Integrity",
        description=(
            "⚠️ **Academic Standard:** ResearchLens AI never presents synthetic or pre-baked scores. "
            "Metrics are only calculated and displayed after an evaluation run is executed against ground-truth benchmarks."
        )
    )

    tab_eval_overview, tab_metrics_guide, tab_run_eval = st.tabs([
        "📈 Current Evaluation Status",
        "📐 Metric Definitions & Framework",
        "⚙️ Run Benchmark Evaluation"
    ])

    with tab_eval_overview:
        st.subheader("Benchmark Status")
        eval_run_completed = st.session_state.get("evaluation_executed", False)

        if not eval_run_completed:
            st.warning("⚠️ No benchmark evaluation has been run yet for this session. Scores are pending.")
            st.markdown(
                """
                | Metric Category | Target Dimension | Status | Current Score |
                | :--- | :--- | :--- | :--- |
                | **Retrieval** | Precision@K | `Pending Run` | — |
                | **Retrieval** | Recall@K | `Pending Run` | — |
                | **Retrieval** | Mean Reciprocal Rank (MRR) | `Pending Run` | — |
                | **Generation** | Groundedness (Faithfulness) | `Pending Run` | — |
                | **Generation** | Answer Relevance | `Pending Run` | — |
                | **Generation** | Completeness | `Pending Run` | — |
                | **Extraction** | Paper Analysis Accuracy | `Pending Run` | — |
                | **Extraction** | Dataset Extraction Accuracy | `Pending Run` | — |
                | **Extraction** | Method Extraction Accuracy | `Pending Run` | — |
                | **Extraction** | Citation / Evidence Traceability | `Pending Run` | — |
                | **Gaps & Questions** | Research Gap Relevance | `Pending Run` | — |
                | **Translation** | BLEU / COMET Score (4 Languages) | `Pending Run` | — |
                """
            )
        else:
            st.success("Evaluation Report active. (Will be calculated in Phase 12).")

    with tab_metrics_guide:
        st.subheader("Evaluation Methodology")
        st.markdown(
            """
            ### 1. Retrieval Metrics
            - **Precision@K:** Fraction of retrieved document chunks in top-K that contain ground-truth answers.
            - **Recall@K:** Fraction of relevant evidence chunks retrieved across the entire paper.

            ### 2. Grounded Generation
            - **Groundedness Score:** Verification that every sentence in the response is mathematically entailed by retrieved chunks.
            - **Relevance:** Measuring whether the model addressed the user's specific research inquiry.

            ### 3. Extraction & Citation
            - **Evidence Traceability:** Percentage of extracted parameters with verifiable page numbers and sections.
            - **Hallucination Detection:** Flagging any claim lacking source grounding.
            """
        )

    with tab_run_eval:
        st.subheader("Execute Evaluation Suite")
        st.write("Run evaluation against a gold-standard annotated benchmark of research papers.")
        if st.button("🚀 Run Phase 12 Benchmark Suite", type="primary"):
            st.info("Evaluation framework will be fully wired to Microsoft Foundry evaluators in Phase 12.")
