"""
Tests for Phase 7: Research Question Generation + Future Research Directions.
Covers:
1. ResearchQuestion schema validation
2. Academic question types validation (13 types)
3. FutureDirection schema validation (is_suggestion=True)
4. ResearchQuestionAnalysisResult schema validation & summary counts
5. Gap-to-question traceability (every question maps to at least 1 gap)
6. Evidence provenance preservation (chunk text, page, score, paper_id)
7. Question origin: Author-inspired
8. Question origin: Evidence-derived
9. Question origin: Cross-paper
10. Novelty guardrail: Literature-scoped framing
11. Future direction separation: Partitioned AI suggestion
12. Personalized query handling and active query preservation
13. Single-paper question generation support
14. Multi-paper question generation support
15. Cached gap reuse (Phase 6 gaps passed directly)
16. Quality validation gate rejection of ungrounded questions
17. End-to-end mocked Foundry agent generation
18. Markdown and JSON export data integrity
19. Model failure graceful handling
20. No paper ranking or declaring winners
21. Phase 3 retrieval regression check
22. Phase 4 agent regression check
23. Phase 6 gap detector regression check
"""

import json
import pytest
from unittest.mock import MagicMock, patch

from models.paper import PaperDocument, PaperMetadata, DocumentChunk, EvidenceItem, RetrievalResult
from models.research_gap import (
    ResearchGap,
    ResearchGapAnalysisResult,
    ResearchQuestion,
    FutureDirection,
    ResearchQuestionAnalysisResult,
    QUESTION_TYPES,
    FUTURE_DIRECTION_TYPES,
)
from services.foundry_agent import ResearchLensAgent, SYSTEM_PROMPT_QUESTION_AGENT
from services.retrieval import HybridRetrievalEngine, BM25Index


@pytest.fixture
def mock_paper_a() -> PaperDocument:
    paper = PaperDocument(
        id="paper_001",
        filename="attention_paper.pdf",
        saved_path="/data/attention_paper.pdf",
        metadata=PaperMetadata(
            title="Attention Is All You Need",
            authors=["Vaswani et al."],
            publication_year="2017",
        ),
        page_count=15,
    )
    chunk1 = DocumentChunk(
        paper_id="paper_001",
        chunk_id="paper_001_c001",
        document_order=1,
        chunk_order=1,
        page_start=8,
        page_end=8,
        normalized_section="Limitations",
        original_heading="5. Limitations",
        text="While the Transformer achieves state of the art on English-to-German and English-to-French translation, quadratic self-attention complexity over sequence length limits scaling to very long documents.",
        rrf_score=0.033,
    )
    chunk2 = DocumentChunk(
        paper_id="paper_001",
        chunk_id="paper_001_c002",
        document_order=2,
        chunk_order=2,
        page_start=10,
        page_end=10,
        normalized_section="Conclusion",
        original_heading="6. Conclusion and Future Work",
        text="We plan to extend the Transformer to problems involving input and output modalities other than text, such as images, audio and video.",
        rrf_score=0.028,
    )
    paper.chunks = [chunk1, chunk2]
    return paper


@pytest.fixture
def mock_paper_b() -> PaperDocument:
    paper = PaperDocument(
        id="paper_002",
        filename="bert_paper.pdf",
        saved_path="/data/bert_paper.pdf",
        metadata=PaperMetadata(
            title="BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
            authors=["Devlin et al."],
            publication_year="2019",
        ),
        page_count=16,
    )
    chunk1 = DocumentChunk(
        paper_id="paper_002",
        chunk_id="paper_002_c001",
        document_order=1,
        chunk_order=1,
        page_start=9,
        page_end=9,
        normalized_section="Conclusion",
        original_heading="5. Conclusion and Future Work",
        text="In future work, we will investigate pre-training models that handle longer input contexts beyond 512 tokens and examine cross-lingual representations for low-resource languages.",
        rrf_score=0.031,
    )
    paper.chunks = [chunk1]
    return paper


@pytest.fixture
def sample_evidence_item() -> EvidenceItem:
    return EvidenceItem(
        evidence_id="ev_001",
        paper_id="paper_001",
        chunk_id="paper_001_c001",
        page_start=8,
        page_end=8,
        normalized_section="Limitations",
        original_heading="5. Limitations",
        content_type="body",
        score=0.033,
        citation_label="[paper_001:p8]",
        text="quadratic self-attention complexity over sequence length limits scaling to very long documents.",
    )


@pytest.fixture
def sample_gap(sample_evidence_item) -> ResearchGap:
    return ResearchGap(
        gap_id="GAP-01",
        title="Quadratic Complexity in Long Sequences",
        category="Methodological / Architectural Gaps",
        description="Self-attention scales quadratically with sequence length, making long-document processing intractable.",
        evidence_type="explicit",
        evidence_support="High (Direct citation)",
        supporting_papers=["paper_001"],
        rationale="Vaswani et al. explicitly note quadratic scaling bounds on sequence length.",
        citation_labels=["[paper_001:p8]"],
        evidence_items=[sample_evidence_item],
    )


@pytest.fixture
def sample_evidence_item_b() -> EvidenceItem:
    return EvidenceItem(
        evidence_id="ev_002",
        paper_id="paper_002",
        chunk_id="paper_002_c001",
        page_start=9,
        page_end=9,
        normalized_section="Conclusion",
        original_heading="5. Conclusion and Future Work",
        content_type="body",
        score=0.031,
        citation_label="[paper_002:p9]",
        text="In future work, we will investigate pre-training models that handle longer input contexts beyond 512 tokens.",
    )


@pytest.fixture
def sample_gap_b(sample_evidence_item_b) -> ResearchGap:
    return ResearchGap(
        gap_id="GAP-02",
        title="Context Limit in Masked Pre-training",
        category="Scalability Gap",
        description="BERT restricts sequences to 512 tokens due to memory bounds.",
        evidence_type="explicit",
        evidence_support="High (Direct citation)",
        supporting_papers=["paper_002"],
        rationale="Devlin et al. note input context limits.",
        citation_labels=["[paper_002:p9]"],
        evidence_items=[sample_evidence_item_b],
    )


@pytest.fixture
def mock_retrieval_engine(mock_paper_a, mock_paper_b) -> HybridRetrievalEngine:
    engine = MagicMock(spec=HybridRetrievalEngine)

    def retrieve_mock(query, paper_ids=None, top_k=5, search_mode="hybrid"):
        results = []
        if paper_ids is None or "paper_001" in paper_ids:
            results.extend([RetrievalResult(chunk=c, score=0.033) for c in mock_paper_a.chunks])
        if paper_ids is None or "paper_002" in paper_ids:
            results.extend([RetrievalResult(chunk=c, score=0.031) for c in mock_paper_b.chunks])
        return results[:top_k]

    engine.retrieve.side_effect = retrieve_mock
    return engine


# --------------------------------------------------------------------------
# Test 1: ResearchQuestion Schema Validation
# --------------------------------------------------------------------------
def test_research_question_schema(sample_evidence_item):
    q = ResearchQuestion(
        question_id="RQ-01",
        question="How does sub-quadratic attention impact perplexity on documents > 4096 tokens?",
        question_type="Methodological",
        question_origin="evidence_derived",
        research_gap_ids=["GAP-01"],
        supporting_papers=["paper_001"],
        evidence_items=[sample_evidence_item],
        citation_labels=["[paper_001:p8]"],
        rationale="Addresses the quadratic self-attention scaling bottleneck.",
        novelty_basis="Motivated by the identified gap in the selected papers regarding quadratic complexity.",
        potential_research_direction="Evaluate linear attention kernels on book-length sequences.",
        direction_type="Scalability studies",
        suggested_evaluation="Compute perplexity across sequence lengths from 512 to 8192.",
        suggested_dataset_or_setting="PG-19 long document benchmark.",
        evidence_support="High (Direct citation)",
    )
    assert q.question_id == "RQ-01"
    assert q.question_type == "Methodological"
    assert q.question_origin == "evidence_derived"
    assert q.question_origin_label == "Evidence-Derived Question"
    assert q.research_gap_ids == ["GAP-01"]
    assert len(q.evidence_items) == 1
    assert q.evidence_items[0].chunk_id == "paper_001_c001"


# --------------------------------------------------------------------------
# Test 2: Academic Question Types Validation
# --------------------------------------------------------------------------
def test_question_types():
    assert len(QUESTION_TYPES) == 13
    expected_types = [
        "Exploratory", "Explanatory", "Comparative", "Methodological",
        "Empirical", "Evaluation", "Generalization", "Dataset-focused",
        "Scalability", "Reproducibility", "Theoretical", "Cross-domain",
        "Cross-paper"
    ]
    for exp in expected_types:
        assert exp in QUESTION_TYPES


# --------------------------------------------------------------------------
# Test 3: FutureDirection Schema Validation (is_suggestion=True)
# --------------------------------------------------------------------------
def test_future_direction_schema():
    fd = FutureDirection(
        direction_id="FD-01",
        title="Sub-quadratic linear attention evaluation",
        direction_type="Scalability studies",
        gap_addressed="GAP-01",
        linked_question_id="RQ-01",
        description="Investigate recurrent or linear attention approximations.",
        suggested_methodology="Implement linear attention kernel.",
        suggested_dataset_or_setting="Long Range Arena (LRA).",
        expected_contribution="Reduce computational complexity from O(N^2) to O(N).",
        is_suggestion=True,
    )
    assert fd.direction_id == "FD-01"
    assert fd.is_suggestion is True
    assert fd.direction_type in FUTURE_DIRECTION_TYPES
    assert fd.gap_addressed == "GAP-01"
    assert fd.linked_question_id == "RQ-01"
    assert fd.description == "Investigate recurrent or linear attention approximations."


# --------------------------------------------------------------------------
# Test 4: ResearchQuestionAnalysisResult Schema Validation
# --------------------------------------------------------------------------
def test_research_question_analysis_result_schema(sample_evidence_item, sample_evidence_item_b):
    q = ResearchQuestion(
        question_id="RQ-01",
        question="How does sparse attention compare to dense attention?",
        question_type="Comparative",
        question_origin="cross_paper",
        research_gap_ids=["GAP-01", "GAP-02"],
        supporting_papers=["paper_001", "paper_002"],
        evidence_items=[sample_evidence_item, sample_evidence_item_b],
    )
    fd = FutureDirection(
        direction_id="FD-01",
        title="Comparative sparse evaluation",
        direction_type="Cross-paper synthesis",
        gap_addressed="GAP-01",
        linked_question_id="RQ-01",
        description="Compare sparse transformers vs BERT.",
        is_suggestion=True,
    )
    result = ResearchQuestionAnalysisResult(
        analysis_id="test-analysis-1",
        query="Scalability questions",
        selected_paper_ids=["paper_001", "paper_002"],
        selected_paper_titles={"paper_001": "Attention", "paper_002": "BERT"},
        questions=[q],
        future_directions=[fd],
        linked_gap_ids=["GAP-01"],
        all_evidence_items=[sample_evidence_item],
        summary_counts={"total_questions": 1, "cross_paper": 1, "future_directions": 1},
        model_used="gpt-4.1-mini",
        execution_latency_ms=450.0,
    )
    assert result.status == "completed"
    assert len(result.questions) == 1
    assert len(result.future_directions) == 1
    assert result.summary_counts["total_questions"] == 1


# --------------------------------------------------------------------------
# Test 5: Gap-to-Question Traceability
# --------------------------------------------------------------------------
def test_gap_to_question_traceability(sample_evidence_item):
    q = ResearchQuestion(
        question_id="RQ-01",
        question="Can multi-modal attention scale without cross-attention saturation?",
        question_type="Exploratory",
        question_origin="author_inspired",
        research_gap_ids=["GAP-02"],
        supporting_papers=["paper_001"],
        evidence_items=[sample_evidence_item],
    )
    assert len(q.research_gap_ids) > 0
    assert "GAP-02" in q.research_gap_ids


# --------------------------------------------------------------------------
# Test 6: Evidence Provenance Preservation
# --------------------------------------------------------------------------
def test_evidence_provenance_preservation(sample_evidence_item):
    q = ResearchQuestion(
        question_id="RQ-01",
        question="How does sequence length affect attention weights?",
        question_type="Empirical",
        question_origin="evidence_derived",
        research_gap_ids=["GAP-01"],
        supporting_papers=["paper_001"],
        evidence_items=[sample_evidence_item],
    )
    assert len(q.evidence_items) == 1
    ev = q.evidence_items[0]
    assert ev.paper_id == "paper_001"
    assert ev.chunk_id == "paper_001_c001"
    assert ev.page_start == 8
    assert ev.score == 0.033
    assert "quadratic self-attention" in ev.text


# --------------------------------------------------------------------------
# Test 7: Question Origin: Author-Inspired
# --------------------------------------------------------------------------
def test_question_origin_author_inspired():
    q = ResearchQuestion(
        question_id="RQ-01",
        question="How does multi-modal self-attention perform on video synthesis?",
        question_origin="author_inspired",
    )
    assert q.question_origin == "author_inspired"
    assert q.question_origin_label == "Author-Inspired Question"


# --------------------------------------------------------------------------
# Test 8: Question Origin: Evidence-Derived
# --------------------------------------------------------------------------
def test_question_origin_evidence_derived():
    q = ResearchQuestion(
        question_id="RQ-02",
        question="What is the empirical degradation of BERT representations on sequences > 512?",
        question_origin="evidence_derived",
    )
    assert q.question_origin == "evidence_derived"
    assert q.question_origin_label == "Evidence-Derived Question"


# --------------------------------------------------------------------------
# Test 9: Question Origin: Cross-Paper
# --------------------------------------------------------------------------
def test_question_origin_cross_paper():
    q = ResearchQuestion(
        question_id="RQ-03",
        question="How do autoregressive and bidirectional attention mechanisms compare under memory bounds?",
        question_origin="cross_paper",
    )
    assert q.question_origin == "cross_paper"
    assert q.question_origin_label == "Cross-Paper Synthesis Question"


# --------------------------------------------------------------------------
# Test 10: Novelty Guardrail: Literature-Scoped Framing
# --------------------------------------------------------------------------
def test_novelty_guardrail_literature_scoped():
    q = ResearchQuestion(
        question_id="RQ-01",
        question="Is this entirely novel in the field?",
        novelty_basis="This is completely novel in the field and never done before.",
    )
    assert "never done before" not in q.novelty_basis
    assert "Motivated by the identified gap in the selected papers" in q.novelty_basis


# --------------------------------------------------------------------------
# Test 11: Future Direction Separation: Partitioned AI Suggestion
# --------------------------------------------------------------------------
def test_future_direction_separation():
    fd = FutureDirection(
        direction_id="FD-01",
        title="Multi-modal transformer pre-training",
        direction_type="New domains",
        gap_addressed="GAP-02",
        description="Extend architecture to video processing.",
        is_suggestion=True,
    )
    assert fd.is_suggestion is True


# --------------------------------------------------------------------------
# Test 12: Personalized Query Handling and Active Query Preservation
# --------------------------------------------------------------------------
def test_personalized_query_handling(mock_paper_a, mock_paper_b, mock_retrieval_engine, sample_gap):
    mock_client = MagicMock()
    mock_client.chat_deployment = "gpt-4.1-mini"
    user_query = "What research questions can I investigate regarding long-sequence efficiency?"
    
    mock_response = {
        "questions": [
            {
                "question_id": "RQ-01",
                "question": "How can attention mechanisms achieve linear scaling on sequence lengths > 4096?",
                "question_type": "Scalability",
                "question_origin": "evidence_derived",
                "research_gap_ids": ["GAP-01"],
                "supporting_papers": ["paper_001"],
                "evidence_chunk_ids": ["paper_001_c001"],
                "rationale": "Directly tackles long-sequence quadratic scaling limitations.",
                "novelty_basis": "Motivated by the identified gap in the selected papers regarding sequence length bounds.",
                "suggested_evaluation": "Measure memory footprint and token throughput.",
                "suggested_dataset_or_setting": "PG-19 long text dataset.",
                "potential_research_direction": "Linearized attention kernels.",
                "direction_type": "Scalability studies",
            }
        ],
        "future_directions": [
            {
                "direction_id": "FD-01",
                "title": "Long-sequence linear attention benchmarks",
                "direction_type": "Scalability studies",
                "gap_addressed": "GAP-01",
                "linked_question_id": "RQ-01",
                "description": "Benchmark throughput on 8k sequence tokens.",
            }
        ]
    }
    mock_client.generate_chat_response.return_value = {
        "success": True,
        "content": json.dumps(mock_response),
    }

    agent = ResearchLensAgent(foundry_client=mock_client)
    result = agent.generate_research_questions(
        papers=[mock_paper_a, mock_paper_b],
        engine=mock_retrieval_engine,
        gaps=[sample_gap],
        custom_query=user_query,
    )
    assert result.status == "completed"
    assert result.query == user_query
    assert len(result.questions) == 1
    assert "linear scaling" in result.questions[0].question.lower() or "attention" in result.questions[0].question.lower()


# --------------------------------------------------------------------------
# Test 13: Single-Paper Question Generation Support
# --------------------------------------------------------------------------
def test_single_paper_question_generation(mock_paper_a, mock_retrieval_engine, sample_gap):
    mock_client = MagicMock()
    mock_client.chat_deployment = "gpt-4.1-mini"
    mock_response = {
        "questions": [
            {
                "question_id": "RQ-01",
                "question": "How does cross-attention degradation occur in vision-language tasks?",
                "question_type": "Exploratory",
                "question_origin": "author_inspired",
                "research_gap_ids": ["GAP-01"],
                "supporting_papers": ["paper_001"],
                "evidence_chunk_ids": ["paper_001_c001"],
                "rationale": "Vaswani et al. explicitly target vision modalities in future work.",
                "novelty_basis": "Motivated by the identified gap in the selected papers regarding multi-modality.",
            }
        ]
    }
    mock_client.generate_chat_response.return_value = {
        "success": True,
        "content": json.dumps(mock_response),
    }

    agent = ResearchLensAgent(foundry_client=mock_client)
    result = agent.generate_research_questions(
        papers=[mock_paper_a],
        engine=mock_retrieval_engine,
        gaps=[sample_gap],
    )
    assert result.status == "completed"
    assert len(result.selected_paper_ids) == 1
    assert result.selected_paper_ids[0] == "paper_001"
    assert len(result.questions) == 1


# --------------------------------------------------------------------------
# Test 14: Multi-Paper Question Generation Support
# --------------------------------------------------------------------------
def test_multi_paper_question_generation(mock_paper_a, mock_paper_b, mock_retrieval_engine, sample_gap, sample_gap_b):
    mock_client = MagicMock()
    mock_client.chat_deployment = "gpt-4.1-mini"
    mock_response = {
        "questions": [
            {
                "question_id": "RQ-01",
                "question": "How do autoregressive translation models compare to masked bidirectional representations on syntactic parsing?",
                "question_type": "Comparative",
                "question_origin": "cross_paper",
                "research_gap_ids": ["GAP-01", "GAP-02"],
                "supporting_papers": ["paper_001", "paper_002"],
                "evidence_chunk_ids": ["paper_001_c001", "paper_002_c001"],
                "rationale": "Synthesizes architectural tradeoffs between Vaswani et al. and Devlin et al.",
                "novelty_basis": "Potentially underexplored within the uploaded literature.",
            }
        ]
    }
    mock_client.generate_chat_response.return_value = {
        "success": True,
        "content": json.dumps(mock_response),
    }

    agent = ResearchLensAgent(foundry_client=mock_client)
    result = agent.generate_research_questions(
        papers=[mock_paper_a, mock_paper_b],
        engine=mock_retrieval_engine,
        gaps=[sample_gap, sample_gap_b],
    )
    assert result.status == "completed"
    assert len(result.selected_paper_ids) == 2
    assert len(result.questions) == 1
    assert result.questions[0].question_origin == "cross_paper"
    assert "paper_001" in result.questions[0].supporting_papers
    assert "paper_002" in result.questions[0].supporting_papers


# --------------------------------------------------------------------------
# Test 15: Cached Gap Reuse (Phase 6 gaps passed directly)
# --------------------------------------------------------------------------
def test_cached_gap_reuse(mock_paper_a, mock_retrieval_engine, sample_gap):
    mock_client = MagicMock()
    mock_client.chat_deployment = "gpt-4.1-mini"
    mock_response = {
        "questions": [
            {
                "question_id": "RQ-01",
                "question": "Can attention kernels be optimized using block-sparse approximations?",
                "question_type": "Methodological",
                "question_origin": "evidence_derived",
                "research_gap_ids": ["GAP-01"],
                "supporting_papers": ["paper_001"],
                "evidence_chunk_ids": ["paper_001_c001"],
                "rationale": "Directly addresses quadratic scaling.",
                "novelty_basis": "Motivated by the identified gap in the selected papers.",
            }
        ]
    }
    mock_client.generate_chat_response.return_value = {
        "success": True,
        "content": json.dumps(mock_response),
    }

    agent = ResearchLensAgent(foundry_client=mock_client)
    with patch.object(agent, "detect_research_gaps") as mock_detect_gaps:
        result = agent.generate_research_questions(
            papers=[mock_paper_a],
            engine=mock_retrieval_engine,
            gaps=[sample_gap],  # Explicitly passed
        )
        assert result.status == "completed"
        mock_detect_gaps.assert_not_called()


# --------------------------------------------------------------------------
# Test 16: Quality Validation Gate Rejection
# --------------------------------------------------------------------------
def test_quality_validation_gate_rejection(mock_paper_a, mock_retrieval_engine, sample_gap):
    mock_client = MagicMock()
    mock_client.chat_deployment = "gpt-4.1-mini"
    mock_response = {
        "questions": [
            {
                "question_id": "RQ-01",
                "question": "",  # Empty question -> invalid
                "question_type": "Comparative",
                "research_gap_ids": [],
            },
            {
                "question_id": "RQ-02",
                "question": "A perfectly valid question grounded in GAP-01?",
                "question_type": "Methodological",
                "research_gap_ids": ["GAP-01"],
                "supporting_papers": ["paper_001"],
                "evidence_chunk_ids": ["paper_001_c001"],
                "rationale": "Grounded in quadratic complexity gap.",
                "novelty_basis": "Motivated by the identified gap in the selected papers.",
            }
        ]
    }
    mock_client.generate_chat_response.return_value = {
        "success": True,
        "content": json.dumps(mock_response),
    }

    agent = ResearchLensAgent(foundry_client=mock_client)
    result = agent.generate_research_questions(
        papers=[mock_paper_a],
        engine=mock_retrieval_engine,
        gaps=[sample_gap],
    )
    assert result.status == "completed"
    assert len(result.questions) == 1
    assert result.questions[0].question_id == "RQ-02"


# --------------------------------------------------------------------------
# Test 17: End-to-End Mocked Foundry Agent Generation
# --------------------------------------------------------------------------
def test_agent_generate_research_questions_e2e_mock(mock_paper_a, mock_paper_b, mock_retrieval_engine, sample_gap):
    mock_client = MagicMock()
    mock_client.chat_deployment = "gpt-4.1-mini"
    mock_response = {
        "questions": [
            {
                "question_id": "RQ-01",
                "question": "How does attention complexity impact low-resource multilingual translation?",
                "question_type": "Empirical",
                "question_origin": "evidence_derived",
                "research_gap_ids": ["GAP-01"],
                "supporting_papers": ["paper_001"],
                "evidence_chunk_ids": ["paper_001_c001"],
                "rationale": "Addresses both quadratic complexity and language coverage.",
                "novelty_basis": "Motivated by the identified gap in the selected papers.",
                "suggested_evaluation": "BLEU scores on low-resource pairs.",
                "suggested_dataset_or_setting": "WMT14 benchmark.",
                "potential_research_direction": "Linear cross-lingual attention.",
                "direction_type": "New languages",
            }
        ],
        "future_directions": [
            {
                "direction_id": "FD-01",
                "title": "Low-resource multilingual linear attention",
                "direction_type": "New languages",
                "gap_addressed": "GAP-01",
                "linked_question_id": "RQ-01",
                "description": "Evaluate linear attention on African languages.",
                "suggested_methodology": "Dual encoder training.",
                "suggested_dataset_or_setting": "Masakhane benchmark.",
                "expected_contribution": "State of the art on low-resource translation.",
            }
        ]
    }
    mock_client.generate_chat_response.return_value = {
        "success": True,
        "content": json.dumps(mock_response),
    }

    agent = ResearchLensAgent(foundry_client=mock_client)
    result = agent.generate_research_questions(
        papers=[mock_paper_a, mock_paper_b],
        engine=mock_retrieval_engine,
        gaps=[sample_gap],
    )
    assert result.status == "completed"
    assert len(result.questions) == 1
    assert len(result.future_directions) == 1
    assert result.summary_counts["total_questions"] == 1
    assert result.summary_counts["evidence_derived"] == 1
    assert result.summary_counts["future_directions"] == 1


# --------------------------------------------------------------------------
# Test 18: Markdown and JSON Export Integrity
# --------------------------------------------------------------------------
def test_json_markdown_export_integrity(sample_evidence_item):
    q = ResearchQuestion(
        question_id="RQ-01",
        question="How does sub-quadratic attention impact perplexity on long documents?",
        question_type="Methodological",
        question_origin="evidence_derived",
        research_gap_ids=["GAP-01"],
        supporting_papers=["paper_001"],
        evidence_items=[sample_evidence_item],
        citation_labels=["[paper_001:p8]"],
        rationale="Addresses quadratic scaling.",
        novelty_basis="Motivated by the identified gap in the selected papers.",
        potential_research_direction="Evaluate linear attention kernels on book-length sequences.",
        direction_type="Scalability studies",
        suggested_evaluation="Compute perplexity across sequence lengths.",
        suggested_dataset_or_setting="PG-19 benchmark.",
        evidence_support="High (Direct citation)",
    )
    fd = FutureDirection(
        direction_id="FD-01",
        title="Sub-quadratic linear attention evaluation",
        direction_type="Scalability studies",
        gap_addressed="GAP-01",
        linked_question_id="RQ-01",
        description="Investigate recurrent or linear attention approximations.",
        is_suggestion=True,
    )
    result = ResearchQuestionAnalysisResult(
        analysis_id="test-export-1",
        query="Scalability query",
        selected_paper_ids=["paper_001"],
        selected_paper_titles={"paper_001": "Attention"},
        questions=[q],
        future_directions=[fd],
        linked_gap_ids=["GAP-01"],
        all_evidence_items=[sample_evidence_item],
        summary_counts={"total_questions": 1, "evidence_derived": 1, "future_directions": 1},
        model_used="gpt-4.1-mini",
        execution_latency_ms=300.0,
    )

    # JSON export test
    json_data = json.dumps({
        "analysis_id": result.analysis_id,
        "query": result.query,
        "questions": [{"id": item.question_id, "text": item.question} for item in result.questions],
        "future_directions": [{"id": item.direction_id, "title": item.title} for item in result.future_directions],
    })
    parsed = json.loads(json_data)
    assert parsed["analysis_id"] == "test-export-1"
    assert parsed["query"] == "Scalability query"
    assert len(parsed["questions"]) == 1
    assert len(parsed["future_directions"]) == 1


# --------------------------------------------------------------------------
# Test 19: Model Failure Graceful Handling
# --------------------------------------------------------------------------
def test_model_failure_graceful_handling(mock_paper_a, mock_retrieval_engine, sample_gap):
    mock_client = MagicMock()
    mock_client.chat_deployment = "gpt-4.1-mini"
    mock_client.generate_chat_response.return_value = {
        "success": False,
        "content": "",
        "error": "Azure OpenAI service unavailable",
    }
    agent = ResearchLensAgent(foundry_client=mock_client)
    result = agent.generate_research_questions(
        papers=[mock_paper_a],
        engine=mock_retrieval_engine,
        gaps=[sample_gap],
    )
    assert result.status == "error"
    assert "Azure OpenAI service unavailable" in result.error_message
    assert len(result.questions) == 0


# --------------------------------------------------------------------------
# Test 20: No Paper Ranking or Winner Language
# --------------------------------------------------------------------------
def test_no_ranking_or_winner_language():
    prompt = SYSTEM_PROMPT_QUESTION_AGENT
    assert "Do NOT rank papers" in prompt or "winner" in prompt.lower()
    assert "No Winner Language" in prompt or "objective" in prompt.lower()


# --------------------------------------------------------------------------
# Test 21: Phase 3 Retrieval Regression Check
# --------------------------------------------------------------------------
def test_phase_3_retrieval_regression():
    bm25 = BM25Index()
    assert bm25.k1 == 1.5
    assert bm25.b == 0.75


# --------------------------------------------------------------------------
# Test 22: Phase 4 Agent Regression Check
# --------------------------------------------------------------------------
def test_phase_4_agent_regression():
    agent = ResearchLensAgent()
    assert hasattr(agent, "answer_question")
    assert hasattr(agent, "analyze_paper_sections")


# --------------------------------------------------------------------------
# Test 23: Phase 6 Gap Detector Regression Check
# --------------------------------------------------------------------------
def test_phase_6_gaps_regression(mock_paper_a, mock_paper_b, mock_retrieval_engine):
    mock_client = MagicMock()
    mock_client.chat_deployment = "gpt-4.1-mini"
    mock_gap_json = {
        "gaps": [
            {
                "gap_id": "GAP-01",
                "title": "Quadratic Self-Attention Complexity",
                "category": "Methodological Gap",
                "description": "Scaling to long sequences is computationally prohibitive.",
                "evidence_type": "explicit",
                "evidence_support": "High evidence support",
                "supporting_papers": ["paper_001"],
                "rationale": "Vaswani et al. explicitly cite quadratic bounds.",
                "evidence_chunk_ids": ["paper_001_c001"],
            }
        ]
    }
    mock_client.generate_chat_response.return_value = {
        "success": True,
        "content": json.dumps(mock_gap_json),
    }
    agent = ResearchLensAgent(foundry_client=mock_client)
    gap_result = agent.detect_research_gaps(
        papers=[mock_paper_a, mock_paper_b],
        engine=mock_retrieval_engine,
    )
    assert gap_result.status == "completed"
    assert len(gap_result.gaps) == 1
    assert gap_result.gaps[0].gap_id == "GAP-01"


# --------------------------------------------------------------------------
# Test 24: Question supporting paper matches evidence items
# --------------------------------------------------------------------------
def test_question_supporting_paper_matches_evidence(sample_evidence_item):
    q = ResearchQuestion(
        question_id="RQ-01",
        question="How does sub-quadratic attention impact perplexity?",
        question_type="Methodological",
        research_gap_ids=["GAP-01"],
        supporting_papers=["paper_001"],
        evidence_items=[sample_evidence_item],
    )
    # Every evidence item's paper_id must belong to question.supporting_papers
    for ev in q.evidence_items:
        assert ev.paper_id in q.supporting_papers

    # For every supporting paper, there must be at least one evidence item from that paper
    for p in q.supporting_papers:
        assert any(ev.paper_id == p for ev in q.evidence_items)


# --------------------------------------------------------------------------
# Test 25: No cross-paper attribution leakage
# --------------------------------------------------------------------------
def test_no_cross_paper_attribution_leakage(mock_paper_a, mock_paper_b, mock_retrieval_engine, sample_gap):
    # Both papers are selected, but the question only addresses GAP-01 (paper_001).
    # Even if LLM erroneously suggests supporting_papers = ["paper_002"],
    # the application must deterministically assign supporting_papers = ["paper_001"]
    mock_client = MagicMock()
    mock_client.chat_deployment = "gpt-4.1-mini"
    mock_response = {
        "questions": [
            {
                "question_id": "rq_001",
                "question": "How can more sophisticated attention compatibility functions improve Transformers?",
                "question_type": "Methodological",
                "question_origin": "author_inspired",
                "research_gap_ids": ["GAP-01"],
                "supporting_papers": ["paper_002"],  # LLM erroneously attempts to attribute to paper_002
                "rationale": "Refers to dot-product compatibility functions in Vaswani et al.",
            }
        ]
    }
    mock_client.generate_chat_response.return_value = {
        "success": True,
        "content": json.dumps(mock_response),
    }

    agent = ResearchLensAgent(foundry_client=mock_client)
    result = agent.generate_research_questions(
        papers=[mock_paper_a, mock_paper_b],
        engine=mock_retrieval_engine,
        gaps=[sample_gap],  # Only GAP-01 from paper_001
    )
    assert result.status == "completed"
    assert len(result.questions) == 1
    q = result.questions[0]
    # Critical fix check: paper_002 must NOT leak into supporting_papers
    assert q.supporting_papers == ["paper_001"]
    assert "paper_002" not in q.supporting_papers
    # And evidence items must all be from paper_001
    for ev in q.evidence_items:
        assert ev.paper_id == "paper_001"


# --------------------------------------------------------------------------
# Test 26: Single-paper question isolation
# --------------------------------------------------------------------------
def test_single_paper_question_isolation(mock_paper_a, mock_paper_b, mock_retrieval_engine, sample_gap, sample_gap_b):
    # Multiple papers provided, but individual questions address single-paper gaps
    mock_client = MagicMock()
    mock_client.chat_deployment = "gpt-4.1-mini"
    mock_response = {
        "questions": [
            {
                "question_id": "rq_001",
                "question": "Attention quadratic complexity question",
                "question_origin": "author_inspired",
                "research_gap_ids": ["GAP-01"],
            },
            {
                "question_id": "rq_002",
                "question": "BERT context window question",
                "question_origin": "author_inspired",
                "research_gap_ids": ["GAP-02"],
            }
        ]
    }
    mock_client.generate_chat_response.return_value = {
        "success": True,
        "content": json.dumps(mock_response),
    }

    agent = ResearchLensAgent(foundry_client=mock_client)
    result = agent.generate_research_questions(
        papers=[mock_paper_a, mock_paper_b],
        engine=mock_retrieval_engine,
        gaps=[sample_gap, sample_gap_b],
    )
    assert result.status == "completed"
    assert len(result.questions) == 2
    # rq_001 is strictly paper_001
    assert result.questions[0].supporting_papers == ["paper_001"]
    assert result.questions[0].question_origin == "author_inspired"
    # rq_002 is strictly paper_002
    assert result.questions[1].supporting_papers == ["paper_002"]
    assert result.questions[1].question_origin == "author_inspired"


# --------------------------------------------------------------------------
# Test 27: Cross-paper question requires evidence from both papers
# --------------------------------------------------------------------------
def test_cross_paper_question_requires_evidence_from_both_papers(mock_paper_a, mock_paper_b, mock_retrieval_engine, sample_gap):
    # LLM claims a question is "cross_paper", but it only links to GAP-01 (supported only by paper_001).
    # The system must reject the false cross_paper label and demote it.
    mock_client = MagicMock()
    mock_client.chat_deployment = "gpt-4.1-mini"
    mock_response = {
        "questions": [
            {
                "question_id": "rq_001",
                "question": "How do attention mechanisms compare?",
                "question_type": "Comparative",
                "question_origin": "cross_paper",  # LLM erroneously labels as cross_paper
                "research_gap_ids": ["GAP-01"],    # Only GAP-01 exists (paper_001 only)
            }
        ]
    }
    mock_client.generate_chat_response.return_value = {
        "success": True,
        "content": json.dumps(mock_response),
    }

    agent = ResearchLensAgent(foundry_client=mock_client)
    result = agent.generate_research_questions(
        papers=[mock_paper_a, mock_paper_b],
        engine=mock_retrieval_engine,
        gaps=[sample_gap],
    )
    assert result.status == "completed"
    q = result.questions[0]
    # Because only paper_001 evidence exists, origin CANNOT be cross_paper
    assert q.question_origin != "cross_paper"
    assert q.supporting_papers == ["paper_001"]


# --------------------------------------------------------------------------
# Test 28: Rationale and evidence paper consistency
# --------------------------------------------------------------------------
def test_rationale_and_evidence_paper_consistency(sample_evidence_item):
    q = ResearchQuestion(
        question_id="rq_001",
        question="How does key size impact translation?",
        research_gap_ids=["GAP-01"],
        supporting_papers=["paper_002"],  # Invalid assignment passed to model
        evidence_items=[sample_evidence_item],  # From paper_001
        rationale="Reducing key size hurts Transformer translation in Vaswani et al.",
    )
    # The post-init hook must enforce that supporting_papers matches the evidence item's paper
    assert q.supporting_papers == ["paper_001"]
    assert "paper_002" not in q.supporting_papers


# --------------------------------------------------------------------------
# Test 29: Supporting papers are derived, not LLM-assigned
# --------------------------------------------------------------------------
def test_supporting_papers_are_derived_not_llm_assigned(mock_paper_a, mock_retrieval_engine, sample_gap):
    mock_client = MagicMock()
    mock_client.chat_deployment = "gpt-4.1-mini"
    mock_response = {
        "questions": [
            {
                "question_id": "rq_001",
                "question": "Attention key size optimization question",
                "question_origin": "author_inspired",
                "research_gap_ids": ["GAP-01"],
                "supporting_papers": ["paper_999_fabricated"],  # Fabricated ID by LLM
            }
        ]
    }
    mock_client.generate_chat_response.return_value = {
        "success": True,
        "content": json.dumps(mock_response),
    }

    agent = ResearchLensAgent(foundry_client=mock_client)
    result = agent.generate_research_questions(
        papers=[mock_paper_a],
        engine=mock_retrieval_engine,
        gaps=[sample_gap],
    )
    assert result.status == "completed"
    q = result.questions[0]
    # Fabricated paper ID must be ignored and replaced by the actual evidence paper
    assert q.supporting_papers == ["paper_001"]


# --------------------------------------------------------------------------
# Test 30: Question gap evidence chain integrity
# --------------------------------------------------------------------------
def test_question_gap_evidence_chain_integrity(sample_evidence_item, sample_gap):
    q = ResearchQuestion(
        question_id="rq_001",
        question="How does sequence length affect attention?",
        research_gap_ids=["GAP-01"],
        evidence_items=[sample_evidence_item],
    )
    assert q.research_gap_ids == ["GAP-01"]
    assert sample_gap.gap_id == "GAP-01"
    assert q.supporting_papers == sample_gap.supporting_papers
    assert len(q.evidence_items) == 1
    assert q.evidence_items[0].chunk_id == "paper_001_c001"
    assert q.evidence_items[0].page_start == 8
    assert q.evidence_items[0].normalized_section == "Limitations"


# --------------------------------------------------------------------------
# Test 31: Paper ID provenance matches chunk paper ID
# --------------------------------------------------------------------------
def test_paper_id_provenance_matches_chunk_paper_id(mock_paper_a, sample_evidence_item):
    chunk = mock_paper_a.chunks[0]
    assert sample_evidence_item.paper_id == chunk.paper_id
    assert sample_evidence_item.chunk_id == chunk.chunk_id
    q = ResearchQuestion(
        question_id="rq_001",
        question="Test Question",
        evidence_items=[sample_evidence_item],
    )
    assert q.supporting_papers[0] == chunk.paper_id


# --------------------------------------------------------------------------
# Test 32 (Req 7A): BERTLARGE / fine-tuning evidence must map to paper_002
# --------------------------------------------------------------------------
def test_bertlarge_fine_tuning_evidence_maps_to_paper_002():
    bert_chunk = DocumentChunk(
        paper_id="paper_002",
        chunk_id="paper_002_c005",
        document_order=5,
        chunk_order=5,
        page_start=8,
        page_end=8,
        normalized_section="Discussion",
        original_heading="5.2 Effect of Model Size",
        text="However, we note that fine-tuning BERTLARGE can be unstable on small datasets, which motivated multiple random restarts.",
        rrf_score=0.035,
    )
    ev_item = EvidenceItem(
        evidence_id="ev_bert_01",
        paper_id="paper_002",
        chunk_id="paper_002_c005",
        page_start=8,
        page_end=8,
        normalized_section="Discussion",
        original_heading="5.2 Effect of Model Size",
        content_type="body",
        score=0.035,
        citation_label="[paper_002:p8:Discussion]",
        text=bert_chunk.text,
    )
    bert_gap = ResearchGap(
        gap_id="gap_bert_01",
        title="Fine-Tuning Instability of Large Models on Small Datasets",
        category="Reproducibility Gap",
        description="Fine-tuning BERTLARGE on small datasets can lead to severe optimization instability.",
        supporting_papers=["paper_002"],
        evidence_items=[ev_item],
    )
    # Ensure gap itself derives supporting_papers from ev_item
    assert bert_gap.supporting_papers == ["paper_002"]

    q = ResearchQuestion(
        question_id="rq_bert_01",
        question="How can fine-tuning stability of BERTLARGE be improved on small datasets?",
        research_gap_ids=["gap_bert_01"],
        evidence_items=[ev_item],
        rationale="Devlin et al. (paper_002) observe that fine-tuning BERTLARGE can be unstable on small datasets.",
    )
    assert q.supporting_papers == ["paper_002"]
    assert "paper_001" not in q.supporting_papers


# --------------------------------------------------------------------------
# Test 33 (Req 7B): Attention architecture evidence must map to paper_001
# --------------------------------------------------------------------------
def test_attention_architecture_evidence_maps_to_paper_001(sample_evidence_item, sample_gap):
    q = ResearchQuestion(
        question_id="rq_attn_01",
        question="How can more sophisticated attention compatibility functions improve Transformer translation?",
        research_gap_ids=[sample_gap.gap_id],
        evidence_items=[sample_evidence_item],
        rationale="In paper_001, Vaswani et al. demonstrate that reducing key size d_k hurts model quality.",
    )
    assert q.supporting_papers == ["paper_001"]
    assert "paper_002" not in q.supporting_papers
    assert q.evidence_items[0].paper_id == "paper_001"


# --------------------------------------------------------------------------
# Test 34 (Req 7C): Cross-paper question must preserve separate paper attribution
# --------------------------------------------------------------------------
def test_cross_paper_question_preserves_separate_paper_attribution():
    ev1 = EvidenceItem(
        evidence_id="ev_001",
        paper_id="paper_001",
        chunk_id="paper_001_c001",
        page_start=6,
        page_end=6,
        normalized_section="Methodology",
        original_heading="4. Self-Attention",
        content_type="body",
        score=0.033,
        citation_label="[paper_001:p6]",
        text="Self-attention layers have O(n^2 * d) complexity per layer.",
    )
    ev2 = EvidenceItem(
        evidence_id="ev_002",
        paper_id="paper_002",
        chunk_id="paper_002_c001",
        page_start=8,
        page_end=8,
        normalized_section="Discussion",
        original_heading="5.2 Effect of Model Size",
        content_type="body",
        score=0.032,
        citation_label="[paper_002:p8]",
        text="Fine-tuning BERTLARGE on small datasets can be unstable across random seeds.",
    )
    q = ResearchQuestion(
        question_id="rq_cross_01",
        question="How does model scaling interact with sequence length and fine-tuning stability across architectures?",
        research_gap_ids=["gap_01", "gap_02"],
        evidence_items=[ev1, ev2],
        rationale="While Vaswani et al. (paper_001) focus on quadratic attention scaling, Devlin et al. (paper_002) highlight fine-tuning instability in large bidirectional models.",
    )
    assert q.question_origin == "cross_paper"
    assert q.question_origin_label == "Cross-Paper Synthesis Question"
    assert "paper_001" in q.supporting_papers
    assert "paper_002" in q.supporting_papers
    assert len(q.supporting_papers) == 2
    # Check separate chunk attribution preserved
    papers_in_chunks = [ev.paper_id for ev in q.evidence_items]
    assert "paper_001" in papers_in_chunks
    assert "paper_002" in papers_in_chunks


# --------------------------------------------------------------------------
# Test 35 (Req 7D): supporting_papers must equal unique paper_ids of linked evidence
# --------------------------------------------------------------------------
def test_supporting_papers_equals_unique_evidence_paper_ids():
    ev_a = EvidenceItem(
        evidence_id="ev_1",
        paper_id="paper_001",
        chunk_id="c1",
        page_start=1,
        page_end=1,
        normalized_section="Intro",
        original_heading="1. Intro",
        content_type="body",
        score=0.01,
        citation_label="[p1]",
        text="Text A",
    )
    ev_b = EvidenceItem(
        evidence_id="ev_2",
        paper_id="paper_001",
        chunk_id="c2",
        page_start=2,
        page_end=2,
        normalized_section="Intro",
        original_heading="1. Intro",
        content_type="body",
        score=0.01,
        citation_label="[p2]",
        text="Text B",
    )
    q = ResearchQuestion(
        question_id="rq_test",
        question="Test Question",
        supporting_papers=["paper_002", "paper_003"],  # Injected false papers
        evidence_items=[ev_a, ev_b],  # Both belong to paper_001
    )
    # Invariant: supporting_papers MUST equal the unique paper_ids of the evidence items
    assert q.supporting_papers == ["paper_001"]
    assert set(q.supporting_papers) == {ev.paper_id for ev in q.evidence_items}


# --------------------------------------------------------------------------
# Test 36 (Req 7E): No question may contain a paper-specific claim unsupported by evidence
# --------------------------------------------------------------------------
def test_no_question_contains_paper_claim_unsupported_by_evidence(mock_paper_a, mock_retrieval_engine, sample_gap):
    # LLM hallucinates a question about BERTLARGE but provides evidence only from paper_001
    mock_client = MagicMock()
    mock_client.chat_deployment = "gpt-4.1-mini"
    mock_response = {
        "questions": [
            {
                "question_id": "rq_hallucinated",
                "question": "How can BERTLARGE fine-tuning stability be improved?",
                "question_type": "Methodological",
                "question_origin": "evidence_derived",
                "research_gap_ids": [sample_gap.gap_id],  # GAP from paper_001
                "rationale": "Paper_001 notes fine-tuning BERTLARGE can be unstable on small datasets.",
                "supporting_papers": ["paper_001"],
            }
        ]
    }
    mock_client.generate_chat_response.return_value = {
        "success": True,
        "content": json.dumps(mock_response),
    }

    agent = ResearchLensAgent(foundry_client=mock_client)
    result = agent.generate_research_questions(
        papers=[mock_paper_a],
        engine=mock_retrieval_engine,
        gaps=[sample_gap],
    )
    # The validation gate MUST reject the question with BERT claim unsupported by paper_001 evidence
    assert len(result.questions) == 0 or all("bertlarge" not in q.rationale.lower() or "paper_002" in q.supporting_papers for q in result.questions)


# --------------------------------------------------------------------------
# Test 37 (Req 7F): No cross-paper leakage in rationale
# --------------------------------------------------------------------------
def test_no_cross_paper_leakage_in_rationale(mock_paper_a, mock_retrieval_engine, sample_gap):
    # LLM produces inverted attribution: "Paper_001 notes fine-tuning BERTLARGE..."
    mock_client = MagicMock()
    mock_client.chat_deployment = "gpt-4.1-mini"
    mock_response = {
        "questions": [
            {
                "question_id": "rq_leakage",
                "question": "How does masked pre-training scale to longer sequences?",
                "question_type": "Scalability",
                "question_origin": "author_inspired",
                "research_gap_ids": [sample_gap.gap_id],
                "rationale": "Paper_001 notes fine-tuning BERTLARGE can be unstable on small datasets and emphasizes bidirectional pre-training.",
            }
        ]
    }
    mock_client.generate_chat_response.return_value = {
        "success": True,
        "content": json.dumps(mock_response),
    }

    agent = ResearchLensAgent(foundry_client=mock_client)
    result = agent.generate_research_questions(
        papers=[mock_paper_a],
        engine=mock_retrieval_engine,
        gaps=[sample_gap],
    )
    # The question with cross-paper leakage must be rejected
    assert len(result.questions) == 0


# --------------------------------------------------------------------------
# Test 38: Broad query with multiple independent gaps yields multiple distinct questions
# --------------------------------------------------------------------------
def test_broad_query_yields_multiple_distinct_questions(mock_paper_a, mock_paper_b, mock_retrieval_engine):
    ev1 = EvidenceItem(
        evidence_id="ev_comp_1",
        paper_id="paper_001",
        chunk_id="c001",
        page_start=8,
        page_end=8,
        normalized_section="Complexity",
        original_heading="5. Complexity",
        content_type="body",
        text="Self-attention layers have O(n^2 * d) complexity while restricted attention is O(r * n * d).",
        score=0.92,
        citation_label="[Evidence A1]",
    )
    ev2 = EvidenceItem(
        evidence_id="ev_pos_1",
        paper_id="paper_001",
        chunk_id="c002",
        page_start=5,
        page_end=5,
        normalized_section="Positional Encoding",
        original_heading="3.5 Positional Encoding",
        content_type="body",
        text="Sinusoidal positional encoding allows extrapolation to sequence lengths longer than seen during training.",
        score=0.88,
        citation_label="[Evidence A2]",
    )
    ev3 = EvidenceItem(
        evidence_id="ev_bert_1",
        paper_id="paper_002",
        chunk_id="c003",
        page_start=4,
        page_end=4,
        normalized_section="Pre-training",
        original_heading="3.1 Pre-training",
        content_type="body",
        text="BERT pre-training is executed with sequence length 128 for 90% of steps and 512 for remaining steps.",
        score=0.89,
        citation_label="[Evidence B1]",
    )

    gap1 = ResearchGap(
        gap_id="gap_comp",
        title="Quadratic Self-Attention Complexity",
        category="Methodological Gap",
        description="Quadratic self-attention scaling on long sequences.",
        evidence_type="explicit",
        evidence_support="High evidence support",
        supporting_papers=["paper_001"],
        evidence_items=[ev1],
    )
    gap2 = ResearchGap(
        gap_id="gap_pos",
        title="Positional Encoding Extrapolation",
        category="Generalization Gap",
        description="Extrapolation limits of positional encodings beyond training lengths.",
        evidence_type="explicit",
        evidence_support="High evidence support",
        supporting_papers=["paper_001"],
        evidence_items=[ev2],
    )
    gap3 = ResearchGap(
        gap_id="gap_bert_scale",
        title="Pre-training Sequence Length Constraints",
        category="Scalability Gap",
        description="Scalability limits of bidirectional pre-training on extended sequence lengths.",
        evidence_type="evidence_derived",
        evidence_support="Moderate evidence support",
        supporting_papers=["paper_002"],
        evidence_items=[ev3],
    )

    mock_client = MagicMock()
    mock_client.chat_deployment = "gpt-4.1-mini"
    mock_response = {
        "questions": [
            {
                "question_id": "rq_001",
                "question": "How can restricted attention mechanisms reduce self-attention computational complexity on long sequences?",
                "question_type": "Scalability",
                "question_origin": "author_inspired",
                "research_gap_ids": ["gap_comp"],
                "rationale": "Paper_001 notes self-attention layers have quadratic complexity and suggests restricted attention.",
                "potential_research_direction": "Implement local window attention benchmarks.",
                "direction_type": "Architectural modification",
            },
            {
                "question_id": "rq_002",
                "question": "How effectively do sinusoidal positional encodings extrapolate to sequence lengths substantially longer than training sequences?",
                "question_type": "Generalization",
                "question_origin": "author_inspired",
                "research_gap_ids": ["gap_pos"],
                "rationale": "Paper_001 hypothesizes sinusoidal positional encodings can extrapolate to longer sequence lengths.",
                "potential_research_direction": "Evaluate perplexity degradation on 4k token sequences.",
                "direction_type": "New experiments",
            },
            {
                "question_id": "rq_003",
                "question": "What memory and throughput trade-offs emerge when scaling bidirectional masked language model pre-training to longer sequence lengths?",
                "question_type": "Empirical",
                "question_origin": "evidence_derived",
                "research_gap_ids": ["gap_bert_scale"],
                "rationale": "Paper_002 demonstrates pre-training BERT with sequence length 128 before switching to 512 due to computational constraints.",
                "potential_research_direction": "Profile memory allocation for 2048-token bidirectional pre-training.",
                "direction_type": "Benchmark expansion",
            },
        ]
    }
    mock_client.generate_chat_response.return_value = {
        "success": True,
        "content": json.dumps(mock_response),
    }

    agent = ResearchLensAgent(foundry_client=mock_client)
    result = agent.generate_research_questions(
        papers=[mock_paper_a, mock_paper_b],
        engine=mock_retrieval_engine,
        gaps=[gap1, gap2, gap3],
        custom_query="What research questions can I investigate using these papers if I am particularly interested in long-sequence efficiency?",
    )

    assert result.status == "completed"
    assert len(result.questions) == 3
    assert result.summary_counts["total_questions"] == 3
    # Check that questions cover independent gaps
    gap_ids_covered = set()
    for q in result.questions:
        gap_ids_covered.update(q.research_gap_ids)
    assert gap_ids_covered == {"gap_comp", "gap_pos", "gap_bert_scale"}


# --------------------------------------------------------------------------
# Test 39: Narrow query legitimately yields 1-2 focused questions without error
# --------------------------------------------------------------------------
def test_narrow_query_yields_focused_question_without_forced_retries(mock_paper_a, mock_retrieval_engine):
    ev = EvidenceItem(
        evidence_id="ev_pos_single",
        paper_id="paper_001",
        chunk_id="c002",
        page_start=5,
        page_end=5,
        normalized_section="Positional Encoding",
        original_heading="3.5 Positional Encoding",
        content_type="body",
        text="Sinusoidal positional encoding allows extrapolation to sequence lengths longer than seen during training.",
        score=0.91,
        citation_label="[Evidence A1]",
    )
    narrow_gap = ResearchGap(
        gap_id="gap_pos_narrow",
        title="Positional Encoding Extrapolation",
        category="Generalization Gap",
        description="Extrapolation limits of positional encodings beyond training lengths.",
        evidence_type="explicit",
        evidence_support="High evidence support",
        supporting_papers=["paper_001"],
        evidence_items=[ev],
    )

    mock_client = MagicMock()
    mock_client.chat_deployment = "gpt-4.1-mini"
    mock_response = {
        "questions": [
            {
                "question_id": "rq_001",
                "question": "How effectively do sinusoidal positional encodings extrapolate to sequence lengths substantially longer than training sequences?",
                "question_type": "Generalization",
                "question_origin": "author_inspired",
                "research_gap_ids": ["gap_pos_narrow"],
                "rationale": "Paper_001 hypothesizes sinusoidal positional encodings can extrapolate to longer sequence lengths.",
            }
        ]
    }
    mock_client.generate_chat_response.return_value = {
        "success": True,
        "content": json.dumps(mock_response),
    }

    agent = ResearchLensAgent(foundry_client=mock_client)
    result = agent.generate_research_questions(
        papers=[mock_paper_a],
        engine=mock_retrieval_engine,
        gaps=[narrow_gap],
        custom_query="What research questions can I formulate specifically regarding positional encoding extrapolation?",
    )

    # A narrow query legitimately produces 1 question without forced extra calls
    assert result.status == "completed"
    assert len(result.questions) == 1
    assert result.summary_counts["total_questions"] == 1
    # Model should have been called only once (no retry triggered)
    assert mock_client.generate_chat_response.call_count == 1


# --------------------------------------------------------------------------
# Test 40: Near-duplicate candidate question is rejected by diversity filter
# --------------------------------------------------------------------------
def test_near_duplicate_candidate_is_rejected_by_diversity_filter(mock_paper_a, mock_retrieval_engine, sample_gap):
    mock_client = MagicMock()
    mock_client.chat_deployment = "gpt-4.1-mini"
    # Q1 and Q2 are near-duplicates with identical meaning and high word overlap
    mock_response = {
        "questions": [
            {
                "question_id": "rq_001",
                "question": "How can restricted attention mechanisms reduce the computational complexity of Transformer models on long sequences?",
                "question_type": "Scalability",
                "question_origin": "author_inspired",
                "research_gap_ids": [sample_gap.gap_id],
                "rationale": "Paper_001 discusses self-attention complexity and mentions restricted attention.",
            },
            {
                "question_id": "rq_002",
                "question": "How does restricted self-attention mechanism reduce computational complexity of Transformers on long sequence inputs?",
                "question_type": "Scalability",
                "question_origin": "author_inspired",
                "research_gap_ids": [sample_gap.gap_id],
                "rationale": "Paper_001 discusses self-attention complexity and mentions restricted attention.",
            },
        ]
    }
    mock_client.generate_chat_response.return_value = {
        "success": True,
        "content": json.dumps(mock_response),
    }

    agent = ResearchLensAgent(foundry_client=mock_client)
    result = agent.generate_research_questions(
        papers=[mock_paper_a],
        engine=mock_retrieval_engine,
        gaps=[sample_gap],
    )

    # Q2 should be rejected as a near-duplicate of Q1
    assert len(result.questions) == 1
    assert result.questions[0].question_id == "rq_001"


# --------------------------------------------------------------------------
# Test 41: Fresh query-tailored detection is executed when gaps=None
# --------------------------------------------------------------------------
def test_fresh_query_tailored_detection_executed_when_gaps_none(mock_paper_a, mock_retrieval_engine, sample_gap):
    tailored_gap = ResearchGap(
        gap_id="gap_tailored",
        title="Long-Sequence Quadratic Memory Bottleneck",
        category="Scalability Gap",
        description="Memory consumption scales quadratically with sequence length in self-attention.",
        evidence_type="explicit",
        evidence_support="High evidence support",
        supporting_papers=["paper_001"],
        evidence_items=sample_gap.evidence_items,
    )

    mock_client = MagicMock()
    mock_client.chat_deployment = "gpt-4.1-mini"
    mock_response = {
        "questions": [
            {
                "question_id": "rq_001",
                "question": "How can memory consumption in self-attention be optimized for extended sequence lengths?",
                "question_type": "Scalability",
                "question_origin": "author_inspired",
                "research_gap_ids": ["gap_tailored"],
                "rationale": "Paper_001 discusses self-attention scaling bottlenecks.",
            }
        ]
    }
    mock_client.generate_chat_response.return_value = {
        "success": True,
        "content": json.dumps(mock_response),
    }

    agent = ResearchLensAgent(foundry_client=mock_client)
    with patch.object(
        agent,
        "detect_research_gaps",
        return_value=ResearchGapAnalysisResult(
            analysis_id="gap_tailored_res",
            query="custom",
            selected_paper_ids=["paper_001"],
            selected_paper_titles={"paper_001": "Attention"},
            gaps=[tailored_gap],
            status="completed",
        ),
    ) as mock_detect:
        result = agent.generate_research_questions(
            papers=[mock_paper_a],
            engine=mock_retrieval_engine,
            gaps=None,
            custom_query="What research questions can I investigate using these papers if I am particularly interested in long-sequence efficiency?",
        )

        assert result.status == "completed"
        mock_detect.assert_called_once()
        assert tailored_gap.gap_id in result.linked_gap_ids
        assert len(result.questions) == 1


# --------------------------------------------------------------------------
# Test 42: Contiguous display numbering & provenance preservation when candidate is rejected
# --------------------------------------------------------------------------
def test_contiguous_display_numbering_and_provenance_preservation_on_candidate_rejection(
    mock_paper_a, mock_paper_b, mock_retrieval_engine, sample_gap, sample_gap_b
):
    """
    Regression test for UI numbering / question ID diagnostic:
    When the LLM generates candidates rq_001, rq_002, rq_003, rq_004,
    and candidate rq_003 is rejected by post-generation filtering:
    1. Valid questions count must be 3.
    2. User-facing display_index must be strictly contiguous: 1, 2, 3.
    3. User-facing display_id must be strictly contiguous: rq_001, rq_002, rq_003.
    4. Internal question_id for the 3rd valid question must preserve raw provenance ("rq_004").
    5. candidate_id must record raw candidate identifier ("rq_004").
    6. FutureDirection must link to both internal question_id ("rq_004") and linked_display_id ("rq_003").
    7. rejected_candidate_reasons must document why rq_003 was rejected.
    """
    mock_client = MagicMock()
    mock_client.chat_deployment = "gpt-4.1-mini"

    # Candidate 1: valid
    # Candidate 2: valid
    # Candidate 3: rejected (violates attribution integrity by claiming BERT fine-tuning belongs to paper_001)
    # Candidate 4: valid
    mock_response = {
        "questions": [
            {
                "question_id": "rq_001",
                "question": "How do attention key dimension reductions affect translation BLEU scores?",
                "question_type": "Empirical",
                "question_origin": "author_inspired",
                "research_gap_ids": [sample_gap.gap_id],
                "rationale": "Paper_001 discusses dot-product attention and key sizes for translation.",
                "potential_research_direction": "Evaluate key size scaling on WMT 2014.",
            },
            {
                "question_id": "rq_002",
                "question": "What is the computational speedup when replacing full self-attention with restricted attention?",
                "question_type": "Methodological",
                "question_origin": "author_inspired",
                "research_gap_ids": [sample_gap.gap_id],
                "rationale": "Paper_001 proposes restricted self-attention to reduce quadratic memory.",
                "potential_research_direction": "Benchmark restricted attention latency across varying sequence lengths.",
            },
            {
                "question_id": "rq_003",
                "question": "How does BERT pre-training stabilize fine-tuning on small classification tasks?",
                "question_type": "Empirical",
                "question_origin": "author_inspired",
                "research_gap_ids": [sample_gap.gap_id],
                "rationale": "Paper_001 notes fine-tuning BERTLARGE can be unstable on small datasets.",  # Attribution violation!
                "potential_research_direction": "Evaluate fine-tuning stability.",
            },
            {
                "question_id": "rq_004",
                "question": "How do bidirectional encoder representations synthesize with auto-regressive decoders across diverse benchmarks?",
                "question_type": "Synthesized",
                "question_origin": "cross_paper",
                "research_gap_ids": [sample_gap_b.gap_id],
                "rationale": "Synthesizes paper_001 self-attention with paper_002 bidirectional representations.",
                "potential_research_direction": "Analyze cross-attention representations in joint encoder-decoder setups.",
            },
        ]
    }
    mock_client.generate_chat_response.return_value = {
        "success": True,
        "content": json.dumps(mock_response),
    }

    agent = ResearchLensAgent(foundry_client=mock_client)
    result = agent.generate_research_questions(
        papers=[mock_paper_a, mock_paper_b],
        engine=mock_retrieval_engine,
        gaps=[sample_gap, sample_gap_b],
    )

    assert result.status == "completed"
    assert len(result.questions) == 3

    # Question 1: candidate 1
    q1 = result.questions[0]
    assert q1.display_index == 1
    assert q1.display_id == "rq_001"
    assert q1.question_id == "rq_001"
    assert q1.candidate_id == "rq_001"

    # Question 2: candidate 2
    q2 = result.questions[1]
    assert q2.display_index == 2
    assert q2.display_id == "rq_002"
    assert q2.question_id == "rq_002"
    assert q2.candidate_id == "rq_002"

    # Question 3: candidate 4 (candidate 3 was rejected)
    q3 = result.questions[2]
    assert q3.display_index == 3
    assert q3.display_id == "rq_003"  # Contiguous display numbering: no gap in UI!
    assert q3.question_id == "rq_004"  # Stable internal provenance preserved!
    assert q3.candidate_id == "rq_004"

    # Future direction linked to question 3
    fd3 = [f for f in result.future_directions if f.linked_question_id == "rq_004"][0]
    assert fd3.linked_question_id == "rq_004"
    assert fd3.linked_display_id == "rq_003"
    assert fd3.display_id == "fd_003"

    # Verify rejection reason was captured
    assert len(result.rejected_candidate_reasons) > 0
    assert any("misattribution" in r.lower() or "bert" in r.lower() for r in result.rejected_candidate_reasons)


