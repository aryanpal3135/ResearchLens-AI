"""
Tests for Phase 6: Research Gap Detection.
Covers:
1. Gap schema validation
2. Explicit gap parsing
3. Evidence-derived gap parsing
4. Cross-paper gap attribution
5. Paper isolation in retrieval
6. Evidence provenance preservation
7. Missing evidence handling
8. No false absence claims
9. Evidence-support validation (categorical, no fake percentages)
10. Contradiction detection ("Potentially conflicting evidence")
11. Future-direction separation (AI suggestion clearly partitioned)
12. Model failure handling
13. Structured Foundry response parsing
14. End-to-end mocked gap detection
15. Single-paper gap detection support
16. Phase 3 retrieval regression check
17. Phase 4 agent regression check
18. Phase 5 comparison regression check / No ranking/winner language
"""

import json
import pytest
from unittest.mock import MagicMock, patch

from models.paper import PaperDocument, PaperMetadata, DocumentChunk, EvidenceItem
from models.research_gap import (
    ResearchGap,
    ResearchGapAnalysisResult,
    GAP_CATEGORIES,
)
from services.foundry_agent import ResearchLensAgent, SYSTEM_PROMPT_GAP_AGENT
from services.retrieval import HybridRetrievalEngine
from services.gap_detector import ResearchGapDetector


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
def mock_retrieval_engine(mock_paper_a, mock_paper_b) -> HybridRetrievalEngine:
    from models.paper import RetrievalResult
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


# 1. Gap schema validation
def test_gap_schema_validation():
    gap = ResearchGap(
        gap_id="gap_001",
        title="Quadratic Complexity for Long Sequences",
        category="Methodological Gap",
        description="Quadratic self-attention scaling limits sequence lengths to short windows.",
        evidence_type="explicit",
        evidence_type_label="Explicitly stated by the paper",
        evidence_support="High evidence support",
        supporting_papers=["paper_001"],
        rationale="Author explicitly mentions quadratic complexity in limitations.",
        affected_dimension="Architecture",
        potential_research_direction="Explore sub-quadratic linear or sparse attention mechanisms.",
    )
    assert gap.gap_id == "gap_001"
    assert gap.category in GAP_CATEGORIES
    assert gap.evidence_support == "High evidence support"
    assert gap.is_explicit is False  # Default unless set or parsed
    assert gap.potential_research_direction is not None


# 2. Explicit gap parsing
def test_explicit_gap_parsing(mock_paper_a, mock_retrieval_engine):
    mock_client = MagicMock()
    mock_client.chat_deployment = "gpt-4.1-mini"
    mock_client.generate_chat_response.return_value = {
        "success": True,
        "content": json.dumps({
            "gaps": [
                {
                    "gap_id": "gap_001",
                    "title": "Quadratic Memory Bottleneck",
                    "category": "Scalability Gap",
                    "description": "Quadratic self-attention complexity restricts scaling to long contexts.",
                    "evidence_type": "explicit",
                    "evidence_support": "High evidence support",
                    "supporting_papers": ["paper_001"],
                    "citation_labels": ["[Evidence A1: paper_001, p. 8, Limitations]"],
                    "rationale": "Author directly discusses memory constraints in section 5.",
                    "affected_dimension": "Architecture",
                    "potential_research_direction": "Apply factorized attention kernels."
                }
            ]
        })
    }

    agent = ResearchLensAgent(foundry_client=mock_client)
    res = agent.detect_research_gaps(papers=[mock_paper_a], engine=mock_retrieval_engine)

    assert res.status == "completed"
    assert len(res.gaps) == 1
    assert res.gaps[0].evidence_type == "explicit"
    assert res.gaps[0].evidence_type_label == "Explicitly stated by the paper"
    assert res.gaps[0].is_explicit is True
    assert res.summary_counts["explicit_gaps"] == 1


# 3. Evidence-derived gap parsing
def test_evidence_derived_gap_parsing(mock_paper_a, mock_retrieval_engine):
    mock_client = MagicMock()
    mock_client.chat_deployment = "gpt-4.1-mini"
    mock_client.generate_chat_response.return_value = {
        "success": True,
        "content": json.dumps({
            "gaps": [
                {
                    "gap_id": "gap_002",
                    "title": "Limited Translation Benchmark Scope",
                    "category": "Evaluation Gap",
                    "description": "Evaluation is documented only on WMT 2014 English-to-German and English-to-French. No evidence of low-resource languages was identified in retrieved passages.",
                    "evidence_type": "evidence_derived",
                    "evidence_support": "Moderate evidence support",
                    "supporting_papers": ["paper_001"],
                    "rationale": "Retrieved evaluation tables reflect only high-resource pairs.",
                    "affected_dimension": "Evaluation",
                    "potential_research_direction": "Benchmark across typologically diverse low-resource languages."
                }
            ]
        })
    }

    agent = ResearchLensAgent(foundry_client=mock_client)
    res = agent.detect_research_gaps(papers=[mock_paper_a], engine=mock_retrieval_engine)

    assert res.status == "completed"
    assert res.gaps[0].evidence_type == "evidence_derived"
    assert res.gaps[0].evidence_type_label == "Derived from retrieved evidence"
    assert res.gaps[0].is_explicit is False


# 4. Cross-paper gap attribution
def test_cross_paper_gap_attribution(mock_paper_a, mock_paper_b, mock_retrieval_engine):
    mock_client = MagicMock()
    mock_client.chat_deployment = "gpt-4.1-mini"
    mock_client.generate_chat_response.return_value = {
        "success": True,
        "content": json.dumps({
            "gaps": [
                {
                    "gap_id": "gap_003",
                    "title": "Absence of Unified Multimodal Pre-training Evaluation",
                    "category": "Cross-Paper Gap",
                    "description": "Paper A focuses on sequence transduction for text translation, while Paper B focuses on bidirectional text representation. Neither study evaluates joint text-vision pre-training.",
                    "evidence_type": "cross_paper",
                    "evidence_support": "High evidence support",
                    "supporting_papers": ["paper_001", "paper_002"],
                    "rationale": "Both studies explicitly state multimodal and longer context as future directions.",
                    "potential_research_direction": "Construct a joint vision-language masked pre-training objective."
                }
            ]
        })
    }

    agent = ResearchLensAgent(foundry_client=mock_client)
    res = agent.detect_research_gaps(papers=[mock_paper_a, mock_paper_b], engine=mock_retrieval_engine)

    assert res.gaps[0].evidence_type == "cross_paper"
    assert "paper_001" in res.gaps[0].supporting_papers
    assert "paper_002" in res.gaps[0].supporting_papers
    assert res.summary_counts["cross_paper_gaps"] == 1


# 5. Paper isolation in retrieval
def test_per_paper_retrieval_isolation(mock_paper_a, mock_paper_b):
    isolated_engine = MagicMock(spec=HybridRetrievalEngine)
    isolated_engine.retrieve.return_value = []

    mock_client = MagicMock()
    mock_client.generate_chat_response.return_value = {"success": True, "content": json.dumps({"gaps": []})}

    agent = ResearchLensAgent(foundry_client=mock_client)
    agent.detect_research_gaps(papers=[mock_paper_a, mock_paper_b], engine=isolated_engine)

    # Check that retrieve was called with explicit single paper_ids
    for call_args in isolated_engine.retrieve.call_args_list:
        p_ids = call_args.kwargs.get("paper_ids")
        assert p_ids is not None
        assert len(p_ids) == 1
        assert p_ids[0] in ["paper_001", "paper_002"]


# 6. Evidence provenance preservation
def test_evidence_provenance_preservation(mock_paper_a, mock_retrieval_engine):
    mock_client = MagicMock()
    mock_client.chat_deployment = "gpt-4.1-mini"
    mock_client.generate_chat_response.return_value = {
        "success": True,
        "content": json.dumps({
            "gaps": [
                {
                    "gap_id": "gap_001",
                    "title": "Scaling Limitations",
                    "category": "Scalability Gap",
                    "description": "Quadratic self-attention scaling restricts longer documents.",
                    "evidence_type": "explicit",
                    "evidence_support": "High evidence support",
                    "supporting_papers": ["paper_001"],
                    "citation_labels": ["[Evidence A1: paper_001, p. 8, Limitations]"],
                    "rationale": "Directly stated.",
                }
            ]
        })
    }

    agent = ResearchLensAgent(foundry_client=mock_client)
    res = agent.detect_research_gaps(papers=[mock_paper_a], engine=mock_retrieval_engine)

    assert len(res.gaps[0].evidence_items) > 0
    ev = res.gaps[0].evidence_items[0]
    assert ev.paper_id == "paper_001"
    assert ev.chunk_id == "paper_001_c001"
    assert ev.page_start == 8
    assert ev.score > 0
    assert "quadratic" in ev.text.lower()


# 7. Missing evidence handling
def test_missing_evidence_handling(mock_paper_a):
    empty_engine = MagicMock(spec=HybridRetrievalEngine)
    empty_engine.retrieve.return_value = []

    mock_client = MagicMock()
    agent = ResearchLensAgent(foundry_client=mock_client)
    res = agent.detect_research_gaps(papers=[mock_paper_a], engine=empty_engine)

    assert res.status == "insufficient_evidence"
    assert len(res.gaps) == 0
    mock_client.generate_chat_response.assert_not_called()


# 8. No false absence claims
def test_no_false_absence_claims():
    bad_phrase = "The paper does not evaluate multilingual tasks."
    good_phrase = "No evidence of multilingual evaluation was identified in the retrieved passages."
    assert "no evidence" in good_phrase.lower()
    assert SYSTEM_PROMPT_GAP_AGENT is not None
    assert "No evidence of X was identified in the retrieved passages" in SYSTEM_PROMPT_GAP_AGENT


# 9. Evidence-support validation (categorical, no fake percentages)
def test_evidence_support_categorical(mock_paper_a, mock_retrieval_engine):
    mock_client = MagicMock()
    mock_client.generate_chat_response.return_value = {
        "success": True,
        "content": json.dumps({
            "gaps": [
                {
                    "gap_id": "gap_001",
                    "title": "Ablation Scope",
                    "category": "Evaluation Gap",
                    "description": "Limited ablation across optimizer hyperparameter grids.",
                    "evidence_type": "evidence_derived",
                    "evidence_support": "0.89 confidence",  # Non-standard numerical
                    "supporting_papers": ["paper_001"],
                }
            ]
        })
    }

    agent = ResearchLensAgent(foundry_client=mock_client)
    res = agent.detect_research_gaps(papers=[mock_paper_a], engine=mock_retrieval_engine)

    # Should be normalized to valid categorical support
    assert res.gaps[0].evidence_support in [
        "High evidence support",
        "Moderate evidence support",
        "Limited evidence support",
    ]


# 10. Contradiction detection ("Potentially conflicting evidence")
def test_contradiction_detection_labeling(mock_paper_a, mock_paper_b, mock_retrieval_engine):
    mock_client = MagicMock()
    mock_client.generate_chat_response.return_value = {
        "success": True,
        "content": json.dumps({
            "gaps": [
                {
                    "gap_id": "gap_004",
                    "title": "Divergent Context Conditioning",
                    "category": "Contradictory Evidence",
                    "description": "Paper A claims unidirectional autoregressive masking is necessary for sequence generation, while Paper B demonstrates bidirectional conditioning produces stronger representations for language understanding.",
                    "evidence_type": "cross_paper",
                    "evidence_support": "High evidence support",
                    "supporting_papers": ["paper_001", "paper_002"],
                    "potential_tension": "Autoregressive generation vs bidirectional representation pre-training trade-off.",
                }
            ]
        })
    }

    agent = ResearchLensAgent(foundry_client=mock_client)
    res = agent.detect_research_gaps(papers=[mock_paper_a, mock_paper_b], engine=mock_retrieval_engine)

    assert res.gaps[0].category == "Contradictory Evidence"
    assert res.gaps[0].potential_tension is not None
    assert res.summary_counts["contradictory_evidence"] == 1


# 11. Future-direction separation (AI suggestion clearly partitioned)
def test_future_direction_separation(mock_paper_a, mock_retrieval_engine):
    mock_client = MagicMock()
    mock_client.generate_chat_response.return_value = {
        "success": True,
        "content": json.dumps({
            "gaps": [
                {
                    "gap_id": "gap_005",
                    "title": "Hardware Generalization",
                    "category": "Scalability Gap",
                    "description": "Training relied exclusively on 8 NVIDIA P100 GPUs.",
                    "evidence_type": "explicit",
                    "evidence_support": "High evidence support",
                    "supporting_papers": ["paper_001"],
                    "potential_research_direction": "Evaluate training efficiency on heterogeneous consumer hardware."
                }
            ]
        })
    }

    agent = ResearchLensAgent(foundry_client=mock_client)
    res = agent.detect_research_gaps(papers=[mock_paper_a], engine=mock_retrieval_engine)

    gap = res.gaps[0]
    assert gap.potential_research_direction is not None
    assert "heterogeneous" in gap.potential_research_direction


# 12. Model failure handling
def test_model_failure_handling(mock_paper_a, mock_retrieval_engine):
    mock_client = MagicMock()
    mock_client.chat_deployment = "gpt-4.1-mini"
    mock_client.generate_chat_response.return_value = {
        "success": False,
        "error": "Execution timeout after 45 seconds",
    }

    agent = ResearchLensAgent(foundry_client=mock_client)
    res = agent.detect_research_gaps(papers=[mock_paper_a], engine=mock_retrieval_engine)

    assert res.status == "error"
    assert "timeout" in res.error_message.lower()


# 13. Structured Foundry response parsing (markdown fences)
def test_structured_foundry_response_parsing(mock_paper_a, mock_retrieval_engine):
    fenced_content = """```json
{
  "gaps": [
    {
      "gap_id": "gap_001",
      "title": "Subword Vocabulary Trade-off",
      "category": "Methodological Gap",
      "description": "Fixed 37,000 subword vocabulary limits rare token generalization.",
      "evidence_type": "evidence_derived",
      "evidence_support": "Moderate evidence support",
      "supporting_papers": ["paper_001"]
    }
  ]
}
```"""
    mock_client = MagicMock()
    mock_client.generate_chat_response.return_value = {"success": True, "content": fenced_content}

    agent = ResearchLensAgent(foundry_client=mock_client)
    res = agent.detect_research_gaps(papers=[mock_paper_a], engine=mock_retrieval_engine)

    assert res.status == "completed"
    assert len(res.gaps) == 1
    assert res.gaps[0].title == "Subword Vocabulary Trade-off"


# 14. End-to-end mocked gap detection via ResearchGapDetector service
def test_end_to_end_mocked_gap_detection(mock_paper_a, mock_paper_b, mock_retrieval_engine):
    mock_client = MagicMock()
    mock_client.generate_chat_response.return_value = {
        "success": True,
        "content": json.dumps({
            "gaps": [
                {
                    "gap_id": "gap_001",
                    "title": "Long Context Limitation",
                    "category": "Scalability Gap",
                    "description": "Both papers report token context bounds (Paper A quadratic, Paper B max 512 tokens).",
                    "evidence_type": "cross_paper",
                    "evidence_support": "High evidence support",
                    "supporting_papers": ["paper_001", "paper_002"],
                    "potential_research_direction": "Investigate recurrent memory or state-space augmentation."
                }
            ]
        })
    }

    agent = ResearchLensAgent(foundry_client=mock_client)
    detector = ResearchGapDetector(agent=agent)
    res = detector.detect_gaps_for_papers(
        papers=[mock_paper_a, mock_paper_b],
        engine=mock_retrieval_engine,
        custom_query="What long context limitations exist?",
    )

    assert res.status == "completed"
    assert len(res.gaps) == 1
    assert res.gaps[0].category == "Scalability Gap"


# 15. Single-paper gap detection support
def test_single_paper_gap_detection_support(mock_paper_a, mock_retrieval_engine):
    mock_client = MagicMock()
    mock_client.generate_chat_response.return_value = {
        "success": True,
        "content": json.dumps({
            "gaps": [
                {
                    "gap_id": "gap_001",
                    "title": "Recurrence-Free Dependency",
                    "category": "Theoretical Gap",
                    "description": "Entirely relying on attention requires explicit positional encodings.",
                    "evidence_type": "explicit",
                    "evidence_support": "High evidence support",
                    "supporting_papers": ["paper_001"]
                }
            ]
        })
    }

    agent = ResearchLensAgent(foundry_client=mock_client)
    res = agent.detect_research_gaps(papers=[mock_paper_a], engine=mock_retrieval_engine)

    assert res.status == "completed"
    assert res.selected_paper_ids == ["paper_001"]
    assert len(res.gaps) == 1


# 16. Existing Phase 3 retrieval regression check
def test_phase3_retrieval_regression():
    from services.retrieval import BM25Index, HybridRetrievalEngine
    bm25 = BM25Index()
    assert bm25.k1 == 1.5
    assert bm25.b == 0.75


# 17. Existing Phase 4 agent regression check
def test_phase4_agent_regression():
    from services.foundry_agent import ResearchLensAgent
    agent = ResearchLensAgent()
    assert hasattr(agent, "answer_question")
    assert hasattr(agent, "analyze_paper_sections")


# 18. Phase 5 comparison regression check & No ranking/winner language
def test_phase5_comparison_regression_and_no_ranking():
    from services.foundry_agent import ResearchLensAgent, SYSTEM_PROMPT_COMPARISON_AGENT
    agent = ResearchLensAgent()
    assert hasattr(agent, "compare_papers")
    assert "winner" in SYSTEM_PROMPT_COMPARISON_AGENT.lower()
    assert "Do NOT declare a winner" in SYSTEM_PROMPT_COMPARISON_AGENT
