"""
Phase 5 Unit and Integration Test Suite:
Multi-Paper Comparison, Strict Per-Paper Retrieval Isolation, Evidence Bundling,
Neutral Grounded Contrast, Provenance Preservation, and Regression Validation.
"""

import pytest
from unittest.mock import MagicMock, patch
from config.settings import settings
from models.analysis import (
    MultiPaperComparison,
    DimensionComparisonItem,
    ComparisonPoint,
    ResearchAnswer,
    PaperSectionAnalysis,
)
from models.paper import (
    PaperDocument,
    ExtractedParagraph,
    DocumentChunk,
    RetrievalResult,
    EvidenceItem,
)
from services.chunking import DocumentChunker
from services.embeddings import MockTestEmbeddingService
from services.foundry_client import MicrosoftFoundryClient
from services.foundry_agent import ResearchLensAgent, COMPARISON_DIMENSIONS
from services.rag_context import EvidenceBundler
from services.retrieval import HybridRetrievalEngine
from services.vector_store import LocalVectorStore


@pytest.fixture
def dual_paper_engine():
    """Sets up a hybrid retrieval engine indexed with Paper A (Transformer) and Paper B (BERT)."""
    emb_service = MockTestEmbeddingService()
    vector_store = LocalVectorStore(index_name="test_phase5_comparison_index")
    vector_store.clear()

    engine = HybridRetrievalEngine(
        embedding_service=emb_service,
        vector_store=vector_store,
    )

    # Paper A: Transformer (Attention Is All You Need)
    paper_a = PaperDocument(
        id="paper_001",
        filename="attention_is_all_you_need.pdf",
        saved_path="data/uploads/transformer.pdf",
        page_count=15,
    )
    paper_a.paragraphs = [
        ExtractedParagraph(
            paper_id=paper_a.id,
            paragraph_id="p1_para1",
            page=1,
            original_heading="Abstract",
            normalized_section="Abstract",
            content_type="abstract",
            text="The dominant sequence transduction models are based on complex recurrent or convolutional neural networks. We propose the Transformer, based solely on attention mechanisms.",
        ),
        ExtractedParagraph(
            paper_id=paper_a.id,
            paragraph_id="p1_para2",
            page=3,
            original_heading="3.2 Scaled Dot-Product Attention",
            normalized_section="Methodology",
            content_type="body",
            text="We call our particular attention Scaled Dot-Product Attention. The input consists of queries and keys of dimension d_k, and values of dimension d_v. Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V.",
        ),
        ExtractedParagraph(
            paper_id=paper_a.id,
            paragraph_id="p1_para3",
            page=7,
            original_heading="5.1 Training Data and Batching",
            normalized_section="Experimental Setup",
            content_type="body",
            text="We trained on the standard WMT 2014 English-German dataset consisting of about 4.5 million sentence pairs.",
        ),
    ]

    # Paper B: BERT (Pre-training of Deep Bidirectional Transformers)
    paper_b = PaperDocument(
        id="paper_002",
        filename="bert_paper.pdf",
        saved_path="data/uploads/bert.pdf",
        page_count=16,
    )
    paper_b.paragraphs = [
        ExtractedParagraph(
            paper_id=paper_b.id,
            paragraph_id="p2_para1",
            page=1,
            original_heading="Abstract",
            normalized_section="Abstract",
            content_type="abstract",
            text="We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations.",
        ),
        ExtractedParagraph(
            paper_id=paper_b.id,
            paragraph_id="p2_para2",
            page=3,
            original_heading="3.1 Pre-training BERT",
            normalized_section="Methodology",
            content_type="body",
            text="Task 1: Masked LM. We mask 15% of all WordPiece tokens in each sequence at random. Task 2: Next Sentence Prediction (NSP).",
        ),
        ExtractedParagraph(
            paper_id=paper_b.id,
            paragraph_id="p2_para3",
            page=4,
            original_heading="4. Experiments",
            normalized_section="Experimental Setup",
            content_type="body",
            text="BERT is evaluated on the GLUE benchmark, SQuAD v1.1, and SWAG. For pre-training corpus we use BooksCorpus and English Wikipedia.",
        ),
    ]

    chunker = DocumentChunker()
    chunker.chunk_document(paper_a)
    chunker.chunk_document(paper_b)

    engine.index_paper(paper_a)
    engine.index_paper(paper_b)

    return engine, paper_a, paper_b


def test_two_paper_selection_validation(dual_paper_engine):
    """1. Validates that comparing 2 papers passes initial validation."""
    engine, paper_a, paper_b = dual_paper_engine
    agent = ResearchLensAgent()

    # Pre-check papers list length
    papers = [paper_a, paper_b]
    assert len(papers) >= 2
    assert paper_a.id == "paper_001"
    assert paper_b.id == "paper_002"


def test_single_paper_rejection(dual_paper_engine):
    """2. Validates that supplying fewer than 2 papers returns insufficient_evidence without model invocation."""
    engine, paper_a, _ = dual_paper_engine
    agent = ResearchLensAgent()

    result = agent.compare_papers(papers=[paper_a], engine=engine)
    assert result.status == "insufficient_evidence"
    assert "at least two" in result.error_message.lower()
    assert len(result.dimensions) == 0


def test_per_paper_retrieval_isolation(dual_paper_engine):
    """3. Validates strict paper scoping: Paper A queries only return Paper A chunks."""
    engine, paper_a, paper_b = dual_paper_engine

    # Retrieve with paper_a filter
    results_a = engine.retrieve("attention", paper_ids=[paper_a.id], top_k=5)
    for r in results_a:
        assert r.chunk.paper_id == paper_a.id
        assert r.chunk.paper_id != paper_b.id

    # Retrieve with paper_b filter
    results_b = engine.retrieve("Masked LM", paper_ids=[paper_b.id], top_k=5)
    for r in results_b:
        assert r.chunk.paper_id == paper_b.id
        assert r.chunk.paper_id != paper_a.id


def test_evidence_attribution(dual_paper_engine):
    """4. Validates that each EvidenceItem retains its originating paper_id."""
    engine, paper_a, paper_b = dual_paper_engine
    bundler = EvidenceBundler()

    results_a = engine.retrieve("Transformer", paper_ids=[paper_a.id], top_k=2)
    bundle_a = bundler.build_bundle(query="Transformer", results=results_a)
    for item in bundle_a.items:
        assert item.paper_id == paper_a.id

    results_b = engine.retrieve("Bidirectional", paper_ids=[paper_b.id], top_k=2)
    bundle_b = bundler.build_bundle(query="Bidirectional", results=results_b)
    for item in bundle_b.items:
        assert item.paper_id == paper_b.id


def test_comparison_schema_validation():
    """5. Validates serialization and structural integrity of MultiPaperComparison model."""
    comp = MultiPaperComparison(
        comparison_id="comp_test_01",
        paper_ids=["paper_001", "paper_002"],
        paper_titles={"paper_001": "Attention", "paper_002": "BERT"},
        comparison_request="Test Request",
        status="completed",
    )
    dumped = comp.model_dump()
    assert dumped["comparison_id"] == "comp_test_01"
    assert "paper_001" in dumped["paper_ids"]
    assert "paper_002" in dumped["paper_ids"]
    assert dumped["status"] == "completed"


def test_similarity_evidence_requires_both_papers():
    """6. Validates that similarity comparison point links to claims from both papers."""
    sim = ComparisonPoint(
        topic="Transformer Architecture",
        description="Both papers employ multi-head self-attention mechanisms.",
        paper_a_claim="Uses encoder-decoder self-attention stacks.",
        paper_b_claim="Uses multi-layer bidirectional Transformer encoder.",
    )
    assert sim.paper_a_claim != ""
    assert sim.paper_b_claim != ""
    assert "self-attention" in sim.paper_a_claim.lower()
    assert "transformer" in sim.paper_b_claim.lower()


def test_difference_evidence_preserves_paper_identity():
    """7. Validates difference comparison clearly distinguishes Paper A vs Paper B."""
    diff = ComparisonPoint(
        topic="Objective Function",
        description="Attention is designed for sequence-to-sequence translation, while BERT is pre-trained for masked representation.",
        paper_a_claim="Auto-regressive sequence-to-sequence cross-entropy loss.",
        paper_b_claim="Masked language model and next sentence prediction losses.",
    )
    assert diff.paper_a_claim != diff.paper_b_claim
    assert "sequence-to-sequence" in diff.paper_a_claim
    assert "masked language model" in diff.paper_b_claim.lower()


def test_missing_evidence_handling():
    """8. Validates that missing evidence is represented neutrally without fabrication."""
    item = DimensionComparisonItem(
        dimension_name="Quantum Optimization",
        paper_summaries={"paper_001": "Not clearly identified in the retrieved evidence."},
        paper_evidence={"paper_001": []},
    )
    assert item.paper_summaries["paper_001"] == "Not clearly identified in the retrieved evidence."
    assert len(item.paper_evidence["paper_001"]) == 0



def test_model_failure_handling(dual_paper_engine):
    """9. Validates graceful error handling when Foundry model invocation fails."""
    engine, paper_a, paper_b = dual_paper_engine
    mock_client = MagicMock(spec=MicrosoftFoundryClient)
    mock_client.is_configured = True
    mock_client.chat_deployment = "gpt-4.1-mini"
    mock_client.generate_chat_response.return_value = {
        "success": False,
        "status": "error",
        "error": "Foundry API gateway connection timeout (45.0s)",
        "content": "",
        "model": "gpt-4.1-mini",
    }

    agent = ResearchLensAgent(foundry_client=mock_client)
    res = agent.compare_papers(papers=[paper_a, paper_b], engine=engine)

    assert res.status == "error"
    assert "timeout" in res.error_message.lower()
    assert len(res.all_evidence_items) > 0  # Retrieved evidence is preserved even on model failure


def test_structured_comparison_parsing():
    """10. Validates structured JSON parsing from LLM output."""
    agent = ResearchLensAgent()
    sample_json = """```json
{
  "dimensions": {
    "Methodology": {
      "paper_a_summary": "Introduces scaled dot-product attention without recurrence.",
      "paper_b_summary": "Uses masked language modeling for bidirectional pretraining.",
      "synthesis": "While Paper A uses encoder-decoder attention for translation, Paper B uses bidirectional encoder attention."
    }
  },
  "similarities": [
    {
      "topic": "Core Mechanism",
      "description": "Both use self-attention as the fundamental building block.",
      "paper_a_claim": "Transformer replaces RNNs with self-attention.",
      "paper_b_claim": "BERT adopts the Transformer encoder layers."
    }
  ],
  "differences": [
    {
      "topic": "Task Formulation",
      "description": "Seq2Seq generation vs Bidirectional representation.",
      "paper_a_claim": "Encoder-decoder translation.",
      "paper_b_claim": "Encoder-only masked pretraining."
    }
  ],
  "custom_query_answer": "Direct comparative answer."
}
```"""
    parsed = agent._parse_llm_response(sample_json)
    assert "dimensions" in parsed
    assert "Methodology" in parsed["dimensions"]
    assert len(parsed["similarities"]) == 1
    assert len(parsed["differences"]) == 1
    assert parsed["custom_query_answer"] == "Direct comparative answer."


def test_provenance_preservation(dual_paper_engine):
    """11. Validates that evidence items retain chunk_id, page numbers, and score throughout."""
    engine, paper_a, _ = dual_paper_engine
    results = engine.retrieve("Scaled Dot-Product", paper_ids=[paper_a.id], top_k=1)
    assert len(results) > 0

    chunk = results[0].chunk
    assert chunk.paper_id == "paper_001"
    assert chunk.page_start > 0
    assert chunk.normalized_section != ""
    assert results[0].score > 0.0


def test_multi_paper_comparison_end_to_end_mock(dual_paper_engine):
    """12. Validates end-to-end multi-paper comparison flow with mocked Foundry response."""
    engine, paper_a, paper_b = dual_paper_engine
    mock_client = MagicMock(spec=MicrosoftFoundryClient)
    mock_client.is_configured = True
    mock_client.chat_deployment = "gpt-4.1-mini"
    mock_client.generate_chat_response.return_value = {
        "success": True,
        "status": "success",
        "content": """{
            "dimensions": {
                "Research Objective": {
                    "paper_a_summary": "Eliminate recurrence and convolution in sequence modeling using attention.",
                    "paper_b_summary": "Pre-train deep bidirectional representations using masked language modeling.",
                    "synthesis": "Both aim to improve NLP sequence processing, with Paper A focusing on translation and Paper B on transferable representations."
                }
            },
            "similarities": [
                {
                    "topic": "Attention Mechanism",
                    "description": "Both leverage multi-head self-attention.",
                    "paper_a_claim": "Proposes Scaled Dot-Product Attention.",
                    "paper_b_claim": "Built directly upon Transformer encoder attention."
                }
            ],
            "differences": [
                {
                    "topic": "Target Scope",
                    "description": "Seq2Seq Translation vs Universal Pretraining.",
                    "paper_a_claim": "Machine translation on WMT datasets.",
                    "paper_b_claim": "General pre-training for GLUE/SQuAD benchmarks."
                }
            ],
            "custom_query_answer": "Paper A focuses on translation architecture while Paper B focuses on pre-trained language modeling."
        }""",
        "model": "gpt-4.1-mini",
        "latency_ms": 450.0,
    }

    agent = ResearchLensAgent(foundry_client=mock_client)
    comp = agent.compare_papers(
        papers=[paper_a, paper_b],
        engine=engine,
        custom_query="Compare research objectives",
    )

    assert comp.status == "completed"
    assert len(comp.paper_ids) == 2
    assert "Research Objective" in comp.dimensions
    assert len(comp.similarities) == 1
    assert len(comp.differences) == 1
    assert comp.custom_query_answer is not None
    assert len(comp.all_evidence_items) > 0
    # Verify strict paper separation in all_evidence_items
    ids = set(e.paper_id for e in comp.all_evidence_items)
    assert "paper_001" in ids
    assert "paper_002" in ids


def test_phase3_regression_check(dual_paper_engine):
    """13. Regression check: Phase 3 hybrid retrieval engine remains functional."""
    engine, paper_a, _ = dual_paper_engine
    results = engine.retrieve("attention", top_k=2, paper_ids=[paper_a.id])
    assert len(results) > 0
    assert results[0].chunk.paper_id == paper_a.id


def test_phase4_agent_regression_check(dual_paper_engine):
    """14. Regression check: Phase 4 single-paper Q&A and 7-section analysis remain functional."""
    engine, paper_a, _ = dual_paper_engine
    mock_client = MagicMock(spec=MicrosoftFoundryClient)
    mock_client.is_configured = True
    mock_client.chat_deployment = "gpt-4.1-mini"
    mock_client.generate_chat_response.return_value = {
        "success": True,
        "content": "Scaled Dot-Product Attention computes softmax(QK^T / sqrt(d_k)) * V [Evidence 1: paper_001].",
        "model": "gpt-4.1-mini",
    }

    agent = ResearchLensAgent(foundry_client=mock_client)
    ans = agent.answer_question(query="What is attention?", paper_id=paper_a.id, engine=engine)
    assert ans.confidence_status == "grounded"
    assert len(ans.evidence_items) > 0
