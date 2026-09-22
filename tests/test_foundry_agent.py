"""
Phase 4 Unit and Integration Test Suite:
Microsoft Foundry Client, Application-Level ResearchLensAgent, Evidence Bundling,
Strict Paper Scoping, Zero-Hallucination Guardrails, and Verifiable Provenance.
"""

import pytest
from unittest.mock import MagicMock, patch
from config.settings import settings
from models.analysis import ResearchAnswer, AnalysisSectionItem, PaperSectionAnalysis
from models.paper import (
    PaperDocument,
    ExtractedParagraph,
    DocumentChunk,
    RetrievalResult,
)
from services.chunking import DocumentChunker
from services.embeddings import MockTestEmbeddingService
from services.foundry_client import FoundryClient, MicrosoftFoundryClient
from services.foundry_agent import ResearchLensAgent
from services.rag_context import EvidenceBundler
from services.retrieval import HybridRetrievalEngine
from services.vector_store import LocalVectorStore


@pytest.fixture
def multi_paper_engine():
    """Sets up a hybrid retrieval engine containing two distinct papers."""
    emb_service = MockTestEmbeddingService()
    vector_store = LocalVectorStore(index_name="test_foundry_agent_index")
    vector_store.clear()

    engine = HybridRetrievalEngine(
        embedding_service=emb_service,
        vector_store=vector_store,
    )


    # Paper A: Transformer
    paper_a = PaperDocument(
        id="paper_transformer_01",
        filename="attention_is_all_you_need.pdf",
        saved_path="data/uploads/transformer.pdf",
        page_count=15,
    )
    paper_a.paragraphs = [
        ExtractedParagraph(
            paper_id=paper_a.id,
            paragraph_id="t_p1",
            page=1,
            original_heading="Abstract",
            normalized_section="Abstract",
            content_type="abstract",
            text="The Transformer is an architecture based solely on attention mechanisms.",
        ),
        ExtractedParagraph(
            paper_id=paper_a.id,
            paragraph_id="t_p2",
            page=3,
            original_heading="3.2 Attention",
            normalized_section="Methodology",
            content_type="body",
            text="Scaled Dot-Product Attention computes softmax(QK^T / sqrt(d_k)) * V.",
        ),
    ]

    # Paper B: BERT
    paper_b = PaperDocument(
        id="paper_bert_02",
        filename="bert_paper.pdf",
        saved_path="data/uploads/bert.pdf",
        page_count=16,
    )
    paper_b.paragraphs = [
        ExtractedParagraph(
            paper_id=paper_b.id,
            paragraph_id="b_p1",
            page=1,
            original_heading="Abstract",
            normalized_section="Abstract",
            content_type="abstract",
            text="BERT stands for Bidirectional Encoder Representations from Transformers.",
        ),
        ExtractedParagraph(
            paper_id=paper_b.id,
            paragraph_id="b_p2",
            page=4,
            original_heading="3.1 Masked LM",
            normalized_section="Methodology",
            content_type="body",
            text="Masked Language Model randomly masks 15 percent of input tokens to train bidirectional representations.",
        ),
    ]

    chunker = DocumentChunker()
    chunker.chunk_document(paper_a)
    chunker.chunk_document(paper_b)


    engine.index_paper(paper_a)
    engine.index_paper(paper_b)

    return engine, paper_a, paper_b


def test_foundry_configuration_validation():
    """1. Validates that settings properly expose required Foundry configuration."""
    assert hasattr(settings, "AZURE_OPENAI_ENDPOINT")
    assert hasattr(settings, "AZURE_OPENAI_API_KEY")
    assert hasattr(settings, "AZURE_OPENAI_CHAT_DEPLOYMENT")
    assert settings.AZURE_OPENAI_CHAT_DEPLOYMENT == "gpt-4.1-mini"
    assert "openai.azure.com" in settings.AZURE_OPENAI_ENDPOINT or settings.AZURE_OPENAI_ENDPOINT == ""


def test_foundry_service_initialization():
    """2. Validates that FoundryClient and ResearchLensAgent initialize and return status."""
    client = FoundryClient()
    agent = ResearchLensAgent(client=client)

    status = agent.get_connection_status()
    assert "configured" in status
    assert "endpoint" in status
    assert "deployment" in status
    assert status["deployment"] == "gpt-4.1-mini"
    assert "model_attribution" in status
    assert "gpt-4.1-mini" in status["model_attribution"]


def test_missing_credentials_handling():
    """3. Validates graceful unconfigured status when credentials are empty."""
    client = FoundryClient(endpoint="", api_key="")
    assert not client.is_configured
    status = client.get_configuration_status()
    assert status["configured"] is False
    assert "Missing" in status["message"]

    # Invocation should return unconfigured dictionary without unhandled exception
    res = client.generate_chat_response(system_prompt="Test", user_prompt="Test")
    assert res["status"] == "unconfigured" or res["success"] is False
    assert "not configured" in res.get("content", "").lower() or "not configured" in res.get("error", "").lower()


def test_evidence_bundle_formatting():
    """4. Validates that EvidenceBundler produces structured grounded context."""
    bundler = EvidenceBundler(default_token_budget=1000)
    chunk = DocumentChunk(
        chunk_id="chunk_test_01",
        paper_id="paper_test",
        chunk_index=0,
        text="The Adam optimizer uses learning rate warmup steps.",
        token_count=10,
        page_start=5,
        page_end=5,
        normalized_section="Experimental Setup",
    )
    result = RetrievalResult(
        chunk=chunk,
        score=0.88,
        bm25_score=1.2,
        dense_score=0.95,
        match_type="hybrid",
    )

    bundle = bundler.build_bundle(
        query="warmup steps",
        results=[result],
    )

    assert bundle.total_found == 1
    assert len(bundle.items) == 1
    assert "RETRIEVED RESEARCH EVIDENCE" in bundle.formatted_context
    assert "chunk_test_01" in bundle.formatted_context
    assert "p. 5" in bundle.formatted_context
    assert "The Adam optimizer uses learning rate warmup steps" in bundle.formatted_context


def test_structured_response_parsing():
    """5. Validates parsing structured model responses into ResearchAnswer."""
    agent = ResearchLensAgent()
    json_content = """```json
{
  "answer": "Scaled Dot-Product Attention operates over queries, keys, and values [Chunk t_p2].",
  "is_explicit": true,
  "citation_labels": ["[Chunk t_p2]"]
}
```"""
    parsed = agent._parse_llm_response(json_content)
    assert "Scaled Dot-Product Attention" in parsed["answer"]
    assert parsed["is_explicit"] is True
    assert "[Chunk t_p2]" in parsed["citation_labels"]


def test_provenance_preservation(multi_paper_engine):
    """6. Validates full provenance retention across retrieval and answer synthesis."""
    engine, paper_a, _ = multi_paper_engine
    agent = ResearchLensAgent()

    results = engine.retrieve("attention mechanism", top_k=2, paper_ids=[paper_a.id])
    assert len(results) > 0

    bundle = agent.bundler.build_bundle(query="attention mechanism", results=results)
    assert len(bundle.items) == len(results)

    first_item = bundle.items[0]
    assert first_item.paper_id == paper_a.id
    assert first_item.chunk_id == results[0].chunk.chunk_id
    assert first_item.page_start == results[0].chunk.page_start
    assert first_item.page_end == results[0].chunk.page_end
    assert first_item.normalized_section == results[0].chunk.normalized_section
    assert first_item.score == pytest.approx(results[0].score, rel=1e-3)
    assert first_item.text == results[0].chunk.text


def test_paper_isolation_in_agent(multi_paper_engine):
    """7. Validates strict paper scoping: queries to Paper A never retrieve Paper B."""
    engine, paper_a, paper_b = multi_paper_engine
    agent = ResearchLensAgent()

    # Query Paper A for BERT concepts with strict paper filter
    results_a = engine.retrieve("Masked Language Model", top_k=5, paper_ids=[paper_a.id])
    for r in results_a:
        assert r.chunk.paper_id == paper_a.id
        assert r.chunk.paper_id != paper_b.id

    # Query Paper B for Transformer concepts with strict paper filter
    results_b = engine.retrieve("Scaled Dot-Product Attention", top_k=5, paper_ids=[paper_b.id])
    for r in results_b:
        assert r.chunk.paper_id == paper_b.id
        assert r.chunk.paper_id != paper_a.id


def test_no_evidence_handling(multi_paper_engine):
    """8. Validates zero-hallucination guardrail when no relevant evidence exists."""
    engine, paper_a, _ = multi_paper_engine
    agent = ResearchLensAgent()

    with patch.object(engine, "retrieve", return_value=[]):
        response: ResearchAnswer = agent.answer_question(
            query="Quantum teleportation in black holes",
            paper_id=paper_a.id,
            engine=engine,
        )

        assert response.confidence_status == "insufficient_evidence"
        assert "I could not find sufficient evidence" in response.answer
        assert len(response.evidence_items) == 0


def test_model_failure_handling(multi_paper_engine):
    """9. Validates graceful error handling when LLM invocation throws an error."""
    engine, paper_a, _ = multi_paper_engine
    mock_client = MagicMock(spec=MicrosoftFoundryClient)
    mock_client.is_configured = True
    mock_client.chat_deployment = "gpt-4.1-mini"
    mock_client.generate_chat_response.return_value = {
        "success": False,
        "status": "error",
        "error": "Microsoft Foundry request failed: Connection timed out after 45.0s",
        "content": "",
        "model": "gpt-4.1-mini",
        "latency_ms": 45000.0,
    }

    agent = ResearchLensAgent(foundry_client=mock_client)
    response: ResearchAnswer = agent.answer_question(
        query="Attention mechanism",
        paper_id=paper_a.id,
        engine=engine,
    )

    assert response.confidence_status == "error"
    assert "failed" in response.answer.lower()
    assert len(response.evidence_items) > 0  # Evidence was retrieved even if generation failed


def test_end_to_end_mocked_foundry_response(multi_paper_engine):
    """10. Validates end-to-end question answering pipeline with mocked Foundry response."""
    engine, paper_a, _ = multi_paper_engine
    mock_client = MagicMock(spec=MicrosoftFoundryClient)
    mock_client.is_configured = True
    mock_client.chat_deployment = "gpt-4.1-mini"
    mock_client.generate_chat_response.return_value = {
        "success": True,
        "status": "success",
        "content": "Scaled Dot-Product Attention computes softmax(QK^T / sqrt(d_k)) * V [Evidence 1: paper_transformer_01, pp. 3-3, Methodology | Chunk t_p2].",
        "model": "gpt-4.1-mini",
        "latency_ms": 280.0,
    }

    agent = ResearchLensAgent(foundry_client=mock_client)
    response: ResearchAnswer = agent.answer_question(
        query="What is the equation for Scaled Dot-Product Attention?",
        paper_id=paper_a.id,
        engine=engine,
    )

    assert response.confidence_status == "grounded"
    assert "Scaled Dot-Product Attention computes" in response.answer
    assert len(response.evidence_items) > 0
    assert response.evidence_items[0].paper_id == paper_a.id
    assert response.model_used == "gpt-4.1-mini"
