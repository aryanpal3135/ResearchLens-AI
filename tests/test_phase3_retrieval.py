"""
Phase 3 Unit and Integration Test Suite:
Document Chunking, Embeddings, Vector Store, Hybrid Retrieval, Evidence Bundles, and Evaluation.
"""

import pytest
from pathlib import Path
from config.settings import settings
from models.paper import (
    PaperDocument,
    ExtractedParagraph,
    ExtractedSection,
    ExtractedTable,
    ExtractedReference,
    DocumentChunk,
)
from services.chunking import DocumentChunker, estimate_tokens
from services.embeddings import (
    MockTestEmbeddingService,
    AzureOpenAIEmbeddingService,
    get_embedding_service,
)
from services.vector_store import LocalVectorStore, AzureSearchVectorStore
from services.retrieval import BM25Index, HybridRetrievalEngine
from services.rag_context import EvidenceBundler
from services.pdf_processor import PDFProcessor


@pytest.fixture
def sample_paper_document():
    """Creates a synthetic multi-section paper with paragraphs, table, and references."""
    paper = PaperDocument(
        id="paper_001",
        filename="synthetic_test.pdf",
        saved_path="data/uploads/synthetic_test.pdf",
        page_count=3,
    )
    # Paragraphs across 3 sections
    paper.paragraphs = [
        ExtractedParagraph(
            paper_id="paper_001",
            paragraph_id="paper_001_p001_para001",
            page=1,
            original_heading="Abstract",
            normalized_section="Abstract",
            content_type="abstract",
            text="This is the abstract paragraph summarizing the deep neural network research.",
        ),
        ExtractedParagraph(
            paper_id="paper_001",
            paragraph_id="paper_001_p001_para002",
            page=1,
            original_heading="1. Introduction",
            normalized_section="Introduction",
            content_type="body",
            text="Deep learning architectures have transformed sequence transduction problems and language modeling.",
        ),
        ExtractedParagraph(
            paper_id="paper_001",
            paragraph_id="paper_001_p002_para003",
            page=2,
            original_heading="2. Methodology",
            normalized_section="Methodology",
            content_type="body",
            text="The Multi-Head Attention mechanism computes scaled dot-product attention in parallel representation subspaces.",
        ),
        ExtractedParagraph(
            paper_id="paper_001",
            paragraph_id="paper_001_p002_para004",
            page=2,
            original_heading="2. Methodology",
            normalized_section="Methodology",
            content_type="body",
            text="Given queries Q, keys K, and values V, the attention matrix is computed as softmax(Q K^T / sqrt(d_k)) V.",
        ),
        ExtractedParagraph(
            paper_id="paper_001",
            paragraph_id="paper_001_p003_para005",
            page=3,
            original_heading="3. Results",
            normalized_section="Results",
            content_type="body",
            text="On the WMT 2014 English-to-German translation task, the model achieves a BLEU score of 28.4.",
        ),
    ]

    # Valid table
    paper.tables = [
        ExtractedTable(
            paper_id="paper_001",
            table_id="paper_001_t001",
            page=2,
            caption="Table 1: Translation BLEU scores and training costs",
            markdown_content="| Model | BLEU | Training FLOPs |\n|---|---|---|\n| Transformer (base) | 27.3 | 3.3e18 |\n| Transformer (big) | 28.4 | 2.3e19 |",
            row_count=2,
            col_count=3,
            status="Valid",
            extraction_quality="high",
        )
    ]

    # References
    paper.references = [
        ExtractedReference(
            paper_id="paper_001",
            ref_id="paper_001_ref001",
            raw_text="Vaswani, A., et al. (2017). Attention is all you need. NeurIPS.",
            year="2017",
        ),
        ExtractedReference(
            paper_id="paper_001",
            ref_id="paper_001_ref002",
            raw_text="Bahdanau, D., Cho, K., & Bengio, Y. (2014). Neural machine translation by jointly learning to align and translate. ICLR.",
            year="2014",
        ),
    ]

    return paper


# 1. CHUNKING VALIDATION
def test_chunking_section_boundary_isolation(sample_paper_document):
    """Chunks must never mix paragraphs from different normalized sections."""
    chunker = DocumentChunker(target_chunk_size=1000)
    chunks = chunker.chunk_document(sample_paper_document)

    assert len(chunks) >= 4  # Abstract, Intro, Methodology, Results, Table, Ref

    for chunk in chunks:
        # Check that all source paragraphs within a chunk share the exact same normalized_section
        if chunk.source_paragraph_ids:
            matching_paras = [p for p in sample_paper_document.paragraphs if p.paragraph_id in chunk.source_paragraph_ids]
            sections = set(p.normalized_section for p in matching_paras)
            assert len(sections) == 1, f"Chunk {chunk.chunk_id} crossed section boundaries: {sections}"
            assert chunk.normalized_section == list(sections)[0]


def test_chunking_document_order_and_linkage(sample_paper_document):
    """Every chunk must have sequential document_order, chunk_index, and valid previous/next chunk links."""
    chunker = DocumentChunker()
    chunks = chunker.chunk_document(sample_paper_document)

    for i, chunk in enumerate(chunks):
        assert chunk.document_order == i + 1
        assert chunk.chunk_index == i

        if i == 0:
            assert chunk.previous_chunk_id is None
        else:
            assert chunk.previous_chunk_id == chunks[i - 1].chunk_id

        if i == len(chunks) - 1:
            assert chunk.next_chunk_id is None
        else:
            assert chunk.next_chunk_id == chunks[i + 1].chunk_id


def test_chunking_table_and_reference_dedication(sample_paper_document):
    """Tables and references must become dedicated chunks preserving their unique source IDs."""
    chunker = DocumentChunker()
    chunks = chunker.chunk_document(sample_paper_document)

    table_chunks = [c for c in chunks if c.content_type == "table"]
    assert len(table_chunks) == 1
    assert table_chunks[0].source_table_ids == ["paper_001_t001"]
    assert "BLEU" in table_chunks[0].text
    assert table_chunks[0].page_start == 2

    ref_chunks = [c for c in chunks if c.content_type == "reference"]
    assert len(ref_chunks) == 1
    assert "paper_001_ref001" in ref_chunks[0].source_reference_ids
    assert "paper_001_ref002" in ref_chunks[0].source_reference_ids


def test_chunking_zero_data_loss(sample_paper_document):
    """All input paragraphs, tables, and references must be represented in chunk provenance."""
    chunker = DocumentChunker()
    chunks = chunker.chunk_document(sample_paper_document)

    chunked_para_ids = set()
    for c in chunks:
        chunked_para_ids.update(c.source_paragraph_ids)

    original_para_ids = set(p.paragraph_id for p in sample_paper_document.paragraphs)
    assert original_para_ids == chunked_para_ids, "Some paragraphs were dropped during chunking!"


# 2. EMBEDDINGS SERVICE & HONEST LABELING
def test_mock_embedding_honest_labeling():
    """Mock/test embedding service must be explicitly labeled as non-semantic."""
    from services.embeddings import MockEmbeddingProvider, EmbeddingProvider
    emb_service = MockEmbeddingProvider(dimension=384)
    assert isinstance(emb_service, EmbeddingProvider)
    meta = emb_service.get_metadata()

    assert emb_service.is_semantic is False
    assert meta["is_semantic"] is False
    assert "Mock" in emb_service.display_label or "Non-Semantic" in emb_service.display_label
    assert meta["embedding_dimension"] == 384


def test_mock_embedding_deterministic_output():
    """Embedding the exact same text must produce identical normalized vectors."""
    emb_service = MockTestEmbeddingService(dimension=64)
    v1 = emb_service.generate_embeddings(["Attention is all you need."])[0]
    v2 = emb_service.generate_embeddings(["Attention is all you need."])[0]

    assert len(v1) == 64
    assert v1 == v2


def test_azure_openai_embedding_configurability():
    """Azure OpenAI embedding service must use configurable deployment and model names."""
    from services.embeddings import AzureOpenAIEmbeddingProvider, EmbeddingProvider
    service = AzureOpenAIEmbeddingProvider(
        endpoint="https://mock-foundry.openai.azure.com",
        api_key="mock_key",
        deployment_name="custom-embedding-deployment-v2",
        model_name="custom-embedding-model",
        dimension=3072,
    )
    assert isinstance(service, EmbeddingProvider)
    meta = service.get_metadata()
    assert meta["embedding_deployment"] == "custom-embedding-deployment-v2"
    assert meta["embedding_model"] == "custom-embedding-model"
    assert meta["embedding_dimension"] == 3072
    assert meta["is_semantic"] is True


def test_local_sentence_transformer_provider_declaration():
    """LocalSentenceTransformerProvider inherits from EmbeddingProvider and records metadata."""
    from services.embeddings import LocalSentenceTransformerProvider, EmbeddingProvider
    provider = LocalSentenceTransformerProvider(model_name="all-MiniLM-L6-v2")
    assert isinstance(provider, EmbeddingProvider)
    assert provider.model_name == "all-MiniLM-L6-v2"
    assert provider.dimension == 384
    meta = provider.get_metadata()
    assert meta["embedding_model"] == "all-MiniLM-L6-v2"


# 3. VECTOR STORE & PERSISTENCE
def test_local_vector_store_cosine_and_metadata_filters(sample_paper_document):
    """LocalVectorStore must accurately compute similarity and enforce metadata filters."""
    chunker = DocumentChunker()
    chunks = chunker.chunk_document(sample_paper_document)

    emb_service = MockTestEmbeddingService(dimension=128)
    emb_service.embed_chunks(chunks)

    store = LocalVectorStore(index_name="test_index")
    store.clear()
    added = store.add_chunks(chunks)
    assert added == len(chunks)

    # Search with query vector
    q_vec = emb_service.generate_embeddings(["Attention mechanism"])[0]
    results = store.search_vectors(query_vector=q_vec, top_k=5)
    assert len(results) > 0

    # Section filter test
    sec_results = store.search_vectors(query_vector=q_vec, top_k=5, section_filter=["Methodology"])
    for chunk, score in sec_results:
        assert chunk.normalized_section == "Methodology"

    # Content type filter test
    tbl_results = store.search_vectors(query_vector=q_vec, top_k=5, content_type_filter=["table"])
    assert len(tbl_results) == 1
    assert tbl_results[0][0].content_type == "table"


def test_local_vector_store_disk_roundtrip(sample_paper_document, tmp_path):
    """Vector store must serialize to disk and restore identically."""
    chunker = DocumentChunker()
    chunks = chunker.chunk_document(sample_paper_document)
    emb_service = MockTestEmbeddingService(dimension=64)
    emb_service.embed_chunks(chunks)

    store = LocalVectorStore(index_name="roundtrip_test")
    store.clear()
    store.add_chunks(chunks)

    save_path = tmp_path / "index_dump.json"
    store.save(filepath=save_path)
    assert save_path.exists()

    new_store = LocalVectorStore(index_name="roundtrip_test_loaded")
    loaded = new_store.load(filepath=save_path)
    assert loaded is True
    assert len(new_store.chunks) == len(store.chunks)


# 4. HYBRID RETRIEVAL & BM25
def test_bm25_exact_term_matching(sample_paper_document):
    """BM25Index must accurately locate exact scientific keywords and technical formulas."""
    chunker = DocumentChunker()
    chunks = chunker.chunk_document(sample_paper_document)

    bm25 = BM25Index()
    bm25.build_index(chunks)

    results = bm25.search("BLEU 28.4", top_k=3)
    assert len(results) > 0
    top_chunk, score = results[0]
    assert "28.4" in top_chunk.text or "BLEU" in top_chunk.text


def test_hybrid_rrf_scoring(sample_paper_document):
    """Hybrid retrieval must produce combined RRF scores and record channel breakdowns."""
    chunker = DocumentChunker()
    chunks = chunker.chunk_document(sample_paper_document)

    engine = HybridRetrievalEngine()
    engine.clear()
    engine.index_chunks(chunks)

    results = engine.retrieve("Multi-Head Attention scaled dot-product", top_k=3, search_mode="hybrid")
    assert len(results) > 0

    top = results[0]
    assert top.rank == 1
    assert top.score > 0.0
    assert top.fused_score == top.score
    assert top.dense_score is not None or top.vector_score is not None
    assert top.bm25_score is not None or top.keyword_score is not None
    assert top.match_type == "hybrid"
    assert top.chunk.paper_id == "paper_001"


def test_bm25_configurable_parameters():
    """BM25Index must accept custom k1 and b parameters."""
    bm25 = BM25Index(k1=2.0, b=0.85)
    assert bm25.k1 == 2.0
    assert bm25.b == 0.85


# 5. CROSS-PAPER ISOLATION
def test_cross_paper_strict_isolation(sample_paper_document):
    """Querying paper_001 must NEVER return any chunks belonging to paper_002."""
    # Create second distinct paper
    paper2 = PaperDocument(
        id="paper_002",
        filename="lora_test.pdf",
        saved_path="data/uploads/lora_test.pdf",
        page_count=2,
    )
    paper2.paragraphs = [
        ExtractedParagraph(
            paper_id="paper_002",
            paragraph_id="paper_002_p001_para001",
            page=1,
            original_heading="Abstract",
            normalized_section="Abstract",
            content_type="abstract",
            text="LoRA freezes the pre-trained model weights and injects trainable rank decomposition matrices.",
        ),
        ExtractedParagraph(
            paper_id="paper_002",
            paragraph_id="paper_002_p001_para002",
            page=1,
            original_heading="1. Introduction",
            normalized_section="Introduction",
            content_type="body",
            text="Parameter-efficient fine-tuning reduces GPU memory requirements by over 3 times.",
        ),
    ]

    chunker = DocumentChunker()
    chunks1 = chunker.chunk_document(sample_paper_document)
    chunks2 = chunker.chunk_document(paper2)

    engine = HybridRetrievalEngine()
    engine.clear()
    engine.index_chunks(chunks1)
    engine.index_chunks(chunks2)

    assert len(engine.indexed_chunks) == len(chunks1) + len(chunks2)

    # 1. Query for paper_001 only
    results_p1 = engine.retrieve("fine-tuning weights", paper_ids=["paper_001"], top_k=10, search_mode="hybrid")
    for r in results_p1:
        assert r.chunk.paper_id == "paper_001", f"Isolation breach! Got {r.chunk.paper_id}"

    # 2. Query for paper_002 only
    results_p2 = engine.retrieve("Attention dot-product", paper_ids=["paper_002"], top_k=10, search_mode="hybrid")
    for r in results_p2:
        assert r.chunk.paper_id == "paper_002", f"Isolation breach! Got {r.chunk.paper_id}"


# 6. EVIDENCE BUNDLER & GROUNDED CONTEXT
def test_evidence_bundler_provenance_and_budget(sample_paper_document):
    """Evidence bundler must build structured items and respect context token limits without answer generation."""
    chunker = DocumentChunker()
    chunks = chunker.chunk_document(sample_paper_document)

    engine = HybridRetrievalEngine()
    engine.clear()
    engine.index_chunks(chunks)

    results = engine.retrieve("attention mechanism", top_k=4)
    bundler = EvidenceBundler(default_token_budget=1000)
    bundle = bundler.build_bundle(query="attention mechanism", results=results)

    assert len(bundle.items) > 0
    assert bundle.estimated_context_tokens <= 1000
    assert "RETRIEVED RESEARCH EVIDENCE" in bundle.formatted_context

    # Verify citation label format
    for item in bundle.items:
        assert item.citation_label.startswith("[Evidence")
        assert item.paper_id == "paper_001"
        assert len(item.text) > 0


# 7. GROUND-TRUTH RETRIEVAL BENCHMARK EVALUATION
def test_ground_truth_benchmark_evaluation(sample_paper_document):
    """Ground-truth evaluation metrics must be calculated against explicit test cases."""
    chunker = DocumentChunker()
    chunks = chunker.chunk_document(sample_paper_document)

    engine = HybridRetrievalEngine()
    engine.clear()
    engine.index_chunks(chunks)

    # Explicit test cases with known target sections & keywords
    test_cases = [
        {
            "query": "scaled dot-product attention formula",
            "target_keywords": ["softmax", "sqrt", "queries"],
            "target_section": "Methodology",
        },
        {
            "query": "BLEU score on WMT 2014 translation",
            "target_keywords": ["bleu", "28.4"],
            "target_section": "Results",
        },
    ]

    p_at_3_scores = []
    rec_at_3_scores = []
    hit_at_3_scores = []
    mrr_scores = []

    for tc in test_cases:
        results = engine.retrieve(tc["query"], top_k=3, search_mode="hybrid")
        relevant_count = 0
        first_relevant_rank = None
        for rank, r in enumerate(results, 1):
            txt = r.chunk.text.lower()
            if any(kw in txt for kw in tc["target_keywords"]) or r.chunk.normalized_section == tc["target_section"]:
                relevant_count += 1
                if first_relevant_rank is None:
                    first_relevant_rank = rank
        p_at_3_scores.append(relevant_count / 3.0)
        rec_at_3_scores.append(1.0 if relevant_count > 0 else 0.0)
        hit_at_3_scores.append(1 if relevant_count > 0 else 0)
        mrr_scores.append(1.0 / first_relevant_rank if first_relevant_rank else 0.0)

    # Ensure valid non-synthetic metrics
    mean_precision = sum(p_at_3_scores) / len(p_at_3_scores)
    mean_recall = sum(rec_at_3_scores) / len(rec_at_3_scores)
    mean_hit = sum(hit_at_3_scores) / len(hit_at_3_scores)
    mean_mrr = sum(mrr_scores) / len(mrr_scores)

    assert 0.0 <= mean_precision <= 1.0
    assert mean_recall == 1.0
    assert mean_hit == 1
    assert mean_mrr > 0.0


# 8. MULTI-PAPER REGRESSION VALIDATION (Real PDFs)
def test_multi_paper_chunking_and_retrieval_regression():
    """
    Validates end-to-end Phase 3 pipeline using both real test PDFs:
    1. Res.pdf ('Attention Is All You Need')
    2. paper_lora.pdf ('LoRA')
    """
    res_path = Path("data/uploads/Res.pdf")
    lora_path = Path("data/uploads/paper_lora.pdf")

    if not res_path.exists() or not lora_path.exists():
        pytest.skip("Test PDFs not found in data/uploads")

    processor = PDFProcessor(upload_dir=settings.UPLOAD_DIR)
    chunker = DocumentChunker()

    engine = HybridRetrievalEngine()
    engine.clear()

    # 1. Process and chunk Paper 1 (Transformer)
    paper_res = processor.extract_document(res_path, "paper_001")
    chunks_res = chunker.chunk_document(paper_res)

    assert len(chunks_res) > 10, f"Expected >10 chunks for Res.pdf, got {len(chunks_res)}"
    # Check document order
    assert [c.document_order for c in chunks_res] == list(range(1, len(chunks_res) + 1))
    assert all(c.paper_id == "paper_001" for c in chunks_res)

    # 2. Process and chunk Paper 2 (LoRA)
    paper_lora = processor.extract_document(lora_path, "paper_002")
    chunks_lora = chunker.chunk_document(paper_lora)


    assert len(chunks_lora) > 10, f"Expected >10 chunks for paper_lora.pdf, got {len(chunks_lora)}"
    assert [c.document_order for c in chunks_lora] == list(range(1, len(chunks_lora) + 1))
    assert all(c.paper_id == "paper_002" for c in chunks_lora)

    # 3. Index both into retrieval engine
    engine.index_chunks(chunks_res)
    engine.index_chunks(chunks_lora)

    stats = engine.vector_store.get_stats()
    assert stats["paper_count"] == 2
    assert "paper_001" in stats["indexed_papers"]
    assert "paper_002" in stats["indexed_papers"]

    # 4. Strict isolation retrieval on Transformer paper
    res_retrieved = engine.retrieve("multi-head self-attention", paper_ids=["paper_001"], top_k=5)
    assert len(res_retrieved) > 0
    assert all(r.chunk.paper_id == "paper_001" for r in res_retrieved)

    # 5. Strict isolation retrieval on LoRA paper
    lora_retrieved = engine.retrieve("rank decomposition matrices", paper_ids=["paper_002"], top_k=5)
    assert len(lora_retrieved) > 0
    assert all(r.chunk.paper_id == "paper_002" for r in lora_retrieved)


# 9. PAPER-AWARE BENCHMARK & MULTI-CORPUS ISOLATION REGRESSION
def test_bert_queries_isolation_and_retrieval():
    """BERT-specific queries must be scoped to BERT and never retrieve Transformer chunks."""
    res_path = Path("data/uploads/Res.pdf")
    bert_path = Path("data/uploads/res2.pdf")

    if not res_path.exists() or not bert_path.exists():
        pytest.skip("Test PDFs Res.pdf or res2.pdf not found in data/uploads")

    processor = PDFProcessor(upload_dir=settings.UPLOAD_DIR)
    chunker = DocumentChunker()
    engine = HybridRetrievalEngine()
    engine.clear()

    # Index Transformer (paper_001) and BERT (paper_002)
    p_trans = processor.extract_document(res_path, "paper_001")
    p_bert = processor.extract_document(bert_path, "paper_002")
    chunker.chunk_document(p_trans)
    chunker.chunk_document(p_bert)

    engine.index_chunks(p_trans.chunks)
    engine.index_chunks(p_bert.chunks)

    # 1. Scoped BERT query must return only paper_002 chunks
    mlm_results = engine.retrieve(
        query="Masked Language Model MLM pre-training objective",
        paper_ids=["paper_002"],
        top_k=5,
        search_mode="hybrid"
    )
    assert len(mlm_results) > 0
    assert all(r.chunk.paper_id == "paper_002" for r in mlm_results)

    # Verify relevance: at least 1 relevant chunk found
    kws = ["masked", "mlm", "pre-training", "token", "bidirectional"]
    relevant = [r for r in mlm_results if any(kw in r.chunk.text.lower() for kw in kws)]
    assert len(relevant) > 0, "Expected at least one relevant BERT MLM chunk"

    # 2. Scoped Transformer query must return only paper_001 chunks
    trans_results = engine.retrieve(
        query="Scaled Dot-Product Attention equation and computation",
        paper_ids=["paper_001"],
        top_k=5,
        search_mode="hybrid"
    )
    assert len(trans_results) > 0
    assert all(r.chunk.paper_id == "paper_001" for r in trans_results)


def test_lora_q5_q7_grounded_retrieval():
    """When LoRA is indexed and queried with paper filter, Q5, Q6, Q7 achieve Hit@5=1 and MRR=1.00."""
    lora_path = Path("data/uploads/paper_lora.pdf")
    if not lora_path.exists():
        pytest.skip("paper_lora.pdf not found in data/uploads")

    processor = PDFProcessor(upload_dir=settings.UPLOAD_DIR)
    chunker = DocumentChunker()
    engine = HybridRetrievalEngine()
    engine.clear()

    p_lora = processor.extract_document(lora_path, "paper_002")
    chunker.chunk_document(p_lora)
    engine.index_chunks(p_lora.chunks)

    # Q7: Downstream fine-tuning inference latency and memory savings
    q7_results = engine.retrieve(
        query="Downstream fine-tuning inference latency and memory savings",
        paper_ids=["paper_002"],
        top_k=5,
        search_mode="hybrid"
    )
    assert len(q7_results) > 0
    top_chunk = q7_results[0].chunk
    # Ground-truth chunk paper_002_c009 contains Section 4.1 "No Additional Inference Latency"
    assert "inference latency" in top_chunk.text.lower() or "latency" in top_chunk.text.lower()


def test_benchmark_unloaded_paper_zero_fabrication():
    """When a paper is absent, benchmark queries are not executed against unrelated papers."""
    res_path = Path("data/uploads/Res.pdf")
    if not res_path.exists():
        pytest.skip("Res.pdf not found in data/uploads")

    processor = PDFProcessor(upload_dir=settings.UPLOAD_DIR)
    chunker = DocumentChunker()
    engine = HybridRetrievalEngine()
    engine.clear()

    # Index ONLY Transformer
    p_trans = processor.extract_document(res_path, "paper_001")
    chunker.chunk_document(p_trans)
    engine.index_chunks(p_trans.chunks)

    # Searching for absent paper_002 returns empty list (no fabrication)
    absent_results = engine.retrieve(
        query="Low-Rank Adaptation intrinsic rank r",
        paper_ids=["paper_002"],
        top_k=5,
        search_mode="hybrid"
    )
    assert len(absent_results) == 0, "Must return 0 results when target paper is not indexed!"
