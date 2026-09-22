"""
Unit tests for Chunking and Retrieval engine.
"""

from models.paper import DocumentChunk
from services.retrieval import RAGRetrievalEngine


def test_retrieval_engine_indexing():
    engine = RAGRetrievalEngine()
    chunks = [
        DocumentChunk(
            paper_id="p1",
            chunk_id="p1_c1",
            page=1,
            section="Abstract",
            text="This paper investigates retrieval augmented generation."
        ),
        DocumentChunk(
            paper_id="p1",
            chunk_id="p1_c2",
            page=2,
            section="Methodology",
            text="We deploy Microsoft Foundry models for synthesis."
        ),
    ]

    total_indexed = engine.index_chunks(chunks)
    assert total_indexed == 2
    retrieved = engine.retrieve("retrieval", top_k=1)
    assert len(retrieved) == 1
