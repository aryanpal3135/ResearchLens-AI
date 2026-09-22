"""
Services package for ResearchLens AI.
"""

from .pdf_processor import PDFProcessor
from .chunking import DocumentChunker, estimate_tokens
from .embeddings import (
    EmbeddingService,
    BaseEmbeddingService,
    MockTestEmbeddingService,
    AzureOpenAIEmbeddingService,
    get_embedding_service,
)
from .vector_store import (
    BaseVectorStore,
    LocalVectorStore,
    AzureSearchVectorStore,
    get_vector_store,
)
from .retrieval import RAGRetrievalEngine, HybridRetrievalEngine, BM25Index
from .rag_context import EvidenceBundler
from .foundry_client import MicrosoftFoundryClient
from .paper_analyzer import PaperAnalyzer
from .comparison_engine import ComparisonEngine
from .gap_detector import ResearchGapDetector
from .question_generator import ResearchQuestionGenerator
from .literature_review import LiteratureReviewGenerator
from .translation import MultilingualTranslator
from .citation_service import CitationService

__all__ = [
    "PDFProcessor",
    "DocumentChunker",
    "estimate_tokens",
    "EmbeddingService",
    "BaseEmbeddingService",
    "MockTestEmbeddingService",
    "AzureOpenAIEmbeddingService",
    "get_embedding_service",
    "BaseVectorStore",
    "LocalVectorStore",
    "AzureSearchVectorStore",
    "get_vector_store",
    "RAGRetrievalEngine",
    "HybridRetrievalEngine",
    "BM25Index",
    "EvidenceBundler",
    "MicrosoftFoundryClient",
    "PaperAnalyzer",
    "ComparisonEngine",
    "ResearchGapDetector",
    "ResearchQuestionGenerator",
    "LiteratureReviewGenerator",
    "MultilingualTranslator",
    "CitationService",
]

