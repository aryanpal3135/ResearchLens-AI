"""
Embeddings Service for ResearchLens AI.
Supports Azure OpenAI (Production) and deterministic Mock/Test embeddings with disk caching.
Adheres strictly to Phase 3 honesty principles:
- Non-neural/hash embeddings are explicitly labeled as mock/non-semantic test embeddings.
- Full provenance metadata recorded: model, deployment, dimension, version, is_semantic.
"""

import os
import json
import hashlib
import numpy as np
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Optional

from config.settings import settings
from models.paper import DocumentChunk


# Module-level fast RAM cache for embedding vectors across all instances and reruns
_GLOBAL_EMBED_CACHE: Dict[str, List[float]] = {}


class EmbeddingProvider(ABC):
    """Abstract Base Class for Document and Query Embedding Providers."""


    def __init__(
        self,
        model_name: str,
        deployment_name: Optional[str] = None,
        dimension: int = 384,
        version: str = "1.0.0",
        is_semantic: bool = False,
        display_label: str = "Base Embedding"
    ):
        self.model_name = model_name
        self.deployment_name = deployment_name or model_name
        self.dimension = dimension
        self.version = version
        self.is_semantic = is_semantic
        self.display_label = display_label
        
        # Cache directory setup
        self.cache_dir = settings.CACHE_STORAGE_DIR / "embeddings"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_metadata(self) -> Dict[str, Any]:
        """Returns provenance dictionary for embedding vectors."""
        return {
            "embedding_model": self.model_name,
            "embedding_deployment": self.deployment_name,
            "embedding_dimension": self.dimension,
            "embedding_version": self.version,
            "is_semantic": self.is_semantic,
            "display_label": self.display_label
        }

    @abstractmethod
    def _compute_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Underlying computation method for raw texts."""
        pass

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generates normalized embedding vectors for a list of texts,
        using RAM and disk caching to avoid redundant computation.
        """
        if not texts:
            return []

        results: List[Optional[List[float]]] = [None] * len(texts)
        missing_indices: List[int] = []
        missing_texts: List[str] = []

        for idx, text in enumerate(texts):
            cache_key = hashlib.sha256(f"{self.model_name}:{text}".encode("utf-8")).hexdigest()
            # 1. Global in-memory RAM cache check (instant <0.01ms)
            if cache_key in _GLOBAL_EMBED_CACHE:
                results[idx] = _GLOBAL_EMBED_CACHE[cache_key]
                continue

            # 2. Disk file cache check
            cache_file = self.cache_dir / f"{cache_key}.json"
            if cache_file.exists():
                try:
                    with open(cache_file, "r", encoding="utf-8") as f:
                        cached_vec = json.load(f)
                        if len(cached_vec) == self.dimension:
                            results[idx] = cached_vec
                            _GLOBAL_EMBED_CACHE[cache_key] = cached_vec
                            continue
                except Exception:
                    pass

            missing_indices.append(idx)
            missing_texts.append(text)

        if missing_texts:
            computed_vectors = self._compute_embeddings(missing_texts)
            for idx, vec in zip(missing_indices, computed_vectors):
                # Ensure L2 normalization
                arr = np.array(vec, dtype=np.float32)
                norm = np.linalg.norm(arr)
                if norm > 0:
                    arr = arr / norm
                normalized_vec = arr.tolist()

                results[idx] = normalized_vec
                _GLOBAL_EMBED_CACHE[cache_key] = normalized_vec

                # Save to cache
                cache_key = hashlib.sha256(f"{self.model_name}:{texts[idx]}".encode("utf-8")).hexdigest()
                cache_file = self.cache_dir / f"{cache_key}.json"
                try:
                    with open(cache_file, "w", encoding="utf-8") as f:
                        json.dump(normalized_vec, f)
                except Exception:
                    pass

        return [r if r is not None else [0.0] * self.dimension for r in results]

    def embed_chunks(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """
        Computes embeddings for chunks and attaches vectors and metadata.
        Preserves all chunk content and provenance.
        """
        if not chunks:
            return chunks

        texts = [c.text for c in chunks]
        vectors = self.generate_embeddings(texts)
        meta = self.get_metadata()

        for chunk, vec in zip(chunks, vectors):
            chunk.embedding = vec
            chunk.embedding_metadata = meta.copy()

        return chunks


class MockEmbeddingProvider(EmbeddingProvider):
    """
    Deterministic Lexical/Hash-Based Mock Embedding Provider.
    Explicitly labeled as NON-SEMANTIC (is_semantic=False).
    Used for local testing, rapid unit testing, and offline environments without neural weights.
    Never mislabeled as semantic.
    """

    def __init__(self, dimension: Optional[int] = None):
        dim = dimension or settings.EMBEDDING_DIMENSION or 384
        super().__init__(
            model_name=settings.LOCAL_EMBEDDING_MODEL or "deterministic_hash_mock",
            deployment_name="local_mock",
            dimension=dim,
            version="1.0.0",
            is_semantic=False,
            display_label="Deterministic Hash (Mock/Non-Semantic Test Embedding)"
        )

    def _compute_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generates deterministic pseudo-vectors using MD5 and token hashing."""
        vectors: List[List[float]] = []
        for text in texts:
            vec = np.zeros(self.dimension, dtype=np.float32)
            words = text.lower().split()
            if not words:
                vectors.append(vec.tolist())
                continue

            for word in words:
                # Hash token to multiple buckets with alternating signs
                h = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16)
                idx1 = h % self.dimension
                idx2 = (h >> 16) % self.dimension
                vec[idx1] += 1.0
                vec[idx2] += 0.5

            # L2 normalize
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            vectors.append(vec.tolist())
        return vectors


class LocalSentenceTransformerProvider(EmbeddingProvider):
    """
    Local Neural Semantic Embedding Provider using sentence-transformers.
    is_semantic = True.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", dimension: int = 384):
        self._model = None
        dim = dimension
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(model_name)
            dim = self._model.get_sentence_embedding_dimension()
        except (ImportError, ModuleNotFoundError):
            dim = dimension

        super().__init__(
            model_name=model_name,
            deployment_name=model_name,
            dimension=dim,
            version="st-1.0",
            is_semantic=True,
            display_label=f"SentenceTransformers ({model_name})"
        )

    def _compute_embeddings(self, texts: List[str]) -> List[List[float]]:
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        embeddings = self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return embeddings.tolist()


class AzureOpenAIEmbeddingProvider(EmbeddingProvider):
    """
    Microsoft Foundry / Azure OpenAI Embedding Provider (Production).
    Configurable deployment name and model name from environment.
    is_semantic = True.
    """

    def __init__(
        self,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        deployment_name: Optional[str] = None,
        model_name: Optional[str] = None,
        api_version: Optional[str] = None,
        dimension: Optional[int] = None,
    ):
        self.endpoint = endpoint or settings.AZURE_OPENAI_ENDPOINT
        self.api_key = api_key or settings.AZURE_OPENAI_API_KEY
        self.deployment = deployment_name or settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT
        self.model = model_name or settings.AZURE_OPENAI_EMBEDDING_MODEL
        self.api_version = api_version or settings.AZURE_OPENAI_API_VERSION
        
        # Determine dimension based on parameter, deployment or settings
        if dimension is not None:
            dim = dimension
        elif "large" in self.deployment.lower():
            dim = 3072
        elif "small" in self.deployment.lower() or "ada" in self.deployment.lower():
            dim = 1536
        else:
            dim = settings.EMBEDDING_DIMENSION or 3072

        super().__init__(
            model_name=self.model,
            deployment_name=self.deployment,
            dimension=dim,
            version="azure-openai-2024",
            is_semantic=True,
            display_label=f"Azure OpenAI ({self.deployment})"
        )
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import AzureOpenAI
            self._client = AzureOpenAI(
                azure_endpoint=self.endpoint,
                api_key=self.api_key,
                api_version=self.api_version,
            )
        return self._client

    def _compute_embeddings(self, texts: List[str]) -> List[List[float]]:
        if not self.endpoint or not self.api_key:
            raise ValueError("Azure OpenAI Endpoint and API Key must be configured.")

        client = self._get_client()
        # Batch API calls up to 128 texts per batch for high-speed document indexing
        batch_size = 128
        all_embeddings: List[List[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = client.embeddings.create(
                input=batch,
                model=self.deployment
            )
            sorted_data = sorted(response.data, key=lambda x: x.index)
            all_embeddings.extend([item.embedding for item in sorted_data])

        return all_embeddings


# Backwards compatibility aliases
BaseEmbeddingService = EmbeddingProvider
MockTestEmbeddingService = MockEmbeddingProvider
SentenceTransformerEmbeddingService = LocalSentenceTransformerProvider
AzureOpenAIEmbeddingService = AzureOpenAIEmbeddingProvider


def get_embedding_service(provider: Optional[str] = None) -> EmbeddingProvider:
    """
    Factory function to instantiate the configured embedding provider.
    - Explicit provider argument takes highest precedence.
    - If Azure OpenAI is configured and available, it is selected as the default production provider.
    - Otherwise falls back to explicitly labeled MockEmbeddingProvider.
    - Raises clear configuration errors if a real provider is explicitly requested but unconfigured.
    """
    chosen = (provider or settings.EMBEDDING_PROVIDER).lower()

    if chosen == "azure_openai":
        if settings.is_foundry_configured:
            return AzureOpenAIEmbeddingProvider()
        if provider is not None:
            # User explicitly requested azure_openai but credentials missing
            raise ValueError(
                "Azure OpenAI embedding credentials are not configured in settings (.env). "
                "Please set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY, or set EMBEDDING_PROVIDER=mock for offline testing."
            )

    if chosen in ("sentence_transformers", "local_neural"):
        try:
            return LocalSentenceTransformerProvider()
        except ImportError:
            if provider is not None:
                raise ImportError(
                    "sentence-transformers is not installed. Run 'pip install sentence-transformers' "
                    "or configure EMBEDDING_PROVIDER=azure_openai or mock."
                )

    if chosen in ("mock", "test", "local_mock"):
        return MockEmbeddingProvider()

    # Default fallback: check if Azure OpenAI is configured
    if settings.is_foundry_configured:
        try:
            return AzureOpenAIEmbeddingProvider()
        except Exception:
            pass

    return MockEmbeddingProvider()


class EmbeddingService:
    """
    Backwards-compatible wrapper alias for get_embedding_service.
    Ensures seamless compatibility with existing Phase 1 & 2 callers.
    """

    def __init__(self, deployment_name: Optional[str] = None):
        self._service = get_embedding_service()
        if deployment_name:
            self._service.deployment_name = deployment_name

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self._service.generate_embeddings(texts)

    def embed_chunks(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        return self._service.embed_chunks(chunks)

    @property
    def is_semantic(self) -> bool:
        return self._service.is_semantic

    @property
    def display_label(self) -> str:
        return self._service.display_label


