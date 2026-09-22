"""
Global settings and environment management for ResearchLens AI.
Designed for seamless Microsoft Foundry and Azure AI integration.
"""

import os
from pathlib import Path
from typing import List
from dotenv import load_dotenv

# Base Directory of the Project
BASE_DIR = Path(__file__).resolve().parent.parent

# Load .env if present
env_path = BASE_DIR / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)


class Settings:
    """Application Settings dataclass/object."""

    # Project Information
    APP_NAME: str = "ResearchLens AI"
    APP_TAGLINE: str = "Multilingual AI-Powered Research Paper Analysis & Gap Detection"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = os.getenv("DEBUG", "True").lower() in ("true", "1", "yes")

    # Supported Languages (Strictly 4)
    SUPPORTED_LANGUAGES: List[str] = ["English", "Hindi", "French", "Spanish"]
    DEFAULT_LANGUAGE: str = "English"

    # Microsoft Foundry / Azure AI Agent Service Config
    FOUNDRY_PROJECT_ENDPOINT: str = os.getenv(
        "FOUNDRY_PROJECT_ENDPOINT",
        "https://researchmate-resource.services.ai.azure.com/api/projects/researchmate"
    )
    FOUNDRY_RESEARCH_AGENT_NAME: str = os.getenv("FOUNDRY_RESEARCH_AGENT_NAME", "researchmate-gpt4-1-mini")
    FOUNDRY_RESEARCH_AGENT_VERSION: str = os.getenv("FOUNDRY_RESEARCH_AGENT_VERSION", "1")
    FOUNDRY_CHAT_AGENT_NAME: str = os.getenv("FOUNDRY_CHAT_AGENT_NAME", "Paper-Chat-Agent")
    FOUNDRY_CHAT_AGENT_VERSION: str = os.getenv("FOUNDRY_CHAT_AGENT_VERSION", "2")

    # Azure OpenAI Config (strictly for embeddings and direct vector model resources)
    AZURE_OPENAI_ENDPOINT: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    AZURE_OPENAI_API_KEY: str = os.getenv("AZURE_OPENAI_API_KEY", "")
    AZURE_OPENAI_API_VERSION: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
    AZURE_OPENAI_CHAT_DEPLOYMENT: str = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4.1-mini")
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT: str = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large")
    AZURE_OPENAI_EMBEDDING_MODEL: str = os.getenv("AZURE_OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")

    # Azure AI Speech Configuration (Text-to-Speech / Read Aloud)
    AZURE_SPEECH_ENDPOINT: str = os.getenv(
        "AZURE_SPEECH_ENDPOINT",
        "https://researchmate-resource.cognitiveservices.azure.com/"
    )
    AZURE_SPEECH_API_KEY: str = os.getenv("AZURE_SPEECH_API_KEY", os.getenv("AZURE_OPENAI_API_KEY", ""))
    AZURE_SPEECH_REGION: str = os.getenv("AZURE_SPEECH_REGION", "southeastasia")

    # Phase 3 Chunking Configuration
    CHUNK_TARGET_TOKENS: int = int(os.getenv("CHUNK_TARGET_TOKENS", "650"))
    CHUNK_OVERLAP_TOKENS: int = int(os.getenv("CHUNK_OVERLAP_TOKENS", "80"))
    CHUNK_MIN_TOKENS: int = int(os.getenv("CHUNK_MIN_TOKENS", "120"))
    CHUNK_MAX_TOKENS: int = int(os.getenv("CHUNK_MAX_TOKENS", "900"))

    # Phase 3 Embeddings Configuration
    EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "azure_openai" if os.getenv("AZURE_OPENAI_ENDPOINT") and os.getenv("AZURE_OPENAI_API_KEY") else "mock")
    LOCAL_EMBEDDING_MODEL: str = os.getenv("LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    EMBEDDING_DIMENSION: int = int(os.getenv("EMBEDDING_DIMENSION", "3072" if "large" in os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "large").lower() else "1536"))

    # Phase 3 Vector Store & Retrieval Configuration
    VECTOR_STORE_BACKEND: str = os.getenv("VECTOR_STORE_BACKEND", "local")  # "local", "azure_search"
    AZURE_SEARCH_ENDPOINT: str = os.getenv("AZURE_SEARCH_ENDPOINT", "")
    AZURE_SEARCH_API_KEY: str = os.getenv("AZURE_SEARCH_API_KEY", "")
    AZURE_SEARCH_INDEX_NAME: str = os.getenv("AZURE_SEARCH_INDEX_NAME", "researchlens-chunks")
    BM25_K1: float = float(os.getenv("BM25_K1", "1.5"))
    BM25_B: float = float(os.getenv("BM25_B", "0.75"))
    RRF_K: int = int(os.getenv("RRF_K", "60"))


    # Storage Paths
    UPLOAD_DIR: Path = BASE_DIR / os.getenv("UPLOAD_DIR", "data/uploads")
    PROCESSED_DIR: Path = BASE_DIR / os.getenv("PROCESSED_DIR", "data/processed")
    INDEX_STORAGE_DIR: Path = BASE_DIR / os.getenv("INDEX_STORAGE_DIR", "data/indices")
    CACHE_STORAGE_DIR: Path = BASE_DIR / os.getenv("CACHE_STORAGE_DIR", "data/cache")
    MAX_UPLOAD_SIZE_MB: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "50"))

    def __init__(self):
        # Ensure directories exist
        self.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        self.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        self.INDEX_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        self.CACHE_STORAGE_DIR.mkdir(parents=True, exist_ok=True)

    @property
    def is_foundry_configured(self) -> bool:
        """Returns True if Microsoft Foundry / Azure credentials are provided."""
        return bool(self.AZURE_OPENAI_ENDPOINT and self.AZURE_OPENAI_API_KEY)


settings = Settings()

