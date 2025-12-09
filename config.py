"""
Configuration management for GraphLLM system
"""
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator
from typing import Optional
import os


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # Application
    app_name: str = "GraphLLM"
    app_version: str = "1.0.0"
    environment: str = "development"
    debug: bool = True

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4

    # LLM Settings - Gemini (Primary)
    gemini_api_key: str = Field(default="", env="GEMINI_API_KEY")
    gemini_model: str = "gemini-2.5-flash"

    # LLM Settings - Mistral (Fallback)
    mistral_api_key: str = Field(default="", env="MISTRAL_API_KEY")
    mistral_model: str = "mistral-7b-instruct-v0.1"

    # LLM Parameters
    llm_temperature: float = 0.0
    llm_max_tokens: int = 2048
    llm_timeout: int = 120

    # Embedding Settings
    embedding_model: str = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
    embedding_dimension: int = 384
    embedding_batch_size: int = 32

    # FAISS Vector DB
    faiss_index_path: str = "./data/faiss_index"
    faiss_metric: str = "cosine"

    # Neo4j Graph DB
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = Field(default="", env="NEO4J_PASSWORD")
    neo4j_database: str = "neo4j"

    # PostgreSQL
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "graphllm"
    postgres_user: str = "postgres"
    postgres_password: str = Field(default="", env="POSTGRES_PASSWORD")

    # MongoDB (optional)
    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_database: str = "graphllm"

    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 128
    min_chunk_size: int = 100

    # Triplet Extraction
    triplet_confidence_threshold: float = 0.6
    entity_similarity_threshold: float = 0.85
    max_triples_per_chunk: int = 10

    # Graph Pruning
    node_importance_threshold: float = 0.3
    edge_confidence_threshold: float = 0.5
    min_node_mentions: int = 2

    # RAG
    rag_top_k: int = 10
    rag_rerank_top_k: int = 5
    max_context_length: int = 4000

    # File Upload
    max_file_size_mb: int = 50
    allowed_extensions: str = "pdf"
    upload_dir: str = "./data/uploads"

    # Storage
    data_dir: str = "./data"
    logs_dir: str = "./logs"
    cache_dir: str = "./cache"

    # Monitoring
    enable_metrics: bool = True
    metrics_port: int = 9090
    log_level: str = "INFO"

    @property
    def postgres_url(self) -> str:
        """Build PostgreSQL connection URL"""
        return f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"

    @property
    def max_file_size_bytes(self) -> int:
        """Convert MB to bytes"""
        return self.max_file_size_mb * 1024 * 1024

    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()


def ensure_directories():
    """Ensure all required directories exist"""
    dirs = [
        settings.data_dir,
        settings.upload_dir,
        settings.logs_dir,
        settings.cache_dir,
        settings.faiss_index_path,
    ]
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)
