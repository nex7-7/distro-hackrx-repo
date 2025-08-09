"""
Configuration management for the RAG Application API.

This module provides type-safe configuration loading from environment variables
using Pydantic's BaseSettings. All configuration parameters are centralized here
and automatically validated at startup.
"""

from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    This class follows the Single Responsibility Principle by handling
    only configuration management and validation.
    """
    
    # API Configuration
    api_bearer_token: str = Field(..., env="API_BEARER_TOKEN")
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    
    # LLM Configuration
    google_api_key: str = Field(..., env="GOOGLE_API_KEY")
    llm_model: str = Field(default="gemini-2.5-flash", env="LLM_MODEL")
    
    # Weaviate Configuration
    weaviate_url: str = Field(default="http://localhost:8080", env="WEAVIATE_URL")
    weaviate_class_name: str = Field(default="DocumentChunks", env="WEAVIATE_CLASS_NAME")
    
    # Embedding Configuration
    embedding_model: str = Field(default="BAAI/bge-m3", env="EMBEDDING_MODEL")
    
    # OCR Configuration (Tesseract)
    tesseract_cmd: Optional[str] = Field(default=None, env="TESSERACT_CMD")
    
    # Chunking Configuration
    min_chunk_size: int = Field(default=100, env="MIN_CHUNK_SIZE")
    max_chunk_size: int = Field(default=1000, env="MAX_CHUNK_SIZE")
    chunk_overlap_percentage: int = Field(default=5, env="CHUNK_OVERLAP_PERCENTAGE")
    
    # Logging Configuration
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: str = Field(default="logs/app.log", env="LOG_FILE")
    disable_file_logging: bool = Field(default=False, env="DISABLE_FILE_LOGGING")
    
    # Performance Configuration
    max_workers: int = Field(default=4, env="MAX_WORKERS")
    retrieval_top_k: int = Field(default=30, env="RETRIEVAL_TOP_K")  # Changed to 30 for reranking
    reranking_top_k: int = Field(default=7, env="RERANKING_TOP_K")  # Top chunks after reranking
    
    # Derived properties
    @property
    def chunk_overlap_size(self) -> int:
        """Calculate actual overlap size in characters."""
        return int(self.max_chunk_size * self.chunk_overlap_percentage / 100)
    
    @property
    def log_file_path(self) -> Path:
        """Get log file as Path object."""
        return Path(self.log_file)
    
    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
# This follows the Singleton pattern for configuration access
settings = Settings()


def get_settings() -> Settings:
    """
    Dependency injection function for FastAPI.
    
    Returns:
        Settings: The global settings instance
    """
    return settings
