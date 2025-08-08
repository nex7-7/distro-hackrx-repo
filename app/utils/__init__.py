"""
Utilities package for the RAG Application.

This package contains utility functions, logging, exception handling,
and other helper functionality used across the application.
"""

from .logger import get_logger, logger, log_function_entry, log_function_exit
from .exceptions import (
    RAGApplicationError,
    DocumentProcessingError,
    DocumentDownloadError,
    UnsupportedFileTypeError,
    DocumentExtractionError,
    ChunkingError,
    EmbeddingError,
    VectorStoreError,
    LLMGenerationError,
    QueryClassificationError,
    ResponseGenerationError,
    RetrievalError,
    ContextNotFoundError,
    AuthenticationError,
    ValidationError,
    ConfigurationError,
    create_error
)

__all__ = [
    # Logging
    "get_logger",
    "logger",
    "log_function_entry", 
    "log_function_exit",
    
    # Exceptions
    "RAGApplicationError",
    "DocumentProcessingError",
    "DocumentDownloadError",
    "UnsupportedFileTypeError",
    "DocumentExtractionError",
    "ChunkingError",
    "EmbeddingError",
    "VectorStoreError",
    "LLMGenerationError",
    "QueryClassificationError", 
    "ResponseGenerationError",
    "RetrievalError",
    "ContextNotFoundError",
    "AuthenticationError",
    "ValidationError",
    "ConfigurationError",
    "create_error"
]
