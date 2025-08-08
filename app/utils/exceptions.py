"""
Custom exception classes for the RAG Application API.

This module defines a hierarchical exception structure that provides
clear error categorization and enables proper error handling throughout
the RAG pipeline stages.
"""

from typing import Optional, Any, Dict


class RAGApplicationError(Exception):
    """
    Base exception class for all RAG application errors.
    
    This follows the Open/Closed Principle by providing a extensible
    base for all application-specific exceptions.
    """
    
    def __init__(
        self, 
        message: str, 
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
    
    def __str__(self) -> str:
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message


# Stage 1: Document Processing Errors
class DocumentProcessingError(RAGApplicationError):
    """Raised when document ingestion or preprocessing fails."""
    pass


class DocumentDownloadError(DocumentProcessingError):
    """Raised when document download from URL fails."""
    pass


class UnsupportedFileTypeError(DocumentProcessingError):
    """Raised when the file type is not supported for processing."""
    pass


class DocumentExtractionError(DocumentProcessingError):
    """Raised when text extraction from document fails."""
    pass


# Stage 2: Chunking and Vectorization Errors
class ChunkingError(RAGApplicationError):
    """Raised when document chunking fails."""
    pass


class EmbeddingError(RAGApplicationError):
    """Raised when generating embeddings fails."""
    pass


class VectorStoreError(RAGApplicationError):
    """Raised when vector database operations fail."""
    pass


# Stage 3 & 5: LLM Integration Errors
class LLMGenerationError(RAGApplicationError):
    """Raised when LLM generation fails."""
    pass


class QueryClassificationError(LLMGenerationError):
    """Raised when query classification by LLM fails."""
    pass


class ResponseGenerationError(LLMGenerationError):
    """Raised when final response generation fails."""
    pass


# Stage 4: Retrieval Errors
class RetrievalError(RAGApplicationError):
    """Raised when vector search and context retrieval fails."""
    pass


class ContextNotFoundError(RetrievalError):
    """Raised when no relevant context is found for a query."""
    pass


# API and Authentication Errors
class AuthenticationError(RAGApplicationError):
    """Raised when API authentication fails."""
    pass


class ValidationError(RAGApplicationError):
    """Raised when request validation fails."""
    pass


# Configuration Errors
class ConfigurationError(RAGApplicationError):
    """Raised when application configuration is invalid."""
    pass


# Utility function for error creation
def create_error(
    error_class: type[RAGApplicationError],
    message: str,
    error_code: Optional[str] = None,
    **details: Any
) -> RAGApplicationError:
    """
    Factory function for creating structured errors.
    
    Args:
        error_class: The exception class to instantiate
        message: Error message
        error_code: Optional error code
        **details: Additional error details
        
    Returns:
        RAGApplicationError: Configured exception instance
    """
    return error_class(
        message=message,
        error_code=error_code,
        details=details
    )
