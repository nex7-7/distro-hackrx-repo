"""
RAG Application API

A high-performance, containerized Retrieval-Augmented Generation (RAG) Application API
that processes documents and answers questions using vector search and LLM integration.

This package contains:
- FastAPI application with authentication
- 6-stage RAG pipeline for document processing
- Weaviate vector database integration
- Multiprocessing for performance optimization
- Comprehensive error handling and logging
"""

__version__ = "1.0.0"
__author__ = "RAG Development Team"
__description__ = "High-performance RAG Application API"

from typing import Final

# Application constants
APP_NAME: Final[str] = "RAG Application API"
APP_VERSION: Final[str] = __version__
API_V1_PREFIX: Final[str] = "/api/v1"

# Export key components (will be populated as we build them)
__all__ = [
    "__version__",
    "__author__", 
    "__description__",
    "APP_NAME",
    "APP_VERSION", 
    "API_V1_PREFIX"
]
