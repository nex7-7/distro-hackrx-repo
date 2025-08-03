"""
Core components package for the RAG pipeline.

This package contains the main pipeline components for document ingestion,
retrieval, and response generation.
"""

from .data_ingestion import DocumentIngestionPipeline, ChunkingConfig

__all__ = [
    'DocumentIngestionPipeline',
    'ChunkingConfig'
]
