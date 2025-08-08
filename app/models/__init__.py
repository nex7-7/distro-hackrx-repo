"""
Models package for the RAG Application.

This package contains Pydantic models for request/response validation
and internal data structures used throughout the RAG pipeline.
"""

from .schemas import (
    HackRXRequest,
    HackRXResponse,
    DocumentInfo,
    ChunkData,
    QueryAnalysis,
    RetrievalResult,
    GeneratedResponse,
    ErrorResponse,
    HealthCheckResponse,
    QueryClassification
)

__all__ = [
    "HackRXRequest",
    "HackRXResponse", 
    "DocumentInfo",
    "ChunkData",
    "QueryAnalysis",
    "RetrievalResult",
    "GeneratedResponse",
    "ErrorResponse",
    "HealthCheckResponse",
    "QueryClassification"
]
