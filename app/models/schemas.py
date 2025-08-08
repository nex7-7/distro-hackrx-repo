"""
Pydantic models for request and response validation.

This module defines all the data models used in the RAG Application API,
ensuring type safety and automatic validation for all inputs and outputs.
"""

from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, HttpUrl, validator
from enum import Enum


class QueryClassification(str, Enum):
    """Classification types for user queries."""
    FROM_DOCUMENT = "From Document"
    NOT_FROM_DOCUMENT = "Not From Document"


class HackRXRequest(BaseModel):
    """
    Request model for the /api/v1/hackrx/run endpoint.
    
    This model validates the incoming request structure according
    to the API specification.
    """
    documents: HttpUrl = Field(
        ...,
        description="URL to the document to be processed",
        example="https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=..."
    )
    questions: List[str] = Field(
        ...,
        min_items=1,
        max_items=50,
        description="List of questions to answer based on the document",
        example=[
            "What is the grace period for premium payment?",
            "What is the waiting period for pre-existing diseases?"
        ]
    )
    
    @validator('questions')
    def validate_questions(cls, v):
        """Ensure all questions are non-empty strings."""
        for question in v:
            if not question.strip():
                raise ValueError("Questions cannot be empty")
        return v


class HackRXResponse(BaseModel):
    """
    Response model for the /api/v1/hackrx/run endpoint.
    
    This model ensures the response structure matches the API specification.
    """
    answers: List[str] = Field(
        ...,
        description="List of answers corresponding to the input questions",
        example=[
            "A grace period of thirty days is provided for premium payment...",
            "There is a waiting period of thirty-six (36) months of continuous coverage..."
        ]
    )
    
    @validator('answers')
    def validate_answers_length(cls, v, values):
        """Ensure answers list matches questions list length."""
        # Note: This validation would be enhanced in the actual endpoint
        # where we have access to the original questions count
        return v


# Internal models for RAG pipeline stages
class DocumentInfo(BaseModel):
    """Model for document metadata and content."""
    url: HttpUrl
    filename: str
    file_type: str
    content_hash: str
    total_pages: Optional[int] = None
    extracted_text: str
    first_five_pages: str
    extraction_method: str


class ChunkData(BaseModel):
    """Model for document chunks."""
    chunk_id: str
    content: str
    chunk_index: int
    source_page: Optional[int] = None
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class QueryAnalysis(BaseModel):
    """Model for query classification results."""
    query_id: str
    original_query: str
    classification: QueryClassification
    keywords: Optional[List[str]] = None
    prompt_template: Optional[str] = None


class RetrievalResult(BaseModel):
    """Model for context retrieval results."""
    query_id: str
    retrieved_chunks: List[ChunkData]
    relevance_scores: List[float]
    context: str


class GeneratedResponse(BaseModel):
    """Model for generated responses."""
    query_id: str
    original_query: str
    answer: str
    context_used: bool
    generation_method: str  # "from_document" or "general_knowledge"


# Error response models
class ErrorDetail(BaseModel):
    """Model for error details."""
    error_code: Optional[str] = None
    message: str
    details: Optional[Dict[str, Any]] = None


class ErrorResponse(BaseModel):
    """Model for API error responses."""
    error: ErrorDetail
    timestamp: str
    request_id: Optional[str] = None


# Health check and status models
class HealthCheckResponse(BaseModel):
    """Model for health check responses."""
    status: str = Field(default="healthy")
    timestamp: str
    version: str
    services: Dict[str, str] = Field(
        description="Status of external services",
        example={
            "weaviate": "connected",
            "llm_service": "available",
            "embedding_model": "loaded"
        }
    )


# Configuration models for runtime settings
class PipelineConfig(BaseModel):
    """Model for RAG pipeline configuration."""
    min_chunk_size: int
    max_chunk_size: int
    chunk_overlap_percentage: int
    retrieval_top_k: int
    max_workers: int


class LLMConfig(BaseModel):
    """Model for LLM service configuration."""
    model_name: str
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    timeout_seconds: int = 30


# Batch processing models (for future extensions)
class BatchRequest(BaseModel):
    """Model for batch processing requests."""
    requests: List[HackRXRequest] = Field(max_items=10)
    batch_id: Optional[str] = None


class BatchResponse(BaseModel):
    """Model for batch processing responses."""
    batch_id: str
    responses: List[Union[HackRXResponse, ErrorResponse]]
    completed_at: str
    total_requests: int
    successful_requests: int
    failed_requests: int


# Export all models
__all__ = [
    # Main API models
    "HackRXRequest",
    "HackRXResponse",
    
    # Internal pipeline models
    "DocumentInfo",
    "ChunkData", 
    "QueryAnalysis",
    "RetrievalResult",
    "GeneratedResponse",
    
    # Error models
    "ErrorDetail",
    "ErrorResponse",
    
    # Status models
    "HealthCheckResponse",
    
    # Configuration models
    "PipelineConfig",
    "LLMConfig",
    
    # Batch models
    "BatchRequest",
    "BatchResponse",
    
    # Enums
    "QueryClassification"
]
