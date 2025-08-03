"""
Pydantic models for structured data validation and serialization.

This module defines the data models used throughout the RAG pipeline,
ensuring type safety and data validation.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator, model_validator
from datetime import datetime


class ChunkMetadata(BaseModel):
    """
    Metadata associated with a document chunk.
    
    Critical for source citation and justification in responses.
    """
    
    source_document: str = Field(..., description="Name of the source document")
    clause_id: str = Field(..., description="Unique identifier for this chunk/clause")
    chunk_index: int = Field(..., ge=0, description="Index of chunk within document")
    original_text: str = Field(..., description="Original text content of the chunk")
    char_count: int = Field(..., ge=0, description="Character count of the chunk")
    page_number: Optional[int] = Field(None, description="Page number if available")
    
    class Config:
        """Pydantic configuration."""
        frozen = True  # Make immutable


class DocumentChunk(BaseModel):
    """
    Represents a semantically coherent chunk of document content.
    
    Used for vector storage and retrieval in the RAG pipeline.
    """
    
    content: str = Field(..., min_length=1, description="The chunk content text")
    metadata: ChunkMetadata = Field(..., description="Associated metadata")
    embedding: Optional[List[float]] = Field(None, description="Vector embedding (populated during storage)")
    
    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True


class QueryEntity(BaseModel):
    """
    Structured representation of extracted entities from user query.
    
    Ensures consistent parsing of natural language queries.
    """
    
    age: Optional[int] = Field(None, ge=0, le=150, description="Age of the person")
    gender: Optional[str] = Field(None, description="Gender (M/F/Male/Female)")
    procedure: Optional[str] = Field(None, description="Medical procedure or treatment")
    location: Optional[str] = Field(None, description="City or location")
    policy_duration: Optional[str] = Field(None, description="Policy age/duration")
    policy_type: Optional[str] = Field(None, description="Type of insurance policy")
    amount: Optional[float] = Field(None, ge=0, description="Claim amount if specified")
    
    @field_validator('gender')
    @classmethod
    def normalize_gender(cls, v):
        """Normalize gender values to consistent format."""
        if v is None:
            return v
        
        gender_map = {
            'M': 'Male',
            'F': 'Female',
            'male': 'Male',
            'female': 'Female'
        }
        return gender_map.get(v, v)


class ParsedQuery(BaseModel):
    """
    Complete structured representation of a user query.
    
    Combines the original query with extracted entities.
    """
    
    original_query: str = Field(..., min_length=1, description="Original user input")
    entities: QueryEntity = Field(..., description="Extracted structured entities")
    query_type: str = Field(default="claim_evaluation", description="Type of query")
    timestamp: datetime = Field(default_factory=datetime.now, description="When query was processed")


class RetrievedContext(BaseModel):
    """
    Context retrieved from vector database for query processing.
    
    Contains the relevant document chunks and their relevance scores.
    """
    
    chunks: List[DocumentChunk] = Field(..., description="Retrieved document chunks")
    relevance_scores: List[float] = Field(..., description="Relevance scores for each chunk")
    total_retrieved: int = Field(..., ge=0, description="Total number of chunks retrieved")
    
    @model_validator(mode='after')
    def validate_scores_length(self):
        """Ensure relevance scores match number of chunks."""
        if len(self.relevance_scores) != len(self.chunks):
            raise ValueError("Number of relevance scores must match number of chunks")
        return self


class JustificationEntry(BaseModel):
    """
    Single justification entry mapping decision to source clause.
    
    Essential for explainable AI and audit trails.
    """
    
    decision_aspect: str = Field(..., description="What aspect of decision this justifies")
    clause_id: str = Field(..., description="ID of the clause supporting this decision")
    source_document: str = Field(..., description="Source document name")
    relevant_text: str = Field(..., description="Relevant text from the clause")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in this justification")


class PolicyDecision(BaseModel):
    """
    Final structured decision response from the RAG pipeline.
    
    This is the main output format that downstream systems will consume.
    """
    
    decision: str = Field(..., description="Decision (approved/rejected/partial)")
    amount: Optional[float] = Field(None, ge=0, description="Approved amount if applicable")
    justification: List[JustificationEntry] = Field(
        ..., 
        min_length=1, 
        description="Detailed justification with source citations"
    )
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Overall confidence in decision")
    query_id: Optional[str] = Field(None, description="Unique identifier for this query")
    processing_time: Optional[float] = Field(None, ge=0, description="Processing time in seconds")
    
    @field_validator('decision')
    @classmethod
    def validate_decision(cls, v):
        """Ensure decision is one of allowed values."""
        allowed_decisions = {'approved', 'rejected', 'partial', 'insufficient_info'}
        if v.lower() not in allowed_decisions:
            raise ValueError(f"Decision must be one of: {allowed_decisions}")
        return v.lower()


class ProcessingError(BaseModel):
    """
    Structured error information for failed processing attempts.
    
    Helps with debugging and error tracking in production.
    """
    
    error_type: str = Field(..., description="Type of error encountered")
    error_message: str = Field(..., description="Detailed error message")
    query: Optional[str] = Field(None, description="Original query that caused error")
    timestamp: datetime = Field(default_factory=datetime.now, description="When error occurred")
    stack_trace: Optional[str] = Field(None, description="Stack trace if available")


class IngestionResult(BaseModel):
    """
    Result of document ingestion process.
    
    Provides summary statistics and status information.
    """
    
    total_documents: int = Field(..., ge=0, description="Total documents processed")
    total_chunks: int = Field(..., ge=0, description="Total chunks created")
    successful_documents: int = Field(..., ge=0, description="Successfully processed documents")
    failed_documents: int = Field(..., ge=0, description="Failed document count")
    processing_time: float = Field(..., ge=0, description="Total processing time in seconds")
    average_chunk_size: float = Field(..., ge=0, description="Average chunk size in characters")
    errors: List[ProcessingError] = Field(default_factory=list, description="Any errors encountered")
    
    @model_validator(mode='after')
    def validate_success_count(self):
        """Ensure successful + failed = total."""
        expected_total = self.successful_documents + self.failed_documents
        if expected_total != self.total_documents:
            raise ValueError("Successful + failed documents must equal total")
        return self
