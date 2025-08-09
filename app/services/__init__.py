"""
Services package for the RAG Application.

This package contains external service integrations including
document processing, embedding generation, vector database,
and LLM service clients.
"""

from .document_processor import DocumentProcessor
from .embedding_service import (
    EmbeddingService,
    embedding_service,
    get_embedding_service,
    embed_text,
    embed_texts,
    embed_query
)
from .vector_store import VectorStore, vector_store, get_vector_store
from .reranking_service import (
    RerankingService,
    reranking_service,
    get_reranking_service,
    rerank_chunks
)
from .llm_service import (
    LLMService,
    llm_service,
    get_llm_service,
    restructure_queries,
    generate_responses
)

__all__ = [
    # Document Processing
    "DocumentProcessor",
    
    # Embedding Services
    "EmbeddingService",
    "embedding_service",
    "get_embedding_service",
    "embed_text",
    "embed_texts",
    "embed_query",
    
    # Vector Store
    "VectorStore",
    "vector_store",
    "get_vector_store",
    
    # Reranking Services
    "RerankingService",
    "reranking_service",
    "get_reranking_service",
    "rerank_chunks",
    
    # LLM Services
    "LLMService",
    "llm_service",
    "get_llm_service",
    "restructure_queries",
    "generate_responses"
]
