"""
Vector store service for the RAG Application API.

This module provides Weaviate vector database integration for storing and
retrieving document chunks. It handles Stage 2 (chunk storage) and Stage 4
(context retrieval) of the RAG pipeline with multiprocessing optimization.
"""

import asyncio
import time
import uuid
from typing import List, Optional, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import weaviate
from weaviate import Client
from weaviate.exceptions import (
    WeaviateBaseError,
    WeaviateConnectionError,
    WeaviateQueryError,
    WeaviateTimeoutError
)

from config.settings import settings
from app.models.schemas import ChunkData, DocumentInfo
from app.utils.logger import get_logger
from app.utils.exceptions import VectorStoreError, create_error
from app.services.embedding_service import embedding_service

logger = get_logger(__name__)


class VectorStore:
    """
    Weaviate vector database integration for document chunk storage and retrieval.
    
    This class provides high-performance vector operations with multiprocessing
    support for concurrent chunk storage and parallel query execution.
    """
    
    def __init__(self) -> None:
        """Initialize the vector store service."""
        self.client: Optional[Client] = None
        self.class_name = settings.weaviate_class_name
        self.weaviate_url = settings.weaviate_url
        self.executor = ThreadPoolExecutor(max_workers=settings.max_workers)
        self._client_lock = threading.Lock()
        
        logger.info("Vector store initialized", 
                   url=self.weaviate_url,
                   class_name=self.class_name)
    
    async def connect(self) -> None:
        """
        Connect to Weaviate and ensure schema exists.
        
        Raises:
            VectorStoreError: If connection or schema setup fails
        """
        if self.client is None:
            with self._client_lock:
                if self.client is None:
                    try:
                        logger.info("Connecting to Weaviate", url=self.weaviate_url)
                        
                        # Create Weaviate client
                        self.client = weaviate.Client(
                            url=self.weaviate_url,
                            timeout_config=(5, 15)  # (connection, read) timeouts
                        )
                        
                        # Test connection
                        if not self.client.is_ready():
                            raise create_error(
                                VectorStoreError,
                                "Weaviate is not ready",
                                "WEAVIATE_NOT_READY"
                            )
                        
                        # Ensure schema exists
                        await self._ensure_schema()
                        
                        logger.info("Successfully connected to Weaviate")
                        
                    except WeaviateBaseError as e:
                        raise create_error(
                            VectorStoreError,
                            f"Failed to connect to Weaviate: {str(e)}",
                            "WEAVIATE_CONNECTION_FAILED",
                            url=self.weaviate_url
                        )
                    except Exception as e:
                        raise create_error(
                            VectorStoreError,
                            f"Unexpected error connecting to Weaviate: {str(e)}",
                            "WEAVIATE_UNEXPECTED_ERROR"
                        )
    
    async def _ensure_schema(self) -> None:
        """Ensure the required schema exists in Weaviate."""
        try:
            # Check if class exists
            schema = self.client.schema.get()
            existing_classes = [cls['class'] for cls in schema.get('classes', [])]
            
            if self.class_name not in existing_classes:
                logger.info("Creating Weaviate schema", class_name=self.class_name)
                
                # Define schema for document chunks
                class_definition = {
                    "class": self.class_name,
                    "description": "Document chunks for RAG retrieval",
                    "vectorizer": "none",  # We provide our own vectors
                    "properties": [
                        {
                            "name": "content",
                            "dataType": ["text"],
                            "description": "Text content of the chunk"
                        },
                        {
                            "name": "chunk_id",
                            "dataType": ["string"],
                            "description": "Unique identifier for the chunk"
                        },
                        {
                            "name": "chunk_index",
                            "dataType": ["int"],
                            "description": "Index of chunk within document"
                        },
                        {
                            "name": "document_hash",
                            "dataType": ["string"],
                            "description": "Hash of the source document"
                        },
                        {
                            "name": "document_url",
                            "dataType": ["string"],
                            "description": "URL of the source document"
                        },
                        {
                            "name": "document_filename",
                            "dataType": ["string"],
                            "description": "Filename of the source document"
                        },
                        {
                            "name": "source_page",
                            "dataType": ["int"],
                            "description": "Source page number (if applicable)"
                        },
                        {
                            "name": "metadata",
                            "dataType": ["object"],
                            "description": "Additional metadata"
                        }
                    ]
                }
                
                self.client.schema.create_class(class_definition)
                logger.info("Schema created successfully")
            else:
                logger.debug("Schema already exists")
                
        except Exception as e:
            raise create_error(
                VectorStoreError,
                f"Failed to ensure schema: {str(e)}",
                "SCHEMA_SETUP_FAILED"
            )
    
    async def document_exists(self, document_hash: str) -> bool:
        """
        Check if document already exists in vector store.
        
        Args:
            document_hash: SHA-256 hash of document content
            
        Returns:
            bool: True if document exists
            
        Raises:
            VectorStoreError: If query fails
        """
        try:
            await self.connect()
            
            logger.debug("Checking document existence", hash=document_hash)
            
            result = (
                self.client.query
                .get(self.class_name, ["chunk_id"])
                .with_where({
                    "path": ["document_hash"],
                    "operator": "Equal",
                    "valueString": document_hash
                })
                .with_limit(1)
                .do()
            )
            
            chunks = result.get("data", {}).get("Get", {}).get(self.class_name, [])
            exists = len(chunks) > 0
            
            logger.debug("Document existence check completed", 
                        hash=document_hash,
                        exists=exists)
            
            return exists
            
        except Exception as e:
            raise create_error(
                VectorStoreError,
                f"Failed to check document existence: {str(e)}",
                "EXISTENCE_CHECK_FAILED",
                document_hash=document_hash
            )
    
    async def store_chunks(
        self, 
        chunks: List[ChunkData], 
        document_info: DocumentInfo
    ) -> None:
        """
        Store document chunks in vector store with multiprocessing.
        
        Args:
            chunks: List of document chunks with embeddings
            document_info: Document metadata
            
        Raises:
            VectorStoreError: If storage fails
        """
        start_time = time.time()
        await self.connect()
        
        logger.log_stage("Vector Storage", "Starting", 
                        chunk_count=len(chunks),
                        document=document_info.filename)
        
        try:
            # Use multiprocessing for concurrent storage
            batch_size = max(1, len(chunks) // settings.max_workers)
            futures = []
            
            # Create batches for parallel processing
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                future = asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    self._store_chunk_batch,
                    batch,
                    document_info
                )
                futures.append(future)
            
            # Wait for all batches to complete
            results = await asyncio.gather(*futures, return_exceptions=True)
            
            # Check for errors
            failed_batches = [r for r in results if isinstance(r, Exception)]
            if failed_batches:
                error_msgs = [str(e) for e in failed_batches]
                raise create_error(
                    VectorStoreError,
                    f"Failed to store {len(failed_batches)} batches: {error_msgs}",
                    "BATCH_STORAGE_FAILED"
                )
            
            duration = time.time() - start_time
            logger.log_performance(
                "Vector Storage",
                duration,
                chunk_count=len(chunks),
                chunks_per_second=len(chunks) / duration
            )
            
        except Exception as e:
            logger.error(f"Chunk storage failed: {str(e)}", 
                        chunk_count=len(chunks))
            
            if isinstance(e, VectorStoreError):
                raise
            
            raise create_error(
                VectorStoreError,
                f"Failed to store chunks: {str(e)}",
                "STORAGE_FAILED",
                chunk_count=len(chunks)
            )
    
    def _store_chunk_batch(
        self, 
        chunk_batch: List[ChunkData], 
        document_info: DocumentInfo
    ) -> None:
        """
        Store a batch of chunks synchronously (runs in thread pool).
        
        Args:
            chunk_batch: Batch of chunks to store
            document_info: Document metadata
        """
        try:
            # Use batch import for efficiency
            with self.client.batch as batch:
                batch.batch_size = len(chunk_batch)
                
                for chunk in chunk_batch:
                    # Prepare properties
                    properties = {
                        "content": chunk.content,
                        "chunk_id": chunk.chunk_id,
                        "chunk_index": chunk.chunk_index,
                        "document_hash": document_info.content_hash,
                        "document_url": str(document_info.url),
                        "document_filename": document_info.filename,
                        "source_page": chunk.source_page,
                        "metadata": chunk.metadata
                    }
                    
                    # Add to batch with vector
                    batch.add_data_object(
                        data_object=properties,
                        class_name=self.class_name,
                        vector=chunk.embedding
                    )
            
            logger.debug("Stored chunk batch", batch_size=len(chunk_batch))
            
        except Exception as e:
            logger.error(f"Failed to store chunk batch: {str(e)}", 
                        batch_size=len(chunk_batch))
            raise
    
    async def search_similar_chunks(
        self, 
        query_embedding: List[float], 
        limit: int = None
    ) -> List[Tuple[ChunkData, float]]:
        """
        Search for similar chunks using vector similarity.
        
        Args:
            query_embedding: Query vector embedding
            limit: Maximum number of results (defaults to settings.retrieval_top_k)
            
        Returns:
            List[Tuple[ChunkData, float]]: List of (chunk, similarity_score) tuples
            
        Raises:
            VectorStoreError: If search fails
        """
        if limit is None:
            limit = settings.retrieval_top_k
        
        try:
            await self.connect()
            
            logger.debug("Searching similar chunks", limit=limit)
            
            # Perform vector search
            result = (
                self.client.query
                .get(self.class_name, [
                    "content", "chunk_id", "chunk_index", "document_hash",
                    "document_url", "document_filename", "source_page", "metadata"
                ])
                .with_near_vector({
                    "vector": query_embedding
                })
                .with_limit(limit)
                .with_additional(["distance"])
                .do()
            )
            
            # Parse results
            chunks_data = result.get("data", {}).get("Get", {}).get(self.class_name, [])
            
            results = []
            for item in chunks_data:
                # Extract chunk data
                chunk = ChunkData(
                    chunk_id=item["chunk_id"],
                    content=item["content"],
                    chunk_index=item["chunk_index"],
                    source_page=item.get("source_page"),
                    metadata=item.get("metadata", {})
                )
                
                # Extract similarity score (distance -> similarity)
                distance = item.get("_additional", {}).get("distance", 1.0)
                similarity = 1.0 - distance  # Convert distance to similarity
                
                results.append((chunk, similarity))
            
            logger.debug("Vector search completed", 
                        results_found=len(results),
                        limit=limit)
            
            return results
            
        except Exception as e:
            raise create_error(
                VectorStoreError,
                f"Vector search failed: {str(e)}",
                "SEARCH_FAILED",
                limit=limit
            )
    
    async def search_by_keywords(
        self, 
        keywords: List[str], 
        limit: int = None
    ) -> List[Tuple[ChunkData, float]]:
        """
        Search chunks by keywords using hybrid search.
        
        Args:
            keywords: List of keywords to search for
            limit: Maximum number of results
            
        Returns:
            List[Tuple[ChunkData, float]]: List of (chunk, relevance_score) tuples
        """
        if limit is None:
            limit = settings.retrieval_top_k
        
        try:
            await self.connect()
            
            logger.debug("Searching by keywords", 
                        keywords=keywords,
                        limit=limit)
            
            # Generate embedding for keyword query
            keyword_query = " ".join(keywords)
            query_embedding = await embedding_service.generate_query_embedding(keyword_query)
            
            # Use vector search (which is more effective than keyword search)
            return await self.search_similar_chunks(query_embedding, limit)
            
        except Exception as e:
            raise create_error(
                VectorStoreError,
                f"Keyword search failed: {str(e)}",
                "KEYWORD_SEARCH_FAILED",
                keywords=keywords
            )
    
    async def get_document_chunks(self, document_hash: str) -> List[ChunkData]:
        """
        Retrieve all chunks for a specific document.
        
        Args:
            document_hash: Document content hash
            
        Returns:
            List[ChunkData]: All chunks for the document
        """
        try:
            await self.connect()
            
            result = (
                self.client.query
                .get(self.class_name, [
                    "content", "chunk_id", "chunk_index", 
                    "source_page", "metadata"
                ])
                .with_where({
                    "path": ["document_hash"],
                    "operator": "Equal", 
                    "valueString": document_hash
                })
                .with_sort([{"path": ["chunk_index"], "order": "asc"}])
                .do()
            )
            
            chunks_data = result.get("data", {}).get("Get", {}).get(self.class_name, [])
            
            chunks = []
            for item in chunks_data:
                chunk = ChunkData(
                    chunk_id=item["chunk_id"],
                    content=item["content"],
                    chunk_index=item["chunk_index"],
                    source_page=item.get("source_page"),
                    metadata=item.get("metadata", {})
                )
                chunks.append(chunk)
            
            logger.debug("Retrieved document chunks",
                        document_hash=document_hash,
                        chunk_count=len(chunks))
            
            return chunks
            
        except Exception as e:
            raise create_error(
                VectorStoreError,
                f"Failed to retrieve document chunks: {str(e)}",
                "DOCUMENT_CHUNKS_RETRIEVAL_FAILED",
                document_hash=document_hash
            )
    
    async def delete_document(self, document_hash: str) -> int:
        """
        Delete all chunks for a document.
        
        Args:
            document_hash: Document content hash
            
        Returns:
            int: Number of chunks deleted
        """
        try:
            await self.connect()
            
            logger.info("Deleting document chunks", document_hash=document_hash)
            
            # Delete by document hash
            result = self.client.batch.delete_objects(
                class_name=self.class_name,
                where={
                    "path": ["document_hash"],
                    "operator": "Equal",
                    "valueString": document_hash
                }
            )
            
            deleted_count = result.get("results", {}).get("successful", 0)
            
            logger.info("Document chunks deleted",
                       document_hash=document_hash,
                       deleted_count=deleted_count)
            
            return deleted_count
            
        except Exception as e:
            raise create_error(
                VectorStoreError,
                f"Failed to delete document: {str(e)}",
                "DOCUMENT_DELETION_FAILED",
                document_hash=document_hash
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get vector store statistics.
        
        Returns:
            Dict[str, Any]: Store statistics
        """
        try:
            if self.client is None:
                return {"status": "disconnected"}
            
            # Get object count
            result = (
                self.client.query
                .aggregate(self.class_name)
                .with_meta_count()
                .do()
            )
            
            count = result.get("data", {}).get("Aggregate", {}).get(self.class_name, [{}])[0].get("meta", {}).get("count", 0)
            
            return {
                "status": "connected",
                "total_chunks": count,
                "class_name": self.class_name,
                "weaviate_url": self.weaviate_url
            }
            
        except Exception as e:
            logger.warning(f"Failed to get vector store stats: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    def __del__(self) -> None:
        """Cleanup resources when service is destroyed."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)


# Global vector store instance
vector_store = VectorStore()


async def get_vector_store() -> VectorStore:
    """
    Get the global vector store instance.
    
    Returns:
        VectorStore: The vector store service
    """
    return vector_store
