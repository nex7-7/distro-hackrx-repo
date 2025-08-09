"""
Vector store service for the RAG Application API.

This module provides Weaviate vector database integration for storing and
retrieving document chunks. It handles Stage 2 (chunk storage) and Stage 4
(context retrieval) of the RAG pipeline with multiprocessing optimization.
"""

import asyncio
import time
import uuid
from typing import List, Optional, Dict, Any, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import weaviate
import weaviate.classes as wvc
from weaviate.classes.config import Property, DataType
from weaviate.exceptions import WeaviateBaseError

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
    Each document gets its own collection for better isolation and performance.
    """
    
    def __init__(self) -> None:
        """Initialize the vector store service."""
        self.client: Optional[weaviate.WeaviateClient] = None
        self.weaviate_url = settings.weaviate_url
        self.executor = ThreadPoolExecutor(max_workers=settings.max_workers)
        self._client_lock = threading.Lock()
        
        logger.info("Vector store initialized", 
                   url=self.weaviate_url)
    
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
                        
                        # Create Weaviate client (v4 API)
                        # Parse URL to extract host
                        from urllib.parse import urlparse
                        parsed_url = urlparse(self.weaviate_url)
                        host = parsed_url.hostname or 'localhost'
                        port = parsed_url.port or 8080
                        
                        self.client = weaviate.connect_to_local(
                            host=host,
                            port=port,
                            grpc_port=50051
                        )
                        
                        # Ensure connection works
                        if not self.client.is_ready():
                            raise create_error(
                                VectorStoreError,
                                "Weaviate is not ready",
                                "WEAVIATE_NOT_READY"
                            )
                        
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
    
    def _get_collection_name(self, document_hash: str) -> str:
        """
        Generate collection name for a document.
        
        Args:
            document_hash: SHA-256 hash of document content
            
        Returns:
            str: Collection name for the document
        """
        # Use first 16 characters of hash to keep collection names reasonable
        # Prefix with 'Doc_' to ensure valid collection name
        return f"Doc_{document_hash[:16]}"
    
    async def _ensure_document_collection(self, document_hash: str, document_info: DocumentInfo) -> str:
        """
        Ensure a collection exists for the document.
        
        Args:
            document_hash: SHA-256 hash of document content
            document_info: Document metadata
            
        Returns:
            str: Collection name for the document
            
        Raises:
            VectorStoreError: If collection creation fails
        """
        collection_name = self._get_collection_name(document_hash)
        
        try:
            # Check if collection exists
            if not self.client.collections.exists(collection_name):
                logger.info("Creating document collection", 
                          collection_name=collection_name,
                          document=document_info.filename)
                
                # Define properties for document chunks
                properties = [
                    Property(name="content", data_type=DataType.TEXT),
                    Property(name="chunk_id", data_type=DataType.TEXT),
                    Property(name="chunk_index", data_type=DataType.INT),
                    Property(name="source_page", data_type=DataType.INT),
                    Property(name="metadata", data_type=DataType.TEXT),
                ]
                
                # Create collection with document metadata
                collection_description = f"Chunks for document: {document_info.filename}"
                
                self.client.collections.create(
                    name=collection_name,
                    description=collection_description,
                    properties=properties,
                    vector_config=wvc.config.Configure.Vectors.self_provided()
                )
                
                logger.info("Document collection created", collection_name=collection_name)
            else:
                logger.debug("Document collection already exists", collection_name=collection_name)
            
            return collection_name
            
        except Exception as e:
            raise create_error(
                VectorStoreError,
                f"Failed to ensure document collection: {str(e)}",
                "COLLECTION_CREATION_FAILED",
                collection_name=collection_name
            )
    
    async def document_exists(self, document_hash: str) -> bool:
        """
        Check if document already exists in vector store by checking collection existence.
        
        Args:
            document_hash: SHA-256 hash of document content
            
        Returns:
            bool: True if document exists (collection exists)
            
        Raises:
            VectorStoreError: If check fails
        """
        try:
            await self.connect()
            
            collection_name = self._get_collection_name(document_hash)
            exists = self.client.collections.exists(collection_name)
            
            logger.debug("Document existence check completed", 
                        hash=document_hash,
                        collection_name=collection_name,
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
        Store document chunks in document-specific vector store collection.
        
        Args:
            chunks: List of document chunks with embeddings
            document_info: Document metadata
            
        Raises:
            VectorStoreError: If storage fails
        """
        start_time = time.time()
        await self.connect()
        
        # Ensure document collection exists
        collection_name = await self._ensure_document_collection(
            document_info.content_hash, 
            document_info
        )
        
        logger.log_stage("Vector Storage", "Starting", 
                        chunk_count=len(chunks),
                        document=document_info.filename,
                        collection=collection_name)
        
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
                    collection_name
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
                chunks_per_second=len(chunks) / duration,
                collection=collection_name
            )
            
        except Exception as e:
            logger.error(f"Chunk storage failed: {str(e)}", 
                        chunk_count=len(chunks),
                        collection=collection_name)
            
            if isinstance(e, VectorStoreError):
                raise
            
            raise create_error(
                VectorStoreError,
                f"Failed to store chunks: {str(e)}",
                "STORAGE_FAILED",
                chunk_count=len(chunks),
                collection=collection_name
            )
    
    def _store_chunk_batch(
        self, 
        chunk_batch: List[ChunkData], 
        collection_name: str
    ) -> None:
        """
        Store a batch of chunks synchronously (runs in thread pool).
        
        Args:
            chunk_batch: Batch of chunks to store
            collection_name: Name of the document collection
        """
        try:
            import json
            collection = self.client.collections.get(collection_name)
            
            # Prepare data objects for batch insertion
            objects = []
            for chunk in chunk_batch:
                # Prepare properties (no need for document-level metadata)
                properties = {
                    "content": chunk.content,
                    "chunk_id": chunk.chunk_id,
                    "chunk_index": chunk.chunk_index,
                    "source_page": chunk.source_page,
                    "metadata": json.dumps(chunk.metadata) if chunk.metadata else "{}"
                }
                
                # Create data object with vector
                objects.append(
                    wvc.data.DataObject(
                        properties=properties,
                        vector=chunk.embedding
                    )
                )
            
            # Insert batch
            collection.data.insert_many(objects)
            
            logger.debug("Stored chunk batch", 
                        batch_size=len(chunk_batch),
                        collection=collection_name)
            
        except Exception as e:
            logger.error(f"Failed to store chunk batch: {str(e)}", 
                        batch_size=len(chunk_batch),
                        collection=collection_name)
            raise
    
    async def search_similar_chunks(
        self, 
        query_embedding: List[float], 
        document_hash: str,
        limit: int = None
    ) -> List[Tuple[ChunkData, float]]:
        """
        Search for similar chunks within a specific document's collection.
        
        Args:
            query_embedding: Query vector embedding
            document_hash: Hash of the document to search within
            limit: Maximum number of results (defaults to settings.retrieval_top_k)
            
        Returns:
            List[Tuple[ChunkData, float]]: List of (chunk, similarity_score) tuples
            
        Raises:
            VectorStoreError: If search fails
        """
        if limit is None:
            limit = settings.retrieval_top_k
        
        try:
            import json
            await self.connect()
            
            collection_name = self._get_collection_name(document_hash)
            
            # Check if document collection exists
            if not self.client.collections.exists(collection_name):
                logger.warning("Document collection not found", 
                             document_hash=document_hash,
                             collection_name=collection_name)
                return []
            
            logger.debug("Searching similar chunks", 
                        limit=limit,
                        collection=collection_name)
            
            collection = self.client.collections.get(collection_name)
            
            # Perform vector search
            result = collection.query.near_vector(
                near_vector=query_embedding,
                limit=limit,
                return_metadata=wvc.query.MetadataQuery(distance=True)
            )
            
            results = []
            for obj in result.objects:
                # Parse metadata from JSON string
                metadata_str = obj.properties.get("metadata", "{}")
                try:
                    metadata = json.loads(metadata_str) if metadata_str else {}
                except (json.JSONDecodeError, TypeError):
                    metadata = {}
                
                # Extract chunk data
                chunk = ChunkData(
                    chunk_id=obj.properties["chunk_id"],
                    content=obj.properties["content"],
                    chunk_index=obj.properties["chunk_index"],
                    source_page=obj.properties.get("source_page"),
                    metadata=metadata
                )
                
                # Extract similarity score (distance -> similarity)
                distance = obj.metadata.distance if obj.metadata else 1.0
                similarity = 1.0 - distance  # Convert distance to similarity
                
                results.append((chunk, similarity))
            
            logger.debug("Vector search completed", 
                        results_found=len(results),
                        limit=limit,
                        collection=collection_name)
            
            return results
            
        except Exception as e:
            raise create_error(
                VectorStoreError,
                f"Vector search failed: {str(e)}",
                "SEARCH_FAILED",
                limit=limit,
                document_hash=document_hash
            )
    
    async def search_by_keywords(
        self, 
        keywords: Union[List[str], str], 
        document_hash: str,
        limit: int = 30  # Changed default to 30 for reranking
    ) -> List[Tuple[ChunkData, float]]:
        """
        Search chunks by keywords within a specific document's collection.
        
        Args:
            keywords: List of keywords or single keyword string to search for
            document_hash: Hash of the document to search within
            limit: Maximum number of results (default 30 for reranking)
            
        Returns:
            List[Tuple[ChunkData, float]]: List of (chunk, relevance_score) tuples
        """
        try:
            await self.connect()
            
            # Handle both single string and list of keywords
            if isinstance(keywords, str):
                keyword_query = keywords
            else:
                keyword_query = " ".join(keywords)
            
            logger.debug("Searching by keywords", 
                        keywords=keyword_query,
                        limit=limit,
                        document_hash=document_hash)
            
            # Generate embedding for keyword query
            query_embedding = await embedding_service.generate_query_embedding(keyword_query)
            
            # Use vector search within the specific document
            return await self.search_similar_chunks(query_embedding, document_hash, limit)
            
        except Exception as e:
            raise create_error(
                VectorStoreError,
                f"Keyword search failed: {str(e)}",
                "KEYWORD_SEARCH_FAILED",
                keywords=keywords,
                document_hash=document_hash
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
            import json
            await self.connect()
            
            collection_name = self._get_collection_name(document_hash)
            
            # Check if document collection exists
            if not self.client.collections.exists(collection_name):
                logger.warning("Document collection not found", 
                             document_hash=document_hash,
                             collection_name=collection_name)
                return []
            
            collection = self.client.collections.get(collection_name)
            
            # Use iterator to get all chunks for the document
            objects_iterator = collection.iterator()
            
            chunks = []
            for obj in objects_iterator:
                # Parse metadata from JSON string
                metadata_str = obj.properties.get("metadata", "{}")
                try:
                    metadata = json.loads(metadata_str) if metadata_str else {}
                except (json.JSONDecodeError, TypeError):
                    metadata = {}
                
                chunk = ChunkData(
                    chunk_id=obj.properties["chunk_id"],
                    content=obj.properties["content"],
                    chunk_index=obj.properties["chunk_index"],
                    source_page=obj.properties.get("source_page"),
                    metadata=metadata
                )
                chunks.append(chunk)
            
            # Sort by chunk index
            chunks.sort(key=lambda x: x.chunk_index)
            
            logger.debug("Retrieved document chunks",
                        document_hash=document_hash,
                        chunk_count=len(chunks),
                        collection=collection_name)
            
            return chunks
            
        except Exception as e:
            raise create_error(
                VectorStoreError,
                f"Failed to retrieve document chunks: {str(e)}",
                "DOCUMENT_CHUNKS_RETRIEVAL_FAILED",
                document_hash=document_hash
            )
    
    async def delete_document(self, document_hash: str) -> bool:
        """
        Delete all chunks for a specific document by deleting its collection.
        
        Args:
            document_hash: Document content hash
            
        Returns:
            bool: True if deletion was successful
        """
        try:
            await self.connect()
            
            collection_name = self._get_collection_name(document_hash)
            
            logger.info("Deleting document collection", 
                       document_hash=document_hash,
                       collection_name=collection_name)
            
            # Check if document collection exists
            if not self.client.collections.exists(collection_name):
                logger.warning("Document collection not found for deletion",
                             document_hash=document_hash,
                             collection_name=collection_name)
                return True  # Already deleted
            
            # Delete the entire collection for this document
            self.client.collections.delete(collection_name)
            
            logger.info("Document collection deleted successfully",
                       document_hash=document_hash,
                       collection_name=collection_name)
            
            return True
            
        except Exception as e:
            raise create_error(
                VectorStoreError,
                f"Failed to delete document: {str(e)}",
                "DOCUMENT_DELETION_FAILED",
                document_hash=document_hash
            )
            
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
            
            # Get all collections and count total objects
            all_collections = list(self.client.collections.list_all())
            total_count = 0
            
            for collection_name in all_collections:
                try:
                    collection = self.client.collections.get(collection_name)
                    result = collection.aggregate.over_all(total_count=True)
                    total_count += result.total_count or 0
                except Exception:
                    # Skip collections that can't be accessed
                    continue
            
            return {
                "status": "connected",
                "total_chunks": total_count,
                "total_collections": len(all_collections),
                "weaviate_url": self.weaviate_url
            }
            
        except Exception as e:
            logger.warning(f"Failed to get vector store stats: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    def __del__(self) -> None:
        """Cleanup resources when service is destroyed."""
        if hasattr(self, 'client') and self.client is not None:
            try:
                self.client.close()
            except:
                pass
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
