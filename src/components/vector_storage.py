"""
Vector database storage component for document chunks using ChromaDB.

This module handles the storage and retrieval of document chunks in ChromaDB,
following Clean Code and SOLID principles for maintainable vector operations.
"""

import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import uuid
import re
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

# Import our schemas
from ..schemas.models import DocumentChunk, ChunkMetadata


# Configure logging
logger = logging.getLogger(__name__)


# ===== CUSTOM EXCEPTIONS =====
class VectorStorageError(Exception):
    """Raised when vector storage operations fail."""
    pass


class CollectionError(Exception):
    """Raised when collection operations fail."""
    pass


class EmbeddingError(Exception):
    """Raised when embedding generation fails."""
    pass


# ===== CONFIGURATION CLASS =====
class VectorStorageConfig:
    """
    Configuration for vector database storage.
    
    Follows Single Responsibility Principle by containing only
    vector storage related configuration.
    """
    
    def __init__(
        self,
        chroma_db_path: str = "./chroma_db",
        collection_name: str = "policy_documents",
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        distance_metric: str = "cosine",
        max_workers: int = 4,  # For async operations
        batch_size: int = 100  # For batch processing
    ):
        """
        Initialize vector storage configuration.
        
        Args:
            chroma_db_path: Path to ChromaDB storage directory
            collection_name: Name of the collection to store chunks
            embedding_model: Name of the embedding model to use
            chunk_size: Maximum size of chunks for embedding
            chunk_overlap: Overlap between chunks
            distance_metric: Distance metric for similarity search
            max_workers: Maximum number of worker threads for async operations
            batch_size: Default batch size for processing operations
        """
        self.chroma_db_path = Path(chroma_db_path)
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.distance_metric = distance_metric
        self.max_workers = max_workers
        self.batch_size = batch_size
        
        # Ensure directory exists
        self.chroma_db_path.mkdir(parents=True, exist_ok=True)


# ===== VECTOR STORAGE CLASS =====
class VectorStorage:
    """
    Handles vector storage operations using ChromaDB with async support.
    
    Follows Single Responsibility Principle by focusing solely on
    vector database operations and chunk storage.
    """
    
    def __init__(self, config: VectorStorageConfig):
        """
        Initialize vector storage with ChromaDB.
        
        Args:
            config: Configuration for vector storage
        """
        self.config = config
        self.client = None
        self.collection = None
        self.embedding_function = None
        self.executor = ThreadPoolExecutor(max_workers=config.max_workers)
        
        self._initialize_client()
        self._initialize_collection()
    
    def _initialize_client(self) -> None:
        """
        Initialize ChromaDB client.
        
        Raises:
            VectorStorageError: If client initialization fails
        """
        try:
            # Initialize ChromaDB client with persistent storage
            self.client = chromadb.PersistentClient(
                path=str(self.config.chroma_db_path),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Initialize embedding function
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.config.embedding_model
            )
            
            logger.info(f"ChromaDB client initialized at: {self.config.chroma_db_path}")
            
        except Exception as e:
            raise VectorStorageError(f"Failed to initialize ChromaDB client: {e}")
    
    def _initialize_collection(self) -> None:
        """
        Initialize or get existing collection with document tracking metadata.
        
        Raises:
            CollectionError: If collection initialization fails
        """
        try:
            if not self.client or not self.embedding_function:
                raise CollectionError("Client or embedding function not initialized")
                
            # Try to get existing collection first
            try:
                self.collection = self.client.get_collection(
                    name=self.config.collection_name,
                    embedding_function=self.embedding_function  # type: ignore
                )
                logger.info(f"Retrieved existing collection: {self.config.collection_name}")
                
            except Exception as get_error:
                # Collection doesn't exist, create it with document tracking metadata
                logger.info(f"Collection doesn't exist ({get_error}), creating new collection...")
                try:
                    initial_metadata = {
                        "hnsw:space": self.config.distance_metric,
                        "processed_documents": "[]",  # JSON string of processed document names
                        "created_at": datetime.now().isoformat(),
                        "version": "1.0"
                    }
                    
                    self.collection = self.client.create_collection(
                        name=self.config.collection_name,
                        embedding_function=self.embedding_function,  # type: ignore
                        metadata=initial_metadata
                    )
                    logger.info(f"Created new collection: {self.config.collection_name}")
                    
                except Exception as create_error:
                    raise CollectionError(f"Failed to create collection: {create_error}")
                
        except CollectionError:
            raise  # Re-raise CollectionError as-is
        except Exception as e:
            raise CollectionError(f"Failed to initialize collection: {e}")
    
    def get_processed_documents(self) -> List[str]:
        """
        Get list of documents that have already been processed.
        
        Returns:
            List of document names that have been processed
        """
        try:
            if not self.collection:
                return []
                
            # ChromaDB doesn't store collection-level metadata in an easily accessible way
            # We'll track processed documents by querying for unique source documents
            # This is more reliable than trying to maintain metadata
            try:
                # Query all documents to get unique source document names
                results = self.collection.get(
                    include=["metadatas"]
                )
                
                if (results and 
                    isinstance(results, dict) and 
                    results.get("metadatas") and 
                    isinstance(results["metadatas"], list)):
                    
                    source_docs = set()
                    for metadata in results["metadatas"]:
                        if (metadata and 
                            isinstance(metadata, dict) and 
                            "source_document" in metadata):
                            source_docs.add(metadata["source_document"])
                    return list(source_docs)
                    
            except Exception as query_error:
                logger.debug(f"Could not query existing documents: {query_error}")
                
            return []
            
        except Exception as e:
            logger.warning(f"Failed to get processed documents: {e}")
            return []
    
    def is_document_processed(self, document_name: str) -> bool:
        """
        Check if a document has already been processed.
        
        Args:
            document_name: Name of the document to check
            
        Returns:
            True if document has been processed, False otherwise
        """
        processed_docs = self.get_processed_documents()
        return document_name in processed_docs
    
    def _add_processed_document(self, document_name: str) -> None:
        """
        Add a document to the list of processed documents.
        
        This is automatically handled when chunks are stored with proper metadata.
        The document tracking is done via the source_document field in chunk metadata.
        
        Args:
            document_name: Name of the document to mark as processed
        """
        # Since we're tracking documents via chunk metadata (source_document field),
        # this method just logs that the document has been processed.
        # The actual tracking happens when chunks are stored.
        logger.info(f"Document processing completed: {document_name}")
    
    async def store_chunks_async(self, chunks: List[DocumentChunk], document_name: str) -> Dict[str, Any]:
        """
        Store document chunks asynchronously in the vector database.
        
        Args:
            chunks: List of DocumentChunk objects to store
            document_name: Name of the source document
            
        Returns:
            Dictionary with storage statistics
            
        Raises:
            VectorStorageError: If storage operation fails
        """
        if not chunks:
            logger.warning("No chunks provided for storage")
            return {"stored_count": 0, "failed_count": 0, "document_name": document_name}
        
        logger.info(f"Storing {len(chunks)} chunks for document '{document_name}' asynchronously")
        
        # Run the synchronous storage in a thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.executor,
            self._store_chunks_sync,
            chunks,
            document_name
        )
        
        return result
    
    def _store_chunks_sync(self, chunks: List[DocumentChunk], document_name: str) -> Dict[str, Any]:
        """
        Synchronous helper for storing chunks.
        
        Args:
            chunks: List of DocumentChunk objects to store
            document_name: Name of the source document
            
        Returns:
            Dictionary with storage statistics
        """
        start_time = datetime.now()
        
        # First, store the chunks using the existing method
        result = self.store_chunks(chunks)
        
        # Then mark the document as processed
        if result.get("stored_count", 0) > 0:
            self._add_processed_document(document_name)
        
        result["document_name"] = document_name
        return result
    
    async def store_multiple_documents_async(
        self, 
        document_chunks_map: Dict[str, List[DocumentChunk]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Store multiple documents' chunks asynchronously with concurrent processing.
        
        Args:
            document_chunks_map: Dictionary mapping document names to their chunks
            
        Returns:
            Dictionary mapping document names to their storage results
        """
        if not document_chunks_map:
            return {}
        
        logger.info(f"Starting async storage for {len(document_chunks_map)} documents")
        
        # Create async tasks for each document
        tasks = []
        for doc_name, chunks in document_chunks_map.items():
            task = self.store_chunks_async(chunks, doc_name)
            tasks.append((doc_name, task))
        
        # Wait for all tasks to complete
        results = {}
        for doc_name, task in tasks:
            try:
                result = await task
                results[doc_name] = result
                logger.info(f"Completed storage for {doc_name}: {result.get('stored_count', 0)} chunks")
            except Exception as e:
                logger.error(f"Failed to store chunks for {doc_name}: {e}")
                results[doc_name] = {
                    "stored_count": 0,
                    "failed_count": len(document_chunks_map.get(doc_name, [])),
                    "error": str(e),
                    "document_name": doc_name
                }
        
        return results
    
    def cleanup(self) -> None:
        """
        Cleanup resources, including the thread pool executor.
        """
        if hasattr(self, 'executor') and self.executor:
            self.executor.shutdown(wait=True)
            logger.info("Thread pool executor shut down")
    
    def store_chunks(self, chunks: List[DocumentChunk]) -> Dict[str, Any]:
        """
        Store document chunks in the vector database.
        
        Args:
            chunks: List of DocumentChunk objects to store
            
        Returns:
            Dictionary with storage statistics
            
        Raises:
            VectorStorageError: If storage operation fails
        """
        if not chunks:
            logger.warning("No chunks provided for storage")
            return {"stored_count": 0, "failed_count": 0}
        
        logger.info(f"Storing {len(chunks)} chunks in vector database")
        start_time = datetime.now()
        
        # Prepare data for ChromaDB
        documents = []
        metadatas = []
        ids = []
        
        stored_count = 0
        failed_count = 0
        
        for chunk in chunks:
            try:
                # Generate unique ID for the chunk
                chunk_id = self._generate_chunk_id(chunk.metadata)
                
                # Prepare document content (only the text content)
                documents.append(chunk.content)
                
                # Prepare metadata (everything except the content)
                metadata = self._prepare_metadata(chunk.metadata)
                metadatas.append(metadata)
                
                # Store the generated ID
                ids.append(chunk_id)
                
                stored_count += 1
                
            except Exception as e:
                logger.error(f"Failed to prepare chunk {chunk.metadata.clause_id}: {e}")
                failed_count += 1
                continue
        
        # Store in ChromaDB
        try:
            if documents and self.collection:  # Only store if we have valid documents and collection
                self.collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
                
                processing_time = (datetime.now() - start_time).total_seconds()
                
                logger.info(f"Successfully stored {stored_count} chunks in {processing_time:.2f}s")
                
                return {
                    "stored_count": stored_count,
                    "failed_count": failed_count,
                    "processing_time": processing_time,
                    "collection_name": self.config.collection_name
                }
            else:
                logger.warning("No valid documents to store")
                return {"stored_count": 0, "failed_count": failed_count}
                
        except Exception as e:
            raise VectorStorageError(f"Failed to store chunks in ChromaDB: {e}")
    
    def _generate_chunk_id(self, metadata: ChunkMetadata) -> str:
        """
        Generate a unique ID for a chunk.
        
        Args:
            metadata: Chunk metadata
            
        Returns:
            Unique string ID for the chunk
        """
        # Use clause_id as base, but make it globally unique
        base_id = f"{metadata.source_document}_{metadata.clause_id}_{metadata.chunk_index}"
        
        # Clean the ID to be ChromaDB compatible (no special characters)
        clean_id = re.sub(r'[^a-zA-Z0-9_-]', '_', base_id)
        
        return clean_id
    
    def _prepare_metadata(self, chunk_metadata: ChunkMetadata) -> Dict[str, Any]:
        """
        Prepare metadata for ChromaDB storage.
        
        ChromaDB metadata must be JSON serializable and contain only
        basic types (str, int, float, bool).
        
        Args:
            chunk_metadata: ChunkMetadata object
            
        Returns:
            Dictionary of metadata ready for ChromaDB
        """
        metadata = {
            "source_document": chunk_metadata.source_document,
            "clause_id": chunk_metadata.clause_id,
            "chunk_index": chunk_metadata.chunk_index,
            "char_count": chunk_metadata.char_count,
            "page_number": chunk_metadata.page_number or -1,  # ChromaDB doesn't handle None
            "created_at": datetime.now().isoformat()
        }
        
        return metadata
    
    def search_similar_chunks(
        self, 
        query: str, 
        n_results: int = 5,
        where_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks in the vector database.
        
        Args:
            query: Query text to search for
            n_results: Number of results to return
            where_filter: Optional metadata filter
            
        Returns:
            List of similar chunks with metadata and distances
            
        Raises:
            VectorStorageError: If search operation fails
        """
        try:
            if not self.collection:
                raise VectorStorageError("Collection not initialized")
                
            logger.info(f"Searching for similar chunks: '{query[:50]}...'")
            
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where_filter
            )
            
            # Format results for easier consumption
            formatted_results = []
            
            if (results and isinstance(results, dict) and 
                results.get('documents') and 
                isinstance(results['documents'], list) and 
                len(results['documents']) > 0):
                
                documents = results['documents'][0]
                metadatas = []
                distances = []
                
                if results.get('metadatas') and isinstance(results['metadatas'], list) and len(results['metadatas']) > 0:
                    metadatas = results['metadatas'][0]
                    
                if results.get('distances') and isinstance(results['distances'], list) and len(results['distances']) > 0:
                    distances = results['distances'][0]
                
                for i, doc in enumerate(documents):
                    metadata = metadatas[i] if i < len(metadatas) else {}
                    distance = distances[i] if i < len(distances) else 1.0
                    
                    formatted_results.append({
                        "content": doc,
                        "metadata": metadata,
                        "similarity_score": 1 - distance,  # Convert distance to similarity
                        "rank": i + 1
                    })
            
            logger.info(f"Found {len(formatted_results)} similar chunks")
            return formatted_results
            
        except Exception as e:
            raise VectorStorageError(f"Failed to search similar chunks: {e}")
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection.
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            if not self.collection:
                return {"error": "Collection not initialized"}
                
            count = self.collection.count()
            
            return {
                "collection_name": self.config.collection_name,
                "total_chunks": count,
                "embedding_model": self.config.embedding_model,
                "distance_metric": self.config.distance_metric
            }
            
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {"error": str(e)}
    
    def delete_collection(self) -> bool:
        """
        Delete the entire collection.
        
        WARNING: This will permanently delete all stored chunks.
        
        Returns:
            True if deletion was successful, False otherwise
        """
        try:
            if not self.client:
                logger.error("Client not initialized")
                return False
                
            self.client.delete_collection(name=self.config.collection_name)
            logger.info(f"Deleted collection: {self.config.collection_name}")
            
            # Reinitialize collection
            self._initialize_collection()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            return False
    
    def reset_collection(self) -> bool:
        """
        Reset the collection (delete and recreate).
        
        Returns:
            True if reset was successful, False otherwise
        """
        try:
            if not self.client:
                logger.error("Client not initialized")
                return False
                
            logger.info(f"Resetting collection: {self.config.collection_name}")
            
            # Delete existing collection if it exists
            try:
                self.client.delete_collection(name=self.config.collection_name)
                logger.info(f"Deleted existing collection: {self.config.collection_name}")
            except Exception as e:
                # Collection doesn't exist or other error, that's fine
                logger.info(f"Collection deletion skipped: {e}")
            
            # Create new collection
            self._initialize_collection()
            
            logger.info("Collection reset successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to reset collection: {e}")
            return False


# ===== UTILITY FUNCTIONS =====
def create_vector_storage(
    chroma_db_path: str = "./chroma_db",
    collection_name: str = "policy_documents"
) -> VectorStorage:
    """
    Factory function to create a VectorStorage instance with default configuration.
    
    Args:
        chroma_db_path: Path to ChromaDB storage
        collection_name: Name of the collection
        
    Returns:
        Configured VectorStorage instance
    """
    config = VectorStorageConfig(
        chroma_db_path=chroma_db_path,
        collection_name=collection_name
    )
    
    return VectorStorage(config)


def batch_store_chunks(
    storage: VectorStorage,
    chunks: List[DocumentChunk],
    batch_size: int = 100
) -> Dict[str, Any]:
    """
    Store chunks in batches for better performance.
    
    Args:
        storage: VectorStorage instance
        chunks: List of chunks to store
        batch_size: Number of chunks per batch
        
    Returns:
        Combined storage statistics
    """
    total_stored = 0
    total_failed = 0
    total_time = 0.0
    
    logger.info(f"Storing {len(chunks)} chunks in batches of {batch_size}")
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        
        logger.info(f"Processing batch {batch_num} ({len(batch)} chunks)")
        
        try:
            result = storage.store_chunks(batch)
            total_stored += result.get("stored_count", 0)
            total_failed += result.get("failed_count", 0)
            total_time += result.get("processing_time", 0.0)
            
        except Exception as e:
            logger.error(f"Failed to store batch {batch_num}: {e}")
            total_failed += len(batch)
    
    return {
        "total_stored": total_stored,
        "total_failed": total_failed,
        "total_processing_time": total_time,
        "batches_processed": (len(chunks) + batch_size - 1) // batch_size
    }


# Make classes available for import
__all__ = [
    'VectorStorage',
    'VectorStorageConfig',
    'VectorStorageError',
    'CollectionError', 
    'EmbeddingError',
    'create_vector_storage',
    'batch_store_chunks'
]
