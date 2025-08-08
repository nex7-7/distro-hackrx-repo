"""
Embedding service for the RAG Application API.

This module handles text embedding generation using the BAAI/bge-m3 model.
It provides efficient batch processing and manages the embedding model lifecycle
for Stage 2 (vectorization) and Stage 4 (retrieval) of the RAG pipeline.
"""

import asyncio
import time
from typing import List, Optional, Union
from concurrent.futures import ThreadPoolExecutor
import threading

from sentence_transformers import SentenceTransformer
import numpy as np

from config.settings import settings
from app.utils.logger import get_logger
from app.utils.exceptions import EmbeddingError, create_error

logger = get_logger(__name__)


class EmbeddingService:
    """
    Embedding service that generates vector embeddings using BAAI/bge-m3 model.
    
    This service implements a singleton pattern for efficient model management
    and provides both synchronous and asynchronous embedding generation.
    """
    
    _instance: Optional['EmbeddingService'] = None
    _lock = threading.Lock()
    
    def __new__(cls) -> 'EmbeddingService':
        """Singleton pattern implementation."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self) -> None:
        """Initialize the embedding service (called only once due to singleton)."""
        if hasattr(self, '_initialized'):
            return
        
        self._initialized = True
        self.model: Optional[SentenceTransformer] = None
        self.model_name = settings.embedding_model
        self.executor = ThreadPoolExecutor(max_workers=settings.max_workers)
        self._model_lock = threading.Lock()
        
        logger.info("Embedding service initialized", model=self.model_name)
    
    def _load_model(self) -> SentenceTransformer:
        """
        Load the embedding model with thread safety.
        
        Returns:
            SentenceTransformer: Loaded model instance
            
        Raises:
            EmbeddingError: If model loading fails
        """
        if self.model is None:
            with self._model_lock:
                if self.model is None:
                    try:
                        logger.info("Loading embedding model", model=self.model_name)
                        start_time = time.time()
                        
                        self.model = SentenceTransformer(self.model_name)
                        
                        # Warm up the model with a dummy embedding
                        _ = self.model.encode("warm up text", show_progress_bar=False)
                        
                        load_time = time.time() - start_time
                        logger.log_performance(
                            "Model Loading",
                            load_time,
                            model=self.model_name
                        )
                        
                    except Exception as e:
                        raise create_error(
                            EmbeddingError,
                            f"Failed to load embedding model: {str(e)}",
                            "MODEL_LOAD_FAILED",
                            model=self.model_name,
                            error_type=type(e).__name__
                        )
        
        return self.model
    
    async def generate_embeddings(
        self, 
        texts: Union[str, List[str]], 
        batch_size: int = 32
    ) -> Union[List[float], List[List[float]]]:
        """
        Generate embeddings for text(s) asynchronously.
        
        Args:
            texts: Single text string or list of texts
            batch_size: Batch size for processing multiple texts
            
        Returns:
            Union[List[float], List[List[float]]]: Single embedding or list of embeddings
            
        Raises:
            EmbeddingError: If embedding generation fails
        """
        start_time = time.time()
        is_single_text = isinstance(texts, str)
        
        if is_single_text:
            texts = [texts]
        
        logger.debug("Starting embedding generation", 
                    text_count=len(texts),
                    batch_size=batch_size)
        
        try:
            # Run embedding generation in thread pool to avoid blocking
            embeddings = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self._generate_embeddings_sync,
                texts,
                batch_size
            )
            
            duration = time.time() - start_time
            logger.log_performance(
                "Embedding Generation",
                duration,
                text_count=len(texts),
                avg_text_length=sum(len(t) for t in texts) // len(texts),
                embeddings_per_second=len(texts) / duration
            )
            
            # Return single embedding for single text
            if is_single_text:
                return embeddings[0]
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {str(e)}", 
                        text_count=len(texts))
            
            if isinstance(e, EmbeddingError):
                raise
            
            raise create_error(
                EmbeddingError,
                f"Embedding generation failed: {str(e)}",
                "EMBEDDING_FAILED",
                text_count=len(texts),
                error_type=type(e).__name__
            )
    
    def _generate_embeddings_sync(
        self, 
        texts: List[str], 
        batch_size: int
    ) -> List[List[float]]:
        """
        Synchronous embedding generation with batching.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            
        Returns:
            List[List[float]]: List of embeddings
        """
        model = self._load_model()
        
        # Process in batches to manage memory
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Generate embeddings for batch
            batch_embeddings = model.encode(
                batch_texts,
                convert_to_numpy=True,
                show_progress_bar=False,
                batch_size=len(batch_texts)
            )
            
            # Convert to list format
            if isinstance(batch_embeddings, np.ndarray):
                batch_embeddings = batch_embeddings.tolist()
            
            all_embeddings.extend(batch_embeddings)
            
            logger.debug("Processed embedding batch",
                        batch_start=i,
                        batch_size=len(batch_texts),
                        total_texts=len(texts))
        
        return all_embeddings
    
    async def generate_query_embedding(self, query: str) -> List[float]:
        """
        Generate embedding for a single query (convenience method).
        
        Args:
            query: Query text
            
        Returns:
            List[float]: Query embedding
        """
        logger.debug("Generating query embedding", query_length=len(query))
        return await self.generate_embeddings(query)
    
    async def generate_chunk_embeddings(self, chunks: List[str]) -> List[List[float]]:
        """
        Generate embeddings for document chunks with optimized batching.
        
        Args:
            chunks: List of text chunks
            
        Returns:
            List[List[float]]: List of chunk embeddings
        """
        logger.log_stage("Embedding Generation", "Starting", chunk_count=len(chunks))
        
        # Use larger batch size for chunks (they're typically shorter)
        chunk_batch_size = min(64, len(chunks))
        
        embeddings = await self.generate_embeddings(chunks, batch_size=chunk_batch_size)
        
        logger.log_stage("Embedding Generation", "Completed", 
                        chunk_count=len(chunks),
                        embedding_dimension=len(embeddings[0]) if embeddings else 0)
        
        return embeddings
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by the model.
        
        Returns:
            int: Embedding dimension
        """
        if self.model is None:
            # For BAAI/bge-m3, the dimension is 1024
            # We'll verify this when the model is loaded
            return 1024
        
        # Get dimension from model
        try:
            # Generate a dummy embedding to get dimension
            dummy_embedding = self.model.encode("test", show_progress_bar=False)
            return len(dummy_embedding)
        except Exception:
            # Fallback to known dimension for BAAI/bge-m3
            return 1024
    
    def is_model_loaded(self) -> bool:
        """
        Check if the embedding model is loaded.
        
        Returns:
            bool: True if model is loaded
        """
        return self.model is not None
    
    async def preload_model(self) -> None:
        """
        Preload the embedding model to reduce first-request latency.
        """
        logger.info("Preloading embedding model")
        
        # Load model in thread pool
        await asyncio.get_event_loop().run_in_executor(
            self.executor,
            self._load_model
        )
        
        logger.info("Embedding model preloaded successfully")
    
    def get_model_info(self) -> dict:
        """
        Get information about the embedding model.
        
        Returns:
            dict: Model information
        """
        return {
            "model_name": self.model_name,
            "is_loaded": self.is_model_loaded(),
            "embedding_dimension": self.get_embedding_dimension(),
            "max_workers": settings.max_workers
        }
    
    def __del__(self) -> None:
        """Cleanup resources when service is destroyed."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)


# Global embedding service instance
embedding_service = EmbeddingService()


async def get_embedding_service() -> EmbeddingService:
    """
    Get the global embedding service instance.
    
    Returns:
        EmbeddingService: The singleton embedding service
    """
    return embedding_service


# Convenience functions for common operations
async def embed_text(text: str) -> List[float]:
    """
    Generate embedding for a single text.
    
    Args:
        text: Text to embed
        
    Returns:
        List[float]: Text embedding
    """
    service = await get_embedding_service()
    return await service.generate_embeddings(text)


async def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for multiple texts.
    
    Args:
        texts: List of texts to embed
        
    Returns:
        List[List[float]]: List of embeddings
    """
    service = await get_embedding_service()
    return await service.generate_embeddings(texts)


async def embed_query(query: str) -> List[float]:
    """
    Generate embedding for a query (alias for embed_text).
    
    Args:
        query: Query text
        
    Returns:
        List[float]: Query embedding
    """
    return await embed_text(query)
