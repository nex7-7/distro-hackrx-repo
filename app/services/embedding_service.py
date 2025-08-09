"""
Embedding service for the RAG Application API.

This module handles text embedding generation using the BAAI/bge-m3 model.
It provides efficient batch processing and manages the embedding model lifecycle
for Stage 2 (vectorization) and Stage 4 (retrieval) of the RAG pipeline.
"""

import asyncio
import time
import os
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
        
        # Setup cache directories for HuggingFace models
        self._setup_cache_directories()
        
        # Use a more conservative thread pool to avoid resource contention
        max_workers = max(1, min(settings.max_workers, 4))  # Cap at 4 for embedding
        self.executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="embedding_worker"
        )
        self._model_lock = threading.Lock()
        
        logger.info("Embedding service initialized", 
                   model=self.model_name,
                   max_workers=max_workers)
    
    def _setup_cache_directories(self) -> None:
        """Setup cache directories for HuggingFace models."""
        import os
        import tempfile
        from pathlib import Path
        
        # Try to create cache directories in the designated app cache location
        cache_base = Path("/app/cache/huggingface")
        
        try:
            cache_base.mkdir(parents=True, exist_ok=True, mode=0o777)
            (cache_base / "transformers").mkdir(exist_ok=True)
            (cache_base / "sentence_transformers").mkdir(exist_ok=True)
            
            # Also setup torch cache
            torch_cache = Path("/app/cache/torch")
            torch_cache.mkdir(parents=True, exist_ok=True, mode=0o777)
            
            # Set environment variables for HuggingFace libraries
            os.environ["HF_HOME"] = str(cache_base)
            os.environ["TRANSFORMERS_CACHE"] = str(cache_base / "transformers")
            os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(cache_base / "sentence_transformers")
            os.environ["TORCH_HOME"] = str(torch_cache)
            
            logger.info("HuggingFace cache directories configured", cache_dir=str(cache_base))
            
        except (PermissionError, OSError) as e:
            # Fall back to temp directory
            temp_cache = Path(tempfile.gettempdir()) / "huggingface_cache"
            temp_cache.mkdir(parents=True, exist_ok=True)
            (temp_cache / "transformers").mkdir(exist_ok=True)
            (temp_cache / "sentence_transformers").mkdir(exist_ok=True)
            
            torch_temp_cache = Path(tempfile.gettempdir()) / "torch_cache"
            torch_temp_cache.mkdir(parents=True, exist_ok=True)
            
            os.environ["HF_HOME"] = str(temp_cache)
            os.environ["TRANSFORMERS_CACHE"] = str(temp_cache / "transformers")
            os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(temp_cache / "sentence_transformers")
            os.environ["TORCH_HOME"] = str(torch_temp_cache)
            
            logger.warning(f"Could not create app cache directory: {e}. Using temp cache: {temp_cache}")
    
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
                        
                        logger.debug("Creating SentenceTransformer instance")
                        self.model = SentenceTransformer(self.model_name)
                        logger.debug("SentenceTransformer instance created successfully")
                        
                        # Warm up the model with a dummy embedding
                        logger.debug("Warming up model with dummy text")
                        dummy_result = self.model.encode("warm up text", show_progress_bar=False)
                        logger.debug("Model warm-up completed", 
                                   dummy_embedding_shape=dummy_result.shape if hasattr(dummy_result, 'shape') else len(dummy_result))
                        
                        load_time = time.time() - start_time
                        logger.log_performance(
                            "Model Loading",
                            load_time,
                            model=self.model_name
                        )
                        
                    except Exception as e:
                        logger.error("Failed to load embedding model",
                                   model=self.model_name,
                                   error=str(e),
                                   error_type=type(e).__name__)
                        logger.exception("Full model loading exception")
                        raise create_error(
                            EmbeddingError,
                            f"Failed to load embedding model: {str(e)}",
                            "MODEL_LOAD_FAILED",
                            model=self.model_name,
                            error_type=type(e).__name__
                        )
        
        logger.debug("Model loading check completed - model is ready")
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
        
        logger.info("Starting embedding generation", 
                    text_count=len(texts),
                    batch_size=batch_size,
                    avg_text_length=sum(len(t) for t in texts) // len(texts))
        
        try:
            logger.info("Submitting embedding task to thread pool",
                       executor_active=not self.executor._shutdown,
                       thread_count=len(self.executor._threads) if hasattr(self.executor, '_threads') else 'unknown')
            
            # Run embedding generation in thread pool to avoid blocking
            # Add timeout to prevent hanging indefinitely
            embeddings = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    self._generate_embeddings_sync,
                    texts,
                    batch_size
                ),
                timeout=300.0  # 5 minute timeout for large batches
            )
            
            logger.info("Embedding generation completed successfully",
                       embedding_count=len(embeddings),
                       first_embedding_dim=len(embeddings[0]) if embeddings else 0)
            
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
            
        except asyncio.TimeoutError:
            logger.error("Embedding generation timed out",
                        text_count=len(texts),
                        timeout_seconds=300)
            raise create_error(
                EmbeddingError,
                "Embedding generation timed out after 5 minutes",
                "EMBEDDING_TIMEOUT",
                text_count=len(texts)
            )
        except Exception as e:
            logger.error(f"Embedding generation failed: {str(e)}", 
                        text_count=len(texts),
                        error_type=type(e).__name__)
            logger.exception("Full exception traceback")
            
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
        logger.info("Starting synchronous embedding generation",
                   total_texts=len(texts),
                   batch_size=batch_size,
                   thread_id=threading.current_thread().name)
        
        try:
            model = self._load_model()
            logger.info("Model loaded successfully for embedding generation")
            
            # Process in batches to manage memory
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_num = i // batch_size + 1
                total_batches = (len(texts) + batch_size - 1) // batch_size
                
                logger.info("Processing embedding batch",
                           batch_num=batch_num,
                           total_batches=total_batches,
                           batch_start=i,
                           batch_size=len(batch_texts),
                           total_texts=len(texts))
                
                # Log some sample text characteristics for debugging
                sample_text = batch_texts[0] if batch_texts else ""
                logger.debug("Batch text sample",
                            batch_num=batch_num,
                            sample_length=len(sample_text),
                            sample_preview=sample_text[:100] + "..." if len(sample_text) > 100 else sample_text)
                
                try:
                    # Generate embeddings for batch
                    logger.info("Calling model.encode for batch", batch_num=batch_num)
                    batch_start_time = time.time()
                    
                    batch_embeddings = model.encode(
                        batch_texts,
                        convert_to_numpy=True,
                        show_progress_bar=False,
                        batch_size=len(batch_texts)
                    )
                    
                    batch_duration = time.time() - batch_start_time
                    logger.info("Model.encode completed for batch",
                               batch_num=batch_num,
                               duration_seconds=round(batch_duration, 2),
                               embedding_shape=batch_embeddings.shape if hasattr(batch_embeddings, 'shape') else 'unknown')
                    
                    # Convert to list format
                    if isinstance(batch_embeddings, np.ndarray):
                        logger.debug("Converting numpy array to list", batch_num=batch_num)
                        batch_embeddings = batch_embeddings.tolist()
                        logger.debug("Conversion completed", batch_num=batch_num)
                    
                    all_embeddings.extend(batch_embeddings)
                    
                    logger.info("Batch processing completed",
                               batch_num=batch_num,
                               batch_embeddings_count=len(batch_embeddings),
                               total_embeddings_so_far=len(all_embeddings))
                               
                except Exception as e:
                    logger.error("Error processing embedding batch",
                                batch_num=batch_num,
                                error=str(e),
                                error_type=type(e).__name__)
                    logger.exception("Full batch processing exception")
                    raise
            
            logger.info("All embedding batches completed",
                       total_embeddings=len(all_embeddings),
                       expected_count=len(texts))
            
            return all_embeddings
            
        except Exception as e:
            logger.error("Synchronous embedding generation failed",
                        error=str(e),
                        error_type=type(e).__name__)
            logger.exception("Full synchronous generation exception")
            raise
    
    async def generate_query_embedding(self, query: str) -> List[float]:
        """
        Generate embedding for a single query (convenience method).
        
        Args:
            query: Query text
            
        Returns:
            List[float]: Query embedding
        """
        logger.info("Generating query embedding", query_length=len(query))
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
