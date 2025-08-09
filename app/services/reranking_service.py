"""
Reranking service using cross-encoder for the RAG Application API.

This module provides cross-encoder reranking functionality to improve
retrieval quality by reordering chunks based on query-chunk relevance.
"""

import asyncio
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

from app.models.schemas import ChunkData
from app.utils.logger import get_logger
from app.utils.exceptions import RetrievalError, create_error

logger = get_logger(__name__)


class RerankingService:
    """
    Cross-encoder reranking service for improving chunk relevance.
    
    Uses cross-encoder/ms-marco-MiniLM-L-6-v2 to rerank retrieved chunks
    based on query-chunk similarity scores.
    """
    
    def __init__(self) -> None:
        """Initialize the reranking service."""
        self.model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.executor = ThreadPoolExecutor(max_workers=2)
        self._model_loaded = False
        
        # Setup cache directories for HuggingFace models (same as embedding service)
        self._setup_cache_directories()
        
        logger.info("Reranking service initialized", 
                   model=self.model_name,
                   device=str(self.device))
    
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
            
            # Also setup torch cache
            torch_cache = Path("/app/cache/torch")
            torch_cache.mkdir(parents=True, exist_ok=True, mode=0o777)
            
            # Set environment variables for HuggingFace libraries
            os.environ["HF_HOME"] = str(cache_base)
            os.environ["TRANSFORMERS_CACHE"] = str(cache_base / "transformers")
            os.environ["TORCH_HOME"] = str(torch_cache)
            
            logger.info("HuggingFace cache directories configured for reranking", cache_dir=str(cache_base))
            
        except (PermissionError, OSError) as e:
            # Fall back to temp directory
            temp_cache = Path(tempfile.gettempdir()) / "huggingface_cache"
            temp_cache.mkdir(parents=True, exist_ok=True)
            (temp_cache / "transformers").mkdir(exist_ok=True)
            
            torch_temp_cache = Path(tempfile.gettempdir()) / "torch_cache"
            torch_temp_cache.mkdir(parents=True, exist_ok=True)
            
            os.environ["HF_HOME"] = str(temp_cache)
            os.environ["TRANSFORMERS_CACHE"] = str(temp_cache / "transformers")
            os.environ["TORCH_HOME"] = str(torch_temp_cache)
            
            logger.warning(f"Could not create app cache directory for reranking: {e}. Using temp cache: {temp_cache}")
    
    async def load_model(self) -> None:
        """Load the cross-encoder model asynchronously."""
        if not self._model_loaded:
            try:
                logger.info("Loading cross-encoder model", model=self.model_name)
                
                # Load model in thread pool to avoid blocking
                await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    self._load_model_sync
                )
                
                self._model_loaded = True
                logger.info("Cross-encoder model loaded successfully")
                
            except Exception as e:
                logger.error(f"Failed to load cross-encoder model: {str(e)}")
                raise create_error(
                    RetrievalError,
                    f"Failed to load reranking model: {str(e)}",
                    "RERANKING_MODEL_LOAD_FAILED"
                )
    
    def _load_model_sync(self) -> None:
        """Synchronously load the model."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
    
    async def rerank_chunks(
        self,
        query: str,
        chunks: List[ChunkData],
        top_k: int = 7
    ) -> List[ChunkData]:
        """
        Rerank chunks using cross-encoder scores.
        
        Args:
            query: Original user query
            chunks: List of retrieved chunks to rerank
            top_k: Number of top chunks to return
            
        Returns:
            List[ChunkData]: Reranked chunks (top_k)
            
        Raises:
            RetrievalError: If reranking fails
        """
        if not self._model_loaded:
            await self.load_model()
        
        if not chunks:
            return []
        
        if len(chunks) <= top_k:
            return chunks
        
        try:
            logger.debug("Reranking chunks", 
                        query_length=len(query),
                        chunk_count=len(chunks),
                        top_k=top_k)
            
            # Score chunks in thread pool
            scores = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self._score_chunks_sync,
                query,
                chunks
            )
            
            # Create pairs of (chunk, score) and sort by score descending
            chunk_score_pairs = list(zip(chunks, scores))
            chunk_score_pairs.sort(key=lambda x: x[1], reverse=True)
            
            # Return top_k chunks
            reranked_chunks = [chunk for chunk, _ in chunk_score_pairs[:top_k]]
            
            logger.debug("Reranking completed",
                        original_count=len(chunks),
                        reranked_count=len(reranked_chunks),
                        top_score=chunk_score_pairs[0][1] if chunk_score_pairs else 0.0)
            
            return reranked_chunks
            
        except Exception as e:
            logger.error(f"Chunk reranking failed: {str(e)}")
            
            # Fallback to original order if reranking fails
            logger.warning("Using fallback ranking due to reranking failure")
            return chunks[:top_k]
    
    def _score_chunks_sync(self, query: str, chunks: List[ChunkData]) -> List[float]:
        """
        Synchronously score chunks using cross-encoder.
        
        Args:
            query: User query
            chunks: Chunks to score
            
        Returns:
            List[float]: Relevance scores for each chunk
        """
        try:
            # Prepare query-chunk pairs
            pairs = [(query, chunk.content) for chunk in chunks]
            
            # Tokenize in batches to handle memory efficiently
            batch_size = 8
            all_scores = []
            
            for i in range(0, len(pairs), batch_size):
                batch_pairs = pairs[i:i + batch_size]
                
                # Tokenize batch
                inputs = self.tokenizer(
                    batch_pairs,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                )
                
                # Move to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Get scores
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    scores = torch.nn.functional.sigmoid(outputs.logits).squeeze(-1)
                    
                    # Convert to CPU and numpy
                    batch_scores = scores.cpu().numpy().tolist()
                    
                    # Handle single item case
                    if not isinstance(batch_scores, list):
                        batch_scores = [batch_scores]
                    
                    all_scores.extend(batch_scores)
            
            return all_scores
            
        except Exception as e:
            logger.error(f"Synchronous chunk scoring failed: {str(e)}")
            # Return uniform scores as fallback
            return [0.5] * len(chunks)
    
    def get_model_info(self) -> dict:
        """Get information about the reranking model."""
        return {
            "model_name": self.model_name,
            "device": str(self.device),
            "model_loaded": self._model_loaded,
            "cuda_available": torch.cuda.is_available()
        }
    
    def __del__(self) -> None:
        """Cleanup resources when service is destroyed."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)


# Global reranking service instance
reranking_service = RerankingService()


async def get_reranking_service() -> RerankingService:
    """
    Get the global reranking service instance.
    
    Returns:
        RerankingService: The reranking service
    """
    return reranking_service


# Convenience function
async def rerank_chunks(
    query: str,
    chunks: List[ChunkData],
    top_k: int = 7
) -> List[ChunkData]:
    """
    Rerank chunks using the global reranking service.
    
    Args:
        query: User query
        chunks: Chunks to rerank
        top_k: Number of top chunks to return
        
    Returns:
        List[ChunkData]: Reranked chunks
    """
    service = await get_reranking_service()
    return await service.rerank_chunks(query, chunks, top_k)
