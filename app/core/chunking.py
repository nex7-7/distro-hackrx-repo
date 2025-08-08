"""
Document chunking service for the RAG Application API.

This module handles Stage 2 of the RAG pipeline: document chunking and vectorization.
It implements smart chunking with paragraph-wise strategy, 5% overlap, and 
efficient embedding generation with multiprocessing.
"""

import asyncio
import hashlib
import time
import uuid
from typing import List, Tuple
import re

from app.models.schemas import DocumentInfo, ChunkData
from app.services.embedding_service import embedding_service
from app.services.vector_store import vector_store
from app.utils.logger import get_logger
from app.utils.exceptions import ChunkingError, create_error
from config.settings import settings

logger = get_logger(__name__)


class DocumentChunker:
    """
    Document chunking service that implements smart paragraph-wise chunking
    with configurable overlap and size constraints.
    """
    
    def __init__(self) -> None:
        """Initialize the document chunker."""
        self.min_chunk_size = settings.min_chunk_size
        self.max_chunk_size = settings.max_chunk_size
        self.overlap_size = settings.chunk_overlap_size
        
        logger.info("Document chunker initialized",
                   min_size=self.min_chunk_size,
                   max_size=self.max_chunk_size,
                   overlap_percentage=settings.chunk_overlap_percentage)
    
    async def chunk_and_store_document(self, document_info: DocumentInfo) -> bool:
        """
        Main entry point for Stage 2: chunking and vectorization.
        
        Args:
            document_info: Processed document information
            
        Returns:
            bool: True if document was processed, False if already existed
            
        Raises:
            ChunkingError: If chunking or storage fails
        """
        start_time = time.time()
        
        logger.log_stage("Document Chunking", "Starting",
                        document=document_info.filename,
                        content_hash=document_info.content_hash)
        
        try:
            # Check if document already exists in vector store
            if await vector_store.document_exists(document_info.content_hash):
                logger.info("Document already exists in vector store, skipping",
                           document_hash=document_info.content_hash)
                return False
            
            # Step 1: Chunk the document
            chunks = self._chunk_document(document_info.extracted_text)
            
            # Step 2: Generate embeddings for chunks
            chunk_texts = [chunk.content for chunk in chunks]
            embeddings = await embedding_service.generate_chunk_embeddings(chunk_texts)
            
            # Step 3: Attach embeddings to chunks
            for chunk, embedding in zip(chunks, embeddings):
                chunk.embedding = embedding
            
            # Step 4: Store chunks in vector database with multiprocessing
            await vector_store.store_chunks(chunks, document_info)
            
            duration = time.time() - start_time
            logger.log_performance(
                "Document Chunking and Storage",
                duration,
                chunk_count=len(chunks),
                chunks_per_second=len(chunks) / duration,
                text_length=len(document_info.extracted_text)
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Document chunking failed: {str(e)}",
                        document=document_info.filename)
            
            if isinstance(e, ChunkingError):
                raise
            
            raise create_error(
                ChunkingError,
                f"Failed to chunk and store document: {str(e)}",
                "CHUNKING_FAILED",
                document=document_info.filename,
                error_type=type(e).__name__
            )
    
    def _chunk_document(self, text: str) -> List[ChunkData]:
        """
        Chunk document using smart paragraph-wise strategy.
        
        Args:
            text: Document text to chunk
            
        Returns:
            List[ChunkData]: List of document chunks
        """
        logger.debug("Starting document chunking", text_length=len(text))
        
        # Step 1: Split into paragraphs
        paragraphs = self._split_into_paragraphs(text)
        
        # Step 2: Group paragraphs into chunks
        chunks = self._group_paragraphs_into_chunks(paragraphs)
        
        # Step 3: Apply overlap
        chunks_with_overlap = self._apply_chunk_overlap(chunks)
        
        # Step 4: Create ChunkData objects
        chunk_objects = []
        for i, chunk_text in enumerate(chunks_with_overlap):
            chunk_id = str(uuid.uuid4())
            
            chunk_data = ChunkData(
                chunk_id=chunk_id,
                content=chunk_text.strip(),
                chunk_index=i,
                source_page=None,  # Could be enhanced to track pages
                metadata={
                    "original_length": len(chunk_text),
                    "paragraph_count": len(chunk_text.split('\n\n'))
                }
            )
            chunk_objects.append(chunk_data)
        
        logger.debug("Document chunking completed",
                    chunk_count=len(chunk_objects),
                    avg_chunk_size=sum(len(c.content) for c in chunk_objects) // len(chunk_objects))
        
        return chunk_objects
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs based on double newlines."""
        # Split on double newlines, but preserve single newlines within paragraphs
        paragraphs = re.split(r'\n\s*\n', text)
        
        # Clean up paragraphs: strip whitespace and filter empty ones
        cleaned_paragraphs = []
        for para in paragraphs:
            cleaned = para.strip()
            if cleaned and len(cleaned) > 10:  # Filter very short paragraphs
                cleaned_paragraphs.append(cleaned)
        
        return cleaned_paragraphs
    
    def _group_paragraphs_into_chunks(self, paragraphs: List[str]) -> List[str]:
        """Group paragraphs into chunks respecting size constraints."""
        chunks = []
        current_chunk = []
        current_size = 0
        
        for paragraph in paragraphs:
            para_size = len(paragraph)
            
            # If adding this paragraph would exceed max size, finalize current chunk
            if current_size + para_size > self.max_chunk_size and current_chunk:
                chunk_text = '\n\n'.join(current_chunk)
                chunks.append(chunk_text)
                current_chunk = []
                current_size = 0
            
            # Add paragraph to current chunk
            current_chunk.append(paragraph)
            current_size += para_size + 2  # +2 for the newlines
            
            # If current chunk is large enough and we have content, could start new chunk
            # But prefer to include complete paragraphs, so continue
        
        # Add final chunk if it has content
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            chunks.append(chunk_text)
        
        # Post-process: merge chunks that are too small
        chunks = self._merge_small_chunks(chunks)
        
        return chunks
    
    def _merge_small_chunks(self, chunks: List[str]) -> List[str]:
        """Merge chunks that are smaller than min_chunk_size."""
        if not chunks:
            return chunks
        
        merged_chunks = []
        current_chunk = chunks[0]
        
        for next_chunk in chunks[1:]:
            # If current chunk is too small, try to merge with next
            if len(current_chunk) < self.min_chunk_size:
                combined = current_chunk + '\n\n' + next_chunk
                
                # If combined size is acceptable, merge
                if len(combined) <= self.max_chunk_size:
                    current_chunk = combined
                    continue
            
            # Otherwise, finalize current chunk and start new one
            merged_chunks.append(current_chunk)
            current_chunk = next_chunk
        
        # Add the final chunk
        merged_chunks.append(current_chunk)
        
        return merged_chunks
    
    def _apply_chunk_overlap(self, chunks: List[str]) -> List[str]:
        """Apply 5% overlap between consecutive chunks."""
        if len(chunks) <= 1:
            return chunks
        
        overlapped_chunks = []
        
        for i in range(len(chunks)):
            chunk = chunks[i]
            
            # For chunks after the first, add overlap from previous chunk
            if i > 0:
                prev_chunk = chunks[i - 1]
                overlap_start = max(0, len(prev_chunk) - self.overlap_size)
                overlap_text = prev_chunk[overlap_start:]
                
                # Add overlap prefix
                chunk = overlap_text + '\n\n' + chunk
            
            overlapped_chunks.append(chunk)
        
        return overlapped_chunks


# Global chunker instance
document_chunker = DocumentChunker()


async def chunk_and_store_document(document_info: DocumentInfo) -> bool:
    """
    Chunk and store document using the global chunker.
    
    Args:
        document_info: Document information
        
    Returns:
        bool: True if processed, False if already existed
    """
    return await document_chunker.chunk_and_store_document(document_info)
