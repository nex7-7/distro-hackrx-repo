"""
Complete document ingestion pipeline combining extraction and vector storage.

This module orchestrates the entire RAG pipeline from PDF extraction through 
vector storage, with async processing and intelligent duplicate detection.
"""

import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Set, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import traceback

# Import extraction components
from .data_extraction import (
    DocumentExtractionPipeline,
    ChunkingConfig,
    PDFParsingError,
    ExtractionError,
    ChunkingError,
    validate_pdf_file,
    estimate_processing_time
)

# Import storage components
from .vector_storage import (
    VectorStorage,
    VectorStorageConfig,
    VectorStorageError,
    CollectionError,
    EmbeddingError
)

# Import schemas
from ..schemas.models import (
    DocumentChunk,
    IngestionResult,
    ProcessingError
)

# Configure logging
logger = logging.getLogger(__name__)


# ===== CUSTOM EXCEPTIONS =====
class IngestionError(Exception):
    """Raised when the ingestion pipeline fails."""
    pass


class DuplicateDocumentError(Exception):
    """Raised when attempting to process an already processed document."""
    pass


# ===== CONFIGURATION CLASS =====
@dataclass
class IngestionConfig:
    """
    Complete configuration for the document ingestion pipeline.
    
    Combines extraction and storage configurations with pipeline-specific settings.
    """
    
    # Extraction configuration
    chunking_config: ChunkingConfig
    
    # Storage configuration  
    storage_config: VectorStorageConfig
    
    # Pipeline-specific settings
    enable_duplicate_checking: bool = True
    force_reprocessing: bool = False  # Override duplicate checking
    max_concurrent_documents: int = 3  # Maximum documents to process simultaneously
    max_concurrent_storage: int = 2   # Maximum concurrent storage operations
    
    # Error handling
    continue_on_error: bool = True     # Continue processing other docs if one fails
    max_retries: int = 2              # Maximum retries for failed operations
    
    # Progress tracking
    log_progress_interval: int = 5     # Log progress every N documents
    
    @classmethod
    def create_default(
        cls, 
        data_dir: str = "./data/raw",
        db_dir: str = "./chroma_db",
        collection_name: str = "policy_documents"
    ) -> "IngestionConfig":
        """
        Create a default configuration with sensible defaults.
        
        Args:
            data_dir: Directory containing PDF files
            db_dir: Directory for ChromaDB storage
            collection_name: Name of the ChromaDB collection
            
        Returns:
            IngestionConfig with default settings
        """
        chunking_config = ChunkingConfig()
        storage_config = VectorStorageConfig(
            chroma_db_path=db_dir,
            collection_name=collection_name
        )
        
        return cls(
            chunking_config=chunking_config,
            storage_config=storage_config
        )
    
    def validate(self) -> None:
        """
        Validate the configuration parameters.
        
        Raises:
            ValueError: If configuration is invalid
        """
        # Chunking config validates itself in __post_init__
        # We can test this by trying to create a new instance with the same params
        try:
            from dataclasses import asdict
            ChunkingConfig(**asdict(self.chunking_config))
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid chunking configuration: {e}")
        
        # Validate concurrency settings
        if self.max_concurrent_documents < 1:
            raise ValueError("max_concurrent_documents must be at least 1")
        if self.max_concurrent_storage < 1:
            raise ValueError("max_concurrent_storage must be at least 1")
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if self.log_progress_interval < 1:
            raise ValueError("log_progress_interval must be at least 1")
        
        # Validate paths
        if not self.storage_config.chroma_db_path.parent.exists():
            logger.warning(f"Parent directory for ChromaDB path does not exist: {self.storage_config.chroma_db_path.parent}")
        
        logger.info("Configuration validation passed")
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the configuration.
        
        Returns:
            Dictionary with configuration summary
        """
        return {
            "extraction": {
                "page_based_chunking": self.chunking_config.use_page_based_chunking,
                "merge_small_pages": self.chunking_config.merge_small_pages,
                "max_chunk_chars": self.chunking_config.max_chunk_chars,
                "extraction_threshold": self.chunking_config.extraction_threshold,
                "include_tables": self.chunking_config.include_tables
            },
            "storage": {
                "chroma_db_path": str(self.storage_config.chroma_db_path),
                "collection_name": self.storage_config.collection_name,
                "embedding_model": self.storage_config.embedding_model,
                "distance_metric": self.storage_config.distance_metric
            },
            "pipeline": {
                "enable_duplicate_checking": self.enable_duplicate_checking,
                "force_reprocessing": self.force_reprocessing,
                "max_concurrent_documents": self.max_concurrent_documents,
                "max_concurrent_storage": self.max_concurrent_storage,
                "continue_on_error": self.continue_on_error,
                "max_retries": self.max_retries,
                "log_progress_interval": self.log_progress_interval
            }
        }


# ===== MAIN INGESTION PIPELINE =====
class DocumentIngestionPipeline:
    """
    Complete document ingestion pipeline with async processing and duplicate detection.
    
    Orchestrates the entire process from PDF files to vector storage,
    following Clean Code principles and SOLID design patterns.
    """
    
    def __init__(self, config: IngestionConfig):
        """
        Initialize the complete ingestion pipeline.
        
        Args:
            config: Configuration for the ingestion pipeline
        """
        self.config = config
        
        # Validate configuration
        logger.info("Initializing Document Ingestion Pipeline...")
        try:
            self.config.validate()
        except ValueError as e:
            logger.error(f"Configuration validation failed: {e}")
            raise IngestionError(f"Invalid configuration: {e}")
        
        # Log configuration summary
        config_summary = self.config.get_summary()
        logger.info("Pipeline configuration:")
        for category, settings in config_summary.items():
            logger.info(f"  {category.upper()}:")
            for key, value in settings.items():
                logger.info(f"    {key}: {value}")
        
        # Initialize extraction pipeline
        try:
            self.extraction_pipeline = DocumentExtractionPipeline(config.chunking_config)
            logger.info("✅ Extraction pipeline initialized")
        except Exception as e:
            logger.error(f"Failed to initialize extraction pipeline: {e}")
            raise IngestionError(f"Extraction pipeline initialization failed: {e}")
        
        # Initialize vector storage
        try:
            self.vector_storage = VectorStorage(config.storage_config)
            logger.info("✅ Vector storage initialized")
            
            # Check storage connection
            stats = self.vector_storage.get_collection_stats()
            logger.info(f"Vector storage stats: {stats}")
            
        except Exception as e:
            logger.error(f"Failed to initialize vector storage: {e}")
            raise IngestionError(f"Vector storage initialization failed: {e}")
        
        # Processing statistics
        self.stats = {
            "total_documents": 0,
            "processed_documents": 0,
            "skipped_documents": 0,
            "failed_documents": 0,
            "total_chunks": 0,
            "processing_start_time": None,
            "processing_end_time": None,
            "errors": []
        }
        
        # Background storage tracking
        self._background_storage_futures: Dict[str, asyncio.Task] = {}
        self._storage_executor = ThreadPoolExecutor(
            max_workers=config.max_concurrent_storage,
            thread_name_prefix="storage-"
        )
        
        # Check for existing processed documents
        try:
            existing_docs = self.get_processed_documents()
            if existing_docs:
                logger.info(f"Found {len(existing_docs)} previously processed documents")
                if logger.isEnabledFor(logging.DEBUG):
                    for doc in existing_docs[:5]:  # Show first 5
                        logger.debug(f"  - {doc}")
                    if len(existing_docs) > 5:
                        logger.debug(f"  ... and {len(existing_docs) - 5} more")
            else:
                logger.info("No previously processed documents found")
        except Exception as e:
            logger.warning(f"Could not check for existing documents: {e}")
        
        logger.info("✅ Document Ingestion Pipeline initialized successfully")
    
    def get_processed_documents(self) -> List[str]:
        """
        Get list of documents already processed and stored in vector database.
        
        Returns:
            List of document names that have been processed
        """
        return self.vector_storage.get_processed_documents()
    
    def should_process_document(self, pdf_path: Path) -> bool:
        """
        Determine if a document should be processed based on duplicate checking.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            True if document should be processed, False if it should be skipped
        """
        document_name = pdf_path.name
        
        # If force reprocessing is enabled, always process
        if self.config.force_reprocessing:
            logger.info(f"Force reprocessing enabled - will process {document_name}")
            return True
        
        # If duplicate checking is disabled, always process
        if not self.config.enable_duplicate_checking:
            logger.info(f"Duplicate checking disabled - will process {document_name}")
            return True
        
        # Check if document has already been processed
        if self.vector_storage.is_document_processed(document_name):
            logger.info(f"Document already processed - skipping {document_name}")
            return False
        
        logger.info(f"Document not yet processed - will process {document_name}")
        return True
    
    async def process_single_document(self, pdf_path: Path) -> Dict[str, Any]:
        """
        Process a single PDF document through the complete pipeline.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary with processing results
        """
        document_name = pdf_path.name
        start_time = datetime.now()
        
        logger.info(f"Starting processing of document: {document_name}")
        
        try:
            # Validate PDF file
            if not validate_pdf_file(pdf_path):
                raise PDFParsingError(f"Invalid or corrupted PDF: {document_name}")
            
            # Check if document should be processed
            if not self.should_process_document(pdf_path):
                self.stats["skipped_documents"] += 1
                return {
                    "document_name": document_name,
                    "status": "skipped",
                    "reason": "already_processed",
                    "processing_time": 0.0
                }
            
            # Extract and chunk the document
            logger.info(f"Extracting chunks from {document_name}")
            extraction_start = datetime.now()
            
            chunks = self.extraction_pipeline.process_document(pdf_path)
            
            extraction_time = (datetime.now() - extraction_start).total_seconds()
            logger.info(f"Extracted {len(chunks)} chunks from {document_name} in {extraction_time:.2f}s")
            
            if not chunks:
                logger.warning(f"No chunks extracted from {document_name}")
                return {
                    "document_name": document_name,
                    "status": "completed",
                    "chunks_created": 0,
                    "chunks_stored": 0,
                    "processing_time": (datetime.now() - start_time).total_seconds()
                }
            
            # Store chunks asynchronously
            logger.info(f"Storing {len(chunks)} chunks for {document_name}")
            storage_result = await self.vector_storage.store_chunks_async(chunks, document_name)
            
            total_time = (datetime.now() - start_time).total_seconds()
            
            # Update statistics
            self.stats["processed_documents"] += 1
            self.stats["total_chunks"] += len(chunks)
            
            logger.info(f"Successfully processed {document_name}: {len(chunks)} chunks in {total_time:.2f}s")
            
            return {
                "document_name": document_name,
                "status": "completed",
                "chunks_created": len(chunks),
                "chunks_stored": storage_result.get("stored_count", 0),
                "processing_time": total_time,
                "extraction_time": extraction_time,
                "storage_time": storage_result.get("processing_time", 0.0)
            }
            
        except Exception as e:
            # Record error
            error = ProcessingError(
                error_type=type(e).__name__,
                error_message=str(e),
                query=document_name,
                stack_trace=traceback.format_exc()
            )
            self.stats["errors"].append(error)
            self.stats["failed_documents"] += 1
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            logger.error(f"Failed to process {document_name}: {e}")
            
            return {
                "document_name": document_name,
                "status": "failed",
                "error": str(e),
                "error_type": type(e).__name__,
                "processing_time": processing_time
            }
    
    async def process_single_document_with_background_storage(self, pdf_path: Path) -> Dict[str, Any]:
        """
        Process a single document with background storage optimization.
        
        This method starts storage in background and returns immediately after extraction,
        allowing the next document to begin processing while this one's chunks are being stored.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary with processing results (storage may still be in progress)
        """
        document_name = pdf_path.name
        start_time = datetime.now()
        
        logger.info(f"Starting processing of document with background storage: {document_name}")
        
        try:
            # Validate PDF file
            if not validate_pdf_file(pdf_path):
                raise PDFParsingError(f"Invalid or corrupted PDF: {document_name}")
            
            # Check if document should be processed
            if not self.should_process_document(pdf_path):
                self.stats["skipped_documents"] += 1
                return {
                    "document_name": document_name,
                    "status": "skipped",
                    "reason": "already_processed",
                    "processing_time": 0.0
                }
            
            # Extract and chunk the document
            logger.info(f"Extracting chunks from {document_name}")
            extraction_start = datetime.now()
            
            chunks = self.extraction_pipeline.process_document(pdf_path)
            
            extraction_time = (datetime.now() - extraction_start).total_seconds()
            logger.info(f"Extracted {len(chunks)} chunks from {document_name} in {extraction_time:.2f}s")
            
            if not chunks:
                logger.warning(f"No chunks extracted from {document_name}")
                return {
                    "document_name": document_name,
                    "status": "completed",
                    "chunks_created": 0,
                    "chunks_stored": 0,
                    "processing_time": (datetime.now() - start_time).total_seconds()
                }
            
            # Start background storage
            logger.info(f"Starting background storage for {len(chunks)} chunks from {document_name}")
            storage_future = asyncio.create_task(
                self.vector_storage.store_chunks_async(chunks, document_name)
            )
            
            # Store the future for later monitoring
            self._background_storage_futures[document_name] = storage_future
            
            # Update extraction statistics immediately
            self.stats["processed_documents"] += 1
            self.stats["total_chunks"] += len(chunks)
            
            extraction_time_total = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"Extraction completed for {document_name}: {len(chunks)} chunks in {extraction_time:.2f}s (storage in background)")
            
            return {
                "document_name": document_name,
                "status": "extraction_completed",
                "chunks_created": len(chunks),
                "chunks_stored": "in_progress",
                "processing_time": extraction_time_total,
                "extraction_time": extraction_time,
                "storage_future": storage_future  # For monitoring
            }
            
        except Exception as e:
            # Record error
            error = ProcessingError(
                error_type=type(e).__name__,
                error_message=str(e),
                query=document_name,
                stack_trace=traceback.format_exc()
            )
            self.stats["errors"].append(error)
            self.stats["failed_documents"] += 1
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            logger.error(f"Failed to process {document_name}: {e}")
            
            return {
                "document_name": document_name,
                "status": "failed",
                "error": str(e),
                "error_type": type(e).__name__,
                "processing_time": processing_time
            }
    
    async def wait_for_background_storage(self, timeout: Optional[float] = None) -> Dict[str, Dict[str, Any]]:
        """
        Wait for all background storage operations to complete.
        
        Args:
            timeout: Optional timeout in seconds
            
        Returns:
            Dictionary mapping document names to their storage results
        """
        if not self._background_storage_futures:
            logger.info("No background storage operations to wait for")
            return {}
        
        logger.info(f"Waiting for {len(self._background_storage_futures)} background storage operations to complete")
        
        storage_results = {}
        
        try:
            # Wait for all storage operations to complete
            if timeout:
                done, pending = await asyncio.wait(
                    list(self._background_storage_futures.values()),
                    timeout=timeout,
                    return_when=asyncio.ALL_COMPLETED
                )
                
                # Cancel any pending operations
                for task in pending:
                    task.cancel()
                    logger.warning("Background storage operation timed out and was cancelled")
            else:
                results = await asyncio.gather(
                    *self._background_storage_futures.values(),
                    return_exceptions=True
                )
            
            # Collect results
            for doc_name, future in self._background_storage_futures.items():
                try:
                    if future.done():
                        result = await future
                        storage_results[doc_name] = result
                        logger.info(f"Background storage completed for {doc_name}: {result.get('stored_count', 0)} chunks")
                    else:
                        storage_results[doc_name] = {
                            "stored_count": 0,
                            "error": "timeout",
                            "document_name": doc_name
                        }
                except Exception as e:
                    logger.error(f"Background storage failed for {doc_name}: {e}")
                    storage_results[doc_name] = {
                        "stored_count": 0,
                        "error": str(e),
                        "document_name": doc_name
                    }
        
        finally:
            # Clear the futures
            self._background_storage_futures.clear()
        
        return storage_results
    
    async def process_documents_with_background_storage(
        self,
        pdf_paths: List[Path],
        progress_callback: Optional[Callable[[int, int, Dict[str, Any]], None]] = None
    ) -> IngestionResult:
        """
        Process multiple PDF documents with optimized background storage.
        
        This method starts storage for each document in the background while continuing
        to process the next document, maximizing throughput by overlapping I/O and computation.
        
        Args:
            pdf_paths: List of paths to PDF files
            progress_callback: Optional callback function for progress updates
            
        Returns:
            IngestionResult with comprehensive statistics
        """
        if not pdf_paths:
            logger.warning("No PDF files provided for processing")
            return self._create_empty_result()
        
        logger.info(f"Starting optimized batch processing of {len(pdf_paths)} documents with background storage")
        
        # Initialize statistics
        self.stats["total_documents"] = len(pdf_paths)
        self.stats["processing_start_time"] = datetime.now()
        
        # Process documents with concurrency control for extraction
        # Storage happens in background automatically
        semaphore = asyncio.Semaphore(self.config.max_concurrent_documents)
        
        async def process_with_semaphore(pdf_path: Path) -> Dict[str, Any]:
            async with semaphore:
                return await self.process_single_document_with_background_storage(pdf_path)
        
        # Create tasks for all documents (extraction phase)
        extraction_tasks = [process_with_semaphore(pdf_path) for pdf_path in pdf_paths]
        
        # Process documents and collect extraction results
        extraction_results = []
        completed = 0
        
        logger.info("Starting extraction phase for all documents...")
        
        for future in asyncio.as_completed(extraction_tasks):
            try:
                result = await future
                extraction_results.append(result)
                completed += 1
                
                # Call progress callback if provided
                if progress_callback:
                    progress_callback(completed, len(pdf_paths), result)
                
                # Log progress at intervals
                if completed % self.config.log_progress_interval == 0:
                    logger.info(f"Extraction progress: {completed}/{len(pdf_paths)} documents processed")
                    
            except Exception as e:
                logger.error(f"Unexpected error in document extraction: {e}")
                self.stats["failed_documents"] += 1
                continue
        
        logger.info(f"Extraction phase completed for {len(extraction_results)} documents")
        
        # Now wait for all background storage operations to complete
        logger.info("Waiting for background storage operations to complete...")
        storage_results = await self.wait_for_background_storage(timeout=300)  # 5 minute timeout
        
        # Merge extraction and storage results
        final_results = []
        for extraction_result in extraction_results:
            doc_name = extraction_result["document_name"]
            storage_result = storage_results.get(doc_name, {})
            
            # Merge the results
            merged_result = extraction_result.copy()
            if "stored_count" in storage_result:
                merged_result["chunks_stored"] = storage_result["stored_count"]
                merged_result["storage_time"] = storage_result.get("processing_time", 0.0)
                merged_result["status"] = "completed"
            elif "error" in storage_result:
                merged_result["storage_error"] = storage_result["error"]
                merged_result["status"] = "storage_failed"
            
            final_results.append(merged_result)
        
        self.stats["processing_end_time"] = datetime.now()
        
        # Create final result
        return self._create_ingestion_result(final_results)
    
    async def process_single_document_with_retry(
        self, 
        pdf_path: Path, 
        use_background_storage: bool = True
    ) -> Dict[str, Any]:
        """
        Process a single document with retry logic for robustness.
        
        Args:
            pdf_path: Path to the PDF file
            use_background_storage: Whether to use background storage optimization
            
        Returns:
            Dictionary with processing results
        """
        document_name = pdf_path.name
        last_error = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                if attempt > 0:
                    logger.info(f"Retry attempt {attempt} for {document_name}")
                    # Small delay before retry to allow transient issues to resolve
                    await asyncio.sleep(min(attempt * 2, 10))  # Exponential backoff, max 10s
                
                if use_background_storage:
                    return await self.process_single_document_with_background_storage(pdf_path)
                else:
                    return await self.process_single_document(pdf_path)
                    
            except (PDFParsingError, ExtractionError, ChunkingError) as e:
                # These are likely permanent errors, don't retry
                logger.error(f"Permanent error processing {document_name}: {e}")
                last_error = e
                break
                
            except (VectorStorageError, CollectionError, EmbeddingError) as e:
                # Storage errors might be transient, worth retrying
                logger.warning(f"Storage error processing {document_name} (attempt {attempt + 1}): {e}")
                last_error = e
                if attempt == self.config.max_retries:
                    logger.error(f"Max retries exceeded for {document_name}")
                continue
                
            except Exception as e:
                # Unknown errors, worth retrying but with caution
                logger.warning(f"Unknown error processing {document_name} (attempt {attempt + 1}): {e}")
                last_error = e
                if attempt == self.config.max_retries:
                    logger.error(f"Max retries exceeded for {document_name}")
                continue
        
        # If we get here, all retries failed
        error = ProcessingError(
            error_type=type(last_error).__name__ if last_error else "UnknownError",
            error_message=str(last_error) if last_error else "Maximum retries exceeded",
            query=document_name,
            stack_trace=traceback.format_exc()
        )
        self.stats["errors"].append(error)
        self.stats["failed_documents"] += 1
        
        return {
            "document_name": document_name,
            "status": "failed_after_retries",
            "error": str(last_error) if last_error else "Maximum retries exceeded",
            "error_type": type(last_error).__name__ if last_error else "UnknownError",
            "attempts": self.config.max_retries + 1,
            "processing_time": 0.0
        }
    
    async def process_documents_robust(
        self,
        pdf_paths: List[Path],
        use_background_storage: bool = True,
        progress_callback: Optional[Callable[[int, int, Dict[str, Any]], None]] = None
    ) -> IngestionResult:
        """
        Process multiple documents with full error handling and retry logic.
        
        This is the most robust processing method, recommended for production use.
        
        Args:
            pdf_paths: List of paths to PDF files
            use_background_storage: Whether to use background storage optimization
            progress_callback: Optional callback function for progress updates
            
        Returns:
            IngestionResult with comprehensive statistics
        """
        if not pdf_paths:
            logger.warning("No PDF files provided for processing")
            return self._create_empty_result()
        
        logger.info(f"Starting robust batch processing of {len(pdf_paths)} documents")
        
        # Initialize statistics
        self.stats["total_documents"] = len(pdf_paths)
        self.stats["processing_start_time"] = datetime.now()
        
        # Process documents with concurrency control
        semaphore = asyncio.Semaphore(self.config.max_concurrent_documents)
        
        async def process_with_semaphore_and_retry(pdf_path: Path) -> Dict[str, Any]:
            async with semaphore:
                return await self.process_single_document_with_retry(pdf_path, use_background_storage)
        
        # Create tasks for all documents
        tasks = [process_with_semaphore_and_retry(pdf_path) for pdf_path in pdf_paths]
        
        # Process documents and collect results
        results = []
        completed = 0
        
        for future in asyncio.as_completed(tasks):
            try:
                result = await future
                results.append(result)
                completed += 1
                
                # Call progress callback if provided
                if progress_callback:
                    progress_callback(completed, len(pdf_paths), result)
                
                # Log progress at intervals
                if completed % self.config.log_progress_interval == 0:
                    logger.info(f"Robust processing progress: {completed}/{len(pdf_paths)} documents")
                    
                # Log individual results
                status = result.get("status", "unknown")
                doc_name = result.get("document_name", "unknown")
                
                if status == "completed":
                    chunks = result.get("chunks_created", 0)
                    logger.debug(f"✅ {doc_name}: {chunks} chunks")
                elif status == "skipped":
                    logger.debug(f"⏭️  {doc_name}: skipped ({result.get('reason', 'unknown')})")
                elif status in ["failed", "failed_after_retries"]:
                    error = result.get("error", "unknown")
                    logger.debug(f"❌ {doc_name}: failed ({error})")
                    
            except Exception as e:
                logger.error(f"Unexpected error in robust document processing: {e}")
                # Create error result for unexpected failures
                error_result = {
                    "document_name": "unknown",
                    "status": "unexpected_failure",
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "processing_time": 0.0
                }
                results.append(error_result)
                self.stats["failed_documents"] += 1
                continue
        
        # If using background storage, wait for all storage operations
        if use_background_storage:
            logger.info("Waiting for background storage operations to complete...")
            storage_results = await self.wait_for_background_storage(timeout=600)  # 10 minute timeout
            
            # Update results with storage information
            for result in results:
                doc_name = result.get("document_name")
                if doc_name and doc_name in storage_results:
                    storage_result = storage_results[doc_name]
                    if "stored_count" in storage_result:
                        result["chunks_stored"] = storage_result["stored_count"]
                        result["storage_time"] = storage_result.get("processing_time", 0.0)
                        if result.get("status") == "extraction_completed":
                            result["status"] = "completed"
                    elif "error" in storage_result:
                        result["storage_error"] = storage_result["error"]
                        if result.get("status") == "extraction_completed":
                            result["status"] = "storage_failed"
        
        self.stats["processing_end_time"] = datetime.now()
        
        # Log final summary
        total_time = (self.stats["processing_end_time"] - self.stats["processing_start_time"]).total_seconds()
        successful = sum(1 for r in results if r.get("status") == "completed")
        failed = sum(1 for r in results if r.get("status", "").startswith("failed"))
        skipped = sum(1 for r in results if r.get("status") == "skipped")
        
        logger.info(f"Robust processing completed in {total_time:.2f}s:")
        logger.info(f"  ✅ Successful: {successful}")
        logger.info(f"  ❌ Failed: {failed}")
        logger.info(f"  ⏭️  Skipped: {skipped}")
        logger.info(f"  📊 Total chunks: {self.stats['total_chunks']}")
        
        return self._create_ingestion_result(results)
    
    async def process_documents(
        self, 
        pdf_paths: List[Path],
        progress_callback: Optional[Callable[[int, int, Dict[str, Any]], None]] = None
    ) -> IngestionResult:
        """
        Process multiple PDF documents with concurrent processing.
        
        Args:
            pdf_paths: List of paths to PDF files
            progress_callback: Optional callback function for progress updates
            
        Returns:
            IngestionResult with comprehensive statistics
        """
        if not pdf_paths:
            logger.warning("No PDF files provided for processing")
            return self._create_empty_result()
        
        logger.info(f"Starting batch processing of {len(pdf_paths)} documents")
        
        # Initialize statistics
        self.stats["total_documents"] = len(pdf_paths)
        self.stats["processing_start_time"] = datetime.now()
        
        # Process documents with concurrency control
        semaphore = asyncio.Semaphore(self.config.max_concurrent_documents)
        
        async def process_with_semaphore(pdf_path: Path) -> Dict[str, Any]:
            async with semaphore:
                return await self.process_single_document(pdf_path)
        
        # Create tasks for all documents
        tasks = [process_with_semaphore(pdf_path) for pdf_path in pdf_paths]
        
        # Process documents and collect results
        results = []
        completed = 0
        
        for future in asyncio.as_completed(tasks):
            try:
                result = await future
                results.append(result)
                completed += 1
                
                # Call progress callback if provided
                if progress_callback:
                    progress_callback(completed, len(pdf_paths), result)
                
                # Log progress at intervals
                if completed % self.config.log_progress_interval == 0:
                    logger.info(f"Progress: {completed}/{len(pdf_paths)} documents processed")
                    
            except Exception as e:
                logger.error(f"Unexpected error in document processing: {e}")
                self.stats["failed_documents"] += 1
                continue
        
        self.stats["processing_end_time"] = datetime.now()
        
        # Create final result
        return self._create_ingestion_result(results)
    
    async def process_directory(
        self, 
        directory_path: Path,
        pattern: str = "*.pdf",
        progress_callback: Optional[Callable[[int, int, Dict[str, Any]], None]] = None
    ) -> IngestionResult:
        """
        Process all PDF files in a directory.
        
        Args:
            directory_path: Path to directory containing PDF files
            pattern: File pattern to match (default: "*.pdf")
            progress_callback: Optional callback for progress updates
            
        Returns:
            IngestionResult with processing statistics
        """
        if not directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        if not directory_path.is_dir():
            raise ValueError(f"Path is not a directory: {directory_path}")
        
        # Find all PDF files
        pdf_files = list(directory_path.glob(pattern))
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {directory_path} matching pattern '{pattern}'")
            return self._create_empty_result()
        
        logger.info(f"Found {len(pdf_files)} PDF files in {directory_path}")
        
        # Process the files
        return await self.process_documents(pdf_files, progress_callback)
    
    def _create_ingestion_result(self, processing_results: List[Dict[str, Any]]) -> IngestionResult:
        """
        Create an IngestionResult from processing results.
        
        Args:
            processing_results: List of individual document processing results
            
        Returns:
            IngestionResult with comprehensive statistics
        """
        total_time = 0.0
        if (self.stats["processing_start_time"] and 
            self.stats["processing_end_time"]):
            total_time = (
                self.stats["processing_end_time"] - 
                self.stats["processing_start_time"]
            ).total_seconds()
        
        # Calculate average chunk size (approximation)
        avg_chunk_size = 1000.0  # Default estimate
        if self.stats["total_chunks"] > 0:
            # This could be improved by tracking actual chunk sizes
            avg_chunk_size = 1200.0  # Slightly better estimate
        
        return IngestionResult(
            total_documents=self.stats["total_documents"],
            total_chunks=self.stats["total_chunks"],
            successful_documents=self.stats["processed_documents"],
            failed_documents=self.stats["failed_documents"],
            processing_time=total_time,
            average_chunk_size=avg_chunk_size,
            errors=self.stats["errors"]
        )
    
    def _create_empty_result(self) -> IngestionResult:
        """Create an empty IngestionResult for edge cases."""
        return IngestionResult(
            total_documents=0,
            total_chunks=0,
            successful_documents=0,
            failed_documents=0,
            processing_time=0.0,
            average_chunk_size=0.0,
            errors=[]
        )
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """
        Get detailed processing statistics.
        
        Returns:
            Dictionary with comprehensive processing statistics
        """
        stats = self.stats.copy()
        
        # Add derived statistics
        if stats["total_documents"] > 0:
            stats["success_rate"] = stats["processed_documents"] / stats["total_documents"]
            stats["skip_rate"] = stats["skipped_documents"] / stats["total_documents"]
            stats["failure_rate"] = stats["failed_documents"] / stats["total_documents"]
        else:
            stats["success_rate"] = 0.0
            stats["skip_rate"] = 0.0
            stats["failure_rate"] = 0.0
        
        if stats["processed_documents"] > 0:
            stats["avg_chunks_per_document"] = stats["total_chunks"] / stats["processed_documents"]
        else:
            stats["avg_chunks_per_document"] = 0.0
        
        return stats
    
    def reset_statistics(self) -> None:
        """
        Reset processing statistics.
        
        Useful when reusing the same pipeline instance for multiple processing runs.
        """
        logger.info("Resetting processing statistics")
        self.stats = {
            "total_documents": 0,
            "processed_documents": 0,
            "skipped_documents": 0,
            "failed_documents": 0,
            "total_chunks": 0,
            "processing_start_time": None,
            "processing_end_time": None,
            "errors": []
        }
    
    def get_processing_progress(self) -> Dict[str, Any]:
        """
        Get current processing progress information.
        
        Returns:
            Dictionary with current progress metrics
        """
        total = self.stats["total_documents"]
        completed = (self.stats["processed_documents"] + 
                    self.stats["skipped_documents"] + 
                    self.stats["failed_documents"])
        
        progress = {
            "total_documents": total,
            "completed_documents": completed,
            "processed_documents": self.stats["processed_documents"],
            "skipped_documents": self.stats["skipped_documents"],
            "failed_documents": self.stats["failed_documents"],
            "total_chunks": self.stats["total_chunks"],
            "completion_percentage": (completed / total * 100) if total > 0 else 0.0,
            "active_background_storage": len(self._background_storage_futures)
        }
        
        # Add timing information if available
        if self.stats["processing_start_time"]:
            elapsed = (datetime.now() - self.stats["processing_start_time"]).total_seconds()
            progress["elapsed_time"] = elapsed
            
            if completed > 0:
                rate = completed / elapsed
                progress["processing_rate"] = rate  # documents per second
                
                if total > completed:
                    remaining = total - completed
                    estimated_remaining_time = remaining / rate
                    progress["estimated_remaining_time"] = estimated_remaining_time
        
        return progress
    
    def log_progress_summary(self) -> None:
        """
        Log a comprehensive progress summary.
        """
        progress = self.get_processing_progress()
        
        logger.info("=== PROCESSING PROGRESS SUMMARY ===")
        logger.info(f"Total documents: {progress['total_documents']}")
        logger.info(f"Completed: {progress['completed_documents']} ({progress['completion_percentage']:.1f}%)")
        logger.info(f"  ✅ Processed: {progress['processed_documents']}")
        logger.info(f"  ⏭️  Skipped: {progress['skipped_documents']}")
        logger.info(f"  ❌ Failed: {progress['failed_documents']}")
        logger.info(f"📊 Total chunks created: {progress['total_chunks']}")
        
        if "elapsed_time" in progress:
            logger.info(f"⏱️  Elapsed time: {progress['elapsed_time']:.1f}s")
            if "processing_rate" in progress:
                logger.info(f"🚀 Processing rate: {progress['processing_rate']:.2f} docs/sec")
            if "estimated_remaining_time" in progress:
                logger.info(f"⏳ Estimated remaining: {progress['estimated_remaining_time']:.1f}s")
        
        if progress["active_background_storage"] > 0:
            logger.info(f"🔄 Active background storage operations: {progress['active_background_storage']}")
        
        # Show recent errors if any
        recent_errors = self.stats["errors"][-3:]  # Last 3 errors
        if recent_errors:
            logger.info("Recent errors:")
            for error in recent_errors:
                logger.info(f"  ❌ {error.query}: {error.error_type} - {error.error_message}")
    
    def create_progress_callback(self, log_level: int = logging.INFO) -> Callable[[int, int, Dict[str, Any]], None]:
        """
        Create a progress callback function with customizable logging.
        
        Args:
            log_level: Logging level for progress messages
            
        Returns:
            Progress callback function
        """
        def progress_callback(completed: int, total: int, result: Dict[str, Any]) -> None:
            doc_name = result.get("document_name", "unknown")
            status = result.get("status", "unknown")
            
            # Create status emoji
            status_emoji = {
                "completed": "✅",
                "skipped": "⏭️",
                "failed": "❌",
                "failed_after_retries": "❌",
                "extraction_completed": "🔄",
                "storage_failed": "⚠️",
                "unexpected_failure": "💥"
            }.get(status, "❓")
            
            # Log individual completion
            message = f"{status_emoji} [{completed}/{total}] {doc_name}"
            
            if status == "completed":
                chunks = result.get("chunks_created", 0)
                time_taken = result.get("processing_time", 0)
                message += f" ({chunks} chunks, {time_taken:.2f}s)"
            elif status == "skipped":
                reason = result.get("reason", "unknown")
                message += f" (skipped: {reason})"
            elif status in ["failed", "failed_after_retries"]:
                error = result.get("error", "unknown")
                message += f" (error: {error})"
            
            logger.log(log_level, message)
            
            # Log summary at intervals
            if completed % (self.config.log_progress_interval * 2) == 0:
                progress = self.get_processing_progress()
                completion_pct = progress.get("completion_percentage", 0)
                rate = progress.get("processing_rate", 0)
                logger.log(log_level, f"📊 Progress: {completion_pct:.1f}% complete, {rate:.2f} docs/sec")
        
        return progress_callback
    
    async def process_directory_comprehensive(
        self,
        directory_path: Path,
        pattern: str = "*.pdf",
        use_background_storage: bool = True,
        auto_progress_logging: bool = True
    ) -> IngestionResult:
        """
        Most comprehensive directory processing method with all optimizations.
        
        This is the recommended method for production use, combining all features:
        - Background storage optimization
        - Retry logic
        - Progress tracking
        - Error handling
        - Duplicate detection
        
        Args:
            directory_path: Path to directory containing PDF files
            pattern: File pattern to match (default: "*.pdf")
            use_background_storage: Whether to use background storage optimization
            auto_progress_logging: Whether to automatically log progress
            
        Returns:
            IngestionResult with comprehensive statistics
        """
        logger.info(f"Starting comprehensive processing of directory: {directory_path}")
        
        # Validate directory
        if not directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        if not directory_path.is_dir():
            raise ValueError(f"Path is not a directory: {directory_path}")
        
        # Find PDF files
        pdf_files = list(directory_path.glob(pattern))
        if not pdf_files:
            logger.warning(f"No PDF files found in {directory_path} matching pattern '{pattern}'")
            return self._create_empty_result()
        
        # Log preprocessing information
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        # Estimate processing time
        estimated_time = estimate_total_processing_time(directory_path)
        logger.info(f"Estimated processing time: {estimated_time:.1f} seconds")
        
        # Create progress callback if auto logging is enabled
        progress_callback = None
        if auto_progress_logging:
            progress_callback = self.create_progress_callback()
        
        # Reset statistics for this run
        self.reset_statistics()
        
        # Process the files using robust method
        try:
            result = await self.process_documents_robust(
                pdf_files, 
                use_background_storage=use_background_storage,
                progress_callback=progress_callback
            )
            
            # Log final comprehensive summary
            self.log_progress_summary()
            
            return result
            
        except Exception as e:
            logger.error(f"Comprehensive processing failed: {e}")
            raise IngestionError(f"Directory processing failed: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check of the ingestion pipeline.
        
        Returns:
            Dictionary with health check results
        """
        health = {
            "status": "unknown",
            "timestamp": datetime.now().isoformat(),
            "components": {},
            "errors": []
        }
        
        try:
            # Check extraction pipeline
            try:
                # Test with a simple config validation
                test_config = ChunkingConfig()
                health["components"]["extraction"] = {
                    "status": "healthy",
                    "config_valid": True
                }
            except Exception as e:
                health["components"]["extraction"] = {
                    "status": "error",
                    "error": str(e)
                }
                health["errors"].append(f"Extraction pipeline: {e}")
            
            # Check vector storage
            try:
                storage_stats = self.vector_storage.get_collection_stats()
                health["components"]["storage"] = {
                    "status": "healthy",
                    "stats": storage_stats
                }
            except Exception as e:
                health["components"]["storage"] = {
                    "status": "error",
                    "error": str(e)
                }
                health["errors"].append(f"Vector storage: {e}")
            
            # Check processed documents
            try:
                processed_docs = self.get_processed_documents()
                health["components"]["document_tracking"] = {
                    "status": "healthy",
                    "processed_document_count": len(processed_docs)
                }
            except Exception as e:
                health["components"]["document_tracking"] = {
                    "status": "error",
                    "error": str(e)
                }
                health["errors"].append(f"Document tracking: {e}")
            
            # Overall status
            if health["errors"]:
                health["status"] = "degraded" if len(health["errors"]) < len(health["components"]) else "error"
            else:
                health["status"] = "healthy"
            
        except Exception as e:
            health["status"] = "error"
            health["errors"].append(f"Health check failed: {e}")
        
        return health
    
    def cleanup(self) -> None:
        """
        Cleanup resources and shut down thread pools.
        """
        logger.info("Cleaning up ingestion pipeline resources...")
        
        # Shutdown storage executor
        if hasattr(self, '_storage_executor') and self._storage_executor:
            self._storage_executor.shutdown(wait=True)
        
        # Cleanup vector storage
        if hasattr(self, 'vector_storage') and self.vector_storage:
            self.vector_storage.cleanup()
        
        logger.info("Ingestion pipeline cleanup completed")


# ===== UTILITY FUNCTIONS =====
async def process_documents_simple(
    pdf_directory: Union[str, Path],
    db_directory: Union[str, Path] = "./chroma_db",
    collection_name: str = "policy_documents",
    force_reprocessing: bool = False,
    max_concurrent: int = 3
) -> IngestionResult:
    """
    Simple utility function to process documents with default configuration.
    
    Args:
        pdf_directory: Directory containing PDF files
        db_directory: Directory for ChromaDB storage
        collection_name: Name of the ChromaDB collection
        force_reprocessing: Whether to reprocess already-processed documents
        max_concurrent: Maximum concurrent documents to process
        
    Returns:
        IngestionResult with processing statistics
    """
    # Create configuration
    config = IngestionConfig.create_default(
        data_dir=str(pdf_directory),
        db_dir=str(db_directory),
        collection_name=collection_name
    )
    config.force_reprocessing = force_reprocessing
    config.max_concurrent_documents = max_concurrent
    
    # Create pipeline and process
    pipeline = DocumentIngestionPipeline(config)
    
    try:
        result = await pipeline.process_directory_comprehensive(
            Path(pdf_directory),
            use_background_storage=True,
            auto_progress_logging=True
        )
        return result
    finally:
        pipeline.cleanup()


async def process_documents_advanced(
    pdf_directory: Union[str, Path],
    config: Optional[IngestionConfig] = None,
    **kwargs
) -> IngestionResult:
    """
    Advanced utility function with full configuration control.
    
    Args:
        pdf_directory: Directory containing PDF files
        config: Optional custom configuration (will create default if not provided)
        **kwargs: Additional configuration overrides
        
    Returns:
        IngestionResult with processing statistics
    """
    # Create or use provided configuration
    if config is None:
        config = IngestionConfig.create_default()
    
    # Apply any keyword argument overrides
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            logger.warning(f"Unknown configuration parameter: {key}")
    
    # Create pipeline and process
    pipeline = DocumentIngestionPipeline(config)
    
    try:
        # Perform health check first
        health = await pipeline.health_check()
        if health["status"] == "error":
            raise IngestionError(f"Pipeline health check failed: {health['errors']}")
        elif health["status"] == "degraded":
            logger.warning(f"Pipeline health check shows degraded status: {health['errors']}")
        
        # Process documents
        result = await pipeline.process_directory_comprehensive(
            Path(pdf_directory),
            use_background_storage=True,
            auto_progress_logging=True
        )
        return result
    finally:
        pipeline.cleanup()


def check_processing_prerequisites(
    pdf_directory: Union[str, Path],
    db_directory: Union[str, Path] = "./chroma_db"
) -> Dict[str, Any]:
    """
    Check prerequisites for document processing.
    
    Args:
        pdf_directory: Directory containing PDF files
        db_directory: Directory for ChromaDB storage
        
    Returns:
        Dictionary with prerequisite check results
    """
    checks = {
        "status": "unknown",
        "pdf_directory": {},
        "db_directory": {},
        "pdf_files": {},
        "recommendations": []
    }
    
    pdf_path = Path(pdf_directory)
    db_path = Path(db_directory)
    
    # Check PDF directory
    if not pdf_path.exists():
        checks["pdf_directory"] = {
            "exists": False,
            "error": f"Directory does not exist: {pdf_path}"
        }
        checks["recommendations"].append(f"Create PDF directory: {pdf_path}")
    elif not pdf_path.is_dir():
        checks["pdf_directory"] = {
            "exists": True,
            "is_directory": False,
            "error": f"Path is not a directory: {pdf_path}"
        }
        checks["recommendations"].append(f"Ensure path is a directory: {pdf_path}")
    else:
        checks["pdf_directory"] = {
            "exists": True,
            "is_directory": True,
            "readable": pdf_path.is_dir() and pdf_path.stat().st_mode & 0o444
        }
    
    # Check DB directory
    if not db_path.parent.exists():
        checks["db_directory"] = {
            "parent_exists": False,
            "error": f"Parent directory does not exist: {db_path.parent}"
        }
        checks["recommendations"].append(f"Create parent directory: {db_path.parent}")
    else:
        checks["db_directory"] = {
            "parent_exists": True,
            "writable": db_path.parent.is_dir() and db_path.parent.stat().st_mode & 0o222
        }
    
    # Check PDF files
    if checks["pdf_directory"].get("exists") and checks["pdf_directory"].get("is_directory"):
        pdf_files = list(pdf_path.glob("*.pdf"))
        checks["pdf_files"] = {
            "count": len(pdf_files),
            "files": [f.name for f in pdf_files[:10]]  # First 10 files
        }
        
        if len(pdf_files) == 0:
            checks["recommendations"].append("No PDF files found - add PDF files to process")
        elif len(pdf_files) > 100:
            checks["recommendations"].append(f"Large number of PDFs ({len(pdf_files)}) - consider batch processing")
        
        # Check a few files for validity
        invalid_files = []
        for pdf_file in pdf_files[:5]:  # Check first 5
            if not validate_pdf_file(pdf_file):
                invalid_files.append(pdf_file.name)
        
        if invalid_files:
            checks["pdf_files"]["invalid_files"] = invalid_files
            checks["recommendations"].append(f"Some PDF files appear to be invalid: {invalid_files[:3]}")
    
    # Overall status
    if checks["pdf_directory"].get("exists") and checks["pdf_directory"].get("is_directory"):
        if checks["pdf_files"].get("count", 0) > 0:
            if len(checks["recommendations"]) == 0:
                checks["status"] = "ready"
            else:
                checks["status"] = "ready_with_warnings"
        else:
            checks["status"] = "no_files"
    else:
        checks["status"] = "not_ready"
    
    return checks


def estimate_total_processing_time(pdf_directory: Path) -> float:
    """
    Estimate total processing time for all PDFs in a directory.
    
    Args:
        pdf_directory: Directory containing PDF files
        
    Returns:
        Estimated processing time in seconds
    """
    if not pdf_directory.exists():
        return 0.0
    
    pdf_files = list(pdf_directory.glob("*.pdf"))
    total_time = 0.0
    
    for pdf_path in pdf_files:
        total_time += estimate_processing_time(pdf_path)
    
    return total_time


# Make classes available for import
__all__ = [
    # Main classes
    "DocumentIngestionPipeline",
    "IngestionConfig", 
    
    # Exceptions
    "IngestionError",
    "DuplicateDocumentError",
    
    # Utility functions
    "process_documents_simple",
    "process_documents_advanced",
    "estimate_total_processing_time",
    "check_processing_prerequisites"
]


# ===== USAGE EXAMPLES =====
"""
Example usage of the Document Ingestion Pipeline:

# Simple usage - process all PDFs in a directory
import asyncio
from src.components.data_ingestion import process_documents_simple

async def main():
    result = await process_documents_simple(
        pdf_directory="./data/raw",
        db_directory="./chroma_db",
        force_reprocessing=False
    )
    print(f"Processed {result.successful_documents} documents, {result.total_chunks} chunks")

asyncio.run(main())

# Advanced usage with custom configuration
from src.components.data_ingestion import DocumentIngestionPipeline, IngestionConfig
from src.components.data_extraction import ChunkingConfig
from src.components.vector_storage import VectorStorageConfig

async def advanced_processing():
    # Custom configuration
    chunking_config = ChunkingConfig(
        max_chunk_chars=3000,
        merge_small_pages=True,
        extraction_threshold=0.85
    )
    
    storage_config = VectorStorageConfig(
        chroma_db_path="./custom_db",
        collection_name="my_documents",
        embedding_model="all-MiniLM-L12-v2"
    )
    
    config = IngestionConfig(
        chunking_config=chunking_config,
        storage_config=storage_config,
        max_concurrent_documents=5,
        enable_duplicate_checking=True,
        force_reprocessing=False
    )
    
    # Create and use pipeline
    pipeline = DocumentIngestionPipeline(config)
    
    try:
        # Health check
        health = await pipeline.health_check()
        print(f"Pipeline health: {health['status']}")
        
        # Process documents
        result = await pipeline.process_directory_comprehensive(
            Path("./data/raw"),
            use_background_storage=True,
            auto_progress_logging=True
        )
        
        # Get detailed statistics
        stats = pipeline.get_processing_statistics()
        print(f"Success rate: {stats['success_rate']:.2%}")
        print(f"Average chunks per document: {stats['avg_chunks_per_document']:.1f}")
        
    finally:
        pipeline.cleanup()

# Check prerequisites before processing
from src.components.data_ingestion import check_processing_prerequisites

def check_before_processing():
    checks = check_processing_prerequisites(
        pdf_directory="./data/raw",
        db_directory="./chroma_db"
    )
    
    print(f"Prerequisite check status: {checks['status']}")
    if checks['recommendations']:
        print("Recommendations:")
        for rec in checks['recommendations']:
            print(f"  - {rec}")
    
    return checks['status'] in ['ready', 'ready_with_warnings']

# Monitor processing progress
async def processing_with_monitoring():
    config = IngestionConfig.create_default()
    config.log_progress_interval = 2  # Log every 2 documents
    
    pipeline = DocumentIngestionPipeline(config)
    
    def custom_progress_callback(completed, total, result):
        doc_name = result.get('document_name', 'unknown')
        status = result.get('status', 'unknown')
        print(f"[{completed}/{total}] {doc_name}: {status}")
        
        # Log summary every 5 documents
        if completed % 5 == 0:
            progress = pipeline.get_processing_progress()
            print(f"Progress: {progress['completion_percentage']:.1f}% complete")
    
    try:
        result = await pipeline.process_documents_robust(
            pdf_paths=list(Path("./data/raw").glob("*.pdf")),
            use_background_storage=True,
            progress_callback=custom_progress_callback
        )
        
        # Final summary
        pipeline.log_progress_summary()
        
    finally:
        pipeline.cleanup()
"""
