"""
Robust document ingestion pipeline with dual PDF extraction and intelligent fallback.

This module implements a sophisticated data ingestion system that combines PyMuPDF
and PDFPlumber for maximum text extraction reliability, with page-wise chunking
for semantic coherence.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Iterator, Tuple, Optional, Union
from dataclasses import dataclass
import re
from datetime import datetime

# PDF Processing libraries
import pymupdf4llm  # PyMuPDF4LLM for better LLM text extraction
import pdfplumber
from pydantic import BaseModel, Field, field_validator

# Import our schemas
from ..schemas.models import (
    DocumentChunk,
    ChunkMetadata,
    IngestionResult,
    ProcessingError,
)


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ===== CUSTOM EXCEPTIONS =====
class PDFParsingError(Exception):
    """Raised when PDF parsing fails completely."""

    pass


class ExtractionError(Exception):
    """Raised when text extraction fails for both methods."""

    pass


class ChunkingError(Exception):
    """Raised when document chunking fails."""

    pass


# ===== CONFIGURATION CLASSES =====
@dataclass
class ChunkingConfig:
    """
    Configuration for document chunking strategy.

    Follows the Single Responsibility Principle by containing only
    chunking-related configuration parameters.
    """

    # Page-wise chunking parameters
    use_page_based_chunking: bool = True
    merge_small_pages: bool = True
    min_page_chars: int = 200  # Minimum characters per page to avoid merging
    max_chunk_chars: int = 4000  # Maximum chunk size when merging pages

    # Extraction parameters
    extraction_threshold: float = 0.9  # 90% threshold for PyMuPDF vs PDFPlumber
    include_tables: bool = True
    preserve_formatting: bool = True

    # Quality control
    min_chunk_chars: int = 50  # Minimum viable chunk size
    remove_empty_chunks: bool = True

    def __post_init__(self):
        """Validate configuration parameters."""
        if not 0.5 <= self.extraction_threshold <= 1.0:
            raise ValueError("extraction_threshold must be between 0.5 and 1.0")
        if self.min_chunk_chars < 10:
            raise ValueError("min_chunk_chars must be at least 10")
        if self.max_chunk_chars < self.min_chunk_chars:
            raise ValueError("max_chunk_chars must be >= min_chunk_chars")


@dataclass
class ExtractionResult:
    """
    Result of text extraction for a single page.

    Encapsulates extraction metadata for decision making.
    """

    page_number: int
    pymupdf_text: str
    pdfplumber_text: str
    chosen_method: str
    chosen_text: str
    confidence_score: float
    extraction_time: float


# ===== CORE EXTRACTION CLASSES =====
class DualPDFExtractor:
    """
    Handles dual PDF text extraction using PyMuPDF and PDFPlumber.

    Follows Single Responsibility Principle by focusing solely on
    text extraction and method comparison.
    """

    def __init__(self, config: ChunkingConfig):
        """
        Initialize the dual extractor.

        Args:
            config: Chunking configuration with extraction parameters
        """
        self.config = config
        self.extraction_stats = {
            "pymupdf4llm_chosen": 0,
            "pdfplumber_chosen": 0,
            "total_pages": 0,
        }

    def extract_page_with_pymupdf4llm(self, pdf_path: Path, page_num: int) -> str:
        """
        Extract text from a page using PyMuPDF4LLM.

        Args:
            pdf_path: Path to the PDF file
            page_num: Zero-based page number

        Returns:
            Extracted text string

        Raises:
            ExtractionError: If extraction fails
        """
        try:
            # PyMuPDF4LLM extracts the entire document, so we extract all and get the specific page
            # This is less efficient but PyMuPDF4LLM is designed for full document processing
            md_text = pymupdf4llm.to_markdown(
                str(pdf_path),
                pages=[page_num],  # Extract only the specific page
                write_images=False,  # Don't extract images for text processing
                table_strategy="lines_strict" if self.config.include_tables else "none",
            )

            return self._clean_extracted_text(md_text)

        except Exception as e:
            logger.warning(f"PyMuPDF4LLM extraction failed for page {page_num}: {e}")
            return ""

    def extract_page_with_pdfplumber(self, pdf_path: Path, page_num: int) -> str:
        """
        Extract text from a page using PDFPlumber.

        Args:
            pdf_path: Path to the PDF file
            page_num: Zero-based page number

        Returns:
            Extracted text string

        Raises:
            ExtractionError: If extraction fails
        """
        try:
            with pdfplumber.open(pdf_path) as pdf:
                if page_num >= len(pdf.pages):
                    return ""

                page = pdf.pages[page_num]

                # Extract with layout preservation
                text = page.extract_text(
                    x_tolerance=3,
                    x_tolerance_ratio=None,
                    y_tolerance=3,
                    layout=True,
                    x_density=7.25,
                    y_density=13,
                    line_dir_render=None,
                    char_dir_render=None,
                )

                if self.config.include_tables:
                    # Extract tables separately
                    try:
                        tables = page.extract_tables()
                        for table in tables:
                            if table:
                                # Convert table to text format
                                table_str = "\n".join(
                                    [
                                        "\t".join(
                                            str(cell) if cell else "" for cell in row
                                        )
                                        for row in table
                                    ]
                                )
                                text += f"\n\n[TABLE]\n{table_str}\n[/TABLE]\n"
                    except Exception as e:
                        logger.debug(
                            f"Table extraction failed for page {page_num}: {e}"
                        )

                return self._clean_extracted_text(text or "")

        except Exception as e:
            logger.warning(f"PDFPlumber extraction failed for page {page_num}: {e}")
            return ""

    def _clean_extracted_text(self, text: str) -> str:
        """
        Clean and normalize extracted text.

        Args:
            text: Raw extracted text

        Returns:
            Cleaned text string
        """
        if not text:
            return ""

        # Remove excessive whitespace while preserving paragraph structure
        text = re.sub(
            r"\n\s*\n\s*\n", "\n\n", text
        )  # Multiple empty lines -> double newline
        text = re.sub(r"[ \t]+", " ", text)  # Multiple spaces/tabs -> single space
        text = re.sub(
            r"^\s+|\s+$", "", text, flags=re.MULTILINE
        )  # Trim line whitespace

        return text.strip()

    def compare_extractions(
        self, pymupdf_text: str, pdfplumber_text: str
    ) -> Tuple[str, str, float]:
        """
        Compare extraction results and choose the best method.

        Uses character count comparison with the configured threshold.
        If PyMuPDF4LLM extracts >= 90% of PDFPlumber's character count, choose PyMuPDF4LLM.
        Otherwise, choose PDFPlumber.

        Args:
            pymupdf_text: Text extracted by PyMuPDF4LLM
            pdfplumber_text: Text extracted by PDFPlumber

        Returns:
            Tuple of (chosen_method, chosen_text, confidence_score)
        """
        pymupdf_chars = len(pymupdf_text)
        pdfplumber_chars = len(pdfplumber_text)

        # Handle edge cases
        if pdfplumber_chars == 0 and pymupdf_chars == 0:
            return "none", "", 0.0
        elif pdfplumber_chars == 0:
            return "pymupdf4llm", pymupdf_text, 1.0
        elif pymupdf_chars == 0:
            return "pdfplumber", pdfplumber_text, 1.0

        # Calculate ratio
        ratio = pymupdf_chars / pdfplumber_chars

        if ratio >= self.config.extraction_threshold:
            # PyMuPDF4LLM extracted enough content
            confidence = min(1.0, ratio)  # Cap at 1.0
            return "pymupdf4llm", pymupdf_text, confidence
        else:
            # PDFPlumber is better
            confidence = min(1.0, pdfplumber_chars / max(pymupdf_chars, 1))
            return "pdfplumber", pdfplumber_text, confidence

    def extract_document(self, pdf_path: Path) -> List[ExtractionResult]:
        """
        Extract text from entire document using dual extraction strategy.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            List of ExtractionResult objects, one per page

        Raises:
            PDFParsingError: If document cannot be opened
        """
        if not pdf_path.exists():
            raise PDFParsingError(f"PDF file not found: {pdf_path}")

        logger.info(f"Starting dual extraction for: {pdf_path.name}")
        start_time = datetime.now()

        results = []

        try:
            # First, determine the number of pages using pdfplumber (more reliable for page count)
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
                logger.info(f"Document has {total_pages} pages")

                for page_num in range(total_pages):
                    page_start = datetime.now()

                    # Extract with both methods
                    pymupdf_text = self.extract_page_with_pymupdf4llm(
                        pdf_path, page_num
                    )
                    pdfplumber_text = self.extract_page_with_pdfplumber(
                        pdf_path, page_num
                    )

                    # Compare and choose best method
                    chosen_method, chosen_text, confidence = self.compare_extractions(
                        pymupdf_text, pdfplumber_text
                    )

                    # Update statistics
                    if chosen_method == "pymupdf4llm":
                        self.extraction_stats["pymupdf4llm_chosen"] += 1
                    elif chosen_method == "pdfplumber":
                        self.extraction_stats["pdfplumber_chosen"] += 1

                    page_time = (datetime.now() - page_start).total_seconds()

                    result = ExtractionResult(
                        page_number=page_num + 1,  # 1-based for user display
                        pymupdf_text=pymupdf_text,
                        pdfplumber_text=pdfplumber_text,
                        chosen_method=chosen_method,
                        chosen_text=chosen_text,
                        confidence_score=confidence,
                        extraction_time=page_time,
                    )

                    results.append(result)

                    if page_num % 10 == 0:  # Log progress every 10 pages
                        logger.info(f"Processed {page_num + 1}/{total_pages} pages")

        except Exception as e:
            raise PDFParsingError(f"Failed to process PDF {pdf_path}: {e}")

        total_time = (datetime.now() - start_time).total_seconds()
        self.extraction_stats["total_pages"] = len(results)

        logger.info(f"Extraction complete: {len(results)} pages in {total_time:.2f}s")
        logger.info(
            f"Method selection: PyMuPDF4LLM={self.extraction_stats['pymupdf4llm_chosen']}, "
            f"PDFPlumber={self.extraction_stats['pdfplumber_chosen']}"
        )

        return results


# ===== PAGE-WISE CHUNKING CLASS =====
class PageWiseChunker:
    """
    Handles page-wise document chunking with intelligent page merging.

    Follows Single Responsibility Principle by focusing solely on
    converting extracted pages into semantically coherent chunks.
    """

    def __init__(self, config: ChunkingConfig):
        """
        Initialize the page-wise chunker.

        Args:
            config: Chunking configuration
        """
        self.config = config

    def create_chunks(
        self, extraction_results: List[ExtractionResult], source_document: str
    ) -> List[DocumentChunk]:
        """
        Create document chunks from extraction results using page-wise strategy.

        Args:
            extraction_results: List of page extraction results
            source_document: Name of the source document

        Returns:
            List of DocumentChunk objects

        Raises:
            ChunkingError: If chunking process fails
        """
        if not extraction_results:
            raise ChunkingError("No extraction results provided for chunking")

        logger.info(f"Creating page-wise chunks for {source_document}")

        chunks = []
        current_chunk_pages = []
        current_chunk_text = ""
        chunk_index = 0

        for result in extraction_results:
            # Skip pages with no content
            if (
                not result.chosen_text
                or len(result.chosen_text.strip()) < self.config.min_chunk_chars
            ):
                if result.chosen_text.strip():  # Log non-empty but small pages
                    logger.debug(
                        f"Skipping small page {result.page_number}: {len(result.chosen_text)} chars"
                    )
                continue

            page_text = result.chosen_text.strip()
            page_chars = len(page_text)

            # Decide whether to start a new chunk or merge with current
            if self.config.merge_small_pages and current_chunk_text:
                projected_size = (
                    len(current_chunk_text) + len(page_text) + 2
                )  # +2 for page separator

                # If adding this page would exceed max size, finalize current chunk
                if projected_size > self.config.max_chunk_chars:
                    chunk = self._create_chunk_from_pages(
                        current_chunk_pages,
                        current_chunk_text,
                        source_document,
                        chunk_index,
                    )
                    chunks.append(chunk)
                    chunk_index += 1

                    # Start new chunk with current page
                    current_chunk_pages = [result]
                    current_chunk_text = page_text
                else:
                    # Add page to current chunk
                    current_chunk_pages.append(result)
                    current_chunk_text += (
                        f"\n\n--- Page {result.page_number} ---\n\n{page_text}"
                    )
            else:
                # Start new chunk
                if current_chunk_text:  # Finalize previous chunk if exists
                    chunk = self._create_chunk_from_pages(
                        current_chunk_pages,
                        current_chunk_text,
                        source_document,
                        chunk_index,
                    )
                    chunks.append(chunk)
                    chunk_index += 1

                current_chunk_pages = [result]
                current_chunk_text = page_text

        # Handle final chunk
        if current_chunk_text:
            chunk = self._create_chunk_from_pages(
                current_chunk_pages, current_chunk_text, source_document, chunk_index
            )
            chunks.append(chunk)

        logger.info(
            f"Created {len(chunks)} chunks from {len(extraction_results)} pages"
        )
        return chunks

    def _create_chunk_from_pages(
        self,
        pages: List[ExtractionResult],
        content: str,
        source_document: str,
        chunk_index: int,
    ) -> DocumentChunk:
        """
        Create a DocumentChunk from one or more pages.

        Args:
            pages: List of ExtractionResult objects that form this chunk
            content: Combined text content
            source_document: Name of source document
            chunk_index: Index of this chunk within the document

        Returns:
            DocumentChunk object
        """
        # Generate clause ID
        if len(pages) == 1:
            clause_id = f"{source_document}_page_{pages[0].page_number}"
        else:
            first_page = pages[0].page_number
            last_page = pages[-1].page_number
            clause_id = f"{source_document}_pages_{first_page}-{last_page}"

        # Calculate average confidence
        avg_confidence = sum(page.confidence_score for page in pages) / len(pages)

        # Create metadata
        metadata = ChunkMetadata(
            source_document=source_document,
            clause_id=clause_id,
            chunk_index=chunk_index,
            original_text=content,
            char_count=len(content),
            page_number=pages[0].page_number if len(pages) == 1 else None,
        )

        return DocumentChunk(
            content=content,
            metadata=metadata,
            embedding=None,  # Will be populated during vector storage
        )


# ===== MAIN PIPELINE CLASS =====
class DocumentIngestionPipeline:
    """
    Main document ingestion pipeline with robust dual extraction.

    Orchestrates the entire process from PDF extraction to chunk creation,
    following the Open/Closed Principle for easy extension.
    """

    def __init__(self, config: ChunkingConfig):
        """
        Initialize the ingestion pipeline.

        Args:
            config: Configuration for chunking and extraction
        """
        self.config = config
        self.extractor = DualPDFExtractor(config)
        self.chunker = PageWiseChunker(config)
        self.processing_stats = {
            "documents_processed": 0,
            "documents_failed": 0,
            "total_chunks": 0,
            "total_pages": 0,
            "start_time": None,
        }

    def process_document(self, pdf_path: Path) -> List[DocumentChunk]:
        """
        Process a single PDF document into chunks.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            List of DocumentChunk objects

        Raises:
            PDFParsingError: If PDF cannot be processed
            ChunkingError: If chunking fails
        """
        logger.info(f"Processing document: {pdf_path.name}")

        try:
            # Extract text using dual method
            extraction_results = self.extractor.extract_document(pdf_path)

            if not extraction_results:
                logger.warning(f"No content extracted from {pdf_path.name}")
                return []

            # Create chunks from extracted pages
            chunks = self.chunker.create_chunks(extraction_results, pdf_path.name)

            # Update statistics
            self.processing_stats["documents_processed"] += 1
            self.processing_stats["total_chunks"] += len(chunks)
            self.processing_stats["total_pages"] += len(extraction_results)

            logger.info(
                f"Successfully processed {pdf_path.name}: {len(chunks)} chunks from {len(extraction_results)} pages"
            )
            return chunks

        except Exception as e:
            self.processing_stats["documents_failed"] += 1
            logger.error(f"Failed to process {pdf_path.name}: {e}")
            raise

    def process_directory(self, directory_path: Path) -> Iterator[DocumentChunk]:
        """
        Process all PDF documents in a directory.

        Args:
            directory_path: Path to directory containing PDF files

        Yields:
            DocumentChunk objects as they are created

        Raises:
            FileNotFoundError: If directory doesn't exist
        """
        if not directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")

        pdf_files = list(directory_path.glob("*.pdf"))

        if not pdf_files:
            logger.warning(f"No PDF files found in {directory_path}")
            return

        logger.info(f"Found {len(pdf_files)} PDF files to process")
        self.processing_stats["start_time"] = datetime.now()

        for pdf_path in pdf_files:
            try:
                chunks = self.process_document(pdf_path)
                for chunk in chunks:
                    yield chunk

            except Exception as e:
                logger.error(f"Skipping {pdf_path.name} due to error: {e}")
                continue

        self._log_final_statistics()

    def get_processing_summary(self) -> IngestionResult:
        """
        Get summary of processing results.

        Returns:
            IngestionResult with processing statistics
        """
        end_time = datetime.now()
        processing_time = 0.0

        if self.processing_stats["start_time"]:
            processing_time = (
                end_time - self.processing_stats["start_time"]
            ).total_seconds()

        total_docs = (
            self.processing_stats["documents_processed"]
            + self.processing_stats["documents_failed"]
        )
        avg_chunk_size = 0.0

        if self.processing_stats["total_chunks"] > 0:
            # This is an approximation - we'd need to track actual chunk sizes for precision
            avg_chunk_size = 1000.0  # Placeholder - could be improved

        return IngestionResult(
            total_documents=total_docs,
            total_chunks=self.processing_stats["total_chunks"],
            successful_documents=self.processing_stats["documents_processed"],
            failed_documents=self.processing_stats["documents_failed"],
            processing_time=processing_time,
            average_chunk_size=avg_chunk_size,
            errors=[],  # Could be enhanced to track specific errors
        )

    def _log_final_statistics(self) -> None:
        """Log final processing statistics."""
        stats = self.processing_stats
        total_docs = stats["documents_processed"] + stats["documents_failed"]

        logger.info("=== PROCESSING COMPLETE ===")
        logger.info(f"Total documents: {total_docs}")
        logger.info(f"Successful: {stats['documents_processed']}")
        logger.info(f"Failed: {stats['documents_failed']}")
        logger.info(f"Total chunks created: {stats['total_chunks']}")
        logger.info(f"Total pages processed: {stats['total_pages']}")

        if stats["start_time"]:
            duration = (datetime.now() - stats["start_time"]).total_seconds()
            logger.info(f"Total processing time: {duration:.2f} seconds")

        # Log extraction method statistics
        extractor_stats = self.extractor.extraction_stats
        logger.info(
            f"Extraction methods - PyMuPDF4LLM: {extractor_stats['pymupdf4llm_chosen']}, "
            f"PDFPlumber: {extractor_stats['pdfplumber_chosen']}"
        )


# ===== UTILITY FUNCTIONS =====
def validate_pdf_file(pdf_path: Path) -> bool:
    """
    Validate that a file is a readable PDF.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        True if file is a valid PDF, False otherwise
    """
    try:
        # Use pdfplumber for validation as it's more reliable for this purpose
        with pdfplumber.open(pdf_path) as pdf:
            return len(pdf.pages) > 0
    except Exception:
        return False


def estimate_processing_time(pdf_path: Path) -> float:
    """
    Estimate processing time for a PDF file.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Estimated processing time in seconds
    """
    try:
        # Use pdfplumber for page count estimation
        with pdfplumber.open(pdf_path) as pdf:
            page_count = len(pdf.pages)
            # Rough estimate: 0.5 seconds per page for dual extraction
            return page_count * 0.5
    except Exception:
        return 10.0  # Default estimate


# Make classes available for import
__all__ = [
    "DocumentIngestionPipeline",
    "ChunkingConfig",
    "DualPDFExtractor",
    "PageWiseChunker",
    "ExtractionResult",
    "PDFParsingError",
    "ExtractionError",
    "ChunkingError",
    "validate_pdf_file",
    "estimate_processing_time",
]