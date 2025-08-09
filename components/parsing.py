"""
Parsing Module

This module contains functions for parsing PDF documents.
"""
import os
import time
import uuid
import tempfile
from typing import List
from datetime import datetime
from fastapi import HTTPException

# Import local modules
from logger import logger, log_service_event, log_error


def parse_pdf_with_pymupdf(pdf_content: bytes, document_id: str, filename: str) -> List[str]:
    """
    Primary function to parse PDF using PyMuPDF for text extraction.

    This function is the preferred method for PDF parsing due to its superior
    performance and accuracy compared to API-based parsing services.

    Parameters:
        pdf_content (bytes): The PDF file content as bytes
        document_id (str): Unique identifier for tracking this document
        filename (str): Name of the PDF file for logging

    Returns:
        List[str]: List of text chunks extracted from the PDF

    Raises:
        HTTPException: If PyMuPDF parsing fails
    """
    parsing_start_time = time.time()

    print(
        f"� Parsing PDF with PyMuPDF for '{filename}'...")
    log_service_event("primary_parsing_start", "Starting primary PDF parsing with PyMuPDF", {
        "document_id": document_id,
        "filename": filename,
        "pdf_size_bytes": len(pdf_content),
        "parsing_method": "PyMuPDF"
    })

    try:
        # Create a temporary file to write the PDF content
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
            temp_file.write(pdf_content)
            temp_path = temp_file.name

        try:
            # Import fitz from PyMuPDF
            import fitz

            # Open the PDF document
            doc = fitz.open(temp_path)

            # Extract text from each page
            page_texts = []
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                page_text = page.get_text("text")
                if page_text.strip():
                    page_texts.append(page_text)

            # Close the document
            doc.close()

        finally:
            # Clean up the temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)

        if not page_texts:
            error_msg = "No text could be extracted from the PDF."
            log_error("pdf_extraction_empty", {
                "document_id": document_id,
                "filename": filename,
                "parsing_method": "PyMuPDF"
            })
            raise HTTPException(status_code=422, detail=error_msg)

        # Split page texts into smaller chunks based on paragraphs and sentences
        chunk_texts = []
        for page_num, page_text in enumerate(page_texts):
            # Split by paragraphs (double newlines)
            paragraphs = page_text.split("\n\n")

            # Process each paragraph
            for para in paragraphs:
                if not para.strip():
                    continue

                # If paragraph is very long, split it further
                if len(para) > 1000:
                    # Split by sentences or other natural breaks
                    sentences = para.split(". ")
                    current_chunk = ""

                    for sentence in sentences:
                        if len(current_chunk) + len(sentence) < 1000:
                            current_chunk += sentence + ". "
                        else:
                            if current_chunk:
                                chunk_texts.append(current_chunk.strip())
                            current_chunk = sentence + ". "

                    if current_chunk:
                        chunk_texts.append(current_chunk.strip())
                else:
                    chunk_texts.append(para.strip())

        # Filter out very short chunks
        chunk_texts = [
            chunk for chunk in chunk_texts if len(chunk.strip()) > 50]

        parsing_time = time.time() - parsing_start_time

        # Log successful parsing
        log_service_event("primary_parsing_complete", "Successfully parsed PDF with PyMuPDF", {
            "document_id": document_id,
            "filename": filename,
            "chunks_extracted": len(chunk_texts),
            "total_chars": sum(len(chunk) for chunk in chunk_texts),
            "avg_chunk_length": sum(len(chunk) for chunk in chunk_texts) / len(chunk_texts) if chunk_texts else 0,
            "parsing_time_seconds": parsing_time,
            "chunk_size_distribution": {
                "short_chunks": sum(1 for chunk in chunk_texts if len(chunk) < 200),
                "medium_chunks": sum(1 for chunk in chunk_texts if 200 <= len(chunk) < 500),
                "long_chunks": sum(1 for chunk in chunk_texts if len(chunk) >= 500)
            }
        })

        print(
            f"✅ PyMuPDF extracted {len(chunk_texts)} text chunks from '{filename}'")
        return chunk_texts

    except Exception as e:
        error_msg = f"PyMuPDF parsing failed: {e}"
        log_error("primary_parsing_failed", {
            "document_id": document_id,
            "filename": filename,
            "error": str(e),
            "parsing_time_seconds": time.time() - parsing_start_time
        })
        raise HTTPException(status_code=503, detail=error_msg)
