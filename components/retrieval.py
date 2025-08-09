"""
Retrieval Function Module

This module contains functions related to retrieving documents and parsing them.
"""
import os
import time
import uuid
import tempfile
import requests
import zipfile
from typing import List
from datetime import datetime
from urllib.parse import urlparse, unquote
from fastapi import HTTPException

# Import local modules
from logger import logger, log_service_event, log_error


async def fetch_and_parse_pdf(api_url: str, file_url: str) -> List[str]:
    """
    Downloads and processes document files for text extraction.

    Supported formats:
    - PDF (.pdf) - processed with PyMuPDF as primary, extract_unstructured as fallback
    - Word Documents (.docx, .doc) - processed with extract_unstructured
    - Text Files (.txt) - processed with extract_unstructured
    - Markdown Files (.md) - processed with extract_unstructured
    - Rich Text Format (.rtf) - processed with extract_unstructured
    - ZIP archives containing any of the above formats

    For PDF files, PyMuPDF is used as the primary parser for better performance and accuracy.
    If PyMuPDF fails, extract_unstructured is used as a fallback.
    For non-PDF files, extract_unstructured is used directly.
    If the file is a ZIP archive, extracts it and processes all supported documents inside.

    Parameters:
        api_url (str): URL of the parsing API (legacy parameter, no longer used).
        file_url (str): URL of the file to download.

    Returns:
        List[str]: List of text chunks extracted from the document(s).

    Raises:
        HTTPException: If download fails, parsing fails, or no text chunks are found.
    """
    document_id = str(uuid.uuid4())
    start_time = time.time()

    print(f"\n⬇️  Downloading file from: {file_url}")
    log_service_event("download_start", f"Downloading document", {
        "url": file_url,
        "document_id": document_id,
        "timestamp": datetime.now().isoformat()
    })

    try:
        download_start = time.time()
        file_response = requests.get(file_url)
        file_response.raise_for_status()
        download_time = time.time() - download_start

        # Log download completion with detailed metrics
        log_service_event("download_complete", f"Document downloaded successfully", {
            "document_id": document_id,
            "size_bytes": len(file_response.content),
            "download_time_seconds": download_time,
            "content_type": file_response.headers.get('Content-Type'),
            "status_code": file_response.status_code
        })
    except requests.RequestException as e:
        error_msg = f"Failed to download document from URL: {e}"
        log_error("document_download_failed", {
            "document_id": document_id,
            "url": file_url,
            "error": str(e),
            "elapsed_time": time.time() - start_time
        })
        raise HTTPException(status_code=400, detail=error_msg)

    filename = os.path.basename(unquote(urlparse(file_url).path))
    file_extension = os.path.splitext(filename.lower())[1]

    # Check if the downloaded file is a ZIP
    if file_extension == '.zip':
        print(f"📦 Detected ZIP file: '{filename}'")
        log_service_event("zip_file_detected", "ZIP file detected, will extract and process", {
            "document_id": document_id,
            "filename": filename,
            "file_size_bytes": len(file_response.content)
        })

        # Process the ZIP file
        return await extract_and_process_zip(api_url, file_response.content, filename)

    # Determine content type based on file extension
    content_type_map = {
        '.pdf': 'application/pdf',
        '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        '.doc': 'application/msword',
        '.txt': 'text/plain',
        '.md': 'text/markdown',
        '.rtf': 'application/rtf'
    }

    # Get appropriate content type or default to PDF
    content_type = content_type_map.get(file_extension, 'application/pdf')

    # Log the detected file type
    if file_extension in content_type_map:
        print(f"📄 Detected {file_extension.upper()} file: '{filename}'")
        log_service_event("supported_file_type_detected", f"Detected supported file type: {file_extension}", {
            "document_id": document_id,
            "filename": filename,
            "extension": file_extension,
            "content_type": content_type
        })
    else:
        print(
            f"⚠️ Unknown file extension '{file_extension}', treating as PDF: '{filename}'")
        log_service_event("unknown_file_extension", "Unknown file extension, treating as PDF", {
            "document_id": document_id,
            "filename": filename,
            "extension": file_extension
        })

    if file_extension == ".pdf":
        try:
            from components.parsing import parse_pdf_with_pymupdf
            return parse_pdf_with_pymupdf(file_response.content, document_id, filename)
        except Exception as e:
            error_msg = f"PDF parsing failed with both methods: {e}"
            log_error("pdf_parsing_failed", {
                "document_id": document_id,
                "filename": filename,
                "error": str(e)
            })
            raise HTTPException(status_code=422, detail=error_msg)
    else:
        # For non-PDF files, use extract_unstructured directly
        try:
            from extract_unstructured import extract_text_from_file
            return extract_text_from_file(file_response.content, content_type, document_id, filename)
        except Exception as e:
            error_msg = f"Failed to extract text from file: {e}"
            log_error("non_pdf_extraction_failed", {
                "document_id": document_id,
                "filename": filename,
                "extension": file_extension,
                "error": str(e)
            })
            raise HTTPException(status_code=422, detail=error_msg)


async def extract_and_process_zip(api_url: str, zip_content: bytes, original_filename: str) -> List[str]:
    """
    Extracts a ZIP file and processes any supported document files found inside.

    Supported formats include: PDF, DOCX, DOC, TXT, MD, RTF

    For PDF files within the ZIP:
    - Primary: PyMuPDF for better performance and accuracy
    - Fallback: extract_unstructured if PyMuPDF fails

    For non-PDF files within the ZIP:
    - Uses extract_unstructured directly

    Parameters:
        api_url (str): URL of the parsing API (legacy parameter, no longer used).
        zip_content (bytes): The content of the ZIP file.
        original_filename (str): The original filename of the ZIP file.

    Returns:
        List[str]: Combined list of text chunks from all documents in the ZIP.

    Raises:
        HTTPException: If ZIP extraction fails or no supported documents are found.
    """
    zip_id = str(uuid.uuid4())
    combined_chunks = []

    # Create a temporary directory for extraction
    with tempfile.TemporaryDirectory() as temp_dir:
        zip_path = os.path.join(temp_dir, "downloaded.zip")

        # Log zip extraction start
        log_service_event("zip_extraction_start", "Starting ZIP file extraction", {
            "zip_id": zip_id,
            "original_filename": original_filename,
            "content_size_bytes": len(zip_content),
            "temp_dir": temp_dir
        })

        # Write ZIP content to temporary file
        with open(zip_path, 'wb') as f:
            f.write(zip_content)

        try:
            # Extract the ZIP file
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
                file_list = zip_ref.namelist()
                log_service_event("zip_extracted", "ZIP file extracted", {
                    "zip_id": zip_id,
                    "file_count": len(file_list),
                    "files": file_list[:10] + (["..."] if len(file_list) > 10 else [])
                })

            # Process each file in the ZIP
            doc_files = []
            supported_extensions = ['.pdf', '.docx',
                                    '.doc', '.txt', '.md', '.rtf']

            for root, _, files in os.walk(temp_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    file_extension = os.path.splitext(file.lower())[1]

                    # Skip the original ZIP file and non-supported file types
                    if file_path == zip_path or file_extension not in supported_extensions:
                        continue

                    doc_files.append(file_path)

            # Process each document
            for doc_file in doc_files:
                file_extension = os.path.splitext(doc_file.lower())[1]
                doc_id = str(uuid.uuid4())
                doc_filename = os.path.basename(doc_file)

                try:
                    # Read file content
                    with open(doc_file, 'rb') as f:
                        file_content = f.read()

                    # Process based on file type
                    if file_extension == '.pdf':
                        from components.parsing import parse_pdf_with_pymupdf
                        try:
                            chunks = parse_pdf_with_pymupdf(
                                file_content, doc_id, doc_filename)
                            combined_chunks.extend(chunks)
                        except Exception as e:
                            # Fallback to extract_unstructured for PDF
                            log_service_event("pdf_parse_fallback", f"Falling back to extract_unstructured for PDF", {
                                "zip_id": zip_id,
                                "doc_id": doc_id,
                                "filename": doc_filename,
                                "error": str(e)
                            })
                            from extract_unstructured import extract_text_from_file
                            chunks = extract_text_from_file(
                                file_content, 'application/pdf', doc_id, doc_filename)
                            combined_chunks.extend(chunks)
                    else:
                        # For non-PDF files, use extract_unstructured
                        content_type_map = {
                            '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                            '.doc': 'application/msword',
                            '.txt': 'text/plain',
                            '.md': 'text/markdown',
                            '.rtf': 'application/rtf'
                        }
                        content_type = content_type_map.get(
                            file_extension, 'application/octet-stream')

                        from extract_unstructured import extract_text_from_file
                        chunks = extract_text_from_file(
                            file_content, content_type, doc_id, doc_filename)
                        combined_chunks.extend(chunks)

                    log_service_event("zip_file_processed", f"Processed file from ZIP", {
                        "zip_id": zip_id,
                        "doc_id": doc_id,
                        "filename": doc_filename,
                        "extension": file_extension,
                        "chunks_extracted": len(chunks),
                        "total_chars": sum(len(chunk) for chunk in chunks)
                    })

                except Exception as e:
                    log_error("zip_file_processing_error", {
                        "zip_id": zip_id,
                        "doc_id": doc_id,
                        "filename": doc_filename,
                        "extension": file_extension,
                        "error": str(e)
                    })
                    print(f"⚠️ Error processing {doc_filename} from ZIP: {e}")

        except zipfile.BadZipFile as e:
            error_msg = f"Invalid ZIP file: {e}"
            log_error("invalid_zip_file", {
                "zip_id": zip_id,
                "original_filename": original_filename,
                "error": str(e)
            })
            raise HTTPException(status_code=422, detail=error_msg)

    # Check if we got any chunks
    if not combined_chunks:
        error_msg = "No text could be extracted from the documents in the ZIP file."
        log_error("no_text_from_zip_documents", {
            "zip_id": zip_id,
            "original_filename": original_filename
        })
        raise HTTPException(status_code=422, detail=error_msg)

    log_service_event("zip_processing_complete", "Completed processing of ZIP archive", {
        "zip_id": zip_id,
        "total_chunks": len(combined_chunks),
        "documents_processed": len(doc_files)
    })

    return combined_chunks
