"""Ingestion Engine

Implements a unified file ingestion pipeline with a parser registry and a main
`ingest_from_url` function. Keeps outward behaviour (returning List[str] of
raw text chunks) consistent with the existing retrieval pipeline.

Features:
 - Download remote file (or accept local path) to temp storage
 - Detect extension and route to appropriate parser via registry
 - Handle ZIP archives recursively
 - Parse PDFs with fast PyMuPDF pathway
 - Use unstructured's partitioning directly (if installed) for generic documents
     / spreadsheets / images (best‑effort OCR)
 - Graceful fallbacks & detailed logging
"""
from __future__ import annotations

import os
import uuid
import time
import zipfile
import tempfile
import requests
from typing import Callable, Dict, List, Optional
from urllib.parse import urlparse, unquote

from fastapi import HTTPException

from components.utils.logger import log_service_event, log_error

# Optional direct import of unstructured partition function; engine degrades gracefully if absent.
try:  # pragma: no cover
    from unstructured.partition.auto import partition as _unstructured_partition  # type: ignore
    _UNSTRUCTURED_AVAILABLE = True
except Exception:  # pragma: no cover
    _unstructured_partition = None  # type: ignore
    _UNSTRUCTURED_AVAILABLE = False

log_service_event(
    "unstructured_availability",
    f"unstructured partition module {'available' if _UNSTRUCTURED_AVAILABLE else 'unavailable'}",
    {"available": _UNSTRUCTURED_AVAILABLE}
)

# ------------------------------ Parser Functions ------------------------------


def parse_standard_document(file_path: str, source_url: str) -> List[str]:
    """Parse a standard text / office document into list[str].

    Uses unstructured.process_document to obtain chunk groups; flattens each
    chunk (list of element dicts) into a single string joined by double newlines.
    Falls back to naive read for .txt if unstructured unavailable.
    """
    start = time.time()
    ext = os.path.splitext(file_path.lower())[1]
    chunks: List[str] = []
    if not _UNSTRUCTURED_AVAILABLE:
        if ext == ".txt":  # basic text fallback
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
            chunks = [seg.strip() for seg in text.split('\n\n') if seg.strip()]
            log_service_event("plain_text_fallback", "Processed TXT without unstructured", {
                "file_path": file_path,
                "chunks": len(chunks)
            })
        else:
            log_service_event("unstructured_unavailable_skip", "Skipping non-txt file (unstructured missing)", {
                "file_path": file_path,
                "ext": ext
            })
            return []
    else:
        try:
            # Lightweight internal grouping to mimic previous behaviour
            elements = _unstructured_partition(
                filename=file_path, languages=["eng"]) or []
            if not elements:
                try:
                    elements = _unstructured_partition(
                        filename=file_path, languages=None) or []
                except Exception:
                    pass
            groups = []
            current = []
            current_words = 0
            for el in elements:
                text = str(el).strip()
                if not text:
                    continue
                words = text.split()
                if current_words + len(words) > 500 and current:
                    groups.append(current)
                    current = []
                    current_words = 0
                # Capture dict form if available
                try:
                    d = el.to_dict()  # type: ignore[attr-defined]
                    if 'text' not in d:
                        d['text'] = text
                except Exception:
                    d = {'text': text}
                current.append(d)
                current_words += len(words)
            if current:
                groups.append(current)
            for group in groups:
                texts = []
                for el in group:
                    txt = el.get('text') if isinstance(el, dict) else None
                    if not txt:
                        # element objects handled by str(el)
                        try:
                            txt = el.get('text')  # type: ignore
                        except Exception:
                            txt = str(el)
                    if txt and txt.strip():
                        texts.append(txt.strip())
                if texts:
                    chunks.append("\n\n".join(texts))
        except Exception as e:  # pragma: no cover
            log_error("standard_doc_parse_failed", {
                "file_path": file_path,
                "error": str(e)
            })
            raise HTTPException(
                status_code=422, detail=f"Failed to parse document: {e}")

    log_service_event("standard_doc_parsed", "Parsed standard document", {
        "file_path": file_path,
        "source_url": source_url,
        "ext": ext,
        "chunks": len(chunks),
        "time_seconds": time.time() - start
    })
    return chunks


def parse_spreadsheet(file_path: str, source_url: str) -> List[str]:
    """Parse spreadsheet (.xlsx) into list[str] using pandas.

    Each sheet will be processed separately, and columns/rows will be formatted
    with appropriate headers.
    """
    try:
        import pandas as pd
    except ImportError:
        log_error("pandas_import_failed", {
                  "file_path": file_path, "error": "pandas not installed"})
        # Fall back to standard parser
        return parse_standard_document(file_path, source_url)

    start_time = time.time()
    chunks = []

    try:
        # Read all sheets from the Excel file
        xl = pd.ExcelFile(file_path)
        sheet_names = xl.sheet_names

        for sheet_name in sheet_names:
            # Read each sheet into a dataframe
            df = pd.read_excel(file_path, sheet_name=sheet_name)

            # Skip empty sheets
            if df.empty:
                continue

            # Format the sheet content
            sheet_header = f"Sheet: {sheet_name}"

            # Convert DataFrame to a formatted string with headers and rows
            table_str = df.to_string(index=False)

            # Add to chunks with appropriate headers
            chunks.append(f"{sheet_header}\n\n{table_str}")

            # If the sheet is large, create additional chunks with specific column groups
            # to improve search relevance
            if df.shape[1] > 5:  # More than 5 columns
                # Process column groups to create additional chunks for better searchability
                for i in range(0, len(df.columns), 3):
                    col_group = df.iloc[:, i:i+3]
                    if not col_group.empty:
                        group_str = col_group.to_string(index=False)
                        chunks.append(
                            f"{sheet_name} (columns {i+1}-{i+3}):\n\n{group_str}")

    except Exception as e:
        log_error("excel_pandas_parse_failed", {
                  "file_path": file_path, "error": str(e)})
        # Fall back to standard parser
        return parse_standard_document(file_path, source_url)

    # Filter out very short chunks
    chunks = [chunk for chunk in chunks if len(chunk) > 50]

    parsing_time = time.time() - start_time
    log_service_event("excel_pandas_parsed", "Parsed Excel with pandas", {
        "file_path": file_path,
        "source_url": source_url,
        "sheets_found": len(sheet_names),
        "chunks": len(chunks),
        "time_seconds": parsing_time
    })

    return chunks


def parse_image_with_ocr(file_path: str, source_url: str) -> List[str]:
    """Parse image using unstructured (which may invoke OCR if deps present).

    If OCR backend not available, returns empty list (logged) instead of failing.
    """
    try:
        return parse_standard_document(file_path, source_url)
    except HTTPException as e:
        log_service_event("image_ocr_unavailable", "Image OCR unavailable or failed", {
            "file_path": file_path,
            "source_url": source_url,
            "detail": str(e)
        })
        return []


def parse_pdf_file(file_path: str, source_url: str) -> List[str]:
    """Parse PDF using PyMuPDF directly (inlined) and return chunk list.

    Mirrors previous parse_pdf_with_pymupdf behaviour to keep identical
    splitting heuristic (paragraphs / long paragraph sentence splitting).
    """
    import fitz  # PyMuPDF
    start_time = time.time()
    document_id = str(uuid.uuid4())
    filename = os.path.basename(file_path)

    log_service_event("primary_parsing_start", "Starting primary PDF parsing with PyMuPDF", {
        "document_id": document_id,
        "filename": filename,
        "parsing_method": "PyMuPDF"
    })

    try:
        doc = fitz.open(file_path)
        page_texts: List[str] = []
        for pno in range(len(doc)):
            page = doc.load_page(pno)
            txt = page.get_text("text")
            if txt.strip():
                page_texts.append(txt)
        doc.close()

        if not page_texts:
            log_error("pdf_extraction_empty", {
                      "document_id": document_id, "filename": filename})
            return []

        chunks: List[str] = []
        for page_text in page_texts:
            paragraphs = page_text.split("\n\n")
            for para in paragraphs:
                if not para.strip():
                    continue
                if len(para) > 1000:
                    sentences = para.split(". ")
                    current = ""
                    for s in sentences:
                        if len(current) + len(s) < 1000:
                            current += s + ". "
                        else:
                            if current:
                                chunks.append(current.strip())
                            current = s + ". "
                    if current:
                        chunks.append(current.strip())
                else:
                    chunks.append(para.strip())

        # filter short
        chunks = [c for c in chunks if len(c.strip()) > 50]
        parsing_time = time.time() - start_time
        log_service_event("primary_parsing_complete", "Successfully parsed PDF with PyMuPDF", {
            "document_id": document_id,
            "filename": filename,
            "chunks_extracted": len(chunks),
            "total_chars": sum(len(c) for c in chunks),
            "avg_chunk_length": (sum(len(c) for c in chunks) / len(chunks)) if chunks else 0,
            "parsing_time_seconds": parsing_time
        })
        return chunks
    except Exception as e:  # pragma: no cover
        log_error("primary_parsing_failed", {
                  "document_id": document_id, "filename": filename, "error": str(e)})
        return []


# --- PPTX Parser ---
def parse_pptx_file(file_path: str, source_url: str) -> List[str]:
    """Parse PPTX file and return list of text chunks (one per slide)."""
    try:
        from pptx import Presentation
    except ImportError:
        log_error("pptx_import_failed", {
                  "file_path": file_path, "error": "python-pptx not installed"})
        return []
    chunks = []
    try:
        prs = Presentation(file_path)
        for slide in prs.slides:
            slide_text = []
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    txt = shape.text.strip()
                    if txt:
                        slide_text.append(txt)
            if slide_text:
                chunks.append("\n".join(slide_text))
    except Exception as e:
        log_error("pptx_parse_failed", {
                  "file_path": file_path, "error": str(e)})
        return []
    log_service_event("pptx_parsed", "Parsed PPTX file", {
        "file_path": file_path,
        "source_url": source_url,
        "chunks": len(chunks)
    })
    return chunks


# ------------------------------ Parser Registry ------------------------------
Parser_Registry: Dict[str, Callable[[str, str], List[str]]] = {
    "xlsx": parse_spreadsheet,
    "jpg": parse_image_with_ocr,
    "jpeg": parse_image_with_ocr,
    "png": parse_image_with_ocr,
    "pdf": parse_pdf_file,
    "docx": parse_standard_document,
    "txt": parse_standard_document,
    "pptx": parse_pptx_file,
}

# ------------------------------ Helper Functions ------------------------------


def _is_url(path_or_url: str) -> bool:
    parsed = urlparse(path_or_url)
    return bool(parsed.scheme and parsed.netloc)


def _download_to_temp(url: str) -> str:
    dl_id = str(uuid.uuid4())
    log_service_event("download_start", "Starting file download", {
        "url": url,
        "download_id": dl_id
    })
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
    except Exception as e:
        log_error("download_failed", {"url": url, "error": str(e)})
        raise HTTPException(
            status_code=400, detail=f"Failed to download file: {e}")

    name = os.path.basename(unquote(urlparse(url).path)) or f"file_{dl_id}"
    suffix = os.path.splitext(name)[1] or ''
    tmp_fd, tmp_path = tempfile.mkstemp(prefix="ingest_", suffix=suffix)
    with os.fdopen(tmp_fd, 'wb') as f:
        f.write(resp.content)

    log_service_event("download_complete", "File downloaded", {
        "url": url,
        "path": tmp_path,
        "bytes": len(resp.content)
    })
    return tmp_path


def handle_zip_file(zip_path: str, source_url: str) -> List[str]:
    zip_id = str(uuid.uuid4())
    start = time.time()
    log_service_event("zip_extraction_start", "Extracting ZIP", {
        "zip_id": zip_id,
        "zip_path": zip_path
    })
    combined: List[str] = []
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(temp_dir)
                members = zf.namelist()
            log_service_event("zip_extracted", "ZIP extracted", {
                "zip_id": zip_id,
                "file_count": len(members),
                "files_sample": members[:10]
            })
        except zipfile.BadZipFile as e:
            log_error("invalid_zip", {"zip_path": zip_path, "error": str(e)})
            raise HTTPException(
                status_code=422, detail=f"Invalid ZIP file: {e}")

        for root, _, files in os.walk(temp_dir):
            for fname in files:
                if fname.startswith("._"):
                    continue
                fpath = os.path.join(root, fname)
                try:
                    chunks = ingest_from_url(
                        fpath, _is_recursive=True, _source_url=source_url)
                    combined.extend(chunks)
                except Exception as e:
                    log_error("zip_member_ingest_failed", {
                        "zip_id": zip_id,
                        "member": fname,
                        "error": str(e)
                    })
        log_service_event("zip_processing_complete", "ZIP processed", {
            "zip_id": zip_id,
            "total_chunks": len(combined),
            "elapsed": time.time() - start
        })
    return combined

# ------------------------------ Main Engine ------------------------------


def ingest_from_url(url_or_path: str, *, _is_recursive: bool = False, _source_url: Optional[str] = None) -> List[str]:
    """Ingest a document (remote URL or local path) and return list[str] chunks.

    Implements required steps:
      1. Download file if remote
      2. Detect extension
      3. If ZIP -> recurse into contents
      4. Route to parser via registry
      5. Execute parser & collect chunks
    """
    original_input = url_or_path
    start = time.time()
    temp_path = None
    is_local = os.path.exists(url_or_path)
    source_url = _source_url or (url_or_path if _is_url(url_or_path) else None)

    try:
        if not is_local:
            temp_path = _download_to_temp(url_or_path)
        else:
            temp_path = url_or_path

        ext = os.path.splitext(temp_path)[1].lower().lstrip('.')
        log_service_event("file_type_detected", "Detected file type", {
            "input": original_input,
            "local_path": temp_path,
            "extension": ext or "(none)"
        })

        if ext == 'zip':
            chunks = handle_zip_file(temp_path, source_url or original_input)
            return chunks

        parser_func = Parser_Registry.get(ext)
        if not parser_func:
            log_service_event("unsupported_file_type", "Unsupported file type encountered", {
                "extension": ext,
                "path": temp_path
            })
            return []

        chunks = parser_func(temp_path, source_url or original_input)
        log_service_event("ingest_complete", "Completed ingestion", {
            "input": original_input,
            "chunks": len(chunks),
            "elapsed": time.time() - start
        })
        return chunks
    finally:
        if temp_path and os.path.exists(temp_path) and not is_local and not _is_recursive:
            try:
                os.unlink(temp_path)
            except OSError:
                pass


__all__ = ["ingest_from_url", "Parser_Registry", "parse_pdf_file"]
