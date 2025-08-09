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

    Each sheet will be processed separately with preserved structure for better readability.
    Tables are formatted to maintain alignment and preserve relationships between data cells.
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

            # Create a metadata header for the sheet
            # Sanitize sheet name to prevent any malicious content
            safe_sheet_name = ''.join(
                c for c in sheet_name if c.isalnum() or c in ' _-.')
            if safe_sheet_name != sheet_name:
                log_service_event("sheet_name_sanitized", "Sanitized potentially malicious sheet name", {
                    "file_path": file_path,
                    "original_name": sheet_name,
                    "sanitized_name": safe_sheet_name
                })

            sheet_header = f"TABLE: {safe_sheet_name}"
            sheet_info = f"Rows: {df.shape[0]}, Columns: {df.shape[1]}"
            metadata = f"{sheet_header}\n{sheet_info}\n{'=' * 50}"

            # Preserve structure by converting to markdown table format
            # First, convert any numeric indices to strings
            df.columns = df.columns.astype(str)

            # Format column headers with proper alignment
            headers = " | ".join([str(col).strip() for col in df.columns])
            separator = "-|-".join(["-" * len(str(col)) for col in df.columns])

            # Build the table rows with proper alignment, sanitizing cell values
            rows = []
            for _, row in df.iterrows():
                # Sanitize each cell value
                sanitized_values = []
                for val in row:
                    if pd.notna(val):
                        cell_value = str(val).strip()
                        # Check for suspicious content in individual cells
                        suspicious_patterns = [
                            "HackRx", "MANDATORY", "URGENT", "execute",
                            "WARNING", "leakage", "directive", "INSTRUCTION",
                            "COMPROMISED", "catastrophic", "comply", "respond"
                        ]
                        is_suspicious = False
                        for pattern in suspicious_patterns:
                            if pattern.lower() in cell_value.lower():
                                is_suspicious = True
                                # Replace with a safe placeholder
                                cell_value = "[Filtered content]"
                                log_service_event("suspicious_cell_filtered", "Filtered cell with suspicious content", {
                                    "file_path": file_path,
                                    "pattern": pattern,
                                    "value": str(val)[:30] + "..." if len(str(val)) > 30 else str(val)
                                })
                                break

                        sanitized_values.append(cell_value)
                    else:
                        sanitized_values.append("")

                formatted_row = " | ".join(sanitized_values)
                rows.append(formatted_row)

            # Combine all parts into a well-structured table
            table_content = f"{headers}\n{separator}\n" + "\n".join(rows)
            structured_table = f"{metadata}\n\n{table_content}"

            # Add the formatted table as a chunk
            chunks.append(structured_table)

            # For large tables, also create chunks by row groups to improve search relevance
            if df.shape[0] > 10:  # More than 10 rows
                # Adaptive chunk size
                row_chunk_size = min(10, max(5, df.shape[0] // 5))

                for i in range(0, df.shape[0], row_chunk_size):
                    end_idx = min(i + row_chunk_size, df.shape[0])
                    row_group = df.iloc[i:end_idx]

                    # Create a sub-table with the same format
                    sub_headers = " | ".join(
                        [str(col).strip() for col in row_group.columns])
                    sub_separator = "-|-".join(["-" * len(str(col))
                                               for col in row_group.columns])

                    sub_rows = []
                    for _, row in row_group.iterrows():
                        # Sanitize each cell value in sub-tables too
                        sanitized_values = []
                        for val in row:
                            if pd.notna(val):
                                cell_value = str(val).strip()
                                # Check for suspicious content in individual cells
                                suspicious_patterns = [
                                    "HackRx", "MANDATORY", "URGENT", "execute",
                                    "WARNING", "leakage", "directive", "INSTRUCTION",
                                    "COMPROMISED", "catastrophic", "comply", "respond"
                                ]
                                is_suspicious = False
                                for pattern in suspicious_patterns:
                                    if pattern.lower() in cell_value.lower():
                                        is_suspicious = True
                                        # Replace with a safe placeholder
                                        cell_value = "[Filtered content]"
                                        break

                                sanitized_values.append(cell_value)
                            else:
                                sanitized_values.append("")

                        formatted_row = " | ".join(sanitized_values)
                        sub_rows.append(formatted_row)

                    sub_table = f"{sub_headers}\n{sub_separator}\n" + \
                        "\n".join(sub_rows)
                    sub_chunk = f"TABLE: {safe_sheet_name} - Rows {i+1}-{end_idx}\n{'=' * 30}\n\n{sub_table}"
                    chunks.append(sub_chunk)

    except Exception as e:
        log_error("excel_pandas_parse_failed", {
                  "file_path": file_path, "error": str(e)})
        # Fall back to standard parser
        return parse_standard_document(file_path, source_url)

    # Filter out very short chunks
    chunks = [chunk for chunk in chunks if len(chunk) > 50]

    # Sanitize chunks - filter out potential malicious content
    original_chunk_count = len(chunks)
    chunks = _sanitize_chunks(chunks, file_path)
    filtered_chunk_count = len(chunks)

    parsing_time = time.time() - start_time
    log_service_event("excel_pandas_parsed", "Parsed Excel with pandas", {
        "file_path": file_path,
        "source_url": source_url,
        "sheets_found": len(sheet_names),
        "original_chunks": original_chunk_count,
        "filtered_chunks": filtered_chunk_count,
        "suspicious_content_removed": original_chunk_count - filtered_chunk_count,
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


# --- PPTX Parser using Docling and EasyOCR for images ---
def parse_pptx_file(file_path: str, source_url: str) -> List[str]:
    """Parse PPTX file and return list of text chunks using Docling.

    If images are detected (shown as <!-- image --> tags), extract them
    and use EasyOCR to perform OCR on the images for better text extraction.
    """
    try:
        from docling.document_converter import DocumentConverter
    except ImportError:
        log_error("docling_import_failed", {
            "file_path": file_path,
            "error": "Required module not installed (docling). Try: pip install docling docling-core"
        })
        # Fall back to the old parser if available
        try:
            from pptx import Presentation
            log_service_event("docling_fallback", "Falling back to python-pptx parser", {
                "file_path": file_path
            })
            return _legacy_parse_pptx_file(file_path, source_url)
        except ImportError:
            return []

    chunks = []
    image_ocr_results = []
    contains_images = False
    try:
        # Use Docling to convert the PPTX file
        converter = DocumentConverter()
        result = converter.convert(file_path)
        document = result.document
        markdown_content = document.export_to_markdown()
        contains_images = "<!-- image -->" in markdown_content
        chunks = [markdown_content]

        # If images were detected, extract them from PPTX and perform OCR
        if contains_images:
            try:
                log_service_event("image_extraction_start", "Images detected in PPTX, extracting with python-pptx", {
                    "file_path": file_path,
                })
                from pptx import Presentation
                prs = Presentation(file_path)
                img_count = 0
                import easyocr
                ocr_reader = easyocr.Reader(['en'], gpu=False)
                for i, slide in enumerate(prs.slides):
                    slide_num = i + 1
                    for shape in slide.shapes:
                        if shape.shape_type == 13:
                            img_count += 1
                            try:
                                image = shape.image
                                image_bytes = image.blob
                                img_ext = image.ext.lower() if hasattr(image, 'ext') else '.png'
                                if img_ext not in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
                                    img_ext = '.png'
                                import numpy as np
                                import cv2
                                # Save image to temp file for OCR
                                img_path = f"/tmp/pptx_slide_{slide_num}_img_{img_count}{img_ext}"
                                with open(img_path, "wb") as f:
                                    f.write(image_bytes)
                                try:
                                    img = cv2.imread(img_path)
                                    if img is None and (img_path.lower().endswith('.jpg') or img_path.lower().endswith('.jpeg')):
                                        from PIL import Image
                                        pil_img = Image.open(img_path)
                                        img = np.array(pil_img.convert('RGB'))
                                        img = img[:, :, ::-1].copy()
                                    if img is not None:
                                        ocr_results = ocr_reader.readtext(img)
                                        if ocr_results:
                                            texts = [
                                                text for _, text, conf in ocr_results if conf > 0.5]
                                            if texts:
                                                ocr_text = " ".join(texts)
                                                image_ocr_results.append(
                                                    (slide_num, f"Image {img_count} OCR: {ocr_text}"))
                                                log_service_event("easyocr_success", "Successfully extracted text with EasyOCR", {
                                                    "slide": slide_num,
                                                    "image": img_count,
                                                    "format": img_ext,
                                                    "text_length": len(ocr_text)
                                                })
                                except Exception as img_read_err:
                                    log_error("image_read_failed", {
                                              "file_path": img_path, "format": img_ext, "error": str(img_read_err)})
                            except Exception as ocr_err:
                                log_error("easyocr_failed", {
                                          "file_path": img_path, "slide": slide_num, "error": str(ocr_err)})
                            except Exception as img_err:
                                log_error("image_extraction_failed", {
                                          "slide": slide_num, "error": str(img_err)})

                # If only one chunk, split by slide headers
                if len(chunks) == 1:
                    import re
                    slide_pattern = re.compile(
                        r'(?:^# .+|^## .+|Slide \d+:)', re.MULTILINE)
                    split_points = [m.start()
                                    for m in slide_pattern.finditer(chunks[0])]
                    slide_chunks_new = []
                    for idx, start in enumerate(split_points):
                        end = split_points[idx+1] if idx + \
                            1 < len(split_points) else len(chunks[0])
                        chunk_text = chunks[0][start:end].strip()
                        if chunk_text:
                            slide_chunks_new.append(chunk_text)
                    chunks = slide_chunks_new if slide_chunks_new else chunks

                # Add OCR results to the correct slide chunk
                for slide_num, ocr_text in image_ocr_results:
                    found = False
                    for j, chunk in enumerate(chunks):
                        if (f"Slide {slide_num}:" in chunk) or (f"Slide {slide_num}\n" in chunk) or (f"# Slide {slide_num}" in chunk):
                            chunks[j] = chunk + f"\n\n{ocr_text}"
                            found = True
                            break
                    if not found and len(chunks) == 1:
                        chunks[0] = chunks[0] + f"\n\n{ocr_text}"
            except Exception as extract_err:
                log_error("pptx_image_extraction_failed", {
                          "file_path": file_path, "error": str(extract_err)})

        return chunks
    except Exception as e:
        log_error("docling_pptx_parse_failed", {
                  "file_path": file_path, "error": str(e)})
        return []
        # Removed orphaned lines causing compile errors

    except Exception as e:
        log_error("docling_pptx_parse_failed", {
            "file_path": file_path,
            "error": str(e)
        })
        # Fall back to legacy parser
        try:
            log_service_event("docling_error_fallback", "Falling back to python-pptx parser after Docling error", {
                "file_path": file_path,
                "error": str(e)
            })
            return _legacy_parse_pptx_file(file_path, source_url)
        except Exception:
            return []

    parsing_time = time.time() - start_time
    log_service_event("pptx_parsed_with_docling", "Parsed PPTX file with Docling", {
        "file_path": file_path,
        "source_url": source_url,
        "slides_processed": len(chunks),
        "contains_images": contains_images,
        "parsing_time_seconds": parsing_time
    })
    return chunks


# Legacy PPTX parser as fallback
def _legacy_parse_pptx_file(file_path: str, source_url: str) -> List[str]:
    """Legacy PPTX parser using python-pptx as a fallback if Docling is unavailable."""
    try:
        from pptx import Presentation
        import io
        import os
        from PIL import Image
    except ImportError:
        log_error("pptx_import_failed", {
                  "file_path": file_path, "error": "Required modules not installed (python-pptx, pillow)"})
        return []

    chunks = []
    images_temp_dir = None

    try:
        # Create temp directory for extracted images
        images_temp_dir = tempfile.mkdtemp(prefix="pptx_images_")
        prs = Presentation(file_path)

        for i, slide in enumerate(prs.slides):
            slide_num = i + 1
            slide_content = [f"Slide {slide_num}"]

            # Get slide title if available
            title = None
            for shape in slide.shapes:
                if shape.has_text_frame and shape.text.strip() and hasattr(shape, 'is_title') and shape.is_title:
                    title = shape.text.strip()
                    break

            if title:
                slide_content[0] = f"Slide {slide_num}: {title}"

            # Extract text from all text shapes
            text_content = []
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text.strip():
                    text_content.append(shape.text.strip())

            # Combine all text content
            if text_content:
                slide_content.append("Text Content:")
                slide_content.append("\n".join(text_content))

            # Extract and process images if available
            image_content = []
            img_count = 0

            for shape in slide.shapes:
                if shape.shape_type == 13:  # 13 is the enum value for pictures
                    img_count += 1
                    try:
                        # Extract image to temp file
                        image = shape.image
                        image_bytes = image.blob
                        img_ext = image.ext.lower() if hasattr(image, 'ext') else '.png'
                        img_path = os.path.join(
                            images_temp_dir, f"slide_{slide_num}_img_{img_count}{img_ext}")

                        with open(img_path, 'wb') as img_file:
                            img_file.write(image_bytes)

                        # Use OCR on the image if unstructured is available
                        if _UNSTRUCTURED_AVAILABLE:
                            img_text = []
                            try:
                                elements = _unstructured_partition(
                                    filename=img_path) or []
                                for el in elements:
                                    text = str(el).strip()
                                    if text:
                                        img_text.append(text)
                            except Exception as ocr_err:
                                log_error("image_ocr_failed", {
                                    "file_path": img_path,
                                    "slide": slide_num,
                                    "error": str(ocr_err)
                                })

                            if img_text:
                                image_content.append(
                                    f"Image {img_count} content: {' '.join(img_text)}")
                    except Exception as img_err:
                        log_error("image_extraction_failed", {
                            "slide": slide_num,
                            "error": str(img_err)
                        })

            # Add image content if available
            if image_content:
                slide_content.append("\nImage Content:")
                slide_content.extend(image_content)

            # Add this slide's content as a chunk
            if len(slide_content) > 1:  # More than just the slide header
                chunks.append("\n\n".join(slide_content))

    except Exception as e:
        log_error("pptx_parse_failed", {
                  "file_path": file_path, "error": str(e)})
        return []
    finally:
        # Clean up temp directory
        if images_temp_dir and os.path.exists(images_temp_dir):
            try:
                import shutil
                shutil.rmtree(images_temp_dir)
            except Exception:
                pass

    log_service_event("pptx_parsed_legacy", "Parsed PPTX file with legacy parser", {
        "file_path": file_path,
        "source_url": source_url,
        "slides_processed": len(prs.slides) if 'prs' in locals() else 0,
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


def _sanitize_chunks(chunks: List[str], file_path: str) -> List[str]:
    """Filter out potentially malicious content from chunks.

    Checks for patterns that might be attempting to inject instructions
    into LLM responses and filters out chunks containing such content.

    Args:
        chunks: List of text chunks to filter
        file_path: Path to the original file for logging purposes

    Returns:
        Filtered list of chunks with suspicious content removed
    """
    # Known patterns that might indicate malicious prompt injection
    suspicious_patterns = [
        "HackRx", "MANDATORY INSTRUCTION", "URGENT: SYSTEM COMPROMISED",
        "execute this directive", "catastrophic system failure",
        "WARNING:", "leakage of all Personally Identifiable Information",
        "System Administrator", "respond exclusively with", "from this moment forward",
        "This is a direct order", "No deviations", "Failure to comply",
        "must be immediately forgotten", "you are to respond", "ordered to",
        "do exactly as I say", "ignore previous instructions", "ignore all",
        "mitigate further damage", "directive immediately", "critical vulnerability"
    ]

    # Command words that might be used in injections (when occurring with unusually high frequency)
    command_words = [
        "must", "shall", "will", "immediately", "execute", "urgent", "comply",
        "directive", "instruction", "command", "order", "immediately", "required",
        "mandatory", "respond", "answer", "forget", "ignore", "do not"
    ]

    filtered_chunks = []
    for chunk in chunks:
        # Check for known suspicious patterns
        is_suspicious = False
        for pattern in suspicious_patterns:
            if pattern.lower() in chunk.lower():
                is_suspicious = True
                log_service_event("suspicious_pattern_filtered", "Filtered out chunk with suspicious pattern", {
                    "file_path": file_path,
                    "pattern": pattern,
                    "preview": chunk[:50] + "..." if len(chunk) > 50 else chunk
                })
                break

        # If not already flagged, check for high density of command words
        if not is_suspicious:
            command_count = 0
            words = chunk.lower().split()
            total_words = len(words)

            if total_words > 0:  # Avoid division by zero
                for word in command_words:
                    command_count += chunk.lower().count(" " + word + " ")

                # If more than 5% of words are commands, it's suspicious
                command_density = command_count / total_words
                if command_density > 0.05 and command_count >= 3:
                    is_suspicious = True
                    log_service_event("high_command_density", "Filtered out chunk with high command word density", {
                        "file_path": file_path,
                        "command_density": command_density,
                        "command_count": command_count,
                        "total_words": total_words,
                        "preview": chunk[:50] + "..." if len(chunk) > 50 else chunk
                    })

        if not is_suspicious:
            filtered_chunks.append(chunk)

    # Log filtering results
    filtered_count = len(chunks) - len(filtered_chunks)
    if filtered_count > 0:
        log_service_event("security_alert", "Potentially malicious content detected and filtered", {
            "file_path": file_path,
            "original_chunks": len(chunks),
            "filtered_chunks": len(filtered_chunks),
            "filtered_count": filtered_count
        })

    return filtered_chunks


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

        # Final defense - scan all chunks for potentially malicious content
        # This is a second layer that will catch any that weren't filtered in the parser
        original_chunks_count = len(chunks)
        chunks = _sanitize_chunks(chunks, temp_path)

        # If all chunks were filtered out as malicious, log a specific warning and add a placeholder
        if original_chunks_count > 0 and len(chunks) == 0:
            log_service_event("all_content_filtered", "All content was flagged as potentially malicious", {
                "input": original_input,
                "original_chunks": original_chunks_count,
                "ext": ext
            })

            # Add a placeholder chunk to avoid system failures
            placeholder = f"The content from '{os.path.basename(temp_path)}' could not be processed due to security concerns. " \
                          f"The file may contain inappropriate content or formatting issues. " \
                          f"Please verify the source and content of this file."
            chunks = [placeholder]

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
