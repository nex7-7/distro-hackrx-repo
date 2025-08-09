"""
Extract and chunk data from a document using unstructured's partitioning API.
Supports all major file types (PDF, DOCX, PPTX, XLSX, images, etc.) and outputs structured JSON.
Single-process extraction for all file types. Optionally strips metadata from output.
Follows PEP 8, PEP 257, and project coding standards.
"""

import os
import json
import time
from typing import List, Any
from unstructured.partition.auto import partition

def extract_doc_elements(doc_path: str) -> List[Any]:
    """
    Extract structured elements from a document using unstructured's auto partition.
    Tries English first, then auto-detects language if nothing found.
    Returns a list of element objects.
    """
    try:
        elements = partition(filename=doc_path, languages=["eng"])
        if not elements or all((not str(el).strip()) for el in elements):
            elements = partition(filename=doc_path, languages=None)
        return elements
    except Exception as e:
        print(f"[ERROR] Failed to extract elements from {doc_path}: {e}")
        return []

def chunk_elements(elements: List[Any], chunk_size: int = 500, strip_metadata: bool = False) -> List[List[dict]]:
    """
    Chunk a list of element objects into larger blocks, preserving structure.
    Each chunk is a list of element dicts.
    Args:
        elements (List[Any]): List of element objects.
        chunk_size (int): Max words per chunk.
        strip_metadata (bool): If True, removes metadata from output dicts.
    Returns:
        List[List[dict]]: List of chunks (each chunk is a list of element dicts).
    """
    chunks = []
    current_chunk = []
    current_len = 0
    for el in elements:
        text = str(el)
        words = text.split()
        if current_len + len(words) > chunk_size and current_chunk:
            if strip_metadata:
                chunk = []
                for e in current_chunk:
                    d = e.to_dict()
                    d.pop('metadata', None)
                    chunk.append(d)
                chunks.append(chunk)
            else:
                chunks.append([e.to_dict() for e in current_chunk])
            current_chunk = []
            current_len = 0
        current_chunk.append(el)
        current_len += len(words)
    if current_chunk:
        if strip_metadata:
            chunk = []
            for e in current_chunk:
                d = e.to_dict()
                d.pop('metadata', None)
                chunk.append(d)
            chunks.append(chunk)
        else:
            chunks.append([e.to_dict() for e in current_chunk])
    return chunks

def process_document(
    doc_path: str,
    chunk_size: int = 500,
    strip_metadata: bool = False
) -> List[List[dict]]:
    """
    Extract and chunk structured elements from a document (any supported type).
    Returns a list of chunks, each chunk is a list of element dicts.
    Handles errors gracefully and prints timing info.
    Args:
        doc_path (str): Path to the document.
        chunk_size (int): Max words per chunk.
        strip_metadata (bool): If True, removes metadata from output.
    """
    start_time = time.time()
    elements = extract_doc_elements(doc_path)
    chunks = chunk_elements(elements, chunk_size=chunk_size, strip_metadata=strip_metadata)
    elapsed = time.time() - start_time
    print(f"[INFO] Processed {os.path.basename(doc_path)} in {elapsed:.2f} seconds.")
    return chunks

if __name__ == "__main__":
    # Example usage for a single document (any supported type)
    doc_path = "data/principia_newton.pdf"  # Change as needed
    output_file = os.path.splitext(os.path.basename(doc_path))[0] + "_chunks.json"
    STRIP_METADATA = True  # Remove metadata from output chunks
    chunks = process_document(doc_path, chunk_size=500, strip_metadata=STRIP_METADATA)
    print(f"Extracted {len(chunks)} chunks from {os.path.basename(doc_path)}")
    # Save chunks to JSON file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Chunks saved to {output_file}")
    # Print preview of first 3 chunks
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n--- Chunk {i+1} ---\n{json.dumps(chunk, ensure_ascii=False, indent=2)[:500]}...")