"""
Extract and chunk data from a document using unstructured's partitioning API.
Supports all major file types and outputs structured JSON.
"""

import os
from typing import List
from multiprocessing import Pool, cpu_count
from unstructured.partition.auto import partition
import json


def extract_doc_elements(doc_path: str):
    """
    Extracts structured elements from a document using unstructured's auto partition.
    Tries English first, then auto-detects language if nothing found.
    Returns a list of element objects.
    """
    elements = partition(filename=doc_path, languages=["eng"])
    if not elements or all((not str(el).strip()) for el in elements):
        elements = partition(filename=doc_path, languages=None)
    return elements

def chunk_elements(elements, chunk_size: int = 500):
    """
    Chunks a list of element objects into larger blocks, preserving structure.
    Each chunk is a list of element dicts.
    """
    chunks = []
    current_chunk = []
    current_len = 0
    for el in elements:
        text = str(el)
        words = text.split()
        if current_len + len(words) > chunk_size and current_chunk:
            chunks.append([e.to_dict() for e in current_chunk])
            current_chunk = []
            current_len = 0
        current_chunk.append(el)
        current_len += len(words)
    if current_chunk:
        chunks.append([e.to_dict() for e in current_chunk])
    return chunks


def process_document(doc_path: str, chunk_size: int = 500):
    """
    Extracts and chunks structured elements from a document (any supported type).
    Returns a list of chunks, each chunk is a list of element dicts.
    """
    import time
    start_time = time.time()
    elements = extract_doc_elements(doc_path)
    chunks = chunk_elements(elements, chunk_size=chunk_size)
    elapsed = time.time() - start_time
    print(f"[INFO] Processed {os.path.basename(doc_path)} in {elapsed:.2f} seconds.")
    return chunks

if __name__ == "__main__":
    # Example usage for a single document (any supported type)
    doc_path = "/Users/aayushshah/Programming/hackrx-repo/40-kWp-Technical-Bid-ZO-1.docx"
    output_file = os.path.splitext(os.path.basename(doc_path))[0] + "_chunks.json"
    chunks = process_document(doc_path, chunk_size=500)
    print(f"Extracted {len(chunks)} chunks from {os.path.basename(doc_path)}")
    # Save chunks to JSON file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Chunks saved to {output_file}")
    # Print preview of first 3 chunks
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n--- Chunk {i+1} ---\n{json.dumps(chunk, ensure_ascii=False, indent=2)[:500]}...")
