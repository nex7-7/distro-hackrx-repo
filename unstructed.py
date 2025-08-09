"""
Extract and chunk data from a PDF using unstructured's partitioning API.
Supports parallel processing for speed.
"""

import os
from typing import List
from multiprocessing import Pool, cpu_count
from unstructured.partition.pdf import partition_pdf

def extract_pdf_elements(pdf_path: str) -> List[str]:
    """
    Extracts text elements from a PDF file using unstructured.
    
    Parameters:
        pdf_path (str): Path to the PDF file.
    
    Returns:
        List[str]: List of text elements (strings) from the PDF.
    """
    elements = partition_pdf(filename=pdf_path)
    # Each element is a TextElement, Table, Title, etc.
    # We'll extract the text from each element.
    return [str(el) for el in elements if str(el).strip()]

def chunk_text(elements: List[str], chunk_size: int = 500) -> List[str]:
    """
    Chunks a list of text elements into larger text blocks.
    
    Parameters:
        elements (List[str]): List of text elements.
        chunk_size (int): Approximate number of words per chunk.
    
    Returns:
        List[str]: List of chunked text blocks.
    """
    chunks = []
    current_chunk = []
    current_len = 0
    for el in elements:
        words = el.split()
        if current_len + len(words) > chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_len = 0
        current_chunk.append(el)
        current_len += len(words)
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def process_pdf(pdf_path: str, chunk_size: int = 500) -> List[str]:
    """
    Extracts and chunks text from a single PDF.
    """
    import time
    start_time = time.time()
    elements = extract_pdf_elements(pdf_path)
    chunks = chunk_text(elements, chunk_size=chunk_size)
    elapsed = time.time() - start_time
    print(f"[INFO] Processed {os.path.basename(pdf_path)} in {elapsed:.2f} seconds.")
    return chunks

def process_pdfs_in_parallel(pdf_paths: List[str], chunk_size: int = 500) -> List[List[str]]:
    """
    Processes multiple PDFs in parallel.
    """
    import time
    start_time = time.time()
    with Pool(processes=min(cpu_count(), len(pdf_paths))) as pool:
        results = pool.starmap(process_pdf, [(path, chunk_size) for path in pdf_paths])
    elapsed = time.time() - start_time
    print(f"[INFO] Processed {len(pdf_paths)} PDFs in {elapsed:.2f} seconds (parallel).")
    return results

if __name__ == "__main__":
    # Example usage for a single PDF
    pdf_path = "/Users/aayushshah/Programming/hackrx-repo/principia_newton.pdf"
    output_file = os.path.splitext(os.path.basename(pdf_path))[0] + "_chunks.txt"
    chunks = process_pdf(pdf_path, chunk_size=500)
    print(f"Extracted {len(chunks)} chunks from {os.path.basename(pdf_path)}")
    # Save chunks to file
    with open(output_file, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks):
            f.write(f"--- Chunk {i+1} ---\n{chunk}\n\n")
    print(f"[INFO] Chunks saved to {output_file}")
    # Print preview of first 3 chunks
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n--- Chunk {i+1} ---\n{chunk[:500]}...")  # Print first 500 chars of each chunk