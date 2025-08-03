"""
Example usage of the document ingestion pipeline.

This script demonstrates how to use the data extraction and chunking
components to process policy documents.
"""

from pathlib import Path
import sys
from typing import List
from src.components import DocumentIngestionPipeline, ChunkingConfig
from src.schemas.models import DocumentChunk


def save_chunks_to_markdown(chunks: List[DocumentChunk], output_dir: Path) -> None:
    """
    Save document chunks to individual markdown files.
    
    Creates a separate .md file for each source document containing all its chunks.
    Follows SRP by focusing solely on chunk persistence.
    
    Args:
        chunks: List of DocumentChunk objects to save
        output_dir: Directory where markdown files will be saved
    """
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Group chunks by source document
    chunks_by_document = {}
    for chunk in chunks:
        doc_name = chunk.metadata.source_document
        if doc_name not in chunks_by_document:
            chunks_by_document[doc_name] = []
        chunks_by_document[doc_name].append(chunk)
    
    # Save each document's chunks to a separate markdown file
    for doc_name, doc_chunks in chunks_by_document.items():
        # Create filename: remove .pdf extension and add .md
        base_name = doc_name.replace('.pdf', '').replace('.docx', '')
        output_file = output_dir / f"{base_name}_chunks.md"
        
        # Sort chunks by chunk_index to maintain order
        doc_chunks.sort(key=lambda x: x.metadata.chunk_index)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            # Write document header
            f.write(f"# Document Chunks: {doc_name}\n\n")
            f.write(f"**Total Chunks:** {len(doc_chunks)}\n")
            f.write(f"**Processing Date:** {Path().cwd()}\n\n")
            f.write("---\n\n")
            
            # Write each chunk
            for chunk in doc_chunks:
                f.write(f"## Chunk {chunk.metadata.chunk_index + 1}\n\n")
                f.write(f"**Clause ID:** `{chunk.metadata.clause_id}`\n")
                f.write(f"**Character Count:** {chunk.metadata.char_count}\n")
                if chunk.metadata.page_number:
                    f.write(f"**Page Number:** {chunk.metadata.page_number}\n")
                f.write("\n### Content\n\n")
                f.write(chunk.content)
                f.write("\n\n---\n\n")
        
        print(f"💾 Saved {len(doc_chunks)} chunks to: {output_file}")


def main():
    """
    Example of processing policy documents from the data/raw directory.
    """
    # Configure chunking parameters
    config = ChunkingConfig(
        include_tables=True,
        use_page_based_chunking =True,
    )
    
    # Initialize the pipeline
    pipeline = DocumentIngestionPipeline(config)
    
    # Path to your policy documents
    data_dir = Path("data/raw")
    
    print(f"Processing documents from: {data_dir}")
    print(f"Chunking config: min={config.min_chunk_chars}, max={config.max_chunk_chars}")
    print("-" * 50)
    
    # Process all documents and collect chunks
    total_chunks = 0
    processed_docs = 0
    all_chunks = []  # Collect all chunks for saving
    
    try:
        for chunk in pipeline.process_directory(data_dir):
            total_chunks += 1
            all_chunks.append(chunk)  # Store chunk for later saving
            
            # Print sample information for first few chunks
            if total_chunks <= 5:
                print(f"Chunk {total_chunks}:")
                print(f"  Source: {chunk.metadata.source_document}")
                print(f"  Clause ID: {chunk.metadata.clause_id}")
                print(f"  Size: {chunk.metadata.char_count} chars")
                print(f"  Preview: {chunk.content[:150]}...")
                print()
            
            # Count documents processed (approximate)
            if chunk.metadata.chunk_index == 0:
                processed_docs += 1
                print(f"Started processing: {chunk.metadata.source_document}")
    
    except Exception as e:
        print(f"Error during processing: {e}")
        return
    
    print("-" * 50)
    print(f"Processing complete!")
    print(f"Documents processed: {processed_docs}")
    print(f"Total chunks created: {total_chunks}")
    print(f"Average chunks per document: {total_chunks / processed_docs:.1f}")
    
    # Save all chunks to markdown files
    if all_chunks:
        print("\n📁 Saving chunks to markdown files...")
        output_dir = Path("data/chunked")
        save_chunks_to_markdown(all_chunks, output_dir)
        print(f"✅ All chunks saved to: {output_dir}")
    else:
        print("⚠️ No chunks to save.")


def process_single_document_example():
    """
    Example of processing a single document.
    """
    config = ChunkingConfig(
        min_chunk_chars=500,
        max_chunk_chars=2000
    )
    
    pipeline = DocumentIngestionPipeline(config)
    
    # Process a specific document
    pdf_path = Path("data/raw/BAJHLIP23020V012223.pdf")
    
    if not pdf_path.exists():
        print(f"Document not found: {pdf_path}")
        return
    
    try:
        chunks = pipeline.process_document(pdf_path)
        
        print(f"Processed: {pdf_path.name}")
        print(f"Created {len(chunks)} chunks")
        
        # Show details of first chunk
        if chunks:
            first_chunk = chunks[0]
            print(f"\nFirst chunk details:")
            print(f"  Clause ID: {first_chunk.metadata.clause_id}")
            print(f"  Size: {first_chunk.metadata.char_count} chars")
            print(f"  Content preview:")
            print(f"    {first_chunk.content[:300]}...")
            
            # Save chunks to markdown
            print(f"\n📁 Saving chunks for {pdf_path.name}...")
            output_dir = Path("data/chunked")
            save_chunks_to_markdown(chunks, output_dir)
            print(f"✅ Chunks saved to: {output_dir}")
            
    except Exception as e:
        print(f"Error processing document: {e}")


if __name__ == "__main__":
    print("=== Document Ingestion Pipeline Testing ===\n")
    
    # print("1. Processing all documents in directory:")
    main()
    
    # print("\n" + "="*60 + "\n")
    
    # print("2. Processing single document:")
    # process_single_document_example()
