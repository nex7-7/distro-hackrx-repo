"""
Embeddings Module

This module contains functions for creating embeddings from text chunks.
"""
import time
import uuid
from typing import List, Dict, Any
from datetime import datetime

# Import local modules
from logger import log_service_event


def create_embeddings(chunks: List[str], model, model_name: str) -> Any:
    """
    Generate embeddings for text chunks using the specified model.

    Parameters:
        chunks (List[str]): List of text chunks to embed
        model: The embedding model to use
        model_name (str): Name of the embedding model

    Returns:
        Any: The generated embeddings
    """
    ingest_id = str(uuid.uuid4())
    overall_start_time = time.time()

    # Log detailed information about the embedding process starting
    log_service_event("embedding_process_start", "Starting embedding generation process", {
        "ingest_id": ingest_id,
        "chunks_count": len(chunks),
        "total_text_size": sum(len(chunk) for chunk in chunks),
        "embedding_model": model_name,
        "timestamp": datetime.now().isoformat()
    })

    # Generate embeddings with BGE's recommended prompt wrapper
    print(f"\n🧠 Generating embeddings for {len(chunks)} chunks...")

    # Log embedding generation parameters
    log_service_event("embedding_generation_start", "Starting chunk embedding generation", {
        "ingest_id": ingest_id,
        "chunks_count": len(chunks),
        "bge_prompt_used": True,
        "model_name": model_name,
        "avg_chunk_length": sum(len(chunk) for chunk in chunks) / len(chunks) if chunks else 0,
        "max_chunk_length": max(len(chunk) for chunk in chunks) if chunks else 0,
        "min_chunk_length": min(len(chunk) for chunk in chunks) if chunks else 0
    })

    # Apply the BGE prompt wrapper to each chunk
    wrapping_start_time = time.time()
    wrapped_chunks = [
        f"Represent this sentence for searching relevant passages: {chunk}" for chunk in chunks]
    wrapping_time = time.time() - wrapping_start_time

    # Log prompt wrapping completion
    log_service_event("prompt_wrapping_complete", "Completed BGE prompt wrapping", {
        "ingest_id": ingest_id,
        "chunks_count": len(wrapped_chunks),
        "avg_wrapped_length": sum(len(chunk) for chunk in wrapped_chunks) / len(wrapped_chunks) if wrapped_chunks else 0,
        "wrapping_time_seconds": wrapping_time,
        "wrapper_template": "Represent this sentence for searching relevant passages: {chunk}"
    })

    print("🔍 Using BGE recommended prompt wrapper for better embedding quality")

    # Generate embeddings with timing
    embedding_start_time = time.time()
    embeddings = model.encode(wrapped_chunks, show_progress_bar=True)
    embedding_time = time.time() - embedding_start_time
    total_embedding_process_time = time.time() - overall_start_time

    # Calculate embedding statistics
    embedding_norms = [float(sum(x**2 for x in emb)**0.5)
                       for emb in embeddings]

    # Log detailed embedding generation statistics
    log_service_event("embedding_generation_complete", "Completed chunk embedding generation", {
        "ingest_id": ingest_id,
        "chunks_count": len(chunks),
        "embedding_dimension": embeddings.shape[1],
        "embedding_time_seconds": embedding_time,
        "embedding_throughput": len(chunks) / embedding_time if embedding_time > 0 else 0,
        "total_embedding_process_time": total_embedding_process_time,
        "embedding_stats": {
            "mean_norm": sum(embedding_norms) / len(embedding_norms) if embedding_norms else 0,
            "max_norm": max(embedding_norms) if embedding_norms else 0,
            "min_norm": min(embedding_norms) if embedding_norms else 0,
            "memory_usage_mb": embeddings.nbytes / (1024 * 1024)
        }
    })

    # --- DEBUG PRINT 1: EMBEDDINGS RECEIVED ---
    print("\n" + "="*50)
    print("1. EMBEDDINGS RECEIVED")
    print("="*50)
    print(f"Shape of embeddings array: {embeddings.shape}")
    print("="*50 + "\n")
    # ----------------------------------------

    return embeddings


def create_query_embedding(question: str, model, model_name: str) -> List[float]:
    """
    Generate embedding for a query using the specified model.

    Parameters:
        question (str): Question to embed
        model: The embedding model to use
        model_name (str): Name of the embedding model

    Returns:
        List[float]: The generated embedding as a list
    """
    # Use BGE's recommended prompt wrapper
    bge_wrapped_query = f"Represent this sentence for searching relevant passages: {question}"

    # Generate embedding
    query_vector = model.encode(bge_wrapped_query).tolist()

    # Log embedding generation
    log_service_event("query_embedding_generated", "Question embedding generated", {
        "question": question,
        "vector_dimension": len(query_vector),
        "bge_prompt_used": True,
        "model_name": model_name
    })

    return query_vector
