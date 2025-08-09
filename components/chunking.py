"""
Chunking Module

This module contains functions for text chunking and preprocessing.
"""
import uuid
import time
import nltk
from typing import List
from nltk.tokenize import word_tokenize

# Import local modules
from components.utils.logger import log_service_event


def semantic_chunk_texts(
    chunk_texts: List[str],
    embedding_model=None,
    model_name: str = None,
    similarity_threshold: float = 0.8,
    min_chunk_size: int = 3,
    max_chunk_size: int = 12
) -> List[str]:
    """
    Splits text into semantically coherent chunks using sentence embeddings and similarity.

    Parameters:
        chunk_texts (List[str]): List of raw text chunks (usually one big chunk)
        embedding_model: SentenceTransformer model for embeddings
        model_name (str): Name of the embedding model
        similarity_threshold (float): Cosine similarity threshold to merge sentences
        min_chunk_size (int): Minimum number of sentences per chunk
        max_chunk_size (int): Maximum number of sentences per chunk

    Returns:
        List[str]: List of semantically chunked texts
    """
    import numpy as np
    from nltk.tokenize import sent_tokenize
    from components.utils.logger import log_service_event
    import uuid
    import time

    rechunk_id = str(uuid.uuid4())
    start_time = time.time()
    log_service_event("semantic_chunking_start", "Starting semantic chunking process", {
        "rechunk_id": rechunk_id,
        "original_chunks": len(chunk_texts)
    })

    # Concatenate all input chunks into one text
    full_text = " ".join([" ".join(chunk.split()).replace("\n", " ")
                         for chunk in chunk_texts])
    sentences = sent_tokenize(full_text)
    log_service_event("sentence_tokenization", "Tokenized text into sentences", {
        "rechunk_id": rechunk_id,
        "sentence_count": len(sentences)
    })

    # Generate embeddings for all sentences
    embedding_start = time.time()
    if embedding_model is None:
        raise ValueError(
            "embedding_model must be provided for semantic chunking.")
    sentence_embeddings = embedding_model.encode(
        sentences, show_progress_bar=False, normalize_embeddings=True)
    embedding_time = time.time() - embedding_start
    log_service_event("sentence_embedding", "Generated sentence embeddings", {
        "rechunk_id": rechunk_id,
        "embedding_time_seconds": embedding_time,
        "embedding_shape": str(np.shape(sentence_embeddings))
    })

    # Group sentences into semantic chunks
    chunks = []
    current_chunk = [sentences[0]] if sentences else []
    for i in range(1, len(sentences)):
        prev_emb = sentence_embeddings[i-1]
        curr_emb = sentence_embeddings[i]
        similarity = float(np.dot(prev_emb, curr_emb))
        # If similar enough and chunk not too big, merge
        if similarity >= similarity_threshold and len(current_chunk) < max_chunk_size:
            current_chunk.append(sentences[i])
        else:
            # If chunk too small, force merge
            if len(current_chunk) < min_chunk_size and i < len(sentences)-1:
                current_chunk.append(sentences[i])
            else:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentences[i]]
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    chunk_lengths = [len(chunk) for chunk in chunks]
    log_service_event("semantic_chunking_complete", "Completed semantic chunking", {
        "rechunk_id": rechunk_id,
        "new_chunks": len(chunks),
        "avg_chunk_length": sum(chunk_lengths)/len(chunk_lengths) if chunk_lengths else 0,
        "max_chunk_length": max(chunk_lengths) if chunk_lengths else 0,
        "min_chunk_length": min(chunk_lengths) if chunk_lengths else 0,
        "total_processing_time": time.time() - start_time
    })

    # Log sample chunks
    for i, chunk in enumerate(chunks[:2]):
        log_service_event("semantic_chunk_sample", f"Sample semantic chunk {i+1}", {
            "rechunk_id": rechunk_id,
            "chunk_index": i,
            "char_length": len(chunk),
            "word_count": len(chunk.split()),
            "preview": chunk[:100] + ("..." if len(chunk) > 100 else "")
        })

    return chunks
