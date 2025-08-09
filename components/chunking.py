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
from logger import log_service_event


def clean_and_rechunk_texts(chunk_texts: List[str], chunk_token_size: int = 450, overlap: int = 30) -> List[str]:
    """
    Cleans up chunk texts and re-chunks them to a specified token size with overlap.
    Removes extra spaces and newlines, then combines chunks to reach the target token size.

    Parameters:
        chunk_texts (List[str]): Original text chunks to process
        chunk_token_size (int): Target token size for each chunk
        overlap (int): Number of tokens to overlap between chunks

    Returns:
        List[str]: List of cleaned and rechunked texts
    """
    rechunk_id = str(uuid.uuid4())
    start_time = time.time()

    # Log start of rechunking process
    log_service_event("rechunking_start", "Starting document rechunking process", {
        "rechunk_id": rechunk_id,
        "original_chunks": len(chunk_texts),
        "target_chunk_token_size": chunk_token_size,
        "overlap_tokens": overlap,
        "original_total_chars": sum(len(chunk) for chunk in chunk_texts)
    })

    # Clean up each chunk: remove extra spaces and newlines
    cleaning_start = time.time()
    cleaned = [" ".join(chunk.split()).replace("\n", " ")
               for chunk in chunk_texts]
    cleaning_time = time.time() - cleaning_start

    # Log cleaning results
    log_service_event("chunk_cleaning_complete", "Completed chunk text cleaning", {
        "rechunk_id": rechunk_id,
        "chunks_count": len(cleaned),
        "cleaning_time_seconds": cleaning_time,
        "chars_before_cleaning": sum(len(chunk) for chunk in chunk_texts),
        "chars_after_cleaning": sum(len(chunk) for chunk in cleaned),
        "reduction_percentage": round((1 - sum(len(chunk) for chunk in cleaned) /
                                       sum(len(chunk) for chunk in chunk_texts)) * 100, 2) if sum(len(chunk) for chunk in chunk_texts) > 0 else 0
    })

    # Concatenate all cleaned chunks into one big text
    tokenization_start = time.time()
    full_text = " ".join(cleaned)
    tokens = word_tokenize(full_text)
    tokenization_time = time.time() - tokenization_start

    # Log tokenization stats
    log_service_event("text_tokenization", "Tokenized full text for rechunking", {
        "rechunk_id": rechunk_id,
        "total_tokens": len(tokens),
        "tokens_per_char": len(tokens) / len(full_text) if len(full_text) > 0 else 0,
        "tokenization_time_seconds": tokenization_time
    })

    # Perform rechunking with overlap
    rechunking_start = time.time()
    new_chunks = []
    i = 0
    while i < len(tokens):
        chunk = tokens[i:i+chunk_token_size]
        new_chunks.append(" ".join(chunk))
        i += chunk_token_size - overlap
    rechunking_time = time.time() - rechunking_start

    # Calculate rechunking statistics
    new_chunk_lengths = [len(chunk) for chunk in new_chunks]
    new_chunk_word_counts = [len(chunk.split()) for chunk in new_chunks]

    # Log rechunking completion with detailed statistics
    log_service_event("rechunking_complete", "Completed text rechunking process", {
        "rechunk_id": rechunk_id,
        "original_chunks": len(chunk_texts),
        "new_chunks": len(new_chunks),
        "rechunking_time_seconds": rechunking_time,
        "total_processing_time": time.time() - start_time,
        "avg_new_chunk_length": sum(new_chunk_lengths) / len(new_chunk_lengths) if new_chunk_lengths else 0,
        "max_new_chunk_length": max(new_chunk_lengths) if new_chunk_lengths else 0,
        "min_new_chunk_length": min(new_chunk_lengths) if new_chunk_lengths else 0,
        "avg_tokens_per_chunk": chunk_token_size - overlap,
        "avg_words_per_chunk": sum(new_chunk_word_counts) / len(new_chunk_word_counts) if new_chunk_word_counts else 0,
        "total_chars": sum(new_chunk_lengths),
        "distribution": {
            "short_chunks": sum(1 for length in new_chunk_lengths if length < 200),
            "medium_chunks": sum(1 for length in new_chunk_lengths if 200 <= length < 500),
            "long_chunks": sum(1 for length in new_chunk_lengths if length >= 500)
        }
    })

    # Log sample chunks (first 2 only)
    for i, chunk in enumerate(new_chunks[:2]):
        log_service_event("rechunked_sample", f"Sample rechunked text {i+1}", {
            "rechunk_id": rechunk_id,
            "chunk_index": i,
            "char_length": len(chunk),
            "word_count": len(chunk.split()),
            "preview": chunk[:100] + ("..." if len(chunk) > 100 else "")
        })

    return new_chunks
