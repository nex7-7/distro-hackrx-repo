"""
Search Module

This module contains functions for searching in Weaviate.
"""
import time
import uuid
from typing import List, Dict, Any, Tuple
import weaviate
from weaviate.exceptions import WeaviateClosedClientError

# Import local modules
from components.utils.logger import log_service_event


def hybrid_search(
    question: str,
    query_vector: List[float],
    collection_name: str,
    weaviate_client: weaviate.WeaviateClient,
    limit: int = 8,
    alpha: float = 0.5
) -> Tuple[List[str], List[str], List[float]]:
    """
    Perform hybrid search in Weaviate combining vector and keyword search.

    Parameters:
        question (str): The question to search for
        query_vector (List[float]): The vector representation of the question
        collection_name (str): Name of the collection to search in
        weaviate_client (weaviate.WeaviateClient): Weaviate client
        limit (int): Maximum number of results to return
        alpha (float): Balance between vector (0) and keyword (1) search

    Returns:
        Tuple[List[str], List[str], List[float]]: Tuple containing (context_chunks, chunk_ids, chunk_scores)
    """
    question_id = str(uuid.uuid4())

    # Log retrieval parameters before executing query
    log_service_event("chunk_retrieval_params", "Preparing to retrieve chunks using hybrid search", {
        "question_id": question_id,
        "query_text": question,
        "vector_dimension": len(query_vector),
        "alpha": alpha,
        "limit": limit,
        "collection_name": collection_name,
        "reranking_disabled": True  # Flag to indicate reranking is disabled
    })

    retrieval_start = time.time()

    try:
        # Check if client is still connected
        weaviate_client.is_ready()

        # Get collection from client
        policy_collection = weaviate_client.collections.get(collection_name)

        # Execute hybrid search
        response = policy_collection.query.hybrid(
            query=question,  # Text for BM25 keyword search
            vector=query_vector,  # Vector for semantic search
            alpha=alpha,  # Balance between vector (0) and keyword (1) search
            limit=limit,  # Number of results to return
            return_properties=["content"]
        )
    except WeaviateClosedClientError:
        log_service_event("weaviate_client_closed", "Weaviate client is closed during search", {
            "question_id": question_id,
            "collection_name": collection_name
        })
        raise ConnectionError(
            "Weaviate client is closed. Unable to perform search.")
    retrieval_time = time.time() - retrieval_start

    # Extract context chunks and their object IDs
    context_chunks = []
    chunk_ids = []
    chunk_scores = []

    # Process the response objects to extract content and metadata
    for obj in response.objects:
        context_chunks.append(obj.properties['content'])
        chunk_ids.append(str(obj.uuid))

        # Extract hybrid search score if available
        if hasattr(obj, 'metadata') and hasattr(obj.metadata, 'hybrid_score'):
            chunk_scores.append(obj.metadata.hybrid_score)
        else:
            chunk_scores.append(None)

    # Log detailed retrieval results
    log_service_event("context_retrieved", "Retrieved context chunks from Weaviate", {
        "question_id": question_id,
        "chunks_count": len(context_chunks),
        "chunks_avg_length": sum(len(chunk) for chunk in context_chunks) / len(context_chunks) if context_chunks else 0,
        "retrieval_time_seconds": retrieval_time,
        "has_scores": any(score is not None for score in chunk_scores),
        "top_chunk_length": len(context_chunks[0]) if context_chunks else 0,
        "bottom_chunk_length": len(context_chunks[-1]) if context_chunks else 0,
        "chunk_count_distribution": {
            "short_chunks": sum(1 for chunk in context_chunks if len(chunk) < 200),
            "medium_chunks": sum(1 for chunk in context_chunks if 200 <= len(chunk) < 500),
            "long_chunks": sum(1 for chunk in context_chunks if len(chunk) >= 500)
        }
    })

    # Log individual chunk details (first 5 chunks only to avoid excessive logging)
    for i, (chunk, chunk_id) in enumerate(zip(context_chunks[:5], chunk_ids[:5])):
        score = chunk_scores[i] if i < len(
            chunk_scores) and chunk_scores[i] is not None else "N/A"
        log_service_event("chunk_details", f"Details for retrieved chunk {i+1}", {
            "question_id": question_id,
            "chunk_index": i,
            "chunk_id": chunk_id,
            "chunk_length": len(chunk),
            "word_count": len(chunk.split()),
            "hybrid_score": score,
            "chunk_preview": chunk[:100] + ("..." if len(chunk) > 100 else "")
        })

    return context_chunks, chunk_ids, chunk_scores
