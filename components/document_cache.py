"""
Document Cache Module

This module provides functionality for caching documents based on their content hash.
Each document is stored in its own collection with metadata including the document hash.
"""
import hashlib
import time
from typing import List, Optional, Tuple
import weaviate
import weaviate.classes.config as wvc
from weaviate.exceptions import WeaviateClosedClientError

from components.utils.logger import log_service_event, log_error


def generate_file_hash(file_path: str) -> str:
    """
    Generate a SHA-256 hash for the raw file content.
    
    Parameters:
        file_path (str): Path to the file to hash
        
    Returns:
        str: SHA-256 hash of the file content
    """
    import os
    
    file_size = os.path.getsize(file_path)
    
    # Generate SHA-256 hash from file bytes
    hash_obj = hashlib.sha256()
    
    with open(file_path, 'rb') as f:
        # Read file in chunks to handle large files efficiently
        for chunk in iter(lambda: f.read(8192), b""):
            hash_obj.update(chunk)
    
    file_hash = hash_obj.hexdigest()
    
    log_service_event("file_hash_generated", "Generated file hash", {
        "hash": file_hash,
        "file_path": file_path,
        "file_size_bytes": file_size
    })
    
    return file_hash


def create_collection_name_from_hash(file_hash: str) -> str:
    """
    Create a collection name based on the file hash.
    
    Parameters:
        file_hash (str): The file hash
        
    Returns:
        str: Collection name for the document
    """
    # Use first 16 characters of hash to keep collection name reasonable
    return f"Doc_{file_hash[:16]}"


async def find_cached_document(
    client: weaviate.WeaviateClient, 
    file_hash: str
) -> Optional[str]:
    """
    Check if a document with the given file hash already exists in any collection.
    
    Parameters:
        client (weaviate.WeaviateClient): Weaviate client
        file_hash (str): Hash of the file to search for
        
    Returns:
        Optional[str]: Collection name if found, None otherwise
    """
    search_start = time.time()
    
    try:
        # List all collections
        collections = client.collections.list_all()
        
        # Check each collection that follows our naming pattern
        for collection_name in collections:
            if not collection_name.startswith("Doc_"):
                continue
                
            try:
                collection = client.collections.get(collection_name)
                
                # Get the first object to check the file hash
                response = collection.query.fetch_objects(limit=1)
                
                if response.objects and response.objects[0].properties.get("file_hash") == file_hash:
                    search_time = time.time() - search_start
                    log_service_event("document_cache_hit", "Found cached document", {
                        "file_hash": file_hash,
                        "collection_name": collection_name,
                        "search_time_seconds": search_time
                    })
                    return collection_name
                    
            except Exception as e:
                # Log warning but continue searching other collections
                log_error("cache_search_collection_error", {
                    "collection_name": collection_name,
                    "file_hash": file_hash,
                    "error": str(e)
                })
                continue
        
        search_time = time.time() - search_start
        log_service_event("document_cache_miss", "Document not found in cache", {
            "file_hash": file_hash,
            "collections_searched": len([c for c in collections if c.startswith("Doc_")]),
            "search_time_seconds": search_time
        })
        
        return None
        
    except Exception as e:
        log_error("cache_search_failed", {
            "file_hash": file_hash,
            "error": str(e)
        })
        return None


async def create_cached_collection(
    client: weaviate.WeaviateClient,
    collection_name: str,
    chunks: List[str],
    embeddings,
    file_hash: str,
    source_url: str
) -> weaviate.WeaviateClient:
    """
    Create a new collection for caching a document with its file hash in metadata.
    
    Parameters:
        client (weaviate.WeaviateClient): Weaviate client
        collection_name (str): Name for the new collection
        chunks (List[str]): Document chunks
        embeddings: Document embeddings
        file_hash (str): Hash of the original file
        source_url (str): Original document URL
        
    Returns:
        weaviate.WeaviateClient: The Weaviate client (potentially reconnected)
    """
    creation_start = time.time()
    
    log_service_event("cached_collection_creation_start", "Creating new cached collection", {
        "collection_name": collection_name,
        "file_hash": file_hash,
        "chunks_count": len(chunks),
        "source_url": source_url
    })
    
    try:
        # Check if collection already exists (shouldn't happen but safety check)
        if client.collections.exists(collection_name):
            log_service_event("cached_collection_exists", "Collection already exists, deleting", {
                "collection_name": collection_name,
                "file_hash": file_hash
            })
            client.collections.delete(collection_name)
        
        # Create new collection with enhanced schema for caching
        client.collections.create(
            name=collection_name,
            vectorizer_config=wvc.Configure.Vectorizer.none(),
            properties=[
                wvc.Property(name="content", data_type=wvc.DataType.TEXT),
                wvc.Property(name="file_hash", data_type=wvc.DataType.TEXT),
                wvc.Property(name="source_url", data_type=wvc.DataType.TEXT),
                wvc.Property(name="created_at", data_type=wvc.DataType.DATE),
                wvc.Property(name="chunk_index", data_type=wvc.DataType.INT)
            ]
        )
        
        # Get the collection for batch insertion
        collection = client.collections.get(collection_name)
        
        # Insert chunks with metadata
        from datetime import datetime
        current_time = datetime.now()
        
        with collection.batch.dynamic() as batch:
            for i, chunk in enumerate(chunks):
                batch.add_object(
                    properties={
                        "content": chunk,
                        "file_hash": file_hash,
                        "source_url": source_url,
                        "created_at": current_time,
                        "chunk_index": i
                    },
                    vector=embeddings[i].tolist()
                )
        
        creation_time = time.time() - creation_start
        log_service_event("cached_collection_created", "Successfully created cached collection", {
            "collection_name": collection_name,
            "file_hash": file_hash,
            "chunks_inserted": len(chunks),
            "creation_time_seconds": creation_time,
            "source_url": source_url
        })
        
        return client
        
    except Exception as e:
        log_error("cached_collection_creation_failed", {
            "collection_name": collection_name,
            "file_hash": file_hash,
            "error": str(e)
        })
        raise


async def get_cached_document_stats(client: weaviate.WeaviateClient) -> dict:
    """
    Get statistics about cached documents in Weaviate.
    
    Parameters:
        client (weaviate.WeaviateClient): Weaviate client
        
    Returns:
        dict: Statistics about cached documents
    """
    try:
        collections = client.collections.list_all()
        doc_collections = [c for c in collections if c.startswith("Doc_")]
        
        total_chunks = 0
        unique_documents = len(doc_collections)
        oldest_cache = None
        newest_cache = None
        
        for collection_name in doc_collections:
            try:
                collection = client.collections.get(collection_name)
                # Get total count by fetching objects with a limit
                response = collection.query.fetch_objects(limit=1000)  # Reasonable limit for counting
                total_chunks += len(response.objects)
                
                # Get the creation date from the first object
                if response.objects:
                    created_at = response.objects[0].properties.get("created_at")
                    if created_at:
                        if oldest_cache is None or created_at < oldest_cache:
                            oldest_cache = created_at
                        if newest_cache is None or created_at > newest_cache:
                            newest_cache = created_at
                            
            except Exception as e:
                log_error("cache_stats_collection_error", {
                    "collection_name": collection_name,
                    "error": str(e)
                })
                continue
        
        stats = {
            "unique_documents": unique_documents,
            "total_chunks": total_chunks,
            "oldest_cache": oldest_cache.isoformat() if oldest_cache else None,
            "newest_cache": newest_cache.isoformat() if newest_cache else None,
            "cache_collections": doc_collections
        }
        
        log_service_event("cache_stats_retrieved", "Retrieved cache statistics", stats)
        return stats
        
    except Exception as e:
        log_error("cache_stats_failed", {"error": str(e)})
        return {
            "unique_documents": 0,
            "total_chunks": 0,
            "oldest_cache": None,
            "newest_cache": None,
            "cache_collections": [],
            "error": str(e)
        }


async def process_document_with_cache(
    file_path: str,
    original_source_url: str,
    weaviate_client,
    embedding_model,
    model_name: str
):
    """
    Process document with caching - check cache first, only parse/chunk/embed if cache miss.
    
    Parameters:
        file_path (str): Path to the downloaded file
        original_source_url (str): Original document URL
        weaviate_client: Weaviate client
        embedding_model: The embedding model
        model_name (str): Name of the embedding model
        
    Returns:
        Tuple[str, bool]: (collection_name, was_cached)
    """
    from components.ingest_engine import ingest_from_url
    from components.chunking import semantic_chunk_texts
    from components.embeddings import create_embeddings
    
    cache_check_start = time.time()
    
    # Step 1: Generate file hash immediately after download
    file_hash = generate_file_hash(file_path)
    print(f"🔍 Generated file hash: {file_hash[:16]}...")
    
    # Step 2: Check if document is already cached
    cached_collection_name = await find_cached_document(weaviate_client, file_hash)
    
    if cached_collection_name:
        cache_hit_time = time.time() - cache_check_start
        print(f"🎯 Document found in cache! Using collection: {cached_collection_name}")
        
        log_service_event("document_cache_used", "Using cached document - skipping all processing", {
            "file_hash": file_hash,
            "collection_name": cached_collection_name,
            "source_url": original_source_url,
            "cache_check_time_seconds": cache_hit_time
        })
        
        return cached_collection_name, True
    
    else:
        cache_miss_time = time.time() - cache_check_start
        print(f"💾 Document not cached. Processing and storing...")
        
        log_service_event("document_cache_miss", "Document not cached - proceeding with full processing", {
            "file_hash": file_hash,
            "source_url": original_source_url,
            "cache_check_time_seconds": cache_miss_time
        })
        
        # Step 3: Process document (parse, chunk, embed) - only on cache miss
        processing_start = time.time()
        
        # Parse document
        chunks, _ = ingest_from_url(file_path, _source_url=original_source_url)
        print(f"📝 Parsed {len(chunks)} initial text chunks.")
        
        if not chunks:
            raise ValueError("No content extracted from document")
        
        # Semantic chunking
        chunks = semantic_chunk_texts(
            chunks,
            embedding_model=embedding_model,
            model_name=model_name,
            similarity_threshold=0.8,
            min_chunk_size=3,
            max_chunk_size=12
        )
        print(f"🧩 After semantic chunking: {len(chunks)} chunks.")
        
        if not chunks:
            raise ValueError("No content after semantic chunking")
        
        # Generate embeddings
        embeddings = create_embeddings(chunks, embedding_model, model_name)
        print(f"🔢 Generated embeddings for {len(chunks)} chunks.")
        
        processing_time = time.time() - processing_start
        
        # Step 4: Store in cache
        collection_name = create_collection_name_from_hash(file_hash)
        
        weaviate_client = await create_cached_collection(
            weaviate_client,
            collection_name,
            chunks,
            embeddings,
            file_hash,
            original_source_url
        )
        
        total_time = time.time() - cache_check_start
        
        log_service_event("document_processed_and_cached", "Document processed and stored in cache", {
            "file_hash": file_hash,
            "collection_name": collection_name,
            "source_url": original_source_url,
            "chunks_count": len(chunks),
            "processing_time_seconds": processing_time,
            "total_time_seconds": total_time
        })
        
        return collection_name, False
