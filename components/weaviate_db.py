"""
Weaviate Database Module

This module contains functions for interacting with Weaviate database.
"""
import time
import uuid
import weaviate
import weaviate.classes.config as wvc
import weaviate.connect
from weaviate.exceptions import WeaviateClosedClientError
from typing import List, Dict, Any
from tqdm import tqdm
from datetime import datetime

# Import local modules
from logger import log_service_event, log_error


async def connect_to_weaviate(host: str, port: int, grpc_port: int) -> weaviate.WeaviateClient:
    """
    Connects to Weaviate with a retry mechanism.

    Parameters:
        host (str): Weaviate host address
        port (int): Weaviate HTTP port
        grpc_port (int): Weaviate gRPC port

    Returns:
        weaviate.WeaviateClient: An initialized Weaviate client.

    Raises:
        ConnectionError: If unable to connect to Weaviate after retries.
    """
    print(f"\n🔗 Connecting to Weaviate at {host}:{port}...")
    log_service_event("connection_attempt",
                      f"Connecting to Weaviate at {host}:{port}")

    for i in range(5):  # 5 retries
        try:
            # Create Weaviate client using provided parameters
            client = weaviate.WeaviateClient(
                connection_params=weaviate.connect.ConnectionParams.from_params(
                    http_host=host,
                    http_port=port,
                    http_secure=False,
                    grpc_host=host,
                    grpc_port=grpc_port,
                    grpc_secure=False,
                )
            )
            client.connect()

            # Check if connection is successful
            client.is_ready()
            print("✅ Connected to Weaviate successfully!")
            log_service_event("connection_success",
                              f"Successfully connected to Weaviate")
            return client

        except Exception as e:
            retry_delay = 2 ** i  # Exponential backoff
            print(f"❌ Failed to connect to Weaviate: {e}")
            print(f"Retrying in {retry_delay} seconds...")
            log_service_event("connection_retry", f"Retrying Weaviate connection", {
                "attempt": i + 1,
                "retry_delay": retry_delay,
                "error": str(e)
            })
            time.sleep(retry_delay)

    error_msg = "❌ Could not connect to Weaviate after multiple retries."
    log_error("Weaviate connection failed permanently")
    raise ConnectionError(error_msg)


async def ingest_to_weaviate(client: weaviate.WeaviateClient, collection_name: str, chunks: List[str], embeddings: Any, host: str = None, port: int = None, grpc_port: int = None) -> weaviate.WeaviateClient:
    """
    Ingest text chunks and their embeddings into Weaviate.

    Parameters:
        client (weaviate.WeaviateClient): The Weaviate client.
        collection_name (str): Name of the collection to create.
        chunks (List[str]): List of text chunks to ingest.
        embeddings: The embeddings corresponding to the chunks.
        host (str, optional): Weaviate host address for reconnection if needed.
        port (int, optional): Weaviate HTTP port for reconnection if needed.
        grpc_port (int, optional): Weaviate gRPC port for reconnection if needed.

    Raises:
        ConnectionError: If unable to connect or reconnect to Weaviate.
    """
    ingest_id = str(uuid.uuid4())
    overall_start_time = time.time()

    # Log detailed information about the ingestion process starting
    log_service_event("ingestion_process_start", "Starting vector ingestion process", {
        "ingest_id": ingest_id,
        "collection_name": collection_name,
        "chunks_count": len(chunks),
        "total_text_size": sum(len(chunk) for chunk in chunks),
        "timestamp": datetime.now().isoformat()
    })

    # Create collection with timing
    collection_start_time = time.time()

    # Log collection creation start
    log_service_event("collection_creation_start", "Starting Weaviate collection creation", {
        "ingest_id": ingest_id,
        "collection_name": collection_name,
        "vector_dimension": embeddings.shape[1]
    })

    # Check if client is closed and reconnect if necessary
    try:
        # Attempt to check if client is ready
        client.is_ready()
    except weaviate.exceptions.WeaviateClosedClientError:
        # If client is closed and we have connection parameters, try to reconnect
        if host and port and grpc_port:
            print(
                f"⚠️ Weaviate client is closed. Attempting to reconnect to {host}:{port}...")
            log_service_event("connection_retry", "Reconnecting to Weaviate - client was closed", {
                "host": host,
                "port": port
            })
            try:
                # Create a new client using provided parameters
                client = weaviate.WeaviateClient(
                    connection_params=weaviate.connect.ConnectionParams.from_params(
                        http_host=host,
                        http_port=port,
                        http_secure=False,
                        grpc_host=host,
                        grpc_port=grpc_port,
                        grpc_secure=False,
                    )
                )
                client.connect()
                # Check if new connection is successful
                client.is_ready()
                print("✅ Reconnected to Weaviate successfully!")
                log_service_event("connection_success",
                                  "Successfully reconnected to Weaviate")
            except Exception as e:
                error_msg = f"❌ Failed to reconnect to Weaviate: {e}"
                log_error("weaviate_reconnection_failed", {"error": str(e)})
                raise ConnectionError(error_msg)
        else:
            error_msg = "❌ Weaviate client is closed and no connection parameters provided for reconnection."
            log_error("weaviate_client_closed", {"error": error_msg})
            raise ConnectionError(error_msg)

    # Check if collection exists and delete if necessary
    collection_exists = False
    try:
        if client.collections.exists(collection_name):
            collection_exists = True
            delete_start_time = time.time()
            client.collections.delete(collection_name)
            delete_time = time.time() - delete_start_time
            log_service_event("collection_deleted", f"Deleted existing collection", {
                "ingest_id": ingest_id,
                "collection_name": collection_name,
                "deletion_time_seconds": delete_time
            })
    except Exception as e:
        error_msg = f"❌ Error checking/deleting collection {collection_name}: {e}"
        log_error("collection_operation_failed", {
                  "error": str(e), "collection_name": collection_name})
        raise

    # Create new collection
    creation_start_time = time.time()
    client.collections.create(
        name=collection_name,
        vectorizer_config=wvc.Configure.Vectorizer.none(),
        properties=[wvc.Property(name="content", data_type=wvc.DataType.TEXT)]
    )
    creation_time = time.time() - creation_start_time
    total_collection_time = time.time() - collection_start_time

    print(f"✅ Created Weaviate collection: '{collection_name}'")
    log_service_event("collection_created", f"Created Weaviate collection", {
        "ingest_id": ingest_id,
        "collection_name": collection_name,
        "creation_time_seconds": creation_time,
        "total_collection_time_seconds": total_collection_time,
        "had_to_delete_existing": collection_exists
    })

    # Prepare for batch insertion
    ingest_start_time = time.time()
    policy_collection = client.collections.get(collection_name)
    print(f"🚀 Pushing {len(chunks)} objects to Weaviate...")

    # Log detailed batch insertion start
    log_service_event("batch_insertion_start", "Starting batch insertion to Weaviate", {
        "ingest_id": ingest_id,
        "objects_count": len(chunks),
        "collection_name": collection_name,
        "avg_vector_bytes": sum(len(embedding.tobytes()) for embedding in embeddings) / len(embeddings) if embeddings.shape[0] > 0 else 0,
        "total_data_size_mb": (
            sum(len(chunk) for chunk in chunks) +  # Text size
            sum(len(embedding.tobytes())
                for embedding in embeddings)  # Vector size
        ) / (1024 * 1024)  # Convert to MB
    })

    # Batch insertion with progress tracking
    batch_start_time = time.time()
    objects_added = 0
    batch_size_tracker = []

    with policy_collection.batch.dynamic() as batch:
        for i, text in enumerate(tqdm(chunks, desc="Batching objects")):
            # Each object needs a unique ID and properties
            batch.add_object(
                properties={"content": text},
                vector=embeddings[i].tolist()
            )
            objects_added += 1

            # Track batch size and timing every 100 objects
            if i > 0 and i % 100 == 0:
                batch_size_tracker.append({
                    "batch_size": 100,
                    "objects_processed": i,
                    "time": time.time() - batch_start_time
                })

    # Calculate batch processing statistics
    batch_time = time.time() - batch_start_time
    failed_count = len(policy_collection.batch.failed_objects) if hasattr(
        policy_collection.batch, 'failed_objects') else 0
    success_count = objects_added - failed_count

    # Log any failures
    if failed_count > 0:
        print(f"⚠️ Failed to push {failed_count} objects.")

        # Log detailed failure information
        failure_info = {
            "failed_objects_count": failed_count,
            "failure_percentage": (failed_count / objects_added) * 100 if objects_added > 0 else 0,
            "collection_name": collection_name
        }

        # Add sample of failed objects if available (up to 5)
        if hasattr(policy_collection.batch, 'failed_objects') and policy_collection.batch.failed_objects:
            failure_info["sample_failures"] = [
                str(obj) for obj in policy_collection.batch.failed_objects[:5]
            ]

        log_error("batch_insertion_failures", failure_info)

    # Calculate final statistics and log completion
    ingest_time = time.time() - ingest_start_time
    total_process_time = time.time() - overall_start_time
    throughput = success_count / batch_time if batch_time > 0 else 0

    print(f"✅ Pushed {success_count} objects to Weaviate successfully.")
    log_service_event("batch_insertion_complete", "Completed batch insertion to Weaviate", {
        "ingest_id": ingest_id,
        "successful_objects_count": success_count,
        "failed_objects_count": failed_count,
        "success_rate_percentage": (success_count / objects_added) * 100 if objects_added > 0 else 0,
        "collection_name": collection_name,
        "batch_time_seconds": batch_time,
        "total_ingest_time_seconds": ingest_time,
        "total_process_time_seconds": total_process_time,
        "objects_per_second": throughput,
        "batch_performance": batch_size_tracker
    })

    # Log overall ingestion process summary
    log_service_event("ingestion_process_complete", "Completed full vector ingestion process", {
        "ingest_id": ingest_id,
        "collection_name": collection_name,
        "total_chunks": len(chunks),
        "successful_chunks": success_count,
        "failed_chunks": failed_count,
        "insertion_time_seconds": ingest_time,
        "total_time_seconds": total_process_time,
        "timestamp": datetime.now().isoformat()
    })

    # Return the client (potentially reconnected)
    return client
