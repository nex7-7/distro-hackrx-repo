"""
Main RAG API Module

This module is the entry point for the RAG API system.
"""
import os
import uuid
import time
import json
import nltk
import httpx
import weaviate
import weaviate.exceptions
import uvicorn
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager
from sentence_transformers import SentenceTransformer, CrossEncoder
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from tqdm import tqdm
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Body, Header, Depends, Security, HTTPException, APIRouter
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# Import custom logger module
from components.utils.logger import (
    setup_file_logging,
    log_api_request,
    log_api_response,
    log_error,
    log_service_event,
)

# Import components
# legacy (kept for compatibility)
from components.ingest_engine import ingest_from_url, download_and_check_file
from components.chunking import semantic_chunk_texts
from components.embeddings import create_embeddings, create_query_embedding
from components.weaviate_db import connect_to_weaviate, ingest_to_weaviate
from components.search import hybrid_search
from components.prompt_template import create_prompt
from components.gemini_api import (
    generate_gemini_response_httpx,
    generate_mistral_response_httpx,
    generate_github_models_response_httpx
)
from components.reranker_utils import diagnose_reranker_model
from components.riddle_solver import solve_riddle
from components.document_cache import (
    generate_file_hash,
    create_collection_name_from_hash,
    find_cached_document,
    create_cached_collection,
    get_cached_document_stats,
    process_document_with_cache
)

nltk.data.find('tokenizers/punkt')
nltk.download('punkt_tab')
nltk.download('punk')


# Load environment variables
load_dotenv()
WEAVIATE_HOST = os.getenv("WEAVIATE_HOST", "127.0.0.1")
WEAVIATE_PORT = int(os.getenv("WEAVIATE_PORT", 8080))
WEAVIATE_GRPC_PORT = int(os.getenv("WEAVIATE_GRPC_PORT", 50051))
AUTH_TOKEN = os.getenv(
    "AUTH_TOKEN", "fd8defb3118175da9553e106c05f40bc476971f0b46a400db1e625eaffa1fc08")

# RAG pipeline configuration
INITIAL_RETRIEVAL_K = 25  # Fetch more candidates for reranking
TOP_K_RESULTS = 15  # Keep top 15 results after reranking for LLM
RERANKER_THRESHOLD = 0.3  # Minimum score threshold for chunks after reranking
CACHE_DIR = "./huggingface_cache"
MODEL_NAME = os.getenv("EMBEDDING_MODEL", 'BAAI/bge-base-en-v1.5')
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", 'gemini-2.5-flash')

# Special PDF trigger URL for riddle solver diversion
SPECIAL_PDF_URL = (
    "https://hackrx.blob.core.windows.net/hackrx/rounds/FinalRound4SubmissionPDF.pdf?sv=2023-01-03&spr=https&st=2025-08-07T14%3A23%3A48Z&se=2027-08-08T14%3A23%3A00Z&sr=b&sp=r&sig=nMtZ2x9aBvz%2FPjRWboEOZIGB%2FaGfNf5TfBOrhGqSv4M%3D"
)

# Verify Gemini API keys
GEMINI_API_KEYS_STR = os.getenv("GEMINI_API_KEYS")
if not GEMINI_API_KEYS_STR:
    raise ValueError("GEMINI_API_KEYS not found in .env file.")
GEMINI_API_KEY_LIST = [key.strip()
                       for key in GEMINI_API_KEYS_STR.split(',') if key.strip()]
if not GEMINI_API_KEY_LIST:
    raise ValueError("GEMINI_API_KEYS variable is empty or invalid.")

# Set API URLs
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL_NAME}:generateContent"
PARSING_API_URL = os.getenv(
    "PARSING_API_URL", "http://127.0.0.1:5010/api/parseDocument?renderFormat=all")

# Load reranker model name
RERANKER_MODEL_NAME = os.getenv(
    "RERANKER_MODEL", 'BAAI/bge-reranker-base')

# Global dictionary to store ML models and clients
ml_models = {}
weaviate_client = None

# --- FastAPI Lifespan for Model Pre-loading ---


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Application starting up...")
    # Set up file logging at application startup
    setup_file_logging()
    log_service_event("startup", "Application starting up")

    print(f"🤖 Loading embedding model '{MODEL_NAME}'...")
    log_service_event(
        "model_loading", f"Loading embedding model: {MODEL_NAME}")
    ml_models['embedding_model'] = SentenceTransformer(
        MODEL_NAME, device='cuda', cache_folder=CACHE_DIR)
    print("✅ Embedding model loaded.")
    log_service_event("model_loaded", f"Embedding model loaded: {MODEL_NAME}")

    # Load reranker model
    print(f"🤖 Loading reranker model '{RERANKER_MODEL_NAME}'...")
    log_service_event(
        "reranker_loading", f"Loading reranker model: {RERANKER_MODEL_NAME}")
    from sentence_transformers import CrossEncoder
    ml_models['reranker_model'] = CrossEncoder(
        RERANKER_MODEL_NAME, device='cuda')
    print("✅ Reranker model loaded.")
    log_service_event("reranker_loaded",
                      f"Reranker model loaded: {RERANKER_MODEL_NAME}")

    # Connect to Weaviate once during startup
    global weaviate_client
    try:
        print(
            f"🔗 Connecting to Weaviate at {WEAVIATE_HOST}:{WEAVIATE_PORT}...")
        log_service_event("weaviate_connection_attempt",
                          "Connecting to Weaviate during startup")
        weaviate_client = weaviate.WeaviateClient(
            connection_params=weaviate.connect.ConnectionParams.from_params(
                http_host=WEAVIATE_HOST,
                http_port=WEAVIATE_PORT,
                http_secure=False,
                grpc_host=WEAVIATE_HOST,
                grpc_port=WEAVIATE_GRPC_PORT,
                grpc_secure=False,
            )
        )
        weaviate_client.connect()
        weaviate_client.is_ready()
        print("✅ Connected to Weaviate successfully!")
        log_service_event("weaviate_connection_success",
                          "Successfully connected to Weaviate during startup")
    except Exception as e:
        log_error("weaviate_connection_failed", {
            "error": str(e),
            "host": WEAVIATE_HOST,
            "port": WEAVIATE_PORT
        })
        print(f"❌ Failed to connect to Weaviate during startup: {e}")
        print("⚠️ Will try to connect again during request processing")
        weaviate_client = None

    yield

    print("👋 Application shutting down...")
    # Close Weaviate client if it exists
    if weaviate_client is not None:
        try:
            weaviate_client.close()
            print("✅ Closed Weaviate connection")
        except Exception as e:
            print(f"⚠️ Error closing Weaviate connection: {e}")

    log_service_event("shutdown", "Application shutting down")
    ml_models.clear()

app = FastAPI(
    title="Main RAG API (with Debugging)",
    description="A high-performance API that prints key data points during processing.",
    version="2.1.0",
    lifespan=lifespan
)

# Create API router with /api/v1 prefix
router = APIRouter(prefix="/api/v1")

security = HTTPBearer()


# --- Pydantic Models ---

class QueryRequest(BaseModel):
    documents: str = Field(..., example="https://some-url.com/document.pdf")
    questions: List[str] = Field(..., example=["What is the grace period?"])


class Answer(BaseModel):
    question: str
    answer: str


class QueryResponse(BaseModel):
    answers: List[str]


class LLMTestRequest(BaseModel):
    prompt: str = Field(..., example="What is the capital of France?")
    questions: List[str] = Field(default_factory=list, example=[
                                 "What is the capital of France?"])


class LLMTestResponse(BaseModel):
    response: str


class ClearWeaviateResponse(BaseModel):
    message: str


class CacheStatsResponse(BaseModel):
    unique_documents: int
    total_chunks: int
    oldest_cache: Optional[str]
    newest_cache: Optional[str]
    cache_collections: List[str]

# --- Authentication Function ---


def _is_url(path_or_url: str) -> bool:
    """Check if a string is a URL."""
    parsed = urlparse(path_or_url)
    return bool(parsed.scheme and parsed.netloc)


def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Verify that the provided token matches the expected token."""
    if credentials.credentials != AUTH_TOKEN:
        print(f"❌ Invalid authorization token: {credentials.credentials}")
        raise HTTPException(
            status_code=401,
            detail="Invalid authorization token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials


async def process_single_question(
    question: str,
    collection_name: str,
    weaviate_client: weaviate.WeaviateClient,
    http_session: httpx.AsyncClient
) -> Answer:
    """
    Process a single question using RAG pipeline with reranking.

    Parameters:
        question (str): The question to answer.
        collection_name (str): Name of the Weaviate collection to query.
        weaviate_client (weaviate.WeaviateClient): Client for Weaviate operations.
        http_session (httpx.AsyncClient): HTTP client for external API calls.

    Returns:
        Answer: An Answer object containing the question and generated answer.
    """
    question_id = str(uuid.uuid4())
    start_time = time.time()
    print(f"🔍 Processing question: '{question}'")
    log_service_event("question_processing", "Processing question", {
        "question_id": question_id,
        "question": question,
        "collection_name": collection_name
    })

    # Generate embedding for the question
    embedding_model = ml_models['embedding_model']
    query_vector = create_query_embedding(
        question, embedding_model, MODEL_NAME)

    # Perform hybrid search in Weaviate
    try:
        context_chunks, chunk_ids, chunk_scores = hybrid_search(
            question=question,
            query_vector=query_vector,
            collection_name=collection_name,
            weaviate_client=weaviate_client,
            limit=INITIAL_RETRIEVAL_K,  # Get 25 chunks for reranking
            alpha=0.5  # Balance between vector and keyword search
        )
    except ConnectionError as e:
        # Try to reconnect and retry
        log_service_event("search_connection_retry", "Attempting to reconnect and retry search", {
            "question_id": question_id,
            "collection_name": collection_name
        })

        # Reconnect to Weaviate
        global WEAVIATE_HOST, WEAVIATE_PORT, WEAVIATE_GRPC_PORT
        try:
            weaviate_client = await connect_to_weaviate(WEAVIATE_HOST, WEAVIATE_PORT, WEAVIATE_GRPC_PORT)

            # Retry the hybrid search
            context_chunks, chunk_ids, chunk_scores = hybrid_search(
                question=question,
                query_vector=query_vector,
                collection_name=collection_name,
                weaviate_client=weaviate_client,
                limit=INITIAL_RETRIEVAL_K,  # Get 25 chunks for reranking
                alpha=0.5
            )
        except Exception as retry_error:
            log_error("search_retry_failed", {
                "question_id": question_id,
                "error": str(retry_error)
            })
            context_chunks, chunk_ids, chunk_scores = [], [], []

    # Rerank chunks if we have results
    if context_chunks and len(context_chunks) > TOP_K_RESULTS:
        reranker_start = time.time()
        reranker_model = ml_models['reranker_model']

        # Create query-chunk pairs for reranking
        query_chunk_pairs = [[question, chunk] for chunk in context_chunks]

        # Get reranking scores
        rerank_scores = reranker_model.predict(query_chunk_pairs)

        # Sort by reranking scores and take top K
        ranked_indices = sorted(range(len(rerank_scores)),
                                key=lambda i: rerank_scores[i], reverse=True)
        top_indices = ranked_indices[:TOP_K_RESULTS]

        # Reorder chunks based on reranking
        context_chunks = [context_chunks[i] for i in top_indices]
        chunk_ids = [chunk_ids[i] for i in top_indices] if chunk_ids else []
        chunk_scores = [float(rerank_scores[i]) for i in top_indices]

        reranker_time = time.time() - reranker_start
        log_service_event("reranking_complete", "Chunks reranked successfully", {
            "question_id": question_id,
            "initial_chunks": len(query_chunk_pairs),
            "final_chunks": len(context_chunks),
            "reranking_time_seconds": reranker_time,
            "top_rerank_score": max(chunk_scores) if chunk_scores else 0,
            "avg_rerank_score": sum(chunk_scores) / len(chunk_scores) if chunk_scores else 0
        })
        print(
            f"🔄 Reranked {len(query_chunk_pairs)} chunks → top {len(context_chunks)} selected")

    print(
        f"🔍 Using top {len(context_chunks)} chunks for LLM context")

    if not context_chunks:
        answer_text = "Based on the provided documents, I cannot find any relevant information to answer this question."
        log_service_event("empty_context", "No context chunks found for question", {
            "question_id": question_id,
            "rag_pipeline_failure": True,
            "failure_reason": "No context chunks retrieved from vector store"
        })
    else:
        # Create prompt with formatted context
        context_formatting_start = time.time()
        prompt = create_prompt(question, context_chunks, chunk_scores)
        context_formatting_time = time.time() - context_formatting_start

        # Log context preparation metrics
        log_service_event("context_formatted", "Context formatted for LLM prompt", {
            "question_id": question_id,
            "prompt_length": len(prompt),
            "context_count": len(context_chunks),
            "formatting_time_seconds": context_formatting_time
        })

        # Generate answer using Gemini API
        answer_text = await generate_gemini_response_httpx(http_session, prompt)

    processing_time = time.time() - start_time
    log_service_event("question_answered", "Question processing completed", {
        "question_id": question_id,
        "processing_time_seconds": processing_time,
        "answer_length": len(answer_text),
        "answer": answer_text
    })
    return Answer(question=question, answer=answer_text)


# --- API Endpoints ---

@router.post("/hackrx/run", response_model=QueryResponse, tags=["RAG Pipeline"])
async def process_document_and_answer_questions(
    request: QueryRequest = Body(...),
    token: str = Depends(verify_token)
):
    """
    Process a document and answer questions using the RAG pipeline.

    This endpoint:
    1. Downloads and processes the document from the provided URL
    2. Splits the document into chunks
    3. Generates embeddings for the chunks
    4. Stores the chunks and embeddings in Weaviate
    5. For each question:
       - Generates an embedding for the question
       - Retrieves relevant chunks using hybrid search
       - Formats the prompt with the retrieved chunks
       - Generates an answer using Gemini API

    Parameters:
        request (QueryRequest): The request containing document URL and questions
        token (str): The authentication token

    Returns:
        QueryResponse: The response containing answers to the questions
    """
    # Create a unique ID for this request
    request_id = str(uuid.uuid4())

    # Log the API request
    log_api_request(
        endpoint="/api/v1/hackrx/run",
        request_data={
            "documents": request.documents,
            "questions_count": len(request.questions),
            "questions": request.questions,
            "request_id": request_id
        }
    )

    start_time = time.time()
    
    # Declare global weaviate_client at the start of the function
    global weaviate_client

    try:
        # Extract document URL and questions
        document_url = request.documents.strip()
        questions = request.questions

        print(f"📄 Processing document: {document_url}")
        print(f"❓ Questions to answer: {len(questions)}")

        # Create HTTP client session for making requests
        async with httpx.AsyncClient() as http_client:
            # --- Branch: Special Riddle Solver Path ---
            if document_url == SPECIAL_PDF_URL:
                log_service_event("riddle_solver_triggered", "Special PDF detected; diverting to riddle solver", {
                    "request_id": request_id,
                    "document_url": document_url,
                    "questions_count": len(questions)
                })
                try:
                    riddle_answer = await solve_riddle(http_client)
                except Exception as e:
                    log_error("riddle_solver_failure", {"error": str(e), "request_id": request_id})
                    riddle_answer = "Failed to retrieve flight number"

                answers = [riddle_answer for _ in questions]

                processing_time = time.time() - start_time
                log_api_response(
                    endpoint="/api/v1/hackrx/run",
                    response_data={
                        "answers_count": len(answers),
                        "processing_time": processing_time,
                        "request_id": request_id,
                        "status_code": 200,
                        "riddle_solver": True
                    },
                    duration=processing_time
                )
                return QueryResponse(answers=answers)
            # 1. Download and check file security first
            try:
                file_path, original_source_url, is_temp_file = download_and_check_file(document_url)
            except HTTPException as he:
                # Handle security violations and download errors
                log_error("file_download_failed", {
                    "document_url": document_url,
                    "request_id": request_id,
                    "status_code": he.status_code,
                    "detail": he.detail
                })
                print(f"❌ File download/check failed: {he.detail}")
                
                # Return "Files Not found" for security violations or processing errors
                answers = ["Incorrect file type submitted"] * len(questions)
                
                processing_time = time.time() - start_time
                log_api_response(
                    endpoint="/api/v1/hackrx/run",
                    response_data={
                        "answers_count": len(answers),
                        "processing_time": processing_time,
                        "request_id": request_id,
                        "status_code": 200,
                        "download_failed": True,
                        "failure_reason": he.detail
                    },
                    duration=processing_time
                )
                
                return QueryResponse(answers=answers)
            
            # Handle webpages separately (can't be cached by file hash)
            if file_path == original_source_url and _is_url(file_path):
                print(f"🌐 Processing webpage: {file_path}")
                
                try:
                    chunks, _ = ingest_from_url(file_path)
                    print(f"📝 Extracted {len(chunks)} chunks from webpage.")
                    
                    if not chunks:
                        answers = ["Files Not found"] * len(questions)
                        processing_time = time.time() - start_time
                        log_api_response(
                            endpoint="/api/v1/hackrx/run",
                            response_data={
                                "answers_count": len(answers),
                                "processing_time": processing_time,
                                "request_id": request_id,
                                "status_code": 200,
                                "webpage_no_content": True
                            },
                            duration=processing_time
                        )
                        return QueryResponse(answers=answers)
                    
                    # For webpages, process without caching
                    chunks = semantic_chunk_texts(
                        chunks,
                        embedding_model=ml_models['embedding_model'],
                        model_name=MODEL_NAME,
                        similarity_threshold=0.8,
                        min_chunk_size=3,
                        max_chunk_size=12
                    )
                    
                    if not chunks:
                        answers = ["Files Not found"] * len(questions)
                        processing_time = time.time() - start_time
                        log_api_response(
                            endpoint="/api/v1/hackrx/run",
                            response_data={
                                "answers_count": len(answers),
                                "processing_time": processing_time,
                                "request_id": request_id,
                                "status_code": 200,
                                "webpage_no_content_after_chunking": True
                            },
                            duration=processing_time
                        )
                        return QueryResponse(answers=answers)
                    
                    # Create temporary collection for webpage (no caching)
                    embeddings = create_embeddings(chunks, ml_models['embedding_model'], MODEL_NAME)
                    collection_name = f"Webpage_{request_id.replace('-', '')[:12]}"
                    
                    # Use global Weaviate client
                    if weaviate_client is None:
                        weaviate_client = await connect_to_weaviate(WEAVIATE_HOST, WEAVIATE_PORT, WEAVIATE_GRPC_PORT)
                    else:
                        try:
                            weaviate_client.is_ready()
                        except weaviate.exceptions.WeaviateClosedClientError:
                            weaviate_client = await connect_to_weaviate(WEAVIATE_HOST, WEAVIATE_PORT, WEAVIATE_GRPC_PORT)
                    
                    # Store in temporary collection (will be cleared later if needed)
                    weaviate_client = await ingest_to_weaviate(
                        weaviate_client,
                        collection_name,
                        chunks,
                        embeddings,
                        host=WEAVIATE_HOST,
                        port=WEAVIATE_PORT,
                        grpc_port=WEAVIATE_GRPC_PORT
                    )
                    
                    was_cached = False
                    
                except Exception as e:
                    log_error("webpage_processing_failed", {
                        "url": file_path,
                        "error": str(e)
                    })
                    answers = ["Files Not found"] * len(questions)
                    processing_time = time.time() - start_time
                    log_api_response(
                        endpoint="/api/v1/hackrx/run",
                        response_data={
                            "answers_count": len(answers),
                            "processing_time": processing_time,
                            "request_id": request_id,
                            "status_code": 200,
                            "webpage_processing_failed": True
                        },
                        duration=processing_time
                    )
                    return QueryResponse(answers=answers)
            
            else:
                # 2. Use global Weaviate client
                if weaviate_client is None:
                    weaviate_client = await connect_to_weaviate(WEAVIATE_HOST, WEAVIATE_PORT, WEAVIATE_GRPC_PORT)
                else:
                    try:
                        weaviate_client.is_ready()
                    except weaviate.exceptions.WeaviateClosedClientError:
                        weaviate_client = await connect_to_weaviate(WEAVIATE_HOST, WEAVIATE_PORT, WEAVIATE_GRPC_PORT)
                
                # 3. Process document with caching (cache check happens here)
                try:
                    collection_name, was_cached = await process_document_with_cache(
                        file_path,
                        original_source_url,
                        weaviate_client,
                        ml_models['embedding_model'],
                        MODEL_NAME
                    )
                except ValueError as ve:
                    # Handle case where document has no content
                    log_service_event("document_no_content", str(ve), {
                        "file_path": file_path,
                        "request_id": request_id
                    })
                    
                    answers = ["Files Not found"] * len(questions)
                    processing_time = time.time() - start_time
                    log_api_response(
                        endpoint="/api/v1/hackrx/run",
                        response_data={
                            "answers_count": len(answers),
                            "processing_time": processing_time,
                            "request_id": request_id,
                            "status_code": 200,
                            "document_no_content": True
                        },
                        duration=processing_time
                    )
                    
                    # Clean up temp file if needed
                    if is_temp_file and os.path.exists(file_path):
                        try:
                            os.unlink(file_path)
                        except OSError:
                            pass
                    
                    return QueryResponse(answers=answers)
                
                # Clean up temp file if needed (after processing/caching)
                if is_temp_file and os.path.exists(file_path):
                    try:
                        os.unlink(file_path)
                        log_service_event("temp_file_cleanup", "Temporary file cleaned up", {
                            "temp_path": file_path
                        })
                    except OSError as e:
                        log_error("temp_file_cleanup_failed", {
                            "temp_path": file_path,
                            "error": str(e)
                        })

            # 7. Process each question
            answers = []
            for question in questions:
                answer_obj = await process_single_question(
                    question=question,
                    collection_name=collection_name,
                    weaviate_client=weaviate_client,
                    http_session=http_client
                )
                answers.append(answer_obj.answer)

        # Calculate total processing time
        processing_time = time.time() - start_time

        # Log successful API response
        log_api_response(
            endpoint="/api/v1/hackrx/run",
            response_data={
                "answers_count": len(answers),
                "processing_time": processing_time,
                "request_id": request_id,
                "status_code": 200,
                "was_cached": was_cached,
                "cache_hit": was_cached
            },
            duration=processing_time
        )        
        # Return the answers
        return QueryResponse(answers=answers)

    except Exception as e:
        # Log error
        log_error(
            "api_error",
            {
                "error": str(e),
                "request_id": request_id,
                "endpoint": "/api/v1/hackrx/run"
            }
        )

        # Re-raise the exception to return appropriate error response
        raise


@router.post("/github-models", response_model=LLMTestResponse, tags=["LLM Testing"])
async def test_github_models_api(request: LLMTestRequest):
    """
    Test the GitHub Models API directly with a prompt and optional questions.
    """
    async with httpx.AsyncClient() as http_client:
        prompt = request.prompt
        response = await generate_github_models_response_httpx(http_client, prompt)
        return LLMTestResponse(response=response)


@router.post("/gemini", response_model=LLMTestResponse, tags=["LLM Testing"])
async def test_gemini_api(request: LLMTestRequest):
    """
    Test the Gemini API directly with a prompt and optional questions.
    """
    async with httpx.AsyncClient() as http_client:
        prompt = request.prompt
        response = await generate_gemini_response_httpx(http_client, prompt)
        return LLMTestResponse(response=response)


@router.post("/mistral", response_model=LLMTestResponse, tags=["LLM Testing"])
async def test_mistral_api(request: LLMTestRequest):
    """
    Test the Mistral API directly with a prompt and optional questions.
    """
    async with httpx.AsyncClient() as http_client:
        prompt = request.prompt
        response = await generate_mistral_response_httpx(http_client, prompt)
        return LLMTestResponse(response=response)


@router.post("/clear-weaviate", response_model=ClearWeaviateResponse, tags=["Weaviate Admin"])
async def clear_weaviate_collections():
    """
    Delete all collections in Weaviate (admin operation).
    """
    try:
        import weaviate
        client = await connect_to_weaviate(WEAVIATE_HOST, WEAVIATE_PORT, WEAVIATE_GRPC_PORT)
        schema = client.collections.list_all()
        for collection in schema:
            client.collections.delete(collection)
        client.close()
        return ClearWeaviateResponse(message="All collections deleted. Weaviate is now empty.")
    except Exception as e:
        log_error("clear_weaviate_error", {"error": str(e)})
        raise HTTPException(
            status_code=500, detail=f"Failed to clear Weaviate: {e}")


@router.get("/cache-stats", response_model=CacheStatsResponse, tags=["Document Cache"])
async def get_document_cache_statistics():
    """
    Get statistics about cached documents in Weaviate.
    """
    try:
        global weaviate_client
        if weaviate_client is None:
            weaviate_client = await connect_to_weaviate(WEAVIATE_HOST, WEAVIATE_PORT, WEAVIATE_GRPC_PORT)
        
        stats = await get_cached_document_stats(weaviate_client)
        return CacheStatsResponse(**stats)
    except Exception as e:
        log_error("cache_stats_error", {"error": str(e)})
        raise HTTPException(
            status_code=500, detail=f"Failed to get cache statistics: {e}")

# Include the router in the main app
app.include_router(router)

# --- Main Execution Block ---
if __name__ == "__main__":
    # Set up file logging before starting the application
    setup_file_logging()
    log_service_event("application_launch", "Starting application directly", {
        "host": "0.0.0.0",
        "port": 8000,
        "reload": True
    })
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
