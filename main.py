import os
import time
import uuid
import asyncio
import random
import json  # --- Added for pretty printing JSON ---
from typing import List, Dict, Any
from contextlib import asynccontextmanager

import weaviate
import weaviate.classes.config as wvc
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

import requests
import httpx
from urllib.parse import urlparse
from tqdm import tqdm

from fastapi import FastAPI, HTTPException, Body, Header, Depends, Security, HTTPException, APIRouter
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import uvicorn
from requests_toolbelt.multipart.encoder import MultipartEncoder, MultipartEncoderMonitor

# Import custom logger module
from logger import (
    logger, 
    setup_file_logging, 
    log_api_request, 
    log_api_response, 
    log_error, 
    log_service_event
)

# --- 1. Configuration ---
load_dotenv()
# ... (Configuration section is unchanged) ...
WEAVIATE_HOST = os.getenv("WEAVIATE_HOST", "127.0.0.1")
WEAVIATE_PORT = int(os.getenv("WEAVIATE_PORT", 8080))
WEAVIATE_GRPC_PORT = int(os.getenv("WEAVIATE_GRPC_PORT", 50051))
AUTH_TOKEN = "fd8defb3118175da9553e106c05f40bc476971f0b46a400db1e625eaffa1fc08"
TOP_K_RESULTS = 5
CACHE_DIR = "./huggingface_cache"
MODEL_NAME = 'BAAI/bge-base-en-v1.5'
GEMINI_MODEL_NAME = 'gemini-2.5-flash'
GEMINI_API_KEYS_STR = os.getenv("GEMINI_API_KEYS")
if not GEMINI_API_KEYS_STR:
    raise ValueError("GEMINI_API_KEYS not found in .env file.")
GEMINI_API_KEY_LIST = [key.strip()
                       for key in GEMINI_API_KEYS_STR.split(',') if key.strip()]
if not GEMINI_API_KEY_LIST:
    raise ValueError("GEMINI_API_KEYS variable is empty or invalid.")
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL_NAME}:generateContent"
PARSING_API_URL = "http://127.0.0.1:5010/api/parseDocument?renderFormat=all"

PROMPT_TEMPLATE = """
You are a meticulous document analyst. Your task is to answer the user’s question with precision and clarity, using only the information in the provided CONTEXT block.

---
RULES:
1. Base your answer exclusively on the information within the “--- CONTEXT ---” block. Do not use any external knowledge or assumptions.
2. If the context does not contain the answer, reply exactly: “Based on the provided documents, I cannot find a definitive answer to this question.”
3. When present in the context, include specific data points such as dates, durations, quantities, monetary amounts, percentages, definitions, conditions, eligibility criteria, exceptions, or exclusions.
4. Synthesize a single, standalone paragraph without line breaks or tabs. Keep it concise and self-contained.
5. If you can only answer part of the question, state what you can answer and specify which part is unresolved due to missing context.

---
Example:
--- CONTEXT ---
Context 1: A Hospital is an institution with at least 15 inpatient beds in towns with a population over one million.
Context 2: It must maintain qualified nursing staff 24/7 and daily patient records.
--- END CONTEXT ---
--- QUESTION ---
How is “Hospital” defined?
--- END QUESTION ---
**Answer:** A Hospital is defined as an institution with at least 15 inpatient beds in towns with over one million residents, qualified nursing staff available 24/7, and daily patient record maintenance.

---
--- CONTEXT ---
{context}
--- END CONTEXT ---
--- QUESTION ---
{question}
--- END QUESTION ---
**Answer:**
"""

ml_models = {}

# --- 2. FastAPI Lifespan for Model Pre-loading ---


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Application starting up...")
    # Set up file logging at application startup
    setup_file_logging()
    log_service_event("startup", "Application starting up")
    
    print(f"🤖 Loading embedding model '{MODEL_NAME}'...")
    log_service_event("model_loading", f"Loading embedding model: {MODEL_NAME}")
    ml_models['embedding_model'] = SentenceTransformer(
        MODEL_NAME, device='cpu', cache_folder=CACHE_DIR)
    print("✅ Embedding model loaded.")
    log_service_event("model_loaded", f"Embedding model loaded: {MODEL_NAME}")
    
    yield
    
    print("👋 Application shutting down...")
    log_service_event("shutdown", "Application shutting down")
    ml_models.clear()

app = FastAPI(
    title="Optimized Document RAG API (with Debugging)",
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
    # results: List[Answer] = None
    # processing_time_seconds: float = None

# --- 3. Core Logic Functions ---

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

async def connect_to_weaviate() -> weaviate.WeaviateClient:
    """
    Connects to Weaviate with a retry mechanism.
    
    Returns:
        weaviate.WeaviateClient: An initialized Weaviate client.
        
    Raises:
        ConnectionError: If unable to connect to Weaviate after retries.
    """
    print(f"\n🔗 Connecting to Weaviate at {WEAVIATE_HOST}:{WEAVIATE_PORT}...")
    log_service_event("connection_attempt", f"Connecting to Weaviate at {WEAVIATE_HOST}:{WEAVIATE_PORT}")
    
    for i in range(5):  # 5 retries
        try:
            client = weaviate.connect_to_local(
                host=WEAVIATE_HOST,
                port=WEAVIATE_PORT,
                grpc_port=WEAVIATE_GRPC_PORT,
            )
            if client.is_ready():
                print("✅ Weaviate is ready!")
                log_service_event("connection_success", "Weaviate connection established")
                return client
            client.close()
            await asyncio.sleep(3)
        except Exception as e:
            error_msg = f"⏳ Weaviate not ready, retrying... (Attempt {i+1}/5). Error: {e}"
            print(error_msg)
            log_error("Weaviate connection failed", {"attempt": i+1, "error": str(e)})
            await asyncio.sleep(3)
    
    error_msg = "❌ Could not connect to Weaviate after multiple retries."
    log_error("Weaviate connection failed permanently")
    raise ConnectionError(error_msg)


async def fetch_and_parse_pdf(api_url: str, file_url: str) -> List[str]:
    """
    Downloads a PDF and sends it to a parsing API to get text chunks.
    
    Parameters:
        api_url (str): URL of the parsing API.
        file_url (str): URL of the PDF file to download.
        
    Returns:
        List[str]: List of text chunks extracted from the PDF.
        
    Raises:
        HTTPException: If download fails, parsing fails, or no text chunks are found.
    """
    print(f"\n⬇️  Downloading file from: {file_url}")
    log_service_event("download_start", f"Downloading document", {"url": file_url})
    
    try:
        pdf_response = requests.get(file_url)
        pdf_response.raise_for_status()
        log_service_event("download_complete", f"Document downloaded successfully", {"size_bytes": len(pdf_response.content)})
    except requests.RequestException as e:
        error_msg = f"Failed to download document from URL: {e}"
        log_error("document_download_failed", {"url": file_url, "error": str(e)})
        raise HTTPException(status_code=400, detail=error_msg)

    filename = os.path.basename(urlparse(file_url).path)
    encoder = MultipartEncoder(
        fields={'file': (filename, pdf_response.content, 'application/pdf')})

    print(f"➡️  Sending '{filename}' to parsing API at {api_url}...")
    log_service_event("parsing_start", f"Sending document to parsing API", {"filename": filename, "api": api_url})
    
    try:
        response = requests.post(api_url, data=encoder, headers={
                                 'Content-Type': encoder.content_type})
        response.raise_for_status()
        log_service_event("parsing_response_received", "Received response from parsing API", {"status_code": response.status_code})
    except requests.RequestException as e:
        error_msg = f"Document parsing service failed: {e}"
        log_error("parsing_failed", {"api": api_url, "error": str(e)})
        raise HTTPException(status_code=503, detail=error_msg)

    # Log parsed response structure (without full content)
    response_json = response.json()
    log_service_event("parsing_result_structure", "Structure of parsing result", {
        "has_return_dict": "return_dict" in response_json,
        "has_result": "result" in response_json.get("return_dict", {}),
        "has_blocks": "blocks" in response_json.get("return_dict", {}).get("result", {})
    })
    
    blocks = response_json.get('return_dict', {}).get('result', {}).get('blocks', [])
    if not blocks:
        error_msg = "No text chunks were found in the parsed document."
        log_error("no_text_chunks", {"filename": filename})
        raise HTTPException(status_code=422, detail=error_msg)

    chunk_texts = [" ".join(block.get('sentences', [])) for block in blocks]
    print(f"✅ Parsed document into {len(chunk_texts)} text chunks.")
    log_service_event("parsing_complete", f"Document parsed successfully", {
        "chunks_count": len(chunk_texts),
        "avg_chunk_length": sum(len(chunk) for chunk in chunk_texts) / len(chunk_texts) if chunk_texts else 0
    })
    return chunk_texts


async def ingest_to_weaviate(client: weaviate.WeaviateClient, collection_name: str, chunks: List[str]):
    """
    Generate embeddings for text chunks and ingest them into Weaviate.
    
    Parameters:
        client (weaviate.WeaviateClient): The Weaviate client.
        collection_name (str): Name of the collection to create.
        chunks (List[str]): List of text chunks to embed and ingest.
    """
    ingest_start_time = time.time()
    
    # Generate embeddings
    model = ml_models['embedding_model']
    print(f"\n🧠 Generating embeddings for {len(chunks)} chunks...")
    log_service_event("embedding_generation_start", "Starting chunk embedding generation", {
        "chunks_count": len(chunks),
        "collection_name": collection_name
    })
    
    embeddings = model.encode(chunks, show_progress_bar=True)
    embedding_time = time.time() - ingest_start_time
    
    # Log embedding generation statistics
    log_service_event("embedding_generation_complete", "Completed chunk embedding generation", {
        "chunks_count": len(chunks),
        "embedding_dimension": embeddings.shape[1],
        "time_seconds": embedding_time
    })

    # --- DEBUG PRINT 1: EMBEDDINGS RECEIVED ---
    print("\n" + "="*50)
    print("1. EMBEDDINGS RECEIVED")
    print("="*50)
    print(f"Shape of embeddings array: {embeddings.shape}")
    print("="*50 + "\n")
    # ----------------------------------------

    # Create collection
    collection_start_time = time.time()
    if client.collections.exists(collection_name):
        client.collections.delete(collection_name)
        log_service_event("collection_deleted", f"Deleted existing collection: {collection_name}")
    
    client.collections.create(
        name=collection_name,
        vectorizer_config=wvc.Configure.Vectorizer.none(),
        properties=[wvc.Property(name="content", data_type=wvc.DataType.TEXT)]
    )
    print(f"✅ Created Weaviate collection: '{collection_name}'")
    log_service_event("collection_created", f"Created Weaviate collection", {
        "collection_name": collection_name,
        "time_seconds": time.time() - collection_start_time
    })
    
    # Ingest data
    ingest_start_time = time.time()
    policy_collection = client.collections.get(collection_name)
    print(f"🚀 Pushing {len(chunks)} objects to Weaviate...")
    log_service_event("batch_insertion_start", "Starting batch insertion to Weaviate", {
        "objects_count": len(chunks),
        "collection_name": collection_name
    })
    
    with policy_collection.batch.dynamic() as batch:
        for i, text in enumerate(tqdm(chunks, desc="Batching objects")):
            batch.add_object(
                properties={"content": text},
                uuid=uuid.uuid4(),
                vector=embeddings[i].tolist()
            )
            
    failed_count = len(policy_collection.batch.failed_objects)
    if failed_count > 0:
        print(f"⚠️ Failed to push {failed_count} objects.")
        log_error("batch_insertion_failures", {
            "failed_objects_count": failed_count,
            "collection_name": collection_name
        })
    
    ingest_time = time.time() - ingest_start_time
    print(f"✅ Pushed {len(chunks)} objects to Weaviate successfully.")
    log_service_event("batch_insertion_complete", "Completed batch insertion to Weaviate", {
        "successful_objects_count": len(chunks) - failed_count,
        "failed_objects_count": failed_count,
        "collection_name": collection_name,
        "time_seconds": ingest_time
    })


async def generate_gemini_response_httpx(session: httpx.AsyncClient, prompt: str) -> str:
    """
    Generate a response using the Gemini API.
    
    Parameters:
        session (httpx.AsyncClient): HTTP client session for making requests.
        prompt (str): The prompt to send to the Gemini API.
        
    Returns:
        str: The generated text response from Gemini.
    """
    start_time = time.time()
    api_key = random.choice(GEMINI_API_KEY_LIST)
    headers = {'Content-Type': 'application/json'}
    payload = {'contents': [{'parts': [{'text': prompt}]}]}

    # Log Gemini request (excluding the full prompt for brevity)
    log_service_event("gemini_request", "Sending request to Gemini API", {
        "prompt_length": len(prompt),
        "model": GEMINI_MODEL_NAME
    })

    # --- DEBUG PRINT 2: PAYLOAD GIVEN ---
    print("\n" + "="*50)
    print("2. PAYLOAD GIVEN TO GEMINI")
    print("="*50)
    # Use json.dumps for pretty printing the dictionary
    print(json.dumps(payload, indent=2))
    print("="*50 + "\n")
    # ------------------------------------

    try:
        response = await session.post(f"{GEMINI_API_URL}?key={api_key}", json=payload, headers=headers, timeout=90)
        duration = time.time() - start_time

        # Log response timing
        log_service_event("gemini_response_received", "Received response from Gemini API", {
            "status_code": response.status_code,
            "duration_seconds": duration
        })

        # --- DEBUG PRINT 3: OUTPUT RECEIVED ---
        print("\n" + "="*50)
        print("3. OUTPUT RECEIVED FROM GEMINI")
        print("="*50)
        print(f"Status Code: {response.status_code}")
        # Try to pretty print if it's JSON, otherwise print as text
        try:
            print(json.dumps(response.json(), indent=2))
        except json.JSONDecodeError:
            print(response.text)
        print("="*50 + "\n")
        # --------------------------------------

        response.raise_for_status()
        data = response.json()
        response_text = data['candidates'][0]['content']['parts'][0]['text']
        
        # Log successful response (only first 100 chars of text for brevity)
        log_service_event("gemini_response_success", "Successfully processed Gemini response", {
            "response_length": len(response_text),
            "response_preview": response_text[:100] + ("..." if len(response_text) > 100 else ""),
            "duration_seconds": duration
        })
        
        return response_text
    except httpx.HTTPStatusError as e:
        error_msg = f"Gemini API error: {e.response.status_code} - {e.response.text}"
        print(f"❌ {error_msg}")
        log_error("gemini_api_error", {
            "status_code": e.response.status_code,
            "response_text": e.response.text,
            "duration_seconds": time.time() - start_time
        })
        return error_msg
    except (KeyError, IndexError) as e:
        error_msg = f"Error parsing Gemini response: {e}. Response: {response.text}"
        print(f"❌ {error_msg}")
        log_error("gemini_response_parse_error", {
            "error": str(e),
            "response_text": response.text if hasattr(response, 'text') else None,
            "duration_seconds": time.time() - start_time
        })
        return error_msg
    except Exception as e:
        error_msg = f"An unexpected error occurred during Gemini call: {e}"
        print(f"❌ {error_msg}")
        log_error("gemini_unexpected_error", {
            "error": str(e),
            "duration_seconds": time.time() - start_time
        })
        return error_msg


async def process_single_question(
    question: str,
    collection_name: str,
    weaviate_client: weaviate.WeaviateClient,
    http_session: httpx.AsyncClient
) -> Answer:
    """
    Process a single question using RAG pipeline.
    
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
    query_vector = embedding_model.encode(question).tolist()
    log_service_event("embedding_generated", "Question embedding generated", {
        "question_id": question_id,
        "vector_dimension": len(query_vector)
    })
    
    # Query Weaviate for relevant chunks
    policy_collection = weaviate_client.collections.get(collection_name)
    response = policy_collection.query.near_vector(
        near_vector=query_vector, limit=TOP_K_RESULTS, return_properties=["content"]
    )
    context_chunks = [obj.properties['content'] for obj in response.objects]
    
    log_service_event("context_retrieved", "Retrieved context chunks from Weaviate", {
        "question_id": question_id,
        "chunks_count": len(context_chunks),
        "chunks_avg_length": sum(len(chunk) for chunk in context_chunks) / len(context_chunks) if context_chunks else 0
    })
    
    if not context_chunks:
        answer_text = "Based on the provided documents, I cannot find any relevant information to answer this question."
        log_service_event("empty_context", "No context chunks found for question", {"question_id": question_id})
    else:
        context_str = "\n\n---\n\n".join(context_chunks)
        prompt = PROMPT_TEMPLATE.format(context=context_str, question=question)
        answer_text = await generate_gemini_response_httpx(http_session, prompt)
    
    processing_time = time.time() - start_time
    
    log_service_event("question_answered", "Question processing completed", {
        "question_id": question_id,
        "processing_time_seconds": processing_time,
        "answer_length": len(answer_text),
        "answer_preview": answer_text[:100] + ("..." if len(answer_text) > 100 else "")
    })
    
    return Answer(question=question, answer=answer_text)

# --- 4. FastAPI Endpoint ---


@router.post("/hackrx/run", response_model=QueryResponse, tags=["RAG Pipeline"])
async def process_document_and_answer_questions(
    request: QueryRequest = Body(...),
    token: str = Depends(verify_token)
):
    """
    Process a document and answer questions using RAG.
    
    This endpoint takes a document URL and a list of questions, processes the document
    using a parsing API, stores the content in Weaviate, and generates answers using
    the Gemini API.
    
    Parameters:
        request (QueryRequest): Request body containing document URL and questions.
        token (str): Authentication token from HTTP Bearer.
        
    Returns:
        QueryResponse: Object containing answers to the questions.
        
    Raises:
        HTTPException: For various errors during processing.
    """
    start_time = time.time()
    weaviate_client = None
    collection_name = f"Doc_{uuid.uuid4().hex}"
    request_id = str(uuid.uuid4())

    print("\n" + "="*50)
    print("✓ TOKEN VERIFICATION SUCCESSFUL")
    print("🚀 STARTING DOCUMENT PROCESSING")
    print(f"📄 Document URL: {request.documents}")
    print(f"❓ Number of questions: {len(request.questions)}")
    print("="*50 + "\n")
    
    # Log API request
    log_api_request("/api/v1/hackrx/run", {
        "request_id": request_id,
        "document_url": request.documents,
        "questions_count": len(request.questions),
        "questions": request.questions
    })
    
    try:
        weaviate_client = await connect_to_weaviate()
        chunks = await fetch_and_parse_pdf(PARSING_API_URL, request.documents)
        await ingest_to_weaviate(weaviate_client, collection_name, chunks)
        
        # Log successful ingestion
        log_service_event("document_ingested", "Document successfully ingested to Weaviate", {
            "request_id": request_id,
            "collection_name": collection_name,
            "chunks_count": len(chunks)
        })
        
        async with httpx.AsyncClient() as http_session:
            tasks = [
                process_single_question(
                    q, collection_name, weaviate_client, http_session)
                for q in request.questions
            ]
            final_answers = await asyncio.gather(*tasks)
        
        end_time = time.time()
        processing_time = end_time - start_time
        answer_texts = [a.answer for a in final_answers]

        print("\n" + "="*50)
        print("✅ PROCESSING COMPLETED SUCCESSFULLY")
        print(f"⏱️  Total processing time: {processing_time:.2f} seconds")
        print(f"📊 Number of questions processed: {len(final_answers)}")
        print("="*50 + "\n")

        # Create response object
        response = QueryResponse(answers=answer_texts)
        
        # Log API response
        log_api_response("/api/v1/hackrx/run", {
            "request_id": request_id,
            "answers_count": len(answer_texts),
            "answers": [a[:100] + ("..." if len(a) > 100 else "") for a in answer_texts],  # Log previews only for brevity
            "processing_time_seconds": processing_time
        }, duration=processing_time)
        
        # Just return answers as requested
        return response
    except Exception as e:
        error_msg = f"A critical error occurred: {e}"
        print(f"❌ {error_msg}")
        
        # Log error
        log_error("critical_processing_error", {
            "request_id": request_id,
            "error": str(e),
            "document_url": request.documents,
            "questions_count": len(request.questions),
            "processing_time_seconds": time.time() - start_time
        })
        
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if weaviate_client and weaviate_client.is_connected():
            if weaviate_client.collections.exists(collection_name):
                weaviate_client.collections.delete(collection_name)
                print(f"\n✅ Cleaned up and deleted collection: '{collection_name}'")
                log_service_event("collection_deleted", "Cleaned up Weaviate collection", {
                    "request_id": request_id,
                    "collection_name": collection_name
                })
            weaviate_client.close()
            print("✅ Weaviate client connection closed.")
            log_service_event("connection_closed", "Weaviate client connection closed", {"request_id": request_id})

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
