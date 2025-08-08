from nltk.tokenize import word_tokenize
import os
import time
import uuid
import asyncio
import random
import json  # --- Added for pretty printing JSON ---
from datetime import datetime  # --- Added for timestamp formatting ---
from typing import List, Dict, Any
from contextlib import asynccontextmanager

import weaviate
import weaviate.classes.config as wvc
from sentence_transformers import SentenceTransformer, CrossEncoder
from dotenv import load_dotenv

import requests
import httpx
import zipfile
import tempfile
import shutil
import os.path
from urllib.parse import urlparse, unquote
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

import nltk
nltk.data.find('tokenizers/punkt')
nltk.download('punkt_tab')


load_dotenv()
WEAVIATE_HOST = os.getenv("WEAVIATE_HOST", "127.0.0.1")
WEAVIATE_PORT = int(os.getenv("WEAVIATE_PORT", 8080))
WEAVIATE_GRPC_PORT = int(os.getenv("WEAVIATE_GRPC_PORT", 50051))
AUTH_TOKEN = "fd8defb3118175da9553e106c05f40bc476971f0b46a400db1e625eaffa1fc08"
# RAG pipeline configuration
INITIAL_RETRIEVAL_K = 15  # Increased from 13 to retrieve more candidates for reranking
TOP_K_RESULTS = 8  # Keep only the top K results after reranking
RERANKER_THRESHOLD = 0.3  # Minimum score threshold for chunks after reranking
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
You are a meticulous document analyst. Your task is to answer the user's question with precision and clarity, using only the information in the provided CONTEXT blocks.

---
RULES:
1. Base your answer exclusively on the information within the "--- CONTEXT ---" section. Do not use any external knowledge or assumptions.
2. If the context does not contain the answer, reply exactly: "Based on the provided documents, I cannot find a definitive answer to this question."
3. When present in the context, include specific data points such as dates, durations, quantities, monetary amounts, percentages, definitions, conditions, eligibility criteria, exceptions, or exclusions.
4. Synthesize a single, standalone paragraph without line breaks or tabs. Keep it concise and self-contained.
5. If you can only answer part of the question, state what you can answer and specify which part is unresolved due to missing context.
6. Pay special attention to Context blocks with higher relevance scores - they are more likely to contain information directly related to the question.

---
Example:
--- CONTEXT ---
Context 1 (Relevance: 0.87): A Hospital is an institution with at least 15 inpatient beds in towns with a population over one million.
Context 2 (Relevance: 0.76): It must maintain qualified nursing staff 24/7 and daily patient records.
--- END CONTEXT ---
--- QUESTION ---
How is "Hospital" defined?
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

# Load reranker model name (can be changed as needed)
RERANKER_MODEL_NAME = 'cross-encoder/ms-marco-MiniLM-L-6-v2'

# --- 2. FastAPI Lifespan for Model Pre-loading ---


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
        MODEL_NAME, device='cpu', cache_folder=CACHE_DIR)
    print("✅ Embedding model loaded.")
    log_service_event("model_loaded", f"Embedding model loaded: {MODEL_NAME}")

    print(f"🤖 Loading reranker model '{RERANKER_MODEL_NAME}'...")
    try:
        ml_models['reranker_model'] = CrossEncoder(
            RERANKER_MODEL_NAME, device='cpu', cache_folder=CACHE_DIR)
        print("✅ Reranker model loaded.")

        # Verify the reranker model is working properly
        print("🔍 Testing reranker model with sample data...")
        test_pairs = [
            ("What is diabetes?",
             "Diabetes is a disease that occurs when your blood glucose is too high."),
            ("What are the symptoms of flu?",
             "Flu symptoms include fever, cough, sore throat, body aches.")
        ]
        scores = ml_models['reranker_model'].predict(test_pairs)

        # Check if scores contain NaN values
        scores_list = scores.tolist() if hasattr(scores, 'tolist') else list(scores)
        valid_scores = [s for s in scores_list if s == s]  # Filter out NaN

        if len(valid_scores) == len(scores_list):
            print(f"✅ Reranker model verified. Sample scores: {valid_scores}")
            model_status = "verified"
        else:
            print(
                f"⚠️ Reranker model loaded but returned {len(scores_list) - len(valid_scores)} NaN values out of {len(scores_list)}.")
            model_status = "loaded_with_nan_issue"

        log_service_event("model_loaded", f"Reranker model loaded: {RERANKER_MODEL_NAME}", {
            "model_status": model_status,
            "valid_scores": len(valid_scores),
            "total_scores": len(scores_list)
        })

    except Exception as e:
        print(f"⚠️ Error loading reranker model: {e}")
        ml_models['reranker_model'] = None
        log_error("model_loading_failed", {
            "model": RERANKER_MODEL_NAME,
            "error": str(e)
        })

    yield

    print("👋 Application shutting down...")
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
    log_service_event("connection_attempt",
                      f"Connecting to Weaviate at {WEAVIATE_HOST}:{WEAVIATE_PORT}")

    for i in range(5):  # 5 retries
        try:
            client = weaviate.connect_to_local(
                host="127.0.0.1",
                port="8080",
                grpc_port="50051",
            )
            if client.is_ready():
                print("✅ Weaviate is ready!")
                log_service_event("connection_success",
                                  "Weaviate connection established")
                return client
            client.close()
            await asyncio.sleep(3)
        except Exception as e:
            error_msg = f"⏳ Weaviate not ready, retrying... (Attempt {i+1}/5). Error: {e}"
            print(error_msg)
            log_error("Weaviate connection failed", {
                      "attempt": i+1, "error": str(e)})
            await asyncio.sleep(3)

    error_msg = "❌ Could not connect to Weaviate after multiple retries."
    log_error("Weaviate connection failed permanently")
    raise ConnectionError(error_msg)


async def extract_and_process_zip(api_url: str, zip_content: bytes, original_filename: str) -> List[str]:
    """
    Extracts a ZIP file and processes any supported document files found inside.

    Supported formats include: PDF, DOCX, DOC, TXT, MD, RTF

    Parameters:
        api_url (str): URL of the parsing API.
        zip_content (bytes): The content of the ZIP file.
        original_filename (str): The original filename of the ZIP file.

    Returns:
        List[str]: Combined list of text chunks from all documents in the ZIP.

    Raises:
        HTTPException: If ZIP extraction fails or no supported documents are found.
    """
    zip_id = str(uuid.uuid4())
    combined_chunks = []

    # Create a temporary directory for extraction
    with tempfile.TemporaryDirectory() as temp_dir:
        zip_path = os.path.join(temp_dir, "downloaded.zip")

        # Log zip extraction start
        log_service_event("zip_extraction_start", "Starting ZIP file extraction", {
            "zip_id": zip_id,
            "original_filename": original_filename,
            "content_size_bytes": len(zip_content),
            "temp_dir": temp_dir
        })

        # Write ZIP content to temporary file
        with open(zip_path, 'wb') as f:
            f.write(zip_content)

        try:
            # Extract ZIP file
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
                file_list = zip_ref.namelist()

                # Log extraction completion
                log_service_event("zip_extraction_complete", "ZIP file extracted successfully", {
                    "zip_id": zip_id,
                    "files_count": len(file_list),
                    "extracted_files": file_list[:10] + (["..."] if len(file_list) > 10 else [])
                })

                # Define supported document file extensions
                supported_extensions = [
                    '.pdf', '.docx', '.doc', '.txt', '.md', '.rtf']

                # Map file extensions to content types
                content_type_map = {
                    '.pdf': 'application/pdf',
                    '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                    '.doc': 'application/msword',
                    '.txt': 'text/plain',
                    '.md': 'text/markdown',
                    '.rtf': 'application/rtf'
                }

                # Find all supported document files in the extracted content
                doc_files = []
                for f in file_list:
                    ext = os.path.splitext(f.lower())[1]
                    if ext in supported_extensions:
                        doc_files.append(f)

                if not doc_files:
                    error_msg = "No supported document files found in the ZIP archive."
                    log_error("no_documents_in_zip", {
                        "zip_id": zip_id,
                        "original_filename": original_filename,
                        "supported_extensions": supported_extensions
                    })
                    raise HTTPException(status_code=422, detail=error_msg)

                print(
                    f"📁 Found {len(doc_files)} document files in ZIP archive")
                log_service_event("documents_found_in_zip", f"Found document files in ZIP archive", {
                    "zip_id": zip_id,
                    "document_count": len(doc_files),
                    "document_files": doc_files[:5] + (["..."] if len(doc_files) > 5 else [])
                })

                # Process each document file
                for pdf_file in doc_files:
                    file_path = os.path.join(temp_dir, pdf_file)
                    file_ext = os.path.splitext(pdf_file.lower())[1]

                    # Determine content type based on extension
                    content_type = content_type_map.get(
                        file_ext, 'application/pdf')

                    # Read the file content
                    with open(file_path, 'rb') as f:
                        pdf_content = f.read()

                    print(
                        f"📄 Processing {file_ext.upper()} file '{pdf_file}' from ZIP")

                    # Create a MultipartEncoder for the document
                    encoder = MultipartEncoder(
                        fields={'file': (os.path.basename(
                            pdf_file), pdf_content, content_type)}
                    )

                    # Send the document to the parsing API
                    print(
                        f"➡️  Sending '{pdf_file}' from ZIP to parsing API...")
                    log_service_event("parsing_zip_document", f"Sending document from ZIP to parsing API", {
                        "zip_id": zip_id,
                        "document_file": pdf_file,
                        "file_extension": file_ext,
                        "content_type": content_type,
                        "file_size_bytes": len(pdf_content)
                    })

                    # Parse the PDF
                    try:
                        response = requests.post(api_url, data=encoder, headers={
                            'Content-Type': encoder.content_type
                        })
                        response.raise_for_status()

                        # Extract chunks from response
                        response_json = response.json()
                        blocks = response_json.get('return_dict', {}).get(
                            'result', {}).get('blocks', [])

                        if blocks:
                            pdf_chunks = [
                                " ".join(block.get('sentences', [])) for block in blocks]
                            combined_chunks.extend(pdf_chunks)

                            log_service_event("zip_document_parsed", f"Successfully parsed document from ZIP", {
                                "zip_id": zip_id,
                                "document_file": pdf_file,
                                "file_extension": file_ext,
                                "chunks_count": len(pdf_chunks)
                            })
                        else:
                            log_service_event("zip_document_empty", f"No text chunks found in document from ZIP", {
                                "zip_id": zip_id,
                                "document_file": pdf_file,
                                "file_extension": file_ext
                            })

                    except Exception as e:
                        # Log error but continue with other documents
                        log_error("zip_document_parsing_error", {
                            "zip_id": zip_id,
                            "document_file": pdf_file,
                            "file_extension": file_ext,
                            "error": str(e)
                        })
                        print(f"⚠️ Error parsing '{pdf_file}' from ZIP: {e}")
                        continue

        except zipfile.BadZipFile as e:
            error_msg = f"Invalid ZIP file: {e}"
            log_error("invalid_zip_file", {
                "zip_id": zip_id,
                "error": str(e)
            })
            raise HTTPException(status_code=422, detail=error_msg)

    # Check if we got any chunks
    if not combined_chunks:
        error_msg = "No text could be extracted from the documents in the ZIP file."
        log_error("no_text_from_zip_documents", {
            "zip_id": zip_id,
            "original_filename": original_filename
        })
        raise HTTPException(status_code=422, detail=error_msg)

    log_service_event("zip_processing_complete", "Completed processing of ZIP archive", {
        "zip_id": zip_id,
        "total_chunks": len(combined_chunks),
        "documents_processed": len(doc_files)
    })

    return combined_chunks


async def fetch_and_parse_pdf(api_url: str, file_url: str) -> List[str]:
    """
    Downloads and processes document files for text extraction.

    Supported formats:
    - PDF (.pdf)
    - Word Documents (.docx, .doc)
    - Text Files (.txt)
    - Markdown Files (.md)
    - Rich Text Format (.rtf)
    - ZIP archives containing any of the above formats

    If the file is a supported document, sends it to the parsing API to extract text.
    If the file is a ZIP archive, extracts it and processes all supported documents inside.

    Parameters:
        api_url (str): URL of the parsing API.
        file_url (str): URL of the file to download.

    Returns:
        List[str]: List of text chunks extracted from the document(s).

    Raises:
        HTTPException: If download fails, parsing fails, or no text chunks are found.
    """
    document_id = str(uuid.uuid4())
    start_time = time.time()

    print(f"\n⬇️  Downloading file from: {file_url}")
    log_service_event("download_start", f"Downloading document", {
        "url": file_url,
        "document_id": document_id,
        "timestamp": datetime.now().isoformat()
    })

    try:
        download_start = time.time()
        file_response = requests.get(file_url)
        file_response.raise_for_status()
        download_time = time.time() - download_start

        # Log download completion with detailed metrics
        log_service_event("download_complete", f"Document downloaded successfully", {
            "document_id": document_id,
            "size_bytes": len(file_response.content),
            "download_time_seconds": download_time,
            "content_type": file_response.headers.get('Content-Type'),
            "status_code": file_response.status_code
        })
    except requests.RequestException as e:
        error_msg = f"Failed to download document from URL: {e}"
        log_error("document_download_failed", {
            "document_id": document_id,
            "url": file_url,
            "error": str(e),
            "elapsed_time": time.time() - start_time
        })
        raise HTTPException(status_code=400, detail=error_msg)

    filename = os.path.basename(unquote(urlparse(file_url).path))
    file_extension = os.path.splitext(filename.lower())[1]

    # Check if the downloaded file is a ZIP
    if file_extension == '.zip':
        print(f"📦 Detected ZIP file: '{filename}'")
        log_service_event("zip_file_detected", "ZIP file detected, will extract and process", {
            "document_id": document_id,
            "filename": filename,
            "file_size_bytes": len(file_response.content)
        })

        # Process the ZIP file
        return await extract_and_process_zip(api_url, file_response.content, filename)

    # Determine content type based on file extension
    content_type_map = {
        '.pdf': 'application/pdf',
        '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        '.doc': 'application/msword',
        '.txt': 'text/plain',
        '.md': 'text/markdown',
        '.rtf': 'application/rtf'
    }

    # Get appropriate content type or default to PDF
    content_type = content_type_map.get(file_extension, 'application/pdf')

    # Log the detected file type
    if file_extension in content_type_map:
        print(f"📄 Detected {file_extension.upper()} file: '{filename}'")
        log_service_event("supported_file_type_detected", f"Detected supported file type: {file_extension}", {
            "document_id": document_id,
            "filename": filename,
            "extension": file_extension,
            "content_type": content_type
        })
    else:
        print(
            f"⚠️ Unknown file extension '{file_extension}', treating as PDF: '{filename}'")
        log_service_event("unknown_file_extension", "Unknown file extension, treating as PDF", {
            "document_id": document_id,
            "filename": filename,
            "extension": file_extension
        })

    encoder = MultipartEncoder(
        fields={'file': (filename, file_response.content, content_type)})

    print(f"➡️  Sending '{filename}' to parsing API at {api_url}...")
    log_service_event("parsing_start", f"Sending document to parsing API", {
        "document_id": document_id,
        "filename": filename,
        "api": api_url,
        "file_size_bytes": len(file_response.content)
    })

    try:
        parsing_start = time.time()
        response = requests.post(api_url, data=encoder, headers={
                                 'Content-Type': encoder.content_type})
        response.raise_for_status()
        parsing_time = time.time() - parsing_start

        # Log parsing API response metrics
        log_service_event("parsing_response_received", "Received response from parsing API", {
            "document_id": document_id,
            "status_code": response.status_code,
            "parsing_time_seconds": parsing_time,
            "response_size_bytes": len(response.content)
        })
    except requests.RequestException as e:
        error_msg = f"Document parsing service failed: {e}"
        log_error("parsing_failed", {
            "document_id": document_id,
            "api": api_url,
            "error": str(e),
            "elapsed_time": time.time() - start_time
        })
        raise HTTPException(status_code=503, detail=error_msg)

    # Parse the response and log structure details
    try:
        response_json = response.json()

        # Log parsed response structure (without full content)
        structure_details = {
            "document_id": document_id,
            "has_return_dict": "return_dict" in response_json,
            "has_result": "result" in response_json.get("return_dict", {}),
            "has_blocks": "blocks" in response_json.get("return_dict", {}).get("result", {})
        }

        # Add additional structure information if available
        if "return_dict" in response_json and "result" in response_json.get("return_dict", {}):
            result = response_json.get("return_dict", {}).get("result", {})
            structure_details.update({
                "metadata_keys": list(result.keys()) if isinstance(result, dict) else [],
                "has_metadata": "metadata" in result if isinstance(result, dict) else False,
                "pages_count": len(result.get("pages", [])) if isinstance(result, dict) else 0
            })

        log_service_event("parsing_result_structure",
                          "Structure of parsing result", structure_details)
    except Exception as e:
        error_msg = f"Failed to parse API response: {e}"
        log_error("parsing_response_parse_error", {
            "document_id": document_id,
            "error": str(e)
        })
        raise HTTPException(status_code=500, detail=error_msg)

    # Extract blocks from the response
    blocks = response_json.get('return_dict', {}).get(
        'result', {}).get('blocks', [])

    if not blocks:
        error_msg = "No text chunks were found in the parsed document."
        log_error("no_text_chunks", {
            "document_id": document_id,
            "filename": filename
        })
        raise HTTPException(status_code=422, detail=error_msg)

    # Extract and analyze chunk texts
    chunk_texts = [" ".join(block.get('sentences', [])) for block in blocks]

    # Calculate chunk statistics
    chunk_lengths = [len(chunk) for chunk in chunk_texts]
    chunk_word_counts = [len(chunk.split()) for chunk in chunk_texts]

    # Log detailed chunk statistics
    log_service_event("chunks_extracted", "Extracted raw text chunks from document", {
        "document_id": document_id,
        "raw_chunks_count": len(chunk_texts),
        "total_chars": sum(chunk_lengths),
        "total_words": sum(chunk_word_counts),
        "chars_statistics": {
            "min": min(chunk_lengths) if chunk_lengths else 0,
            "max": max(chunk_lengths) if chunk_lengths else 0,
            "mean": sum(chunk_lengths) / len(chunk_lengths) if chunk_lengths else 0
        },
        "words_statistics": {
            "min": min(chunk_word_counts) if chunk_word_counts else 0,
            "max": max(chunk_word_counts) if chunk_word_counts else 0,
            "mean": sum(chunk_word_counts) / len(chunk_word_counts) if chunk_word_counts else 0
        },
        "short_chunks": sum(1 for length in chunk_lengths if length < 100),
        "medium_chunks": sum(1 for length in chunk_lengths if 100 <= length < 500),
        "long_chunks": sum(1 for length in chunk_lengths if length >= 500)
    })

    # Log sample of first few chunks for debugging (only log first 3 chunks to avoid excessive logging)
    for i, chunk in enumerate(chunk_texts[:3]):
        preview = chunk[:100] + ("..." if len(chunk) > 100 else "")
        log_service_event("chunk_sample", f"Sample of extracted chunk {i+1}", {
            "document_id": document_id,
            "chunk_index": i,
            "chunk_length": len(chunk),
            "word_count": len(chunk.split()),
            "preview": preview
        })

    print(f"✅ Parsed document into {len(chunk_texts)} text chunks.")
    log_service_event("parsing_complete", f"Document parsed successfully", {
        "document_id": document_id,
        "chunks_count": len(chunk_texts),
        "avg_chunk_length": sum(chunk_lengths) / len(chunk_lengths) if chunk_lengths else 0,
        "total_processing_time": time.time() - start_time
    })
    return chunk_texts


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


async def ingest_to_weaviate(client: weaviate.WeaviateClient, collection_name: str, chunks: List[str]):
    """
    Generate embeddings for text chunks and ingest them into Weaviate.

    Parameters:
        client (weaviate.WeaviateClient): The Weaviate client.
        collection_name (str): Name of the collection to create.
        chunks (List[str]): List of text chunks to embed and ingest.
    """
    ingest_id = str(uuid.uuid4())
    overall_start_time = time.time()

    # Log detailed information about the ingestion process starting
    log_service_event("ingestion_process_start", "Starting vector ingestion process", {
        "ingest_id": ingest_id,
        "collection_name": collection_name,
        "chunks_count": len(chunks),
        "total_text_size": sum(len(chunk) for chunk in chunks),
        "embedding_model": MODEL_NAME,
        "timestamp": datetime.now().isoformat()
    })

    # Generate embeddings with BGE's recommended prompt wrapper
    model = ml_models['embedding_model']
    print(f"\n🧠 Generating embeddings for {len(chunks)} chunks...")

    # Log embedding generation parameters
    log_service_event("embedding_generation_start", "Starting chunk embedding generation", {
        "ingest_id": ingest_id,
        "chunks_count": len(chunks),
        "collection_name": collection_name,
        "bge_prompt_used": True,
        "model_name": MODEL_NAME,
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

    # Create collection with timing
    collection_start_time = time.time()

    # Log collection creation start
    log_service_event("collection_creation_start", "Starting Weaviate collection creation", {
        "ingest_id": ingest_id,
        "collection_name": collection_name,
        "weaviate_host": WEAVIATE_HOST,
        "weaviate_port": WEAVIATE_PORT,
        "vector_dimension": embeddings.shape[1]
    })

    # Check if collection exists and delete if necessary
    collection_exists = False
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
            start_obj_time = time.time()
            batch.add_object(
                properties={"content": text},
                uuid=uuid.uuid4(),
                vector=embeddings[i].tolist()
            )
            objects_added += 1

            # Track batch performance (every 100 objects to avoid excessive logging)
            if i % 100 == 0 and i > 0:
                current_batch_time = time.time() - batch_start_time
                throughput = i / current_batch_time if current_batch_time > 0 else 0
                batch_size_tracker.append({
                    "objects_processed": i,
                    "elapsed_time": current_batch_time,
                    "throughput_objects_per_second": throughput
                })

                # Log batch progress update (every 500 objects)
                if i % 500 == 0:
                    log_service_event("batch_insertion_progress", f"Batch insertion progress update", {
                        "ingest_id": ingest_id,
                        "objects_processed": i,
                        "total_objects": len(chunks),
                        "percent_complete": round((i / len(chunks)) * 100, 2) if len(chunks) > 0 else 0,
                        "elapsed_time_seconds": current_batch_time,
                        "estimated_time_remaining": (len(chunks) - i) / throughput if throughput > 0 else None
                    })

    # Calculate batch processing statistics
    batch_time = time.time() - batch_start_time
    failed_count = len(policy_collection.batch.failed_objects)
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
            sample_failures = []
            for i, fail_obj in enumerate(policy_collection.batch.failed_objects[:5]):
                sample_failures.append({
                    "index": i,
                    "error": str(fail_obj.get('error', 'Unknown error')),
                    "object_id": str(fail_obj.get('id', 'Unknown ID'))
                })
            failure_info["sample_failures"] = sample_failures

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
        "embedding_time_seconds": embedding_time,
        "collection_time_seconds": total_collection_time,
        "insertion_time_seconds": ingest_time,
        "total_time_seconds": total_process_time,
        "timestamp": datetime.now().isoformat()
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


def diagnose_reranker_model(model, test_pairs=None):
    """
    Diagnose the reranker model to check if it's working properly.

    Parameters:
        model: The reranker model to test
        test_pairs: Optional test pairs to use; if None, some default pairs will be created

    Returns:
        dict: Diagnostic results
    """
    if test_pairs is None:
        # Create some simple test pairs
        test_pairs = [
            ("What is diabetes?",
             "Diabetes is a disease that occurs when your blood glucose is too high."),
            ("What are the symptoms of flu?",
             "Flu symptoms include fever, cough, sore throat, body aches."),
            ("How to lose weight?",
             "Exercise regularly and maintain a healthy diet to lose weight.")
        ]

    try:
        print("\n" + "="*50)
        print("RERANKER MODEL DIAGNOSTICS")
        print("="*50)
        print(f"Model: {model.__class__.__name__}")
        print(f"Test pairs: {len(test_pairs)}")

        # Test prediction
        scores = model.predict(test_pairs)

        # Check for NaN values
        scores_list = scores.tolist() if hasattr(scores, 'tolist') else list(scores)
        valid_scores = [s for s in scores_list if s == s]  # Filter out NaN

        print(
            f"Scores shape/length: {scores.shape if hasattr(scores, 'shape') else len(scores)}")
        print(f"Valid scores: {len(valid_scores)}/{len(scores_list)}")
        print(f"Score stats: min={min(valid_scores) if valid_scores else 'N/A'}, "
              f"max={max(valid_scores) if valid_scores else 'N/A'}, "
              f"mean={sum(valid_scores)/len(valid_scores) if valid_scores else 'N/A'}")
        print(f"Sample scores: {valid_scores[:3]}")
        print("="*50 + "\n")

        return {
            "success": True,
            "valid_scores": len(valid_scores),
            "total_scores": len(scores_list),
            "has_nan": len(valid_scores) != len(scores_list),
            "stats": {
                "min": min(valid_scores) if valid_scores else None,
                "max": max(valid_scores) if valid_scores else None,
                "mean": sum(valid_scores)/len(valid_scores) if valid_scores else None,
            }
        }
    except Exception as e:
        print(f"❌ Reranker diagnostic error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e)
        }


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

    # Generate embedding for the question with BGE's recommended prompt wrapper
    embedding_model = ml_models['embedding_model']
    # Use BGE's recommended prompt wrapper
    bge_wrapped_query = f"Represent this sentence for searching relevant passages: {question}"
    query_vector = embedding_model.encode(bge_wrapped_query).tolist()
    log_service_event("embedding_generated", "Question embedding generated", {
        "question_id": question_id,
        "vector_dimension": len(query_vector),
        "bge_prompt_used": True
    })

    # Query Weaviate using hybrid search (combines vector and keyword search)
    # Note: We're directly using TOP_K_RESULTS here since we're skipping reranking
    policy_collection = weaviate_client.collections.get(collection_name)

    # Log retrieval parameters before executing query
    log_service_event("chunk_retrieval_params", "Preparing to retrieve chunks using hybrid search", {
        "question_id": question_id,
        "query_text": question,
        "vector_dimension": len(query_vector),
        "alpha": 0.5,  # Balance between vector and keyword search
        "limit": TOP_K_RESULTS,  # Using TOP_K_RESULTS directly
        "collection_name": collection_name,
        "reranking_disabled": True  # Flag to indicate reranking is disabled
    })

    retrieval_start = time.time()
    response = policy_collection.query.hybrid(
        query=question,  # Text for BM25 keyword search
        vector=query_vector,  # Vector for semantic search
        alpha=0.5,  # Balance between vector (0) and keyword (1) search
        limit=TOP_K_RESULTS,  # Using TOP_K_RESULTS directly since we're skipping reranking
        return_properties=["content"]
    )
    retrieval_time = time.time() - retrieval_start

    # Extract context chunks and their object IDs
    context_chunks = []
    chunk_ids = []
    chunk_scores = []

    # Process the response objects to extract content and metadata
    for i, obj in enumerate(response.objects):
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

    # [RERANKING REMOVED] - Skipping reranking and directly using chunks from vector search
    print(
        f"🔍 Using top {len(context_chunks)} chunks directly from vector search (reranking disabled)")
    log_service_event("reranking_skipped", "Skipping reranking process as requested", {
        "question_id": question_id,
        "using_chunks_count": len(context_chunks),
        "using_direct_vector_search": True
    })
    if not context_chunks:
        answer_text = "Based on the provided documents, I cannot find any relevant information to answer this question."
        log_service_event("empty_context", "No context chunks found for question", {
            "question_id": question_id,
            "rag_pipeline_failure": True,
            "failure_reason": "No context chunks retrieved from vector store"
        })
    else:
        # Format context with chunk numbering for better structure
        context_entries = []
        context_formatting_start = time.time()

        # Log the start of context formatting
        log_service_event("context_formatting_start", "Starting context formatting for LLM prompt", {
            "question_id": question_id,
            "chunks_to_format": len(context_chunks)
        })

        # Track word and character counts for detailed logging
        total_words = 0
        total_chars = 0
        context_metadata = []

        for i, chunk in enumerate(context_chunks):
            # Calculate chunk statistics
            chunk_words = len(chunk.split())
            chunk_chars = len(chunk)
            total_words += chunk_words
            total_chars += chunk_chars

            # Store metadata about this chunk for logging
            chunk_meta = {
                "chunk_index": i,
                "words": chunk_words,
                "chars": chunk_chars
            }

            # Since we're skipping reranking, just use simple context numbering
            # No relevance scores to include (avoiding NoneType.__format__ error)
            context_entries.append(f"Context {i+1}: {chunk}")

            context_metadata.append(chunk_meta)

        # Build context string with clear separation between chunks
        context_str = "\n\n---\n\n".join(context_entries)
        context_formatting_time = time.time() - context_formatting_start

        # Log detailed context preparation metrics
        log_service_event("context_prepared", "Context prepared for LLM", {
            "question_id": question_id,
            "chunks_count": len(context_chunks),
            "total_context_length": len(context_str),
            "total_words": total_words,
            "total_chars": total_chars,
            "avg_words_per_chunk": total_words / len(context_chunks) if context_chunks else 0,
            "avg_chars_per_chunk": total_chars / len(context_chunks) if context_chunks else 0,
            "formatting_time_seconds": context_formatting_time,
            "has_relevance_scores": 'chunk_scores' in locals(),
            # Log metadata for first 5 chunks only
            "chunk_metadata": context_metadata[:5]
        })

        # Log token estimation (approximate)
        estimated_tokens = total_chars / 4  # Simple approximation of token count
        log_service_event("context_token_estimation", "Estimated token count for context", {
            "question_id": question_id,
            "estimated_tokens": estimated_tokens,
            "estimated_tokens_with_prompt": estimated_tokens + 200,  # Add rough prompt overhead
            "estimation_method": "chars_divided_by_4"
        })

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
        # Clean and rechunk
        chunks = clean_and_rechunk_texts(
            chunks, chunk_token_size=450, overlap=30)
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
            # Log previews only for brevity
            "answers": [a[:100] + ("..." if len(a) > 100 else "") for a in answer_texts],
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
                print(
                    f"\n✅ Cleaned up and deleted collection: '{collection_name}'")
                log_service_event("collection_deleted", "Cleaned up Weaviate collection", {
                    "request_id": request_id,
                    "collection_name": collection_name
                })
            weaviate_client.close()
            print("✅ Weaviate client connection closed.")
            log_service_event("connection_closed", "Weaviate client connection closed", {
                              "request_id": request_id})

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
