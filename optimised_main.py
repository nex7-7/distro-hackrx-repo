import os
import time
import uuid
import asyncio
import random
import json
import logging
from typing import List, Set
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

import weaviate
import weaviate.classes.config as wvc
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

import requests
import httpx
from urllib.parse import urlparse
from tqdm import tqdm

from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, Field
import uvicorn

# --- 1. LOGGING SETUP ---


def setup_logging():
    """Configures the application logger."""
    log_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Create a handler for file output
    file_handler = logging.FileHandler("rag_pipeline.log")
    file_handler.setFormatter(log_formatter)

    # Create a handler for console output
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)

    # Get the root logger and add handlers
    # Using a specific name avoids interfering with uvicorn's root logger
    logger = logging.getLogger("rag_pipeline")
    logger.setLevel(logging.INFO)

    # Prevent log messages from being propagated to the root logger
    logger.propagate = False

    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger


# Get the logger instance
logger = setup_logging()

# --- 2. Configuration ---
load_dotenv()
WEAVIATE_HOST = os.getenv("WEAVIATE_HOST", "127.0.0.1")
WEAVIATE_PORT = int(os.getenv("WEAVIATE_PORT", 8080))
WEAVIATE_GRPC_PORT = int(os.getenv("WEAVIATE_GRPC_PORT", 50051))
TOP_K_RESULTS = 10
CACHE_DIR = "./huggingface_cache"
MODEL_NAME = 'BAAI/bge-base-en-v1.5'
GEMINI_MODEL_NAME = 'gemini-1.5-flash-latest'
GEMINI_API_KEYS_STR = os.getenv("GEMINI_API_KEYS")
if not GEMINI_API_KEYS_STR:
    raise ValueError("GEMINI_API_KEYS not found in .env file.")
GEMINI_API_KEY_LIST = [key.strip()
                       for key in GEMINI_API_KEYS_STR.split(',') if key.strip()]
if not GEMINI_API_KEY_LIST:
    raise ValueError("GEMINI_API_KEYS variable is empty or invalid.")
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL_NAME}:generateContent"
PARSING_API_URL = "http://127.0.0.1:5010/api/parseDocument?renderFormat=all"

# --- Data Structures to support the PromptTemplate class ---


@dataclass
class Chunk:
    """Represents a single chunk of text."""
    content: str


@dataclass
class RetrievedContext:
    """Represents the context retrieved from the database."""
    chunks: List[Chunk] = field(default_factory=list)

# --- Integrated PromptTemplate Class ---


class PromptTemplate:
    """
    Manages prompt templates for the RAG pipeline.
    """
    @staticmethod
    def create_policy_evaluation_prompt(query: str, context: RetrievedContext) -> str:
        """
        Create a structured prompt for policy evaluation.
        """
        system_prompt = """You are an expert insurance policy evaluator. Your role is to analyze insurance claims based on policy documents and provide clear, accurate responses.

INSTRUCTIONS:
1. Analyze the user's query against the provided policy context.
2. Do not mention the chunk ids, and reference
3. The answer will be in the context provided. If there's some ambiguity, assume context, and answer
4. If Policy Name in the query does not match, assume it is the same policy as mentioned in the context (IMPORTANT)
5. You MUST provide a concise, single-paragraph answer. Your entire response MUST NOT exceed 75 words. Do not explain your reasoning or mention the source document.
6. Answer what is asked, and nothing more.

EXAMPLE:
Query: What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?
Expected Response: A grace period of thirty days is provided for premium payment after the due date to renew or continue the policy without losing continuity benefits.
"""
        context_section = "\n=== POLICY CONTEXT ===\n"
        if context.chunks:
            for i, chunk in enumerate(context.chunks, 1):
                context_section += f"\n--- Context {i} ---\n"
                context_section += f"Content: {chunk.content}\n"
        else:
            context_section += "\nNo relevant policy context found.\n"

        query_section = f"\n=== USER QUERY ===\n{query}\n"
        instructions_section = "\n=== EVALUATION REQUEST ===\nDirectly and concisely answer the user's query using only the provided policy context. Emulate the style of the example provided."
        full_prompt = system_prompt + context_section + \
            query_section + instructions_section
        return full_prompt


ml_models = {}

# --- 3. FastAPI Lifespan ---


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Application starting up...")
    logger.info(f"🤖 Loading embedding model '{MODEL_NAME}'...")
    ml_models['embedding_model'] = SentenceTransformer(
        MODEL_NAME, device='cpu', cache_folder=CACHE_DIR)
    logger.info("✅ Embedding model loaded.")
    yield
    logger.info("👋 Application shutting down...")
    ml_models.clear()

app = FastAPI(
    title="Advanced RAG API with Logging",
    description="A production-ready RAG pipeline with file and console logging.",
    version="2.4.0",
    lifespan=lifespan
)

# --- Pydantic Models ---


class QueryRequest(BaseModel):
    documents: str = Field(..., example="https://some-url.com/document.pdf")
    questions: List[str] = Field(..., example=["What is the grace period?"])


class Answer(BaseModel):
    question: str
    answer: str


class QueryResponse(BaseModel):
    results: List[Answer]
    processing_time_seconds: float

# --- 4. Core Logic Functions ---


async def connect_to_weaviate() -> weaviate.WeaviateClient:
    logger.info(
        f"🔗 Connecting to Weaviate at {WEAVIATE_HOST}:{WEAVIATE_PORT}...")
    for i in range(5):
        try:
            client = weaviate.connect_to_local(
                host=WEAVIATE_HOST, port=WEAVIATE_PORT, grpc_port=WEAVIATE_GRPC_PORT,
            )
            if client.is_ready():
                logger.info("✅ Weaviate is ready!")
                return client
            client.close()
            await asyncio.sleep(3)
        except Exception as e:
            logger.warning(
                f"⏳ Weaviate not ready, retrying... (Attempt {i+1}/5). Error: {e}")
            await asyncio.sleep(3)
    raise ConnectionError(
        "❌ Could not connect to Weaviate after multiple retries.")


async def fetch_and_parse_pdf(api_url: str, file_url: str) -> List[str]:
    logger.info(f"⬇️  Downloading file from: {file_url}")
    try:
        pdf_response = requests.get(file_url)
        pdf_response.raise_for_status()
    except requests.RequestException as e:
        raise HTTPException(
            status_code=400, detail=f"Failed to download document from URL: {e}")

    filename = os.path.basename(urlparse(file_url).path)
    files = {'file': (filename, pdf_response.content, 'application/pdf')}

    logger.info(f"➡️  Sending '{filename}' to parsing API at {api_url}...")
    try:
        response = requests.post(api_url, files=files)
        response.raise_for_status()
    except requests.RequestException as e:
        raise HTTPException(
            status_code=503, detail=f"Document parsing service failed: {e}")

    blocks = response.json().get('return_dict', {}).get('result', {}).get('blocks', [])
    if not blocks:
        raise HTTPException(
            status_code=422, detail="No text chunks were found in the parsed document.")
    chunk_texts = [" ".join(block.get('sentences', [])) for block in blocks]
    logger.info(f"✅ Parsed document into {len(chunk_texts)} text chunks.")
    return chunk_texts


async def ingest_to_weaviate(client: weaviate.WeaviateClient, collection_name: str, chunks: List[str]):
    model = ml_models['embedding_model']
    logger.info(f"🧠 Generating embeddings for {len(chunks)} chunks...")
    embeddings = model.encode(chunks, show_progress_bar=True)

    logger.info(
        "\n=================================================="
        "\n1. EMBEDDINGS RECEIVED"
        "\n=================================================="
        f"\nShape of embeddings array: {embeddings.shape}"
        "\n=================================================="
    )

    if client.collections.exists(collection_name):
        client.collections.delete(collection_name)

    client.collections.create(
        name=collection_name,
        vectorizer_config=wvc.Configure.Vectorizer.none(),
        properties=[
            wvc.Property(name="content", data_type=wvc.DataType.TEXT),
            wvc.Property(name="chunk_index", data_type=wvc.DataType.INT)
        ]
    )
    logger.info(f"✅ Created Weaviate collection: '{collection_name}'")
    policy_collection = client.collections.get(collection_name)
    logger.info(f"🚀 Pushing {len(chunks)} objects to Weaviate...")

    with policy_collection.batch.dynamic() as batch:
        for i, text in enumerate(tqdm(chunks, desc="Batching objects")):
            properties = {"content": text, "chunk_index": i}
            batch.add_object(
                properties=properties,
                uuid=uuid.uuid4(),
                vector=embeddings[i].tolist()
            )

    if len(policy_collection.batch.failed_objects) > 0:
        logger.warning(
            f"⚠️ Failed to push {len(policy_collection.batch.failed_objects)} objects.")
    logger.info(f"✅ Pushed {len(chunks)} objects to Weaviate successfully.")


async def generate_gemini_response_httpx(session: httpx.AsyncClient, prompt: str) -> str:
    api_key = random.choice(GEMINI_API_KEY_LIST)
    headers = {'Content-Type': 'application/json'}
    payload = {'contents': [{'parts': [{'text': prompt}]}]}

    logger.info(
        "\n=================================================="
        "\n2. PAYLOAD GIVEN TO GEMINI"
        "\n=================================================="
        f"\n{json.dumps(payload, indent=2)}"
        "\n=================================================="
    )

    try:
        response = await session.post(f"{GEMINI_API_URL}?key={api_key}", json=payload, headers=headers, timeout=90)

        try:
            response_json = response.json()
            response_to_log = json.dumps(response_json, indent=2)
        except json.JSONDecodeError:
            response_to_log = response.text

        logger.info(
            "\n=================================================="
            "\n3. OUTPUT RECEIVED FROM GEMINI"
            "\n=================================================="
            f"\nStatus Code: {response.status_code}"
            f"\nResponse Body:\n{response_to_log}"
            "\n=================================================="
        )

        response.raise_for_status()
        return response_json['candidates'][0]['content']['parts'][0]['text']
    except httpx.HTTPStatusError as e:
        error_msg = f"Gemini API error: {e.response.status_code} - {e.response.text}"
        logger.error(f"❌ {error_msg}")
        return error_msg
    except (KeyError, IndexError) as e:
        error_msg = f"Error parsing Gemini response: {e}. Response: {response.text}"
        logger.error(f"❌ {error_msg}", exc_info=True)
        return error_msg
    except Exception as e:
        error_msg = f"An unexpected error occurred during Gemini call: {e}"
        logger.error(f"❌ {error_msg}", exc_info=True)
        return error_msg


async def process_single_question(
    question: str,
    collection_name: str,
    weaviate_client: weaviate.WeaviateClient,
    http_session: httpx.AsyncClient,
    all_document_chunks: List[str]
) -> Answer:
    logger.info(f"🔍 Processing question: '{question}'")
    embedding_model = ml_models['embedding_model']

    query_vector = embedding_model.encode(question).tolist()
    policy_collection = weaviate_client.collections.get(collection_name)
    response = policy_collection.query.near_vector(
        near_vector=query_vector,
        limit=TOP_K_RESULTS,
        return_properties=["content", "chunk_index"]
    )

    neighbor_indices: Set[int] = set()
    if response.objects:
        retrieved_indices = {
            obj.properties['chunk_index'] for obj in response.objects}
        for index in retrieved_indices:
            neighbor_indices.add(index)
            if index > 0:
                neighbor_indices.add(index - 1)
            if index < len(all_document_chunks) - 1:
                neighbor_indices.add(index + 1)

    final_indices = sorted(list(neighbor_indices))
    context_chunks_str = [all_document_chunks[i] for i in final_indices]
    logger.info(
        f"Retrieved {len(response.objects)} chunks, expanded to {len(context_chunks_str)} with neighbors.")

    if not context_chunks_str:
        answer_text = "Based on the provided documents, I cannot find any relevant information to answer this question."
    else:
        context_obj = RetrievedContext(
            chunks=[Chunk(content=text) for text in context_chunks_str]
        )
        prompt = PromptTemplate.create_policy_evaluation_prompt(
            query=question, context=context_obj)
        answer_text = await generate_gemini_response_httpx(http_session, prompt)

    return Answer(question=question, answer=answer_text)

# --- 5. FastAPI Endpoint ---


@app.post("/process_document/", response_model=QueryResponse, tags=["RAG Pipeline"])
async def process_document_and_answer_questions(request: QueryRequest = Body(...)):
    start_time = time.time()
    weaviate_client = None
    collection_name = f"Doc_{uuid.uuid4().hex}"
    try:
        weaviate_client = await connect_to_weaviate()
        all_chunks = await fetch_and_parse_pdf(PARSING_API_URL, request.documents)
        await ingest_to_weaviate(weaviate_client, collection_name, all_chunks)

        async with httpx.AsyncClient() as http_session:
            tasks = [
                process_single_question(
                    q, collection_name, weaviate_client, http_session, all_chunks)
                for q in request.questions
            ]
            final_answers = await asyncio.gather(*tasks)

        end_time = time.time()
        return QueryResponse(results=final_answers, processing_time_seconds=round(end_time - start_time, 2))
    except Exception as e:
        logger.error(
            f"❌ A critical error occurred in the main endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if weaviate_client and weaviate_client.is_connected():
            if weaviate_client.collections.exists(collection_name):
                weaviate_client.collections.delete(collection_name)
                logger.info(
                    f"✅ Cleaned up and deleted collection: '{collection_name}'")
            weaviate_client.close()
            logger.info("✅ Weaviate client connection closed.")

# --- Main Execution Block ---
if __name__ == "__main__":
    uvicorn.run("optimised_main:app", host="0.0.0.0", port=8000, reload=True)
