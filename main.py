import os
import time
import uuid
import asyncio
import random
import json  # --- Added for pretty printing JSON ---
from typing import List
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
You are a meticulous and expert insurance policy analyst. Your primary task is to answer the user's question with precision and clarity, synthesizing all relevant information from the provided context chunks into a single, comprehensive answer.

---
**RULES:**
1.  **Strictly Adhere to Context:** You MUST base your answer *only* on the information within the provided "--- CONTEXT ---" block. Do not use any external knowledge or make assumptions beyond what is written.
2.  **Handle Missing Information:** If the context does not contain the information needed to answer the question, you MUST respond with the exact phrase: "Based on the provided documents, I cannot find a definitive answer to this question."
3.  **Checklist for Key Details:** When formulating the answer, you **MUST** actively look for and include the following details if they are present in the context:
    - Specific **time periods** (e.g., 30 days, 24 months, 2 years)
    - **Monetary amounts, percentages, or limits** (e.g., INR 50,000, 5%, 1% of Sum Insured)
    - **Key conditions, eligibility criteria, exceptions, or exclusions.**
    - **Definitions** of specific terms (e.g., what constitutes a 'Hospital').
4.  **Synthesize, Don't Just Repeat:** Synthesize information from multiple context chunks if necessary. Do not just copy-paste sentences. Rephrase the information in clear, natural language to form a single, coherent paragraph. Avoid bullet points unless the user's question explicitly asks for a list.

---
**EXAMPLE OF EXCELLENT OUTPUT:**

--- CONTEXT ---
Context 1: A Hospital is an institution which has at least 15 inpatient beds in towns with a population of more than ten lacs.
Context 2: A Hospital must have qualified nursing staff under its employment round the clock and a fully equipped operation theatre of its own. It also must maintain daily records of patients.
Context 3: In towns having a population of less than ten lacs, a Hospital must have at least 10 inpatient beds.
--- END CONTEXT ---

--- QUESTION ---
How does the policy define a 'Hospital'?
--- END QUESTION ---

**Answer:** A hospital is defined as an institution with at least 10 inpatient beds (in towns with a population below ten lakhs) or 15 beds (in all other places), with qualified nursing staff and medical practitioners available 24/7, a fully equipped operation theatre, and which maintains daily records of patients.

---
**USER'S REQUEST:**

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
    print(f"🤖 Loading embedding model '{MODEL_NAME}'...")
    ml_models['embedding_model'] = SentenceTransformer(
        MODEL_NAME, device='cpu', cache_folder=CACHE_DIR)
    print("✅ Embedding model loaded.")
    yield
    print("👋 Application shutting down...")
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
    results: List[Answer] = None
    processing_time_seconds: float = None

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
    """Connects to Weaviate with a retry mechanism."""
    print(f"\n🔗 Connecting to Weaviate at {WEAVIATE_HOST}:{WEAVIATE_PORT}...")
    for i in range(5):  # 5 retries
        try:
            client = weaviate.connect_to_local(
                host=WEAVIATE_HOST,
                port=WEAVIATE_PORT,
                grpc_port=WEAVIATE_GRPC_PORT,
            )
            if client.is_ready():
                print("✅ Weaviate is ready!")
                return client
            client.close()
            await asyncio.sleep(3)
        except Exception as e:
            print(
                f"⏳ Weaviate not ready, retrying... (Attempt {i+1}/5). Error: {e}")
            await asyncio.sleep(3)
    raise ConnectionError(
        "❌ Could not connect to Weaviate after multiple retries.")


async def fetch_and_parse_pdf(api_url: str, file_url: str) -> List[str]:
    """Downloads a PDF and sends it to a parsing API to get text chunks."""
    print(f"\n⬇️  Downloading file from: {file_url}")
    try:
        pdf_response = requests.get(file_url)
        pdf_response.raise_for_status()
    except requests.RequestException as e:
        raise HTTPException(
            status_code=400, detail=f"Failed to download document from URL: {e}")

    filename = os.path.basename(urlparse(file_url).path)
    encoder = MultipartEncoder(
        fields={'file': (filename, pdf_response.content, 'application/pdf')})

    print(f"➡️  Sending '{filename}' to parsing API at {api_url}...")
    try:
        response = requests.post(api_url, data=encoder, headers={
                                 'Content-Type': encoder.content_type})
        response.raise_for_status()
    except requests.RequestException as e:
        raise HTTPException(
            status_code=503, detail=f"Document parsing service failed: {e}")

    blocks = response.json().get('return_dict', {}).get('result', {}).get('blocks', [])
    if not blocks:
        raise HTTPException(
            status_code=422, detail="No text chunks were found in the parsed document.")

    chunk_texts = [" ".join(block.get('sentences', [])) for block in blocks]
    print(f"✅ Parsed document into {len(chunk_texts)} text chunks.")
    return chunk_texts


async def ingest_to_weaviate(client: weaviate.WeaviateClient, collection_name: str, chunks: List[str]):
    model = ml_models['embedding_model']
    print(f"\n🧠 Generating embeddings for {len(chunks)} chunks...")
    embeddings = model.encode(chunks, show_progress_bar=True)

    # --- DEBUG PRINT 1: EMBEDDINGS RECEIVED ---
    print("\n" + "="*50)
    print("1. EMBEDDINGS RECEIVED")
    print("="*50)
    print(f"Shape of embeddings array: {embeddings.shape}")
    print("="*50 + "\n")
    # ----------------------------------------

    if client.collections.exists(collection_name):
        client.collections.delete(collection_name)
    client.collections.create(
        name=collection_name,
        vectorizer_config=wvc.Configure.Vectorizer.none(),
        properties=[wvc.Property(name="content", data_type=wvc.DataType.TEXT)]
    )
    print(f"✅ Created Weaviate collection: '{collection_name}'")
    policy_collection = client.collections.get(collection_name)
    print(f"🚀 Pushing {len(chunks)} objects to Weaviate...")
    with policy_collection.batch.dynamic() as batch:
        for i, text in enumerate(tqdm(chunks, desc="Batching objects")):
            batch.add_object(
                properties={"content": text},
                uuid=uuid.uuid4(),
                vector=embeddings[i].tolist()
            )
    if len(policy_collection.batch.failed_objects) > 0:
        print(
            f"⚠️ Failed to push {len(policy_collection.batch.failed_objects)} objects.")
    print(f"✅ Pushed {len(chunks)} objects to Weaviate successfully.")


async def generate_gemini_response_httpx(session: httpx.AsyncClient, prompt: str) -> str:
    api_key = random.choice(GEMINI_API_KEY_LIST)
    headers = {'Content-Type': 'application/json'}
    payload = {'contents': [{'parts': [{'text': prompt}]}]}

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
        return data['candidates'][0]['content']['parts'][0]['text']
    except httpx.HTTPStatusError as e:
        error_msg = f"Gemini API error: {e.response.status_code} - {e.response.text}"
        print(f"❌ {error_msg}")
        return error_msg
    except (KeyError, IndexError) as e:
        error_msg = f"Error parsing Gemini response: {e}. Response: {response.text}"
        print(f"❌ {error_msg}")
        return error_msg
    except Exception as e:
        error_msg = f"An unexpected error occurred during Gemini call: {e}"
        print(f"❌ {error_msg}")
        return error_msg


async def process_single_question(
    question: str,
    collection_name: str,
    weaviate_client: weaviate.WeaviateClient,
    http_session: httpx.AsyncClient
) -> Answer:
    print(f"🔍 Processing question: '{question}'")
    embedding_model = ml_models['embedding_model']
    query_vector = embedding_model.encode(question).tolist()
    policy_collection = weaviate_client.collections.get(collection_name)
    response = policy_collection.query.near_vector(
        near_vector=query_vector, limit=TOP_K_RESULTS, return_properties=[
            "content"]
    )
    context_chunks = [obj.properties['content'] for obj in response.objects]
    if not context_chunks:
        answer_text = "Based on the provided documents, I cannot find any relevant information to answer this question."
    else:
        context_str = "\n\n---\n\n".join(context_chunks)
        prompt = PROMPT_TEMPLATE.format(context=context_str, question=question)
        answer_text = await generate_gemini_response_httpx(http_session, prompt)
    return Answer(question=question, answer=answer_text)

# --- 4. FastAPI Endpoint ---


@router.post("/hackrx/run", response_model=QueryResponse, tags=["RAG Pipeline"])
async def process_document_and_answer_questions(
    request: QueryRequest = Body(...),
    token: str = Depends(verify_token)
):
    start_time = time.time()
    weaviate_client = None
    collection_name = f"Doc_{uuid.uuid4().hex}"

    print("\n" + "="*50)
    print("� TOKEN VERIFICATION SUCCESSFUL")
    print("�🚀 STARTING DOCUMENT PROCESSING")
    print(f"📄 Document URL: {request.documents}")
    print(f"❓ Number of questions: {len(request.questions)}")
    print("="*50 + "\n")
    try:
        weaviate_client = await connect_to_weaviate()
        chunks = await fetch_and_parse_pdf(PARSING_API_URL, request.documents)
        await ingest_to_weaviate(weaviate_client, collection_name, chunks)
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

        # Just return answers as requested
        return QueryResponse(answers=answer_texts)
    except Exception as e:
        print(f"❌ A critical error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if weaviate_client and weaviate_client.is_connected():
            if weaviate_client.collections.exists(collection_name):
                weaviate_client.collections.delete(collection_name)
                print(
                    f"\n✅ Cleaned up and deleted collection: '{collection_name}'")
            weaviate_client.close()
            print("✅ Weaviate client connection closed.")

# Include the router in the main app
app.include_router(router)

# --- Main Execution Block ---
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
