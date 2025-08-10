# Setup and Installation Guide

This document provides step-by-step instructions for setting up and running the HackRx RAG API system.

## Prerequisites

- Python 3.9+ installed
- Docker and Docker Compose installed (for running Weaviate)
- Git for cloning the repository
- At least 8GB of free RAM (16GB+ recommended)
- 10GB+ of free disk space

## Environment Setup

### 1. Clone the Repository

```bash
git clone https://github.com/nex7-7/hackrx-repo.git
cd hackrx-repo
```

### 2. Create and Activate a Virtual Environment

```bash
# For Linux/MacOS
python -m venv .venv
source .venv/bin/activate

# For Windows
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Create .env File

Create a `.env` file in the root directory with the following configuration:

```
# Weaviate Configuration
WEAVIATE_HOST=weaviate
WEAVIATE_PORT=8080
WEAVIATE_GRPC_PORT=50051

# LLM API Keys
GEMINI_API_KEYS=your_gemini_api_key1,your_gemini_api_key2
MISTRAL_API_KEY=your_mistral_api_key
GITHUB_MODELS_API_KEY=your_github_models_api_key

# Model Configuration
EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
RERANKER_MODEL=BAAI/bge-reranker-base
GEMINI_MODEL_NAME=gemini-2.5-flash

# Authentication
AUTH_TOKEN=fd8defb3118175da9553e106c05f40bc476971f0b46a400db1e625eaffa1fc08

# Logging Configuration
LOG_LEVEL=INFO
```

Replace the API keys with your actual keys. At minimum, you need at least one Gemini API key.

## Running the Application

### Option 1: Using Docker Compose (Recommended for Production)

This method starts both the RAG API and Weaviate database in containers:

```bash
# Build and start all services
docker-compose up -d

# Check logs
docker-compose logs -f rag-api
```

The API will be available at http://localhost:8000

### Option 2: Running Separately (Better for Development)

#### Step 1: Start Weaviate using Docker

```bash
docker-compose up -d weaviate
```

#### Step 2: Update .env for Local Development

Change the `WEAVIATE_HOST` in your `.env` file:

```
WEAVIATE_HOST=localhost
```

#### Step 3: Run the API Locally

```bash
# Set up file logging
mkdir -p logs

# Run the API with auto-reload for development
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

The API will be available at http://localhost:8001

## Accessing the API

### Authentication

All API endpoints require authentication using the Bearer token specified in your `.env` file:

```
Authorization: Bearer fd8defb3118175da9553e106c05f40bc476971f0b46a400db1e625eaffa1fc08
```

### API Endpoints

#### Process Document and Answer Questions

```
POST /api/v1/hackrx/run
```

Request body:
```json
{
  "documents": "https://example.com/document.pdf",
  "questions": [
    "What is the main topic?",
    "What are the key findings?"
  ]
}
```

Response:
```json
{
  "answers": [
    "The main topic is...",
    "The key findings are..."
  ]
}
```

#### Test LLM APIs

```
POST /api/v1/gemini
POST /api/v1/mistral
POST /api/v1/github-models
```

Request body:
```json
{
  "prompt": "What is the capital of France?"
}
```

Response:
```json
{
  "response": "The capital of France is Paris."
}
```

#### Clear Weaviate Collections

```
POST /api/v1/clear-weaviate
```

Response:
```json
{
  "message": "All collections deleted. Weaviate is now empty."
}
```

## Monitoring and Logs

### Log Files

Logs are stored in the `logs` directory with the naming convention `rag_api_YYYYMMDD.log`.

### Docker Container Logs

```bash
# View logs for the RAG API container
docker-compose logs -f rag-api

# View logs for the Weaviate container
docker-compose logs -f weaviate
```

### Resource Monitoring

For production deployments, consider monitoring:
- CPU and memory usage of containers
- Disk usage, especially for the Weaviate volume
- API response times and error rates

## Troubleshooting

### Common Issues and Solutions

#### 1. Weaviate Connection Issues

If the API cannot connect to Weaviate:

```bash
# Check if Weaviate container is running
docker ps | grep weaviate

# Restart Weaviate container
docker-compose restart weaviate

# Check Weaviate logs for errors
docker-compose logs weaviate
```

#### 2. Model Loading Errors

If you encounter errors loading models:

- Check internet connectivity
- Ensure the `huggingface_cache` directory has write permissions
- Try downloading the models manually:

```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-base-en-v1.5', cache_folder='./huggingface_cache')"
```

#### 3. NLTK Resource Errors

If you encounter NLTK errors, run:

```bash
python -c "import nltk; nltk.download('punkt')"
```

#### 4. API Key Issues

If LLM API calls fail:
- Check your API keys in `.env`
- Ensure API keys have proper access permissions
- Check for rate limits or usage quotas

## Next Steps

After successful setup:
- Try processing different document types
- Test with various question formats
- Explore advanced configurations in the code
- Read the [File Structure and Architecture](file_structure.md) document to understand the codebase better
