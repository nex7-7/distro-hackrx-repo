# RAG Application API

A high-performance, containerized RAG (Retrieval-Augmented Generation) Application API built with FastAPI, Weaviate, and Google Gemini.

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose installed
- Google Gemini API key

### Setup & Run

1. **Clone and navigate to the project:**
   ```bash
   cd hackrx-repo
   ```

2. **Configure environment variables:**
   ```bash
   # Copy and edit the .env file
   cp .env .env.local
   # Edit .env.local with your Google API key
   ```

3. **Start the application:**
   ```bash
   docker-compose up -d
   ```

4. **Check health status:**
   ```bash
   # Check if both services are running
   docker-compose ps
   
   # Check application health
   curl http://localhost:8000/health
   
   # Check Weaviate health
   curl http://localhost:8080/v1/.well-known/ready
   ```

5. **Access the API:**
   - **API Documentation**: http://localhost:8000/api/v1/docs
   - **Health Check**: http://localhost:8000/health
   - **Weaviate Console**: http://localhost:8080

## 📋 API Usage

### Authentication
All requests require a Bearer token:
```
Authorization: Bearer fd8defb3118175da9553e106c05f40bc476971f0b46a400db1e625eaffa1fc08
```

### Main Endpoint
**POST** `/api/v1/hackrx/run`

```json
{
    "documents": "https://example.com/document.pdf",
    "questions": [
        "What is the grace period for premium payment?",
        "What are the covered benefits?"
    ]
}
```

**Response:**
```json
{
    "answers": [
        "A grace period of thirty days is provided for premium payment...",
        "The covered benefits include medical expenses..."
    ]
}
```

## 🛠️ Development

### Running Locally (Without Docker)

1. **Set up Python environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Start Weaviate:**
   ```bash
   docker run -d \
     --name weaviate \
     -p 8080:8080 \
     -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true \
     -e PERSISTENCE_DATA_PATH=/var/lib/weaviate \
     semitechnologies/weaviate:1.22.4
   ```

3. **Run the application:**
   ```bash
   python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```

### Configuration

Key environment variables in `.env`:

| Variable | Description | Default |
|----------|-------------|---------|
| `GOOGLE_API_KEY` | Google Gemini API key | Required |
| `API_BEARER_TOKEN` | API authentication token | Required |
| `WEAVIATE_URL` | Weaviate database URL | `http://localhost:8080` |
| `LLM_MODEL` | Google model to use | `gemini-1.5-flash` |
| `LOG_LEVEL` | Logging level | `INFO` |

## 🔧 Architecture

### RAG Pipeline (6 Stages)
1. **Document Ingestion & Preprocessing** - Download and extract content from various file types
2. **Data Chunking & Vectorization** - Smart chunking with BAAI/bge-m3 embeddings
3. **Query Analysis & Classification** - LLM-powered query classification
4. **Context Retrieval** - Vector similarity search in Weaviate
5. **Response Generation** - Parallel LLM calls for answer generation
6. **Final Response Formatting** - Clean and format final responses

### Key Components
- **FastAPI** - Web framework with automatic OpenAPI documentation
- **Weaviate** - Vector database for semantic search
- **Google Gemini** - LLM for query classification and response generation
- **BAAI/bge-m3** - Embedding model for vectorization
- **Multiprocessing** - Concurrent operations for performance

## 📊 Monitoring

### Health Checks
- **Application**: `GET /health`
- **Weaviate**: `GET http://localhost:8080/v1/.well-known/ready`

### Logs
```bash
# View application logs
docker-compose logs -f rag-api

# View Weaviate logs
docker-compose logs -f weaviate

# View logs from host
tail -f logs/app.log
```

### Performance Monitoring
The application includes built-in performance logging for each pipeline stage.

## 🚦 Troubleshooting

### Common Issues

1. **Container fails to start:**
   ```bash
   # Check logs
   docker-compose logs rag-api
   
   # Restart services
   docker-compose restart
   ```

2. **Weaviate connection errors:**
   ```bash
   # Ensure Weaviate is healthy
   curl http://localhost:8080/v1/.well-known/ready
   
   # Check network connectivity
   docker-compose exec rag-api curl http://weaviate:8080/v1/.well-known/ready
   ```

3. **API authentication errors:**
   - Verify the Bearer token in your requests
   - Check that `API_BEARER_TOKEN` is set correctly in `.env`

4. **Document processing failures:**
   - Check file accessibility and format support
   - Review logs for specific error details
   - Ensure sufficient disk space for temporary files

### Reset Everything
```bash
# Stop and remove containers, networks, and volumes
docker-compose down -v

# Remove images (optional)
docker-compose down --rmi all

# Start fresh
docker-compose up -d
```

## 📄 License

This project is built according to the specifications in `copilot-instructions.md`.
