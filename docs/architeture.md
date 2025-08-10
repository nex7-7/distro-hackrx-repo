# RAG_BAJ System Architecture

## System Overview

The RAG_BAJ system is a sophisticated document processing and question-answering pipeline that leverages advanced NLP techniques, vector databases, and multiple LLM providers to deliver accurate, context-aware responses.

## High-Level Architecture

```mermaid
flowchart TD
    subgraph "Document Ingestion Layer"
        A[Document Input] --> B{Security Validation}
        B --> |Pass| C[File Type Detection]
        B --> |Fail| D[Security Error]
        C --> E[Parser Selection]
        E --> F[Content Extraction]
    end

    subgraph "Text Processing Layer"
        F --> G[Semantic Chunking]
        G --> H[Embedding Generation]
        H --> I[Vector Storage]
    end

    subgraph "Query Processing Layer"
        J[User Question] --> K[Query Embedding]
        K --> L[Hybrid Search]
        L --> M[Result Reranking]
        M --> N[Context Assembly]
    end

    subgraph "Response Generation Layer"
        N --> O[Prompt Engineering]
        O --> P{LLM Selection}
        P --> |Primary| Q[Gemini API]
        P --> |Fallback 1| R[Mistral API]
        P --> |Fallback 2| S[GitHub Models]
        Q --> T[Response]
        R --> T
        S --> T
    end

    subgraph "Storage & Infrastructure"
        U[Weaviate Vector DB]
        V[HuggingFace Cache]
        W[Logging System]
    end

    I --> U
    L --> U
    H --> V
    F --> W
    T --> W
```

## Component Architecture

### 1. FastAPI Application Layer (`main.py`)

**Role**: Central orchestrator and API gateway

- Handles HTTP requests and routing
- Manages application lifecycle and model loading
- Implements authentication and rate limiting
- Coordinates between all system components

**Key Features**:

- Async/await support for concurrent processing
- Pre-loaded ML models for optimal performance
- Comprehensive error handling and logging
- Health checks and monitoring endpoints

### 2. Document Ingestion Engine (`components/ingest_engine.py`)

**Role**: Unified document processing with security controls

```mermaid
graph LR
    A[Document URL] --> B[Security Check]
    B --> C{File Type}
    C --> |PDF| D[PyMuPDF Parser]
    C --> |DOCX| E[Unstructured Parser]
    C --> |Image| F[OCR Engine]
    C --> |Web| G[BeautifulSoup]
    C --> |ZIP| H[Recursive Handler]
    D --> I[Text Chunks]
    E --> I
    F --> I
    G --> I
    H --> I
```

**Security Features**:

- File size validation (1GB limit)
- Extension blacklisting for dangerous files
- ZIP bomb protection with depth limits
- Content sanitization and validation

### 3. Semantic Processing Pipeline

#### Chunking Strategy (`components/chunking.py`)

```mermaid
graph TD
    A[Raw Text] --> B[Sentence Tokenization]
    B --> C[Sentence Embeddings]
    C --> D[Similarity Calculation]
    D --> E{Similarity > Threshold?}
    E --> |Yes| F[Merge Sentences]
    E --> |No| G[Create New Chunk]
    F --> H[Check Size Limits]
    G --> H
    H --> I[Add Overlap]
    I --> J[Final Chunks]
```

**Algorithm Details**:

- Uses BGE embeddings for semantic similarity
- Configurable similarity threshold (default: 0.8)
- Size constraints: 3-12 sentences per chunk
- 10% overlap between consecutive chunks

#### Embedding Generation (`components/embeddings.py`)

```mermaid
graph LR
    A[Text Chunks] --> B[BGE Prompt Wrapping]
    B --> C[Batch Processing]
    C --> D[Vector Generation]
    D --> E[Normalization]
    E --> F[Quality Validation]
    F --> G[Output Vectors]
```

**Technical Details**:

- Model: BAAI/bge-base-en-v1.5 (768 dimensions)
- Prompt template optimization for retrieval quality
- Batch processing with progress tracking
- GPU acceleration when available

### 4. Vector Database Layer (`components/weaviate_db.py`)

**Architecture**:

```mermaid
graph TD
    A[Vector Embeddings] --> B[Collection Creation]
    B --> C[Batch Ingestion]
    C --> D[Index Building]
    D --> E[Query Interface]
    E --> F[Hybrid Search]
    F --> G[Result Ranking]
```

**Features**:

- Dynamic collection management
- Optimized batch insertion
- Connection pooling and retry logic
- Performance monitoring and metrics

### 5. Search & Retrieval System (`components/search.py`)

**Hybrid Search Architecture**:

```mermaid
graph LR
    A[Query] --> B[Vector Search]
    A --> C[BM25 Search]
    B --> D[Semantic Results]
    C --> E[Keyword Results]
    D --> F[Score Fusion]
    E --> F
    F --> G[Ranked Results]
```

**Implementation**:

- Alpha parameter controls vector/keyword balance
- Configurable result limits
- Score normalization and fusion
- Relevance scoring for downstream processing

### 6. Reranking System (`components/reranker_utils.py`)

**Cross-Encoder Reranking**:

```mermaid
graph TD
    A[Initial Results] --> B[Query-Document Pairs]
    B --> C[BGE Reranker Model]
    C --> D[Relevance Scores]
    D --> E[Score Thresholding]
    E --> F[Top-K Selection]
    F --> G[Reranked Results]
```

**Benefits**:

- Improved relevance over initial retrieval
- Cross-attention between query and documents
- Configurable score thresholds
- Performance diagnostics and monitoring

### 7. LLM Integration Layer (`components/gemini_api.py`)

**Multi-Provider Fallback Chain**:

```mermaid
graph TD
    A[Prompt] --> B{Gemini API}
    B --> |Success| C[Response]
    B --> |Failure| D{Mistral API}
    D --> |Success| C
    D --> |Failure| E{GitHub Models}
    E --> |Success| C
    E --> |Failure| F[Error Response]
```

**Features**:

- API key rotation for rate limiting
- Automatic failover between providers
- Response quality monitoring
- Timeout and retry handling

### 8. Prompt Engineering (`components/prompt_template.py`)

**Template Structure**:

```
System Instructions
├── Task Definition
├── Response Rules
├── Quality Guidelines
└── Safety Constraints

Context Section
├── Relevance-Scored Chunks
├── Source Attribution
└── Metadata Preservation

Query Section
├── User Question
├── Answer Format
└── Response Trigger
```

### 9. Logging & Monitoring (`components/utils/logger.py`)

**Structured Logging Architecture**:

```mermaid
graph LR
    A[System Events] --> B[Structured Logger]
    B --> C[JSON Formatting]
    C --> D[File Output]
    C --> E[Console Output]
    D --> F[Log Analysis]
    E --> G[Real-time Monitoring]
```

**Metrics Tracked**:

- Processing latencies by component
- API response times and errors
- Vector operations performance
- Memory and resource utilization
- Error rates and failure patterns

## Data Flow Architecture

### Complete Pipeline Flow

```mermaid
sequenceDiagram
    participant U as User
    participant API as FastAPI
    participant I as Ingestion
    participant C as Chunking
    participant E as Embeddings
    participant W as Weaviate
    participant S as Search
    participant R as Reranker
    participant L as LLM

    U->>API: POST /hackrx/run
    API->>I: Process document URL
    I->>I: Security validation
    I->>I: Parse & extract text
    I->>C: Raw text chunks
    C->>C: Semantic chunking
    C->>E: Chunked text
    E->>E: Generate embeddings
    E->>W: Store vectors

    loop For each question
        API->>E: Generate query embedding
        E->>S: Query vector
        S->>W: Hybrid search
        W->>S: Retrieved chunks
        S->>R: Initial results
        R->>R: Rerank by relevance
        R->>L: Top contexts + question
        L->>L: Generate response
        L->>API: Answer
    end

    API->>U: All answers
```

## Performance Characteristics

### Latency Breakdown

- Document ingestion: 2-10 seconds (varies by size/type)
- Semantic chunking: 1-3 seconds per document
- Embedding generation: 0.1-0.5 seconds per chunk
- Vector storage: 0.5-2 seconds for batch
- Search & retrieval: 100-500ms per query
- Reranking: 200-800ms for 25 candidates
- LLM generation: 1-5 seconds per answer

### Scalability Considerations

- Horizontal scaling via load balancers
- Vector database sharding for large collections
- Model serving optimization with GPU clusters
- Caching strategies for frequent queries
- Async processing for batch operations

## Security Architecture

### Defense in Depth

1. **Input Validation**: File type, size, and content validation
2. **Processing Isolation**: Sandboxed parsing and execution
3. **Resource Limits**: Memory, CPU, and time constraints
4. **API Security**: Authentication, rate limiting, CORS
5. **Data Protection**: Encryption in transit and at rest

### Threat Mitigation

- ZIP bomb detection and prevention
- Malicious file content sanitization
- API abuse protection via rate limiting
- Input injection prevention
- Resource exhaustion safeguards

## Deployment Architecture

### Container Strategy

```mermaid
graph TD
    A[Load Balancer] --> B[FastAPI Containers]
    B --> C[Weaviate Cluster]
    B --> D[Model Cache Volume]
    B --> E[Log Aggregation]
    C --> F[Persistent Storage]
    E --> G[Monitoring Stack]
```

### Infrastructure Requirements

- **Compute**: 4+ CPU cores, 16GB+ RAM per instance
- **GPU**: Optional for faster embedding generation
- **Storage**: SSD for vector database, shared volume for models
- **Network**: High bandwidth for document processing
- **Monitoring**: Prometheus, Grafana, ELK stack

## Quality Assurance

### Testing Strategy

- Unit tests for individual components
- Integration tests for pipeline flows
- Performance benchmarks and load testing
- Security vulnerability scanning
- API contract testing

### Monitoring & Alerting

- Real-time performance dashboards
- Error rate and latency alerts
- Resource utilization monitoring
- Quality score tracking
- User satisfaction metrics

This architecture provides a robust, scalable foundation for intelligent document processing and question answering, with built-in redundancy, security, and performance optimization.
