# Testing Guide

This document outlines how to run tests for the HackRx RAG API system and provides information about the test cases implemented.

## Test Setup

### Prerequisites

Before running the tests, ensure you have:

1. Installed all dependencies, including dev dependencies:
   ```bash
   pip install -r requirements.txt
   pip install pytest pytest-cov httpx pytest-asyncio
   ```

2. Set up a running instance of Weaviate (for integration tests):
   ```bash
   docker-compose up -d weaviate
   ```

3. Created a `.env.test` file with test configuration:
   ```
   WEAVIATE_HOST=localhost
   WEAVIATE_PORT=8080
   WEAVIATE_GRPC_PORT=50051
   GEMINI_API_KEYS=your_test_api_key
   EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
   RERANKER_MODEL=BAAI/bge-reranker-base
   ```

## Running Tests

### Running All Tests

```bash
pytest test/ -v
```

### Running Specific Test Files

```bash
# Run HTTP integration tests
pytest test/test_http_integration.py -v

# Run Weaviate and reranker tests
pytest test/test_weaviate_rerank.py -v
```

### Running Tests with Coverage

```bash
pytest test/ --cov=components --cov-report=term --cov-report=html
```

This will generate a coverage report in the terminal and a detailed HTML report in the `htmlcov` directory.

## Test Categories

### Unit Tests

These test individual functions and methods in isolation.

### Integration Tests

These test interactions between different components, such as:
- HTTP client integration tests
- Weaviate database integration tests
- LLM API integration tests

### End-to-End Tests

These test the complete RAG pipeline, from document ingestion to answer generation.

## Test Files and Their Purpose

### `test/test_http_integration.py`

**Purpose**: Tests the HTTP tools functionality.

**Key Test Cases**:
1. `test_http_request_get`: Tests GET requests to external endpoints
2. `test_http_request_post`: Tests POST requests with various payload types
3. `test_http_tool_error_handling`: Tests proper handling of HTTP errors

**Running the Tests**:
```bash
pytest test/test_http_integration.py::test_http_request_get -v
```

### `test/test_weaviate_rerank.py`

**Purpose**: Tests Weaviate integration and reranking functionality.

**Key Test Cases**:
1. `test_weaviate_connection`: Tests successful connection to Weaviate
2. `test_ingest_to_weaviate`: Tests document ingestion into Weaviate
3. `test_hybrid_search`: Tests hybrid search functionality
4. `test_reranker`: Tests reranking functionality

**Running the Tests**:
```bash
pytest test/test_weaviate_rerank.py -v
```

### `test/zip.py`

**Purpose**: Utility for testing ZIP file handling.

**Key Features**:
1. Functions for creating test ZIP files
2. Functions for testing recursive ZIP extraction

## Creating New Tests

### Creating Unit Tests

1. Create a new file in the `test` directory, following the naming convention `test_*.py`
2. Import the module and functions you want to test
3. Write test functions with clear assertions

Example:
```python
def test_semantic_chunk_texts():
    from components.chunking import semantic_chunk_texts
    
    # Test with simple input
    chunks = ["This is a test sentence. This is another test sentence."]
    result = semantic_chunk_texts(
        chunks, 
        embedding_model=mock_model, 
        model_name="test-model",
        similarity_threshold=0.8
    )
    
    assert len(result) > 0
    assert isinstance(result[0], str)
```

### Creating Integration Tests

For integration tests, you'll often need to use `pytest-asyncio` for testing asynchronous functions:

```python
import pytest
import asyncio

@pytest.mark.asyncio
async def test_connect_to_weaviate():
    from components.weaviate_db import connect_to_weaviate
    
    # Test connection
    client = await connect_to_weaviate("localhost", 8080, 50051)
    
    # Assertions
    assert client is not None
    assert client.is_ready()
```

### Creating Mocks

For tests that require external dependencies, use mocking:

```python
from unittest.mock import patch, MagicMock

def test_generate_gemini_response():
    with patch('components.gemini_api.httpx.AsyncClient') as mock_client:
        # Setup the mock
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"candidates": [{"content": {"parts": [{"text": "Test response"}]}}]}
        mock_client.return_value.__aenter__.return_value.post.return_value = mock_response
        
        # Call the function
        from components.gemini_api import generate_gemini_response_httpx
        import asyncio
        
        response = asyncio.run(generate_gemini_response_httpx(mock_client, "Test prompt"))
        
        # Assertions
        assert response == "Test response"
```

## Additional Test Resources

### Test Data

Test data files can be placed in a `test/data` directory:
- Sample PDFs for testing document ingestion
- Sample embeddings for testing vector operations
- Mock HTTP responses for testing API integrations

### Test Configuration

The test suite automatically uses the `.env.test` configuration file when available. This allows you to separate your test environment from your development environment.

## Best Practices for Testing

1. **Isolation**: Ensure each test is isolated and doesn't depend on other tests
2. **Naming**: Use descriptive test names that indicate what's being tested
3. **Fixtures**: Use pytest fixtures for common setup and teardown
4. **Mocking**: Use mocks for external dependencies to keep tests fast
5. **Coverage**: Aim for high test coverage, especially for critical components
6. **Edge Cases**: Include tests for edge cases and error scenarios
7. **Cleanup**: Clean up after tests, especially when creating data in Weaviate

## Continuous Integration

For CI/CD pipelines, the tests can be integrated into GitHub Actions or other CI platforms:

```yaml
# Example GitHub Actions workflow
name: Run Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install pytest pytest-cov pytest-asyncio
      - name: Start Weaviate
        run: docker-compose up -d weaviate
      - name: Run tests
        run: pytest test/ --cov=components --cov-report=xml
      - name: Upload coverage report
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
```

## Troubleshooting Tests

### Common Issues

1. **Weaviate Connection Failures**:
   - Ensure Weaviate is running: `docker-compose ps`
   - Check Weaviate logs: `docker-compose logs weaviate`
   - Verify host and port settings

2. **Timeout Errors**:
   - Increase timeout parameters in pytest.ini:
     ```ini
     [pytest]
     asyncio_timeout = 30
     ```

3. **Rate Limiting in API Tests**:
   - Use mock responses for LLM API tests
   - Implement retries with exponential backoff

4. **Resource Cleanup**:
   - Use pytest fixtures for proper cleanup:
     ```python
     @pytest.fixture(autouse=True)
     async def clear_weaviate_after_test():
         yield  # Run the test
         # Cleanup code here
     ```

## Example: Complete Test Case

Here's an example of a complete test case for testing the document ingestion pipeline:

```python
import pytest
import os
import tempfile
from unittest.mock import patch
import asyncio

@pytest.mark.asyncio
async def test_full_document_ingestion():
    """Test the full document ingestion pipeline with a sample PDF"""
    
    # Import required components
    from components.ingest_engine import ingest_from_file
    from components.chunking import semantic_chunk_texts
    from components.embeddings import create_embeddings
    from components.weaviate_db import connect_to_weaviate, ingest_to_weaviate
    
    # Create a sample PDF file for testing
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(b"%PDF-1.7\nSample test content\n%%EOF")
        tmp_path = tmp.name
    
    try:
        # Mock embedding model
        mock_model = pytest.MockSentenceTransformer()
        
        # Test ingest_from_file
        chunks = ingest_from_file(tmp_path)
        assert len(chunks) > 0
        
        # Test semantic chunking
        semantic_chunks = semantic_chunk_texts(
            chunks, 
            embedding_model=mock_model,
            model_name="test-model",
            similarity_threshold=0.8
        )
        assert len(semantic_chunks) > 0
        
        # Test embedding creation
        embeddings = create_embeddings(
            semantic_chunks,
            embedding_model=mock_model,
            model_name="test-model"
        )
        assert len(embeddings) == len(semantic_chunks)
        
        # Test Weaviate ingestion
        client = await connect_to_weaviate("localhost", 8080, 50051)
        assert client is not None
        
        collection_name = "TestCollection"
        result = await ingest_to_weaviate(
            client,
            collection_name,
            semantic_chunks,
            embeddings
        )
        assert result is not None
        
        # Clean up the collection
        client.schema.delete_class(collection_name)
        
    finally:
        # Clean up the temporary file
        os.unlink(tmp_path)
```

This test covers the entire ingestion pipeline from file parsing to storage in the vector database.
