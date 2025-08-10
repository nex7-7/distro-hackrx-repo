# API Reference

This document provides detailed information about the API endpoints available in the HackRx RAG API system.

## Base URL

When running locally or in Docker:
```
http://localhost:8000/api/v1
```

## Authentication

All API endpoints require authentication using a Bearer token:

```
Authorization: Bearer fd8defb3118175da9553e106c05f40bc476971f0b46a400db1e625eaffa1fc08
```

The default token is defined in the `.env` file as `AUTH_TOKEN`.

## API Endpoints

### RAG Pipeline

#### Process Document and Answer Questions

```
POST /api/v1/hackrx/run
```

Process a document from a URL and answer questions about its content.

**Request Body**:

```json
{
  "documents": "string",  // URL to the document
  "questions": [          // List of questions to answer
    "string"
  ]
}
```

**Example**:

```json
{
  "documents": "https://example.com/policy.pdf",
  "questions": [
    "What is the refund policy?",
    "How long is the warranty period?"
  ]
}
```

**Response**:

```json
{
  "answers": [
    "string",  // Answer to first question
    "string"   // Answer to second question
  ]
}
```

**Example Response**:

```json
{
  "answers": [
    "The refund policy allows returns within 30 days of purchase with a valid receipt.",
    "The warranty period is 12 months from the date of purchase for all electronic items."
  ]
}
```

**Status Codes**:
- `200 OK`: Successfully processed document and generated answers
- `400 Bad Request`: Invalid request format
- `401 Unauthorized`: Invalid or missing authentication token
- `500 Internal Server Error`: Server-side error during processing

### LLM Testing

#### Test Gemini API

```
POST /api/v1/gemini
```

Test the Gemini LLM API with a direct prompt.

**Request Body**:

```json
{
  "prompt": "string",         // Prompt to send to the LLM
  "questions": ["string"]     // Optional list of additional questions
}
```

**Response**:

```json
{
  "response": "string"        // LLM-generated response
}
```

#### Test Mistral API

```
POST /api/v1/mistral
```

Test the Mistral LLM API with a direct prompt.

**Request Body**:

```json
{
  "prompt": "string",         // Prompt to send to the LLM
  "questions": ["string"]     // Optional list of additional questions
}
```

**Response**:

```json
{
  "response": "string"        // LLM-generated response
}
```

#### Test GitHub Models API

```
POST /api/v1/github-models
```

Test the GitHub Models LLM API with a direct prompt.

**Request Body**:

```json
{
  "prompt": "string",         // Prompt to send to the LLM
  "questions": ["string"]     // Optional list of additional questions
}
```

**Response**:

```json
{
  "response": "string"        // LLM-generated response
}
```

### Weaviate Admin

#### Clear Weaviate Collections

```
POST /api/v1/clear-weaviate
```

Delete all collections in the Weaviate database.

**Request Body**: None

**Response**:

```json
{
  "message": "string"        // Confirmation message
}
```

**Example Response**:

```json
{
  "message": "All collections deleted. Weaviate is now empty."
}
```

## Data Models

### QueryRequest

```json
{
  "documents": "string",     // URL to the document
  "questions": [             // List of questions to answer
    "string"
  ]
}
```

### QueryResponse

```json
{
  "answers": [               // List of answers corresponding to questions
    "string"
  ]
}
```

### LLMTestRequest

```json
{
  "prompt": "string",        // Prompt to send to the LLM
  "questions": [             // Optional list of additional questions
    "string"
  ]
}
```

### LLMTestResponse

```json
{
  "response": "string"       // LLM-generated response
}
```

### ClearWeaviateResponse

```json
{
  "message": "string"        // Confirmation message
}
```

## Error Handling

Errors are returned in the following format:

```json
{
  "detail": "string"         // Error description
}
```

Common error scenarios:

1. **Invalid Authentication**:
   ```json
   {
     "detail": "Invalid authorization token"
   }
   ```

2. **Document Processing Error**:
   ```json
   {
     "detail": "Failed to process document: invalid file format"
   }
   ```

3. **LLM API Error**:
   ```json
   {
     "detail": "Failed to generate response from LLM API: rate limit exceeded"
   }
   ```

4. **Weaviate Connection Error**:
   ```json
   {
     "detail": "Failed to connect to Weaviate database"
   }
   ```

## Rate Limits

Default rate limits:
- Maximum 10 requests per minute per client
- Maximum 5 concurrent requests per client
- Maximum document size: 50MB
- Maximum 10 questions per request

## API Versioning

The API version is specified in the URL path: `/api/v1/...`

Future versions will use incrementing version numbers: `/api/v2/...`, etc.

## API Examples

### cURL Examples

#### Process Document with Questions

```bash
curl -X POST "http://localhost:8000/api/v1/hackrx/run" \
  -H "Authorization: Bearer fd8defb3118175da9553e106c05f40bc476971f0b46a400db1e625eaffa1fc08" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": "https://example.com/document.pdf",
    "questions": [
      "What is the main topic?",
      "What are the key findings?"
    ]
  }'
```

#### Test Gemini API

```bash
curl -X POST "http://localhost:8000/api/v1/gemini" \
  -H "Authorization: Bearer fd8defb3118175da9553e106c05f40bc476971f0b46a400db1e625eaffa1fc08" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is the capital of France?"
  }'
```

#### Clear Weaviate Collections

```bash
curl -X POST "http://localhost:8000/api/v1/clear-weaviate" \
  -H "Authorization: Bearer fd8defb3118175da9553e106c05f40bc476971f0b46a400db1e625eaffa1fc08"
```

### Python Examples

#### Process Document with Questions

```python
import requests
import json

url = "http://localhost:8000/api/v1/hackrx/run"
headers = {
    "Authorization": "Bearer fd8defb3118175da9553e106c05f40bc476971f0b46a400db1e625eaffa1fc08",
    "Content-Type": "application/json"
}
payload = {
    "documents": "https://example.com/document.pdf",
    "questions": [
        "What is the main topic?",
        "What are the key findings?"
    ]
}

response = requests.post(url, headers=headers, data=json.dumps(payload))
print(response.json())
```

#### Test Gemini API

```python
import requests
import json

url = "http://localhost:8000/api/v1/gemini"
headers = {
    "Authorization": "Bearer fd8defb3118175da9553e106c05f40bc476971f0b46a400db1e625eaffa1fc08",
    "Content-Type": "application/json"
}
payload = {
    "prompt": "What is the capital of France?"
}

response = requests.post(url, headers=headers, data=json.dumps(payload))
print(response.json())
```

## Best Practices

1. **Error Handling**: Always check for and handle error responses from the API
2. **Rate Limiting**: Implement exponential backoff for retries on rate limit errors
3. **Document Size**: Keep documents under 50MB for optimal processing
4. **Questions**: Formulate clear, specific questions for better answers
5. **Authentication**: Keep the authentication token secure
6. **Caching**: Consider caching responses for identical document/question pairs
7. **Monitoring**: Monitor API response times and error rates for production use
