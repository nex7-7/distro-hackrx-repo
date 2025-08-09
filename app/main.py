"""
Main FastAPI application for the RAG Application API.

This module sets up the FastAPI application with proper configuration,
middleware, error handling, and routing for the RAG pipeline endpoints.
"""

import time
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, Request, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi

from app import APP_NAME, APP_VERSION, API_V1_PREFIX
from app.api.endpoints import hackrx_router
from app.utils.logger import get_logger
from app.utils.exceptions import RAGApplicationError
from app.services.embedding_service import embedding_service
from app.services.vector_store import vector_store
from app.services.reranking_service import reranking_service
from config.settings import settings

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager for startup and shutdown tasks.
    
    Handles:
    - Service initialization on startup
    - Resource cleanup on shutdown
    """
    # Startup
    logger.info("Starting RAG Application API", version=APP_VERSION)
    
    try:
        # Initialize services
        logger.info("Initializing services...")
        
        # Connect to vector store
        await vector_store.connect()
        logger.info("Vector store connected")
        
        # Preload embedding model
        await embedding_service.preload_model()
        logger.info("Embedding model preloaded")
        
        # Preload reranking model
        await reranking_service.load_model()
        logger.info("Reranking model preloaded")
        
        logger.info("RAG Application API started successfully")
        
        yield
        
    except Exception as e:
        logger.critical(f"Failed to start application: {str(e)}")
        raise
    finally:
        # Shutdown
        logger.info("Shutting down RAG Application API...")
        logger.info("RAG Application API shutdown complete")


# Create FastAPI application
app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description="High-performance RAG Application API for document processing and question answering",
    docs_url=None,  # Disable default docs
    redoc_url=None,  # Disable redoc
    openapi_url=f"{API_V1_PREFIX}/openapi.json",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    """Request logging middleware."""
    start_time = time.time()
    
    # Log request
    logger.info("HTTP Request",
               method=request.method,
               url=str(request.url),
               client_ip=request.client.host if request.client else "unknown")
    
    try:
        response = await call_next(request)
        
        # Log response
        process_time = time.time() - start_time
        logger.info("HTTP Response",
                   status_code=response.status_code,
                   process_time=f"{process_time:.3f}s")
        
        # Add timing header
        response.headers["X-Process-Time"] = str(process_time)
        
        return response
        
    except Exception as e:
        process_time = time.time() - start_time
        logger.error("HTTP Request failed",
                    error=str(e),
                    process_time=f"{process_time:.3f}s")
        raise


@app.exception_handler(RAGApplicationError)
async def rag_exception_handler(request: Request, exc: RAGApplicationError):
    """Handle RAG application specific exceptions."""
    logger.error("RAG Application Error",
                error_code=exc.error_code,
                error_message=exc.message,  # Changed from 'message' to 'error_message'
                details=exc.details)
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "error_code": exc.error_code,
                "message": exc.message,
                "details": exc.details
            },
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "request_id": None  # Could add request ID tracking
        }
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    logger.warning("HTTP Exception",
                  status_code=exc.status_code,
                  detail=exc.detail,
                  url=str(request.url))
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": exc.detail,
                "error_code": f"HTTP_{exc.status_code}"
            },
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    logger.exception("Unexpected error occurred",
                    error_type=type(exc).__name__,
                    url=str(request.url))
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "message": "An unexpected error occurred",
                "error_code": "INTERNAL_SERVER_ERROR"
            },
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    )


# Custom OpenAPI documentation endpoint
@app.get(f"{API_V1_PREFIX}/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    """Custom Swagger UI endpoint."""
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=f"{APP_NAME} - Documentation",
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css",
    )


def custom_openapi():
    """Custom OpenAPI schema generation."""
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title=APP_NAME,
        version=APP_VERSION,
        description="High-performance RAG Application API for document processing and question answering",
        routes=app.routes,
    )
    
    # Add security scheme
    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "string"
        }
    }
    
    # Add global security requirement
    openapi_schema["security"] = [{"BearerAuth": []}]
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


# Health check endpoint
@app.get("/health", tags=["Health"])
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint to verify service status.
    
    Returns:
        Dict[str, Any]: Service health information
    """
    try:
        # Check service statuses
        vector_stats = vector_store.get_stats()
        embedding_info = embedding_service.get_model_info()
        reranking_info = reranking_service.get_model_info()
        
        return {
            "status": "healthy",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "version": APP_VERSION,
            "services": {
                "vector_store": vector_stats.get("status", "unknown"),
                "embedding_service": "loaded" if embedding_info["is_loaded"] else "not_loaded",
                "reranking_service": "loaded" if reranking_info["model_loaded"] else "not_loaded",
                "llm_service": "available"
            },
            "metrics": {
                "total_chunks": vector_stats.get("total_chunks", 0),
                "embedding_dimension": embedding_info["embedding_dimension"],
                "reranking_model": reranking_info["model_name"]
            }
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "error": str(e)
        }


# Include API routers
app.include_router(hackrx_router, prefix=API_V1_PREFIX)

# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": APP_NAME,
        "version": APP_VERSION,
        "docs_url": f"{API_V1_PREFIX}/docs",
        "health_url": "/health"
    }
