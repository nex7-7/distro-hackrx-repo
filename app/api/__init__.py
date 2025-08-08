"""
API package for the RAG Application.

This package contains FastAPI routers and endpoint definitions,
handling HTTP requests and responses for the RAG pipeline.
"""

from .endpoints import hackrx_router

__all__ = [
    "hackrx_router"
]
