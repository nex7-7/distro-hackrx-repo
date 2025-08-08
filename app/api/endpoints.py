"""
API endpoints for the RAG Application.

This module defines the FastAPI router with the main HackRX endpoint
for processing documents and answering questions.
"""

import time
from typing import Dict, Any

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials

from app.models.schemas import HackRXRequest, HackRXResponse
from app.core.pipeline import process_hackrx_request
from app.utils.auth import verify_bearer_token, get_current_user
from app.utils.logger import get_logger
from app.utils.exceptions import RAGApplicationError

logger = get_logger(__name__)

# Create router
hackrx_router = APIRouter(
    prefix="/hackrx",
    tags=["HackRX"],
    dependencies=[Depends(verify_bearer_token)]
)


@hackrx_router.post(
    "/run",
    response_model=HackRXResponse,
    summary="Process document and answer questions",
    description="Main endpoint for processing documents and generating answers using the RAG pipeline"
)
async def run_hackrx(
    request: HackRXRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> HackRXResponse:
    """
    Process document and answer questions using the RAG pipeline.
    
    This endpoint executes the complete 6-stage RAG pipeline:
    1. Document Ingestion & Preprocessing
    2. Data Chunking & Vectorization
    3. Query Analysis & Classification  
    4. Context Retrieval
    5. Response Generation
    6. Final Response Formatting
    
    Args:
        request: HackRX request with document URL and questions
        current_user: Authenticated user information
        
    Returns:
        HackRXResponse: Response with answers for each question
        
    Raises:
        HTTPException: If processing fails
    """
    start_time = time.time()
    
    logger.info("HackRX request received",
               document_url=str(request.documents),
               question_count=len(request.questions),
               user=current_user.get("token_prefix", "unknown"))
    
    try:
        # Process request through RAG pipeline
        response = await process_hackrx_request(request)
        
        duration = time.time() - start_time
        logger.info("HackRX request completed successfully",
                   duration=f"{duration:.3f}s",
                   answer_count=len(response.answers))
        
        return response
        
    except RAGApplicationError as e:
        # RAG-specific errors are handled by the global exception handler
        logger.error(f"RAG processing failed: {str(e)}")
        raise
        
    except Exception as e:
        # Unexpected errors
        logger.exception("Unexpected error in HackRX endpoint")
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while processing your request"
        )
