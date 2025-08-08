"""
Authentication utilities for the RAG Application API.

This module provides Bearer token authentication as specified in the requirements.
The API accepts only one hardcoded token for security.
"""

from typing import Optional
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from config.settings import settings
from app.utils.logger import get_logger
from app.utils.exceptions import AuthenticationError, create_error

logger = get_logger(__name__)

# Bearer token security scheme
security = HTTPBearer()


async def verify_bearer_token(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> str:
    """
    Verify the Bearer token against the hardcoded token.
    
    Args:
        credentials: HTTP authorization credentials
        
    Returns:
        str: The verified token
        
    Raises:
        HTTPException: If token is invalid
    """
    try:
        token = credentials.credentials
        
        # Check against hardcoded token
        if token != settings.api_bearer_token:
            logger.warning("Invalid bearer token attempt", 
                          provided_token_prefix=token[:10] if token else "empty")
            
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid bearer token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        logger.debug("Bearer token verified successfully")
        return token
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Token verification failed: {str(e)}")
        
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed",
            headers={"WWW-Authenticate": "Bearer"},
        )


def get_current_user(token: str = Depends(verify_bearer_token)) -> dict:
    """
    Get current user information (placeholder for future user management).
    
    Args:
        token: Verified bearer token
        
    Returns:
        dict: User information
    """
    return {
        "authenticated": True,
        "token_prefix": token[:10],
        "permissions": ["hackrx:run"]
    }
