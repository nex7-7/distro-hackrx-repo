"""
Development entry point for the RAG Application API.

This script runs the FastAPI application using uvicorn with development settings.
Use this for local development when running Weaviate separately in Docker.

Usage:
    python run_dev.py

Prerequisites:
    1. Start Weaviate: docker compose up weaviate
    2. Activate virtual environment: .venv\\Scripts\\Activate.ps1 (Windows)
    3. Install dependencies: pip install -r requirements.txt
    4. Set up .env file with required variables
"""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    import uvicorn
    from config.settings import settings
    from app.utils.logger import get_logger

    # Configure logger for startup
    logger = get_logger(__name__)

    def main():
        """Main entry point for development server."""
        
        # Environment validation
        required_env_vars = [
            "API_BEARER_TOKEN",
            "GOOGLE_API_KEY",
        ]
        
        missing_vars = [var for var in required_env_vars if not os.getenv(var)]
        if missing_vars:
            logger.error("Missing required environment variables", missing=missing_vars)
            logger.info("Please ensure your .env file contains all required variables")
            sys.exit(1)
        
        # Startup information
        logger.info("=== RAG Application API - Development Mode ===")
        logger.info("Starting development server", 
                   host=settings.api_host,
                   port=settings.api_port,
                   weaviate_url=settings.weaviate_url)
        
        # Weaviate connection check
        try:
            import requests
            response = requests.get(f"{settings.weaviate_url}/v1/.well-known/ready", timeout=5)
            if response.status_code == 200:
                logger.info("✓ Weaviate is ready at", url=settings.weaviate_url)
            else:
                logger.warning("✗ Weaviate health check failed", 
                             status_code=response.status_code)
        except Exception as e:
            logger.warning("✗ Cannot connect to Weaviate", 
                         error=str(e),
                         url=settings.weaviate_url)
            logger.info("Make sure Weaviate is running: docker compose up weaviate")
        
        # Start the server
        try:
            uvicorn.run(
                "app.main:app",
                host=settings.api_host,
                port=settings.api_port,
                reload=True,  # Enable auto-reload for development
                log_level="info",
                access_log=True,
                workers=1  # Single worker for development
            )
        except KeyboardInterrupt:
            logger.info("Server stopped by user")
        except Exception as e:
            logger.error("Failed to start server", error=str(e))
            sys.exit(1)

    if __name__ == "__main__":
        main()

except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you have activated your virtual environment and installed dependencies:")
    print("  .venv\\Scripts\\Activate.ps1  # Windows")
    print("  pip install -r requirements.txt")
    sys.exit(1)
