"""
Logger module for the RAG application.

This module provides logging functionality for tracking API responses,
requests, and other important events in the application.
"""

import os
import json
import logging
from typing import Any, Dict, Union, Optional
from datetime import datetime
from pathlib import Path

# Configure the basic logging format and level
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Create logger for this module
logger = logging.getLogger('rag_api')


def setup_file_logging(log_dir: str = "logs") -> None:
    """
    Set up logging to a file in addition to console logging.

    Parameters:
    log_dir (str): Directory where log files will be stored.

    Returns:
    None
    """
    # Create logs directory if it doesn't exist
    Path(log_dir).mkdir(exist_ok=True)

    # Define log file name based on current date
    log_file = os.path.join(
        log_dir, f"rag_api_{datetime.now().strftime('%Y%m%d')}.log")

    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(file_handler)
    logger.info(f"File logging set up at: {log_file}")


def log_api_request(endpoint: str, request_data: Any) -> None:
    """
    Log API request data.

    Parameters:
    endpoint (str): The API endpoint being called.
    request_data (Any): The request data to log, typically a dict or Pydantic model.

    Returns:
    None
    """
    try:
        # Convert Pydantic models to dict if necessary
        if hasattr(request_data, "dict"):
            request_data = request_data.dict()

        logger.info(
            f"API Request to {endpoint}: {json.dumps(request_data, default=str)}")
    except Exception as e:
        logger.error(f"Error logging API request: {str(e)}")


def log_api_response(endpoint: str, response_data: Any, duration: Optional[float] = None) -> None:
    """
    Log API response data and duration if provided.

    Parameters:
    endpoint (str): The API endpoint that was called.
    response_data (Any): The response data to log.
    duration (Optional[float]): The time taken to process the request in seconds.

    Returns:
    None
    """
    try:
        # Convert Pydantic models to dict if necessary
        if hasattr(response_data, "dict"):
            response_data = response_data.dict()

        log_message = f"API Response from {endpoint}"
        if duration is not None:
            log_message += f" (duration: {duration:.2f}s)"

        log_message += f": {json.dumps(response_data, default=str)}"
        logger.info(log_message)
    except Exception as e:
        logger.error(f"Error logging API response: {str(e)}")


def log_error(error_msg: str, details: Optional[Dict[str, Any]] = None) -> None:
    """
    Log an error message with optional details.

    Parameters:
    error_msg (str): The main error message.
    details (Optional[Dict[str, Any]]): Additional error details.

    Returns:
    None
    """
    try:
        if details:
            logger.error(f"{error_msg}: {json.dumps(details, default=str)}")
        else:
            logger.error(error_msg)
    except Exception as e:
        logger.error(
            f"Error while logging error: {str(e)}. Original error: {error_msg}")


def log_service_event(event_type: str, description: str, details: Optional[Dict[str, Any]] = None) -> None:
    """
    Log a service event such as startup, shutdown, or configuration changes.

    Parameters:
    event_type (str): The type of event (e.g., "startup", "shutdown").
    description (str): A description of the event.
    details (Optional[Dict[str, Any]]): Additional event details.

    Returns:
    None
    """
    try:
        log_message = f"Service Event [{event_type}]: {description}"
        if details:
            log_message += f" - Details: {json.dumps(details, default=str)}"
        logger.info(log_message)
    except Exception as e:
        logger.error(f"Error logging service event: {str(e)}")
