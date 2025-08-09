"""
Centralized logging system for the RAG Application API.

This module provides a configurable logging system that writes to both
console and files with structured formatting. It follows the Single
Responsibility Principle by handling only logging concerns.
"""

import logging
import logging.handlers
from pathlib import Path
from typing import Optional
from datetime import datetime
import sys
import tempfile

from config.settings import settings


class RAGLogger:
    """
    Centralized logger class for the RAG application.
    
    This class provides structured logging with both console and file output,
    configurable log levels, and proper formatting for debugging and monitoring.
    """
    
    def __init__(self, name: str = "rag_app") -> None:
        """
        Initialize the logger with console and file handlers.
        
        Args:
            name: Logger name (typically module name)
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, settings.log_level.upper()))
        
        # Prevent duplicate handlers if logger is reinitialized
        if not self.logger.handlers:
            self._setup_handlers()
    
    def _setup_handlers(self) -> None:
        """Setup console and file handlers with proper formatting."""
        # Console handler for immediate feedback
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        
        # Always add console handler first
        self.logger.addHandler(console_handler)
        
        # Skip file logging if disabled
        if settings.disable_file_logging:
            print("File logging disabled via configuration")
            self.logger.setLevel(getattr(logging, settings.log_level.upper()))
            return
        
        # Try to set up file handler with proper error handling
        try:
            # Create logs directory if it doesn't exist
            log_path = settings.log_file_path
            
            # Try to create the directory with full permissions
            try:
                log_path.parent.mkdir(parents=True, exist_ok=True, mode=0o777)
            except (PermissionError, OSError):
                # If we can't create in the default location, try /tmp
                log_path = Path(tempfile.gettempdir()) / "app.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)
            
            # File handler for persistent logging with rotation
            file_handler = logging.handlers.RotatingFileHandler(
                filename=log_path,
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=5,
                encoding='utf-8'
            )
            file_handler.setLevel(getattr(logging, settings.log_level.upper()))
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
            print(f"Logging to file: {log_path}")
            
        except (PermissionError, OSError) as e:
            # If we can't create the file handler at all, log to console only
            print(f"Warning: Could not create file handler for logging: {e}. Logging to console only.")
        
        # Set logger level
        self.logger.setLevel(getattr(logging, settings.log_level.upper()))
    
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message with optional context."""
        self.logger.debug(self._format_message(message, **kwargs))
    
    def info(self, message: str, **kwargs) -> None:
        """Log info message with optional context."""
        self.logger.info(self._format_message(message, **kwargs))
    
    def warning(self, message: str, **kwargs) -> None:
        """Log warning message with optional context."""
        self.logger.warning(self._format_message(message, **kwargs))
    
    def error(self, message: str, **kwargs) -> None:
        """Log error message with optional context."""
        self.logger.error(self._format_message(message, **kwargs))
    
    def critical(self, message: str, **kwargs) -> None:
        """Log critical message with optional context."""
        self.logger.critical(self._format_message(message, **kwargs))
    
    def exception(self, message: str, **kwargs) -> None:
        """Log exception with full traceback."""
        self.logger.exception(self._format_message(message, **kwargs))
    
    def _format_message(self, message: str, **kwargs) -> str:
        """
        Format log message with additional context.
        
        Args:
            message: Base log message
            **kwargs: Additional context to include
            
        Returns:
            str: Formatted message with context
        """
        if not kwargs:
            return message
        
        context_parts = [f"{k}={v}" for k, v in kwargs.items()]
        context_str = " | ".join(context_parts)
        return f"{message} | {context_str}"
    
    def log_stage(self, stage: str, action: str, **context) -> None:
        """
        Log RAG pipeline stage information.
        
        Args:
            stage: Pipeline stage name (e.g., "Document Processing")
            action: Action being performed (e.g., "Starting", "Completed")
            **context: Additional context information
        """
        self.info(f"[{stage}] {action}", **context)
    
    def log_performance(self, operation: str, duration: float, **context) -> None:
        """
        Log performance metrics.
        
        Args:
            operation: Name of the operation
            duration: Time taken in seconds
            **context: Additional performance context
        """
        self.info(
            f"Performance: {operation} completed",
            duration_seconds=f"{duration:.3f}",
            **context
        )


# Global logger instance
logger = RAGLogger()


def get_logger(name: Optional[str] = None) -> RAGLogger:
    """
    Get a logger instance for a specific module.
    
    Args:
        name: Logger name (typically __name__ from calling module)
        
    Returns:
        RAGLogger: Configured logger instance
    """
    if name:
        return RAGLogger(name)
    return logger


def log_function_entry(func_name: str, **params) -> None:
    """
    Utility function to log function entry with parameters.
    
    Args:
        func_name: Name of the function being entered
        **params: Function parameters to log
    """
    logger.debug(f"Entering {func_name}", **params)


def log_function_exit(func_name: str, result: Optional[str] = None) -> None:
    """
    Utility function to log function exit.
    
    Args:
        func_name: Name of the function being exited
        result: Optional result description
    """
    context = {"result": result} if result else {}
    logger.debug(f"Exiting {func_name}", **context)
