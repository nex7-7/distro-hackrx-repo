"""
Configuration package for the RAG Application.

This package contains configuration management and settings
for the entire application.
"""

from .settings import settings, get_settings, Settings

__all__ = [
    "settings",
    "get_settings", 
    "Settings"
]
