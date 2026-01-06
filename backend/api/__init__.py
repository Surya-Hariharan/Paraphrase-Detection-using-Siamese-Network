"""
FastAPI application and routing.

This module contains the REST API endpoints for the paraphrase detection service.
"""

from .app import app
from .routes import router

__all__ = ["app", "router"]
