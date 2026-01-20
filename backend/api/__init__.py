"""
FastAPI application and routing.

This module contains the REST API endpoints for the paraphrase detection service.
"""

from .inference import app

__all__ = ["app"]
