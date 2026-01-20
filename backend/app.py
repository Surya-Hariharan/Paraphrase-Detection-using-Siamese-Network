"""
Main FastAPI Application Entry Point
=====================================

This module serves as the entry point for the backend API server.
It imports the FastAPI app from the api module.

Usage:
    uvicorn app:app --reload
    OR
    uvicorn app:app --host 0.0.0.0 --port 8000
"""

from api.inference import app

__all__ = ["app"]
