"""
FastAPI Application Entry Point
================================

Main entry point for the FastAPI application.
Import the app from api.main for uvicorn to run.

Usage (from project root):
    uvicorn backend.app:app --reload
    uvicorn backend.app:app --host 0.0.0.0 --port 8000
"""

from backend.api.main import app

__all__ = ["app"]
