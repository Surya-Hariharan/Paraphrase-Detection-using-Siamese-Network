"""API routes"""

from .auth import router as auth_router
from .inference import router as inference_router

__all__ = ["auth_router", "inference_router"]
