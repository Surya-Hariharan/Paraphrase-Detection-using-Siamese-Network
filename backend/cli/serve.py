"""Run the FastAPI server"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import uvicorn
from backend.config.settings import settings


if __name__ == "__main__":
    uvicorn.run(
        "backend.api.main:app",
        host=settings.app.HOST,
        port=settings.app.PORT,
        reload=settings.app.DEBUG,
        workers=1 if settings.app.DEBUG else settings.app.WORKERS
    )
