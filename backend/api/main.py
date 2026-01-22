"""Main FastAPI application"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from backend.config.settings import settings
from backend.db.session import init_db
from backend.api.routes import auth, inference
from backend.api.middleware.security import SecurityMiddleware, LoggingMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup
    print("🚀 Starting Paraphrase Detection API")
    
    # Initialize database
    init_db()
    print("✓ Database initialized")
    
    # Model is loaded in inference_service module
    print("✓ Application ready")
    
    yield
    
    # Shutdown
    print("👋 Shutting down")


# Create FastAPI app
app = FastAPI(
    title=settings.app.APP_NAME,
    version=settings.app.APP_VERSION,
    description="Production-ready paraphrase detection API with authentication and security",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.security.CORS_ORIGINS,
    allow_credentials=settings.security.CORS_ALLOW_CREDENTIALS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security middleware
app.add_middleware(SecurityMiddleware)
app.add_middleware(LoggingMiddleware)

# Routes
app.include_router(auth.router)
app.include_router(inference.router)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": settings.app.APP_NAME,
        "version": settings.app.APP_VERSION,
        "status": "online",
        "docs": "/docs"
    }


@app.get("/health")
async def health():
    """Health check"""
    from backend.services.inference_service import inference_engine
    
    return {
        "status": "healthy",
        "model_ready": inference_engine.is_ready()
    }


@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    from backend.api.middleware.security import ddos_protector
    
    return {
        "security": ddos_protector.get_stats()
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "backend.api.main:app",
        host=settings.app.HOST,
        port=settings.app.PORT,
        reload=settings.app.DEBUG
    )
