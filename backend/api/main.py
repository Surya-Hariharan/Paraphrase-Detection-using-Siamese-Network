"""Optimized FastAPI application with production configurations"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import time

from backend.config.settings import settings
from backend.api.routes import inference
from backend.api.middleware.security import SecurityMiddleware, LoggingMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup
    print("=" * 70)
    print("🚀 Starting Paraphrase Detection API")
    print("=" * 70)
    
    # Preload model for faster first request
    from backend.services.inference_service import inference_engine
    if inference_engine.is_ready():
        print("✓ Model preloaded and ready")
    else:
        print("⚠ Model not loaded - will load on first request")
    
    print("=" * 70)
    print(f"✓ API ready at http://{settings.app.HOST}:{settings.app.PORT}")
    print(f"📚 Docs at http://{settings.app.HOST}:{settings.app.PORT}/docs")
    print("=" * 70)
    
    yield
    
    # Shutdown
    print("\n👋 Shutting down API")


# Create FastAPI app with optimizations
app = FastAPI(
    title=settings.app.APP_NAME,
    version=settings.app.APP_VERSION,
    description="Production-grade paraphrase detection API with caching, authentication, and DDoS protection",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Compression middleware for faster responses
app.add_middleware(GZipMiddleware, minimum_size=1000)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.security.CORS_ORIGINS,
    allow_credentials=settings.security.CORS_ALLOW_CREDENTIALS,
    allow_methods=["*"],
    allow_headers=["*"],
    max_age=3600  # Cache preflight requests
)

# Security middleware
app.add_middleware(SecurityMiddleware)
app.add_middleware(LoggingMiddleware)


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle unexpected errors gracefully"""
    print(f"❌ Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "message": "An unexpected error occurred. Please try again later."
        }
    )


# Include routers
app.include_router(inference.router)


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": settings.app.APP_NAME,
        "version": settings.app.APP_VERSION,
        "status": "online",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "auth": "/auth",
            "inference": "/inference"
        }
    }


@app.get("/health")
async def health():
    """Fast health check endpoint"""
    from backend.services.inference_service import inference_engine
    
    return {
        "status": "healthy",
        "model_ready": inference_engine.is_ready(),
        "timestamp": time.time()
    }


@app.get("/stats")
async def get_stats():
    """System statistics endpoint"""
    from backend.api.middleware.security import ddos_protector
    from backend.services.inference_service import inference_engine
    
    return {
        "security": ddos_protector.get_stats(),
        "inference": inference_engine.get_performance_stats()
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "backend.api.main:app",
        host=settings.app.HOST,
        port=settings.app.PORT,
        reload=settings.app.DEBUG,
        log_level="info"
    )
