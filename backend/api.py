"""
FastAPI REST API for Paraphrase Detection System

Production-ready API with endpoints for:
- Document comparison
- Batch processing
- Health checks
- Model info

Deploy to: Vercel, Railway, Render, Heroku, or any cloud platform
"""

import os
import logging
from typing import Optional, List
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Import our modules
try:
    from document_siamese_pipeline import DocumentLevelSiameseModel
    from document_processor import DocumentProcessor
except ImportError:
    # For deployment contexts
    from backend.document_siamese_pipeline import DocumentLevelSiameseModel
    from backend.document_processor import DocumentProcessor

# =============================================================================
# Configuration
# =============================================================================

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Logging configuration
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# API Configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

# Model Configuration
MODEL_PATH = os.getenv("MODEL_PATH", "models/siamese_trained.pt")
THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.75"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# =============================================================================
# FastAPI App Initialization
# =============================================================================

app = FastAPI(
    title="Paraphrase Detection API",
    description="Document-level paraphrase detection using Siamese neural networks with SBERT",
    version="1.0.0",
    docs_url="/docs" if DEBUG else None,
    redoc_url="/redoc" if DEBUG else None,
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# Global Model Instance (loaded once at startup)
# =============================================================================

pipeline: Optional[DocumentLevelSiameseModel] = None


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    global pipeline
    try:
        logger.info("Loading paraphrase detection model...")
        
        # Check if model file exists
        model_path = MODEL_PATH if Path(MODEL_PATH).exists() else None
        
        pipeline = DocumentLevelSiameseModel(
            model_path=model_path,
            threshold=THRESHOLD,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        
        logger.info("✓ Model loaded successfully")
        logger.info(f"Environment: {ENVIRONMENT}")
        logger.info(f"Threshold: {THRESHOLD}")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        # Don't crash the app, just log the error
        pipeline = None


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down API...")


# =============================================================================
# Request/Response Models
# =============================================================================

class CompareRequest(BaseModel):
    """Request model for document comparison"""
    doc_a: str = Field(..., description="First document text")
    doc_b: str = Field(..., description="Second document text")
    threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Custom threshold")
    
    class Config:
        schema_extra = {
            "example": {
                "doc_a": "Machine learning is a subset of AI that enables computers to learn from data.",
                "doc_b": "AI includes machine learning, which allows systems to learn automatically from data.",
                "threshold": 0.75
            }
        }


class CompareResponse(BaseModel):
    """Response model for document comparison"""
    cosine_similarity: float
    is_paraphrase: bool
    decision: str
    confidence: str
    threshold: float
    doc_a_length: int
    doc_b_length: int
    doc_a_chunks: int
    doc_b_chunks: int


class BatchCompareRequest(BaseModel):
    """Request model for batch comparison"""
    pairs: List[dict] = Field(..., description="List of document pairs")
    
    class Config:
        schema_extra = {
            "example": {
                "pairs": [
                    {"doc_a": "First document", "doc_b": "Second document"},
                    {"doc_a": "Third document", "doc_b": "Fourth document"}
                ]
            }
        }


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    environment: str
    model_loaded: bool
    version: str


# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "message": "Paraphrase Detection API",
        "version": "1.0.0",
        "docs": "/docs" if DEBUG else "Documentation disabled in production",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy" if pipeline is not None else "degraded",
        "environment": ENVIRONMENT,
        "model_loaded": pipeline is not None,
        "version": "1.0.0"
    }


@app.post("/api/compare", response_model=CompareResponse)
async def compare_documents(request: CompareRequest):
    """
    Compare two documents for paraphrase detection.
    
    Returns similarity score and decision (PARAPHRASE or NOT PARAPHRASE).
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Use custom threshold if provided
        if request.threshold is not None:
            pipeline.threshold = request.threshold
        
        # Compare documents
        result = pipeline.compare_documents(
            doc_a=request.doc_a,
            doc_b=request.doc_b,
            verbose=False
        )
        
        return CompareResponse(**result)
        
    except Exception as e:
        logger.error(f"Comparison error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/compare/batch")
async def batch_compare(request: BatchCompareRequest):
    """
    Compare multiple document pairs in batch.
    
    Returns list of comparison results.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Extract document pairs
        document_pairs = [
            (pair["doc_a"], pair["doc_b"]) 
            for pair in request.pairs
        ]
        
        # Batch compare
        result = pipeline.batch_compare(
            document_pairs=document_pairs,
            verbose=False
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Batch comparison error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/compare/files")
async def compare_files(
    file_a: UploadFile = File(...),
    file_b: UploadFile = File(...),
    threshold: Optional[float] = Form(None)
):
    """
    Compare two uploaded files.
    
    Supports: .txt, .pdf, .docx
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Create temp directory
    temp_dir = Path("temp")
    temp_dir.mkdir(exist_ok=True)
    
    try:
        # Save uploaded files temporarily
        file_a_path = temp_dir / file_a.filename
        file_b_path = temp_dir / file_b.filename
        
        with open(file_a_path, "wb") as f:
            content = await file_a.read()
            f.write(content)
        
        with open(file_b_path, "wb") as f:
            content = await file_b.read()
            f.write(content)
        
        # Use custom threshold if provided
        if threshold is not None:
            pipeline.threshold = threshold
        
        # Compare files
        result = pipeline.compare_from_files(
            file_a=str(file_a_path),
            file_b=str(file_b_path),
            verbose=False
        )
        
        return result
        
    except Exception as e:
        logger.error(f"File comparison error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Cleanup temp files
        try:
            if file_a_path.exists():
                file_a_path.unlink()
            if file_b_path.exists():
                file_b_path.unlink()
        except Exception as e:
            logger.warning(f"Cleanup error: {e}")


@app.get("/api/model/info")
async def model_info():
    """Get information about the loaded model"""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "architecture": "Doc → SBERT (frozen) → NN (Siamese) → FV → Cosine Similarity",
        "sbert_model": "all-MiniLM-L6-v2",
        "projection_dim": 256,
        "threshold": pipeline.threshold,
        "chunk_size": pipeline.chunk_size,
        "chunk_overlap": pipeline.chunk_overlap,
        "aggregation_method": pipeline.aggregation_method,
        "device": str(pipeline.device)
    }


# =============================================================================
# Error Handlers
# =============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host=API_HOST,
        port=API_PORT,
        reload=DEBUG,
        log_level=os.getenv("LOG_LEVEL", "info").lower()
    )
