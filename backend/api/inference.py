"""
FastAPI Inference Server for Paraphrase Detection
==================================================

Production-ready API with proper error handling and preprocessing.

Usage:
    uvicorn backend.api.inference:app --reload --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import torch
from typing import Dict, Optional
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.core.neural_engine import TrainableSiameseModel

# Initialize FastAPI app
app = FastAPI(
    title="Paraphrase Detection API",
    description="BiLSTM + Attention enhanced Siamese Network for paraphrase detection",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model globally
MODEL_PATH = "checkpoints/best_model.pt"
model = None

@app.on_event("startup")
async def load_model():
    """Load the trained model on startup."""
    global model
    try:
        model = TrainableSiameseModel(
            projection_dim=256,
            unfreeze_last_sbert_layer=True,
            unfreeze_last_n_layers=6  # Entire model unfrozen
        )
        model.load_checkpoint(MODEL_PATH)
        model.eval()
        print(f"✓ Model loaded from {MODEL_PATH}")
    except Exception as e:
        print(f"⚠️  Warning: Could not load model: {e}")
        print("   API will start but predictions will fail until model is trained.")

# Request/Response models
class PairRequest(BaseModel):
    text_a: str = Field(..., description="First text to compare", min_length=1)
    text_b: str = Field(..., description="Second text to compare", min_length=1)
    threshold: Optional[float] = Field(0.8, description="Similarity threshold (0.0-1.0)", ge=0.0, le=1.0)

class PairResponse(BaseModel):
    similarity: float = Field(..., description="Cosine similarity score")
    is_paraphrase: bool = Field(..., description="Whether texts are paraphrases")
    threshold: float = Field(..., description="Threshold used")
    confidence: float = Field(..., description="Confidence score (0.0-1.0)")

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str

# Endpoints
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Paraphrase Detection API",
        "version": "2.0.0",
        "endpoints": {
            "POST /compare": "Compare two texts for paraphrasing",
            "GET /health": "Health check endpoint"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy" if model is not None else "model_not_loaded",
        "model_loaded": model is not None,
        "device": str(model.device) if model is not None else "unknown"
    }

@app.post("/compare", response_model=PairResponse)
async def compare_texts(request: PairRequest):
    """
    Compare two texts and determine if they are paraphrases.
    
    Args:
        request: PairRequest containing text_a, text_b, and optional threshold
    
    Returns:
        PairResponse with similarity score and paraphrase prediction
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first."
        )
    
    try:
        # Get similarity score
        result = model.calculate_similarity(request.text_a, request.text_b)
        
        similarity = float(result['similarity'])
        is_paraphrase = similarity >= request.threshold
        
        # Calculate confidence (distance from threshold)
        confidence = abs(similarity - request.threshold) / (1.0 - request.threshold) if is_paraphrase else abs(request.threshold - similarity) / request.threshold
        confidence = min(1.0, max(0.0, confidence))
        
        return PairResponse(
            similarity=similarity,
            is_paraphrase=is_paraphrase,
            threshold=request.threshold,
            confidence=confidence
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )

@app.post("/batch_compare")
async def batch_compare(pairs: list[PairRequest]):
    """
    Compare multiple text pairs in batch.
    
    Args:
        pairs: List of PairRequest objects
    
    Returns:
        List of PairResponse objects
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first."
        )
    
    results = []
    for pair in pairs:
        try:
            result = await compare_texts(pair)
            results.append(result)
        except Exception as e:
            results.append({
                "error": str(e),
                "text_a": pair.text_a[:50] + "...",
                "text_b": pair.text_b[:50] + "..."
            })
    
    return results

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
