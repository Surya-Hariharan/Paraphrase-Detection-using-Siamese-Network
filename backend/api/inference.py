"""
FastAPI Inference Server for Paraphrase Detection
==================================================

Production-ready API with proper error handling and preprocessing.

Usage:
    uvicorn backend.api.inference:app --reload --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import torch
from typing import Dict, Optional, List
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.core.model import TrainableSiameseModel
from backend.utils.document_processor import extract_text_from_bytes, validate_text

# Try to load AI agents
try:
    from backend.agents import InferenceValidatorAgent, AgentConfig
    AGENTS_AVAILABLE = True
except ImportError:
    AGENTS_AVAILABLE = False

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

# Load model and agent globally
# Use relative path from backend to project root
MODEL_PATH = "../checkpoints/best_model.pt"
model = None
inference_agent = None

@app.on_event("startup")
async def load_model():
    """Load the trained model and AI agent on startup."""
    global model, inference_agent
    
    # Load model
    try:
        model = TrainableSiameseModel(
            projection_dim=256,
            unfreeze_all=True
        )
        model.load_checkpoint(MODEL_PATH)
        model.eval()
        print(f"✓ Model loaded from {MODEL_PATH}")
    except Exception as e:
        print(f"⚠️  Warning: Could not load model: {e}")
        print("   API will start but predictions will fail until model is trained.")
    
    # Load AI agent
    if AGENTS_AVAILABLE:
        try:
            agent_config = AgentConfig()
            inference_agent = InferenceValidatorAgent(agent_config, confidence_threshold=0.7)
            print("✓ AI Inference Validator loaded")
        except Exception as e:
            print(f"⚠️  AI Agent disabled: {str(e)}")
            inference_agent = None

# Request/Response models
class PairRequest(BaseModel):
    text_a: str = Field(..., description="First text to compare", min_length=1)
    text_b: str = Field(..., description="Second text to compare", min_length=1)
    threshold: Optional[float] = Field(0.8, description="Similarity threshold (0.0-1.0)", ge=0.0, le=1.0)
    use_agent: Optional[bool] = Field(True, description="Use AI agent validation for edge cases")

class PairResponse(BaseModel):
    similarity: float = Field(..., description="Cosine similarity score")
    is_paraphrase: bool = Field(..., description="Whether texts are paraphrases")
    threshold: float = Field(..., description="Threshold used")
    confidence: float = Field(..., description="Confidence score (0.0-1.0)")
    agent_validation: Optional[Dict] = Field(None, description="AI agent validation results")

class HealthResponse(BaseModel):
    status: str
    is_model_ready: bool
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
        "is_model_ready": model is not None,
        "device": str(model.device) if model is not None else "unknown"
    }

@app.post("/compare", response_model=PairResponse)
async def compare_texts(request: PairRequest):
    """
    Compare two texts and determine if they are paraphrases.
    
    Args:
        request: PairRequest containing text_a, text_b, threshold, and use_agent flag
    
    Returns:
        PairResponse with similarity score, prediction, and optional agent validation
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first."
        )
    
    try:
        # Get similarity score from model
        result = model.calculate_similarity(request.text_a, request.text_b)
        
        # Extract similarity score (cosine_similarity is in range [-1, 1])
        # Convert to [0, 1] range for easier interpretation
        cosine_sim = float(result['cosine_similarity'])
        similarity = (cosine_sim + 1) / 2  # Convert [-1, 1] to [0, 1]
        
        is_paraphrase = similarity >= request.threshold
        
        # Calculate confidence (distance from threshold)
        confidence = abs(similarity - request.threshold) / (1.0 - request.threshold) if is_paraphrase else abs(request.threshold - similarity) / request.threshold
        confidence = min(1.0, max(0.0, confidence))
        
        # AI Agent validation (if enabled)
        agent_validation = None
        if request.use_agent and inference_agent is not None:
            is_valid, validation_result = inference_agent.validate_prediction(
                text_a=request.text_a,
                text_b=request.text_b,
                similarity=similarity,
                processing_time_ms=0.0  # Will be updated when timing is implemented
            )
            
            # Include relevant agent info
            agent_validation = {
                "is_valid": is_valid,
                "issues": validation_result.get("issues", []),
                "warnings": validation_result.get("warnings", []),
                "timestamp": validation_result.get("timestamp", "")
            }
        
        return PairResponse(
            similarity=similarity,
            is_paraphrase=is_paraphrase,
            threshold=request.threshold,
            confidence=confidence,
            agent_validation=agent_validation
        )
    
    except Exception as e:
        import traceback
        print(f"❌ Error in /compare endpoint: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )

@app.post("/batch_compare")
async def batch_compare(pairs: List[PairRequest]):
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

@app.post("/compare_files", response_model=PairResponse)
async def compare_files(
    file_a: UploadFile = File(..., description="First document (.txt, .pdf, .docx)"),
    file_b: UploadFile = File(..., description="Second document (.txt, .pdf, .docx)"),
    threshold: Optional[float] = 0.8,
    use_agent: Optional[bool] = True
):
    """
    Compare two uploaded documents for paraphrasing.
    
    Supports: .txt, .pdf, .docx
    
    Args:
        file_a: First document file
        file_b: Second document file
        threshold: Similarity threshold (0.0-1.0)
        use_agent: Use AI agent validation
        
    Returns:
        PairResponse with similarity analysis
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first."
        )
    
    try:
        # Read file contents
        content_a = await file_a.read()
        content_b = await file_b.read()
        
        # Extract text from files
        text_a = extract_text_from_bytes(content_a, file_a.filename)
        text_b = extract_text_from_bytes(content_b, file_b.filename)
        
        # Validate extracted text
        valid_a, error_a = validate_text(text_a)
        if not valid_a:
            raise HTTPException(status_code=400, detail=f"File A: {error_a}")
        
        valid_b, error_b = validate_text(text_b)
        if not valid_b:
            raise HTTPException(status_code=400, detail=f"File B: {error_b}")
        
        # Create request object and process
        request = PairRequest(
            text_a=text_a,
            text_b=text_b,
            threshold=threshold,
            use_agent=use_agent
        )
        
        return await compare_texts(request)
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"❌ Error in /compare_files endpoint: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"File processing error: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
