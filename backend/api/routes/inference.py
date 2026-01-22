"""Optimized inference API routes with async processing"""

from fastapi import APIRouter, Depends, HTTPException, Request, BackgroundTasks
from pydantic import BaseModel, Field, validator
from sqlalchemy.orm import Session
from typing import Optional
from datetime import datetime

from backend.services.inference_service import inference_engine
from backend.db.session import get_db
from backend.db.models import User, InferenceLog
from backend.security.auth.dependencies import get_current_user_optional

router = APIRouter(prefix="/inference", tags=["inference"])


class CompareRequest(BaseModel):
    text1: str = Field(..., min_length=1, max_length=5000, description="First text")
    text2: str = Field(..., min_length=1, max_length=5000, description="Second text")
    use_cache: bool = Field(True, description="Use inference cache for faster response")
    
    @validator('text1', 'text2')
    def validate_text(cls, v):
        if not v.strip():
            raise ValueError("Text cannot be empty or whitespace only")
        return v.strip()


class CompareResponse(BaseModel):
    similarity: float = Field(..., description="Similarity score (0-1)")
    is_paraphrase: bool = Field(..., description="Whether texts are paraphrases")
    inference_time_ms: float = Field(..., description="Inference time in milliseconds")
    cached: bool = Field(..., description="Whether result was from cache")


class HealthResponse(BaseModel):
    status: str
    model_ready: bool
    performance: dict


def log_inference_async(
    db: Session,
    user_id: Optional[int],
    text1: str,
    text2: str,
    similarity: float,
    inference_time: float
):
    """Background task to log inference"""
    try:
        log = InferenceLog(
            user_id=user_id,
            text_a=text1[:500],  # Truncate for storage
            text_b=text2[:500],
            similarity_score=similarity,
            inference_time_ms=inference_time,
            timestamp=datetime.utcnow()
        )
        db.add(log)
        db.commit()
    except Exception as e:
        print(f"Failed to log inference: {e}")
        db.rollback()


@router.post("/compare", response_model=CompareResponse)
async def compare_texts(
    request: CompareRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user_optional)
):
    """
    Compare two texts for paraphrase detection
    
    - **Optimized with caching** for repeated queries
    - **Fast response times** with async logging
    - **Optional authentication** for tracking
    """
    try:
        similarity, is_paraphrase, inference_time = inference_engine.predict(
            request.text1,
            request.text2,
            use_cache=request.use_cache
        )
        
        cached = inference_time == 0.0
        
        # Log in background to not slow down response
        if current_user:
            background_tasks.add_task(
                log_inference_async,
                db,
                current_user.id,
                request.text1,
                request.text2,
                similarity,
                inference_time
            )
        
        return CompareResponse(
            similarity=similarity,
            is_paraphrase=is_paraphrase,
            inference_time_ms=inference_time,
            cached=cached
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint with performance stats
    """
    return HealthResponse(
        status="healthy" if inference_engine.is_ready() else "model_not_loaded",
        model_ready=inference_engine.is_ready(),
        performance=inference_engine.get_performance_stats()
    )


@router.post("/clear-cache")
async def clear_cache(current_user: User = Depends(get_current_user_optional)):
    """
    Clear inference cache (admin or user for their own use)
    """
    inference_engine.clear_cache()
    return {"status": "cache_cleared", "message": "Inference cache cleared successfully"}


class CompareRequest(BaseModel):
    text_a: str
    text_b: str
    threshold: float = 0.5


class CompareResponse(BaseModel):
    similarity: float
    is_paraphrase: bool
    threshold: float
    inference_time_ms: float


@router.post("/compare", response_model=CompareResponse)
async def compare_texts(
    data: CompareRequest,
    request: Request,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user_optional)
):
    """
    Compare two texts for paraphrase similarity
    
    - **text_a**: First text to compare
    - **text_b**: Second text to compare
    - **threshold**: Similarity threshold (default: 0.5)
    """
    # Check if model is ready
    if not inference_engine.is_ready():
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please try again later."
        )
    
    # Predict
    try:
        similarity, is_paraphrase, inference_time = inference_engine.predict(
            data.text_a,
            data.text_b
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {str(e)}"
        )
    
    # Log inference
    log = InferenceLog(
        user_id=current_user.id if current_user else None,
        text_a=data.text_a,
        text_b=data.text_b,
        similarity_score=similarity,
        is_paraphrase=is_paraphrase,
        inference_time_ms=inference_time,
        ip_address=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent")
    )
    
    db.add(log)
    db.commit()
    
    return {
        "similarity": round(similarity, 3),
        "is_paraphrase": is_paraphrase,
        "threshold": data.threshold,
        "inference_time_ms": round(inference_time, 2)
    }


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if inference_engine.is_ready() else "model_not_loaded",
        "model_ready": inference_engine.is_ready()
    }
