"""Optimized inference API routes - No database dependencies"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator
from typing import Optional

from backend.services.inference_service import inference_engine

router = APIRouter(prefix="/inference", tags=["inference"])


class CompareRequest(BaseModel):
    text1: str = Field(..., min_length=1, max_length=5000, description="First text")
    text2: str = Field(..., min_length=1, max_length=5000, description="Second text")
    use_cache: bool = Field(True, description="Use inference cache for faster response")
    use_agent: bool = Field(True, description="Use agentic AI for edge case validation")
    
    @field_validator('text1', 'text2')
    @classmethod
    def validate_text(cls, v):
        if not v.strip():
            raise ValueError("Text cannot be empty or whitespace only")
        return v.strip()


class CompareResponse(BaseModel):
    similarity: float = Field(..., description="Similarity score (0-1)")
    is_paraphrase: bool = Field(..., description="Whether texts are paraphrases")
    confidence: float = Field(..., description="Confidence score (0-1)")
    inference_time_ms: float = Field(..., description="Inference time in milliseconds")
    cached: bool = Field(False, description="Whether result was from cache")
    agent_used: bool = Field(False, description="Whether agentic AI was used")
    confidence_level: Optional[str] = Field(None, description="Confidence level (HIGH/MEDIUM/LOW/UNCERTAIN)")
    edge_cases: Optional[list] = Field(None, description="Detected edge cases")
    agent_validation: Optional[bool] = Field(None, description="Whether CrewAI validation was used")
    agent_reasoning: Optional[str] = Field(None, description="CrewAI reasoning for decision")
    agent_confidence: Optional[str] = Field(None, description="CrewAI confidence level")
    paraphrase_rescued: Optional[bool] = Field(None, description="Whether agents caught a paraphrase the model missed")
    original_similarity: Optional[float] = Field(None, description="Original similarity before agent adjustment")
    adjusted_similarity: Optional[float] = Field(None, description="Adjusted similarity after agent validation")


class HealthResponse(BaseModel):
    status: str
    model_ready: bool
    performance: dict


@router.post("/compare", response_model=CompareResponse)
async def compare_texts(request: CompareRequest):
    """
    Compare two texts for paraphrase detection
    
    - **Optimized with caching** for repeated queries
    - **Agentic AI validation** for edge case handling
    - **Fast response times** with async processing
    """
    # Check if model is ready
    if not inference_engine.is_ready():
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please try again later."
        )
    
    try:
        similarity, is_paraphrase, inference_time, agent_metadata = inference_engine.predict(
            request.text1,
            request.text2,
            use_cache=request.use_cache,
            use_agent=request.use_agent
        )
        
        cached = agent_metadata.get('cached', False)
        
        return CompareResponse(
            similarity=similarity,
            is_paraphrase=is_paraphrase,
            confidence=similarity,
            inference_time_ms=inference_time,
            cached=cached,
            agent_used=agent_metadata.get('agent_used', False),
            confidence_level=agent_metadata.get('confidence_level'),
            edge_cases=agent_metadata.get('edge_cases'),
            agent_validation=agent_metadata.get('agent_validation'),
            agent_reasoning=agent_metadata.get('agent_reasoning'),
            agent_confidence=agent_metadata.get('agent_confidence'),
            paraphrase_rescued=agent_metadata.get('paraphrase_rescued'),
            original_similarity=agent_metadata.get('original_similarity'),
            adjusted_similarity=agent_metadata.get('adjusted_similarity')
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
async def clear_cache():
    """
    Clear inference cache
    """
    inference_engine.clear_cache()
    return {"status": "cache_cleared", "message": "Inference cache cleared successfully"}
