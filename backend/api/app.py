"""
FastAPI Backend for Paraphrase Detection System
Backend API with endpoints for document indexing, duplicate search, and comparison.
"""

import os
import sys
from pathlib import Path
from typing import List, Optional
from datetime import datetime
import traceback
import asyncio
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.core.neural_engine import SiameseProjectionModel

app = FastAPI(
    title="Paraphrase Detection API",
    description="AI-powered paraphrase and duplicate detection using Siamese networks and vector databases",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
neural_model = None
trained_model = None
executor = ThreadPoolExecutor(max_workers=4)

# Model paths
BACKEND_DIR = Path(__file__).parent
MODEL_DIR = BACKEND_DIR / "models"
TRAINED_MODEL_PATH = MODEL_DIR / "siamese_trained.pt"


def get_neural_model():
    """Lazy load neural model (untrained, for inference)."""
    global neural_model
    if neural_model is None:
        neural_model = SiameseProjectionModel()
    return neural_model


def get_trained_model():
    """Load trained model if available."""
    global trained_model
    if trained_model is None:
        if TRAINED_MODEL_PATH.exists():
            from neural_engine import TrainableSiameseModel
            trained_model = TrainableSiameseModel()
            trained_model.load_model(str(TRAINED_MODEL_PATH))
            print(f"Loaded trained model from {TRAINED_MODEL_PATH}")
        else:
            print(f"Warning: No trained model found at {TRAINED_MODEL_PATH}")
            print("Using untrained projection head. Train the model first with: python backend/train.py")
            trained_model = SiameseProjectionModel()
    return trained_model


# Request/Response Models
class CompareRequest(BaseModel):
    doc_a: str = Field(..., description="First document text")
    doc_b: str = Field(..., description="Second document text")
    provider: str = Field(default="groq", description="LLM provider (groq or ollama)")
    use_agents: bool = Field(default=True, description="Use multi-agent analysis")


class CompareResponse(BaseModel):
    similarity_score: float
    verdict: Optional[str] = None
    confidence: Optional[str] = None
    reasoning: Optional[str] = None
    doc_a_length: int
    doc_b_length: int
    processing_time: float


class SearchRequest(BaseModel):
    query: str = Field(..., description="Query document text")
    collection: str = Field(default="documents", description="ChromaDB collection name")
    top_k: int = Field(default=5, description="Number of top results")
    provider: str = Field(default="groq", description="LLM provider")


class CandidateMatch(BaseModel):
    id: str
    text: str
    vector_similarity: float
    siamese_score: float
    source: Optional[str] = None
    timestamp: Optional[str] = None


class SearchResponse(BaseModel):
    verdict: str
    confidence: str
    reasoning: str
    best_match: Optional[CandidateMatch] = None
    candidates: List[CandidateMatch]
    processing_time: float


class IndexRequest(BaseModel):
    collection: str = Field(default="documents", description="Collection name")
    provider: str = Field(default="groq", description="LLM provider")


class IndexResponse(BaseModel):
    files_processed: int
    documents_added: int
    chunks_created: int
    collection_name: str
    status: str
    strategy: Optional[dict] = None
    processing_time: float


class DocumentInput(BaseModel):
    text: str
    metadata: Optional[dict] = None


class BulkIndexRequest(BaseModel):
    documents: List[DocumentInput]
    collection: str = Field(default="documents", description="Collection name")
    provider: str = Field(default="groq", description="LLM provider")


class CollectionStats(BaseModel):
    name: str
    count: int
    metadata: dict


class HealthResponse(BaseModel):
    status: str
    neural_model_loaded: bool
    trained_model_available: bool
    trained_model_path: Optional[str]
    version: str
    timestamp: str


# API Endpoints
@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        neural_model_loaded=neural_model is not None,
        trained_model_available=TRAINED_MODEL_PATH.exists(),
        trained_model_path=str(TRAINED_MODEL_PATH) if TRAINED_MODEL_PATH.exists() else None,
        version="1.0.0",
        timestamp=datetime.utcnow().isoformat()
    )


@app.post("/api/compare", response_model=CompareResponse)
async def compare_documents(request: CompareRequest):
    """
    Compare two documents for paraphrase detection.
    
    Phase 1: Neural pipeline (similarity score)
    Phase 2: Multi-agent analysis (optional, verdict + reasoning)
    """
    start_time = datetime.now()
    
    try:
        # Phase 1: Neural Pipeline (use trained model)
        model = get_trained_model()
        
        def compute_similarity():
            return model.calculate_similarity(request.doc_a, request.doc_b)
        
        loop = asyncio.get_event_loop()
        similarity_data = await loop.run_in_executor(executor, compute_similarity)
        
        similarity_score = similarity_data['cosine_similarity']
        
        response_data = {
            "similarity_score": similarity_score,
            "doc_a_length": len(request.doc_a),
            "doc_b_length": len(request.doc_b),
            "processing_time": (datetime.now() - start_time).total_seconds()
        }
        
        # Phase 2: Multi-Agent Analysis (if requested)
        if request.use_agents:
            try:
                from agent_crew import create_crew
                
                def run_agent_analysis():
                    crew = create_crew(request.provider)
                    return crew.analyze(request.doc_a, request.doc_b, similarity_score)
                
                agent_result = await loop.run_in_executor(executor, run_agent_analysis)
                
                response_data.update({
                    "verdict": agent_result.get("verdict"),
                    "confidence": agent_result.get("confidence"),
                    "reasoning": agent_result.get("full_analysis")
                })
            except Exception as e:
                # If agents fail, still return Phase 1 results
                response_data["reasoning"] = f"Agent analysis unavailable: {str(e)}"
        
        response_data["processing_time"] = (datetime.now() - start_time).total_seconds()
        return CompareResponse(**response_data)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")


@app.post("/api/search", response_model=SearchResponse)
async def search_duplicates(request: SearchRequest):
    """
    DEPRECATED: Vector database search not implemented in this version.
    Use /api/compare endpoint instead.
    """
    raise HTTPException(
        status_code=501,
        detail="Search endpoint not implemented. Use /api/compare for document comparison."
    )


@app.post("/api/index/files", response_model=IndexResponse)
async def index_files(
    files: List[UploadFile] = File(...),
    collection: str = "documents",
    provider: str = "groq"
):
    """
    DEPRECATED: File indexing not implemented in this version.
    This system compares two documents directly without indexing.
    """
    raise HTTPException(
        status_code=501,
        detail="Indexing not implemented. Use /api/compare for direct document comparison."
    )


@app.post("/api/index/bulk", response_model=IndexResponse)
async def index_bulk(request: BulkIndexRequest):
    """
    DEPRECATED: Bulk indexing not implemented in this version.
    This system compares two documents directly without indexing.
    """
    raise HTTPException(
        status_code=501,
        detail="Bulk indexing not implemented. Use /api/compare for direct document comparison."
    )


@app.get("/api/collections", response_model=List[CollectionStats])
async def list_collections():
    """
    DEPRECATED: Collection listing not implemented in this version.
    No vector database is used in this system.
    """
    raise HTTPException(
        status_code=501,
        detail="Collections not implemented. This system does not use a vector database."
    )


@app.delete("/api/collections/{collection_name}")
async def delete_collection(collection_name: str):
    """
    DEPRECATED: Collection deletion not implemented in this version.
    No vector database is used in this system.
    """
    raise HTTPException(
        status_code=501,
        detail="Collections not implemented. This system does not use a vector database."
    )


if __name__ == "__main__":
    import os
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=[backend_dir],  # Only watch backend directory
        log_level="info"
    )
