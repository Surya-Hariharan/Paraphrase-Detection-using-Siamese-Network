"""Optimized inference service with caching and agentic AI validation"""

import torch
from typing import Tuple, Dict
import time
import hashlib
from threading import Lock

from backend.core.models.siamese import SiameseModel
from backend.config.settings import settings
from backend.services.agentic_validator import agentic_validator


class InferenceEngine:
    """High-performance inference engine with caching and agentic AI for edge cases"""
    
    def __init__(self):
        self.model = None
        self.device = torch.device(settings.model.DEVICE if torch.cuda.is_available() else "cpu")
        
        # Performance optimization
        self._cache: Dict[str, Tuple[float, bool]] = {}
        self._cache_lock = Lock()
        self._cache_hits = 0
        self._total_requests = 0
        self._agent_validations = 0
        self._max_cache_size = 10000
        
        self.load_model()
    
    def load_model(self):
        """Load and optimize model from checkpoint"""
        try:
            self.model = SiameseModel(
                encoder_name=settings.model.ENCODER_NAME,
                embedding_dim=settings.model.EMBEDDING_DIM,
                projection_dim=settings.model.PROJECTION_DIM
            )
            
            self.model.load(settings.model.MODEL_PATH)
            self.model.to(self.device)
            self.model.eval()
            
            # Enable inference optimizations
            if self.device.type == "cuda":
                torch.backends.cudnn.benchmark = True
                print(f"✓ CUDA optimizations enabled")
            
            print(f"✓ Model loaded from {settings.model.MODEL_PATH} on {self.device}")
        except Exception as e:
            print(f"✗ Failed to load model: {e}")
            self.model = None
    
    def _get_cache_key(self, text_a: str, text_b: str) -> str:
        """Generate deterministic cache key (handles A,B and B,A as same)"""
        texts = sorted([text_a.strip().lower(), text_b.strip().lower()])
        combined = "|".join(texts)
        return hashlib.md5(combined.encode()).hexdigest()
    
    def predict(self, text_a: str, text_b: str, use_cache: bool = True, use_agent: bool = True) -> Tuple[float, bool, float, Dict]:
        """
        Predict paraphrase similarity with caching and agentic validation
        
        Args:
            text_a: First text
            text_b: Second text
            use_cache: Whether to use cache (default: True)
            use_agent: Whether to use agentic AI for edge cases (default: True)
        
        Returns:
            similarity_score, is_paraphrase, inference_time_ms, agent_metadata
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        self._total_requests += 1
        
        # Check cache
        if use_cache:
            cache_key = self._get_cache_key(text_a, text_b)
            with self._cache_lock:
                if cache_key in self._cache:
                    self._cache_hits += 1
                    similarity, is_paraphrase = self._cache[cache_key]
                    return similarity, is_paraphrase, 0.0, {'cached': True, 'agent_used': False}
        
        start_time = time.time()
        
        # Inference with no gradient computation
        with torch.no_grad():
            raw_similarity = self.model.predict(text_a, text_b)
        
        inference_time = (time.time() - start_time) * 1000  # ms
        
        # Agentic validation for edge cases
        agent_metadata = {'cached': False, 'agent_used': False}
        
        if use_agent:
            adjusted_similarity, final_decision, validation_metadata = agentic_validator.validate_prediction(
                text_a, text_b, raw_similarity
            )
            
            if validation_metadata.get('used_agent_validation'):
                self._agent_validations += 1
            
            agent_metadata = {
                'cached': False,
                'agent_used': True,
                'confidence_level': validation_metadata['confidence_level'],
                'edge_cases': validation_metadata['edge_cases'],
                'agent_validation': validation_metadata.get('used_agent_validation', False),
                'agent_reasoning': validation_metadata.get('agent_reasoning'),
                'agent_confidence': validation_metadata.get('agent_confidence'),
                'paraphrase_rescued': validation_metadata.get('paraphrase_rescued', False),
                'original_similarity': raw_similarity,
                'adjusted_similarity': adjusted_similarity
            }
            
            similarity = adjusted_similarity
            is_paraphrase = final_decision
        else:
            similarity = raw_similarity
            is_paraphrase = similarity > settings.model.SIMILARITY_THRESHOLD
        
        # Cache result (with size limit)
        if use_cache:
            with self._cache_lock:
                if len(self._cache) < self._max_cache_size:
                    self._cache[cache_key] = (similarity, is_paraphrase)
                elif self._total_requests % 100 == 0:
                    # Periodically clear old cache when full
                    self._cache.clear()
        
        return similarity, is_paraphrase, inference_time, agent_metadata
    
    def is_ready(self) -> bool:
        """Check if model is loaded and ready"""
        return self.model is not None
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        cache_hit_rate = (self._cache_hits / self._total_requests * 100) if self._total_requests > 0 else 0
        agent_usage_rate = (self._agent_validations / self._total_requests * 100) if self._total_requests > 0 else 0
        
        return {
            "total_requests": self._total_requests,
            "cache_hits": self._cache_hits,
            "cache_hit_rate": f"{cache_hit_rate:.2f}%",
            "cache_size": len(self._cache),
            "max_cache_size": self._max_cache_size,
            "agent_validations": self._agent_validations,
            "agent_usage_rate": f"{agent_usage_rate:.2f}%",
            "model_loaded": self.is_ready(),
            "device": str(self.device),
            "agentic_ai": agentic_validator.get_stats()
        }
    
    def clear_cache(self):
            self._agent_validations = 0
        """Clear inference cache"""
        with self._cache_lock:
            self._cache.clear()
            self._cache_hits = 0
            self._total_requests = 0


# Global inference engine
inference_engine = InferenceEngine()
