"""
Inference Validation Agent
===========================

AI agent for validating model inference results and performance.
"""

from typing import Dict, List, Any, Tuple, Optional
import numpy as np
from datetime import datetime
from .agent_config import AgentConfig


class InferenceValidatorAgent:
    """Agent for validating inference results and model performance"""
    
    def __init__(self, config: AgentConfig, confidence_threshold: float = 0.7):
        """
        Initialize inference validation agent
        
        Args:
            config: Agent configuration
            confidence_threshold: Minimum confidence for predictions
        """
        self.config = config
        self.confidence_threshold = confidence_threshold
        self.inference_history = []
        
    def validate_prediction(
        self, 
        text_a: str, 
        text_b: str, 
        similarity: float,
        processing_time_ms: float
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate a single prediction
        
        Args:
            text_a: First text
            text_b: Second text
            similarity: Predicted similarity score
            processing_time_ms: Processing time in milliseconds
            
        Returns:
            Tuple of (is_valid, validation_details)
        """
        validation = {
            'timestamp': datetime.now().isoformat(),
            'is_valid': True,
            'issues': [],
            'warnings': []
        }
        
        # Check similarity range
        if not (0.0 <= similarity <= 1.0):
            validation['is_valid'] = False
            validation['issues'].append(
                f"Similarity out of range: {similarity}"
            )
        
        # Check processing time
        if processing_time_ms > self.config.response_time_threshold_ms:
            validation['warnings'].append(
                f"Slow processing: {processing_time_ms:.1f}ms > "
                f"{self.config.response_time_threshold_ms}ms"
            )
        
        # Check text validity
        if not text_a or not text_b:
            validation['is_valid'] = False
            validation['issues'].append("Empty text provided")
        
        if len(text_a) < self.config.min_text_length or len(text_b) < self.config.min_text_length:
            validation['warnings'].append(
                f"Text length below minimum ({self.config.min_text_length} chars)"
            )
        
        # Check for potential errors
        if similarity == 0.0 and text_a == text_b:
            validation['warnings'].append(
                "Identical texts with 0 similarity - possible error"
            )
        
        if similarity == 1.0 and text_a != text_b:
            if self._texts_very_different(text_a, text_b):
                validation['warnings'].append(
                    "Very different texts with 1.0 similarity - verify"
                )
        
        # Store in history
        self.inference_history.append({
            'timestamp': validation['timestamp'],
            'text_a_length': len(text_a),
            'text_b_length': len(text_b),
            'similarity': similarity,
            'processing_time_ms': processing_time_ms,
            'is_valid': validation['is_valid']
        })
        
        return validation['is_valid'], validation
    
    def _texts_very_different(self, text_a: str, text_b: str) -> bool:
        """Check if texts are very different"""
        # Simple heuristic: check word overlap
        words_a = set(text_a.lower().split())
        words_b = set(text_b.lower().split())
        
        if not words_a or not words_b:
            return True
        
        overlap = len(words_a & words_b) / len(words_a | words_b)
        return overlap < 0.1  # Less than 10% word overlap
    
    def validate_batch(
        self, 
        predictions: List[Dict[str, Any]]
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate a batch of predictions
        
        Args:
            predictions: List of prediction dictionaries
            
        Returns:
            Tuple of (all_valid, validation_summary)
        """
        if len(predictions) > self.config.batch_size_limit:
            return False, {
                'error': f"Batch size {len(predictions)} exceeds limit "
                        f"{self.config.batch_size_limit}"
            }
        
        all_valid = True
        issues = []
        warnings = []
        
        for i, pred in enumerate(predictions):
            is_valid, details = self.validate_prediction(
                pred.get('text_a', ''),
                pred.get('text_b', ''),
                pred.get('similarity', -1.0),
                pred.get('processing_time_ms', 0.0)
            )
            
            if not is_valid:
                all_valid = False
                issues.extend([f"Prediction {i}: {issue}" for issue in details['issues']])
            
            warnings.extend([f"Prediction {i}: {warn}" for warn in details['warnings']])
        
        summary = {
            'total_predictions': len(predictions),
            'valid_predictions': sum(1 for p in predictions if p.get('similarity', -1) >= 0),
            'all_valid': all_valid,
            'issues': issues,
            'warnings': warnings
        }
        
        return all_valid, summary
    
    def analyze_performance(self) -> Dict[str, Any]:
        """
        Analyze inference performance from history
        
        Returns:
            Performance metrics dictionary
        """
        if not self.inference_history:
            return {'error': 'No inference history available'}
        
        processing_times = [h['processing_time_ms'] for h in self.inference_history]
        similarities = [h['similarity'] for h in self.inference_history]
        
        analysis = {
            'total_inferences': len(self.inference_history),
            'valid_inferences': sum(1 for h in self.inference_history if h['is_valid']),
            'avg_processing_time_ms': np.mean(processing_times),
            'max_processing_time_ms': np.max(processing_times),
            'min_processing_time_ms': np.min(processing_times),
            'p95_processing_time_ms': np.percentile(processing_times, 95),
            'avg_similarity': np.mean(similarities),
            'similarity_std': np.std(similarities),
            'slow_inferences': sum(
                1 for t in processing_times 
                if t > self.config.response_time_threshold_ms
            )
        }
        
        return analysis
    
    def generate_report(self) -> str:
        """
        Generate inference performance report
        
        Returns:
            Formatted report string
        """
        analysis = self.analyze_performance()
        
        if 'error' in analysis:
            return f"Error: {analysis['error']}"
        
        report = []
        report.append("=" * 80)
        report.append("INFERENCE VALIDATION REPORT")
        report.append("=" * 80)
        
        report.append(f"\nTotal Inferences: {analysis['total_inferences']:,}")
        report.append(f"Valid Inferences: {analysis['valid_inferences']:,}")
        report.append(f"Success Rate: {analysis['valid_inferences']/analysis['total_inferences']:.2%}")
        
        report.append("\nProcessing Time Metrics:")
        report.append(f"  Average: {analysis['avg_processing_time_ms']:.2f}ms")
        report.append(f"  P95: {analysis['p95_processing_time_ms']:.2f}ms")
        report.append(f"  Max: {analysis['max_processing_time_ms']:.2f}ms")
        report.append(f"  Slow Inferences (>{self.config.response_time_threshold_ms}ms): "
                     f"{analysis['slow_inferences']:,}")
        
        report.append("\nSimilarity Distribution:")
        report.append(f"  Average: {analysis['avg_similarity']:.4f}")
        report.append(f"  Std Dev: {analysis['similarity_std']:.4f}")
        
        report.append("\n" + "=" * 80)
        
        return "\n".join(report)
    
    def clear_history(self):
        """Clear inference history"""
        self.inference_history = []
