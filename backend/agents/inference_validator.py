"""
Inference Validator Agent
==========================

Validates predictions during inference and handles edge cases.

Triggers:
- When model confidence is low
- When input is out-of-distribution
- When input is adversarial/ambiguous
"""

from typing import Dict, Any, Optional
import re
from .agent_config import AgentConfig


class InferenceValidatorAgent:
    """Agent for validating inference predictions."""
    
    def __init__(self, config: AgentConfig, confidence_threshold: float = 0.7):
        """
        Initialize inference validator agent.
        
        Args:
            config: Agent configuration with LLM client
            confidence_threshold: Threshold for flagging low-confidence predictions
        """
        self.config = config
        self.confidence_threshold = confidence_threshold
        self.validation_stats = {
            "total_predictions": 0,
            "low_confidence": 0,
            "ood_detected": 0,
            "llm_overrides": 0
        }
    
    def validate_prediction(
        self,
        text_a: str,
        text_b: str,
        prediction: int,
        confidence: float,
        similarity_score: float
    ) -> Dict[str, Any]:
        """
        Validate a prediction and flag edge cases.
        
        Args:
            text_a: First text
            text_b: Second text
            prediction: Model prediction (0 or 1)
            confidence: Model confidence (0.0 to 1.0)
            similarity_score: Cosine similarity score
        
        Returns:
            Validation results with flags and suggestions
        """
        self.validation_stats["total_predictions"] += 1
        
        result = {
            "validated": True,
            "flags": [],
            "llm_check_needed": False,
            "suggested_action": "accept",
            "final_prediction": prediction,
            "final_confidence": confidence
        }
        
        # Check for low confidence
        if confidence < self.confidence_threshold:
            result["flags"].append("low_confidence")
            result["llm_check_needed"] = True
            self.validation_stats["low_confidence"] += 1
            
            self.config.log_intervention(
                "InferenceValidator",
                f"Low confidence: {confidence:.3f} for prediction {prediction}"
            )
        
        # Check for out-of-distribution inputs
        if self._is_out_of_distribution(text_a, text_b):
            result["flags"].append("ood_input")
            result["llm_check_needed"] = True
            self.validation_stats["ood_detected"] += 1
            
            self.config.log_intervention(
                "InferenceValidator",
                f"OOD detected: text_a_len={len(text_a)}, text_b_len={len(text_b)}"
            )
        
        # Check for edge case patterns
        edge_case = self._check_edge_case(text_a, text_b, prediction)
        if edge_case:
            result["flags"].append(edge_case)
            result["llm_check_needed"] = True
        
        # If LLM check needed, get second opinion
        if result["llm_check_needed"]:
            llm_result = self._get_llm_validation(
                text_a, text_b, prediction, confidence, similarity_score
            )
            
            if llm_result:
                result["llm_prediction"] = llm_result["prediction"]
                result["llm_confidence"] = llm_result["confidence"]
                result["llm_reasoning"] = llm_result["reasoning"]
                
                # If LLM disagrees strongly, flag for review
                if llm_result["prediction"] != prediction and llm_result["confidence"] > 0.8:
                    result["suggested_action"] = "human_review"
                    result["flags"].append("llm_disagrees")
                    self.validation_stats["llm_overrides"] += 1
                    
                    self.config.log_intervention(
                        "InferenceValidator",
                        f"LLM override: Model={prediction}, LLM={llm_result['prediction']}"
                    )
        
        return result
    
    def _is_out_of_distribution(self, text_a: str, text_b: str) -> bool:
        """Check if input is out-of-distribution."""
        # Check text length
        if len(text_a) < 5 or len(text_b) < 5:
            return True
        
        if len(text_a) > 1000 or len(text_b) > 1000:
            return True
        
        # Check for non-English (simple heuristic)
        combined = text_a + " " + text_b
        non_ascii_ratio = sum(1 for c in combined if ord(c) > 127) / len(combined)
        if non_ascii_ratio > 0.3:
            return True
        
        # Check for special patterns (URLs, code, etc.)
        if re.search(r'https?://', combined) or re.search(r'[{}()<>]', combined):
            return True
        
        return False
    
    def _check_edge_case(self, text_a: str, text_b: str, prediction: int) -> Optional[str]:
        """Check for known edge case patterns."""
        # Identical texts
        if text_a.strip() == text_b.strip():
            if prediction == 0:
                return "identical_but_labeled_nonparaphrase"
        
        # Very short texts
        if len(text_a.split()) < 3 or len(text_b.split()) < 3:
            return "very_short_text"
        
        # One text is substring of another
        if text_a in text_b or text_b in text_a:
            return "substring_relationship"
        
        # Negation patterns
        negations = ["not", "no", "never", "neither", "nor", "n't"]
        words_a = text_a.lower().split()
        words_b = text_b.lower().split()
        
        neg_a = any(neg in words_a for neg in negations)
        neg_b = any(neg in words_b for neg in negations)
        
        if neg_a != neg_b and prediction == 1:
            return "negation_difference"
        
        return None
    
    def _get_llm_validation(
        self,
        text_a: str,
        text_b: str,
        model_prediction: int,
        model_confidence: float,
        similarity_score: float
    ) -> Optional[Dict[str, Any]]:
        """Get LLM second opinion on the prediction."""
        prompt = f"""You are a paraphrase detection expert. Validate this prediction:

Text A: "{text_a}"
Text B: "{text_b}"

Model Prediction: {"Paraphrase" if model_prediction == 1 else "Not Paraphrase"}
Model Confidence: {model_confidence:.3f}
Similarity Score: {similarity_score:.3f}

Task: Determine if these texts are paraphrases (same meaning).

Respond in this exact format:
PREDICTION: [PARAPHRASE or NOT_PARAPHRASE]
CONFIDENCE: [0.0 to 1.0]
REASONING: [Brief explanation in one sentence]
"""
        
        try:
            response = self.config.generate_response(prompt, temperature=0.3)
            
            # Parse response
            prediction = 1  # default
            confidence = 0.5
            reasoning = "Unable to parse LLM response"
            
            if "NOT_PARAPHRASE" in response:
                prediction = 0
            elif "PARAPHRASE" in response:
                prediction = 1
            
            conf_match = re.search(r'CONFIDENCE:\s*([0-9.]+)', response)
            if conf_match:
                confidence = float(conf_match.group(1))
            
            reason_match = re.search(r'REASONING:\s*(.+?)(?:\n|$)', response)
            if reason_match:
                reasoning = reason_match.group(1).strip()
            
            return {
                "prediction": prediction,
                "confidence": confidence,
                "reasoning": reasoning,
                "full_response": response
            }
        
        except Exception as e:
            self.config.log_intervention("InferenceValidator", f"LLM validation failed: {str(e)}")
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get validation statistics."""
        stats = self.validation_stats.copy()
        
        if stats["total_predictions"] > 0:
            stats["low_confidence_rate"] = stats["low_confidence"] / stats["total_predictions"]
            stats["ood_rate"] = stats["ood_detected"] / stats["total_predictions"]
            stats["override_rate"] = stats["llm_overrides"] / stats["total_predictions"]
        
        return stats
