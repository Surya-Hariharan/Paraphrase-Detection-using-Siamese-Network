"""Agentic Validator for Intelligent Paraphrase Detection using Google Gemini"""

import os
from enum import Enum
from typing import Dict, Tuple, Optional
from datetime import datetime

try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


class ConfidenceLevel(Enum):
    """Model confidence levels"""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    UNCERTAIN = "UNCERTAIN"


class AgenticValidator:
    """
    Gemini-powered intelligent validator for paraphrase detection edge cases
    
    Features:
    - Smart triggering: Activates when model likely missed paraphrases
    - Edge case detection with semantic reasoning
    - Confidence-based routing
    """
    
    def __init__(self):
        """Initialize Gemini API"""
        gemini_key = os.getenv("GEMINI_API_KEY")
        
        if not GEMINI_AVAILABLE:
            print("⚠️  Warning: google-generativeai not installed. Agentic AI disabled.")
            self.enabled = False
            return
        
        if not gemini_key:
            print("⚠️  Warning: No GEMINI_API_KEY found. Agentic AI disabled.")
            self.enabled = False
            return
        
        self.enabled = True
        
        # Initialize Gemini client
        try:
            self.client = genai.Client(api_key=gemini_key)
            print("✓ Agentic AI initialized with Gemini")
        except Exception as e:
            print(f"⚠️  Warning: Failed to initialize Gemini: {e}")
            self.enabled = False
            return
        
        # Statistics tracking
        self._total_validations = 0
        self._agent_activations = 0
        self._paraphrase_rescues = 0
        self._edge_case_counts = {
            "length_mismatch": 0,
            "short_text": 0,
            "exact_match_low_similarity": 0,
            "numeric_heavy": 0,
            "special_chars_heavy": 0,
            "borderline_case": 0
        }
    
    def get_confidence_level(self, similarity: float) -> ConfidenceLevel:
        """Determine confidence level from similarity score"""
        if similarity >= 0.85:
            return ConfidenceLevel.HIGH
        elif similarity >= 0.70:
            return ConfidenceLevel.MEDIUM
        elif similarity >= 0.55:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.UNCERTAIN
    
    def detect_edge_cases(self, text1: str, text2: str) -> list:
        """Detect potential edge cases that might confuse the model"""
        edge_cases = []
        
        len1, len2 = len(text1), len(text2)
        
        # Length mismatch (3x difference)
        if max(len1, len2) > 3 * min(len1, len2) and min(len1, len2) > 0:
            edge_cases.append("length_mismatch")
            self._edge_case_counts["length_mismatch"] += 1
        
        # Short text (less than 20 chars)
        if min(len1, len2) < 20:
            edge_cases.append("short_text")
            self._edge_case_counts["short_text"] += 1
        
        # Numeric heavy (>30% digits)
        def digit_ratio(text):
            return sum(c.isdigit() for c in text) / max(len(text), 1)
        
        if digit_ratio(text1) > 0.3 or digit_ratio(text2) > 0.3:
            edge_cases.append("numeric_heavy")
            self._edge_case_counts["numeric_heavy"] += 1
        
        # Special chars heavy
        def special_ratio(text):
            special = sum(not c.isalnum() and not c.isspace() for c in text)
            return special / max(len(text), 1)
        
        if special_ratio(text1) > 0.2 or special_ratio(text2) > 0.2:
            edge_cases.append("special_chars_heavy")
            self._edge_case_counts["special_chars_heavy"] += 1
        
        return edge_cases
    
    def should_trigger_agent(
        self,
        similarity: float,
        is_paraphrase: bool,
        edge_cases: list
    ) -> bool:
        """
        Determine if agentic validation should be triggered
        
        Triggers when:
        1. Similarity is in borderline range (0.55-0.75) - might be missing paraphrases
        2. Edge cases detected
        3. Low confidence (< 0.55) but could be a paraphrase
        """
        confidence = self.get_confidence_level(similarity)
        
        # HIGH confidence and paraphrase detected - skip agent
        if confidence == ConfidenceLevel.HIGH and is_paraphrase:
            return False
        
        # Borderline case - agent should verify
        if 0.55 <= similarity <= 0.75:
            self._edge_case_counts["borderline_case"] += 1
            return True
        
        # Edge cases detected - agent should verify
        if edge_cases:
            return True
        
        # UNCERTAIN confidence - agent should help
        if confidence == ConfidenceLevel.UNCERTAIN:
            return True
        
        return False
    
    def validate(
        self,
        text1: str,
        text2: str,
        model_similarity: float,
        model_is_paraphrase: bool
    ) -> Dict:
        """
        Validate paraphrase detection using Gemini
        
        Returns:
            Dict with validation results
        """
        self._total_validations += 1
        
        if not self.enabled:
            return {
                "agent_used": False,
                "confidence_level": self.get_confidence_level(model_similarity).value,
                "edge_cases": [],
                "agent_validation": None,
                "agent_reasoning": None,
                "agent_confidence": None,
                "paraphrase_rescued": False,
                "original_similarity": model_similarity,
                "adjusted_similarity": model_similarity
            }
        
        # Detect edge cases
        edge_cases = self.detect_edge_cases(text1, text2)
        
        # Check if agent should be triggered
        should_trigger = self.should_trigger_agent(
            model_similarity,
            model_is_paraphrase,
            edge_cases
        )
        
        if not should_trigger:
            return {
                "agent_used": False,
                "confidence_level": self.get_confidence_level(model_similarity).value,
                "edge_cases": edge_cases,
                "agent_validation": None,
                "agent_reasoning": None,
                "agent_confidence": None,
                "paraphrase_rescued": False,
                "original_similarity": model_similarity,
                "adjusted_similarity": model_similarity
            }
        
        # Agent validation triggered
        self._agent_activations += 1
        
        try:
            # Create prompt for Gemini
            prompt = f"""You are an expert in paraphrase detection. Analyze if these two texts are paraphrases (convey the same meaning).

Text 1: "{text1}"

Text 2: "{text2}"

Model's similarity score: {model_similarity:.2f}
Model's decision: {"PARAPHRASE" if model_is_paraphrase else "NOT PARAPHRASE"}
Detected edge cases: {edge_cases if edge_cases else "None"}

Analyze these texts and determine:
1. Are they truly paraphrases? (YES/NO)
2. Your confidence level (HIGH/MEDIUM/LOW)
3. Brief reasoning (1-2 sentences)

Respond in this exact format:
PARAPHRASE: YES or NO
CONFIDENCE: HIGH, MEDIUM, or LOW
REASONING: Your brief explanation"""

            # Call Gemini using new API
            response = self.client.models.generate_content(
                model="gemini-1.5-flash",
                contents=prompt
            )
            result_text = response.text.strip()
            
            # Parse response
            lines = result_text.split('\n')
            agent_is_paraphrase = False
            agent_confidence = "MEDIUM"
            agent_reasoning = "Analysis completed"
            
            for line in lines:
                line = line.strip()
                if line.startswith("PARAPHRASE:"):
                    agent_is_paraphrase = "YES" in line.upper()
                elif line.startswith("CONFIDENCE:"):
                    conf = line.split(":", 1)[1].strip().upper()
                    if conf in ["HIGH", "MEDIUM", "LOW"]:
                        agent_confidence = conf
                elif line.startswith("REASONING:"):
                    agent_reasoning = line.split(":", 1)[1].strip()
            
            # Check if agent rescued a paraphrase
            paraphrase_rescued = agent_is_paraphrase and not model_is_paraphrase
            if paraphrase_rescued:
                self._paraphrase_rescues += 1
            
            # Adjust similarity if agent disagrees
            adjusted_similarity = model_similarity
            if agent_is_paraphrase and model_similarity < 0.75:
                adjusted_similarity = max(model_similarity, 0.80)
            elif not agent_is_paraphrase and model_similarity > 0.60:
                adjusted_similarity = min(model_similarity, 0.50)
            
            return {
                "agent_used": True,
                "confidence_level": self.get_confidence_level(model_similarity).value,
                "edge_cases": edge_cases,
                "agent_validation": agent_is_paraphrase,
                "agent_reasoning": agent_reasoning,
                "agent_confidence": agent_confidence,
                "paraphrase_rescued": paraphrase_rescued,
                "original_similarity": model_similarity,
                "adjusted_similarity": adjusted_similarity
            }
            
        except Exception as e:
            print(f"⚠️  Agent validation failed: {e}")
            return {
                "agent_used": False,
                "confidence_level": self.get_confidence_level(model_similarity).value,
                "edge_cases": edge_cases,
                "agent_validation": None,
                "agent_reasoning": f"Agent error: {str(e)}",
                "agent_confidence": None,
                "paraphrase_rescued": False,
                "original_similarity": model_similarity,
                "adjusted_similarity": model_similarity
            }
    
    def get_stats(self) -> Dict:
        """Get validation statistics"""
        activation_rate = 0
        rescue_rate = 0
        
        if self._total_validations > 0:
            activation_rate = (self._agent_activations / self._total_validations) * 100
        
        if self._agent_activations > 0:
            rescue_rate = (self._paraphrase_rescues / self._agent_activations) * 100
        
        return {
            "enabled": self.enabled,
            "total_validations": self._total_validations,
            "agent_activations": self._agent_activations,
            "activation_rate": f"{activation_rate:.1f}%",
            "paraphrase_rescues": self._paraphrase_rescues,
            "rescue_rate": f"{rescue_rate:.1f}%",
            "edge_case_counts": self._edge_case_counts
        }


# Global validator instance
agentic_validator = AgenticValidator()
