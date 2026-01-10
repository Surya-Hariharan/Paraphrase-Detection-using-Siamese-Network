"""
Data Validation Agent
======================

Validates and cleans training data before training begins.

Checks for:
- Label noise (mislabeled pairs)
- Duplicates
- Outliers (very long/short sentences, non-English, etc.)
- Class imbalance
"""

from typing import List, Dict, Any, Tuple
import re
from collections import Counter
from .agent_config import AgentConfig


class DataValidationAgent:
    """Agent for validating training data quality."""
    
    def __init__(self, config: AgentConfig):
        """
        Initialize data validation agent.
        
        Args:
            config: Agent configuration with LLM client
        """
        self.config = config
        self.validation_results = {}
    
    def validate_batch(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate a batch of training data.
        
        Args:
            batch: List of training examples with text_a, text_b, label
        
        Returns:
            Validation results with warnings and suggestions
        """
        results = {
            "valid": True,
            "warnings": [],
            "fixes_applied": [],
            "statistics": {}
        }
        
        # Check for duplicates
        duplicates = self._check_duplicates(batch)
        if duplicates:
            results["warnings"].append(f"Found {len(duplicates)} duplicate pairs")
            results["statistics"]["duplicates"] = len(duplicates)
        
        # Check for outliers
        outliers = self._check_outliers(batch)
        if outliers:
            results["warnings"].append(f"Found {len(outliers)} outlier pairs (length)")
            results["statistics"]["outliers"] = len(outliers)
        
        # Check label distribution
        label_dist = self._check_label_distribution(batch)
        results["statistics"]["label_distribution"] = label_dist
        
        if abs(label_dist[0] - label_dist[1]) > 0.3:
            results["warnings"].append(
                f"Class imbalance detected: {label_dist[0]:.1%} vs {label_dist[1]:.1%}"
            )
        
        # Check for non-English text
        non_english = self._check_language(batch)
        if non_english:
            results["warnings"].append(f"Found {len(non_english)} non-English samples")
            results["statistics"]["non_english"] = len(non_english)
        
        # Log validation
        if results["warnings"]:
            self.config.log_intervention(
                "DataValidator",
                f"Validated batch of {len(batch)} samples. Warnings: {', '.join(results['warnings'])}"
            )
        
        return results
    
    def validate_sample(self, text_a: str, text_b: str, label: int) -> Dict[str, Any]:
        """
        Validate a single sample and suggest if it's potentially mislabeled.
        
        Args:
            text_a: First text
            text_b: Second text
            label: Ground truth label (0 or 1)
        
        Returns:
            Validation result with LLM assessment
        """
        # Basic checks
        if len(text_a) < 5 or len(text_b) < 5:
            return {
                "valid": False,
                "reason": "Text too short (< 5 chars)",
                "suggested_action": "remove"
            }
        
        if text_a == text_b:
            if label == 0:
                return {
                    "valid": False,
                    "reason": "Identical texts but labeled as non-paraphrase",
                    "suggested_action": "fix_label",
                    "suggested_label": 1
                }
        
        # Ask LLM to validate if suspicious
        if self._is_suspicious(text_a, text_b, label):
            assessment = self._get_llm_assessment(text_a, text_b, label)
            return assessment
        
        return {"valid": True}
    
    def _check_duplicates(self, batch: List[Dict[str, Any]]) -> List[int]:
        """Find duplicate text pairs."""
        seen = set()
        duplicates = []
        
        for i, item in enumerate(batch):
            pair = (item['text_a'].strip(), item['text_b'].strip())
            if pair in seen:
                duplicates.append(i)
            seen.add(pair)
        
        return duplicates
    
    def _check_outliers(self, batch: List[Dict[str, Any]]) -> List[int]:
        """Find outliers based on text length."""
        lengths = [(len(item['text_a']) + len(item['text_b'])) for item in batch]
        mean_len = sum(lengths) / len(lengths)
        std_len = (sum((x - mean_len) ** 2 for x in lengths) / len(lengths)) ** 0.5
        
        outliers = []
        for i, length in enumerate(lengths):
            if abs(length - mean_len) > 3 * std_len:
                outliers.append(i)
        
        return outliers
    
    def _check_label_distribution(self, batch: List[Dict[str, Any]]) -> Dict[int, float]:
        """Check label distribution."""
        labels = [item['label'] for item in batch]
        counts = Counter(labels)
        total = len(labels)
        
        return {
            0: counts.get(0, 0) / total,
            1: counts.get(1, 0) / total
        }
    
    def _check_language(self, batch: List[Dict[str, Any]]) -> List[int]:
        """Detect non-English text (simple heuristic)."""
        non_english = []
        
        for i, item in enumerate(batch):
            text = item['text_a'] + " " + item['text_b']
            # Simple check: if more than 30% non-ASCII, likely non-English
            non_ascii = sum(1 for c in text if ord(c) > 127)
            if non_ascii / len(text) > 0.3:
                non_english.append(i)
        
        return non_english
    
    def _is_suspicious(self, text_a: str, text_b: str, label: int) -> bool:
        """Check if a sample is suspicious and needs LLM validation."""
        # Check for very similar texts labeled as non-paraphrase
        if label == 0:
            # Simple similarity check
            words_a = set(text_a.lower().split())
            words_b = set(text_b.lower().split())
            overlap = len(words_a & words_b) / max(len(words_a), len(words_b))
            
            if overlap > 0.8:
                return True
        
        # Check for very different texts labeled as paraphrase
        if label == 1:
            words_a = set(text_a.lower().split())
            words_b = set(text_b.lower().split())
            overlap = len(words_a & words_b) / max(len(words_a), len(words_b))
            
            if overlap < 0.2:
                return True
        
        return False
    
    def _get_llm_assessment(self, text_a: str, text_b: str, label: int) -> Dict[str, Any]:
        """Use LLM to assess if the sample is correctly labeled."""
        prompt = f"""You are a data quality expert. Analyze this paraphrase detection sample:

Text A: "{text_a}"
Text B: "{text_b}"
Current Label: {"Paraphrase (1)" if label == 1 else "Not Paraphrase (0)"}

Task: Determine if this label is correct. Consider:
1. Do the texts convey the same meaning?
2. Is the labeling reasonable?

Respond in this exact format:
ASSESSMENT: [CORRECT or INCORRECT]
CONFIDENCE: [0.0 to 1.0]
REASON: [Brief explanation]
SUGGESTED_LABEL: [0 or 1]
"""
        
        try:
            response = self.config.generate_response(prompt, temperature=0.3)
            
            # Parse response
            assessment = "CORRECT"
            confidence = 0.5
            reason = "Unable to parse LLM response"
            suggested_label = label
            
            if "ASSESSMENT:" in response:
                assessment = "INCORRECT" if "INCORRECT" in response else "CORRECT"
            if "CONFIDENCE:" in response:
                try:
                    conf_match = re.search(r'CONFIDENCE:\s*([0-9.]+)', response)
                    if conf_match:
                        confidence = float(conf_match.group(1))
                except:
                    pass
            if "REASON:" in response:
                reason_match = re.search(r'REASON:\s*(.+?)(?:\n|$)', response)
                if reason_match:
                    reason = reason_match.group(1).strip()
            if "SUGGESTED_LABEL:" in response:
                label_match = re.search(r'SUGGESTED_LABEL:\s*([01])', response)
                if label_match:
                    suggested_label = int(label_match.group(1))
            
            self.config.log_intervention(
                "DataValidator",
                f"LLM Assessment - Label {label}: {assessment} (conf: {confidence:.2f})"
            )
            
            return {
                "valid": assessment == "CORRECT",
                "confidence": confidence,
                "reason": reason,
                "suggested_label": suggested_label,
                "llm_response": response
            }
        
        except Exception as e:
            self.config.log_intervention("DataValidator", f"LLM assessment failed: {str(e)}")
            return {"valid": True, "reason": "LLM assessment unavailable"}
