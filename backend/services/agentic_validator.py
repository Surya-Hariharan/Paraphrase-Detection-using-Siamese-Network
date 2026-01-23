"""CrewAI-powered Agentic Validator for Intelligent Paraphrase Detection"""

import os
from enum import Enum
from typing import Dict, List, Tuple
from datetime import datetime

from crewai import Agent, Task, Crew, Process
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq


class ConfidenceLevel(Enum):
    """Model confidence levels"""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    UNCERTAIN = "UNCERTAIN"


class AgenticValidator:
    """
    CrewAI-powered intelligent validator for paraphrase detection edge cases
    
    Features:
    - Multi-agent analysis with specialized roles
    - Smart triggering: Activates when model likely missed paraphrases
    - Edge case detection with semantic reasoning
    - Confidence-based routing
    """
    
    def __init__(self):
        """Initialize CrewAI agents and LLM"""
        # Get API key (prefer Gemini, fallback to Groq)
        gemini_key = os.getenv("GEMINI_API_KEY")
        groq_key = os.getenv("GROQ_API_KEY")
        
        if not gemini_key and not groq_key:
            print("⚠️  Warning: No GEMINI_API_KEY or GROQ_API_KEY found. Agentic AI disabled.")
            self.enabled = False
            return
        
        self.enabled = True
        
        # Initialize LLM (prefer Gemini, fallback to Groq)
        try:
            if gemini_key:
                self.llm = ChatGoogleGenerativeAI(
                    model="gemini-1.5-flash",
                    google_api_key=gemini_key,
                    temperature=0.3,
                    convert_system_message_to_human=True
                )
                print("✓ CrewAI initialized with Gemini")
            elif groq_key:
                self.llm = ChatGroq(
                    model="llama-3.3-70b-versatile",
                    groq_api_key=groq_key,
                    temperature=0.3
                )
                print("✓ CrewAI initialized with Groq")
        except Exception as e:
            print(f"⚠️  Warning: Failed to initialize LLM: {e}")
            self.enabled = False
            return
        
        # Statistics tracking
        self._total_validations = 0
        self._agent_activations = 0
        self._paraphrase_rescues = 0  # Times agent corrected model
        self._edge_case_counts = {
            "length_mismatch": 0,
            "short_text": 0,
            "exact_match_low_similarity": 0,
            "numeric_heavy": 0,
            "special_chars_heavy": 0,
            "low_confidence_paraphrase": 0,  # New
            "borderline_case": 0  # New
        }
        
        # Create specialized agents
        self._create_agents()
    
    def _create_agents(self):
        """Create specialized CrewAI agents for paraphrase analysis"""
        
        # Agent 1: Paraphrase Analyzer
        self.paraphrase_analyzer = Agent(
            role="Senior Paraphrase Detection Expert",
            goal="Accurately identify if two texts are paraphrases by analyzing semantic meaning, intent, and context",
            backstory="""You are a world-class expert in natural language understanding with 15 years 
            of experience in paraphrase detection, semantic similarity analysis, and linguistic reasoning. 
            You excel at identifying paraphrases even when texts differ significantly in length, structure, 
            or wording. You understand that paraphrases convey the same core meaning despite surface differences.""",
            verbose=False,
            allow_delegation=False,
            llm=self.llm
        )
        
        # Agent 2: Edge Case Specialist
        self.edge_case_specialist = Agent(
            role="Edge Case Detection Specialist",
            goal="Identify edge cases and anomalies in text comparisons that might confuse ML models",
            backstory="""You are an expert at identifying edge cases in NLP systems. You specialize in 
            detecting length mismatches, short texts, numeric-heavy content, special characters, and 
            other anomalies that can cause neural models to produce unreliable predictions. You provide 
            clear explanations of why each edge case matters.""",
            verbose=False,
            allow_delegation=False,
            llm=self.llm
        )
        
        # Agent 3: Semantic Validator
        self.semantic_validator = Agent(
            role="Semantic Equivalence Validator",
            goal="Validate semantic equivalence by analyzing deep meaning, context, and logical relationships",
            backstory="""You are a semantic analysis expert with deep knowledge of linguistics, logic, 
            and meaning representation. You can determine if two texts express the same proposition, 
            even if they use completely different words. You consider context, implications, and 
            pragmatic meaning beyond just surface-level word matching.""",
            verbose=False,
            allow_delegation=False,
            llm=self.llm
        )
    
    def _calculate_confidence(self, similarity: float) -> ConfidenceLevel:
        """Calculate confidence level based on similarity score"""
        if similarity > 0.85:
            return ConfidenceLevel.HIGH
        elif similarity > 0.70:
            return ConfidenceLevel.MEDIUM
        elif similarity > 0.55:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.UNCERTAIN
    
    def check_edge_cases(self, text_a: str, text_b: str, similarity: float) -> List[str]:
        """
        Detect edge cases that might confuse the model
        
        Returns:
            List of detected edge case types
        """
        edge_cases = []
        
        # 1. Length mismatch (3x or more difference)
        len_a, len_b = len(text_a.split()), len(text_b.split())
        if len_a > 0 and len_b > 0:
            ratio = max(len_a, len_b) / min(len_a, len_b)
            if ratio >= 3.0:
                edge_cases.append("length_mismatch")
                self._edge_case_counts["length_mismatch"] += 1
        
        # 2. Short text (< 10 words each)
        if len_a < 10 and len_b < 10:
            edge_cases.append("short_text")
            self._edge_case_counts["short_text"] += 1
        
        # 3. Exact match but low similarity (model error)
        if text_a.strip().lower() == text_b.strip().lower() and similarity < 0.7:
            edge_cases.append("exact_match_low_similarity")
            self._edge_case_counts["exact_match_low_similarity"] += 1
        
        # 4. Numeric heavy (>30% numbers)
        def is_numeric_heavy(text: str) -> bool:
            digits = sum(c.isdigit() for c in text)
            return digits / len(text) > 0.3 if len(text) > 0 else False
        
        if is_numeric_heavy(text_a) or is_numeric_heavy(text_b):
            edge_cases.append("numeric_heavy")
            self._edge_case_counts["numeric_heavy"] += 1
        
        # 5. Special characters heavy (>20% special chars)
        def is_special_char_heavy(text: str) -> bool:
            special = sum(not c.isalnum() and not c.isspace() for c in text)
            return special / len(text) > 0.2 if len(text) > 0 else False
        
        if is_special_char_heavy(text_a) or is_special_char_heavy(text_b):
            edge_cases.append("special_chars_heavy")
            self._edge_case_counts["special_chars_heavy"] += 1
        
        # 6. Low confidence but might be paraphrase (borderline case)
        if 0.55 <= similarity <= 0.70:
            edge_cases.append("borderline_case")
            self._edge_case_counts["borderline_case"] += 1
        
        # 7. Very low similarity but texts look semantically similar
        if similarity < 0.55:
            edge_cases.append("low_confidence_paraphrase")
            self._edge_case_counts["low_confidence_paraphrase"] += 1
        
        return edge_cases
    
    def should_trigger_agent(
        self, 
        text_a: str, 
        text_b: str, 
        similarity: float, 
        edge_cases: List[str]
    ) -> bool:
        """
        Smart triggering logic: Activate agents when they can add value
        
        Triggers when:
        1. Model is uncertain (similarity < 0.55)
        2. Borderline cases (0.55-0.70) - might miss paraphrases
        3. Edge cases detected that could confuse model
        4. Model says "not paraphrase" but might be wrong (0.40-0.75)
        
        Returns:
            True if agents should analyze, False otherwise
        """
        confidence = self._calculate_confidence(similarity)
        
        # HIGH confidence (>0.85) - trust model, no need for agents
        if confidence == ConfidenceLevel.HIGH:
            return False
        
        # UNCERTAIN (<0.55) - always use agents
        if confidence == ConfidenceLevel.UNCERTAIN:
            return True
        
        # Edge cases present - use agents for validation
        if edge_cases:
            return True
        
        # Borderline cases (0.55-0.75) - model might miss paraphrases
        # This is the key for catching false negatives
        if 0.55 <= similarity <= 0.75:
            return True
        
        # Default: trust model
        return False
    
    def validate_with_crew(
        self, 
        text_a: str, 
        text_b: str, 
        similarity: float, 
        edge_cases: List[str]
    ) -> Tuple[bool, str, str]:
        """
        Use CrewAI multi-agent system to validate paraphrase detection
        
        Returns:
            (is_paraphrase, confidence_level, reasoning)
        """
        if not self.enabled:
            return None, None, "CrewAI not enabled"
        
        self._agent_activations += 1
        
        # Task 1: Analyze edge cases
        edge_case_task = Task(
            description=f"""Analyze these two texts for edge cases and potential issues:

Text A: "{text_a}"
Text B: "{text_b}"

ML Model Similarity: {similarity:.2f}
Detected Edge Cases: {', '.join(edge_cases) if edge_cases else 'None'}

Identify any edge cases, anomalies, or factors that might cause the ML model to produce 
unreliable predictions. Consider length differences, special characters, numbers, etc.

Provide a brief analysis (2-3 sentences).""",
            agent=self.edge_case_specialist,
            expected_output="Analysis of edge cases and model reliability factors"
        )
        
        # Task 2: Validate semantic equivalence
        semantic_task = Task(
            description=f"""Determine if these texts are semantically equivalent (paraphrases):

Text A: "{text_a}"
Text B: "{text_b}"

ML Model Similarity: {similarity:.2f}

Analyze the deep semantic meaning and determine if these texts express the same core 
message or proposition. Consider:
- Do they convey the same information?
- Would they have the same truth value in any context?
- Are the intentions and implications equivalent?

Answer with: PARAPHRASE or NOT_PARAPHRASE
Then explain your reasoning (2-3 sentences).""",
            agent=self.semantic_validator,
            expected_output="Semantic equivalence determination with reasoning"
        )
        
        # Task 3: Final paraphrase decision
        final_decision_task = Task(
            description=f"""Based on the edge case analysis and semantic validation, make a final decision:

Text A: "{text_a}"
Text B: "{text_b}"

ML Model Similarity: {similarity:.2f}
Edge Cases: {', '.join(edge_cases) if edge_cases else 'None'}

Determine:
1. Are these texts paraphrases? (YES/NO)
2. Confidence level: HIGH, MEDIUM, or LOW
3. Brief reasoning (2-3 sentences)

Format your response exactly as:
DECISION: [YES/NO]
CONFIDENCE: [HIGH/MEDIUM/LOW]
REASONING: [Your explanation]""",
            agent=self.paraphrase_analyzer,
            expected_output="Final paraphrase decision with confidence and reasoning"
        )
        
        # Create and run crew
        crew = Crew(
            agents=[self.edge_case_specialist, self.semantic_validator, self.paraphrase_analyzer],
            tasks=[edge_case_task, semantic_task, final_decision_task],
            process=Process.sequential,
            verbose=False
        )
        
        try:
            result = crew.kickoff()
            
            # Parse result
            result_text = str(result)
            
            # Extract decision
            is_paraphrase = False
            confidence = "MEDIUM"
            reasoning = result_text
            
            if "DECISION:" in result_text:
                decision_line = [l for l in result_text.split('\n') if 'DECISION:' in l][0]
                is_paraphrase = 'YES' in decision_line.upper()
            else:
                # Fallback: look for keywords
                is_paraphrase = any(word in result_text.upper() for word in ['PARAPHRASE', 'YES', 'EQUIVALENT', 'SAME'])
            
            if "CONFIDENCE:" in result_text:
                conf_line = [l for l in result_text.split('\n') if 'CONFIDENCE:' in l][0]
                if 'HIGH' in conf_line:
                    confidence = "HIGH"
                elif 'LOW' in conf_line:
                    confidence = "LOW"
            
            if "REASONING:" in result_text:
                reasoning_parts = result_text.split("REASONING:")
                if len(reasoning_parts) > 1:
                    reasoning = reasoning_parts[1].strip()
            
            return is_paraphrase, confidence, reasoning
            
        except Exception as e:
            print(f"❌ CrewAI validation error: {e}")
            return None, None, f"Validation failed: {str(e)}"
    
    def validate_prediction(
        self, 
        text_a: str, 
        text_b: str, 
        model_similarity: float
    ) -> Tuple[float, bool, Dict]:
        """
        Main validation pipeline with smart agent triggering
        
        Returns:
            (adjusted_similarity, final_decision, metadata)
        """
        self._total_validations += 1
        
        # Calculate confidence
        confidence = self._calculate_confidence(model_similarity)
        
        # Detect edge cases
        edge_cases = self.check_edge_cases(text_a, text_b, model_similarity)
        
        # Determine if agents should analyze
        should_activate = self.should_trigger_agent(text_a, text_b, model_similarity, edge_cases)
        
        metadata = {
            "confidence_level": confidence.value,
            "edge_cases": edge_cases,
            "used_agent_validation": False,
            "agent_reasoning": None,
            "paraphrase_rescued": False
        }
        
        # If no need for agents, use model prediction
        if not should_activate or not self.enabled:
            model_decision = model_similarity > 0.75
            return model_similarity, model_decision, metadata
        
        # Activate CrewAI agents
        agent_decision, agent_confidence, agent_reasoning = self.validate_with_crew(
            text_a, text_b, model_similarity, edge_cases
        )
        
        if agent_decision is None:
            # Agent validation failed, fallback to model
            model_decision = model_similarity > 0.75
            return model_similarity, model_decision, metadata
        
        metadata["used_agent_validation"] = True
        metadata["agent_reasoning"] = agent_reasoning
        metadata["agent_confidence"] = agent_confidence
        
        # Check if agent rescued a paraphrase the model missed
        model_decision = model_similarity > 0.75
        if agent_decision and not model_decision:
            self._paraphrase_rescues += 1
            metadata["paraphrase_rescued"] = True
        
        # Smart blending: Trust high-confidence agents over uncertain model
        if agent_confidence == "HIGH":
            # High confidence agent - trust it
            adjusted_similarity = 0.90 if agent_decision else 0.30
            final_decision = agent_decision
        elif agent_confidence == "MEDIUM":
            # Medium confidence - blend with model
            if agent_decision:
                adjusted_similarity = max(model_similarity, 0.80)
            else:
                adjusted_similarity = min(model_similarity, 0.65)
            final_decision = agent_decision
        else:
            # Low confidence agent - prefer model if high confidence
            if confidence == ConfidenceLevel.HIGH:
                adjusted_similarity = model_similarity
                final_decision = model_decision
            else:
                # Both uncertain - slight preference to agent
                adjusted_similarity = 0.70 if agent_decision else 0.50
                final_decision = agent_decision
        
        return adjusted_similarity, final_decision, metadata
    
    def get_stats(self) -> Dict:
        """Get validator statistics"""
        if not self.enabled:
            return {"enabled": False}
        
        rescue_rate = (self._paraphrase_rescues / self._agent_activations * 100) if self._agent_activations > 0 else 0
        activation_rate = (self._agent_activations / self._total_validations * 100) if self._total_validations > 0 else 0
        
        return {
            "enabled": True,
            "total_validations": self._total_validations,
            "agent_activations": self._agent_activations,
            "activation_rate": f"{activation_rate:.2f}%",
            "paraphrase_rescues": self._paraphrase_rescues,
            "rescue_rate": f"{rescue_rate:.2f}%",
            "edge_cases_detected": self._edge_case_counts
        }


# Global singleton instance
agentic_validator = AgenticValidator()
