"""
Agent Crew - Multi-Agent System for Paraphrase Detection

This module implements Phase 2 of the Hybrid Architecture:
A Multi-Agent System using CrewAI for making final decisions.

Agents:
- Agent 1: Semantic Investigator - Extracts and compares technical entities
- Agent 2: Final Decision Maker - Makes DUPLICATE/PARAPHRASE/UNIQUE decisions

LLM: Groq (llama3-8b-8192)
"""

import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# CrewAI imports
from crewai import Agent, Task, Crew, Process, LLM

# LLM Provider imports
try:
    from langchain_groq import ChatGroq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    print("langchain-groq not installed. Install with: pip install langchain-groq")

try:
    from langchain_ollama import ChatOllama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False


class ParaphraseDetectionCrew:
    """Multi-agent crew with Semantic Investigator and Final Decision Maker."""
    
    # LLM Configuration
    GROQ_MODEL = "llama-3.3-70b-versatile"  # Free tier model
    OLLAMA_MODEL = "llama3"
    
    def __init__(self, provider: str = "groq"):
        """
        Initialize the Paraphrase Detection Crew.
        
        Args:
            provider: LLM provider - "groq" or "ollama"
        """
        self.provider = provider.lower()
        
        print("-" * 50)
        print("INITIALIZING MULTI-AGENT CREW")
        print("-" * 50)
        
        # Initialize LLM
        self.llm = self._initialize_llm()
        
        # Create Agents
        self.semantic_investigator = self._create_semantic_investigator()
        self.decision_maker = self._create_decision_maker()
        
        print("\nMulti-Agent Crew ready!")
        print("-" * 50)
    
    def _initialize_llm(self):
        """Initialize the LLM based on provider."""
        
        if self.provider == "groq":
            if not GROQ_AVAILABLE:
                raise ImportError(
                    "langchain-groq is not installed. "
                    "Install with: pip install langchain-groq"
                )
            
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError(
                    "GROQ_API_KEY not found in environment. "
                    "Please set it in your .env file."
                )
            
            print(f"\nInitializing Groq LLM: {self.GROQ_MODEL}")
            # Use CrewAI's LLM class for better compatibility
            return LLM(
                model=f"groq/{self.GROQ_MODEL}",
                api_key=api_key,
                temperature=0.1,
                max_tokens=2048
            )
            
        elif self.provider == "ollama":
            if not OLLAMA_AVAILABLE:
                raise ImportError(
                    "langchain-ollama not installed. "
                    "Install with: pip install langchain-ollama"
                )
            
            print(f"\nInitializing Ollama LLM: {self.OLLAMA_MODEL}")
            # Use CrewAI's LLM class for better compatibility
            return LLM(
                model=f"ollama/{self.OLLAMA_MODEL}",
                base_url="http://localhost:11434",
                temperature=0.1
            )
        
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
    
    def _create_semantic_investigator(self) -> Agent:
        """
        Create Agent 1: The Semantic Investigator.
        
        Role: Extract and compare technical entities from documents.
        """
        print("\nCreating Agent 1: Semantic Investigator")
        
        return Agent(
            role="Semantic Investigator",
            goal=(
                "Meticulously analyze two customer support queries to extract "
                "and compare all technical entities including error codes, "
                "device names, software versions, product names, and technical terms. "
                "Identify both similarities and differences in technical content."
            ),
            backstory=(
                "You are a senior technical analyst with 10+ years of experience "
                "in customer support for enterprise software. You have an encyclopedic "
                "knowledge of error codes, device specifications, and technical terminology. "
                "Your specialty is identifying whether two support tickets are describing "
                "the same issue by analyzing the technical details, not just surface-level "
                "word similarity. You are thorough and never miss important technical details."
            ),
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )
    
    def _create_decision_maker(self) -> Agent:
        """
        Create Agent 2: The Final Decision Maker.
        
        Role: Review all evidence and make the final duplicate/unique decision.
        """
        print("Creating Agent 2: Final Decision Maker")
        
        return Agent(
            role="Final Decision Maker",
            goal=(
                "Make the final, authoritative decision on whether two customer "
                "support queries are DUPLICATES, PARAPHRASES, or UNIQUE based on "
                "both the neural similarity score and the Semantic Investigator's analysis. "
                "Provide clear reasoning and confidence level."
            ),
            backstory=(
                "You are the Chief Quality Officer for a large customer support "
                "organization. You make final decisions on ticket deduplication that "
                "affect customer experience and team efficiency. You understand that "
                "high similarity scores don't always mean exact duplicates - context "
                "matters. You weigh both quantitative (similarity score) and qualitative "
                "(entity analysis) evidence before making decisions. Your decisions "
                "are respected because they are well-reasoned and consistent."
            ),
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )
    
    def _create_investigation_task(
        self,
        doc_a: str,
        doc_b: str
    ) -> Task:
        """Create the task for the Semantic Investigator."""
        
        return Task(
            description=f"""
Analyze the following two customer support queries and extract all technical entities.

## DOCUMENT A (Query 1):
"{doc_a}"

## DOCUMENT B (Query 2):
"{doc_b}"

## YOUR TASK:
1. **Extract Technical Entities from Document A:**
   - Error codes (e.g., 0x80070005, ERR_CONNECTION_REFUSED)
   - Device/Hardware names (e.g., iPhone 14, Dell XPS 15, RTX 3080)
   - Software/OS names and versions (e.g., Windows 11, macOS Ventura, Chrome 118)
   - Product names (e.g., Microsoft Office, Adobe Photoshop)
   - Technical actions/symptoms (e.g., crash, freeze, won't boot)

2. **Extract Technical Entities from Document B:**
   - Same categories as above

3. **Compare and Contrast:**
   - List entities that MATCH between documents
   - List entities that DIFFER between documents
   - Note any entities present in one but missing in the other

## REQUIRED OUTPUT FORMAT:
```
### DOCUMENT A ENTITIES:
- Error Codes: [list]
- Devices/Hardware: [list]
- Software/OS: [list]
- Products: [list]
- Symptoms/Actions: [list]

### DOCUMENT B ENTITIES:
- Error Codes: [list]
- Devices/Hardware: [list]
- Software/OS: [list]
- Products: [list]
- Symptoms/Actions: [list]

### COMPARISON:
- MATCHING ENTITIES: [list with explanation]
- DIFFERING ENTITIES: [list with explanation]
- MISSING IN A: [list]
- MISSING IN B: [list]

### TECHNICAL SIMILARITY ASSESSMENT:
[Your assessment of whether these queries describe the same technical issue]
```
""",
            expected_output=(
                "A structured analysis with extracted entities from both documents "
                "and a detailed comparison showing matches and differences."
            ),
            agent=self.semantic_investigator
        )
    
    def _create_decision_task(
        self,
        doc_a: str,
        doc_b: str,
        similarity_score: float,
        investigation_context: str
    ) -> Task:
        """Create the task for the Final Decision Maker."""
        
        # Determine threshold status
        if similarity_score > 0.85:
            score_interpretation = "HIGH (above 0.85 threshold - suggests duplicate)"
        elif similarity_score > 0.70:
            score_interpretation = "MODERATE (0.70-0.85 - possible paraphrase)"
        else:
            score_interpretation = "LOW (below 0.70 - likely unique)"
        
        return Task(
            description=f"""
Make the final decision on whether these two queries are duplicates.

## EVIDENCE PACKAGE:

### 1. NEURAL NETWORK ANALYSIS (Phase 1 - Quantitative):
- **Cosine Similarity Score:** {similarity_score:.4f}
- **Interpretation:** {score_interpretation}

### 2. SEMANTIC INVESTIGATOR'S REPORT (Qualitative):
{investigation_context}

### 3. ORIGINAL DOCUMENTS:
**Document A:** "{doc_a}"
**Document B:** "{doc_b}"

## YOUR DECISION CRITERIA:

| Condition | Decision |
|-----------|----------|
| Score > 0.85 AND same technical entities | **DUPLICATE** |
| Score > 0.70 AND similar but not identical entities | **PARAPHRASE** |
| Score > 0.85 BUT different technical entities | **UNIQUE** (false positive) |
| Score < 0.70 | **UNIQUE** |

## IMPORTANT CONSIDERATIONS:
- A high similarity score with DIFFERENT error codes = NOT a duplicate
- Same error code with different devices = MAY be related but not duplicate
- Same symptoms without specific identifiers = Need careful judgment

## REQUIRED OUTPUT FORMAT:

**VERDICT:** [DUPLICATE / PARAPHRASE / UNIQUE]

**CONFIDENCE:** [HIGH / MEDIUM / LOW]

**SIMILARITY SCORE:** {similarity_score:.4f}

**KEY EVIDENCE:**
- [Bullet points of most important factors]

**REASONING:**
[Detailed explanation of why you made this decision, referencing both the similarity score and the entity analysis]

**RECOMMENDATION:**
[What action should be taken - merge tickets, link as related, keep separate, etc.]
""",
            expected_output=(
                "A clear verdict (DUPLICATE/PARAPHRASE/UNIQUE) with confidence level, "
                "key evidence, detailed reasoning, and actionable recommendation."
            ),
            agent=self.decision_maker,
            context=[self._create_investigation_task(doc_a, doc_b)]  # Depends on investigation
        )
    
    def analyze(
        self,
        doc_a: str,
        doc_b: str,
        similarity_score: float
    ) -> Dict[str, Any]:
        """
        Run the full multi-agent analysis.
        
        Args:
            doc_a: First document text
            doc_b: Second document text
            similarity_score: Cosine similarity from Phase 1 (neural pipeline)
            
        Returns:
            Dictionary with verdict, analysis, and metadata
        """
        print("\n" + "-" * 50)
        print("STARTING MULTI-AGENT ANALYSIS (PHASE 2)")
        print("-" * 50)
        print(f"\nInput Similarity Score: {similarity_score:.4f}")
        print(f"Provider: {self.provider.upper()} ({self.GROQ_MODEL if self.provider == 'groq' else self.OLLAMA_MODEL})")
        
        # Create tasks
        investigation_task = self._create_investigation_task(doc_a, doc_b)
        
        # We need to get the investigation result first, then pass to decision task
        # Create the crew with sequential process
        decision_task = Task(
            description=f"""
Make the final decision on whether these two queries are duplicates.

## EVIDENCE PACKAGE:

### 1. NEURAL NETWORK ANALYSIS (Phase 1 - Quantitative):
- **Cosine Similarity Score:** {similarity_score:.4f}
- **Interpretation:** {"HIGH - suggests duplicate" if similarity_score > 0.85 else "MODERATE - possible paraphrase" if similarity_score > 0.70 else "LOW - likely unique"}

### 2. SEMANTIC INVESTIGATOR'S REPORT:
Use the analysis from the Semantic Investigator above.

### 3. ORIGINAL DOCUMENTS:
**Document A:** "{doc_a}"
**Document B:** "{doc_b}"

## DECISION CRITERIA:
- Score > 0.85 AND same technical entities → DUPLICATE
- Score > 0.70 AND similar entities → PARAPHRASE  
- Score > 0.85 BUT different entities → UNIQUE (false positive)
- Score < 0.70 → UNIQUE

## REQUIRED OUTPUT:

**VERDICT:** [DUPLICATE / PARAPHRASE / UNIQUE]

**CONFIDENCE:** [HIGH / MEDIUM / LOW]

**SIMILARITY SCORE:** {similarity_score:.4f}

**KEY EVIDENCE:**
[Most important factors]

**REASONING:**
[Detailed explanation referencing both score and entity analysis]

**RECOMMENDATION:**
[Suggested action]
""",
            expected_output=(
                "A clear verdict (DUPLICATE/PARAPHRASE/UNIQUE) with confidence, "
                "evidence, reasoning, and recommendation."
            ),
            agent=self.decision_maker
        )
        
        # Create and run the crew
        crew = Crew(
            agents=[self.semantic_investigator, self.decision_maker],
            tasks=[investigation_task, decision_task],
            process=Process.sequential,
            verbose=True
        )
        
        print("\nRunning agent analysis...")
        result = crew.kickoff()
        
        # Extract verdict from result
        result_text = str(result)
        verdict = self._extract_verdict(result_text)
        confidence = self._extract_confidence(result_text)
        
        return {
            "verdict": verdict,
            "confidence": confidence,
            "similarity_score": similarity_score,
            "full_analysis": result_text,
            "provider": self.provider,
            "model": self.GROQ_MODEL if self.provider == "groq" else self.OLLAMA_MODEL,
            "agents_used": ["Semantic Investigator", "Final Decision Maker"]
        }
    
    def _extract_verdict(self, text: str) -> str:
        """Extract verdict from analysis text."""
        text_upper = text.upper()
        
        if "**VERDICT:** DUPLICATE" in text_upper or "VERDICT: DUPLICATE" in text_upper:
            return "DUPLICATE"
        elif "**VERDICT:** PARAPHRASE" in text_upper or "VERDICT: PARAPHRASE" in text_upper:
            return "PARAPHRASE"
        elif "**VERDICT:** UNIQUE" in text_upper or "VERDICT: UNIQUE" in text_upper:
            return "UNIQUE"
        
        # Fallback detection
        if "DUPLICATE" in text_upper:
            return "DUPLICATE"
        elif "PARAPHRASE" in text_upper:
            return "PARAPHRASE"
        return "UNIQUE"
    
    def _extract_confidence(self, text: str) -> str:
        """Extract confidence level from analysis text."""
        text_upper = text.upper()
        
        if "CONFIDENCE:** HIGH" in text_upper or "CONFIDENCE: HIGH" in text_upper:
            return "HIGH"
        elif "CONFIDENCE:** MEDIUM" in text_upper or "CONFIDENCE: MEDIUM" in text_upper:
            return "MEDIUM"
        elif "CONFIDENCE:** LOW" in text_upper or "CONFIDENCE: LOW" in text_upper:
            return "LOW"
        
        return "MEDIUM"  # Default


def create_crew(provider: str = "groq") -> ParaphraseDetectionCrew:
    """
    Factory function to create a ParaphraseDetectionCrew.
    
    Args:
        provider: "groq" for Groq cloud API, "ollama" for local Ollama
        
    Returns:
        Configured ParaphraseDetectionCrew instance
    """
    return ParaphraseDetectionCrew(provider=provider)


# ============================================================
# Testing
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("   PHASE 2: MULTI-AGENT SYSTEM TEST")
    print("=" * 70 + "\n")
    
    # Test documents
    doc_a = "My laptop won't turn on after the Windows update. Error code 0x80070005."
    doc_b = "Computer not starting following system update. Getting error 0x80070005."
    
    # Mock similarity score from Phase 1
    mock_similarity = 0.87
    
    print(f"Document A: {doc_a}")
    print(f"Document B: {doc_b}")
    print(f"Similarity Score (from Phase 1): {mock_similarity}")
    
    try:
        # Create crew and analyze
        crew = create_crew("groq")
        result = crew.analyze(doc_a, doc_b, mock_similarity)
        
        print("\n" + "-" * 50)
        print("FINAL RESULT")
        print("-" * 50)
        print(f"\nVerdict: {result['verdict']}")
        print(f"Confidence: {result['confidence']}")
        print(f"Agents: {', '.join(result['agents_used'])}")
        print(f"\nFull Analysis:\n{result['full_analysis']}")
        
    except Exception as e:
        print(f"\nError: {e}")
        print("Make sure GROQ_API_KEY is set in your .env file")
        print("Or try: crew = create_crew('ollama')")
