"""
Agentic AI for Paraphrase Detection - EVALUATION ONLY

CRITICAL CONSTRAINTS:
- Agentic AI operates OUTSIDE the core SBERT → NN → Cosine Similarity pipeline
- Agents do NOT modify or train the neural network weights
- Agents do NOT interfere with SBERT or projection head parameters
- This is for EVALUATION and TEST GENERATION only

WORKFLOW:
1. Take a trained (or untrained) SBERT + Siamese model
2. Use agents to generate document-level test cases:
   - Agent 1: Generate paraphrase pairs (label=1)
   - Agent 2: Generate adversarial/hard negative pairs (label=0)
3. Run inference ONLY (no training) through the fixed pipeline
4. Compute similarity scores and threshold-based decisions
5. Report robustness, failure patterns, and edge cases

This is an NLP evaluation tool, not an autonomous AI training system.
No reinforcement learning, no online training, no weight updates.
"""

import os
from typing import Dict, Any, List, Tuple, Optional
from dotenv import load_dotenv

load_dotenv()

try:
    from crewai import Agent, Task, Crew, Process, LLM
    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False
    print("Warning: CrewAI not available. Install with: pip install crewai")


class AgenticEvaluator:
    """
    Multi-agent system for generating paraphrase detection test cases.
    
    IMPORTANT: This does NOT train the model.
    It only generates test cases and evaluates the fixed model.
    """
    
    GEMINI_MODEL = "gemini/gemini-2.0-flash-exp"  # Fast and capable model
    GROQ_MODEL = "llama-3.3-70b-versatile"  # Fallback if user wants Groq
    
    def __init__(self, provider: str = "gemini"):
        """
        Initialize agentic evaluator.
        
        Args:
            provider: LLM provider ("gemini" or "groq" - default: gemini)
        """
        if not CREWAI_AVAILABLE:
            raise ImportError("CrewAI required for agentic evaluation")
        
        self.provider = provider.lower()
        self.llm = self._initialize_llm()
        
        # Create agents
        self.paraphrase_generator = self._create_paraphrase_generator()
        self.adversarial_generator = self._create_adversarial_generator()
        self.evaluation_orchestrator = self._create_evaluation_orchestrator()
        
        print(f"\n✓ Agentic Evaluator initialized with {provider.upper()} (EVALUATION MODE ONLY)")
        print("  - No training")
        print("  - No weight updates")
        print("  - Test generation and inference only\n")
    
    def _initialize_llm(self):
        """Initialize LLM for agents."""
        if self.provider == "gemini":
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError(
                    "GEMINI_API_KEY not found in environment.\\n"
                    "Get your free API key at: https://aistudio.google.com/app/apikey"
                )
            
            return LLM(
                model=self.GEMINI_MODEL,
                api_key=api_key,
                temperature=0.7,  # Higher temperature for generation
                max_tokens=2048
            )
        
        elif self.provider == "groq":
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY not found in environment")
            
            return LLM(
                model=f"groq/{self.GROQ_MODEL}",
                api_key=api_key,
                temperature=0.7,
                max_tokens=2048
            )
        
        elif self.provider == "ollama":
            return LLM(
                model="ollama/llama3",
                base_url="http://localhost:11434",
                temperature=0.7
            )
        
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
    
    def _create_paraphrase_generator(self) -> Agent:
        """
        AGENT 1: Document Paraphrase Generator
        
        Generates document-level paraphrases that preserve meaning
        but change wording, structure, and paragraph order.
        """
        return Agent(
            role="Document Paraphrase Generator",
            goal=(
                "Generate document-level paraphrases that preserve factual content "
                "and intent while modifying sentence structure, wording, and "
                "paragraph organization. Create realistic paraphrases suitable for "
                "testing semantic similarity models."
            ),
            backstory=(
                "You are an expert NLP researcher specializing in paraphrase generation. "
                "You understand semantic equivalence at the document level. You can "
                "rewrite documents while preserving meaning, changing:\n"
                "- Vocabulary (synonyms, different terms for same concepts)\n"
                "- Sentence structure (active/passive, clause ordering)\n"
                "- Paragraph organization (different logical flow)\n"
                "- Discourse markers (however → but, therefore → thus)\n"
                "You never change the core meaning or introduce new information."
            ),
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )
    
    def _create_adversarial_generator(self) -> Agent:
        """
        AGENT 2: Adversarial / Hard Negative Generator
        
        Generates documents that appear similar but are NOT paraphrases.
        These are challenging test cases.
        """
        return Agent(
            role="Adversarial Example Generator",
            goal=(
                "Generate hard negative examples: documents that have high lexical "
                "overlap with the original but are NOT semantic paraphrases. Create "
                "challenging test cases that might fool naive similarity measures."
            ),
            backstory=(
                "You are a red team NLP researcher focused on finding failure modes "
                "in semantic similarity systems. You create adversarial examples by:\n"
                "- Keeping same topic/domain but changing intent\n"
                "- Using shared terminology but different conclusions\n"
                "- Partial content overlap with critical differences\n"
                "- Topic drift while maintaining lexical similarity\n"
                "- Negation or contradictory statements\n"
                "Your goal is to stress-test the model, not to train it."
            ),
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )
    
    def _create_evaluation_orchestrator(self) -> Agent:
        """
        AGENT 3: Evaluation Orchestrator
        
        Analyzes model performance on generated test cases.
        """
        return Agent(
            role="Evaluation Orchestrator",
            goal=(
                "Analyze paraphrase detection model performance on generated test cases. "
                "Identify failure patterns, borderline cases, and robustness issues. "
                "Provide actionable insights for model improvement."
            ),
            backstory=(
                "You are a senior ML engineer evaluating NLP systems. You analyze "
                "model predictions, compare them to expected outcomes, and identify "
                "patterns in failures. You understand:\n"
                "- False positives (non-paraphrases detected as paraphrases)\n"
                "- False negatives (paraphrases detected as non-paraphrases)\n"
                "- Borderline cases near decision threshold\n"
                "- Systematic biases or failure modes\n"
                "You provide clear, actionable recommendations."
            ),
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )
    
    def generate_paraphrase_pairs(
        self,
        seed_document: str,
        num_pairs: int = 3
    ) -> List[Tuple[str, str, int]]:
        """
        Generate paraphrase test pairs.
        
        Args:
            seed_document: Original document
            num_pairs: Number of paraphrases to generate
            
        Returns:
            List of (original, paraphrase, label=1) tuples
        """
        task = Task(
            description=f"""
Generate {num_pairs} different paraphrases of the following document.

ORIGINAL DOCUMENT:
{seed_document}

REQUIREMENTS:
1. Preserve all factual information and intent
2. Change sentence structure and wording significantly
3. Reorder paragraphs/sentences when appropriate
4. Use synonyms and different phrasings
5. Maintain document coherence

OUTPUT FORMAT (for each paraphrase):
---PARAPHRASE {"{i}"}---
[Your paraphrase here]
---END---
""",
            expected_output=f"{num_pairs} distinct paraphrases separated by markers",
            agent=self.paraphrase_generator
        )
        
        crew = Crew(
            agents=[self.paraphrase_generator],
            tasks=[task],
            process=Process.sequential,
            verbose=False
        )
        
        result = crew.kickoff()
        
        # Parse generated paraphrases
        pairs = []
        result_text = str(result)
        
        # Simple parsing (improve as needed)
        import re
        paraphrases = re.findall(r'---PARAPHRASE \d+---(.*?)---END---', result_text, re.DOTALL)
        
        for paraphrase in paraphrases:
            pairs.append((seed_document, paraphrase.strip(), 1))
        
        return pairs
    
    def generate_adversarial_pairs(
        self,
        seed_document: str,
        num_pairs: int = 3
    ) -> List[Tuple[str, str, int]]:
        """
        Generate adversarial (hard negative) test pairs.
        
        Args:
            seed_document: Original document
            num_pairs: Number of adversarial examples to generate
            
        Returns:
            List of (original, adversarial, label=0) tuples
        """
        task = Task(
            description=f"""
Generate {num_pairs} adversarial examples based on this document.

ORIGINAL DOCUMENT:
{seed_document}

REQUIREMENTS:
1. High lexical overlap (shared words/phrases) with original
2. Same topic/domain but DIFFERENT meaning or intent
3. Could fool surface-level similarity measures
4. Examples:
   - Contradictory statements
   - Same terms but different conclusions
   - Topic drift with shared vocabulary
   - Partial information mismatch

OUTPUT FORMAT (for each adversarial example):
---ADVERSARIAL {"{i}"}---
[Your adversarial document here]
---END---
""",
            expected_output=f"{num_pairs} adversarial examples separated by markers",
            agent=self.adversarial_generator
        )
        
        crew = Crew(
            agents=[self.adversarial_generator],
            tasks=[task],
            process=Process.sequential,
            verbose=False
        )
        
        result = crew.kickoff()
        
        # Parse generated adversarial examples
        pairs = []
        result_text = str(result)
        
        import re
        adversarials = re.findall(r'---ADVERSARIAL \d+---(.*?)---END---', result_text, re.DOTALL)
        
        for adversarial in adversarials:
            pairs.append((seed_document, adversarial.strip(), 0))
        
        return pairs
    
    def evaluate_model_on_generated_cases(
        self,
        model_pipeline,
        test_cases: List[Tuple[str, str, int]]
    ) -> Dict[str, Any]:
        """
        Evaluate model on generated test cases (INFERENCE ONLY).
        
        Args:
            model_pipeline: DocumentLevelSiameseModel instance
            test_cases: List of (doc_a, doc_b, label) tuples
            
        Returns:
            Evaluation report
        """
        print("\n" + "=" * 70)
        print("AGENTIC EVALUATION - INFERENCE ON GENERATED TEST CASES")
        print("=" * 70)
        print(f"Total test cases: {len(test_cases)}")
        print("Mode: INFERENCE ONLY (no training)\n")
        
        results = []
        
        for i, (doc_a, doc_b, true_label) in enumerate(test_cases):
            print(f"\nTest case {i+1}/{len(test_cases)}")
            
            # INFERENCE ONLY - no gradients, no weight updates
            result = model_pipeline.compare_documents(doc_a, doc_b, verbose=False)
            
            predicted_label = 1 if result["is_paraphrase"] else 0
            correct = (predicted_label == true_label)
            
            result["true_label"] = true_label
            result["predicted_label"] = predicted_label
            result["correct"] = correct
            result["doc_a_preview"] = doc_a[:200] + "..."
            result["doc_b_preview"] = doc_b[:200] + "..."
            
            results.append(result)
            
            status = "✓ CORRECT" if correct else "✗ WRONG"
            print(f"  True label: {true_label}, Predicted: {predicted_label} - {status}")
            print(f"  Similarity: {result['cosine_similarity']:.4f}")
        
        # Compute metrics
        correct = sum(r["correct"] for r in results)
        total = len(results)
        accuracy = correct / total if total > 0 else 0
        
        tp = sum(1 for r in results if r["predicted_label"] == 1 and r["true_label"] == 1)
        fp = sum(1 for r in results if r["predicted_label"] == 1 and r["true_label"] == 0)
        tn = sum(1 for r in results if r["predicted_label"] == 0 and r["true_label"] == 0)
        fn = sum(1 for r in results if r["predicted_label"] == 0 and r["true_label"] == 1)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Identify failure cases
        false_positives = [r for r in results if r["predicted_label"] == 1 and r["true_label"] == 0]
        false_negatives = [r for r in results if r["predicted_label"] == 0 and r["true_label"] == 1]
        
        evaluation_report = {
            "total_cases": total,
            "correct": correct,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": {
                "true_positives": tp,
                "false_positives": fp,
                "true_negatives": tn,
                "false_negatives": fn
            },
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "all_results": results
        }
        
        print("\n" + "=" * 70)
        print("EVALUATION RESULTS")
        print("=" * 70)
        print(f"Accuracy: {accuracy:.2%}")
        print(f"Precision: {precision:.2%}")
        print(f"Recall: {recall:.2%}")
        print(f"F1 Score: {f1:.2%}")
        print(f"\nConfusion Matrix:")
        print(f"  TP: {tp}, FP: {fp}")
        print(f"  FN: {fn}, TN: {tn}")
        print("=" * 70 + "\n")
        
        return evaluation_report


# ============================================================
# Complete Agentic Evaluation Workflow
# ============================================================

def run_agentic_evaluation(
    model_pipeline,
    seed_documents: List[str],
    num_paraphrases_per_doc: int = 2,
    num_adversarial_per_doc: int = 2,
    provider: str = "groq"
) -> Dict[str, Any]:
    """
    Complete agentic evaluation workflow.
    
    Steps:
    1. Initialize agents
    2. Generate paraphrase pairs (positive examples)
    3. Generate adversarial pairs (negative examples)
    4. Run inference on all test cases (NO TRAINING)
    5. Analyze results and identify failure patterns
    
    Args:
        model_pipeline: Trained DocumentLevelSiameseModel
        seed_documents: List of seed documents to generate test cases from
        num_paraphrases_per_doc: Paraphrases to generate per seed
        num_adversarial_per_doc: Adversarial examples to generate per seed
        provider: LLM provider
        
    Returns:
        Complete evaluation report
    """
    print("\n" + "=" * 70)
    print("AGENTIC EVALUATION WORKFLOW")
    print("=" * 70)
    print("This is EVALUATION ONLY - no model training")
    print("Agents generate test cases, model runs inference\n")
    
    # Initialize evaluator
    evaluator = AgenticEvaluator(provider=provider)
    
    # Generate test cases
    all_test_cases = []
    
    for i, seed_doc in enumerate(seed_documents):
        print(f"\nGenerating test cases for seed document {i+1}/{len(seed_documents)}")
        
        # Generate paraphrases (label=1)
        print(f"  Generating {num_paraphrases_per_doc} paraphrases...")
        paraphrase_pairs = evaluator.generate_paraphrase_pairs(
            seed_doc,
            num_pairs=num_paraphrases_per_doc
        )
        all_test_cases.extend(paraphrase_pairs)
        
        # Generate adversarial examples (label=0)
        print(f"  Generating {num_adversarial_per_doc} adversarial examples...")
        adversarial_pairs = evaluator.generate_adversarial_pairs(
            seed_doc,
            num_pairs=num_adversarial_per_doc
        )
        all_test_cases.extend(adversarial_pairs)
    
    print(f"\n✓ Generated {len(all_test_cases)} total test cases")
    
    # Evaluate model (INFERENCE ONLY)
    evaluation_report = evaluator.evaluate_model_on_generated_cases(
        model_pipeline,
        all_test_cases
    )
    
    return evaluation_report


# ============================================================
# Testing
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("AGENTIC EVALUATOR TEST")
    print("=" * 70)
    print("\nThis demonstrates agent-based test case generation")
    print("for evaluating paraphrase detection models.\n")
    
    # Sample seed document
    seed_doc = """
    Machine learning is transforming healthcare by enabling early disease detection
    and personalized treatment plans. AI algorithms analyze medical images with high
    accuracy, often matching or exceeding human radiologists. Predictive models help
    identify patients at risk of serious conditions before symptoms appear.
    """
    
    try:
        evaluator = AgenticEvaluator(provider="groq")
        
        print("\nGenerating paraphrase examples...")
        paraphrase_pairs = evaluator.generate_paraphrase_pairs(seed_doc, num_pairs=2)
        
        print(f"\n✓ Generated {len(paraphrase_pairs)} paraphrase pairs")
        
        print("\nGenerating adversarial examples...")
        adversarial_pairs = evaluator.generate_adversarial_pairs(seed_doc, num_pairs=2)
        
        print(f"\n✓ Generated {len(adversarial_pairs)} adversarial pairs")
        
        print("\nTest cases ready for evaluation!")
        
    except Exception as e:
        print(f"\nError: {e}")
        print("Make sure GROQ_API_KEY is set in .env file")
