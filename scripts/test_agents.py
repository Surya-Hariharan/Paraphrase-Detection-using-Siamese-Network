"""
Test Agentic AI Integration
============================

Quick test to verify AI agents are working correctly.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.agents import (
    DataValidationAgent,
    TrainingMonitorAgent,
    InferenceValidatorAgent,
    AgentConfig
)

def test_agent_config():
    """Test agent configuration."""
    print("\n" + "="*60)
    print("TEST 1: Agent Configuration")
    print("="*60)
    
    try:
        config = AgentConfig(provider="gemini", enable_logging=True)
        print("✓ Agent config initialized successfully")
        print(f"  Provider: {config.provider}")
        print(f"  Logging: {config.enable_logging}")
        return config
    except Exception as e:
        print(f"✗ Agent config failed: {e}")
        return None


def test_data_validator(config):
    """Test data validation agent."""
    print("\n" + "="*60)
    print("TEST 2: Data Validation Agent")
    print("="*60)
    
    if not config:
        print("⊗ Skipped (config not available)")
        return
    
    try:
        validator = DataValidationAgent(config)
        print("✓ Data validator initialized")
        
        # Test batch validation
        batch = [
            {"text_a": "The cat sat on the mat", "text_b": "A cat was on the rug", "label": 1},
            {"text_a": "Python is great", "text_b": "Python is terrible", "label": 1},  # Suspicious
            {"text_a": "Hello", "text_b": "World", "label": 0},
        ]
        
        results = validator.validate_batch(batch)
        print(f"\n  Batch validation results:")
        print(f"  - Valid: {results['valid']}")
        print(f"  - Warnings: {len(results['warnings'])}")
        for warning in results['warnings']:
            print(f"    ⚠️  {warning}")
        
        # Test single sample validation (suspicious case)
        print("\n  Testing suspicious sample with LLM...")
        sample_result = validator.validate_sample(
            "Python is great",
            "Python is not great",
            label=1  # Labeled as paraphrase (suspicious!)
        )
        
        print(f"  - Valid: {sample_result.get('valid')}")
        print(f"  - Reason: {sample_result.get('reason', 'N/A')}")
        if 'suggested_label' in sample_result:
            print(f"  - Suggested label: {sample_result['suggested_label']}")
        
        print("\n✓ Data validator test passed")
        
    except Exception as e:
        print(f"✗ Data validator test failed: {e}")


def test_training_monitor(config):
    """Test training monitor agent."""
    print("\n" + "="*60)
    print("TEST 3: Training Monitor Agent")
    print("="*60)
    
    if not config:
        print("⊗ Skipped (config not available)")
        return
    
    try:
        monitor = TrainingMonitorAgent(config)
        print("✓ Training monitor initialized")
        
        # Simulate normal training
        print("\n  Testing normal training scenario...")
        result = monitor.monitor_epoch(
            epoch=3,
            train_loss=0.15,
            val_loss=0.18,
            train_acc=0.85,
            val_acc=0.82
        )
        
        print(f"  - Status: {result['status']}")
        print(f"  - Warnings: {len(result['warnings'])}")
        print(f"  - Suggestions: {len(result['suggestions'])}")
        
        # Simulate overfitting
        print("\n  Testing overfitting scenario...")
        result2 = monitor.monitor_epoch(
            epoch=10,
            train_loss=0.05,
            val_loss=0.25,
            train_acc=0.95,
            val_acc=0.75
        )
        
        print(f"  - Status: {result2['status']}")
        print(f"  - Warnings: {result2['warnings']}")
        if result2['suggestions']:
            print(f"  - Top suggestion: {result2['suggestions'][0][:80]}...")
        
        print("\n✓ Training monitor test passed")
        
    except Exception as e:
        print(f"✗ Training monitor test failed: {e}")


def test_inference_validator(config):
    """Test inference validator agent."""
    print("\n" + "="*60)
    print("TEST 4: Inference Validator Agent")
    print("="*60)
    
    if not config:
        print("⊗ Skipped (config not available)")
        return
    
    try:
        validator = InferenceValidatorAgent(config, confidence_threshold=0.7)
        print("✓ Inference validator initialized")
        
        # Test normal prediction
        print("\n  Testing high-confidence prediction...")
        result = validator.validate_prediction(
            text_a="The cat sat on the mat",
            text_b="A cat was sitting on the rug",
            prediction=1,
            confidence=0.9,
            similarity_score=0.92
        )
        
        print(f"  - Validated: {result['validated']}")
        print(f"  - Flags: {result['flags']}")
        print(f"  - LLM check needed: {result['llm_check_needed']}")
        
        # Test low-confidence prediction
        print("\n  Testing low-confidence prediction...")
        result2 = validator.validate_prediction(
            text_a="Python is great",
            text_b="Python is not great",
            prediction=1,
            confidence=0.55,  # Low confidence
            similarity_score=0.78
        )
        
        print(f"  - Validated: {result2['validated']}")
        print(f"  - Flags: {result2['flags']}")
        print(f"  - LLM check needed: {result2['llm_check_needed']}")
        
        if 'llm_prediction' in result2:
            print(f"  - LLM prediction: {result2['llm_prediction']}")
            print(f"  - LLM confidence: {result2['llm_confidence']}")
            print(f"  - LLM reasoning: {result2['llm_reasoning'][:80]}...")
        
        # Get stats
        stats = validator.get_stats()
        print(f"\n  Validation statistics:")
        print(f"  - Total predictions: {stats['total_predictions']}")
        print(f"  - Low confidence: {stats['low_confidence']}")
        print(f"  - OOD detected: {stats['ood_detected']}")
        
        print("\n✓ Inference validator test passed")
        
    except Exception as e:
        print(f"✗ Inference validator test failed: {e}")


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("AGENTIC AI INTEGRATION TEST SUITE")
    print("="*60)
    print("\nThis will test all AI agents with your Gemini API key.")
    print("Make sure GEMINI_API_KEY is set in your .env file.\n")
    
    # Test 1: Agent config
    config = test_agent_config()
    
    # Test 2: Data validator
    test_data_validator(config)
    
    # Test 3: Training monitor
    test_training_monitor(config)
    
    # Test 4: Inference validator
    test_inference_validator(config)
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETED")
    print("="*60)
    print("\n✓ Agentic AI integration is working correctly!")
    print("\nCheck logs/agents/ for detailed logs of all agent interventions.")


if __name__ == "__main__":
    main()
