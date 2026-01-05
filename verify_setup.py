"""
Quick Verification Script
=========================

Run this script to verify that the paraphrase detection system is working correctly.
This includes:
1. Model loading
2. Gradient flow verification
3. Training capability check
4. Inference test

Usage:
    python verify_setup.py
"""

import sys
import torch
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / 'backend'))

from neural_engine import TrainableSiameseModel, SiameseProjectionModel, ContrastiveLoss

def check_environment():
    """Check Python and PyTorch environment."""
    print("="*70)
    print("ENVIRONMENT CHECK")
    print("="*70)
    
    print(f"✓ Python version: {sys.version.split()[0]}")
    print(f"✓ PyTorch version: {torch.__version__}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        device = 'cuda'
    else:
        print("⚠ GPU not available, using CPU")
        device = 'cpu'
    
    return device

def check_model_loading():
    """Check if model loads correctly."""
    print("\n" + "="*70)
    print("MODEL LOADING CHECK")
    print("="*70)
    
    try:
        model = TrainableSiameseModel()
        print("✓ TrainableSiameseModel loaded successfully")
        print(f"  - SBERT Model: {model.SBERT_MODEL}")
        print(f"  - Projection Dim: {model.projection_dim}")
        
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  - Trainable parameters: {trainable:,}")
        
        return model
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        return None

def check_gradient_flow(model, device):
    """Verify gradient flow during training."""
    print("\n" + "="*70)
    print("GRADIENT FLOW CHECK")
    print("="*70)
    
    try:
        model = model.to(device)
        model.train()
        
        # Setup
        optimizer = torch.optim.Adam(model.get_trainable_parameters(), lr=0.001)
        criterion = ContrastiveLoss(margin=1.0)
        
        # Test data
        doc_a = "The cat sat on the mat"
        doc_b = "A cat is sitting on a mat"
        label = torch.tensor([1.0], device=device)
        
        # Forward pass
        optimizer.zero_grad()
        vec_a, vec_b = model.forward(doc_a, doc_b)
        
        # Compute loss
        loss = criterion(vec_a.unsqueeze(0), vec_b.unsqueeze(0), label)
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        grad_norms = {}
        for name, param in model.projection_head.named_parameters():
            if param.grad is not None:
                grad_norms[name] = param.grad.norm().item()
        
        if all(g > 0 for g in grad_norms.values()):
            print("✓ Gradient flow verified")
            for name, norm in grad_norms.items():
                print(f"  - {name}: {norm:.6f}")
            return True
        else:
            print("✗ Zero gradients detected!")
            return False
            
    except Exception as e:
        print(f"✗ Gradient check failed: {e}")
        return False

def check_inference(model, device):
    """Check inference capability."""
    print("\n" + "="*70)
    print("INFERENCE CHECK")
    print("="*70)
    
    try:
        model = model.to(device)
        model.eval()
        
        # Test pairs
        pairs = [
            ("A man is eating food", "A person is having a meal", "Paraphrase"),
            ("A man is eating food", "The car engine exploded", "Non-paraphrase"),
        ]
        
        with torch.no_grad():
            for doc_a, doc_b, label in pairs:
                vec_a, vec_b = model.forward(doc_a, doc_b)
                sim = torch.nn.functional.cosine_similarity(vec_a, vec_b, dim=0).item()
                print(f"  {label}: {sim:.4f}")
        
        print("✓ Inference working correctly")
        return True
        
    except Exception as e:
        print(f"✗ Inference check failed: {e}")
        return False

def main():
    """Run all verification checks."""
    print("\n" + "="*70)
    print("PARAPHRASE DETECTION SYSTEM VERIFICATION")
    print("="*70)
    
    # Run checks
    device = check_environment()
    model = check_model_loading()
    
    if model is None:
        print("\n" + "="*70)
        print("✗ VERIFICATION FAILED - Model loading error")
        print("="*70)
        return False
    
    gradient_ok = check_gradient_flow(model, device)
    inference_ok = check_inference(model, device)
    
    # Summary
    print("\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)
    
    checks = {
        "Environment": True,
        "Model Loading": model is not None,
        "Gradient Flow": gradient_ok,
        "Inference": inference_ok,
    }
    
    for check, status in checks.items():
        symbol = "✓" if status else "✗"
        print(f"{symbol} {check}")
    
    all_passed = all(checks.values())
    
    print("\n" + "="*70)
    if all_passed:
        print("✓ ALL CHECKS PASSED - System ready to use!")
        print("="*70)
        print("\nNext steps:")
        print("  1. Place documents in datasets/ folder")
        print("  2. Run: python backend/quick_compare.py")
        print("  3. Or train: python backend/train_on_documents.py")
    else:
        print("✗ SOME CHECKS FAILED - Please review errors above")
        print("="*70)
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
