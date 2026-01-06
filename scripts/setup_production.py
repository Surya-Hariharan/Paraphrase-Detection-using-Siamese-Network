"""
Production Setup Script

Run this script to:
1. Download required models
2. Set up directories
3. Verify environment
4. Run health checks
"""

import os
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_directories():
    """Create necessary directories"""
    directories = [
        "models",
        "checkpoints",
        "logs",
        "uploads",
        "temp",
        "data"
    ]
    
    for directory in directories:
        path = Path(directory)
        path.mkdir(exist_ok=True)
        logger.info(f"✓ Created directory: {directory}")


def download_sbert_model():
    """Download and cache SBERT model"""
    try:
        from sentence_transformers import SentenceTransformer
        
        model_name = os.getenv("SBERT_MODEL_NAME", "all-MiniLM-L6-v2")
        logger.info(f"Downloading SBERT model: {model_name}")
        
        model = SentenceTransformer(model_name)
        logger.info(f"✓ SBERT model downloaded and cached")
        
        return True
    except Exception as e:
        logger.error(f"✗ Failed to download SBERT model: {e}")
        return False


def verify_environment():
    """Verify environment variables"""
    required_vars = []
    optional_vars = [
        "GROQ_API_KEY",
        "SBERT_MODEL_NAME",
        "SIMILARITY_THRESHOLD",
        "API_PORT"
    ]
    
    logger.info("Checking environment variables...")
    
    for var in required_vars:
        if not os.getenv(var):
            logger.error(f"✗ Missing required variable: {var}")
            return False
        logger.info(f"✓ {var} is set")
    
    for var in optional_vars:
        if os.getenv(var):
            logger.info(f"✓ {var} is set")
        else:
            logger.warning(f"⚠ {var} not set (using default)")
    
    return True


def check_dependencies():
    """Check if all dependencies are installed"""
    logger.info("Checking dependencies...")
    
    dependencies = [
        "torch",
        "sentence_transformers",
        "fastapi",
        "uvicorn",
        "pydantic",
        "python-dotenv"
    ]
    
    missing = []
    for dep in dependencies:
        try:
            __import__(dep.replace("-", "_"))
            logger.info(f"✓ {dep}")
        except ImportError:
            logger.error(f"✗ {dep}")
            missing.append(dep)
    
    if missing:
        logger.error(f"Missing dependencies: {', '.join(missing)}")
        logger.error("Run: pip install -r requirements.txt")
        return False
    
    return True


def main():
    """Main setup function"""
    logger.info("=" * 70)
    logger.info("PRODUCTION SETUP - PARAPHRASE DETECTION SYSTEM")
    logger.info("=" * 70)
    
    # Step 1: Create directories
    logger.info("\n[1/4] Creating directories...")
    create_directories()
    
    # Step 2: Verify environment
    logger.info("\n[2/4] Verifying environment...")
    if not verify_environment():
        logger.error("Environment verification failed")
        sys.exit(1)
    
    # Step 3: Check dependencies
    logger.info("\n[3/4] Checking dependencies...")
    if not check_dependencies():
        logger.error("Dependency check failed")
        sys.exit(1)
    
    # Step 4: Download models
    logger.info("\n[4/4] Downloading models...")
    if not download_sbert_model():
        logger.error("Model download failed")
        sys.exit(1)
    
    logger.info("\n" + "=" * 70)
    logger.info("✓ PRODUCTION SETUP COMPLETE")
    logger.info("=" * 70)
    logger.info("\nYou can now start the API server:")
    logger.info("  python -m uvicorn backend.api:app --host 0.0.0.0 --port 8000")
    logger.info("\nOr using Docker:")
    logger.info("  docker-compose up")


if __name__ == "__main__":
    main()
