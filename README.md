# Paraphrase Detection System

[![Training](https://img.shields.io/badge/Training-Verified%20Working-brightgreen)](TRAINING_ARCHITECTURE.md)
[![GPU](https://img.shields.io/badge/GPU-CUDA%20Enabled-blue)](TRAINING_ARCHITECTURE.md)
[![Architecture](https://img.shields.io/badge/Architecture-Siamese%20Network-orange)](TRAINING_ARCHITECTURE.md)

## Overview
Detect paraphrases between two documents using SBERT embeddings, trainable neural networks, and AI agents.

**✅ Training Fixed:** Gradient flow verified, weights update correctly. See [TRAINING_ARCHITECTURE.md](TRAINING_ARCHITECTURE.md) for details.

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Place Your Documents
```bash
datasets/
├── document1.txt    # Your first document
├── document2.pdf    # Your second document
```

### 3. Compare
```bash
# Set API key in .env
echo "GROQ_API_KEY=your_key" > .env

# Compare with AI agents
python backend/quick_compare.py --use-agents
```

## Core Architecture

```
Documents → SBERT → Embeddings → Projection Head → Similarity Score → AI Agents → Verdict
```

## Key Files

- `backend/quick_compare.py` - Main comparison script
- `backend/app.py` - API server
- `backend/neural_engine.py` - Neural network
- `backend/agent_crew.py` - AI agents
- `backend/document_loader.py` - File loading (txt/pdf)

## Usage

### Command Line
```bash
python backend/quick_compare.py --use-agents
```

### Web Interface
```bash
.\start-backend.bat
.\start-frontend.bat
# Visit http://localhost:3000
```

### API
```bash
curl -X POST http://localhost:8000/api/compare \
  -H "Content-Type: application/json" \
  -d '{"doc_a": "text 1", "doc_b": "text 2", "use_agents": true}'
```

## Optional: Training

**Status:** ✅ Verified working with gradient flow validation

The system works out-of-the-box with pre-trained SBERT, but you can fine-tune the projection head:

```bash
# Train on a dataset
python backend/train.py --data-path datasets/train.csv --epochs 10

# Or train on two specific documents
python backend/train_on_documents.py --doc-a file1.txt --doc-b file2.txt --epochs 50
```

**What gets trained:**
- ✅ Projection head (98,560 parameters) - **trainable**
- ❌ SBERT encoder (22M parameters) - **frozen**

**Gradient verification:** Automatic in epoch 1 to ensure training is real.

📖 **Detailed training documentation:** See [TRAINING_ARCHITECTURE.md](TRAINING_ARCHITECTURE.md)

## Requirements

- Python 3.8+
- GROQ API key (for AI agents)
- GPU recommended for training (CUDA support)
- See `requirements.txt` for packages

## Documentation

- [TRAINING_ARCHITECTURE.md](TRAINING_ARCHITECTURE.md) - Training methodology, gradient flow fix, architecture details
- [HOW_IT_WORKS.md](HOW_IT_WORKS.md) - System workflow and components
- [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) - Project structure
- [SETUP_GUIDE.md](SETUP_GUIDE.md) - Installation and setup
