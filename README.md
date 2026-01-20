# Paraphrase Detection using Siamese Network

A production-ready deep learning system for detecting paraphrases and semantic similarity using Siamese Neural Networks with SBERT embeddings.

## 🌟 Features

- **High-Accuracy Detection**: 85%+ accuracy on paraphrase detection
- **Fast Inference**: < 100ms response time for text comparison
- **Document Support**: PDF, DOCX, and TXT file processing
- **REST API**: FastAPI-powered endpoints for easy integration
- **Comprehensive Evaluation**: 9 test categories including edge cases
- **Fine-Tuning Capability**: Address specific weaknesses with targeted training
- **Production-Ready**: CORS support, error handling, and monitoring

## 🏗️ Architecture

### Siamese Network Design

```
Input Text A          Input Text B
     ↓                     ↓
  SBERT Encoder (all-MiniLM-L6-v2)
     ↓                     ↓
  768-dim embeddings
     ↓                     ↓
  Projection Head (768→256)
     ↓                     ↓
  Cosine Similarity
     ↓
Similarity Score [0-1]
```

**Key Components:**
- **Encoder**: Frozen SBERT model (all-MiniLM-L6-v2)
- **Projection Head**: Trainable 768→256 dimension mapping
- **Loss**: Contrastive loss with margin (0.5)
- **Training**: Mixed precision with gradient clipping

## 📁 Project Structure

```
paraphrase-detection/
├── backend/
│   ├── api/                  # FastAPI endpoints
│   │   ├── inference.py     # Main API server
│   │   └── __init__.py
│   ├── core/                 # Neural network core
│   │   ├── model.py         # Siamese architecture
│   │   ├── trainer.py       # Training pipeline
│   │   └── __init__.py
│   ├── scripts/              # Utility scripts
│   │   ├── train.py         # Main training script
│   │   ├── train_enhanced.py # Fine-tuning script
│   │   ├── evaluate.py      # Model evaluation
│   │   └── generate_enhanced_dataset.py
│   ├── utils/                # Utilities
│   │   ├── document_processor.py  # PDF/DOCX processing
│   │   └── config.py
│   ├── app.py               # Application entry point
│   └── __init__.py
├── checkpoints/              # Model checkpoints
├── data/                     # Training data
├── docs/                     # Documentation
├── frontend/                 # React frontend
│   ├── src/
│   │   ├── components/
│   │   │   ├── Compare.js   # Main comparison UI
│   │   │   └── Navbar.js
│   │   ├── api.js           # API client
│   │   └── App.js
│   └── public/
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/Surya-Hariharan/Paraphrase-Detection-using-Siamese-Network.git
cd Paraphrase-Detection-using-Siamese-Network

# Install Python dependencies
pip install -r requirements.txt

# Install frontend dependencies
cd frontend
npm install
cd ..
```

### 2. Train Model from Scratch

```bash
# Basic training (10 epochs)
python backend/scripts/train.py

# Advanced training with custom parameters
python backend/scripts/train.py \
    --epochs 20 \
    --batch-size 64 \
    --lr 5e-5 \
    --projection-dim 256
```

**Expected Output:**
- `checkpoints/best_model.pt` - Best model by validation accuracy
- `checkpoints/final_model.pt` - Final model after all epochs
- `checkpoints/training_history.json` - Training metrics

### 3. Evaluate Model

```bash
python backend/scripts/evaluate.py
```

**Evaluation Categories:**
- Normal paraphrases & non-paraphrases
- Negation cases
- Long text (>500 chars)
- Short text (<20 chars)
- Idioms & expressions
- Homonyms
- Semantic paraphrases
- Hard negatives

### 4. Start Backend Server

```bash
# Development mode
uvicorn backend.app:app --reload --port 8000

# Production mode
uvicorn backend.app:app --host 0.0.0.0 --port 8000 --workers 4
```

### 5. Start Frontend

```bash
cd frontend
npm start
```

Visit `http://localhost:3000` to use the web interface.

## 🔌 API Endpoints

### POST `/compare`
Compare two text strings

```bash
curl -X POST http://localhost:8000/compare \
  -H "Content-Type: application/json" \
  -d '{
    "text_a": "How can I learn Python?",
    "text_b": "What is the best way to learn Python?"
  }'
```

**Response:**
```json
{
  "similarity": 0.89,
  "is_paraphrase": true,
  "confidence": "high",
  "processing_time_ms": 45.2
}
```

### POST `/compare_files`
Upload and compare documents (PDF, DOCX, TXT)

```bash
curl -X POST http://localhost:8000/compare_files \
  -F "file_a=@document1.pdf" \
  -F "file_b=@document2.pdf"
```

### POST `/compare_batch`
Batch comparison of multiple text pairs

```bash
curl -X POST http://localhost:8000/compare_batch \
  -H "Content-Type: application/json" \
  -d '{
    "pairs": [
      {"text_a": "Hello", "text_b": "Hi"},
      {"text_a": "Goodbye", "text_b": "Farewell"}
    ]
  }'
```

### GET `/health`
Health check endpoint

```bash
curl http://localhost:8000/health
```

## 🎓 Training Guide

### Basic Training Workflow

1. **Prepare Data**
   ```csv
   text_a,text_b,label
   "How to code?","How to program?",1
   "What is AI?","What is ML?",0
   ```

2. **Train Model**
   ```bash
   python backend/scripts/train.py --epochs 10
   ```

3. **Evaluate**
   ```bash
   python backend/scripts/evaluate.py
   ```

4. **Fine-Tune (if needed)**
   ```bash
   python backend/scripts/generate_enhanced_dataset.py
   python backend/scripts/train_enhanced.py
   ```

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 10 | Number of training epochs |
| `--batch-size` | 32 | Training batch size |
| `--lr` | 2e-5 | Learning rate |
| `--projection-dim` | 256 | Projection head dimension |
| `--margin` | 0.5 | Contrastive loss margin |
| `--checkpoint-dir` | checkpoints | Checkpoint directory |
| `--resume` | None | Resume from checkpoint |

See [TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md) for complete training documentation.

## 📊 Performance Metrics

### Baseline Performance (After Initial Training)

| Metric | Score |
|--------|-------|
| Overall Accuracy | 85.2% |
| Precision | 84.7% |
| Recall | 86.1% |
| F1-Score | 85.4% |
| ROC-AUC | 0.91 |

### Category-Specific Performance

| Category | Accuracy | Notes |
|----------|----------|-------|
| Normal Paraphrases | 100% | ✓ Perfect |
| Normal Non-Paraphrases | 100% | ✓ Perfect |
| Negation Cases | 50% → 85%* | *After fine-tuning |
| Long Text (>500 chars) | 0% → 75%* | *After fine-tuning |
| Idioms & Expressions | 40% → 80%* | *After fine-tuning |
| Homonyms | 62.5% → 85%* | *After fine-tuning |
| Semantic Paraphrases | 85.7% | ✓ Strong |
| Hard Negatives | 75% → 85%* | *After fine-tuning |

## 🔧 Fine-Tuning

### When to Fine-Tune

Fine-tune if evaluation reveals:
- Negation accuracy < 80%
- Idiom accuracy < 75%
- Long text accuracy < 70%

### Fine-Tuning Process

```bash
# 1. Generate enhanced dataset (edge cases)
python backend/scripts/generate_enhanced_dataset.py

# 2. Fine-tune with weighted sampling
python backend/scripts/train_enhanced.py

# 3. Re-evaluate
python backend/scripts/evaluate.py
```

**Weighted Sampling:**
- 3x weight for negation cases
- 2.5x weight for idioms
- 2.5x weight for homonyms
- 2x weight for hard negatives

## 📦 Dependencies

### Core Dependencies
```
torch>=2.0.0
transformers>=4.30.0
sentence-transformers>=2.2.0
fastapi>=0.100.0
uvicorn>=0.23.0
pydantic>=2.0.0
```

### Document Processing
```
PyMuPDF>=1.24.0
python-docx>=1.1.0
```

### Training & Evaluation
```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
tqdm>=4.65.0
```

### Frontend
```
react>=18.0.0
axios>=1.7.0
```

See [requirements.txt](requirements.txt) for complete list.

## 📖 Documentation

- [Backend Architecture](docs/BACKEND_ARCHITECTURE.md) - Complete backend structure
- [Training Guide](docs/TRAINING_GUIDE.md) - Detailed training instructions
- [Project Overview](docs/PROJECT_OVERVIEW.md) - High-level project overview
- [Training Architecture](docs/TRAINING_ARCHITECTURE.md) - Model architecture details
- [How It Works](docs/HOW_IT_WORKS.md) - Technical deep dive

## 🛠️ Development

### Running Tests

```bash
# Test imports
python -c "from backend.core.model import TrainableSiameseModel; print('✓ Imports OK')"

# Test API
python -c "from backend.api.inference import app; print('✓ API OK')"
```

### Code Quality

```bash
# Format code
black backend/

# Type checking
mypy backend/

# Linting
flake8 backend/
```

## 🐛 Troubleshooting

### Out of Memory

```bash
# Reduce batch size
python backend/scripts/train.py --batch-size 16

# Disable mixed precision
python backend/scripts/train.py --no-mixed-precision
```

### Import Errors

```bash
# Verify Python version (3.8+)
python --version

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### Model Not Loading

```python
# Debug checkpoint
import torch
checkpoint = torch.load('checkpoints/best_model.pt', map_location='cpu')
print(checkpoint.keys())
```

## 📝 Use Cases

### 1. Plagiarism Detection
```python
from backend.core.model import TrainableSiameseModel

model = TrainableSiameseModel()
model.load_checkpoint('checkpoints/best_model.pt')

result = model.compare(
    "The quick brown fox jumps.",
    "A fast brown fox leaps."
)
print(f"Similarity: {result['similarity']}")
```

### 2. Duplicate Question Detection
```python
# Check if two questions are the same
result = model.compare(
    "How do I learn Python?",
    "What's the best way to learn Python?"
)
```

### 3. Document Comparison
```python
from backend.utils.document_processor import extract_text_from_pdf

text_a = extract_text_from_pdf("doc1.pdf")
text_b = extract_text_from_pdf("doc2.pdf")
result = model.compare(text_a, text_b)
```

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file.

## 👨‍💻 Author

**Surya Hariharan**
- GitHub: [@Surya-Hariharan](https://github.com/Surya-Hariharan)

## 🙏 Acknowledgments

- [Sentence-BERT](https://www.sbert.net/) for pre-trained embeddings
- [Quora Question Pairs](https://www.kaggle.com/c/quora-question-pairs) dataset
- [FastAPI](https://fastapi.tiangolo.com/) for the API framework

## 📊 Project Status

- ✅ Core architecture implemented
- ✅ Training pipeline complete
- ✅ API server functional
- ✅ Document processing added
- ✅ Evaluation framework complete
- ✅ Fine-tuning capability added
- ✅ Frontend UI complete
- 🔄 Continuous improvement

## 🔮 Future Enhancements

- [ ] Multi-language support
- [ ] Real-time streaming comparison
- [ ] Advanced caching layer
- [ ] Model quantization for edge deployment
- [ ] Explainability features (attention visualization)
- [ ] Docker containerization
- [ ] Kubernetes deployment configs

---

**Built with ❤️ using PyTorch and FastAPI**

**Last Updated:** January 2026  
**Version:** 2.0.0  
**Status:** Production Ready
