# Paraphrase Detection System 🚀

[![Production Ready](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)](PROJECT_STATUS.md)
[![Docker](https://img.shields.io/badge/Docker-Enabled-blue)](Dockerfile)
[![Architecture](https://img.shields.io/badge/Architecture-Siamese%20Network-orange)](docs/ARCHITECTURE_IMPLEMENTATION.md)
[![API](https://img.shields.io/badge/API-FastAPI-green)](backend/api.py)
[![Deploy](https://img.shields.io/badge/Deploy-Railway%20%7C%20Render-purple)](DEPLOYMENT.md)

## Overview
Production-ready document-level paraphrase detection using SBERT embeddings, Siamese neural networks, and multi-agent AI evaluation. Deployable to Railway, Render, Vercel, or any Docker-compatible platform.

**✅ 100% Complete:** Full implementation with REST API, Docker support, and comprehensive documentation. See [PROJECT_STATUS.md](PROJECT_STATUS.md) for details.

## ⚡ Quick Start

### Option 1: Docker (Recommended)
```bash
# 1. Clone and setup
git clone <your-repo>
cd paraphrase-detection
cp .env.template .env
# Add your GROQ_API_KEY to .env

# 2. Run with Docker
docker-compose up --build

# 3. Test API
curl http://localhost:8000/health
```

### Option 2: Deploy to Railway (5 minutes)
```bash
# 1. Push to GitHub
git push origin main

# 2. Go to railway.app
# 3. Click "Deploy from GitHub"
# 4. Add GROQ_API_KEY in environment variables
# 5. Done! Auto-deploys with Dockerfile
```

See [QUICKSTART.md](QUICKSTART.md) for more deployment options.

---

## 🏗️ Architecture

### Document-Level Pipeline
```
Documents → Chunking → SBERT (frozen) → Aggregation → Neural Network → Feature Vectors → Cosine Similarity → Threshold → Result
                                ↓
                      all-MiniLM-L6-v2 (384-dim)
                                ↓
                    Shared-weight Projection Head
                           (256-dim output)
```

### Multi-Agent Evaluation (Optional)
```
Test Cases → ParaphraseGenerator → AdversarialGenerator → EvaluationOrchestrator → Performance Report
```

**Architecture Details**: See [docs/ARCHITECTURE_IMPLEMENTATION.md](docs/ARCHITECTURE_IMPLEMENTATION.md)

---

## 📁 Project Structure

```
paraphrase-detection/
├── backend/
│   ├── api.py                          # FastAPI REST server ✅
│   ├── config.py                       # Configuration management ✅
│   ├── document_siamese_pipeline.py    # Complete ML pipeline ✅
│   ├── neural_engine.py                # SBERT + Neural Network ✅
│   ├── agentic_evaluator.py            # Multi-agent system ✅
│   └── setup_production.py             # Production setup script ✅
├── docs/                                # Comprehensive documentation
├── .env.template                        # Environment template ✅
├── Dockerfile                           # Multi-stage production image ✅
├── docker-compose.yml                   # Service orchestration ✅
├── railway.json                         # Railway deployment ✅
├── render.yaml                          # Render deployment ✅
├── requirements.txt                     # Python dependencies ✅
└── DEPLOYMENT.md                        # Deployment guides ✅
```

---

## 🌐 API Endpoints

```bash
GET  /                    # API information
GET  /health              # Health check + model status
POST /api/compare         # Compare two documents
POST /api/compare/batch   # Batch comparison
POST /api/compare/files   # File upload comparison
GET  /api/model/info      # Model configuration
```

### Example Usage
```bash
curl -X POST http://localhost:8000/api/compare \
  -H "Content-Type: application/json" \
  -d '{
    "text1": "Machine learning is transforming industries.",
    "text2": "ML is revolutionizing various sectors."
  }'
```

**Response:**
```json
{
  "similarity": 0.87,
  "is_paraphrase": true,
  "confidence": "high",
  "processing_time": 0.45
}
```

---

## 🔧 Configuration

### Required Environment Variables
```bash
# Get free API key at: https://console.groq.com/keys
GROQ_API_KEY=your_groq_api_key_here
```

### Optional Configuration
```bash
ENVIRONMENT=production
API_PORT=8000
LOG_LEVEL=info
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
SIMILARITY_THRESHOLD=0.75
```

See [.env.template](.env.template) for all configuration options.

---

## 📚 Documentation

### Getting Started
- [QUICKSTART.md](QUICKSTART.md) - Deploy in 5 minutes
- [DEPLOYMENT.md](DEPLOYMENT.md) - Platform-specific deployment guides
- [PRODUCTION_GUIDE.md](PRODUCTION_GUIDE.md) - Production best practices

### Technical Details
- [PROJECT_STATUS.md](PROJECT_STATUS.md) - Complete implementation status
- [docs/ARCHITECTURE_IMPLEMENTATION.md](docs/ARCHITECTURE_IMPLEMENTATION.md) - Full architecture guide
- [docs/QUICK_REFERENCE.md](docs/QUICK_REFERENCE.md) - Common commands
- [docs/AUDIT_REPORT.md](docs/AUDIT_REPORT.md) - Senior ML engineer audit

### Training & Development
- [docs/TRAINING_ARCHITECTURE.md](docs/TRAINING_ARCHITECTURE.md) - Training details
- [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md) - System explanation

---

## 🚀 Deployment Options

| Platform | Setup Time | Cost | Best For |
|----------|------------|------|----------|
| **Railway** | 5 min | $5/mo | Production (Recommended) |
| **Render** | 10 min | $7/mo | Production |
| **Docker** | 2 min | Free | Local development |
| **Vercel** | 5 min | $20/mo | Lightweight APIs |

**Detailed guides**: See [DEPLOYMENT.md](DEPLOYMENT.md)

---

## 🧪 Testing

### Local Testing
```bash
# Install dependencies
pip install -r requirements.txt

# Run setup script
python backend/setup_production.py

# Start server
python -m uvicorn backend.api:app --reload

# Test
curl http://localhost:8000/health
```

### Docker Testing
```bash
# Build and run
docker-compose up --build

# Test health
curl http://localhost:8000/health

# Test comparison
curl -X POST http://localhost:8000/api/compare \
  -H "Content-Type: application/json" \
  -d '{"text1": "Hello world", "text2": "Hi there"}'
```

---

## 📊 Performance

### Expected Metrics (Medium Hardware)
- **Single Comparison**: ~0.3-0.5 seconds
- **Batch Processing** (10 docs): ~2-3 seconds
- **Model Loading**: ~5-10 seconds (one-time)
- **Memory Usage**: ~500MB-1GB (SBERT loaded)
- **API Response**: <1 second

### Optimization
- ✅ SBERT weights frozen (no gradient computation)
- ✅ Batch processing for efficiency
- ✅ Configurable chunk sizes
- 💡 Add Redis caching for 10x performance boost

---

## 🛠️ Development

### Local Development
```bash
# Hot reload mode
docker-compose --profile dev up

# Or without Docker
python -m uvicorn backend.api:app --reload --host 0.0.0.0 --port 8000
```

### Adding Features
1. Update code in `backend/`
2. Test locally with `docker-compose up`
3. Update documentation
4. Push to GitHub (auto-deploys on Railway)

---

## 🔒 Security

- ✅ Environment variable management (no secrets in code)
- ✅ `.gitignore` excludes sensitive files
- ✅ Non-root Docker user
- ✅ CORS configuration
- ✅ Input validation with Pydantic
- 💡 Add JWT authentication for production use

---

## 🤝 Contributing

This project is fully documented and ready for contributions:

1. Review [docs/ARCHITECTURE_IMPLEMENTATION.md](docs/ARCHITECTURE_IMPLEMENTATION.md)
2. Check [PROJECT_STATUS.md](PROJECT_STATUS.md) for current status
3. Make changes and test locally
4. Submit pull request

---

## 📝 License

See [LICENSE](LICENSE) file for details.

---

## 🆘 Troubleshooting

### Common Issues

**Port already in use:**
```bash
# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

**Model download fails:**
```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

**Docker issues:**
```bash
docker system prune -a
docker-compose up --build
```

**More help**: See [PRODUCTION_GUIDE.md](PRODUCTION_GUIDE.md) troubleshooting section

---

## 📞 Support

- **Documentation**: Comprehensive guides in `docs/` folder
- **Quick Start**: [QUICKSTART.md](QUICKSTART.md)
- **Deployment**: [DEPLOYMENT.md](DEPLOYMENT.md)
- **Status**: [PROJECT_STATUS.md](PROJECT_STATUS.md)

---

## ✨ What's Included

✅ **Complete ML Pipeline**: SBERT + Siamese Network + Document Processing  
✅ **REST API**: FastAPI with 6 endpoints  
✅ **Multi-Agent AI**: CrewAI-based evaluation system  
✅ **Docker Support**: Multi-stage production image  
✅ **Multiple Platforms**: Railway, Render, Vercel, Heroku  
✅ **Configuration**: 50+ environment variables  
✅ **Documentation**: 10+ comprehensive guides  
✅ **Security**: Best practices for production  

**Status**: 🟢 **100% Production Ready**

---

## 🎯 Next Steps

1. **Quick Test**: `docker-compose up --build`
2. **Deploy**: Follow [QUICKSTART.md](QUICKSTART.md)
3. **Monitor**: Check logs and performance
4. **Scale**: Add Redis caching, authentication, monitoring

**Recommended Platform**: Railway (easiest, affordable, reliable)

---

**Built with ❤️ using PyTorch, FastAPI, and Sentence Transformers**
