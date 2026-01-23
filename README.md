# Paraphrase Detection API

Production-grade paraphrase detection system using Siamese Neural Networks with SBERT embeddings.

## Features

- **High Performance**: Optimized inference with caching (10,000 query cache)
- **CrewAI Multi-Agent System**: Intelligent edge case handling with collaborative AI agents ([Learn more](docs/AGENTIC_AI.md))
- **Smart Triggering**: Agents only activate when needed - catches paraphrases the model misses
- **Secure**: JWT authentication, API keys, DDoS protection
- **Fast**: GZip compression, async processing, background logging
- **Scalable**: Connection pooling, thread-safe caching
- **Professional**: Clean architecture, comprehensive logging

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your configuration
# Add GEMINI_API_KEY for CrewAI multi-agent features (optional but recommended)
```

### 3. Initialize Database

```bash
python backend/manage.py init-db
python backend/manage.py create-admin
```

### 4. Train Model (if needed)

```bash
python backend/manage.py train
```

### 5. Start API Server

```bash
python backend/manage.py serve
```

API will be available at `http://localhost:8000`
- **Docs**: http://localhost:8000/docs
- **Health**: http://localhost:8000/health

## Management Commands

```bash
python backend/manage.py serve          # Start API server
python backend/manage.py train          # Train model
python backend/manage.py init-db        # Initialize database
python backend/manage.py create-admin   # Create admin user
```

## API Usage

### Authentication

```bash
# Register
curl -X POST http://localhost:8000/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email":"user@example.com","username":"user","password":"pass123"}'

# Login
curl -X POST http://localhost:8000/auth/login \
  -d "username=user&password=pass123"
```

### Paraphrase Detection

```bash (with agentic AI)
curl -X POST http://localhost:8000/inference/compare \
  -H "Content-Type: application/json" \
  -d '{
    "text1": "The cat sat on the mat",
    "text2": "A feline rested on the rug",
    "use_cache": true,
    "use_agent": true
  }'
```

**Response:**
```json
{
  "similarity": 0.85,
  "is_paraphrase": true,
  "inference_time_ms": 45.2,
  "cached": false,
  "agent_used": true,
  "confidence_level": "HIGH",
  "edge_cases": [],
  "llm_validation": false,
  "llm_reasoning": null,
  "original_similarity": 0.85,
  "adjusted_similarity": 0.85
}
```

**For edge cases with LLM validation** (short texts, length mismatches, etc.):
```json
{
  "similarity": 0.82,
  "is_paraphrase": true,
  "inference_time_ms": 1250.5,
  "cached": false,
  "agent_used": true,
  "confidence_level": "LOW",
  "edge_cases
│   ├── inference_service.py    # Optimized inference
│   └── agentic_validator.py    # Edge case AI validation
├── db/                     # Database models
├── config/                 # Configuration
└── cli/                    # Management scripts
```

## Agentic AI Pipeline
Agentic AI**: LLM validation only for edge cases (~1-5% of queries)
3. **CUDA Optimization**: cudnn.benchmark for faster GPU inference
4. **Async Processing**: Background logging, non-blocking operations
5. **GZip Compression**: Automatic response compression
6. **Connection Pooling**: Efficient database connections
7 **Edge Case Detection**: Length mismatch, short text, numeric-heavy, etc.
- **LLM Validation**: Google Gemini Pro for uncertain predictions
- **Smart Blending**: Combines model + LLM decisions intelligently

**Edge cases detected**:
1. Length mismatch (3x difference)
2. Short texts (< 10 words)
3. Exact matches with low similarity
4. Numeric-heavy content (> 30% numbers)
5. Special character-heavy (> 20% special chars)

**Performance**: LLM activates for only ~1-5% of queries (edge cases), keeping most requests fast.

📖 **Full documentation**: [Agentic AI Guide](docs/AGENTIC_AI.md)adjusted_similarity": 0.82
}
```

📖 **Learn more**: [Agentic AI Documentation](docs/AGENTIC_AI.md)
```

## Architecture

```
backend/
├── api/
│   ├── main.py              # FastAPI application
│   ├── routes/              # API endpoints
│   │   ├── auth.py         # Authentication
│   │   └── inference.py    # Paraphrase detection
│   └── middleware/          # Security, logging
├── core/
│   ├── models/             # Siamese network
│   └── training/           # Training pipeline
├── security/
│   ├── auth/               # JWT, auth dependencies
│   └── ddos/               # DDoS protection
├── services/               # Business logic
├─Agentic AI
GEMINI_API_KEY=your_gemini_api_key_here  # For edge case validation

# ─ db/                     # Database models
├── config/                 # Configuration
└── cli/                    # Management scripts
```

## Performance Optimizations

1. **Inference Caching**: MD5-based cache for identical queries
2. **CUDA Optimization**: cudnn.benchmark for faster GPU inference
3. **Async Processing**: Background logging, non-blocking operations
4. **GZip Compression**: Automatic response compression
5. **Connection Pooling**: Efficient database connections
6. **Thread Safety**: Lock-based cache management

## Configuration

Key environment variables in `.env`:

```env
# Performance
RATE_LIMIT_PER_MINUTE=60
WORKERS=4

# Model
ENCODER_NAME=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIM=384
PROJECTION_DIM=256
MODEL_PATH=checkpoints/best_model.pt
DEVICE=cuda
SIMILARITY_THRESHOLD=0.75

# Training
NUM_EPOCHS=30
BATCH_SIZE=32
LEARNING_RATE=0.0001

# Security
SECRET_KEY=your-secret-key
DDOS_PROTECTION_ENABLED=true
```

## Production Deployment

### Using uvicorn with multiple workers

```bash
python backend/manage.py serve
```

### Docker (recommended)

```bash
docker build -t paraphrase-api .
docker run -p 8000:8000 --env-file .env paraphrase-api
```

## Security Features

- **JWT Tokens**: Secure authentication with 24h expiry
- **API Keys**: Programmatic access tokens
- **Rate Limiting**: 60 requests/minute per IP
- **DDoS Protection**,
    "agent_validations": 47,
    "agent_usage_rate": "3.81%",
    "agentic_ai": {
      "total_validations": 47,
      "llm_calls": 12,
      "llm_usage_rate": "0.97%"
    }: Threat level tracking
- **CORS**: Configurable cross-origin policies
- **Input Validation**: Pydantic models with size limits

## Monitoring

### Health Check

```bash
curl http://localhost:8000/health
```

### Performance Stats

```bash
curl http://localhost:8000/stats
```

**Response:**
```json
{
  "security": { "blocked_ips": 0, "total_requests": 1523 },
  "inference": {
    "total_requests": 1234,
    "cache_hits": 456,
    "cache_hit_rate": "36.95%",
    "cache_size": 456
  }
}
```

## License

MIT License
