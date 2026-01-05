# =============================================================================
# Multi-stage Dockerfile for Paraphrase Detection System
# Optimized for production deployment
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Base Image with Python and System Dependencies
# -----------------------------------------------------------------------------
FROM python:3.10-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create app user for security (don't run as root)
RUN useradd -m -u 1000 appuser

# Set working directory
WORKDIR /app

# -----------------------------------------------------------------------------
# Stage 2: Dependencies Installation
# -----------------------------------------------------------------------------
FROM base as dependencies

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download SBERT model to cache it in the image
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# -----------------------------------------------------------------------------
# Stage 3: Production Image
# -----------------------------------------------------------------------------
FROM base as production

# Copy Python dependencies from dependencies stage
COPY --from=dependencies /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=dependencies /usr/local/bin /usr/local/bin

# Copy SBERT model cache
COPY --from=dependencies /root/.cache /home/appuser/.cache

# Create necessary directories
RUN mkdir -p /app/backend \
    /app/data \
    /app/models \
    /app/checkpoints \
    /app/logs \
    /app/uploads \
    /app/temp

# Copy application code
COPY --chown=appuser:appuser backend/ /app/backend/
COPY --chown=appuser:appuser data/.gitkeep /app/data/
COPY --chown=appuser:appuser models/.gitkeep /app/models/
COPY --chown=appuser:appuser checkpoints/.gitkeep /app/checkpoints/
COPY --chown=appuser:appuser docs/ /app/docs/
COPY --chown=appuser:appuser README.md /app/
COPY --chown=appuser:appuser LICENSE /app/

# Set ownership
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command (can be overridden)
CMD ["python", "-m", "uvicorn", "backend.api:app", "--host", "0.0.0.0", "--port", "8000"]

# -----------------------------------------------------------------------------
# Stage 4: Development Image (with additional tools)
# -----------------------------------------------------------------------------
FROM production as development

USER root

# Install development tools
RUN apt-get update && apt-get install -y \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Install development Python packages
RUN pip install --no-cache-dir \
    pytest \
    pytest-cov \
    black \
    flake8 \
    mypy \
    ipython \
    jupyter

USER appuser

# Development command
CMD ["python", "-m", "uvicorn", "backend.api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
