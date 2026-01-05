# Procfile for Heroku and Railway deployment

# Web process (API server)
web: uvicorn backend.api:app --host 0.0.0.0 --port $PORT --workers 4

# Worker process (optional, for background tasks)
# worker: celery -A backend.celery_app worker --loglevel=info

# Release phase (run migrations, downloads, etc.)
# release: python backend/setup_production.py
