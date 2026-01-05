"""
Configuration management for production deployment.

Loads configuration from environment variables with sensible defaults.
"""

import os
from typing import Optional
from pathlib import Path
from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # =============================================================================
    # Application
    # =============================================================================
    
    app_name: str = Field(default="Paraphrase Detection API", env="APP_NAME")
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=True, env="DEBUG")
    secret_key: str = Field(default="dev-secret-key", env="SECRET_KEY")
    
    # =============================================================================
    # API Configuration
    # =============================================================================
    
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_workers: int = Field(default=4, env="API_WORKERS")
    
    # CORS
    enable_cors: bool = Field(default=True, env="ENABLE_CORS")
    allowed_origins: str = Field(default="*", env="ALLOWED_ORIGINS")
    
    # =============================================================================
    # Model Configuration
    # =============================================================================
    
    sbert_model_name: str = Field(default="all-MiniLM-L6-v2", env="SBERT_MODEL_NAME")
    projection_dim: int = Field(default=256, env="PROJECTION_DIM")
    similarity_threshold: float = Field(default=0.75, env="SIMILARITY_THRESHOLD")
    
    # =============================================================================
    # Document Processing
    # =============================================================================
    
    chunk_size: int = Field(default=1000, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, env="CHUNK_OVERLAP")
    aggregation_method: str = Field(default="mean", env="AGGREGATION_METHOD")
    
    # =============================================================================
    # Paths
    # =============================================================================
    
    data_dir: Path = Field(default=Path("data"), env="DATA_DIR")
    models_dir: Path = Field(default=Path("models"), env="MODELS_DIR")
    checkpoints_dir: Path = Field(default=Path("checkpoints"), env="CHECKPOINTS_DIR")
    logs_dir: Path = Field(default=Path("logs"), env="LOGS_DIR")
    uploads_dir: Path = Field(default=Path("uploads"), env="UPLOADS_DIR")
    temp_dir: Path = Field(default=Path("temp"), env="TEMP_DIR")
    
    model_path: Optional[str] = Field(default=None, env="MODEL_PATH")
    
    # =============================================================================
    # Training (if needed)
    # =============================================================================
    
    epochs: int = Field(default=10, env="EPOCHS")
    batch_size: int = Field(default=32, env="BATCH_SIZE")
    learning_rate: float = Field(default=0.0001, env="LEARNING_RATE")
    device: str = Field(default="cuda", env="DEVICE")
    freeze_sbert: bool = Field(default=True, env="FREEZE_SBERT")
    unfreeze_last_layer: bool = Field(default=False, env="UNFREEZE_LAST_LAYER")
    
    # =============================================================================
    # LLM APIs (for agentic evaluation)
    # =============================================================================
    
    groq_api_key: Optional[str] = Field(default=None, env="GROQ_API_KEY")
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    
    # =============================================================================
    # Database (optional)
    # =============================================================================
    
    database_url: Optional[str] = Field(default=None, env="DATABASE_URL")
    redis_url: Optional[str] = Field(default=None, env="REDIS_URL")
    
    # =============================================================================
    # Monitoring & Logging
    # =============================================================================
    
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    enable_telemetry: bool = Field(default=False, env="ENABLE_TELEMETRY")
    sentry_dsn: Optional[str] = Field(default=None, env="SENTRY_DSN")
    
    # =============================================================================
    # File Upload
    # =============================================================================
    
    max_upload_size_mb: int = Field(default=10, env="MAX_UPLOAD_SIZE_MB")
    
    class Config:
        env_file = ".env"
        case_sensitive = False
    
    @property
    def is_production(self) -> bool:
        """Check if running in production"""
        return self.environment.lower() == "production"
    
    @property
    def allowed_origins_list(self) -> list:
        """Get list of allowed origins for CORS"""
        if self.allowed_origins == "*":
            return ["*"]
        return [origin.strip() for origin in self.allowed_origins.split(",")]
    
    def create_directories(self):
        """Create necessary directories"""
        for directory in [
            self.data_dir,
            self.models_dir,
            self.checkpoints_dir,
            self.logs_dir,
            self.uploads_dir,
            self.temp_dir
        ]:
            directory.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()

# Create directories on import
settings.create_directories()
