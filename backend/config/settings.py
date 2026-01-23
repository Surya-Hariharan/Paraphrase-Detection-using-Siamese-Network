"""
Configuration Management
========================

Centralized settings for the entire application using Pydantic settings.
Supports environment variables and .env files.
"""

from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from typing import Optional, List
from pathlib import Path


class DatabaseSettings(BaseSettings):
    """Database configuration"""
    DB_TYPE: str = "sqlite"  # sqlite, postgresql, mysql
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_NAME: str = "paraphrase_db"
    DB_USER: str = "admin"
    DB_PASSWORD: str = ""
    
    @property
    def database_url(self) -> str:
        if self.DB_TYPE == "sqlite":
            return f"sqlite:///./data/{self.DB_NAME}.db"
        elif self.DB_TYPE == "postgresql":
            return f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        return ""
    
    model_config = ConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="allow"
    )


class SecuritySettings(BaseSettings):
    """Security and authentication configuration"""
    SECRET_KEY: str = "change-this-secret-key-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    
    # Password hashing
    BCRYPT_ROUNDS: int = 12
    
    # API Keys
    API_KEY_ENABLED: bool = False
    ADMIN_API_KEY: Optional[str] = None
    
    # Agentic AI API Keys
    GEMINI_API_KEY: Optional[str] = None
    GROQ_API_KEY: Optional[str] = None
    
    # CORS
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:5173"]
    CORS_ALLOW_CREDENTIALS: bool = True
    
    # Rate limiting
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_PER_MINUTE: int = 60
    
    # DDoS protection
    DDOS_PROTECTION_ENABLED: bool = True
    MAX_CLIENTS: int = 10000
    
    model_config = ConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="allow"
    )


class ModelSettings(BaseSettings):
    """ML Model configuration"""
    MODEL_PATH: str = "checkpoints/best_model.pt"
    ENCODER_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIM: int = 768
    PROJECTION_DIM: int = 256
    DEVICE: str = "cuda"
    
    # Inference settings
    BATCH_SIZE: int = 32
    MAX_SEQUENCE_LENGTH: int = 512
    SIMILARITY_THRESHOLD: float = 0.5
    
    model_config = ConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="allow"
    )


class TrainingSettings(BaseSettings):
    """Training configuration"""
    DATA_PATH: str = "data/quora_siamese_train.csv"
    CHECKPOINT_DIR: str = "checkpoints"
    
    # Hyperparameters
    NUM_EPOCHS: int = 50
    MIN_EPOCHS: int = 30
    BATCH_SIZE: int = 32
    LEARNING_RATE: float = 2e-4
    WEIGHT_DECAY: float = 0.01
    
    # Early stopping
    EARLY_STOPPING_PATIENCE: int = 8
    SAVE_EVERY_N_EPOCHS: int = 5
    
    # Optimization
    USE_MIXED_PRECISION: bool = True
    GRADIENT_ACCUMULATION_STEPS: int = 1
    WARMUP_RATIO: float = 0.1
    
    model_config = ConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="allow"
    )


class AppSettings(BaseSettings):
    """Main application settings"""
    APP_NAME: str = "Paraphrase Detection API"
    APP_VERSION: str = "2.0.0"
    DEBUG: bool = False
    
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 4
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/app.log"
    
    # Features
    ENABLE_TRAINING_API: bool = False  # Disable in production
    ENABLE_ADMIN_API: bool = False
    ENABLE_METRICS: bool = True
    
    # File uploads
    MAX_UPLOAD_SIZE_MB: int = 10
    ALLOWED_EXTENSIONS: List[str] = [".txt", ".pdf", ".docx"]
    
    model_config = ConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="allow"
    )


class Settings:
    """Combined settings"""
    def __init__(self):
        self.app = AppSettings()
        self.db = DatabaseSettings()
        self.security = SecuritySettings()
        self.model = ModelSettings()
        self.training = TrainingSettings()


# Global settings instance
settings = Settings()
