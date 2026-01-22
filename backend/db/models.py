"""Database models and ORM setup using SQLAlchemy"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

Base = declarative_base()


class User(Base):
    """User model for authentication"""
    __tablename__ = "users"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    is_verified = Column(Boolean, default=False)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)
    
    # Relationships
    api_keys = relationship("APIKey", back_populates="user", cascade="all, delete-orphan")
    inference_logs = relationship("InferenceLog", back_populates="user", cascade="all, delete-orphan")
    training_jobs = relationship("TrainingJob", back_populates="user", cascade="all, delete-orphan")


class APIKey(Base):
    """API keys for programmatic access"""
    __tablename__ = "api_keys"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    key = Column(String(64), unique=True, index=True, nullable=False)
    name = Column(String(100), nullable=False)
    
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)
    last_used = Column(DateTime, nullable=True)
    
    # Rate limiting
    rate_limit = Column(Integer, default=1000)  # requests per hour
    
    user = relationship("User", back_populates="api_keys")


class InferenceLog(Base):
    """Logs for inference requests"""
    __tablename__ = "inference_logs"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("users.id"), nullable=True)
    
    text_a = Column(Text, nullable=False)
    text_b = Column(Text, nullable=False)
    similarity_score = Column(Float, nullable=False)
    is_paraphrase = Column(Boolean, nullable=False)
    
    inference_time_ms = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(String(255), nullable=True)
    
    user = relationship("User", back_populates="inference_logs")


class TrainingJob(Base):
    """Training job tracking"""
    __tablename__ = "training_jobs"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    
    status = Column(String(20), default="pending")  # pending, running, completed, failed
    dataset_path = Column(String(255), nullable=False)
    model_path = Column(String(255), nullable=True)
    
    # Training params
    num_epochs = Column(Integer, default=50)
    batch_size = Column(Integer, default=32)
    learning_rate = Column(Float, default=2e-4)
    
    # Results
    best_loss = Column(Float, nullable=True)
    best_accuracy = Column(Float, nullable=True)
    current_epoch = Column(Integer, default=0)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    error_message = Column(Text, nullable=True)
    
    user = relationship("User", back_populates="training_jobs")


class SecurityEvent(Base):
    """Security events and audit logs"""
    __tablename__ = "security_events"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    
    event_type = Column(String(50), nullable=False)  # login, logout, failed_login, rate_limit, blocked
    user_id = Column(String(36), ForeignKey("users.id"), nullable=True)
    ip_address = Column(String(45), nullable=False)
    user_agent = Column(String(255), nullable=True)
    
    severity = Column(String(20), default="info")  # info, warning, critical
    description = Column(Text, nullable=True)
    metadata = Column(Text, nullable=True)  # JSON string
    
    created_at = Column(DateTime, default=datetime.utcnow)
