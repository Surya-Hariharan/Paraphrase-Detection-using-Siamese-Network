"""User service"""

from sqlalchemy.orm import Session
from typing import Optional
import secrets

from backend.db.models import User, APIKey
from backend.security.auth.jwt import get_password_hash, verify_password
from datetime import datetime, timedelta


class UserService:
    """User management service"""
    
    @staticmethod
    def create_user(
        db: Session,
        email: str,
        username: str,
        password: str,
        is_admin: bool = False
    ) -> User:
        """Create new user"""
        hashed_password = get_password_hash(password)
        
        user = User(
            email=email,
            username=username,
            hashed_password=hashed_password,
            is_admin=is_admin,
            is_active=True
        )
        
        db.add(user)
        db.commit()
        db.refresh(user)
        
        return user
    
    @staticmethod
    def authenticate_user(db: Session, username: str, password: str) -> Optional[User]:
        """Authenticate user with username/password"""
        user = db.query(User).filter(User.username == username).first()
        
        if not user:
            return None
        
        if not verify_password(password, user.hashed_password):
            return None
        
        return user
    
    @staticmethod
    def get_user_by_email(db: Session, email: str) -> Optional[User]:
        """Get user by email"""
        return db.query(User).filter(User.email == email).first()
    
    @staticmethod
    def get_user_by_username(db: Session, username: str) -> Optional[User]:
        """Get user by username"""
        return db.query(User).filter(User.username == username).first()
    
    @staticmethod
    def create_api_key(
        db: Session,
        user_id: str,
        name: str,
        expires_days: Optional[int] = None
    ) -> APIKey:
        """Create API key for user"""
        key = secrets.token_urlsafe(32)
        
        api_key = APIKey(
            user_id=user_id,
            key=key,
            name=name,
            is_active=True
        )
        
        if expires_days:
            api_key.expires_at = datetime.utcnow() + timedelta(days=expires_days)
        
        db.add(api_key)
        db.commit()
        db.refresh(api_key)
        
        return api_key
    
    @staticmethod
    def revoke_api_key(db: Session, key_id: str) -> bool:
        """Revoke API key"""
        api_key = db.query(APIKey).filter(APIKey.id == key_id).first()
        
        if not api_key:
            return False
        
        api_key.is_active = False
        db.commit()
        
        return True
