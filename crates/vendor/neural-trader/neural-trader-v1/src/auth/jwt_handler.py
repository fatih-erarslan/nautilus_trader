"""
JWT Authentication Handler for Neural Trader
Optional authentication that can be enabled/disabled via environment variables
"""

import os
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import HTTPException, Security, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import structlog

logger = structlog.get_logger()

# Configuration from environment
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_EXPIRATION_HOURS = int(os.getenv("JWT_EXPIRATION_HOURS", "24"))
AUTH_ENABLED = os.getenv("AUTH_ENABLED", "false").lower() == "true"
AUTH_USERNAME = os.getenv("AUTH_USERNAME", "admin")
AUTH_PASSWORD = os.getenv("AUTH_PASSWORD", "changeme")
API_KEY = os.getenv("API_KEY", "")  # Optional API key for simple auth

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Security scheme
security = HTTPBearer(auto_error=False)

class JWTHandler:
    """Handler for JWT operations"""
    
    @staticmethod
    def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """Create a new JWT access token"""
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS)
        
        to_encode.update({"exp": expire, "iat": datetime.utcnow()})
        
        encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
        logger.info("Created new JWT token", expires_at=expire.isoformat())
        
        return encoded_jwt
    
    @staticmethod
    def verify_token(token: str) -> Dict[str, Any]:
        """Verify and decode a JWT token"""
        try:
            payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
            return payload
        except JWTError as e:
            logger.warning("JWT verification failed", error=str(e))
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
    
    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash"""
        return pwd_context.verify(plain_password, hashed_password)
    
    @staticmethod
    def get_password_hash(password: str) -> str:
        """Generate password hash"""
        return pwd_context.hash(password)

def get_current_user_optional(credentials: HTTPAuthorizationCredentials = Security(security)) -> Optional[Dict[str, Any]]:
    """
    Optional authentication - returns user if authenticated, None otherwise
    This allows endpoints to work both with and without authentication
    """
    if not AUTH_ENABLED:
        # Authentication disabled, allow all requests
        return {"username": "anonymous", "authenticated": False}
    
    if not credentials:
        # No credentials provided, but auth is enabled
        return None
    
    try:
        # Check if it's an API key
        if API_KEY and credentials.credentials == API_KEY:
            return {"username": "api_key_user", "authenticated": True, "auth_type": "api_key"}
        
        # Try to verify as JWT
        payload = JWTHandler.verify_token(credentials.credentials)
        return {
            "username": payload.get("sub", "unknown"),
            "authenticated": True,
            "auth_type": "jwt",
            **payload
        }
    except HTTPException:
        return None

def get_current_user_required(credentials: HTTPAuthorizationCredentials = Security(security)) -> Dict[str, Any]:
    """
    Required authentication - raises exception if not authenticated
    Use this for endpoints that must be protected
    """
    if not AUTH_ENABLED:
        # Authentication disabled, allow all requests
        return {"username": "anonymous", "authenticated": False}
    
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Check if it's an API key
    if API_KEY and credentials.credentials == API_KEY:
        return {"username": "api_key_user", "authenticated": True, "auth_type": "api_key"}
    
    # Verify as JWT
    payload = JWTHandler.verify_token(credentials.credentials)
    return {
        "username": payload.get("sub", "unknown"),
        "authenticated": True,
        "auth_type": "jwt",
        **payload
    }

def check_auth_optional(user: Optional[Dict[str, Any]] = Depends(get_current_user_optional)):
    """
    Dependency for optional authentication
    Logs access but doesn't block unauthenticated requests
    """
    if user:
        logger.info("Authenticated request", username=user.get("username"), auth_type=user.get("auth_type"))
    else:
        logger.info("Unauthenticated request")
    return user

def check_auth_required(user: Dict[str, Any] = Depends(get_current_user_required)):
    """
    Dependency for required authentication
    Blocks unauthenticated requests
    """
    logger.info("Authenticated request", username=user.get("username"), auth_type=user.get("auth_type"))
    return user

# Authentication configuration helper
def get_auth_config() -> Dict[str, Any]:
    """Get current authentication configuration"""
    return {
        "enabled": AUTH_ENABLED,
        "jwt_algorithm": JWT_ALGORITHM,
        "token_expiration_hours": JWT_EXPIRATION_HOURS,
        "api_key_enabled": bool(API_KEY),
        "username_configured": bool(AUTH_USERNAME != "admin"),
    }

# Token validation helper
def validate_token(token: str) -> bool:
    """Simple token validation check"""
    try:
        JWTHandler.verify_token(token)
        return True
    except:
        return False