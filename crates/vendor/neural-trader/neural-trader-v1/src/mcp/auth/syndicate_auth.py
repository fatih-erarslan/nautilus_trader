"""
Syndicate Authentication and Access Control System
Implements JWT-based authentication with role-based access control
"""

import os
import jwt
import hashlib
import secrets
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from functools import wraps
import json
import logging
from enum import Enum

from fastapi import HTTPException, Request, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# Import member roles from syndicate module
from src.syndicate.member_management import MemberRole, MemberPermissions


# Configure logging
logger = logging.getLogger(__name__)


class AuthenticationError(Exception):
    """Custom authentication error"""
    pass


class AuthorizationError(Exception):
    """Custom authorization error"""
    pass


@dataclass
class AuthToken:
    """Authentication token data structure"""
    member_id: str
    syndicate_id: str
    role: MemberRole
    permissions: MemberPermissions
    issued_at: datetime
    expires_at: datetime
    token_id: str
    
    @property
    def is_expired(self) -> bool:
        """Check if token is expired"""
        return datetime.now(timezone.utc) > self.expires_at
    
    @property
    def remaining_time(self) -> timedelta:
        """Get remaining time before expiration"""
        return self.expires_at - datetime.now(timezone.utc)


@dataclass
class Session:
    """User session information"""
    session_id: str
    member_id: str
    syndicate_id: str
    role: MemberRole
    created_at: datetime
    last_activity: datetime
    ip_address: str
    user_agent: str
    is_active: bool = True


class AuditAction(Enum):
    """Audit log action types"""
    LOGIN = "login"
    LOGOUT = "logout"
    FAILED_LOGIN = "failed_login"
    TOKEN_REFRESH = "token_refresh"
    PERMISSION_DENIED = "permission_denied"
    FUND_TRANSFER = "fund_transfer"
    VOTE_CAST = "vote_cast"
    MEMBER_ADDED = "member_added"
    MEMBER_REMOVED = "member_removed"
    ROLE_CHANGED = "role_changed"
    STRATEGY_MODIFIED = "strategy_modified"
    BET_PLACED = "bet_placed"
    PROFIT_DISTRIBUTED = "profit_distributed"
    DATA_ACCESS = "data_access"
    SETTINGS_CHANGED = "settings_changed"


class SyndicateAuthenticator:
    """Main authentication handler for syndicate operations"""
    
    def __init__(self, secret_key: Optional[str] = None, token_expiry_hours: int = 24):
        self.secret_key = secret_key or os.environ.get('JWT_SECRET_KEY', self._generate_secret_key())
        self.token_expiry_hours = token_expiry_hours
        self.algorithm = "HS256"
        self.sessions: Dict[str, Session] = {}
        self.rate_limiter = RateLimiter()
        self.audit_logger = AuditLogger()
        
    def _generate_secret_key(self) -> str:
        """Generate a secure secret key"""
        return secrets.token_urlsafe(32)
    
    def hash_password(self, password: str, salt: Optional[str] = None) -> Tuple[str, str]:
        """Hash password with salt"""
        if not salt:
            salt = secrets.token_hex(16)
        
        # Use PBKDF2 for password hashing
        key = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000  # iterations
        )
        
        return key.hex(), salt
    
    def verify_password(self, password: str, hashed_password: str, salt: str) -> bool:
        """Verify password against hash"""
        key, _ = self.hash_password(password, salt)
        return key == hashed_password
    
    def generate_token(self, member_id: str, syndicate_id: str, role: MemberRole,
                      permissions: MemberPermissions) -> str:
        """Generate JWT token for authenticated member"""
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(hours=self.token_expiry_hours)
        token_id = secrets.token_urlsafe(16)
        
        # Create token payload
        payload = {
            'member_id': member_id,
            'syndicate_id': syndicate_id,
            'role': role.value,
            'permissions': self._serialize_permissions(permissions),
            'iat': now.timestamp(),
            'exp': expires_at.timestamp(),
            'jti': token_id
        }
        
        # Generate token
        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        
        # Log token generation
        self.audit_logger.log(
            AuditAction.TOKEN_REFRESH,
            member_id=member_id,
            syndicate_id=syndicate_id,
            details={'token_id': token_id}
        )
        
        return token
    
    def verify_token(self, token: str) -> AuthToken:
        """Verify and decode JWT token"""
        try:
            # Decode token
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            # Extract data
            auth_token = AuthToken(
                member_id=payload['member_id'],
                syndicate_id=payload['syndicate_id'],
                role=MemberRole(payload['role']),
                permissions=self._deserialize_permissions(payload['permissions']),
                issued_at=datetime.fromtimestamp(payload['iat'], timezone.utc),
                expires_at=datetime.fromtimestamp(payload['exp'], timezone.utc),
                token_id=payload['jti']
            )
            
            # Check if expired
            if auth_token.is_expired:
                raise AuthenticationError("Token has expired")
            
            return auth_token
            
        except jwt.ExpiredSignatureError:
            raise AuthenticationError("Token has expired")
        except jwt.InvalidTokenError as e:
            raise AuthenticationError(f"Invalid token: {str(e)}")
    
    def create_session(self, member_id: str, syndicate_id: str, role: MemberRole,
                      ip_address: str, user_agent: str) -> Session:
        """Create new user session"""
        session_id = secrets.token_urlsafe(32)
        now = datetime.now(timezone.utc)
        
        session = Session(
            session_id=session_id,
            member_id=member_id,
            syndicate_id=syndicate_id,
            role=role,
            created_at=now,
            last_activity=now,
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        self.sessions[session_id] = session
        
        # Log session creation
        self.audit_logger.log(
            AuditAction.LOGIN,
            member_id=member_id,
            syndicate_id=syndicate_id,
            details={
                'session_id': session_id,
                'ip_address': ip_address
            }
        )
        
        return session
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """Get active session"""
        session = self.sessions.get(session_id)
        
        if session and session.is_active:
            # Update last activity
            session.last_activity = datetime.now(timezone.utc)
            return session
        
        return None
    
    def end_session(self, session_id: str):
        """End user session"""
        session = self.sessions.get(session_id)
        
        if session:
            session.is_active = False
            
            # Log logout
            self.audit_logger.log(
                AuditAction.LOGOUT,
                member_id=session.member_id,
                syndicate_id=session.syndicate_id,
                details={'session_id': session_id}
            )
    
    def _serialize_permissions(self, permissions: MemberPermissions) -> Dict[str, bool]:
        """Serialize permissions to dict"""
        return {
            'create_syndicate': permissions.create_syndicate,
            'modify_strategy': permissions.modify_strategy,
            'approve_large_bets': permissions.approve_large_bets,
            'manage_members': permissions.manage_members,
            'distribute_profits': permissions.distribute_profits,
            'access_all_analytics': permissions.access_all_analytics,
            'veto_power': permissions.veto_power,
            'propose_bets': permissions.propose_bets,
            'access_advanced_analytics': permissions.access_advanced_analytics,
            'create_models': permissions.create_models,
            'vote_on_strategy': permissions.vote_on_strategy,
            'manage_junior_analysts': permissions.manage_junior_analysts,
            'view_bets': permissions.view_bets,
            'vote_on_major_decisions': permissions.vote_on_major_decisions,
            'access_basic_analytics': permissions.access_basic_analytics,
            'propose_ideas': permissions.propose_ideas,
            'withdraw_own_funds': permissions.withdraw_own_funds
        }
    
    def _deserialize_permissions(self, permissions_dict: Dict[str, bool]) -> MemberPermissions:
        """Deserialize permissions from dict"""
        return MemberPermissions(**permissions_dict)


class RateLimiter:
    """Rate limiting for sensitive operations"""
    
    def __init__(self):
        self.attempts: Dict[str, List[datetime]] = {}
        self.limits = {
            'login': (5, 300),  # 5 attempts per 5 minutes
            'fund_transfer': (10, 3600),  # 10 per hour
            'api_general': (100, 60),  # 100 per minute
            'bet_placement': (20, 60)  # 20 per minute
        }
    
    def check_rate_limit(self, identifier: str, action: str) -> bool:
        """Check if action is within rate limit"""
        if action not in self.limits:
            return True
        
        max_attempts, time_window = self.limits[action]
        key = f"{identifier}:{action}"
        now = datetime.now(timezone.utc)
        
        # Clean old attempts
        if key in self.attempts:
            self.attempts[key] = [
                attempt for attempt in self.attempts[key]
                if (now - attempt).total_seconds() < time_window
            ]
        else:
            self.attempts[key] = []
        
        # Check limit
        if len(self.attempts[key]) >= max_attempts:
            return False
        
        # Record attempt
        self.attempts[key].append(now)
        return True
    
    def get_remaining_time(self, identifier: str, action: str) -> int:
        """Get remaining time until rate limit resets (in seconds)"""
        if action not in self.limits:
            return 0
        
        _, time_window = self.limits[action]
        key = f"{identifier}:{action}"
        
        if key not in self.attempts or not self.attempts[key]:
            return 0
        
        oldest_attempt = min(self.attempts[key])
        elapsed = (datetime.now(timezone.utc) - oldest_attempt).total_seconds()
        
        return max(0, int(time_window - elapsed))


class AuditLogger:
    """Audit logging for security-sensitive operations"""
    
    def __init__(self, log_file: str = "syndicate_audit.log"):
        self.log_file = log_file
        self.logger = logging.getLogger("syndicate.audit")
        
        # Configure file handler
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def log(self, action: AuditAction, member_id: str, syndicate_id: str,
            details: Optional[Dict[str, Any]] = None):
        """Log an audit event"""
        event = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'action': action.value,
            'member_id': member_id,
            'syndicate_id': syndicate_id,
            'details': details or {}
        }
        
        # Log to file
        self.logger.info(json.dumps(event))
        
        # In production, also send to centralized logging system
        # self._send_to_central_logging(event)
    
    def log_financial_operation(self, operation_type: str, member_id: str,
                              syndicate_id: str, amount: float, currency: str,
                              details: Dict[str, Any]):
        """Log financial operations with extra detail"""
        event = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'action': AuditAction.FUND_TRANSFER.value,
            'operation_type': operation_type,
            'member_id': member_id,
            'syndicate_id': syndicate_id,
            'amount': amount,
            'currency': currency,
            'details': details
        }
        
        self.logger.info(json.dumps(event))


# FastAPI dependencies
security = HTTPBearer()


def get_current_member(credentials: HTTPAuthorizationCredentials = Depends(security)) -> AuthToken:
    """FastAPI dependency to get current authenticated member"""
    authenticator = get_authenticator()
    
    try:
        token = authenticator.verify_token(credentials.credentials)
        return token
    except AuthenticationError as e:
        raise HTTPException(status_code=401, detail=str(e))


def require_permission(permission: str):
    """Decorator to require specific permission"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get current member from request context
            request = kwargs.get('request')
            if not request:
                raise AuthorizationError("No request context available")
            
            # Get auth token
            auth_token = getattr(request.state, 'auth_token', None)
            if not auth_token:
                raise AuthorizationError("Not authenticated")
            
            # Check permission
            if not hasattr(auth_token.permissions, permission) or \
               not getattr(auth_token.permissions, permission):
                raise AuthorizationError(f"Permission denied: {permission}")
            
            # Log permission check
            get_audit_logger().log(
                AuditAction.DATA_ACCESS,
                member_id=auth_token.member_id,
                syndicate_id=auth_token.syndicate_id,
                details={'permission': permission, 'granted': True}
            )
            
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator


def require_role(min_role: MemberRole):
    """Decorator to require minimum role level"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get current member from request context
            request = kwargs.get('request')
            if not request:
                raise AuthorizationError("No request context available")
            
            # Get auth token
            auth_token = getattr(request.state, 'auth_token', None)
            if not auth_token:
                raise AuthorizationError("Not authenticated")
            
            # Define role hierarchy
            role_hierarchy = {
                MemberRole.OBSERVER: 0,
                MemberRole.CONTRIBUTING_MEMBER: 1,
                MemberRole.JUNIOR_ANALYST: 2,
                MemberRole.SENIOR_ANALYST: 3,
                MemberRole.LEAD_INVESTOR: 4
            }
            
            # Check role level
            if role_hierarchy.get(auth_token.role, 0) < role_hierarchy.get(min_role, 0):
                raise AuthorizationError(f"Insufficient role: requires {min_role.value}")
            
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator


# Global instances
_authenticator = None
_audit_logger = None


def get_authenticator() -> SyndicateAuthenticator:
    """Get global authenticator instance"""
    global _authenticator
    if _authenticator is None:
        _authenticator = SyndicateAuthenticator()
    return _authenticator


def get_audit_logger() -> AuditLogger:
    """Get global audit logger instance"""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger