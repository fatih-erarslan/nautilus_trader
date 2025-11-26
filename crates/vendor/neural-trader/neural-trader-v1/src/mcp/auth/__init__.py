"""
Syndicate Authentication and Access Control Module
Provides JWT-based authentication, role-based access control, and security features
"""

from .syndicate_auth import (
    SyndicateAuthenticator,
    AuthToken,
    Session,
    AuditAction,
    AuditLogger,
    RateLimiter,
    AuthenticationError,
    AuthorizationError,
    get_authenticator,
    get_audit_logger,
    get_current_member,
    require_permission,
    require_role
)

from .middleware import (
    AuthenticationMiddleware,
    RateLimitMiddleware,
    SecurityHeadersMiddleware,
    AuditLoggingMiddleware,
    CORSMiddleware
)

from .encryption import (
    DataEncryptor,
    SecureDataStore,
    TokenEncryption
)

from .routes import router as auth_router

__all__ = [
    # Auth classes
    'SyndicateAuthenticator',
    'AuthToken',
    'Session',
    'AuditAction',
    'AuditLogger',
    'RateLimiter',
    
    # Exceptions
    'AuthenticationError',
    'AuthorizationError',
    
    # Helper functions
    'get_authenticator',
    'get_audit_logger',
    'get_current_member',
    'require_permission',
    'require_role',
    
    # Middleware
    'AuthenticationMiddleware',
    'RateLimitMiddleware',
    'SecurityHeadersMiddleware',
    'AuditLoggingMiddleware',
    'CORSMiddleware',
    
    # Encryption
    'DataEncryptor',
    'SecureDataStore',
    'TokenEncryption',
    
    # Router
    'auth_router'
]