"""
Authentication Middleware for Syndicate Operations
Handles request authentication, rate limiting, and security headers
"""

import time
from typing import Callable, Optional
from fastapi import Request, Response, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from .syndicate_auth import (
    SyndicateAuthenticator, AuthenticationError, AuthorizationError,
    get_authenticator, get_audit_logger, AuditAction
)


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """Middleware to handle JWT authentication for all requests"""
    
    def __init__(self, app: ASGIApp, authenticator: Optional[SyndicateAuthenticator] = None,
                 exclude_paths: Optional[list] = None):
        super().__init__(app)
        self.authenticator = authenticator or get_authenticator()
        self.exclude_paths = exclude_paths or [
            "/health",
            "/docs",
            "/openapi.json",
            "/auth/login",
            "/auth/register"
        ]
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process each request for authentication"""
        
        # Skip authentication for excluded paths
        if request.url.path in self.exclude_paths:
            return await call_next(request)
        
        # Extract token from Authorization header
        auth_header = request.headers.get("Authorization")
        
        if not auth_header or not auth_header.startswith("Bearer "):
            return JSONResponse(
                status_code=401,
                content={"detail": "Missing authentication token"}
            )
        
        token = auth_header.split(" ")[1]
        
        try:
            # Verify token
            auth_token = self.authenticator.verify_token(token)
            
            # Attach auth token to request state
            request.state.auth_token = auth_token
            
            # Process request
            response = await call_next(request)
            
            return response
            
        except AuthenticationError as e:
            return JSONResponse(
                status_code=401,
                content={"detail": str(e)}
            )
        except Exception as e:
            # Log unexpected errors
            get_audit_logger().log(
                AuditAction.PERMISSION_DENIED,
                member_id="unknown",
                syndicate_id="unknown",
                details={"error": str(e), "path": request.url.path}
            )
            return JSONResponse(
                status_code=500,
                content={"detail": "Internal server error"}
            )


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware to handle rate limiting"""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.rate_limiter = get_authenticator().rate_limiter
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Check rate limits for each request"""
        
        # Get identifier (IP address or member ID)
        identifier = request.client.host
        
        if hasattr(request.state, 'auth_token'):
            identifier = request.state.auth_token.member_id
        
        # Determine action type based on path
        action = self._get_action_type(request.url.path)
        
        # Check rate limit
        if not self.rate_limiter.check_rate_limit(identifier, action):
            retry_after = self.rate_limiter.get_remaining_time(identifier, action)
            
            return JSONResponse(
                status_code=429,
                content={
                    "detail": "Rate limit exceeded",
                    "retry_after": retry_after
                },
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(self.rate_limiter.limits[action][0]),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(time.time()) + retry_after)
                }
            )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        if action in self.rate_limiter.limits:
            max_attempts, _ = self.rate_limiter.limits[action]
            key = f"{identifier}:{action}"
            current_attempts = len(self.rate_limiter.attempts.get(key, []))
            remaining = max(0, max_attempts - current_attempts)
            
            response.headers["X-RateLimit-Limit"] = str(max_attempts)
            response.headers["X-RateLimit-Remaining"] = str(remaining)
        
        return response
    
    def _get_action_type(self, path: str) -> str:
        """Determine action type from request path"""
        if path.startswith("/auth/login"):
            return "login"
        elif path.startswith("/funds/transfer"):
            return "fund_transfer"
        elif path.startswith("/bets/place"):
            return "bet_placement"
        else:
            return "api_general"


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware to add security headers to all responses"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Add security headers to response"""
        
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        
        # Remove sensitive headers
        response.headers.pop("Server", None)
        response.headers.pop("X-Powered-By", None)
        
        return response


class AuditLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log all requests for audit purposes"""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.audit_logger = get_audit_logger()
        self.sensitive_paths = [
            "/funds/",
            "/members/",
            "/votes/",
            "/strategy/",
            "/profits/"
        ]
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Log requests to sensitive endpoints"""
        
        # Check if this is a sensitive path
        is_sensitive = any(request.url.path.startswith(path) for path in self.sensitive_paths)
        
        if is_sensitive:
            # Get member info if authenticated
            member_id = "anonymous"
            syndicate_id = "unknown"
            
            if hasattr(request.state, 'auth_token'):
                member_id = request.state.auth_token.member_id
                syndicate_id = request.state.auth_token.syndicate_id
            
            # Log the request
            self.audit_logger.log(
                AuditAction.DATA_ACCESS,
                member_id=member_id,
                syndicate_id=syndicate_id,
                details={
                    "method": request.method,
                    "path": request.url.path,
                    "ip_address": request.client.host,
                    "user_agent": request.headers.get("User-Agent", "unknown")
                }
            )
        
        # Process request
        start_time = time.time()
        response = await call_next(request)
        duration = time.time() - start_time
        
        # Log response for sensitive operations
        if is_sensitive and response.status_code >= 400:
            self.audit_logger.log(
                AuditAction.PERMISSION_DENIED,
                member_id=member_id if 'member_id' in locals() else "anonymous",
                syndicate_id=syndicate_id if 'syndicate_id' in locals() else "unknown",
                details={
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "duration": duration
                }
            )
        
        return response


class CORSMiddleware(BaseHTTPMiddleware):
    """Custom CORS middleware with proper security"""
    
    def __init__(self, app: ASGIApp, allowed_origins: Optional[list] = None):
        super().__init__(app)
        self.allowed_origins = allowed_origins or ["http://localhost:3000"]
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Handle CORS headers"""
        
        # Handle preflight requests
        if request.method == "OPTIONS":
            return self._preflight_response(request)
        
        # Process request
        response = await call_next(request)
        
        # Add CORS headers
        origin = request.headers.get("Origin")
        
        if origin in self.allowed_origins:
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Credentials"] = "true"
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = "Authorization, Content-Type"
            response.headers["Access-Control-Max-Age"] = "3600"
        
        return response
    
    def _preflight_response(self, request: Request) -> Response:
        """Handle preflight OPTIONS requests"""
        origin = request.headers.get("Origin")
        
        if origin not in self.allowed_origins:
            return Response(status_code=403)
        
        return Response(
            status_code=200,
            headers={
                "Access-Control-Allow-Origin": origin,
                "Access-Control-Allow-Credentials": "true",
                "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
                "Access-Control-Allow-Headers": "Authorization, Content-Type",
                "Access-Control-Max-Age": "3600"
            }
        )