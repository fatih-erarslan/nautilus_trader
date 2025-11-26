"""
Authentication fixtures for Polymarket API testing.

Contains mock API keys, signatures, and authentication responses
for testing security and authentication flows.
"""

import hashlib
import hmac
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import base64
import secrets


class AuthFixtures:
    """Authentication-related test fixtures."""
    
    # Test API credentials (NOT real credentials)
    TEST_API_KEY = "test_api_key_12345678901234567890"
    TEST_PRIVATE_KEY = "0x" + secrets.token_hex(32)
    TEST_SECRET_KEY = secrets.token_hex(32)
    
    @staticmethod
    def valid_api_key() -> str:
        """Return a valid test API key."""
        return AuthFixtures.TEST_API_KEY
    
    @staticmethod
    def invalid_api_key() -> str:
        """Return an invalid API key for testing error cases."""
        return "invalid_key_123"
    
    @staticmethod
    def expired_api_key() -> str:
        """Return an expired API key."""
        return "expired_key_" + str(int(time.time() - 86400))  # 1 day ago
    
    @staticmethod
    def test_private_key() -> str:
        """Return a test private key (Ethereum format)."""
        return AuthFixtures.TEST_PRIVATE_KEY
    
    @staticmethod
    def generate_signature(
        method: str,
        path: str,
        body: str = "",
        timestamp: Optional[int] = None
    ) -> Dict[str, str]:
        """Generate a mock signature for API requests."""
        if timestamp is None:
            timestamp = int(time.time())
        
        # Create message to sign (similar to real Polymarket format)
        message = f"{timestamp}{method.upper()}{path}{body}"
        
        # Generate HMAC signature
        signature = hmac.new(
            AuthFixtures.TEST_SECRET_KEY.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return {
            "signature": signature,
            "timestamp": str(timestamp),
            "message": message
        }
    
    @staticmethod
    def valid_auth_headers(
        method: str = "GET",
        path: str = "/markets",
        body: str = ""
    ) -> Dict[str, str]:
        """Generate valid authentication headers."""
        sig_data = AuthFixtures.generate_signature(method, path, body)
        
        return {
            "Authorization": f"Bearer {AuthFixtures.TEST_API_KEY}",
            "X-Signature": sig_data["signature"],
            "X-Timestamp": sig_data["timestamp"],
            "Content-Type": "application/json"
        }
    
    @staticmethod
    def invalid_auth_headers() -> Dict[str, str]:
        """Generate invalid authentication headers."""
        return {
            "Authorization": f"Bearer {AuthFixtures.invalid_api_key()}",
            "X-Signature": "invalid_signature",
            "X-Timestamp": str(int(time.time())),
            "Content-Type": "application/json"
        }
    
    @staticmethod
    def expired_auth_headers() -> Dict[str, str]:
        """Generate expired authentication headers."""
        old_timestamp = int(time.time()) - 3600  # 1 hour ago
        sig_data = AuthFixtures.generate_signature("GET", "/markets", timestamp=old_timestamp)
        
        return {
            "Authorization": f"Bearer {AuthFixtures.TEST_API_KEY}",
            "X-Signature": sig_data["signature"],
            "X-Timestamp": str(old_timestamp),
            "Content-Type": "application/json"
        }
    
    @staticmethod
    def jwt_token(
        expiry_minutes: int = 60,
        user_id: str = "test_user_123"
    ) -> str:
        """Generate a mock JWT token."""
        import json
        
        # Header
        header = {
            "alg": "HS256",
            "typ": "JWT"
        }
        
        # Payload
        now = datetime.now()
        payload = {
            "sub": user_id,
            "iat": int(now.timestamp()),
            "exp": int((now + timedelta(minutes=expiry_minutes)).timestamp()),
            "aud": "polymarket-api",
            "iss": "polymarket-test"
        }
        
        # Encode (simplified, not cryptographically secure)
        header_b64 = base64.urlsafe_b64encode(json.dumps(header).encode()).decode().rstrip('=')
        payload_b64 = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip('=')
        
        # Mock signature
        signature = base64.urlsafe_b64encode(
            hashlib.sha256(f"{header_b64}.{payload_b64}".encode()).digest()
        ).decode().rstrip('=')
        
        return f"{header_b64}.{payload_b64}.{signature}"
    
    @staticmethod
    def expired_jwt_token() -> str:
        """Generate an expired JWT token."""
        return AuthFixtures.jwt_token(expiry_minutes=-60)  # Expired 1 hour ago
    
    @staticmethod
    def authentication_response(success: bool = True) -> Dict[str, Any]:
        """Generate authentication response."""
        if success:
            return {
                "status": "success",
                "data": {
                    "authenticated": True,
                    "user_id": "test_user_123",
                    "api_key": AuthFixtures.TEST_API_KEY,
                    "permissions": [
                        "read_markets",
                        "place_orders",
                        "cancel_orders",
                        "view_positions"
                    ],
                    "rate_limits": {
                        "requests_per_minute": 1000,
                        "orders_per_minute": 100,
                        "websocket_connections": 5
                    },
                    "session_expires_at": (datetime.now() + timedelta(hours=24)).isoformat() + "Z"
                },
                "timestamp": datetime.now().isoformat() + "Z"
            }
        else:
            return {
                "status": "error",
                "error": {
                    "code": "AUTHENTICATION_FAILED",
                    "message": "Invalid API key or signature",
                    "details": {
                        "reason": "Invalid signature",
                        "timestamp": datetime.now().isoformat() + "Z"
                    }
                },
                "timestamp": datetime.now().isoformat() + "Z"
            }
    
    @staticmethod
    def rate_limit_headers(
        limit: int = 1000,
        remaining: int = 995,
        reset_time: Optional[int] = None
    ) -> Dict[str, str]:
        """Generate rate limit headers."""
        if reset_time is None:
            reset_time = int(time.time()) + 3600  # 1 hour from now
        
        return {
            "X-RateLimit-Limit": str(limit),
            "X-RateLimit-Remaining": str(remaining),
            "X-RateLimit-Reset": str(reset_time),
            "X-RateLimit-RetryAfter": str(max(0, reset_time - int(time.time())))
        }
    
    @staticmethod
    def api_key_info_response() -> Dict[str, Any]:
        """Generate API key information response."""
        return {
            "status": "success",
            "data": {
                "api_key": AuthFixtures.TEST_API_KEY,
                "name": "Test API Key",
                "created_at": (datetime.now() - timedelta(days=30)).isoformat() + "Z",
                "last_used": datetime.now().isoformat() + "Z",
                "permissions": [
                    "read_markets",
                    "place_orders",
                    "cancel_orders",
                    "view_positions",
                    "view_trades"
                ],
                "rate_limits": {
                    "requests_per_minute": 1000,
                    "orders_per_minute": 100,
                    "websocket_connections": 5
                },
                "ip_whitelist": [],
                "status": "active",
                "expires_at": (datetime.now() + timedelta(days=365)).isoformat() + "Z"
            },
            "timestamp": datetime.now().isoformat() + "Z"
        }
    
    @staticmethod
    def wallet_connection_data() -> Dict[str, Any]:
        """Generate wallet connection test data."""
        return {
            "wallet_address": "0x" + secrets.token_hex(20),
            "chain_id": 137,  # Polygon
            "network": "polygon",
            "connected": True,
            "balance": {
                "USDC": "10000.50",
                "MATIC": "100.25",
                "native": "100.25"
            },
            "nonce": secrets.randbelow(1000000),
            "signature": "0x" + secrets.token_hex(65),
            "message": "Sign this message to authenticate with Polymarket",
            "timestamp": datetime.now().isoformat() + "Z"
        }
    
    @staticmethod
    def session_data(valid: bool = True) -> Dict[str, Any]:
        """Generate session data for testing."""
        if valid:
            return {
                "session_id": "sess_" + secrets.token_hex(16),
                "user_id": "test_user_123",
                "authenticated": True,
                "created_at": datetime.now().isoformat() + "Z",
                "expires_at": (datetime.now() + timedelta(hours=24)).isoformat() + "Z",
                "last_activity": datetime.now().isoformat() + "Z",
                "permissions": [
                    "read_markets",
                    "place_orders",
                    "cancel_orders",
                    "view_positions"
                ],
                "metadata": {
                    "user_agent": "PolymarketClient/1.0",
                    "ip_address": "192.168.1.100",
                    "device_id": "device_" + secrets.token_hex(8)
                }
            }
        else:
            return {
                "session_id": "sess_invalid",
                "authenticated": False,
                "error": "Session expired or invalid",
                "timestamp": datetime.now().isoformat() + "Z"
            }