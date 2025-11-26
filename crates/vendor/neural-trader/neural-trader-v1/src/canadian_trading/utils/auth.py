"""
OAuth2 Authentication Utilities for Canadian Trading Platform

Provides secure token management, refresh mechanisms, and authentication
helpers for various broker APIs.
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Any, Tuple
import aiofiles
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

logger = logging.getLogger(__name__)


class TokenEncryption:
    """Secure token encryption/decryption using Fernet"""
    
    def __init__(self, password: Optional[str] = None):
        """
        Initialize encryption with password
        
        Args:
            password: Password for encryption. If None, uses environment variable
        """
        if password is None:
            password = os.environ.get("TOKEN_ENCRYPTION_PASSWORD", "default-dev-password")
        
        # Derive key from password
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'stable-salt-for-tokens',  # In production, use random salt
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        self.cipher = Fernet(key)
    
    def encrypt(self, data: str) -> str:
        """Encrypt a string"""
        return self.cipher.encrypt(data.encode()).decode()
    
    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt a string"""
        return self.cipher.decrypt(encrypted_data.encode()).decode()


class TokenStorage:
    """Secure storage for OAuth2 tokens"""
    
    def __init__(self, storage_path: Optional[Path] = None):
        """
        Initialize token storage
        
        Args:
            storage_path: Path to store tokens. Defaults to user home directory
        """
        if storage_path is None:
            storage_path = Path.home() / ".canadian_trading" / "tokens"
        
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.encryption = TokenEncryption()
        self._lock = asyncio.Lock()
    
    async def save_tokens(self, broker: str, tokens: Dict[str, Any]):
        """
        Save tokens securely for a broker
        
        Args:
            broker: Broker identifier (e.g., 'questrade', 'ibkr')
            tokens: Token data to store
        """
        async with self._lock:
            file_path = self.storage_path / f"{broker}_tokens.json"
            
            # Add timestamp
            tokens["saved_at"] = datetime.now().isoformat()
            
            # Encrypt sensitive fields
            encrypted_tokens = tokens.copy()
            for field in ["access_token", "refresh_token"]:
                if field in encrypted_tokens:
                    encrypted_tokens[field] = self.encryption.encrypt(encrypted_tokens[field])
            
            # Save to file
            async with aiofiles.open(file_path, 'w') as f:
                await f.write(json.dumps(encrypted_tokens, indent=2))
            
            # Set file permissions (Unix-like systems)
            try:
                os.chmod(file_path, 0o600)  # Read/write for owner only
            except:
                pass  # Windows doesn't support chmod
    
    async def load_tokens(self, broker: str) -> Optional[Dict[str, Any]]:
        """
        Load tokens for a broker
        
        Args:
            broker: Broker identifier
            
        Returns:
            Token data or None if not found
        """
        async with self._lock:
            file_path = self.storage_path / f"{broker}_tokens.json"
            
            if not file_path.exists():
                return None
            
            try:
                async with aiofiles.open(file_path, 'r') as f:
                    encrypted_tokens = json.loads(await f.read())
                
                # Decrypt sensitive fields
                tokens = encrypted_tokens.copy()
                for field in ["access_token", "refresh_token"]:
                    if field in tokens and tokens[field]:
                        tokens[field] = self.encryption.decrypt(tokens[field])
                
                return tokens
                
            except Exception as e:
                logger.error(f"Failed to load tokens for {broker}: {e}")
                return None
    
    async def delete_tokens(self, broker: str):
        """Delete stored tokens for a broker"""
        async with self._lock:
            file_path = self.storage_path / f"{broker}_tokens.json"
            if file_path.exists():
                file_path.unlink()


class OAuth2Manager:
    """
    Generic OAuth2 manager for broker authentication
    """
    
    def __init__(self, broker: str, client_id: Optional[str] = None, 
                 client_secret: Optional[str] = None):
        """
        Initialize OAuth2 manager
        
        Args:
            broker: Broker identifier
            client_id: OAuth2 client ID
            client_secret: OAuth2 client secret
        """
        self.broker = broker
        self.client_id = client_id or os.environ.get(f"{broker.upper()}_CLIENT_ID")
        self.client_secret = client_secret or os.environ.get(f"{broker.upper()}_CLIENT_SECRET")
        self.storage = TokenStorage()
        self.token_cache = {}
        self._refresh_lock = asyncio.Lock()
    
    async def get_valid_token(self) -> Optional[str]:
        """
        Get a valid access token, refreshing if necessary
        
        Returns:
            Valid access token or None
        """
        # Check cache
        if self.broker in self.token_cache:
            cached = self.token_cache[self.broker]
            if cached.get("expires_at") and datetime.now() < cached["expires_at"]:
                return cached["access_token"]
        
        # Load from storage
        tokens = await self.storage.load_tokens(self.broker)
        if not tokens:
            return None
        
        # Check if token is still valid
        expires_at = self._calculate_expiry(tokens)
        if expires_at and datetime.now() < expires_at:
            # Cache and return
            self.token_cache[self.broker] = {
                "access_token": tokens["access_token"],
                "expires_at": expires_at
            }
            return tokens["access_token"]
        
        # Token expired, try to refresh
        if "refresh_token" in tokens:
            return await self.refresh_token(tokens["refresh_token"])
        
        return None
    
    async def refresh_token(self, refresh_token: str) -> Optional[str]:
        """
        Refresh an expired token
        
        Args:
            refresh_token: OAuth2 refresh token
            
        Returns:
            New access token or None
        """
        async with self._refresh_lock:
            # Implementation depends on specific broker
            # This is a placeholder that should be overridden
            logger.warning(f"Token refresh not implemented for {self.broker}")
            return None
    
    async def save_new_tokens(self, access_token: str, refresh_token: Optional[str] = None,
                            expires_in: Optional[int] = None, **kwargs):
        """
        Save new tokens
        
        Args:
            access_token: OAuth2 access token
            refresh_token: OAuth2 refresh token
            expires_in: Token lifetime in seconds
            **kwargs: Additional token data
        """
        tokens = {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "expires_in": expires_in,
            **kwargs
        }
        
        await self.storage.save_tokens(self.broker, tokens)
        
        # Update cache
        expires_at = self._calculate_expiry(tokens)
        self.token_cache[self.broker] = {
            "access_token": access_token,
            "expires_at": expires_at
        }
    
    def _calculate_expiry(self, tokens: Dict[str, Any]) -> Optional[datetime]:
        """Calculate token expiry time"""
        if "expires_in" in tokens and tokens["expires_in"]:
            saved_at = tokens.get("saved_at")
            if saved_at:
                saved_time = datetime.fromisoformat(saved_at)
                return saved_time + timedelta(seconds=tokens["expires_in"])
        
        if "expires_at" in tokens:
            return datetime.fromisoformat(tokens["expires_at"])
        
        return None


class QuestradeOAuth2Manager(OAuth2Manager):
    """
    Questrade-specific OAuth2 implementation
    """
    
    def __init__(self):
        super().__init__("questrade")
        self.auth_url = "https://login.questrade.com/oauth2/token"
    
    async def refresh_token(self, refresh_token: str) -> Optional[str]:
        """
        Refresh Questrade token
        
        Args:
            refresh_token: Current refresh token
            
        Returns:
            New access token or None
        """
        import aiohttp
        
        async with self._refresh_lock:
            params = {
                "grant_type": "refresh_token",
                "refresh_token": refresh_token
            }
            
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(self.auth_url, params=params) as response:
                        if response.status != 200:
                            logger.error(f"Token refresh failed: {response.status}")
                            return None
                        
                        data = await response.json()
                        
                        # Save new tokens
                        await self.save_new_tokens(
                            access_token=data["access_token"],
                            refresh_token=data["refresh_token"],
                            expires_in=data.get("expires_in", 1800),
                            api_server=data.get("api_server")
                        )
                        
                        return data["access_token"]
                        
            except Exception as e:
                logger.error(f"Token refresh error: {e}")
                return None
    
    async def authenticate_with_refresh_token(self, refresh_token: str) -> Dict[str, Any]:
        """
        Initial authentication with a refresh token
        
        Args:
            refresh_token: Manual refresh token from Questrade
            
        Returns:
            Authentication response data
        """
        import aiohttp
        
        params = {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(self.auth_url, params=params) as response:
                if response.status != 200:
                    error_data = await response.json()
                    raise Exception(f"Authentication failed: {error_data}")
                
                data = await response.json()
                
                # Save tokens
                await self.save_new_tokens(
                    access_token=data["access_token"],
                    refresh_token=data["refresh_token"],
                    expires_in=data.get("expires_in", 1800),
                    api_server=data.get("api_server")
                )
                
                return data


class IBKROAuth2Manager(OAuth2Manager):
    """
    Interactive Brokers OAuth2 implementation
    """
    
    def __init__(self, client_id: Optional[str] = None, client_secret: Optional[str] = None):
        super().__init__("ibkr", client_id, client_secret)
        # IBKR uses different authentication mechanism (TWS/Gateway)
        # This is a placeholder for future OAuth2 implementation
    
    async def get_session_token(self) -> Optional[str]:
        """Get IBKR session token from TWS/Gateway"""
        # IBKR typically uses local TWS/Gateway connection
        # This would interface with the running TWS/Gateway instance
        return "ibkr-session-token"


class TokenValidator:
    """Validate and verify tokens"""
    
    @staticmethod
    def is_jwt(token: str) -> bool:
        """Check if token appears to be a JWT"""
        parts = token.split('.')
        return len(parts) == 3
    
    @staticmethod
    def decode_jwt_payload(token: str) -> Optional[Dict[str, Any]]:
        """
        Decode JWT payload (without verification)
        
        Note: This only decodes the payload for inspection.
        Actual verification should be done by the API server.
        """
        try:
            parts = token.split('.')
            if len(parts) != 3:
                return None
            
            # Add padding if needed
            payload = parts[1]
            padding = 4 - len(payload) % 4
            if padding != 4:
                payload += '=' * padding
            
            decoded = base64.urlsafe_b64decode(payload)
            return json.loads(decoded)
            
        except Exception as e:
            logger.error(f"Failed to decode JWT: {e}")
            return None
    
    @staticmethod
    def get_token_expiry(token: str) -> Optional[datetime]:
        """Get expiry time from JWT token"""
        if not TokenValidator.is_jwt(token):
            return None
        
        payload = TokenValidator.decode_jwt_payload(token)
        if payload and "exp" in payload:
            return datetime.fromtimestamp(payload["exp"])
        
        return None


class AuthenticationError(Exception):
    """Custom exception for authentication errors"""
    pass


async def setup_broker_authentication(broker: str, **kwargs) -> Dict[str, Any]:
    """
    Setup authentication for a specific broker
    
    Args:
        broker: Broker identifier ('questrade', 'ibkr', 'oanda')
        **kwargs: Broker-specific authentication parameters
        
    Returns:
        Authentication result
    """
    if broker.lower() == "questrade":
        manager = QuestradeOAuth2Manager()
        
        if "refresh_token" in kwargs:
            # Initial setup with manual refresh token
            return await manager.authenticate_with_refresh_token(kwargs["refresh_token"])
        else:
            # Try to use stored tokens
            token = await manager.get_valid_token()
            if token:
                return {"access_token": token, "status": "authenticated"}
            else:
                return {"status": "authentication_required", 
                       "message": "Please provide a refresh token from Questrade"}
    
    elif broker.lower() == "ibkr":
        manager = IBKROAuth2Manager(
            client_id=kwargs.get("client_id"),
            client_secret=kwargs.get("client_secret")
        )
        token = await manager.get_session_token()
        return {"session_token": token, "status": "connected"}
    
    else:
        raise ValueError(f"Unsupported broker: {broker}")


async def revoke_broker_authentication(broker: str):
    """
    Revoke authentication for a broker
    
    Args:
        broker: Broker identifier
    """
    storage = TokenStorage()
    await storage.delete_tokens(broker)
    
    # Additional broker-specific revocation logic can be added here
    logger.info(f"Authentication revoked for {broker}")