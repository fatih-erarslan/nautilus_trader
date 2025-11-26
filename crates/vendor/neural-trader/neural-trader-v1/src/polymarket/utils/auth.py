"""
Authentication utilities for Polymarket API

This module provides authentication and signing functionality for Polymarket API requests,
including order signing, request authentication, and signature verification.
"""

import hashlib
import hmac
import json
import time
from typing import Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)


async def authenticate(
    api_key: Optional[str] = None,
    private_key: Optional[str] = None,
    **kwargs
) -> Dict[str, str]:
    """
    Authenticate with Polymarket API and return authentication headers
    
    Args:
        api_key: API key for authentication
        private_key: Private key for signing requests
        **kwargs: Additional authentication parameters
        
    Returns:
        Dictionary of authentication headers
        
    Raises:
        Exception: If authentication fails
    """
    if not api_key or not private_key:
        raise Exception("API key and private key are required for authentication")
    
    try:
        # Generate timestamp
        timestamp = str(int(time.time() * 1000))
        
        # Create signature
        message = f"{timestamp}{api_key}"
        signature = hmac.new(
            private_key.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        # Return authentication headers
        return {
            'Authorization': f'Bearer {api_key}',
            'X-Signature': signature,
            'X-Timestamp': timestamp,
            'X-API-Key': api_key
        }
        
    except Exception as e:
        logger.error(f"Authentication failed: {e}")
        raise Exception(f"Authentication failed: {e}")


def sign_order(
    order_data: Dict[str, Any],
    private_key: str,
    timestamp: Optional[str] = None
) -> str:
    """
    Sign an order for submission to Polymarket
    
    Args:
        order_data: Order data to sign
        private_key: Private key for signing
        timestamp: Optional timestamp (current time if not provided)
        
    Returns:
        Signature string
        
    Raises:
        Exception: If signing fails
    """
    try:
        if timestamp is None:
            timestamp = str(int(time.time() * 1000))
        
        # Create canonical representation
        canonical_data = json.dumps(order_data, sort_keys=True, separators=(',', ':'))
        
        # Create message to sign
        message = f"{timestamp}{canonical_data}"
        
        # Generate signature
        signature = hmac.new(
            private_key.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return signature
        
    except Exception as e:
        logger.error(f"Order signing failed: {e}")
        raise Exception(f"Order signing failed: {e}")


def verify_signature(
    data: str,
    signature: str,
    public_key: str
) -> bool:
    """
    Verify a signature against data and public key
    
    Args:
        data: Original data that was signed
        signature: Signature to verify
        public_key: Public key for verification
        
    Returns:
        True if signature is valid, False otherwise
    """
    try:
        # This is a simplified implementation
        # Real implementation would use proper cryptographic verification
        expected_signature = hmac.new(
            public_key.encode('utf-8'),
            data.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return hmac.compare_digest(signature, expected_signature)
        
    except Exception as e:
        logger.error(f"Signature verification failed: {e}")
        return False