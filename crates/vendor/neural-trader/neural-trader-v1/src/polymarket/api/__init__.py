"""
Polymarket API Clients

This module provides API clients for interacting with different Polymarket endpoints:
- CLOBClient: Central Limit Order Book API for trading
- GammaClient: Gamma API for market data
"""

from .clob_client import CLOBClient
from .base import PolymarketClient, PolymarketAPIError, RateLimitError, AuthenticationError, ValidationError

# Placeholder for GammaClient (to be implemented)
try:
    from .gamma_client import GammaClient
except ImportError:
    GammaClient = None

__all__ = [
    "CLOBClient",
    "PolymarketClient", 
    "PolymarketAPIError",
    "RateLimitError",
    "AuthenticationError",
    "ValidationError",
]

if GammaClient:
    __all__.append("GammaClient")