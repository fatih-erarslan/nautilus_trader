"""
Sports Betting API Integration Package

This package provides comprehensive sports betting API integrations including:
- TheOddsAPI for real-time odds and market data
- Betfair Exchange for trading capabilities 
- Unified API layer for provider abstraction

The package includes WebSocket streaming, rate limiting, error handling,
failover mechanisms, and data normalization across providers.
"""

from .the_odds_api import TheOddsAPI
from .betfair_api import BetfairAPI
from .unified_api import UnifiedSportsAPI

__all__ = [
    'TheOddsAPI',
    'BetfairAPI', 
    'UnifiedSportsAPI'
]

__version__ = "1.0.0"