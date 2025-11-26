"""
Canadian Trading Brokers Module

This module provides production-ready integrations with Canadian brokers.
Currently supported:
- Interactive Brokers Canada (Full trading and data)
- Questrade (Canadian trading)
- OANDA Canada (Forex trading)
"""

from .ib_canada import (
    IBCanadaClient,
    ConnectionConfig,
    ConnectionState,
    OrderType,
    MarketDataConfig,
    PositionInfo,
    OrderInfo
)

from .questrade import (
    QuestradeAPI,
    QuestradeAPIError,
    QuestradeDataFeed,
    QuestradeOrderManager
)

from .oanda_canada import (
    OANDACanada,
    ForexSignal,
    SpreadAnalysis,
    MarginInfo
)

__all__ = [
    # Interactive Brokers
    'IBCanadaClient',
    'ConnectionConfig',
    'ConnectionState',
    'OrderType',
    'MarketDataConfig',
    'PositionInfo',
    'OrderInfo',
    
    # Questrade
    'QuestradeAPI',
    'QuestradeAPIError',
    'QuestradeDataFeed',
    'QuestradeOrderManager',
    
    # OANDA
    'OANDACanada',
    'ForexSignal',
    'SpreadAnalysis',
    'MarginInfo',
]

# Version info
__version__ = '1.0.0'
__author__ = 'AI News Trading Platform'