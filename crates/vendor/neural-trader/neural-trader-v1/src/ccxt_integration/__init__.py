"""
CCXT Integration Module for AI News Trader

This module provides unified cryptocurrency exchange trading through CCXT library,
supporting 100+ exchanges with a single interface.
"""

from .interfaces.ccxt_interface import CCXTInterface
from .core.client_manager import ClientManager
from .core.exchange_registry import ExchangeRegistry
from .streaming.websocket_manager import WebSocketManager
from .execution.order_router import OrderRouter

__version__ = "1.0.0"
__all__ = [
    "CCXTInterface",
    "ClientManager",
    "ExchangeRegistry",
    "WebSocketManager",
    "OrderRouter",
]