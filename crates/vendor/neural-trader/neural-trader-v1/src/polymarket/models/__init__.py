"""
Polymarket Data Models

This module provides data models for Polymarket entities including markets,
orders, trades, events, and analytics. All models use dataclasses with 
validation and type hints for robust data handling.
"""

from .market import (
    Market,
    OrderBook,
    PricePoint,
    MarketMetadata,
    LiquidityMetrics,
    MarketStatus,
)

from .order import (
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Trade,
    TradeStatus,
)

from .event import (
    Event,
    Outcome,
    EventStatus,
    EventCategory,
    Resolution,
)

from .position import (
    Position,
    Portfolio,
    PositionStatus,
    PnLMetrics,
)

from .analytics import (
    TradingSignal,
    SignalStrength,
    MarketAnalysis,
    SentimentScore,
    RiskMetrics,
    PerformanceMetrics,
)

__all__ = [
    # Market models
    "Market",
    "OrderBook",
    "PricePoint",
    "MarketMetadata",
    "LiquidityMetrics",
    "MarketStatus",
    
    # Order and trade models
    "Order",
    "OrderSide",
    "OrderStatus", 
    "OrderType",
    "Trade",
    "TradeStatus",
    
    # Event models
    "Event",
    "Outcome",
    "EventStatus",
    "EventCategory",
    "Resolution",
    
    # Position models
    "Position",
    "Portfolio",
    "PositionStatus",
    "PnLMetrics",
    
    # Analytics models
    "TradingSignal",
    "SignalStrength",
    "MarketAnalysis",
    "SentimentScore",
    "RiskMetrics",
    "PerformanceMetrics",
]