"""
Polymarket Trading Strategies

This module provides various trading strategies for prediction markets:
- News-based sentiment strategies
- Market making strategies
- Arbitrage opportunities
- Momentum-based strategies
- Ensemble strategies that combine multiple approaches
"""

from .base import (
    PolymarketStrategy,
    StrategyConfig,
    TradingSignal,
    SignalStrength,
    StrategyError,
    StrategyMetrics,
)

from .news_sentiment import NewsSentimentStrategy
from .market_maker import MarketMakerStrategy
from .arbitrage import ArbitrageStrategy
from .momentum import MomentumStrategy
from .ensemble import EnsembleStrategy

__all__ = [
    # Base classes
    "PolymarketStrategy",
    "StrategyConfig",
    "TradingSignal",
    "SignalStrength", 
    "StrategyError",
    "StrategyMetrics",
    
    # Strategy implementations
    "NewsSentimentStrategy",
    "MarketMakerStrategy",
    "ArbitrageStrategy",
    "MomentumStrategy",
    "EnsembleStrategy",
]