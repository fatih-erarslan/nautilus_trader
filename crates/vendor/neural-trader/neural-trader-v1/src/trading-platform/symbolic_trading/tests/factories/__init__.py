"""Test factories for AI News Trading Platform."""

from .market_data import MarketDataFactory, CandleDataFactory, OrderBookFactory
from .trading import TradeFactory, OrderFactory, PositionFactory
from .news import NewsArticleFactory, NewsSourceFactory, SentimentFactory
from .api_mocks import (
    MockCryptoAPIFactory,
    MockLLMClientFactory,
    MockNewsAPIFactory,
    MockMarketDataAPIFactory,
    MockBrokerAPIFactory
)

__all__ = [
    # Market Data
    'MarketDataFactory',
    'CandleDataFactory',
    'OrderBookFactory',
    
    # Trading
    'TradeFactory',
    'OrderFactory',
    'PositionFactory',
    
    # News
    'NewsArticleFactory',
    'NewsSourceFactory',
    'SentimentFactory',
    
    # API Mocks
    'MockCryptoAPIFactory',
    'MockLLMClientFactory',
    'MockNewsAPIFactory',
    'MockMarketDataAPIFactory',
    'MockBrokerAPIFactory'
]