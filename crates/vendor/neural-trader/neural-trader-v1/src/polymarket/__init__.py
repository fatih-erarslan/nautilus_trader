"""
Polymarket integration for AI News Trading Platform
Provides prediction market functionality with GPU-accelerated analytics
"""

# Import MCP tools for backwards compatibility
from .mcp_tools import (
    get_prediction_markets,
    analyze_market_sentiment,
    get_market_orderbook,
    place_prediction_order,
    get_prediction_positions,
    calculate_expected_value,
    GPU_AVAILABLE
)

# Import API clients and models for TDD implementation
try:
    from .api.clob_client import CLOBClient
    from .models import Market, Order, OrderBook, OrderSide, OrderStatus, OrderType, MarketStatus
    from .utils import PolymarketConfig
    
    __all__ = [
        # MCP Tools (existing)
        'get_prediction_markets',
        'analyze_market_sentiment',
        'get_market_orderbook',
        'place_prediction_order',
        'get_prediction_positions',
        'calculate_expected_value',
        'GPU_AVAILABLE',
        
        # New API clients and models
        'CLOBClient',
        'Market',
        'Order',
        'OrderBook',
        'OrderSide',
        'OrderStatus',
        'OrderType',
        'MarketStatus',
        'PolymarketConfig',
    ]
except ImportError:
    # Fallback if new modules are not fully implemented yet
    __all__ = [
        'get_prediction_markets',
        'analyze_market_sentiment',
        'get_market_orderbook',
        'place_prediction_order',
        'get_prediction_positions',
        'calculate_expected_value',
        'GPU_AVAILABLE'
    ]

__version__ = "1.0.0"