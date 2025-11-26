"""Trading strategies for news-based trading."""

from .swing_trading import SwingTradingStrategy
from .momentum_trading import MomentumTradingStrategy
from .mirror_trading import MirrorTradingStrategy

__all__ = [
    "SwingTradingStrategy",
    "MomentumTradingStrategy",
    "MirrorTradingStrategy",
]