"""
Crypto Trading Strategies for Beefy Finance

This module provides various trading strategies for yield optimization across
multiple blockchain networks and DeFi protocols.
"""

from .base_strategy import BaseStrategy
from .yield_chaser import YieldChaserStrategy
from .stable_farmer import StableFarmerStrategy
from .risk_balanced import RiskBalancedStrategy
from .news_driven import NewsDrivenStrategy
from .portfolio_optimizer import PortfolioOptimizer
from .risk_calculator import RiskCalculator

__all__ = [
    'BaseStrategy',
    'YieldChaserStrategy',
    'StableFarmerStrategy',
    'RiskBalancedStrategy',
    'NewsDrivenStrategy',
    'PortfolioOptimizer',
    'RiskCalculator'
]