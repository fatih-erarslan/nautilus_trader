"""
Market simulation scenarios for benchmarking.
"""
from .bull_market import BullMarketScenario
from .bear_market import BearMarketScenario
from .high_volatility import HighVolatilityScenario
from .flash_crash import FlashCrashScenario

__all__ = [
    "BullMarketScenario",
    "BearMarketScenario", 
    "HighVolatilityScenario",
    "FlashCrashScenario"
]