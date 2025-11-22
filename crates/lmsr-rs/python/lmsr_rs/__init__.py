"""
LMSR-RS: High-Performance Logarithmic Market Scoring Rule

A fast, numerically stable implementation of LMSR for prediction markets
and market making applications.
"""

from .lmsr_rs import (
    # Core classes
    LMSRMarket,
    MarketStatistics, 
    Position,
    MarketState,
    MarketSimulation,
    LMSRBenchmark,
    
    # Utility functions
    py_calculate_price as calculate_price,
    py_calculate_cost as calculate_cost,
)

__version__ = "0.1.0"
__author__ = "Financial Systems Team"
__email__ = "dev@company.com"

__all__ = [
    # Classes
    "LMSRMarket",
    "MarketStatistics",
    "Position", 
    "MarketState",
    "MarketSimulation",
    "LMSRBenchmark",
    
    # Functions
    "calculate_price",
    "calculate_cost",
]

# Package metadata
__doc__ = """
LMSR-RS: High-Performance Logarithmic Market Scoring Rule

Key Features:
- 100-200x faster than pure Python implementations
- Numerically stable for extreme market conditions  
- Thread-safe concurrent market access
- Real-time market state management
- Comprehensive position tracking
- Financial-grade precision and safety

Quick Start:
```python
import lmsr_rs

# Create a binary market
market = lmsr_rs.LMSRMarket.create_binary(
    "Will it rain tomorrow?",
    "Weather prediction market", 
    liquidity=1000.0
)

# Execute a trade
cost = market.trade("trader1", [10.0, 0.0])
print(f"Trade cost: ${cost:.2f}")

# Get current prices  
prices = market.get_prices()
print(f"Prices: {prices}")
```

For more examples and documentation, visit:
https://github.com/company/lmsr-rs
"""