"""
Market Data Simulator - Real-time market data simulation
"""

import asyncio
import logging
import numpy as np
from typing import Dict, Any, List, Generator
from datetime import datetime
import time

class MarketDataSimulator:
    """
    High-frequency market data simulator
    Generates realistic market data streams for testing
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
        self.base_prices = {symbol: np.random.uniform(100, 500) for symbol in self.symbols}
        self.volatility = {symbol: np.random.uniform(0.001, 0.01) for symbol in self.symbols}
        
    async def initialize(self):
        """Initialize market data simulator"""
        self.logger.info("Market Data Simulator initialized")
        
    async def generate_stream(self, duration_seconds: int, frequency_hz: int) -> List[Dict[str, Any]]:
        """Generate market data stream"""
        total_updates = duration_seconds * frequency_hz
        interval = 1.0 / frequency_hz
        
        data_stream = []
        
        self.logger.info(f"Generating {total_updates} market updates at {frequency_hz}Hz")
        
        for i in range(total_updates):
            # Generate market update
            update = await self._generate_market_update()
            data_stream.append(update)
            
            # Simulate real-time streaming
            if i < total_updates - 1:  # Don't sleep on last iteration
                await asyncio.sleep(interval / 1000)  # Scale down for testing
                
        return data_stream
        
    async def _generate_market_update(self) -> Dict[str, Any]:
        """Generate single market data update"""
        symbol = np.random.choice(self.symbols)
        
        # Price movement using geometric Brownian motion
        current_price = self.base_prices[symbol]
        vol = self.volatility[symbol]
        
        # Random walk
        price_change = np.random.normal(0, vol) * current_price
        new_price = max(current_price + price_change, 0.01)  # Prevent negative prices
        
        self.base_prices[symbol] = new_price
        
        # Generate order book data
        bid_price = new_price * (1 - np.random.uniform(0.0001, 0.001))
        ask_price = new_price * (1 + np.random.uniform(0.0001, 0.001))
        
        update = {
            'timestamp': time.time(),
            'symbol': symbol,
            'prices': [new_price],
            'bid': bid_price,
            'ask': ask_price,
            'volume': np.random.randint(100, 10000),
            'last_trade_size': np.random.randint(1, 1000)
        }
        
        return update
        
    async def get_current_prices(self) -> Dict[str, float]:
        """Get current prices for all symbols"""
        return self.base_prices.copy()