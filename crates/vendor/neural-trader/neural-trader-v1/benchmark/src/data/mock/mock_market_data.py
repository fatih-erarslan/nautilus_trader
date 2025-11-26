"""Mock market data generator for realistic testing.

Generates realistic stock, crypto, and forex data with configurable
volatility, trends, and market events.
"""

import asyncio
import random
import time
import math
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
import numpy as np

from ..realtime_feed import DataUpdate, DataSource

class MarketCondition(Enum):
    """Market conditions for simulation."""
    NORMAL = auto()
    VOLATILE = auto()
    TRENDING_UP = auto()
    TRENDING_DOWN = auto()
    FLASH_CRASH = auto()
    SQUEEZE = auto()


@dataclass
class MockSymbolConfig:
    """Configuration for a mock symbol."""
    symbol: str
    base_price: float
    volatility: float = 0.02  # 2% daily volatility
    drift: float = 0.0001  # Slight upward drift
    min_price: float = 0.01
    max_price: float = None
    tick_size: float = 0.01
    market_hours: Optional[tuple] = None  # (open_hour, close_hour) in UTC
    asset_type: str = "stock"  # stock, crypto, forex, index


@dataclass
class MockMarketConfig:
    """Configuration for mock market data generator."""
    symbols: List[MockSymbolConfig]
    update_rate_hz: float = 100.0  # Updates per second
    enable_spreads: bool = True
    spread_basis_points: float = 10.0  # 0.1% spread
    enable_market_hours: bool = True
    enable_random_gaps: bool = True
    enable_fat_tails: bool = True  # Realistic extreme events
    correlation_matrix: Optional[np.ndarray] = None
    
    
class PriceGenerator:
    """Generate realistic price movements."""
    
    def __init__(self, config: MockSymbolConfig):
        self.config = config
        self.current_price = config.base_price
        self.last_update = time.time()
        
        # State for various price models
        self._momentum = 0.0
        self._mean_reversion_target = config.base_price
        self._volatility_regime = 1.0
        
    def generate_next_price(self, dt: float, market_condition: MarketCondition) -> float:
        """Generate next price using geometric Brownian motion with enhancements."""
        # Base random walk
        random_shock = np.random.normal(0, 1) * math.sqrt(dt)
        
        # Adjust for market conditions
        condition_multiplier = self._get_condition_multiplier(market_condition)
        
        # Add momentum (creates more realistic trends)
        self._momentum = 0.9 * self._momentum + 0.1 * random_shock
        
        # Calculate return
        drift_component = self.config.drift * dt
        volatility_component = self.config.volatility * condition_multiplier * (random_shock + 0.3 * self._momentum)
        
        # Add jump component for fat tails (rare large moves)
        jump = 0
        if random.random() < 0.001 * dt:  # Rare jumps
            jump = np.random.normal(0, self.config.volatility * 5)
        
        # Calculate new price
        price_return = drift_component + volatility_component * math.sqrt(dt) + jump
        new_price = self.current_price * (1 + price_return)
        
        # Apply constraints
        if self.config.min_price:
            new_price = max(new_price, self.config.min_price)
        if self.config.max_price:
            new_price = min(new_price, self.config.max_price)
        
        # Round to tick size
        new_price = round(new_price / self.config.tick_size) * self.config.tick_size
        
        self.current_price = new_price
        return new_price
    
    def _get_condition_multiplier(self, condition: MarketCondition) -> float:
        """Get volatility multiplier based on market condition."""
        multipliers = {
            MarketCondition.NORMAL: 1.0,
            MarketCondition.VOLATILE: 3.0,
            MarketCondition.TRENDING_UP: 1.5,
            MarketCondition.TRENDING_DOWN: 1.5,
            MarketCondition.FLASH_CRASH: 10.0,
            MarketCondition.SQUEEZE: 5.0,
        }
        return multipliers.get(condition, 1.0)
    
    def apply_gap(self, gap_percent: float):
        """Apply a price gap (for market open, news events, etc.)."""
        self.current_price *= (1 + gap_percent / 100)
        
        
class MockMarketDataGenerator:
    """Generate realistic mock market data."""
    
    def __init__(self, config: MockMarketConfig):
        self.config = config
        self._running = False
        self._update_task = None
        self._callbacks: List[Callable] = []
        
        # Price generators for each symbol
        self._price_generators: Dict[str, PriceGenerator] = {}
        for symbol_config in config.symbols:
            self._price_generators[symbol_config.symbol] = PriceGenerator(symbol_config)
        
        # Market state
        self._market_condition = MarketCondition.NORMAL
        self._condition_duration = 0
        self._last_condition_change = time.time()
        
        # Order book simulation
        self._order_books: Dict[str, Dict] = {}
        
        # Metrics
        self._total_updates = 0
        self._start_time = time.time()
        
    async def start(self):
        """Start generating mock data."""
        if self._running:
            return
        
        self._running = True
        self._update_task = asyncio.create_task(self._generation_loop())
        
    async def stop(self):
        """Stop generating mock data."""
        self._running = False
        
        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass
    
    def add_callback(self, callback: Callable[[DataUpdate], None]):
        """Add callback for data updates."""
        self._callbacks.append(callback)
    
    async def _generation_loop(self):
        """Main generation loop."""
        update_interval = 1.0 / self.config.update_rate_hz
        last_update = time.time()
        
        while self._running:
            try:
                current_time = time.time()
                dt = current_time - last_update
                
                # Update market condition periodically
                self._update_market_condition()
                
                # Generate updates for all symbols
                tasks = []
                for symbol_config in self.config.symbols:
                    if self._should_update_symbol(symbol_config, current_time):
                        task = self._generate_symbol_update(symbol_config, dt)
                        tasks.append(task)
                
                # Process all updates in parallel
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
                
                last_update = current_time
                
                # Sleep to maintain update rate
                sleep_time = update_interval - (time.time() - current_time)
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Generation loop error: {e}")
                await asyncio.sleep(0.1)
    
    def _should_update_symbol(self, symbol_config: MockSymbolConfig, current_time: float) -> bool:
        """Check if symbol should be updated based on market hours."""
        if not self.config.enable_market_hours:
            return True
        
        if symbol_config.asset_type == "crypto":
            return True  # 24/7 market
        
        if symbol_config.market_hours:
            current_hour = datetime.utcnow().hour
            open_hour, close_hour = symbol_config.market_hours
            
            if open_hour < close_hour:
                return open_hour <= current_hour < close_hour
            else:  # Handles overnight sessions
                return current_hour >= open_hour or current_hour < close_hour
        
        return True
    
    async def _generate_symbol_update(self, symbol_config: MockSymbolConfig, dt: float):
        """Generate update for a single symbol."""
        try:
            generator = self._price_generators[symbol_config.symbol]
            
            # Generate new price
            new_price = generator.generate_next_price(dt, self._market_condition)
            
            # Generate spread if enabled
            bid = new_price
            ask = new_price
            if self.config.enable_spreads:
                spread = new_price * self.config.spread_basis_points / 10000
                bid = new_price - spread / 2
                ask = new_price + spread / 2
            
            # Generate volume (more volume during volatile conditions)
            base_volume = random.randint(100, 10000)
            condition_volume_mult = {
                MarketCondition.NORMAL: 1.0,
                MarketCondition.VOLATILE: 3.0,
                MarketCondition.FLASH_CRASH: 10.0,
                MarketCondition.SQUEEZE: 5.0,
            }.get(self._market_condition, 1.0)
            
            volume = int(base_volume * condition_volume_mult * random.uniform(0.5, 2.0))
            
            # Create metadata
            metadata = {
                "source": "mock_generator",
                "bid": round(bid, symbol_config.tick_size),
                "ask": round(ask, symbol_config.tick_size),
                "bid_size": random.randint(100, 5000),
                "ask_size": random.randint(100, 5000),
                "volume": volume,
                "market_condition": self._market_condition.name,
                "asset_type": symbol_config.asset_type,
            }
            
            # Create data update
            update = DataUpdate(
                symbol=symbol_config.symbol,
                price=new_price,
                timestamp=time.time(),
                source=DataSource.WEBSOCKET,  # Simulate WebSocket feed
                metadata=metadata
            )
            
            self._total_updates += 1
            
            # Notify callbacks
            for callback in self._callbacks:
                try:
                    await callback(update)
                except Exception as e:
                    print(f"Callback error: {e}")
                    
        except Exception as e:
            print(f"Error generating update for {symbol_config.symbol}: {e}")
    
    def _update_market_condition(self):
        """Randomly change market conditions to simulate different scenarios."""
        current_time = time.time()
        
        # Change condition every 30-300 seconds
        if current_time - self._last_condition_change > self._condition_duration:
            # Random new condition with weights
            conditions_weights = [
                (MarketCondition.NORMAL, 0.7),
                (MarketCondition.VOLATILE, 0.15),
                (MarketCondition.TRENDING_UP, 0.05),
                (MarketCondition.TRENDING_DOWN, 0.05),
                (MarketCondition.FLASH_CRASH, 0.02),
                (MarketCondition.SQUEEZE, 0.03),
            ]
            
            conditions, weights = zip(*conditions_weights)
            self._market_condition = random.choices(conditions, weights=weights)[0]
            
            # Set duration for this condition
            if self._market_condition == MarketCondition.FLASH_CRASH:
                self._condition_duration = random.uniform(5, 30)  # Short duration
            else:
                self._condition_duration = random.uniform(30, 300)
            
            self._last_condition_change = current_time
    
    def trigger_event(self, event_type: str, symbols: Optional[List[str]] = None):
        """Trigger specific market events for testing."""
        if event_type == "flash_crash":
            self._market_condition = MarketCondition.FLASH_CRASH
            self._condition_duration = 10
            self._last_condition_change = time.time()
            
            # Apply immediate gap to affected symbols
            affected_symbols = symbols or [s.symbol for s in self.config.symbols]
            for symbol in affected_symbols:
                if symbol in self._price_generators:
                    self._price_generators[symbol].apply_gap(random.uniform(-5, -15))
        
        elif event_type == "gap_up":
            affected_symbols = symbols or [s.symbol for s in self.config.symbols]
            for symbol in affected_symbols:
                if symbol in self._price_generators:
                    self._price_generators[symbol].apply_gap(random.uniform(1, 5))
        
        elif event_type == "halt":
            # Simulate trading halt by temporarily removing from updates
            pass
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get generator metrics."""
        runtime = time.time() - self._start_time
        updates_per_second = self._total_updates / runtime if runtime > 0 else 0
        
        return {
            "total_updates": self._total_updates,
            "updates_per_second": updates_per_second,
            "runtime_seconds": runtime,
            "active_symbols": len(self._price_generators),
            "current_market_condition": self._market_condition.name,
            "target_update_rate": self.config.update_rate_hz,
        }
    
    def get_current_prices(self) -> Dict[str, float]:
        """Get current prices for all symbols."""
        return {
            symbol: generator.current_price
            for symbol, generator in self._price_generators.items()
        }


def create_default_mock_config() -> MockMarketConfig:
    """Create default mock configuration with common symbols."""
    symbols = [
        # Stocks
        MockSymbolConfig("AAPL", 150.0, volatility=0.02, asset_type="stock"),
        MockSymbolConfig("GOOGL", 2800.0, volatility=0.025, asset_type="stock"),
        MockSymbolConfig("MSFT", 300.0, volatility=0.018, asset_type="stock"),
        MockSymbolConfig("TSLA", 250.0, volatility=0.04, asset_type="stock"),
        MockSymbolConfig("SPY", 450.0, volatility=0.015, asset_type="index"),
        
        # Crypto
        MockSymbolConfig("BTC-USD", 45000.0, volatility=0.04, tick_size=1.0, asset_type="crypto"),
        MockSymbolConfig("ETH-USD", 3000.0, volatility=0.05, tick_size=0.1, asset_type="crypto"),
        
        # Forex
        MockSymbolConfig("EUR-USD", 1.08, volatility=0.01, tick_size=0.0001, asset_type="forex"),
        MockSymbolConfig("GBP-USD", 1.26, volatility=0.012, tick_size=0.0001, asset_type="forex"),
    ]
    
    return MockMarketConfig(
        symbols=symbols,
        update_rate_hz=100.0,
        enable_spreads=True,
        enable_market_hours=True,
        enable_random_gaps=True,
        enable_fat_tails=True,
    )