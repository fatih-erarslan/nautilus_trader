"""
Complete market simulation engine combining order books, price generation, and market participants.
"""
import asyncio
import time
import numpy as np
import psutil
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import random

from .order_book import OrderBook, Order, OrderSide, OrderType, Trade
from .price_generator import PriceGenerator, PriceGeneratorConfig, MarketRegime


@dataclass
class SimulationConfig:
    """Configuration for market simulation."""
    symbols: List[str]
    duration: float  # Simulation duration in seconds
    tick_rate: int = 1000  # Ticks per second
    initial_prices: Optional[Dict[str, float]] = None
    participant_counts: Optional[Dict[str, int]] = None
    correlation_matrix: Optional[np.ndarray] = None
    
    def __post_init__(self):
        if self.initial_prices is None:
            self.initial_prices = {symbol: 100.0 for symbol in self.symbols}
        if self.participant_counts is None:
            self.participant_counts = {
                "market_maker": 2,
                "random_trader": 10,
                "momentum_trader": 5
            }


@dataclass
class MarketStats:
    """Statistics for a single market/symbol."""
    symbol: str
    total_volume: int
    total_trades: int
    average_spread: float
    price_volatility: float
    vwap: float  # Volume-weighted average price
    high_price: float = 0.0
    low_price: float = float('inf')
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "total_volume": self.total_volume,
            "total_trades": self.total_trades,
            "average_spread": self.average_spread,
            "price_volatility": self.price_volatility,
            "vwap": self.vwap,
            "high_price": self.high_price,
            "low_price": self.low_price
        }


@dataclass
class SimulationResult:
    """Results from a simulation run."""
    duration: float
    total_ticks: int
    total_trades: int
    market_stats: Dict[str, MarketStats]
    price_series: Dict[str, List[float]] = field(default_factory=dict)
    events_processed: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for export."""
        return {
            "duration": self.duration,
            "total_ticks": self.total_ticks,
            "total_trades": self.total_trades,
            "market_stats": {
                symbol: stats.to_dict() 
                for symbol, stats in self.market_stats.items()
            },
            "price_series": self.price_series,
            "events_processed": self.events_processed
        }


class MarketParticipant:
    """Base class for market participants."""
    
    def __init__(self, participant_id: str, capital: float = 1000000):
        self.participant_id = participant_id
        self.capital = capital
        self.positions: Dict[str, int] = defaultdict(int)
        self.orders_sent = 0
        self.trades_executed = 0
    
    def generate_orders(self, symbol: str, current_price: float, 
                       order_book_snapshot: Any) -> List[Order]:
        """Generate orders based on strategy. Override in subclasses."""
        raise NotImplementedError
    
    def on_trade(self, trade: Trade):
        """Handle trade execution notification."""
        self.trades_executed += 1


class MarketMaker(MarketParticipant):
    """Market maker providing liquidity."""
    
    def __init__(self, participant_id: str, symbol: str, spread: float = 0.001):
        super().__init__(participant_id)
        self.symbol = symbol
        self.spread = spread
        self.order_size = 100
        self.max_position = 10000
    
    def generate_orders(self, symbol: str, current_price: float,
                       order_book_snapshot: Any) -> List[Order]:
        """Generate bid and ask orders around current price."""
        if symbol != self.symbol:
            return []
        
        orders = []
        position = self.positions[symbol]
        
        # Adjust spread based on position
        position_adjustment = abs(position) / self.max_position * 0.001
        
        # Generate buy order if not too long
        if position < self.max_position:
            buy_price = current_price * (1 - self.spread - position_adjustment)
            buy_order = Order(
                order_id=f"{self.participant_id}_BUY_{self.orders_sent}",
                side=OrderSide.BUY,
                price=round(buy_price, 2),
                quantity=self.order_size,
                order_type=OrderType.LIMIT,
                timestamp=time.time()
            )
            orders.append(buy_order)
        
        # Generate sell order if not too short
        if position > -self.max_position:
            sell_price = current_price * (1 + self.spread + position_adjustment)
            sell_order = Order(
                order_id=f"{self.participant_id}_SELL_{self.orders_sent + 1}",
                side=OrderSide.SELL,
                price=round(sell_price, 2),
                quantity=self.order_size,
                order_type=OrderType.LIMIT,
                timestamp=time.time()
            )
            orders.append(sell_order)
        
        self.orders_sent += len(orders)
        return orders


class RandomTrader(MarketParticipant):
    """Random trader for noise/liquidity."""
    
    def __init__(self, participant_id: str, capital: float = 50000):
        super().__init__(participant_id, capital)
        self.trade_probability = 0.1
        self.max_order_size = 500
    
    def generate_orders(self, symbol: str, current_price: float,
                       order_book_snapshot: Any) -> List[Order]:
        """Generate random orders."""
        if random.random() > self.trade_probability:
            return []
        
        # Random side
        side = random.choice([OrderSide.BUY, OrderSide.SELL])
        
        # Random order type
        if random.random() < 0.7:  # 70% limit orders
            # Limit order with random price offset
            offset = random.uniform(-0.02, 0.02)  # ±2%
            price = current_price * (1 + offset)
            order_type = OrderType.LIMIT
        else:
            # Market order
            price = None
            order_type = OrderType.MARKET
        
        # Random quantity
        quantity = random.randint(10, self.max_order_size)
        
        order = Order(
            order_id=f"{self.participant_id}_{self.orders_sent}",
            side=side,
            price=round(price, 2) if price else None,
            quantity=quantity,
            order_type=order_type,
            timestamp=time.time()
        )
        
        self.orders_sent += 1
        return [order]


class MomentumTrader(MarketParticipant):
    """Momentum-based trader."""
    
    def __init__(self, participant_id: str, capital: float = 100000):
        super().__init__(participant_id, capital)
        self.lookback_period = 20
        self.price_history: Dict[str, List[float]] = defaultdict(list)
        self.momentum_threshold = 0.001
        self.position_size = 200
    
    def calculate_signal(self, symbol: str, prices: List[float]) -> float:
        """Calculate momentum signal."""
        if len(prices) < 2:
            return 0.0
        
        # Simple momentum: recent return
        returns = []
        for i in range(1, min(len(prices), self.lookback_period)):
            ret = (prices[-i] - prices[-i-1]) / prices[-i-1]
            returns.append(ret)
        
        if not returns:
            return 0.0
        
        # Average return as signal
        return sum(returns) / len(returns)
    
    def generate_orders(self, symbol: str, current_price: float,
                       order_book_snapshot: Any) -> List[Order]:
        """Generate orders based on momentum."""
        # Update price history
        self.price_history[symbol].append(current_price)
        if len(self.price_history[symbol]) > self.lookback_period * 2:
            self.price_history[symbol] = self.price_history[symbol][-self.lookback_period:]
        
        # Calculate signal
        signal = self.calculate_signal(symbol, self.price_history[symbol])
        
        # Generate order if signal is strong
        if abs(signal) < self.momentum_threshold:
            return []
        
        # Determine side based on signal
        side = OrderSide.BUY if signal > 0 else OrderSide.SELL
        
        # Use aggressive pricing for momentum
        if side == OrderSide.BUY:
            price = current_price * 1.001  # Pay up to get filled
        else:
            price = current_price * 0.999  # Sell below market
        
        order = Order(
            order_id=f"{self.participant_id}_{symbol}_{self.orders_sent}",
            side=side,
            price=round(price, 2),
            quantity=self.position_size,
            order_type=OrderType.LIMIT,
            timestamp=time.time()
        )
        
        self.orders_sent += 1
        return [order]


class MarketSimulator:
    """Main market simulation engine."""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.symbols = config.symbols
        
        # Initialize order books
        self.order_books: Dict[str, OrderBook] = {
            symbol: OrderBook(symbol, tick_size=0.01)
            for symbol in self.symbols
        }
        
        # Initialize price generators
        self.price_generators: Dict[str, PriceGenerator] = {}
        self._init_price_generators()
        
        # Initialize participants
        self.participants: List[MarketParticipant] = []
        self._init_participants()
        
        # Tracking
        self.tick_count = 0
        self.total_trades = 0
        self.price_history: Dict[str, List[float]] = defaultdict(list)
        self.spread_history: Dict[str, List[float]] = defaultdict(list)
        
        # Circuit breakers
        self.circuit_breaker_enabled = False
        self.circuit_breaker_threshold = 0.07
        
        # Scheduled events
        self.scheduled_events: List[Tuple[float, Dict[str, Any]]] = []
    
    def _init_price_generators(self):
        """Initialize price generators with correlation if specified."""
        base_config = PriceGeneratorConfig(
            initial_price=100.0,
            volatility=0.20,
            drift=0.05,
            tick_size=0.01,
            update_frequency=self.config.tick_rate
        )
        
        if self.config.correlation_matrix is not None:
            # Create correlated generators
            self.price_generators = PriceGenerator.create_correlated(
                symbols=self.symbols,
                correlation_matrix=self.config.correlation_matrix,
                base_config=base_config
            )
        else:
            # Create independent generators
            for symbol in self.symbols:
                config = PriceGeneratorConfig(
                    initial_price=self.config.initial_prices[symbol],
                    volatility=0.20,
                    drift=0.05,
                    tick_size=0.01,
                    update_frequency=self.config.tick_rate
                )
                self.price_generators[symbol] = PriceGenerator(symbol, config)
    
    def _init_participants(self):
        """Initialize market participants."""
        # Market makers (per symbol)
        for symbol in self.symbols:
            for i in range(self.config.participant_counts.get("market_maker", 2)):
                mm = MarketMaker(f"MM_{symbol}_{i}", symbol, spread=0.001)
                self.participants.append(mm)
        
        # Random traders
        for i in range(self.config.participant_counts.get("random_trader", 10)):
            trader = RandomTrader(f"RANDOM_{i}")
            self.participants.append(trader)
        
        # Momentum traders
        for i in range(self.config.participant_counts.get("momentum_trader", 5)):
            trader = MomentumTrader(f"MOMENTUM_{i}")
            self.participants.append(trader)
    
    async def run(self) -> SimulationResult:
        """Run the simulation."""
        start_time = time.time()
        tick_interval = 1.0 / self.config.tick_rate
        
        # Main simulation loop
        while time.time() - start_time < self.config.duration:
            await self._simulation_tick()
            await asyncio.sleep(tick_interval)
        
        # Calculate final statistics
        end_time = time.time()
        duration = end_time - start_time
        
        market_stats = {}
        for symbol in self.symbols:
            stats = self.calculate_market_stats(symbol)
            market_stats[symbol] = stats
        
        return SimulationResult(
            duration=duration,
            total_ticks=self.tick_count,
            total_trades=self.total_trades,
            market_stats=market_stats,
            price_series=dict(self.price_history),
            events_processed=[e[1] for e in self.scheduled_events if e[0] <= duration]
        )
    
    async def _simulation_tick(self):
        """Execute one simulation tick."""
        self.tick_count += 1
        current_time = time.time()
        
        # Process scheduled events
        self._process_events(current_time)
        
        # Update prices
        if self.config.correlation_matrix is not None:
            price_updates = PriceGenerator.generate_correlated_updates(self.price_generators)
        else:
            price_updates = [gen.generate_update() for gen in self.price_generators.values()]
        
        # Process each symbol
        for update in price_updates:
            symbol = update.symbol
            
            # Record price
            self.price_history[symbol].append(update.price)
            
            # Get order book snapshot
            snapshot = self.order_books[symbol].get_snapshot()
            
            # Record spread
            if snapshot.spread is not None:
                self.spread_history[symbol].append(snapshot.spread)
            
            # Generate participant orders
            for participant in self.participants:
                orders = participant.generate_orders(symbol, update.price, snapshot)
                
                # Submit orders to order book
                for order in orders:
                    trades = self.order_books[symbol].add_order(order)
                    self.total_trades += len(trades)
                    
                    # Notify participant of trades
                    for trade in trades:
                        participant.on_trade(trade)
    
    def calculate_market_stats(self, symbol: str) -> MarketStats:
        """Calculate statistics for a symbol."""
        prices = self.price_history[symbol]
        spreads = self.spread_history[symbol]
        order_book = self.order_books[symbol]
        
        # Calculate volume and trades
        total_volume = sum(trade.quantity for trade in self._get_all_trades(symbol))
        total_trades = order_book.total_trades
        
        # Average spread
        avg_spread = np.mean(spreads) if spreads else 0.0
        
        # Price volatility
        if len(prices) > 1:
            returns = np.diff(prices) / prices[:-1]
            volatility = np.std(returns) * np.sqrt(252 * self.config.tick_rate)
        else:
            volatility = 0.0
        
        # VWAP calculation would require trade data
        vwap = np.mean(prices) if prices else 0.0
        
        # High/Low
        high = max(prices) if prices else 0.0
        low = min(prices) if prices else float('inf')
        
        return MarketStats(
            symbol=symbol,
            total_volume=total_volume,
            total_trades=total_trades,
            average_spread=avg_spread,
            price_volatility=volatility,
            vwap=vwap,
            high_price=high,
            low_price=low
        )
    
    def _get_all_trades(self, symbol: str) -> List[Trade]:
        """Get all trades for a symbol (mock implementation)."""
        # In real implementation, order book would track trades
        return []
    
    def enable_circuit_breakers(self, threshold: float):
        """Enable circuit breakers."""
        self.circuit_breaker_enabled = True
        self.circuit_breaker_threshold = threshold
        
        # Configure price generators
        for generator in self.price_generators.values():
            generator.set_circuit_breaker(threshold)
    
    def schedule_event(self, time: float, event_type: str, data: Dict[str, Any]):
        """Schedule a market event."""
        self.scheduled_events.append((time, {"event_type": event_type, **data}))
        self.scheduled_events.sort(key=lambda x: x[0])
    
    def _process_events(self, current_time: float):
        """Process scheduled events."""
        # This is a placeholder - in real implementation would process events
        pass