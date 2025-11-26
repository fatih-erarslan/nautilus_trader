"""
Market data factories for testing.

This module provides factory classes for generating test market data
including OHLCV data, order books, and other market-related data structures.
"""

import random
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import factory
from factory import Factory, LazyAttribute, LazyFunction, SubFactory, Trait
from faker import Faker

fake = Faker()


class MarketDataFactory(Factory):
    """Factory for creating market data snapshots."""
    
    class Meta:
        model = dict
    
    symbol = factory.LazyFunction(lambda: random.choice(["BTC_USDT", "ETH_USDT", "XRP_USDT", "ADA_USDT"]))
    timestamp = factory.LazyFunction(lambda: datetime.now(timezone.utc))
    price = factory.LazyAttribute(lambda obj: {
        "BTC_USDT": random.uniform(40000, 60000),
        "ETH_USDT": random.uniform(2500, 4000),
        "XRP_USDT": random.uniform(0.5, 1.5),
        "ADA_USDT": random.uniform(0.3, 0.8)
    }.get(obj.symbol, random.uniform(1, 100)))
    
    bid = factory.LazyAttribute(lambda obj: round(obj.price * 0.9995, 4))
    ask = factory.LazyAttribute(lambda obj: round(obj.price * 1.0005, 4))
    spread = factory.LazyAttribute(lambda obj: round(obj.ask - obj.bid, 4))
    
    volume_24h = factory.LazyFunction(lambda: round(random.uniform(1000000, 10000000), 2))
    volume_quote_24h = factory.LazyAttribute(lambda obj: round(obj.volume_24h * obj.price, 2))
    
    high_24h = factory.LazyAttribute(lambda obj: round(obj.price * random.uniform(1.02, 1.08), 2))
    low_24h = factory.LazyAttribute(lambda obj: round(obj.price * random.uniform(0.92, 0.98), 2))
    open_24h = factory.LazyAttribute(lambda obj: round(obj.price * random.uniform(0.95, 1.05), 2))
    
    price_change_24h = factory.LazyAttribute(lambda obj: round(obj.price - obj.open_24h, 2))
    price_change_percent_24h = factory.LazyAttribute(
        lambda obj: round((obj.price_change_24h / obj.open_24h) * 100, 2)
    )
    
    market_cap = factory.LazyFunction(lambda: random.uniform(100000000, 1000000000000))
    circulating_supply = factory.LazyFunction(lambda: random.uniform(1000000, 100000000000))
    
    class Params:
        """Trait parameters for different market conditions."""
        bullish = Trait(
            price_change_percent_24h=factory.LazyFunction(lambda: random.uniform(2, 10)),
            volume_24h=factory.LazyFunction(lambda: random.uniform(5000000, 20000000))
        )
        bearish = Trait(
            price_change_percent_24h=factory.LazyFunction(lambda: random.uniform(-10, -2)),
            volume_24h=factory.LazyFunction(lambda: random.uniform(5000000, 20000000))
        )
        volatile = Trait(
            high_24h=factory.LazyAttribute(lambda obj: round(obj.price * random.uniform(1.10, 1.20), 2)),
            low_24h=factory.LazyAttribute(lambda obj: round(obj.price * random.uniform(0.80, 0.90), 2))
        )


class CandleDataFactory(Factory):
    """Factory for creating OHLCV candle data."""
    
    class Meta:
        model = dict
    
    symbol = factory.LazyFunction(lambda: random.choice(["BTC_USDT", "ETH_USDT", "XRP_USDT"]))
    interval = factory.LazyFunction(lambda: random.choice(["1m", "5m", "15m", "1h", "4h", "1d"]))
    timestamp = factory.LazyFunction(lambda: datetime.now(timezone.utc))
    
    # Generate realistic OHLC values
    open = factory.LazyAttribute(lambda obj: {
        "BTC_USDT": random.uniform(40000, 60000),
        "ETH_USDT": random.uniform(2500, 4000),
        "XRP_USDT": random.uniform(0.5, 1.5)
    }.get(obj.symbol, random.uniform(10, 1000)))
    
    # Close can be higher or lower than open
    close = factory.LazyAttribute(
        lambda obj: obj.open * random.uniform(0.98, 1.02)
    )
    
    # High should be >= max(open, close)
    high = factory.LazyAttribute(
        lambda obj: max(obj.open, obj.close) * random.uniform(1.0, 1.02)
    )
    
    # Low should be <= min(open, close)
    low = factory.LazyAttribute(
        lambda obj: min(obj.open, obj.close) * random.uniform(0.98, 1.0)
    )
    
    volume = factory.LazyFunction(lambda: round(random.uniform(100, 10000), 4))
    quote_volume = factory.LazyAttribute(lambda obj: round(obj.volume * obj.close, 2))
    trades = factory.LazyFunction(lambda: random.randint(10, 1000))
    
    # Technical indicators
    rsi = factory.LazyFunction(lambda: round(random.uniform(20, 80), 2))
    macd = factory.LazyFunction(lambda: round(random.uniform(-100, 100), 2))
    macd_signal = factory.LazyFunction(lambda: round(random.uniform(-100, 100), 2))
    
    class Params:
        """Trait parameters for different candle types."""
        green = Trait(
            close=factory.LazyAttribute(lambda obj: obj.open * random.uniform(1.01, 1.05))
        )
        red = Trait(
            close=factory.LazyAttribute(lambda obj: obj.open * random.uniform(0.95, 0.99))
        )
        doji = Trait(
            close=factory.LazyAttribute(lambda obj: obj.open * random.uniform(0.999, 1.001)),
            high=factory.LazyAttribute(lambda obj: obj.open * random.uniform(1.01, 1.02)),
            low=factory.LazyAttribute(lambda obj: obj.open * random.uniform(0.98, 0.99))
        )
        hammer = Trait(
            open=factory.LazyAttribute(lambda obj: obj.close * 1.01),
            low=factory.LazyAttribute(lambda obj: obj.close * 0.97),
            high=factory.LazyAttribute(lambda obj: obj.close * 1.005)
        )


class OrderBookFactory(Factory):
    """Factory for creating order book data."""
    
    class Meta:
        model = dict
    
    symbol = factory.LazyFunction(lambda: random.choice(["BTC_USDT", "ETH_USDT", "XRP_USDT"]))
    timestamp = factory.LazyFunction(lambda: datetime.now(timezone.utc))
    
    @factory.lazy_attribute
    def bids(self):
        """Generate realistic bid orders."""
        base_price = {
            "BTC_USDT": 50000,
            "ETH_USDT": 3000,
            "XRP_USDT": 1.0
        }.get(self.symbol, 100)
        
        bids = []
        current_price = base_price
        
        for i in range(20):
            # Price decreases as we go down the bid book
            current_price *= (1 - random.uniform(0.0001, 0.001))
            quantity = random.uniform(0.1, 10) * (1 + i * 0.1)  # Larger quantities at lower prices
            
            bids.append({
                "price": round(current_price, 4),
                "quantity": round(quantity, 4),
                "total": round(current_price * quantity, 2)
            })
        
        return bids
    
    @factory.lazy_attribute
    def asks(self):
        """Generate realistic ask orders."""
        base_price = {
            "BTC_USDT": 50000,
            "ETH_USDT": 3000,
            "XRP_USDT": 1.0
        }.get(self.symbol, 100)
        
        asks = []
        current_price = base_price
        
        for i in range(20):
            # Price increases as we go up the ask book
            current_price *= (1 + random.uniform(0.0001, 0.001))
            quantity = random.uniform(0.1, 10) * (1 + i * 0.1)  # Larger quantities at higher prices
            
            asks.append({
                "price": round(current_price, 4),
                "quantity": round(quantity, 4),
                "total": round(current_price * quantity, 2)
            })
        
        return asks
    
    @factory.lazy_attribute
    def spread(self):
        """Calculate the spread."""
        if self.bids and self.asks:
            return round(self.asks[0]["price"] - self.bids[0]["price"], 4)
        return 0
    
    @factory.lazy_attribute
    def mid_price(self):
        """Calculate the mid price."""
        if self.bids and self.asks:
            return round((self.asks[0]["price"] + self.bids[0]["price"]) / 2, 4)
        return 0
    
    @factory.lazy_attribute
    def liquidity_imbalance(self):
        """Calculate liquidity imbalance."""
        bid_liquidity = sum(bid["total"] for bid in self.bids[:5])
        ask_liquidity = sum(ask["total"] for ask in self.asks[:5])
        total_liquidity = bid_liquidity + ask_liquidity
        
        if total_liquidity > 0:
            return round((bid_liquidity - ask_liquidity) / total_liquidity, 4)
        return 0
    
    class Params:
        """Trait parameters for different order book conditions."""
        thin = Trait(
            bids=factory.LazyAttribute(lambda obj: obj.bids[:5]),
            asks=factory.LazyAttribute(lambda obj: obj.asks[:5])
        )
        deep = Trait(
            bids=factory.LazyAttribute(lambda obj: [
                {**bid, "quantity": bid["quantity"] * 10}
                for bid in obj.bids
            ]),
            asks=factory.LazyAttribute(lambda obj: [
                {**ask, "quantity": ask["quantity"] * 10}
                for ask in obj.asks
            ])
        )
        buy_pressure = Trait(
            bids=factory.LazyAttribute(lambda obj: [
                {**bid, "quantity": bid["quantity"] * 2}
                for bid in obj.bids
            ])
        )
        sell_pressure = Trait(
            asks=factory.LazyAttribute(lambda obj: [
                {**ask, "quantity": ask["quantity"] * 2}
                for ask in obj.asks
            ])
        )


def create_candle_series(
    symbol: str = "BTC_USDT",
    interval: str = "1h",
    count: int = 100,
    start_price: Optional[float] = None,
    trend: str = "random"  # "up", "down", "sideways", "random"
) -> List[Dict[str, Any]]:
    """
    Create a series of candles with realistic price movement.
    
    Args:
        symbol: Trading symbol
        interval: Candle interval
        count: Number of candles to generate
        start_price: Starting price (random if not specified)
        trend: Overall trend direction
        
    Returns:
        List of candle dictionaries
    """
    if start_price is None:
        start_price = {
            "BTC_USDT": 50000,
            "ETH_USDT": 3000,
            "XRP_USDT": 1.0
        }.get(symbol, 100)
    
    candles = []
    current_price = start_price
    current_time = datetime.now(timezone.utc)
    
    # Determine interval in minutes
    interval_minutes = {
        "1m": 1, "5m": 5, "15m": 15, "30m": 30,
        "1h": 60, "4h": 240, "1d": 1440
    }.get(interval, 60)
    
    for i in range(count):
        # Determine price movement based on trend
        if trend == "up":
            price_change = random.uniform(0.99, 1.02)
        elif trend == "down":
            price_change = random.uniform(0.98, 1.01)
        elif trend == "sideways":
            price_change = random.uniform(0.995, 1.005)
        else:  # random
            price_change = random.uniform(0.98, 1.02)
        
        # Add some volatility
        volatility = random.uniform(0.98, 1.02)
        
        open_price = current_price
        close_price = open_price * price_change
        high_price = max(open_price, close_price) * volatility
        low_price = min(open_price, close_price) / volatility
        
        candle = CandleDataFactory(
            symbol=symbol,
            interval=interval,
            timestamp=current_time - timedelta(minutes=interval_minutes * (count - i - 1)),
            open=round(open_price, 4),
            close=round(close_price, 4),
            high=round(high_price, 4),
            low=round(low_price, 4)
        )
        
        candles.append(candle)
        current_price = close_price
    
    return candles


def create_market_depth_snapshot(
    symbol: str = "BTC_USDT",
    depth: int = 50,
    spread_percent: float = 0.01,
    liquidity_factor: float = 1.0
) -> Dict[str, Any]:
    """
    Create a detailed market depth snapshot.
    
    Args:
        symbol: Trading symbol
        depth: Number of price levels
        spread_percent: Spread as percentage of mid price
        liquidity_factor: Multiplier for liquidity (higher = more liquidity)
        
    Returns:
        Market depth snapshot dictionary
    """
    base_price = {
        "BTC_USDT": 50000,
        "ETH_USDT": 3000,
        "XRP_USDT": 1.0
    }.get(symbol, 100)
    
    mid_price = base_price
    half_spread = mid_price * spread_percent / 2
    
    # Generate bids
    bids = []
    current_bid = mid_price - half_spread
    for i in range(depth):
        price_step = mid_price * 0.0001 * (i + 1)
        current_bid -= price_step
        
        # Liquidity increases with distance from mid price
        base_quantity = random.uniform(0.5, 2.0) * liquidity_factor
        quantity = base_quantity * (1 + i * 0.05)
        
        bids.append({
            "price": round(current_bid, 4),
            "quantity": round(quantity, 4),
            "orders": random.randint(1, 10)
        })
    
    # Generate asks
    asks = []
    current_ask = mid_price + half_spread
    for i in range(depth):
        price_step = mid_price * 0.0001 * (i + 1)
        current_ask += price_step
        
        # Liquidity increases with distance from mid price
        base_quantity = random.uniform(0.5, 2.0) * liquidity_factor
        quantity = base_quantity * (1 + i * 0.05)
        
        asks.append({
            "price": round(current_ask, 4),
            "quantity": round(quantity, 4),
            "orders": random.randint(1, 10)
        })
    
    return {
        "symbol": symbol,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "bids": bids,
        "asks": asks,
        "mid_price": round(mid_price, 4),
        "spread": round(asks[0]["price"] - bids[0]["price"], 4),
        "spread_percent": round(spread_percent * 100, 2),
        "total_bid_volume": round(sum(bid["quantity"] for bid in bids), 4),
        "total_ask_volume": round(sum(ask["quantity"] for ask in asks), 4),
        "liquidity_score": round(liquidity_factor * 100, 2)
    }