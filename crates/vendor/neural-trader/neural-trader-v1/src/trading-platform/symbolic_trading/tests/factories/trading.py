"""
Trading-related factories for testing.

This module provides factory classes for generating test trading data
including trades, orders, positions, and portfolios.
"""

import random
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

import factory
from factory import Factory, LazyAttribute, LazyFunction, SubFactory, Trait
from faker import Faker

fake = Faker()


class OrderFactory(Factory):
    """Factory for creating trading orders."""
    
    class Meta:
        model = dict
    
    order_id = factory.LazyFunction(lambda: str(uuid4()))
    symbol = factory.LazyFunction(lambda: random.choice(["BTC_USDT", "ETH_USDT", "XRP_USDT", "ADA_USDT"]))
    side = factory.LazyFunction(lambda: random.choice(["buy", "sell"]))
    order_type = factory.LazyFunction(lambda: random.choice(["market", "limit", "stop", "stop_limit"]))
    
    quantity = factory.LazyFunction(lambda: round(random.uniform(0.01, 10), 4))
    
    @factory.lazy_attribute
    def price(self):
        """Generate price based on order type."""
        if self.order_type == "market":
            return None
        
        base_prices = {
            "BTC_USDT": 50000,
            "ETH_USDT": 3000,
            "XRP_USDT": 1.0,
            "ADA_USDT": 0.5
        }
        base_price = base_prices.get(self.symbol, 100)
        
        if self.order_type == "limit":
            # Limit orders slightly off market price
            if self.side == "buy":
                return round(base_price * random.uniform(0.98, 0.995), 2)
            else:
                return round(base_price * random.uniform(1.005, 1.02), 2)
        elif self.order_type in ["stop", "stop_limit"]:
            # Stop orders further from market price
            if self.side == "buy":
                return round(base_price * random.uniform(1.01, 1.05), 2)
            else:
                return round(base_price * random.uniform(0.95, 0.99), 2)
        
        return round(base_price, 2)
    
    @factory.lazy_attribute
    def stop_price(self):
        """Generate stop price for stop orders."""
        if self.order_type in ["stop", "stop_limit"]:
            return self.price
        return None
    
    status = factory.LazyFunction(lambda: random.choice(["new", "pending", "partial", "filled", "cancelled", "rejected"]))
    
    @factory.lazy_attribute
    def filled_quantity(self):
        """Generate filled quantity based on status."""
        if self.status == "filled":
            return self.quantity
        elif self.status == "partial":
            return round(self.quantity * random.uniform(0.1, 0.9), 4)
        else:
            return 0
    
    @factory.lazy_attribute
    def remaining_quantity(self):
        """Calculate remaining quantity."""
        return round(self.quantity - self.filled_quantity, 4)
    
    @factory.lazy_attribute
    def average_fill_price(self):
        """Generate average fill price."""
        if self.filled_quantity > 0:
            if self.price:
                # Slight slippage from limit price
                slippage = random.uniform(0.999, 1.001)
                return round(self.price * slippage, 2)
            else:
                # Market order fill price
                base_prices = {
                    "BTC_USDT": 50000,
                    "ETH_USDT": 3000,
                    "XRP_USDT": 1.0,
                    "ADA_USDT": 0.5
                }
                return round(base_prices.get(self.symbol, 100) * random.uniform(0.999, 1.001), 2)
        return None
    
    time_in_force = factory.LazyFunction(lambda: random.choice(["GTC", "IOC", "FOK", "DAY"]))
    
    created_at = factory.LazyFunction(lambda: datetime.now(timezone.utc) - timedelta(hours=random.randint(0, 24)))
    updated_at = factory.LazyAttribute(lambda obj: obj.created_at + timedelta(minutes=random.randint(0, 60)))
    
    commission = factory.LazyAttribute(
        lambda obj: round(obj.filled_quantity * obj.average_fill_price * 0.001, 4) if obj.filled_quantity > 0 else 0
    )
    commission_currency = "USDT"
    
    metadata = factory.LazyFunction(lambda: {
        "strategy": random.choice(["swing", "momentum", "mirror", "arbitrage"]),
        "risk_score": round(random.uniform(1, 5), 1),
        "confidence": round(random.uniform(0.6, 0.95), 2)
    })
    
    class Params:
        """Trait parameters for different order scenarios."""
        market_buy = Trait(
            side="buy",
            order_type="market",
            price=None,
            status="filled"
        )
        limit_sell = Trait(
            side="sell",
            order_type="limit",
            status="pending"
        )
        stop_loss = Trait(
            side="sell",
            order_type="stop",
            metadata={"strategy": "risk_management", "stop_type": "stop_loss"}
        )
        take_profit = Trait(
            side="sell",
            order_type="limit",
            metadata={"strategy": "risk_management", "stop_type": "take_profit"}
        )


class TradeFactory(Factory):
    """Factory for creating executed trades."""
    
    class Meta:
        model = dict
    
    trade_id = factory.LazyFunction(lambda: str(uuid4()))
    order_id = factory.LazyFunction(lambda: str(uuid4()))
    symbol = factory.LazyFunction(lambda: random.choice(["BTC_USDT", "ETH_USDT", "XRP_USDT", "ADA_USDT"]))
    side = factory.LazyFunction(lambda: random.choice(["buy", "sell"]))
    
    @factory.lazy_attribute
    def price(self):
        """Generate trade price based on symbol."""
        base_prices = {
            "BTC_USDT": 50000,
            "ETH_USDT": 3000,
            "XRP_USDT": 1.0,
            "ADA_USDT": 0.5
        }
        base_price = base_prices.get(self.symbol, 100)
        return round(base_price * random.uniform(0.98, 1.02), 2)
    
    quantity = factory.LazyFunction(lambda: round(random.uniform(0.01, 10), 4))
    
    @factory.lazy_attribute
    def value(self):
        """Calculate trade value."""
        return round(self.price * self.quantity, 2)
    
    commission = factory.LazyAttribute(lambda obj: round(obj.value * 0.001, 4))
    commission_currency = "USDT"
    
    executed_at = factory.LazyFunction(lambda: datetime.now(timezone.utc) - timedelta(hours=random.randint(0, 48)))
    
    # P&L tracking
    @factory.lazy_attribute
    def entry_price(self):
        """Entry price for P&L calculation."""
        if self.side == "sell":
            # For sells, entry was a previous buy
            return round(self.price * random.uniform(0.9, 0.99), 2)
        return None
    
    @factory.lazy_attribute
    def realized_pnl(self):
        """Calculate realized P&L for closing trades."""
        if self.side == "sell" and self.entry_price:
            pnl = (self.price - self.entry_price) * self.quantity
            return round(pnl - self.commission, 2)
        return None
    
    @factory.lazy_attribute
    def pnl_percent(self):
        """Calculate P&L percentage."""
        if self.realized_pnl is not None and self.entry_price:
            return round((self.realized_pnl / (self.entry_price * self.quantity)) * 100, 2)
        return None
    
    metadata = factory.LazyFunction(lambda: {
        "strategy": random.choice(["swing", "momentum", "mirror", "arbitrage"]),
        "signal_strength": round(random.uniform(0.6, 0.95), 2),
        "market_impact": round(random.uniform(0.0001, 0.001), 4)
    })
    
    class Params:
        """Trait parameters for different trade scenarios."""
        profitable = Trait(
            side="sell",
            entry_price=factory.LazyAttribute(lambda obj: round(obj.price * 0.9, 2))
        )
        loss = Trait(
            side="sell",
            entry_price=factory.LazyAttribute(lambda obj: round(obj.price * 1.1, 2))
        )


class PositionFactory(Factory):
    """Factory for creating trading positions."""
    
    class Meta:
        model = dict
    
    position_id = factory.LazyFunction(lambda: str(uuid4()))
    symbol = factory.LazyFunction(lambda: random.choice(["BTC_USDT", "ETH_USDT", "XRP_USDT", "ADA_USDT"]))
    side = factory.LazyFunction(lambda: random.choice(["long", "short"]))
    
    quantity = factory.LazyFunction(lambda: round(random.uniform(0.1, 10), 4))
    
    @factory.lazy_attribute
    def entry_price(self):
        """Generate entry price based on symbol."""
        base_prices = {
            "BTC_USDT": 50000,
            "ETH_USDT": 3000,
            "XRP_USDT": 1.0,
            "ADA_USDT": 0.5
        }
        base_price = base_prices.get(self.symbol, 100)
        return round(base_price * random.uniform(0.95, 1.05), 2)
    
    @factory.lazy_attribute
    def current_price(self):
        """Generate current price with some movement from entry."""
        if self.side == "long":
            # Could be profit or loss
            return round(self.entry_price * random.uniform(0.9, 1.1), 2)
        else:
            # Short position - inverse relationship
            return round(self.entry_price * random.uniform(0.9, 1.1), 2)
    
    @factory.lazy_attribute
    def market_value(self):
        """Calculate current market value."""
        return round(self.quantity * self.current_price, 2)
    
    @factory.lazy_attribute
    def cost_basis(self):
        """Calculate cost basis."""
        return round(self.quantity * self.entry_price, 2)
    
    @factory.lazy_attribute
    def unrealized_pnl(self):
        """Calculate unrealized P&L."""
        if self.side == "long":
            return round((self.current_price - self.entry_price) * self.quantity, 2)
        else:  # short
            return round((self.entry_price - self.current_price) * self.quantity, 2)
    
    @factory.lazy_attribute
    def unrealized_pnl_percent(self):
        """Calculate unrealized P&L percentage."""
        return round((self.unrealized_pnl / self.cost_basis) * 100, 2)
    
    opened_at = factory.LazyFunction(lambda: datetime.now(timezone.utc) - timedelta(days=random.randint(0, 30)))
    
    # Risk management
    stop_loss = factory.LazyAttribute(
        lambda obj: round(obj.entry_price * (0.95 if obj.side == "long" else 1.05), 2)
    )
    take_profit = factory.LazyAttribute(
        lambda obj: round(obj.entry_price * (1.05 if obj.side == "long" else 0.95), 2)
    )
    
    leverage = factory.LazyFunction(lambda: random.choice([1, 2, 3, 5, 10]))
    margin_used = factory.LazyAttribute(lambda obj: round(obj.cost_basis / obj.leverage, 2))
    
    metadata = factory.LazyFunction(lambda: {
        "strategy": random.choice(["swing", "momentum", "mirror", "arbitrage"]),
        "confidence": round(random.uniform(0.6, 0.95), 2),
        "risk_score": round(random.uniform(1, 5), 1)
    })
    
    class Params:
        """Trait parameters for different position scenarios."""
        profitable_long = Trait(
            side="long",
            current_price=factory.LazyAttribute(lambda obj: round(obj.entry_price * random.uniform(1.02, 1.10), 2))
        )
        losing_long = Trait(
            side="long",
            current_price=factory.LazyAttribute(lambda obj: round(obj.entry_price * random.uniform(0.90, 0.98), 2))
        )
        profitable_short = Trait(
            side="short",
            current_price=factory.LazyAttribute(lambda obj: round(obj.entry_price * random.uniform(0.90, 0.98), 2))
        )
        at_risk = Trait(
            current_price=factory.LazyAttribute(lambda obj: round(obj.stop_loss * 1.01, 2))
        )


class PortfolioFactory(Factory):
    """Factory for creating portfolio snapshots."""
    
    class Meta:
        model = dict
    
    portfolio_id = factory.LazyFunction(lambda: str(uuid4()))
    account_id = factory.LazyFunction(lambda: str(uuid4()))
    
    # Cash balances
    cash_balance = factory.LazyFunction(lambda: round(random.uniform(10000, 100000), 2))
    reserved_cash = factory.LazyFunction(lambda: round(random.uniform(0, 10000), 2))
    available_cash = factory.LazyAttribute(lambda obj: round(obj.cash_balance - obj.reserved_cash, 2))
    
    # Generate positions
    @factory.lazy_attribute
    def positions(self):
        """Generate a list of positions."""
        num_positions = random.randint(3, 10)
        return [PositionFactory() for _ in range(num_positions)]
    
    @factory.lazy_attribute
    def total_value(self):
        """Calculate total portfolio value."""
        positions_value = sum(pos["market_value"] for pos in self.positions)
        return round(self.cash_balance + positions_value, 2)
    
    @factory.lazy_attribute
    def total_cost_basis(self):
        """Calculate total cost basis of positions."""
        return round(sum(pos["cost_basis"] for pos in self.positions), 2)
    
    @factory.lazy_attribute
    def total_unrealized_pnl(self):
        """Calculate total unrealized P&L."""
        return round(sum(pos["unrealized_pnl"] for pos in self.positions), 2)
    
    @factory.lazy_attribute
    def total_unrealized_pnl_percent(self):
        """Calculate total unrealized P&L percentage."""
        if self.total_cost_basis > 0:
            return round((self.total_unrealized_pnl / self.total_cost_basis) * 100, 2)
        return 0
    
    # Historical performance
    @factory.lazy_attribute
    def realized_pnl_today(self):
        """Generate today's realized P&L."""
        return round(random.uniform(-1000, 2000), 2)
    
    @factory.lazy_attribute
    def realized_pnl_week(self):
        """Generate week's realized P&L."""
        return round(random.uniform(-5000, 10000), 2)
    
    @factory.lazy_attribute
    def realized_pnl_month(self):
        """Generate month's realized P&L."""
        return round(random.uniform(-10000, 20000), 2)
    
    @factory.lazy_attribute
    def realized_pnl_year(self):
        """Generate year's realized P&L."""
        return round(random.uniform(-20000, 50000), 2)
    
    # Risk metrics
    @factory.lazy_attribute
    def margin_used(self):
        """Calculate margin used."""
        return round(sum(pos.get("margin_used", 0) for pos in self.positions), 2)
    
    @factory.lazy_attribute
    def margin_available(self):
        """Calculate available margin."""
        return round(self.available_cash * 2, 2)  # 2x leverage available
    
    @factory.lazy_attribute
    def risk_score(self):
        """Calculate portfolio risk score."""
        if self.positions:
            avg_risk = sum(pos.get("metadata", {}).get("risk_score", 3) for pos in self.positions) / len(self.positions)
            leverage_factor = self.margin_used / self.total_value if self.total_value > 0 else 0
            return round(avg_risk * (1 + leverage_factor), 1)
        return 0
    
    timestamp = factory.LazyFunction(lambda: datetime.now(timezone.utc))
    
    class Params:
        """Trait parameters for different portfolio scenarios."""
        profitable = Trait(
            total_unrealized_pnl=factory.LazyFunction(lambda: round(random.uniform(1000, 10000), 2)),
            realized_pnl_today=factory.LazyFunction(lambda: round(random.uniform(100, 1000), 2))
        )
        at_loss = Trait(
            total_unrealized_pnl=factory.LazyFunction(lambda: round(random.uniform(-10000, -1000), 2)),
            realized_pnl_today=factory.LazyFunction(lambda: round(random.uniform(-1000, -100), 2))
        )
        high_risk = Trait(
            margin_used=factory.LazyAttribute(lambda obj: round(obj.total_value * 0.8, 2)),
            risk_score=factory.LazyFunction(lambda: round(random.uniform(4, 5), 1))
        )


def create_trade_history(
    symbol: str = "BTC_USDT",
    count: int = 100,
    days_back: int = 30,
    win_rate: float = 0.6
) -> List[Dict[str, Any]]:
    """
    Create a realistic trade history.
    
    Args:
        symbol: Trading symbol
        count: Number of trades to generate
        days_back: How many days back to generate trades
        win_rate: Percentage of winning trades
        
    Returns:
        List of trade dictionaries
    """
    trades = []
    current_time = datetime.now(timezone.utc)
    
    for i in range(count):
        # Determine if this is a winning trade
        is_winner = random.random() < win_rate
        
        # Generate trade timing
        hours_back = random.uniform(0, days_back * 24)
        trade_time = current_time - timedelta(hours=hours_back)
        
        # Create trade with appropriate P&L
        if is_winner:
            trade = TradeFactory(
                symbol=symbol,
                side="sell",  # Closing trade
                executed_at=trade_time,
                profitable=True
            )
        else:
            trade = TradeFactory(
                symbol=symbol,
                side="sell",  # Closing trade
                executed_at=trade_time,
                loss=True
            )
        
        trades.append(trade)
    
    # Sort by execution time
    trades.sort(key=lambda x: x["executed_at"])
    
    return trades


def create_portfolio_history(
    days: int = 30,
    initial_value: float = 100000,
    volatility: float = 0.02,
    trend: float = 0.001  # Daily growth rate
) -> List[Dict[str, Any]]:
    """
    Create portfolio value history.
    
    Args:
        days: Number of days of history
        initial_value: Starting portfolio value
        volatility: Daily volatility (standard deviation)
        trend: Daily growth trend
        
    Returns:
        List of portfolio snapshots
    """
    history = []
    current_value = initial_value
    current_time = datetime.now(timezone.utc)
    
    for i in range(days):
        # Calculate daily return with trend and volatility
        daily_return = random.gauss(trend, volatility)
        current_value *= (1 + daily_return)
        
        # Create portfolio snapshot
        snapshot_time = current_time - timedelta(days=days - i - 1)
        
        portfolio = {
            "timestamp": snapshot_time.isoformat(),
            "total_value": round(current_value, 2),
            "daily_return": round(daily_return * 100, 2),
            "cumulative_return": round(((current_value / initial_value) - 1) * 100, 2)
        }
        
        history.append(portfolio)
    
    return history