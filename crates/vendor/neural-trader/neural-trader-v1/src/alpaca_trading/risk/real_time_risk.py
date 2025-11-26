"""Real-time risk management with stream-based updates.

Provides ultra-low latency risk calculations with < 2ms response time.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from decimal import Decimal
import numpy as np
import redis.asyncio as redis
from alpaca.trading.models import Position
from alpaca.data.models import Trade, Quote
import logging

logger = logging.getLogger(__name__)


@dataclass
class RiskMetrics:
    """Real-time risk metrics for a position."""
    symbol: str
    position_value: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal
    var_95: Decimal  # Value at Risk at 95% confidence
    var_99: Decimal  # Value at Risk at 99% confidence
    exposure: Decimal
    beta: float
    delta: Optional[float] = None  # For options
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    timestamp_ms: int = field(default_factory=lambda: int(time.time() * 1000))


class RealTimeRiskManager:
    """Ultra-low latency risk management system."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis: Optional[redis.Redis] = None
        self.redis_url = redis_url
        self.positions: Dict[str, Position] = {}
        self.latest_prices: Dict[str, Decimal] = {}
        self.latest_quotes: Dict[str, Quote] = {}
        self.risk_metrics: Dict[str, RiskMetrics] = {}
        self.historical_returns: Dict[str, List[float]] = {}
        self._running = False
        
    async def initialize(self):
        """Initialize Redis connection and data structures."""
        self.redis = await redis.from_url(self.redis_url, decode_responses=True)
        await self.redis.ping()
        logger.info("Redis connection established for real-time risk")
        
    async def close(self):
        """Clean up resources."""
        self._running = False
        if self.redis:
            await self.redis.close()
            
    async def update_position(self, position: Position):
        """Update position with < 2ms latency."""
        start_time = time.perf_counter()
        
        # Store position in memory
        self.positions[position.symbol] = position
        
        # Calculate risk metrics if we have price data
        if position.symbol in self.latest_prices:
            metrics = await self._calculate_risk_metrics(position)
            self.risk_metrics[position.symbol] = metrics
            
            # Store in Redis for persistence
            await self._store_metrics_redis(metrics)
            
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        if elapsed_ms > 2:
            logger.warning(f"Position update took {elapsed_ms:.2f}ms (> 2ms target)")
            
    async def update_price(self, symbol: str, price: Decimal):
        """Update latest price and recalculate risk."""
        start_time = time.perf_counter()
        
        self.latest_prices[symbol] = price
        
        # Recalculate risk for affected position
        if symbol in self.positions:
            position = self.positions[symbol]
            metrics = await self._calculate_risk_metrics(position)
            self.risk_metrics[symbol] = metrics
            await self._store_metrics_redis(metrics)
            
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        if elapsed_ms > 2:
            logger.warning(f"Price update took {elapsed_ms:.2f}ms (> 2ms target)")
            
    async def update_quote(self, quote: Quote):
        """Update quote data for options Greeks calculation."""
        self.latest_quotes[quote.symbol] = quote
        
    async def _calculate_risk_metrics(self, position: Position) -> RiskMetrics:
        """Calculate all risk metrics for a position."""
        symbol = position.symbol
        current_price = self.latest_prices.get(symbol, position.current_price)
        
        # Position value and P&L
        position_value = abs(position.qty) * current_price
        cost_basis = abs(position.qty) * position.avg_entry_price
        unrealized_pnl = position_value - cost_basis
        if position.side == "short":
            unrealized_pnl = -unrealized_pnl
            
        # Get historical returns for VaR calculation
        returns = await self._get_historical_returns(symbol)
        
        # Calculate VaR using parametric method for speed
        if returns:
            returns_array = np.array(returns)
            std_dev = np.std(returns_array)
            mean_return = np.mean(returns_array)
            
            # VaR at 95% and 99% confidence
            var_95 = position_value * (mean_return - 1.645 * std_dev)
            var_99 = position_value * (mean_return - 2.326 * std_dev)
        else:
            # Default VaR if no historical data
            var_95 = position_value * Decimal("0.05")
            var_99 = position_value * Decimal("0.10")
            
        # Calculate beta (simplified - would need market returns in production)
        beta = await self._calculate_beta(symbol, returns)
        
        # Greeks for options (if applicable)
        delta, gamma, theta, vega = None, None, None, None
        if self._is_option(symbol):
            delta, gamma, theta, vega = await self._calculate_greeks(symbol, position)
            
        return RiskMetrics(
            symbol=symbol,
            position_value=position_value,
            unrealized_pnl=unrealized_pnl,
            realized_pnl=Decimal("0"),  # Would track from trade history
            var_95=Decimal(str(var_95)),
            var_99=Decimal(str(var_99)),
            exposure=position_value * Decimal(str(beta)),
            beta=beta,
            delta=delta,
            gamma=gamma,
            theta=theta,
            vega=vega
        )
        
    async def _get_historical_returns(self, symbol: str) -> List[float]:
        """Get historical returns from Redis cache."""
        key = f"returns:{symbol}"
        
        # Try to get from Redis first
        cached = await self.redis.lrange(key, 0, -1)
        if cached:
            return [float(r) for r in cached]
            
        # In production, would fetch from data provider
        # For now, return synthetic data
        returns = np.random.normal(0.0001, 0.02, 252).tolist()
        
        # Cache in Redis
        pipe = self.redis.pipeline()
        for ret in returns:
            pipe.rpush(key, ret)
        pipe.expire(key, 3600)  # 1 hour cache
        await pipe.execute()
        
        return returns
        
    async def _calculate_beta(self, symbol: str, returns: List[float]) -> float:
        """Calculate beta against market (simplified)."""
        if not returns:
            return 1.0
            
        # In production, would correlate with SPY returns
        # For now, use a heuristic based on volatility
        volatility = np.std(returns) if returns else 0.02
        market_vol = 0.015  # Typical market volatility
        
        return min(max(volatility / market_vol, 0.5), 2.0)
        
    def _is_option(self, symbol: str) -> bool:
        """Check if symbol is an option."""
        # Simple check - in production would use proper option symbol parsing
        return len(symbol) > 10 and any(c in symbol for c in ['C', 'P'])
        
    async def _calculate_greeks(self, symbol: str, position: Position) -> tuple:
        """Calculate option Greeks (simplified Black-Scholes)."""
        # This is a simplified implementation
        # In production, would use proper option pricing models
        
        quote = self.latest_quotes.get(symbol)
        if not quote:
            return None, None, None, None
            
        # Extract option parameters from symbol
        # This is highly simplified - real implementation would parse properly
        underlying_price = float(self.latest_prices.get(symbol[:3], 100))
        strike = 100  # Would parse from symbol
        time_to_expiry = 30 / 365  # 30 days, would calculate properly
        volatility = 0.25  # Would calculate implied volatility
        risk_free_rate = 0.05
        
        # Simplified Greeks calculation
        moneyness = underlying_price / strike
        
        # Delta
        if 'C' in symbol:  # Call
            delta = min(0.5 + 0.3 * (moneyness - 1), 1.0)
        else:  # Put
            delta = max(-0.5 + 0.3 * (moneyness - 1), -1.0)
            
        # Gamma (rate of change of delta)
        gamma = 0.05 * np.exp(-0.5 * ((moneyness - 1) / 0.1) ** 2)
        
        # Theta (time decay)
        theta = -0.05 * volatility * np.sqrt(1 / time_to_expiry)
        
        # Vega (volatility sensitivity)
        vega = 0.4 * np.sqrt(time_to_expiry)
        
        return delta, gamma, theta, vega
        
    async def _store_metrics_redis(self, metrics: RiskMetrics):
        """Store risk metrics in Redis for persistence."""
        key = f"risk:metrics:{metrics.symbol}"
        
        # Store as hash for efficient updates
        await self.redis.hset(key, mapping={
            "position_value": str(metrics.position_value),
            "unrealized_pnl": str(metrics.unrealized_pnl),
            "realized_pnl": str(metrics.realized_pnl),
            "var_95": str(metrics.var_95),
            "var_99": str(metrics.var_99),
            "exposure": str(metrics.exposure),
            "beta": str(metrics.beta),
            "delta": str(metrics.delta) if metrics.delta else "",
            "gamma": str(metrics.gamma) if metrics.gamma else "",
            "theta": str(metrics.theta) if metrics.theta else "",
            "vega": str(metrics.vega) if metrics.vega else "",
            "timestamp_ms": str(metrics.timestamp_ms)
        })
        
        # Set expiry
        await self.redis.expire(key, 86400)  # 24 hours
        
    async def get_portfolio_risk(self) -> Dict[str, Any]:
        """Get aggregated portfolio risk metrics."""
        start_time = time.perf_counter()
        
        total_value = Decimal("0")
        total_exposure = Decimal("0")
        total_unrealized_pnl = Decimal("0")
        total_var_95 = Decimal("0")
        total_var_99 = Decimal("0")
        
        for metrics in self.risk_metrics.values():
            total_value += metrics.position_value
            total_exposure += metrics.exposure
            total_unrealized_pnl += metrics.unrealized_pnl
            total_var_95 += metrics.var_95
            total_var_99 += metrics.var_99
            
        portfolio_risk = {
            "total_value": float(total_value),
            "total_exposure": float(total_exposure),
            "total_unrealized_pnl": float(total_unrealized_pnl),
            "portfolio_var_95": float(total_var_95),
            "portfolio_var_99": float(total_var_99),
            "position_count": len(self.positions),
            "calculation_time_ms": (time.perf_counter() - start_time) * 1000
        }
        
        # Store in Redis
        await self.redis.hset("risk:portfolio", mapping={
            k: str(v) for k, v in portfolio_risk.items()
        })
        
        return portfolio_risk
        
    async def stream_risk_updates(self, callback):
        """Stream real-time risk updates to callback function."""
        self._running = True
        
        while self._running:
            try:
                # Get all risk metrics
                risk_update = {
                    "timestamp": int(time.time() * 1000),
                    "positions": {
                        symbol: {
                            "value": float(metrics.position_value),
                            "pnl": float(metrics.unrealized_pnl),
                            "var_95": float(metrics.var_95),
                            "exposure": float(metrics.exposure),
                            "beta": metrics.beta
                        }
                        for symbol, metrics in self.risk_metrics.items()
                    },
                    "portfolio": await self.get_portfolio_risk()
                }
                
                # Call the callback
                await callback(risk_update)
                
                # Stream at 10Hz (100ms intervals)
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in risk stream: {e}")
                await asyncio.sleep(1)