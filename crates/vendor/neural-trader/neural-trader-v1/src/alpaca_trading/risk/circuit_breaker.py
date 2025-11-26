"""Circuit breakers for risk management with automatic shutdown.

Implements loss-based, volatility, and rate limiting circuit breakers.
"""

import asyncio
import time
from typing import Dict, List, Optional, Set, Callable
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from collections import deque
import redis.asyncio as redis
import logging

logger = logging.getLogger(__name__)


class BreakerType(Enum):
    """Types of circuit breakers."""
    LOSS_LIMIT = "loss_limit"
    VOLATILITY = "volatility"
    ORDER_RATE = "order_rate"
    DRAWDOWN = "drawdown"
    CORRELATION = "correlation"
    SYSTEM_ERROR = "system_error"


class BreakerState(Enum):
    """Circuit breaker states."""
    ACTIVE = "active"
    WARNING = "warning"
    TRIPPED = "tripped"
    COOLDOWN = "cooldown"
    DISABLED = "disabled"


@dataclass
class CircuitBreakerConfig:
    """Configuration for a circuit breaker."""
    breaker_type: BreakerType
    threshold: Decimal
    warning_threshold: Optional[Decimal] = None
    cooldown_seconds: int = 300  # 5 minutes default
    auto_reset: bool = True
    escalation_action: Optional[str] = None


@dataclass
class BreakerStatus:
    """Current status of a circuit breaker."""
    breaker_type: BreakerType
    state: BreakerState
    current_value: Decimal
    threshold: Decimal
    trips_count: int = 0
    last_trip_ms: Optional[int] = None
    cooldown_end_ms: Optional[int] = None
    message: str = ""


@dataclass
class TradingMetrics:
    """Real-time trading metrics for breaker evaluation."""
    total_pnl: Decimal = Decimal("0")
    daily_pnl: Decimal = Decimal("0")
    max_drawdown: Decimal = Decimal("0")
    current_volatility: float = 0.0
    order_count_1min: int = 0
    order_count_5min: int = 0
    error_count: int = 0
    last_update_ms: int = field(default_factory=lambda: int(time.time() * 1000))


class CircuitBreaker:
    """Advanced circuit breaker system with recovery protocols."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis: Optional[redis.Redis] = None
        self.redis_url = redis_url
        
        # Breaker configurations
        self.breakers: Dict[BreakerType, CircuitBreakerConfig] = {}
        self.breaker_status: Dict[BreakerType, BreakerStatus] = {}
        
        # Trading metrics
        self.metrics = TradingMetrics()
        self.order_history = deque(maxlen=1000)  # Last 1000 orders
        self.pnl_history = deque(maxlen=1440)  # 24 hours at 1-min intervals
        
        # Callbacks
        self.trip_callbacks: List[Callable] = []
        self.recovery_callbacks: List[Callable] = []
        
        # Emergency shutdown flag
        self.emergency_shutdown = False
        self._monitor_task: Optional[asyncio.Task] = None
        
        # Load default breakers
        self._load_default_breakers()
        
    async def initialize(self):
        """Initialize Redis and start monitoring."""
        self.redis = await redis.from_url(self.redis_url, decode_responses=True)
        await self.redis.ping()
        
        # Load saved states
        await self._load_breaker_states()
        
        # Start monitoring
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        
        logger.info("Circuit breaker system initialized")
        
    async def close(self):
        """Clean up resources."""
        if self._monitor_task:
            self._monitor_task.cancel()
        if self.redis:
            await self.redis.close()
            
    def _load_default_breakers(self):
        """Load default circuit breaker configurations."""
        # Loss limit breaker - trip at 2% daily loss
        self.add_breaker(CircuitBreakerConfig(
            breaker_type=BreakerType.LOSS_LIMIT,
            threshold=Decimal("-0.02"),  # -2% daily loss
            warning_threshold=Decimal("-0.015"),  # -1.5% warning
            cooldown_seconds=1800,  # 30 minutes
            escalation_action="halt_all_trading"
        ))
        
        # Volatility breaker - trip at extreme volatility
        self.add_breaker(CircuitBreakerConfig(
            breaker_type=BreakerType.VOLATILITY,
            threshold=Decimal("0.05"),  # 5% volatility
            warning_threshold=Decimal("0.03"),  # 3% warning
            cooldown_seconds=600,  # 10 minutes
            escalation_action="reduce_position_sizes"
        ))
        
        # Order rate breaker - prevent algo gone wild
        self.add_breaker(CircuitBreakerConfig(
            breaker_type=BreakerType.ORDER_RATE,
            threshold=Decimal("100"),  # 100 orders per minute
            warning_threshold=Decimal("75"),
            cooldown_seconds=300,
            escalation_action="halt_new_orders"
        ))
        
        # Drawdown breaker - trip at 5% drawdown
        self.add_breaker(CircuitBreakerConfig(
            breaker_type=BreakerType.DRAWDOWN,
            threshold=Decimal("-0.05"),  # -5% from peak
            warning_threshold=Decimal("-0.03"),
            cooldown_seconds=3600,  # 1 hour
            escalation_action="close_all_positions"
        ))
        
        # System error breaker
        self.add_breaker(CircuitBreakerConfig(
            breaker_type=BreakerType.SYSTEM_ERROR,
            threshold=Decimal("10"),  # 10 errors
            cooldown_seconds=600,
            auto_reset=False,  # Require manual reset
            escalation_action="emergency_shutdown"
        ))
        
    def add_breaker(self, config: CircuitBreakerConfig):
        """Add or update a circuit breaker."""
        self.breakers[config.breaker_type] = config
        
        # Initialize status
        self.breaker_status[config.breaker_type] = BreakerStatus(
            breaker_type=config.breaker_type,
            state=BreakerState.ACTIVE,
            current_value=Decimal("0"),
            threshold=config.threshold
        )
        
    async def _load_breaker_states(self):
        """Load saved breaker states from Redis."""
        states = await self.redis.hgetall("breakers:states")
        
        for breaker_type_str, state_data in states.items():
            try:
                breaker_type = BreakerType(breaker_type_str)
                if breaker_type in self.breaker_status:
                    # Parse saved state
                    import json
                    data = json.loads(state_data)
                    status = self.breaker_status[breaker_type]
                    status.state = BreakerState(data.get("state", "active"))
                    status.trips_count = data.get("trips_count", 0)
                    status.last_trip_ms = data.get("last_trip_ms")
                    status.cooldown_end_ms = data.get("cooldown_end_ms")
            except Exception as e:
                logger.error(f"Error loading breaker state: {e}")
                
    async def _monitor_loop(self):
        """Main monitoring loop for circuit breakers."""
        while True:
            try:
                # Check all breakers
                await self._check_all_breakers()
                
                # Check for cooldown expiry
                await self._check_cooldowns()
                
                # Update metrics
                await self._update_metrics()
                
                # Sleep for 100ms (10Hz monitoring)
                await asyncio.sleep(0.1)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in breaker monitor: {e}")
                await asyncio.sleep(1)
                
    async def update_pnl(self, pnl: Decimal, is_realized: bool = False):
        """Update P&L metrics with < 2ms latency."""
        start_time = time.perf_counter()
        
        # Update metrics
        self.metrics.daily_pnl = pnl
        self.metrics.total_pnl += pnl if is_realized else Decimal("0")
        
        # Add to history
        self.pnl_history.append({
            "pnl": float(pnl),
            "timestamp": int(time.time() * 1000),
            "is_realized": is_realized
        })
        
        # Calculate drawdown
        if self.pnl_history:
            peak_pnl = max(h["pnl"] for h in self.pnl_history)
            current_pnl = self.pnl_history[-1]["pnl"]
            self.metrics.max_drawdown = Decimal(str(min(0, current_pnl - peak_pnl)))
            
        # Store in Redis
        await self.redis.hset("metrics:trading", mapping={
            "daily_pnl": str(self.metrics.daily_pnl),
            "total_pnl": str(self.metrics.total_pnl),
            "max_drawdown": str(self.metrics.max_drawdown),
            "timestamp": str(self.metrics.last_update_ms)
        })
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        if elapsed_ms > 2:
            logger.warning(f"PnL update took {elapsed_ms:.2f}ms")
            
    async def record_order(self, order_id: str, symbol: str, 
                          quantity: int, side: str):
        """Record an order for rate monitoring."""
        timestamp = int(time.time() * 1000)
        
        self.order_history.append({
            "order_id": order_id,
            "symbol": symbol,
            "quantity": quantity,
            "side": side,
            "timestamp": timestamp
        })
        
        # Calculate order rates
        current_time = time.time()
        self.metrics.order_count_1min = sum(
            1 for o in self.order_history 
            if o["timestamp"] > (current_time - 60) * 1000
        )
        self.metrics.order_count_5min = sum(
            1 for o in self.order_history 
            if o["timestamp"] > (current_time - 300) * 1000
        )
        
    async def record_error(self, error_type: str, message: str):
        """Record a system error."""
        self.metrics.error_count += 1
        
        # Store error details
        await self.redis.hset(
            f"errors:{int(time.time())}",
            mapping={
                "type": error_type,
                "message": message,
                "timestamp": str(int(time.time() * 1000))
            }
        )
        
    async def _check_all_breakers(self):
        """Check all circuit breakers against current metrics."""
        # Loss limit breaker
        if BreakerType.LOSS_LIMIT in self.breakers:
            await self._check_loss_breaker()
            
        # Volatility breaker
        if BreakerType.VOLATILITY in self.breakers:
            await self._check_volatility_breaker()
            
        # Order rate breaker
        if BreakerType.ORDER_RATE in self.breakers:
            await self._check_order_rate_breaker()
            
        # Drawdown breaker
        if BreakerType.DRAWDOWN in self.breakers:
            await self._check_drawdown_breaker()
            
        # System error breaker
        if BreakerType.SYSTEM_ERROR in self.breakers:
            await self._check_error_breaker()
            
    async def _check_loss_breaker(self):
        """Check loss limit breaker."""
        breaker = self.breakers[BreakerType.LOSS_LIMIT]
        status = self.breaker_status[BreakerType.LOSS_LIMIT]
        
        # Calculate daily loss percentage
        # In production, would use actual account equity
        account_value = Decimal("100000")  # Example
        daily_loss_pct = self.metrics.daily_pnl / account_value
        
        status.current_value = daily_loss_pct
        
        # Check thresholds
        if daily_loss_pct <= breaker.threshold and status.state == BreakerState.ACTIVE:
            await self._trip_breaker(BreakerType.LOSS_LIMIT, 
                                   f"Daily loss exceeded: {float(daily_loss_pct):.2%}")
        elif breaker.warning_threshold and daily_loss_pct <= breaker.warning_threshold:
            status.state = BreakerState.WARNING
            status.message = f"Approaching loss limit: {float(daily_loss_pct):.2%}"
            
    async def _check_volatility_breaker(self):
        """Check volatility breaker."""
        breaker = self.breakers[BreakerType.VOLATILITY]
        status = self.breaker_status[BreakerType.VOLATILITY]
        
        # Calculate volatility from PnL history
        if len(self.pnl_history) >= 10:
            recent_pnls = [h["pnl"] for h in list(self.pnl_history)[-60:]]  # Last hour
            import numpy as np
            volatility = np.std(recent_pnls) / np.mean(np.abs(recent_pnls)) if recent_pnls else 0
            self.metrics.current_volatility = volatility
            
            status.current_value = Decimal(str(volatility))
            
            # Check thresholds
            if status.current_value >= breaker.threshold and status.state == BreakerState.ACTIVE:
                await self._trip_breaker(BreakerType.VOLATILITY,
                                       f"Extreme volatility detected: {volatility:.2%}")
                                       
    async def _check_order_rate_breaker(self):
        """Check order rate breaker."""
        breaker = self.breakers[BreakerType.ORDER_RATE]
        status = self.breaker_status[BreakerType.ORDER_RATE]
        
        status.current_value = Decimal(str(self.metrics.order_count_1min))
        
        # Check thresholds
        if status.current_value >= breaker.threshold and status.state == BreakerState.ACTIVE:
            await self._trip_breaker(BreakerType.ORDER_RATE,
                                   f"Order rate too high: {self.metrics.order_count_1min}/min")
                                   
    async def _check_drawdown_breaker(self):
        """Check drawdown breaker."""
        breaker = self.breakers[BreakerType.DRAWDOWN]
        status = self.breaker_status[BreakerType.DRAWDOWN]
        
        # Drawdown as percentage
        account_value = Decimal("100000")  # Example
        drawdown_pct = self.metrics.max_drawdown / account_value
        
        status.current_value = drawdown_pct
        
        # Check thresholds
        if drawdown_pct <= breaker.threshold and status.state == BreakerState.ACTIVE:
            await self._trip_breaker(BreakerType.DRAWDOWN,
                                   f"Maximum drawdown exceeded: {float(drawdown_pct):.2%}")
                                   
    async def _check_error_breaker(self):
        """Check system error breaker."""
        breaker = self.breakers[BreakerType.SYSTEM_ERROR]
        status = self.breaker_status[BreakerType.SYSTEM_ERROR]
        
        status.current_value = Decimal(str(self.metrics.error_count))
        
        # Check threshold
        if status.current_value >= breaker.threshold and status.state == BreakerState.ACTIVE:
            await self._trip_breaker(BreakerType.SYSTEM_ERROR,
                                   f"Too many system errors: {self.metrics.error_count}")
                                   
    async def _trip_breaker(self, breaker_type: BreakerType, reason: str):
        """Trip a circuit breaker."""
        config = self.breakers[breaker_type]
        status = self.breaker_status[breaker_type]
        
        # Update status
        status.state = BreakerState.TRIPPED
        status.trips_count += 1
        status.last_trip_ms = int(time.time() * 1000)
        status.cooldown_end_ms = status.last_trip_ms + (config.cooldown_seconds * 1000)
        status.message = reason
        
        logger.warning(f"Circuit breaker tripped: {breaker_type.value} - {reason}")
        
        # Store in Redis
        await self._save_breaker_state(breaker_type, status)
        
        # Execute escalation action
        if config.escalation_action:
            await self._execute_escalation(config.escalation_action, reason)
            
        # Notify callbacks
        for callback in self.trip_callbacks:
            await callback(breaker_type, status, reason)
            
    async def _execute_escalation(self, action: str, reason: str):
        """Execute escalation action."""
        logger.info(f"Executing escalation: {action} due to {reason}")
        
        if action == "halt_all_trading":
            await self._halt_all_trading()
        elif action == "reduce_position_sizes":
            await self._reduce_position_sizes()
        elif action == "halt_new_orders":
            await self._halt_new_orders()
        elif action == "close_all_positions":
            await self._close_all_positions()
        elif action == "emergency_shutdown":
            await self._emergency_shutdown()
            
    async def _halt_all_trading(self):
        """Halt all trading activity."""
        await self.redis.set("trading:halted", "true", ex=3600)
        logger.critical("ALL TRADING HALTED")
        
    async def _reduce_position_sizes(self):
        """Signal to reduce position sizes."""
        await self.redis.set("trading:reduced_size", "0.5", ex=1800)  # 50% size for 30min
        logger.warning("Position sizes reduced to 50%")
        
    async def _halt_new_orders(self):
        """Halt new order creation."""
        await self.redis.set("trading:no_new_orders", "true", ex=600)
        logger.warning("New orders halted")
        
    async def _close_all_positions(self):
        """Signal to close all positions."""
        await self.redis.set("trading:close_all", "true", ex=300)
        logger.critical("CLOSING ALL POSITIONS")
        
    async def _emergency_shutdown(self):
        """Execute emergency shutdown."""
        self.emergency_shutdown = True
        await self.redis.set("trading:emergency_shutdown", "true")
        logger.critical("EMERGENCY SHUTDOWN INITIATED")
        
    async def _check_cooldowns(self):
        """Check and reset breakers after cooldown."""
        current_time = int(time.time() * 1000)
        
        for breaker_type, status in self.breaker_status.items():
            if status.state == BreakerState.TRIPPED and status.cooldown_end_ms:
                if current_time >= status.cooldown_end_ms:
                    config = self.breakers[breaker_type]
                    
                    if config.auto_reset:
                        # Auto reset after cooldown
                        status.state = BreakerState.COOLDOWN
                        status.message = "Cooldown period - monitoring for reset"
                        
                        # If metrics are good, fully reset
                        if await self._can_reset_breaker(breaker_type):
                            await self.reset_breaker(breaker_type, "Auto-reset after cooldown")
                            
    async def _can_reset_breaker(self, breaker_type: BreakerType) -> bool:
        """Check if breaker can be safely reset."""
        status = self.breaker_status[breaker_type]
        config = self.breakers[breaker_type]
        
        # Check if current value is below warning threshold
        if config.warning_threshold:
            if breaker_type in [BreakerType.LOSS_LIMIT, BreakerType.DRAWDOWN]:
                return status.current_value > config.warning_threshold
            else:
                return status.current_value < config.warning_threshold
                
        # Default: reset if value is significantly better than threshold
        return abs(status.current_value) < abs(config.threshold) * Decimal("0.5")
        
    async def reset_breaker(self, breaker_type: BreakerType, reason: str):
        """Manually reset a circuit breaker."""
        status = self.breaker_status[breaker_type]
        
        status.state = BreakerState.ACTIVE
        status.cooldown_end_ms = None
        status.message = f"Reset: {reason}"
        
        # Clear any escalation actions
        if breaker_type == BreakerType.LOSS_LIMIT:
            await self.redis.delete("trading:halted")
        elif breaker_type == BreakerType.ORDER_RATE:
            await self.redis.delete("trading:no_new_orders")
            
        # Save state
        await self._save_breaker_state(breaker_type, status)
        
        # Notify recovery callbacks
        for callback in self.recovery_callbacks:
            await callback(breaker_type, status, reason)
            
        logger.info(f"Circuit breaker reset: {breaker_type.value} - {reason}")
        
    async def _save_breaker_state(self, breaker_type: BreakerType, status: BreakerStatus):
        """Save breaker state to Redis."""
        import json
        state_data = {
            "state": status.state.value,
            "trips_count": status.trips_count,
            "last_trip_ms": status.last_trip_ms,
            "cooldown_end_ms": status.cooldown_end_ms,
            "current_value": str(status.current_value),
            "message": status.message
        }
        
        await self.redis.hset("breakers:states", breaker_type.value, json.dumps(state_data))
        
    async def _update_metrics(self):
        """Update trading metrics in Redis."""
        await self.redis.hset("metrics:breakers", mapping={
            "order_rate_1min": str(self.metrics.order_count_1min),
            "order_rate_5min": str(self.metrics.order_count_5min),
            "volatility": str(self.metrics.current_volatility),
            "error_count": str(self.metrics.error_count),
            "emergency_shutdown": str(self.emergency_shutdown)
        })
        
    def add_trip_callback(self, callback: Callable):
        """Add callback for breaker trips."""
        self.trip_callbacks.append(callback)
        
    def add_recovery_callback(self, callback: Callable):
        """Add callback for breaker recovery."""
        self.recovery_callbacks.append(callback)
        
    async def get_status(self) -> Dict[str, Any]:
        """Get current status of all circuit breakers."""
        return {
            "emergency_shutdown": self.emergency_shutdown,
            "breakers": [
                {
                    "type": status.breaker_type.value,
                    "state": status.state.value,
                    "current_value": float(status.current_value),
                    "threshold": float(status.threshold),
                    "trips_count": status.trips_count,
                    "message": status.message,
                    "cooldown_remaining": (
                        int((status.cooldown_end_ms - time.time() * 1000) / 1000)
                        if status.cooldown_end_ms and status.cooldown_end_ms > time.time() * 1000
                        else 0
                    )
                }
                for status in self.breaker_status.values()
            ],
            "metrics": {
                "daily_pnl": float(self.metrics.daily_pnl),
                "max_drawdown": float(self.metrics.max_drawdown),
                "volatility": self.metrics.current_volatility,
                "order_rate_1min": self.metrics.order_count_1min,
                "error_count": self.metrics.error_count
            }
        }
        
    async def test_breaker(self, breaker_type: BreakerType):
        """Test a circuit breaker (for testing/debugging)."""
        await self._trip_breaker(breaker_type, "Manual test trip")
        logger.info(f"Test trip executed for {breaker_type.value}")