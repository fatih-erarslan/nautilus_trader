"""
Circuit Breaker System for Sports Betting Risk Management

Automated protection mechanisms including:
- Stop-loss triggers
- Drawdown protection
- Consecutive loss limits
- Emergency shutdown procedures
- Dynamic risk adjustments
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime, timedelta
import json
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class CircuitBreakerType(Enum):
    """Types of circuit breakers"""
    DRAWDOWN = "drawdown"
    CONSECUTIVE_LOSSES = "consecutive_losses"
    DAILY_LOSS = "daily_loss"
    VELOCITY = "velocity"
    CONCENTRATION = "concentration"
    VOLATILITY = "volatility"
    BANKROLL_THRESHOLD = "bankroll_threshold"


class BreakerStatus(Enum):
    """Circuit breaker status"""
    NORMAL = "normal"
    WARNING = "warning"
    TRIGGERED = "triggered"
    COOLING_DOWN = "cooling_down"
    DISABLED = "disabled"


class ActionType(Enum):
    """Circuit breaker actions"""
    ALERT = "alert"
    REDUCE_STAKES = "reduce_stakes"
    HALT_BETTING = "halt_betting"
    EMERGENCY_STOP = "emergency_stop"
    FORCE_CLOSE = "force_close"


@dataclass
class BreakerTrigger:
    """Circuit breaker trigger event"""
    breaker_id: str
    trigger_time: datetime
    trigger_value: float
    threshold: float
    action: ActionType
    message: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BreakerConfig:
    """Circuit breaker configuration"""
    breaker_type: CircuitBreakerType
    threshold: float
    action: ActionType
    cooldown_minutes: int = 30
    enabled: bool = True
    severity_levels: Optional[Dict[float, ActionType]] = None
    custom_params: Dict[str, Any] = field(default_factory=dict)


class CircuitBreaker(ABC):
    """Base class for circuit breakers"""
    
    def __init__(self, 
                 breaker_id: str,
                 config: BreakerConfig):
        self.breaker_id = breaker_id
        self.config = config
        self.status = BreakerStatus.NORMAL
        self.trigger_history: List[BreakerTrigger] = []
        self.last_trigger_time: Optional[datetime] = None
        self.consecutive_triggers = 0
        
    @abstractmethod
    def check(self, 
              current_value: float, 
              context: Dict[str, Any]) -> Optional[BreakerTrigger]:
        """Check if circuit breaker should trigger"""
        pass
    
    def is_in_cooldown(self) -> bool:
        """Check if breaker is in cooldown period"""
        if self.last_trigger_time is None:
            return False
        
        cooldown_end = self.last_trigger_time + timedelta(minutes=self.config.cooldown_minutes)
        return datetime.now() < cooldown_end
    
    def trigger(self, value: float, message: str, metadata: Dict[str, Any] = None) -> BreakerTrigger:
        """Trigger the circuit breaker"""
        trigger = BreakerTrigger(
            breaker_id=self.breaker_id,
            trigger_time=datetime.now(),
            trigger_value=value,
            threshold=self.config.threshold,
            action=self.config.action,
            message=message,
            metadata=metadata or {}
        )
        
        self.trigger_history.append(trigger)
        self.last_trigger_time = trigger.trigger_time
        self.status = BreakerStatus.TRIGGERED
        self.consecutive_triggers += 1
        
        logger.warning(f"Circuit breaker triggered: {self.breaker_id} - {message}")
        
        return trigger
    
    def reset(self):
        """Reset circuit breaker to normal status"""
        self.status = BreakerStatus.NORMAL
        self.consecutive_triggers = 0
        logger.info(f"Circuit breaker reset: {self.breaker_id}")
    
    def disable(self):
        """Disable circuit breaker"""
        self.status = BreakerStatus.DISABLED
        logger.info(f"Circuit breaker disabled: {self.breaker_id}")
    
    def enable(self):
        """Enable circuit breaker"""
        if self.status == BreakerStatus.DISABLED:
            self.status = BreakerStatus.NORMAL
            logger.info(f"Circuit breaker enabled: {self.breaker_id}")


class DrawdownBreaker(CircuitBreaker):
    """Circuit breaker for maximum drawdown protection"""
    
    def check(self, current_drawdown: float, context: Dict[str, Any]) -> Optional[BreakerTrigger]:
        if not self.config.enabled or self.is_in_cooldown():
            return None
        
        if current_drawdown > self.config.threshold:
            # Check severity levels
            if self.config.severity_levels:
                for threshold, action in sorted(self.config.severity_levels.items()):
                    if current_drawdown >= threshold:
                        # Update action based on severity
                        original_action = self.config.action
                        self.config.action = action
                        
                        trigger = self.trigger(
                            value=current_drawdown,
                            message=f"Drawdown limit exceeded: {current_drawdown:.2%} > {threshold:.2%}",
                            metadata={
                                "bankroll": context.get("bankroll", 0),
                                "peak_bankroll": context.get("peak_bankroll", 0),
                                "severity_level": threshold
                            }
                        )
                        
                        # Restore original action
                        self.config.action = original_action
                        return trigger
            else:
                return self.trigger(
                    value=current_drawdown,
                    message=f"Drawdown limit exceeded: {current_drawdown:.2%} > {self.config.threshold:.2%}",
                    metadata={
                        "bankroll": context.get("bankroll", 0),
                        "peak_bankroll": context.get("peak_bankroll", 0)
                    }
                )
        
        return None


class ConsecutiveLossBreaker(CircuitBreaker):
    """Circuit breaker for consecutive losses"""
    
    def check(self, consecutive_losses: float, context: Dict[str, Any]) -> Optional[BreakerTrigger]:
        if not self.config.enabled or self.is_in_cooldown():
            return None
        
        if consecutive_losses >= self.config.threshold:
            return self.trigger(
                value=consecutive_losses,
                message=f"Consecutive losses limit exceeded: {consecutive_losses} >= {self.config.threshold}",
                metadata={
                    "recent_results": context.get("recent_results", []),
                    "total_loss_amount": context.get("total_loss_amount", 0)
                }
            )
        
        return None


class DailyLossBreaker(CircuitBreaker):
    """Circuit breaker for daily loss limits"""
    
    def check(self, daily_loss: float, context: Dict[str, Any]) -> Optional[BreakerTrigger]:
        if not self.config.enabled or self.is_in_cooldown():
            return None
        
        daily_loss_pct = abs(daily_loss) / context.get("starting_bankroll", 1)
        
        if daily_loss_pct > self.config.threshold:
            return self.trigger(
                value=daily_loss_pct,
                message=f"Daily loss limit exceeded: {daily_loss_pct:.2%} > {self.config.threshold:.2%}",
                metadata={
                    "daily_loss_amount": daily_loss,
                    "starting_bankroll": context.get("starting_bankroll", 0),
                    "trades_today": context.get("trades_today", 0)
                }
            )
        
        return None


class VelocityBreaker(CircuitBreaker):
    """Circuit breaker for rapid loss velocity"""
    
    def check(self, loss_velocity: float, context: Dict[str, Any]) -> Optional[BreakerTrigger]:
        if not self.config.enabled or self.is_in_cooldown():
            return None
        
        # Loss velocity is loss per hour
        if loss_velocity > self.config.threshold:
            return self.trigger(
                value=loss_velocity,
                message=f"Loss velocity exceeded: ${loss_velocity:.2f}/hour > ${self.config.threshold:.2f}/hour",
                metadata={
                    "time_window_hours": context.get("time_window_hours", 1),
                    "recent_losses": context.get("recent_losses", [])
                }
            )
        
        return None


class ConcentrationBreaker(CircuitBreaker):
    """Circuit breaker for position concentration"""
    
    def check(self, concentration_ratio: float, context: Dict[str, Any]) -> Optional[BreakerTrigger]:
        if not self.config.enabled or self.is_in_cooldown():
            return None
        
        if concentration_ratio > self.config.threshold:
            return self.trigger(
                value=concentration_ratio,
                message=f"Concentration ratio exceeded: {concentration_ratio:.3f} > {self.config.threshold:.3f}",
                metadata={
                    "largest_position": context.get("largest_position", 0),
                    "total_exposure": context.get("total_exposure", 0),
                    "num_positions": context.get("num_positions", 0)
                }
            )
        
        return None


class BankrollThresholdBreaker(CircuitBreaker):
    """Circuit breaker for absolute bankroll thresholds"""
    
    def check(self, current_bankroll: float, context: Dict[str, Any]) -> Optional[BreakerTrigger]:
        if not self.config.enabled or self.is_in_cooldown():
            return None
        
        if current_bankroll < self.config.threshold:
            return self.trigger(
                value=current_bankroll,
                message=f"Bankroll below threshold: ${current_bankroll:.2f} < ${self.config.threshold:.2f}",
                metadata={
                    "initial_bankroll": context.get("initial_bankroll", 0),
                    "bankroll_percentage": current_bankroll / context.get("initial_bankroll", 1)
                }
            )
        
        return None


class CircuitBreakerSystem:
    """
    Comprehensive circuit breaker system for sports betting risk management
    """
    
    def __init__(self, initial_bankroll: float):
        self.initial_bankroll = initial_bankroll
        self.current_bankroll = initial_bankroll
        self.peak_bankroll = initial_bankroll
        
        # Circuit breakers
        self.breakers: Dict[str, CircuitBreaker] = {}
        
        # State tracking
        self.consecutive_losses = 0
        self.last_bet_result: Optional[str] = None
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.last_reset_date = datetime.now().date()
        
        # Performance tracking
        self.performance_history = []
        self.trigger_history: List[BreakerTrigger] = []
        
        # System status
        self.system_halted = False
        self.emergency_stop = False
        
        # Initialize default breakers
        self._initialize_default_breakers()
        
        logger.info(f"Circuit breaker system initialized with ${initial_bankroll:,.2f} bankroll")
    
    def _initialize_default_breakers(self):
        """Initialize default circuit breakers"""
        
        # Drawdown breaker with severity levels
        drawdown_config = BreakerConfig(
            breaker_type=CircuitBreakerType.DRAWDOWN,
            threshold=0.20,  # 20% drawdown
            action=ActionType.HALT_BETTING,
            cooldown_minutes=60,
            severity_levels={
                0.10: ActionType.ALERT,
                0.15: ActionType.REDUCE_STAKES, 
                0.20: ActionType.HALT_BETTING,
                0.30: ActionType.EMERGENCY_STOP
            }
        )
        self.add_breaker("drawdown_protection", DrawdownBreaker("drawdown_protection", drawdown_config))
        
        # Consecutive losses breaker
        consecutive_config = BreakerConfig(
            breaker_type=CircuitBreakerType.CONSECUTIVE_LOSSES,
            threshold=5,  # 5 consecutive losses
            action=ActionType.REDUCE_STAKES,
            cooldown_minutes=30
        )
        self.add_breaker("consecutive_losses", ConsecutiveLossBreaker("consecutive_losses", consecutive_config))
        
        # Daily loss breaker
        daily_loss_config = BreakerConfig(
            breaker_type=CircuitBreakerType.DAILY_LOSS,
            threshold=0.10,  # 10% daily loss
            action=ActionType.HALT_BETTING,
            cooldown_minutes=24 * 60  # 24 hours
        )
        self.add_breaker("daily_loss_limit", DailyLossBreaker("daily_loss_limit", daily_loss_config))
        
        # Velocity breaker
        velocity_config = BreakerConfig(
            breaker_type=CircuitBreakerType.VELOCITY,
            threshold=1000,  # $1000/hour loss rate
            action=ActionType.REDUCE_STAKES,
            cooldown_minutes=60
        )
        self.add_breaker("loss_velocity", VelocityBreaker("loss_velocity", velocity_config))
        
        # Concentration breaker
        concentration_config = BreakerConfig(
            breaker_type=CircuitBreakerType.CONCENTRATION,
            threshold=0.50,  # 50% concentration ratio
            action=ActionType.ALERT,
            cooldown_minutes=30
        )
        self.add_breaker("concentration_risk", ConcentrationBreaker("concentration_risk", concentration_config))
        
        # Bankroll threshold breaker
        bankroll_config = BreakerConfig(
            breaker_type=CircuitBreakerType.BANKROLL_THRESHOLD,
            threshold=self.initial_bankroll * 0.50,  # 50% of initial bankroll
            action=ActionType.EMERGENCY_STOP,
            cooldown_minutes=0  # No cooldown for critical threshold
        )
        self.add_breaker("bankroll_threshold", BankrollThresholdBreaker("bankroll_threshold", bankroll_config))
    
    def add_breaker(self, breaker_id: str, breaker: CircuitBreaker):
        """Add a circuit breaker to the system"""
        self.breakers[breaker_id] = breaker
        logger.info(f"Circuit breaker added: {breaker_id}")
    
    def remove_breaker(self, breaker_id: str):
        """Remove a circuit breaker from the system"""
        if breaker_id in self.breakers:
            del self.breakers[breaker_id]
            logger.info(f"Circuit breaker removed: {breaker_id}")
    
    def check_all_breakers(self, context: Dict[str, Any] = None) -> List[BreakerTrigger]:
        """Check all circuit breakers and return any triggered"""
        if context is None:
            context = {}
        
        # Update daily tracking
        self._update_daily_tracking()
        
        # Calculate current metrics
        current_drawdown = (self.peak_bankroll - self.current_bankroll) / self.peak_bankroll
        loss_velocity = self._calculate_loss_velocity()
        
        # Build context
        full_context = {
            "bankroll": self.current_bankroll,
            "initial_bankroll": self.initial_bankroll,
            "peak_bankroll": self.peak_bankroll,
            "starting_bankroll": self._get_daily_starting_bankroll(),
            "trades_today": self.daily_trades,
            "recent_results": self._get_recent_results(),
            "total_loss_amount": self._get_consecutive_loss_amount(),
            "time_window_hours": 1,
            "recent_losses": self._get_recent_losses(),
            **context
        }
        
        triggered_breakers = []
        
        # Check each breaker
        for breaker_id, breaker in self.breakers.items():
            trigger = None
            
            if isinstance(breaker, DrawdownBreaker):
                trigger = breaker.check(current_drawdown, full_context)
            elif isinstance(breaker, ConsecutiveLossBreaker):
                trigger = breaker.check(self.consecutive_losses, full_context)
            elif isinstance(breaker, DailyLossBreaker):
                trigger = breaker.check(self.daily_pnl, full_context)
            elif isinstance(breaker, VelocityBreaker):
                trigger = breaker.check(loss_velocity, full_context)
            elif isinstance(breaker, ConcentrationBreaker):
                concentration_ratio = context.get("concentration_ratio", 0)
                trigger = breaker.check(concentration_ratio, full_context)
            elif isinstance(breaker, BankrollThresholdBreaker):
                trigger = breaker.check(self.current_bankroll, full_context)
            
            if trigger:
                triggered_breakers.append(trigger)
                self.trigger_history.append(trigger)
                
                # Execute action
                self._execute_action(trigger)
        
        return triggered_breakers
    
    def _execute_action(self, trigger: BreakerTrigger):
        """Execute circuit breaker action"""
        action = trigger.action
        
        if action == ActionType.ALERT:
            logger.warning(f"Circuit breaker alert: {trigger.message}")
        
        elif action == ActionType.REDUCE_STAKES:
            self._reduce_stakes()
            logger.warning(f"Stakes reduced due to: {trigger.message}")
        
        elif action == ActionType.HALT_BETTING:
            self.system_halted = True
            logger.critical(f"Betting halted due to: {trigger.message}")
        
        elif action == ActionType.EMERGENCY_STOP:
            self.emergency_stop = True
            self.system_halted = True
            logger.critical(f"EMERGENCY STOP activated: {trigger.message}")
        
        elif action == ActionType.FORCE_CLOSE:
            self._force_close_positions()
            logger.critical(f"Positions force closed due to: {trigger.message}")
    
    def _reduce_stakes(self):
        """Reduce stake sizes by 50%"""
        # This would integrate with the Kelly optimizer to reduce fractional factor
        # Implementation depends on integration with other systems
        logger.info("Stake reduction triggered - reducing position sizes by 50%")
    
    def _force_close_positions(self):
        """Force close all active positions"""
        # This would integrate with position management system
        # Implementation depends on integration with other systems
        logger.critical("Force closing all active positions")
    
    def update_bankroll(self, new_bankroll: float, bet_result: Optional[str] = None):
        """Update bankroll and betting state"""
        old_bankroll = self.current_bankroll
        self.current_bankroll = new_bankroll
        
        # Update peak bankroll
        if new_bankroll > self.peak_bankroll:
            self.peak_bankroll = new_bankroll
        
        # Track P&L
        pnl = new_bankroll - old_bankroll
        self.daily_pnl += pnl
        
        # Track consecutive losses
        if bet_result:
            if bet_result == "loss":
                self.consecutive_losses += 1
            elif bet_result in ["win", "push"]:
                self.consecutive_losses = 0
            
            self.last_bet_result = bet_result
            self.daily_trades += 1
        
        # Record performance
        self.performance_history.append({
            "timestamp": datetime.now(),
            "bankroll": new_bankroll,
            "pnl": pnl,
            "bet_result": bet_result,
            "consecutive_losses": self.consecutive_losses
        })
        
        logger.info(f"Bankroll updated: ${old_bankroll:.2f} -> ${new_bankroll:.2f} (P&L: {pnl:+.2f})")
    
    def _update_daily_tracking(self):
        """Update daily tracking variables"""
        current_date = datetime.now().date()
        
        if current_date != self.last_reset_date:
            # Reset daily counters
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.last_reset_date = current_date
            logger.info("Daily counters reset")
    
    def _get_daily_starting_bankroll(self) -> float:
        """Get starting bankroll for today"""
        # Find first entry for today
        today = datetime.now().date()
        
        for entry in reversed(self.performance_history):
            if entry["timestamp"].date() != today:
                continue
            return entry["bankroll"] - entry["pnl"]
        
        return self.current_bankroll
    
    def _calculate_loss_velocity(self) -> float:
        """Calculate recent loss velocity (loss per hour)"""
        cutoff_time = datetime.now() - timedelta(hours=1)
        
        recent_losses = [
            entry["pnl"] for entry in self.performance_history
            if entry["timestamp"] > cutoff_time and entry["pnl"] < 0
        ]
        
        return sum(abs(loss) for loss in recent_losses)
    
    def _get_recent_results(self, count: int = 10) -> List[str]:
        """Get recent bet results"""
        results = []
        for entry in reversed(self.performance_history):
            if entry.get("bet_result") and len(results) < count:
                results.append(entry["bet_result"])
        
        return list(reversed(results))
    
    def _get_consecutive_loss_amount(self) -> float:
        """Get total amount lost in consecutive losses"""
        loss_amount = 0.0
        
        for entry in reversed(self.performance_history):
            if entry.get("bet_result") == "loss":
                loss_amount += abs(entry["pnl"])
            elif entry.get("bet_result") in ["win", "push"]:
                break
        
        return loss_amount
    
    def _get_recent_losses(self, hours: int = 1) -> List[float]:
        """Get recent losses within time window"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        return [
            abs(entry["pnl"]) for entry in self.performance_history
            if entry["timestamp"] > cutoff_time and entry["pnl"] < 0
        ]
    
    def reset_system(self):
        """Reset system halt status"""
        self.system_halted = False
        self.emergency_stop = False
        
        # Reset all breakers
        for breaker in self.breakers.values():
            breaker.reset()
        
        logger.info("Circuit breaker system reset")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        current_drawdown = (self.peak_bankroll - self.current_bankroll) / self.peak_bankroll
        
        breaker_status = {}
        for breaker_id, breaker in self.breakers.items():
            breaker_status[breaker_id] = {
                "status": breaker.status.value,
                "enabled": breaker.config.enabled,
                "triggers": len(breaker.trigger_history),
                "last_trigger": breaker.last_trigger_time.isoformat() if breaker.last_trigger_time else None,
                "in_cooldown": breaker.is_in_cooldown()
            }
        
        return {
            "system_status": {
                "halted": self.system_halted,
                "emergency_stop": self.emergency_stop,
                "total_triggers": len(self.trigger_history)
            },
            "bankroll": {
                "current": self.current_bankroll,
                "initial": self.initial_bankroll,
                "peak": self.peak_bankroll,
                "drawdown": current_drawdown
            },
            "daily_stats": {
                "pnl": self.daily_pnl,
                "trades": self.daily_trades,
                "consecutive_losses": self.consecutive_losses
            },
            "breakers": breaker_status,
            "recent_triggers": [
                {
                    "breaker_id": t.breaker_id,
                    "trigger_time": t.trigger_time.isoformat(),
                    "action": t.action.value,
                    "message": t.message
                }
                for t in self.trigger_history[-5:]  # Last 5 triggers
            ]
        }


# Example usage
if __name__ == "__main__":
    # Initialize circuit breaker system
    cb_system = CircuitBreakerSystem(initial_bankroll=50000)
    
    # Simulate some losses
    cb_system.update_bankroll(48000, "loss")  # 2k loss
    cb_system.update_bankroll(46000, "loss")  # 2k loss  
    cb_system.update_bankroll(44000, "loss")  # 2k loss
    
    # Check breakers
    triggers = cb_system.check_all_breakers()
    
    if triggers:
        print(f"Triggered {len(triggers)} circuit breakers:")
        for trigger in triggers:
            print(f"- {trigger.breaker_id}: {trigger.message}")
    
    # Get system status
    status = cb_system.get_system_status()
    print(json.dumps(status, indent=2, default=str))