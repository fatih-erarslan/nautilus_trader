"""
Betting Limits and Controls for Sports Betting

Implements maximum exposure limits, drawdown controls, circuit breakers,
automated stop-loss mechanisms, and position sizing algorithms.
"""

import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import logging


logger = logging.getLogger(__name__)


class LimitType(Enum):
    """Types of betting limits"""
    PER_BET = "per_bet"
    PER_SPORT = "per_sport"
    PER_DAY = "per_day"
    PER_WEEK = "per_week"
    PER_MONTH = "per_month"
    TOTAL_EXPOSURE = "total_exposure"


class CircuitBreakerStatus(Enum):
    """Circuit breaker status"""
    ACTIVE = "active"
    TRIGGERED = "triggered"
    COOLDOWN = "cooldown"
    DISABLED = "disabled"


@dataclass
class BettingLimit:
    """Represents a betting limit"""
    limit_type: LimitType
    value: float
    current_usage: float = 0.0
    sport: Optional[str] = None
    reset_time: Optional[datetime.datetime] = None
    
    @property
    def remaining(self) -> float:
        """Calculate remaining limit"""
        return max(0, self.value - self.current_usage)
    
    @property
    def utilization_percentage(self) -> float:
        """Calculate utilization percentage"""
        return (self.current_usage / self.value * 100) if self.value > 0 else 0


@dataclass
class DrawdownLimit:
    """Drawdown limit configuration"""
    max_drawdown_percentage: float
    lookback_period_hours: int
    action: str  # 'stop', 'reduce', 'alert'
    reduction_factor: float = 0.5  # For 'reduce' action


@dataclass
class CircuitBreaker:
    """Circuit breaker configuration"""
    name: str
    trigger_condition: str  # 'consecutive_losses', 'loss_amount', 'loss_percentage'
    trigger_value: float
    cooldown_minutes: int
    status: CircuitBreakerStatus = CircuitBreakerStatus.ACTIVE
    triggered_at: Optional[datetime.datetime] = None
    trigger_count: int = 0


@dataclass
class PositionSizingRule:
    """Position sizing rule"""
    name: str
    base_percentage: float
    confidence_scaling: bool = True
    volatility_adjustment: bool = True
    max_percentage: float = 0.05
    min_percentage: float = 0.001


@dataclass
class BetRecord:
    """Record of a placed bet"""
    bet_id: str
    sport: str
    amount: float
    timestamp: datetime.datetime
    odds: float
    result: Optional[str] = None  # 'win', 'loss', 'push', 'pending'
    pnl: Optional[float] = None


class BettingLimitsController:
    """
    Controls betting limits, drawdowns, and position sizing for sports betting operations.
    """
    
    def __init__(self,
                 bankroll: float,
                 max_bet_percentage: float = 0.05,
                 max_daily_loss_percentage: float = 0.10,
                 max_drawdown_percentage: float = 0.20):
        """
        Initialize Betting Limits Controller
        
        Args:
            bankroll: Total bankroll
            max_bet_percentage: Maximum percentage per bet
            max_daily_loss_percentage: Maximum daily loss percentage
            max_drawdown_percentage: Maximum drawdown percentage
        """
        self.bankroll = bankroll
        self.initial_bankroll = bankroll
        
        # Initialize limits
        self.limits: Dict[str, BettingLimit] = {}
        self._initialize_default_limits(
            max_bet_percentage,
            max_daily_loss_percentage
        )
        
        # Drawdown controls
        self.drawdown_limits = [
            DrawdownLimit(max_drawdown_percentage, 24, 'stop'),
            DrawdownLimit(max_drawdown_percentage * 0.5, 6, 'reduce', 0.5),
            DrawdownLimit(max_drawdown_percentage * 0.3, 1, 'alert')
        ]
        
        # Circuit breakers
        self.circuit_breakers = [
            CircuitBreaker('consecutive_losses', 'consecutive_losses', 5, 60),
            CircuitBreaker('daily_loss', 'loss_percentage', 0.08, 1440),
            CircuitBreaker('rapid_loss', 'loss_amount', bankroll * 0.05, 30)
        ]
        
        # Position sizing rules
        self.position_sizing_rules = {
            'default': PositionSizingRule('default', 0.02),
            'conservative': PositionSizingRule('conservative', 0.01, max_percentage=0.03),
            'aggressive': PositionSizingRule('aggressive', 0.03, max_percentage=0.08),
            'kelly': PositionSizingRule('kelly', 0.025, confidence_scaling=True)
        }
        
        # Bet history
        self.bet_history: List[BetRecord] = []
        self.consecutive_losses = 0
        self.daily_stats = {}
        
    def _initialize_default_limits(self,
                                   max_bet_percentage: float,
                                   max_daily_loss_percentage: float):
        """Initialize default betting limits"""
        # Per bet limit
        self.limits['per_bet'] = BettingLimit(
            LimitType.PER_BET,
            self.bankroll * max_bet_percentage
        )
        
        # Daily limits
        self.limits['per_day'] = BettingLimit(
            LimitType.PER_DAY,
            self.bankroll * 0.20,  # 20% daily exposure
            reset_time=self._get_next_reset_time('daily')
        )
        
        # Sport-specific limits (can be customized)
        sports = ['football', 'basketball', 'baseball', 'soccer', 'tennis']
        for sport in sports:
            self.limits[f'per_sport_{sport}'] = BettingLimit(
                LimitType.PER_SPORT,
                self.bankroll * 0.10,  # 10% per sport
                sport=sport
            )
            
        # Total exposure limit
        self.limits['total_exposure'] = BettingLimit(
            LimitType.TOTAL_EXPOSURE,
            self.bankroll * 0.30  # 30% max total exposure
        )
        
    def check_bet_limits(self,
                         amount: float,
                         sport: str,
                         existing_exposure: float = 0
                         ) -> Tuple[bool, List[str]]:
        """
        Check if a bet amount is within all applicable limits
        
        Args:
            amount: Proposed bet amount
            sport: Sport category
            existing_exposure: Current total exposure
            
        Returns:
            Tuple of (is_allowed, list_of_violations)
        """
        violations = []
        
        # Check per-bet limit
        if amount > self.limits['per_bet'].remaining:
            violations.append(f"Exceeds per-bet limit: ${amount:.2f} > ${self.limits['per_bet'].remaining:.2f}")
            
        # Check daily limit
        if amount > self.limits['per_day'].remaining:
            violations.append(f"Exceeds daily limit: ${amount:.2f} > ${self.limits['per_day'].remaining:.2f}")
            
        # Check sport-specific limit
        sport_key = f'per_sport_{sport.lower()}'
        if sport_key in self.limits and amount > self.limits[sport_key].remaining:
            violations.append(
                f"Exceeds {sport} limit: ${amount:.2f} > ${self.limits[sport_key].remaining:.2f}"
            )
            
        # Check total exposure
        new_exposure = existing_exposure + amount
        if new_exposure > self.limits['total_exposure'].value:
            violations.append(
                f"Exceeds total exposure limit: ${new_exposure:.2f} > "
                f"${self.limits['total_exposure'].value:.2f}"
            )
            
        return len(violations) == 0, violations
    
    def check_circuit_breakers(self) -> Tuple[bool, List[str]]:
        """
        Check if any circuit breakers are triggered
        
        Returns:
            Tuple of (is_clear, list_of_triggered_breakers)
        """
        triggered = []
        current_time = datetime.datetime.now()
        
        for breaker in self.circuit_breakers:
            # Check if in cooldown
            if breaker.status == CircuitBreakerStatus.TRIGGERED:
                if breaker.triggered_at:
                    cooldown_end = breaker.triggered_at + datetime.timedelta(
                        minutes=breaker.cooldown_minutes
                    )
                    if current_time >= cooldown_end:
                        breaker.status = CircuitBreakerStatus.ACTIVE
                        breaker.triggered_at = None
                        logger.info(f"Circuit breaker '{breaker.name}' cooldown ended")
                    else:
                        remaining = (cooldown_end - current_time).total_seconds() / 60
                        triggered.append(
                            f"{breaker.name}: In cooldown for {remaining:.1f} more minutes"
                        )
                        
            # Check trigger conditions
            if breaker.status == CircuitBreakerStatus.ACTIVE:
                if self._check_breaker_condition(breaker):
                    breaker.status = CircuitBreakerStatus.TRIGGERED
                    breaker.triggered_at = current_time
                    breaker.trigger_count += 1
                    triggered.append(
                        f"{breaker.name}: Triggered (count: {breaker.trigger_count})"
                    )
                    logger.warning(f"Circuit breaker '{breaker.name}' triggered")
                    
        return len(triggered) == 0, triggered
    
    def _check_breaker_condition(self, breaker: CircuitBreaker) -> bool:
        """Check if a circuit breaker condition is met"""
        if breaker.trigger_condition == 'consecutive_losses':
            return self.consecutive_losses >= breaker.trigger_value
            
        elif breaker.trigger_condition == 'loss_percentage':
            recent_pnl = self._calculate_recent_pnl(hours=1)
            loss_percentage = abs(recent_pnl / self.bankroll) if recent_pnl < 0 else 0
            return loss_percentage >= breaker.trigger_value
            
        elif breaker.trigger_condition == 'loss_amount':
            recent_pnl = self._calculate_recent_pnl(hours=0.5)  # 30 minutes
            return recent_pnl <= -breaker.trigger_value
            
        return False
    
    def check_drawdown_limits(self) -> Tuple[str, Optional[float]]:
        """
        Check drawdown limits and return required action
        
        Returns:
            Tuple of (action, reduction_factor)
            Actions: 'continue', 'reduce', 'stop', 'alert'
        """
        for limit in self.drawdown_limits:
            drawdown = self._calculate_drawdown(limit.lookback_period_hours)
            
            if drawdown >= limit.max_drawdown_percentage:
                logger.warning(
                    f"Drawdown limit triggered: {drawdown:.2%} over "
                    f"{limit.lookback_period_hours}h (limit: {limit.max_drawdown_percentage:.2%})"
                )
                
                if limit.action == 'stop':
                    return 'stop', None
                elif limit.action == 'reduce':
                    return 'reduce', limit.reduction_factor
                elif limit.action == 'alert':
                    return 'alert', None
                    
        return 'continue', None
    
    def calculate_position_size(self,
                                edge: float,
                                confidence: float,
                                volatility: float,
                                sizing_rule: str = 'default'
                                ) -> float:
        """
        Calculate position size based on rules and current conditions
        
        Args:
            edge: Expected edge (percentage)
            confidence: Confidence level (0-1)
            volatility: Market volatility measure
            sizing_rule: Name of sizing rule to use
            
        Returns:
            Position size as percentage of bankroll
        """
        rule = self.position_sizing_rules.get(sizing_rule, self.position_sizing_rules['default'])
        
        # Base size
        size = rule.base_percentage
        
        # Apply confidence scaling
        if rule.confidence_scaling:
            size *= confidence
            
        # Apply volatility adjustment
        if rule.volatility_adjustment:
            # Reduce size in high volatility
            volatility_factor = 1 / (1 + volatility)
            size *= volatility_factor
            
        # Apply edge-based scaling
        if edge > 0:
            edge_factor = min(1 + edge, 2.0)  # Cap at 2x
            size *= edge_factor
            
        # Apply limits
        size = max(rule.min_percentage, min(size, rule.max_percentage))
        
        # Check against current limits
        max_allowed = self.limits['per_bet'].remaining / self.bankroll
        size = min(size, max_allowed)
        
        return size
    
    def record_bet(self, bet_record: BetRecord):
        """Record a placed bet and update limits"""
        self.bet_history.append(bet_record)
        
        # Update limits
        self.limits['per_bet'].current_usage = bet_record.amount
        self.limits['per_day'].current_usage += bet_record.amount
        
        sport_key = f'per_sport_{bet_record.sport.lower()}'
        if sport_key in self.limits:
            self.limits[sport_key].current_usage += bet_record.amount
            
        # Update daily stats
        date_key = bet_record.timestamp.date().isoformat()
        if date_key not in self.daily_stats:
            self.daily_stats[date_key] = {
                'bets': 0,
                'volume': 0,
                'pnl': 0,
                'wins': 0,
                'losses': 0
            }
        self.daily_stats[date_key]['bets'] += 1
        self.daily_stats[date_key]['volume'] += bet_record.amount
        
    def update_bet_result(self, bet_id: str, result: str, pnl: float):
        """Update bet result and check stop-loss conditions"""
        for bet in self.bet_history:
            if bet.bet_id == bet_id:
                bet.result = result
                bet.pnl = pnl
                
                # Update consecutive losses
                if result == 'loss':
                    self.consecutive_losses += 1
                else:
                    self.consecutive_losses = 0
                    
                # Update daily stats
                date_key = bet.timestamp.date().isoformat()
                if date_key in self.daily_stats:
                    self.daily_stats[date_key]['pnl'] += pnl
                    if result == 'win':
                        self.daily_stats[date_key]['wins'] += 1
                    elif result == 'loss':
                        self.daily_stats[date_key]['losses'] += 1
                        
                # Update bankroll
                self.bankroll += pnl
                
                break
                
    def _calculate_recent_pnl(self, hours: float) -> float:
        """Calculate P&L over recent hours"""
        cutoff_time = datetime.datetime.now() - datetime.timedelta(hours=hours)
        recent_pnl = sum(
            bet.pnl or 0 for bet in self.bet_history
            if bet.timestamp >= cutoff_time and bet.pnl is not None
        )
        return recent_pnl
    
    def _calculate_drawdown(self, hours: int) -> float:
        """Calculate drawdown percentage over period"""
        if hours == 0:
            # Current drawdown from initial bankroll
            return (self.initial_bankroll - self.bankroll) / self.initial_bankroll
            
        cutoff_time = datetime.datetime.now() - datetime.timedelta(hours=hours)
        
        # Find peak bankroll in period
        peak_bankroll = self.bankroll
        running_bankroll = self.bankroll
        
        # Work backwards through bet history
        for bet in reversed(self.bet_history):
            if bet.timestamp < cutoff_time:
                break
                
            if bet.pnl is not None:
                running_bankroll -= bet.pnl
                peak_bankroll = max(peak_bankroll, running_bankroll)
                
        # Calculate drawdown from peak
        if peak_bankroll > 0:
            return (peak_bankroll - self.bankroll) / peak_bankroll
        return 0
    
    def _get_next_reset_time(self, period: str) -> datetime.datetime:
        """Get next reset time for periodic limits"""
        now = datetime.datetime.now()
        
        if period == 'daily':
            # Reset at midnight
            tomorrow = now.date() + datetime.timedelta(days=1)
            return datetime.datetime.combine(tomorrow, datetime.time.min)
        elif period == 'weekly':
            # Reset on Monday
            days_until_monday = (7 - now.weekday()) % 7
            if days_until_monday == 0:
                days_until_monday = 7
            next_monday = now.date() + datetime.timedelta(days=days_until_monday)
            return datetime.datetime.combine(next_monday, datetime.time.min)
            
        return now
    
    def reset_periodic_limits(self):
        """Reset limits that have reached their reset time"""
        now = datetime.datetime.now()
        
        for limit in self.limits.values():
            if limit.reset_time and now >= limit.reset_time:
                limit.current_usage = 0
                
                # Set next reset time
                if limit.limit_type == LimitType.PER_DAY:
                    limit.reset_time = self._get_next_reset_time('daily')
                elif limit.limit_type == LimitType.PER_WEEK:
                    limit.reset_time = self._get_next_reset_time('weekly')
                    
    def get_limits_summary(self) -> Dict:
        """Get summary of all limits and their current status"""
        return {
            'bankroll': self.bankroll,
            'limits': {
                name: {
                    'type': limit.limit_type.value,
                    'value': limit.value,
                    'used': limit.current_usage,
                    'remaining': limit.remaining,
                    'utilization': f"{limit.utilization_percentage:.1f}%"
                }
                for name, limit in self.limits.items()
            },
            'circuit_breakers': [
                {
                    'name': breaker.name,
                    'status': breaker.status.value,
                    'trigger_count': breaker.trigger_count,
                    'triggered_at': breaker.triggered_at.isoformat() if breaker.triggered_at else None
                }
                for breaker in self.circuit_breakers
            ],
            'consecutive_losses': self.consecutive_losses,
            'current_drawdown': f"{self._calculate_drawdown(0):.2%}"
        }