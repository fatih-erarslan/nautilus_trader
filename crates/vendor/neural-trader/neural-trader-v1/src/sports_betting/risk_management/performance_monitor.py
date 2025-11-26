"""
Performance Monitoring for Sports Betting

Implements real-time P&L tracking, Sharpe ratio calculation,
maximum drawdown alerts, and risk-adjusted returns monitoring.
"""

import datetime
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import pandas as pd
import logging


logger = logging.getLogger(__name__)


class TimeFrame(Enum):
    """Time frames for performance analysis"""
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    YEARLY = "yearly"
    ALL_TIME = "all_time"


class AlertType(Enum):
    """Performance alert types"""
    DRAWDOWN = "drawdown"
    LOSING_STREAK = "losing_streak"
    VOLATILITY = "volatility"
    UNDERPERFORMANCE = "underperformance"
    RISK_BREACH = "risk_breach"


@dataclass
class PerformanceMetrics:
    """Performance metrics snapshot"""
    timestamp: datetime.datetime
    total_pnl: float
    win_rate: float
    average_win: float
    average_loss: float
    profit_factor: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    current_drawdown: float
    consecutive_wins: int
    consecutive_losses: int
    total_bets: int
    winning_bets: int
    losing_bets: int
    roi: float
    volatility: float


@dataclass
class DrawdownInfo:
    """Drawdown information"""
    peak_value: float
    trough_value: float
    current_value: float
    drawdown_amount: float
    drawdown_percentage: float
    start_date: datetime.datetime
    trough_date: Optional[datetime.datetime]
    recovery_date: Optional[datetime.datetime]
    duration_days: int
    is_active: bool


@dataclass
class PerformanceAlert:
    """Performance alert"""
    alert_id: str
    alert_type: AlertType
    severity: str  # 'info', 'warning', 'critical'
    message: str
    timestamp: datetime.datetime
    metrics: Dict[str, float]
    resolved: bool = False
    resolved_at: Optional[datetime.datetime] = None


@dataclass
class BettingTransaction:
    """Record of a betting transaction"""
    transaction_id: str
    timestamp: datetime.datetime
    sport: str
    event: str
    selection: str
    bet_type: str
    stake: float
    odds: float
    result: Optional[str] = None  # 'win', 'loss', 'push', 'void'
    pnl: Optional[float] = None
    running_balance: Optional[float] = None


class PerformanceMonitor:
    """
    Monitors and analyzes betting performance with real-time metrics
    and risk-adjusted return calculations.
    """
    
    def __init__(self,
                 initial_bankroll: float,
                 risk_free_rate: float = 0.02,
                 target_return: float = 0.20,
                 max_drawdown_threshold: float = 0.20):
        """
        Initialize Performance Monitor
        
        Args:
            initial_bankroll: Starting bankroll
            risk_free_rate: Annual risk-free rate for Sharpe calculations
            target_return: Target annual return
            max_drawdown_threshold: Maximum acceptable drawdown
        """
        self.initial_bankroll = initial_bankroll
        self.current_bankroll = initial_bankroll
        self.risk_free_rate = risk_free_rate
        self.target_return = target_return
        self.max_drawdown_threshold = max_drawdown_threshold
        
        # Transaction history
        self.transactions: List[BettingTransaction] = []
        self.transaction_index: Dict[str, BettingTransaction] = {}
        
        # Performance tracking
        self.daily_returns: deque = deque(maxlen=365)
        self.hourly_pnl: deque = deque(maxlen=24*7)  # 1 week of hourly data
        self.balance_history: List[Tuple[datetime.datetime, float]] = [
            (datetime.datetime.now(), initial_bankroll)
        ]
        
        # Drawdown tracking
        self.peak_balance = initial_bankroll
        self.current_drawdown_info: Optional[DrawdownInfo] = None
        self.historical_drawdowns: List[DrawdownInfo] = []
        
        # Alert management
        self.active_alerts: Dict[str, PerformanceAlert] = {}
        self.alert_history: List[PerformanceAlert] = []
        self.alert_thresholds = {
            'max_drawdown': max_drawdown_threshold,
            'max_consecutive_losses': 5,
            'min_sharpe_ratio': 0.5,
            'max_volatility': 0.30
        }
        
        # Performance by category
        self.performance_by_sport: Dict[str, Dict] = defaultdict(
            lambda: {'pnl': 0, 'bets': 0, 'wins': 0}
        )
        self.performance_by_bet_type: Dict[str, Dict] = defaultdict(
            lambda: {'pnl': 0, 'bets': 0, 'wins': 0}
        )
        
        # Initialize metrics
        self._last_metrics_update = datetime.datetime.now()
        self._cached_metrics: Optional[PerformanceMetrics] = None
        
    def record_transaction(self, transaction: BettingTransaction) -> bool:
        """
        Record a betting transaction and update performance metrics
        
        Args:
            transaction: Betting transaction to record
            
        Returns:
            True if recorded successfully
        """
        # Validate transaction
        if transaction.transaction_id in self.transaction_index:
            logger.warning(f"Transaction {transaction.transaction_id} already exists")
            return False
            
        # Add to history
        self.transactions.append(transaction)
        self.transaction_index[transaction.transaction_id] = transaction
        
        # Update bankroll and balance history
        if transaction.pnl is not None:
            self.current_bankroll += transaction.pnl
            transaction.running_balance = self.current_bankroll
            self.balance_history.append(
                (transaction.timestamp, self.current_bankroll)
            )
            
            # Update category performance
            if transaction.result in ['win', 'loss']:
                sport_stats = self.performance_by_sport[transaction.sport]
                sport_stats['pnl'] += transaction.pnl
                sport_stats['bets'] += 1
                if transaction.result == 'win':
                    sport_stats['wins'] += 1
                    
                bet_type_stats = self.performance_by_bet_type[transaction.bet_type]
                bet_type_stats['pnl'] += transaction.pnl
                bet_type_stats['bets'] += 1
                if transaction.result == 'win':
                    bet_type_stats['wins'] += 1
                    
            # Update returns tracking
            self._update_returns_tracking(transaction)
            
            # Check for alerts
            self._check_performance_alerts()
            
        return True
        
    def update_transaction_result(self,
                                  transaction_id: str,
                                  result: str,
                                  pnl: float) -> bool:
        """
        Update the result of a pending transaction
        
        Args:
            transaction_id: Transaction identifier
            result: Result ('win', 'loss', 'push', 'void')
            pnl: Profit/loss amount
            
        Returns:
            True if updated successfully
        """
        if transaction_id not in self.transaction_index:
            logger.warning(f"Transaction {transaction_id} not found")
            return False
            
        transaction = self.transaction_index[transaction_id]
        transaction.result = result
        transaction.pnl = pnl
        
        # Update bankroll
        self.current_bankroll += pnl
        transaction.running_balance = self.current_bankroll
        self.balance_history.append(
            (datetime.datetime.now(), self.current_bankroll)
        )
        
        # Update category performance
        if result in ['win', 'loss']:
            sport_stats = self.performance_by_sport[transaction.sport]
            sport_stats['pnl'] += pnl
            sport_stats['bets'] += 1
            if result == 'win':
                sport_stats['wins'] += 1
                
            bet_type_stats = self.performance_by_bet_type[transaction.bet_type]
            bet_type_stats['pnl'] += pnl
            bet_type_stats['bets'] += 1
            if result == 'win':
                bet_type_stats['wins'] += 1
                
        # Update tracking
        self._update_returns_tracking(transaction)
        self._check_performance_alerts()
        
        # Invalidate cached metrics
        self._cached_metrics = None
        
        return True
        
    def get_performance_metrics(self,
                                timeframe: TimeFrame = TimeFrame.ALL_TIME,
                                force_recalculate: bool = False
                                ) -> PerformanceMetrics:
        """
        Get comprehensive performance metrics
        
        Args:
            timeframe: Time period for metrics
            force_recalculate: Force recalculation of metrics
            
        Returns:
            PerformanceMetrics object
        """
        # Use cache if available and recent
        if (not force_recalculate and 
            self._cached_metrics and
            (datetime.datetime.now() - self._last_metrics_update).seconds < 60):
            return self._cached_metrics
            
        # Filter transactions by timeframe
        filtered_transactions = self._filter_transactions_by_timeframe(timeframe)
        
        if not filtered_transactions:
            return self._create_empty_metrics()
            
        # Calculate basic metrics
        total_bets = len([t for t in filtered_transactions if t.result])
        wins = [t for t in filtered_transactions if t.result == 'win']
        losses = [t for t in filtered_transactions if t.result == 'loss']
        
        winning_bets = len(wins)
        losing_bets = len(losses)
        win_rate = winning_bets / total_bets if total_bets > 0 else 0
        
        # Calculate P&L metrics
        total_pnl = sum(t.pnl or 0 for t in filtered_transactions)
        average_win = np.mean([t.pnl for t in wins]) if wins else 0
        average_loss = np.mean([t.pnl for t in losses]) if losses else 0
        
        # Profit factor
        gross_wins = sum(t.pnl for t in wins if t.pnl > 0)
        gross_losses = abs(sum(t.pnl for t in losses if t.pnl < 0))
        profit_factor = gross_wins / gross_losses if gross_losses > 0 else float('inf')
        
        # Calculate returns series
        returns = self._calculate_returns_series(filtered_transactions)
        
        # Risk metrics
        sharpe_ratio = self._calculate_sharpe_ratio(returns)
        sortino_ratio = self._calculate_sortino_ratio(returns)
        calmar_ratio = self._calculate_calmar_ratio(returns, total_pnl)
        
        # Drawdown metrics
        max_dd, current_dd = self._calculate_drawdown_metrics()
        
        # Streak tracking
        consecutive_wins, consecutive_losses = self._calculate_streaks()
        
        # ROI and volatility
        roi = (self.current_bankroll - self.initial_bankroll) / self.initial_bankroll
        volatility = np.std(returns) if len(returns) > 1 else 0
        
        metrics = PerformanceMetrics(
            timestamp=datetime.datetime.now(),
            total_pnl=total_pnl,
            win_rate=win_rate,
            average_win=average_win,
            average_loss=average_loss,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            max_drawdown=max_dd,
            current_drawdown=current_dd,
            consecutive_wins=consecutive_wins,
            consecutive_losses=consecutive_losses,
            total_bets=total_bets,
            winning_bets=winning_bets,
            losing_bets=losing_bets,
            roi=roi,
            volatility=volatility
        )
        
        # Cache metrics
        self._cached_metrics = metrics
        self._last_metrics_update = datetime.datetime.now()
        
        return metrics
        
    def _calculate_sharpe_ratio(self, returns: List[float]) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) < 2:
            return 0.0
            
        # Annualize returns (assuming daily returns)
        mean_return = np.mean(returns) * 252
        std_return = np.std(returns) * np.sqrt(252)
        
        if std_return == 0:
            return 0.0
            
        return (mean_return - self.risk_free_rate) / std_return
        
    def _calculate_sortino_ratio(self, returns: List[float]) -> float:
        """Calculate Sortino ratio (downside deviation)"""
        if len(returns) < 2:
            return 0.0
            
        # Calculate downside deviation
        negative_returns = [r for r in returns if r < 0]
        if not negative_returns:
            return float('inf')  # No downside
            
        downside_std = np.std(negative_returns) * np.sqrt(252)
        
        if downside_std == 0:
            return 0.0
            
        mean_return = np.mean(returns) * 252
        return (mean_return - self.risk_free_rate) / downside_std
        
    def _calculate_calmar_ratio(self, returns: List[float], total_pnl: float) -> float:
        """Calculate Calmar ratio (return over max drawdown)"""
        max_dd, _ = self._calculate_drawdown_metrics()
        
        if max_dd == 0:
            return float('inf')
            
        # Annualized return
        if self.transactions:
            days = (self.transactions[-1].timestamp - self.transactions[0].timestamp).days
            if days > 0:
                annual_return = (total_pnl / self.initial_bankroll) * (365 / days)
                return annual_return / max_dd
                
        return 0.0
        
    def _calculate_drawdown_metrics(self) -> Tuple[float, float]:
        """Calculate maximum and current drawdown"""
        if not self.balance_history:
            return 0.0, 0.0
            
        max_drawdown = 0.0
        current_peak = self.initial_bankroll
        
        for timestamp, balance in self.balance_history:
            if balance > current_peak:
                current_peak = balance
                
            drawdown = (current_peak - balance) / current_peak
            max_drawdown = max(max_drawdown, drawdown)
            
        # Current drawdown
        current_balance = self.balance_history[-1][1]
        current_drawdown = (self.peak_balance - current_balance) / self.peak_balance
        
        # Update peak if needed
        if current_balance > self.peak_balance:
            self.peak_balance = current_balance
            current_drawdown = 0.0
            
        return max_drawdown, current_drawdown
        
    def _calculate_streaks(self) -> Tuple[int, int]:
        """Calculate current winning and losing streaks"""
        if not self.transactions:
            return 0, 0
            
        consecutive_wins = 0
        consecutive_losses = 0
        current_streak_type = None
        current_streak = 0
        
        for transaction in reversed(self.transactions):
            if transaction.result == 'win':
                if current_streak_type == 'win':
                    current_streak += 1
                else:
                    current_streak_type = 'win'
                    current_streak = 1
                    break
            elif transaction.result == 'loss':
                if current_streak_type == 'loss':
                    current_streak += 1
                else:
                    current_streak_type = 'loss'
                    current_streak = 1
                    break
                    
        if current_streak_type == 'win':
            consecutive_wins = current_streak
        elif current_streak_type == 'loss':
            consecutive_losses = current_streak
            
        return consecutive_wins, consecutive_losses
        
    def _update_returns_tracking(self, transaction: BettingTransaction):
        """Update returns tracking data"""
        if transaction.pnl is not None and self.current_bankroll > 0:
            # Calculate return percentage
            return_pct = transaction.pnl / (self.current_bankroll - transaction.pnl)
            
            # Add to daily returns
            self.daily_returns.append(return_pct)
            
            # Add to hourly P&L
            self.hourly_pnl.append((transaction.timestamp, transaction.pnl))
            
    def _check_performance_alerts(self):
        """Check for performance alerts"""
        metrics = self.get_performance_metrics()
        
        # Check drawdown alert
        if metrics.current_drawdown > self.alert_thresholds['max_drawdown']:
            self._create_alert(
                AlertType.DRAWDOWN,
                'critical',
                f"Current drawdown {metrics.current_drawdown:.1%} exceeds threshold",
                {'drawdown': metrics.current_drawdown}
            )
            
        # Check losing streak
        if metrics.consecutive_losses >= self.alert_thresholds['max_consecutive_losses']:
            self._create_alert(
                AlertType.LOSING_STREAK,
                'warning',
                f"Consecutive losses: {metrics.consecutive_losses}",
                {'streak': metrics.consecutive_losses}
            )
            
        # Check Sharpe ratio
        if metrics.sharpe_ratio < self.alert_thresholds['min_sharpe_ratio']:
            self._create_alert(
                AlertType.UNDERPERFORMANCE,
                'warning',
                f"Sharpe ratio {metrics.sharpe_ratio:.2f} below threshold",
                {'sharpe_ratio': metrics.sharpe_ratio}
            )
            
        # Check volatility
        if metrics.volatility > self.alert_thresholds['max_volatility']:
            self._create_alert(
                AlertType.VOLATILITY,
                'warning',
                f"Volatility {metrics.volatility:.1%} exceeds threshold",
                {'volatility': metrics.volatility}
            )
            
    def _create_alert(self,
                      alert_type: AlertType,
                      severity: str,
                      message: str,
                      metrics: Dict[str, float]):
        """Create a performance alert"""
        alert_id = f"{alert_type.value}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Check if similar alert already active
        for active_alert in self.active_alerts.values():
            if active_alert.alert_type == alert_type and not active_alert.resolved:
                return  # Don't create duplicate
                
        alert = PerformanceAlert(
            alert_id=alert_id,
            alert_type=alert_type,
            severity=severity,
            message=message,
            timestamp=datetime.datetime.now(),
            metrics=metrics
        )
        
        self.active_alerts[alert_id] = alert
        logger.warning(f"Performance alert: {message}")
        
    def _filter_transactions_by_timeframe(self,
                                          timeframe: TimeFrame
                                          ) -> List[BettingTransaction]:
        """Filter transactions by timeframe"""
        if timeframe == TimeFrame.ALL_TIME:
            return self.transactions
            
        now = datetime.datetime.now()
        
        timeframe_deltas = {
            TimeFrame.HOURLY: datetime.timedelta(hours=1),
            TimeFrame.DAILY: datetime.timedelta(days=1),
            TimeFrame.WEEKLY: datetime.timedelta(weeks=1),
            TimeFrame.MONTHLY: datetime.timedelta(days=30),
            TimeFrame.YEARLY: datetime.timedelta(days=365)
        }
        
        if timeframe in timeframe_deltas:
            cutoff_time = now - timeframe_deltas[timeframe]
            return [t for t in self.transactions if t.timestamp >= cutoff_time]
            
        return self.transactions
        
    def _calculate_returns_series(self,
                                  transactions: List[BettingTransaction]
                                  ) -> List[float]:
        """Calculate returns series from transactions"""
        returns = []
        
        for i, transaction in enumerate(transactions):
            if transaction.pnl is not None and i > 0:
                prev_balance = transactions[i-1].running_balance or self.initial_bankroll
                if prev_balance > 0:
                    return_pct = transaction.pnl / prev_balance
                    returns.append(return_pct)
                    
        return returns
        
    def _create_empty_metrics(self) -> PerformanceMetrics:
        """Create empty metrics object"""
        return PerformanceMetrics(
            timestamp=datetime.datetime.now(),
            total_pnl=0.0,
            win_rate=0.0,
            average_win=0.0,
            average_loss=0.0,
            profit_factor=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            max_drawdown=0.0,
            current_drawdown=0.0,
            consecutive_wins=0,
            consecutive_losses=0,
            total_bets=0,
            winning_bets=0,
            losing_bets=0,
            roi=0.0,
            volatility=0.0
        )
        
    def get_performance_summary(self) -> Dict:
        """Get comprehensive performance summary"""
        metrics = self.get_performance_metrics()
        
        return {
            'current_bankroll': self.current_bankroll,
            'total_pnl': metrics.total_pnl,
            'roi': f"{metrics.roi:.1%}",
            'win_rate': f"{metrics.win_rate:.1%}",
            'sharpe_ratio': metrics.sharpe_ratio,
            'max_drawdown': f"{metrics.max_drawdown:.1%}",
            'current_drawdown': f"{metrics.current_drawdown:.1%}",
            'total_bets': metrics.total_bets,
            'active_alerts': len([a for a in self.active_alerts.values() if not a.resolved]),
            'performance_by_sport': dict(self.performance_by_sport),
            'performance_by_bet_type': dict(self.performance_by_bet_type)
        }