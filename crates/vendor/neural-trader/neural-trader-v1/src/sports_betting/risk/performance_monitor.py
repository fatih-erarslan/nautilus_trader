"""
Performance Monitoring for Sports Betting Operations

Comprehensive performance tracking including:
- Real-time P&L tracking
- Sharpe ratio calculations
- Risk-adjusted returns
- Alert system for threshold breaches
- Advanced performance analytics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime, timedelta
import json
from scipy import stats
import warnings

logger = logging.getLogger(__name__)


class AlertType(Enum):
    """Performance alert types"""
    POOR_PERFORMANCE = "poor_performance"
    HIGH_VOLATILITY = "high_volatility"
    DRAWDOWN_WARNING = "drawdown_warning"
    WIN_RATE_DECLINE = "win_rate_decline"
    SHARPE_DECLINE = "sharpe_decline"
    STREAK_ALERT = "streak_alert"
    VARIANCE_SPIKE = "variance_spike"


class TimeFrame(Enum):
    """Performance analysis time frames"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"
    ALL_TIME = "all_time"


@dataclass
class BettingTransaction:
    """Individual betting transaction"""
    transaction_id: str
    timestamp: datetime
    sport: str
    event: str
    selection: str
    bet_type: str
    stake: float
    odds: float
    result: Optional[str] = None  # win, loss, push, void, pending
    pnl: float = 0.0
    commission: float = 0.0
    
    def is_settled(self) -> bool:
        """Check if transaction is settled"""
        return self.result in ["win", "loss", "push", "void"]
    
    def get_roi(self) -> float:
        """Get return on investment"""
        if self.stake <= 0:
            return 0.0
        return self.pnl / self.stake


@dataclass
class PerformanceAlert:
    """Performance monitoring alert"""
    alert_id: str
    alert_type: AlertType
    timestamp: datetime
    severity: str  # low, medium, high, critical
    message: str
    current_value: float
    threshold: float
    metric_name: str
    resolved: bool = False
    resolution_time: Optional[datetime] = None


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics"""
    # Basic metrics
    total_pnl: float
    total_stakes: float
    total_roi: float
    win_rate: float
    avg_win: float
    avg_loss: float
    
    # Risk metrics
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    current_drawdown: float
    volatility: float
    
    # Streaks
    current_streak: int
    current_streak_type: str  # win, loss
    longest_winning_streak: int
    longest_losing_streak: int
    
    # Advanced metrics
    profit_factor: float
    kelly_criterion: float
    expectancy: float
    consecutive_losses: int
    consecutive_wins: int
    
    # Time-based
    days_active: int
    trades_per_day: float
    best_day: float
    worst_day: float


class PerformanceMonitor:
    """
    Comprehensive performance monitoring system for sports betting
    with real-time analytics and alerting
    """
    
    def __init__(self,
                 initial_bankroll: float,
                 risk_free_rate: float = 0.02,
                 target_return: float = 0.20,
                 max_drawdown_threshold: float = 0.15,
                 min_sharpe_threshold: float = 1.0,
                 min_win_rate_threshold: float = 0.52):
        """
        Initialize Performance Monitor
        
        Args:
            initial_bankroll: Starting bankroll
            risk_free_rate: Risk-free rate for Sharpe calculation
            target_return: Target annual return
            max_drawdown_threshold: Maximum acceptable drawdown
            min_sharpe_threshold: Minimum acceptable Sharpe ratio
            min_win_rate_threshold: Minimum acceptable win rate
        """
        self.initial_bankroll = initial_bankroll
        self.current_bankroll = initial_bankroll
        self.peak_bankroll = initial_bankroll
        self.risk_free_rate = risk_free_rate
        self.target_return = target_return
        self.max_drawdown_threshold = max_drawdown_threshold
        self.min_sharpe_threshold = min_sharpe_threshold
        self.min_win_rate_threshold = min_win_rate_threshold
        
        # Transaction tracking
        self.transactions: List[BettingTransaction] = []
        self.daily_pnl: Dict[str, float] = {}
        self.bankroll_history: List[Tuple[datetime, float]] = [(datetime.now(), initial_bankroll)]
        
        # Performance tracking
        self.performance_history: List[PerformanceMetrics] = []
        self.active_alerts: Dict[str, PerformanceAlert] = {}
        
        # Streak tracking
        self.current_streak = 0
        self.current_streak_type = ""
        self.consecutive_wins = 0
        self.consecutive_losses = 0
        
        logger.info(f"Performance monitor initialized with ${initial_bankroll:,.2f} bankroll")
    
    def record_transaction(self, transaction: BettingTransaction):
        """Record a new betting transaction"""
        self.transactions.append(transaction)
        
        # Update bankroll if settled
        if transaction.is_settled():
            self.current_bankroll += transaction.pnl
            self.bankroll_history.append((transaction.timestamp, self.current_bankroll))
            
            # Update peak bankroll
            if self.current_bankroll > self.peak_bankroll:
                self.peak_bankroll = self.current_bankroll
            
            # Update daily P&L
            date_key = transaction.timestamp.date().isoformat()
            self.daily_pnl[date_key] = self.daily_pnl.get(date_key, 0) + transaction.pnl
            
            # Update streaks
            self._update_streaks(transaction.result)
            
            logger.info(f"Transaction recorded: {transaction.transaction_id} - {transaction.result} - P&L: ${transaction.pnl:.2f}")
    
    def update_transaction_result(self, transaction_id: str, result: str, pnl: float):
        """Update the result of an existing transaction"""
        for transaction in self.transactions:
            if transaction.transaction_id == transaction_id:
                transaction.result = result
                transaction.pnl = pnl
                
                # Update bankroll and tracking
                self.current_bankroll += pnl
                self.bankroll_history.append((datetime.now(), self.current_bankroll))
                
                if self.current_bankroll > self.peak_bankroll:
                    self.peak_bankroll = self.current_bankroll
                
                # Update daily P&L
                date_key = transaction.timestamp.date().isoformat()
                self.daily_pnl[date_key] = self.daily_pnl.get(date_key, 0) + pnl
                
                # Update streaks
                self._update_streaks(result)
                
                logger.info(f"Transaction updated: {transaction_id} - {result} - P&L: ${pnl:.2f}")
                return True
        
        logger.warning(f"Transaction not found for update: {transaction_id}")
        return False
    
    def calculate_performance_metrics(self, timeframe: TimeFrame = TimeFrame.ALL_TIME) -> PerformanceMetrics:
        """
        Calculate comprehensive performance metrics
        
        Args:
            timeframe: Time frame for analysis
            
        Returns:
            PerformanceMetrics object
        """
        # Filter transactions by timeframe
        filtered_transactions = self._filter_transactions_by_timeframe(timeframe)
        settled_transactions = [t for t in filtered_transactions if t.is_settled()]
        
        if not settled_transactions:
            return self._create_empty_metrics()
        
        # Basic calculations
        total_pnl = sum(t.pnl for t in settled_transactions)
        total_stakes = sum(t.stake for t in settled_transactions)
        
        wins = [t for t in settled_transactions if t.result == "win"]
        losses = [t for t in settled_transactions if t.result == "loss"]
        
        win_rate = len(wins) / len(settled_transactions) if settled_transactions else 0
        avg_win = np.mean([t.pnl for t in wins]) if wins else 0
        avg_loss = np.mean([abs(t.pnl) for t in losses]) if losses else 0
        
        # ROI calculation
        total_roi = total_pnl / total_stakes if total_stakes > 0 else 0
        
        # Risk metrics
        daily_returns = self._calculate_daily_returns(timeframe)
        sharpe_ratio = self._calculate_sharpe_ratio(daily_returns)
        sortino_ratio = self._calculate_sortino_ratio(daily_returns)
        calmar_ratio = self._calculate_calmar_ratio(daily_returns)
        
        max_drawdown = self._calculate_max_drawdown(timeframe)
        current_drawdown = (self.peak_bankroll - self.current_bankroll) / self.peak_bankroll
        volatility = np.std(daily_returns) * np.sqrt(252) if len(daily_returns) > 1 else 0
        
        # Streak analysis
        streak_info = self._analyze_streaks(settled_transactions)
        
        # Advanced metrics
        profit_factor = self._calculate_profit_factor(wins, losses)
        kelly_criterion = self._calculate_kelly_criterion(settled_transactions)
        expectancy = self._calculate_expectancy(wins, losses, win_rate)
        
        # Time-based metrics
        if settled_transactions:
            start_date = min(t.timestamp for t in settled_transactions).date()
            end_date = max(t.timestamp for t in settled_transactions).date()
            days_active = (end_date - start_date).days + 1
            trades_per_day = len(settled_transactions) / days_active if days_active > 0 else 0
        else:
            days_active = 0
            trades_per_day = 0
        
        # Best/worst day
        daily_pnls = list(self.daily_pnl.values()) if self.daily_pnl else [0]
        best_day = max(daily_pnls) if daily_pnls else 0
        worst_day = min(daily_pnls) if daily_pnls else 0
        
        metrics = PerformanceMetrics(
            total_pnl=total_pnl,
            total_stakes=total_stakes,
            total_roi=total_roi,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            max_drawdown=max_drawdown,
            current_drawdown=current_drawdown,
            volatility=volatility,
            current_streak=self.current_streak,
            current_streak_type=self.current_streak_type,
            longest_winning_streak=streak_info["longest_winning_streak"],
            longest_losing_streak=streak_info["longest_losing_streak"],
            profit_factor=profit_factor,
            kelly_criterion=kelly_criterion,
            expectancy=expectancy,
            consecutive_losses=self.consecutive_losses,
            consecutive_wins=self.consecutive_wins,
            days_active=days_active,
            trades_per_day=trades_per_day,
            best_day=best_day,
            worst_day=worst_day
        )
        
        # Store in history
        self.performance_history.append(metrics)
        
        return metrics
    
    def check_performance_alerts(self) -> List[PerformanceAlert]:
        """Check for performance-based alerts"""
        current_metrics = self.calculate_performance_metrics()
        new_alerts = []
        
        # Drawdown alert
        if current_metrics.current_drawdown > self.max_drawdown_threshold:
            alert = self._create_alert(
                AlertType.DRAWDOWN_WARNING,
                "high",
                f"Current drawdown ({current_metrics.current_drawdown:.2%}) exceeds threshold ({self.max_drawdown_threshold:.2%})",
                current_metrics.current_drawdown,
                self.max_drawdown_threshold,
                "current_drawdown"
            )
            new_alerts.append(alert)
        
        # Sharpe ratio alert
        if current_metrics.sharpe_ratio < self.min_sharpe_threshold:
            alert = self._create_alert(
                AlertType.SHARPE_DECLINE,
                "medium",
                f"Sharpe ratio ({current_metrics.sharpe_ratio:.2f}) below threshold ({self.min_sharpe_threshold:.2f})",
                current_metrics.sharpe_ratio,
                self.min_sharpe_threshold,
                "sharpe_ratio"
            )
            new_alerts.append(alert)
        
        # Win rate alert
        if current_metrics.win_rate < self.min_win_rate_threshold:
            alert = self._create_alert(
                AlertType.WIN_RATE_DECLINE,
                "medium",
                f"Win rate ({current_metrics.win_rate:.2%}) below threshold ({self.min_win_rate_threshold:.2%})",
                current_metrics.win_rate,
                self.min_win_rate_threshold,
                "win_rate"
            )
            new_alerts.append(alert)
        
        # Consecutive losses alert
        if current_metrics.consecutive_losses >= 5:
            severity = "high" if current_metrics.consecutive_losses >= 10 else "medium"
            alert = self._create_alert(
                AlertType.STREAK_ALERT,
                severity,
                f"Consecutive losses: {current_metrics.consecutive_losses}",
                current_metrics.consecutive_losses,
                5,
                "consecutive_losses"
            )
            new_alerts.append(alert)
        
        # High volatility alert
        if current_metrics.volatility > 0.5:  # 50% annualized volatility
            alert = self._create_alert(
                AlertType.HIGH_VOLATILITY,
                "medium",
                f"High volatility detected: {current_metrics.volatility:.2%}",
                current_metrics.volatility,
                0.5,
                "volatility"
            )
            new_alerts.append(alert)
        
        # Store new alerts
        for alert in new_alerts:
            self.active_alerts[alert.alert_id] = alert
        
        return new_alerts
    
    def _update_streaks(self, result: str):
        """Update winning/losing streaks"""
        if result == "win":
            if self.current_streak_type == "win":
                self.current_streak += 1
            else:
                self.current_streak = 1
                self.current_streak_type = "win"
            
            self.consecutive_wins += 1
            self.consecutive_losses = 0
            
        elif result == "loss":
            if self.current_streak_type == "loss":
                self.current_streak += 1
            else:
                self.current_streak = 1
                self.current_streak_type = "loss"
            
            self.consecutive_losses += 1
            self.consecutive_wins = 0
            
        else:  # push, void
            # Don't break streaks for pushes/voids
            pass
    
    def _filter_transactions_by_timeframe(self, timeframe: TimeFrame) -> List[BettingTransaction]:
        """Filter transactions by timeframe"""
        if timeframe == TimeFrame.ALL_TIME:
            return self.transactions
        
        now = datetime.now()
        
        if timeframe == TimeFrame.DAILY:
            cutoff = now - timedelta(days=1)
        elif timeframe == TimeFrame.WEEKLY:
            cutoff = now - timedelta(weeks=1)
        elif timeframe == TimeFrame.MONTHLY:
            cutoff = now - timedelta(days=30)
        elif timeframe == TimeFrame.QUARTERLY:
            cutoff = now - timedelta(days=90)
        elif timeframe == TimeFrame.YEARLY:
            cutoff = now - timedelta(days=365)
        else:
            return self.transactions
        
        return [t for t in self.transactions if t.timestamp >= cutoff]
    
    def _calculate_daily_returns(self, timeframe: TimeFrame) -> np.ndarray:
        """Calculate daily returns for the given timeframe"""
        if not self.daily_pnl:
            return np.array([])
        
        # Filter daily P&L by timeframe
        if timeframe == TimeFrame.ALL_TIME:
            daily_pnls = list(self.daily_pnl.values())
        else:
            # Get relevant dates based on timeframe
            now = datetime.now()
            if timeframe == TimeFrame.DAILY:
                cutoff = now - timedelta(days=1)
            elif timeframe == TimeFrame.WEEKLY:
                cutoff = now - timedelta(weeks=1)
            elif timeframe == TimeFrame.MONTHLY:
                cutoff = now - timedelta(days=30)
            elif timeframe == TimeFrame.QUARTERLY:
                cutoff = now - timedelta(days=90)
            elif timeframe == TimeFrame.YEARLY:
                cutoff = now - timedelta(days=365)
            else:
                cutoff = datetime.min
            
            daily_pnls = [
                pnl for date_str, pnl in self.daily_pnl.items()
                if datetime.fromisoformat(date_str) >= cutoff.date()
            ]
        
        # Convert to returns
        if not daily_pnls:
            return np.array([])
        
        # Calculate returns as percentage of average bankroll
        avg_bankroll = np.mean([broll for _, broll in self.bankroll_history])
        returns = np.array(daily_pnls) / avg_bankroll
        
        return returns
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) < 2:
            return 0.0
        
        excess_returns = returns - (self.risk_free_rate / 252)  # Daily risk-free rate
        
        if np.std(returns) == 0:
            return 0.0
        
        sharpe = np.mean(excess_returns) / np.std(returns) * np.sqrt(252)
        
        return sharpe
    
    def _calculate_sortino_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sortino ratio (downside deviation)"""
        if len(returns) < 2:
            return 0.0
        
        excess_returns = returns - (self.risk_free_rate / 252)
        negative_returns = excess_returns[excess_returns < 0]
        
        if len(negative_returns) == 0:
            return float('inf')
        
        downside_deviation = np.std(negative_returns)
        
        if downside_deviation == 0:
            return 0.0
        
        sortino = np.mean(excess_returns) / downside_deviation * np.sqrt(252)
        
        return sortino
    
    def _calculate_calmar_ratio(self, returns: np.ndarray) -> float:
        """Calculate Calmar ratio (return/max drawdown)"""
        if len(returns) < 2:
            return 0.0
        
        annual_return = np.mean(returns) * 252
        max_dd = self._calculate_max_drawdown(TimeFrame.ALL_TIME)
        
        if max_dd == 0:
            return float('inf')
        
        return annual_return / max_dd
    
    def _calculate_max_drawdown(self, timeframe: TimeFrame) -> float:
        """Calculate maximum drawdown"""
        if len(self.bankroll_history) < 2:
            return 0.0
        
        # Filter bankroll history by timeframe
        if timeframe == TimeFrame.ALL_TIME:
            history = self.bankroll_history
        else:
            now = datetime.now()
            if timeframe == TimeFrame.DAILY:
                cutoff = now - timedelta(days=1)
            elif timeframe == TimeFrame.WEEKLY:
                cutoff = now - timedelta(weeks=1)
            elif timeframe == TimeFrame.MONTHLY:
                cutoff = now - timedelta(days=30)
            elif timeframe == TimeFrame.QUARTERLY:
                cutoff = now - timedelta(days=90)
            elif timeframe == TimeFrame.YEARLY:
                cutoff = now - timedelta(days=365)
            else:
                cutoff = datetime.min
            
            history = [(ts, broll) for ts, broll in self.bankroll_history if ts >= cutoff]
        
        if len(history) < 2:
            return 0.0
        
        peak = history[0][1]
        max_drawdown = 0.0
        
        for _, bankroll in history:
            if bankroll > peak:
                peak = bankroll
            
            drawdown = (peak - bankroll) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        return max_drawdown
    
    def _analyze_streaks(self, transactions: List[BettingTransaction]) -> Dict[str, int]:
        """Analyze winning and losing streaks"""
        longest_winning_streak = 0
        longest_losing_streak = 0
        current_winning_streak = 0
        current_losing_streak = 0
        
        for transaction in transactions:
            if transaction.result == "win":
                current_winning_streak += 1
                current_losing_streak = 0
                longest_winning_streak = max(longest_winning_streak, current_winning_streak)
            elif transaction.result == "loss":
                current_losing_streak += 1
                current_winning_streak = 0
                longest_losing_streak = max(longest_losing_streak, current_losing_streak)
        
        return {
            "longest_winning_streak": longest_winning_streak,
            "longest_losing_streak": longest_losing_streak
        }
    
    def _calculate_profit_factor(self, wins: List[BettingTransaction], losses: List[BettingTransaction]) -> float:
        """Calculate profit factor (gross profit / gross loss)"""
        gross_profit = sum(t.pnl for t in wins)
        gross_loss = sum(abs(t.pnl) for t in losses)
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0
        
        return gross_profit / gross_loss
    
    def _calculate_kelly_criterion(self, transactions: List[BettingTransaction]) -> float:
        """Calculate Kelly criterion percentage"""
        wins = [t for t in transactions if t.result == "win"]
        losses = [t for t in transactions if t.result == "loss"]
        
        if not wins or not losses:
            return 0.0
        
        win_rate = len(wins) / len(transactions)
        avg_win_ratio = np.mean([t.pnl / t.stake for t in wins])
        avg_loss_ratio = np.mean([abs(t.pnl) / t.stake for t in losses])
        
        # Kelly formula: f* = (bp - q) / b
        # where b = average win ratio, p = win rate, q = loss rate
        kelly = (avg_win_ratio * win_rate - (1 - win_rate)) / avg_win_ratio
        
        return max(0, kelly)
    
    def _calculate_expectancy(self, wins: List[BettingTransaction], losses: List[BettingTransaction], win_rate: float) -> float:
        """Calculate expectancy per bet"""
        if not wins and not losses:
            return 0.0
        
        avg_win = np.mean([t.pnl for t in wins]) if wins else 0
        avg_loss = np.mean([t.pnl for t in losses]) if losses else 0
        
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
        
        return expectancy
    
    def _create_empty_metrics(self) -> PerformanceMetrics:
        """Create empty performance metrics"""
        return PerformanceMetrics(
            total_pnl=0.0,
            total_stakes=0.0,
            total_roi=0.0,
            win_rate=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            max_drawdown=0.0,
            current_drawdown=0.0,
            volatility=0.0,
            current_streak=0,
            current_streak_type="",
            longest_winning_streak=0,
            longest_losing_streak=0,
            profit_factor=0.0,
            kelly_criterion=0.0,
            expectancy=0.0,
            consecutive_losses=0,
            consecutive_wins=0,
            days_active=0,
            trades_per_day=0.0,
            best_day=0.0,
            worst_day=0.0
        )
    
    def _create_alert(self,
                     alert_type: AlertType,
                     severity: str,
                     message: str,
                     current_value: float,
                     threshold: float,
                     metric_name: str) -> PerformanceAlert:
        """Create performance alert"""
        alert_id = f"{alert_type.value}_{datetime.now().isoformat()}"
        
        return PerformanceAlert(
            alert_id=alert_id,
            alert_type=alert_type,
            timestamp=datetime.now(),
            severity=severity,
            message=message,
            current_value=current_value,
            threshold=threshold,
            metric_name=metric_name
        )
    
    def resolve_alert(self, alert_id: str):
        """Resolve an active alert"""
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].resolved = True
            self.active_alerts[alert_id].resolution_time = datetime.now()
            logger.info(f"Alert resolved: {alert_id}")
    
    def get_performance_summary(self, timeframe: TimeFrame = TimeFrame.ALL_TIME) -> Dict:
        """Get comprehensive performance summary"""
        metrics = self.calculate_performance_metrics(timeframe)
        
        return {
            "timeframe": timeframe.value,
            "bankroll": {
                "current": self.current_bankroll,
                "initial": self.initial_bankroll,
                "peak": self.peak_bankroll,
                "change": self.current_bankroll - self.initial_bankroll,
                "change_percent": (self.current_bankroll - self.initial_bankroll) / self.initial_bankroll
            },
            "trading": {
                "total_trades": len([t for t in self.transactions if t.is_settled()]),
                "win_rate": metrics.win_rate,
                "profit_factor": metrics.profit_factor,
                "expectancy": metrics.expectancy,
                "kelly_criterion": metrics.kelly_criterion
            },
            "risk": {
                "sharpe_ratio": metrics.sharpe_ratio,
                "sortino_ratio": metrics.sortino_ratio,
                "max_drawdown": metrics.max_drawdown,
                "current_drawdown": metrics.current_drawdown,
                "volatility": metrics.volatility
            },
            "streaks": {
                "current_streak": metrics.current_streak,
                "current_streak_type": metrics.current_streak_type,
                "longest_winning_streak": metrics.longest_winning_streak,
                "longest_losing_streak": metrics.longest_losing_streak,
                "consecutive_losses": metrics.consecutive_losses
            },
            "alerts": {
                "active_count": len([a for a in self.active_alerts.values() if not a.resolved]),
                "total_count": len(self.active_alerts)
            }
        }


# Example usage
if __name__ == "__main__":
    # Initialize performance monitor
    monitor = PerformanceMonitor(initial_bankroll=25000)
    
    # Record some sample transactions
    transactions = [
        BettingTransaction(
            transaction_id="bet_1",
            timestamp=datetime.now() - timedelta(days=5),
            sport="NFL",
            event="Chiefs vs Bills",
            selection="Chiefs -3.5",
            bet_type="spread",
            stake=500,
            odds=1.91,
            result="win",
            pnl=455
        ),
        BettingTransaction(
            transaction_id="bet_2",
            timestamp=datetime.now() - timedelta(days=4),
            sport="NBA",
            event="Lakers vs Warriors",
            selection="Over 215.5",
            bet_type="total",
            stake=300,
            odds=1.85,
            result="loss",
            pnl=-300
        )
    ]
    
    # Record transactions
    for transaction in transactions:
        monitor.record_transaction(transaction)
    
    # Check for alerts
    alerts = monitor.check_performance_alerts()
    
    # Get performance summary
    summary = monitor.get_performance_summary()
    print(json.dumps(summary, indent=2, default=str))