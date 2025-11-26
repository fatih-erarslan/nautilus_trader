"""Performance analytics implementation."""

import json
import csv
from io import StringIO
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
import numpy as np
import pandas as pd

from .models import TradeResult, TradeStatus


class PerformanceAnalytics:
    """Generate comprehensive performance analytics."""
    
    def __init__(self):
        """Initialize performance analytics."""
        self.trades: List[TradeResult] = []
        self.trade_attributions: Dict[str, Dict[str, float]] = {}
        self._cache = {}
        self._cache_timestamp = None
        self._cache_duration = timedelta(minutes=5)
    
    def add_trade(self, trade: TradeResult) -> None:
        """Add a trade to analytics.
        
        Args:
            trade: Trade result to add
        """
        self.trades.append(trade)
        self._invalidate_cache()
    
    def add_trade_with_attribution(
        self,
        trade: TradeResult,
        attribution: Dict[str, float],
    ) -> None:
        """Add a trade with source attribution.
        
        Args:
            trade: Trade result
            attribution: Source attribution percentages
        """
        self.add_trade(trade)
        self.trade_attributions[trade.trade_id] = attribution
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report.
        
        Returns:
            Report dictionary with all analytics
        """
        if self._is_cache_valid("report"):
            return self._cache["report"]
        
        report = {
            "summary": self._generate_summary(),
            "by_asset": self._analyze_by_asset(),
            "daily_pnl": self._calculate_daily_pnl(),
            "cumulative_pnl": self._calculate_cumulative_pnl(),
            "time_analysis": {
                "hourly": self.get_hourly_performance(),
                "day_of_week": self.get_day_of_week_performance(),
            },
            "risk_metrics": self.calculate_risk_metrics(),
            "generated_at": datetime.now().isoformat(),
        }
        
        self._cache["report"] = report
        self._cache_timestamp = datetime.now()
        
        return report
    
    def get_source_performance(self) -> Dict[str, Dict[str, float]]:
        """Get performance metrics by news source.
        
        Returns:
            Source performance metrics
        """
        source_metrics = defaultdict(lambda: {
            "weighted_pnl": 0.0,
            "total_signals": 0,
            "profitable_signals": 0,
            "total_attribution": 0.0,
        })
        
        for trade in self.trades:
            if trade.trade_id in self.trade_attributions:
                attribution = self.trade_attributions[trade.trade_id]
                
                for source, weight in attribution.items():
                    metrics = source_metrics[source]
                    metrics["weighted_pnl"] += trade.pnl * weight
                    metrics["total_signals"] += 1
                    metrics["total_attribution"] += weight
                    
                    if trade.pnl > 0:
                        metrics["profitable_signals"] += weight
        
        # Calculate accuracy rates
        for source, metrics in source_metrics.items():
            if metrics["total_signals"] > 0:
                metrics["accuracy_rate"] = (
                    metrics["profitable_signals"] / metrics["total_signals"]
                )
                metrics["average_pnl_per_signal"] = (
                    metrics["weighted_pnl"] / metrics["total_signals"]
                )
        
        return dict(source_metrics)
    
    def get_hourly_performance(self) -> Dict[int, Dict[str, float]]:
        """Analyze performance by hour of day.
        
        Returns:
            Hourly performance metrics
        """
        hourly_trades = defaultdict(list)
        
        for trade in self.trades:
            hour = trade.entry_time.hour
            hourly_trades[hour].append(trade)
        
        hourly_performance = {}
        
        for hour in range(24):
            trades = hourly_trades[hour]
            if trades:
                hourly_performance[hour] = {
                    "trade_count": len(trades),
                    "total_pnl": sum(t.pnl for t in trades),
                    "average_pnl": np.mean([t.pnl for t in trades]),
                    "win_rate": sum(1 for t in trades if t.pnl > 0) / len(trades),
                }
            else:
                hourly_performance[hour] = {
                    "trade_count": 0,
                    "total_pnl": 0.0,
                    "average_pnl": 0.0,
                    "win_rate": 0.0,
                }
        
        return hourly_performance
    
    def get_day_of_week_performance(self) -> Dict[str, Dict[str, float]]:
        """Analyze performance by day of week.
        
        Returns:
            Day of week performance metrics
        """
        dow_trades = defaultdict(list)
        dow_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        
        for trade in self.trades:
            dow = trade.entry_time.weekday()
            dow_trades[dow_names[dow]].append(trade)
        
        dow_performance = {}
        
        for day_name in dow_names:
            trades = dow_trades[day_name]
            if trades:
                dow_performance[day_name] = {
                    "trade_count": len(trades),
                    "total_pnl": sum(t.pnl for t in trades),
                    "average_pnl": np.mean([t.pnl for t in trades]),
                    "win_rate": sum(1 for t in trades if t.pnl > 0) / len(trades),
                }
            else:
                dow_performance[day_name] = {
                    "trade_count": 0,
                    "total_pnl": 0.0,
                    "average_pnl": 0.0,
                    "win_rate": 0.0,
                }
        
        return dow_performance
    
    def compare_strategies(self) -> Dict[str, Dict[str, float]]:
        """Compare performance across different strategies.
        
        Returns:
            Strategy comparison metrics
        """
        strategy_trades = defaultdict(list)
        
        for trade in self.trades:
            if "strategy" in trade.metadata:
                strategy = trade.metadata["strategy"]
                strategy_trades[strategy].append(trade)
        
        strategy_comparison = {}
        
        for strategy, trades in strategy_trades.items():
            if trades:
                pnls = [t.pnl for t in trades]
                returns = [t.pnl_percentage for t in trades if t.pnl_percentage != 0]
                
                # Calculate Sharpe ratio
                if returns and len(returns) > 1:
                    sharpe = (
                        np.mean(returns) / np.std(returns) * np.sqrt(252)
                        if np.std(returns) > 0
                        else 0
                    )
                else:
                    sharpe = 0
                
                strategy_comparison[strategy] = {
                    "total_trades": len(trades),
                    "total_pnl": sum(pnls),
                    "average_pnl": np.mean(pnls),
                    "win_rate": sum(1 for t in trades if t.pnl > 0) / len(trades),
                    "sharpe_ratio": sharpe,
                    "max_win": max(pnls),
                    "max_loss": min(pnls),
                }
        
        return strategy_comparison
    
    def calculate_risk_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive risk metrics.
        
        Returns:
            Risk metrics dictionary
        """
        if not self.trades:
            return {}
        
        # Sort trades by time
        sorted_trades = sorted(self.trades, key=lambda t: t.exit_time or t.entry_time)
        pnls = [t.pnl for t in sorted_trades]
        
        # Calculate cumulative P&L
        cumulative_pnl = np.cumsum(pnls)
        
        # Max drawdown calculation
        running_max = np.maximum.accumulate(cumulative_pnl)
        drawdown = cumulative_pnl - running_max
        max_drawdown_idx = np.argmin(drawdown)
        max_drawdown = drawdown[max_drawdown_idx] / running_max[max_drawdown_idx] if running_max[max_drawdown_idx] > 0 else 0
        
        # Recovery time (if recovered)
        recovery_time = None
        if max_drawdown < 0:
            peak_value = running_max[max_drawdown_idx]
            for i in range(max_drawdown_idx + 1, len(cumulative_pnl)):
                if cumulative_pnl[i] >= peak_value:
                    recovery_time = i - max_drawdown_idx
                    break
        
        # Value at Risk (VaR) and Conditional VaR
        returns = [t.pnl_percentage for t in sorted_trades if t.pnl_percentage != 0]
        if returns:
            var_95 = np.percentile(returns, 5)  # 95% VaR
            cvar_95 = np.mean([r for r in returns if r <= var_95])  # CVaR
        else:
            var_95 = 0
            cvar_95 = 0
        
        return {
            "max_drawdown": max_drawdown,
            "recovery_time": recovery_time,
            "var_95": var_95,
            "cvar_95": cvar_95,
            "total_risk": np.std(pnls) if pnls else 0,
            "downside_deviation": np.std([p for p in pnls if p < 0]) if any(p < 0 for p in pnls) else 0,
            "consecutive_losses": self._max_consecutive_losses(),
        }
    
    def _generate_summary(self) -> Dict[str, float]:
        """Generate summary statistics."""
        if not self.trades:
            return {
                "total_trades": 0,
                "total_pnl": 0.0,
                "win_rate": 0.0,
                "average_pnl": 0.0,
            }
        
        total_trades = len(self.trades)
        total_pnl = sum(t.pnl for t in self.trades)
        winning_trades = sum(1 for t in self.trades if t.pnl > 0)
        
        return {
            "total_trades": total_trades,
            "total_pnl": total_pnl,
            "win_rate": winning_trades / total_trades,
            "average_pnl": total_pnl / total_trades,
            "best_trade": max(t.pnl for t in self.trades),
            "worst_trade": min(t.pnl for t in self.trades),
        }
    
    def _analyze_by_asset(self) -> Dict[str, Dict[str, float]]:
        """Analyze performance by asset."""
        asset_trades = defaultdict(list)
        
        for trade in self.trades:
            asset_trades[trade.asset].append(trade)
        
        asset_performance = {}
        
        for asset, trades in asset_trades.items():
            asset_performance[asset] = {
                "total_trades": len(trades),
                "total_pnl": sum(t.pnl for t in trades),
                "average_pnl": np.mean([t.pnl for t in trades]),
                "win_rate": sum(1 for t in trades if t.pnl > 0) / len(trades),
            }
        
        return asset_performance
    
    def _calculate_daily_pnl(self) -> List[Dict[str, Any]]:
        """Calculate daily P&L."""
        daily_pnl = defaultdict(float)
        
        for trade in self.trades:
            date = (trade.exit_time or trade.entry_time).date()
            daily_pnl[date] += trade.pnl
        
        return [
            {"date": date.isoformat(), "pnl": pnl}
            for date, pnl in sorted(daily_pnl.items())
        ]
    
    def _calculate_cumulative_pnl(self) -> List[Dict[str, Any]]:
        """Calculate cumulative P&L over time."""
        sorted_trades = sorted(
            self.trades,
            key=lambda t: t.exit_time or t.entry_time
        )
        
        cumulative = 0
        cumulative_pnl = []
        
        for trade in sorted_trades:
            cumulative += trade.pnl
            cumulative_pnl.append({
                "timestamp": (trade.exit_time or trade.entry_time).isoformat(),
                "cumulative_pnl": cumulative,
                "trade_id": trade.trade_id,
            })
        
        return cumulative_pnl
    
    def _max_consecutive_losses(self) -> int:
        """Calculate maximum consecutive losses."""
        if not self.trades:
            return 0
        
        sorted_trades = sorted(
            self.trades,
            key=lambda t: t.exit_time or t.entry_time
        )
        
        max_consecutive = 0
        current_consecutive = 0
        
        for trade in sorted_trades:
            if trade.pnl < 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def _is_cache_valid(self, key: str) -> bool:
        """Check if cache is valid."""
        if key not in self._cache or self._cache_timestamp is None:
            return False
        
        return datetime.now() - self._cache_timestamp < self._cache_duration
    
    def _invalidate_cache(self) -> None:
        """Invalidate the cache."""
        self._cache = {}
        self._cache_timestamp = None


class ReportGenerator:
    """Generate reports in various formats."""
    
    def generate_json(self, data: Dict[str, Any]) -> str:
        """Generate JSON report.
        
        Args:
            data: Report data
            
        Returns:
            JSON string
        """
        return json.dumps(data, indent=2, default=str)
    
    def generate_csv(self, trades: List[TradeResult]) -> str:
        """Generate CSV report of trades.
        
        Args:
            trades: List of trades
            
        Returns:
            CSV string
        """
        output = StringIO()
        
        if not trades:
            return ""
        
        # Define columns
        columns = [
            "trade_id", "signal_id", "asset", "entry_time", "exit_time",
            "entry_price", "exit_price", "position_size", "pnl", "pnl_percentage",
            "status", "fees",
        ]
        
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        
        for trade in trades:
            row = {
                "trade_id": trade.trade_id,
                "signal_id": trade.signal_id,
                "asset": trade.asset,
                "entry_time": trade.entry_time.isoformat(),
                "exit_time": trade.exit_time.isoformat() if trade.exit_time else "",
                "entry_price": trade.entry_price,
                "exit_price": trade.exit_price or "",
                "position_size": trade.position_size,
                "pnl": trade.pnl,
                "pnl_percentage": trade.pnl_percentage,
                "status": trade.status.value,
                "fees": trade.fees,
            }
            writer.writerow(row)
        
        return output.getvalue()
    
    def generate_summary(self, metrics: Dict[str, Any]) -> str:
        """Generate text summary report.
        
        Args:
            metrics: Performance metrics
            
        Returns:
            Summary text
        """
        summary = "Performance Summary\n"
        summary += "=" * 40 + "\n\n"
        
        if "total_trades" in metrics:
            summary += f"Total Trades: {metrics['total_trades']}\n"
        
        if "win_rate" in metrics:
            summary += f"Win Rate: {metrics['win_rate']:.1%}\n"
        
        if "average_win" in metrics:
            summary += f"Average Win: ${metrics['average_win']:.2f}\n"
        
        if "average_loss" in metrics:
            summary += f"Average Loss: ${metrics['average_loss']:.2f}\n"
        
        if "sharpe_ratio" in metrics:
            summary += f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n"
        
        if "max_drawdown" in metrics:
            summary += f"Max Drawdown: {metrics['max_drawdown']:.1%}\n"
        
        return summary


class RealTimeMetrics:
    """Real-time metrics streaming."""
    
    def __init__(self, update_interval_ms: int = 1000):
        """Initialize real-time metrics.
        
        Args:
            update_interval_ms: Minimum interval between updates
        """
        self.update_interval = timedelta(milliseconds=update_interval_ms)
        self.subscribers: Dict[str, Callable] = {}
        self.last_update = datetime.now()
        self.pending_updates = []
        
        # Current metrics
        self.total_trades = 0
        self.total_pnl = 0.0
        self.open_positions = 0
        self.today_pnl = 0.0
        self.last_trade_time = None
    
    def subscribe(self, client_id: str, callback: Callable) -> None:
        """Subscribe to real-time updates.
        
        Args:
            client_id: Unique client identifier
            callback: Function to call with updates
        """
        self.subscribers[client_id] = callback
    
    def unsubscribe(self, client_id: str) -> None:
        """Unsubscribe from updates.
        
        Args:
            client_id: Client to unsubscribe
        """
        self.subscribers.pop(client_id, None)
    
    def update_trade(self, trade: TradeResult) -> None:
        """Update metrics with new trade.
        
        Args:
            trade: New trade result
        """
        # Update metrics
        self.total_trades += 1
        self.total_pnl += trade.pnl
        self.last_trade_time = trade.exit_time or trade.entry_time
        
        # Update today's P&L
        if self.last_trade_time.date() == datetime.now().date():
            self.today_pnl += trade.pnl
        
        # Queue update
        self.pending_updates.append({
            "type": "trade_update",
            "trade_id": trade.trade_id,
            "timestamp": datetime.now().isoformat(),
        })
        
        # Check if we should send update
        if datetime.now() - self.last_update >= self.update_interval:
            self._send_updates()
        else:
            # For first update or critical updates, send immediately
            if self.total_trades == 1 or len(self.pending_updates) >= 10:
                self._send_updates()
    
    def get_snapshot(self) -> Dict[str, Any]:
        """Get current metrics snapshot.
        
        Returns:
            Current metrics
        """
        return {
            "total_trades": self.total_trades,
            "total_pnl": self.total_pnl,
            "open_positions": self.open_positions,
            "today_pnl": self.today_pnl,
            "last_update": self.last_trade_time.isoformat() if self.last_trade_time else None,
        }
    
    def _send_updates(self) -> None:
        """Send updates to all subscribers."""
        if not self.pending_updates and datetime.now() - self.last_update < self.update_interval:
            return
        
        # Batch updates
        update = {
            "type": "metrics_update",
            "snapshot": self.get_snapshot(),
            "updates": self.pending_updates,
            "timestamp": datetime.now().isoformat(),
        }
        
        # Send to all subscribers
        for callback in self.subscribers.values():
            try:
                callback(update)
            except Exception:
                # Log error but continue
                pass
        
        # Clear pending updates
        self.pending_updates = []
        self.last_update = datetime.now()