"""Performance metrics for swing trading strategies."""

from typing import Dict, List, Any
from datetime import datetime
import numpy as np


class SwingPerformanceTracker:
    """Tracks performance metrics for swing trades."""
    
    def __init__(self):
        """Initialize the swing performance tracker."""
        self.trades = []
        self.asset_classes = ["stock", "bond", "commodity", "crypto"]
        
    def add_trade(self, trade: Dict[str, Any]) -> None:
        """Add a completed swing trade.
        
        Args:
            trade: Trade details including entry, exit, days held
        """
        # Calculate return if not provided
        if "return" not in trade:
            entry = trade.get("entry", 0)
            exit = trade.get("exit", 0)
            if entry > 0:
                trade["return"] = (exit - entry) / entry
            else:
                trade["return"] = 0
        
        # Calculate P&L if not provided
        if "pnl" not in trade:
            trade["pnl"] = trade["return"] * trade.get("position_size", 1000)
        
        # Add timestamp if not provided
        if "timestamp" not in trade:
            trade["timestamp"] = datetime.now()
        
        self.trades.append(trade)
    
    def calculate_metrics(self) -> Dict[str, float]:
        """Calculate overall performance metrics.
        
        Returns:
            Dictionary of performance metrics
        """
        if not self.trades:
            return self._empty_metrics()
        
        # Basic metrics
        returns = [t["return"] for t in self.trades]
        pnls = [t.get("pnl", 0) for t in self.trades]
        days_held = [t.get("days_held", 0) for t in self.trades]
        
        winning_trades = [t for t in self.trades if t["return"] > 0]
        losing_trades = [t for t in self.trades if t["return"] <= 0]
        
        # Win/loss metrics
        win_rate = len(winning_trades) / len(self.trades)
        avg_win = np.mean([t["return"] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t["return"] for t in losing_trades]) if losing_trades else 0
        
        # Profit factor
        gross_profit = sum(t["pnl"] for t in winning_trades)
        gross_loss = abs(sum(t["pnl"] for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Risk metrics
        returns_array = np.array(returns)
        sharpe_ratio = self._calculate_sharpe_ratio(returns_array)
        max_drawdown = self._calculate_max_drawdown(pnls)
        
        return {
            "total_trades": len(self.trades),
            "win_rate": round(win_rate, 3),
            "avg_win": round(avg_win, 4),
            "avg_loss": round(avg_loss, 4),
            "profit_factor": round(profit_factor, 2),
            "sharpe_ratio": round(sharpe_ratio, 2),
            "max_drawdown": round(max_drawdown, 3),
            "avg_holding_days": round(np.mean(days_held), 1) if days_held else 0,
            "total_pnl": round(sum(pnls), 2),
            "avg_return": round(np.mean(returns), 4),
        }
    
    def metrics_by_asset_class(self) -> Dict[str, Dict[str, float]]:
        """Calculate metrics broken down by asset class.
        
        Returns:
            Metrics for each asset class
        """
        asset_metrics = {}
        
        for asset_class in self.asset_classes:
            asset_trades = [t for t in self.trades if t.get("type") == asset_class]
            
            if asset_trades:
                # Store original trades temporarily
                original_trades = self.trades
                self.trades = asset_trades
                
                # Calculate metrics for this asset class
                metrics = self.calculate_metrics()
                asset_metrics[asset_class] = metrics
                
                # Restore original trades
                self.trades = original_trades
            else:
                asset_metrics[asset_class] = self._empty_metrics()
        
        return asset_metrics
    
    def calculate_efficiency_metrics(self) -> Dict[str, float]:
        """Calculate trade entry/exit efficiency metrics.
        
        Returns:
            Efficiency metrics
        """
        if not self.trades:
            return {
                "avg_entry_efficiency": 0,
                "avg_exit_efficiency": 0,
                "timing_score": 0,
            }
        
        entry_efficiencies = []
        exit_efficiencies = []
        
        for trade in self.trades:
            if "entry_efficiency" in trade:
                entry_efficiencies.append(trade["entry_efficiency"])
            if "exit_efficiency" in trade:
                exit_efficiencies.append(trade["exit_efficiency"])
        
        avg_entry = np.mean(entry_efficiencies) if entry_efficiencies else 0.5
        avg_exit = np.mean(exit_efficiencies) if exit_efficiencies else 0.5
        
        # Combined timing score
        timing_score = (avg_entry + avg_exit) / 2
        
        return {
            "avg_entry_efficiency": round(avg_entry, 3),
            "avg_exit_efficiency": round(avg_exit, 3),
            "timing_score": round(timing_score, 3),
            "perfect_timing_gap": round(1.0 - timing_score, 3),
        }
    
    def get_trade_distribution(self) -> Dict[str, Any]:
        """Get distribution of trades by various factors.
        
        Returns:
            Trade distribution statistics
        """
        if not self.trades:
            return {}
        
        # By asset type
        asset_distribution = {}
        for trade in self.trades:
            asset = trade.get("type", "unknown")
            asset_distribution[asset] = asset_distribution.get(asset, 0) + 1
        
        # By holding period
        holding_periods = [t.get("days_held", 0) for t in self.trades]
        
        period_distribution = {
            "1-3_days": sum(1 for d in holding_periods if 1 <= d <= 3),
            "4-7_days": sum(1 for d in holding_periods if 4 <= d <= 7),
            "8-14_days": sum(1 for d in holding_periods if 8 <= d <= 14),
            "15+_days": sum(1 for d in holding_periods if d >= 15),
        }
        
        # By return buckets
        returns = [t["return"] for t in self.trades]
        
        return_distribution = {
            "big_loss": sum(1 for r in returns if r <= -0.05),
            "small_loss": sum(1 for r in returns if -0.05 < r <= -0.02),
            "scratch": sum(1 for r in returns if -0.02 < r <= 0.02),
            "small_win": sum(1 for r in returns if 0.02 < r <= 0.05),
            "big_win": sum(1 for r in returns if r > 0.05),
        }
        
        return {
            "by_asset": asset_distribution,
            "by_holding_period": period_distribution,
            "by_return": return_distribution,
        }
    
    def calculate_rolling_metrics(self, window: int = 20) -> List[Dict[str, float]]:
        """Calculate rolling performance metrics.
        
        Args:
            window: Rolling window size
            
        Returns:
            List of rolling metrics
        """
        if len(self.trades) < window:
            return []
        
        rolling_metrics = []
        
        for i in range(window, len(self.trades) + 1):
            window_trades = self.trades[i-window:i]
            
            # Calculate metrics for window
            win_rate = sum(1 for t in window_trades if t["return"] > 0) / window
            avg_return = np.mean([t["return"] for t in window_trades])
            
            rolling_metrics.append({
                "end_index": i,
                "win_rate": round(win_rate, 3),
                "avg_return": round(avg_return, 4),
                "trades": window,
            })
        
        return rolling_metrics
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray, risk_free: float = 0.02) -> float:
        """Calculate Sharpe ratio.
        
        Args:
            returns: Array of returns
            risk_free: Risk-free rate
            
        Returns:
            Sharpe ratio
        """
        if len(returns) < 2:
            return 0
        
        excess_returns = returns - risk_free / 252  # Daily risk-free
        
        if np.std(excess_returns) == 0:
            return 0
        
        # Annualized Sharpe
        sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
        
        return sharpe
    
    def _calculate_max_drawdown(self, pnls: List[float]) -> float:
        """Calculate maximum drawdown.
        
        Args:
            pnls: List of P&L values
            
        Returns:
            Maximum drawdown (negative value)
        """
        if not pnls:
            return 0
        
        cumulative = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max
        
        return np.min(drawdown) / running_max[np.argmin(drawdown)] if running_max[np.argmin(drawdown)] > 0 else 0
    
    def _empty_metrics(self) -> Dict[str, float]:
        """Return empty metrics structure.
        
        Returns:
            Dictionary with zero values
        """
        return {
            "total_trades": 0,
            "win_rate": 0,
            "avg_win": 0,
            "avg_loss": 0,
            "profit_factor": 0,
            "sharpe_ratio": 0,
            "max_drawdown": 0,
            "avg_holding_days": 0,
            "total_pnl": 0,
            "avg_return": 0,
        }