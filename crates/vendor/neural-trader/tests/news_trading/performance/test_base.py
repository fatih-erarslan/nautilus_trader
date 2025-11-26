"""Tests for performance tracking base classes."""

import pytest
from abc import ABC

from src.news_trading.performance.base import PerformanceTracker
from src.news_trading.performance.models import TradeResult, PerformanceMetrics, TradeStatus


class TestPerformanceTracker:
    """Test the PerformanceTracker abstract interface."""

    def test_performance_tracker_interface(self):
        """Test that PerformanceTracker abstract interface is properly defined."""
        
        class TestTracker(PerformanceTracker):
            """Test implementation without methods."""
            pass
        
        # Should fail - abstract methods not implemented
        with pytest.raises(TypeError):
            tracker = TestTracker()

    def test_performance_tracker_implementation(self):
        """Test a proper implementation of PerformanceTracker."""
        
        class ConcreteTracker(PerformanceTracker):
            """Concrete implementation for testing."""
            
            def __init__(self):
                self.trades = []
            
            def record_trade(self, trade_result: TradeResult) -> None:
                """Record a completed trade."""
                self.trades.append(trade_result)
            
            def calculate_metrics(self, period: str = "all") -> PerformanceMetrics:
                """Calculate performance metrics for a period."""
                return PerformanceMetrics.from_trades(self.trades)
            
            def get_trade_history(self, filters=None):
                """Get filtered trade history."""
                return self.trades
            
            def generate_report(self, format: str = "json"):
                """Generate performance report."""
                return {
                    "metrics": self.calculate_metrics(),
                    "trades": len(self.trades),
                }
        
        # Should work - all abstract methods implemented
        tracker = ConcreteTracker()
        assert isinstance(tracker, PerformanceTracker)
        
        # Test basic functionality
        trade = TradeResult(
            trade_id="test-1",
            signal_id="signal-1",
            asset="BTC",
            entry_time=pytest.helpers.now(),
            entry_price=50000,
            position_size=0.1,
            status=TradeStatus.CLOSED,
            exit_price=51000,
            pnl=100,
        )
        
        tracker.record_trade(trade)
        assert len(tracker.get_trade_history()) == 1
        
        metrics = tracker.calculate_metrics()
        assert metrics.total_trades == 1