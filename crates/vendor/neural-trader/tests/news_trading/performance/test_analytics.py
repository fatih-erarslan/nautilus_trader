"""Tests for performance analytics generation."""

import pytest
from datetime import datetime, timedelta
import json

from src.news_trading.performance.analytics import (
    PerformanceAnalytics,
    ReportGenerator,
    RealTimeMetrics,
)
from src.news_trading.performance.models import TradeResult, TradeStatus


class TestPerformanceAnalytics:
    """Test performance analytics generation."""

    def test_performance_analytics(self):
        """Test performance analytics generation."""
        analytics = PerformanceAnalytics()

        # Add trade history
        trades = [
            TradeResult(
                trade_id="1",
                signal_id="sig-1",
                asset="BTC",
                entry_time=datetime(2024, 1, 1, 10, 0),
                exit_time=datetime(2024, 1, 1, 14, 0),
                entry_price=45000,
                exit_price=45900,
                position_size=0.1,
                pnl=90,
                status=TradeStatus.CLOSED,
            ),
            TradeResult(
                trade_id="2",
                signal_id="sig-2",
                asset="ETH",
                entry_time=datetime(2024, 1, 2, 9, 0),
                exit_time=datetime(2024, 1, 2, 11, 0),
                entry_price=3000,
                exit_price=2950,
                position_size=1,
                pnl=-50,
                status=TradeStatus.CLOSED,
            ),
            TradeResult(
                trade_id="3",
                signal_id="sig-3",
                asset="BTC",
                entry_time=datetime(2024, 1, 3, 15, 0),
                exit_time=datetime(2024, 1, 3, 17, 0),
                entry_price=46000,
                exit_price=46350,
                position_size=0.1,
                pnl=35,
                status=TradeStatus.CLOSED,
            ),
        ]

        for trade in trades:
            analytics.add_trade(trade)

        # Generate report
        report = analytics.generate_report()

        assert report["summary"]["total_pnl"] == 75  # 90 - 50 + 35
        assert report["by_asset"]["BTC"]["total_pnl"] == 125  # 90 + 35
        assert report["by_asset"]["ETH"]["total_pnl"] == -50
        assert "daily_pnl" in report
        assert "cumulative_pnl" in report

    def test_news_source_analytics(self):
        """Test analytics by news source."""
        analytics = PerformanceAnalytics()

        # Add trades with source attribution
        trades_with_attribution = [
            (
                TradeResult(
                    trade_id="1",
                    signal_id="sig-1",
                    asset="BTC",
                    entry_time=datetime.now(),
                    entry_price=50000,
                    exit_price=51000,
                    position_size=0.1,
                    pnl=100,
                    status=TradeStatus.CLOSED,
                ),
                {"reuters": 0.7, "bloomberg": 0.3},
            ),
            (
                TradeResult(
                    trade_id="2",
                    signal_id="sig-2",
                    asset="ETH",
                    entry_time=datetime.now(),
                    entry_price=3000,
                    exit_price=2980,
                    position_size=1,
                    pnl=-20,
                    status=TradeStatus.CLOSED,
                ),
                {"twitter": 0.9, "reuters": 0.1},
            ),
            (
                TradeResult(
                    trade_id="3",
                    signal_id="sig-3",
                    asset="ADA",
                    entry_time=datetime.now(),
                    entry_price=0.5,
                    exit_price=0.55,
                    position_size=1000,
                    pnl=50,
                    status=TradeStatus.CLOSED,
                ),
                {"bloomberg": 1.0},
            ),
        ]

        for trade, attribution in trades_with_attribution:
            analytics.add_trade_with_attribution(trade, attribution)

        source_performance = analytics.get_source_performance()

        assert "reuters" in source_performance
        assert source_performance["reuters"]["weighted_pnl"] > 0  # 0.7*100 + 0.1*(-20)
        assert source_performance["twitter"]["weighted_pnl"] < 0  # 0.9*(-20)
        assert source_performance["bloomberg"]["total_signals"] == 2

    def test_time_based_analytics(self):
        """Test time-based performance analytics."""
        analytics = PerformanceAnalytics()

        # Add trades across different times
        base_time = datetime(2024, 1, 1)
        
        for day in range(30):
            for hour in [9, 14]:  # Morning and afternoon trades
                trade = TradeResult(
                    trade_id=f"trade-{day}-{hour}",
                    signal_id=f"sig-{day}-{hour}",
                    asset="SPY",
                    entry_time=base_time + timedelta(days=day, hours=hour),
                    exit_time=base_time + timedelta(days=day, hours=hour+2),
                    entry_price=400,
                    exit_price=401 if hour == 9 else 399,  # Morning wins, afternoon loses
                    position_size=10,
                    pnl=10 if hour == 9 else -10,
                    status=TradeStatus.CLOSED,
                )
                analytics.add_trade(trade)

        # Get hourly performance
        hourly_performance = analytics.get_hourly_performance()
        
        assert hourly_performance[9]["average_pnl"] > 0
        assert hourly_performance[14]["average_pnl"] < 0
        assert hourly_performance[9]["trade_count"] == 30

        # Get day of week performance
        dow_performance = analytics.get_day_of_week_performance()
        assert len(dow_performance) == 7  # All days of week

    def test_strategy_comparison_analytics(self):
        """Test analytics comparing multiple strategies."""
        analytics = PerformanceAnalytics()

        strategies = ["momentum", "mean_reversion", "arbitrage"]
        
        # Add trades for each strategy
        for i in range(30):
            for strategy in strategies:
                # Different performance characteristics
                if strategy == "momentum":
                    pnl = 100 if i % 3 == 0 else -30  # High reward, lower win rate
                elif strategy == "mean_reversion":
                    pnl = 20 if i % 2 == 0 else -15  # Consistent small wins
                else:  # arbitrage
                    pnl = 5  # Small consistent profit
                
                trade = TradeResult(
                    trade_id=f"{strategy}-{i}",
                    signal_id=f"sig-{strategy}-{i}",
                    asset="AAPL",
                    entry_time=datetime.now() - timedelta(days=30-i),
                    entry_price=150,
                    exit_price=150 + pnl/10,
                    position_size=10,
                    pnl=pnl,
                    status=TradeStatus.CLOSED,
                    metadata={"strategy": strategy},
                )
                analytics.add_trade(trade)

        strategy_comparison = analytics.compare_strategies()
        
        assert "momentum" in strategy_comparison
        assert strategy_comparison["arbitrage"]["win_rate"] == 1.0
        assert "sharpe_ratio" in strategy_comparison["mean_reversion"]

    def test_risk_metrics_calculation(self):
        """Test calculation of risk metrics."""
        analytics = PerformanceAnalytics()

        # Add trades with varying risk profiles
        trades = []
        cumulative = 0
        
        for i in range(100):
            # Simulate realistic P&L with drawdowns
            if i < 20:
                pnl = 50  # Initial winning streak
            elif i < 40:
                pnl = -100  # Drawdown period
            elif i < 60:
                pnl = 75  # Recovery
            else:
                pnl = 25  # Steady gains
            
            cumulative += pnl
            
            trade = TradeResult(
                trade_id=f"trade-{i}",
                signal_id=f"sig-{i}",
                asset="QQQ",
                entry_time=datetime.now() - timedelta(days=100-i),
                entry_price=350,
                exit_price=350 + pnl/10,
                position_size=10,
                pnl=pnl,
                pnl_percentage=pnl/3500 * 100,
                status=TradeStatus.CLOSED,
            )
            trades.append(trade)
            analytics.add_trade(trade)

        risk_metrics = analytics.calculate_risk_metrics()
        
        assert "max_drawdown" in risk_metrics
        assert risk_metrics["max_drawdown"] < 0  # Should be negative
        assert "var_95" in risk_metrics  # Value at Risk
        assert "cvar_95" in risk_metrics  # Conditional VaR
        assert "recovery_time" in risk_metrics


class TestReportGenerator:
    """Test report generation in various formats."""

    def test_json_report_generation(self):
        """Test JSON report generation."""
        generator = ReportGenerator()
        
        # Create sample data
        data = {
            "summary": {
                "total_trades": 100,
                "win_rate": 0.65,
                "total_pnl": 5000,
            },
            "daily_pnl": [
                {"date": "2024-01-01", "pnl": 100},
                {"date": "2024-01-02", "pnl": -50},
            ],
        }
        
        json_report = generator.generate_json(data)
        
        # Should be valid JSON
        parsed = json.loads(json_report)
        assert parsed["summary"]["total_trades"] == 100

    def test_csv_report_generation(self):
        """Test CSV report generation."""
        generator = ReportGenerator()
        
        trades = [
            TradeResult(
                trade_id="1",
                signal_id="sig-1",
                asset="BTC",
                entry_time=datetime(2024, 1, 1),
                exit_time=datetime(2024, 1, 2),
                entry_price=50000,
                exit_price=51000,
                position_size=0.1,
                pnl=100,
                status=TradeStatus.CLOSED,
            ),
        ]
        
        csv_report = generator.generate_csv(trades)
        
        # Check CSV format
        lines = csv_report.strip().split('\n')
        assert len(lines) >= 2  # Header + at least one row
        assert "trade_id" in lines[0]
        assert "1" in lines[1]

    def test_summary_report_generation(self):
        """Test summary report generation."""
        generator = ReportGenerator()
        
        metrics = {
            "total_trades": 150,
            "win_rate": 0.68,
            "average_win": 125,
            "average_loss": -75,
            "sharpe_ratio": 1.8,
            "max_drawdown": -0.12,
        }
        
        summary = generator.generate_summary(metrics)
        
        assert "Performance Summary" in summary
        assert "Win Rate: 68.0%" in summary
        assert "Sharpe Ratio: 1.80" in summary


class TestRealTimeMetrics:
    """Test real-time metrics streaming."""

    def test_real_time_metrics_update(self):
        """Test real-time metrics updates."""
        metrics = RealTimeMetrics()
        
        # Subscribe to updates
        updates_received = []
        
        def callback(update):
            updates_received.append(update)
        
        metrics.subscribe("test_client", callback)
        
        # Add a trade
        trade = TradeResult(
            trade_id="1",
            signal_id="sig-1",
            asset="AAPL",
            entry_time=datetime.now(),
            entry_price=175,
            exit_price=176,
            position_size=100,
            pnl=100,
            status=TradeStatus.CLOSED,
        )
        
        metrics.update_trade(trade)
        
        # Check update was received
        assert len(updates_received) == 1
        assert updates_received[0]["type"] == "metrics_update"
        assert len(updates_received[0]["updates"]) == 1
        assert updates_received[0]["updates"][0]["trade_id"] == "1"

    def test_metrics_snapshot(self):
        """Test getting current metrics snapshot."""
        metrics = RealTimeMetrics()
        
        # Add some trades
        for i in range(5):
            trade = TradeResult(
                trade_id=str(i),
                signal_id=f"sig-{i}",
                asset="MSFT",
                entry_time=datetime.now() - timedelta(hours=i),
                entry_price=400,
                exit_price=402,
                position_size=10,
                pnl=20,
                status=TradeStatus.CLOSED,
            )
            metrics.update_trade(trade)
        
        snapshot = metrics.get_snapshot()
        
        assert snapshot["total_trades"] == 5
        assert snapshot["total_pnl"] == 100
        assert snapshot["last_update"] is not None

    def test_rate_limiting(self):
        """Test rate limiting of updates."""
        metrics = RealTimeMetrics(update_interval_ms=100)
        
        updates_received = []
        
        def callback(update):
            updates_received.append(update)
        
        metrics.subscribe("test_client", callback)
        
        # Add multiple trades quickly
        for i in range(10):
            trade = TradeResult(
                trade_id=str(i),
                signal_id=f"sig-{i}",
                asset="GOOGL",
                entry_time=datetime.now(),
                entry_price=150,
                exit_price=151,
                position_size=10,
                pnl=10,
                status=TradeStatus.CLOSED,
            )
            metrics.update_trade(trade)
        
        # Updates should be batched
        assert len(updates_received) < 10