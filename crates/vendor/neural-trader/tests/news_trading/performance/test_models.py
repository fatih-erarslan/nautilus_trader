"""Tests for performance tracking models."""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal

from src.news_trading.performance.models import (
    TradeStatus,
    TradeResult,
    PerformanceMetrics,
    Attribution,
    SourceMetrics,
    ModelMetrics,
)


class TestTradeResult:
    """Test the TradeResult model."""

    def test_trade_result_creation(self):
        """Test creating a TradeResult instance."""
        result = TradeResult(
            trade_id="trade-123",
            signal_id="signal-456",
            asset="BTC",
            entry_time=datetime.now() - timedelta(hours=2),
            exit_time=datetime.now(),
            entry_price=45000.0,
            exit_price=46500.0,
            position_size=0.05,
            pnl=75.0,
            pnl_percentage=3.33,
            status=TradeStatus.CLOSED,
            news_events=["news-001", "news-002"],
            sentiment_scores=[0.8, 0.75],
            fees=5.0,
        )

        assert result.trade_id == "trade-123"
        assert result.pnl == 75.0
        assert result.pnl_percentage == 3.33
        assert result.status == TradeStatus.CLOSED
        assert len(result.news_events) == 2

    def test_trade_result_validation(self):
        """Test TradeResult validation."""
        # Test negative position size should raise error
        with pytest.raises(ValueError):
            TradeResult(
                trade_id="trade-123",
                signal_id="signal-456",
                asset="BTC",
                entry_time=datetime.now(),
                exit_time=None,
                entry_price=45000.0,
                exit_price=None,
                position_size=-0.05,  # Invalid
                status=TradeStatus.OPEN,
            )

    def test_trade_result_pnl_calculation(self):
        """Test P&L calculation methods."""
        result = TradeResult(
            trade_id="trade-123",
            signal_id="signal-456",
            asset="AAPL",
            entry_time=datetime.now() - timedelta(hours=1),
            exit_time=datetime.now(),
            entry_price=150.0,
            exit_price=153.0,
            position_size=100,  # 100 shares
            fees=2.0,
            status=TradeStatus.CLOSED,
        )

        # Calculate P&L
        result.calculate_pnl()
        
        assert result.pnl == 298.0  # (153-150)*100 - 2
        assert result.pnl_percentage == pytest.approx(1.9866, 0.01)  # 298/(150*100) * 100


class TestPerformanceMetrics:
    """Test the PerformanceMetrics model."""

    def test_performance_metrics_creation(self):
        """Test creating PerformanceMetrics."""
        metrics = PerformanceMetrics(
            total_trades=100,
            winning_trades=60,
            losing_trades=40,
            win_rate=0.60,
            average_win=50.0,
            average_loss=-30.0,
            profit_factor=2.5,
            sharpe_ratio=1.5,
            max_drawdown=-0.15,
            total_pnl=1200.0,
        )

        assert metrics.total_trades == 100
        assert metrics.win_rate == 0.60
        assert metrics.profit_factor == 2.5

    def test_performance_metrics_validation(self):
        """Test PerformanceMetrics validation."""
        # Win rate should be between 0 and 1
        with pytest.raises(ValueError):
            PerformanceMetrics(
                total_trades=100,
                winning_trades=60,
                losing_trades=40,
                win_rate=1.5,  # Invalid
                average_win=50.0,
                average_loss=-30.0,
                profit_factor=2.5,
                sharpe_ratio=1.5,
                max_drawdown=-0.15,
                total_pnl=1200.0,
            )

    def test_performance_metrics_from_trades(self):
        """Test calculating metrics from trade list."""
        trades = [
            TradeResult(
                trade_id=f"trade-{i}",
                signal_id=f"signal-{i}",
                asset="BTC",
                entry_time=datetime.now() - timedelta(days=i),
                exit_time=datetime.now() - timedelta(days=i-1),
                entry_price=45000.0,
                exit_price=45000.0 + (100 if i % 2 == 0 else -50),
                position_size=0.01,
                pnl=1.0 if i % 2 == 0 else -0.5,
                status=TradeStatus.CLOSED,
            )
            for i in range(10)
        ]

        metrics = PerformanceMetrics.from_trades(trades)
        
        assert metrics.total_trades == 10
        assert metrics.winning_trades == 5
        assert metrics.losing_trades == 5
        assert metrics.win_rate == 0.5


class TestAttribution:
    """Test the Attribution model."""

    def test_attribution_creation(self):
        """Test creating Attribution instance."""
        attribution = Attribution(
            source_contributions={
                "reuters": 0.6,
                "bloomberg": 0.3,
                "twitter": 0.1,
            },
            primary_catalyst="news-001",
            news_weights={
                "news-001": 0.7,
                "news-002": 0.2,
                "news-003": 0.1,
            },
            confidence_score=0.85,
        )

        assert attribution.source_contributions["reuters"] == 0.6
        assert attribution.primary_catalyst == "news-001"
        assert attribution.confidence_score == 0.85

    def test_attribution_normalization(self):
        """Test that source contributions sum to 1."""
        attribution = Attribution(
            source_contributions={
                "reuters": 0.6,
                "bloomberg": 0.3,
                "twitter": 0.1,
            },
            primary_catalyst="news-001",
            news_weights={},
        )

        total = sum(attribution.source_contributions.values())
        assert pytest.approx(total, 0.01) == 1.0


class TestSourceMetrics:
    """Test the SourceMetrics model."""

    def test_source_metrics_creation(self):
        """Test creating SourceMetrics."""
        metrics = SourceMetrics(
            source_name="reuters",
            total_signals=1000,
            profitable_signals=650,
            accuracy_rate=0.65,
            average_pnl_per_signal=25.0,
            weighted_pnl=16250.0,
            signal_quality_score=0.75,
        )

        assert metrics.source_name == "reuters"
        assert metrics.accuracy_rate == 0.65
        assert metrics.signal_quality_score == 0.75


class TestModelMetrics:
    """Test the ModelMetrics model."""

    def test_model_metrics_creation(self):
        """Test creating ModelMetrics."""
        metrics = ModelMetrics(
            model_name="finbert_v1",
            prediction_count=5000,
            mae=0.05,
            rmse=0.08,
            confidence_calibration=0.92,
            average_confidence=0.85,
            directional_accuracy=0.73,
        )

        assert metrics.model_name == "finbert_v1"
        assert metrics.mae == 0.05
        assert metrics.directional_accuracy == 0.73