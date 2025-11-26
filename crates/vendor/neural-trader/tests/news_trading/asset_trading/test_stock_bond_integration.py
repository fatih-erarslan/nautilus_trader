"""Tests for stock-bond integration and multi-asset trading."""

import pytest
from datetime import datetime

from src.news_trading.asset_trading.allocation.rotator import AssetRotator
from src.news_trading.asset_trading.allocation.balanced_portfolio import BalancedPortfolioManager
from src.news_trading.asset_trading.performance.swing_metrics import SwingPerformanceTracker


class TestStockBondIntegration:
    """Test stock-bond correlation and rotation strategies."""
    
    def test_risk_off_rotation(self):
        """Test risk-off rotation from stocks to bonds."""
        rotator = AssetRotator()
        
        # Risk-off scenario
        market_conditions = {
            "spy_trend": "declining",
            "vix": 35,
            "yield_curve": "inverting",
            "economic_data": "weakening",
            "credit_spreads": "widening",
            "dollar_strength": "strong",
        }
        
        allocation = rotator.calculate_allocation(market_conditions)
        
        assert allocation["stocks"] < 0.3  # Reduce stock allocation
        assert allocation["bonds"] > 0.5   # Increase bond allocation
        assert allocation["cash"] > 0.1    # Some cash buffer
        assert sum(allocation.values()) == pytest.approx(1.0, 0.01)
        
    def test_balanced_portfolio_signals(self):
        """Test 60/40 portfolio rebalancing signals."""
        manager = BalancedPortfolioManager(target_stock=0.6, target_bond=0.4)
        
        current_allocation = {
            "stocks": 0.70,  # Overweight after rally
            "bonds": 0.30,   # Underweight
        }
        
        rebalance_trades = manager.generate_rebalance_trades(
            current_allocation,
            portfolio_value=100000
        )
        
        assert len(rebalance_trades) == 2
        assert rebalance_trades[0]["action"] == "sell"
        assert rebalance_trades[0]["asset_class"] == "stocks"
        assert rebalance_trades[0]["amount"] == pytest.approx(10000, 1000)
        assert rebalance_trades[1]["action"] == "buy"
        assert rebalance_trades[1]["asset_class"] == "bonds"
        
    def test_correlation_based_allocation(self):
        """Test dynamic allocation based on correlations."""
        rotator = AssetRotator()
        
        # Provide correlation data
        correlation_data = {
            "stock_bond_corr": 0.7,  # High positive correlation
            "rolling_window": 60,    # 60-day correlation
            "historical_avg": -0.2,  # Usually negative
        }
        
        adjustment = rotator.correlation_adjustment(correlation_data)
        
        # When correlation is high, reduce concentration
        assert adjustment["diversification_boost"] > 0
        assert adjustment["alternative_allocation"] > 0.05
        
    def test_tactical_asset_allocation(self):
        """Test tactical asset allocation signals."""
        rotator = AssetRotator()
        
        # Momentum and valuation signals
        signals = {
            "stock_momentum": 0.8,      # Strong momentum
            "bond_momentum": -0.3,      # Weak momentum
            "stock_valuation": -0.5,    # Expensive
            "bond_valuation": 0.3,      # Fair value
            "macro_score": 0.2,         # Slightly positive
        }
        
        allocation = rotator.tactical_allocation(signals, base_stock=0.6, base_bond=0.4)
        
        # Should tilt toward momentum but consider valuation
        assert 0.55 < allocation["stocks"] < 0.70
        assert 0.30 < allocation["bonds"] < 0.45
        
    def test_regime_detection(self):
        """Test market regime detection for allocation."""
        rotator = AssetRotator()
        
        # Different market regimes
        regimes = [
            {
                "data": {"volatility": 12, "trend": "up", "correlation": -0.3},
                "expected": "risk_on",
            },
            {
                "data": {"volatility": 30, "trend": "down", "correlation": 0.5},
                "expected": "risk_off",
            },
            {
                "data": {"volatility": 18, "trend": "sideways", "correlation": 0.1},
                "expected": "neutral",
            },
        ]
        
        for regime in regimes:
            detected = rotator.detect_regime(regime["data"])
            assert detected == regime["expected"]


class TestPerformanceTracking:
    """Test swing trading performance metrics."""
    
    def test_swing_trade_metrics(self):
        """Test swing trading performance metrics."""
        tracker = SwingPerformanceTracker()
        
        # Add completed trades
        trades = [
            {"symbol": "AAPL", "entry": 170, "exit": 175, "days_held": 5, "type": "stock"},
            {"symbol": "MSFT", "entry": 380, "exit": 370, "days_held": 7, "type": "stock"},
            {"symbol": "TLT", "entry": 95, "exit": 98, "days_held": 10, "type": "bond"},
        ]
        
        for trade in trades:
            tracker.add_trade(trade)
            
        metrics = tracker.calculate_metrics()
        
        assert metrics["win_rate"] == pytest.approx(0.67, 0.01)
        assert metrics["avg_win"] > metrics["avg_loss"]
        assert metrics["profit_factor"] > 1.0
        assert metrics["avg_holding_days"] == pytest.approx(7.33, 0.1)
        
    def test_asset_class_performance(self):
        """Test performance breakdown by asset class."""
        tracker = SwingPerformanceTracker()
        
        # Add trades for different asset classes
        trades = [
            {"symbol": "SPY", "entry": 400, "exit": 410, "days_held": 3, "type": "stock"},
            {"symbol": "QQQ", "entry": 350, "exit": 340, "days_held": 5, "type": "stock"},
            {"symbol": "IEF", "entry": 100, "exit": 102, "days_held": 7, "type": "bond"},
            {"symbol": "LQD", "entry": 115, "exit": 116, "days_held": 4, "type": "bond"},
        ]
        
        for trade in trades:
            tracker.add_trade(trade)
        
        asset_metrics = tracker.metrics_by_asset_class()
        
        assert "stock" in asset_metrics
        assert "bond" in asset_metrics
        assert asset_metrics["stock"]["win_rate"] == 0.5
        assert asset_metrics["bond"]["win_rate"] == 1.0
        
    def test_trade_efficiency_metrics(self):
        """Test trade efficiency and timing metrics."""
        tracker = SwingPerformanceTracker()
        
        # Add trades with entry/exit efficiency data
        trades = [
            {
                "symbol": "AAPL",
                "entry": 150,
                "exit": 155,
                "days_held": 4,
                "entry_efficiency": 0.85,  # Entered 85% from low
                "exit_efficiency": 0.90,   # Exited 90% from high
            },
        ]
        
        for trade in trades:
            tracker.add_trade(trade)
        
        efficiency = tracker.calculate_efficiency_metrics()
        
        assert efficiency["avg_entry_efficiency"] == 0.85
        assert efficiency["avg_exit_efficiency"] == 0.90
        assert efficiency["timing_score"] > 0.8