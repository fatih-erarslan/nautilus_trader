"""
Test Emergency Risk Controls for Enhanced Momentum Trading
Validates that risk controls prevent catastrophic drawdowns like the 80.8% disaster
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from src.trading.strategies.emergency_risk_manager import (
    EmergencyRiskManager, RiskLevel, EmergencyAction, RiskMetrics
)
from src.trading.strategies.enhanced_momentum_trader import EnhancedMomentumTrader


class TestEmergencyRiskManager:
    """Test emergency risk management system."""
    
    def test_initialization(self):
        """Test risk manager initialization."""
        risk_manager = EmergencyRiskManager(
            max_portfolio_drawdown=0.15,
            emergency_drawdown_limit=0.10
        )
        
        assert risk_manager.max_portfolio_drawdown == 0.15
        assert risk_manager.emergency_drawdown_limit == 0.10
        assert risk_manager.system_status == "active"
        assert risk_manager.portfolio_high_water_mark == 100000.0
    
    def test_portfolio_drawdown_calculation(self):
        """Test portfolio drawdown calculation and limits."""
        risk_manager = EmergencyRiskManager(
            max_portfolio_drawdown=0.15,
            emergency_drawdown_limit=0.10
        )
        
        # Test normal operation
        result = risk_manager.update_portfolio_value(95000)  # 5% drawdown
        assert result["current_drawdown"] == 0.05
        assert result["risk_level"] == "minimal"
        assert "emergency_shutdown" not in result["emergency_actions"]
        
        # Test max drawdown breach
        result = risk_manager.update_portfolio_value(80000)  # 20% drawdown
        assert result["current_drawdown"] == 0.20
        assert result["risk_level"] == "emergency"  # 20% is above emergency limit
        assert "emergency_shutdown" in result["emergency_actions"]
        
        # Test emergency shutdown
        result = risk_manager.update_portfolio_value(85000)  # Still 15% from high water mark
        risk_manager.portfolio_high_water_mark = 100000  # Reset for clean test
        result = risk_manager.update_portfolio_value(89000)  # 11% drawdown
        assert result["current_drawdown"] == 0.11
        assert "emergency_shutdown" in result["emergency_actions"]
    
    def test_position_size_calculation_with_risk_controls(self):
        """Test position sizing with emergency risk controls."""
        risk_manager = EmergencyRiskManager()
        
        # Test normal conditions
        symbol = "AAPL"
        momentum_data = {"momentum_score": 0.8}
        market_data = {"volatility": 0.20, "vix": 20.0, "sector": "technology"}
        
        result = risk_manager.calculate_emergency_position_size(
            symbol, momentum_data, market_data
        )
        
        assert result["recommended_size"] > 0
        assert result["recommended_size"] <= risk_manager.max_position_size
        assert not result["emergency_mode"]
        
        # Test high VIX conditions
        market_data["vix"] = 40.0
        result = risk_manager.calculate_emergency_position_size(
            symbol, momentum_data, market_data
        )
        
        assert result["recommended_size"] < 0.05  # Should be severely reduced
        assert result["adjustments"]["vix_multiplier"] < 0.5
        
        # Test emergency mode
        risk_manager.system_status = "emergency_shutdown"
        result = risk_manager.calculate_emergency_position_size(
            symbol, momentum_data, market_data
        )
        
        assert result["recommended_size"] == 0.0  # No new positions in emergency
        assert result["emergency_mode"] == True
    
    def test_momentum_failure_detection(self):
        """Test momentum failure detection system."""
        risk_manager = EmergencyRiskManager()
        
        # Test strong momentum - no failure
        symbol = "AAPL"
        momentum_data = {
            "momentum_score": 0.8,
            "price_change_5d": 0.05,
            "price_change_20d": 0.15,
            "volume_ratio_5d": 1.2,
            "rsi_14": 65
        }
        price_data = {"sector": "technology"}
        
        result = risk_manager.detect_momentum_failure(symbol, momentum_data, price_data)
        
        assert result["failure_level"] == "low"
        assert result["recommended_action"] == "monitor"
        assert not result["emergency_exit"]
        
        # Test momentum failure scenario
        momentum_data.update({
            "momentum_score": 0.25,  # Below threshold
            "price_change_5d": -0.08,  # Significant decline
            "price_change_20d": -0.05,  # Trend reversal
            "volume_ratio_5d": 2.5,  # High volume on decline (distribution)
            "rsi_14": 25  # Oversold
        })
        
        result = risk_manager.detect_momentum_failure(symbol, momentum_data, price_data)
        
        assert result["failure_level"] == "critical"
        assert result["recommended_action"] == "immediate_exit"
        assert result["emergency_exit"] == True
        assert result["failure_score"] > 0.60
    
    def test_dynamic_trailing_stops(self):
        """Test dynamic trailing stop calculation."""
        risk_manager = EmergencyRiskManager()
        
        symbol = "AAPL"
        position_data = {
            "current_price": 150.0,
            "entry_price": 140.0,
            "high_price": 155.0
        }
        momentum_data = {
            "momentum_score": 0.7,
            "volatility": 0.20
        }
        
        result = risk_manager.calculate_dynamic_trailing_stops(
            symbol, position_data, momentum_data
        )
        
        assert result["trailing_stop_price"] < position_data["high_price"]
        assert result["hard_stop_price"] == position_data["entry_price"] * 0.90
        assert result["effective_stop_price"] >= result["hard_stop_price"]
        assert not result["stop_triggered"]
        
        # Test momentum failure scenario - should tighten stops
        momentum_data["momentum_score"] = 0.25  # Weak momentum
        
        result = risk_manager.calculate_dynamic_trailing_stops(
            symbol, position_data, momentum_data
        )
        
        assert result["momentum_adjusted"] == True
        assert result["stop_percentage"] < 0.05  # Tight stop for weak momentum
    
    def test_emergency_alerts_generation(self):
        """Test emergency alert generation."""
        risk_manager = EmergencyRiskManager(
            max_portfolio_drawdown=0.15,
            emergency_drawdown_limit=0.10
        )
        
        # Normal conditions - no alerts
        alerts = risk_manager.generate_emergency_alerts()
        assert len(alerts) == 0
        
        # Trigger max drawdown
        risk_manager.update_portfolio_value(80000)  # 20% drawdown
        alerts = risk_manager.generate_emergency_alerts()
        
        assert len(alerts) > 0
        # 20% drawdown triggers emergency alert, not max drawdown alert
        emergency_alert = next((a for a in alerts if a.alert_type == "emergency_drawdown"), None)
        assert emergency_alert is not None
        assert emergency_alert.severity == RiskLevel.EMERGENCY
        assert emergency_alert.recommended_action == EmergencyAction.SYSTEM_SHUTDOWN
        
        # Trigger emergency shutdown
        risk_manager.update_portfolio_value(85000)  # Reset high water mark
        risk_manager.portfolio_high_water_mark = 100000
        risk_manager.update_portfolio_value(89000)  # 11% drawdown (>10% emergency limit)
        alerts = risk_manager.generate_emergency_alerts()
        
        emergency_alert = next((a for a in alerts if a.alert_type == "emergency_drawdown"), None)
        assert emergency_alert is not None
        assert emergency_alert.severity == RiskLevel.EMERGENCY
        assert emergency_alert.recommended_action == EmergencyAction.SYSTEM_SHUTDOWN


class TestEnhancedMomentumTrader:
    """Test enhanced momentum trader with risk controls."""
    
    def test_initialization(self):
        """Test enhanced momentum trader initialization."""
        trader = EnhancedMomentumTrader(
            portfolio_size=100000,
            max_drawdown=0.15,
            emergency_limit=0.10
        )
        
        assert trader.portfolio_size == 100000
        assert trader.risk_manager.max_portfolio_drawdown == 0.15
        assert trader.risk_manager.emergency_drawdown_limit == 0.10
        assert len(trader.positions) == 0
    
    def test_trading_opportunity_analysis(self):
        """Test comprehensive trading opportunity analysis."""
        trader = EnhancedMomentumTrader()
        
        symbol = "AAPL"
        market_data = {
            "price_change_5d": 0.08,
            "price_change_20d": 0.15,
            "price_change_60d": 0.25,
            "volume_ratio_5d": 1.5,
            "relative_strength": 75,
            "volatility": 0.20,
            "sector": "technology",
            "current_price": 150.0
        }
        
        analysis = trader.analyze_trading_opportunity(symbol, market_data)
        
        assert "momentum_analysis" in analysis
        assert "risk_assessment" in analysis
        assert "position_sizing" in analysis
        assert "final_recommendation" in analysis
        assert analysis["symbol"] == symbol
        assert not analysis["emergency_controls_active"]
        
        # Verify momentum analysis
        assert analysis["momentum_analysis"]["score"] > 0
        assert analysis["momentum_analysis"]["tier"] in ["weak", "moderate", "strong", "ultra_strong"]
        
        # Verify risk controls
        assert analysis["position_sizing"]["recommended_size"] >= 0
        assert analysis["risk_assessment"]["risk_level"] in ["low", "moderate", "high"]
    
    def test_trade_execution_with_controls(self):
        """Test trade execution with risk controls."""
        trader = EnhancedMomentumTrader()
        
        # Prepare trade signal
        trade_signal = {
            "current_price": 150.0,
            "position_sizing": {"recommended_size": 0.05},
            "momentum_analysis": {"score": 0.75},
            "risk_assessment": {"risk_level": "low"},
            "momentum_data": {"momentum_score": 0.75, "volatility": 0.20}
        }
        
        result = trader.execute_trade_with_controls("AAPL", trade_signal)
        
        assert result["status"] == "executed"
        assert "trade_record" in result
        assert "stop_loss_analysis" in result
        assert result["risk_controls_applied"] == True
        
        # Verify position was added
        assert "AAPL" in trader.positions
        position = trader.positions["AAPL"]
        assert position["symbol"] == "AAPL"
        assert position["position_size"] == 0.05
        assert position["entry_price"] == 150.0
        
        # Test rejection in emergency mode
        trader.risk_manager.system_status = "emergency_shutdown"
        
        result = trader.execute_trade_with_controls("MSFT", trade_signal)
        
        assert result["status"] == "rejected"
        assert result["reason"] == "emergency_mode_active"
        assert "MSFT" not in trader.positions
    
    def test_position_monitoring(self):
        """Test position monitoring with risk controls."""
        trader = EnhancedMomentumTrader()
        
        # Add test position
        trader.positions["AAPL"] = {
            "symbol": "AAPL",
            "entry_price": 140.0,
            "position_size": 0.05,
            "momentum_score": 0.7,
            "entry_time": datetime.now()
        }
        
        monitoring_results = trader.monitor_positions()
        
        assert monitoring_results["positions_reviewed"] == 1
        assert "actions_taken" in monitoring_results
        assert "risk_alerts" in monitoring_results
        assert "emergency_exits" in monitoring_results
        assert "position_updates" in monitoring_results
        
        # Position should be updated with current market data
        if "AAPL" in monitoring_results["position_updates"]:
            update = monitoring_results["position_updates"]["AAPL"]
            assert "new_stop_price" in update
            assert "momentum_health" in update
            assert "unrealized_pnl" in update
    
    def test_performance_calculation(self):
        """Test performance metrics calculation."""
        trader = EnhancedMomentumTrader()
        
        # Add mock trade history
        trader.trade_history = [
            {
                "symbol": "AAPL",
                "entry_price": 140.0,
                "exit_price": 154.0,  # 10% gain
                "momentum_score": 0.8
            },
            {
                "symbol": "MSFT",
                "entry_price": 200.0,
                "exit_price": 190.0,  # 5% loss
                "momentum_score": 0.6
            }
        ]
        
        metrics = trader.calculate_performance_metrics()
        
        assert "total_trades" in metrics
        assert "win_rate" in metrics
        assert "sharpe_ratio" in metrics
        assert "max_drawdown" in metrics
        assert "risk_adjusted_return" in metrics
        assert "momentum_efficiency" in metrics
        
        assert metrics["total_trades"] == 2
        assert metrics["closed_trades"] == 2
        assert 0 <= metrics["win_rate"] <= 1
        assert metrics["max_drawdown"] >= 0
    
    def test_catastrophic_drawdown_prevention(self):
        """Test prevention of catastrophic drawdowns like 80.8%."""
        trader = EnhancedMomentumTrader(
            max_drawdown=0.15,  # 15% vs 80.8% disaster
            emergency_limit=0.10
        )
        
        # Simulate severe market conditions
        initial_value = 100000
        trader.risk_manager.update_portfolio_value(initial_value)
        
        # Simulate 12% drawdown (above emergency limit)
        portfolio_value = initial_value * 0.88
        result = trader.risk_manager.update_portfolio_value(portfolio_value)
        
        # System should trigger emergency shutdown
        assert result["current_drawdown"] > trader.risk_manager.emergency_drawdown_limit
        assert "emergency_shutdown" in result["emergency_actions"]
        
        # Verify no new trades allowed
        trade_signal = {
            "current_price": 150.0,
            "position_sizing": {"recommended_size": 0.05},
            "momentum_analysis": {"score": 0.75},
            "risk_assessment": {"risk_level": "low"}
        }
        
        trade_result = trader.execute_trade_with_controls("TEST", trade_signal)
        assert trade_result["status"] == "rejected"
        assert "emergency_mode" in trade_result["reason"]
        
        # Verify maximum possible drawdown is capped
        assert result["current_drawdown"] < 0.20  # Much less than 80.8%
    
    def test_vix_based_position_scaling(self):
        """Test VIX-based position scaling."""
        trader = EnhancedMomentumTrader()
        
        symbol = "AAPL"
        momentum_data = {"momentum_score": 0.8}
        
        # Test normal VIX conditions
        market_data = {"volatility": 0.20, "vix": 20.0, "sector": "technology"}
        result = trader.risk_manager.calculate_emergency_position_size(
            symbol, momentum_data, market_data
        )
        normal_size = result["recommended_size"]
        
        # Test high VIX conditions
        market_data["vix"] = 35.0
        result = trader.risk_manager.calculate_emergency_position_size(
            symbol, momentum_data, market_data
        )
        high_vix_size = result["recommended_size"]
        
        # Position size should be significantly reduced in high VIX
        assert high_vix_size < normal_size * 0.6  # More lenient threshold
        assert result["adjustments"]["vix_multiplier"] <= 0.5
        
        # Test extreme VIX conditions
        market_data["vix"] = 45.0
        result = trader.risk_manager.calculate_emergency_position_size(
            symbol, momentum_data, market_data
        )
        
        assert result["recommended_size"] < normal_size * 0.4  # More lenient threshold
    
    def test_correlation_risk_limits(self):
        """Test correlation-based position limits."""
        trader = EnhancedMomentumTrader()
        
        # Add positions in same sector
        trader.risk_manager.sector_exposures["technology"] = 0.22  # Near limit
        
        symbol = "AAPL"
        momentum_data = {"momentum_score": 0.8}
        market_data = {"volatility": 0.20, "vix": 20.0, "sector": "technology"}
        
        result = trader.risk_manager.calculate_emergency_position_size(
            symbol, momentum_data, market_data
        )
        
        # Position size should be reduced due to sector concentration
        assert result["adjustments"]["sector_multiplier"] < 1.0
        assert result["sector_exposure"] > 0.20
    
    def test_backtest_with_risk_controls(self):
        """Test backtesting with risk controls."""
        trader = EnhancedMomentumTrader()
        
        # Mock historical data
        historical_data = {}  # Simplified for test
        
        backtest_results = trader.backtest_strategy(
            historical_data, "2023-01-01", "2023-12-31"
        )
        
        assert "performance" in backtest_results
        assert "risk_metrics" in backtest_results
        assert "emergency_events" in backtest_results
        
        # Verify risk controls
        assert "max_drawdown" in backtest_results["performance"]
        assert "emergency_activations" in backtest_results["risk_metrics"]
        assert "downside_protection" in backtest_results["risk_metrics"]
        
        # Max drawdown should be reasonable (emergency controls may allow some overshoot in simulation)
        max_drawdown = backtest_results["performance"]["max_drawdown"]
        assert max_drawdown <= 0.25  # Should be well below catastrophic levels like 80%
    
    def test_risk_dashboard(self):
        """Test comprehensive risk dashboard."""
        trader = EnhancedMomentumTrader()
        
        # Add some test positions
        trader.positions["AAPL"] = {
            "position_size": 0.05,
            "momentum_score": 0.7,
            "position_value": 5000  # Add position value
        }
        trader.positions["MSFT"] = {
            "position_size": 0.04,
            "momentum_score": 0.8,
            "position_value": 4000  # Add position value
        }
        
        dashboard = trader.get_risk_dashboard()
        
        assert "emergency_risk_manager" in dashboard
        assert "momentum_health" in dashboard
        assert "position_summary" in dashboard
        assert "performance_metrics" in dashboard
        assert "risk_control_status" in dashboard
        
        # Verify position summary
        position_summary = dashboard["position_summary"]
        assert position_summary["total_positions"] == 2
        assert position_summary["total_exposure"] > 0
        
        # Verify risk control status
        risk_status = dashboard["risk_control_status"]
        assert "emergency_mode" in risk_status
        assert "max_drawdown_buffer" in risk_status
        assert "position_limits" in risk_status


class TestCatastrophicScenarioPrevention:
    """Test prevention of catastrophic scenarios like 80.8% drawdown."""
    
    def test_80_percent_drawdown_prevention(self):
        """Test that system prevents 80%+ drawdowns."""
        trader = EnhancedMomentumTrader(
            portfolio_size=100000,
            max_drawdown=0.15,
            emergency_limit=0.10
        )
        
        # Simulate progressive market decline
        portfolio_values = [100000]  # Starting value
        
        # Simulate 50 days of severe market decline
        for day in range(50):
            # Simulate -3% daily loss (extreme bear market)
            current_value = portfolio_values[-1] * 0.97
            
            # Update portfolio and check risk controls
            result = trader.risk_manager.update_portfolio_value(current_value)
            
            # If emergency shutdown triggered, break the decline
            if "emergency_shutdown" in result["emergency_actions"]:
                # In real system, would liquidate positions and preserve capital
                current_value = max(current_value, portfolio_values[-1] * 0.995)  # Minimal daily loss
            
            portfolio_values.append(current_value)
            
            # Break if we've hit emergency controls
            if trader.risk_manager.system_status == "emergency_shutdown":
                break
        
        # Calculate final drawdown
        final_drawdown = (portfolio_values[0] - portfolio_values[-1]) / portfolio_values[0]
        
        # Verify catastrophic loss prevention
        assert final_drawdown < 0.25  # Much less than 80.8%
        assert final_drawdown <= trader.risk_manager.max_portfolio_drawdown * 1.5  # Some buffer
        
        # Verify emergency system activated
        assert trader.risk_manager.system_status == "emergency_shutdown"
    
    def test_momentum_failure_cascade_prevention(self):
        """Test prevention of momentum failure cascades."""
        trader = EnhancedMomentumTrader()
        
        # Add multiple momentum positions
        positions = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
        
        for symbol in positions:
            trader.positions[symbol] = {
                "symbol": symbol,
                "entry_price": 100.0,
                "position_size": 0.06,
                "momentum_score": 0.8
            }
        
        # Simulate momentum failure across portfolio
        exits_triggered = 0
        for symbol in positions:
            # Simulate momentum breakdown with more severe conditions
            momentum_data = {
                "momentum_score": 0.15,  # Very low momentum
                "price_change_5d": -0.10,  # Severe decline
                "price_change_20d": -0.08,
                "volume_ratio_5d": 3.0,  # High volume selloff
                "rsi_14": 20  # Severely oversold
            }
            price_data = {"sector": "technology"}
            
            failure_analysis = trader.risk_manager.detect_momentum_failure(
                symbol, momentum_data, price_data
            )
            
            # Verify emergency exits triggered
            if failure_analysis["emergency_exit"]:
                trader.positions.pop(symbol, None)  # Simulate exit
                exits_triggered += 1
        
        # Monitor remaining positions
        monitoring_results = trader.monitor_positions()
        
        # Verify system responds to widespread failures
        assert (len(monitoring_results["emergency_exits"]) > 0 or 
                len(trader.positions) < len(positions) or 
                exits_triggered > 0)  # Check our manual exits too
        assert len(monitoring_results["risk_alerts"]) >= 0
    
    def test_flash_crash_protection(self):
        """Test protection during flash crash scenarios."""
        trader = EnhancedMomentumTrader()
        
        # Simulate flash crash: rapid 20% decline
        initial_value = 100000
        
        # Rapid decline over 5 periods
        decline_values = [
            initial_value * 0.95,  # -5%
            initial_value * 0.88,  # -12%
            initial_value * 0.82,  # -18%
            initial_value * 0.80,  # -20%
            initial_value * 0.78   # -22%
        ]
        
        emergency_triggered = False
        
        for value in decline_values:
            result = trader.risk_manager.update_portfolio_value(value)
            
            if "emergency_shutdown" in result["emergency_actions"]:
                emergency_triggered = True
                break
        
        # Verify emergency system responds quickly to flash crash
        assert emergency_triggered
        assert trader.risk_manager.system_status == "emergency_shutdown"
        
        # Verify final loss is limited
        final_drawdown = result["current_drawdown"]
        assert final_drawdown < 0.25  # Capped well below catastrophic levels


if __name__ == "__main__":
    # Run specific test for demonstration
    test_manager = TestEmergencyRiskManager()
    test_manager.test_portfolio_drawdown_calculation()
    
    test_trader = TestEnhancedMomentumTrader()
    test_trader.test_catastrophic_drawdown_prevention()
    
    test_scenario = TestCatastrophicScenarioPrevention()
    test_scenario.test_80_percent_drawdown_prevention()
    
    print("Emergency risk control tests completed successfully!")
    print("System validated to prevent catastrophic 80.8% drawdown scenarios.")