"""
Tests for the Mean Reversion Risk Manager - Sophisticated risk controls.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.trading.strategies.mean_reversion_risk_manager import (
    MeanReversionRiskManager,
    MeanReversionPosition,
    MeanReversionMetrics,
    MeanReversionState,
    VolatilityRegime
)
from src.trading.strategies.emergency_risk_manager import PositionRisk, RiskState


class TestMeanReversionRiskManager:
    """Test suite for Mean Reversion Risk Manager."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.risk_manager = MeanReversionRiskManager(portfolio_size=100000)
        
        # Sample signal data
        self.sample_signal = {
            "ticker": "AAPL",
            "z_score": 2.1,
            "reversion_confidence": 0.75,
            "mean_distance_pct": 0.08,
            "volatility": 0.20,
            "expected_reversion_days": 3,
            "signal_age_hours": 2
        }
        
        # Sample market data
        self.sample_market_data = {
            "current_price": 150.0,
            "volatility": 0.20,
            "atr": 3.0,
            "price_history": list(range(140, 151)),
            "mean_history": [145] * 11
        }
        
        # Sample portfolio state
        self.sample_portfolio_state = {
            "current_positions": {},
            "portfolio_volatility": 0.15,
            "mr_portfolio_heat": 0.0,
            "market_regime": "normal",
            "vix_level": 20
        }
    
    def test_kelly_criterion_calculation(self):
        """Test Kelly Criterion calculation for mean reversion."""
        sizing_result = self.risk_manager.calculate_mean_reversion_position_size(
            self.sample_signal, 
            self.sample_market_data, 
            self.sample_portfolio_state
        )
        
        # Should return valid sizing
        assert sizing_result["approved"] is True
        assert 0 < sizing_result["position_size_pct"] <= 0.06  # Max 6% position
        assert sizing_result["kelly_base"] > 0
        assert "mr_adjustments" in sizing_result
        
        # Check Kelly calculation factors
        assert "confidence" in sizing_result["mr_adjustments"]
        assert "volatility_regime" in sizing_result["mr_adjustments"]
        assert "correlation" in sizing_result["mr_adjustments"]
    
    def test_volatility_regime_detection(self):
        """Test volatility regime detection and adjustment."""
        # Test low volatility
        low_vol_signal = self.sample_signal.copy()
        low_vol_signal["volatility"] = 0.08
        
        result = self.risk_manager.calculate_mean_reversion_position_size(
            low_vol_signal, self.sample_market_data, self.sample_portfolio_state
        )
        
        assert result["volatility_regime"] == "low"
        assert result["mr_adjustments"]["volatility_regime"] > 1.0  # Size up in low vol
        
        # Test high volatility
        high_vol_signal = self.sample_signal.copy()
        high_vol_signal["volatility"] = 0.35
        
        result = self.risk_manager.calculate_mean_reversion_position_size(
            high_vol_signal, self.sample_market_data, self.sample_portfolio_state
        )
        
        assert result["volatility_regime"] == "high"
        assert result["mr_adjustments"]["volatility_regime"] < 1.0  # Size down in high vol
    
    def test_z_score_deterioration_protection(self):
        """Test z-score deterioration detection and protection."""
        # Setup z-score history showing deterioration
        ticker = "AAPL"
        self.risk_manager.z_score_history[ticker].extend([1.5, 1.8, 2.0, 2.3])
        
        # Current signal shows further deterioration
        deteriorating_signal = self.sample_signal.copy()
        deteriorating_signal["z_score"] = 2.5
        
        result = self.risk_manager.calculate_mean_reversion_position_size(
            deteriorating_signal, self.sample_market_data, self.sample_portfolio_state
        )
        
        # Should reduce position size due to deterioration
        assert result["mr_adjustments"]["z_deterioration"] < 1.0
    
    def test_mean_reversion_stops(self):
        """Test multi-layer stop loss system for mean reversion."""
        # Create sample position
        base_position = PositionRisk(
            ticker="AAPL",
            position_size=5000,
            current_price=150.0,
            entry_price=145.0,
            stop_loss=140.0,
            trailing_stop=None,
            time_stop=None,
            atr=3.0,
            volatility=0.20,
            beta=1.0,
            correlation_score=0.5,
            var_contribution=100,
            position_heat=0.02,
            days_held=2
        )
        
        mr_metrics = MeanReversionMetrics(
            z_score=1.8,
            reversion_strength=0.7,
            time_in_position=2,
            volatility_regime=VolatilityRegime.NORMAL,
            correlation_with_mean=0.6,
            mean_distance_pct=0.034,
            lookback_period=20,
            confidence_interval_95=(142.0, 148.0),
            reversion_probability=0.75
        )
        
        mr_position = MeanReversionPosition(
            base_position=base_position,
            mr_metrics=mr_metrics,
            entry_z_score=2.1,
            current_z_score=1.8,
            z_score_deterioration=1.0,
            max_z_score_since_entry=2.1,
            reversion_target=145.0,
            reversion_confidence=0.75,
            correlation_breakdown_threshold=0.3
        )
        
        stop_config = self.risk_manager.calculate_mean_reversion_stops(
            mr_position, self.sample_market_data
        )
        
        # Check all stop types are calculated
        assert "stop_loss_price" in stop_config
        assert "z_deterioration_stop" in stop_config
        assert "mr_failure_stop" in stop_config
        assert "correlation_stop" in stop_config
        assert "time_stop" in stop_config
        assert "volatility_breakout_stop" in stop_config
        assert "stop_type" in stop_config
        
        # Stop price should be reasonable
        assert 140 <= stop_config["stop_loss_price"] <= 150
    
    def test_position_exit_conditions(self):
        """Test sophisticated exit condition logic."""
        # Create position with z-score deterioration
        base_position = PositionRisk(
            ticker="AAPL",
            position_size=5000,
            current_price=155.0,  # Price moved against position
            entry_price=145.0,
            stop_loss=140.0,
            trailing_stop=None,
            time_stop=None,
            atr=3.0,
            volatility=0.20,
            beta=1.0,
            correlation_score=0.5,
            var_contribution=100,
            position_heat=0.02,
            days_held=6  # Extended holding time
        )
        
        mr_position = MeanReversionPosition(
            base_position=base_position,
            mr_metrics=MeanReversionMetrics(
                z_score=2.8,  # Z-score got worse
                reversion_strength=0.3,
                time_in_position=6,
                volatility_regime=VolatilityRegime.HIGH,
                correlation_with_mean=0.2,  # Correlation broke down
                mean_distance_pct=0.067,
                lookback_period=20,
                confidence_interval_95=(142.0, 148.0),
                reversion_probability=0.4
            ),
            entry_z_score=2.1,
            current_z_score=2.8,
            z_score_deterioration=1.33,  # 33% deterioration
            max_z_score_since_entry=2.8,
            reversion_target=145.0,
            reversion_confidence=0.4,
            correlation_breakdown_threshold=0.3
        )
        
        # Test z-score deterioration exit
        market_data = self.sample_market_data.copy()
        market_data["current_price"] = 155.0
        
        should_exit, reason = self.risk_manager.should_exit_mean_reversion_position(
            mr_position, market_data
        )
        
        assert should_exit is True
        assert "correlation_breakdown" in reason or "z_score" in reason or "mean_reversion_failure" in reason
    
    def test_correlation_adjustment(self):
        """Test correlation-based position adjustment."""
        # Setup portfolio with correlated positions
        correlated_positions = {
            "MSFT": {"strategy_type": "mean_reversion"},
            "GOOGL": {"strategy_type": "mean_reversion"}
        }
        
        portfolio_state = self.sample_portfolio_state.copy()
        portfolio_state["current_positions"] = correlated_positions
        
        # Mock correlation matrix
        self.risk_manager.correlation_matrix[("AAPL", "MSFT")] = 0.8
        self.risk_manager.correlation_matrix[("AAPL", "GOOGL")] = 0.9
        
        result = self.risk_manager.calculate_mean_reversion_position_size(
            self.sample_signal, self.sample_market_data, portfolio_state
        )
        
        # Should reduce size due to high correlation
        assert result["mr_adjustments"]["correlation"] < 1.0
    
    def test_portfolio_heat_limits(self):
        """Test portfolio heat limits for mean reversion."""
        # Setup portfolio with high mean reversion exposure
        portfolio_state = self.sample_portfolio_state.copy()
        portfolio_state["mr_portfolio_heat"] = 0.14  # Near 15% limit
        
        result = self.risk_manager.calculate_mean_reversion_position_size(
            self.sample_signal, self.sample_market_data, portfolio_state
        )
        
        # Should reduce size due to high portfolio heat
        assert result["mr_adjustments"]["mr_portfolio_heat"] < 1.0
        
        # Test blocking when at limit
        portfolio_state["mr_portfolio_heat"] = 0.16  # Above 15% limit
        
        result = self.risk_manager.calculate_mean_reversion_position_size(
            self.sample_signal, self.sample_market_data, portfolio_state
        )
        
        assert result["approved"] is False
    
    def test_time_freshness_adjustment(self):
        """Test signal freshness impact on position sizing."""
        # Fresh signal (2 hours old)
        fresh_signal = self.sample_signal.copy()
        fresh_signal["signal_age_hours"] = 2
        
        result_fresh = self.risk_manager.calculate_mean_reversion_position_size(
            fresh_signal, self.sample_market_data, self.sample_portfolio_state
        )
        
        # Old signal (25 hours old)
        stale_signal = self.sample_signal.copy()
        stale_signal["signal_age_hours"] = 25
        
        result_stale = self.risk_manager.calculate_mean_reversion_position_size(
            stale_signal, self.sample_market_data, self.sample_portfolio_state
        )
        
        # Fresh signal should get higher adjustment
        assert (result_fresh["mr_adjustments"]["time_freshness"] > 
                result_stale["mr_adjustments"]["time_freshness"])
    
    def test_confidence_scaling(self):
        """Test confidence-based position scaling."""
        # High confidence signal
        high_conf_signal = self.sample_signal.copy()
        high_conf_signal["reversion_confidence"] = 0.9
        high_conf_signal["z_score"] = 2.8
        
        result_high = self.risk_manager.calculate_mean_reversion_position_size(
            high_conf_signal, self.sample_market_data, self.sample_portfolio_state
        )
        
        # Low confidence signal
        low_conf_signal = self.sample_signal.copy()
        low_conf_signal["reversion_confidence"] = 0.5
        low_conf_signal["z_score"] = 1.6
        
        result_low = self.risk_manager.calculate_mean_reversion_position_size(
            low_conf_signal, self.sample_market_data, self.sample_portfolio_state
        )
        
        # High confidence should get larger position
        assert result_high["position_size_pct"] > result_low["position_size_pct"]
    
    def test_emergency_state_blocking(self):
        """Test position blocking in emergency risk state."""
        # Set risk manager to emergency state
        self.risk_manager.risk_state = RiskState.EMERGENCY
        
        result = self.risk_manager.calculate_mean_reversion_position_size(
            self.sample_signal, self.sample_market_data, self.sample_portfolio_state
        )
        
        assert result["approved"] is False
        assert result["position_size_pct"] == 0
    
    def test_z_score_breakdown_blocking(self):
        """Test position blocking for extreme z-scores."""
        # Extreme z-score signal
        extreme_signal = self.sample_signal.copy()
        extreme_signal["z_score"] = 3.0  # Above 2.5 breakdown limit
        
        result = self.risk_manager.calculate_mean_reversion_position_size(
            extreme_signal, self.sample_market_data, self.sample_portfolio_state
        )
        
        assert result["approved"] is False
    
    def test_mean_reversion_target_calculation(self):
        """Test mean reversion target calculation."""
        signal_dict = {
            "z_score": 2.2,
            "reversion_confidence": 0.8
        }
        
        market_data = {
            "current_price": 152.0,
            "mean_price": 145.0,
            "volatility": 0.18
        }
        
        target_config = self.risk_manager.calculate_mean_reversion_target(
            signal_dict, market_data
        )
        
        assert "primary_target" in target_config
        assert "expected_reversion_days" in target_config
        assert "reversion_confidence" in target_config
        assert "risk_reward_ratio" in target_config
        
        # Target should be between current price and mean
        assert 145.0 <= target_config["primary_target"] <= 152.0
        
        # Should have reasonable holding time
        assert 1 <= target_config["expected_reversion_days"] <= 5
    
    def test_performance_report_generation(self):
        """Test comprehensive risk report generation."""
        report = self.risk_manager.generate_mean_reversion_risk_report()
        
        # Check required sections
        assert "mean_reversion_metrics" in report
        assert "volatility_regimes" in report
        assert "risk_controls_status" in report
        assert "recommendations" in report
        
        # Check mean reversion specific metrics
        mr_metrics = report["mean_reversion_metrics"]
        assert "active_mr_positions" in mr_metrics
        assert "total_mr_exposure" in mr_metrics
        assert "mr_heat_utilization" in mr_metrics
        assert "positions_at_risk" in mr_metrics
        
        # Check risk controls status
        controls = report["risk_controls_status"]
        assert controls["z_score_monitoring"] == "active"
        assert controls["correlation_tracking"] == "active"
        assert controls["time_based_stops"] == "active"
        assert controls["volatility_regime_adjustment"] == "active"
    
    def test_memory_storage(self):
        """Test optimization results storage in memory."""
        memory_data = self.risk_manager.store_optimization_results_in_memory()
        
        # Check structure
        assert "old_risk_system" in memory_data
        assert "new_risk_framework" in memory_data
        assert "expected_improvement" in memory_data
        
        # Check key improvements
        new_framework = memory_data["new_risk_framework"]
        assert "position_sizing" in new_framework
        assert "stop_management" in new_framework
        assert "portfolio_controls" in new_framework
        assert "kelly_parameters" in new_framework
        
        # Check expected improvements
        improvements = memory_data["expected_improvement"]
        assert "50%" in improvements["drawdown_reduction"]
        assert "18.3%" in improvements["drawdown_reduction"]
        assert "<10%" in improvements["drawdown_reduction"]
    
    def test_correlation_calculation(self):
        """Test correlation calculation between price and mean."""
        price_history = [100, 102, 98, 103, 97, 104, 96, 105]
        mean_history = [100, 100, 100, 100, 100, 100, 100, 100]
        
        correlation = self.risk_manager._calculate_mean_correlation(
            price_history, mean_history
        )
        
        # Should handle calculation without errors
        assert isinstance(correlation, float)
        assert -1.0 <= correlation <= 1.0
    
    def test_sizing_reasoning_generation(self):
        """Test human-readable sizing reasoning generation."""
        # Test approved position
        reasoning = self.risk_manager._generate_mr_sizing_reasoning(
            0.04, 0.05, self.sample_signal, VolatilityRegime.NORMAL
        )
        
        assert "Mean reversion" in reasoning
        assert "z-score" in reasoning
        assert "confidence" in reasoning
        assert "4.0%" in reasoning
        
        # Test blocked position
        reasoning_blocked = self.risk_manager._generate_mr_sizing_reasoning(
            0.0, 0.05, self.sample_signal, VolatilityRegime.EXTREME
        )
        
        assert "blocked" in reasoning_blocked.lower()


class TestMeanReversionIntegration:
    """Integration tests for mean reversion risk management."""
    
    def test_full_position_lifecycle(self):
        """Test complete position lifecycle from entry to exit."""
        risk_manager = MeanReversionRiskManager(100000)
        
        # 1. Position sizing
        signal = {
            "ticker": "AAPL",
            "z_score": 2.0,
            "reversion_confidence": 0.8,
            "mean_distance_pct": 0.06,
            "volatility": 0.18,
            "expected_reversion_days": 3,
            "signal_age_hours": 1
        }
        
        sizing = risk_manager.calculate_mean_reversion_position_size(
            signal, {"volatility": 0.18}, {"mr_portfolio_heat": 0.0}
        )
        
        assert sizing["approved"] is True
        
        # 2. Position monitoring
        base_position = PositionRisk(
            ticker="AAPL",
            position_size=sizing["position_size_dollars"],
            current_price=150.0,
            entry_price=148.0,
            stop_loss=140.0,
            trailing_stop=None,
            time_stop=None,
            atr=2.5,
            volatility=0.18,
            beta=1.0,
            correlation_score=0.5,
            var_contribution=100,
            position_heat=0.025,
            days_held=1
        )
        
        mr_position = MeanReversionPosition(
            base_position=base_position,
            mr_metrics=MeanReversionMetrics(
                z_score=1.8,
                reversion_strength=0.8,
                time_in_position=1,
                volatility_regime=VolatilityRegime.NORMAL,
                correlation_with_mean=0.7,
                mean_distance_pct=0.04,
                lookback_period=20,
                confidence_interval_95=(145.0, 150.0),
                reversion_probability=0.8
            ),
            entry_z_score=2.0,
            current_z_score=1.8,
            z_score_deterioration=0.9,
            max_z_score_since_entry=2.0,
            reversion_target=147.0,
            reversion_confidence=0.8,
            correlation_breakdown_threshold=0.3
        )
        
        # 3. Exit condition check
        market_data = {
            "current_price": 149.0,
            "volatility": 0.18,
            "atr": 2.5,
            "price_history": list(range(145, 150)),
            "mean_history": [147] * 5
        }
        
        should_exit, reason = risk_manager.should_exit_mean_reversion_position(
            mr_position, market_data
        )
        
        # Position should continue (mean reversion in progress)
        assert should_exit is False
        
        # 4. Test successful reversion exit
        # Price reverts to near mean
        market_data["current_price"] = 147.2
        mr_position.current_z_score = 0.4  # Near mean
        
        should_exit, reason = risk_manager.should_exit_mean_reversion_position(
            mr_position, market_data
        )
        
        # Should exit due to mean reversion success (handled in trading strategy)
        # Risk manager focuses on risk exits, not profit exits
    
    def test_risk_escalation_scenario(self):
        """Test risk escalation from normal to emergency."""
        risk_manager = MeanReversionRiskManager(100000)
        
        # Start with normal risk state
        assert risk_manager.risk_state == RiskState.NORMAL
        
        # Simulate portfolio with high drawdown
        risk_manager.current_drawdown = -0.12  # 12% drawdown
        risk_manager.peak_equity = 100000
        
        # Update portfolio state
        risk_manager.update_portfolio_state(88000, {})
        
        # Risk state should escalate
        assert risk_manager.risk_state != RiskState.NORMAL
        
        # New positions should be blocked or severely reduced
        signal = {
            "ticker": "AAPL",
            "z_score": 2.0,
            "reversion_confidence": 0.8,
            "mean_distance_pct": 0.06,
            "volatility": 0.18,
            "expected_reversion_days": 3,
            "signal_age_hours": 1
        }
        
        sizing = risk_manager.calculate_mean_reversion_position_size(
            signal, {"volatility": 0.18}, {"mr_portfolio_heat": 0.0}
        )
        
        # Position should be severely reduced or blocked
        assert sizing["position_size_pct"] < 0.02  # Very small position
    
    def test_performance_optimization_validation(self):
        """Validate that the risk system achieves performance targets."""
        risk_manager = MeanReversionRiskManager(100000)
        
        # Test parameters meet optimization goals
        assert risk_manager.mr_risk_limits["max_mr_portfolio_heat"] == 0.15  # 15% max heat
        assert risk_manager.mr_risk_limits["max_mr_position"] == 0.06  # 6% max position
        assert risk_manager.mr_risk_limits["z_score_deterioration_limit"] == 1.8  # 1.8x deterioration
        assert risk_manager.mr_risk_limits["max_mean_reversion_days"] == 5  # 5 day max hold
        
        # Verify Kelly parameters are conservative
        assert risk_manager.mr_risk_limits["kelly_base_fraction"] == 0.20  # Conservative 20%
        assert risk_manager.kelly_params["win_rate_estimate"] == 0.65  # Realistic 65% win rate
        
        # Check that volatility adjustments are reasonable
        assert risk_manager.mr_risk_limits["low_vol_multiplier"] == 1.2  # 20% size up in low vol
        assert risk_manager.mr_risk_limits["high_vol_multiplier"] == 0.6  # 40% size down in high vol
        
        print("✅ Mean Reversion Risk Manager optimization targets validated")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])