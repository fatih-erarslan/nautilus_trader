"""Tests for optimized swing trading strategy."""

import pytest
from datetime import datetime, timedelta
import numpy as np
from src.trading.strategies.swing_trader_optimized import (
    OptimizedSwingTradingEngine, MarketRegime, TradingSignal
)


class TestOptimizedSwingTrader:
    """Test suite for optimized swing trading engine."""
    
    @pytest.fixture
    def engine(self):
        """Create test engine instance."""
        return OptimizedSwingTradingEngine(account_size=100000)
    
    @pytest.fixture
    def trending_market_data(self):
        """Create trending market data."""
        return {
            "price": 100,
            "ma_20": 98,
            "ma_50": 96,
            "ma_200": 92,
            "rsi_14": 55,
            "atr_14": 1.5,
            "volume": 1500000,
            "volume_ma_20": 1000000,
            "volume_ratio": 1.5,
            "macd": 0.8,
            "macd_signal": 0.6,
            "support_level": 95,
            "resistance_level": 105
        }
    
    @pytest.fixture
    def ranging_market_data(self):
        """Create ranging market data."""
        return {
            "price": 98.5,
            "ma_20": 100,
            "ma_50": 100.5,
            "ma_200": 99.8,
            "rsi_14": 32,
            "atr_14": 0.8,
            "volume": 900000,
            "volume_ma_20": 1000000,
            "volume_ratio": 0.9,
            "macd": -0.1,
            "macd_signal": -0.05,
            "support_level": 98,
            "resistance_level": 102
        }
    
    def test_market_regime_detection(self, engine, trending_market_data, ranging_market_data):
        """Test market regime detection accuracy."""
        # Test trending market
        regime = engine.detect_market_regime(trending_market_data)
        assert regime == MarketRegime.TRENDING_UP
        
        # Test ranging market
        regime = engine.detect_market_regime(ranging_market_data)
        assert regime == MarketRegime.RANGING
        
        # Test high volatility
        volatile_data = trending_market_data.copy()
        volatile_data["atr_14"] = 4.0  # 4% ATR
        regime = engine.detect_market_regime(volatile_data)
        assert regime == MarketRegime.HIGH_VOLATILITY
    
    def test_signal_strength_calculation(self, engine, trending_market_data):
        """Test composite signal strength calculation."""
        strength = engine.calculate_signal_strength(trending_market_data)
        
        # Should have strong signal in trending market
        assert 0.6 < strength < 0.9
        
        # Test weak signal conditions
        weak_data = trending_market_data.copy()
        weak_data["rsi_14"] = 85  # Overbought
        weak_data["volume_ratio"] = 0.5  # Low volume
        weak_strength = engine.calculate_signal_strength(weak_data)
        
        assert weak_strength < strength
        assert weak_strength < 0.5
    
    def test_advanced_signal_generation(self, engine, trending_market_data):
        """Test advanced signal generation with confirmations."""
        signal = engine.generate_advanced_signal(trending_market_data)
        
        assert isinstance(signal, TradingSignal)
        assert signal.action == "buy"
        assert signal.setup_type == "trend_continuation"
        assert "strong_uptrend" in signal.confirmations
        assert "volume_confirmation" in signal.confirmations
        
        # Check risk management
        assert signal.risk_reward_ratio >= 2.0
        assert len(signal.take_profit_levels) == 3
        assert signal.position_size_pct <= engine.max_position_pct
    
    def test_volatility_based_position_sizing(self, engine):
        """Test volatility-adjusted position sizing."""
        # Low volatility market
        low_vol_data = {
            "price": 100,
            "atr_14": 0.5,  # 0.5% ATR
            "rsi_14": 45,
            "ma_20": 99,
            "ma_50": 98,
            "ma_200": 96,
            "volume_ratio": 1.2
        }
        
        signal_low_vol = engine.generate_advanced_signal(low_vol_data)
        
        # High volatility market
        high_vol_data = low_vol_data.copy()
        high_vol_data["atr_14"] = 3.0  # 3% ATR
        
        signal_high_vol = engine.generate_advanced_signal(high_vol_data)
        
        # Position size should be smaller in high volatility
        if signal_low_vol.action == "buy" and signal_high_vol.action == "buy":
            assert signal_high_vol.position_size_pct < signal_low_vol.position_size_pct
    
    def test_portfolio_heat_calculation(self, engine):
        """Test portfolio heat (risk) calculation."""
        open_positions = [
            {
                "entry_price": 100,
                "stop_loss": 98,
                "position_value": 10000
            },
            {
                "entry_price": 50,
                "stop_loss": 48.5,
                "position_value": 15000
            }
        ]
        
        heat = engine.calculate_portfolio_heat(open_positions)
        
        # Calculate expected heat
        # Position 1: 2% risk on 10k = $200
        # Position 2: 3% risk on 15k = $450
        # Total risk: $650 on 100k = 0.65%
        expected_heat = 0.0065
        
        assert abs(heat - expected_heat) < 0.0001
    
    def test_portfolio_risk_overlay(self, engine):
        """Test portfolio-level risk management."""
        signal = TradingSignal(
            action="buy",
            strength=0.8,
            entry_price=100,
            stop_loss=98,
            take_profit_levels=[102, 104, 106],
            position_size_pct=0.2,
            risk_reward_ratio=3.0,
            confidence=0.85,
            setup_type="trend_continuation",
            confirmations=["strong_signal"],
            time_to_target=5
        )
        
        # Test with high portfolio heat
        open_positions = [
            {"entry_price": 100, "stop_loss": 95, "position_value": 30000},
            {"entry_price": 200, "stop_loss": 190, "position_value": 25000}
        ]
        
        adjusted_signal = engine.apply_portfolio_risk_overlay(signal, open_positions)
        
        # Position should be reduced due to high portfolio heat
        assert adjusted_signal.position_size_pct < signal.position_size_pct
        assert "position_reduced_portfolio_heat" in adjusted_signal.confirmations
    
    def test_dynamic_exit_management(self, engine):
        """Test advanced exit management with partial profits."""
        position = {
            "entry_price": 100,
            "initial_stop_loss": 98,
            "current_stop_loss": 98,
            "take_profit_levels": [102.5, 104, 106],
            "shares": 100,
            "entry_date": datetime.now() - timedelta(days=3),
            "expected_holding_days": 5
        }
        
        # Test partial profit taking
        market_data = {
            "current_price": 102.6,
            "atr_14": 1.5
        }
        
        exit_decision = engine.dynamic_exit_management(position, market_data)
        
        assert exit_decision["partial_exit"] is True
        assert exit_decision["shares_to_exit"] == 50  # 50% exit
        assert exit_decision["reason"] == "partial_profit_level_1"
        assert exit_decision["new_stop_loss"] > position["entry_price"]  # Moved to breakeven
        
        # Test trailing stop activation
        position["current_stop_loss"] = 100.15  # Breakeven stop
        position["exits_taken"] = 1
        market_data["current_price"] = 105
        
        exit_decision = engine.dynamic_exit_management(position, market_data)
        
        assert exit_decision["new_stop_loss"] > position["current_stop_loss"]
        assert exit_decision["exit"] is False  # Still holding remaining position
    
    def test_time_based_exit(self, engine):
        """Test time-based exit conditions."""
        position = {
            "entry_price": 100,
            "initial_stop_loss": 98,
            "current_stop_loss": 98,
            "take_profit_levels": [103, 105, 107],
            "shares": 100,
            "entry_date": datetime.now() - timedelta(days=12),  # Held too long
            "expected_holding_days": 5
        }
        
        market_data = {
            "current_price": 100.5,  # Minimal profit
            "atr_14": 1.0
        }
        
        exit_decision = engine.dynamic_exit_management(position, market_data)
        
        assert exit_decision["exit"] is True
        assert exit_decision["reason"] == "time_based_exit"
    
    def test_backtest_metrics_calculation(self, engine):
        """Test comprehensive backtest metrics."""
        # Simulate trades with improved performance
        trades = [
            {"profit_pct": 0.025},   # 2.5%
            {"profit_pct": 0.018},   # 1.8%
            {"profit_pct": -0.008},  # -0.8%
            {"profit_pct": 0.032},   # 3.2%
            {"profit_pct": 0.041},   # 4.1%
            {"profit_pct": -0.005},  # -0.5%
            {"profit_pct": 0.022},   # 2.2%
            {"profit_pct": 0.028},   # 2.8%
            {"profit_pct": -0.012},  # -1.2%
            {"profit_pct": 0.035},   # 3.5%
        ]
        
        metrics = engine.backtest_metrics(trades)
        
        # Verify improved metrics
        assert metrics["sharpe_ratio"] > 2.0  # Target 2.0+ Sharpe
        assert metrics["win_rate"] > 65  # 70% win rate
        assert metrics["profit_factor"] > 2.5  # Strong profit factor
        assert metrics["total_return"] > 15  # 15%+ return
        
        # Verify risk metrics
        assert metrics["max_drawdown"] < -5  # Controlled drawdown
        assert metrics["avg_win"] > abs(metrics["avg_loss"]) * 2  # Good RR ratio
    
    def test_regime_specific_strategies(self, engine):
        """Test different strategies for different market regimes."""
        # Trending market strategy
        trending_data = {
            "price": 105,
            "ma_20": 103,
            "ma_50": 101,
            "ma_200": 98,
            "rsi_14": 58,
            "atr_14": 1.8,
            "volume_ratio": 1.3,
            "support_level": 102,
            "resistance_level": 108
        }
        
        trend_signal = engine.generate_advanced_signal(trending_data)
        
        # Ranging market strategy
        ranging_data = {
            "price": 98.2,  # Near support
            "ma_20": 100,
            "ma_50": 100.5,
            "ma_200": 99.8,
            "rsi_14": 28,  # Oversold
            "atr_14": 0.8,
            "volume_ratio": 1.1,
            "support_level": 98,
            "resistance_level": 102
        }
        
        range_signal = engine.generate_advanced_signal(ranging_data)
        
        # Different setups for different regimes
        assert trend_signal.setup_type == "trend_continuation"
        assert range_signal.setup_type == "support_bounce"
        
        # Different time targets
        assert trend_signal.time_to_target > range_signal.time_to_target
    
    def test_multiple_take_profit_levels(self, engine, trending_market_data):
        """Test multiple take profit level generation."""
        signal = engine.generate_advanced_signal(trending_market_data)
        
        if signal.action == "buy":
            assert len(signal.take_profit_levels) == 3
            
            # Verify progressive targets
            assert signal.take_profit_levels[0] < signal.take_profit_levels[1]
            assert signal.take_profit_levels[1] < signal.take_profit_levels[2]
            
            # Verify reasonable R multiples
            risk = signal.entry_price - signal.stop_loss
            r1 = (signal.take_profit_levels[0] - signal.entry_price) / risk
            r2 = (signal.take_profit_levels[1] - signal.entry_price) / risk
            r3 = (signal.take_profit_levels[2] - signal.entry_price) / risk
            
            assert 1.0 < r1 < 2.0
            assert 1.5 < r2 < 3.0
            assert 2.5 < r3 < 4.0