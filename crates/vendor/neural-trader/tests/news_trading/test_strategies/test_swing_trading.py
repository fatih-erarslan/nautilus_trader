"""Tests for Swing Trading Strategy following TDD."""

import pytest
from datetime import datetime, timedelta
from src.news_trading.strategies.swing_trading import SwingTradingStrategy
from src.news_trading.decision_engine.models import (
    TradingSignal, SignalType, RiskLevel, TradingStrategy, AssetType
)


class TestSwingTradingStrategy:
    """Test suite for Swing Trading Strategy."""
    
    def test_swing_strategy_initialization(self):
        """Test swing trading strategy initialization."""
        strategy = SwingTradingStrategy(
            account_size=100000,
            max_risk_per_trade=0.02,
            min_risk_reward=1.5
        )
        
        assert strategy.account_size == 100000
        assert strategy.max_risk_per_trade == 0.02
        assert strategy.min_risk_reward == 1.5
        assert strategy.max_position_pct == 0.10  # Safety limit
        
    def test_swing_setup_detection_bullish(self):
        """Test detection of bullish swing trading setups."""
        strategy = SwingTradingStrategy()
        
        # Valid bullish swing setup
        market_data = {
            "ticker": "AAPL",
            "price": 175.50,
            "ma_50": 172.00,
            "ma_200": 168.00,
            "rsi_14": 55,
            "volume_ratio": 1.3,
            "atr_14": 2.50,
            "support_level": 172.00,
            "resistance_level": 178.00
        }
        
        setup = strategy.identify_swing_setup(market_data)
        
        assert setup["valid"] == True
        assert setup["setup_type"] == "bullish_continuation"
        assert setup["entry_zone"][0] < setup["entry_zone"][1]
        assert setup["confidence"] >= 0.7
        
    def test_swing_setup_detection_oversold_bounce(self):
        """Test detection of oversold bounce setup."""
        strategy = SwingTradingStrategy()
        
        market_data = {
            "ticker": "MSFT",
            "price": 380.00,
            "ma_50": 390.00,
            "ma_200": 375.00,
            "rsi_14": 28,  # Oversold
            "volume_ratio": 1.5,
            "atr_14": 5.00,
            "support_level": 378.00,
            "resistance_level": 395.00
        }
        
        setup = strategy.identify_swing_setup(market_data)
        
        assert setup["valid"] == True
        assert setup["setup_type"] == "oversold_bounce"
        assert setup["confidence"] >= 0.6
        
    def test_invalid_swing_setup(self):
        """Test rejection of invalid swing setups."""
        strategy = SwingTradingStrategy()
        
        # Overbought with no clear trend
        market_data = {
            "ticker": "GOOGL",
            "price": 140.00,
            "ma_50": 138.00,
            "ma_200": 139.00,
            "rsi_14": 78,  # Overbought
            "volume_ratio": 0.8,  # Low volume
            "atr_14": 2.00
        }
        
        setup = strategy.identify_swing_setup(market_data)
        
        assert setup["valid"] == False
        
    def test_position_sizing_with_risk_management(self):
        """Test position sizing based on risk management rules."""
        strategy = SwingTradingStrategy(account_size=100000, max_risk_per_trade=0.02)
        
        trade_setup = {
            "entry_price": 50.00,
            "stop_loss": 48.00,
            "atr": 1.50
        }
        
        position = strategy.calculate_position_size(trade_setup)
        
        # Risk $2000 per trade (2% of $100k)
        # Stop distance is $2, so would be 1000 shares
        # BUT limited by max position size (10% = $10k)
        assert position["shares"] == 200  # Limited to $10k / $50 = 200 shares
        assert position["position_value"] == 10000  # 10% of account
        assert position["risk_amount"] == 400  # 200 shares * $2 risk
        assert position["position_pct"] == 0.10  # Max position size
        
    def test_position_sizing_with_max_limit(self):
        """Test position sizing respects maximum position limits."""
        strategy = SwingTradingStrategy(account_size=100000)
        
        # Very tight stop that would create huge position
        trade_setup = {
            "entry_price": 100.00,
            "stop_loss": 99.50,  # Only $0.50 stop
            "atr": 2.00
        }
        
        position = strategy.calculate_position_size(trade_setup)
        
        # Should be limited by max position size (10% of account)
        assert position["position_value"] <= 10000  # 10% of $100k
        assert position["shares"] <= 100  # Max 100 shares at $100
        
    @pytest.mark.asyncio
    async def test_generate_swing_signal(self):
        """Test complete swing trading signal generation."""
        strategy = SwingTradingStrategy()
        
        market_data = {
            "ticker": "AAPL",
            "price": 175.50,
            "ma_50": 172.00,
            "ma_200": 168.00,
            "rsi_14": 55,
            "volume_ratio": 1.3,
            "atr_14": 2.50,
            "support_level": 172.00,
            "resistance_level": 178.00
        }
        
        news_catalyst = {
            "headline": "Apple announces record iPhone sales",
            "sentiment": 0.8,
            "relevance": "high",
            "source_id": "news-001"
        }
        
        signal = await strategy.generate_signal(market_data, news_catalyst)
        
        assert signal is not None
        assert signal.asset == "AAPL"
        assert signal.strategy == TradingStrategy.SWING
        assert signal.signal_type == SignalType.BUY
        assert signal.holding_period == "3-7 days"
        assert signal.stop_loss < signal.entry_price
        assert signal.take_profit > signal.entry_price
        assert signal.risk_reward_ratio >= 1.5
        
    def test_swing_exit_conditions(self):
        """Test swing trading exit condition checks."""
        strategy = SwingTradingStrategy()
        
        position = {
            "entry_price": 100.00,
            "entry_date": datetime.now() - timedelta(days=5),
            "stop_loss": 97.00,
            "take_profit": 106.00,
            "trailing_stop_pct": 0.02
        }
        
        # Test profit target hit
        market_data = {"current_price": 106.50}
        exit_signal = strategy.check_exit_conditions(position, market_data)
        assert exit_signal["exit"] == True
        assert exit_signal["reason"] == "profit_target_hit"
        
        # Test stop loss hit
        market_data = {"current_price": 96.50}
        exit_signal = strategy.check_exit_conditions(position, market_data)
        assert exit_signal["exit"] == True
        assert exit_signal["reason"] == "stop_loss_hit"
        
        # Test time-based exit (max holding period)
        old_position = position.copy()
        old_position["entry_date"] = datetime.now() - timedelta(days=11)
        market_data = {"current_price": 102.00}
        exit_signal = strategy.check_exit_conditions(old_position, market_data)
        assert exit_signal["exit"] == True
        assert exit_signal["reason"] == "max_holding_period"
        
    @pytest.mark.asyncio
    async def test_bond_swing_trading(self):
        """Test swing trading for bonds."""
        strategy = SwingTradingStrategy()
        
        bond_data = {
            "ticker": "TLT",  # 20+ Year Treasury ETF
            "yield_current": 4.25,
            "yield_ma_50": 4.10,
            "price": 95.50,
            "atr": 0.75,
            "fed_policy": "pause",
            "inflation_trend": "declining",
            "volume_ratio": 1.2
        }
        
        signal = await strategy.generate_bond_swing_signal(bond_data)
        
        assert signal is not None
        assert signal.asset == "TLT"
        assert signal.asset_type == AssetType.BOND
        assert signal.signal_type == SignalType.BUY  # Yields above MA = oversold bonds
        assert "5-15 days" in signal.holding_period
        assert signal.reasoning is not None