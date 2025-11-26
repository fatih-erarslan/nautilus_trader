"""Tests for Momentum Trading Strategy following TDD."""

import pytest
from datetime import datetime, timedelta
from src.news_trading.strategies.momentum_trading import MomentumTradingStrategy
from src.news_trading.decision_engine.models import (
    TradingSignal, SignalType, RiskLevel, TradingStrategy, AssetType
)


class TestMomentumTradingStrategy:
    """Test suite for Momentum Trading Strategy."""
    
    def test_momentum_strategy_initialization(self):
        """Test momentum trading strategy initialization."""
        strategy = MomentumTradingStrategy(
            lookback_periods=[5, 20, 60],
            momentum_threshold=0.50
        )
        
        assert strategy.lookback_periods == [5, 20, 60]
        assert strategy.momentum_threshold == 0.50
        assert strategy.momentum_thresholds["strong"] == 0.75
        assert strategy.momentum_thresholds["moderate"] == 0.50
        assert strategy.momentum_thresholds["weak"] == 0.25
        
    def test_momentum_score_calculation(self):
        """Test momentum scoring algorithm."""
        strategy = MomentumTradingStrategy()
        
        momentum_data = {
            "price_change_5d": 0.08,  # 8% in 5 days
            "price_change_20d": 0.15,  # 15% in 20 days
            "volume_ratio_5d": 1.8,  # 80% above average
            "relative_strength": 82,  # vs S&P 500
            "sector_rank": 2,  # 2nd in sector
            "earnings_revision": "positive",
            "analyst_momentum": 5  # Net upgrades
        }
        
        score = strategy.calculate_momentum_score(momentum_data)
        
        assert 0 <= score <= 1
        assert score > 0.75  # Strong momentum
        assert strategy.get_momentum_tier(score) == "strong"
        
    def test_momentum_tier_classification(self):
        """Test momentum tier classification."""
        strategy = MomentumTradingStrategy()
        
        assert strategy.get_momentum_tier(0.80) == "strong"
        assert strategy.get_momentum_tier(0.60) == "moderate"
        assert strategy.get_momentum_tier(0.30) == "weak"
        assert strategy.get_momentum_tier(0.10) == "weak"
        
    def test_price_momentum_scoring(self):
        """Test price momentum scoring with acceleration."""
        strategy = MomentumTradingStrategy()
        
        # Strong momentum with acceleration
        score1 = strategy._score_price_momentum(0.05, 0.10)  # 5d accelerating
        # Weak momentum
        score2 = strategy._score_price_momentum(0.01, 0.02)
        
        assert score1 > score2
        assert score1 == 0.48  # 0.4 * 1.2 acceleration
        assert score2 == 0  # Below threshold
        
        # Test very strong momentum
        score3 = strategy._score_price_momentum(0.15, 0.25)  # 15% in 5d, 25% in 20d
        assert score3 == 1.0  # Capped at 1.0
        
    @pytest.mark.asyncio
    async def test_earnings_momentum_detection(self):
        """Test earnings-based momentum signals."""
        strategy = MomentumTradingStrategy()
        
        earnings_data = {
            "ticker": "NVDA",
            "eps_actual": 2.50,
            "eps_estimate": 2.00,
            "revenue_actual": 15000000000,
            "revenue_estimate": 14000000000,
            "guidance": "raised",
            "surprise_history": [0.20, 0.15, 0.25, 0.18]  # Last 4 quarters
        }
        
        signal = await strategy.analyze_earnings_momentum(earnings_data)
        
        assert signal is not None
        assert signal["momentum_type"] == "earnings_acceleration"
        assert signal["strength"] > 0.8
        assert signal["suggested_holding"] == "4-8 weeks"
        assert signal["confidence"] > 0.75
        
    @pytest.mark.asyncio
    async def test_sector_rotation_momentum(self):
        """Test sector rotation momentum strategy."""
        strategy = MomentumTradingStrategy()
        
        sector_data = {
            "XLK": {"performance_1m": 0.05, "volume_surge": 1.5},  # Tech
            "XLF": {"performance_1m": -0.02, "volume_surge": 0.8},  # Financials
            "XLE": {"performance_1m": 0.08, "volume_surge": 2.1},  # Energy
            "XLV": {"performance_1m": 0.03, "volume_surge": 1.1},  # Healthcare
        }
        
        rotation_signals = await strategy.identify_sector_rotation(sector_data)
        
        assert rotation_signals["long_sectors"] == ["XLE", "XLK"]
        assert rotation_signals["avoid_sectors"] == ["XLF"]
        assert rotation_signals["rotation_strength"] > 0.7
        assert "energy_leading" in rotation_signals["theme"]
        
    @pytest.mark.asyncio
    async def test_generate_momentum_signal(self):
        """Test complete momentum trading signal generation."""
        strategy = MomentumTradingStrategy()
        
        market_data = {
            "ticker": "TSLA",
            "price": 250.00,
            "price_change_5d": 0.12,
            "price_change_20d": 0.25,
            "volume_ratio_5d": 2.5,
            "relative_strength": 88,
            "atr": 8.00,
            "sector_rank": 1,
            "analyst_momentum": 7
        }
        
        signal = await strategy.generate_signal(market_data)
        
        assert signal is not None
        assert signal.asset == "TSLA"
        assert signal.strategy == TradingStrategy.MOMENTUM
        assert signal.signal_type == SignalType.BUY
        assert signal.momentum_score > 0.8
        assert signal.stop_loss < signal.entry_price
        assert signal.position_size > 0.05  # Larger for strong momentum
        assert "trend following" in signal.reasoning.lower()
        
    def test_momentum_position_sizing(self):
        """Test position sizing based on momentum strength."""
        strategy = MomentumTradingStrategy()
        
        # Strong momentum = larger position
        strong_position = strategy.calculate_position_size(
            momentum_score=0.85,
            volatility=0.02,
            account_size=100000
        )
        
        # Weak momentum = smaller position
        weak_position = strategy.calculate_position_size(
            momentum_score=0.40,
            volatility=0.02,
            account_size=100000
        )
        
        assert strong_position["position_pct"] > weak_position["position_pct"]
        assert strong_position["position_pct"] <= 0.08  # Max 8% for momentum
        assert weak_position["position_pct"] <= 0.03  # Max 3% for weak
        
    def test_momentum_exit_conditions(self):
        """Test momentum exit conditions."""
        strategy = MomentumTradingStrategy()
        
        position = {
            "entry_price": 100.00,
            "entry_momentum": 0.85,
            "stop_loss": 95.00,
            "entry_date": datetime.now() - timedelta(days=10)
        }
        
        # Test momentum exhaustion
        market_data = {
            "current_price": 110.00,
            "current_momentum": 0.30,  # Momentum died
            "volume_ratio": 0.6  # Volume dried up
        }
        
        exit_signal = strategy.check_exit_conditions(position, market_data)
        assert exit_signal["exit"] == True
        assert exit_signal["reason"] == "momentum_exhaustion"
        
        # Test trend reversal
        market_data = {
            "current_price": 98.00,
            "current_momentum": 0.20,
            "price_change_5d": -0.08  # Negative momentum
        }
        
        exit_signal = strategy.check_exit_conditions(position, market_data)
        assert exit_signal["exit"] == True
        assert exit_signal["reason"] == "trend_reversal"
        
    @pytest.mark.asyncio
    async def test_crypto_momentum_trading(self):
        """Test momentum trading for cryptocurrencies."""
        strategy = MomentumTradingStrategy()
        
        crypto_data = {
            "ticker": "BTC",
            "price": 45000,
            "price_change_5d": 0.15,
            "price_change_20d": 0.30,
            "volume_ratio_5d": 3.0,  # High volume surge
            "dominance_change": 0.02,  # BTC dominance increasing
            "funding_rate": 0.01,  # Positive but not extreme
            "atr": 1500
        }
        
        signal = await strategy.generate_crypto_momentum_signal(crypto_data)
        
        assert signal is not None
        assert signal.asset_type == AssetType.CRYPTO
        assert signal.signal_type == SignalType.BUY
        assert signal.risk_level == RiskLevel.HIGH  # Crypto is high risk
        assert signal.position_size <= 0.05  # Conservative for crypto