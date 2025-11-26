"""Test suite for Swing Trading Strategy implementation."""

import pytest
from datetime import datetime, timedelta
from src.trading.strategies.swing_trader import SwingTradingEngine


class TestSwingTradingStrategy:
    """Test cases for swing trading strategy."""
    
    def test_swing_setup_detection(self):
        """Test detection of valid swing trading setups."""
        engine = SwingTradingEngine()
        
        # Valid swing setup: price above 50 & 200 MA, RSI not overbought
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
        
        setup = engine.identify_swing_setup(market_data)
        assert setup["valid"] == True
        assert setup["setup_type"] == "bullish_continuation"
        assert setup["entry_zone"] == (173.00, 174.00)
        assert setup["confidence"] == 0.75
        
        # Test oversold bounce setup
        oversold_data = {
            "ticker": "MSFT",
            "price": 380.00,
            "ma_50": 395.00,
            "ma_200": 390.00,
            "rsi_14": 28,
            "volume_ratio": 1.8,
            "atr_14": 5.50,
            "support_level": 378.00,
            "resistance_level": 400.00
        }
        
        oversold_setup = engine.identify_swing_setup(oversold_data)
        assert oversold_setup["valid"] == True
        assert oversold_setup["setup_type"] == "oversold_bounce"
        assert oversold_setup["confidence"] == 0.65
        
        # Test invalid setup
        invalid_data = {
            "ticker": "GOOGL",
            "price": 140.00,
            "ma_50": 142.00,
            "ma_200": 145.00,
            "rsi_14": 35,  # Not oversold enough
            "volume_ratio": 0.8,
            "atr_14": 2.0,
            "support_level": 135.00,
            "resistance_level": 148.00
        }
        
        invalid_setup = engine.identify_swing_setup(invalid_data)
        assert invalid_setup["valid"] == False
        
    def test_swing_position_sizing(self):
        """Test position sizing based on volatility."""
        engine = SwingTradingEngine(account_size=100000, max_risk_per_trade=0.02)
        
        trade_setup = {
            "entry_price": 50.00,
            "stop_loss": 48.00,
            "atr": 1.50
        }
        
        position = engine.calculate_position_size(trade_setup)
        
        # Risk $2000 per trade (2% of $100k)
        # Stop distance is $2, so 1000 shares
        assert position["shares"] == 1000
        assert position["position_value"] == 50000
        assert position["risk_amount"] == 2000
        assert position["position_pct"] == 0.50  # 50% of account
        
        # Test with tighter stop
        tight_stop_setup = {
            "entry_price": 100.00,
            "stop_loss": 99.00,
            "atr": 2.00
        }
        
        tight_position = engine.calculate_position_size(tight_stop_setup)
        
        # Risk $2000 with $1 stop = 2000 shares
        # But position would be $200k (200% of account), so should be capped
        assert tight_position["shares"] == 500  # Capped at 50% of account
        assert tight_position["position_value"] == 50000
        assert tight_position["risk_amount"] == 500  # Actual risk with position limit
        assert tight_position["position_pct"] == 0.50
        
    def test_swing_exit_rules(self):
        """Test swing trading exit conditions."""
        engine = SwingTradingEngine()
        
        position = {
            "entry_price": 100.00,
            "entry_date": datetime.now() - timedelta(days=5),
            "stop_loss": 97.00,
            "take_profit": 106.00,
            "trailing_stop_pct": 0.03
        }
        
        # Test profit target hit
        market_data = {"current_price": 106.50}
        exit_signal = engine.check_exit_conditions(position, market_data)
        assert exit_signal["exit"] == True
        assert exit_signal["reason"] == "profit_target_hit"
        assert exit_signal["exit_price"] == 106.50
        
        # Test trailing stop
        market_data = {
            "current_price": 104.00,
            "highest_price_since_entry": 105.00
        }
        exit_signal = engine.check_exit_conditions(position, market_data)
        assert exit_signal["exit"] == True
        assert exit_signal["reason"] == "trailing_stop_hit"
        # Trailing stop at 105 * 0.97 = 101.85, current price 104 is above it
        
        # Test with price below trailing stop
        market_data = {
            "current_price": 101.50,
            "highest_price_since_entry": 105.00
        }
        exit_signal = engine.check_exit_conditions(position, market_data)
        assert exit_signal["exit"] == True
        assert exit_signal["reason"] == "trailing_stop_hit"
        
        # Test time-based exit (holding too long)
        old_position = position.copy()
        old_position["entry_date"] = datetime.now() - timedelta(days=12)
        market_data = {"current_price": 102.00}
        
        exit_signal = engine.check_exit_conditions(old_position, market_data)
        assert exit_signal["exit"] == True
        assert exit_signal["reason"] == "max_holding_period"
        
        # Test no exit condition met
        market_data = {
            "current_price": 102.00,
            "highest_price_since_entry": 103.00
        }
        exit_signal = engine.check_exit_conditions(position, market_data)
        assert exit_signal["exit"] == False
        
    @pytest.mark.asyncio
    async def test_bond_swing_trading(self):
        """Test swing trading for bonds."""
        engine = SwingTradingEngine()
        
        bond_data = {
            "ticker": "TLT",  # 20+ Year Treasury ETF
            "yield_current": 4.25,
            "yield_ma_50": 4.10,
            "price": 95.50,
            "fed_policy": "pause",
            "inflation_trend": "declining"
        }
        
        signal = await engine.generate_bond_swing_signal(bond_data)
        
        assert signal["action"] == "buy"
        assert signal["reasoning"] == "Yields above MA suggesting oversold bonds"
        assert signal["holding_period"] == "5-15 days"
        assert signal["stop_loss_yield"] == 4.40  # Stop on yield breakout
        assert signal["take_profit_price"] == 97.14  # ~1.7% gain
        
        # Test bearish bond setup
        bearish_bond_data = {
            "ticker": "TLT",
            "yield_current": 3.85,
            "yield_ma_50": 4.00,
            "price": 98.50,
            "fed_policy": "tightening",
            "inflation_trend": "rising"
        }
        
        bearish_signal = await engine.generate_bond_swing_signal(bearish_bond_data)
        
        assert bearish_signal["action"] == "sell" or bearish_signal["action"] == "avoid"
        assert "yields below ma" in bearish_signal["reasoning"].lower()
        
    def test_swing_risk_reward_calculation(self):
        """Test risk/reward ratio calculation for swing trades."""
        engine = SwingTradingEngine()
        
        trade_setup = {
            "entry_price": 50.00,
            "stop_loss": 48.00,
            "resistance_level": 54.00,
            "atr": 1.50
        }
        
        risk_reward = engine.calculate_risk_reward(trade_setup)
        
        assert risk_reward["risk"] == 2.00  # Entry - Stop
        assert risk_reward["reward"] == 4.00  # Resistance - Entry  
        assert risk_reward["ratio"] == 2.00  # Reward / Risk
        assert risk_reward["acceptable"] == True  # Above minimum 1.5
        
        # Test unacceptable risk/reward
        bad_setup = {
            "entry_price": 50.00,
            "stop_loss": 48.00,
            "resistance_level": 52.00,
            "atr": 1.50
        }
        
        bad_risk_reward = engine.calculate_risk_reward(bad_setup)
        
        assert bad_risk_reward["ratio"] == 1.00
        assert bad_risk_reward["acceptable"] == False
        
    def test_multiple_timeframe_analysis(self):
        """Test multiple timeframe confirmation for swing trades."""
        engine = SwingTradingEngine()
        
        timeframes = {
            "daily": {
                "trend": "bullish",
                "rsi": 55,
                "ma_alignment": "bullish"  # Price > MA50 > MA200
            },
            "4hour": {
                "trend": "bullish", 
                "rsi": 48,
                "ma_alignment": "bullish"
            },
            "1hour": {
                "trend": "neutral",
                "rsi": 45,
                "ma_alignment": "mixed"
            }
        }
        
        confirmation = engine.analyze_multiple_timeframes(timeframes)
        
        assert confirmation["score"] >= 0.6  # Good alignment
        assert confirmation["trade_bias"] == "bullish"
        assert confirmation["entry_timeframe"] == "1hour"  # Enter on smallest TF
        
        # Test conflicting timeframes
        conflicting_tf = {
            "daily": {
                "trend": "bullish",
                "rsi": 70,
                "ma_alignment": "bullish"
            },
            "4hour": {
                "trend": "bearish",
                "rsi": 35,
                "ma_alignment": "bearish"
            },
            "1hour": {
                "trend": "bearish",
                "rsi": 30,
                "ma_alignment": "bearish"
            }
        }
        
        conflict_confirmation = engine.analyze_multiple_timeframes(conflicting_tf)
        
        assert conflict_confirmation["score"] <= 0.45  # Poor alignment
        assert conflict_confirmation["trade_bias"] == "avoid"
        
    def test_sector_etf_swing_trading(self):
        """Test swing trading for sector ETFs."""
        engine = SwingTradingEngine()
        
        sector_data = {
            "ticker": "XLK",  # Technology sector
            "price": 175.00,
            "ma_20": 172.00,
            "ma_50": 170.00,
            "rsi_14": 58,
            "sector_rank": 2,  # 2nd strongest sector
            "relative_strength_vs_spy": 1.15,  # 15% outperformance
            "volume_ratio": 1.4
        }
        
        sector_signal = engine.analyze_sector_etf_swing(sector_data)
        
        assert sector_signal["valid"] == True
        assert sector_signal["position_size_multiplier"] == 1.2  # Stronger position for top sectors
        assert "strong sector" in sector_signal["reasoning"].lower()
        assert sector_signal["expected_holding_days"] <= 10