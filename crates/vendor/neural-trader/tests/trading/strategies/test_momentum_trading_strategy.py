"""Test suite for Momentum Trading Strategy implementation."""

import pytest
from datetime import datetime, timedelta
from src.trading.strategies.momentum_trader import MomentumEngine


class TestMomentumTradingStrategy:
    """Test cases for momentum trading strategy."""
    
    def test_momentum_score_calculation(self):
        """Test momentum scoring algorithm."""
        engine = MomentumEngine()
        
        momentum_data = {
            "price_change_5d": 0.08,  # 8% in 5 days
            "price_change_20d": 0.15,  # 15% in 20 days
            "volume_ratio_5d": 1.8,  # 80% above average
            "relative_strength": 82,  # vs S&P 500
            "sector_rank": 2,  # 2nd in sector
            "earnings_revision": "positive",
            "analyst_momentum": 5  # Net upgrades
        }
        
        score = engine.calculate_momentum_score(momentum_data)
        
        assert score > 0.75  # Strong momentum
        assert engine.get_momentum_tier(score) == "strong"
        
        # Test moderate momentum
        moderate_data = {
            "price_change_5d": 0.04,  # 4% in 5 days
            "price_change_20d": 0.08,  # 8% in 20 days
            "volume_ratio_5d": 1.3,  # 30% above average
            "relative_strength": 65,  # vs S&P 500
            "sector_rank": 5,  # 5th in sector
            "earnings_revision": "neutral",
            "analyst_momentum": 1  # Net upgrades
        }
        
        moderate_score = engine.calculate_momentum_score(moderate_data)
        assert 0.5 <= moderate_score <= 0.75
        assert engine.get_momentum_tier(moderate_score) == "moderate"
        
        # Test weak momentum
        weak_data = {
            "price_change_5d": 0.01,  # 1% in 5 days
            "price_change_20d": 0.02,  # 2% in 20 days
            "volume_ratio_5d": 0.9,  # Below average
            "relative_strength": 45,  # vs S&P 500
            "sector_rank": 8,  # 8th in sector
            "earnings_revision": "negative",
            "analyst_momentum": -2  # Net downgrades
        }
        
        weak_score = engine.calculate_momentum_score(weak_data)
        assert weak_score < 0.25
        assert engine.get_momentum_tier(weak_score) == "weak"
        
    def test_earnings_momentum_detection(self):
        """Test earnings-based momentum signals."""
        engine = MomentumEngine()
        
        earnings_data = {
            "ticker": "NVDA",
            "eps_actual": 2.50,
            "eps_estimate": 2.00,
            "revenue_actual": 15000000000,
            "revenue_estimate": 14000000000,
            "guidance": "raised",
            "surprise_history": [0.20, 0.15, 0.25, 0.18]  # Last 4 quarters
        }
        
        signal = engine.analyze_earnings_momentum(earnings_data)
        
        assert signal["momentum_type"] == "earnings_acceleration"
        assert signal["strength"] > 0.8
        assert signal["suggested_holding"] == "4-8 weeks"
        assert signal["eps_surprise"] > 0.20  # 20% beat
        assert signal["revenue_surprise"] > 0.07  # 7% beat
        
        # Test disappointing earnings
        disappointing_data = {
            "ticker": "INTC",
            "eps_actual": 0.80,
            "eps_estimate": 0.95,
            "revenue_actual": 12000000000,
            "revenue_estimate": 13000000000,
            "guidance": "lowered",
            "surprise_history": [-0.10, -0.05, 0.05, -0.15]  # Mixed history
        }
        
        disappointing_signal = engine.analyze_earnings_momentum(disappointing_data)
        
        assert disappointing_signal["momentum_type"] == "earnings_deceleration"
        assert disappointing_signal["strength"] < 0.3
        assert disappointing_signal["suggested_holding"] == "avoid"
        
    def test_sector_rotation_momentum(self):
        """Test sector rotation momentum strategy."""
        engine = MomentumEngine()
        
        sector_data = {
            "XLK": {"performance_1m": 0.05, "volume_surge": 1.5},  # Tech
            "XLF": {"performance_1m": -0.02, "volume_surge": 0.8},  # Financials
            "XLE": {"performance_1m": 0.08, "volume_surge": 2.1},  # Energy
            "XLV": {"performance_1m": 0.03, "volume_surge": 1.1},  # Healthcare
            "XLI": {"performance_1m": 0.01, "volume_surge": 0.9},  # Industrials
            "XLU": {"performance_1m": -0.05, "volume_surge": 0.7},  # Utilities
        }
        
        rotation_signals = engine.identify_sector_rotation(sector_data)
        
        assert rotation_signals["long_sectors"] == ["XLE", "XLK"]
        assert rotation_signals["avoid_sectors"] == ["XLF", "XLU"]
        assert rotation_signals["rotation_strength"] > 0.7
        assert rotation_signals["dominant_sector"] == "XLE"
        
        # Test low rotation environment
        low_rotation_data = {
            "XLK": {"performance_1m": 0.02, "volume_surge": 1.0},
            "XLF": {"performance_1m": 0.01, "volume_surge": 0.95},
            "XLE": {"performance_1m": 0.015, "volume_surge": 1.05},
            "XLV": {"performance_1m": 0.025, "volume_surge": 1.02},
        }
        
        low_rotation = engine.identify_sector_rotation(low_rotation_data)
        assert low_rotation["rotation_strength"] < 0.3
        assert low_rotation["strategy"] == "broad_market"
        
    def test_momentum_position_sizing(self):
        """Test position sizing based on momentum strength."""
        engine = MomentumEngine(portfolio_size=100000)
        
        # Strong momentum trade
        strong_momentum = {
            "momentum_score": 0.85,
            "volatility": 0.25,  # 25% annualized
            "market_cap": "large",
            "sector": "technology",
            "price": 150.00
        }
        
        position = engine.calculate_momentum_position_size(strong_momentum)
        
        assert position["size_pct"] >= 0.05  # At least 5% for strong momentum
        assert position["size_pct"] <= 0.08  # Max 8% per trade
        assert position["risk_level"] == "moderate"
        
        # Weak momentum trade
        weak_momentum = {
            "momentum_score": 0.35,
            "volatility": 0.40,  # High volatility
            "market_cap": "small",
            "sector": "biotech",
            "price": 45.00
        }
        
        weak_position = engine.calculate_momentum_position_size(weak_momentum)
        
        assert weak_position["size_pct"] <= 0.02  # Small position for weak momentum
        assert weak_position["risk_level"] == "high"
        
    def test_momentum_trend_following(self):
        """Test trend following momentum strategy."""
        engine = MomentumEngine()
        
        trend_data = {
            "price_history": [100, 102, 105, 108, 112, 115, 118, 122],  # Strong uptrend
            "volume_history": [1000, 1200, 1500, 1800, 2000, 2200, 2500, 2800],
            "moving_averages": {
                "ma_10": 115,
                "ma_20": 110,
                "ma_50": 105
            },
            "current_price": 122,
            "breakout_level": 120
        }
        
        trend_signal = engine.analyze_trend_momentum(trend_data)
        
        assert trend_signal["trend_strength"] > 0.8
        assert trend_signal["action"] == "buy"
        assert trend_signal["entry_type"] == "breakout_continuation"
        assert trend_signal["stop_loss"] < 115  # Below recent support
        
    def test_momentum_risk_management(self):
        """Test risk management for momentum trades."""
        engine = MomentumEngine()
        
        # Test position with profit
        profitable_position = {
            "entry_price": 100.00,
            "current_price": 110.00,
            "peak_price": 112.00,
            "momentum_score": 0.75,
            "days_held": 12,
            "trailing_stop_pct": 0.05
        }
        
        risk_check = engine.assess_momentum_risk(profitable_position)
        
        assert risk_check["action"] in ["hold", "reduce", "exit"]
        assert risk_check["trailing_stop_price"] > 0
        
        # Test position with loss
        losing_position = {
            "entry_price": 100.00,
            "current_price": 92.00,
            "peak_price": 101.00,
            "momentum_score": 0.25,  # Momentum fading
            "days_held": 8,
            "trailing_stop_pct": 0.05
        }
        
        losing_risk = engine.assess_momentum_risk(losing_position)
        
        assert losing_risk["action"] == "exit"
        assert losing_risk["reason"] == "momentum_failure"
        
    def test_momentum_backtesting_metrics(self):
        """Test momentum strategy backtesting capabilities."""
        engine = MomentumEngine()
        
        # Mock historical data
        backtest_data = {
            "trades": [
                {"entry": 100, "exit": 115, "days": 15, "momentum_score": 0.8},
                {"entry": 200, "exit": 190, "days": 8, "momentum_score": 0.3},
                {"entry": 50, "exit": 58, "days": 22, "momentum_score": 0.7},
                {"entry": 150, "exit": 145, "days": 5, "momentum_score": 0.4},
                {"entry": 75, "exit": 88, "days": 18, "momentum_score": 0.9}
            ]
        }
        
        metrics = engine.calculate_backtest_metrics(backtest_data)
        
        assert metrics["win_rate"] >= 0.60  # At least 60% win rate
        assert metrics["avg_return"] > 0.05  # Average 5%+ return
        assert metrics["momentum_efficiency"] > 0.70  # 70%+ trend capture
        assert metrics["max_drawdown"] < 0.20  # Max 20% drawdown
        
    @pytest.mark.asyncio
    async def test_real_time_momentum_scanning(self):
        """Test real-time momentum scanning."""
        engine = MomentumEngine()
        
        # Mock real-time market data
        market_scan_data = {
            "AAPL": {"momentum_score": 0.85, "volume_surge": 2.1, "breakout": True},
            "MSFT": {"momentum_score": 0.72, "volume_surge": 1.8, "breakout": False},
            "GOOGL": {"momentum_score": 0.45, "volume_surge": 1.2, "breakout": False},
            "TSLA": {"momentum_score": 0.95, "volume_surge": 3.5, "breakout": True},
            "NVDA": {"momentum_score": 0.88, "volume_surge": 2.8, "breakout": True}
        }
        
        scan_results = await engine.scan_momentum_opportunities(market_scan_data)
        
        assert len(scan_results["high_priority"]) >= 2  # At least 2 strong signals
        assert "TSLA" in [s["ticker"] for s in scan_results["high_priority"]]
        assert "NVDA" in [s["ticker"] for s in scan_results["high_priority"]]
        
        # Check filtering worked
        assert "GOOGL" not in [s["ticker"] for s in scan_results["high_priority"]]
        
    def test_momentum_news_integration(self):
        """Test integration with news sentiment for momentum."""
        engine = MomentumEngine()
        
        news_momentum_data = {
            "ticker": "AAPL",
            "price_momentum": 0.75,
            "news_sentiment": 0.85,
            "news_volume": 150,  # Articles in last 24h
            "analyst_upgrades": 3,
            "social_sentiment": 0.70,
            "earnings_revision": "positive"
        }
        
        integrated_score = engine.integrate_news_momentum(news_momentum_data)
        
        assert integrated_score["combined_score"] > 0.80
        assert integrated_score["confidence"] > 0.75
        assert integrated_score["holding_period"] == "2-4 weeks"
        
        # Test negative news impact
        negative_news_data = {
            "ticker": "FB",
            "price_momentum": 0.60,
            "news_sentiment": 0.25,  # Negative news
            "news_volume": 200,
            "analyst_upgrades": -2,  # Downgrades
            "social_sentiment": 0.30,
            "earnings_revision": "negative"
        }
        
        negative_integrated = engine.integrate_news_momentum(negative_news_data)
        
        assert negative_integrated["combined_score"] < 0.40
        assert negative_integrated["action"] == "avoid"
        
    def test_momentum_strategy_optimization(self):
        """Test momentum strategy parameter optimization."""
        engine = MomentumEngine()
        
        optimization_data = {
            "lookback_periods": [5, 10, 20, 60],
            "volume_thresholds": [1.5, 2.0, 2.5],
            "momentum_filters": [0.5, 0.6, 0.7, 0.8],
            "historical_returns": [0.12, 0.08, 0.15, 0.06, 0.18, 0.03, 0.22]
        }
        
        optimal_params = engine.optimize_parameters(optimization_data)
        
        assert optimal_params["best_lookback"] in [5, 10, 20, 60]
        assert optimal_params["best_volume_threshold"] >= 1.5
        assert optimal_params["best_momentum_filter"] >= 0.5
        assert optimal_params["expected_return"] > 0.10
        assert optimal_params["sharpe_ratio"] > 1.5

    def test_60_day_momentum_analysis(self):
        """Test comprehensive 60-day momentum analysis."""
        engine = MomentumEngine()
        
        # Test with 60D data included
        extended_momentum_data = {
            "price_change_5d": 0.08,   # 8% in 5 days
            "price_change_20d": 0.15,  # 15% in 20 days 
            "price_change_60d": 0.25,  # 25% in 60 days
            "volume_ratio_5d": 1.8,
            "relative_strength": 82,
            "sector_rank": 2,
            "earnings_revision": "positive",
            "analyst_momentum": 5
        }
        
        score = engine.calculate_comprehensive_momentum_score(extended_momentum_data)
        
        # Should give bonus for consistent long-term momentum
        assert score > 0.80  # Higher than basic scoring
        
        # Test momentum deceleration (60D strong, but short-term weak)
        deceleration_data = {
            "price_change_5d": 0.02,   # Slowing down
            "price_change_20d": 0.08,  # Moderate
            "price_change_60d": 0.30,  # Was strong
            "volume_ratio_5d": 1.2,
            "relative_strength": 75,
            "sector_rank": 3,
            "earnings_revision": "neutral",
            "analyst_momentum": 0
        }
        
        decel_score = engine.calculate_comprehensive_momentum_score(deceleration_data)
        
        # Should detect momentum weakening
        assert decel_score < 0.65  # Lower due to deceleration

    def test_momentum_exhaustion_detection(self):
        """Test momentum exhaustion detection algorithms."""
        engine = MomentumEngine()
        
        # Test exhaustion scenario: extremely overbought with slowing momentum
        exhaustion_data = {
            "price_change_5d": 0.12,   # 12% in 5 days (extreme)
            "price_change_20d": 0.35,  # 35% in 20 days (very high)
            "price_change_60d": 0.45,  # 45% in 60 days
            "rsi_14": 88,              # Extremely overbought
            "volume_ratio_5d": 0.7,    # Volume declining (key signal)
            "macd_divergence": True,   # Price vs MACD divergence
            "relative_strength": 95,   # Too high
            "volatility_expansion": True,  # High volatility
            "sector_rank": 1
        }
        
        exhaustion_analysis = engine.detect_momentum_exhaustion(exhaustion_data)
        
        assert exhaustion_analysis["exhaustion_risk"] == "high"
        assert exhaustion_analysis["exhaustion_score"] > 0.8
        assert exhaustion_analysis["recommended_action"] == "reduce_position"
        assert exhaustion_analysis["warning_indicators"] >= 3  # Multiple warning signs
        
        # Test healthy momentum (no exhaustion)
        healthy_data = {
            "price_change_5d": 0.06,
            "price_change_20d": 0.18,
            "price_change_60d": 0.22,
            "rsi_14": 65,
            "volume_ratio_5d": 1.5,
            "macd_divergence": False,
            "relative_strength": 78,
            "volatility_expansion": False,
            "sector_rank": 2
        }
        
        healthy_analysis = engine.detect_momentum_exhaustion(healthy_data)
        
        assert healthy_analysis["exhaustion_risk"] == "minimal"
        assert healthy_analysis["exhaustion_score"] < 0.3
        assert healthy_analysis["recommended_action"] == "hold"

    def test_enhanced_relative_strength_analysis(self):
        """Test enhanced relative strength with multiple benchmarks."""
        engine = MomentumEngine()
        
        multi_benchmark_data = {
            "ticker": "AAPL",
            "sector": "technology",
            "stock_return_1m": 0.12,
            "stock_return_3m": 0.25,
            "benchmarks": {
                "SPY": {"return_1m": 0.04, "return_3m": 0.08},      # S&P 500
                "QQQ": {"return_1m": 0.06, "return_3m": 0.12},      # NASDAQ
                "XLK": {"return_1m": 0.08, "return_3m": 0.15},      # Tech sector
                "VTI": {"return_1m": 0.035, "return_3m": 0.075}     # Total market
            },
            "peer_stocks": {
                "MSFT": {"return_1m": 0.09, "return_3m": 0.18},
                "GOOGL": {"return_1m": 0.07, "return_3m": 0.16},
                "META": {"return_1m": 0.11, "return_3m": 0.22}
            }
        }
        
        rs_analysis = engine.calculate_enhanced_relative_strength(multi_benchmark_data)
        
        # Should show strong outperformance across benchmarks
        assert rs_analysis["overall_rs_score"] > 0.75
        assert rs_analysis["market_outperformance"] > 0.07  # 7%+ vs market
        assert rs_analysis["sector_outperformance"] > 0.04  # 4%+ vs sector
        assert rs_analysis["peer_rank"] <= 2  # Top 2 among peers
        assert len(rs_analysis["outperforming_benchmarks"]) >= 3
        
        # Test underperformance scenario
        underperform_data = {
            "ticker": "INTC",
            "sector": "technology", 
            "stock_return_1m": 0.02,
            "stock_return_3m": 0.05,
            "benchmarks": {
                "SPY": {"return_1m": 0.04, "return_3m": 0.08},
                "QQQ": {"return_1m": 0.06, "return_3m": 0.12},
                "XLK": {"return_1m": 0.08, "return_3m": 0.15}
            },
            "peer_stocks": {
                "AMD": {"return_1m": 0.15, "return_3m": 0.30},
                "NVDA": {"return_1m": 0.18, "return_3m": 0.35}
            }
        }
        
        underperform_rs = engine.calculate_enhanced_relative_strength(underperform_data)
        
        assert underperform_rs["overall_rs_score"] < 0.4
        assert underperform_rs["market_outperformance"] < 0  # Underperforming
        assert underperform_rs["peer_rank"] >= 3  # Bottom rank

    def test_sophisticated_acceleration_bonus(self):
        """Test sophisticated acceleration bonus calculations.""" 
        engine = MomentumEngine()
        
        # Test multi-timeframe acceleration
        acceleration_data = {
            "price_change_1d": 0.025,  # 2.5% daily
            "price_change_5d": 0.08,   # 8% weekly
            "price_change_20d": 0.15,  # 15% monthly
            "price_change_60d": 0.22,  # 22% quarterly
            "volume_acceleration": True,  # Volume increasing with price
            "earnings_momentum": "accelerating",
            "analyst_revisions": "increasing"
        }
        
        acceleration_score = engine.calculate_acceleration_bonus(acceleration_data)
        
        # Should get significant bonus for consistent acceleration
        assert acceleration_score["total_bonus"] > 0.15  # 15%+ bonus
        assert acceleration_score["price_acceleration"] > 0.8  # Strong price acceleration
        assert acceleration_score["volume_confirmation"] == True
        assert acceleration_score["fundamental_support"] == True
        
        # Test deceleration penalty
        deceleration_data = {
            "price_change_1d": 0.005,  # Slowing down
            "price_change_5d": 0.04,   
            "price_change_20d": 0.12,  
            "price_change_60d": 0.25,  # Was stronger
            "volume_acceleration": False,
            "earnings_momentum": "stable",
            "analyst_revisions": "stable"
        }
        
        deceleration_score = engine.calculate_acceleration_bonus(deceleration_data)
        
        # Should penalize for deceleration
        assert deceleration_score["total_bonus"] < 0.05  # Minimal bonus
        assert deceleration_score["price_acceleration"] < 0.4  # Weak acceleration

    def test_momentum_tier_precision(self):
        """Test precise momentum tier classifications with enhanced scoring."""
        engine = MomentumEngine()
        
        # Test ultra-strong momentum (new tier)
        ultra_strong_data = {
            "price_change_5d": 0.12,
            "price_change_20d": 0.28,
            "price_change_60d": 0.35,
            "volume_ratio_5d": 2.5,
            "relative_strength": 92,
            "sector_rank": 1,
            "earnings_revision": "positive",
            "analyst_momentum": 8,
            "rsi_14": 75,
            "macd_divergence": False
        }
        
        ultra_score = engine.calculate_comprehensive_momentum_score(ultra_strong_data)
        tier = engine.get_enhanced_momentum_tier(ultra_score)
        
        assert tier == "ultra_strong"
        assert ultra_score > 0.90
        
        # Test strong momentum (existing)
        strong_data = {
            "price_change_5d": 0.08,
            "price_change_20d": 0.18,
            "price_change_60d": 0.22,
            "volume_ratio_5d": 1.8,
            "relative_strength": 82,
            "sector_rank": 2,
            "earnings_revision": "positive",
            "analyst_momentum": 5
        }
        
        strong_score = engine.calculate_comprehensive_momentum_score(strong_data)
        strong_tier = engine.get_enhanced_momentum_tier(strong_score)
        
        assert strong_tier == "strong"
        assert 0.75 <= strong_score <= 0.90