# Trading Strategies Module - TDD Implementation Plan

## Module Overview
This module implements three core trading strategies for the AI News Trading Platform: Swing Trading, Momentum Trading, and Mirror Trading. Each strategy is designed for specific market conditions and asset classes (stocks, bonds, ETFs) with comprehensive risk management and position sizing.

## Strategy Specifications

### 1. Swing Trading Strategy
- **Holding Period**: 3-10 days
- **Target Assets**: Large-cap stocks, sector ETFs, Treasury bonds
- **Risk/Reward**: Minimum 1.5:1, typically 2:1
- **Position Size**: 2-5% per trade
- **Key Indicators**: Moving averages, RSI, volume patterns

### 2. Momentum Trading Strategy
- **Holding Period**: 1-30 days (trend following)
- **Target Assets**: Growth stocks, sector rotations, commodity ETFs
- **Entry**: Breakouts, earnings beats, analyst upgrades
- **Exit**: Trend exhaustion, momentum divergence
- **Position Size**: 3-8% for strong signals

### 3. Mirror Trading Strategy
- **Holding Period**: Matches institutional investor (1-24 months)
- **Target Assets**: Following 13F filings, insider buys
- **Institutions**: Berkshire, Bridgewater, Soros, Tiger Global
- **Position Size**: 1-3% (scaled to institution's commitment)
- **Entry Timing**: Within 48 hours of filing disclosure

## Test-First Implementation

### Phase 1: Swing Trading Engine

#### RED: Write Swing Trading Tests

```python
# tests/test_swing_trading_strategy.py

import pytest
from datetime import datetime, timedelta
from src.trading.strategies.swing_trader import SwingTradingEngine

class TestSwingTradingStrategy:
    
    def test_swing_setup_detection(self):
        """Test detection of valid swing trading setups"""
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
        
    def test_swing_position_sizing(self):
        """Test position sizing based on volatility"""
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
        
    def test_swing_exit_rules(self):
        """Test swing trading exit conditions"""
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
        
        # Test trailing stop
        market_data = {
            "current_price": 104.00,
            "highest_price_since_entry": 105.00
        }
        exit_signal = engine.check_exit_conditions(position, market_data)
        assert exit_signal["exit"] == True
        assert exit_signal["reason"] == "trailing_stop_hit"
        
    @pytest.mark.asyncio
    async def test_bond_swing_trading(self):
        """Test swing trading for bonds"""
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
```

#### GREEN: Implement Swing Trading

```python
# src/trading/strategies/swing_trader.py

from typing import Dict, List, Optional
from datetime import datetime, timedelta
import numpy as np

class SwingTradingEngine:
    def __init__(self, account_size: float = 100000, max_risk_per_trade: float = 0.02):
        self.account_size = account_size
        self.max_risk_per_trade = max_risk_per_trade
        self.min_risk_reward = 1.5
        
    def identify_swing_setup(self, market_data: Dict) -> Dict:
        """Identify valid swing trading setups"""
        price = market_data["price"]
        ma_50 = market_data["ma_50"]
        ma_200 = market_data["ma_200"]
        rsi = market_data["rsi_14"]
        
        # Bullish continuation setup
        if price > ma_50 > ma_200 and 40 < rsi < 70:
            return {
                "valid": True,
                "setup_type": "bullish_continuation",
                "entry_zone": (ma_50 * 1.005, ma_50 * 1.015),
                "confidence": 0.75
            }
            
        # Oversold bounce setup
        if price < ma_50 and rsi < 30:
            support = market_data.get("support_level", price * 0.98)
            return {
                "valid": True,
                "setup_type": "oversold_bounce",
                "entry_zone": (support * 0.995, support * 1.005),
                "confidence": 0.65
            }
            
        return {"valid": False}
        
    def calculate_position_size(self, trade_setup: Dict) -> Dict:
        """Calculate position size based on risk management"""
        entry_price = trade_setup["entry_price"]
        stop_loss = trade_setup["stop_loss"]
        risk_per_share = entry_price - stop_loss
        
        # Maximum risk amount
        max_risk_amount = self.account_size * self.max_risk_per_trade
        
        # Calculate shares based on risk
        shares = int(max_risk_amount / risk_per_share)
        position_value = shares * entry_price
        
        # Check position size limits (max 50% of account)
        if position_value > self.account_size * 0.5:
            shares = int((self.account_size * 0.5) / entry_price)
            position_value = shares * entry_price
            
        return {
            "shares": shares,
            "position_value": position_value,
            "risk_amount": shares * risk_per_share,
            "position_pct": position_value / self.account_size
        }
```

### Phase 2: Momentum Trading Engine

#### RED: Test Momentum Strategy

```python
# tests/test_momentum_trading_strategy.py

class TestMomentumTradingStrategy:
    
    def test_momentum_score_calculation(self):
        """Test momentum scoring algorithm"""
        from src.trading.strategies.momentum_trader import MomentumEngine
        
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
        
    def test_earnings_momentum_detection(self):
        """Test earnings-based momentum signals"""
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
        
    def test_sector_rotation_momentum(self):
        """Test sector rotation momentum strategy"""
        engine = MomentumEngine()
        
        sector_data = {
            "XLK": {"performance_1m": 0.05, "volume_surge": 1.5},  # Tech
            "XLF": {"performance_1m": -0.02, "volume_surge": 0.8},  # Financials
            "XLE": {"performance_1m": 0.08, "volume_surge": 2.1},  # Energy
            "XLV": {"performance_1m": 0.03, "volume_surge": 1.1},  # Healthcare
        }
        
        rotation_signals = engine.identify_sector_rotation(sector_data)
        
        assert rotation_signals["long_sectors"] == ["XLE", "XLK"]
        assert rotation_signals["avoid_sectors"] == ["XLF"]
        assert rotation_signals["rotation_strength"] > 0.7
```

#### GREEN: Implement Momentum Trading

```python
# src/trading/strategies/momentum_trader.py

class MomentumEngine:
    def __init__(self, lookback_periods: List[int] = [5, 20, 60]):
        self.lookback_periods = lookback_periods
        self.momentum_thresholds = {
            "strong": 0.75,
            "moderate": 0.50,
            "weak": 0.25
        }
        
    def calculate_momentum_score(self, data: Dict) -> float:
        """Calculate composite momentum score"""
        scores = []
        
        # Price momentum (40% weight)
        price_score = self._score_price_momentum(
            data["price_change_5d"],
            data["price_change_20d"]
        )
        scores.append(price_score * 0.4)
        
        # Volume momentum (20% weight)
        volume_score = min(data["volume_ratio_5d"] / 2, 1.0)
        scores.append(volume_score * 0.2)
        
        # Relative strength (25% weight)
        rs_score = data["relative_strength"] / 100
        scores.append(rs_score * 0.25)
        
        # Fundamental momentum (15% weight)
        fundamental_score = self._score_fundamentals(data)
        scores.append(fundamental_score * 0.15)
        
        return sum(scores)
        
    def _score_price_momentum(self, change_5d: float, change_20d: float) -> float:
        """Score price momentum with acceleration"""
        # Acceleration bonus if 5d > 20d annualized
        acceleration = 1.2 if (change_5d * 4) > change_20d else 1.0
        
        # Base score on 20d performance
        if change_20d > 0.20:
            return 1.0 * acceleration
        elif change_20d > 0.10:
            return 0.7 * acceleration
        elif change_20d > 0.05:
            return 0.4 * acceleration
        else:
            return 0.0
```

### Phase 3: Mirror Trading Engine

#### RED: Test Mirror Trading

```python
# tests/test_mirror_trading_strategy.py

class TestMirrorTradingStrategy:
    
    def test_institutional_filing_parser(self):
        """Test parsing of 13F and Form 4 filings"""
        from src.trading.strategies.mirror_trader import MirrorTradingEngine
        
        engine = MirrorTradingEngine()
        
        # Mock 13F filing data
        filing_13f = {
            "filer": "Berkshire Hathaway",
            "quarter": "2024Q1",
            "holdings": [
                {"ticker": "AAPL", "shares": 900000000, "value": 150000000000},
                {"ticker": "BAC", "shares": 1000000000, "value": 35000000000},
                {"ticker": "CVX", "shares": 120000000, "value": 18000000000}
            ],
            "new_positions": ["OXY", "C"],
            "increased_positions": ["CVX"],
            "reduced_positions": ["AAPL"],
            "sold_positions": ["TSM"]
        }
        
        signals = engine.parse_13f_filing(filing_13f)
        
        assert len(signals) > 0
        assert any(s["ticker"] == "OXY" and s["action"] == "buy" for s in signals)
        assert any(s["ticker"] == "TSM" and s["action"] == "sell" for s in signals)
        
    def test_confidence_scoring_by_institution(self):
        """Test confidence scoring based on institution track record"""
        engine = MirrorTradingEngine()
        
        # Test high-confidence institutions
        berkshire_score = engine.get_institution_confidence("Berkshire Hathaway")
        assert berkshire_score > 0.90
        
        # Test insider transactions
        insider_filing = {
            "filer": "Tim Cook",
            "company": "AAPL",
            "role": "CEO",
            "transaction_type": "Purchase",
            "shares": 50000,
            "price": 175.00
        }
        
        insider_score = engine.score_insider_transaction(insider_filing)
        assert insider_score > 0.85  # CEO buying is high confidence
        
    def test_mirror_position_sizing(self):
        """Test position sizing based on institutional commitment"""
        engine = MirrorTradingEngine(portfolio_size=100000)
        
        # Buffett makes a big bet
        institutional_trade = {
            "institution": "Berkshire Hathaway",
            "ticker": "OXY",
            "action": "buy",
            "position_size_pct": 0.15,  # 15% of their portfolio
            "dollar_value": 15000000000
        }
        
        our_position = engine.calculate_mirror_position(institutional_trade)
        
        # We should take a proportional but smaller position
        assert our_position["size_pct"] <= 0.03  # Max 3% for safety
        assert our_position["reasoning"] == "Scaling down institutional position for risk management"
        
    @pytest.mark.asyncio
    async def test_timing_mirror_trades(self):
        """Test entry timing for mirror trades"""
        engine = MirrorTradingEngine()
        
        filing_data = {
            "filing_date": datetime.now() - timedelta(hours=6),
            "ticker": "MSFT",
            "current_price": 400.00,
            "filing_price": 395.00,  # Price when institution bought
            "volume_since_filing": 15000000
        }
        
        timing = await engine.determine_entry_timing(filing_data)
        
        assert timing["entry_strategy"] == "immediate"  # Still close to filing price
        assert timing["max_chase_price"] == 401.75  # Don't chase more than 1.5%
```

#### GREEN: Implement Mirror Trading

```python
# src/trading/strategies/mirror_trader.py

from typing import Dict, List, Optional
from datetime import datetime, timedelta
import pandas as pd

class MirrorTradingEngine:
    def __init__(self, portfolio_size: float = 100000):
        self.portfolio_size = portfolio_size
        self.trusted_institutions = {
            "Berkshire Hathaway": 0.95,
            "Bridgewater Associates": 0.85,
            "Renaissance Technologies": 0.90,
            "Soros Fund Management": 0.80,
            "Tiger Global": 0.75,
            "Third Point": 0.70,
            "Pershing Square": 0.75,
            "Appaloosa Management": 0.80
        }
        
    def parse_13f_filing(self, filing: Dict) -> List[Dict]:
        """Parse 13F filing and generate mirror signals"""
        signals = []
        institution = filing["filer"]
        confidence = self.trusted_institutions.get(institution, 0.5)
        
        # High priority: New positions by trusted institutions
        for ticker in filing.get("new_positions", []):
            signals.append({
                "ticker": ticker,
                "action": "buy",
                "confidence": confidence,
                "priority": "high",
                "reasoning": f"{institution} initiated new position",
                "mirror_type": "new_position"
            })
            
        # Medium priority: Increased positions
        for ticker in filing.get("increased_positions", []):
            signals.append({
                "ticker": ticker,
                "action": "buy",
                "confidence": confidence * 0.8,
                "priority": "medium",
                "reasoning": f"{institution} increased position",
                "mirror_type": "add_position"
            })
            
        # Sell signals from eliminated positions
        for ticker in filing.get("sold_positions", []):
            signals.append({
                "ticker": ticker,
                "action": "sell",
                "confidence": confidence * 0.9,
                "priority": "high",
                "reasoning": f"{institution} eliminated position",
                "mirror_type": "exit_position"
            })
            
        return signals
        
    def calculate_mirror_position(self, institutional_trade: Dict) -> Dict:
        """Calculate our position size based on institutional commitment"""
        inst_position_pct = institutional_trade["position_size_pct"]
        confidence = self.trusted_institutions.get(
            institutional_trade["institution"], 0.5
        )
        
        # Scale down position size
        # Max 20% of institutional position size, further scaled by confidence
        our_position_pct = min(
            inst_position_pct * 0.2 * confidence,
            0.03  # Never more than 3% per position
        )
        
        # Dollar amount
        position_dollars = self.portfolio_size * our_position_pct
        
        return {
            "size_pct": our_position_pct,
            "size_dollars": position_dollars,
            "reasoning": "Scaling down institutional position for risk management",
            "confidence": confidence,
            "expected_holding_period": self._estimate_holding_period(
                institutional_trade["institution"]
            )
        }
        
    def _estimate_holding_period(self, institution: str) -> str:
        """Estimate holding period based on institution's style"""
        long_term_investors = ["Berkshire Hathaway", "Pershing Square"]
        medium_term = ["Tiger Global", "Third Point"]
        
        if institution in long_term_investors:
            return "6-24 months"
        elif institution in medium_term:
            return "3-12 months"
        else:
            return "1-6 months"
```

## Integration Tests

### Cross-Strategy Integration

```python
# tests/test_strategy_integration.py

@pytest.mark.integration
class TestStrategyIntegration:
    
    @pytest.mark.asyncio
    async def test_multi_strategy_portfolio(self):
        """Test running multiple strategies in parallel"""
        from src.trading.strategies.portfolio_manager import MultiStrategyPortfolio
        
        portfolio = MultiStrategyPortfolio(
            capital=100000,
            strategy_allocations={
                "swing": 0.40,     # 40% to swing trading
                "momentum": 0.35,  # 35% to momentum
                "mirror": 0.25     # 25% to mirror trading
            }
        )
        
        # Mock market conditions and signals
        market_conditions = {
            "vix": 18.5,
            "market_trend": "bullish",
            "sector_rotation": "technology",
            "recent_filings": ["BRK", "RENTECH"]
        }
        
        signals = await portfolio.generate_all_signals(market_conditions)
        
        # Should have signals from all strategies
        assert any(s["strategy"] == "swing" for s in signals)
        assert any(s["strategy"] == "momentum" for s in signals)
        assert any(s["strategy"] == "mirror" for s in signals)
        
        # Risk limits should be enforced
        total_exposure = sum(s["position_size"] for s in signals)
        assert total_exposure <= 1.0  # Max 100% invested
        
    def test_strategy_conflict_resolution(self):
        """Test handling conflicting signals from different strategies"""
        from src.trading.strategies.signal_arbiter import SignalArbiter
        
        arbiter = SignalArbiter()
        
        conflicting_signals = [
            {
                "ticker": "AAPL",
                "strategy": "swing",
                "action": "buy",
                "confidence": 0.7,
                "timeframe": "5 days"
            },
            {
                "ticker": "AAPL", 
                "strategy": "momentum",
                "action": "sell",  # Conflict!
                "confidence": 0.6,
                "timeframe": "2 days"
            }
        ]
        
        resolved = arbiter.resolve_conflicts(conflicting_signals)
        
        # Higher confidence should win
        assert resolved["AAPL"]["action"] == "buy"
        assert resolved["AAPL"]["strategy"] == "swing"
```

## Performance Benchmarks

### Strategy Performance Targets

```python
# tests/test_strategy_performance.py

class TestStrategyPerformance:
    
    def test_swing_trading_metrics(self):
        """Test swing trading performance metrics"""
        from src.trading.strategies.performance import StrategyAnalyzer
        
        analyzer = StrategyAnalyzer()
        
        # Historical trades
        swing_trades = load_test_trades("swing_trades.json")
        
        metrics = analyzer.calculate_metrics(swing_trades)
        
        assert metrics["win_rate"] >= 0.55  # 55% minimum win rate
        assert metrics["avg_risk_reward"] >= 1.5  # Minimum R:R ratio
        assert metrics["max_drawdown"] <= 0.15  # Max 15% drawdown
        assert metrics["sharpe_ratio"] >= 1.5  # Risk-adjusted returns
        
    def test_momentum_capture_efficiency(self):
        """Test how well momentum strategy captures trends"""
        analyzer = StrategyAnalyzer()
        
        # Compare to buy-and-hold during trends
        momentum_returns = analyzer.get_strategy_returns("momentum", "2023-01-01", "2024-01-01")
        market_returns = analyzer.get_market_returns("SPY", "2023-01-01", "2024-01-01")
        
        # Should outperform in trending markets
        trend_periods = analyzer.identify_trend_periods(market_returns)
        
        for period in trend_periods:
            momentum_period_return = momentum_returns[period].sum()
            market_period_return = market_returns[period].sum()
            
            # Momentum should capture at least 70% of trend
            assert momentum_period_return >= market_period_return * 0.7
            
    def test_mirror_trading_alpha(self):
        """Test mirror trading performance vs institutions"""
        analyzer = StrategyAnalyzer()
        
        # Compare our returns to institutional returns
        mirror_trades = load_test_trades("mirror_trades.json")
        
        for trade in mirror_trades:
            our_return = trade["our_return"]
            inst_return = trade["institutional_return"]
            
            # Should achieve at least 80% of institutional returns
            assert our_return >= inst_return * 0.8
```

## Risk Management Integration

```python
# tests/test_strategy_risk_management.py

class TestStrategyRiskManagement:
    
    def test_portfolio_heat_map(self):
        """Test portfolio risk heat map across strategies"""
        from src.trading.risk.portfolio_risk import RiskManager
        
        risk_manager = RiskManager()
        
        positions = [
            {"ticker": "AAPL", "size": 0.05, "strategy": "swing", "correlation_to_spy": 0.8},
            {"ticker": "XLE", "size": 0.08, "strategy": "momentum", "correlation_to_spy": 0.6},
            {"ticker": "BRK.B", "size": 0.03, "strategy": "mirror", "correlation_to_spy": 0.7}
        ]
        
        heat_map = risk_manager.calculate_risk_heat_map(positions)
        
        assert heat_map["total_market_exposure"] <= 0.8  # Max 80% correlated to market
        assert heat_map["strategy_concentration"]["momentum"] <= 0.4  # Max 40% in one strategy
        assert heat_map["risk_score"] <= 7  # Max risk score of 7/10
```

## Implementation Timeline

1. **Week 1**: Core strategy engines (swing, momentum, mirror)
2. **Week 2**: Integration with news and market data feeds  
3. **Week 3**: Risk management and position sizing
4. **Week 4**: Performance tracking and optimization
5. **Week 5**: Multi-strategy portfolio management
6. **Week 6**: Production deployment and monitoring

## Success Criteria

- [ ] All strategy tests pass with >95% coverage
- [ ] Swing trading achieves 55%+ win rate with 1.5:1 R:R
- [ ] Momentum strategy captures 70%+ of trending moves
- [ ] Mirror trades achieve 80%+ of institutional returns
- [ ] Portfolio risk metrics stay within defined limits
- [ ] Strategies handle 100+ signals per minute
- [ ] Conflict resolution works for opposing signals
- [ ] Real-money paper trading shows positive alpha