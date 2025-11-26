# Stock & Bond Trading Module - TDD Implementation Plan

## Module Overview
This module implements specialized trading strategies for equities and fixed income securities, with focus on swing trading opportunities in both markets. It includes yield curve analysis, sector rotation strategies, and integration with free market data sources.

## Stock Trading Components

### Phase 1: Equity Market Infrastructure

#### RED: Test Stock Data Collection

```python
# tests/test_stock_trading.py

import pytest
from datetime import datetime, timedelta
from src.trading.stocks.data_collector import StockDataCollector

class TestStockDataCollection:
    
    def test_free_stock_data_sources(self):
        """Test integration with free stock data APIs"""
        collector = StockDataCollector()
        
        # Test Yahoo Finance integration (free)
        data = collector.get_stock_data("AAPL", period="1mo")
        
        assert "open" in data.columns
        assert "high" in data.columns
        assert "low" in data.columns
        assert "close" in data.columns
        assert "volume" in data.columns
        assert len(data) >= 20  # At least 20 trading days
        
    def test_technical_indicators_calculation(self):
        """Test calculation of technical indicators for swing trading"""
        from src.trading.stocks.indicators import TechnicalIndicators
        
        indicators = TechnicalIndicators()
        
        # Mock price data
        prices = [100, 102, 101, 103, 105, 104, 106, 107, 105, 108]
        
        # Test moving averages
        sma_5 = indicators.calculate_sma(prices, period=5)
        assert len(sma_5) == 6  # 10 prices - 4 for SMA calculation
        
        # Test RSI
        rsi = indicators.calculate_rsi(prices, period=5)
        assert 0 <= rsi <= 100
        
        # Test MACD
        macd_line, signal_line = indicators.calculate_macd(prices * 3)  # Need more data
        assert macd_line is not None
        
    def test_support_resistance_detection(self):
        """Test automatic support and resistance level detection"""
        from src.trading.stocks.price_levels import SupportResistanceDetector
        
        detector = SupportResistanceDetector()
        
        # Price data with clear levels
        prices = [100, 98, 99, 98, 102, 103, 102, 98, 99, 98, 104, 105]
        
        levels = detector.detect_levels(prices)
        
        assert 98 in levels["support"]  # Clear support at 98
        assert 102 in levels["resistance"]  # Resistance around 102-103
        assert levels["strength"][98] > 0.7  # Strong support
```

#### GREEN: Implement Stock Data Infrastructure

```python
# src/trading/stocks/data_collector.py

import yfinance as yf
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime, timedelta

class StockDataCollector:
    def __init__(self):
        self.cache = {}
        self.cache_duration = timedelta(minutes=5)
        
    def get_stock_data(self, symbol: str, period: str = "1mo") -> pd.DataFrame:
        """Fetch stock data from Yahoo Finance (free)"""
        cache_key = f"{symbol}_{period}"
        
        # Check cache
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if datetime.now() - timestamp < self.cache_duration:
                return cached_data
                
        # Fetch fresh data
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period)
        
        # Cache the data
        self.cache[cache_key] = (data, datetime.now())
        
        return data
        
    def get_intraday_data(self, symbol: str, interval: str = "5m") -> pd.DataFrame:
        """Fetch intraday data for day trading"""
        ticker = yf.Ticker(symbol)
        
        # Get today's data
        data = ticker.history(period="1d", interval=interval)
        
        return data
        
    def get_sector_data(self) -> Dict[str, float]:
        """Get sector performance data"""
        sectors = {
            "XLK": "Technology",
            "XLF": "Financials", 
            "XLE": "Energy",
            "XLV": "Healthcare",
            "XLI": "Industrials",
            "XLY": "Consumer Discretionary",
            "XLP": "Consumer Staples",
            "XLU": "Utilities",
            "XLRE": "Real Estate"
        }
        
        performance = {}
        for ticker, name in sectors.items():
            data = self.get_stock_data(ticker, period="5d")
            if len(data) >= 2:
                performance[name] = (data['Close'][-1] / data['Close'][0] - 1) * 100
                
        return performance
```

### Phase 2: Stock Swing Trading Strategy

#### RED: Test Swing Trade Setups

```python
# tests/test_stock_swing_trading.py

class TestStockSwingTrading:
    
    def test_bullish_swing_setup(self):
        """Test identification of bullish swing setups"""
        from src.trading.stocks.swing_setups import SwingSetupDetector
        
        detector = SwingSetupDetector()
        
        # Bullish setup scenario
        market_data = {
            "price": 150.00,
            "sma_20": 148.00,
            "sma_50": 145.00,
            "sma_200": 140.00,
            "rsi": 45,  # Oversold bounce
            "volume_avg_ratio": 1.2,
            "recent_low": 147.00,
            "recent_high": 155.00
        }
        
        setup = detector.detect_setup(market_data)
        
        assert setup["type"] == "bullish_reversal"
        assert setup["entry_price"] == pytest.approx(150.50, 0.50)
        assert setup["stop_loss"] == pytest.approx(147.00, 0.50)
        assert setup["target_1"] == pytest.approx(155.00, 1.00)
        
    def test_breakout_swing_setup(self):
        """Test breakout pattern detection"""
        detector = SwingSetupDetector()
        
        # Consolidation breakout scenario
        market_data = {
            "price": 105.00,
            "resistance": 104.50,
            "support": 102.00,
            "consolidation_days": 8,
            "volume_surge": 1.8,
            "atr": 1.50
        }
        
        setup = detector.detect_breakout(market_data)
        
        assert setup["type"] == "resistance_breakout"
        assert setup["confidence"] > 0.7
        assert setup["stop_loss"] == 104.50 - 1.5  # Below breakout level
        
    def test_earnings_gap_swing(self):
        """Test post-earnings gap trading setup"""
        from src.trading.stocks.earnings_trader import EarningsGapTrader
        
        trader = EarningsGapTrader()
        
        earnings_data = {
            "ticker": "MSFT",
            "eps_actual": 2.50,
            "eps_estimate": 2.20,
            "revenue_beat": True,
            "guidance": "raised",
            "gap_percent": 5.5,  # 5.5% gap up
            "pre_earnings_price": 380.00,
            "current_price": 401.00
        }
        
        signal = trader.analyze_earnings_gap(earnings_data)
        
        assert signal["action"] == "buy_pullback"
        assert signal["entry_zone"] == (395.00, 398.00)  # Wait for pullback
        assert signal["holding_period"] == "3-5 days"
```

## Bond Trading Components

### Phase 3: Bond Market Infrastructure

#### RED: Test Bond Data Collection

```python
# tests/test_bond_trading.py

class TestBondDataCollection:
    
    def test_treasury_yield_data(self):
        """Test collection of treasury yield data"""
        from src.trading.bonds.yield_collector import TreasuryYieldCollector
        
        collector = TreasuryYieldCollector()
        
        # Get current yields
        yields = collector.get_current_yields()
        
        assert "1M" in yields
        assert "3M" in yields
        assert "2Y" in yields
        assert "10Y" in yields
        assert "30Y" in yields
        
        # Yields should be reasonable
        assert 0 < yields["10Y"] < 10  # Between 0 and 10%
        
    def test_yield_curve_calculation(self):
        """Test yield curve shape analysis"""
        from src.trading.bonds.yield_curve import YieldCurveAnalyzer
        
        analyzer = YieldCurveAnalyzer()
        
        # Mock yield data
        yields = {
            "3M": 5.50,
            "2Y": 4.80,
            "5Y": 4.50,
            "10Y": 4.60,
            "30Y": 4.80
        }
        
        shape = analyzer.analyze_curve_shape(yields)
        
        assert shape["type"] == "inverted"
        assert shape["2s10s_spread"] == -0.20  # 2Y-10Y spread
        assert shape["recession_probability"] > 0.5  # Inverted = recession risk
        
    def test_bond_etf_analysis(self):
        """Test bond ETF trading analysis"""
        from src.trading.bonds.etf_analyzer import BondETFAnalyzer
        
        analyzer = BondETFAnalyzer()
        
        # Major bond ETFs
        etfs = ["TLT", "IEF", "SHY", "HYG", "LQD"]
        
        analysis = analyzer.analyze_bond_etfs(etfs)
        
        assert "TLT" in analysis  # Long-term treasuries
        assert analysis["TLT"]["duration_risk"] == "high"
        assert "relative_value" in analysis["TLT"]
```

#### GREEN: Implement Bond Infrastructure

```python
# src/trading/bonds/yield_collector.py

import requests
import pandas as pd
from typing import Dict, List
from datetime import datetime
import yfinance as yf

class TreasuryYieldCollector:
    def __init__(self):
        self.treasury_url = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service"
        self.cache = {}
        
    def get_current_yields(self) -> Dict[str, float]:
        """Get current treasury yields across the curve"""
        # Use Treasury Direct API (free)
        yields = {}
        
        # Also can use bond ETF implied yields
        etf_mapping = {
            "SHY": "1-3Y",   # 1-3 year treasuries
            "IEF": "7-10Y",  # 7-10 year treasuries
            "TLT": "20Y+"    # 20+ year treasuries
        }
        
        for etf, maturity in etf_mapping.items():
            ticker = yf.Ticker(etf)
            info = ticker.info
            
            # Calculate implied yield from price and duration
            if "yield" in info:
                yields[maturity] = info["yield"] * 100
                
        # Get specific maturity yields from Fred API or scraping
        yields.update(self._fetch_fred_yields())
        
        return yields
        
    def _fetch_fred_yields(self) -> Dict[str, float]:
        """Fetch yields from FRED (Federal Reserve Economic Data)"""
        # FRED API is free with registration
        series_mapping = {
            "3M": "DGS3MO",
            "2Y": "DGS2", 
            "5Y": "DGS5",
            "10Y": "DGS10",
            "30Y": "DGS30"
        }
        
        yields = {}
        # Implementation would use FRED API
        # Placeholder with realistic values
        yields = {
            "3M": 5.45,
            "2Y": 4.85,
            "5Y": 4.55,
            "10Y": 4.65,
            "30Y": 4.85
        }
        
        return yields

# src/trading/bonds/yield_curve.py

class YieldCurveAnalyzer:
    def analyze_curve_shape(self, yields: Dict[str, float]) -> Dict:
        """Analyze yield curve shape and implications"""
        # Calculate key spreads
        spreads = {
            "2s10s": yields.get("10Y", 0) - yields.get("2Y", 0),
            "2s30s": yields.get("30Y", 0) - yields.get("2Y", 0),
            "5s30s": yields.get("30Y", 0) - yields.get("5Y", 0)
        }
        
        # Determine curve shape
        if spreads["2s10s"] < -0.1:
            curve_type = "inverted"
            recession_prob = 0.7  # Historical probability
        elif spreads["2s10s"] < 0.3:
            curve_type = "flat"
            recession_prob = 0.4
        else:
            curve_type = "normal"
            recession_prob = 0.2
            
        return {
            "type": curve_type,
            "2s10s_spread": spreads["2s10s"],
            "2s30s_spread": spreads["2s30s"],
            "recession_probability": recession_prob,
            "trading_bias": self._get_trading_bias(curve_type)
        }
        
    def _get_trading_bias(self, curve_type: str) -> str:
        """Determine trading bias based on curve shape"""
        if curve_type == "inverted":
            return "long_duration"  # Expect rate cuts
        elif curve_type == "normal":
            return "short_duration"  # Rising rate environment
        else:
            return "neutral"
```

### Phase 4: Bond Swing Trading Strategy

#### RED: Test Bond Trading Setups

```python
# tests/test_bond_swing_trading.py

class TestBondSwingTrading:
    
    def test_duration_trade_setup(self):
        """Test duration-based bond trades"""
        from src.trading.bonds.duration_trader import DurationTrader
        
        trader = DurationTrader()
        
        # Fed pivot scenario
        market_data = {
            "fed_stance": "pivot_to_dovish",
            "10y_yield": 4.75,
            "10y_sma_50": 4.50,
            "inflation_trend": "declining",
            "tlt_price": 92.00,
            "tlt_sma_20": 94.00
        }
        
        signal = trader.analyze_duration_trade(market_data)
        
        assert signal["position"] == "long_tlt"
        assert signal["rationale"] == "Fed pivot supports duration"
        assert signal["stop_yield"] == 4.90  # Stop if yields break higher
        
    def test_yield_curve_trade(self):
        """Test yield curve steepener/flattener trades"""
        from src.trading.bonds.curve_trader import YieldCurveTrader
        
        trader = YieldCurveTrader()
        
        # Curve steepening scenario
        curve_data = {
            "2y_yield": 5.00,
            "10y_yield": 4.50,
            "curve_trend": "steepening",
            "fed_policy": "cutting_short_rates"
        }
        
        trades = trader.generate_curve_trades(curve_data)
        
        assert len(trades) == 2  # Pairs trade
        assert trades[0]["action"] == "long"
        assert trades[0]["instrument"] == "IEF"  # Long intermediate
        assert trades[1]["action"] == "short"
        assert trades[1]["instrument"] == "SHY"  # Short front-end
        
    def test_credit_spread_trade(self):
        """Test corporate bond spread trades"""
        from src.trading.bonds.credit_trader import CreditSpreadTrader
        
        trader = CreditSpreadTrader()
        
        spread_data = {
            "ig_spread": 150,  # Investment grade spread in bps
            "hy_spread": 500,  # High yield spread
            "historical_ig_avg": 100,
            "historical_hy_avg": 400,
            "vix": 25  # Market stress indicator
        }
        
        signal = trader.analyze_credit_opportunity(spread_data)
        
        assert signal["trade"] == "long_credit"
        assert signal["instrument"] == "LQD"  # Investment grade corporate
        assert signal["hedge"] == "TLT"  # Treasury hedge
```

## Integration with Main Platform

### Phase 5: Multi-Asset Coordination

#### RED: Test Stock-Bond Correlation

```python
# tests/test_stock_bond_integration.py

class TestStockBondIntegration:
    
    def test_risk_off_rotation(self):
        """Test risk-off rotation from stocks to bonds"""
        from src.trading.allocation.rotator import AssetRotator
        
        rotator = AssetRotator()
        
        # Risk-off scenario
        market_conditions = {
            "spy_trend": "declining",
            "vix": 35,
            "yield_curve": "inverting",
            "economic_data": "weakening"
        }
        
        allocation = rotator.calculate_allocation(market_conditions)
        
        assert allocation["stocks"] < 0.3  # Reduce stock allocation
        assert allocation["bonds"] > 0.5   # Increase bond allocation
        assert allocation["cash"] > 0.1    # Some cash buffer
        
    def test_balanced_portfolio_signals(self):
        """Test 60/40 portfolio rebalancing signals"""
        from src.trading.allocation.balanced_portfolio import BalancedPortfolioManager
        
        manager = BalancedPortfolioManager(target_stock=0.6, target_bond=0.4)
        
        current_allocation = {
            "stocks": 0.70,  # Overweight after rally
            "bonds": 0.30   # Underweight
        }
        
        rebalance_trades = manager.generate_rebalance_trades(
            current_allocation,
            portfolio_value=100000
        )
        
        assert rebalance_trades[0]["action"] == "sell"
        assert rebalance_trades[0]["asset_class"] == "stocks"
        assert rebalance_trades[0]["amount"] == pytest.approx(10000, 1000)
```

## Performance Monitoring

### Phase 6: Strategy Performance Tracking

```python
# tests/test_performance_tracking.py

class TestPerformanceTracking:
    
    def test_swing_trade_metrics(self):
        """Test swing trading performance metrics"""
        from src.trading.performance.swing_metrics import SwingPerformanceTracker
        
        tracker = SwingPerformanceTracker()
        
        # Add completed trades
        trades = [
            {"symbol": "AAPL", "entry": 170, "exit": 175, "days_held": 5},
            {"symbol": "MSFT", "entry": 380, "exit": 370, "days_held": 7},
            {"symbol": "TLT", "entry": 95, "exit": 98, "days_held": 10}
        ]
        
        for trade in trades:
            tracker.add_trade(trade)
            
        metrics = tracker.calculate_metrics()
        
        assert metrics["win_rate"] == pytest.approx(0.67, 0.01)
        assert metrics["avg_win"] > metrics["avg_loss"]
        assert metrics["profit_factor"] > 1.0
        assert metrics["avg_holding_days"] == pytest.approx(7.33, 0.1)
```

## Implementation Timeline

1. **Week 1**: Stock data infrastructure and technical indicators
2. **Week 2**: Stock swing trading strategies  
3. **Week 3**: Bond market data and yield curve analysis
4. **Week 4**: Bond trading strategies and ETF analysis
5. **Week 5**: Multi-asset integration and rotation strategies
6. **Week 6**: Performance tracking and optimization

## Success Criteria

- [ ] Stock data collection from 3+ free sources
- [ ] Technical indicator accuracy > 99%
- [ ] Swing setup detection rate > 10 per day
- [ ] Bond yield data updates < 5 minute lag
- [ ] Yield curve analysis refreshes hourly
- [ ] Stock-bond correlation tracking in real-time
- [ ] Portfolio rebalancing signals < 1 minute
- [ ] Win rate > 55% for swing trades
- [ ] Risk-adjusted returns > benchmark