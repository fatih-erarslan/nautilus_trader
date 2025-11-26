# Trading Decision Engine - TDD Implementation Plan

## Module Overview
The Trading Decision Engine converts sentiment analysis and market data into actionable trading signals for stocks, bonds, and other securities. It implements swing trading, momentum trading, and mirror trading strategies while managing risk, determining position sizing, and coordinating trade execution. The engine supports multi-asset portfolios including equities, fixed income, and crypto assets.

## Test-First Implementation Sequence

### Phase 1: Core Decision Engine Interface (Red-Green-Refactor)

#### RED: Write failing tests first

```python
# tests/test_trading_decision_engine.py

def test_decision_engine_interface():
    """Test that TradingDecisionEngine abstract interface is properly defined"""
    from src.trading.decision_engine import TradingDecisionEngine
    
    class TestEngine(TradingDecisionEngine):
        pass
    
    # Should fail - abstract methods not implemented
    with pytest.raises(TypeError):
        engine = TestEngine()

def test_trading_signal_model():
    """Test TradingSignal data model"""
    from src.trading.decision_engine.models import TradingSignal, SignalType, RiskLevel, TradingStrategy
    
    signal = TradingSignal(
        id="signal-123",
        timestamp=datetime.now(),
        asset="AAPL",
        asset_type="equity",
        signal_type=SignalType.BUY,
        strategy=TradingStrategy.SWING,
        strength=0.85,  # 0 to 1
        confidence=0.75,
        risk_level=RiskLevel.MEDIUM,
        position_size=0.05,  # 5% of portfolio
        entry_price=175.50,
        stop_loss=171.00,  # 2.5% stop for swing trade
        take_profit=182.00,  # 3.7% target
        holding_period="3-7 days",
        source_events=["news-001", "news-002"],
        reasoning="Technical breakout above 200-day MA with strong volume"
    )
    
    assert signal.signal_type == SignalType.BUY
    assert signal.strategy == TradingStrategy.SWING
    assert signal.risk_level == RiskLevel.MEDIUM
    assert signal.position_size == 0.05
```

#### GREEN: Implement minimal code to pass

```python
# src/trading/decision_engine/models.py
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict
from enum import Enum

class SignalType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    CLOSE = "CLOSE"
    SHORT = "SHORT"
    COVER = "COVER"

class RiskLevel(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    EXTREME = "EXTREME"

class TradingStrategy(Enum):
    SWING = "SWING"  # 3-10 day holds
    MOMENTUM = "MOMENTUM"  # Follow trend strength
    MIRROR = "MIRROR"  # Copy institutional trades
    DAY_TRADE = "DAY_TRADE"  # Intraday only
    POSITION = "POSITION"  # Long-term hold

class AssetType(Enum):
    EQUITY = "EQUITY"
    BOND = "BOND"
    CRYPTO = "CRYPTO"
    COMMODITY = "COMMODITY"
    FOREX = "FOREX"

@dataclass
class TradingSignal:
    id: str
    timestamp: datetime
    asset: str
    asset_type: AssetType
    signal_type: SignalType
    strategy: TradingStrategy
    strength: float  # 0 to 1
    confidence: float  # 0 to 1
    risk_level: RiskLevel
    position_size: float  # Fraction of portfolio
    entry_price: float
    stop_loss: float
    take_profit: float
    holding_period: str  # Expected holding time
    source_events: List[str]
    reasoning: str
    technical_indicators: Dict = None
    mirror_source: str = None  # For mirror trades
    momentum_score: float = None  # For momentum trades
    metadata: Dict = None

# src/trading/decision_engine/base.py
from abc import ABC, abstractmethod
from typing import List, Dict
from .models import TradingSignal

class TradingDecisionEngine(ABC):
    @abstractmethod
    async def process_sentiment(self, sentiment_data: Dict) -> TradingSignal:
        """Process sentiment data into trading signal"""
        pass
    
    @abstractmethod
    async def evaluate_portfolio(self, current_positions: Dict) -> List[TradingSignal]:
        """Evaluate current portfolio and generate rebalancing signals"""
        pass
```

### Phase 2: News-to-Signal Conversion

#### RED: Test signal generation from news sentiment

```python
@pytest.mark.asyncio
async def test_sentiment_to_signal_conversion():
    """Test converting sentiment to trading signal"""
    from src.trading.decision_engine.news_signal_generator import NewsSignalGenerator
    
    generator = NewsSignalGenerator()
    
    sentiment_data = {
        "asset": "BTC",
        "sentiment_score": 0.8,
        "confidence": 0.85,
        "market_impact": {
            "direction": "bullish",
            "magnitude": 0.7,
            "timeframe": "short-term"
        }
    }
    
    signal = await generator.generate_signal(sentiment_data)
    
    assert signal.signal_type == SignalType.BUY
    assert signal.strength > 0.7
    assert signal.position_size > 0

@pytest.mark.asyncio
async def test_bearish_signal_generation():
    """Test bearish signal generation"""
    generator = NewsSignalGenerator()
    
    sentiment_data = {
        "asset": "ETH",
        "sentiment_score": -0.7,
        "confidence": 0.8,
        "market_impact": {
            "direction": "bearish",
            "magnitude": 0.6
        }
    }
    
    signal = await generator.generate_signal(sentiment_data)
    
    assert signal.signal_type == SignalType.SELL
    assert signal.risk_level in [RiskLevel.MEDIUM, RiskLevel.HIGH]
```

#### GREEN: Implement signal generator

```python
# src/trading/decision_engine/news_signal_generator.py
class NewsSignalGenerator:
    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        
    async def generate_signal(self, sentiment_data: Dict) -> TradingSignal:
        sentiment_score = sentiment_data["sentiment_score"]
        confidence = sentiment_data["confidence"]
        
        # Determine signal type based on sentiment
        signal_type = self._determine_signal_type(sentiment_score)
        
        # Calculate signal strength
        strength = abs(sentiment_score) * confidence
        
        # Determine position size based on confidence and risk
        position_size = self._calculate_position_size(strength, confidence)
        
        # Get current market data for pricing
        market_data = await self._fetch_market_data(sentiment_data["asset"])
        
        # Calculate entry, stop loss, and take profit
        entry_price = market_data["current_price"]
        stop_loss, take_profit = self._calculate_risk_levels(
            entry_price, signal_type, sentiment_data
        )
        
        return TradingSignal(
            id=self._generate_signal_id(),
            timestamp=datetime.now(),
            asset=sentiment_data["asset"],
            signal_type=signal_type,
            strength=strength,
            confidence=confidence,
            risk_level=self._assess_risk_level(sentiment_data),
            position_size=position_size,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            source_events=sentiment_data.get("source_events", []),
            reasoning=self._generate_reasoning(sentiment_data)
        )
    
    def _determine_signal_type(self, sentiment_score: float) -> SignalType:
        if sentiment_score > 0.3:
            return SignalType.BUY
        elif sentiment_score < -0.3:
            return SignalType.SELL
        else:
            return SignalType.HOLD
```

### Phase 3: Risk Management System

#### RED: Test risk management

```python
def test_risk_manager_initialization():
    """Test RiskManager initialization"""
    from src.trading.decision_engine.risk_manager import RiskManager
    
    risk_manager = RiskManager(
        max_position_size=0.1,  # 10% max per position
        max_portfolio_risk=0.2,  # 20% max portfolio risk
        max_correlation=0.7     # Max correlation between positions
    )
    
    assert risk_manager.max_position_size == 0.1
    assert risk_manager.max_portfolio_risk == 0.2

@pytest.mark.asyncio
async def test_position_size_validation():
    """Test position size validation against risk limits"""
    from src.trading.decision_engine.risk_manager import RiskManager
    
    risk_manager = RiskManager(max_position_size=0.1)
    
    signal = TradingSignal(
        position_size=0.15,  # Exceeds max
        risk_level=RiskLevel.HIGH,
        # ... other fields
    )
    
    validated_signal = await risk_manager.validate_signal(signal)
    assert validated_signal.position_size <= 0.1

def test_portfolio_risk_calculation():
    """Test portfolio risk calculation"""
    risk_manager = RiskManager()
    
    positions = {
        "BTC": {"size": 0.3, "risk": 0.05},
        "ETH": {"size": 0.2, "risk": 0.04},
        "ADA": {"size": 0.1, "risk": 0.03}
    }
    
    total_risk = risk_manager.calculate_portfolio_risk(positions)
    assert total_risk > 0
    assert total_risk < 1
```

#### GREEN: Implement risk manager

```python
# src/trading/decision_engine/risk_manager.py
class RiskManager:
    def __init__(self, max_position_size=0.1, max_portfolio_risk=0.2, max_correlation=0.7):
        self.max_position_size = max_position_size
        self.max_portfolio_risk = max_portfolio_risk
        self.max_correlation = max_correlation
        
    async def validate_signal(self, signal: TradingSignal, 
                            current_positions: Dict = None) -> TradingSignal:
        # Validate position size
        signal.position_size = min(signal.position_size, self.max_position_size)
        
        # Check portfolio risk if positions provided
        if current_positions:
            portfolio_risk = self.calculate_portfolio_risk(current_positions)
            
            # Adjust position size if portfolio risk too high
            if portfolio_risk + self._signal_risk(signal) > self.max_portfolio_risk:
                signal.position_size *= (self.max_portfolio_risk - portfolio_risk) / self._signal_risk(signal)
                
        # Validate stop loss
        signal = self._validate_stop_loss(signal)
        
        return signal
    
    def calculate_portfolio_risk(self, positions: Dict) -> float:
        total_risk = 0
        for asset, position in positions.items():
            position_risk = position["size"] * position.get("risk", 0.02)
            total_risk += position_risk
            
        return total_risk
```

### Phase 4: Multi-Asset Correlation Analysis

#### RED: Test correlation analysis

```python
def test_correlation_analyzer():
    """Test correlation analysis between assets"""
    from src.trading.decision_engine.correlation_analyzer import CorrelationAnalyzer
    
    analyzer = CorrelationAnalyzer()
    
    # Historical price data
    price_data = {
        "BTC": [45000, 46000, 45500, 47000],
        "ETH": [3000, 3100, 3050, 3200],
        "ADA": [1.2, 1.1, 1.15, 1.0]
    }
    
    correlations = analyzer.calculate_correlations(price_data)
    
    assert "BTC-ETH" in correlations
    assert correlations["BTC-ETH"] > 0  # Typically positive
    assert -1 <= correlations["BTC-ETH"] <= 1

def test_correlation_impact_on_signals():
    """Test how correlations affect signal generation"""
    analyzer = CorrelationAnalyzer()
    
    existing_positions = {"BTC": 0.1}  # 10% in BTC
    new_signal = TradingSignal(asset="ETH", position_size=0.1)
    
    adjusted_signal = analyzer.adjust_for_correlation(
        new_signal, existing_positions, correlation=0.9
    )
    
    # High correlation should reduce position size
    assert adjusted_signal.position_size < new_signal.position_size
```

#### GREEN: Implement correlation analyzer

```python
# src/trading/decision_engine/correlation_analyzer.py
import numpy as np
from typing import Dict, List

class CorrelationAnalyzer:
    def __init__(self, lookback_period: int = 30):
        self.lookback_period = lookback_period
        
    def calculate_correlations(self, price_data: Dict[str, List[float]]) -> Dict[str, float]:
        correlations = {}
        assets = list(price_data.keys())
        
        for i in range(len(assets)):
            for j in range(i + 1, len(assets)):
                asset1, asset2 = assets[i], assets[j]
                
                # Calculate returns
                returns1 = np.diff(price_data[asset1]) / price_data[asset1][:-1]
                returns2 = np.diff(price_data[asset2]) / price_data[asset2][:-1]
                
                # Calculate correlation
                if len(returns1) > 0 and len(returns2) > 0:
                    corr = np.corrcoef(returns1, returns2)[0, 1]
                    correlations[f"{asset1}-{asset2}"] = corr
                    
        return correlations
    
    def adjust_for_correlation(self, signal: TradingSignal, 
                             existing_positions: Dict, 
                             correlation: float) -> TradingSignal:
        if abs(correlation) > 0.7:  # High correlation threshold
            # Reduce position size based on correlation
            reduction_factor = 1 - (abs(correlation) - 0.7) * 2
            signal.position_size *= max(reduction_factor, 0.5)
            
        return signal
```

### Phase 5: Trading Strategy Implementation

#### RED: Test Swing Trading Strategy

```python
@pytest.mark.asyncio
async def test_swing_trading_strategy():
    """Test swing trading signal generation"""
    from src.trading.decision_engine.strategies.swing_trader import SwingTradingStrategy
    
    strategy = SwingTradingStrategy()
    
    # Mock technical setup
    market_data = {
        "ticker": "MSFT",
        "current_price": 380.50,
        "sma_50": 375.00,
        "sma_200": 370.00,
        "rsi": 65,
        "volume_ratio": 1.5,  # 50% above average
        "atr": 5.50  # Average True Range
    }
    
    news_catalyst = {
        "headline": "Microsoft announces major cloud partnership",
        "sentiment": 0.8,
        "relevance": "high"
    }
    
    signal = await strategy.generate_signal(market_data, news_catalyst)
    
    assert signal.strategy == TradingStrategy.SWING
    assert signal.holding_period == "3-7 days"
    assert signal.stop_loss == pytest.approx(380.50 - 2 * 5.50, 0.01)  # 2 ATR stop
    assert signal.take_profit == pytest.approx(380.50 + 3 * 5.50, 0.01)  # 3 ATR target

@pytest.mark.asyncio
async def test_momentum_trading_strategy():
    """Test momentum trading signal generation"""
    from src.trading.decision_engine.strategies.momentum_trader import MomentumTradingStrategy
    
    strategy = MomentumTradingStrategy()
    
    # Strong momentum setup
    momentum_data = {
        "ticker": "NVDA",
        "price_change_5d": 0.12,  # 12% in 5 days
        "volume_surge": 2.5,  # 2.5x average volume
        "relative_strength": 85,  # RS vs market
        "earnings_momentum": {
            "surprise": 0.25,  # 25% beat
            "revision_count": 12,  # 12 upward revisions
            "guidance": "raised"
        }
    }
    
    signal = await strategy.generate_signal(momentum_data)
    
    assert signal.strategy == TradingStrategy.MOMENTUM
    assert signal.momentum_score > 0.8
    assert signal.position_size > 0.05  # Larger position for strong momentum
    assert signal.stop_loss < signal.entry_price * 0.95  # Tight stop for momentum

@pytest.mark.asyncio
async def test_mirror_trading_strategy():
    """Test mirror trading signal generation"""
    from src.trading.decision_engine.strategies.mirror_trader import MirrorTradingStrategy
    
    strategy = MirrorTradingStrategy()
    
    # Buffett-style value investment
    institutional_filing = {
        "institution": "Berkshire Hathaway",
        "ticker": "BAC",
        "action": "purchase",
        "shares": 50000000,
        "avg_price": 32.50,
        "position_change_pct": 0.20,  # 20% increase in position
        "total_value": 1625000000,
        "filing_date": datetime.now() - timedelta(days=1)
    }
    
    signal = await strategy.generate_mirror_signal(institutional_filing)
    
    assert signal.strategy == TradingStrategy.MIRROR
    assert signal.mirror_source == "Berkshire Hathaway"
    assert signal.confidence > 0.8  # High confidence for Buffett
    assert signal.holding_period == "6-12 months"  # Long-term like Buffett
    assert signal.position_size <= 0.02  # Conservative position sizing
```

#### GREEN: Implement Trading Strategies

```python
# src/trading/decision_engine/strategies/swing_trader.py
class SwingTradingStrategy:
    def __init__(self, config: Dict = None):
        self.config = config or self._default_swing_config()
        
    async def generate_signal(self, market_data: Dict, catalyst: Dict = None) -> TradingSignal:
        # Check for swing setup
        if not self._is_valid_swing_setup(market_data):
            return None
            
        # Calculate position parameters
        atr = market_data['atr']
        current_price = market_data['current_price']
        
        # Swing trades use 2-3 ATR stops and 3-5 ATR targets
        stop_loss = current_price - (2 * atr)
        take_profit = current_price + (3.5 * atr)
        
        # Position size based on risk
        position_size = self._calculate_position_size(atr, current_price)
        
        return TradingSignal(
            strategy=TradingStrategy.SWING,
            holding_period="3-7 days",
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=position_size,
            technical_indicators={
                "entry_setup": "MA_crossover",
                "volume_confirmation": True,
                "risk_reward_ratio": 1.75
            }
        )

# src/trading/decision_engine/strategies/mirror_trader.py  
class MirrorTradingStrategy:
    def __init__(self):
        self.trusted_institutions = [
            "Berkshire Hathaway", "Bridgewater", "Renaissance",
            "Soros Fund", "Tiger Global", "Third Point"
        ]
        
    async def generate_mirror_signal(self, filing: Dict) -> TradingSignal:
        # Assess institution credibility
        credibility = self._assess_institution_credibility(filing['institution'])
        
        # Scale position based on their commitment
        their_position_size = filing['total_value'] / self._get_institution_aum(filing['institution'])
        our_position_size = min(their_position_size * 0.1, 0.02)  # Max 2% position
        
        # Match their expected holding period
        holding_period = self._estimate_holding_period(filing['institution'])
        
        return TradingSignal(
            strategy=TradingStrategy.MIRROR,
            mirror_source=filing['institution'],
            confidence=credibility,
            position_size=our_position_size,
            holding_period=holding_period,
            reasoning=f"Mirroring {filing['institution']} {filing['action']} of {filing['shares']:,} shares"
        )
```

### Phase 6: Bond Market Integration

#### RED: Test Bond Trading Signals

```python
@pytest.mark.asyncio
async def test_bond_trading_signals():
    """Test bond market trading signal generation"""
    from src.trading.decision_engine.strategies.bond_trader import BondTradingStrategy
    
    strategy = BondTradingStrategy()
    
    # Fed announcement scenario
    bond_data = {
        "instrument": "US10Y",  # 10-year Treasury
        "current_yield": 4.25,
        "yield_change_1d": 0.15,  # 15 basis points
        "fed_action": "hawkish_pause",
        "inflation_data": {
            "cpi": 3.2,
            "pce": 2.8,
            "expectations": 2.5
        }
    }
    
    signal = await strategy.generate_bond_signal(bond_data)
    
    assert signal.asset_type == AssetType.BOND
    assert signal.signal_type == SignalType.SHORT  # Short bonds on rising yields
    assert signal.holding_period == "1-3 months"  # Longer holds for bonds

@pytest.mark.asyncio
async def test_yield_curve_strategy():
    """Test yield curve trading strategies"""
    from src.trading.decision_engine.strategies.yield_curve_trader import YieldCurveStrategy
    
    strategy = YieldCurveStrategy()
    
    # Steepening yield curve scenario
    curve_data = {
        "2Y_yield": 4.50,
        "10Y_yield": 4.25,
        "30Y_yield": 4.40,
        "curve_change": "inverting",
        "historical_percentile": 95  # Very inverted historically
    }
    
    signals = await strategy.generate_curve_trades(curve_data)
    
    assert len(signals) == 2  # Pairs trade
    assert any(s.asset == "US2Y" and s.signal_type == SignalType.SHORT for s in signals)
    assert any(s.asset == "US10Y" and s.signal_type == SignalType.BUY for s in signals)
```

### Phase 6: Complete Decision Engine Integration

#### RED: Test complete decision engine

```python
@pytest.mark.asyncio
async def test_complete_decision_engine():
    """Test complete trading decision engine"""
    from src.trading.decision_engine import TradingDecisionEngine
    
    engine = TradingDecisionEngine()
    
    # Mock news sentiment data
    sentiment_data = {
        "article_id": "news-123",
        "asset": "BTC",
        "sentiment_score": 0.75,
        "confidence": 0.85,
        "market_impact": {
            "direction": "bullish",
            "magnitude": 0.6,
            "timeframe": "short-term"
        },
        "entities": ["Bitcoin", "SEC", "ETF"]
    }
    
    # Mock current positions
    current_positions = {
        "ETH": {"size": 0.1, "entry_price": 3000},
        "ADA": {"size": 0.05, "entry_price": 1.0}
    }
    
    signal = await engine.process_news_sentiment(
        sentiment_data, 
        current_positions
    )
    
    assert signal is not None
    assert signal.asset == "BTC"
    assert signal.position_size <= 0.1  # Risk limit
    assert signal.stop_loss < signal.entry_price  # For buy signal

@pytest.mark.asyncio
async def test_portfolio_rebalancing():
    """Test portfolio rebalancing decisions"""
    engine = TradingDecisionEngine()
    
    current_positions = {
        "BTC": {"size": 0.5, "entry_price": 40000, "current_price": 45000},
        "ETH": {"size": 0.3, "entry_price": 2500, "current_price": 2000},
        "ADA": {"size": 0.2, "entry_price": 1.0, "current_price": 1.2}
    }
    
    signals = await engine.evaluate_portfolio(current_positions)
    
    # Should generate rebalancing signals
    assert len(signals) > 0
    assert any(s.signal_type == SignalType.SELL for s in signals)  # Take profits
    assert any(s.signal_type == SignalType.BUY for s in signals)   # Rebalance
```

## Interface Contracts and API Design

### TradingDecisionEngine API
```python
class TradingDecisionEngine:
    """Main trading decision engine"""
    
    async def process_news_sentiment(self, sentiment_data: Dict, 
                                   current_positions: Dict = None) -> TradingSignal:
        """Convert news sentiment to trading signal"""
        
    async def evaluate_portfolio(self, current_positions: Dict) -> List[TradingSignal]:
        """Evaluate portfolio and generate rebalancing signals"""
        
    async def process_market_data(self, market_data: Dict) -> List[TradingSignal]:
        """Process market data for technical signals"""
        
    def set_risk_parameters(self, params: Dict):
        """Update risk management parameters"""
        
    def get_active_signals(self) -> List[TradingSignal]:
        """Get currently active trading signals"""
```

### SignalFilter Interface
```python
class SignalFilter(ABC):
    """Base class for signal filters"""
    
    @abstractmethod
    def filter(self, signals: List[TradingSignal]) -> List[TradingSignal]:
        """Filter trading signals based on criteria"""
```

## Dependency Injection Points

1. **Risk Manager**: Configurable risk parameters
2. **Market Data Provider**: Real-time price feeds
3. **Signal Generators**: Multiple signal sources
4. **Portfolio Manager**: Current position tracking

## Mock Object Specifications

### MockMarketDataProvider
```python
class MockMarketDataProvider:
    def __init__(self, prices: Dict[str, float]):
        self.prices = prices
        
    async def get_price(self, asset: str) -> float:
        return self.prices.get(asset, 0.0)
        
    async def get_market_data(self, asset: str) -> Dict:
        return {
            "current_price": self.prices.get(asset, 0.0),
            "volume_24h": 1000000,
            "price_change_24h": 0.05
        }
```

### MockPortfolioManager
```python
class MockPortfolioManager:
    def __init__(self, positions: Dict = None):
        self.positions = positions or {}
        
    def get_position(self, asset: str) -> Dict:
        return self.positions.get(asset, {"size": 0})
        
    def get_all_positions(self) -> Dict:
        return self.positions.copy()
```

## Refactoring Checkpoints

1. **After Phase 2**: Extract signal generation strategies
2. **After Phase 3**: Optimize risk calculations
3. **After Phase 4**: Review correlation algorithms
4. **After Phase 5**: Modularize decision components

## Code Coverage Targets

- **Unit Tests**: 95% coverage for all components
- **Integration Tests**: 90% for full engine
- **Edge Cases**: 100% for risk scenarios
- **Performance Tests**: 100 decisions/second

## Implementation Timeline

1. **Day 1**: Core interfaces and models
2. **Day 2**: News-to-signal conversion
3. **Day 3**: Risk management system
4. **Day 4**: Correlation analysis
5. **Day 5-6**: Complete engine integration
6. **Day 7**: Performance optimization
7. **Day 8**: Integration with existing trader

## Success Criteria

- [ ] Signal generation accuracy > 75% across all strategies
- [ ] Risk limits never exceeded for any asset class
- [ ] Portfolio volatility < configured max
- [ ] Correlation analysis prevents concentration
- [ ] Clear reasoning for all decisions
- [ ] Swing trade win rate > 55% with 1.5:1 risk/reward
- [ ] Momentum trades capture > 70% of strong trends
- [ ] Mirror trades match institutional performance within 10%
- [ ] Bond signals correctly predict yield direction > 60%
- [ ] Multi-asset portfolio optimization working
- [ ] Seamless integration with all trading APIs (stocks, bonds, crypto)