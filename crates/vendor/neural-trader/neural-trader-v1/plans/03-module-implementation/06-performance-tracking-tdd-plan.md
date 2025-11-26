# Performance Tracking Module - TDD Implementation Plan

## Module Overview
The Performance Tracking module monitors trading performance, attributes results to news events, tracks ML model accuracy, provides analytics, and enables A/B testing of different strategies.

## Test-First Implementation Sequence

### Phase 1: Core Performance Metrics (Red-Green-Refactor)

#### RED: Write failing tests first

```python
# tests/test_performance_tracking.py

def test_performance_tracker_interface():
    """Test that PerformanceTracker abstract interface is properly defined"""
    from src.performance.base import PerformanceTracker
    
    class TestTracker(PerformanceTracker):
        pass
    
    # Should fail - abstract methods not implemented
    with pytest.raises(TypeError):
        tracker = TestTracker()

def test_trade_result_model():
    """Test TradeResult data model"""
    from src.performance.models import TradeResult, TradeStatus
    
    result = TradeResult(
        trade_id="trade-123",
        signal_id="signal-456",
        asset="BTC",
        entry_time=datetime.now() - timedelta(hours=2),
        exit_time=datetime.now(),
        entry_price=45000.0,
        exit_price=46500.0,
        position_size=0.05,
        pnl=75.0,  # Profit/Loss in USD
        pnl_percentage=3.33,
        status=TradeStatus.CLOSED,
        news_events=["news-001", "news-002"],
        sentiment_scores=[0.8, 0.75],
        fees=5.0
    )
    
    assert result.pnl == 75.0
    assert result.pnl_percentage == 3.33
    assert result.status == TradeStatus.CLOSED
    assert len(result.news_events) == 2
```

#### GREEN: Implement minimal code to pass

```python
# src/performance/models.py
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict
from enum import Enum

class TradeStatus(Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"

@dataclass
class TradeResult:
    trade_id: str
    signal_id: str
    asset: str
    entry_time: datetime
    exit_time: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    position_size: float
    pnl: float = 0.0
    pnl_percentage: float = 0.0
    status: TradeStatus = TradeStatus.OPEN
    news_events: List[str] = None
    sentiment_scores: List[float] = None
    fees: float = 0.0
    metadata: Dict = None

@dataclass
class PerformanceMetrics:
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    average_win: float
    average_loss: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    total_pnl: float

# src/performance/base.py
from abc import ABC, abstractmethod
from typing import List, Dict
from .models import TradeResult, PerformanceMetrics

class PerformanceTracker(ABC):
    @abstractmethod
    def record_trade(self, trade_result: TradeResult) -> None:
        """Record a completed trade"""
        pass
    
    @abstractmethod
    def calculate_metrics(self, period: str = "all") -> PerformanceMetrics:
        """Calculate performance metrics for a period"""
        pass
```

### Phase 2: Trade Attribution System

#### RED: Test news event attribution

```python
def test_trade_attribution():
    """Test attribution of trades to news events"""
    from src.performance.attribution import TradeAttributor
    
    attributor = TradeAttributor()
    
    trade = TradeResult(
        trade_id="trade-123",
        signal_id="signal-456",
        asset="BTC",
        entry_time=datetime(2024, 1, 15, 10, 0),
        exit_time=datetime(2024, 1, 15, 14, 0),
        pnl=100.0,
        news_events=["news-001", "news-002"]
    )
    
    news_metadata = {
        "news-001": {
            "source": "reuters",
            "sentiment": 0.8,
            "published": datetime(2024, 1, 15, 9, 30)
        },
        "news-002": {
            "source": "bloomberg",
            "sentiment": 0.6,
            "published": datetime(2024, 1, 15, 9, 45)
        }
    }
    
    attribution = attributor.attribute_trade(trade, news_metadata)
    
    assert "reuters" in attribution.source_contributions
    assert attribution.source_contributions["reuters"] > 0
    assert attribution.primary_catalyst == "news-001"

def test_sentiment_accuracy_tracking():
    """Test tracking of sentiment prediction accuracy"""
    from src.performance.attribution import SentimentAccuracyTracker
    
    tracker = SentimentAccuracyTracker()
    
    # Record prediction and outcome
    tracker.record_prediction(
        news_id="news-001",
        predicted_sentiment=0.8,
        predicted_impact="bullish",
        actual_price_change=0.05  # 5% increase
    )
    
    accuracy = tracker.calculate_accuracy()
    assert accuracy["direction_accuracy"] > 0
    assert accuracy["magnitude_mae"] >= 0
```

#### GREEN: Implement attribution system

```python
# src/performance/attribution.py
from typing import Dict, List
from datetime import datetime
from .models import TradeResult

class TradeAttributor:
    def __init__(self):
        self.attribution_window = timedelta(hours=4)  # News impact window
        
    def attribute_trade(self, trade: TradeResult, news_metadata: Dict) -> Attribution:
        source_contributions = {}
        sentiment_weights = {}
        
        for news_id in trade.news_events:
            if news_id in news_metadata:
                news = news_metadata[news_id]
                
                # Calculate time proximity weight
                time_diff = trade.entry_time - news["published"]
                proximity_weight = self._calculate_proximity_weight(time_diff)
                
                # Calculate sentiment weight
                sentiment_weight = abs(news["sentiment"]) * proximity_weight
                
                # Aggregate by source
                source = news["source"]
                if source not in source_contributions:
                    source_contributions[source] = 0
                source_contributions[source] += sentiment_weight
                
                sentiment_weights[news_id] = sentiment_weight
        
        # Normalize contributions
        total_weight = sum(source_contributions.values())
        if total_weight > 0:
            for source in source_contributions:
                source_contributions[source] /= total_weight
                
        # Identify primary catalyst
        primary_catalyst = max(sentiment_weights, key=sentiment_weights.get) if sentiment_weights else None
        
        return Attribution(
            source_contributions=source_contributions,
            primary_catalyst=primary_catalyst,
            news_weights=sentiment_weights
        )
```

### Phase 3: ML Model Performance Tracking

#### RED: Test ML model tracking

```python
def test_ml_model_performance():
    """Test tracking of ML model performance"""
    from src.performance.ml_tracking import MLModelTracker
    
    tracker = MLModelTracker()
    
    # Record model prediction
    tracker.record_prediction(
        model_name="finbert_v1",
        prediction_id="pred-123",
        predicted_value=0.75,
        confidence=0.85,
        actual_value=0.70,
        timestamp=datetime.now()
    )
    
    # Get model metrics
    metrics = tracker.get_model_metrics("finbert_v1")
    
    assert metrics["mae"] >= 0
    assert metrics["rmse"] >= 0
    assert metrics["prediction_count"] == 1
    assert 0 <= metrics["confidence_calibration"] <= 1

def test_model_comparison():
    """Test comparison between multiple models"""
    from src.performance.ml_tracking import MLModelComparator
    
    comparator = MLModelComparator()
    
    # Add model results
    comparator.add_model_results("model_a", predictions=[0.7, 0.8], actuals=[0.75, 0.85])
    comparator.add_model_results("model_b", predictions=[0.6, 0.9], actuals=[0.75, 0.85])
    
    comparison = comparator.compare_models()
    
    assert "model_a" in comparison
    assert "model_b" in comparison
    assert comparison["model_a"]["mae"] < comparison["model_b"]["mae"]
```

#### GREEN: Implement ML tracking

```python
# src/performance/ml_tracking.py
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from typing import Dict, List

class MLModelTracker:
    def __init__(self):
        self.predictions = {}
        
    def record_prediction(self, model_name: str, prediction_id: str,
                         predicted_value: float, confidence: float,
                         actual_value: float, timestamp: datetime):
        if model_name not in self.predictions:
            self.predictions[model_name] = []
            
        self.predictions[model_name].append({
            "id": prediction_id,
            "predicted": predicted_value,
            "confidence": confidence,
            "actual": actual_value,
            "timestamp": timestamp,
            "error": abs(predicted_value - actual_value)
        })
        
    def get_model_metrics(self, model_name: str) -> Dict:
        if model_name not in self.predictions:
            return {}
            
        preds = self.predictions[model_name]
        predicted_values = [p["predicted"] for p in preds]
        actual_values = [p["actual"] for p in preds]
        confidences = [p["confidence"] for p in preds]
        errors = [p["error"] for p in preds]
        
        return {
            "mae": mean_absolute_error(actual_values, predicted_values),
            "rmse": np.sqrt(mean_squared_error(actual_values, predicted_values)),
            "prediction_count": len(preds),
            "confidence_calibration": self._calculate_confidence_calibration(confidences, errors),
            "average_confidence": np.mean(confidences)
        }
```

### Phase 4: A/B Testing Framework

#### RED: Test A/B testing system

```python
def test_ab_test_framework():
    """Test A/B testing framework for strategies"""
    from src.performance.ab_testing import ABTestFramework
    
    ab_test = ABTestFramework(test_name="sentiment_threshold_test")
    
    # Define variants
    ab_test.add_variant("control", {"sentiment_threshold": 0.6})
    ab_test.add_variant("treatment", {"sentiment_threshold": 0.7})
    
    # Assign trades to variants
    trade1 = TradeResult(trade_id="1", pnl=100)
    trade2 = TradeResult(trade_id="2", pnl=-50)
    trade3 = TradeResult(trade_id="3", pnl=75)
    
    ab_test.record_result("control", trade1)
    ab_test.record_result("control", trade2)
    ab_test.record_result("treatment", trade3)
    
    # Analyze results
    results = ab_test.analyze()
    
    assert results["control"]["total_trades"] == 2
    assert results["treatment"]["total_trades"] == 1
    assert results["control"]["average_pnl"] == 25
    assert results["statistical_significance"] is not None

def test_multi_armed_bandit():
    """Test multi-armed bandit for dynamic strategy selection"""
    from src.performance.ab_testing import MultiArmedBandit
    
    bandit = MultiArmedBandit(strategies=["conservative", "moderate", "aggressive"])
    
    # Simulate trades and updates
    for _ in range(100):
        strategy = bandit.select_strategy()
        
        # Simulate trade result based on strategy
        if strategy == "moderate":
            reward = np.random.normal(10, 5)  # Better average
        else:
            reward = np.random.normal(5, 5)
            
        bandit.update(strategy, reward)
    
    # Check that bandit learns the best strategy
    selection_counts = bandit.get_selection_counts()
    assert selection_counts["moderate"] > selection_counts["conservative"]
```

#### GREEN: Implement A/B testing

```python
# src/performance/ab_testing.py
from scipy import stats
import numpy as np
from typing import Dict, List

class ABTestFramework:
    def __init__(self, test_name: str):
        self.test_name = test_name
        self.variants = {}
        self.results = {}
        
    def add_variant(self, name: str, config: Dict):
        self.variants[name] = config
        self.results[name] = []
        
    def record_result(self, variant: str, trade: TradeResult):
        if variant in self.results:
            self.results[variant].append(trade)
            
    def analyze(self) -> Dict:
        analysis = {}
        
        for variant, trades in self.results.items():
            pnls = [t.pnl for t in trades]
            analysis[variant] = {
                "total_trades": len(trades),
                "average_pnl": np.mean(pnls) if pnls else 0,
                "std_pnl": np.std(pnls) if pnls else 0,
                "win_rate": sum(1 for t in trades if t.pnl > 0) / len(trades) if trades else 0
            }
            
        # Statistical significance test
        if len(self.variants) == 2 and all(len(self.results[v]) > 0 for v in self.variants):
            variant_names = list(self.variants.keys())
            pnls_a = [t.pnl for t in self.results[variant_names[0]]]
            pnls_b = [t.pnl for t in self.results[variant_names[1]]]
            
            if len(pnls_a) > 1 and len(pnls_b) > 1:
                t_stat, p_value = stats.ttest_ind(pnls_a, pnls_b)
                analysis["statistical_significance"] = {
                    "t_statistic": t_stat,
                    "p_value": p_value,
                    "significant": p_value < 0.05
                }
                
        return analysis
```

### Phase 5: Performance Analytics Dashboard

#### RED: Test analytics generation

```python
def test_performance_analytics():
    """Test performance analytics generation"""
    from src.performance.analytics import PerformanceAnalytics
    
    analytics = PerformanceAnalytics()
    
    # Add trade history
    trades = [
        TradeResult(trade_id="1", pnl=100, asset="BTC", entry_time=datetime(2024, 1, 1)),
        TradeResult(trade_id="2", pnl=-50, asset="ETH", entry_time=datetime(2024, 1, 2)),
        TradeResult(trade_id="3", pnl=75, asset="BTC", entry_time=datetime(2024, 1, 3))
    ]
    
    for trade in trades:
        analytics.add_trade(trade)
    
    # Generate report
    report = analytics.generate_report()
    
    assert report["summary"]["total_pnl"] == 125
    assert report["by_asset"]["BTC"]["total_pnl"] == 175
    assert report["by_asset"]["ETH"]["total_pnl"] == -50
    assert "daily_pnl" in report
    assert "cumulative_pnl" in report

def test_news_source_analytics():
    """Test analytics by news source"""
    analytics = PerformanceAnalytics()
    
    # Add trades with source attribution
    trades_with_attribution = [
        (TradeResult(trade_id="1", pnl=100), {"reuters": 0.7, "bloomberg": 0.3}),
        (TradeResult(trade_id="2", pnl=-20), {"twitter": 0.9, "reuters": 0.1}),
        (TradeResult(trade_id="3", pnl=50), {"bloomberg": 1.0})
    ]
    
    for trade, attribution in trades_with_attribution:
        analytics.add_trade_with_attribution(trade, attribution)
    
    source_performance = analytics.get_source_performance()
    
    assert "reuters" in source_performance
    assert source_performance["reuters"]["weighted_pnl"] > 0
    assert source_performance["twitter"]["weighted_pnl"] < 0
```

## Interface Contracts and API Design

### PerformanceTracker API
```python
class PerformanceTracker:
    """Main performance tracking interface"""
    
    def record_trade(self, trade_result: TradeResult) -> None:
        """Record completed trade"""
        
    def calculate_metrics(self, period: str = "all") -> PerformanceMetrics:
        """Calculate performance metrics"""
        
    def get_trade_history(self, filters: Dict = None) -> List[TradeResult]:
        """Get filtered trade history"""
        
    def generate_report(self, format: str = "json") -> Dict:
        """Generate performance report"""
```

### Attribution API
```python
class AttributionEngine:
    """Trade attribution interface"""
    
    def attribute_trade(self, trade: TradeResult, context: Dict) -> Attribution:
        """Attribute trade to sources"""
        
    def get_source_performance(self) -> Dict[str, SourceMetrics]:
        """Get performance by source"""
        
    def get_model_performance(self) -> Dict[str, ModelMetrics]:
        """Get performance by ML model"""
```

## Dependency Injection Points

1. **Database Backend**: For persistent storage
2. **Analytics Engine**: Pluggable analytics providers
3. **Visualization Tools**: Chart generation
4. **Export Formats**: CSV, JSON, PDF reports

## Mock Object Specifications

### MockTradeHistory
```python
class MockTradeHistory:
    @staticmethod
    def generate_trades(count: int, seed: int = 42) -> List[TradeResult]:
        np.random.seed(seed)
        trades = []
        
        for i in range(count):
            pnl = np.random.normal(10, 50)
            trades.append(TradeResult(
                trade_id=f"mock-{i}",
                signal_id=f"signal-{i}",
                asset=np.random.choice(["BTC", "ETH", "ADA"]),
                entry_time=datetime.now() - timedelta(days=count-i),
                exit_time=datetime.now() - timedelta(days=count-i-1),
                pnl=pnl,
                pnl_percentage=pnl / 1000,
                status=TradeStatus.CLOSED
            ))
            
        return trades
```

## Refactoring Checkpoints

1. **After Phase 2**: Optimize attribution algorithms
2. **After Phase 3**: Consolidate ML metrics
3. **After Phase 4**: Review statistical tests
4. **After Phase 5**: Extract visualization logic

## Code Coverage Targets

- **Unit Tests**: 95% coverage for calculations
- **Integration Tests**: 90% for data persistence
- **Edge Cases**: 100% for edge scenarios
- **Performance Tests**: Handle 10k+ trades

## Implementation Timeline

1. **Day 1**: Core models and interfaces
2. **Day 2**: Trade attribution system
3. **Day 3**: ML model tracking
4. **Day 4**: A/B testing framework
5. **Day 5-6**: Analytics dashboard
6. **Day 7**: Integration with database
7. **Day 8**: Reporting and exports

## Success Criteria

- [ ] Accurate P&L calculations
- [ ] Attribution accuracy > 90%
- [ ] ML model tracking operational
- [ ] A/B tests with statistical significance
- [ ] Real-time analytics updates
- [ ] Comprehensive reporting suite