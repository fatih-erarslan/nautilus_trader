# AI Sentiment Analysis Module - TDD Implementation Plan

## Module Overview
The AI Sentiment Analysis module analyzes parsed news articles to determine market sentiment, predict impact on asset prices, and provide confidence scores using multiple AI models in an ensemble approach.

## Test-First Implementation Sequence

### Phase 1: Core Sentiment Interface (Red-Green-Refactor)

#### RED: Write failing tests first

```python
# tests/test_sentiment_analysis.py

def test_sentiment_analyzer_interface():
    """Test that SentimentAnalyzer abstract interface is properly defined"""
    from src.sentiment.base import SentimentAnalyzer
    
    class TestAnalyzer(SentimentAnalyzer):
        pass
    
    # Should fail - abstract methods not implemented
    with pytest.raises(TypeError):
        analyzer = TestAnalyzer()

def test_sentiment_result_model():
    """Test SentimentResult data model"""
    from src.sentiment.models import SentimentResult, MarketImpact
    
    result = SentimentResult(
        article_id="test-123",
        overall_sentiment=0.75,  # -1 to 1 scale
        confidence=0.85,
        market_impact=MarketImpact(
            direction="bullish",
            magnitude=0.6,
            timeframe="short-term",
            affected_assets=["BTC", "ETH"]
        ),
        sentiment_breakdown={
            "headline": 0.8,
            "content": 0.7,
            "entities": 0.75
        },
        reasoning="Positive regulatory news typically drives market up"
    )
    
    assert result.overall_sentiment == 0.75
    assert result.market_impact.direction == "bullish"
    assert "BTC" in result.market_impact.affected_assets
```

#### GREEN: Implement minimal code to pass

```python
# src/sentiment/models.py
from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum

class SentimentDirection(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"

@dataclass
class MarketImpact:
    direction: SentimentDirection
    magnitude: float  # 0 to 1
    timeframe: str  # "immediate", "short-term", "long-term"
    affected_assets: List[str]
    confidence: float = 0.5

@dataclass
class SentimentResult:
    article_id: str
    overall_sentiment: float  # -1 to 1
    confidence: float  # 0 to 1
    market_impact: MarketImpact
    sentiment_breakdown: Dict[str, float]
    reasoning: str
    model_scores: Dict[str, float] = None

# src/sentiment/base.py
from abc import ABC, abstractmethod
from typing import Dict, List
from .models import SentimentResult

class SentimentAnalyzer(ABC):
    @abstractmethod
    async def analyze(self, text: str, entities: List[str] = None) -> SentimentResult:
        """Analyze sentiment of text"""
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Get name of the sentiment model"""
        pass
```

### Phase 2: Transformer-based Sentiment Analysis

#### RED: Test transformer sentiment analyzer

```python
def test_transformer_sentiment_init():
    """Test TransformerSentiment initialization"""
    from src.sentiment.transformer_sentiment import TransformerSentiment
    
    analyzer = TransformerSentiment(model_name="finbert")
    assert analyzer.get_model_name() == "finbert"
    assert analyzer.model is not None

@pytest.mark.asyncio
async def test_transformer_sentiment_analysis():
    """Test sentiment analysis with transformer model"""
    from src.sentiment.transformer_sentiment import TransformerSentiment
    
    analyzer = TransformerSentiment(model_name="finbert")
    
    text = "Bitcoin surges to all-time high amid institutional adoption"
    result = await analyzer.analyze(text, entities=["Bitcoin"])
    
    assert result.overall_sentiment > 0  # Positive sentiment
    assert result.confidence > 0.7
    assert result.market_impact.direction == SentimentDirection.BULLISH

@pytest.mark.asyncio
async def test_bearish_sentiment():
    """Test bearish sentiment detection"""
    analyzer = TransformerSentiment()
    
    text = "Crypto market crashes as regulatory crackdown intensifies"
    result = await analyzer.analyze(text, entities=["BTC", "ETH"])
    
    assert result.overall_sentiment < 0  # Negative sentiment
    assert result.market_impact.direction == SentimentDirection.BEARISH
```

#### GREEN: Implement transformer sentiment

```python
# src/sentiment/transformer_sentiment.py
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List
from .base import SentimentAnalyzer
from .models import SentimentResult, MarketImpact, SentimentDirection

class TransformerSentiment(SentimentAnalyzer):
    def __init__(self, model_name: str = "ProsusAI/finbert"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
    async def analyze(self, text: str, entities: List[str] = None) -> SentimentResult:
        # Tokenize and get model predictions
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
        # Convert to sentiment score
        sentiment_score = self._calculate_sentiment_score(predictions)
        confidence = float(torch.max(predictions))
        
        # Determine market impact
        market_impact = self._predict_market_impact(sentiment_score, entities, text)
        
        return SentimentResult(
            article_id=hash(text),  # Temporary ID
            overall_sentiment=sentiment_score,
            confidence=confidence,
            market_impact=market_impact,
            sentiment_breakdown={
                "model_prediction": sentiment_score
            },
            reasoning=self._generate_reasoning(sentiment_score, text)
        )
    
    def _calculate_sentiment_score(self, predictions):
        # Map FinBERT output to -1 to 1 scale
        # Assuming [negative, neutral, positive] order
        scores = predictions[0].tolist()
        return scores[2] - scores[0]  # positive - negative
```

### Phase 3: LLM-based Contextual Analysis

#### RED: Test LLM sentiment analyzer

```python
@pytest.mark.asyncio
async def test_llm_sentiment_analyzer():
    """Test LLM-based sentiment analysis"""
    from src.sentiment.llm_sentiment import LLMSentimentAnalyzer
    
    analyzer = LLMSentimentAnalyzer(api_key="test-key")
    
    # Mock LLM response
    with patch('aiohttp.ClientSession.post') as mock_post:
        mock_post.return_value.__aenter__.return_value.json.return_value = {
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "sentiment": 0.8,
                        "confidence": 0.9,
                        "market_impact": "bullish",
                        "reasoning": "Strong positive indicators"
                    })
                }
            }]
        }
        
        text = "Major investment firm announces Bitcoin allocation"
        result = await analyzer.analyze(text, entities=["Bitcoin"])
        
        assert result.overall_sentiment == 0.8
        assert result.confidence == 0.9

def test_llm_prompt_construction():
    """Test LLM prompt construction for crypto context"""
    from src.sentiment.llm_sentiment import LLMSentimentAnalyzer
    
    analyzer = LLMSentimentAnalyzer()
    prompt = analyzer._build_prompt(
        "Bitcoin ETF approved",
        entities=["Bitcoin"],
        context="regulatory"
    )
    
    assert "Bitcoin" in prompt
    assert "regulatory" in prompt
    assert "sentiment" in prompt.lower()
```

#### GREEN: Implement LLM sentiment analyzer

```python
# src/sentiment/llm_sentiment.py
class LLMSentimentAnalyzer(SentimentAnalyzer):
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.model = "anthropic/claude-3-opus"
        
    async def analyze(self, text: str, entities: List[str] = None) -> SentimentResult:
        prompt = self._build_prompt(text, entities)
        
        async with aiohttp.ClientSession() as session:
            response = await self._call_llm(session, prompt)
            
        parsed = self._parse_llm_response(response)
        
        return SentimentResult(
            article_id=hash(text),
            overall_sentiment=parsed["sentiment"],
            confidence=parsed["confidence"],
            market_impact=self._create_market_impact(parsed, entities),
            sentiment_breakdown=parsed.get("breakdown", {}),
            reasoning=parsed["reasoning"]
        )
    
    def _build_prompt(self, text: str, entities: List[str], context: str = None):
        return f"""
        Analyze the sentiment of this crypto/financial news:
        
        Text: {text}
        Entities mentioned: {', '.join(entities or [])}
        
        Provide JSON response with:
        - sentiment: float between -1 (very bearish) and 1 (very bullish)
        - confidence: float between 0 and 1
        - market_impact: "bullish", "bearish", or "neutral"
        - magnitude: expected price impact 0-1
        - timeframe: "immediate", "short-term", or "long-term"
        - reasoning: brief explanation
        
        Consider crypto market dynamics and historical patterns.
        """
```

### Phase 4: Ensemble Sentiment System

#### RED: Test ensemble system

```python
@pytest.mark.asyncio
async def test_ensemble_sentiment():
    """Test ensemble sentiment analysis"""
    from src.sentiment.ensemble import EnsembleSentiment
    
    # Create mock analyzers
    analyzer1 = Mock(spec=SentimentAnalyzer)
    analyzer1.analyze.return_value = SentimentResult(
        article_id="1",
        overall_sentiment=0.8,
        confidence=0.9,
        market_impact=Mock(direction=SentimentDirection.BULLISH)
    )
    
    analyzer2 = Mock(spec=SentimentAnalyzer)
    analyzer2.analyze.return_value = SentimentResult(
        article_id="1", 
        overall_sentiment=0.6,
        confidence=0.8,
        market_impact=Mock(direction=SentimentDirection.BULLISH)
    )
    
    ensemble = EnsembleSentiment([analyzer1, analyzer2])
    result = await ensemble.analyze("Test text")
    
    # Should average the sentiments weighted by confidence
    expected_sentiment = (0.8 * 0.9 + 0.6 * 0.8) / (0.9 + 0.8)
    assert abs(result.overall_sentiment - expected_sentiment) < 0.01

def test_ensemble_conflict_resolution():
    """Test handling of conflicting sentiments"""
    # Test when models disagree significantly
    pass
```

#### GREEN: Implement ensemble system

```python
# src/sentiment/ensemble.py
class EnsembleSentiment(SentimentAnalyzer):
    def __init__(self, analyzers: List[SentimentAnalyzer], weights: List[float] = None):
        self.analyzers = analyzers
        self.weights = weights or [1.0] * len(analyzers)
        
    async def analyze(self, text: str, entities: List[str] = None) -> SentimentResult:
        # Run all analyzers concurrently
        tasks = [analyzer.analyze(text, entities) for analyzer in self.analyzers]
        results = await asyncio.gather(*tasks)
        
        # Weighted average based on confidence
        total_weight = 0
        weighted_sentiment = 0
        
        for result, weight in zip(results, self.weights):
            effective_weight = weight * result.confidence
            weighted_sentiment += result.overall_sentiment * effective_weight
            total_weight += effective_weight
            
        overall_sentiment = weighted_sentiment / total_weight if total_weight > 0 else 0
        
        # Aggregate market impact
        market_impact = self._aggregate_market_impact(results)
        
        return SentimentResult(
            article_id=results[0].article_id,
            overall_sentiment=overall_sentiment,
            confidence=self._calculate_ensemble_confidence(results),
            market_impact=market_impact,
            sentiment_breakdown=self._aggregate_breakdowns(results),
            reasoning=self._generate_ensemble_reasoning(results),
            model_scores={
                analyzer.get_model_name(): result.overall_sentiment 
                for analyzer, result in zip(self.analyzers, results)
            }
        )
```

### Phase 5: Crypto-Specific Sentiment Features

#### RED: Test crypto-specific features

```python
def test_crypto_specific_patterns():
    """Test detection of crypto-specific sentiment patterns"""
    from src.sentiment.crypto_patterns import CryptoPatternAnalyzer
    
    analyzer = CryptoPatternAnalyzer()
    
    # Test FOMO pattern
    text = "Don't miss out on the next Bitcoin rally, institutions are buying"
    patterns = analyzer.detect_patterns(text)
    assert "FOMO" in patterns
    
    # Test FUD pattern
    text = "Regulatory uncertainty creates fear in crypto markets"
    patterns = analyzer.detect_patterns(text)
    assert "FUD" in patterns

def test_whale_movement_sentiment():
    """Test sentiment adjustment for whale movements"""
    from src.sentiment.crypto_sentiment import CryptoSentimentAdjuster
    
    adjuster = CryptoSentimentAdjuster()
    
    base_sentiment = 0.5
    text = "Whale alert: 1000 BTC moved to exchange"
    
    adjusted = adjuster.adjust_for_whale_activity(base_sentiment, text)
    assert adjusted < base_sentiment  # Bearish signal
```

## Interface Contracts and API Design

### SentimentAnalyzer Interface
```python
class SentimentAnalyzer(ABC):
    """Base class for all sentiment analyzers"""
    
    @abstractmethod
    async def analyze(self, text: str, entities: List[str] = None) -> SentimentResult:
        """Analyze sentiment of text"""
        
    @abstractmethod
    def get_model_name(self) -> str:
        """Get name of the sentiment model"""
        
    async def analyze_batch(self, texts: List[str]) -> List[SentimentResult]:
        """Analyze multiple texts efficiently"""
        
    def calibrate(self, historical_data: List[Tuple[str, float]]):
        """Calibrate model with historical sentiment-outcome pairs"""
```

### MarketImpactPredictor Interface
```python
class MarketImpactPredictor(ABC):
    """Predicts market impact from sentiment"""
    
    @abstractmethod
    def predict(self, sentiment: float, entities: List[str], context: Dict) -> MarketImpact:
        """Predict market impact from sentiment score"""
```

## Dependency Injection Points

1. **Sentiment Models**: Transformers, LLMs, rule-based
2. **Model Weights**: Configurable ensemble weights
3. **Pattern Databases**: Crypto-specific patterns
4. **Historical Data**: For model calibration

## Mock Object Specifications

### MockSentimentModel
```python
class MockSentimentModel:
    def __init__(self, fixed_sentiment=0.5, fixed_confidence=0.8):
        self.fixed_sentiment = fixed_sentiment
        self.fixed_confidence = fixed_confidence
        
    async def analyze(self, text, entities=None):
        return SentimentResult(
            article_id="mock-id",
            overall_sentiment=self.fixed_sentiment,
            confidence=self.fixed_confidence,
            market_impact=MarketImpact(
                direction=SentimentDirection.BULLISH if self.fixed_sentiment > 0 else SentimentDirection.BEARISH,
                magnitude=abs(self.fixed_sentiment),
                timeframe="short-term",
                affected_assets=entities or []
            ),
            sentiment_breakdown={},
            reasoning="Mock analysis"
        )
```

## Refactoring Checkpoints

1. **After Phase 2**: Extract common preprocessing logic
2. **After Phase 3**: Optimize LLM prompt templates
3. **After Phase 4**: Review ensemble weighting strategy
4. **After Phase 5**: Consolidate crypto patterns

## Code Coverage Targets

- **Unit Tests**: 95% coverage for all analyzers
- **Integration Tests**: 90% for ensemble system
- **Edge Cases**: 100% for extreme sentiments
- **Performance Tests**: 50 articles/second minimum

## Implementation Timeline

1. **Day 1**: Core interfaces and models
2. **Day 2-3**: Transformer-based sentiment
3. **Day 4**: LLM sentiment analyzer
4. **Day 5**: Ensemble system
5. **Day 6**: Crypto-specific features
6. **Day 7**: Calibration and optimization
7. **Day 8**: Integration testing

## Success Criteria

- [ ] Sentiment accuracy > 85% on test set
- [ ] Market direction prediction > 70% accuracy
- [ ] Sub-second analysis time per article
- [ ] Handles multiple languages
- [ ] Robust to adversarial inputs
- [ ] Clear reasoning for all predictions