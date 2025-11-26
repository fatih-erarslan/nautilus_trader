"""
Tests for transformer sentiment analysis logic without ML dependencies
"""
import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime
import sys

# Mock the ML dependencies before importing
sys.modules['torch'] = MagicMock()
sys.modules['transformers'] = MagicMock()

from src.news_trading.sentiment_analysis.models import (
    SentimentResult, MarketImpact, SentimentDirection, SentimentBreakdown
)


class TestTransformerSentimentLogic:
    """Test transformer sentiment analysis logic"""
    
    def test_preprocess_text(self):
        """Test text preprocessing logic"""
        from src.news_trading.sentiment_analysis.transformer_sentiment import TransformerSentiment
        
        # Create a minimal instance for testing preprocessing
        with patch('src.news_trading.sentiment_analysis.transformer_sentiment.AutoTokenizer'), \
             patch('src.news_trading.sentiment_analysis.transformer_sentiment.AutoModelForSequenceClassification'), \
             patch('src.news_trading.sentiment_analysis.transformer_sentiment.torch'):
            
            analyzer = TransformerSentiment()
            
            # Test URL removal
            text = "Check this https://example.com Bitcoin is rising!"
            cleaned = analyzer._preprocess_text(text)
            assert "https://" not in cleaned
            assert "Bitcoin" in cleaned
            
            # Test special character removal
            text = "Bitcoin $BTC to the moon! #crypto @user"
            cleaned = analyzer._preprocess_text(text)
            assert "$" not in cleaned
            assert "#" not in cleaned
            assert "@" not in cleaned
            assert "Bitcoin" in cleaned
            
            # Test emoji removal
            text = "Bitcoin 🚀🌙 is awesome!"
            cleaned = analyzer._preprocess_text(text)
            assert "Bitcoin" in cleaned
            assert "awesome" in cleaned
            # Emojis should be removed
            
    def test_calculate_sentiment_metrics(self):
        """Test sentiment metric calculation"""
        from src.news_trading.sentiment_analysis.transformer_sentiment import TransformerSentiment
        
        with patch('src.news_trading.sentiment_analysis.transformer_sentiment.AutoTokenizer'), \
             patch('src.news_trading.sentiment_analysis.transformer_sentiment.AutoModelForSequenceClassification'), \
             patch('src.news_trading.sentiment_analysis.transformer_sentiment.torch'):
            
            analyzer = TransformerSentiment()
            
            # Test bullish sentiment
            # Create a mock predictions object that behaves like a tensor
            class MockPredictions:
                def __getitem__(self, idx):
                    if idx == 0:
                        return np.array([0.1, 0.2, 0.7])  # [neg, neu, pos]
                    return None
            
            predictions = MockPredictions()
            sentiment, confidence = analyzer._calculate_sentiment_metrics(predictions)
            
            assert sentiment == 0.6  # 0.7 - 0.1
            assert confidence == 0.7  # max probability
            
            # Test bearish sentiment
            class MockPredictionsBearish:
                def __getitem__(self, idx):
                    if idx == 0:
                        return np.array([0.8, 0.15, 0.05])
                    return None
            
            predictions = MockPredictionsBearish()
            sentiment, confidence = analyzer._calculate_sentiment_metrics(predictions)
            
            assert sentiment == -0.75  # 0.05 - 0.8
            assert confidence == 0.8
            
            # Test neutral sentiment with low confidence adjustment
            class MockPredictionsNeutral:
                def __getitem__(self, idx):
                    if idx == 0:
                        return np.array([0.35, 0.4, 0.25])
                    return None
            
            predictions = MockPredictionsNeutral()
            sentiment, confidence = analyzer._calculate_sentiment_metrics(predictions)
            
            assert abs(sentiment - (-0.1)) < 0.0001  # 0.25 - 0.35 (with floating point tolerance)
            assert confidence < 0.4  # Adjusted down due to near-neutral
    
    def test_market_impact_prediction(self):
        """Test market impact prediction logic"""
        from src.news_trading.sentiment_analysis.transformer_sentiment import TransformerSentiment
        
        with patch('src.news_trading.sentiment_analysis.transformer_sentiment.AutoTokenizer'), \
             patch('src.news_trading.sentiment_analysis.transformer_sentiment.AutoModelForSequenceClassification'), \
             patch('src.news_trading.sentiment_analysis.transformer_sentiment.torch'):
            
            analyzer = TransformerSentiment()
            
            # Test bullish impact
            impact = analyzer._predict_market_impact(
                sentiment_score=0.8,
                confidence=0.9,
                entities=["BTC", "ETH"],
                text="BREAKING: Bitcoin ETF approved!"
            )
            
            assert impact.direction == SentimentDirection.BULLISH
            assert impact.magnitude > 0.7  # High sentiment * high confidence
            assert impact.timeframe == "immediate"  # "BREAKING" keyword
            assert "BTC" in impact.affected_assets
            assert "regulatory" in impact.catalysts  # "approved" keyword
            
            # Test bearish impact
            impact = analyzer._predict_market_impact(
                sentiment_score=-0.7,
                confidence=0.8,
                entities=["BTC"],
                text="Crypto market crashes as exchange fails"
            )
            
            assert impact.direction == SentimentDirection.BEARISH
            assert impact.volatility_expected in ["extreme", "high"]  # "crashes" keyword
            assert "market" in impact.catalysts
    
    def test_timeframe_determination(self):
        """Test timeframe determination logic"""
        from src.news_trading.sentiment_analysis.transformer_sentiment import TransformerSentiment
        
        with patch('src.news_trading.sentiment_analysis.transformer_sentiment.AutoTokenizer'), \
             patch('src.news_trading.sentiment_analysis.transformer_sentiment.AutoModelForSequenceClassification'), \
             patch('src.news_trading.sentiment_analysis.transformer_sentiment.torch'):
            
            analyzer = TransformerSentiment()
            
            # Immediate timeframe
            assert analyzer._determine_timeframe("BREAKING: Major news") == "immediate"
            assert analyzer._determine_timeframe("Just announced") == "immediate"
            
            # Short-term timeframe
            assert analyzer._determine_timeframe("Upcoming announcement next week") == "short-term"
            assert analyzer._determine_timeframe("Soon to be released") == "short-term"
            
            # Long-term timeframe (default)
            assert analyzer._determine_timeframe("Market analysis report") == "long-term"
    
    def test_catalyst_identification(self):
        """Test catalyst identification"""
        from src.news_trading.sentiment_analysis.transformer_sentiment import TransformerSentiment
        
        with patch('src.news_trading.sentiment_analysis.transformer_sentiment.AutoTokenizer'), \
             patch('src.news_trading.sentiment_analysis.transformer_sentiment.AutoModelForSequenceClassification'), \
             patch('src.news_trading.sentiment_analysis.transformer_sentiment.torch'):
            
            analyzer = TransformerSentiment()
            
            # Multiple catalysts
            text = "SEC approves Bitcoin ETF after institutional investment surge"
            catalysts = analyzer._identify_catalysts(text)
            
            assert "regulatory" in catalysts  # SEC
            assert "institutional" in catalysts  # institutional
            assert len(catalysts) <= 3  # Limited to top 3
            
            # Technical catalyst
            text = "Ethereum upgrade launches with new development"
            catalysts = analyzer._identify_catalysts(text)
            assert "technical" in catalysts
            
            # Market catalyst
            text = "Market crash leads to massive selloff"
            catalysts = analyzer._identify_catalysts(text)
            assert "market" in catalysts
    
    def test_volatility_estimation(self):
        """Test volatility estimation"""
        from src.news_trading.sentiment_analysis.transformer_sentiment import TransformerSentiment
        
        with patch('src.news_trading.sentiment_analysis.transformer_sentiment.AutoTokenizer'), \
             patch('src.news_trading.sentiment_analysis.transformer_sentiment.AutoModelForSequenceClassification'), \
             patch('src.news_trading.sentiment_analysis.transformer_sentiment.torch'):
            
            analyzer = TransformerSentiment()
            
            # Extreme volatility
            assert analyzer._estimate_volatility(0.9, "Market crashes dramatically") == "extreme"
            assert analyzer._estimate_volatility(0.8, "Prices skyrocket") == "extreme"
            
            # High volatility
            assert analyzer._estimate_volatility(0.8, "Strong movement expected") == "high"
            
            # Medium volatility
            assert analyzer._estimate_volatility(0.5, "Moderate changes") == "medium"
            
            # Low volatility
            assert analyzer._estimate_volatility(0.2, "Stable market") == "low"
    
    def test_tone_analysis(self):
        """Test tone analysis"""
        from src.news_trading.sentiment_analysis.transformer_sentiment import TransformerSentiment
        
        with patch('src.news_trading.sentiment_analysis.transformer_sentiment.AutoTokenizer'), \
             patch('src.news_trading.sentiment_analysis.transformer_sentiment.AutoModelForSequenceClassification'), \
             patch('src.news_trading.sentiment_analysis.transformer_sentiment.torch'):
            
            analyzer = TransformerSentiment()
            
            # High intensity with exclamations
            tone = analyzer._analyze_tone("Amazing news!!!")
            assert tone > 0.2
            
            # Uncertain tone with questions
            tone = analyzer._analyze_tone("Is this good? Will it work?")
            assert tone < 0
            
            # High caps intensity
            tone = analyzer._analyze_tone("BREAKING NEWS")
            assert tone > 0.1
    
    def test_language_intensity(self):
        """Test language intensity analysis"""
        from src.news_trading.sentiment_analysis.transformer_sentiment import TransformerSentiment
        
        with patch('src.news_trading.sentiment_analysis.transformer_sentiment.AutoTokenizer'), \
             patch('src.news_trading.sentiment_analysis.transformer_sentiment.AutoModelForSequenceClassification'), \
             patch('src.news_trading.sentiment_analysis.transformer_sentiment.torch'):
            
            analyzer = TransformerSentiment()
            
            # High intensity language
            intensity = analyzer._analyze_intensity("Absolutely massive huge gains")
            assert intensity > 0.7
            
            # Medium intensity
            intensity = analyzer._analyze_intensity("Very significant changes")
            assert intensity > 0.5
            
            # Low intensity
            intensity = analyzer._analyze_intensity("Minor adjustments made")
            assert intensity <= 0.5
    
    def test_reasoning_generation(self):
        """Test reasoning generation"""
        from src.news_trading.sentiment_analysis.transformer_sentiment import TransformerSentiment
        
        with patch('src.news_trading.sentiment_analysis.transformer_sentiment.AutoTokenizer'), \
             patch('src.news_trading.sentiment_analysis.transformer_sentiment.AutoModelForSequenceClassification'), \
             patch('src.news_trading.sentiment_analysis.transformer_sentiment.torch'):
            
            analyzer = TransformerSentiment()
            
            # Bullish reasoning
            reasoning = analyzer._generate_reasoning(0.8, 0.9, "Great news!")
            assert "bullish" in reasoning.lower()
            assert "strong" in reasoning.lower()
            assert "90%" in reasoning
            
            # Bearish reasoning
            reasoning = analyzer._generate_reasoning(-0.7, 0.85, "Bad news")
            assert "bearish" in reasoning.lower()
            
            # Neutral reasoning
            reasoning = analyzer._generate_reasoning(0.1, 0.6, "Market update")
            assert "weak" in reasoning.lower()
    
    def test_supported_languages(self):
        """Test language support detection"""
        from src.news_trading.sentiment_analysis.transformer_sentiment import TransformerSentiment
        
        with patch('src.news_trading.sentiment_analysis.transformer_sentiment.AutoTokenizer'), \
             patch('src.news_trading.sentiment_analysis.transformer_sentiment.AutoModelForSequenceClassification'), \
             patch('src.news_trading.sentiment_analysis.transformer_sentiment.torch'):
            
            # English-only model
            analyzer = TransformerSentiment(model_name="finbert")
            languages = analyzer.get_supported_languages()
            assert languages == ["en"]
            
            # Multilingual model
            analyzer = TransformerSentiment(model_name="xlm-roberta-base")
            languages = analyzer.get_supported_languages()
            assert len(languages) > 1
            assert "en" in languages
            assert "es" in languages