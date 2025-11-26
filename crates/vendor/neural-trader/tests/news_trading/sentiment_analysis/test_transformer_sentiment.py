"""
Tests for transformer-based sentiment analysis
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import asyncio
import numpy as np

# Mock torch and transformers to avoid dependencies in tests
import sys
sys.modules['torch'] = MagicMock()
sys.modules['transformers'] = MagicMock()

from src.news_trading.sentiment_analysis.models import SentimentDirection


class TestTransformerSentiment:
    """Test TransformerSentiment analyzer"""
    
    @patch('src.news_trading.sentiment_analysis.transformer_sentiment.AutoTokenizer')
    @patch('src.news_trading.sentiment_analysis.transformer_sentiment.AutoModelForSequenceClassification')
    def test_transformer_sentiment_init(self, mock_model_class, mock_tokenizer_class):
        """Test TransformerSentiment initialization"""
        # Mock the tokenizer and model
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model_class.from_pretrained.return_value = mock_model
        
        from src.news_trading.sentiment_analysis.transformer_sentiment import TransformerSentiment
        
        analyzer = TransformerSentiment(model_name="finbert")
        assert analyzer.get_model_name() == "finbert"
        assert analyzer.model is not None
        assert analyzer.tokenizer is not None
    
    def test_transformer_sentiment_default_model(self):
        """Test default model initialization"""
        analyzer = TransformerSentiment()
        assert analyzer.get_model_name() == "ProsusAI/finbert"
    
    @pytest.mark.asyncio
    async def test_transformer_sentiment_analysis_bullish(self):
        """Test sentiment analysis with bullish text"""
        analyzer = TransformerSentiment()
        
        text = "Bitcoin surges to all-time high amid institutional adoption and positive regulatory developments"
        result = await analyzer.analyze(text, entities=["Bitcoin"])
        
        assert result.overall_sentiment > 0  # Positive sentiment
        assert result.confidence > 0.7
        assert result.market_impact.direction == SentimentDirection.BULLISH
        assert "Bitcoin" in result.market_impact.affected_assets
    
    @pytest.mark.asyncio
    async def test_transformer_sentiment_analysis_bearish(self):
        """Test bearish sentiment detection"""
        analyzer = TransformerSentiment()
        
        text = "Crypto market crashes as regulatory crackdown intensifies and major exchange faces bankruptcy"
        result = await analyzer.analyze(text, entities=["BTC", "ETH"])
        
        assert result.overall_sentiment < 0  # Negative sentiment
        assert result.market_impact.direction == SentimentDirection.BEARISH
        assert set(["BTC", "ETH"]).issubset(set(result.market_impact.affected_assets))
    
    @pytest.mark.asyncio
    async def test_transformer_sentiment_analysis_neutral(self):
        """Test neutral sentiment detection"""
        analyzer = TransformerSentiment()
        
        text = "Bitcoin trades sideways as market awaits Federal Reserve decision on interest rates"
        result = await analyzer.analyze(text, entities=["Bitcoin"])
        
        assert -0.3 < result.overall_sentiment < 0.3  # Near neutral
        assert result.market_impact.direction == SentimentDirection.NEUTRAL
    
    @pytest.mark.asyncio
    async def test_transformer_batch_analysis(self):
        """Test batch analysis functionality"""
        analyzer = TransformerSentiment()
        
        texts = [
            "Strong bullish momentum in crypto markets",
            "Major selloff hits digital assets",
            "Market consolidates at current levels"
        ]
        
        results = await analyzer.analyze_batch(texts)
        
        assert len(results) == 3
        assert results[0].overall_sentiment > 0  # Bullish
        assert results[1].overall_sentiment < 0  # Bearish
        assert -0.3 < results[2].overall_sentiment < 0.3  # Neutral
    
    def test_transformer_sentiment_preprocessing(self):
        """Test text preprocessing"""
        analyzer = TransformerSentiment()
        
        # Test preprocessing method
        text = "BREAKING: Bitcoin $BTC hits $100k!!! 🚀🚀🚀 #Crypto #ToTheMoon"
        cleaned = analyzer._preprocess_text(text)
        
        assert "$" not in cleaned  # Remove special chars
        assert "🚀" not in cleaned  # Remove emojis
        assert "#" not in cleaned  # Remove hashtags
        assert "bitcoin" in cleaned.lower()  # Preserve important words
    
    @pytest.mark.asyncio
    async def test_transformer_sentiment_long_text(self):
        """Test handling of long text that needs truncation"""
        analyzer = TransformerSentiment()
        
        # Create a very long text
        long_text = "Bitcoin price analysis. " * 100 + "Very bullish outlook."
        
        result = await analyzer.analyze(long_text)
        
        # Should handle without error and maintain sentiment
        assert result is not None
        assert result.overall_sentiment > 0  # Should catch bullish ending
    
    @pytest.mark.asyncio
    async def test_transformer_sentiment_entity_focus(self):
        """Test entity-focused sentiment analysis"""
        analyzer = TransformerSentiment()
        
        text = "While Bitcoin struggles, Ethereum shows strong growth potential with upcoming upgrades"
        
        # Analyze with focus on different entities
        btc_result = await analyzer.analyze(text, entities=["Bitcoin"])
        eth_result = await analyzer.analyze(text, entities=["Ethereum"])
        
        # Bitcoin sentiment should be more negative
        assert btc_result.overall_sentiment < eth_result.overall_sentiment
        assert btc_result.sentiment_breakdown['entity_sentiment'] < 0
        assert eth_result.sentiment_breakdown['entity_sentiment'] > 0
    
    def test_transformer_supported_languages(self):
        """Test language support"""
        analyzer = TransformerSentiment()
        
        languages = analyzer.get_supported_languages()
        assert "en" in languages  # Should support English at minimum
    
    @pytest.mark.asyncio
    async def test_transformer_sentiment_caching(self):
        """Test that sentiment analysis can be cached"""
        analyzer = TransformerSentiment(enable_cache=True)
        
        text = "Bitcoin price rises on positive news"
        
        # First call
        result1 = await analyzer.analyze(text)
        
        # Second call (should be cached)
        result2 = await analyzer.analyze(text)
        
        # Results should be identical
        assert result1.overall_sentiment == result2.overall_sentiment
        assert result1.article_id == result2.article_id
    
    @pytest.mark.asyncio
    async def test_transformer_sentiment_error_handling(self):
        """Test error handling in analysis"""
        analyzer = TransformerSentiment()
        
        # Test with empty text
        with pytest.raises(ValueError, match="Text cannot be empty"):
            await analyzer.analyze("")
        
        # Test with None text
        with pytest.raises(ValueError, match="Text cannot be None"):
            await analyzer.analyze(None)
    
    def test_transformer_model_config(self):
        """Test model configuration options"""
        # Test with custom config
        config = {
            "max_length": 256,
            "batch_size": 16,
            "device": "cpu"
        }
        
        analyzer = TransformerSentiment(model_config=config)
        
        assert analyzer.max_length == 256
        assert analyzer.batch_size == 16
        assert analyzer.device == "cpu"
    
    @pytest.mark.asyncio
    async def test_transformer_confidence_calculation(self):
        """Test confidence score calculation"""
        analyzer = TransformerSentiment()
        
        # Very clear sentiment should have high confidence
        clear_text = "Absolutely terrible news! Market crash imminent! Sell everything!"
        clear_result = await analyzer.analyze(clear_text)
        
        # Mixed sentiment should have lower confidence  
        mixed_text = "Market shows mixed signals with both positive and negative indicators"
        mixed_result = await analyzer.analyze(mixed_text)
        
        assert clear_result.confidence > mixed_result.confidence
        assert clear_result.confidence > 0.8
        assert mixed_result.confidence < 0.7