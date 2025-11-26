"""
Tests for transformer-based sentiment analysis with mocked dependencies
"""
import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import asyncio
import numpy as np
from datetime import datetime

from src.news_trading.sentiment_analysis.models import SentimentDirection, SentimentResult


class TestTransformerSentimentMocked:
    """Test TransformerSentiment analyzer with mocked ML dependencies"""
    
    @pytest.fixture
    def mock_transformer_imports(self):
        """Mock transformer and torch imports"""
        with patch.dict('sys.modules', {
            'torch': MagicMock(),
            'transformers': MagicMock()
        }):
            yield
    
    @pytest.fixture
    def mock_analyzer(self, mock_transformer_imports):
        """Create a mocked TransformerSentiment analyzer"""
        with patch('src.news_trading.sentiment_analysis.transformer_sentiment.AutoTokenizer') as mock_tokenizer_class, \
             patch('src.news_trading.sentiment_analysis.transformer_sentiment.AutoModelForSequenceClassification') as mock_model_class, \
             patch('src.news_trading.sentiment_analysis.transformer_sentiment.torch') as mock_torch:
            
            # Mock tokenizer
            mock_tokenizer = MagicMock()
            mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
            
            # Mock model
            mock_model = MagicMock()
            mock_model_class.from_pretrained.return_value = mock_model
            
            # Mock torch
            mock_torch.cuda.is_available.return_value = False
            mock_torch.no_grad.return_value.__enter__ = MagicMock()
            mock_torch.no_grad.return_value.__exit__ = MagicMock()
            
            # Import and create analyzer
            from src.news_trading.sentiment_analysis.transformer_sentiment import TransformerSentiment
            analyzer = TransformerSentiment()
            
            # Setup tokenizer return value
            mock_inputs = MagicMock()
            mock_inputs.to.return_value = mock_inputs
            mock_tokenizer.return_value = mock_inputs
            
            # Setup model return value
            mock_outputs = MagicMock()
            mock_outputs.logits = MagicMock()
            mock_model.return_value = mock_outputs
            
            # Store mocks for tests
            analyzer._mock_tokenizer = mock_tokenizer
            analyzer._mock_model = mock_model
            analyzer._mock_torch = mock_torch
            
            return analyzer
    
    def test_initialization(self, mock_analyzer):
        """Test analyzer initialization"""
        assert mock_analyzer.get_model_name() == "ProsusAI/finbert"
        assert mock_analyzer.device == "cpu"
        assert mock_analyzer.max_length == 512
        assert mock_analyzer.batch_size == 8
    
    @pytest.mark.asyncio
    async def test_analyze_bullish_sentiment(self, mock_analyzer):
        """Test analysis of bullish text"""
        # Mock softmax output for bullish sentiment
        probs = np.array([0.1, 0.2, 0.7])  # [neg, neu, pos]
        
        # Create a mock tensor-like object
        class MockTensor:
            def __getitem__(self, idx):
                return probs
        
        mock_analyzer._mock_torch.nn.functional.softmax.return_value = MockTensor()
        
        text = "Bitcoin surges to all-time high amid institutional adoption"
        result = await mock_analyzer.analyze(text, entities=["Bitcoin"])
        
        assert isinstance(result, SentimentResult)
        assert result.overall_sentiment > 0  # Positive (0.7 - 0.1 = 0.6)
        assert result.confidence >= 0.7  # Max probability
        assert result.market_impact.direction == SentimentDirection.BULLISH
        assert "Bitcoin" in result.market_impact.affected_assets
    
    @pytest.mark.asyncio
    async def test_analyze_bearish_sentiment(self, mock_analyzer):
        """Test analysis of bearish text"""
        # Mock softmax output for bearish sentiment
        probs = np.array([0.8, 0.15, 0.05])  # [neg, neu, pos]
        
        # Create a mock that returns the numpy array when indexed
        mock_predictions = MagicMock()
        mock_predictions.__getitem__.side_effect = lambda idx: probs
        mock_analyzer._mock_torch.nn.functional.softmax.return_value = mock_predictions
        
        text = "Crypto market crashes as regulatory crackdown intensifies"
        result = await mock_analyzer.analyze(text, entities=["BTC", "ETH"])
        
        assert result.overall_sentiment < 0  # Negative (0.05 - 0.8 = -0.75)
        assert result.market_impact.direction == SentimentDirection.BEARISH
        assert set(["BTC", "ETH"]).issubset(set(result.market_impact.affected_assets))
    
    @pytest.mark.asyncio
    async def test_analyze_neutral_sentiment(self, mock_analyzer):
        """Test analysis of neutral text"""
        # Mock softmax output for neutral sentiment
        probs = np.array([0.3, 0.5, 0.2])  # [neg, neu, pos]
        
        # Create a mock that returns the numpy array when indexed
        mock_predictions = MagicMock()
        mock_predictions.__getitem__.side_effect = lambda idx: probs
        mock_analyzer._mock_torch.nn.functional.softmax.return_value = mock_predictions
        
        text = "Bitcoin trades sideways as market awaits decision"
        result = await mock_analyzer.analyze(text)
        
        assert -0.3 < result.overall_sentiment < 0.3  # Near neutral
        assert result.market_impact.direction == SentimentDirection.NEUTRAL
    
    def test_preprocess_text(self, mock_analyzer):
        """Test text preprocessing"""
        # Test URL removal
        text_with_url = "Check this out: https://example.com Bitcoin is rising!"
        cleaned = mock_analyzer._preprocess_text(text_with_url)
        assert "https://" not in cleaned
        assert "Bitcoin" in cleaned
        
        # Test special character handling
        text_with_special = "Bitcoin $BTC 🚀 to the moon! #crypto"
        cleaned = mock_analyzer._preprocess_text(text_with_special)
        assert "$" not in cleaned
        assert "#" not in cleaned
        assert "Bitcoin" in cleaned
    
    @pytest.mark.asyncio
    async def test_entity_sentiment_analysis(self, mock_analyzer):
        """Test entity-specific sentiment analysis"""
        # Mock different sentiments for different entities
        call_count = 0
        
        def mock_softmax_side_effect(*args, **kwargs):
            nonlocal call_count
            
            # Different probabilities for each call
            if call_count == 0:
                probs = np.array([0.3, 0.4, 0.3])
            elif call_count == 1:
                probs = np.array([0.6, 0.3, 0.1])  # Bitcoin (negative)
            else:
                probs = np.array([0.1, 0.3, 0.6])  # Ethereum (positive)
            
            mock_predictions = MagicMock()
            mock_predictions.__getitem__.side_effect = lambda idx: probs
            
            call_count += 1
            return mock_predictions
        
        mock_analyzer._mock_torch.nn.functional.softmax.side_effect = mock_softmax_side_effect
        
        text = "While Bitcoin struggles with resistance, Ethereum shows strong momentum"
        result = await mock_analyzer.analyze(text, entities=["Bitcoin", "Ethereum"])
        
        # Entity sentiment should be average of Bitcoin (negative) and Ethereum (positive)
        assert result.sentiment_breakdown.entities == pytest.approx(0.0, abs=0.1)  # Average of -0.5 and 0.5
    
    @pytest.mark.asyncio
    async def test_batch_analysis(self, mock_analyzer):
        """Test batch analysis functionality"""
        # Mock different sentiments for batch
        sentiments = [
            np.array([0.1, 0.2, 0.7]),  # Bullish
            np.array([0.7, 0.2, 0.1]),  # Bearish
            np.array([0.3, 0.4, 0.3])   # Neutral
        ]
        
        call_count = 0
        
        def mock_softmax_batch(*args, **kwargs):
            nonlocal call_count
            probs = sentiments[call_count % 3]
            
            mock_predictions = MagicMock()
            mock_predictions.__getitem__.side_effect = lambda idx: probs
            call_count += 1
            return mock_predictions
        
        mock_analyzer._mock_torch.nn.functional.softmax.side_effect = mock_softmax_batch
        
        texts = [
            "Strong bullish momentum",
            "Major selloff incoming", 
            "Market consolidating"
        ]
        
        results = await mock_analyzer.analyze_batch(texts)
        
        assert len(results) == 3
        assert results[0].overall_sentiment > 0  # Bullish
        assert results[1].overall_sentiment < 0  # Bearish
        assert -0.3 < results[2].overall_sentiment < 0.3  # Neutral
    
    @pytest.mark.asyncio
    async def test_error_handling(self, mock_analyzer):
        """Test error handling"""
        # Test empty text
        with pytest.raises(ValueError, match="Text cannot be empty"):
            await mock_analyzer.analyze("")
        
        # Test None text
        with pytest.raises(ValueError, match="Text cannot be None"):
            await mock_analyzer.analyze(None)
    
    @pytest.mark.asyncio
    async def test_caching(self, mock_transformer_imports):
        """Test result caching"""
        with patch('src.news_trading.sentiment_analysis.transformer_sentiment.AutoTokenizer'), \
             patch('src.news_trading.sentiment_analysis.transformer_sentiment.AutoModelForSequenceClassification'), \
             patch('src.news_trading.sentiment_analysis.transformer_sentiment.torch') as mock_torch:
            
            mock_torch.cuda.is_available.return_value = False
            
            from src.news_trading.sentiment_analysis.transformer_sentiment import TransformerSentiment
            analyzer = TransformerSentiment(enable_cache=True)
            
            # Mock predictions
            probs = np.array([0.1, 0.2, 0.7])
            mock_predictions = MagicMock()
            mock_predictions.__getitem__.side_effect = lambda idx: probs
            mock_torch.nn.functional.softmax.return_value = mock_predictions
            
            text = "Bitcoin price rises"
            
            # First call
            result1 = await analyzer.analyze(text)
            call_count1 = mock_torch.nn.functional.softmax.call_count
            
            # Second call (should be cached)
            result2 = await analyzer.analyze(text)
            call_count2 = mock_torch.nn.functional.softmax.call_count
            
            # Model should not be called again
            assert call_count2 == call_count1
            assert result1.overall_sentiment == result2.overall_sentiment
    
    def test_market_impact_prediction(self, mock_analyzer):
        """Test market impact prediction logic"""
        # Test immediate timeframe
        impact = mock_analyzer._predict_market_impact(
            0.8, 0.9, ["BTC"], "BREAKING: Bitcoin ETF approved!"
        )
        assert impact.timeframe == "immediate"
        assert impact.direction == SentimentDirection.BULLISH
        assert impact.magnitude > 0.7
        
        # Test volatility estimation
        volatility = mock_analyzer._estimate_volatility(0.9, "Market crashes dramatically!")
        assert volatility == "extreme"
    
    def test_catalyst_identification(self, mock_analyzer):
        """Test catalyst identification"""
        text = "SEC approves Bitcoin ETF after institutional fund investment"
        catalysts = mock_analyzer._identify_catalysts(text)
        
        assert "regulatory" in catalysts
        assert "institutional" in catalysts
    
    def test_supported_languages(self, mock_analyzer):
        """Test language support"""
        languages = mock_analyzer.get_supported_languages()
        assert "en" in languages
        
        # Test multilingual model
        mock_analyzer.model_name = "xlm-roberta-base"
        languages = mock_analyzer.get_supported_languages()
        assert len(languages) > 1
        assert "es" in languages