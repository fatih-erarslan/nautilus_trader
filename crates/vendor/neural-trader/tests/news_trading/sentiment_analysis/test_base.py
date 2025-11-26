"""
Tests for base sentiment analyzer interface
"""
import pytest
from abc import ABC
from typing import List, Dict, Tuple
import asyncio
from datetime import datetime

from src.news_trading.sentiment_analysis.base import SentimentAnalyzer
from src.news_trading.sentiment_analysis.models import SentimentResult


class TestSentimentAnalyzerInterface:
    """Test the SentimentAnalyzer abstract interface"""
    
    def test_sentiment_analyzer_is_abstract(self):
        """Test that SentimentAnalyzer is abstract and cannot be instantiated"""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            SentimentAnalyzer()
    
    def test_sentiment_analyzer_requires_abstract_methods(self):
        """Test that subclasses must implement abstract methods"""
        
        class IncompleteSentimentAnalyzer(SentimentAnalyzer):
            """Incomplete implementation missing abstract methods"""
            pass
        
        # Should fail - abstract methods not implemented
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteSentimentAnalyzer()
    
    def test_complete_sentiment_analyzer_implementation(self):
        """Test that a complete implementation can be instantiated"""
        
        class CompleteSentimentAnalyzer(SentimentAnalyzer):
            """Complete implementation of SentimentAnalyzer"""
            
            async def analyze(self, text: str, entities: List[str] = None) -> SentimentResult:
                """Analyze sentiment of text"""
                return None  # Dummy implementation
            
            def get_model_name(self) -> str:
                """Get name of the sentiment model"""
                return "test-model"
        
        # Should succeed
        analyzer = CompleteSentimentAnalyzer()
        assert analyzer is not None
        assert analyzer.get_model_name() == "test-model"
    
    @pytest.mark.asyncio
    async def test_analyze_batch_default_implementation(self):
        """Test default batch analysis implementation"""
        
        class TestAnalyzer(SentimentAnalyzer):
            """Test analyzer with simple implementation"""
            
            async def analyze(self, text: str, entities: List[str] = None) -> SentimentResult:
                """Simple sentiment analysis"""
                from src.news_trading.sentiment_analysis.models import (
                    SentimentResult, MarketImpact, SentimentDirection, SentimentBreakdown
                )
                
                return SentimentResult(
                    article_id=f"test-{hash(text)}",
                    overall_sentiment=0.5,
                    confidence=0.8,
                    market_impact=MarketImpact(
                        direction=SentimentDirection.NEUTRAL,
                        magnitude=0.5,
                        timeframe="short-term",
                        affected_assets=entities or []
                    ),
                    sentiment_breakdown=SentimentBreakdown(),
                    reasoning="Test analysis"
                )
            
            def get_model_name(self) -> str:
                return "test-analyzer"
        
        analyzer = TestAnalyzer()
        texts = ["Text 1", "Text 2", "Text 3"]
        
        results = await analyzer.analyze_batch(texts)
        
        assert len(results) == 3
        assert all(isinstance(r, SentimentResult) for r in results)
        assert results[0].article_id != results[1].article_id
    
    def test_calibrate_method(self):
        """Test calibration method interface"""
        
        class CalibratedAnalyzer(SentimentAnalyzer):
            """Analyzer with calibration support"""
            
            def __init__(self):
                self.calibration_data = []
            
            async def analyze(self, text: str, entities: List[str] = None) -> SentimentResult:
                """Dummy analyze method"""
                pass
            
            def get_model_name(self) -> str:
                return "calibrated-analyzer"
            
            def calibrate(self, historical_data: List[Tuple[str, float]]):
                """Store calibration data"""
                self.calibration_data = historical_data
        
        analyzer = CalibratedAnalyzer()
        
        # Test calibration
        historical_data = [
            ("Bullish news", 0.8),
            ("Bearish news", -0.7),
            ("Neutral news", 0.1)
        ]
        
        analyzer.calibrate(historical_data)
        assert len(analyzer.calibration_data) == 3
        assert analyzer.calibration_data[0] == ("Bullish news", 0.8)
    
    @pytest.mark.asyncio
    async def test_concurrent_analysis(self):
        """Test concurrent analysis capabilities"""
        
        class ConcurrentAnalyzer(SentimentAnalyzer):
            """Analyzer that simulates processing delay"""
            
            async def analyze(self, text: str, entities: List[str] = None) -> SentimentResult:
                """Simulate processing with delay"""
                from src.news_trading.sentiment_analysis.models import (
                    SentimentResult, MarketImpact, SentimentDirection, SentimentBreakdown
                )
                
                # Simulate processing time
                await asyncio.sleep(0.1)
                
                return SentimentResult(
                    article_id=f"concurrent-{hash(text)}",
                    overall_sentiment=min(0.9, len(text) / 200),  # Simple sentiment based on length, capped at 0.9
                    confidence=0.8,
                    market_impact=MarketImpact(
                        direction=SentimentDirection.NEUTRAL,
                        magnitude=0.5,
                        timeframe="short-term",
                        affected_assets=[]
                    ),
                    sentiment_breakdown=SentimentBreakdown(),
                    reasoning="Concurrent analysis test"
                )
            
            def get_model_name(self) -> str:
                return "concurrent-analyzer"
        
        analyzer = ConcurrentAnalyzer()
        texts = [f"Text {i}" * (i + 1) * 10 for i in range(5)]
        
        # Measure time for concurrent batch processing
        start_time = datetime.now()
        results = await analyzer.analyze_batch(texts)
        elapsed_time = (datetime.now() - start_time).total_seconds()
        
        # Should complete in less than 0.5 seconds (concurrent, not sequential)
        assert elapsed_time < 0.5
        assert len(results) == 5
        
        # Verify different sentiments based on text length
        assert results[0].overall_sentiment != results[4].overall_sentiment
    
    def test_get_supported_languages(self):
        """Test language support interface"""
        
        class MultilingualAnalyzer(SentimentAnalyzer):
            """Analyzer with language support"""
            
            async def analyze(self, text: str, entities: List[str] = None) -> SentimentResult:
                pass
            
            def get_model_name(self) -> str:
                return "multilingual-analyzer"
            
            def get_supported_languages(self) -> List[str]:
                """Return supported languages"""
                return ["en", "es", "fr", "de", "zh"]
        
        analyzer = MultilingualAnalyzer()
        languages = analyzer.get_supported_languages()
        
        assert "en" in languages
        assert len(languages) == 5