"""
Base classes and interfaces for sentiment analysis
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import asyncio
from datetime import datetime
import logging

from .models import SentimentResult

logger = logging.getLogger(__name__)


class SentimentAnalyzer(ABC):
    """
    Abstract base class for all sentiment analyzers
    
    This interface defines the contract that all sentiment analyzers must follow.
    Implementations can use different approaches (transformers, LLMs, rule-based, etc.)
    but must provide consistent output format.
    """
    
    @abstractmethod
    async def analyze(self, text: str, entities: List[str] = None) -> SentimentResult:
        """
        Analyze sentiment of the given text
        
        Args:
            text: The text to analyze
            entities: Optional list of entities (symbols, companies) mentioned in the text
            
        Returns:
            SentimentResult containing sentiment analysis
        """
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """
        Get the name/identifier of the sentiment model
        
        Returns:
            String identifier for the model
        """
        pass
    
    async def analyze_batch(self, texts: List[str]) -> List[SentimentResult]:
        """
        Analyze multiple texts efficiently
        
        Default implementation runs analyses concurrently.
        Subclasses can override for more efficient batch processing.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of SentimentResult objects
        """
        tasks = [self.analyze(text) for text in texts]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and log them
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error analyzing text {i}: {str(result)}")
            else:
                valid_results.append(result)
        
        return valid_results
    
    def calibrate(self, historical_data: List[Tuple[str, float]]):
        """
        Calibrate model with historical sentiment-outcome pairs
        
        This is an optional method that implementations can use to
        improve accuracy based on historical data.
        
        Args:
            historical_data: List of (text, actual_sentiment) pairs
        """
        logger.info(f"{self.get_model_name()} does not support calibration")
    
    def get_supported_languages(self) -> List[str]:
        """
        Get list of supported languages
        
        Returns:
            List of ISO language codes
        """
        return ["en"]  # Default to English only