#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  6 15:44:28 2025

@author: ashina
"""

# cdfa_extensions/sentiment_analyzer.py
import logging
from typing import Dict, List, Optional, Union, Any
import pandas as pd
import numpy as np
import datetime
import re

class SentimentAnalyzer:
    """
    Sentiment analysis module for financial text data.
    Integrates with Pulsar's Narrative Forecaster for advanced NLP analysis.
    """
    def __init__(self, pulsar_connector=None, config: Optional[Dict[str, Any]] = None):
        self.logger = logging.getLogger(f"{__name__}.SentimentAnalyzer")
        
        # Default configuration
        self.default_config = {
            "use_pulsar": pulsar_connector is not None,
            "use_vader": True,
            "use_textblob": False,
            "cache_ttl": 3600,  # 1 hour
            "min_text_length": 20
        }
        
        # Merge with provided config
        self.config = {**self.default_config, **(config or {})}
        
        # Store Pulsar connector reference
        self.pulsar_connector = pulsar_connector
        
        # Initialize sentiment analysis tools
        self._initialize_tools()
        
        # Cache for sentiment results
        self._sentiment_cache = {}
        
    def _initialize_tools(self):
        """Initialize sentiment analysis tools based on configuration."""
        # VADER Sentiment Analysis
        if self.config["use_vader"]:
            try:
                from nltk.sentiment.vader import SentimentIntensityAnalyzer
                self.vader = SentimentIntensityAnalyzer()
                self.logger.info("VADER sentiment analyzer initialized")
            except ImportError:
                self.logger.warning("NLTK VADER not available. Install with 'pip install nltk' and run nltk.download('vader_lexicon')")
                self.config["use_vader"] = False
                
        # TextBlob
        if self.config["use_textblob"]:
            try:
                from textblob import TextBlob
                self.TextBlob = TextBlob
                self.logger.info("TextBlob initialized")
            except ImportError:
                self.logger.warning("TextBlob not available. Install with 'pip install textblob'")
                self.config["use_textblob"] = False
    
    def analyze_text(self, text: str, use_pulsar: Optional[bool] = None) -> Dict[str, Any]:
        """
        Analyze sentiment of a text string.
        
        Args:
            text: Text to analyze
            use_pulsar: Whether to use Pulsar's NLP capabilities (overrides config)
            
        Returns:
            Dictionary of sentiment metrics
        """
        # Skip short text
        if len(text) < self.config["min_text_length"]:
            return {"error": "Text too short for sentiment analysis"}
            
        # Determine if we should use Pulsar
        use_pulsar = use_pulsar if use_pulsar is not None else self.config["use_pulsar"]
        
        results = {}
        
        # Use Pulsar's Narrative Forecaster for advanced NLP
        if use_pulsar and self.pulsar_connector:
            try:
                pulsar_result = self.pulsar_connector.query_narrative_forecaster(text)
                if pulsar_result and "result" in pulsar_result:
                    # Extract sentiment from Pulsar result
                    results["pulsar"] = {
                        "sentiment": pulsar_result["result"].get("sentiment", 0),
                        "confidence": pulsar_result["result"].get("confidence", 0),
                        "topics": pulsar_result["result"].get("topics", []),
                        "entities": pulsar_result["result"].get("entities", [])
                    }
            except Exception as e:
                self.logger.error(f"Error using Pulsar for sentiment analysis: {e}")
                
        # Use VADER for sentiment analysis
        if self.config["use_vader"]:
            try:
                vader_scores = self.vader.polarity_scores(text)
                results["vader"] = {
                    "compound": vader_scores["compound"],
                    "positive": vader_scores["pos"],
                    "negative": vader_scores["neg"],
                    "neutral": vader_scores["neu"]
                }
            except Exception as e:
                self.logger.error(f"Error using VADER for sentiment analysis: {e}")
                
        # Use TextBlob for sentiment analysis
        if self.config["use_textblob"]:
            try:
                blob = self.TextBlob(text)
                polarity = blob.sentiment.polarity
                subjectivity = blob.sentiment.subjectivity
                
                results["textblob"] = {
                    "polarity": polarity,
                    "subjectivity": subjectivity
                }
            except Exception as e:
                self.logger.error(f"Error using TextBlob for sentiment analysis: {e}")
                
        # Calculate combined sentiment
        if results:
            # Prioritize Pulsar if available
            if "pulsar" in results:
                combined = results["pulsar"]["sentiment"]
            # Otherwise average available methods
            else:
                scores = []
                if "vader" in results:
                    scores.append(results["vader"]["compound"])
                if "textblob" in results:
                    scores.append(results["textblob"]["polarity"])
                    
                combined = sum(scores) / len(scores) if scores else 0
                
            results["combined"] = combined
            
        return results
        
    def analyze_news_sentiment(self, symbol: str, limit: int = 10) -> Dict[str, Any]:
        """
        Analyze sentiment of recent news for a given symbol.
        
        Args:
            symbol: Stock/crypto symbol
            limit: Maximum number of news items to analyze
            
        Returns:
            Sentiment analysis of recent news
        """
        try:
            # This would need to be connected to a news API
            # For now, let's assume we're using Pulsar to get news
            if self.pulsar_connector:
                news_data = self.pulsar_connector.query_narrative_forecaster(
                    f"Get recent news about {symbol}",
                    context={"type": "market_news", "symbol": symbol, "limit": limit}
                )
                
                if news_data and "news_items" in news_data:
                    news_items = news_data["news_items"]
                    sentiments = []
                    
                    for item in news_items:
                        title = item.get("title", "")
                        summary = item.get("summary", "")
                        
                        # Analyze combined text
                        text = f"{title}. {summary}"
                        sentiment = self.analyze_text(text)
                        
                        sentiments.append({
                            "title": title,
                            "date": item.get("date"),
                            "source": item.get("source"),
                            "sentiment": sentiment.get("combined", 0),
                            "details": sentiment
                        })
                    
                    # Calculate aggregate sentiment
                    if sentiments:
                        avg_sentiment = sum(item["sentiment"] for item in sentiments) / len(sentiments)
                        
                        return {
                            "symbol": symbol,
                            "average_sentiment": avg_sentiment,
                            "news_count": len(sentiments),
                            "news_items": sentiments,
                            "timestamp": datetime.datetime.now().isoformat()
                        }
            
            return {"error": "No news data available or Pulsar connector not configured"}
            
        except Exception as e:
            self.logger.error(f"Error analyzing news sentiment: {e}")
            return {"error": str(e)}