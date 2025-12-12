#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualization Adapter

Connects the analyzer connector to the holoviews visualization system,
providing a unified interface for creating market intelligence dashboards.

Created on May 20, 2025
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple, Set, Callable
import time
import os
import datetime
import json

# Import local modules
from cdfa_extensions.analyzer_connector import AnalyzerConnector
from cdfa_extensions.realtime_market_analyzer import MarketRegime, OpportunityScore
from cdfa_extensions.holoviews_visualizer import HoloviewsVisualizer

class VisualizationAdapter:
    """
    Adapter class that connects the analyzer connector to the visualization system.
    
    This class:
    1. Provides data formatting and transformation for visualization
    2. Manages data flow between analyzers and visualizers
    3. Handles caching and optimization for responsive UIs
    4. Provides a simplified interface for creating dashboards
    """
    
    def __init__(self, 
                 analyzer_connector: Optional[AnalyzerConnector] = None,
                 visualizer: Optional[HoloviewsVisualizer] = None,
                 cache_dir: Optional[str] = None,
                 cache_ttl: int = 300,
                 auto_update: bool = True,
                 log_level: str = "INFO"):
        """
        Initialize visualization adapter.
        
        Args:
            analyzer_connector: AnalyzerConnector instance or None to create new
            visualizer: HoloviewsVisualizer instance or None to create new
            cache_dir: Directory for caching visualization data
            cache_ttl: Cache time-to-live in seconds
            auto_update: Whether to automatically update data
            log_level: Logging level
        """
        # Set up logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(getattr(logging, log_level))
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        # Initialize analyzer connector
        self.analyzer_connector = analyzer_connector or AnalyzerConnector(log_level=log_level)
        
        # Initialize visualizer
        self.visualizer = visualizer or HoloviewsVisualizer(
            market_data_fetcher=self, 
            config={"auto_update_interval": cache_ttl if auto_update else None}
        )
        
        # Set up cache
        self.cache_dir = cache_dir or os.path.expanduser("~/cdfa_viz_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_ttl = cache_ttl
        
        # Set up cache dict
        self._cache = {}
        self._cache_timestamps = {}
        
        self.logger.info("VisualizationAdapter initialized")
    
    def get_active_pairs(self) -> List[str]:
        """
        Get list of active trading pairs.
        
        This method is called by the visualizer to get a list of symbols.
        
        Returns:
            List of symbol strings
        """
        return self.analyzer_connector.get_active_pairs()
    
    def get_pair_rankings(self, limit: Optional[int] = None) -> List[Tuple[str, float]]:
        """
        Get ranked list of trading pairs by opportunity score.
        
        This method is called by the visualizer to get ranked pairs.
        
        Args:
            limit: Maximum number of pairs to return
            
        Returns:
            List of (symbol, score) tuples
        """
        return self.analyzer_connector.get_pair_rankings(limit=limit)
    
    def get_pair_metadata(self, symbol: str) -> Dict[str, Any]:
        """
        Get metadata for a specific trading pair.
        
        This method is called by the visualizer to get pair metadata.
        It formats the PairMetadata object to a dict format expected by the visualizer.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Dict with metadata for visualization
        """
        # Get cached metadata if available
        cache_key = f"metadata_{symbol}"
        cached_data = self._get_cached_data(cache_key)
        if cached_data is not None:
            return cached_data
        
        # Get metadata from analyzer connector
        meta = self.analyzer_connector.get_pair_metadata(symbol)
        
        if not meta:
            # Return default metadata
            default_meta = {
                'symbol': symbol,
                'base_currency': symbol.split('-')[0] if '-' in symbol else symbol[:3],
                'quote_currency': symbol.split('-')[1] if '-' in symbol else symbol[3:],
                'price': 0.0,
                'volume_24h': 0.0,
                'change_24h': 0.0,
                'volatility': 0.0,
                'liquidity': 0.0,
                'regime_state': MarketRegime.UNKNOWN.value,
                'regime_confidence': 0.0,
                'opportunity_score': OpportunityScore.UNKNOWN.value,
                'is_trending': False,
                'trend_strength': 0.0,
                'cycle_strength': 0.0,
                'success_rate': 0.5,
                'analyzer_scores': {},
                'detector_signals': {}
            }
            return default_meta
        
        # Convert PairMetadata object to dict
        viz_meta = {
            'symbol': meta.symbol,
            'base_currency': meta.base_currency,
            'quote_currency': meta.quote_currency,
            'exchange': meta.exchange,
            'price': meta.price,
            'volume_24h': meta.volume_24h,
            'change_24h': meta.change_24h,
            'volatility': meta.volatility,
            'liquidity': meta.liquidity,
            'regime_state': meta.regime_state.value if hasattr(meta.regime_state, 'value') else meta.regime_state,
            'regime_confidence': meta.regime_confidence,
            'opportunity_score': meta.opportunity_score.value if hasattr(meta.opportunity_score, 'value') else meta.opportunity_score,
            'is_trending': meta.is_trending,
            'trend_strength': meta.trend_strength,
            'cycle_strength': meta.cycle_strength,
            'success_rate': meta.success_rate,
            'analyzer_scores': dict(meta.analyzer_scores),
            'detector_signals': dict(meta.detector_signals)
        }
        
        # Cache metadata
        self._set_cached_data(cache_key, viz_meta)
        
        return viz_meta
    
    def fetch_data(self, symbols: List[str], timeframe: str = "1d", 
                 lookback: str = "30d") -> Dict[str, pd.DataFrame]:
        """
        Fetch historical data for the specified symbols.
        
        This method is called by the visualizer to get historical data.
        
        Args:
            symbols: List of symbols to fetch data for
            timeframe: Timeframe for data (e.g., "1h", "4h", "1d")
            lookback: Lookback period (e.g., "30d", "90d")
            
        Returns:
            Dictionary of symbol -> DataFrame
        """
        # Get cached data if available
        cache_key = f"ohlcv_{'-'.join(symbols)}_{timeframe}_{lookback}"
        cached_data = self._get_cached_data(cache_key)
        if cached_data is not None:
            return cached_data
        
        # Fetch data from analyzer connector
        data_dict = self.analyzer_connector.fetch_data(symbols, timeframe, lookback)
        
        # Cache data
        self._set_cached_data(cache_key, data_dict)
        
        return data_dict
    
    def query_narrative_forecaster(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Query the narrative forecaster for sentiment analysis.
        
        This method is called by the SentimentAnalyzer.
        It's a placeholder that could be connected to a real NLP service.
        
        Args:
            text: Text to analyze
            context: Optional context information
            
        Returns:
            Dictionary with sentiment analysis results
        """
        # This is a placeholder implementation - in a real system, this would connect to a NLP service
        # For now, we'll generate a simple sentiment score based on keyword matching
        
        positive_words = ['bullish', 'buy', 'up', 'growth', 'profit', 'gain', 'positive', 'increase', 'higher']
        negative_words = ['bearish', 'sell', 'down', 'loss', 'crash', 'negative', 'decrease', 'lower', 'risk']
        
        # Simple word counting
        positive_count = sum(word in text.lower() for word in positive_words)
        negative_count = sum(word in text.lower() for word in negative_words)
        
        total_count = positive_count + negative_count
        if total_count > 0:
            sentiment = (positive_count - negative_count) / total_count
        else:
            sentiment = 0.0
        
        # Create result
        result = {
            'result': {
                'sentiment': sentiment,
                'confidence': 0.5,
                'topics': [],
                'entities': []
            }
        }
        
        # If context provides a symbol, try to enhance with market data
        if context and 'symbol' in context:
            symbol = context['symbol']
            meta = self.analyzer_connector.get_pair_metadata(symbol)
            if meta:
                if hasattr(meta.regime_state, 'value'):
                    result['result']['market_regime'] = meta.regime_state.value
                else:
                    result['result']['market_regime'] = meta.regime_state
                
                result['result']['volatility'] = meta.volatility
                result['result']['trend_strength'] = meta.trend_strength
        
        return result
    
    def _get_cached_data(self, cache_key: str) -> Optional[Any]:
        """
        Get data from cache if available and not expired.
        
        Args:
            cache_key: Cache key
            
        Returns:
            Cached data or None if not available
        """
        # Check in-memory cache
        if cache_key in self._cache:
            timestamp = self._cache_timestamps.get(cache_key, 0)
            if time.time() - timestamp <= self.cache_ttl:
                return self._cache[cache_key]
        
        # Check disk cache
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
        if os.path.exists(cache_file):
            # Check if cache is expired
            if time.time() - os.path.getmtime(cache_file) <= self.cache_ttl:
                try:
                    # Load from disk
                    data = pd.read_pickle(cache_file)
                    
                    # Update in-memory cache
                    self._cache[cache_key] = data
                    self._cache_timestamps[cache_key] = time.time()
                    
                    return data
                except Exception:
                    pass
        
        return None
    
    def _set_cached_data(self, cache_key: str, data: Any):
        """
        Store data in cache.
        
        Args:
            cache_key: Cache key
            data: Data to cache
        """
        # Update in-memory cache
        self._cache[cache_key] = data
        self._cache_timestamps[cache_key] = time.time()
        
        # Update disk cache
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
        try:
            # Save to disk - use different method based on data type
            if isinstance(data, pd.DataFrame) or isinstance(data, Dict) and any(isinstance(v, pd.DataFrame) for v in data.values()):
                pd.to_pickle(data, cache_file)
            else:
                with open(cache_file, 'wb') as f:
                    import pickle
                    pickle.dump(data, f)
                    
        except Exception as e:
            self.logger.warning(f"Error caching data: {e}")
    
    def update_all(self):
        """Update all data and visualizations"""
        self.analyzer_connector.update_all()
        
        # Clear cache
        self._cache.clear()
        self._cache_timestamps.clear()
        
        # Clear disk cache by removing all files older than cache_ttl
        try:
            current_time = time.time()
            for filename in os.listdir(self.cache_dir):
                file_path = os.path.join(self.cache_dir, filename)
                if os.path.isfile(file_path) and current_time - os.path.getmtime(file_path) > self.cache_ttl:
                    os.remove(file_path)
        except Exception as e:
            self.logger.warning(f"Error cleaning cache directory: {e}")
    
    def create_dashboard(self, width=None, height=None):
        """
        Create a complete dashboard with multiple visualizations.
        
        Args:
            width: Width of each component
            height: Height of each component
            
        Returns:
            Holoviews layout or Panel dashboard
        """
        return self.visualizer.create_dashboard(width=width, height=height)
    
    def serve_dashboard(self, port=5006):
        """
        Serve the dashboard using Panel server.
        
        Args:
            port: Server port
            
        Returns:
            Server instance
        """
        return self.visualizer.serve_dashboard(port=port)
    
    def save_dashboard(self, filename="cdfa_dashboard.html", title="CDFA Market Intelligence Dashboard"):
        """
        Save dashboard to HTML file.
        
        Args:
            filename: Output filename
            title: Dashboard title
            
        Returns:
            Path to saved file
        """
        return self.visualizer.save_dashboard(filename=filename, title=title)
    
    def create_tradingview_dashboard(self, symbols=None, timeframe="1d"):
        """
        Create a TradingView-style dashboard with multiple charts.
        
        Args:
            symbols: List of symbols to display (None for top pairs)
            timeframe: Timeframe for charts
            
        Returns:
            Panel dashboard
        """
        return self.visualizer.create_tradingview_dashboard(symbols=symbols, timeframe=timeframe)
    
    def create_market_dashboard(self):
        """
        Create a comprehensive market dashboard with both heatmaps and TradingView charts.
        
        Returns:
            Panel dashboard
        """
        return self.visualizer.create_market_dashboard()
    
    def stop(self):
        """Stop background threads and clean up resources"""
        if hasattr(self, 'visualizer') and self.visualizer:
            self.visualizer.stop()
        
        if hasattr(self, 'analyzer_connector') and self.analyzer_connector:
            self.analyzer_connector.stop()
        
        self.logger.info("VisualizationAdapter stopped")