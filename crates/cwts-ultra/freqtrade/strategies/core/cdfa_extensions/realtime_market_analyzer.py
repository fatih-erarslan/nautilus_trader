#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real-time Market Data Analyzer

Connects to cryptocurrency exchange APIs and processes real-time market data
for visualization and analysis in the CDFA system.

Created on May 20, 2025
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple, Set, Callable
from dataclasses import dataclass, field
import time
import datetime
import threading
import json
import os
from enum import Enum
from collections import defaultdict

# Optional imports - handle gracefully if not available
try:
    import ccxt
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False
    logging.warning("CCXT not available. Install with 'pip install ccxt'")

# Type definitions
class MarketRegime(str, Enum):
    """Market regime classification"""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    RANGING = "ranging"
    VOLATILE = "volatile"
    ACCUMULATION = "accumulation"
    DISTRIBUTION = "distribution"
    UNKNOWN = "unknown"

class OpportunityScore(str, Enum):
    """Opportunity score classification"""
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    NEUTRAL = "neutral"
    SELL = "sell"
    STRONG_SELL = "strong_sell"
    UNKNOWN = "unknown"

@dataclass
class PairMetadata:
    """Trading pair metadata container"""
    symbol: str
    base_currency: str = ""
    quote_currency: str = ""
    exchange: str = ""
    last_updated: float = 0.0
    
    # Market metrics
    price: float = 0.0
    volume_24h: float = 0.0
    change_24h: float = 0.0
    volatility: float = 0.0
    liquidity: float = 0.0
    
    # Analysis results
    regime_state: MarketRegime = MarketRegime.UNKNOWN
    regime_confidence: float = 0.0
    opportunity_score: OpportunityScore = OpportunityScore.UNKNOWN
    is_trending: bool = False
    trend_strength: float = 0.0
    cycle_strength: float = 0.0
    success_rate: float = 0.5
    
    # Analyzer scores
    analyzer_scores: Dict[str, float] = field(default_factory=dict)
    
    # Detector signals
    detector_signals: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)

class RealtimeMarketAnalyzer:
    """
    Real-time market data analyzer for cryptocurrency markets.
    
    Connects to exchange APIs, processes market data, and provides analysis
    for visualization in the CDFA frontend.
    """
    
    def __init__(self, 
                 config: Optional[Dict[str, Any]] = None,
                 cache_dir: Optional[str] = None,
                 analyzers: Optional[Dict[str, Any]] = None,
                 log_level: str = "INFO"):
        """
        Initialize the real-time market data analyzer.
        
        Args:
            config: Configuration dictionary
            cache_dir: Directory for caching data
            analyzers: Dictionary of analyzer instances
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
        
        # Default configuration
        self.default_config = {
            "exchange": "binance",
            "api_key": "",
            "api_secret": "",
            "base_currencies": ["BTC", "ETH", "SOL", "XRP", "ADA"],
            "quote_currencies": ["USDT", "USD", "BUSD", "USDC"],
            "timeframes": ["1h", "4h", "1d"],
            "default_timeframe": "1d",
            "update_interval": 300,  # 5 minutes
            "cache_ttl": 3600,  # 1 hour
            "max_pairs": 100,
            "auto_update": True,
            "use_testnet": False
        }
        
        # Merge with provided config
        self.config = {**self.default_config}
        if config:
            self.config.update({k: v for k, v in config.items() if k in self.default_config})
        
        # Set up cache directory
        self.cache_dir = cache_dir or os.path.expanduser("~/cdfa_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize exchange connection
        self.exchange = None
        self.exchange_markets = {}
        self._initialize_exchange()
        
        # Set up analyzers
        self.analyzers = {}
        if analyzers:
            self.analyzers.update(analyzers)
        
        # Set up data storage
        self.pair_metadata = {}  # Symbol -> PairMetadata
        self.pair_data = {}  # Symbol -> DataFrame of OHLCV
        self.analysis_results = {}  # Symbol -> Analysis results
        
        # Set up update thread
        self.running = True
        self._update_thread = None
        if self.config["auto_update"]:
            self._start_update_thread()
        
        self.logger.info(f"RealtimeMarketAnalyzer initialized with exchange: {self.config['exchange']}")
    
    def _initialize_exchange(self):
        """Initialize connection to cryptocurrency exchange"""
        if not CCXT_AVAILABLE:
            self.logger.error("CCXT is required for exchange connection")
            return
        
        try:
            exchange_id = self.config["exchange"]
            exchange_class = getattr(ccxt, exchange_id)
            
            # Create exchange instance
            exchange_params = {
                'enableRateLimit': True,
            }
            
            # Add API credentials if available
            if self.config["api_key"] and self.config["api_secret"]:
                exchange_params.update({
                    'apiKey': self.config["api_key"],
                    'secret': self.config["api_secret"]
                })
            
            # Use testnet if configured
            if self.config["use_testnet"]:
                if hasattr(exchange_class, 'testnet'):
                    exchange_params['testnet'] = True
            
            # Create exchange instance
            self.exchange = exchange_class(exchange_params)
            
            # Load markets
            self.exchange_markets = self.exchange.load_markets()
            
            self.logger.info(f"Connected to {exchange_id} exchange with {len(self.exchange_markets)} markets")
            
        except Exception as e:
            self.logger.error(f"Error initializing exchange: {e}")
            self.exchange = None
    
    def _start_update_thread(self):
        """Start background thread for automatic updates"""
        def update_loop():
            """Background thread for updating market data"""
            while self.running:
                try:
                    self.update_market_data()
                    
                    # Sleep until next update
                    for _ in range(self.config["update_interval"]):
                        if not self.running:
                            break
                        time.sleep(1)
                        
                except Exception as e:
                    self.logger.error(f"Error in update loop: {e}")
                    time.sleep(10)  # Sleep longer on error
        
        # Start thread
        self._update_thread = threading.Thread(
            target=update_loop,
            daemon=True,
            name="MarketAnalyzer-Update"
        )
        self._update_thread.start()
        
        self.logger.info("Market data update thread started")
    
    def update_market_data(self):
        """Update market data for all tracked pairs"""
        if not self.exchange:
            self.logger.error("Exchange not initialized")
            return False
        
        try:
            # Reload markets if needed
            if not self.exchange_markets:
                self.exchange_markets = self.exchange.load_markets()
            
            # Get all active pairs based on configured base/quote currencies
            active_pairs = []
            for symbol, market in self.exchange_markets.items():
                # Skip non-spot markets
                if market.get('futures') or market.get('swap') or market.get('option'):
                    continue
                
                # Extract base and quote currencies
                base = market.get('base')
                quote = market.get('quote')
                
                if (base in self.config["base_currencies"] and 
                    quote in self.config["quote_currencies"]):
                    active_pairs.append(symbol)
            
            # Limit to configured maximum
            active_pairs = active_pairs[:self.config["max_pairs"]]
            
            # Update metadata for each pair
            updated_count = 0
            for symbol in active_pairs:
                try:
                    # Skip if updated recently (within last 10 seconds)
                    if (symbol in self.pair_metadata and 
                        time.time() - self.pair_metadata[symbol].last_updated < 10):
                        continue
                    
                    # Get ticker data
                    ticker = self.exchange.fetch_ticker(symbol)
                    
                    # Create or update metadata
                    if symbol not in self.pair_metadata:
                        market = self.exchange_markets[symbol]
                        self.pair_metadata[symbol] = PairMetadata(
                            symbol=symbol,
                            base_currency=market.get('base', ''),
                            quote_currency=market.get('quote', ''),
                            exchange=self.config["exchange"]
                        )
                    
                    # Update metadata with ticker data
                    meta = self.pair_metadata[symbol]
                    meta.price = ticker.get('last', 0.0)
                    meta.volume_24h = ticker.get('baseVolume', 0.0)
                    meta.change_24h = ticker.get('percentage', 0.0)
                    meta.last_updated = time.time()
                    
                    updated_count += 1
                    
                except Exception as e:
                    self.logger.warning(f"Error updating {symbol}: {e}")
            
            self.logger.info(f"Updated {updated_count} pairs")
            
            # Fetch OHLCV data for analysis
            self._update_ohlcv_data(active_pairs)
            
            # Run analysis on updated data
            self._run_analysis(active_pairs)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating market data: {e}")
            return False
    
    def _update_ohlcv_data(self, symbols: List[str]):
        """
        Update OHLCV data for the specified symbols.
        
        Args:
            symbols: List of symbols to update
        """
        if not self.exchange:
            return
        
        # Get default timeframe
        timeframe = self.config["default_timeframe"]
        
        for symbol in symbols:
            try:
                # Check if we need to update
                if symbol in self.pair_data:
                    # Get last update time
                    last_update = self.pair_data[symbol].index[-1]
                    if isinstance(last_update, pd.Timestamp):
                        last_update = last_update.timestamp()
                    
                    # Skip if updated recently (within OHLCV interval)
                    now = time.time()
                    if now - last_update < self._get_timeframe_seconds(timeframe):
                        continue
                
                # Fetch OHLCV data
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe)
                
                if not ohlcv:
                    continue
                
                # Convert to DataFrame
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
                # Store data
                self.pair_data[symbol] = df
                
                # Update volatility in metadata
                if symbol in self.pair_metadata and len(df) >= 20:
                    # Calculate volatility as standard deviation of returns
                    returns = df['close'].pct_change().dropna()
                    volatility = returns.std() * np.sqrt(252)  # Annualized
                    
                    self.pair_metadata[symbol].volatility = volatility
                    
                    # Calculate liquidity as volume * price
                    avg_volume = df['volume'].mean()
                    last_price = df['close'].iloc[-1]
                    liquidity = avg_volume * last_price
                    
                    self.pair_metadata[symbol].liquidity = liquidity
                
            except Exception as e:
                self.logger.warning(f"Error fetching OHLCV for {symbol}: {e}")
    
    def _get_timeframe_seconds(self, timeframe: str) -> int:
        """
        Convert timeframe string to seconds.
        
        Args:
            timeframe: Timeframe string (e.g., "1h", "4h", "1d")
            
        Returns:
            Seconds in the timeframe
        """
        # Parse timeframe
        amount = int(timeframe[:-1])
        unit = timeframe[-1]
        
        # Convert to seconds
        if unit == 'm':
            return amount * 60
        elif unit == 'h':
            return amount * 60 * 60
        elif unit == 'd':
            return amount * 24 * 60 * 60
        elif unit == 'w':
            return amount * 7 * 24 * 60 * 60
        else:
            return 86400  # Default to 1 day
    
    def _run_analysis(self, symbols: List[str]):
        """
        Run analysis on the updated market data.
        
        Args:
            symbols: List of symbols to analyze
        """
        # Skip if no analyzers available
        if not self.analyzers:
            return
        
        for symbol in symbols:
            # Skip if no data available
            if symbol not in self.pair_data or symbol not in self.pair_metadata:
                continue
            
            df = self.pair_data[symbol]
            meta = self.pair_metadata[symbol]
            
            # Create metadata for analysis context
            metadata = {
                'symbol': symbol,
                'exchange': self.config["exchange"],
                'base': meta.base_currency,
                'quote': meta.quote_currency
            }
            
            # Run each analyzer
            for analyzer_name, analyzer in self.analyzers.items():
                try:
                    # Check if analyzer has the required method
                    if not hasattr(analyzer, 'analyze'):
                        continue
                    
                    # Run analysis
                    result = analyzer.analyze(df, metadata)
                    
                    # Store result
                    if result is not None:
                        # Update analyzer score
                        if 'score' in result:
                            meta.analyzer_scores[analyzer_name] = result['score']
                        
                        # Update signals if available
                        if 'signals' in result:
                            meta.detector_signals[analyzer_name] = result['signals']
                        
                        # Update regime if available
                        if 'regime' in result:
                            meta.regime_state = result['regime']
                            meta.regime_confidence = result.get('confidence', 0.5)
                        
                        # Update opportunity score if available
                        if 'opportunity' in result:
                            meta.opportunity_score = result['opportunity']
                        
                        # Update trend information if available
                        if 'is_trending' in result:
                            meta.is_trending = result['is_trending']
                            meta.trend_strength = result.get('trend_strength', 0.0)
                        
                        # Update cycle information if available
                        if 'cycle_strength' in result:
                            meta.cycle_strength = result['cycle_strength']
                        
                        # Update success rate if available
                        if 'success_rate' in result:
                            meta.success_rate = result['success_rate']
                    
                except Exception as e:
                    self.logger.warning(f"Error running {analyzer_name} analyzer on {symbol}: {e}")
    
    def get_active_pairs(self) -> List[str]:
        """
        Get list of active trading pairs.
        
        Returns:
            List of symbol strings
        """
        return list(self.pair_metadata.keys())
    
    def get_pair_rankings(self, limit: Optional[int] = None) -> List[Tuple[str, float]]:
        """
        Get ranked list of trading pairs by opportunity score.
        
        Args:
            limit: Maximum number of pairs to return
            
        Returns:
            List of (symbol, score) tuples
        """
        # Convert opportunity score to numeric value
        opportunity_map = {
            OpportunityScore.STRONG_BUY: 1.0,
            OpportunityScore.BUY: 0.75,
            OpportunityScore.NEUTRAL: 0.5,
            OpportunityScore.SELL: 0.25,
            OpportunityScore.STRONG_SELL: 0.0,
            OpportunityScore.UNKNOWN: 0.5
        }
        
        # Create list of (symbol, score) tuples
        rankings = []
        for symbol, meta in self.pair_metadata.items():
            if hasattr(meta.opportunity_score, 'value'):
                opp_score = opportunity_map.get(meta.opportunity_score.value, 0.5)
            else:
                opp_score = opportunity_map.get(meta.opportunity_score, 0.5)
            
            # Combine with analyzer scores for overall ranking
            if meta.analyzer_scores:
                avg_analyzer_score = sum(meta.analyzer_scores.values()) / len(meta.analyzer_scores)
                combined_score = (opp_score + avg_analyzer_score) / 2
            else:
                combined_score = opp_score
            
            rankings.append((symbol, combined_score))
        
        # Sort by score (descending)
        rankings.sort(key=lambda x: x[1], reverse=True)
        
        # Apply limit if specified
        if limit:
            rankings = rankings[:limit]
        
        return rankings
    
    def get_pair_metadata(self, symbol: str) -> Optional[PairMetadata]:
        """
        Get metadata for a specific trading pair.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            PairMetadata object or None if not found
        """
        return self.pair_metadata.get(symbol)
    
    def fetch_data(self, symbols: List[str], timeframe: str = "1d", 
                lookback: str = "30d") -> Dict[str, pd.DataFrame]:
        """
        Fetch historical data for the specified symbols.
        
        Args:
            symbols: List of symbols to fetch data for
            timeframe: Timeframe for data (e.g., "1h", "4h", "1d")
            lookback: Lookback period (e.g., "30d", "90d")
            
        Returns:
            Dictionary of symbol -> DataFrame
        """
        if not self.exchange:
            return {}
        
        # Parse lookback period
        lookback_days = int(lookback[:-1]) if lookback.endswith('d') else 30
        
        # Calculate number of candles needed
        timeframe_days = self._timeframe_to_days(timeframe)
        limit = min(1000, int(lookback_days / timeframe_days) + 10)  # Add buffer
        
        result = {}
        for symbol in symbols:
            try:
                # Try to get from cache first
                cache_key = f"{symbol}_{timeframe}_{lookback}"
                cached_data = self._get_cached_data(cache_key)
                
                if cached_data is not None:
                    result[symbol] = cached_data
                    continue
                
                # Fetch from exchange
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
                
                if not ohlcv:
                    continue
                
                # Convert to DataFrame
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
                # Cache and return
                self._set_cached_data(cache_key, df)
                result[symbol] = df
                
            except Exception as e:
                self.logger.warning(f"Error fetching data for {symbol}: {e}")
        
        return result
    
    def _timeframe_to_days(self, timeframe: str) -> float:
        """
        Convert timeframe string to days.
        
        Args:
            timeframe: Timeframe string (e.g., "1h", "4h", "1d")
            
        Returns:
            Days in the timeframe
        """
        seconds = self._get_timeframe_seconds(timeframe)
        return seconds / 86400.0  # Seconds per day
    
    def _get_cached_data(self, cache_key: str) -> Optional[pd.DataFrame]:
        """
        Get data from cache if available and not expired.
        
        Args:
            cache_key: Cache key
            
        Returns:
            DataFrame or None if not available
        """
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
        if os.path.exists(cache_file):
            # Check if cache is expired
            if time.time() - os.path.getmtime(cache_file) <= self.config["cache_ttl"]:
                try:
                    return pd.read_pickle(cache_file)
                except Exception:
                    pass
        
        return None
    
    def _set_cached_data(self, cache_key: str, data: pd.DataFrame):
        """
        Store data in cache.
        
        Args:
            cache_key: Cache key
            data: DataFrame to cache
        """
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
        try:
            data.to_pickle(cache_file)
        except Exception as e:
            self.logger.warning(f"Error caching data: {e}")
    
    def add_analyzer(self, name: str, analyzer: Any):
        """
        Add an analyzer to the system.
        
        Args:
            name: Analyzer name
            analyzer: Analyzer instance
        """
        self.analyzers[name] = analyzer
        self.logger.info(f"Added analyzer: {name}")
    
    def remove_analyzer(self, name: str):
        """
        Remove an analyzer from the system.
        
        Args:
            name: Analyzer name to remove
        """
        if name in self.analyzers:
            del self.analyzers[name]
            self.logger.info(f"Removed analyzer: {name}")
    
    def get_analyzer_names(self) -> List[str]:
        """
        Get list of available analyzers.
        
        Returns:
            List of analyzer names
        """
        return list(self.analyzers.keys())
    
    def stop(self):
        """Stop background threads and clean up resources."""
        self.logger.info("Stopping RealtimeMarketAnalyzer...")
        self.running = False
        
        # Wait for threads to terminate
        if self._update_thread and self._update_thread.is_alive():
            self._update_thread.join(timeout=5.0)
        
        self.logger.info("RealtimeMarketAnalyzer stopped")
    
    def __del__(self):
        """Destructor to ensure clean shutdown."""
        if hasattr(self, 'running') and self.running:
            self.stop()