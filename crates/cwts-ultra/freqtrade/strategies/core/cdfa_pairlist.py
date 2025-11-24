#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  4 23:36:04 2025

@author: ashina
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CDFA Pairlist Generator for FreqTrade

This module creates a dynamic pairlist generator that uses the Cognitive
Diversity Fusion Analysis (CDFA) system to evaluate and select the most promising
trading pairs for FreqTrade.

Author: Created on May 4, 2025
"""

import os
import json
import time
import logging
import threading
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
from datetime import datetime, timedelta
from functools import lru_cache, wraps

# Third-party imports with error handling
try:
    import ccxt
except ImportError:
    raise ImportError("ccxt is required for exchange connectivity. Install with: pip install ccxt")

# Import CDFA and extensions
from enhanced_cdfa import CognitiveDiversityFusionAnalysis

# Import analyzers from cdfa_extensions
from cdfa_extensions.analyzers.soc_analyzer import SOCAnalyzer
from cdfa_extensions.analyzers.panarchy_analyzer import PanarchyAnalyzer
from cdfa_extensions.analyzers.fibonacci_analyzer import FibonacciAnalyzer
from cdfa_extensions.analyzers.antifragility_analyzer import AntifragilityAnalyzer

# Import detectors from cdfa_extensions
from cdfa_extensions.detectors.pattern_recognizer import PatternRecognizer
from cdfa_extensions.detectors.whale_detector import WhaleDetector
from cdfa_extensions.detectors.black_swan_detector import BlackSwanDetector
from cdfa_extensions.detectors.fibonacci_pattern_detector import FibonacciPatternDetector

# Constants
DEFAULT_UPDATE_INTERVAL = 3600  # 1 hour in seconds
DEFAULT_MAX_PAIRS = 30
DEFAULT_TIMEFRAMES = ["1h", "4h", "1d"]
DEFAULT_CANDLE_LIMIT = 600
MIN_CONFIDENCE_THRESHOLD = 0.75


class CdfaPairlistGenerator:
    """
    Dynamic pairlist generator for FreqTrade using CDFA analysis.
    
    This class analyzes trading pairs using the Cognitive Diversity Fusion Analysis
    system to select the most promising pairs based on multiple technical indicators,
    market regimes, and pattern recognition.
    """
    
    def __init__(self, 
                 config_path: str = "config.json",
                 output_path: str = "user_data/pairlist.json",
                 update_interval: int = DEFAULT_UPDATE_INTERVAL,
                 max_pairs: int = DEFAULT_MAX_PAIRS,
                 timeframes: List[str] = None,
                 enable_redis: bool = False,
                 enable_ml: bool = True,
                 cache_dir: str = "user_data/cdfa_cache"):
        """
        Initialize the CDFA Pairlist Generator.
        
        Args:
            config_path: Path to FreqTrade configuration file
            output_path: Path where pairlist.json will be saved
            update_interval: How often to update the pairlist (seconds)
            max_pairs: Maximum number of pairs to include in pairlist
            timeframes: List of timeframes to analyze
            enable_redis: Whether to enable Redis for communication
            enable_ml: Whether to enable ML components
            cache_dir: Directory to store cache files
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing CDFA Pairlist Generator")
        
        # Configuration
        self.config_path = config_path
        self.output_path = output_path
        self.update_interval = update_interval
        self.max_pairs = max_pairs
        self.timeframes = timeframes or DEFAULT_TIMEFRAMES
        self.enable_redis = enable_redis
        self.enable_ml = enable_ml
        self.cache_dir = cache_dir
        
        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # State variables
        self.running = False
        self._stop_event = threading.Event()
        self.update_thread = None
        self.last_update_time = 0
        self.pair_metrics = {}
        self.all_analyzed_pairs = []
        
        # Initialize components
        self._load_freqtrade_config()
        self._initialize_cdfa()
        self._initialize_components()
        self._initialize_exchange()
        
        self.logger.info("CDFA Pairlist Generator initialized successfully")
        
    def _load_freqtrade_config(self):
        """Load FreqTrade configuration file"""
        try:
            with open(self.config_path, 'r') as f:
                self.freqtrade_config = json.load(f)
                
            # Extract key configuration
            exchange_config = self.freqtrade_config.get('exchange', {})
            self.exchange_name = exchange_config.get('name', 'binance')
            self.stake_currency = self.freqtrade_config.get('stake_currency', 'USDT')
            self.dry_run = self.freqtrade_config.get('dry_run', True)
            
            # Get API keys if available
            self.api_key = exchange_config.get('key', '')
            self.api_secret = exchange_config.get('secret', '')
            
            # Get blacklist if available
            self.blacklist = self.freqtrade_config.get('pair_blacklist', [])
            
            self.logger.info(f"Loaded FreqTrade config: exchange={self.exchange_name}, "
                             f"stake_currency={self.stake_currency}")
                             
        except Exception as e:
            self.logger.error(f"Error loading FreqTrade config: {e}")
            # Use default values
            self.exchange_name = 'binance'
            self.stake_currency = 'USDT'
            self.blacklist = []
            self.dry_run = True
            self.api_key = ''
            self.api_secret = ''
            
    def _initialize_cdfa(self):
        """Initialize CDFA with optimal settings for pairlist generation"""
        from enhanced_cdfa import CDFAConfig
        
        # Create custom configuration
        cdfa_config = CDFAConfig(
            enable_caching=True,
            cache_size=512,
            min_signals_required=3,
            enable_redis=self.enable_redis,
            enable_ml=self.enable_ml,
            enable_adaptive_learning=True,
            enable_visualization=False,
            # Performance optimization settings
            parallelization_threshold=4,
            max_workers=max(1, os.cpu_count() - 1),
            use_numba=True,
            use_vectorization=True
        )
        
        # Initialize CDFA with config
        self.cdfa = CognitiveDiversityFusionAnalysis(cdfa_config)
        
        # Initialize integration frameworks
        self.cdfa.integrate_external_analyzers()
        self.cdfa.integrate_external_detectors()
        
        self.logger.info("CDFA system initialized")
        
    def _initialize_components(self):
        """Initialize all analyzers and detectors"""
        try:
            # Initialize SOC analyzer
            self.soc_analyzer = SOCAnalyzer()
            self.cdfa.connect_soc_analyzer(self.soc_analyzer)
            
            # Initialize Panarchy analyzer
            self.panarchy_analyzer = PanarchyAnalyzer()
            self.cdfa.connect_panarchy_analyzer(self.panarchy_analyzer)
            
            # Initialize Fibonacci analyzer
            self.fibonacci_analyzer = FibonacciAnalyzer()
            self.cdfa.connect_fibonacci_analyzer(self.fibonacci_analyzer)
            
            # Initialize Antifragility analyzer
            self.antifragility_analyzer = AntifragilityAnalyzer()
            self.cdfa.connect_antifragility_analyzer(self.antifragility_analyzer)
            
            # Initialize Pattern recognizer
            self.pattern_recognizer = PatternRecognizer()
            self.cdfa.connect_pattern_recognizer(self.pattern_recognizer)
            
            # Initialize detectors
            self.whale_detector = WhaleDetector()
            self.cdfa.integrate_whale_detector(self.whale_detector)
            
            self.black_swan_detector = BlackSwanDetector()
            self.cdfa.integrate_black_swan_detector(self.black_swan_detector)
            
            self.fibonacci_detector = FibonacciPatternDetector()
            self.cdfa.integrate_fibonacci_detector(self.fibonacci_detector)
            
            self.logger.info("All analyzers and detectors initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing components: {e}")
            raise RuntimeError(f"Failed to initialize CDFA components: {e}")
            
    def _initialize_exchange(self):
        """Initialize exchange connection"""
        try:
            # Determine exchange class from CCXT
            exchange_id = self.exchange_name.lower()
            exchange_class = getattr(ccxt, exchange_id)
            
            # Create exchange instance
            exchange_params = {
                'enableRateLimit': True,
                'nonce': lambda: int(time.time() * 1000),
            }
            
            # Add API keys if available
            if self.api_key and self.api_secret:
                exchange_params['apiKey'] = self.api_key
                exchange_params['secret'] = self.api_secret
                
            # Initialize exchange
            self.exchange = exchange_class(exchange_params)
            
            # Test connection
            self.exchange.load_markets()
            self.logger.info(f"Successfully connected to {self.exchange_name}")
            
        except Exception as e:
            self.logger.error(f"Error connecting to exchange: {e}")
            raise ConnectionError(f"Failed to connect to exchange {self.exchange_name}: {e}")
            
    def get_available_pairs(self) -> List[str]:
        """Get list of available trading pairs for the stake currency"""
        try:
            # Reload markets to get latest data
            markets = self.exchange.load_markets()
            
            # Define stablecoins list
            STABLECOINS = ['USDT', 'USDC', 'BUSD', 'DAI', 'TUSD', 'PAX', 'GUSD', 'USDP', 'FRAX', 'USDN']
            
            # Filter for pairs with stake currency
            available_pairs = []
            
            for symbol in markets.keys():
                # Check if symbol is in correct format
                if '/' not in symbol:
                    continue
                    
                base, quote = symbol.split('/')
                
                # Skip stablecoin-to-stablecoin pairs
                if base in STABLECOINS and quote in STABLECOINS:
                    self.logger.debug(f"Skipping stablecoin pair: {symbol}")
                    continue
                
                # Check if pair has stake currency as quote
                if quote == self.stake_currency:
                    # Check if pair is active
                    market = markets[symbol]
                    
                    # Skip inactive or delisted markets
                    if not market.get('active', False):
                        self.logger.debug(f"Skipping inactive market: {symbol}")
                        continue
                    
                    # Skip if market is marked as delisted or deprecated
                    if market.get('status') in ['delisted', 'deprecated', 'terminated', 'closed']:
                        self.logger.debug(f"Skipping delisted market: {symbol}")
                        continue
                    
                    available_pairs.append(symbol)
                    
            self.logger.info(f"Found {len(available_pairs)} available pairs for {self.stake_currency}")
            return available_pairs
            
        except Exception as e:
            self.logger.error(f"Error getting available pairs: {e}")
            return []
            
    def fetch_ohlcv_data(self, pair: str, timeframe: str) -> pd.DataFrame:
        """
        Fetch OHLCV data for a trading pair and timeframe with robust error handling.
        
        Args:
            pair: Trading pair symbol (e.g., 'BTC/USDT')
            timeframe: Timeframe string (e.g., '1h', '4h', '1d')
            
        Returns:
            DataFrame with OHLCV data or empty DataFrame with error metadata
        """
        # Set default result with error info
        empty_df = pd.DataFrame()
        empty_df.attrs['error'] = True
        empty_df.attrs['error_message'] = "Not initialized"
        
        try:
            # Validate exchange object
            if not hasattr(self, 'exchange') or not callable(getattr(self.exchange, 'fetch_ohlcv', None)):
                error_msg = f"Exchange object or fetch_ohlcv method not available for {pair}"
                self.logger.error(error_msg)
                empty_df.attrs['error_message'] = error_msg
                return empty_df
            
            # Define fetch parameters
            limit = DEFAULT_CANDLE_LIMIT  # Use class constant (1200 candles)
            params = {}
            
            # Add retry mechanism
            max_retries = 3
            current_retry = 0
            
            while current_retry < max_retries:
                try:
                    # Fetch the data with timeout
                    ohlcv = self.exchange.fetch_ohlcv(
                        pair, 
                        timeframe,
                        limit=limit,
                        params=params
                    )
                    
                    # Validate data
                    if not ohlcv or len(ohlcv) < 20:  # Minimum viable data
                        error_msg = f"Insufficient OHLCV data for {pair} {timeframe}: got {len(ohlcv) if ohlcv else 0} candles"
                        if current_retry < max_retries - 1:
                            self.logger.warning(f"{error_msg}, retrying ({current_retry+1}/{max_retries})...")
                            current_retry += 1
                            time.sleep(1)  # Delay between retries
                            continue
                        else:
                            self.logger.error(error_msg)
                            empty_df.attrs['error_message'] = error_msg
                            return empty_df
                    
                    # Data looks good, process it
                    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    
                    # Ensure numeric types
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    # Drop rows with any NaN values in critical columns
                    df.dropna(subset=['open', 'high', 'low', 'close', 'volume'], inplace=True)
                    
                    # Validate again after cleaning
                    if len(df) < 20:
                        error_msg = f"Insufficient clean data for {pair} {timeframe}: {len(df)} candles after cleaning"
                        self.logger.error(error_msg)
                        empty_df.attrs['error_message'] = error_msg
                        return empty_df
                    
                    # Convert timestamp to datetime and set as index
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
                    df.set_index('timestamp', inplace=True)
                    
                    # Preserve metadata consistently
                    df.attrs['metadata'] = {
                        'pair': pair,
                        'timeframe': timeframe,
                        'exchange': self.exchange_name,
                        'candles': len(df)
                    }
                    df.attrs['error'] = False
                    
                    # Log success
                    self.logger.debug(f"Fetched {len(df)} candles for {pair} {timeframe}")
                    print(f"Fetched {len(df)} candles for {pair} {timeframe}")
                    
                    return df
                    
                except ccxt.NetworkError as e:
                    # Retry network errors
                    if current_retry < max_retries - 1:
                        self.logger.warning(f"Network error fetching {pair} {timeframe}: {e}, retrying ({current_retry+1}/{max_retries})...")
                        current_retry += 1
                        time.sleep(2)  # Longer delay for network errors
                    else:
                        self.logger.error(f"Network error fetching {pair} {timeframe} after {max_retries} retries: {e}")
                        empty_df.attrs['error_message'] = f"Network error: {str(e)}"
                        return empty_df
                        
                except Exception as e:
                    # Don't retry other exceptions
                    self.logger.error(f"Error fetching OHLCV for {pair} ({timeframe}): {e}")
                    empty_df.attrs['error_message'] = str(e)
                    return empty_df
            
            # This should not be reached due to returns within the loop
            return empty_df
            
        except Exception as e:
            # Catch any unexpected errors
            self.logger.error(f"Unexpected error fetching OHLCV for {pair} ({timeframe}): {e}", exc_info=True)
            empty_df.attrs['error_message'] = f"Unexpected error: {str(e)}"
            return empty_df
            
    def analyze_pair(self, pair: str) -> Optional[Dict[str, Any]]:
        """
        Analyze a trading pair across multiple timeframes using CDFA with enhanced validation.
        
        Args:
            pair: Trading pair symbol (e.g., 'BTC/USDT')
            
        Returns:
            Analysis results dictionary or None if analysis failed
        """
        try:
            self.logger.debug(f"Starting analysis of {pair}")
            
            # Generate cache path
            cache_dir = getattr(self, 'cache_dir', 'user_data/cdfa_cache')
            os.makedirs(cache_dir, exist_ok=True)
            cache_path = os.path.join(cache_dir, f"{pair.replace('/', '_')}_analysis.json")
            
            # Check for recent cache file
            if os.path.exists(cache_path):
                cache_age = time.time() - os.path.getmtime(cache_path)
                if cache_age < self.update_interval * 0.8:  # Use cache if relatively fresh
                    try:
                        with open(cache_path, 'r') as f:
                            cached_data = json.load(f)
                        self.logger.debug(f"Using cached analysis for {pair}")
                        return cached_data
                    except Exception as e:
                        self.logger.warning(f"Error reading cache for {pair}: {e}")
                        # Continue with fresh analysis
            
            # Verify if pair should be analyzed (volume and volatility)
            is_valid, validation_data = self._validate_pair_viability(pair)
            if not is_valid:
                reason = validation_data.get('reason', 'Unknown reason')
                self.logger.debug(f"Skipping {pair}: {reason}")
                return None
                
            # Prepare for analysis
            pair_analysis = {
                'pair': pair,
                'timestamp': time.time(),
                'timeframe_results': {},
                'action': 'none',
                'strength': 0.0,
                'confidence': 0.0,
                'score': 0.0,
                'validation': validation_data  # Include validation data for reference
            }
                
            # Get all timeframes for analysis
            weighted_signals = []
            timeframe_weights = {'1h': 0.3, '4h': 0.4, '1d': 0.5}
            
            # Track if analysis succeeded for at least one timeframe
            any_timeframe_succeeded = False
            
            # Analyze each timeframe
            for timeframe in self.timeframes:
                self.logger.debug(f"Analyzing {pair} on {timeframe} timeframe")
                
                # Fetch OHLCV data
                df = self.fetch_ohlcv_data(pair, timeframe)
                
                # Check if fetch was successful
                if df.empty or df.attrs.get('error', False):
                    error_msg = df.attrs.get('error_message', 'Empty dataframe')
                    self.logger.warning(f"Failed to get data for {pair} on {timeframe}: {error_msg}")
                    continue
                
                # Ensure we have enough data
                if len(df) < 100:  # Minimum required for most complex indicators
                    self.logger.warning(f"Insufficient data for {pair} on {timeframe}: {len(df)} candles")
                    continue
                    
                try:
                    # Process with CDFA
                    result = self.cdfa.process_signals_from_dataframe(df, pair, calculate_fusion=True)
                    
                    # Extract key metrics
                    signals = result.get("signals", {})
                    fusion_result = result.get("fusion_result", {})
                    market_regime = result.get("market_regime", "unknown")
                    volatility = result.get("volatility", 0.5)
                    
                    # Verify that we have signals and fusion results
                    if not signals or not fusion_result:
                        self.logger.warning(f"No signals or fusion results for {pair} on {timeframe}")
                        continue
                    
                    # Generate trading recommendation
                    recommendation = self.cdfa._generate_action_recommendation(
                        signals, 
                        fusion_result.get("fused_signal", [0.5]), 
                        market_regime
                    )
                    
                    # Store timeframe result
                    timeframe_result = {
                        'action': recommendation.get('action', 'none'),
                        'strength': recommendation.get('strength', 0.0),
                        'confidence': fusion_result.get('confidence', 0.5),
                        'signal_value': recommendation.get('signal_value', 0.5),
                        'market_regime': market_regime,
                        'volatility': volatility,
                        'signals_count': len(signals)
                    }
                    
                    pair_analysis['timeframe_results'][timeframe] = timeframe_result
                    
                    # Weight by timeframe
                    weight = timeframe_weights.get(timeframe, 0.3)
                    weighted_signals.append({
                        'action': timeframe_result['action'],
                        'strength': timeframe_result['strength'],
                        'confidence': timeframe_result['confidence'],
                        'weight': weight
                    })
                    
                    any_timeframe_succeeded = True
                    
                except Exception as e:
                    self.logger.error(f"Error analyzing {pair} on {timeframe}: {e}", exc_info=True)
                    continue
                    
            # If no timeframes were successfully analyzed, return None
            if not any_timeframe_succeeded:
                self.logger.warning(f"No successful timeframe analysis for {pair}")
                return None
                
            # Calculate combined metrics across timeframes
            actions = {a: 0.0 for a in ['buy', 'sell', 'hold', 'none']}
            total_weight = 0.0
            total_confidence = 0.0
            
            for ws in weighted_signals:
                action = ws['action']
                # Weight by both timeframe weight and signal confidence
                weight = ws['weight'] * ws['confidence']
                actions[action] += weight
                total_weight += weight
                total_confidence += ws['confidence'] * ws['weight']
                
            # Normalize confidence
            if total_weight > 0:
                pair_analysis['confidence'] = total_confidence / total_weight
            else:
                pair_analysis['confidence'] = 0.0
                
            # Determine final action
            if total_weight > 0:
                norm_actions = {a: w/total_weight for a, w in actions.items()}
                best_action = max(norm_actions.items(), key=lambda x: x[1])
                pair_analysis['action'] = best_action[0]
                pair_analysis['strength'] = best_action[1]
            
            # Calculate final score (prioritize buy signals)
            if pair_analysis['action'] == 'buy':
                pair_analysis['score'] = pair_analysis['strength'] * pair_analysis['confidence'] * 2.0
            elif pair_analysis['action'] == 'hold':
                pair_analysis['score'] = 0.3 * pair_analysis['confidence']
            elif pair_analysis['action'] == 'sell':
                pair_analysis['score'] = -1.0 * pair_analysis['strength'] * pair_analysis['confidence']
            else:
                pair_analysis['score'] = 0.0
                
            # Cache results
            try:
                with open(cache_path, 'w') as f:
                    json.dump(pair_analysis, f, indent=2)
            except Exception as e:
                self.logger.warning(f"Failed to cache analysis for {pair}: {e}")
                
            self.logger.debug(f"Completed analysis of {pair}: action={pair_analysis['action']}, score={pair_analysis['score']:.4f}")
            return pair_analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing {pair}: {e}", exc_info=True)
            return None
            
    def analyze_all_pairs(self) -> List[Dict[str, Any]]:
        """Analyze all available trading pairs using CDFA"""
        # Get available pairs
        pairs = self.get_available_pairs()
        if not pairs:
            self.logger.warning("No pairs available for analysis")
            return []
            
        # Filter blacklisted pairs
        filtered_pairs = [p for p in pairs if p not in self.blacklist]
        self.logger.info(f"Analyzing {len(filtered_pairs)} pairs (excluded {len(pairs) - len(filtered_pairs)} blacklisted pairs)")
        
        # Add debugging and error tracking
        analyzed_pairs = []
        error_counts = {}
        empty_data_counts = {}
        
        # Process in smaller batches with better error handling
        batch_size = 50
        for i in range(0, len(filtered_pairs), batch_size):
            batch = filtered_pairs[i:i+batch_size]
            self.logger.info(f"Processing batch {i//batch_size + 1}/{(len(filtered_pairs)-1)//batch_size + 1} ({len(batch)} pairs)")
            
            for pair in batch:
                self.logger.debug(f"Starting analysis of {pair}")
                
                try:
                    result = self.analyze_pair(pair)
                    if result:
                        analyzed_pairs.append(result)
                        self.logger.debug(f"Successfully analyzed {pair}")
                    else:
                        self.logger.debug(f"No results for {pair}")
                except Exception as e:
                    error_msg = str(e)
                    if error_msg not in error_counts:
                        error_counts[error_msg] = 0
                    error_counts[error_msg] += 1
                    self.logger.error(f"Error analyzing {pair}: {e}")
        
        # Log error statistics
        if error_counts:
            self.logger.warning(f"Analysis errors by type: {error_counts}")
            
        # Sort by score
        self.all_analyzed_pairs = sorted(analyzed_pairs, key=lambda x: x['score'], reverse=True)
        self.logger.info(f"Completed analysis of {len(self.all_analyzed_pairs)} pairs")
        
        return self.all_analyzed_pairs
        
    def select_best_pairs(self, min_confidence_threshold=0.6) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Select the best pairs based on CDFA analysis"""
        # Analyze pairs if not already done
        if not hasattr(self, 'all_analyzed_pairs') or not self.all_analyzed_pairs:
            self.analyze_all_pairs()
        
        # If no pairs passed analysis, implement fallback
        if not self.all_analyzed_pairs:
            self.logger.warning("No pairs passed CDFA analysis criteria. Using fallback selection.")
            try:
                fallback_pairs = []
                markets = self.exchange.load_markets()
                tickers = self.exchange.fetch_tickers()
                
                # Define stablecoins to filter out stablecoin pairs
                STABLECOINS = ['USDT', 'USDC', 'BUSD', 'DAI', 'TUSD', 'PAX', 'GUSD', 'USDP', 'FRAX', 'USDN']
                
                for symbol in markets:
                    if '/' not in symbol:
                        continue
                        
                    base, quote = symbol.split('/')
                    
                    # Skip stablecoin-to-stablecoin pairs
                    if base in STABLECOINS and quote in STABLECOINS:
                        continue
                    
                    if symbol in tickers and quote == self.stake_currency:
                        market = markets[symbol]
                        # Skip inactive markets
                        if not market.get('active', False):
                            continue
                        # Skip markets with problematic status
                        if market.get('status') in ['delisted', 'deprecated', 'terminated', 'closed']:
                            continue
                            
                        # Get volume
                        volume = tickers[symbol].get('quoteVolume', 0)
                        # Only include pairs with minimum volume
                        if volume >= 10000:  # $10k minimum volume
                            fallback_pairs.append({
                                'pair': symbol,
                                'action': 'hold',
                                'strength': 0.5,
                                'confidence': 0.5,
                                'score': volume  # Use volume as score for fallback
                            })
                
                # Sort by volume and take top pairs
                sorted_pairs = sorted(fallback_pairs, key=lambda x: x['score'], reverse=True)
                selected_pairs = sorted_pairs[:self.max_pairs]
                
                # Extract pair names - NO replace here to keep proper format
                pair_names = [p['pair'] for p in selected_pairs if p['pair']]
                
                self.logger.info(f"Selected {len(pair_names)} fallback pairs based on volume")
                return pair_names, selected_pairs
            except Exception as e:
                self.logger.error(f"Error in fallback pair selection: {e}")
                return [], []
            
        # Filter by minimum confidence
        confident_pairs = [p for p in self.all_analyzed_pairs if p.get('confidence', 0) >= min_confidence_threshold]
        
        # If we don't have enough confident pairs, lower the threshold
        if len(confident_pairs) < self.max_pairs / 2:
            self.logger.warning(f"Not enough pairs with confidence >= {min_confidence_threshold}. Lowering threshold.")
            min_confidence_threshold = 0.4
            confident_pairs = [p for p in self.all_analyzed_pairs if p.get('confidence', 0) >= min_confidence_threshold]
            
        # Get buy and hold recommendations
        potential_pairs = [p for p in confident_pairs if p.get('action') in ['buy', 'hold']]
        
        # If we still don't have enough pairs, include some with lower confidence
        if len(potential_pairs) < self.max_pairs / 2 and self.all_analyzed_pairs:
            self.logger.warning(f"Not enough buy/hold pairs. Including some with lower confidence.")
            # Sort by score and take top regardless of action/confidence
            additional_pairs = sorted(
                [p for p in self.all_analyzed_pairs if p not in potential_pairs],
                key=lambda x: x.get('score', 0), 
                reverse=True
            )
            potential_pairs.extend(additional_pairs[:max(5, self.max_pairs // 2)])
        
        # Prioritize buys, then holds with good confidence
        sorted_pairs = sorted(potential_pairs, key=lambda x: x.get('score', 0), reverse=True)
        
        # Select top pairs up to max_pairs
        selected_pairs = sorted_pairs[:self.max_pairs]
        
        # Extract just pair names for pairlist
        # IMPORTANT CHANGE: Remove the replace('/', '_') to keep proper format
        pair_names = [p.get('pair', '') for p in selected_pairs]
        
        # Remove any empty strings that might have crept in
        pair_names = [p for p in pair_names if p]
        
        # Log details
        buy_count = sum(1 for p in selected_pairs if p.get('action') == 'buy')
        hold_count = sum(1 for p in selected_pairs if p.get('action') == 'hold')
        other_count = len(selected_pairs) - buy_count - hold_count
        
        self.logger.info(f"Selected {len(pair_names)} pairs: {buy_count} buy, {hold_count} hold, {other_count} other")
        
        return pair_names, selected_pairs
    
    def _implement_fallback_selection(self) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Implement fallback pair selection based on volume when CDFA analysis fails.
        
        Returns:
            Tuple of (pair_names, pair_details)
        """
        try:
            fallback_pairs = []
            self.logger.info("Fetching market data for fallback selection")
            
            # Load markets with error handling
            try:
                markets = self.exchange.load_markets()
            except Exception as e:
                self.logger.error(f"Error loading markets: {e}")
                markets = {}
                
            # Fetch tickers with error handling
            try:
                tickers = self.exchange.fetch_tickers()
            except Exception as e:
                self.logger.error(f"Error fetching tickers: {e}")
                tickers = {}
            
            if not markets or not tickers:
                self.logger.error("Cannot perform fallback selection: missing market data")
                return [], []
                
            # Define stablecoins to filter out stablecoin pairs
            STABLECOINS = ['USDT', 'USDC', 'BUSD', 'DAI', 'TUSD', 'PAX', 'GUSD', 'USDP', 'FRAX', 'USDN']
            
            # Track processed pairs
            processed_count = 0
            
            for symbol in markets:
                processed_count += 1
                if processed_count % 100 == 0:
                    self.logger.debug(f"Processed {processed_count} markets for fallback selection")
                    
                # Basic validation
                if '/' not in symbol:
                    continue
                    
                base, quote = symbol.split('/')
                
                # Skip stablecoin-to-stablecoin pairs
                if base in STABLECOINS and quote in STABLECOINS:
                    continue
                
                # Ensure pair matches our stake currency
                if quote != self.stake_currency:
                    continue
                    
                # Basic market validation
                market = markets[symbol]
                if not market.get('active', False):
                    continue
                    
                # Skip markets with problematic status
                if market.get('status') in ['delisted', 'deprecated', 'terminated', 'closed']:
                    continue
                    
                # Check if ticker exists
                if symbol not in tickers:
                    continue
                    
                # Get volume with safety checks
                ticker = tickers[symbol]
                volume = ticker.get('quoteVolume', 0)
                if not isinstance(volume, (int, float)):
                    continue
                    
                # Only include pairs with minimum volume
                min_fallback_volume = 10000  # $10k minimum volume for fallback
                if volume >= min_fallback_volume:
                    # Create simple pair entry
                    fallback_pairs.append({
                        'pair': symbol,
                        'action': 'hold',
                        'strength': 0.5,
                        'confidence': 0.5,
                        'score': volume,  # Use volume as score for fallback
                        'fallback': True  # Mark as fallback selection
                    })
            
            # Sort by volume and take top pairs
            sorted_pairs = sorted(fallback_pairs, key=lambda x: x['score'], reverse=True)
            selected_pairs = sorted_pairs[:self.max_pairs]
            
            # Extract pair names - using the correct format
            pair_names = [p['pair'] for p in selected_pairs if p['pair']]
            
            self.logger.info(f"Selected {len(pair_names)} fallback pairs based on volume")
            return pair_names, selected_pairs
            
        except Exception as e:
            self.logger.error(f"Error in fallback pair selection: {e}", exc_info=True)
            return [], []
        
    
    def _validate_pair_viability(self, pair: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate if a pair is viable for analysis by checking volume, volatility and other factors.
        
        Args:
            pair: Trading pair symbol (e.g., 'BTC/USDT')
            
        Returns:
            Tuple of (is_valid, validation_data)
        """
        validation_data = {
            'pair': pair,
            'timestamp': time.time(),
            'is_valid': False,
            'reason': 'Not validated',
            'checks': {}
        }
        
        try:
            # 1. Check if pair is tradable on exchange
            try:
                market = self.exchange.market(pair)
                if not market.get('active', False):
                    validation_data['reason'] = 'Market inactive'
                    validation_data['checks']['active'] = False
                    return False, validation_data
                    
                validation_data['checks']['active'] = True
            except Exception as e:
                validation_data['reason'] = f'Market info error: {str(e)}'
                validation_data['checks']['market_info'] = False
                return False, validation_data
            
            # 2. Check 24h volume
            try:
                ticker = self.exchange.fetch_ticker(pair)
                volume_usd = ticker.get('quoteVolume', 0)
                
                validation_data['checks']['volume'] = {
                    'value': volume_usd,
                    'min_required': 100000  # $100k minimum 24h volume
                }
                
                if volume_usd < 100000:  # $100k minimum 24h volume
                    validation_data['reason'] = f'Low volume: ${volume_usd:.2f}'
                    validation_data['checks']['volume']['passed'] = False
                    return False, validation_data
                    
                validation_data['checks']['volume']['passed'] = True
            except Exception as e:
                validation_data['reason'] = f'Volume check error: {str(e)}'
                validation_data['checks']['volume_check'] = False
                return False, validation_data
                
            # 3. Check historical volatility using 1h timeframe
            try:
                df_1h = self.fetch_ohlcv_data(pair, '1h')
                
                # Check if data fetch was successful
                if df_1h.empty or df_1h.attrs.get('error', False):
                    validation_data['reason'] = f'No 1h data: {df_1h.attrs.get("error_message", "Unknown error")}'
                    validation_data['checks']['data_1h'] = False
                    return False, validation_data
                    
                # Check if we have enough data points
                if len(df_1h) < 24:  # At least 24h of hourly data
                    validation_data['reason'] = f'Insufficient 1h data: {len(df_1h)} candles'
                    validation_data['checks']['data_1h_count'] = False
                    return False, validation_data
                    
                # Calculate volatility (annualized standard deviation of returns)
                returns = df_1h['close'].pct_change().dropna()
                if len(returns) < 20:
                    validation_data['reason'] = f'Insufficient return data: {len(returns)} values'
                    validation_data['checks']['returns_data'] = False
                    return False, validation_data
                    
                volatility = returns.std() * (252 ** 0.5)  # Annualized
                
                validation_data['checks']['volatility'] = {
                    'value': volatility,
                    'min_required': 0.01  # 1% minimum volatility
                }
                
                if volatility < 0.01:  # 1% minimum volatility
                    validation_data['reason'] = f'Low volatility: {volatility:.4f}'
                    validation_data['checks']['volatility']['passed'] = False
                    return False, validation_data
                    
                validation_data['checks']['volatility']['passed'] = True
            except Exception as e:
                validation_data['reason'] = f'Volatility check error: {str(e)}'
                validation_data['checks']['volatility_check'] = False
                return False, validation_data
                
            # 4. All checks passed
            validation_data['is_valid'] = True
            validation_data['reason'] = 'All checks passed'
            return True, validation_data
            
        except Exception as e:
            validation_data['reason'] = f'Validation error: {str(e)}'
            return False, validation_data
        
    def generate_pairlist_json(self) -> Dict[str, Any]:
        """Generate and save the pairlist.json file for FreqTrade"""
        self.logger.info("Generating pairlist.json")
        
        # Get selected pairs
        pair_names, pair_details = self.select_best_pairs()
        
        # Create pairlist JSON structure
        now = int(time.time())
        pairlist = {
            "pairlist": pair_names,
            "generated": now,
            "timestamp": datetime.fromtimestamp(now).strftime('%Y-%m-%d %H:%M:%S'),
            "details": [{
                "pair": p['pair'],
                "action": p['action'],
                "strength": round(p['strength'], 4),
                "confidence": round(p['confidence'], 4),
                "score": round(p['score'], 4)
            } for p in pair_details]
        }
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        
        # Save to file
        with open(self.output_path, 'w') as f:
            json.dump(pairlist, f, indent=2)
            
        self.logger.info(f"Pairlist saved to {self.output_path} with {len(pair_names)} pairs")
        self.last_update_time = now
        
        return pairlist
        
    def update_thread_function(self):
        """Thread function for periodic updates"""
        self.logger.info("Starting pairlist update thread")
        
        while not self._stop_event.is_set():
            try:
                # Generate pairlist
                self.generate_pairlist_json()
                self.logger.info(f"Pairlist updated. Next update in {self.update_interval} seconds.")
                
                # Sleep until next update
                for _ in range(int(self.update_interval / 10)):
                    if self._stop_event.is_set():
                        break
                    time.sleep(10)
                    
            except Exception as e:
                self.logger.error(f"Error updating pairlist: {e}")
                time.sleep(60)  # Wait a minute and try again
                
        self.logger.info("Pairlist update thread stopped")
        
    def start(self):
        """Start periodic pairlist updates"""
        if self.running:
            self.logger.warning("Already running")
            return
            
        # Generate initial pairlist
        self.generate_pairlist_json()
        
        # Start update thread
        self._stop_event.clear()
        self.update_thread = threading.Thread(target=self.update_thread_function)
        self.update_thread.daemon = True
        self.update_thread.start()
        
        self.running = True
        self.logger.info(f"Started periodic updates every {self.update_interval} seconds")
        
    def stop(self):
        """Stop periodic pairlist updates"""
        if not self.running:
            self.logger.warning("Not running")
            return
            
        # Signal thread to stop
        self._stop_event.set()
        
        # Wait for thread to terminate
        if self.update_thread and self.update_thread.is_alive():
            self.update_thread.join(timeout=30)
            
        self.running = False
        self.logger.info("Stopped periodic updates")
        
    def get_status(self) -> Dict[str, Any]:
        """Get current status of pairlist generator"""
        status = {
            "running": self.running,
            "last_update": self.last_update_time,
            "time_since_update": time.time() - self.last_update_time if self.last_update_time else None,
            "update_interval": self.update_interval,
            "max_pairs": self.max_pairs,
            "analyzed_pairs": len(self.all_analyzed_pairs) if hasattr(self, 'all_analyzed_pairs') else 0,
            "exchange": self.exchange_name,
            "stake_currency": self.stake_currency,
            "output_path": self.output_path
        }
        
        return status
        
    def __enter__(self):
        """Context manager enter - start the generator"""
        self.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - stop the generator"""
        self.stop()

    def debug_wrapper(func):
        """Wrapper to debug function execution and catch errors"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            import gc
            import traceback
            import logging
            
            logging.info(f"Starting {func.__name__}")
            gc.collect()  # Force garbage collection before
            
            try:
                # Run the function
                result = func(*args, **kwargs)
                logging.info(f"Completed {func.__name__}")
                return result
                
            except Exception as e:
                # Log detailed error information
                logging.error(f"Error in {func.__name__}: {e}")
                logging.error(traceback.format_exc())
                # Return a safe default value
                return None
            finally:
                # Clean up
                gc.collect()
                


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )
    
    # Parse command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description='CDFA Pairlist Generator for FreqTrade')
    parser.add_argument('--config', type=str, default='/home/ashina/freqtrade/user_data/config.json', help='Path to FreqTrade config file')
    parser.add_argument('--output', type=str, default='/home/ashina/freqtrade/user_data/cdfa_pairlist.json', help='Output path for pairlist')
    parser.add_argument('--interval', type=int, default=300, help='Update interval in seconds')
    parser.add_argument('--max-pairs', type=int, default=34, help='Maximum number of pairs in pairlist')
    parser.add_argument('--timeframes', type=str, default='1h,4h,1d', help='Timeframes to analyze (comma-separated)')
    parser.add_argument('--redis', action='store_true', help='Enable Redis integration')
    parser.add_argument('--no-ml', dest='ml', action='store_false', help='Disable ML components')
    parser.add_argument('--run-once', action='store_true', help='Run once and exit')
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = CdfaPairlistGenerator(
        config_path=args.config,
        output_path=args.output,
        update_interval=args.interval,
        max_pairs=args.max_pairs,
        timeframes=args.timeframes.split(','),
        enable_redis=args.redis,
        enable_ml=args.ml
    )
    
    if args.run_once:
        # Run once and exit
        generator.generate_pairlist_json()
    else:
        # Start periodic updates
        try:
            generator.start()
            
            # Keep main thread alive
            while True:
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("Stopping...")
            generator.stop()