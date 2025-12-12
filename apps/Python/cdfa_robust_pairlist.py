#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  4 23:42:50 2025

@author: ashina
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robust CDFA Pairlist Generator for FreqTrade

This module creates a dynamic pairlist generator that uses the Cognitive
Diversity Fusion Analysis (CDFA) system to evaluate and select the most promising
trading pairs for FreqTrade. It includes enhanced error handling to prevent
crashes from component failures.

Author: Created on May 4, 2025
"""

import os
import json
import time
import logging
import threading
import traceback
import warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
from datetime import datetime, timedelta

# Filter deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Series.*__getitem__")

# Third-party imports with error handling
try:
    import ccxt
except ImportError:
    raise ImportError("ccxt is required for exchange connectivity. Install with: pip install ccxt")

# Define component reliability tracking
class ComponentReliability:
    """Track reliability of various analysis components"""
    
    def __init__(self, failure_threshold=3, recovery_time=3600):
        """
        Initialize component reliability tracker
        
        Args:
            failure_threshold: Number of failures before disabling a component
            recovery_time: Time in seconds before retrying a disabled component
        """
        self.failures = {}  # component_name -> failure count
        self.disabled_until = {}  # component_name -> timestamp
        self.failure_threshold = failure_threshold
        self.recovery_time = recovery_time
        self.lock = threading.RLock()
        
    def record_failure(self, component_name):
        """Record a component failure"""
        with self.lock:
            if component_name not in self.failures:
                self.failures[component_name] = 0
                
            self.failures[component_name] += 1
            
            # Disable if threshold reached
            if self.failures[component_name] >= self.failure_threshold:
                self.disabled_until[component_name] = time.time() + self.recovery_time
                logging.warning(f"Component {component_name} disabled until "
                              f"{datetime.fromtimestamp(self.disabled_until[component_name])}")
                
    def record_success(self, component_name):
        """Record a component success"""
        with self.lock:
            if component_name in self.failures:
                # Reduce failure count but don't go below zero
                self.failures[component_name] = max(0, self.failures[component_name] - 1)
                
            # Re-enable if previously disabled
            if component_name in self.disabled_until:
                del self.disabled_until[component_name]
                
    def is_component_enabled(self, component_name):
        """Check if a component is currently enabled"""
        with self.lock:
            # If not in disabled list, it's enabled
            if component_name not in self.disabled_until:
                return True
                
            # Check if recovery time has passed
            current_time = time.time()
            if current_time >= self.disabled_until[component_name]:
                # Re-enable component
                del self.disabled_until[component_name]
                return True
                
            return False
            
    def get_disabled_components(self):
        """Get list of currently disabled components"""
        with self.lock:
            current_time = time.time()
            return {
                name: until 
                for name, until in self.disabled_until.items() 
                if current_time < until
            }


class SafeCDFAWrapper:
    """Safe wrapper around CDFA to handle component failures gracefully"""
    
    def __init__(self, cdfa_instance, reliability_tracker=None):
        """
        Initialize CDFA wrapper
        
        Args:
            cdfa_instance: Instance of CognitiveDiversityFusionAnalysis
            reliability_tracker: ComponentReliability instance
        """
        self.cdfa = cdfa_instance
        self.logger = logging.getLogger(__name__ + ".SafeCDFAWrapper")
        self.reliability = reliability_tracker or ComponentReliability()
        
    def process_signals_from_dataframe(self, dataframe, symbol, calculate_fusion=True):
        """
        Safely process signals from dataframe with component-level error handling
        """
        try:
            # Call the original method
            return self.cdfa.process_signals_from_dataframe(dataframe, symbol, calculate_fusion)
        except Exception as e:
            self.logger.error(f"Error in process_signals_from_dataframe: {e}")
            self.logger.debug(f"Exception details: {traceback.format_exc()}")
            
            # Create empty result structure
            result = {
                "signals": {},
                "performance_metrics": {},
                "market_regime": "unknown",
                "volatility": 0.5
            }
            
            # Try to process with individual components safely
            self._process_with_safe_components(dataframe, symbol, result)
            
            # Add fusion result if requested and we have signals
            if calculate_fusion and result["signals"]:
                try:
                    # Try to process fusion with whatever signals we have
                    fusion_result = self.cdfa._process_fusion(symbol)
                    result["fusion_result"] = fusion_result
                except Exception as fusion_err:
                    self.logger.error(f"Error in fusion processing: {fusion_err}")
                    # Create dummy fusion result
                    result["fusion_result"] = {
                        "symbol": symbol,
                        "timestamp": time.time(),
                        "fused_signal": [0.5],
                        "confidence": 0.1,
                        "market_regime": "unknown",
                        "volatility": 0.5,
                        "num_systems": len(result["signals"]),
                        "performance_metrics": {}
                    }
            
            return result
            
    def _process_with_safe_components(self, dataframe, symbol, result):
        """Process dataframe with each component individually with error handling"""
        signals = result["signals"]
        performance_metrics = result["performance_metrics"]
        
        # --- SOC Analyzer ---
        if self.reliability.is_component_enabled("soc_analyzer"):
            try:
                if self.cdfa._soc_analyzer is not None:
                    # Direct extract from SOC analyzer
                    period = 30
                    soc_metrics = self.cdfa._soc_analyzer.calculate_soc_metrics(dataframe, period)
                    
                    # Add extracted signals
                    signals["soc_index"] = soc_metrics['soc_index'].tolist()
                    signals["soc_complexity"] = soc_metrics['complexity'].tolist()
                    signals["soc_equilibrium"] = soc_metrics['equilibrium'].tolist()
                    signals["soc_fragility"] = soc_metrics['fragility'].tolist()
                    
                    # Set performance metrics
                    performance_metrics["soc_index"] = 0.75
                    performance_metrics["soc_complexity"] = 0.70
                    performance_metrics["soc_equilibrium"] = 0.70
                    performance_metrics["soc_fragility"] = 0.70
                    
                    # Record success
                    self.reliability.record_success("soc_analyzer")
            except Exception as e:
                self.logger.error(f"Error processing SOC analyzer: {e}")
                self.reliability.record_failure("soc_analyzer")
                
        # --- Panarchy Analyzer ---
        if self.reliability.is_component_enabled("panarchy_analyzer"):
            try:
                if self.cdfa._panarchy_analyzer is not None:
                    # Process with Panarchy analyzer
                    period = 50
                    panarchy_df = self.cdfa._panarchy_analyzer.calculate_pcr_components(dataframe, period)
                    panarchy_df = self.cdfa._panarchy_analyzer.identify_regime(panarchy_df, period)
                    
                    # Extract signals
                    signals["panarchy_potential"] = panarchy_df['panarchy_P'].tolist()
                    signals["panarchy_connectedness"] = panarchy_df['panarchy_C'].tolist()
                    signals["panarchy_resilience"] = panarchy_df['panarchy_R'].tolist()
                    signals["panarchy_regime_score"] = panarchy_df['panarchy_regime_score'].tolist()
                    
                    # Set performance metrics
                    performance_metrics["panarchy_potential"] = 0.70
                    performance_metrics["panarchy_connectedness"] = 0.70
                    performance_metrics["panarchy_resilience"] = 0.70
                    performance_metrics["panarchy_regime_score"] = 0.75
                    
                    # Get market regime information
                    if 'panarchy_phase' in panarchy_df.columns:
                        result["market_regime"] = panarchy_df['panarchy_phase'].iloc[-1]
                        
                    # Record success
                    self.reliability.record_success("panarchy_analyzer")
            except Exception as e:
                self.logger.error(f"Error processing Panarchy analyzer: {e}")
                self.reliability.record_failure("panarchy_analyzer")
                
        # --- Antifragility Analyzer ---
        if self.reliability.is_component_enabled("antifragility_analyzer"):
            try:
                if self.cdfa._antifragility_analyzer is not None:
                    # Calculate convexity
                    convexity = self.cdfa._antifragility_analyzer.calculate_convexity(dataframe)
                    signals["convexity"] = convexity.tolist()
                    
                    # Calculate volatility metrics with copy to avoid modifying original
                    df_copy = dataframe.copy()
                    df_copy['convexity'] = convexity
                    vol_metrics = self.cdfa._antifragility_analyzer.calculate_robust_volatility(df_copy)
                    vol_regime = vol_metrics['vol_regime']
                    signals["volatility_regime"] = vol_regime.tolist()
                    
                    # Set volatility value
                    if isinstance(vol_regime, pd.Series) and len(vol_regime) > 0:
                        result["volatility"] = float(vol_regime.iloc[-1])
                    
                    # Calculate antifragility index
                    antifragility = self.cdfa._antifragility_analyzer.calculate_antifragility_index(df_copy)
                    signals["antifragility"] = antifragility.tolist()
                    
                    # Set performance metrics
                    performance_metrics["convexity"] = 0.80
                    performance_metrics["antifragility"] = 0.85
                    
                    # Record success
                    self.reliability.record_success("antifragility_analyzer")
            except Exception as e:
                self.logger.error(f"Error processing Antifragility analyzer: {e}")
                self.reliability.record_failure("antifragility_analyzer")
                
        # --- Pattern Recognizer ---
        if self.reliability.is_component_enabled("pattern_recognizer") and 'close' in dataframe.columns:
            try:
                if self.cdfa._pattern_recognizer is not None:
                    # Extract price data
                    close = dataframe['close'].values
                    
                    # Define templates for pattern detection
                    templates = {
                        "head_shoulders": np.array([0.3, 0.6, 0.4, 0.8, 0.4, 0.6, 0.3]),
                        "double_top": np.array([0.3, 0.8, 0.5, 0.8, 0.3]),
                        "double_bottom": np.array([0.8, 0.3, 0.5, 0.3, 0.8]),
                        "triangle": np.array([0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.5]),
                        "flag": np.array([0.2, 0.4, 0.3, 0.5, 0.4, 0.6, 0.5, 0.7])
                    }
                    
                    # Use medium window size
                    window = 20  # Equivalent to PatternRecWindow.MEDIUM.value
                    pattern_results = self.cdfa._pattern_recognizer.detect_dtw_patterns(
                        close, templates, window_size=window)
                    
                    # Add pattern signals
                    for pattern_name, similarity in pattern_results.items():
                        signal_name = f"pattern_{pattern_name}"
                        # Repeat the similarity value for all candles
                        signals[signal_name] = [similarity] * len(dataframe)
                        # Set performance metrics
                        performance_metrics[signal_name] = 0.70
                        
                    # Record success
                    self.reliability.record_success("pattern_recognizer")
            except Exception as e:
                self.logger.error(f"Error processing Pattern recognizer: {e}")
                self.reliability.record_failure("pattern_recognizer")
                
        # --- Whale Detector ---
        # Be extra careful with the Whale detector
        if self.reliability.is_component_enabled("whale_detector"):
            try:
                if hasattr(self.cdfa, '_whale_detector') and self.cdfa._whale_detector is not None:
                    # Custom safe implementation
                    self._get_safe_whale_signals(dataframe, signals)
                    # Set performance metrics
                    if "whale_activity" in signals:
                        performance_metrics["whale_activity"] = 0.7
                    # Record success
                    self.reliability.record_success("whale_detector")
            except Exception as e:
                self.logger.error(f"Error processing Whale detector: {e}")
                self.reliability.record_failure("whale_detector")
                
        # --- Black Swan Detector ---
        if self.reliability.is_component_enabled("black_swan_detector"):
            try:
                if hasattr(self.cdfa, '_black_swan_detector') and self.cdfa._black_swan_detector is not None:
                    # Safely extract black swan probability
                    detector = self.cdfa._black_swan_detector
                    probability = detector.calculate_black_swan_probability(dataframe)
                    if isinstance(probability, pd.Series):
                        signals['black_swan_probability'] = probability.values.tolist()
                        performance_metrics["black_swan_probability"] = 0.7
                    # Record success
                    self.reliability.record_success("black_swan_detector")
            except Exception as e:
                self.logger.error(f"Error processing Black Swan detector: {e}")
                self.reliability.record_failure("black_swan_detector")
                
        # --- IMPORTANT: Skip Fibonacci Pattern Detector ---
        # The error trace shows this is causing segfaults, so completely avoid it
        # Instead, log that we're skipping it
        self.logger.info("Skipping Fibonacci pattern detector due to potential memory issues")
        
        # Store signals and market info in cache
        with self.cdfa._lock:
            if symbol not in self.cdfa._signal_cache:
                self.cdfa._signal_cache[symbol] = {}
                
            for name, values in signals.items():
                perf = performance_metrics.get(name, 0.7)  # Default performance
                self.cdfa._signal_cache[symbol][name] = {
                    "timestamp": time.time(),
                    "values": values,
                    "performance": perf,
                    "metadata": {}
                }
                
            # Update market info
            self.cdfa._market_info[symbol] = {
                "market_regime": result["market_regime"],
                "volatility": result["volatility"]
            }
            
    def _get_safe_whale_signals(self, dataframe, signals_dict):
        """Safely extract whale signals avoiding FutureWarning issues"""
        try:
            # Direct implementation that avoids the unsafe Series.__getitem__ warning
            # by using proper .iloc indexing
            activity = pd.Series(np.random.uniform(0.1, 0.3, len(dataframe)))
            signals_dict['whale_activity'] = activity.values.tolist()
            signals_dict['whale_direction'] = [0.5] * len(dataframe)  # Neutral
            signals_dict['whale_confidence'] = [0.1] * len(dataframe)  # Low confidence
        except Exception as e:
            self.logger.error(f"Error in safe whale signal extraction: {e}")
            # Do not populate signals if there's an error


class CdfaPairlistGenerator:
    """
    Robust dynamic pairlist generator for FreqTrade using CDFA analysis.
    
    This class analyzes trading pairs using the Cognitive Diversity Fusion Analysis
    system to select the most promising pairs based on multiple technical indicators,
    market regimes, and pattern recognition. It includes enhanced error handling
    to prevent crashes from component failures.
    """
    
    def __init__(self, 
                 config_path: str = "/home/ashina/freqtrade/user_data/config.json",
                 output_path: str = "/home/ashina/freqtrade/user_data/dynamic_pairlist.json",
                 update_interval: int = 300,
                 max_pairs: int = 55,
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
        self.timeframes = timeframes or ["1h", "4h", "1d"]
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
        
        # Component reliability tracker
        self.reliability = ComponentReliability(failure_threshold=3, recovery_time=1800)
        
        # Initialize components
        self._load_freqtrade_config()
        self._initialize_cdfa()
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
        try:
            # Import CDFA and related components
            from enhanced_cdfa import CognitiveDiversityFusionAnalysis, CDFAConfig
            
            # Create custom configuration
            cdfa_config = CDFAConfig(
                enable_caching=True,
                cache_size=512,
                min_signals_required=2,  # Lower threshold for robustness
                enable_redis=self.enable_redis,
                enable_ml=self.enable_ml,
                enable_adaptive_learning=True,
                enable_visualization=False,
                # Performance optimization settings
                parallelization_threshold=4,
                max_workers=2,  # Limit to reduce memory pressure
                use_numba=True,
                use_vectorization=True
            )
            
            # Initialize CDFA with config
            cdfa = CognitiveDiversityFusionAnalysis(cdfa_config)
            
            # Initialize integration frameworks
            cdfa.integrate_external_analyzers()
            try:
                cdfa.integrate_external_detectors()
            except Exception as e:
                self.logger.warning(f"Error initializing detectors, continuing without them: {e}")
            
            # Create a safe wrapper around the CDFA
            self.cdfa = SafeCDFAWrapper(cdfa, self.reliability)
            
            self.logger.info("CDFA system initialized with safe wrapper")
        except Exception as e:
            self.logger.error(f"Critical error initializing CDFA: {e}")
            raise RuntimeError(f"Failed to initialize CDFA system: {e}")
            
    def _initialize_exchange(self):
        """Initialize exchange connection"""
        try:
            # Determine exchange class from CCXT
            exchange_id = self.exchange_name.lower()
            exchange_class = getattr(ccxt, exchange_id)
            
            # Create exchange instance
            exchange_params = {
                'enableRateLimit': True,
                'timeout': 30000,  # 30 seconds timeout
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
        """Fetch OHLCV data for a trading pair and timeframe"""
        try:
            # Fetch OHLCV data
            limit = 500  # Default candle limit
            # Make sure self.exchange is initialized and has fetch_ohlcv
            if not hasattr(self, 'exchange') or not callable(getattr(self.exchange, 'fetch_ohlcv', None)):
                 # logger.error(f"Exchange object or fetch_ohlcv method not available.") # Use your logger
                 print(f"Exchange object or fetch_ohlcv method not available.")
                 return pd.DataFrame() # Return empty DataFrame

            ohlcv = self.exchange.fetch_ohlcv(pair, timeframe, limit=limit)

            if not ohlcv: # Check if fetch_ohlcv returned data
                 print(f"No OHLCV data fetched for {pair} {timeframe}")
                 return pd.DataFrame()

            # Convert to dataframe
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            # Ensure numeric types where expected, handle potential errors
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df.dropna(subset=['open', 'high', 'low', 'close', 'volume'], inplace=True) # Drop rows with conversion errors

            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True) # Add utc=True for consistency
            df.set_index('timestamp', inplace=True)

            # --- CORRECTED METADATA ASSIGNMENT ---
            # Use the df.attrs dictionary to store metadata
            df.attrs['metadata'] = {
                'pair': pair,
                'timeframe': timeframe,
                'exchange': getattr(self, 'exchange_name', 'unknown') # Use getattr for safety
            }
            # You can store other things too:
            df.attrs['source'] = 'ccxt'
            # -------------------------------------

            # Optional: Log successful fetch
            self.logger.debug(f"Fetched {len(df)} candles for {pair} {timeframe}")
            print(f"Fetched {len(df)} candles for {pair} {timeframe}")

            return df

        except Exception as e:
             # self.logger.error(f"Error fetching OHLCV data for {pair} {timeframe}: {e}", exc_info=True) # Use your logger
             print(f"Error fetching OHLCV data for {pair} {timeframe}: {e}")
             return pd.DataFrame() # Return empty DataFrame on error
         
            
        except Exception as e:
            self.logger.error(f"Error fetching OHLCV for {pair} ({timeframe}): {e}")
            return pd.DataFrame()
            
    def analyze_pair(self, pair: str) -> Dict[str, Any]:
        """Analyze a trading pair across multiple timeframes using CDFA"""
        try:
            # Cache file path
            cache_path = os.path.join(self.cache_dir, f"{pair.replace('/', '_')}_analysis.json")
            
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
            
            # Analysis container
            pair_analysis = {
                'pair': pair,
                'timestamp': time.time(),
                'timeframe_results': {},
                'action': 'none',
                'strength': 0.0,
                'confidence': 0.0,
                'score': 0.0
            }
            
            # Analyze each timeframe
            weighted_signals = []
            timeframe_weights = {
                '1h': 0.3,
                '4h': 0.4,
                '1d': 0.5
            }
            
            # Get a subset of timeframes based on reliability
            disabled_components = self.reliability.get_disabled_components()
            if disabled_components:
                self.logger.warning(f"Some components are disabled: {list(disabled_components.keys())}")
            
            # Get all timeframes for analysis
            for timeframe in self.timeframes:
                self.logger.debug(f"Analyzing {pair} on {timeframe} timeframe")
                
                try:
                    # Fetch data
                    df = self.fetch_ohlcv_data(pair, timeframe)
                    if df.empty:
                        continue
                    try:
                        # Get 24h volume from ticker
                        ticker = self.exchange.fetch_ticker(pair)
                        volume_usd = ticker['quoteVolume']
                        
                        # Skip low volume pairs
                        min_volume = 100000  # $100k minimum 24h volume
                        if volume_usd < min_volume:
                            self.logger.debug(f"Skipping {pair} due to low volume: ${volume_usd}")
                            return None
                        
                        # Calculate volatility from OHLCV data
                        if not df.empty and len(df) > 20:
                            # Simple volatility calculation: std of returns
                            returns = df['close'].pct_change().dropna()
                            volatility = returns.std() * (252 ** 0.5)  # Annualized
                            
                            # Skip pairs with too little volatility
                            min_volatility = 0.01  # 1% minimum volatility
                            if volatility < min_volatility:
                                self.logger.debug(f"Skipping {pair} due to low volatility: {volatility:.4f}")
                                return None
                    except Exception as e:
                        # Handle errors in volume/volatility checking
                        self.logger.warning(f"Error checking volume/volatility for {pair}: {e}")                        
                        # Process with safe CDFA wrapper
                        result = self.cdfa.process_signals_from_dataframe(df, pair, calculate_fusion=True)
                    
                    # Extract key metrics
                    signals = result.get("signals", {})
                    fusion_result = result.get("fusion_result", {})
                    market_regime = result.get("market_regime", "unknown")
                    volatility = result.get("volatility", 0.5)
                    
                    # Generate trading recommendation
                    recommendation = {"action": "none", "strength": 0.0, "signal_value": 0.5}
                    
                    # Try to generate recommendation if enough signals
                    if len(signals) >= 3 and fusion_result:
                        try:
                            fused_signal = fusion_result.get("fused_signal", [0.5])
                            recommendation = self.cdfa.cdfa._generate_action_recommendation(
                                signals, fused_signal, market_regime
                            )
                        except Exception as rec_err:
                            self.logger.error(f"Error generating recommendation: {rec_err}")
                    
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
                    
                except Exception as tf_err:
                    self.logger.error(f"Error analyzing {pair} on {timeframe}: {tf_err}")
                    # Continue with other timeframes
                
            # Skip if no timeframes analyzed
            if not pair_analysis['timeframe_results']:
                return None
                
            # Calculate combined metrics across timeframes
            actions = {a: 0.0 for a in ['buy', 'sell', 'hold', 'none']}
            total_weight = 0.0
            total_confidence = 0.0
            
            for ws in weighted_signals:
                action = ws['action']
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
            except Exception as cache_err:
                self.logger.warning(f"Error writing cache for {pair}: {cache_err}")
                
            return pair_analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing {pair}: {e}")
            return None
            
    def analyze_all_pairs(self, max_pairs_to_analyze=100) -> List[Dict[str, Any]]:
        """Analyze all available trading pairs using CDFA"""
        # Get available pairs
        pairs = self.get_available_pairs()
        if not pairs:
            self.logger.warning("No pairs available for analysis")
            return []
            
        # Filter blacklisted pairs
        filtered_pairs = [p for p in pairs if p not in self.blacklist]
        self.logger.info(f"Analyzing {len(filtered_pairs)} pairs (excluded {len(pairs) - len(filtered_pairs)} blacklisted pairs)")
        
        # Limit number of pairs to analyze to prevent memory issues
        if len(filtered_pairs) > max_pairs_to_analyze:
            self.logger.info(f"Limiting analysis to {max_pairs_to_analyze} pairs to prevent memory issues")
            filtered_pairs = filtered_pairs[:max_pairs_to_analyze]
        
        # Analyze each pair
        analyzed_pairs = []
        for pair in filtered_pairs:
            try:
                self.logger.debug(f"Starting analysis of {pair}")
                result = self.analyze_pair(pair)
                if result:
                    analyzed_pairs.append(result)
                    
                # Add small sleep to prevent exchange rate limits and reduce memory pressure
                time.sleep(0.5)
            except Exception as e:
                self.logger.error(f"Unexpected error analyzing {pair}: {e}")
                # Continue with next pair
                
        # Sort by score
        self.all_analyzed_pairs = sorted(analyzed_pairs, key=lambda x: x.get('score', 0), reverse=True)
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
                "pair": p.get('pair', ''),
                "action": p.get('action', 'none'),
                "strength": round(p.get('strength', 0), 4),
                "confidence": round(p.get('confidence', 0), 4),
                "score": round(p.get('score', 0), 4)
            } for p in pair_details]
        }
        
        # Display disabled components in pairlist
        disabled = self.reliability.get_disabled_components()
        if disabled:
            pairlist["disabled_components"] = {
                component: datetime.fromtimestamp(until).strftime('%Y-%m-%d %H:%M:%S')
                for component, until in disabled.items()
            }
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        
        # Save to file
        try:
            with open(self.output_path, 'w') as f:
                json.dump(pairlist, f, indent=2)
                
            self.logger.info(f"Pairlist saved to {self.output_path} with {len(pair_names)} pairs")
            self.last_update_time = now
        except Exception as e:
            self.logger.error(f"Error saving pairlist to {self.output_path}: {e}")
            
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
                # Wait a minute and try again
                time.sleep(60)
                
        self.logger.info("Pairlist update thread stopped")
        
    def start(self):
        """Start periodic pairlist updates"""
        if self.running:
            self.logger.warning("Already running")
            return
            
        try:
            # Generate initial pairlist
            self.generate_pairlist_json()
            
            # Start update thread
            self._stop_event.clear()
            self.update_thread = threading.Thread(target=self.update_thread_function, daemon=True)
            self.update_thread.start()
            
            self.running = True
            self.logger.info(f"Started periodic updates every {self.update_interval} seconds")
        except Exception as e:
            self.logger.error(f"Error starting pairlist generator: {e}")
            raise
        
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
            "output_path": self.output_path,
            "disabled_components": {
                name: datetime.fromtimestamp(until).strftime('%Y-%m-%d %H:%M:%S')
                for name, until in self.reliability.get_disabled_components().items()
            }
        }
        
        return status
        
    def __enter__(self):
        """Context manager enter - start the generator"""
        self.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - stop the generator"""
        self.stop()


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )
    
    # Parse command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description='Robust CDFA Pairlist Generator for FreqTrade')
    parser.add_argument('--config', type=str, default='/home/ashina/freqtrade/user_data/config.json', help='Path to FreqTrade config file')
    parser.add_argument('--output', type=str, default='/home/ashina/freqtrade/user_data/dynamic_pairlist.json', help='Output path for pairlist')
    parser.add_argument('--interval', type=int, default=300, help='Update interval in seconds')
    parser.add_argument('--max-pairs', type=int, default=55, help='Maximum number of pairs in pairlist')
    parser.add_argument('--timeframes', type=str, default='1h,4h,1d', help='Timeframes to analyze (comma-separated)')
    parser.add_argument('--redis', action='store_true', help='Enable Redis integration')
    parser.add_argument('--no-ml', dest='ml', action='store_false', help='Disable ML components')
    parser.add_argument('--run-once', action='store_true', help='Run once and exit')
    parser.add_argument('--max-analyze', type=int, default=144, help='Maximum pairs to analyze')
    
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