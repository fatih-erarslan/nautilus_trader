#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AdaptiveMarketDataFetcher with FreqTrade Integration (Fixed Version)

An enterprise-grade market data fetcher that dynamically selects trading pairs 
based on feedback from the CDFA framework. Provides seamless integration with 
FreqTrade through formatted pairlists.

Author: Created on May 8, 2025
Updated: May 18, 2025 (Fixed issues with data fetching and enum handling)
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple, Set, TypedDict, Callable, Iterator
from enum import Enum, auto, IntEnum
from dataclasses import dataclass, field, asdict
import threading
import queue
import time
import os
import json
import datetime
import traceback
from pathlib import Path
import concurrent.futures
from threading import Lock, Event
import copy
import warnings
import heapq


try:
    import ccxt
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False
    warnings.warn("CCXT is not available")

# Import base MarketDataFetcher - adjust import path based on your project structure
from cdfa_extensions import MarketDataFetcher

# Type definitions for configuration
class ConfigDict(TypedDict, total=False):
    max_active_pairs: int
    min_active_pairs: int
    update_interval: int
    exploration_ratio: float
    bootstrap_pairs: List[str]
    data_sources: List[str]
    timeframes: List[str]
    default_lookback: str
    feedback_window: int
    score_decay_rate: float
    success_threshold: float
    max_universe_size: int
    discovery_interval: int
    discovery_limit: int
    analyzer_weights: Dict[str, float]
    detector_weights: Dict[str, float]
    metadata_file: str
    cache_dir: str
    enable_auto_update: bool
    enable_auto_discovery: bool
    use_parallel_processing: bool
    max_workers: int
    ccxt_exchanges: List[str]
    preferred_quote_currencies: List[str]
    freqtrade_export_path: Optional[str]
    freqtrade_export_interval: int
    log_level: int
    config_version: str

class OpportunityScore(IntEnum):
    """Scoring levels for trading opportunities."""
    VERY_LOW = 0.0
    LOW = 2.5
    MEDIUM_LOW = 4.0
    MEDIUM = 5.0
    MODERATE = 5.5
    MEDIUM_HIGH = 6.5
    HIGH = 8.0
    VERY_HIGH = 10.0

class MarketRegime(str, Enum):
    """Market regime classifications."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    ACCUMULATION = "accumulation"
    DISTRIBUTION = "distribution"
    RANGING = "ranging"
    VOLATILE = "volatile"
    BREAKOUT = "breakout"
    REVERSAL = "reversal"
    TRENDING = "trending"
    UNKNOWN = "unknown"
    
@dataclass
class PairMetadata:
    """Metadata for a trading pair including feedback scores."""
    symbol: str
    last_updated: float = field(default_factory=time.time)
    opportunity_score: OpportunityScore = OpportunityScore.MODERATE
    analyzer_scores: Dict[str, float] = field(default_factory=dict)
    detector_signals: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    regime_state: MarketRegime = MarketRegime.UNKNOWN
    regime_confidence: float = 0.0
    cycle_strength: float = 0.0
    volatility: float = 0.0
    liquidity: float = 0.0
    is_trending: bool = False
    recent_anomalies: int = 0
    feedback_count: int = 0
    success_rate: float = 0.5  # 0.0 to 1.0
    priority: int = 3  # 1 (highest) to 5 (lowest)
    last_analysis: Dict[str, Any] = field(default_factory=dict)
    custom_metadata: Dict[str, Any] = field(default_factory=dict)
    quote_currency: Optional[str] = None
    base_currency: Optional[str] = None
    
    def __post_init__(self):
        """Parse symbol into base and quote currencies if not provided."""
        if self.base_currency is None or self.quote_currency is None:
            if '/' in self.symbol:
                self.base_currency, self.quote_currency = self.symbol.split('/')
            else:
                # Try to guess the split if standard format
                # Assumes quote currencies like USDT, BTC, ETH
                common_quotes = ['USDT', 'USD', 'BTC', 'ETH', 'BUSD', 'USDC']
                found = False
                for quote in common_quotes:
                    if self.symbol.endswith(quote):
                        self.base_currency = self.symbol[:-len(quote)]
                        self.quote_currency = quote
                        found = True
                        break
                
                if not found:
                    self.base_currency = self.symbol
                    self.quote_currency = "UNKNOWN"
    
    def update_score(self, new_data: Dict[str, Any]):
        """Update metadata with new analysis results."""
        for key, value in new_data.items():
            if hasattr(self, key):
                setattr(self, key, value)
                
        # Update timestamp
        self.last_updated = time.time()
        
    def calculate_priority(self) -> int:
        """Calculate priority score (lower is higher priority)."""
        # Base priority on opportunity score
        if hasattr(self.opportunity_score, 'value'):
            base_priority = 6 - self.opportunity_score.value  # Invert so 1 is highest
        else:
            # Handle case where opportunity_score might be a raw value
            opportunity_score_value = self.opportunity_score
            if isinstance(opportunity_score_value, str) and opportunity_score_value.isdigit():
                opportunity_score_value = int(opportunity_score_value)
            base_priority = 6 - int(opportunity_score_value)
            
        # Adjust based on other factors
        if self.recent_anomalies > 0:
            base_priority -= 1  # Higher priority for anomalies
            
        if self.regime_confidence > 0.8:
            base_priority -= 1  # Higher priority for clear regimes
            
        if self.success_rate > 0.7:
            base_priority -= 1  # Higher priority for successful pairs
            
        # Ensure in valid range
        return max(1, min(5, base_priority))
        
    def get_composite_score(self) -> float:
        """Get composite score for visualization and ranking."""
        # Weighted combination of factors
        score = 0.0
        
        # Base on opportunity score
        if hasattr(self.opportunity_score, 'value'):
            score += self.opportunity_score.value * 20  # 0-100 scale
        else:
            # Handle case where opportunity_score might be a raw value
            opportunity_score_value = self.opportunity_score
            if isinstance(opportunity_score_value, str) and opportunity_score_value.isdigit():
                opportunity_score_value = int(opportunity_score_value)
            score += int(opportunity_score_value) * 20
        
        # Adjust based on regime clarity
        score += self.regime_confidence * 20
        
        # Adjust based on cycle strength
        score += self.cycle_strength * 15
        
        # Adjust based on success rate
        score += self.success_rate * 25
        
        # Normalize to 0-100
        return min(100, max(0, score))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary format."""
        data = asdict(self)
        # Convert enums to string values for serialization
        data['opportunity_score'] = self.opportunity_score.value if hasattr(self.opportunity_score, 'value') else self.opportunity_score
        data['regime_state'] = self.regime_state.value if hasattr(self.regime_state, 'value') else self.regime_state
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PairMetadata':
        """Create PairMetadata from dictionary."""
        # Make a copy of data to avoid modifying the original
        data_copy = data.copy()
        
        # Convert string values back to enums
        if 'opportunity_score' in data_copy:
            if isinstance(data_copy['opportunity_score'], int):
                data_copy['opportunity_score'] = OpportunityScore(data_copy['opportunity_score'])
            elif isinstance(data_copy['opportunity_score'], str) and data_copy['opportunity_score'].isdigit():
                data_copy['opportunity_score'] = OpportunityScore(int(data_copy['opportunity_score']))
        
        if 'regime_state' in data_copy and isinstance(data_copy['regime_state'], str):
            try:
                data_copy['regime_state'] = MarketRegime(data_copy['regime_state'])
            except ValueError:
                data_copy['regime_state'] = MarketRegime.UNKNOWN
        
        # Create instance with available data
        return cls(**data_copy)


class AdaptiveMarketDataFetcher:
    """
    Enhanced MarketDataFetcher with dynamic pair selection based on 
    feedback from CDFA analyzers and detectors. Provides seamless
    integration with FreqTrade through formatted pairlists.
    
    Features:
    - Dynamic trading pair selection based on multi-factor scoring
    - Feedback-driven prioritization system
    - Exploration vs. exploitation balance for discovering opportunities
    - Direct FreqTrade integration via pairlists
    - Thread-safe, robust error handling
    - Configurable for frontend integration
    """
    
    # Class-level attribute for default configuration parameters
    DEFAULT_CONFIG: ConfigDict = {
        # General settings
        "max_active_pairs": 50,        # Maximum actively monitored pairs
        "min_active_pairs": 0,         # Minimum actively monitored pairs - set to 0 to allow empty list
        "update_interval": 3600,       # Seconds between reevaluating pairs
        "exploration_ratio": 0.2,      # Percentage of pairs for exploration
        "bootstrap_pairs": [],         # No default pairs - let user add them manually
        
        # Data fetching settings
        "data_sources": ["yahoo", "cryptocompare", "ccxt"],
        "timeframes": ["1d", "4h", "1h"],
        "default_lookback": "90d",
        
        # Feedback settings
        "feedback_window": 7 * 86400,  # 7 days in seconds
        "score_decay_rate": 0.95,      # Daily score decay rate
        "success_threshold": 0.03,     # 3% price move for success
        
        # Pair universe settings
        "max_universe_size": 500,      # Maximum pairs to track metadata for
        "discovery_interval": 86400,   # How often to discover new pairs
        "discovery_limit": 100,        # Max pairs to add in each discovery
        
        # Analysis settings
        "analyzer_weights": {
            "antifragility": 1.0,
            "fibonacci": 1.0,
            "panarchy": 1.0,
            "soc": 1.0,
            "mra": 1.5  # Higher weight for MRA
        },
        "detector_weights": {
            "black_swan": 1.0,
            "fibonacci_pattern": 1.0,
            "pattern": 1.0,
            "whale": 1.0
        },
        
        # File paths
        "metadata_file": "~/.cdfa/pair_metadata.json",
        "cache_dir": "~/.cdfa/cache",
        
        # Automation settings
        "enable_auto_update": True,
        "enable_auto_discovery": True,
        "use_parallel_processing": True,
        "max_workers": 4,
        
        # Exchange settings
        "ccxt_exchanges": ["binance", "coinbase", "kraken"],
        "preferred_quote_currencies": ["USDT", "USD", "BUSD", "USDC"],
        
        # FreqTrade integration
        "freqtrade_export_path": None,
        "freqtrade_export_interval": 3600,  # 1 hour
        
        # Misc settings
        "log_level": logging.INFO,
        "config_version": "1.0.0"
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the AdaptiveMarketDataFetcher.
        
        Args:
            config: Configuration dictionary to override defaults
        """
        # Set up logging
        self.logger = logging.getLogger(f"{__name__}.AdaptiveMarketDataFetcher")
        
        # Merge with provided config
        self.config = {**self.DEFAULT_CONFIG}
        if config:
            self._update_config(config)
            
        # Set log level
        self.logger.setLevel(self.config["log_level"])
        
        # Resolve file paths
        self._resolve_file_paths()
        
        # Initialize base data fetcher
        self.base_fetcher = MarketDataFetcher()
        
        # Share access to exchanges with base_fetcher
        self.exchanges = {}
        if hasattr(self.base_fetcher, 'exchanges'):
            self.exchanges = self.base_fetcher.exchanges
        
        # Initialize analyzers and detectors
        self.analyzers = {}
        self.detectors = {}
        self.mra = None
        
        # Dictionary of all pair metadata
        self.pair_metadata: Dict[str, PairMetadata] = {}
        self.pair_metadata_lock = Lock()
        
        # Set of actively monitored pairs
        self.active_pairs: Set[str] = set()
        
        # Set of universe pairs (all known pairs)
        self.universe_pairs: Set[str] = set()
        
        # Queue for analyzer results
        self.result_queue: queue.Queue = queue.Queue()
        
        # Initialization flags
        self.analyzers_initialized = False
        self.detectors_initialized = False
        self.mra_initialized = False
        self._is_initialized = False
        
        # Thread control
        self.running = True
        self._update_thread = None
        self._discovery_thread = None
        self._freqtrade_thread = None
        self._result_thread = None
        self._initialize_threads_event = Event()
        
        # Load existing metadata if available
        self._load_metadata()
        
        # Initialize active pairs with bootstrap pairs
        for pair in self.config["bootstrap_pairs"]:
            if pair not in self.pair_metadata:
                with self.pair_metadata_lock:
                    self.pair_metadata[pair] = PairMetadata(symbol=pair)
            self.active_pairs.add(pair)
            self.universe_pairs.add(pair)
        
        # Initialization completed
        self._is_initialized = True
        
        # Initialize components (analyzers, detectors, MRA)
        self._initialize_components()
        
        # Start background threads if auto-update enabled
        if self.config["enable_auto_update"]:
            self._start_background_threads()
    
    def _update_config(self, config: Dict[str, Any]):
        """
        Update configuration with provided values.
        
        Args:
            config: New configuration values
        """
        for key, value in config.items():
            if key in self.config:
                # Handle nested dictionaries
                if isinstance(self.config[key], dict) and isinstance(value, dict):
                    self.config[key].update(value)
                else:
                    self.config[key] = value
            else:
                self.logger.warning(f"Unknown configuration parameter: {key}")
    
    def _resolve_file_paths(self):
        """Resolve and create file paths from configuration."""
        for key in ["metadata_file", "cache_dir"]:
            if isinstance(self.config[key], str):
                # Expand user path (~/...)
                path = os.path.expanduser(self.config[key])
                self.config[key] = path
                
                # Create directories if needed
                if key.endswith("_dir"):
                    os.makedirs(path, exist_ok=True)
                else:
                    dir_path = os.path.dirname(path)
                    if dir_path:
                        os.makedirs(dir_path, exist_ok=True)
    
    def _initialize_components(self):
        """Initialize analyzers, detectors, and MRA."""
        try:
            if not self.analyzers_initialized:
                self.analyzers = self._initialize_analyzers()
                self.analyzers_initialized = True
                
            if not self.detectors_initialized:
                self.detectors = self._initialize_detectors()
                self.detectors_initialized = True
                
            if not self.mra_initialized:
                self.mra = self._initialize_mra()
                self.mra_initialized = True
                
            self._initialize_threads_event.set()
            self.logger.info("All components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing components: {e}")
            self.logger.debug(traceback.format_exc())
    
    def _initialize_analyzers(self) -> Dict[str, Any]:
        """Initialize analyzer instances."""
        analyzers = {}
        try:
            # Import from analyzers directory first (standalone directory structure)
            try:
                import sys
                import os
                sys.path.append(os.path.join(os.path.dirname(__file__), 'analyzers'))
                from antifragility_analyzer import AntifragilityAnalyzer
                from fibonacci_analyzer import FibonacciAnalyzer
                from panarchy_analyzer import PanarchyAnalyzer
                from soc_analyzer import SOCAnalyzer
                
                analyzers = {
                    "antifragility": AntifragilityAnalyzer(),
                    "fibonacci": FibonacciAnalyzer(),
                    "panarchy": PanarchyAnalyzer(),
                    "soc": SOCAnalyzer()
                }
                
                self.logger.info(f"Initialized {len(analyzers)} analyzers")
                return analyzers
            except ImportError:
                # Try to import analyzers from cdfa_extensions
                from cdfa_extensions.analyzers import (
                    AntifragilityAnalyzer, FibonacciAnalyzer,
                    PanarchyAnalyzer, SOCAnalyzer
                )
                
                analyzers = {
                    "antifragility": AntifragilityAnalyzer(),
                    "fibonacci": FibonacciAnalyzer(),
                    "panarchy": PanarchyAnalyzer(),
                    "soc": SOCAnalyzer()
                }
                
                self.logger.info(f"Initialized {len(analyzers)} analyzers")
            
        except ImportError as e:
            self.logger.warning(f"Could not import analyzers: {e}")
            
        return analyzers
    
    def _initialize_detectors(self) -> Dict[str, Any]:
        """Initialize detector instances."""
        detectors = {}
        try:
            # Import from detectors directory first (standalone directory structure)
            try:
                import sys
                import os
                sys.path.append(os.path.join(os.path.dirname(__file__), 'detectors'))
                from black_swan_detector import BlackSwanDetector
                from fibonacci_pattern_detector import FibonacciPatternDetector
                from pattern_recognizer import PatternRecognizer
                from whale_detector import WhaleDetector
                
                detectors = {
                    "black_swan": BlackSwanDetector(),
                    "fibonacci_pattern": FibonacciPatternDetector(),
                    "pattern": PatternRecognizer(),
                    "whale": WhaleDetector()
                }
                
                self.logger.info(f"Initialized {len(detectors)} detectors")
                return detectors
            except ImportError:
                # Try to import detectors from cdfa_extensions
                from cdfa_extensions.detectors import (
                    BlackSwanDetector, FibonacciPatternDetector,
                    PatternRecognizer, WhaleDetector
                )
                
                detectors = {
                    "black_swan": BlackSwanDetector(),
                    "fibonacci_pattern": FibonacciPatternDetector(),
                    "pattern": PatternRecognizer(),
                    "whale": WhaleDetector()
                }
                
                self.logger.info(f"Initialized {len(detectors)} detectors")
            
        except ImportError as e:
            self.logger.warning(f"Could not import detectors: {e}")
            
        return detectors
    
    def _initialize_mra(self) -> Any:
        """Initialize MultiResolutionAnalyzer."""
        try:
            # Try to import MRA from cdfa_extensions
            from cdfa_extensions.mra_analyzer import MultiResolutionAnalyzer
            
            mra = MultiResolutionAnalyzer()
            self.logger.info("Initialized MultiResolutionAnalyzer")
            return mra
            
        except ImportError as e:
            self.logger.warning(f"Could not import MultiResolutionAnalyzer: {e}")
            return None
    
    def _start_background_threads(self):
        """Start background processing threads."""
        # Create and start update thread
        self._update_thread = threading.Thread(
            target=self._update_loop,
            daemon=True,
            name="MarketDataFetcher-Update"
        )
        self._update_thread.start()
        
        # Create and start discovery thread
        if self.config["enable_auto_discovery"]:
            self._discovery_thread = threading.Thread(
                target=self._discovery_loop,
                daemon=True,
                name="MarketDataFetcher-Discovery"
            )
            self._discovery_thread.start()
        
        # Create and start FreqTrade export thread if path is configured
        if self.config.get("freqtrade_export_path"):
            self._freqtrade_thread = threading.Thread(
                target=self._freqtrade_export_loop,
                daemon=True,
                name="MarketDataFetcher-FreqTrade"
            )
            self._freqtrade_thread.start()
        
        # Create and start result processing thread
        self._result_thread = threading.Thread(
            target=self._result_processing_loop,
            daemon=True,
            name="MarketDataFetcher-Results"
        )
        self._result_thread.start()
        
        self.logger.info("Background threads started")
    
    def _load_metadata(self):
        """Load pair metadata from file."""
        metadata_path = self.config["metadata_file"]
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    data = json.load(f)
                    
                with self.pair_metadata_lock:
                    for symbol, meta_dict in data.items():
                        try:
                            # Convert from dict to PairMetadata
                            self.pair_metadata[symbol] = PairMetadata.from_dict(meta_dict)
                            
                            # Add to universe and active pairs
                            self.universe_pairs.add(symbol)
                            self.active_pairs.add(symbol)
                        except Exception as e:
                            self.logger.warning(f"Error loading metadata for {symbol}: {e}")
                    
                self.logger.info(f"Loaded metadata for {len(self.pair_metadata)} pairs ({len(self.active_pairs)} active)")
                
            except Exception as e:
                self.logger.error(f"Error loading metadata: {e}")
                self.logger.debug(traceback.format_exc())
    
    def _save_metadata(self):
        """Save pair metadata to file."""
        try:
            metadata_path = self.config["metadata_file"]
            tmp_path = f"{metadata_path}.tmp"
            
            # Convert to dict for serialization
            data = {}
            with self.pair_metadata_lock:
                for symbol, meta in self.pair_metadata.items():
                    try:
                        data[symbol] = meta.to_dict()
                    except Exception as e:
                        self.logger.warning(f"Error converting metadata for {symbol}: {e}")
            
            # Write to temporary file first
            with open(tmp_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            # Rename to actual file (atomic operation)
            os.replace(tmp_path, metadata_path)
            
            self.logger.info(f"Saved metadata for {len(self.pair_metadata)} pairs")
            
        except Exception as e:
            self.logger.error(f"Error saving metadata: {e}")
            self.logger.debug(traceback.format_exc())
    
    def _update_loop(self):
        """Background thread for updating pair selection."""
        # Wait for initialization to complete
        self._initialize_threads_event.wait()
        
        last_update = time.time()
        
        while self.running:
            try:
                current_time = time.time()
                
                # Check if it's time to update selection
                if current_time - last_update >= self.config["update_interval"]:
                    self._update_pair_selection()
                    self._apply_score_decay()
                    self._save_metadata()
                    last_update = current_time
                
                # Sleep to prevent CPU overuse (but check running flag frequently)
                for _ in range(30):  # Check every 2 seconds for 1 minute
                    if not self.running:
                        break
                    time.sleep(2)
                
            except Exception as e:
                self.logger.error(f"Error in update loop: {e}")
                self.logger.debug(traceback.format_exc())
                time.sleep(60)  # Longer sleep on error
    
    def _discovery_loop(self):
        """Background thread for discovering new pairs."""
        # Wait for initialization to complete
        self._initialize_threads_event.wait()
        
        last_discovery = time.time()
        
        while self.running:
            try:
                current_time = time.time()
                
                # Check if it's time to discover new pairs
                if current_time - last_discovery >= self.config["discovery_interval"]:
                    self._discover_new_pairs()
                    last_discovery = current_time
                
                # Sleep to prevent CPU overuse (but check running flag frequently)
                for _ in range(30):  # Check every 2 seconds for 1 minute
                    if not self.running:
                        break
                    time.sleep(2)
                
            except Exception as e:
                self.logger.error(f"Error in discovery loop: {e}")
                self.logger.debug(traceback.format_exc())
                time.sleep(60)  # Longer sleep on error
    
    def _freqtrade_export_loop(self):
        """Background thread for exporting to FreqTrade."""
        # Wait for initialization to complete
        self._initialize_threads_event.wait()
        
        last_export = time.time()
        
        while self.running:
            try:
                current_time = time.time()
                
                # Check if it's time to export
                if current_time - last_export >= self.config.get("freqtrade_export_interval", 3600):
                    self.export_to_freqtrade_pairlist(
                        limit=self.config.get("max_active_pairs", 50),
                        quote_currencies=self.config.get("preferred_quote_currencies"),
                        filename=self.config.get("freqtrade_export_path")
                    )
                    last_export = current_time
                
                # Sleep to prevent CPU overuse (but check running flag frequently)
                for _ in range(30):  # Check every 2 seconds for 1 minute
                    if not self.running:
                        break
                    time.sleep(2)
                
            except Exception as e:
                self.logger.error(f"Error in FreqTrade export loop: {e}")
                self.logger.debug(traceback.format_exc())
                time.sleep(60)  # Longer sleep on error
    
    def _result_processing_loop(self):
        """Background thread for processing analysis results."""
        # Wait for initialization to complete
        self._initialize_threads_event.wait()
        
        while self.running:
            try:
                # Get result from queue with timeout
                try:
                    result_type, symbol, data = self.result_queue.get(timeout=2.0)
                    
                    # Process result based on type
                    if result_type == "analyzer":
                        analyzer_name = data.pop("analyzer_name", "unknown")
                        self.register_analyzer_feedback(analyzer_name, symbol, data.pop("score", 0.5), data)
                    elif result_type == "detector":
                        detector_name = data.pop("detector_name", "unknown")
                        self.register_detector_signal(detector_name, symbol, data)
                    elif result_type == "mra":
                        self.register_mra_results(symbol, data)
                    
                    # Mark task as done
                    self.result_queue.task_done()
                    
                except queue.Empty:
                    continue
                
            except Exception as e:
                self.logger.error(f"Error in result processing loop: {e}")
                self.logger.debug(traceback.format_exc())
                time.sleep(1)  # Short sleep on error
    
    def _update_pair_selection(self):
        """Update the set of actively monitored pairs."""
        try:
            self.logger.info("Updating pair selection...")
            
            # Calculate priority for all pairs
            with self.pair_metadata_lock:
                for symbol, meta in self.pair_metadata.items():
                    meta.priority = meta.calculate_priority()
            
            # Sort pairs by priority
            prioritized_pairs = []
            with self.pair_metadata_lock:
                for symbol, meta in self.pair_metadata.items():
                    prioritized_pairs.append((meta.priority, -meta.last_updated, symbol))
            
            # Sort by priority (lower is better), then by recency (newer is better)
            prioritized_pairs.sort()
            
            # Determine exploitation set (high priority pairs)
            exploitation_count = int(self.config["max_active_pairs"] * (1 - self.config["exploration_ratio"]))
            exploitation_pairs = set([p[2] for p in prioritized_pairs[:exploitation_count]])
            
            # Determine exploration set (random selection of other pairs)
            exploration_count = self.config["max_active_pairs"] - len(exploitation_pairs)
            exploration_candidates = [p[2] for p in prioritized_pairs[exploitation_count:]]
            
            import random
            random.seed(int(time.time()))  # Ensure randomness
            
            exploration_pairs = set()
            if exploration_candidates and exploration_count > 0:
                exploration_pairs = set(random.sample(
                    exploration_candidates, 
                    min(exploration_count, len(exploration_candidates))
                ))
            
            # Update active pairs
            self.active_pairs = exploitation_pairs.union(exploration_pairs)
            
            # Ensure minimum number of pairs
            if len(self.active_pairs) < self.config["min_active_pairs"] and len(prioritized_pairs) >= self.config["min_active_pairs"]:
                # Add more pairs if needed
                additional_needed = self.config["min_active_pairs"] - len(self.active_pairs)
                additional_pairs = [p[2] for p in prioritized_pairs 
                                  if p[2] not in self.active_pairs][:additional_needed]
                self.active_pairs.update(additional_pairs)
            
            self.logger.info(f"Pair selection updated: {len(self.active_pairs)} pairs selected")
            if exploitation_pairs:
                self.logger.info(f"Top 5 pairs: {list(exploitation_pairs)[:5]}")
            
        except Exception as e:
            self.logger.error(f"Error updating pair selection: {e}")
            self.logger.debug(traceback.format_exc())
    
    def _discover_new_pairs(self):
        """Discover new trading pairs to add to the universe."""
        try:
            self.logger.info("Discovering new pairs...")
            
            # Check if we're under the universe size limit
            if len(self.universe_pairs) >= self.config["max_universe_size"]:
                self.logger.info(f"Universe size limit reached ({len(self.universe_pairs)} pairs)")
                return
            
            # Get top pairs by volume from exchanges
            new_pairs = []
            
            try:
                import ccxt
                
                exchanges = self.config["ccxt_exchanges"]
                for exchange_id in exchanges:
                    try:
                        # Use existing exchange connection if available
                        if exchange_id in self.exchanges:
                            exchange = self.exchanges[exchange_id]
                        else:
                            exchange_class = getattr(ccxt, exchange_id)
                            exchange = exchange_class()
                        
                        markets = exchange.fetch_markets()
                        
                        # Filter for preferred quote currencies
                        quote_currencies = self.config["preferred_quote_currencies"]
                        filtered_markets = []
                        
                        for market in markets:
                            if 'quote' in market and market['quote'] in quote_currencies:
                                if 'active' not in market or market['active']:
                                    filtered_markets.append(market)
                        
                        # Sort by priority (active first, then by quote currency preference)
                        filtered_markets.sort(
                            key=lambda m: (
                                0 if m.get('active', True) else 1,
                                quote_currencies.index(m['quote']) if m['quote'] in quote_currencies else 999
                            )
                        )
                        
                        # Get symbols
                        market_symbols = [m['symbol'] for m in filtered_markets]
                        new_pairs.extend(market_symbols)
                        
                        self.logger.info(f"Found {len(market_symbols)} pairs from {exchange_id}")
                        
                    except Exception as e:
                        self.logger.warning(f"Error fetching pairs from {exchange_id}: {e}")
                        
            except ImportError:
                self.logger.warning("CCXT not available, using fallback discovery method")
                # Fallback to another method (e.g., predefined list or another API)
                new_pairs.extend(self._discover_new_pairs_fallback())
            
            # Deduplicate and standardize format
            new_pairs = set(self._standardize_pair_format(pair) for pair in new_pairs)
            
            # Filter out existing pairs
            new_pairs = new_pairs - self.universe_pairs
            
            # Limit to discovery limit
            new_pairs = list(new_pairs)[:self.config["discovery_limit"]]
            
            # Add to universe and create metadata
            for pair in new_pairs:
                self.universe_pairs.add(pair)
                with self.pair_metadata_lock:
                    self.pair_metadata[pair] = PairMetadata(symbol=pair)
            
            self.logger.info(f"Discovered {len(new_pairs)} new pairs, universe size: {len(self.universe_pairs)}")
            
        except Exception as e:
            self.logger.error(f"Error discovering new pairs: {e}")
            self.logger.debug(traceback.format_exc())
    
    def _discover_new_pairs_fallback(self) -> List[str]:
        """Fallback method for discovering pairs when CCXT is not available."""
        # This could use another data source, API, or a predefined list
        fallback_pairs = [
            "ADA/USDT", "DOT/USDT", "LINK/USDT", "AVAX/USDT", "MATIC/USDT",
            "DOGE/USDT", "UNI/USDT", "AAVE/USDT", "ATOM/USDT", "LTC/USDT",
            "XLM/USDT", "VET/USDT", "FIL/USDT", "XTZ/USDT", "ALGO/USDT",
            "EOS/USDT", "TRX/USDT", "THETA/USDT", "NEO/USDT", "IOTA/USDT"
        ]
        
        # Add to new pairs list
        self.logger.info(f"Using fallback list of {len(fallback_pairs)} pairs")
        return fallback_pairs
    
    def _standardize_pair_format(self, pair: str) -> str:
        """Standardize pair format to BASE/QUOTE."""
        # Already in standard format
        if '/' in pair:
            return pair
            
        # Try to derive from common formats
        # Format: BTCUSDT -> BTC/USDT
        for quote in self.config["preferred_quote_currencies"]:
            if pair.endswith(quote):
                base = pair[:-len(quote)]
                return f"{base}/{quote}"
        
        # Unknown format, return as is
        return pair
    
    def _apply_score_decay(self):
        """Apply decay to scores to prioritize recent information."""
        try:
            # Calculate daily decay factor
            daily_factor = self.config["score_decay_rate"]
            
            # Current time
            current_time = time.time()
            
            # Apply decay to each pair's score
            with self.pair_metadata_lock:
                for symbol, meta in self.pair_metadata.items():
                    # Calculate days since last update
                    days_since_update = (current_time - meta.last_updated) / 86400
                    
                    if days_since_update > 0:
                        # Apply decay to analyzer scores
                        for analyzer, score in meta.analyzer_scores.items():
                            meta.analyzer_scores[analyzer] = score * (daily_factor ** days_since_update)
                        
                        # Decay recent anomalies
                        meta.recent_anomalies = int(meta.recent_anomalies * (daily_factor ** days_since_update))
            
        except Exception as e:
            self.logger.error(f"Error applying score decay: {e}")
            self.logger.debug(traceback.format_exc())
    
    def register_signal_feedback(self, signal: Dict[str, Any], success: Optional[bool] = None):
        """
        Register feedback for a trading signal.
        
        Args:
            signal: Signal dictionary
            success: Whether the signal was successful (None if unknown yet)
        """
        try:
            symbol = signal.get("symbol")
            if not symbol:
                return
            
            with self.pair_metadata_lock:
                # Create metadata if it doesn't exist
                if symbol not in self.pair_metadata:
                    self.pair_metadata[symbol] = PairMetadata(symbol=symbol)
                    self.universe_pairs.add(symbol)
                
                # Get metadata
                meta = self.pair_metadata[symbol]
                
                # Update feedback count
                meta.feedback_count += 1
                
                # Update success rate if success is provided
                if success is not None:
                    # Calculate new success rate with exponential smoothing
                    alpha = 0.3  # Smoothing factor
                    meta.success_rate = alpha * (1.0 if success else 0.0) + (1 - alpha) * meta.success_rate
                
                # Update opportunity score based on signal strength and confidence
                strength = signal.get("strength", 0.5)
                confidence = signal.get("confidence", 0.5)
                
                # Combined score
                combined = strength * confidence
                
                # Map to opportunity score
                if combined > 0.8:
                    meta.opportunity_score = OpportunityScore.VERY_HIGH
                elif combined > 0.6:
                    meta.opportunity_score = OpportunityScore.HIGH
                elif combined > 0.4:
                    meta.opportunity_score = OpportunityScore.MODERATE
                elif combined > 0.2:
                    meta.opportunity_score = OpportunityScore.LOW
                else:
                    meta.opportunity_score = OpportunityScore.VERY_LOW
                
                # Update last updated timestamp
                meta.last_updated = time.time()
                
                # Recalculate priority
                meta.priority = meta.calculate_priority()
            
        except Exception as e:
            self.logger.error(f"Error registering signal feedback: {e}")
            self.logger.debug(traceback.format_exc())
    
    def register_analyzer_feedback(self, analyzer: str, symbol: str, score: float, data: Optional[Dict[str, Any]] = None):
        """
        Register feedback from an analyzer.
        
        Args:
            analyzer: Analyzer name
            symbol: Symbol
            score: Score value (0.0 to 1.0)
            data: Additional data
        """
        try:
            with self.pair_metadata_lock:
                if symbol not in self.pair_metadata:
                    # Create metadata if it doesn't exist
                    self.pair_metadata[symbol] = PairMetadata(symbol=symbol)
                    self.universe_pairs.add(symbol)
                
                # Get metadata
                meta = self.pair_metadata[symbol]
                
                # Update analyzer score
                meta.analyzer_scores[analyzer] = score
                
                # Update metadata if additional data provided
                if data:
                    # Special handling for regime_state if it's a string
                    if "regime_state" in data and isinstance(data["regime_state"], str):
                        try:
                            data["regime_state"] = MarketRegime(data["regime_state"])
                        except ValueError:
                            data["regime_state"] = MarketRegime.UNKNOWN
                    
                    meta.update_score(data)
                
                # Update last updated timestamp
                meta.last_updated = time.time()
                
                # Recalculate priority
                meta.priority = meta.calculate_priority()
            
        except Exception as e:
            self.logger.error(f"Error registering analyzer feedback: {e}")
            self.logger.debug(traceback.format_exc())
    
    def register_detector_signal(self, detector: str, symbol: str, signal: Dict[str, Any]):
        """
        Register a signal from a detector.
        
        Args:
            detector: Detector name
            symbol: Symbol
            signal: Signal data
        """
        try:
            with self.pair_metadata_lock:
                if symbol not in self.pair_metadata:
                    # Create metadata if it doesn't exist
                    self.pair_metadata[symbol] = PairMetadata(symbol=symbol)
                    self.universe_pairs.add(symbol)
                
                # Get metadata
                meta = self.pair_metadata[symbol]
                
                # Initialize detector signals list if needed
                if detector not in meta.detector_signals:
                    meta.detector_signals[detector] = []
                
                # Add signal to list (with timestamp if not already present)
                if "timestamp" not in signal:
                    signal["timestamp"] = time.time()
                    
                meta.detector_signals[detector].append(signal)
                
                # Trim list if needed
                max_signals = 10
                if len(meta.detector_signals[detector]) > max_signals:
                    meta.detector_signals[detector] = meta.detector_signals[detector][-max_signals:]
                
                # Increment recent anomalies
                meta.recent_anomalies += 1
                
                # Update last updated timestamp
                meta.last_updated = time.time()
                
                # Recalculate priority
                meta.priority = meta.calculate_priority()
            
        except Exception as e:
            self.logger.error(f"Error registering detector signal: {e}")
            self.logger.debug(traceback.format_exc())
    
    def register_mra_results(self, symbol: str, results: Dict[str, Any]):
        """
        Register results from the MultiResolutionAnalyzer.
        
        Args:
            symbol: Symbol
            results: MRA results
        """
        try:
            with self.pair_metadata_lock:
                if symbol not in self.pair_metadata:
                    # Create metadata if it doesn't exist
                    self.pair_metadata[symbol] = PairMetadata(symbol=symbol)
                    self.universe_pairs.add(symbol)
                
                # Get metadata
                meta = self.pair_metadata[symbol]
                
                # Update relevant fields from MRA results
                if "regime" in results:
                    try:
                        meta.regime_state = MarketRegime(results["regime"])
                    except ValueError:
                        meta.regime_state = MarketRegime.UNKNOWN
                
                if "regime_confidence" in results:
                    meta.regime_confidence = results["regime_confidence"]
                
                if "cycle_strength" in results:
                    meta.cycle_strength = results["cycle_strength"]
                
                if "trending" in results:
                    meta.is_trending = results["trending"]
                
                # Store all results in last_analysis
                meta.last_analysis["mra"] = results
                
                # Update analyzer score
                meta.analyzer_scores["mra"] = results.get("score", 0.5)
                
                # Update last updated timestamp
                meta.last_updated = time.time()
                
                # Recalculate priority
                meta.priority = meta.calculate_priority()
            
        except Exception as e:
            self.logger.error(f"Error registering MRA results: {e}")
            self.logger.debug(traceback.format_exc())
    
    def fetch_data(self, symbols: Optional[Union[str, List[str]]] = None, 
                  timeframe: str = "1d", lookback: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for given symbols or active pairs using the most appropriate source.
        
        Args:
            symbols: Symbol or list of symbols (None for active pairs)
            timeframe: Timeframe for data
            lookback: Lookback period
            
        Returns:
            Dictionary of symbol -> dataframe
        """
        try:
            # Use active pairs if symbols not provided
            if symbols is None:
                symbols = list(self.active_pairs)
                
            # Ensure symbols is a list
            if isinstance(symbols, str):
                symbols = [symbols]
                
            if not symbols:
                return {}
                
            # Default lookback
            if lookback is None:
                lookback = self.config["default_lookback"]
            
            # Separate symbols by type for optimal data source selection
            crypto_symbols = [s for s in symbols if '/' in s]
            stock_symbols = [s for s in symbols if '/' not in s]
            
            result_dict = {}
            
            # For cryptocurrencies, try specialized sources first
            if crypto_symbols:
                # First try CCXT for real-time crypto data
                if CCXT_AVAILABLE and (self.exchanges or hasattr(self.base_fetcher, 'exchanges')):
                    try:
                        # Use base_fetcher's fetch_crypto_data directly
                        if hasattr(self.base_fetcher, 'fetch_crypto_data'):
                            crypto_data = self.base_fetcher.fetch_crypto_data(
                                crypto_symbols,
                                prefer_ccxt=True,
                                period=lookback,
                                interval=timeframe
                            )
                            result_dict.update(crypto_data)
                            
                            # Remove successfully fetched symbols
                            crypto_symbols = [s for s in crypto_symbols if s not in result_dict]
                        else:
                            self.logger.warning("base_fetcher does not have fetch_crypto_data method")
                    except Exception as e:
                        self.logger.warning(f"Error fetching from CCXT: {e}")
                
                # Then try CryptoCompare for any remaining
                if crypto_symbols and "cryptocompare" in self.config["data_sources"]:
                    try:
                        # Assuming cryptocompare functionality is implemented in the base fetcher
                        if hasattr(self.base_fetcher, 'fetch_data_for_cdfa'):
                            crypto_data = self.base_fetcher.fetch_data_for_cdfa(
                                crypto_symbols,
                                source='cryptocompare',
                                period=lookback,
                                interval=timeframe
                            )
                            result_dict.update(crypto_data)
                            
                            # Remove successfully fetched symbols
                            crypto_symbols = [s for s in crypto_symbols if s not in result_dict]
                    except Exception as e:
                        self.logger.warning(f"Error fetching from CryptoCompare: {e}")
                
                # Finally use Yahoo Finance as fallback for any remaining crypto
                if crypto_symbols:
                    try:
                        if hasattr(self.base_fetcher, 'fetch_yahoo_data'):
                            crypto_data = self.base_fetcher.fetch_yahoo_data(
                                crypto_symbols,
                                period=lookback,
                                interval=timeframe
                            )
                            result_dict.update(crypto_data)
                    except Exception as e:
                        self.logger.warning(f"Error fetching crypto from Yahoo Finance: {e}")
            
            # For stocks, use Yahoo Finance directly
            if stock_symbols:
                try:
                    if hasattr(self.base_fetcher, 'fetch_yahoo_data'):
                        stock_data = self.base_fetcher.fetch_yahoo_data(
                            stock_symbols,
                            period=lookback,
                            interval=timeframe
                        )
                        result_dict.update(stock_data)
                except Exception as e:
                    self.logger.warning(f"Error fetching stocks from Yahoo Finance: {e}")
            
            # Update metadata with volatility and liquidity
            for symbol, df in result_dict.items():
                if symbol in self.pair_metadata:
                    meta = self.pair_metadata[symbol]
                    
                    # Calculate volatility
                    if "returns" in df.columns:
                        meta.volatility = df["returns"].std() * np.sqrt(252)  # Annualized
                        
                    # Estimate liquidity from volume
                    if "volume" in df.columns and "close" in df.columns:
                        avg_volume = df["volume"].mean()
                        avg_price = df["close"].mean()
                        meta.liquidity = avg_volume * avg_price
                        
                    # Update timestamp
                    meta.last_updated = time.time()
            
            return result_dict
            
        except Exception as e:
            self.logger.error(f"Error fetching data: {e}")
            self.logger.debug(traceback.format_exc())
            return {}
    
    def analyze_all_pairs(self, symbols: Optional[List[str]] = None):
        """
        Analyze specified pairs or all active pairs with all analyzers and detectors.
        
        Args:
            symbols: List of symbols to analyze (None for active pairs)
        """
        try:
            # Use active pairs if symbols not provided
            if symbols is None:
                symbols = list(self.active_pairs)
                
            if not symbols:
                self.logger.warning("No pairs available for analysis")
                return
                
            self.logger.info(f"Analyzing {len(symbols)} pairs")
            
            # Fetch data for all pairs
            data_dict = self.fetch_data(symbols)
            
            if not data_dict:
                self.logger.warning("No data available for analysis")
                return
                
            # Process pairs in parallel if enabled
            if self.config["use_parallel_processing"] and len(data_dict) > 1:
                self._analyze_pairs_parallel(data_dict)
            else:
                self._analyze_pairs_sequential(data_dict)
            
            self.logger.info(f"Completed analysis for {len(data_dict)} pairs")
            
        except Exception as e:
            self.logger.error(f"Error analyzing pairs: {e}")
            self.logger.debug(traceback.format_exc())
    
    def _analyze_pairs_sequential(self, data_dict: Dict[str, pd.DataFrame]):
        """
        Analyze pairs sequentially.
        
        Args:
            data_dict: Dictionary of symbol -> dataframe
        """
        # Run MRA analysis
        if self.mra:
            for symbol, df in data_dict.items():
                try:
                    # Run MRA
                    results = self.mra.analyze_regimes(df)
                    
                    # Register results
                    self.register_mra_results(symbol, results)
                    
                except Exception as e:
                    self.logger.error(f"Error in MRA analysis for {symbol}: {e}")
        
        # Run analyzer analysis
        for analyzer_name, analyzer in self.analyzers.items():
            for symbol, df in data_dict.items():
                try:
                    # Run analyzer
                    results = analyzer.analyze(df)
                    
                    # Register results
                    self.register_analyzer_feedback(analyzer_name, symbol, results.get("score", 0.5), results)
                    
                except Exception as e:
                    self.logger.error(f"Error in {analyzer_name} for {symbol}: {e}")
        
        # Run detector analysis
        for detector_name, detector in self.detectors.items():
            for symbol, df in data_dict.items():
                try:
                    # Run detector
                    signals = detector.detect(df)
                    
                    # Register signals
                    for signal in signals:
                        self.register_detector_signal(detector_name, symbol, signal)
                        
                except Exception as e:
                    self.logger.error(f"Error in {detector_name} for {symbol}: {e}")
    
    def _analyze_pairs_parallel(self, data_dict: Dict[str, pd.DataFrame]):
        """
        Analyze pairs in parallel using ThreadPoolExecutor.
        
        Args:
            data_dict: Dictionary of symbol -> dataframe
        """
        max_workers = min(self.config["max_workers"], len(data_dict))
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit MRA analysis tasks
            if self.mra:
                mra_futures = {
                    executor.submit(self._analyze_mra_task, symbol, df): symbol
                    for symbol, df in data_dict.items()
                }
            
            # Submit analyzer tasks
            analyzer_futures = []
            for analyzer_name, analyzer in self.analyzers.items():
                for symbol, df in data_dict.items():
                    future = executor.submit(
                        self._analyze_analyzer_task,
                        analyzer_name,
                        analyzer,
                        symbol,
                        df
                    )
                    analyzer_futures.append(future)
            
            # Submit detector tasks
            detector_futures = []
            for detector_name, detector in self.detectors.items():
                for symbol, df in data_dict.items():
                    future = executor.submit(
                        self._analyze_detector_task,
                        detector_name,
                        detector,
                        symbol,
                        df
                    )
                    detector_futures.append(future)
            
            # Wait for all futures to complete
            for future in concurrent.futures.as_completed(analyzer_futures + detector_futures):
                try:
                    future.result()
                except Exception as e:
                    self.logger.error(f"Error in analysis task: {e}")
    
    def _analyze_mra_task(self, symbol: str, df: pd.DataFrame):
        """
        MRA analysis task for parallel execution.
        
        Args:
            symbol: Symbol
            df: Data frame
        """
        try:
            # Run MRA
            results = self.mra.analyze_regimes(df)
            
            # Add to result queue
            self.result_queue.put(("mra", symbol, results))
            
        except Exception as e:
            self.logger.error(f"Error in MRA analysis task for {symbol}: {e}")
            self.logger.debug(traceback.format_exc())
    
    def _analyze_analyzer_task(self, analyzer_name: str, analyzer: Any, symbol: str, df: pd.DataFrame):
        """
        Analyzer task for parallel execution.
        
        Args:
            analyzer_name: Analyzer name
            analyzer: Analyzer instance
            symbol: Symbol
            df: Data frame
        """
        try:
            # Run analyzer
            results = analyzer.analyze(df)
            
            # Add analyzer name to results
            results["analyzer_name"] = analyzer_name
            
            # Add to result queue
            self.result_queue.put(("analyzer", symbol, results))
            
        except Exception as e:
            self.logger.error(f"Error in {analyzer_name} task for {symbol}: {e}")
            self.logger.debug(traceback.format_exc())
    
    def _analyze_detector_task(self, detector_name: str, detector: Any, symbol: str, df: pd.DataFrame):
        """
        Detector task for parallel execution.
        
        Args:
            detector_name: Detector name
            detector: Detector instance
            symbol: Symbol
            df: Data frame
        """
        try:
            # Run detector
            signals = detector.detect(df)
            
            # Add to result queue
            for signal in signals:
                signal["detector_name"] = detector_name
                self.result_queue.put(("detector", symbol, signal))
                
        except Exception as e:
            self.logger.error(f"Error in {detector_name} task for {symbol}: {e}")
            self.logger.debug(traceback.format_exc())
    
    def get_pair_rankings(self, limit: int = 20) -> List[Tuple[str, float]]:
        """
        Get top ranked pairs by composite score.
        
        Args:
            limit: Maximum number of pairs to return
            
        Returns:
            List of (symbol, score) tuples
        """
        try:
            # Calculate composite scores
            scores = []
            with self.pair_metadata_lock:
                for symbol, meta in self.pair_metadata.items():
                    scores.append((symbol, meta.get_composite_score()))
            
            # Sort by score (descending)
            scores.sort(key=lambda x: x[1], reverse=True)
            
            # Return top pairs
            return scores[:limit]
            
        except Exception as e:
            self.logger.error(f"Error getting pair rankings: {e}")
            self.logger.debug(traceback.format_exc())
            return []
    
    def get_pair_metadata(self, symbol: str) -> Optional[PairMetadata]:
        """
        Get metadata for a specific pair.
        
        Args:
            symbol: Symbol
            
        Returns:
            Pair metadata or None if not found
        """
        with self.pair_metadata_lock:
            return self.pair_metadata.get(symbol)
    
    def get_active_pairs(self) -> List[str]:
        """
        Get list of actively monitored pairs.
        
        Returns:
            List of active pairs
        """
        return list(self.active_pairs)
    
    def get_universe_pairs(self) -> List[str]:
        """
        Get list of all known pairs in the universe.
        
        Returns:
            List of universe pairs
        """
        return list(self.universe_pairs)
    
    def export_to_freqtrade_pairlist(self, 
                                   limit: int = 50, 
                                   quote_currencies: Optional[List[str]] = None,
                                   filename: Optional[str] = None) -> Dict[str, List[str]]:
        """
        Export top pairs to FreqTrade pairlist JSON format.
        
        Args:
            limit: Maximum number of pairs to include
            quote_currencies: Filter by quote currencies (e.g., ["USDT"])
            filename: Optional filename to save JSON
            
        Returns:
            Dictionary in FreqTrade pairlist format
        """
        try:
            # Get top ranked pairs
            ranked_pairs = self.get_pair_rankings(limit=limit*2)  # Get extra pairs for filtering
            
            # Extract symbols
            pairs = [symbol for symbol, _ in ranked_pairs]
            
            # Filter by quote currency if specified
            if quote_currencies:
                filtered_pairs = []
                for pair in pairs:
                    # Check if pair has one of the specified quote currencies
                    if '/' in pair:
                        _, quote = pair.split('/')
                        if quote in quote_currencies:
                            filtered_pairs.append(pair)
                    else:
                        # Try to get quote currency from metadata
                        meta = self.get_pair_metadata(pair)
                        if meta and meta.quote_currency in quote_currencies:
                            filtered_pairs.append(pair)
                
                pairs = filtered_pairs
            
            # Limit to requested number
            pairs = pairs[:limit]
            
            # Create FreqTrade format
            freqtrade_format = {"pairs": pairs}
            
            # Save to file if requested
            if filename:
                # Ensure directory exists
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                
                # Write to temp file first
                temp_filename = f"{filename}.tmp"
                with open(temp_filename, 'w') as f:
                    json.dump(freqtrade_format, f, indent=2)
                
                # Rename to actual file (atomic operation)
                os.replace(temp_filename, filename)
                
                self.logger.info(f"Exported {len(pairs)} pairs to {filename}")
            
            return freqtrade_format
            
        except Exception as e:
            self.logger.error(f"Error exporting to FreqTrade format: {e}")
            self.logger.debug(traceback.format_exc())
            return {"pairs": []}
    
    def export_extended_freqtrade_data(self, limit: int = 50, filename: Optional[str] = None) -> Dict[str, Any]:
        """
        Export top pairs with extended metadata for advanced FreqTrade integration.
        
        Args:
            limit: Maximum number of pairs to include
            filename: Optional filename to save JSON
            
        Returns:
            Dictionary with extended FreqTrade data
        """
        try:
            # Get standard pairlist
            standard_pairlist = self.export_to_freqtrade_pairlist(limit=limit)
            pairs = standard_pairlist["pairs"]
            
            # Add extended data
            extended_data = {
                "pairs": pairs,
                "metadata": {},
                "regime_data": {},
                "opportunity_scores": {},
                "generator": "CDFA AdaptiveMarketDataFetcher",
                "timestamp": datetime.datetime.now().isoformat(),
                "version": self.config["config_version"]
            }
            
            # Fill in metadata for each pair
            with self.pair_metadata_lock:
                for pair in pairs:
                    if pair in self.pair_metadata:
                        meta = self.pair_metadata[pair]
                        
                        # Basic metadata
                        extended_data["metadata"][pair] = {
                            "volatility": meta.volatility,
                            "liquidity": meta.liquidity,
                            "success_rate": meta.success_rate,
                            "recent_anomalies": meta.recent_anomalies,
                            "base_currency": meta.base_currency,
                            "quote_currency": meta.quote_currency
                        }
                        
                        # Regime data
                        regime_state_value = meta.regime_state.value if hasattr(meta.regime_state, 'value') else meta.regime_state
                        extended_data["regime_data"][pair] = {
                            "regime": regime_state_value,
                            "confidence": meta.regime_confidence,
                            "is_trending": meta.is_trending,
                            "cycle_strength": meta.cycle_strength
                        }
                        
                        # Opportunity score
                        extended_data["opportunity_scores"][pair] = meta.get_composite_score()
            
            # Save to file if requested
            if filename:
                # Ensure directory exists
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                
                # Write to temp file first
                temp_filename = f"{filename}.tmp"
                with open(temp_filename, 'w') as f:
                    json.dump(extended_data, f, indent=2)
                
                # Rename to actual file (atomic operation)
                os.replace(temp_filename, filename)
            
            return extended_data
            
        except Exception as e:
            self.logger.error(f"Error exporting extended FreqTrade data: {e}")
            self.logger.debug(traceback.format_exc())
            return {"pairs": []}
    
    def get_config_parameters(self) -> Dict[str, Any]:
        """
        Get configuration parameters for frontend integration.
        
        Returns:
            Dictionary of configuration parameters with metadata
        """
        return {
            "max_active_pairs": {
                "type": "int",
                "min": 5,
                "max": 200,
                "default": self.DEFAULT_CONFIG["max_active_pairs"],
                "description": "Maximum number of actively monitored pairs"
            },
            "min_active_pairs": {
                "type": "int",
                "min": 1,
                "max": 100,
                "default": self.DEFAULT_CONFIG["min_active_pairs"],
                "description": "Minimum number of actively monitored pairs"
            },
            "update_interval": {
                "type": "int",
                "min": 300,
                "max": 86400,
                "default": self.DEFAULT_CONFIG["update_interval"],
                "description": "Seconds between reevaluating pairs"
            },
            "exploration_ratio": {
                "type": "float",
                "min": 0.0,
                "max": 0.5,
                "default": self.DEFAULT_CONFIG["exploration_ratio"],
                "description": "Percentage of pairs for exploration"
            },
            "preferred_quote_currencies": {
                "type": "list",
                "options": ["USDT", "USD", "BUSD", "USDC", "BTC", "ETH"],
                "default": self.DEFAULT_CONFIG["preferred_quote_currencies"],
                "description": "Preferred quote currencies"
            },
            "enable_auto_update": {
                "type": "bool",
                "default": self.DEFAULT_CONFIG["enable_auto_update"],
                "description": "Automatically update pair selection"
            },
            "enable_auto_discovery": {
                "type": "bool",
                "default": self.DEFAULT_CONFIG["enable_auto_discovery"],
                "description": "Automatically discover new pairs"
            },
            "analyzer_weights": {
                "type": "dict",
                "default": self.DEFAULT_CONFIG["analyzer_weights"],
                "description": "Weights for different analyzers"
            }
        }
    
    def update_config_parameter(self, parameter: str, value: Any) -> bool:
        """
        Update a configuration parameter.
        
        Args:
            parameter: Parameter name
            value: New value
            
        Returns:
            Success flag
        """
        try:
            if parameter in self.config:
                old_value = self.config[parameter]
                self.config[parameter] = value
                self.logger.info(f"Updated parameter {parameter}: {old_value} -> {value}")
                return True
            else:
                self.logger.warning(f"Unknown parameter: {parameter}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error updating parameter {parameter}: {e}")
            return False
    
    def stop(self):
        """Stop background threads and clean up resources."""
        self.logger.info("Stopping AdaptiveMarketDataFetcher...")
        self.running = False
        
        # Save metadata before stopping
        try:
            self._save_metadata()
        except Exception as e:
            self.logger.error(f"Error saving metadata: {e}")
        
        # Wait for threads to terminate
        threads = [
            self._update_thread,
            self._discovery_thread,
            self._freqtrade_thread,
            self._result_thread
        ]
        
        for thread in threads:
            if thread and thread.is_alive():
                thread.join(timeout=5.0)
                
        self.logger.info("AdaptiveMarketDataFetcher stopped")
    
    def __del__(self):
        """Destructor to ensure clean shutdown."""
        if hasattr(self, 'running') and self.running:
            self.stop()