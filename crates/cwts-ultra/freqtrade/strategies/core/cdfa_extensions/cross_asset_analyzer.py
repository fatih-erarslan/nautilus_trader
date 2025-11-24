#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cross-Asset Analyzer for CDFA Extensions

Provides comprehensive cross-asset analysis capabilities including:
- Correlation matrix calculation with time-varying methods
- Lead-lag relationship identification
- Contagion risk assessment
- Market structure visualization via MST
- Flow of funds analysis
- Cross-asset regime consistency detection

Author: Created on May 6, 2025
"""

import logging
import time
import numpy as np
import pandas as pd
import threading
from typing import Dict, List, Tuple, Optional, Union, Any, Callable, Set, TYPE_CHECKING
from enum import Enum, auto
from dataclasses import dataclass, field
import warnings
import queue
from datetime import datetime, timedelta
import uuid
import os
from collections import defaultdict
import json

# Import from cdfa_extensions
from .hw_acceleration import HardwareAccelerator

import importlib
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from .hw_acceleration import HardwareAccelerator

# Add a lazy-loading function
def _get_hardware_accelerator():
    """Get the hardware accelerator instance."""
    try:
        hw_module = importlib.import_module('.hw_acceleration', package='cdfa_extensions')
        return hw_module.HardwareAccelerator()
    except ImportError as e:
        import warnings
        warnings.warn(f"Failed to import HardwareAccelerator: {e}", DeprecationWarning, DeprecationWarning)
        return None
# ---- Optional dependencies with graceful fallbacks ----

# SciPy for statistical computations
try:
    from scipy.stats import spearmanr, kendalltau, pearsonr
    from scipy.cluster import hierarchy
    from scipy.spatial.distance import squareform, pdist
    import scipy.signal as signal
    from scipy.ndimage import gaussian_filter1d
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("SciPy not available. Some cross-asset analysis methods will be limited.", DeprecationWarning, DeprecationWarning)

# NetworkX for graph-based analysis
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    warnings.warn("NetworkX not available. Graph-based analysis will be limited.", DeprecationWarning, DeprecationWarning)

# PyWavelets for multi-resolution analysis
try:
    import pywt
    PYWAVELETS_AVAILABLE = True
except ImportError:
    PYWAVELETS_AVAILABLE = False
    warnings.warn("PyWavelets not available. Wavelet-based analysis will be limited.", DeprecationWarning, DeprecationWarning)

# Visualization tools
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    warnings.warn("Matplotlib not available. Visualization features will be limited.", DeprecationWarning, DeprecationWarning)

class CorrelationMethod(Enum):
    """Methods for calculating correlation between assets."""
    PEARSON = auto()
    SPEARMAN = auto()
    KENDALL = auto()
    DISTANCE = auto()
    TAIL = auto()
    WAVELET = auto()
    
    def __str__(self) -> str:
        return self.name.lower()
    
    @classmethod
    def from_string(cls, s: str) -> 'CorrelationMethod':
        """Create enum from string representation."""
        s_upper = s.upper()
        for item in cls:
            if item.name == s_upper:
                return item
        for item in cls:
            if item.name.startswith(s_upper):
                return item
        raise ValueError(f"Unknown CorrelationMethod: {s}")

class TimeScale(Enum):
    """Time scales for analysis."""
    TICK = auto()
    MINUTE = auto()
    HOUR = auto()
    DAY = auto()
    WEEK = auto()
    MONTH = auto()
    
    def __str__(self) -> str:
        return self.name.lower()
    
    @classmethod
    def from_string(cls, s: str) -> 'TimeScale':
        """Create enum from string representation."""
        s_upper = s.upper()
        for item in cls:
            if item.name == s_upper:
                return item
        for item in cls:
            if item.name.startswith(s_upper):
                return item
        raise ValueError(f"Unknown TimeScale: {s}")

class AssetClass(Enum):
    """Asset classes for categorization."""
    STOCK = auto()
    BOND = auto()
    COMMODITY = auto()
    CURRENCY = auto()
    CRYPTO = auto()
    REAL_ESTATE = auto()
    INDEX = auto()
    ETF = auto()
    DERIVATIVE = auto()
    OTHER = auto()
    
    def __str__(self) -> str:
        return self.name.lower()
    
    @classmethod
    def from_string(cls, s: str) -> 'AssetClass':
        """Create enum from string representation."""
        s_upper = s.upper()
        for item in cls:
            if item.name == s_upper:
                return item
        for item in cls:
            if item.name.startswith(s_upper):
                return item
        raise ValueError(f"Unknown AssetClass: {s}")

@dataclass
class AssetInfo:
    """Information about an asset."""
    symbol: str
    name: str = ""
    asset_class: AssetClass = AssetClass.OTHER
    sub_class: str = ""
    sector: str = ""
    industry: str = ""
    country: str = ""
    exchange: str = ""
    currency: str = "USD"
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CrossAssetRelationship:
    """Relationship between assets."""
    from_symbol: str
    to_symbol: str
    relationship_type: str
    strength: float
    direction: float  # -1 to 1, negative for inverse
    lag: int = 0  # Lag in time units
    confidence: float = 0.7
    method: str = ""
    timeframe: str = ""
    timestamp: float = field(default_factory=time.time)
    p_value: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ContagionRisk:
    """Contagion risk assessment."""
    symbol: str
    risk_score: float  # 0 to 1
    impact_symbols: List[Tuple[str, float]]  # (symbol, impact)
    systemic_impact: float  # 0 to 1
    timeframe: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

class CrossAssetAnalyzer:
    """
    Cross-Asset Analyzer for the CDFA system.
    
    Analyzes relationships between multiple assets to identify correlations,
    lead-lag relationships, contagion risks, and market structure changes.
    """
    
    def __init__(self, hw_accelerator: Optional[Any] = None,
                config: Optional[Dict[str, Any]] = None,
                log_level: int = logging.INFO):
        """
        Initialize the cross-asset analyzer.
        
        Args:
            hw_accelerator: Optional hardware accelerator
            config: Configuration parameters
            log_level: Logging level
        """
        # Setup logging
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.setLevel(log_level)
        
        # Initialize hardware accelerator
        self.hw_accelerator = hw_accelerator if hw_accelerator is not None else _get_hardware_accelerator()        
        # Default configuration
        self.default_config = {
            # Correlation parameters
            "default_correlation_method": "pearson",
            "default_timeframe": "1d",
            "correlation_threshold": 0.5,
            "use_absolute_correlation": True,
            "correlation_window": 30,  # Days
            "rolling_window": 20,  # Days
            "ewma_alpha": 0.1,  # EWMA decay factor
            "tail_threshold": 0.1,  # Threshold for tail events
            
            # Lead-lag parameters
            "max_lag": 10,  # Maximum lag to check
            "min_lag_significance": 0.05,  # p-value threshold
            "lag_step": 1,  # Step size for lag checks
            
            # Graph parameters
            "mst_threshold": 0.3,  # MST edge weight threshold
            "graph_layout": "spring",  # Graph layout algorithm
            "use_community_detection": True,
            "max_communities": 10,
            
            # Contagion parameters
            "contagion_threshold": 0.7,
            "max_contagion_path": 3,  # Maximum path length for contagion
            "systemic_importance_threshold": 0.6,
            
            # Performance parameters
            "cache_results": True,
            "cache_ttl": 3600,  # 1 hour
            "use_parallel": True,
            "chunk_size": 100,
            "max_symbols": 500,
            
            # Visualization parameters
            "heatmap_cmap": "coolwarm",
            "network_cmap": "viridis",
            "plot_width": 10,
            "plot_height": 8,
            "plot_dpi": 100
        }
        
        # Merge with provided config
        self.config = {**self.default_config, **(config or {})}
        
        # Initialize state
        self._lock = threading.RLock()
        self._asset_info = {}  # symbol -> AssetInfo
        self._price_data = {}  # symbol -> DataFrame
        self._relationship_cache = {}  # (symbol1, symbol2, type) -> CrossAssetRelationship
        self._contagion_cache = {}  # symbol -> ContagionRisk
        self._cache_timestamps = {}  # key -> timestamp
        self._mst_cache = {}  # timeframe -> (MST, timestamp)
        
        # Background processing 
        self._is_running = True
        self._processing_queue = queue.PriorityQueue()
        self._processing_thread = None
        
        # Start background thread if needed
        if self.config["use_parallel"]:
            self._start_background_processing()
            
        self.logger.info("CrossAssetAnalyzer initialized")

        
    def _start_background_processing(self):
        """Start background processing thread."""
        if self._processing_thread is not None and self._processing_thread.is_alive():
            return
            
        self._processing_thread = threading.Thread(
            target=self._background_worker,
            daemon=True,
            name="CrossAssetAnalyzerWorker"
        )
        self._processing_thread.start()
        
    def _background_worker(self):
        """Background worker for processing analysis tasks."""
        self.logger.info("Starting background worker thread")
        
        while self._is_running:
            try:
                # Get next task from queue
                try:
                    # Format: (priority, task_function, args, kwargs, callback)
                    _, task_func, args, kwargs, callback = self._processing_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Process task
                try:
                    result = task_func(*args, **kwargs)
                    
                    # Call callback if provided
                    if callback is not None:
                        try:
                            callback(result)
                        except Exception as e:
                            self.logger.error(f"Error in task callback: {e}")
                except Exception as e:
                    self.logger.error(f"Error processing task: {e}")
                    if callback is not None:
                        try:
                            callback(None, exception=e)
                        except Exception as e2:
                            self.logger.error(f"Error in error callback: {e2}")
                
                # Mark task as done
                self._processing_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"Error in background worker: {e}")
                time.sleep(0.1)
                
        self.logger.info("Background worker thread stopped")
        
    def add_asset_info(self, asset_info: Union[AssetInfo, Dict[str, Any]]) -> bool:
        """
        Add asset information to the analyzer.
        
        Args:
            asset_info: Asset information object or dictionary
            
        Returns:
            Success flag
        """
        try:
            # Convert dict to AssetInfo if needed
            if isinstance(asset_info, dict):
                # Convert asset_class string to enum if needed
                if "asset_class" in asset_info and isinstance(asset_info["asset_class"], str):
                    asset_info["asset_class"] = AssetClass.from_string(asset_info["asset_class"])
                    
                asset_info = AssetInfo(**asset_info)
                
            # Store in asset info dictionary
            with self._lock:
                self._asset_info[asset_info.symbol] = asset_info
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding asset info: {e}")
            return False
            
    def get_asset_info(self, symbol: str) -> Optional[AssetInfo]:
        """
        Get asset information.
        
        Args:
            symbol: Asset symbol
            
        Returns:
            Asset information or None if not found
        """
        with self._lock:
            return self._asset_info.get(symbol)
            
    def add_price_data(self, symbol: str, df: pd.DataFrame, replace: bool = False) -> bool:
        """
        Add price data for an asset.
        
        Args:
            symbol: Asset symbol
            df: Price dataframe (should have OHLCV columns)
            replace: Whether to replace existing data
            
        Returns:
            Success flag
        """
        try:
            with self._lock:
                # Check if we already have data
                existing_df = self._price_data.get(symbol)
                
                if existing_df is not None and not replace:
                    # Merge with existing data
                    merged_df = pd.concat([existing_df, df])
                    
                    # Remove duplicates
                    merged_df = merged_df[~merged_df.index.duplicated(keep='last')]
                    
                    # Sort by index
                    merged_df = merged_df.sort_index()
                    
                    # Store merged dataframe
                    self._price_data[symbol] = merged_df
                else:
                    # Store new dataframe
                    self._price_data[symbol] = df.copy()
                    
            # Invalidate cache for this symbol
            self._invalidate_cache_for_symbol(symbol)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding price data: {e}")
            return False
                
    def get_price_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Get price data for an asset with improved validation.
        
        Args:
            symbol: Asset symbol
            
        Returns:
            Price dataframe or None if not found
        """
        with self._lock:
            df = self._price_data.get(symbol)
            
            if df is None:
                return None
                
            # Check for empty dataframe
            if df.empty:
                return None
                
            # Ensure DataFrame has required columns for correlation analysis
            if 'close' not in df.columns and 'Close' not in df.columns:
                # Try to identify a suitable price column
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                
                if not numeric_cols.empty:
                    # Rename first numeric column to 'close' for consistency
                    df = df.copy()
                    df['close'] = df[numeric_cols[0]]
                    self.logger.debug(f"Created 'close' column for {symbol} from {numeric_cols[0]}")
                    
            return df
            
    def _invalidate_cache_for_symbol(self, symbol: str):
        """
        Invalidate cached relationships for a symbol.
        
        Args:
            symbol: Symbol to invalidate
        """
        with self._lock:
            # Find all relationship keys that involve this symbol
            keys_to_remove = []
            
            for key in self._relationship_cache:
                if isinstance(key, tuple) and len(key) >= 3:
                    if key[0] == symbol or key[1] == symbol:
                        keys_to_remove.append(key)
                        
            # Remove from cache
            for key in keys_to_remove:
                self._relationship_cache.pop(key, None)
                self._cache_timestamps.pop(key, None)
                
            # Remove from contagion cache
            self._contagion_cache.pop(symbol, None)
            
            # Invalidate all MST caches
            self._mst_cache.clear()
            
    def _get_cached_result(self, key: Any) -> Optional[Any]:
        """
        Get cached result if valid.
        
        Args:
            key: Cache key
            
        Returns:
            Cached result or None if not found or expired
        """
        if not self.config["cache_results"]:
            return None
            
        with self._lock:
            # Check if result is in cache
            result = self._relationship_cache.get(key)
            timestamp = self._cache_timestamps.get(key)
            
            if result is None or timestamp is None:
                return None
                
            # Check if expired
            current_time = time.time()
            if current_time - timestamp > self.config["cache_ttl"]:
                # Remove from cache
                self._relationship_cache.pop(key, None)
                self._cache_timestamps.pop(key, None)
                return None
                
            return result
            
    def _cache_result(self, key: Any, result: Any):
        """
        Cache analysis result.
        
        Args:
            key: Cache key
            result: Result to cache
        """
        if not self.config["cache_results"]:
            return
            
        with self._lock:
            self._relationship_cache[key] = result
            self._cache_timestamps[key] = time.time()

    def _calculate_leadership_scores(self, lead_lag_matrix, symbols):
        """
        Calculate leadership scores based on lead-lag relationships.
        
        Args:
            lead_lag_matrix: Matrix of lead-lag relationships
            symbols: List of asset symbols
            
        Returns:
            dict: Leadership scores for each asset
        """
        leadership_scores = {sym: 0.0 for sym in symbols}
        
        # Count how many assets each symbol leads
        for sym1 in symbols:
            for sym2 in symbols:
                if sym1 == sym2:
                    continue
                
                relationship = lead_lag_matrix.get(sym1, {}).get(sym2, {})
                lag = relationship.get('lag', 0)
                corr = relationship.get('correlation', 0)
                
                # Positive lag means sym1 leads sym2
                if lag > 0 and abs(corr) >= self.config["correlation_threshold"]:
                    # Add weighted leadership score
                    leadership_scores[sym1] += lag * abs(corr)
                # Negative lag means sym1 follows sym2
                elif lag < 0 and abs(corr) >= self.config["correlation_threshold"]:
                    # Reduce leadership score
                    leadership_scores[sym1] -= abs(lag) * abs(corr)
        
        # Normalize scores to [-1, 1] range
        max_abs_score = max(abs(score) for score in leadership_scores.values()) if leadership_scores else 1.0
        if max_abs_score > 0:
            for sym in leadership_scores:
                leadership_scores[sym] /= max_abs_score
        
        return leadership_scores

    def _calculate_causality(self, price_df, window, max_lag=5):
        """
        Calculate Granger causality between assets.
        
        Args:
            price_df: DataFrame with price data for multiple assets
            window: Analysis window size
            max_lag: Maximum lag for Granger test
            
        Returns:
            dict: Causality relationships
        """
        try:
            from statsmodels.tsa.stattools import grangercausalitytests
            
            # Calculate returns
            returns_df = price_df.pct_change().dropna()
            
            # Get recent data
            if len(returns_df) > window:
                recent_returns = returns_df.tail(window)
            else:
                recent_returns = returns_df
            
            # Get symbols
            symbols = price_df.columns.tolist()
            
            # Initialize results
            causality_matrix = {}
            
            # Calculate Granger causality for each pair
            for sym1 in symbols:
                causality_matrix[sym1] = {}
                
                for sym2 in symbols:
                    if sym1 == sym2:
                        continue
                    
                    # Prepare data for Granger test
                    data = recent_returns[[sym1, sym2]].dropna()
                    
                    # Skip if not enough data
                    if len(data) <= max_lag + 1:
                        continue
                    
                    # Test for Granger causality
                    try:
                        # Test if sym1 Granger-causes sym2
                        test_result = grangercausalitytests(data, max_lag, verbose=False)
                        
                        # Extract p-values for each lag
                        p_values = [test_result[i+1][0]['ssr_ftest'][1] for i in range(max_lag)]
                        
                        # Find minimum p-value and corresponding lag
                        min_p_value = min(p_values)
                        best_lag = p_values.index(min_p_value) + 1
                        
                        # Store result if significant
                        if min_p_value < 0.05:  # 5% significance level
                            causality_matrix[sym1][sym2] = {
                                'p_value': min_p_value,
                                'lag': best_lag,
                                'strength': 1.0 - min_p_value  # Convert p-value to strength
                            }
                    except Exception as inner_e:
                        self.logger.debug(f"Granger test failed for {sym1}->{sym2}: {inner_e}")
            
            return {
                'matrix': causality_matrix,
                'max_lag': max_lag
            }
        except Exception as e:
            self.logger.error(f"Error calculating causality: {e}")
            return {'matrix': {}, 'max_lag': max_lag}

    def _generate_signals(self, correlation_matrix, lead_lag, causality, price_df):
        """
        Generate trading signals based on cross-asset analysis.
        
        Args:
            correlation_matrix: Correlation matrix data
            lead_lag: Lead-lag relationship data
            causality: Causality relationship data
            price_df: DataFrame with price data
            
        Returns:
            dict: Signals for each asset
        """
        signals = {}
        
        try:
            symbols = correlation_matrix.get('symbols', [])
            
            for symbol in symbols:
                # Initialize signal array (same length as price data)
                signal_values = np.zeros(len(price_df))
                
                # Get leadership score
                leadership_score = lead_lag.get('leadership_scores', {}).get(symbol, 0.0)
                
                # Look for assets that lead this symbol
                for other_symbol in symbols:
                    if other_symbol == symbol:
                        continue
                    
                    # Check if other_symbol leads symbol
                    lead_lag_rel = lead_lag.get('matrix', {}).get(other_symbol, {}).get(symbol, {})
                    lag = lead_lag_rel.get('lag', 0)
                    corr = lead_lag_rel.get('correlation', 0.0)
                    
                    if lag > 0 and abs(corr) >= self.min_correlation:
                        # other_symbol leads symbol with positive lag
                        # Calculate the signal based on the leading asset's returns
                        other_returns = price_df[other_symbol].pct_change().fillna(0).values
                        
                        # Apply the lag
                        for i in range(lag, len(signal_values)):
                            # Add to signal based on correlation direction
                            signal_values[i] += np.sign(corr) * other_returns[i-lag] * abs(corr)
                
                # Normalize signal values
                if np.max(np.abs(signal_values)) > 0:
                    signal_values = signal_values / np.max(np.abs(signal_values))
                
                # Apply strength based on leadership score (stronger if asset is a follower)
                strength = 0.5 - leadership_score/2  # Scale from 0-1, lower for leaders, higher for followers
                signal_values = signal_values * strength
                
                # Store signals
                signals[symbol] = signal_values.tolist()
        except Exception as e:
            self.logger.error(f"Error generating cross-asset signals: {e}")
        
        return signals
   
    def _clean_old_data(self, max_age=86400):  # 24 hours by default
        """
        Clean old data from storage.
        
        Args:
            max_age: Maximum age in seconds
        """
        now = time.time()
        cutoff = now - max_age
        
        with self.lock:
            # Clean correlation matrices
            self.correlation_matrices = {
                ts: data for ts, data in self.correlation_matrices.items()
                if ts >= cutoff
            }
            
            # Clean lead-lag data
            self.lead_lag_data = {
                ts: data for ts, data in self.lead_lag_data.items()
                if ts >= cutoff
            }
            
            # Clean causality data
            self.causality_data = {
                ts: data for ts, data in self.causality_data.items()
                if ts >= cutoff
            }
    
    
    def _identify_clusters(self, correlation_matrix, threshold=None):
        """
        Identify clusters of correlated assets.
        
        Args:
            correlation_matrix: Asset correlation matrix (numpy array or dictionary)
            threshold: Correlation threshold (defaults to min_correlation)
            
        Returns:
            list: Clusters of related assets
        """
        if threshold is None:
            threshold = self.min_correlation
        
        # Handle case when correlation_matrix is a dictionary instead of a numpy array
        symbols = None
        if isinstance(correlation_matrix, dict):
            # Extract relevant data based on dictionary structure
            matrix_dict = None
            if 'recent' in correlation_matrix and isinstance(correlation_matrix['recent'], dict):
                matrix_dict = correlation_matrix['recent']
                symbols = correlation_matrix.get('symbols', list(matrix_dict.keys()))
            elif 'filtered' in correlation_matrix:
                matrix_dict = correlation_matrix['filtered']
                symbols = correlation_matrix.get('symbols', list(matrix_dict.keys()))
            elif 'full' in correlation_matrix:
                matrix_dict = correlation_matrix['full']
                symbols = correlation_matrix.get('symbols', list(matrix_dict.keys()))
            else:
                # Try to interpret the dictionary directly as a correlation matrix
                matrix_dict = correlation_matrix
                symbols = list(matrix_dict.keys())
            
            # Create a numpy array from the dictionary
            n = len(symbols)
            matrix = np.zeros((n, n))
            
            # Fill in correlation values
            for i, sym1 in enumerate(symbols):
                for j, sym2 in enumerate(symbols):
                    if i == j:
                        matrix[i, j] = 1.0  # Self-correlation is always 1.0
                    elif sym1 in matrix_dict and sym2 in matrix_dict[sym1]:
                        matrix[i, j] = matrix_dict[sym1][sym2]
                    elif sym2 in matrix_dict and sym1 in matrix_dict[sym2]:
                        matrix[i, j] = matrix_dict[sym2][sym1]
        else:
            # Standard case - matrix is already a numpy array
            matrix = np.array(correlation_matrix)
            n = matrix.shape[0]
        
        # Simple clustering based on correlation threshold
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if abs(matrix[i, j]) >= threshold:
                    edges.append((i, j, abs(matrix[i, j])))
        
        # Sort edges by weight (correlation strength)
        edges.sort(key=lambda x: x[2], reverse=True)
        
        # Union-find for clustering
        parent = list(range(n))
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            parent[find(x)] = find(y)
        
        # Form clusters
        for i, j, _ in edges:
            union(i, j)
        
        # Collect clusters
        clusters = {}
        for i in range(n):
            root = find(i)
            if root not in clusters:
                clusters[root] = []
            clusters[root].append(i if symbols is None else symbols[i])
        
        # Convert to list of clusters
        return list(clusters.values())
    
    def _compute_cross_correlation(self, series1, series2, max_lag):
        """
        Compute cross-correlation function between two series.
        
        Args:
            series1: First time series
            series2: Second time series
            max_lag: Maximum lag to consider
            
        Returns:
            array: Cross-correlation values for each lag
        """
        # Ensure series are numpy arrays
        s1 = np.array(series1)
        s2 = np.array(series2)
        
        # Standardize series
        s1 = (s1 - np.mean(s1)) / (np.std(s1) if np.std(s1) > 0 else 1)
        s2 = (s2 - np.mean(s2)) / (np.std(s2) if np.std(s2) > 0 else 1)
        
        # Calculate cross-correlation for different lags
        xcorr = np.zeros(2 * max_lag + 1)
        
        for i in range(2 * max_lag + 1):
            lag = i - max_lag
            if lag < 0:
                # s1 lags behind s2
                xcorr[i] = np.corrcoef(s1[:lag], s2[-lag:])[0, 1]
            elif lag > 0:
                # s1 leads s2
                xcorr[i] = np.corrcoef(s1[lag:], s2[:-lag])[0, 1]
            else:
                # No lag
                xcorr[i] = np.corrcoef(s1, s2)[0, 1]
        
        return xcorr
    
    def get_signals_for_symbol(self, symbol):
        """
        Get cross-asset signals for a specific symbol.
        
        Args:
            symbol: Asset symbol
            
        Returns:
            dict: Signal data
        """
        with self.lock:
            # Get most recent data
            if not self.correlation_matrices:
                return {}
            
            latest_ts = max(self.correlation_matrices.keys())
            corr_matrix = self.correlation_matrices.get(latest_ts, {})
            
            lead_lag = self.lead_lag_data.get(latest_ts, {})
            leadership_score = lead_lag.get('leadership_scores', {}).get(symbol, 0.0)
            
            # Generate simple signals based on leadership
            signals = {
                'leadership': [leadership_score] * 10,  # Constant value for simplicity
                'follower_influence': [0.0] * 10
            }
            
            # Check for specific lead-lag relationships
            for other_symbol, rel in lead_lag.get('matrix', {}).get(symbol, {}).items():
                lag = rel.get('lag', 0)
                corr = rel.get('correlation', 0.0)
                
                if lag > 0 and abs(corr) > self.min_correlation:
                    # This symbol leads other_symbol
                    if 'lead_influence' not in signals:
                        signals['lead_influence'] = [0.2] * 10  # Positive influence
                
                if lag < 0 and abs(corr) > self.min_correlation:
                    # This symbol follows other_symbol
                    signals['follower_influence'] = [0.5] * 10  # Stronger influence
            
            return signals
        
    
            
    def enqueue_task(self, task_func: Callable, args: Tuple = (), kwargs: Dict[str, Any] = None,
                   callback: Optional[Callable] = None, priority: int = 1):
        """
        Enqueue a task for background processing.
        
        Args:
            task_func: Task function to execute
            args: Positional arguments
            kwargs: Keyword arguments
            callback: Optional callback for result
            priority: Task priority (lower number = higher priority)
        """
        if not self.config["use_parallel"]:
            # Execute immediately
            try:
                result = task_func(*args, **(kwargs or {}))
                if callback:
                    callback(result)
            except Exception as e:
                self.logger.error(f"Error executing task: {e}")
                if callback:
                    callback(None, exception=e)
        else:
            # Add to queue
            try:
                self._processing_queue.put((priority, task_func, args, kwargs or {}, callback))
            except queue.Full:
                self.logger.warning("Task queue full, executing task immediately")
                try:
                    result = task_func(*args, **(kwargs or {}))
                    if callback:
                        callback(result)
                except Exception as e:
                    self.logger.error(f"Error executing task: {e}")
                    if callback:
                        callback(None, exception=e)
                            
    def calculate_correlation_matrix(self, symbols: Union[List[str], pd.DataFrame], method: Optional[str] = None,
                                  timeframe: Optional[str] = None, window: Optional[int] = None,
                                  start_date: Optional[Union[str, pd.Timestamp]] = None,
                                  end_date: Optional[Union[str, pd.Timestamp]] = None) -> pd.DataFrame:
        """
        Calculate correlation matrix for the given symbols or dataframe.
        
        Args:
            symbols: List of symbols or DataFrame with price data
            method: Correlation method (default from config)
            timeframe: Timeframe for analysis (default from config)
            window: Window size for rolling correlation (default from config)
            start_date: Start date for analysis
            end_date: End date for analysis
            
        Returns:
            Correlation matrix dataframe
        """
        # Get defaults from config if not provided
        method = method or self.config["default_correlation_method"]
        timeframe = timeframe or self.config["default_timeframe"]
        window = window or self.config["correlation_window"]
        
        # Handle DataFrame input vs symbol list input
        if isinstance(symbols, pd.DataFrame):
            # If symbols is a DataFrame, use it directly as price_df
            price_df = symbols
            missing_symbols = []  # No missing symbols when DataFrame is provided directly
        else:
            # Check cache for exact match when using symbol list
            cache_key = (tuple(sorted(symbols)), method, timeframe, window, start_date, end_date)
            cached_result = self._get_cached_result(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Validate input symbols
            if not symbols:  # This check is safe because symbols is a list here
                self.logger.error("No symbols provided for correlation analysis")
                return pd.DataFrame()
                
            # Get price data for symbols with enhanced validation
            dfs = []
            valid_symbols = []
            missing_symbols = []
            
            for symbol in symbols:
                df = self.get_price_data(symbol)
                if df is None or df.empty:
                    missing_symbols.append(symbol)
                    continue
                    
                # Check for close column and try alternatives if not found
                if 'close' in df.columns:
                    series = df['close'].copy()
                elif 'Close' in df.columns:
                    series = df['Close'].copy()
                elif len(df.columns) > 0:
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        series = df[numeric_cols[0]].copy()
                        self.logger.warning(f"No close price found for {symbol}, using {numeric_cols[0]} instead")
                    else:
                        missing_symbols.append(symbol)
                        continue
                else:
                    missing_symbols.append(symbol)
                    continue
                    
                # Check if series has enough data
                if len(series.dropna()) < 2:
                    self.logger.warning(f"Insufficient data for {symbol}, skipping")
                    continue
                    
                # Valid data found
                series.name = symbol
                dfs.append(series)
                valid_symbols.append(symbol)
            
            # Log issues once rather than repeatedly
            if missing_symbols:
                self.logger.warning(f"No price data found for {len(missing_symbols)} symbols: {', '.join(missing_symbols[:5])}" + 
                                  (f" and {len(missing_symbols) - 5} more" if len(missing_symbols) > 5 else ""))
            
            # Check if we have enough data to proceed
            if not dfs:
                self.logger.error("No valid price data available for any symbols")
                return pd.DataFrame()
            
            # Combine price series into a dataframe
            price_df = pd.concat(dfs, axis=1)
        
        # Now we have price_df regardless of input type
        
        # Check if dataframe is empty using proper pandas method
        if price_df.empty:
            self.logger.error("Empty price dataframe, cannot calculate correlation")
            return pd.DataFrame()
        
        # Filter by date range if provided
        if start_date is not None:
            price_df = price_df[price_df.index >= start_date]
        if end_date is not None:
            price_df = price_df[price_df.index <= end_date]
        
        # Check if filtered dataframe is empty
        if price_df.empty:
            self.logger.error("No data available after applying date filters")
            return pd.DataFrame()
            
        # Calculate returns
        returns_df = price_df.pct_change().dropna()
        
        # Check if returns dataframe is empty
        if returns_df.empty:
            self.logger.error("Cannot calculate correlation: no valid returns data")
            return pd.DataFrame()
        
        # Calculate correlation based on method
        if method.lower() == 'pearson':
            corr_matrix = returns_df.corr(method='pearson')
        elif method.lower() == 'spearman':
            corr_matrix = returns_df.corr(method='spearman')
        elif method.lower() == 'kendall':
            corr_matrix = returns_df.corr(method='kendall')
        elif method.lower() == 'distance':
            # Distance correlation
            if SCIPY_AVAILABLE:
                # Calculate pairwise distances
                distances = pdist(returns_df.T, metric='correlation')
                # Convert to square matrix
                corr_matrix = pd.DataFrame(1 - squareform(distances), 
                                        index=returns_df.columns, 
                                        columns=returns_df.columns)
            else:
                self.logger.warning("SciPy not available, using Pearson instead")
                corr_matrix = returns_df.corr(method='pearson')
        else:
            # Default to Pearson
            self.logger.warning(f"Unknown correlation method: {method}, using Pearson")
            corr_matrix = returns_df.corr(method='pearson')
                
        # Explicit check of DataFrame emptiness using pandas methods
        if corr_matrix is None or corr_matrix.empty:
            self.logger.error("Failed to calculate correlation matrix")
            return pd.DataFrame()
        
        # Only handle missing symbols when working with symbol list input
        if not isinstance(symbols, pd.DataFrame) and missing_symbols:
            # Add rows and columns of NaN for missing symbols
            for symbol in missing_symbols:
                if symbol not in corr_matrix.columns:
                    corr_matrix[symbol] = np.nan
                    corr_matrix.loc[symbol] = np.nan
                
            # Set diagonal to 1
            np.fill_diagonal(corr_matrix.values, 1.0)
                
            # Cache result when using symbol list (not when using DataFrame input)
            self._cache_result(cache_key, corr_matrix)
        
        return corr_matrix
    
    def calculate_rolling_correlation(self, symbol1: str, symbol2: str, method: Optional[str] = None,
                                   timeframe: Optional[str] = None, window: Optional[int] = None,
                                   start_date: Optional[Union[str, pd.Timestamp]] = None,
                                   end_date: Optional[Union[str, pd.Timestamp]] = None) -> pd.Series:
        """
        Calculate rolling correlation between two symbols.
        
        Args:
            symbol1: First symbol
            symbol2: Second symbol
            method: Correlation method (default from config)
            timeframe: Timeframe for analysis (default from config)
            window: Window size for rolling correlation (default from config)
            start_date: Start date for analysis
            end_date: End date for analysis
            
        Returns:
            Rolling correlation series
        """
        # Get defaults from config if not provided
        method = method or self.config["default_correlation_method"]
        timeframe = timeframe or self.config["default_timeframe"]
        window = window or self.config["rolling_window"]
        
        # Check cache for exact match
        cache_key = (symbol1, symbol2, "rolling", method, timeframe, window, start_date, end_date)
        cached_result = self._get_cached_result(cache_key)
        if cached_result is not None:
            return cached_result
            
        # Get price data for symbols
        df1 = self.get_price_data(symbol1)
        df2 = self.get_price_data(symbol2)
        
        if df1 is None or df2 is None:
            self.logger.error(f"Missing price data for {symbol1 if df1 is None else symbol2}")
            return pd.Series(dtype=float)
            
        # Extract close prices
        if 'close' in df1.columns and 'close' in df2.columns:
            prices1 = df1['close']
            prices2 = df2['close']
        else:
            self.logger.error(f"Missing close prices for {symbol1 if 'close' not in df1.columns else symbol2}")
            return pd.Series(dtype=float)
            
        # Combine price series
        price_df = pd.DataFrame({symbol1: prices1, symbol2: prices2})
        
        # Filter by date range if provided
        if start_date is not None:
            price_df = price_df[price_df.index >= start_date]
        if end_date is not None:
            price_df = price_df[price_df.index <= end_date]
            
        # Calculate returns
        returns_df = price_df.pct_change().dropna()
        
        # Calculate rolling correlation based on method
        if method.lower() in ('pearson', 'spearman', 'kendall'):
            rolling_corr = returns_df[symbol1].rolling(window=window).corr(returns_df[symbol2], method=method.lower())
        elif method.lower() == 'exponential':
            # Exponential weighted correlation
            ewma1 = returns_df[symbol1].ewm(alpha=self.config["ewma_alpha"]).std()
            ewma2 = returns_df[symbol2].ewm(alpha=self.config["ewma_alpha"]).std()
            ewma_cov = returns_df[symbol1].ewm(alpha=self.config["ewma_alpha"]).cov(returns_df[symbol2])
            rolling_corr = ewma_cov / (ewma1 * ewma2)
        else:
            # Default to Pearson
            self.logger.warning(f"Unknown correlation method: {method}, using Pearson")
            rolling_corr = returns_df[symbol1].rolling(window=window).corr(returns_df[symbol2])
            
        # Cache result
        self._cache_result(cache_key, rolling_corr)
        
        return rolling_corr
    
    def find_correlated_assets(self, symbol: str, threshold: Optional[float] = None,
                            method: Optional[str] = None, timeframe: Optional[str] = None,
                            asset_classes: Optional[List[Union[str, AssetClass]]] = None,
                            limit: int = 10) -> List[Tuple[str, float]]:
        """
        Find assets correlated with the given symbol.
        
        Args:
            symbol: Symbol to find correlations for
            threshold: Correlation threshold (default from config)
            method: Correlation method (default from config)
            timeframe: Timeframe for analysis (default from config)
            asset_classes: Optional filter by asset classes
            limit: Maximum number of results
            
        Returns:
            List of (symbol, correlation) tuples sorted by correlation strength
        """
        # Get defaults from config if not provided
        threshold = threshold if threshold is not None else self.config["correlation_threshold"]
        method = method or self.config["default_correlation_method"]
        timeframe = timeframe or self.config["default_timeframe"]
        
        # Convert asset_classes to enums if needed
        if asset_classes is not None:
            asset_classes = [
                ac if isinstance(ac, AssetClass) else AssetClass.from_string(ac)
                for ac in asset_classes
            ]
            
        # Get all available symbols
        with self._lock:
            all_symbols = list(self._price_data.keys())
            
        # Filter by asset class if provided
        if asset_classes is not None:
            filtered_symbols = []
            for sym in all_symbols:
                asset_info = self.get_asset_info(sym)
                if asset_info is not None and asset_info.asset_class in asset_classes:
                    filtered_symbols.append(sym)
        else:
            filtered_symbols = all_symbols
            
        # Remove the target symbol itself
        if symbol in filtered_symbols:
            filtered_symbols.remove(symbol)
            
        if not filtered_symbols:
            self.logger.warning(f"No symbols available for correlation analysis with {symbol}")
            return []
            
        # Calculate correlation with target symbol
        correlations = []
        
        for other_symbol in filtered_symbols:
            # Calculate correlation between the two symbols
            cache_key = (symbol, other_symbol, "correlation", method, timeframe)
            cached_result = self._get_cached_result(cache_key)
            
            if cached_result is not None:
                correlation = cached_result
            else:
                try:
                    # Get price data
                    df1 = self.get_price_data(symbol)
                    df2 = self.get_price_data(other_symbol)
                    
                    if df1 is None or df2 is None:
                        continue
                        
                    # Extract close prices
                    if 'close' in df1.columns and 'close' in df2.columns:
                        prices1 = df1['close']
                        prices2 = df2['close']
                    else:
                        continue
                        
                    # Align series and calculate returns
                    combined = pd.DataFrame({symbol: prices1, other_symbol: prices2})
                    returns = combined.pct_change().dropna()
                    
                    # Calculate correlation
                    if method.lower() in ('pearson', 'spearman', 'kendall'):
                        corr_obj = returns.corr(method=method.lower())
                        correlation = corr_obj.loc[symbol, other_symbol]
                    else:
                        # Default to Pearson
                        corr_obj = returns.corr(method='pearson')
                        correlation = corr_obj.loc[symbol, other_symbol]
                        
                    # Cache result
                    self._cache_result(cache_key, correlation)
                    
                except Exception as e:
                    self.logger.error(f"Error calculating correlation for {symbol} and {other_symbol}: {e}")
                    continue
                    
            # Apply threshold filter
            if self.config["use_absolute_correlation"]:
                if abs(correlation) >= threshold:
                    correlations.append((other_symbol, correlation))
            else:
                if correlation >= threshold:
                    correlations.append((other_symbol, correlation))
                    
        # Sort by absolute correlation strength (descending)
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # Limit results
        return correlations[:limit]
    
    def analyze_market_correlation_structure(self, symbols: Optional[List[str]] = None,
                                         method: Optional[str] = None,
                                         timeframe: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze the correlation structure of the market.
        
        Args:
            symbols: List of symbols to analyze (all available if None)
            method: Correlation method (default from config)
            timeframe: Timeframe for analysis (default from config)
            
        Returns:
            Dictionary with structure analysis results
        """
        # Get defaults from config if not provided
        method = method or self.config["default_correlation_method"]
        timeframe = timeframe or self.config["default_timeframe"]
        
        # Get all available symbols if not provided
        if symbols is None:
            with self._lock:
                symbols = list(self._price_data.keys())
                
        # Limit number of symbols for performance
        if len(symbols) > self.config["max_symbols"]:
            self.logger.warning(f"Too many symbols ({len(symbols)}), limiting to {self.config['max_symbols']}")
            symbols = symbols[:self.config["max_symbols"]]
            
        # Calculate correlation matrix
        corr_matrix = self.calculate_correlation_matrix(symbols, method=method, timeframe=timeframe)
        
        if corr_matrix.empty:
            self.logger.error("Failed to calculate correlation matrix")
            return {}
            
        # Calculate average correlation
        avg_corr = np.nanmean(np.abs(corr_matrix.values))
        
        # Calculate dispersion (standard deviation of correlations)
        corr_std = np.nanstd(corr_matrix.values)
        
        # Cluster analysis
        clusters = []
        
        if SCIPY_AVAILABLE:
            try:
                # Convert correlation to distance
                distance_matrix = 1 - np.abs(corr_matrix.values)
                
                # Set diagonal to zero
                np.fill_diagonal(distance_matrix, 0)
                
                # Hierarchical clustering
                Z = hierarchy.linkage(squareform(distance_matrix), method='ward')
                
                # Determine optimal number of clusters
                max_clusters = min(10, len(symbols) // 2)
                
                if max_clusters > 1:
                    clusters_counts = {}
                    
                    for n_clusters in range(2, max_clusters + 1):
                        cluster_labels = hierarchy.fcluster(Z, n_clusters, criterion='maxclust')
                        
                        # Count symbols in each cluster
                        cluster_counts = np.bincount(cluster_labels)
                        
                        # Calculate cluster quality (silhouette score)
                        clusters_counts[n_clusters] = list(cluster_counts)
                        
                    # Find clusters for optimal number (simplified)
                    n_clusters = min(max_clusters, 5)  # Default to 5 clusters
                    cluster_labels = hierarchy.fcluster(Z, n_clusters, criterion='maxclust')
                    
                    # Create clusters
                    for i in range(1, n_clusters + 1):
                        cluster_symbols = [symbols[j] for j, label in enumerate(cluster_labels) if label == i]
                        
                        if cluster_symbols:
                            # Calculate internal correlation
                            within_corr = 0
                            n_pairs = 0
                            
                            for idx1, s1 in enumerate(cluster_symbols):
                                for idx2, s2 in enumerate(cluster_symbols[idx1+1:], idx1+1):
                                    if s1 in corr_matrix.index and s2 in corr_matrix.columns:
                                        within_corr += abs(corr_matrix.loc[s1, s2])
                                        n_pairs += 1
                                        
                            avg_within_corr = within_corr / n_pairs if n_pairs > 0 else 0
                                
                            clusters.append({
                                "id": i,
                                "symbols": cluster_symbols,
                                "size": len(cluster_symbols),
                                "avg_correlation": avg_within_corr
                            })
            except Exception as e:
                self.logger.error(f"Error in cluster analysis: {e}")
                
        # Calculate Minimum Spanning Tree (MST)
        mst_edges = []
        if NETWORKX_AVAILABLE:
            try:
                # Create network from correlation matrix
                G = nx.Graph()
                
                # Add nodes
                for symbol in corr_matrix.index:
                    G.add_node(symbol)
                    
                # Add edges with weights based on correlation distance
                for i, symbol1 in enumerate(corr_matrix.index):
                    for j, symbol2 in enumerate(corr_matrix.columns):
                        if i < j:  # Avoid duplicates
                            # Higher correlation = shorter distance
                            distance = 1 - abs(corr_matrix.loc[symbol1, symbol2])
                            G.add_edge(symbol1, symbol2, weight=distance)
                            
                # Calculate Minimum Spanning Tree
                mst = nx.minimum_spanning_tree(G)
                
                # Extract edges
                for u, v, data in mst.edges(data=True):
                    weight = data['weight']
                    correlation = 1 - weight
                    mst_edges.append({
                        "source": u,
                        "target": v,
                        "correlation": correlation
                    })
            except Exception as e:
                self.logger.error(f"Error calculating MST: {e}")
                
        # Calculate node centrality
        central_nodes = []
        if NETWORKX_AVAILABLE and mst_edges:
            try:
                # Create network from MST
                mst_graph = nx.Graph()
                
                # Add edges
                for edge in mst_edges:
                    mst_graph.add_edge(edge["source"], edge["target"], weight=edge["correlation"])
                    
                # Calculate degree centrality
                centrality = nx.degree_centrality(mst_graph)
                
                # Sort by centrality
                sorted_centrality = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
                
                # Get top 10 central nodes
                central_nodes = [{"symbol": s, "centrality": c} for s, c in sorted_centrality[:10]]
            except Exception as e:
                self.logger.error(f"Error calculating centrality: {e}")
                
        # Create result
        result = {
            "average_correlation": float(avg_corr),
            "correlation_dispersion": float(corr_std),
            "clusters": clusters,
            "central_nodes": central_nodes,
            "mst_edges": mst_edges,
            "symbols_count": len(symbols),
            "timestamp": time.time(),
            "method": method,
            "timeframe": timeframe
        }
        
        return result
    
    # ----- Lead-Lag Analysis Methods -----
    
    def analyze_lead_lag_relationship(self, symbol1: str, symbol2: str, method: Optional[str] = None,
                                   max_lag: Optional[int] = None, timeframe: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze lead-lag relationship between two symbols.
        
        Args:
            symbol1: First symbol
            symbol2: Second symbol
            method: Analysis method ('correlation', 'granger', 'mutual_info')
            max_lag: Maximum lag to check (default from config)
            timeframe: Timeframe for analysis (default from config)
            
        Returns:
            Dictionary with lead-lag analysis results
        """
        # Get defaults from config if not provided
        method = method or 'correlation'
        max_lag = max_lag or self.config["max_lag"]
        timeframe = timeframe or self.config["default_timeframe"]
        
        # Check cache for exact match
        cache_key = (symbol1, symbol2, "lead_lag", method, max_lag, timeframe)
        cached_result = self._get_cached_result(cache_key)
        if cached_result is not None:
            return cached_result
            
        # Get price data for symbols
        df1 = self.get_price_data(symbol1)
        df2 = self.get_price_data(symbol2)
        
        if df1 is None or df2 is None:
            self.logger.error(f"Missing price data for {symbol1 if df1 is None else symbol2}")
            return {}
            
        # Extract close prices
        if 'close' in df1.columns and 'close' in df2.columns:
            prices1 = df1['close']
            prices2 = df2['close']
        else:
            self.logger.error(f"Missing close prices for {symbol1 if 'close' not in df1.columns else symbol2}")
            return {}
            
        # Align series
        combined = pd.DataFrame({symbol1: prices1, symbol2: prices2})
        
        # Calculate returns
        returns = combined.pct_change().dropna()
        
        # Initialize result dictionary
        result = {
            "symbol1": symbol1,
            "symbol2": symbol2,
            "method": method,
            "max_lag": max_lag,
            "timeframe": timeframe,
            "timestamp": time.time(),
            "lag_results": []
        }
        
        # Analyze based on method
        if method.lower() == 'correlation':
            # Cross-correlation analysis
            if SCIPY_AVAILABLE:
                # Get return series
                x = returns[symbol1].values
                y = returns[symbol2].values
                
                # Calculate cross-correlation for different lags
                lags = np.arange(-max_lag, max_lag + 1, self.config["lag_step"])
                lag_results = []
                
                for lag in lags:
                    # Calculate cross-correlation at this lag
                    cross_corr = self._calculate_cross_correlation(x, y, lag)
                    
                    # Calculate p-value (simplified)
                    n = len(x) - abs(lag)
                    t_stat = cross_corr * np.sqrt(n - 2) / np.sqrt(1 - cross_corr**2)
                    p_value = 2 * (1 - abs(t_stat) / np.sqrt(n))
                    
                    lag_results.append({
                        "lag": int(lag),
                        "correlation": float(cross_corr),
                        "p_value": float(min(1.0, max(0.0, p_value)))
                    })
                    
                # Sort by absolute correlation (highest first)
                lag_results.sort(key=lambda x: abs(x["correlation"]), reverse=True)
                
                # Find optimal lag
                significant_lags = [
                    l for l in lag_results 
                    if l["p_value"] < self.config["min_lag_significance"]
                ]
                
                if significant_lags:
                    optimal_lag = significant_lags[0]
                    result["optimal_lag"] = optimal_lag["lag"]
                    result["optimal_correlation"] = optimal_lag["correlation"]
                    result["significant"] = True
                else:
                    # No significant lag found
                    result["optimal_lag"] = 0
                    result["optimal_correlation"] = lag_results[0]["correlation"] if lag_results else 0
                    result["significant"] = False
                    
                # Store all lag results
                result["lag_results"] = lag_results
            else:
                self.logger.warning("SciPy not available, cannot calculate cross-correlation")
                
        elif method.lower() == 'granger':
            # Granger causality analysis
            try:
                from statsmodels.tsa.stattools import grangercausalitytests
                
                # Prepare data
                data = returns[[symbol1, symbol2]].dropna()
                
                # Test Granger causality in both directions
                maxlag = max_lag
                
                # Test if symbol1 Granger-causes symbol2
                gc_1_to_2 = grangercausalitytests(data[[symbol2, symbol1]], maxlag=maxlag, verbose=False)
                
                # Test if symbol2 Granger-causes symbol1
                gc_2_to_1 = grangercausalitytests(data[[symbol1, symbol2]], maxlag=maxlag, verbose=False)
                
                # Parse results
                forward_results = []
                backward_results = []
                
                for lag in range(1, maxlag + 1):
                    # Get p-values for F-tests
                    p_1_to_2 = gc_1_to_2[lag][0]['ssr_ftest'][1]
                    p_2_to_1 = gc_2_to_1[lag][0]['ssr_ftest'][1]
                    
                    forward_results.append({
                        "lag": lag,
                        "p_value": float(p_1_to_2),
                        "significant": p_1_to_2 < self.config["min_lag_significance"]
                    })
                    
                    backward_results.append({
                        "lag": lag,
                        "p_value": float(p_2_to_1),
                        "significant": p_2_to_1 < self.config["min_lag_significance"]
                    })
                    
                # Find optimal lags
                forward_significant = [r for r in forward_results if r["significant"]]
                backward_significant = [r for r in backward_results if r["significant"]]
                
                # Determine causality direction
                if forward_significant and not backward_significant:
                    direction = 1  # Symbol1 leads Symbol2
                    optimal_lag = forward_significant[0]["lag"]
                elif backward_significant and not forward_significant:
                    direction = -1  # Symbol2 leads Symbol1
                    optimal_lag = backward_significant[0]["lag"]
                elif forward_significant and backward_significant:
                    # Bidirectional causality - determine the stronger direction
                    best_forward = min(forward_significant, key=lambda x: x["p_value"])
                    best_backward = min(backward_significant, key=lambda x: x["p_value"])
                    
                    if best_forward["p_value"] < best_backward["p_value"]:
                        direction = 1  # Symbol1 leads Symbol2 (stronger)
                        optimal_lag = best_forward["lag"]
                    else:
                        direction = -1  # Symbol2 leads Symbol1 (stronger)
                        optimal_lag = best_backward["lag"]
                else:
                    direction = 0  # No significant causality
                    optimal_lag = 0
                    
                # Store results
                result["causality_direction"] = direction
                result["optimal_lag"] = optimal_lag
                result["forward_results"] = forward_results
                result["backward_results"] = backward_results
                result["significant"] = bool(forward_significant or backward_significant)
                
            except ImportError:
                self.logger.warning("statsmodels not available, cannot perform Granger causality test")
                
        elif method.lower() == 'mutual_info':
            # Mutual information analysis
            if SCIPY_AVAILABLE:
                from sklearn.feature_selection import mutual_info_regression
                
                # Get return series
                x = returns[symbol1].values.reshape(-1, 1)
                y = returns[symbol2].values
                
                # Calculate mutual information for different lags
                lags = np.arange(-max_lag, max_lag + 1, self.config["lag_step"])
                lag_results = []
                
                for lag in lags:
                    # Shift data according to lag
                    if lag > 0:
                        # Positive lag: y is shifted forward, x is fixed
                        x_lag = x[:-lag] if lag < len(x) else np.array([]).reshape(-1, 1)
                        y_lag = y[lag:] if lag < len(y) else np.array([])
                    elif lag < 0:
                        # Negative lag: x is shifted forward, y is fixed
                        x_lag = x[-lag:] if -lag < len(x) else np.array([]).reshape(-1, 1)
                        y_lag = y[:lag] if -lag < len(y) else np.array([])
                    else:
                        # No lag
                        x_lag = x
                        y_lag = y
                        
                    # Calculate mutual information if we have enough data
                    if len(x_lag) > 10 and len(y_lag) > 10:
                        mi = float(mutual_info_regression(x_lag, y_lag, random_state=42)[0])
                        
                        # Determine significance (heuristic)
                        # Mutual information is always positive, so we compare to shuffle
                        y_shuffled = np.random.permutation(y_lag)
                        mi_shuffled = float(mutual_info_regression(x_lag, y_shuffled, random_state=42)[0])
                        p_value = float(mi_shuffled / (mi + 1e-10))
                        
                        lag_results.append({
                            "lag": int(lag),
                            "mutual_info": mi,
                            "p_value": p_value
                        })
                    else:
                        lag_results.append({
                            "lag": int(lag),
                            "mutual_info": 0.0,
                            "p_value": 1.0
                        })
                        
                # Sort by mutual information (highest first)
                lag_results.sort(key=lambda x: x["mutual_info"], reverse=True)
                
                # Find optimal lag
                significant_lags = [
                    l for l in lag_results 
                    if l["p_value"] < self.config["min_lag_significance"]
                ]
                
                if significant_lags:
                    optimal_lag = significant_lags[0]
                    result["optimal_lag"] = optimal_lag["lag"]
                    result["optimal_mutual_info"] = optimal_lag["mutual_info"]
                    result["significant"] = True
                else:
                    # No significant lag found
                    result["optimal_lag"] = 0
                    result["optimal_mutual_info"] = lag_results[0]["mutual_info"] if lag_results else 0
                    result["significant"] = False
                    
                # Store all lag results
                result["lag_results"] = lag_results
            else:
                self.logger.warning("SciPy and scikit-learn not available, cannot calculate mutual information")
                
        else:
            self.logger.warning(f"Unknown lead-lag method: {method}")
            
        # Cache result
        self._cache_result(cache_key, result)
        
        return result
    
    def _calculate_cross_correlation(self, x: np.ndarray, y: np.ndarray, lag: int) -> float:
        """
        Calculate cross-correlation between two series at given lag.
        
        Args:
            x: First series
            y: Second series
            lag: Lag value (positive: y is shifted forward)
            
        Returns:
            Cross-correlation value
        """
        if not SCIPY_AVAILABLE:
            return 0.0
            
        try:
            n = len(x)
            
            if lag > 0:
                # Positive lag: y is shifted forward, x is fixed
                if lag >= n:
                    return 0.0
                x_lag = x[:-lag]
                y_lag = y[lag:]
            elif lag < 0:
                # Negative lag: x is shifted forward, y is fixed
                if -lag >= n:
                    return 0.0
                x_lag = x[-lag:]
                y_lag = y[:lag]
            else:
                # No lag
                x_lag = x
                y_lag = y
                
            # Calculate correlation
            corr, _ = pearsonr(x_lag, y_lag)
            return corr
            
        except Exception as e:
            self.logger.error(f"Error calculating cross-correlation: {e}")
            return 0.0
    
    def find_leading_assets(self, symbol: str, asset_classes: Optional[List[Union[str, AssetClass]]] = None,
                         method: Optional[str] = None, max_lag: Optional[int] = None,
                         min_significance: Optional[float] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Find assets that lead the given symbol.
        
        Args:
            symbol: Symbol to analyze
            asset_classes: Optional filter by asset classes
            method: Analysis method (default: 'correlation')
            max_lag: Maximum lag to check (default from config)
            min_significance: Minimum significance level (default from config)
            limit: Maximum number of results
            
        Returns:
            List of leading assets with analysis results
        """
        # Get defaults from config if not provided
        method = method or 'correlation'
        max_lag = max_lag or self.config["max_lag"]
        min_significance = min_significance or self.config["min_lag_significance"]
        
        # Convert asset_classes to enums if needed
        if asset_classes is not None:
            asset_classes = [
                ac if isinstance(ac, AssetClass) else AssetClass.from_string(ac)
                for ac in asset_classes
            ]
            
        # Get all available symbols
        with self._lock:
            all_symbols = list(self._price_data.keys())
            
        # Filter by asset class if provided
        if asset_classes is not None:
            filtered_symbols = []
            for sym in all_symbols:
                asset_info = self.get_asset_info(sym)
                if asset_info is not None and asset_info.asset_class in asset_classes:
                    filtered_symbols.append(sym)
        else:
            filtered_symbols = all_symbols
            
        # Remove the target symbol itself
        if symbol in filtered_symbols:
            filtered_symbols.remove(symbol)
            
        if not filtered_symbols:
            self.logger.warning(f"No symbols available for lead-lag analysis with {symbol}")
            return []
            
        # Analyze lead-lag relationships
        lead_lag_results = []
        
        for other_symbol in filtered_symbols:
            # Check cache first
            cache_key = (other_symbol, symbol, "lead_lag", method, max_lag, self.config["default_timeframe"])
            cached_result = self._get_cached_result(cache_key)
            
            if cached_result is not None:
                # Use cached result
                result = cached_result
            else:
                # Calculate lead-lag relationship
                result = self.analyze_lead_lag_relationship(
                    other_symbol, symbol, method=method, max_lag=max_lag
                )
                
            # Check if this is a leading relationship
            if result.get("significant", False):
                optimal_lag = result.get("optimal_lag", 0)
                
                if method.lower() == 'correlation':
                    # For correlation, positive lag means other_symbol leads symbol
                    if optimal_lag > 0:
                        correlation = result.get("optimal_correlation", 0)
                        
                        lead_lag_results.append({
                            "symbol": other_symbol,
                            "lag": optimal_lag,
                            "correlation": correlation,
                            "asset_class": self._get_asset_class_str(other_symbol),
                            "strength": abs(correlation)
                        })
                elif method.lower() == 'granger':
                    # For Granger, check causality direction
                    direction = result.get("causality_direction", 0)
                    
                    if direction > 0:  # other_symbol (symbol1) leads symbol (symbol2)
                        # Get p-value for optimal lag
                        forward_results = result.get("forward_results", [])
                        optimal_result = next((r for r in forward_results if r["lag"] == optimal_lag), None)
                        
                        if optimal_result and optimal_result.get("p_value", 1.0) < min_significance:
                            lead_lag_results.append({
                                "symbol": other_symbol,
                                "lag": optimal_lag,
                                "p_value": optimal_result["p_value"],
                                "asset_class": self._get_asset_class_str(other_symbol),
                                "strength": 1.0 - optimal_result["p_value"]
                            })
                elif method.lower() == 'mutual_info':
                    # For mutual info, positive lag means other_symbol leads symbol
                    if optimal_lag > 0:
                        mutual_info = result.get("optimal_mutual_info", 0)
                        
                        lag_results = result.get("lag_results", [])
                        optimal_result = next((r for r in lag_results if r["lag"] == optimal_lag), None)
                        
                        if optimal_result and optimal_result.get("p_value", 1.0) < min_significance:
                            lead_lag_results.append({
                                "symbol": other_symbol,
                                "lag": optimal_lag,
                                "mutual_info": mutual_info,
                                "p_value": optimal_result["p_value"],
                                "asset_class": self._get_asset_class_str(other_symbol),
                                "strength": mutual_info
                            })
                            
        # Sort by strength (descending)
        lead_lag_results.sort(key=lambda x: x.get("strength", 0), reverse=True)
        
        # Limit results
        return lead_lag_results[:limit]
    
    def _get_asset_class_str(self, symbol: str) -> str:
        """Get asset class string for a symbol."""
        asset_info = self.get_asset_info(symbol)
        if asset_info is not None:
            return str(asset_info.asset_class)
        return "unknown"

    
    def fetch_and_analyze_yfinance(self, symbols: List[str], 
                                 period: str = '1y', 
                                 interval: str = '1d',
                                 analysis_type: str = 'correlation') -> Dict[str, Any]:
        """
        Fetch data from Yahoo Finance and perform cross-asset analysis.
        
        Args:
            symbols: List of symbols to analyze
            period: Time period to analyze
            interval: Data interval
            analysis_type: Type of analysis to perform
            
        Returns:
            Analysis results
        """
        # Create data fetcher if not already available
        if not hasattr(self, '_data_fetcher'):
            from .data_fetcher import MarketDataFetcher
            self._data_fetcher = MarketDataFetcher()
            
        # Fetch data
        data_dict = self._data_fetcher.fetch_data_for_cdfa(
            symbols, 
            source='yahoo', 
            period=period, 
            interval=interval
        )
        
        if not data_dict:
            self.logger.error("Failed to fetch data from Yahoo Finance")
            return {"error": "Failed to fetch data"}
            
        # Add data to analyzer
        for symbol, df in data_dict.items():
            self.add_price_data(symbol, df)
            
        # Perform requested analysis
        if analysis_type == 'correlation':
            result = self.analyze_market_correlation_structure(symbols)
            # Ensure 'clusters' key exists with default empty list
            if 'clusters' not in result:
                result['clusters'] = []
            return result
            
        elif analysis_type == 'lead_lag':
            # For lead-lag, we need pairs of symbols
            results = {}
            for i, sym1 in enumerate(symbols):
                for sym2 in symbols[i+1:]:
                    result = self.analyze_lead_lag_relationship(sym1, sym2)
                    results[f"{sym1}_{sym2}"] = result
            return results
            
        elif analysis_type == 'regime':
            return self.analyze_regime_consistency(symbols)
            
        else:
            self.logger.error(f"Unsupported analysis type: {analysis_type}")
            return {"error": f"Unsupported analysis type: {analysis_type}"}
    
    # ----- Contagion Risk Analysis Methods -----
    
    def calculate_contagion_risk(self, symbol: str, threshold: Optional[float] = None,
                              max_path: Optional[int] = None, timeframe: Optional[str] = None) -> Dict[str, Any]:
        """
        Calculate contagion risk from the given symbol to other assets.
        
        Args:
            symbol: Source symbol for contagion
            threshold: Correlation threshold (default from config)
            max_path: Maximum path length for contagion (default from config)
            timeframe: Timeframe for analysis (default from config)
            
        Returns:
            Dictionary with contagion risk analysis
        """
        # Get defaults from config if not provided
        threshold = threshold if threshold is not None else self.config["contagion_threshold"]
        max_path = max_path if max_path is not None else self.config["max_contagion_path"]
        timeframe = timeframe or self.config["default_timeframe"]
        
        # Check cache
        cache_key = (symbol, "contagion", threshold, max_path, timeframe)
        cached_result = self._get_cached_result(cache_key)
        if cached_result is not None:
            return cached_result
            
        # Get all available symbols
        with self._lock:
            all_symbols = list(self._price_data.keys())
            
        if symbol not in all_symbols:
            self.logger.error(f"Symbol {symbol} not found in price data")
            return {}
            
        # Calculate correlation matrix
        corr_matrix = self.calculate_correlation_matrix(all_symbols, timeframe=timeframe)
        
        if corr_matrix.empty:
            self.logger.error("Failed to calculate correlation matrix")
            return {}
            
        # For contagion analysis, we need a graph representation
        if not NETWORKX_AVAILABLE:
            self.logger.warning("NetworkX not available, using simplified contagion analysis")
            
            # Simplified contagion analysis
            impact_symbols = []
            
            for other_symbol in all_symbols:
                if other_symbol != symbol and other_symbol in corr_matrix.index and symbol in corr_matrix.columns:
                    corr_value = abs(corr_matrix.loc[symbol, other_symbol])
                    
                    if corr_value >= threshold:
                        impact_symbols.append((other_symbol, corr_value))
                        
            # Sort by correlation (descending)
            impact_symbols.sort(key=lambda x: x[1], reverse=True)
            
            # Calculate system impact (simplified)
            systemic_impact = sum(corr for _, corr in impact_symbols) / len(all_symbols) if impact_symbols else 0
            
            # Create result
            result = {
                "symbol": symbol,
                "risk_score": len(impact_symbols) / len(all_symbols),
                "impact_symbols": impact_symbols,
                "systemic_impact": systemic_impact,
                "timeframe": timeframe,
                "timestamp": time.time()
            }
            
            # Cache result
            self._cache_result(cache_key, result)
            
            return result
            
        else:
            # Full graph-based contagion analysis
            # Create network from correlation matrix
            G = nx.Graph()
            
            # Add nodes
            for sym in corr_matrix.index:
                G.add_node(sym)
                
            # Add edges with weights based on correlation
            for i, sym1 in enumerate(corr_matrix.index):
                for j, sym2 in enumerate(corr_matrix.columns):
                    if i < j:  # Avoid duplicates
                        corr_value = abs(corr_matrix.loc[sym1, sym2])
                        
                        if corr_value >= threshold:
                            G.add_edge(sym1, sym2, weight=corr_value)
                            
            # Check if symbol is in the graph
            if symbol not in G:
                self.logger.warning(f"Symbol {symbol} not in the correlation network")
                return {}
                
            # Calculate shortest paths from the source symbol
            impact_symbols = []
            
            try:
                # Get all shortest paths from source within max_path
                paths = nx.single_source_shortest_path_length(G, symbol, cutoff=max_path)
                
                for target, path_length in paths.items():
                    if target != symbol:
                        # Calculate impact based on path length
                        if path_length > 0:
                            impact = 1.0 / path_length
                        else:
                            impact = 1.0
                            
                        impact_symbols.append((target, impact))
                        
                # Sort by impact (descending)
                impact_symbols.sort(key=lambda x: x[1], reverse=True)
                
                # Calculate centrality for systemic importance
                centrality = nx.degree_centrality(G)
                source_centrality = centrality.get(symbol, 0)
                
                # Measure system impact as a combination of direct connections and centrality
                direct_connections = len([n for n in G.neighbors(symbol)])
                systemic_impact = (direct_connections / len(G)) * source_centrality
                
                # Create result
                result = {
                    "symbol": symbol,
                    "risk_score": source_centrality,
                    "impact_symbols": impact_symbols,
                    "systemic_impact": systemic_impact,
                    "direct_connections": direct_connections,
                    "total_nodes": len(G),
                    "timeframe": timeframe,
                    "timestamp": time.time()
                }
                
                # Cache result
                self._cache_result(cache_key, result)
                self._contagion_cache[symbol] = ContagionRisk(
                    symbol=symbol,
                    risk_score=source_centrality,
                    impact_symbols=impact_symbols,
                    systemic_impact=systemic_impact,
                    timeframe=timeframe,
                    timestamp=time.time()
                )
                
                return result
                
            except Exception as e:
                self.logger.error(f"Error calculating contagion paths: {e}")
                return {}
    
    def find_systemic_assets(self, threshold: Optional[float] = None, asset_classes: Optional[List[Union[str, AssetClass]]] = None,
                         limit: int = 10) -> List[Dict[str, Any]]:
        """
        Find systemically important assets based on contagion risk.
        
        Args:
            threshold: Importance threshold (default from config)
            asset_classes: Optional filter by asset classes
            limit: Maximum number of results
            
        Returns:
            List of systemically important assets
        """
        # Get defaults from config if not provided
        threshold = threshold if threshold is not None else self.config["systemic_importance_threshold"]
        
        # Convert asset_classes to enums if needed
        if asset_classes is not None:
            asset_classes = [
                ac if isinstance(ac, AssetClass) else AssetClass.from_string(ac)
                for ac in asset_classes
            ]
            
        # Get all available symbols
        with self._lock:
            all_symbols = list(self._price_data.keys())
            
        # Filter by asset class if provided
        if asset_classes is not None:
            filtered_symbols = []
            for sym in all_symbols:
                asset_info = self.get_asset_info(sym)
                if asset_info is not None and asset_info.asset_class in asset_classes:
                    filtered_symbols.append(sym)
        else:
            filtered_symbols = all_symbols
            
        if not filtered_symbols:
            self.logger.warning("No symbols available for systemic asset analysis")
            return []
            
        # Calculate minimum spanning tree (MST) for the market
        mst_result = self.calculate_market_mst(filtered_symbols)
        
        if not mst_result:
            self.logger.error("Failed to calculate market MST")
            return []
            
        # Get centrality measures
        centrality = mst_result.get("centrality", {})
        
        # Calculate contagion risk for each node with high centrality
        systemic_assets = []
        
        for symbol, central_value in sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:limit*2]:
            # Skip symbols with centrality below threshold
            if central_value < threshold:
                continue
                
            # Check if we have cached contagion risk
            if symbol in self._contagion_cache:
                contagion_risk = self._contagion_cache[symbol]
                
                systemic_assets.append({
                    "symbol": symbol,
                    "centrality": central_value,
                    "systemic_impact": contagion_risk.systemic_impact,
                    "risk_score": contagion_risk.risk_score,
                    "impact_count": len(contagion_risk.impact_symbols),
                    "asset_class": self._get_asset_class_str(symbol)
                })
            else:
                # Calculate contagion risk
                result = self.calculate_contagion_risk(symbol)
                
                if result:
                    systemic_assets.append({
                        "symbol": symbol,
                        "centrality": central_value,
                        "systemic_impact": result.get("systemic_impact", 0),
                        "risk_score": result.get("risk_score", 0),
                        "impact_count": len(result.get("impact_symbols", [])),
                        "asset_class": self._get_asset_class_str(symbol)
                    })
                    
        # Sort by systemic impact (descending)
        systemic_assets.sort(key=lambda x: x["systemic_impact"], reverse=True)
        
        # Limit results
        return systemic_assets[:limit]
    
    def calculate_market_mst(self, symbols: Optional[List[str]] = None, timeframe: Optional[str] = None) -> Dict[str, Any]:
        """
        Calculate Minimum Spanning Tree (MST) for the market.
        
        Args:
            symbols: List of symbols to include (all available if None)
            timeframe: Timeframe for analysis (default from config)
            
        Returns:
            Dictionary with MST analysis results
        """
        # Get defaults from config if not provided
        timeframe = timeframe or self.config["default_timeframe"]
        
        # Get all available symbols if not provided
        if symbols is None:
            with self._lock:
                symbols = list(self._price_data.keys())
                
        # Limit number of symbols for performance
        if len(symbols) > self.config["max_symbols"]:
            self.logger.warning(f"Too many symbols ({len(symbols)}), limiting to {self.config['max_symbols']}")
            symbols = symbols[:self.config["max_symbols"]]
            
        # Check cache
        cache_key = (tuple(sorted(symbols)), "mst", timeframe)
        cached_result = self._get_cached_result(cache_key)
        if cached_result is not None:
            return cached_result
            
        # Calculate correlation matrix
        corr_matrix = self.calculate_correlation_matrix(symbols, timeframe=timeframe)
        
        if corr_matrix.empty:
            self.logger.error("Failed to calculate correlation matrix")
            return {}
            
        # Need NetworkX for MST calculation
        if not NETWORKX_AVAILABLE:
            self.logger.error("NetworkX not available, cannot calculate MST")
            return {}
            
        # Create network from correlation matrix
        G = nx.Graph()
        
        # Add nodes
        for symbol in corr_matrix.index:
            G.add_node(symbol)
            
        # Add edges with weights based on correlation distance
        for i, symbol1 in enumerate(corr_matrix.index):
            for j, symbol2 in enumerate(corr_matrix.columns):
                if i < j:  # Avoid duplicates
                    # Higher correlation = shorter distance
                    distance = 1 - abs(corr_matrix.loc[symbol1, symbol2])
                    G.add_edge(symbol1, symbol2, weight=distance)
                    
        # Calculate Minimum Spanning Tree
        mst = nx.minimum_spanning_tree(G)
        
        # Extract edges
        mst_edges = []
        for u, v, data in mst.edges(data=True):
            weight = data['weight']
            correlation = 1 - weight
            mst_edges.append({
                "source": u,
                "target": v,
                "correlation": correlation,
                "distance": weight
            })
            
        # Calculate centrality measures
        degree_centrality = nx.degree_centrality(mst)
        betweenness_centrality = nx.betweenness_centrality(mst)
        
        # Combined centrality
        combined_centrality = {}
        for node in mst.nodes():
            combined_centrality[node] = (degree_centrality.get(node, 0) * 0.5 + 
                                       betweenness_centrality.get(node, 0) * 0.5)
            
        # Detect communities
        communities = []
        if self.config["use_community_detection"]:
            try:
                import community as community_louvain
                
                # Use Louvain method for community detection
                partition = community_louvain.best_partition(mst)
                
                # Group nodes by community
                community_nodes = {}
                for node, com_id in partition.items():
                    if com_id not in community_nodes:
                        community_nodes[com_id] = []
                    community_nodes[com_id].append(node)
                    
                # Create communities list
                for com_id, nodes in community_nodes.items():
                    communities.append({
                        "id": com_id,
                        "nodes": nodes,
                        "size": len(nodes)
                    })
                    
                # Sort by size (descending)
                communities.sort(key=lambda x: x["size"], reverse=True)
                
                # Limit to max communities
                if len(communities) > self.config["max_communities"]:
                    communities = communities[:self.config["max_communities"]]
                    
            except ImportError:
                self.logger.warning("python-louvain not available, skipping community detection")
                
        # Create result
        result = {
            "edges": mst_edges,
            "nodes": list(mst.nodes()),
            "centrality": combined_centrality,
            "degree_centrality": degree_centrality,
            "betweenness_centrality": betweenness_centrality,
            "communities": communities,
            "diameter": nx.diameter(mst),
            "average_shortest_path": nx.average_shortest_path_length(mst),
            "timeframe": timeframe,
            "timestamp": time.time()
        }
        
        # Cache result
        self._cache_result(cache_key, result)
        self._mst_cache[timeframe] = (result, time.time())
        
        return result
    
    # ----- Cross-Asset Flow Analysis Methods -----
    
    def analyze_market_flows(self, asset_classes: Optional[List[Union[str, AssetClass]]] = None,
                          timeframe: Optional[str] = None, window: Optional[int] = None) -> Dict[str, Any]:
        """
        Analyze flows between different asset classes or sectors.
        
        Args:
            asset_classes: List of asset classes to analyze (all if None)
            timeframe: Timeframe for analysis (default from config)
            window: Window size for rolling analysis (default from config)
            
        Returns:
            Dictionary with market flow analysis
        """
        # Get defaults from config if not provided
        timeframe = timeframe or self.config["default_timeframe"]
        window = window or self.config["rolling_window"]
        
        # Convert asset_classes to enums if needed
        if asset_classes is not None:
            asset_classes = [
                ac if isinstance(ac, AssetClass) else AssetClass.from_string(ac)
                for ac in asset_classes
            ]
        else:
            # Use all asset classes
            asset_classes = list(AssetClass)
            
        # Get all available symbols
        with self._lock:
            all_symbols = list(self._price_data.keys())
            
        # Group symbols by asset class
        asset_class_symbols = {ac: [] for ac in asset_classes}
        
        for symbol in all_symbols:
            asset_info = self.get_asset_info(symbol)
            
            if asset_info is not None and asset_info.asset_class in asset_classes:
                asset_class_symbols[asset_info.asset_class].append(symbol)
            elif asset_info is None:
                # Unknown asset class - try to guess from symbol name (simplified)
                if symbol.endswith('USD') or symbol.endswith('USDT'):
                    asset_class_symbols[AssetClass.CRYPTO].append(symbol)
                elif len(symbol) <= 5 and symbol.isupper():
                    asset_class_symbols[AssetClass.STOCK].append(symbol)
                elif 'BOND' in symbol or 'TREASURY' in symbol:
                    asset_class_symbols[AssetClass.BOND].append(symbol)
                elif 'OIL' in symbol or 'GOLD' in symbol or 'SILVER' in symbol:
                    asset_class_symbols[AssetClass.COMMODITY].append(symbol)
                elif 'USD' in symbol or 'EUR' in symbol or 'JPY' in symbol:
                    asset_class_symbols[AssetClass.CURRENCY].append(symbol)
                elif 'ETF' in symbol:
                    asset_class_symbols[AssetClass.ETF].append(symbol)
                else:
                    # Skip unknown
                    continue
                    
        # Filter out empty asset classes
        asset_class_symbols = {ac: symbols for ac, symbols in asset_class_symbols.items() if symbols}
        
        if not asset_class_symbols:
            self.logger.warning("No symbols available for market flow analysis")
            return {}
            
        # Calculate average price for each asset class
        asset_class_prices = {}
        
        for ac, symbols in asset_class_symbols.items():
            ac_prices = []
            
            for symbol in symbols:
                df = self.get_price_data(symbol)
                
                if df is not None and 'close' in df.columns:
                    # Normalize to start at 100
                    prices = df['close'].copy()
                    if not prices.empty:
                        prices = 100 * prices / prices.iloc[0]
                        ac_prices.append(prices)
                        
            if ac_prices:
                # Calculate average price for this asset class
                ac_prices_df = pd.concat(ac_prices, axis=1)
                asset_class_prices[ac] = ac_prices_df.mean(axis=1)
                
        if not asset_class_prices:
            self.logger.error("Failed to calculate asset class prices")
            return {}
            
        # Create dataframe with all asset class prices
        flows_df = pd.DataFrame({str(ac): prices for ac, prices in asset_class_prices.items()})
        
        # Calculate returns
        returns_df = flows_df.pct_change().dropna()
        
        # Calculate rolling correlations between asset classes
        correlations = {}
        
        for i, ac1 in enumerate(asset_class_prices.keys()):
            ac1_str = str(ac1)
            correlations[ac1_str] = {}
            
            for j, ac2 in enumerate(asset_class_prices.keys()):
                if i != j:
                    ac2_str = str(ac2)
                    
                    # Calculate rolling correlation
                    rolling_corr = returns_df[ac1_str].rolling(window=window).corr(returns_df[ac2_str])
                    
                    # Get last value
                    last_corr = rolling_corr.iloc[-1] if not rolling_corr.empty else 0.0
                    
                    # Calculate 30-day trend
                    trend_window = min(30, len(rolling_corr))
                    if trend_window > 5:
                        recent_corr = rolling_corr.iloc[-trend_window:]
                        trend_slope = np.polyfit(np.arange(trend_window), recent_corr, 1)[0]
                    else:
                        trend_slope = 0.0
                        
                    correlations[ac1_str][ac2_str] = {
                        "correlation": float(last_corr),
                        "trend": float(trend_slope)
                    }
                    
        # Calculate relative performance
        performance = {}
        
        for ac, prices in asset_class_prices.items():
            ac_str = str(ac)
            
            # Calculate 7-day, 30-day, and 90-day performance
            if len(prices) > 90:
                perf_90d = (prices.iloc[-1] / prices.iloc[-90] - 1) * 100
            else:
                perf_90d = 0.0
                
            if len(prices) > 30:
                perf_30d = (prices.iloc[-1] / prices.iloc[-30] - 1) * 100
            else:
                perf_30d = 0.0
                
            if len(prices) > 7:
                perf_7d = (prices.iloc[-1] / prices.iloc[-7] - 1) * 100
            else:
                perf_7d = 0.0
                
            performance[ac_str] = {
                "7d": float(perf_7d),
                "30d": float(perf_30d),
                "90d": float(perf_90d)
            }
            
        # Detect flow patterns
        flow_patterns = []
        
        # Sort asset classes by 7-day performance
        sorted_by_perf = sorted(performance.items(), key=lambda x: x[1]["7d"], reverse=True)
        
        # Check for flows from bottom performers to top performers
        if len(sorted_by_perf) >= 2:
            top_performers = sorted_by_perf[:len(sorted_by_perf)//3 + 1]
            bottom_performers = sorted_by_perf[-(len(sorted_by_perf)//3 + 1):]
            
            for bottom_ac, bottom_perf in bottom_performers:
                for top_ac, top_perf in top_performers:
                    # Check correlation trend
                    if bottom_ac in correlations and top_ac in correlations[bottom_ac]:
                        corr_data = correlations[bottom_ac][top_ac]
                        
                        # Negative correlation and negative trend suggests flow
                        if corr_data["correlation"] < -0.3 and corr_data["trend"] < 0:
                            flow_patterns.append({
                                "from": bottom_ac,
                                "to": top_ac,
                                "strength": abs(corr_data["correlation"]),
                                "trend": abs(corr_data["trend"]),
                                "performance_gap": top_perf["7d"] - bottom_perf["7d"]
                            })
                            
        # Sort flow patterns by strength
        flow_patterns.sort(key=lambda x: x["strength"], reverse=True)
        
        # Create result
        result = {
            "correlations": correlations,
            "performance": performance,
            "flow_patterns": flow_patterns,
            "timeframe": timeframe,
            "window": window,
            "timestamp": time.time()
        }
        
        return result
    
    # ----- Cross-Asset Regime Analysis Methods -----
    
    def analyze_regime_consistency(self, symbols: Optional[List[str]] = None,
                               timeframe: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze regime consistency across different assets.
        
        Args:
            symbols: List of symbols to analyze (all available if None)
            timeframe: Timeframe for analysis (default from config)
            
        Returns:
            Dictionary with regime consistency analysis
        """
        # Get defaults from config if not provided
        timeframe = timeframe or self.config["default_timeframe"]
        
        # Get all available symbols if not provided
        if symbols is None:
            with self._lock:
                symbols = list(self._price_data.keys())
                
        # Limit number of symbols for performance
        if len(symbols) > self.config["max_symbols"]:
            self.logger.warning(f"Too many symbols ({len(symbols)}), limiting to {self.config['max_symbols']}")
            symbols = symbols[:self.config["max_symbols"]]
            
        # Detect regimes for each symbol
        regimes = {}
        
        for symbol in symbols:
            df = self.get_price_data(symbol)
            
            if df is None:
                continue
                
            if 'close' not in df.columns:
                continue
                
            # Detect regime (simplified using rolling volatility)
            prices = df['close']
            returns = prices.pct_change().dropna()
            
            # Calculate rolling volatility
            rolling_vol = returns.rolling(window=20).std() * (252 ** 0.5)  # Annualized
            
            # Smooth volatility
            if SCIPY_AVAILABLE:
                smoothed_vol = gaussian_filter1d(rolling_vol.values, sigma=3)
                rolling_vol = pd.Series(smoothed_vol, index=rolling_vol.index)
                
            # Define regimes based on volatility
            vol_regimes = pd.Series(index=rolling_vol.index)
            
            # Calculate vol percentiles for this symbol
            low_vol = rolling_vol.quantile(0.25)
            high_vol = rolling_vol.quantile(0.75)
            
            # Determine trend
            ma_short = prices.rolling(window=20).mean()
            ma_long = prices.rolling(window=50).mean()
            trend = (ma_short > ma_long).astype(int) * 2 - 1  # 1 for uptrend, -1 for downtrend
            
            # Combine volatility and trend for regime
            # 4 regimes: low vol uptrend, high vol uptrend, low vol downtrend, high vol downtrend
            for i in range(len(rolling_vol)):
                if i < 50:  # Not enough history
                    vol_regimes.iloc[i] = 'unknown'
                else:
                    vol = rolling_vol.iloc[i]
                    tr = trend.iloc[i]
                    
                    if vol < low_vol:
                        if tr > 0:
                            vol_regimes.iloc[i] = 'low_vol_uptrend'
                        else:
                            vol_regimes.iloc[i] = 'low_vol_downtrend'
                    elif vol > high_vol:
                        if tr > 0:
                            vol_regimes.iloc[i] = 'high_vol_uptrend'
                        else:
                            vol_regimes.iloc[i] = 'high_vol_downtrend'
                    else:
                        if tr > 0:
                            vol_regimes.iloc[i] = 'medium_vol_uptrend'
                        else:
                            vol_regimes.iloc[i] = 'medium_vol_downtrend'
                            
            # Store results
            regimes[symbol] = vol_regimes
            
        if not regimes:
            self.logger.error("No regime data available for any symbols")
            return {}
            
        # Align all series to common index
        all_series = list(regimes.values())
        common_idx = all_series[0].index
        
        for series in all_series[1:]:
            common_idx = common_idx.intersection(series.index)
            
        # Filter series to common index
        aligned_regimes = {}
        for symbol, regime_series in regimes.items():
            aligned_regimes[symbol] = regime_series.loc[common_idx]
            
        # Analyze consistency at each point in time
        consistency = pd.Series(index=common_idx, dtype=float)
        
        for idx in common_idx:
            # Count regime frequencies at this timestamp
            regimes_at_t = [aligned_regimes[s].loc[idx] for s in aligned_regimes]
            regime_counts = {}
            
            for r in regimes_at_t:
                if r not in regime_counts:
                    regime_counts[r] = 0
                regime_counts[r] += 1
                
            # Calculate consistency as ratio of most frequent regime
            max_count = max(regime_counts.values()) if regime_counts else 0
            consistency.loc[idx] = max_count / len(aligned_regimes)
            
        # Calculate overall consistency
        avg_consistency = float(consistency.mean())
        
        # Get the latest regime for each symbol
        latest_regimes = {symbol: series.iloc[-1] for symbol, series in aligned_regimes.items()}
        
        # Count regime frequencies in latest data
        latest_counts = {}
        for regime in latest_regimes.values():
            if regime not in latest_counts:
                latest_counts[regime] = 0
            latest_counts[regime] += 1
            
        # Find dominant regime
        dominant_regime = max(latest_counts.items(), key=lambda x: x[1]) if latest_counts else (None, 0)
        
        # Group symbols by regime
        regime_groups = {}
        for symbol, regime in latest_regimes.items():
            if regime not in regime_groups:
                regime_groups[regime] = []
            regime_groups[regime].append(symbol)
            
        # Create result
        result = {
            "average_consistency": avg_consistency,
            "latest_consistency": float(consistency.iloc[-1]) if not consistency.empty else 0.0,
            "dominant_regime": dominant_regime[0],
            "dominant_ratio": dominant_regime[1] / len(latest_regimes) if latest_regimes else 0.0,
            "regime_groups": regime_groups,
            "regime_counts": latest_counts,
            "timeframe": timeframe,
            "timestamp": time.time()
        }
        
        return result
    
    def get_metrics_for_symbol(self, symbol):
        """
        Get cross-asset metrics for a specific symbol.
        
        Args:
            symbol: Asset symbol
            
        Returns:
            dict: Metric data
        """
        with self.lock:
            # Get most recent data
            if not self.correlation_matrices:
                return {}
            
            latest_ts = max(self.correlation_matrices.keys())
            corr_matrix = self.correlation_matrices.get(latest_ts, {})
            
            lead_lag = self.lead_lag_data.get(latest_ts, {})
            causality = self.causality_data.get(latest_ts, {})
            
            # Extract relevant metrics
            metrics = {
                'leadership_score': lead_lag.get('leadership_scores', {}).get(symbol, 0.0),
                'correlations': corr_matrix.get('filtered', {}).get(symbol, {}),
                'lead_relationships': lead_lag.get('matrix', {}).get(symbol, {}),
                'causes': causality.get('matrix', {}).get(symbol, {})
            }
            
            return metrics
        
        
    # ----- General Utilities -----
    
    def stop(self):
        """Stop background processing and cleanup resources."""
        self._is_running = False
        
        # Wait for processing queue to finish
        if hasattr(self, '_processing_queue'):
            try:
                self._processing_queue.join(timeout=1.0)
            except:
                pass
                
        # Clear caches to free memory
        with self._lock:
            self._relationship_cache.clear()
            self._cache_timestamps.clear()
            self._contagion_cache.clear()
            self._mst_cache.clear()
