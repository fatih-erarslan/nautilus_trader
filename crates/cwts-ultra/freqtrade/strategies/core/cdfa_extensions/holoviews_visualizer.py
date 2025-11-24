#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Holoviews Visualization for Market Analysis

Enterprise-grade visualization system using Holoviews to create interactive
heatmaps and other visualizations for analyzing CDFA and market data.

Author: Created on May 8, 2025
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple, Set, Callable, TypedDict
import time
import os
import json
import colorcet as cc
import datetime
from enum import Enum
import warnings
import pathlib
from threading import Lock, Event

# Import Holoviews and supporting libraries
try:
    import holoviews as hv
    from holoviews import opts
    from bokeh.models import HoverTool, Range1d, ColumnDataSource, LinearColorMapper, ColorBar
    from bokeh.plotting import figure
    from bokeh.palettes import Spectral11, Viridis256, Inferno256, RdYlGn11
    import panel as pn
    import param
    HOLOVIEWS_AVAILABLE = True
except ImportError:
    HOLOVIEWS_AVAILABLE = False
    warnings.warn("Holoviews not available. Install with 'pip install holoviews bokeh panel colorcet'")

# Type definitions for configuration
class VisualizerConfigDict(TypedDict, total=False):
    default_width: int
    default_height: int
    default_cmap: str
    default_tools: List[str]
    show_toolbar: bool
    heatmap_nbins: int
    interactive_mode: bool
    custom_theme: str
    background_color: str
    font_size: str
    tooltips_enabled: bool
    enable_widgets: bool
    auto_update_interval: Optional[int]
    data_cache_ttl: int
    export_formats: List[str]
    default_output_dir: str
    high_dpi_mode: bool
    max_items_per_plot: int
    log_level: int
    config_version: str

class HeatmapType(str, Enum):
    """Types of heatmaps available in the visualization system."""
    OPPORTUNITY = "opportunity"
    REGIME = "regime"
    CORRELATION = "correlation"
    ANALYZER = "analyzer"
    WHALE = "whale"
    VOLATILITY = "volatility"
    FEEDBACK = "feedback"
    CUSTOM = "custom"

class HoloviewsVisualizer:
    """
    Advanced visualization system using Holoviews to create interactive
    heatmaps and other visualizations for market data and analysis results.
    
    Features:
    - Interactive heatmaps for different metrics (opportunity, regime, correlation)
    - Customizable color maps and visualization options
    - Dashboard creation with multiple views
    - Export capabilities for integration with other tools
    - Multi-visualization composing for complex data relationships
    """
    
    # Class-level attribute for default configuration parameters
    DEFAULT_CONFIG: VisualizerConfigDict = {
        "default_width": 800,
        "default_height": 600,
        "default_cmap": "fire",
        "default_tools": ["hover", "pan", "wheel_zoom", "box_zoom", "reset", "save"],
        "show_toolbar": True,
        "heatmap_nbins": 50,
        "interactive_mode": True,
        "custom_theme": "default",
        "background_color": "#ffffff",
        "font_size": "12pt",
        "tooltips_enabled": True,
        "enable_widgets": True,
        "auto_update_interval": None,
        "data_cache_ttl": 300,  # 5 minutes
        "export_formats": ["html", "png", "svg"],
        "default_output_dir": "~/cdfa_visualizations",
        "high_dpi_mode": False,
        "max_items_per_plot": 50,
        "log_level": logging.INFO,
        "config_version": "1.0.0"
    }
    
    # Available color maps
    AVAILABLE_CMAPS = {
        "fire": cc.fire,
        "rainbow": cc.rainbow,
        "blues": cc.CET_L7,   # A proper blue colormap
        "reds": cc.CET_R1,    # A proper red colormap
        "greens": cc.CET_L8,  # A proper green colormap
        "coolwarm": cc.coolwarm,
        "bgy": cc.bgy,
        "bjy": cc.bjy,
        "kbc": cc.kbc,
        "viridis": Viridis256,
        "inferno": Inferno256,
        "spectral": Spectral11,
        "rdylgn": RdYlGn11
    }
    
    def __init__(self, market_data_fetcher=None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize visualizer.
        
        Args:
            market_data_fetcher: Optional market data fetcher instance
            config: Optional configuration dictionary
        """
        # Set up logging
        self.logger = logging.getLogger(f"{__name__}.HoloviewsVisualizer")
        
        # Check if Holoviews is available
        if not HOLOVIEWS_AVAILABLE:
            self.logger.error("Holoviews is not available. Please install required dependencies.")
            raise ImportError("Holoviews is required for visualization")
        
        # Merge with provided config
        self.config = {**self.DEFAULT_CONFIG}
        if config:
            self._update_config(config)
            
        # Set log level
        self.logger.setLevel(self.config["log_level"])
        
        # Store market data fetcher
        self.market_data_fetcher = market_data_fetcher
        
        # Initialize Holoviews with Bokeh backend
        hv.extension('bokeh')
        
        # Set default options
        self.width = self.config["default_width"]
        self.height = self.config["default_height"]
        self.cmap = self._get_colormap(self.config["default_cmap"])
        
        # Set up data cache
        self._data_cache = {}
        self._data_cache_timestamps = {}
        self._data_cache_lock = Lock()
        
        # Set up panel dashboard if widgets enabled
        if self.config["enable_widgets"]:
            self._setup_panel_dashboard()
            
        # Initialize update thread if auto-update enabled
        self._auto_update_thread = None
        self.running = True
        if self.config["auto_update_interval"]:
            self._start_auto_update()
            
        self.logger.info("HoloviewsVisualizer initialized")
    
    def _update_config(self, config: Dict[str, Any]):
        """
        Update configuration with provided values.
        
        Args:
            config: New configuration values
        """
        for key, value in config.items():
            if key in self.config:
                self.config[key] = value
            else:
                self.logger.warning(f"Unknown configuration parameter: {key}")
    
    def _get_colormap(self, cmap_name: str) -> Any:
        """
        Get colormap by name.
        
        Args:
            cmap_name: Colormap name
            
        Returns:
            Colormap
        """
        if cmap_name in self.AVAILABLE_CMAPS:
            return self.AVAILABLE_CMAPS[cmap_name]
        else:
            self.logger.warning(f"Unknown colormap: {cmap_name}, using default")
            return self.AVAILABLE_CMAPS[self.DEFAULT_CONFIG["default_cmap"]]
    
    def _setup_panel_dashboard(self):
        """Set up Panel dashboard for interactive widgets."""
        try:
            # Create parameter class for widgets
            class VisualizerParams(param.Parameterized):
                heatmap_type = param.ObjectSelector(default=HeatmapType.OPPORTUNITY, 
                                                 objects=[t.value for t in HeatmapType])
                colormap = param.ObjectSelector(default=self.config["default_cmap"], 
                                              objects=list(self.AVAILABLE_CMAPS.keys()))
                width = param.Integer(default=self.config["default_width"], bounds=(200, 2000))
                height = param.Integer(default=self.config["default_height"], bounds=(200, 2000))
                max_items = param.Integer(default=self.config["max_items_per_plot"], bounds=(5, 200))
                
                @param.depends('heatmap_type', 'colormap', 'width', 'height', 'max_items')
                def view(self):
                    """Generate visualization based on parameters."""
                    width = self.width
                    height = self.height
                    cmap = self._get_colormap(self.colormap)
                    
                    if self.heatmap_type == HeatmapType.OPPORTUNITY.value:
                        return self.create_opportunity_heatmap(width=width, height=height, cmap=cmap, 
                                                            limit=self.max_items)
                    elif self.heatmap_type == HeatmapType.REGIME.value:
                        return self.create_regime_heatmap(width=width, height=height, cmap=cmap, 
                                                        limit=self.max_items)
                    elif self.heatmap_type == HeatmapType.CORRELATION.value:
                        return self.create_correlation_heatmap(width=width, height=height, cmap=cmap, 
                                                            limit=self.max_items)
                    elif self.heatmap_type == HeatmapType.ANALYZER.value:
                        return self.create_analyzer_heatmap(width=width, height=height, cmap=cmap, 
                                                         limit=self.max_items)
                    elif self.heatmap_type == HeatmapType.WHALE.value:
                        return self.create_whale_activity_heatmap(width=width, height=height, cmap=cmap, 
                                                              limit=self.max_items)
                    else:
                        return hv.Text(0, 0, f"Visualization type '{self.heatmap_type}' not implemented yet")
                
                def _get_colormap(self, cmap_name):
                    """Get colormap from name."""
                    return HoloviewsVisualizer.AVAILABLE_CMAPS.get(
                        cmap_name, 
                        HoloviewsVisualizer.AVAILABLE_CMAPS[HoloviewsVisualizer.DEFAULT_CONFIG["default_cmap"]]
                    )
            
            # Create instance with reference to self
            params = VisualizerParams()
            params.create_opportunity_heatmap = self.create_opportunity_heatmap
            params.create_regime_heatmap = self.create_regime_heatmap
            params.create_correlation_heatmap = self.create_correlation_heatmap
            params.create_analyzer_heatmap = self.create_analyzer_heatmap
            params.create_whale_activity_heatmap = self.create_whale_activity_heatmap
            
            # Create dashboard
            self.dashboard = pn.Column(
                pn.Row(
                    pn.Column(
                        "# CDFA Market Intelligence Dashboard",
                        pn.Param(params, 
                                widgets={
                                    'heatmap_type': pn.widgets.Select,
                                    'colormap': pn.widgets.Select,
                                    'width': pn.widgets.IntSlider,
                                    'height': pn.widgets.IntSlider,
                                    'max_items': pn.widgets.IntInput
                                }),
                        width=300
                    ),
                    params.view
                )
            )
            
            self.logger.info("Panel dashboard initialized")
            
        except Exception as e:
            self.logger.error(f"Error setting up Panel dashboard: {e}")
            self.dashboard = None
    
    def _start_auto_update(self):
        """Start auto-update thread for visualizations."""
        import threading
        
        def update_loop():
            """Background thread for updating visualizations."""
            while self.running:
                try:
                    # Clear expired cache entries
                    self._clear_expired_cache()
                    
                    # Sleep until next update
                    for _ in range(self.config["auto_update_interval"]):
                        if not self.running:
                            break
                        time.sleep(1)
                        
                except Exception as e:
                    self.logger.error(f"Error in auto-update loop: {e}")
                    time.sleep(10)  # Sleep longer on error
        
        # Start thread
        self._auto_update_thread = threading.Thread(
            target=update_loop,
            daemon=True,
            name="Visualizer-AutoUpdate"
        )
        self._auto_update_thread.start()
        
        self.logger.info("Auto-update thread started")
    
    def _clear_expired_cache(self):
        """Clear expired entries from data cache."""
        with self._data_cache_lock:
            current_time = time.time()
            expired_keys = []
            
            for key, timestamp in self._data_cache_timestamps.items():
                if current_time - timestamp > self.config["data_cache_ttl"]:
                    expired_keys.append(key)
                    
            for key in expired_keys:
                self._data_cache.pop(key, None)
                self._data_cache_timestamps.pop(key, None)
                
            if expired_keys:
                self.logger.debug(f"Cleared {len(expired_keys)} expired cache entries")
    
    def _get_cached_data(self, cache_key: str) -> Optional[Any]:
        """
        Get data from cache if available and not expired.
        
        Args:
            cache_key: Cache key
            
        Returns:
            Cached data or None if not available
        """
        with self._data_cache_lock:
            if cache_key in self._data_cache:
                timestamp = self._data_cache_timestamps.get(cache_key, 0)
                if time.time() - timestamp <= self.config["data_cache_ttl"]:
                    return self._data_cache[cache_key]
            return None
    
    def _set_cached_data(self, cache_key: str, data: Any):
        """
        Store data in cache.
        
        Args:
            cache_key: Cache key
            data: Data to cache
        """
        with self._data_cache_lock:
            self._data_cache[cache_key] = data
            self._data_cache_timestamps[cache_key] = time.time()
    
    def create_opportunity_heatmap(self, data=None, width=None, height=None, 
                                cmap=None, limit=None):
        """
        Create heatmap of trading opportunities.
        
        Args:
            data: Optional data dictionary, will fetch from market_data_fetcher if None
            width: Plot width
            height: Plot height
            cmap: Colormap
            limit: Maximum number of items to include
            
        Returns:
            Holoviews object
        """
        # Use default parameters if not provided
        width = width or self.width
        height = height or self.height
        cmap = cmap or self.cmap
        limit = limit or self.config["max_items_per_plot"]
        
        # Check cache first
        cache_key = f"opportunity_heatmap_{width}_{height}_{limit}"
        cached_result = self._get_cached_data(cache_key)
        if cached_result is not None:
            return cached_result
            
        if data is None and self.market_data_fetcher:
            # Get top pairs
            pairs = self.market_data_fetcher.get_pair_rankings(limit=limit)
            
            # Prepare data for visualization
            data = []
            for symbol, score in pairs:
                meta = self.market_data_fetcher.get_pair_metadata(symbol)
                if not meta:
                    continue
                
                # Get base and quote from symbol
                base = meta.base_currency
                quote = meta.quote_currency
                
                # Only include if base and quote are valid
                if not base or not quote:
                    continue
                
                data.append({
                    'base': base,
                    'quote': quote,
                    'symbol': symbol,
                    'score': score,
                    'regime': meta.regime_state.value if hasattr(meta.regime_state, 'value') else meta.regime_state,
                    'volatility': meta.volatility,
                    'success_rate': meta.success_rate,
                    'opportunity': meta.opportunity_score.value if hasattr(meta.opportunity_score, 'value') else meta.opportunity_score,
                    'liquidity': meta.liquidity
                })
        
        if not data:
            result = hv.Text(0, 0, "No data available")
            self._set_cached_data(cache_key, result)
            return result
            
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(data)
        
        # Aggregate by base/quote if there are duplicates
        if df.duplicated(['base', 'quote']).any():
            df = df.groupby(['base', 'quote']).agg({
                'score': 'mean',
                'volatility': 'mean',
                'success_rate': 'mean',
                'opportunity': 'mean',
                'liquidity': 'sum',
                'symbol': lambda x: ', '.join(x[:3]) + ('...' if len(x) > 3 else '')
            }).reset_index()
        
        # Create heatmap
        heatmap = hv.HeatMap(
            df, 
            kdims=['base', 'quote'], 
            vdims=['score', 'regime', 'volatility', 'success_rate', 'opportunity', 'liquidity', 'symbol']
        ).opts(
            width=width,
            height=height,
            colorbar=True,
            cmap=cmap,
            tools=self.config["default_tools"],
            xrotation=45,
            toolbar='above' if self.config["show_toolbar"] else None,
            title="Trading Pair Opportunity Heatmap",
            ylabel="Quote Currency",
            xlabel="Base Currency"
        )
        
        # Add hover tooltip if enabled
        if self.config["tooltips_enabled"]:
            hover = HoverTool(tooltips=[
                ('Symbol', '@symbol'),
                ('Score', '@score{0.00}'),
                ('Regime', '@regime'),
                ('Volatility', '@volatility{0.00}'),
                ('Success Rate', '@success_rate{0.00}'),
                ('Opportunity', '@opportunity'),
                ('Liquidity', '@liquidity{0.00 a}')
            ])
            
            heatmap = heatmap.opts(tools=[hover])
        
        # Store in cache
        self._set_cached_data(cache_key, heatmap)
        
        return heatmap
    
    def create_regime_heatmap(self, data=None, width=None, height=None, 
                           cmap=None, limit=None):
        """
        Create heatmap of market regimes.
        
        Args:
            data: Optional data dictionary, will fetch from market_data_fetcher if None
            width: Plot width
            height: Plot height
            cmap: Colormap
            limit: Maximum number of items to include
            
        Returns:
            Holoviews object
        """
        # Use default parameters if not provided
        width = width or self.width
        height = height or self.height
        cmap = cmap or self.AVAILABLE_CMAPS["rdylgn"]  # Special colormap for regimes
        limit = limit or self.config["max_items_per_plot"]
        
        # Check cache first
        cache_key = f"regime_heatmap_{width}_{height}_{limit}"
        cached_result = self._get_cached_data(cache_key)
        if cached_result is not None:
            return cached_result
            
        if data is None and self.market_data_fetcher:
            # Get active pairs
            symbols = self.market_data_fetcher.get_active_pairs()
            if limit:
                symbols = symbols[:limit]
            
            # Prepare data for visualization
            data = []
            for symbol in symbols:
                meta = self.market_data_fetcher.get_pair_metadata(symbol)
                if not meta:
                    continue
                
                # Get base and quote from symbol
                base = meta.base_currency
                quote = meta.quote_currency
                
                # Only include if base and quote are valid
                if not base or not quote:
                    continue
                
                # Map regime to numeric value for coloring
                regime_map = {
                    "bullish": 1.0,
                    "accumulation": 0.75,
                    "neutral": 0.5,
                    "ranging": 0.5,
                    "distribution": 0.25,
                    "bearish": 0.0,
                    "volatile": 0.3,
                    "unknown": 0.5
                }
                
                regime = meta.regime_state
                if hasattr(regime, 'value'):
                    regime = regime.value
                
                regime_value = regime_map.get(regime.lower(), 0.5)
                
                data.append({
                    'base': base,
                    'quote': quote,
                    'symbol': symbol,
                    'regime': regime,
                    'regime_value': regime_value,
                    'confidence': meta.regime_confidence,
                    'trending': 'Yes' if meta.is_trending else 'No',
                    'cycle_strength': meta.cycle_strength
                })
        
        if not data:
            result = hv.Text(0, 0, "No data available")
            self._set_cached_data(cache_key, result)
            return result
            
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(data)
        
        # Aggregate by base/quote if there are duplicates
        if df.duplicated(['base', 'quote']).any():
            df = df.groupby(['base', 'quote']).agg({
                'regime_value': 'mean',
                'regime': lambda x: max(set(x), key=list(x).count),  # Most common regime
                'confidence': 'mean',
                'trending': lambda x: 'Yes' if sum(y == 'Yes' for y in x) / len(x) > 0.5 else 'No',
                'cycle_strength': 'mean',
                'symbol': lambda x: ', '.join(x[:3]) + ('...' if len(x) > 3 else '')
            }).reset_index()
        
        # Create heatmap
        heatmap = hv.HeatMap(
            df, 
            kdims=['base', 'quote'], 
            vdims=['regime_value', 'regime', 'confidence', 'trending', 'cycle_strength', 'symbol']
        ).opts(
            width=width,
            height=height,
            colorbar=True,
            cmap=cmap,
            tools=self.config["default_tools"],
            xrotation=45,
            toolbar='above' if self.config["show_toolbar"] else None,
            title="Market Regime Heatmap",
            ylabel="Quote Currency",
            xlabel="Base Currency",
            clim=(0, 1)  # Fixed range for regime values
        )
        
        # Add hover tooltip if enabled
        if self.config["tooltips_enabled"]:
            hover = HoverTool(tooltips=[
                ('Symbol', '@symbol'),
                ('Regime', '@regime'),
                ('Confidence', '@confidence{0.00}'),
                ('Trending', '@trending'),
                ('Cycle Strength', '@cycle_strength{0.00}')
            ])
            
            heatmap = heatmap.opts(tools=[hover])
        
        # Store in cache
        self._set_cached_data(cache_key, heatmap)
        
        return heatmap
    
    def create_correlation_heatmap(self, symbols=None, timeframe="1d", lookback="30d",
                                width=None, height=None, cmap=None, limit=None):
        """
        Create correlation heatmap for selected symbols.
        
        Args:
            symbols: List of symbols (None for active pairs)
            timeframe: Timeframe for data
            lookback: Lookback period
            width: Plot width
            height: Plot height
            cmap: Colormap
            limit: Maximum number of items to include
            
        Returns:
            Holoviews object
        """
        # Use default parameters if not provided
        width = width or self.width
        height = height or self.height
        cmap = cmap or self.AVAILABLE_CMAPS["coolwarm"]  # Special colormap for correlation
        limit = limit or self.config["max_items_per_plot"]
        
        # Check cache first
        cache_key = f"correlation_heatmap_{timeframe}_{lookback}_{width}_{height}_{limit}"
        if symbols:
            cache_key += f"_{','.join(sorted(symbols[:5]))}"
            
        cached_result = self._get_cached_data(cache_key)
        if cached_result is not None:
            return cached_result
            
        if symbols is None and self.market_data_fetcher:
            # Get active pairs
            symbols = self.market_data_fetcher.get_active_pairs()
            
            # Limit to a reasonable number
            if limit:
                symbols = symbols[:limit]
        
        if not symbols:
            result = hv.Text(0, 0, "No symbols available")
            self._set_cached_data(cache_key, result)
            return result
            
        # Fetch data
        if self.market_data_fetcher:
            data_dict = self.market_data_fetcher.fetch_data(
                symbols=symbols,
                timeframe=timeframe,
                lookback=lookback
            )
        else:
            result = hv.Text(0, 0, "No data fetcher available")
            self._set_cached_data(cache_key, result)
            return result
            
        if not data_dict:
            result = hv.Text(0, 0, "No data available")
            self._set_cached_data(cache_key, result)
            return result
            
        # Extract close prices and standardize across time
        prices = {}
        for symbol, df in data_dict.items():
            if 'close' in df.columns:
                # Resample to ensure consistent timestamps
                if not isinstance(df.index, pd.DatetimeIndex):
                    if 'timestamp' in df.columns:
                        df.set_index('timestamp', inplace=True)
                    else:
                        # Skip if no proper time index
                        continue
                        
                # Resample to daily if not already
                if timeframe != '1d':
                    price_series = df['close'].resample('1D').last()
                else:
                    price_series = df['close']
                    
                # Forward fill any missing values
                price_series = price_series.fillna(method='ffill')
                
                # Only keep if we have sufficient data
                if len(price_series) > 10:
                    prices[symbol] = price_series
                
        if not prices:
            result = hv.Text(0, 0, "Insufficient price data for correlation analysis")
            self._set_cached_data(cache_key, result)
            return result
            
        try:
            # Create DataFrame of prices with aligned timestamps
            price_df = pd.DataFrame(prices)
            
            # Drop any rows with NaN (times where not all assets have data)
            price_df = price_df.dropna(axis=0, how='any')
            
            # Compute returns
            returns_df = price_df.pct_change().dropna()
            
            # Calculate correlation matrix
            corr_matrix = returns_df.corr()
        except Exception as e:
            self.logger.error(f"Error calculating correlation matrix: {e}")
            result = hv.Text(0, 0, f"Error calculating correlations: {str(e)}")
            self._set_cached_data(cache_key, result)
            return result
        
        # Convert to long format for HeatMap
        corr_data = []
        for i, row in enumerate(corr_matrix.index):
            for j, col in enumerate(corr_matrix.columns):
                corr_data.append({
                    'x': row, 
                    'y': col, 
                    'correlation': corr_matrix.iloc[i, j]
                })
                
        # Convert to DataFrame
        corr_df = pd.DataFrame(corr_data)
        
        # Create heatmap
        heatmap = hv.HeatMap(
            corr_df, 
            kdims=['x', 'y'], 
            vdims=['correlation']
        ).opts(
            width=width,
            height=height,
            colorbar=True,
            cmap=cmap,
            tools=self.config["default_tools"],
            xrotation=45,
            toolbar='above' if self.config["show_toolbar"] else None,
            title=f"Correlation Heatmap ({lookback} lookback, {timeframe} interval)",
            ylabel="Symbol",
            xlabel="Symbol",
            clim=(-1, 1)  # Correlation range
        )
        
        # Add hover tooltip if enabled
        if self.config["tooltips_enabled"]:
            hover = HoverTool(tooltips=[
                ('Symbols', '@{x} vs @{y}'),
                ('Correlation', '@{correlation}{0.00}')
            ])
            
            heatmap = heatmap.opts(tools=[hover])
        
        # Store in cache
        self._set_cached_data(cache_key, heatmap)
        
        return heatmap
    
    def create_analyzer_heatmap(self, analyzer_type=None, width=None, height=None, 
                             cmap=None, limit=None):
        """
        Create heatmap of analyzer scores.
        
        Args:
            analyzer_type: Specific analyzer to visualize (None for composite)
            width: Plot width
            height: Plot height
            cmap: Colormap
            limit: Maximum number of items to include
            
        Returns:
            Holoviews object
        """
        # Use default parameters if not provided
        width = width or self.width
        height = height or self.height
        cmap = cmap or self.cmap
        limit = limit or self.config["max_items_per_plot"]
        
        # Check cache first
        cache_key = f"analyzer_heatmap_{analyzer_type}_{width}_{height}_{limit}"
        cached_result = self._get_cached_data(cache_key)
        if cached_result is not None:
            return cached_result
            
        if not self.market_data_fetcher:
            result = hv.Text(0, 0, "No data fetcher available")
            self._set_cached_data(cache_key, result)
            return result
            
        # Get active pairs
        symbols = self.market_data_fetcher.get_active_pairs()
        if limit:
            symbols = symbols[:limit]
        
        if not symbols:
            result = hv.Text(0, 0, "No symbols available")
            self._set_cached_data(cache_key, result)
            return result
            
        # Prepare data for visualization
        data = []
        for symbol in symbols:
            meta = self.market_data_fetcher.get_pair_metadata(symbol)
            if not meta:
                continue
            
            # Get base and quote from symbol
            base = meta.base_currency
            quote = meta.quote_currency
            
            # Only include if base and quote are valid
            if not base or not quote:
                continue
            
            if analyzer_type and analyzer_type in meta.analyzer_scores:
                score = meta.analyzer_scores[analyzer_type]
                
                data.append({
                    'base': base,
                    'quote': quote,
                    'symbol': symbol,
                    'score': score,
                    'analyzer': analyzer_type
                })
            elif not analyzer_type:
                # Composite score from all analyzers
                scores = meta.analyzer_scores
                if scores:
                    avg_score = sum(scores.values()) / len(scores)
                    
                    data.append({
                        'base': base,
                        'quote': quote,
                        'symbol': symbol,
                        'score': avg_score,
                        'analyzers': ', '.join(scores.keys()),
                        'analyzer_count': len(scores)
                    })
        
        if not data:
            result = hv.Text(0, 0, "No analyzer data available")
            self._set_cached_data(cache_key, result)
            return result
            
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(data)
        
        # Aggregate by base/quote if there are duplicates
        if df.duplicated(['base', 'quote']).any():
            if analyzer_type:
                df = df.groupby(['base', 'quote']).agg({
                    'score': 'mean',
                    'analyzer': 'first',
                    'symbol': lambda x: ', '.join(x[:3]) + ('...' if len(x) > 3 else '')
                }).reset_index()
            else:
                df = df.groupby(['base', 'quote']).agg({
                    'score': 'mean',
                    'analyzers': 'first',
                    'analyzer_count': 'mean',
                    'symbol': lambda x: ', '.join(x[:3]) + ('...' if len(x) > 3 else '')
                }).reset_index()
        
        # Title based on analyzer type
        title = f"{analyzer_type.capitalize()} Analyzer Heatmap" if analyzer_type else "Composite Analyzer Heatmap"
        
        # Create heatmap
        if analyzer_type:
            heatmap = hv.HeatMap(
                df, 
                kdims=['base', 'quote'], 
                vdims=['score', 'analyzer', 'symbol']
            ).opts(
                width=width,
                height=height,
                colorbar=True,
                cmap=cmap,
                tools=self.config["default_tools"],
                xrotation=45,
                toolbar='above' if self.config["show_toolbar"] else None,
                title=title,
                ylabel="Quote Currency",
                xlabel="Base Currency"
            )
            
            # Add hover tooltip if enabled
            if self.config["tooltips_enabled"]:
                hover = HoverTool(tooltips=[
                    ('Symbol', '@symbol'),
                    ('Score', '@score{0.00}'),
                    ('Analyzer', '@analyzer')
                ])
                
                heatmap = heatmap.opts(tools=[hover])
        else:
            heatmap = hv.HeatMap(
                df, 
                kdims=['base', 'quote'], 
                vdims=['score', 'analyzers', 'analyzer_count', 'symbol']
            ).opts(
                width=width,
                height=height,
                colorbar=True,
                cmap=cmap,
                tools=self.config["default_tools"],
                xrotation=45,
                toolbar='above' if self.config["show_toolbar"] else None,
                title=title,
                ylabel="Quote Currency",
                xlabel="Base Currency"
            )
            
            # Add hover tooltip if enabled
            if self.config["tooltips_enabled"]:
                hover = HoverTool(tooltips=[
                    ('Symbol', '@symbol'),
                    ('Score', '@score{0.00}'),
                    ('Analyzers', '@analyzers'),
                    ('Analyzer Count', '@analyzer_count{0}')
                ])
                
                heatmap = heatmap.opts(tools=[hover])
        
        # Store in cache
        self._set_cached_data(cache_key, heatmap)
        
        return heatmap
    
    def create_whale_activity_heatmap(self, width=None, height=None, cmap=None, limit=None):
        """
        Create heatmap of whale activity.
        
        Args:
            width: Plot width
            height: Plot height
            cmap: Colormap
            limit: Maximum number of items to include
            
        Returns:
            Holoviews object
        """
        # Use default parameters if not provided
        width = width or self.width
        height = height or self.height
        cmap = cmap or self.AVAILABLE_CMAPS["plasma"]  # Special colormap for whale activity
        limit = limit or self.config["max_items_per_plot"]
        
        # Check cache first
        cache_key = f"whale_heatmap_{width}_{height}_{limit}"
        cached_result = self._get_cached_data(cache_key)
        if cached_result is not None:
            return cached_result
            
        if not self.market_data_fetcher:
            result = hv.Text(0, 0, "No data fetcher available")
            self._set_cached_data(cache_key, result)
            return result
            
        # Get active pairs
        symbols = self.market_data_fetcher.get_active_pairs()
        if limit:
            symbols = symbols[:limit]
        
        if not symbols:
            result = hv.Text(0, 0, "No symbols available")
            self._set_cached_data(cache_key, result)
            return result
            
        # Prepare data for visualization
        data = []
        for symbol in symbols:
            meta = self.market_data_fetcher.get_pair_metadata(symbol)
            if not meta:
                continue
            
            # Get base and quote from symbol
            base = meta.base_currency
            quote = meta.quote_currency
            
            # Only include if base and quote are valid
            if not base or not quote:
                continue
            
            # Check for whale detector signals
            whale_signals = meta.detector_signals.get("whale", [])
            
            if whale_signals:
                # Calculate activity score based on recency and magnitude
                current_time = time.time()
                activity_score = 0.0
                most_recent_days = float('inf')
                
                for signal in whale_signals:
                    # Age in days
                    age_days = (current_time - signal.get("timestamp", 0)) / 86400
                    most_recent_days = min(most_recent_days, age_days)
                    
                    # Decay factor based on age
                    decay = np.exp(-0.5 * age_days)  # Exponential decay
                    
                    # Signal strength
                    strength = signal.get("strength", 0.5)
                    
                    # Contribute to overall score
                    activity_score += strength * decay
                
                # Normalize to 0-1 range
                activity_score = min(1.0, activity_score)
                
                data.append({
                    'base': base,
                    'quote': quote,
                    'symbol': symbol,
                    'activity_score': activity_score,
                    'signal_count': len(whale_signals),
                    'recent_signal': "Yes" if most_recent_days < 1 else "No",
                    'days_since': f"{most_recent_days:.1f}" if most_recent_days < float('inf') else "N/A"
                })
        
        if not data:
            result = hv.Text(0, 0, "No whale activity data available")
            self._set_cached_data(cache_key, result)
            return result
            
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(data)
        
        # Aggregate by base/quote if there are duplicates
        if df.duplicated(['base', 'quote']).any():
            df = df.groupby(['base', 'quote']).agg({
                'activity_score': 'max',  # Take maximum activity score
                'signal_count': 'sum',
                'recent_signal': lambda x: 'Yes' if 'Yes' in x.values else 'No',
                'days_since': lambda x: min(float(y) if y != 'N/A' else float('inf') for y in x),
                'symbol': lambda x: ', '.join(x[:3]) + ('...' if len(x) > 3 else '')
            }).reset_index()
            
            # Convert days_since back to string
            df['days_since'] = df['days_since'].apply(lambda x: f"{x:.1f}" if x < float('inf') else "N/A")
        
        # Create heatmap
        heatmap = hv.HeatMap(
            df, 
            kdims=['base', 'quote'], 
            vdims=['activity_score', 'signal_count', 'recent_signal', 'days_since', 'symbol']
        ).opts(
            width=width,
            height=height,
            colorbar=True,
            cmap=cmap,
            tools=self.config["default_tools"],
            xrotation=45,
            toolbar='above' if self.config["show_toolbar"] else None,
            title="Whale Activity Heatmap",
            ylabel="Quote Currency",
            xlabel="Base Currency"
        )
        
        # Add hover tooltip if enabled
        if self.config["tooltips_enabled"]:
            hover = HoverTool(tooltips=[
                ('Symbol', '@symbol'),
                ('Activity Score', '@activity_score{0.00}'),
                ('Signal Count', '@signal_count'),
                ('Recent Signal', '@recent_signal'),
                ('Days Since Latest', '@days_since')
            ])
            
            heatmap = heatmap.opts(tools=[hover])
        
        # Store in cache
        self._set_cached_data(cache_key, heatmap)
        
        return heatmap
    
    def create_custom_heatmap(self, data, x_dim, y_dim, value_dim, 
                           title="Custom Heatmap", width=None, height=None, cmap=None,
                           tooltips=None):
        """
        Create custom heatmap from provided data.
        
        Args:
            data: Data for heatmap
            x_dim: X dimension name
            y_dim: Y dimension name
            value_dim: Value dimension name
            title: Plot title
            width: Plot width
            height: Plot height
            cmap: Colormap
            tooltips: Custom tooltips
            
        Returns:
            Holoviews object
        """
        # Use default parameters if not provided
        width = width or self.width
        height = height or self.height
        cmap = cmap or self.cmap
        
        # Convert to DataFrame if not already
        if not isinstance(data, pd.DataFrame):
            df = pd.DataFrame(data)
        else:
            df = data
            
        # Ensure required dimensions exist
        if x_dim not in df.columns or y_dim not in df.columns or value_dim not in df.columns:
            return hv.Text(0, 0, f"Required dimensions not found in data: {x_dim}, {y_dim}, {value_dim}")
        
        # Create list of all value dimensions
        vdims = [col for col in df.columns if col not in [x_dim, y_dim]]
        
        # Create heatmap
        heatmap = hv.HeatMap(
            df, 
            kdims=[x_dim, y_dim], 
            vdims=vdims
        ).opts(
            width=width,
            height=height,
            colorbar=True,
            cmap=cmap,
            tools=self.config["default_tools"],
            xrotation=45,
            toolbar='above' if self.config["show_toolbar"] else None,
            title=title,
            ylabel=y_dim,
            xlabel=x_dim
        )
        
        # Add hover tooltip if enabled and provided
        if self.config["tooltips_enabled"] and tooltips:
            hover = HoverTool(tooltips=tooltips)
            heatmap = heatmap.opts(tools=[hover])
        
        return heatmap
    
    def create_tradingview_chart(self, symbol: str, timeframe: str = "1d", 
                              lookback: str = "90d", width: int = 1000, 
                              height: int = 600, indicators: List[str] = None):
        """
        Create a TradingView-style interactive chart for a trading pair.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe for data
            lookback: Lookback period
            width: Chart width
            height: Chart height
            indicators: List of indicators to add (e.g., ["ma", "bollinger", "volume"])
            
        Returns:
            Holoviews Layout with interactive chart
        """
        if not self.market_data_fetcher:
            return hv.Text(0, 0, "No data fetcher available")
        
        # Default indicators if not provided
        if indicators is None:
            indicators = ["volume", "ma", "bollinger"]
        
        # Fetch data
        data_dict = self.market_data_fetcher.fetch_data(
            symbols=[symbol],
            timeframe=timeframe,
            lookback=lookback
        )
        
        if not data_dict or symbol not in data_dict:
            return hv.Text(0, 0, f"No data available for {symbol}")
        
        # Get data frame
        df = data_dict[symbol]
        
        # Create hover tool for candlesticks
        hover = HoverTool(tooltips=[
            ('Date', '@date{%F}'),
            ('Open', '@open{0.00}'),
            ('High', '@high{0.00}'),
            ('Low', '@low{0.00}'),
            ('Close', '@close{0.00}'),
            ('Volume', '@volume{0.00 a}')
        ], formatters={'@date': 'datetime'})
        
        # Convert DataFrame for Bokeh/Holoviews
        source_data = {
            'date': df.index,
            'open': df['open'],
            'high': df['high'],
            'low': df['low'],
            'close': df['close'],
            'volume': df['volume'] if 'volume' in df.columns else np.zeros(len(df)),
            'color': np.where(df['close'] > df['open'], 'green', 'red')
        }
        
        # Create source
        source = ColumnDataSource(source_data)
        
        # Create candlestick chart
        candle_width = 0.8  # Width of candles (0-1)
        if timeframe in ['1d', '1h', '4h']:
            # Calculate candle width based on timeframe
            if timeframe == '1d':
                # 12 hours in milliseconds (half a day)
                candle_width = 12 * 60 * 60 * 1000
            elif timeframe == '4h':
                # 2 hours in milliseconds (half of 4h)
                candle_width = 2 * 60 * 60 * 1000
            elif timeframe == '1h':
                # 30 minutes in milliseconds (half an hour)
                candle_width = 30 * 60 * 1000
        
        # Create figure for main chart
        p = figure(
            x_axis_type='datetime',
            width=width,
            height=int(height * 0.7),
            title=f"{symbol} - {timeframe}",
            tools=[hover, 'pan', 'wheel_zoom', 'box_zoom', 'reset', 'save'],
            toolbar_location='above',
            active_drag='pan',
            active_scroll='wheel_zoom'
        )
        
        # Add candlesticks
        # Stems
        p.segment('date', 'high', 'date', 'low', color='color', source=source)
        
        # Bodies
        p.vbar(
            'date', candle_width, 'open', 'close',
            source=source,
            fill_color='color',
            line_color='color'
        )
        
        # Customize appearance
        p.grid.grid_line_alpha = 0.3
        p.xaxis.major_label_orientation = 1.0
        p.xaxis.axis_label = 'Date'
        p.yaxis.axis_label = 'Price'
        p.background_fill_color = "#f5f5f5"
        p.border_fill_color = "#ffffff"
        
        # Create plots dictionary for layout
        plots = {'main': p}
        
        # Add indicators as requested
        if 'ma' in indicators:
            # Add moving averages
            df['ma20'] = df['close'].rolling(window=20).mean()
            df['ma50'] = df['close'].rolling(window=50).mean()
            df['ma200'] = df['close'].rolling(window=200).mean()
            
            p.line(df.index, df['ma20'], color='blue', line_width=1, legend_label='MA(20)')
            p.line(df.index, df['ma50'], color='orange', line_width=1, legend_label='MA(50)')
            p.line(df.index, df['ma200'], color='purple', line_width=1.5, legend_label='MA(200)')
            
            # Configure legend
            p.legend.location = "top_left"
            p.legend.click_policy = "hide"
        
        if 'bollinger' in indicators:
            # Add Bollinger Bands
            window = 20
            df['ma'] = df['close'].rolling(window=window).mean()
            df['std'] = df['close'].rolling(window=window).std()
            df['upper_band'] = df['ma'] + 2 * df['std']
            df['lower_band'] = df['ma'] - 2 * df['std']
            
            p.line(df.index, df['upper_band'], color='rgba(0,0,255,0.7)', line_width=1, legend_label='BB Upper')
            p.line(df.index, df['lower_band'], color='rgba(0,0,255,0.7)', line_width=1, legend_label='BB Lower')
        
        if 'volume' in indicators and 'volume' in df.columns:
            # Create volume chart
            v = figure(
                x_axis_type='datetime',
                width=width,
                height=int(height * 0.2),
                tools=['pan', 'wheel_zoom', 'box_zoom', 'reset', 'save'],
                toolbar_location=None,
                x_range=p.x_range,  # Link x range with main chart
            )
            
            # Add volume bars
            v.vbar(
                'date', candle_width, 0, 'volume',
                source=source,
                fill_color='color',
                line_color='color',
                alpha=0.5
            )
            
            # Customize appearance
            v.grid.grid_line_alpha = 0.3
            v.xaxis.major_label_orientation = 1.0
            v.xaxis.axis_label = 'Date'
            v.yaxis.axis_label = 'Volume'
            v.background_fill_color = "#f5f5f5"
            v.border_fill_color = "#ffffff"
            
            plots['volume'] = v
        
        if 'rsi' in indicators:
            # Calculate RSI
            delta = df['close'].diff()
            up = delta.clip(lower=0)
            down = -1 * delta.clip(upper=0)
            ma_up = up.rolling(window=14).mean()
            ma_down = down.rolling(window=14).mean()
            rsi = 100 - (100 / (1 + ma_up / ma_down))
            df['rsi'] = rsi
            
            # Create RSI chart
            r = figure(
                x_axis_type='datetime',
                width=width,
                height=int(height * 0.15),
                tools=['pan', 'wheel_zoom', 'box_zoom', 'reset', 'save'],
                toolbar_location=None,
                x_range=p.x_range,  # Link x range with main chart
                y_range=(0, 100)  # RSI range is 0-100
            )
            
            # Add RSI line
            r.line(df.index, df['rsi'], color='blue', line_width=1)
            
            # Add overbought/oversold lines
            r.line(df.index, [70] * len(df), color='red', line_width=1, line_dash='dashed')
            r.line(df.index, [30] * len(df), color='green', line_width=1, line_dash='dashed')
            
            # Customize appearance
            r.grid.grid_line_alpha = 0.3
            r.xaxis.major_label_orientation = 1.0
            r.xaxis.axis_label = 'Date'
            r.yaxis.axis_label = 'RSI'
            r.background_fill_color = "#f5f5f5"
            r.border_fill_color = "#ffffff"
            
            plots['rsi'] = r
        
        if 'macd' in indicators:
            # Calculate MACD
            df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
            df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = df['ema12'] - df['ema26']
            df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['histogram'] = df['macd'] - df['signal']
            
            # Create MACD chart
            m = figure(
                x_axis_type='datetime',
                width=width,
                height=int(height * 0.15),
                tools=['pan', 'wheel_zoom', 'box_zoom', 'reset', 'save'],
                toolbar_location=None,
                x_range=p.x_range  # Link x range with main chart
            )
            
            # Add MACD and signal lines
            m.line(df.index, df['macd'], color='blue', line_width=1, legend_label='MACD')
            m.line(df.index, df['signal'], color='orange', line_width=1, legend_label='Signal')
            
            # Add histogram
            m.vbar(
                x=df.index,
                top=df['histogram'],
                width=candle_width,
                color=['green' if v >= 0 else 'red' for v in df['histogram']],
                alpha=0.5
            )
            
            # Customize appearance
            m.grid.grid_line_alpha = 0.3
            m.xaxis.major_label_orientation = 1.0
            m.xaxis.axis_label = 'Date'
            m.yaxis.axis_label = 'MACD'
            m.background_fill_color = "#f5f5f5"
            m.border_fill_color = "#ffffff"
            m.legend.location = "top_left"
            
            plots['macd'] = m
        
        # Create layout
        if len(plots) == 1:
            # Only main chart
            layout = plots['main']
        else:
            # Stack charts vertically
            layout = pn.Column(
                plots['main'], 
                *[plots[k] for k in plots if k != 'main'],
                sizing_mode='stretch_width'
            )
        
        return layout
    
    def create_tradingview_dashboard(self, symbols: List[str] = None, timeframe: str = "1d"):
        """
        Create a TradingView-style dashboard with multiple charts.
        
        Args:
            symbols: List of symbols to display (None for top pairs)
            timeframe: Timeframe for charts
            
        Returns:
            Panel dashboard
        """
        if symbols is None and self.market_data_fetcher:
            # Get top pairs
            pairs = self.market_data_fetcher.get_pair_rankings(limit=6)
            symbols = [symbol for symbol, _ in pairs]
        
        if not symbols:
            return pn.Column(pn.pane.Markdown("# No symbols available"))
        
        # Limit to 6 symbols to avoid overloading
        symbols = symbols[:6]
        
        # Create charts for each symbol
        charts = []
        for symbol in symbols:
            charts.append(
                self.create_tradingview_chart(
                    symbol=symbol,
                    timeframe=timeframe,
                    width=900,
                    height=500
                )
            )
        
        # Create tabs for symbols
        tabs = pn.Tabs(
            *[(symbol, chart) for symbol, chart in zip(symbols, charts)]
        )
        
        # Add title
        dashboard = pn.Column(
            pn.pane.Markdown("# TradingView-Style Market Dashboard"),
            tabs
        )
        
        return dashboard
    
    def create_market_dashboard(self):
        """
        Create a comprehensive market dashboard with both heatmaps and TradingView charts.
        
        Returns:
            Panel dashboard
        """
        # Create opportunity heatmap
        opportunity_heatmap = self.create_opportunity_heatmap()
        
        # Create regime heatmap
        regime_heatmap = self.create_regime_heatmap()
        
        # Get top pairs
        if self.market_data_fetcher:
            pairs = self.market_data_fetcher.get_pair_rankings(limit=3)
            symbols = [symbol for symbol, _ in pairs]
        else:
            symbols = ["BTC-USD", "ETH-USD", "SOL-USD"]
        
        # Create TradingView charts for top pairs
        charts = []
        for symbol in symbols:
            charts.append(
                self.create_tradingview_chart(
                    symbol=symbol,
                    timeframe="1d",
                    width=900,
                    height=500
                )
            )
        
        # Create dashboard
        dashboard = pn.Column(
            pn.pane.Markdown("# CDFA Market Intelligence Dashboard"),
            pn.Tabs(
                ("Market Opportunities", opportunity_heatmap),
                ("Market Regimes", regime_heatmap),
                *([(symbol, chart) for symbol, chart in zip(symbols, charts)])
            )
        )
        
        return dashboard

    
    def create_dashboard(self, width=None, height=None):
        """
        Create a complete dashboard with multiple visualizations.
        
        Args:
            width: Width of each component
            height: Height of each component
            
        Returns:
            Holoviews layout or Panel dashboard
        """
        # Use default parameters if not provided
        width = width or self.width
        height = height or self.height
        
        # Check if we have a Panel dashboard already
        if hasattr(self, 'dashboard') and self.dashboard:
            return self.dashboard
        
        # Create individual visualizations
        opportunity_heatmap = self.create_opportunity_heatmap(width=width, height=height)
        regime_heatmap = self.create_regime_heatmap(width=width, height=height)
        correlation_heatmap = self.create_correlation_heatmap(width=width, height=height)
        analyzer_heatmap = self.create_analyzer_heatmap(width=width, height=height)
        whale_heatmap = self.create_whale_activity_heatmap(width=width, height=height)
        
        # Create tabs for different visualizations
        tabs = hv.Tabs(
            ('Opportunity', opportunity_heatmap),
            ('Market Regime', regime_heatmap),
            ('Correlation', correlation_heatmap),
            ('Analyzer Scores', analyzer_heatmap),
            ('Whale Activity', whale_heatmap)
        )
        
        return tabs
    
    def create_specialized_mra_visualization(self, symbol, timeframe="1d", lookback="90d",
                                          width=None, height=None):
        """
        Create specialized MRA visualization for a specific symbol.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe for data
            lookback: Lookback period
            width: Plot width
            height: Plot height
            
        Returns:
            Holoviews layout
        """
        # Use default parameters if not provided
        width = width or self.width
        height = height or (self.height // 2)
        
        # Check cache first
        cache_key = f"mra_viz_{symbol}_{timeframe}_{lookback}_{width}_{height}"
        cached_result = self._get_cached_data(cache_key)
        if cached_result is not None:
            return cached_result
            
        if not self.market_data_fetcher:
            result = hv.Text(0, 0, "No data fetcher available")
            self._set_cached_data(cache_key, result)
            return result
            
        # Fetch data
        data_dict = self.market_data_fetcher.fetch_data(
            symbols=[symbol],
            timeframe=timeframe,
            lookback=lookback
        )
        
        if not data_dict or symbol not in data_dict:
            result = hv.Text(0, 0, "No data available")
            self._set_cached_data(cache_key, result)
            return result
            
        df = data_dict[symbol]
        
        # Get MRA instance
        mra = getattr(self.market_data_fetcher, 'mra', None)
        if not mra:
            result = hv.Text(0, 0, "MultiResolutionAnalyzer not available")
            self._set_cached_data(cache_key, result)
            return result
        
        try:
            # Perform decomposition
            decomp = mra.decompose(df['close'])
            
            # Perform regime analysis
            regime_results = mra.analyze_regimes(df)
            
            # Get scales for plotting
            scales = decomp.get('scales', [])
            if not scales:
                scales = list(range(1, len(decomp.get('wavelet_coeffs', [])) + 1))
                
            # Create curve for price
            price_curve = hv.Curve(
                (df.index, df['close']), 
                kdims=['Date'], 
                vdims=['Price']
            ).opts(
                width=width,
                height=height,
                line_width=2,
                tools=self.config["default_tools"],
                toolbar='above' if self.config["show_toolbar"] else None,
                title=f"{symbol} Price"
            )
            
            # Create heatmap for wavelet coefficients
            coefs = decomp.get('wavelet_coeffs', [])
            if coefs:
                # Convert to 2D array suitable for heatmap
                coef_data = []
                for i, scale in enumerate(scales):
                    for j, t in enumerate(range(len(coefs[i]))):
                        # Use absolute value of coefficients for power
                        coef_data.append({
                            'Time': j,
                            'Scale': scale,
                            'Power': abs(coefs[i][j])
                        })
                        
                # Convert to DataFrame
                coef_df = pd.DataFrame(coef_data)
                
                # Create heatmap
                wavelet_heatmap = hv.HeatMap(
                    coef_df, 
                    kdims=['Time', 'Scale'], 
                    vdims=['Power']
                ).opts(
                    width=width,
                    height=height,
                    colorbar=True,
                    cmap='viridis',
                    tools=self.config["default_tools"],
                    toolbar='above' if self.config["show_toolbar"] else None,
                    title=f"{symbol} Wavelet Transform"
                )
            else:
                wavelet_heatmap = hv.Text(0, 0, "No wavelet data available")
                
            # Create regime plot
            if 'regimes' in regime_results:
                regimes = regime_results['regimes']
                
                # Create data for regime plot
                if isinstance(df.index, pd.DatetimeIndex) and len(df.index) == len(regimes):
                    # Use actual dates
                    regime_data = [(df.index[i], regimes[i]) for i in range(len(regimes))]
                    x_dim = 'Date'
                else:
                    # Use indices
                    regime_data = [(i, regimes[i]) for i in range(len(regimes))]
                    x_dim = 'Time'
                
                # Convert to DataFrame for better handling
                regime_df = pd.DataFrame(regime_data, columns=[x_dim, 'Regime'])
                
                # Create curve
                regime_curve = hv.Curve(
                    regime_df, 
                    kdims=[x_dim], 
                    vdims=['Regime']
                ).opts(
                    width=width,
                    height=height // 2,
                    line_width=2,
                    tools=self.config["default_tools"],
                    toolbar='above' if self.config["show_toolbar"] else None,
                    title=f"{symbol} Regime"
                )
            else:
                regime_curve = hv.Text(0, 0, "No regime data available")
                
            # Combine visualizations
            layout = (price_curve + wavelet_heatmap + regime_curve).cols(1)
            
            # Store in cache
            self._set_cached_data(cache_key, layout)
            
            return layout
            
        except Exception as e:
            self.logger.error(f"Error creating MRA visualization: {e}")
            return hv.Text(0, 0, f"Error creating visualization: {str(e)}")
    
    def serve_dashboard(self, port=5006):
        """
        Serve the dashboard using Panel server.
        
        Args:
            port: Server port
            
        Returns:
            Server instance
        """
        try:
            import panel as pn
            
            # Check if we have a panel dashboard
            if hasattr(self, 'dashboard') and self.dashboard:
                dashboard = self.dashboard
            else:
                # Create dashboard using tabs
                dashboard = pn.panel(self.create_dashboard())
            
            # Serve the dashboard
            server = pn.serve(dashboard, port=port, show=False, title="CDFA Market Intelligence Dashboard")
            
            self.logger.info(f"Dashboard served at http://localhost:{port}")
            
            return server
            
        except ImportError:
            self.logger.error("Panel is required for serving dashboard")
            raise ImportError("Panel is required for serving dashboard")
        except Exception as e:
            self.logger.error(f"Error serving dashboard: {e}")
            raise
    
    def save_dashboard(self, filename="cdfa_dashboard.html", title="CDFA Market Intelligence Dashboard"):
        """
        Save dashboard to HTML file.
        
        Args:
            filename: Output filename
            title: Dashboard title
            
        Returns:
            Path to saved file
        """
        try:
            import panel as pn
            
            # Create dashboard if needed
            if hasattr(self, 'dashboard') and self.dashboard:
                dashboard = self.dashboard
            else:
                # Create dashboard using tabs
                dashboard = pn.panel(self.create_dashboard())
            
            # Add title
            full_dashboard = pn.Column(
                pn.pane.Markdown(f"# {title}"),
                dashboard
            )
            
            # Save to file
            path = os.path.expanduser(filename)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            full_dashboard.save(path, title=title, embed=True)
            
            self.logger.info(f"Dashboard saved to {path}")
            
            return path
            
        except ImportError:
            self.logger.error("Panel is required for saving dashboard")
            raise ImportError("Panel is required for saving dashboard")
        except Exception as e:
            self.logger.error(f"Error saving dashboard: {e}")
            raise
    
    def export_visualization(self, visualization, filename, format="html", 
                           width=None, height=None, dpi=None):
        """
        Export visualization to file.
        
        Args:
            visualization: Holoviews object
            filename: Output filename
            format: Output format ('html', 'png', 'svg')
            width: Image width (for png/svg)
            height: Image height (for png/svg)
            dpi: Image DPI (for png)
            
        Returns:
            Path to saved file
        """
        try:
            # Set width and height if provided
            if width is not None or height is not None:
                w = width or self.width
                h = height or self.height
                visualization = visualization.opts(width=w, height=h)
            
            # Create directory if needed
            path = os.path.expanduser(filename)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Get renderer
            renderer = hv.renderer('bokeh')
            
            if format == "html":
                # Save to HTML
                renderer.save(visualization, path)
            elif format == "png":
                # Save to PNG
                kwargs = {}
                if dpi:
                    kwargs['dpi'] = dpi
                renderer.save(visualization, path, fmt='png', **kwargs)
            elif format == "svg":
                # Save to SVG
                renderer.save(visualization, path, fmt='svg')
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            self.logger.info(f"Visualization exported to {path}")
            
            return path
            
        except Exception as e:
            self.logger.error(f"Error exporting visualization: {e}")
            raise
    
    def get_config_parameters(self) -> Dict[str, Any]:
        """
        Get configuration parameters for frontend integration.
        
        Returns:
            Dictionary of configuration parameters with metadata
        """
        return {
            "default_width": {
                "type": "int",
                "min": 200,
                "max": 2000,
                "default": self.DEFAULT_CONFIG["default_width"],
                "description": "Default plot width"
            },
            "default_height": {
                "type": "int",
                "min": 200,
                "max": 2000,
                "default": self.DEFAULT_CONFIG["default_height"],
                "description": "Default plot height"
            },
            "default_cmap": {
                "type": "str",
                "options": list(self.AVAILABLE_CMAPS.keys()),
                "default": self.DEFAULT_CONFIG["default_cmap"],
                "description": "Default colormap"
            },
            "show_toolbar": {
                "type": "bool",
                "default": self.DEFAULT_CONFIG["show_toolbar"],
                "description": "Show Bokeh toolbar"
            },
            "tooltips_enabled": {
                "type": "bool",
                "default": self.DEFAULT_CONFIG["tooltips_enabled"],
                "description": "Enable hover tooltips"
            },
            "max_items_per_plot": {
                "type": "int",
                "min": 5,
                "max": 200,
                "default": self.DEFAULT_CONFIG["max_items_per_plot"],
                "description": "Maximum items to include in plots"
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
                
                # Update instance variables if needed
                if parameter == "default_width":
                    self.width = value
                elif parameter == "default_height":
                    self.height = value
                elif parameter == "default_cmap":
                    self.cmap = self._get_colormap(value)
                
                self.logger.info(f"Updated parameter {parameter}: {old_value} -> {value}")
                
                # Clear cache since parameters changed
                with self._data_cache_lock:
                    self._data_cache.clear()
                    self._data_cache_timestamps.clear()
                
                return True
            else:
                self.logger.warning(f"Unknown parameter: {parameter}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error updating parameter {parameter}: {e}")
            return False
    
    def stop(self):
        """Stop background threads and clean up resources."""
        self.logger.info("Stopping HoloviewsVisualizer...")
        self.running = False
        
        # Wait for threads to terminate
        if self._auto_update_thread and self._auto_update_thread.is_alive():
            self._auto_update_thread.join(timeout=5.0)
                
        self.logger.info("HoloviewsVisualizer stopped")
    
    def __del__(self):
        """Destructor to ensure clean shutdown."""
        if hasattr(self, 'running') and self.running:
            self.stop()
