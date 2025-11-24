#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Visualization Framework for CDFA Extensions

Provides a comprehensive visualization framework for the CDFA system with:
- Interactive signal exploration
- Real-time regime visualization
- Pattern recognition visualization
- Fusion confidence heat maps
- Cross-asset relationship graphs
- Market microstructure visualizations

Author: Created on May 6, 2025
"""

import logging
import time
import numpy as np
import pandas as pd
import threading
from typing import Dict, List, Tuple, Optional, Union, Any, Callable, Set
from enum import Enum, auto
from dataclasses import dataclass, field
import warnings
import os
import uuid
import io
import base64
from datetime import datetime, timedelta
import json
import tempfile

# Import from cdfa_extensions
from .hw_acceleration import HardwareAccelerator

# ---- Optional dependencies with graceful fallbacks ----

# Matplotlib for basic visualization
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.figure import Figure
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    warnings.warn("Matplotlib not available. Basic visualization will be limited.")
    
    # Dummy Figure class for typing
    class Figure:
        pass

# Seaborn for enhanced visualization
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    warnings.warn("Seaborn not available. Enhanced visualization will be limited.")

# Add missing AdvancedVisualization class required by core.py
class AdvancedVisualization:
    """Advanced visualization framework for CDFA."""
    
    def __init__(self, **kwargs):
        """Initialize the visualization framework."""
        self.logger = logging.getLogger("advanced_visualization")
        self.config = kwargs.get('config', {})
        self.hardware_accelerator = None
        
        # Initialize hardware acceleration if available
        if 'enable_hw_acceleration' in self.config and self.config['enable_hw_acceleration']:
            try:
                self.hardware_accelerator = HardwareAccelerator()
                self.logger.info("Hardware acceleration enabled for visualization")
            except Exception as e:
                self.logger.warning(f"Failed to initialize hardware acceleration: {e}")
                
        self.logger.info("AdvancedVisualization initialized successfully")

# NetworkX for graph visualization
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    warnings.warn("NetworkX not available. Graph visualization will be limited.")

# Plotly for interactive visualization
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    warnings.warn("Plotly not available. Interactive visualization will be limited.")

# Bokeh for interactive visualization
try:
    from bokeh.plotting import figure as bokeh_figure
    from bokeh.models import ColumnDataSource, HoverTool, ColorBar, LinearColorMapper
    from bokeh.palettes import viridis, Spectral, magma, Spectral10
    from bokeh.layouts import gridplot
    from bokeh.io import output_notebook, show, output_file, save
    BOKEH_AVAILABLE = True
except ImportError:
    BOKEH_AVAILABLE = False
    warnings.warn("Bokeh not available. Interactive visualization will be limited.")

# Holoviews for declarative visualization
try:
    import holoviews as hv
    from holoviews import opts, dim
    HOLOVIEWS_AVAILABLE = True
    hv.extension('bokeh', 'matplotlib')
except ImportError:
    HOLOVIEWS_AVAILABLE = False
    warnings.warn("Holoviews not available. Advanced visualization will be limited.")

class VisualizationType(Enum):
    """Types of visualizations provided by the framework."""
    SIGNAL_EXPLORATION = auto()
    REGIME_VISUALIZATION = auto()
    PATTERN_RECOGNITION = auto()
    FUSION_HEATMAP = auto()
    CORRELATION_NETWORK = auto()
    MARKET_STRUCTURE = auto()
    FLOW_ANALYSIS = auto()
    ASSET_RELATIONSHIP = auto()
    CONTAGION_RISK = auto()
    PERFORMANCE_TRACKING = auto()
    
    def __str__(self) -> str:
        return self.name.lower()
    
    @classmethod
    def from_string(cls, s: str) -> 'VisualizationType':
        """Create enum from string representation."""
        s_upper = s.upper()
        for item in cls:
            if item.name == s_upper:
                return item
        for item in cls:
            if item.name.startswith(s_upper):
                return item
        raise ValueError(f"Unknown VisualizationType: {s}")

class OutputFormat(Enum):
    """Output formats for visualizations."""
    PNG = auto()
    SVG = auto()
    HTML = auto()
    JSON = auto()
    INTERACTIVE = auto()
    EMBED = auto()
    
    def __str__(self) -> str:
        return self.name.lower()
    
    def get_extension(self) -> str:
        """Get file extension for this format."""
        if self == OutputFormat.PNG:
            return "png"
        elif self == OutputFormat.SVG:
            return "svg"
        elif self == OutputFormat.HTML:
            return "html"
        elif self == OutputFormat.JSON:
            return "json"
        else:
            return "html"

class VisualizationEngine:
    """
    Advanced visualization engine for the CDFA system.
    
    Provides comprehensive visualization capabilities for analyzing signals,
    patterns, correlations, and market structure. Supports both static and
    interactive visualizations with multiple backend options.
    """
    
    def __init__(self, hw_accelerator: Optional[HardwareAccelerator] = None,
                config: Optional[Dict[str, Any]] = None,
                log_level: int = logging.INFO):
        """
        Initialize the visualization engine.
        
        Args:
            hw_accelerator: Optional hardware accelerator
            config: Configuration parameters
            log_level: Logging level
        """
        # Setup logging
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.setLevel(log_level)
        
        # Initialize hardware accelerator
        self.hw_accelerator = hw_accelerator if hw_accelerator is not None else HardwareAccelerator()
        
        # Default configuration
        self.default_config = {
            # General visualization parameters
            "default_width": 1000,
            "default_height": 600,
            "default_dpi": 100,
            "default_format": "png",
            "default_interactive": True,
            "default_theme": "dark",
            "default_color_palette": "viridis",
            "title_font_size": 14,
            "axis_font_size": 12,
            "tick_font_size": 10,
            "legend_font_size": 10,
            
            # Plotting preferences
            "use_gridlines": True,
            "use_antialiasing": True,
            "use_tight_layout": True,
            "use_log_scale": False,
            "show_annotations": True,
            "plot_background": "#2E3440",  # Nord color scheme
            "text_color": "#ECEFF4",
            "grid_color": "#4C566A",
            "line_color": "#88C0D0",
            "highlight_color": "#EBCB8B",
            "negative_color": "#BF616A",
            "positive_color": "#A3BE8C",
            "neutral_color": "#B48EAD",
            
            # Interactive visualization
            "enable_hover": True,
            "enable_zoom": True,
            "enable_pan": True,
            "enable_save": True,
            "enable_tools": True,
            "hover_mode": "closest",
            "hover_info": ["x", "y", "text"],
            
            # Backend preferences
            "preferred_backend": "auto",  # auto, matplotlib, plotly, bokeh, holoviews
            "backend_priority": ["plotly", "bokeh", "matplotlib"],
            "fallback_to_raster": True,
            
            # Output settings
            "output_dir": "./visualizations",
            "inline_images": True,
            "image_quality": 90,
            "embed_data": True,
            "image_format": "svg",
            
            # Advanced settings
            "cache_plots": True,
            "cache_ttl": 300,  # 5 minutes
            "max_items_per_plot": 500,
            "downsampling": True,
            "use_webgl": True,
            "use_threading": True
        }
        
        # Merge with provided config
        self.config = {**self.default_config, **(config or {})}
        
        # Create output directory if needed
        os.makedirs(self.config["output_dir"], exist_ok=True)
        
        # Initialize state
        self._lock = threading.RLock()
        self._plot_cache = {}  # key -> (plot, timestamp)
        self._data_cache = {}  # key -> (data, timestamp)
        
        # Configure default plot style
        self._configure_plot_style()
        
        # Determine available backends
        self.available_backends = self._detect_available_backends()
        self.logger.info(f"Available visualization backends: {', '.join(self.available_backends)}")
        
        # Set the active backend
        self.active_backend = self._select_backend()
        self.logger.info(f"Using visualization backend: {self.active_backend}")
    
    def _detect_available_backends(self) -> List[str]:
        """Detect available visualization backends."""
        backends = []
        
        if MATPLOTLIB_AVAILABLE:
            backends.append("matplotlib")
            
        if PLOTLY_AVAILABLE:
            backends.append("plotly")
            
        if BOKEH_AVAILABLE:
            backends.append("bokeh")
            
        if HOLOVIEWS_AVAILABLE:
            backends.append("holoviews")
            
        return backends
    
    def _select_backend(self) -> str:
        """Select the best available backend based on config."""
        preferred = self.config["preferred_backend"].lower()
        
        # If auto, use priority list
        if preferred == "auto":
            for backend in self.config["backend_priority"]:
                if backend in self.available_backends:
                    return backend
            # Fallback to matplotlib if available
            if "matplotlib" in self.available_backends:
                return "matplotlib"
            # Use whatever is available
            if self.available_backends:
                return self.available_backends[0]
            return "none"
        else:
            # Use preferred if available
            if preferred in self.available_backends:
                return preferred
            # Otherwise use first available
            if self.available_backends:
                self.logger.warning(f"Preferred backend '{preferred}' not available, using '{self.available_backends[0]}'")
                return self.available_backends[0]
            return "none"
    
    def _configure_plot_style(self):
        """Configure default plot style based on config."""
        if MATPLOTLIB_AVAILABLE:
            # Set global Matplotlib style
            if SEABORN_AVAILABLE:
                theme = self.config["default_theme"]
                if theme == "dark":
                    sns.set_theme(style="darkgrid")
                else:
                    sns.set_theme(style="whitegrid")
                    
            # Set global preferences
            plt.rcParams['font.size'] = self.config["axis_font_size"]
            plt.rcParams['axes.titlesize'] = self.config["title_font_size"]
            plt.rcParams['axes.labelsize'] = self.config["axis_font_size"]
            plt.rcParams['xtick.labelsize'] = self.config["tick_font_size"]
            plt.rcParams['ytick.labelsize'] = self.config["tick_font_size"]
            plt.rcParams['legend.fontsize'] = self.config["legend_font_size"]
            plt.rcParams['figure.figsize'] = (
                self.config["default_width"] / 100,
                self.config["default_height"] / 100
            )
            plt.rcParams['figure.dpi'] = self.config["default_dpi"]
            plt.rcParams['figure.autolayout'] = self.config["use_tight_layout"]
            plt.rcParams['savefig.dpi'] = self.config["default_dpi"]
            plt.rcParams['savefig.format'] = self.config["image_format"]
            plt.rcParams['savefig.bbox'] = 'tight' if self.config["use_tight_layout"] else 'standard'
            
            # Set theme colors for dark theme
            if self.config["default_theme"] == "dark":
                plt.rcParams['axes.facecolor'] = self.config["plot_background"]
                plt.rcParams['figure.facecolor'] = self.config["plot_background"]
                plt.rcParams['text.color'] = self.config["text_color"]
                plt.rcParams['axes.labelcolor'] = self.config["text_color"]
                plt.rcParams['xtick.color'] = self.config["text_color"]
                plt.rcParams['ytick.color'] = self.config["text_color"]
                plt.rcParams['grid.color'] = self.config["grid_color"]
    
    def _get_cached_plot(self, key: Any) -> Optional[Any]:
        """
        Get cached plot if valid.
        
        Args:
            key: Cache key
            
        Returns:
            Cached plot or None if not found or expired
        """
        if not self.config["cache_plots"]:
            return None
            
        with self._lock:
            # Check if plot is in cache
            cache_entry = self._plot_cache.get(key)
            
            if cache_entry is None:
                return None
                
            plot, timestamp = cache_entry
            
            # Check if expired
            current_time = time.time()
            if current_time - timestamp > self.config["cache_ttl"]:
                # Remove from cache
                self._plot_cache.pop(key, None)
                return None
                
            return plot
            
    def _cache_plot(self, key: Any, plot: Any):
        """
        Cache plot for future use.
        
        Args:
            key: Cache key
            plot: Plot object
        """
        if not self.config["cache_plots"]:
            return
            
        with self._lock:
            self._plot_cache[key] = (plot, time.time())
            
    def _get_cached_data(self, key: Any) -> Optional[Any]:
        """
        Get cached data if valid.
        
        Args:
            key: Cache key
            
        Returns:
            Cached data or None if not found or expired
        """
        if not self.config["cache_plots"]:
            return None
            
        with self._lock:
            # Check if data is in cache
            cache_entry = self._data_cache.get(key)
            
            if cache_entry is None:
                return None
                
            data, timestamp = cache_entry
            
            # Check if expired
            current_time = time.time()
            if current_time - timestamp > self.config["cache_ttl"]:
                # Remove from cache
                self._data_cache.pop(key, None)
                return None
                
            return data
            
    def _cache_data(self, key: Any, data: Any):
        """
        Cache data for future use.
        
        Args:
            key: Cache key
            data: Data object
        """
        if not self.config["cache_plots"]:
            return
            
        with self._lock:
            self._data_cache[key] = (data, time.time())
    
    def _fig_to_base64(self, fig: Figure, format: str = 'png', dpi: int = 100) -> str:
        """
        Convert matplotlib figure to base64 string.
        
        Args:
            fig: Matplotlib figure
            format: Image format (png, svg, pdf)
            dpi: Resolution
            
        Returns:
            Base64 encoded image
        """
        if not MATPLOTLIB_AVAILABLE:
            return ""
            
        # Create a bytes buffer for the image
        buf = io.BytesIO()
        
        # Save the figure to the buffer
        fig.savefig(buf, format=format, dpi=dpi, bbox_inches='tight')
        
        # Get the buffer content and encode it to base64
        buf.seek(0)
        img_data = base64.b64encode(buf.read()).decode('utf-8')
        
        # Close the buffer and figure
        buf.close()
        plt.close(fig)
        
        return img_data
    
    def _plotly_to_html(self, fig: Any, include_plotlyjs: str = 'cdn', full_html: bool = False) -> str:
        """
        Convert plotly figure to HTML.
        
        Args:
            fig: Plotly figure
            include_plotlyjs: How to include plotly.js
            full_html: Whether to include full HTML document
            
        Returns:
            HTML string
        """
        if not PLOTLY_AVAILABLE:
            return ""
            
        return fig.to_html(include_plotlyjs=include_plotlyjs, full_html=full_html)
    
    def _bokeh_to_html(self, plot: Any) -> str:
        """
        Convert bokeh plot to HTML.
        
        Args:
            plot: Bokeh plot
            
        Returns:
            HTML string
        """
        if not BOKEH_AVAILABLE:
            return ""
            
        from bokeh.embed import file_html
        from bokeh.resources import CDN
        
        return file_html(plot, CDN)
    
    def save_visualization(self, viz_obj: Any, filename: str, format: Optional[str] = None) -> str:
        """
        Save visualization to file.
        
        Args:
            viz_obj: Visualization object
            filename: Output filename (without extension)
            format: Output format (default from config)
            
        Returns:
            Path to saved file
        """
        if format is None:
            format = self.config["default_format"]
            
        # Create full path
        if os.path.dirname(filename):
            # Already has path
            output_path = filename
        else:
            # Add output directory
            output_path = os.path.join(self.config["output_dir"], filename)
            
        # Add extension if missing
        if not output_path.endswith(f".{format}"):
            output_path = f"{output_path}.{format}"
            
        # Create directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save based on object type
        if MATPLOTLIB_AVAILABLE and isinstance(viz_obj, plt.Figure):
            # Matplotlib figure
            viz_obj.savefig(output_path, format=format, dpi=self.config["default_dpi"], 
                         bbox_inches='tight')
            plt.close(viz_obj)
            
        elif PLOTLY_AVAILABLE and 'plotly.graph_objs' in str(type(viz_obj)):
            # Plotly figure
            if format.lower() == 'html':
                viz_obj.write_html(output_path, include_plotlyjs='cdn')
            else:
                viz_obj.write_image(output_path, format=format, scale=2)
                
        elif BOKEH_AVAILABLE and 'bokeh' in str(type(viz_obj)):
            # Bokeh figure
            from bokeh.io import save as bokeh_save
            bokeh_save(viz_obj, filename=output_path, title="CDFA Visualization")
            
        elif HOLOVIEWS_AVAILABLE and 'holoviews' in str(type(viz_obj)):
            # Holoviews figure
            if format.lower() == 'html':
                hv.save(viz_obj, output_path)
            else:
                hv.save(viz_obj, output_path, fmt=format)
                
        else:
            self.logger.error(f"Unsupported visualization object type: {type(viz_obj)}")
            return ""
            
        self.logger.info(f"Saved visualization to {output_path}")
        return output_path
    
    def get_visualization_data(self, viz_obj: Any, format: str = 'html') -> str:
        """
        Get visualization data as string.
        
        Args:
            viz_obj: Visualization object
            format: Output format ('html', 'base64', 'json')
            
        Returns:
            Visualization data
        """
        # Get data based on object type and format
        if format.lower() == 'html':
            # Return HTML representation
            if MATPLOTLIB_AVAILABLE and isinstance(viz_obj, plt.Figure):
                # Save to temporary file and read back
                with tempfile.NamedTemporaryFile(suffix='.html') as tmp:
                    viz_obj.savefig(tmp.name, format='png', dpi=self.config["default_dpi"])
                    plt.close(viz_obj)
                    
                    # Create HTML with image
                    img_data = self._fig_to_base64(viz_obj, format='png', dpi=self.config["default_dpi"])
                    html = f'<img src="data:image/png;base64,{img_data}" />'
                    return html
                    
            elif PLOTLY_AVAILABLE and 'plotly.graph_objs' in str(type(viz_obj)):
                # Plotly to HTML
                return self._plotly_to_html(viz_obj, include_plotlyjs='cdn')
                
            elif BOKEH_AVAILABLE and 'bokeh' in str(type(viz_obj)):
                # Bokeh to HTML
                return self._bokeh_to_html(viz_obj)
                
            elif HOLOVIEWS_AVAILABLE and 'holoviews' in str(type(viz_obj)):
                # Holoviews to HTML
                return hv.render(viz_obj, backend='bokeh').to_html()
                
            else:
                self.logger.error(f"Unsupported visualization object type: {type(viz_obj)}")
                return ""
                
        elif format.lower() == 'base64':
            # Return base64 encoded image
            if MATPLOTLIB_AVAILABLE and isinstance(viz_obj, plt.Figure):
                return self._fig_to_base64(viz_obj, format='png', dpi=self.config["default_dpi"])
                
            elif PLOTLY_AVAILABLE and 'plotly.graph_objs' in str(type(viz_obj)):
                # Save to temporary file and convert to base64
                with tempfile.NamedTemporaryFile(suffix='.png') as tmp:
                    viz_obj.write_image(tmp.name, format='png', scale=2)
                    
                    # Read and encode
                    with open(tmp.name, 'rb') as f:
                        img_data = base64.b64encode(f.read()).decode('utf-8')
                        
                    return img_data
                    
            elif BOKEH_AVAILABLE and 'bokeh' in str(type(viz_obj)):
                # Save to temporary file and convert to base64
                from bokeh.io import export_png
                
                with tempfile.NamedTemporaryFile(suffix='.png') as tmp:
                    export_png(viz_obj, filename=tmp.name)
                    
                    # Read and encode
                    with open(tmp.name, 'rb') as f:
                        img_data = base64.b64encode(f.read()).decode('utf-8')
                        
                    return img_data
                    
            else:
                self.logger.error(f"Unsupported visualization object type for base64: {type(viz_obj)}")
                return ""
                
        elif format.lower() == 'json':
            # Return JSON representation
            if PLOTLY_AVAILABLE and 'plotly.graph_objs' in str(type(viz_obj)):
                return json.dumps(viz_obj.to_dict())
                
            elif BOKEH_AVAILABLE and 'bokeh' in str(type(viz_obj)):
                from bokeh.model import json_item
                return json.dumps(json_item(viz_obj))
                
            else:
                self.logger.error(f"Unsupported visualization object type for JSON: {type(viz_obj)}")
                return ""
                
        else:
            self.logger.error(f"Unsupported output format: {format}")
            return ""
    
    # ----- Signal Visualization Methods -----
    
    def create_signal_visualization(self, symbols: Union[str, List[str]], signals: Dict[str, List[float]], 
                                 timestamps: Optional[List[Union[str, datetime, float]]] = None,
                                 title: Optional[str] = None, 
                                 output_format: Optional[Union[str, OutputFormat]] = None) -> Any:
        """
        Create signal visualization for one or more signals.
        
        Args:
            symbols: Symbol or list of symbols
            signals: Dictionary of signal name to values
            timestamps: Optional list of timestamps
            title: Optional plot title
            output_format: Output format
            
        Returns:
            Visualization object
        """
        # Standardize inputs
        if isinstance(symbols, str):
            symbols = [symbols]
            
        # Determine output format
        if output_format is None:
            output_format = self.config["default_format"]
        elif isinstance(output_format, str):
            output_format = output_format.lower()
            
        # Get interactive preference
        interactive = self.config["default_interactive"]
        
        # Set default title if not provided
        if title is None:
            if len(symbols) == 1:
                title = f"Signal Analysis for {symbols[0]}"
            else:
                title = f"Signal Analysis for Multiple Symbols"
                
        # Create cache key
        cache_key = ("signal", tuple(symbols), tuple(signals.keys()), output_format, title)
        cached_viz = self._get_cached_plot(cache_key)
        if cached_viz is not None:
            return cached_viz
            
        # Prepare data
        if timestamps is None:
            # Create default timestamps (ascending integers)
            max_len = max(len(values) for values in signals.values()) if signals else 0
            timestamps = list(range(max_len))
            
        # Convert timestamps to datetime if they are strings or floats
        if timestamps and not isinstance(timestamps[0], datetime):
            try:
                if isinstance(timestamps[0], str):
                    timestamps = [pd.to_datetime(ts) for ts in timestamps]
                elif isinstance(timestamps[0], (int, float)):
                    timestamps = [pd.to_datetime(ts, unit='s') for ts in timestamps]
            except Exception as e:
                self.logger.warning(f"Failed to convert timestamps to datetime: {e}")
                # Use integers as fallback
                timestamps = list(range(len(timestamps)))
                
        # Create visualization based on backend and interactivity
        viz_obj = None
        
        if self.active_backend == "plotly" and PLOTLY_AVAILABLE and interactive:
            # Create Plotly figure
            fig = go.Figure()
            
            # Add signals
            for name, values in signals.items():
                if len(values) > len(timestamps):
                    # Trim values to match timestamps
                    values = values[:len(timestamps)]
                elif len(values) < len(timestamps):
                    # Pad values with NaN
                    values = list(values) + [float('nan')] * (len(timestamps) - len(values))
                    
                # Add signal line
                fig.add_trace(go.Scatter(
                    x=timestamps,
                    y=values,
                    mode='lines',
                    name=name,
                    hovertemplate="%{x}<br>%{y:.4f}"
                ))
                
            # Customize layout
            fig.update_layout(
                title=title,
                xaxis_title="Time",
                yaxis_title="Signal Value",
                height=self.config["default_height"],
                width=self.config["default_width"],
                template="plotly_dark" if self.config["default_theme"] == "dark" else "plotly_white",
                hovermode="closest",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            viz_obj = fig
            
        elif self.active_backend == "bokeh" and BOKEH_AVAILABLE and interactive:
            # Create Bokeh figure
            p = bokeh_figure(
                width=self.config["default_width"],
                height=self.config["default_height"],
                title=title,
                x_axis_type="datetime" if isinstance(timestamps[0], datetime) else "auto",
                tools="pan,wheel_zoom,box_zoom,reset,save",
                toolbar_location="above"
            )
            
            # Add signals
            for i, (name, values) in enumerate(signals.items()):
                if len(values) > len(timestamps):
                    # Trim values to match timestamps
                    values = values[:len(timestamps)]
                elif len(values) < len(timestamps):
                    # Pad values with NaN
                    values = list(values) + [float('nan')] * (len(timestamps) - len(values))
                    
                # Get color from palette
                color = Spectral10[i % len(Spectral10)]
                
                # Add signal line
                p.line(
                    x=timestamps,
                    y=values,
                    legend_label=name,
                    line_width=2,
                    line_color=color
                )
                
            # Customize
            p.legend.location = "top_right"
            p.legend.click_policy = "hide"
            p.grid.grid_line_alpha = 0.3
            
            # Add hover tool
            hover = HoverTool(
                tooltips=[
                    ("Signal", "$name"),
                    ("Time", "@x{%F %T}"),
                    ("Value", "@y{0.0000}")
                ],
                formatters={"@x": "datetime"} if isinstance(timestamps[0], datetime) else {}
            )
            p.add_tools(hover)
            
            viz_obj = p
            
        elif self.active_backend == "holoviews" and HOLOVIEWS_AVAILABLE and interactive:
            # Create HoloViews visualization
            curves = []
            
            for name, values in signals.items():
                if len(values) > len(timestamps):
                    # Trim values to match timestamps
                    values = values[:len(timestamps)]
                elif len(values) < len(timestamps):
                    # Pad values with NaN
                    values = list(values) + [float('nan')] * (len(timestamps) - len(values))
                    
                # Create curve
                curve = hv.Curve((timestamps, values), kdims=['Time'], vdims=['Value']).relabel(name)
                curves.append(curve)
                
            # Overlay all curves
            overlay = hv.Overlay(curves)
            
            # Apply options
            overlay = overlay.opts(
                opts.Curve(
                    width=self.config["default_width"],
                    height=self.config["default_height"],
                    title=title,
                    tools=['hover'],
                    legend_position='top_right',
                    xrotation=45,
                    show_grid=True
                )
            )
            
            viz_obj = overlay
            
        else:
            # Fallback to Matplotlib (static)
            if not MATPLOTLIB_AVAILABLE:
                self.logger.error("No visualization backends available")
                return None
                
            # Create figure
            fig, ax = plt.subplots(figsize=(
                self.config["default_width"] / 100,
                self.config["default_height"] / 100
            ))
            
            # Plot signals
            for name, values in signals.items():
                if len(values) > len(timestamps):
                    # Trim values to match timestamps
                    values = values[:len(timestamps)]
                elif len(values) < len(timestamps):
                    # Pad values with NaN
                    values = list(values) + [float('nan')] * (len(timestamps) - len(values))
                    
                # Plot line
                ax.plot(timestamps, values, label=name)
                
            # Customize plot
            ax.set_title(title)
            ax.set_xlabel("Time")
            ax.set_ylabel("Signal Value")
            ax.legend()
            
            # Format x-axis for datetime
            if isinstance(timestamps[0], datetime):
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
                fig.autofmt_xdate()
                
            # Set grid
            ax.grid(self.config["use_gridlines"], alpha=0.3)
            
            # Apply tight layout
            if self.config["use_tight_layout"]:
                fig.tight_layout()
                
            viz_obj = fig
            
        # Cache visualization
        self._cache_plot(cache_key, viz_obj)
        
        return viz_obj
    
    def create_fusion_heatmap(self, correlation_matrix: pd.DataFrame, title: Optional[str] = None,
                           output_format: Optional[Union[str, OutputFormat]] = None) -> Any:
        """
        Create heatmap visualization for correlation or fusion matrix.
        
        Args:
            correlation_matrix: Correlation or fusion matrix dataframe
            title: Optional plot title
            output_format: Output format
            
        Returns:
            Visualization object
        """
        # Determine output format
        if output_format is None:
            output_format = self.config["default_format"]
        elif isinstance(output_format, str):
            output_format = output_format.lower()
            
        # Get interactive preference
        interactive = self.config["default_interactive"]
        
        # Set default title if not provided
        if title is None:
            title = "Correlation Heatmap"
            
        # Create cache key
        cache_key = ("heatmap", hash(str(correlation_matrix)), output_format, title)
        cached_viz = self._get_cached_plot(cache_key)
        if cached_viz is not None:
            return cached_viz
            
        # Create visualization based on backend and interactivity
        viz_obj = None
        
        if self.active_backend == "plotly" and PLOTLY_AVAILABLE and interactive:
            # Create Plotly heatmap
            fig = go.Figure(data=go.Heatmap(
                z=correlation_matrix.values,
                x=correlation_matrix.columns,
                y=correlation_matrix.index,
                colorscale=self.config["default_color_palette"],
                showscale=True,
                hoverongaps=False,
                zmin=-1, zmax=1
            ))
            
            # Customize layout
            fig.update_layout(
                title=title,
                height=self.config["default_height"],
                width=self.config["default_width"],
                template="plotly_dark" if self.config["default_theme"] == "dark" else "plotly_white",
                xaxis=dict(tickangle=-45)
            )
            
            viz_obj = fig
            
        elif self.active_backend == "bokeh" and BOKEH_AVAILABLE and interactive:
            # Create Bokeh heatmap
            # Convert matrix to long format for bokeh
            symbols = correlation_matrix.index.tolist()
            n = len(symbols)
            
            source_data = {
                'x': [],
                'y': [],
                'value': []
            }
            
            for i in range(n):
                for j in range(n):
                    source_data['x'].append(symbols[i])
                    source_data['y'].append(symbols[j])
                    source_data['value'].append(correlation_matrix.iloc[i, j])
                    
            source = ColumnDataSource(source_data)
            
            # Create color mapper
            mapper = LinearColorMapper(palette=viridis(256), low=-1, high=1)
            
            # Create figure
            p = bokeh_figure(
                width=self.config["default_width"],
                height=self.config["default_height"],
                title=title,
                tools="pan,wheel_zoom,box_zoom,reset,save",
                toolbar_location="above",
                x_range=symbols,
                y_range=list(reversed(symbols))
            )
            
            # Create heatmap
            p.rect(
                x='x',
                y='y',
                width=1,
                height=1,
                source=source,
                fill_color={'field': 'value', 'transform': mapper},
                line_color=None
            )
            
            # Add color bar
            color_bar = ColorBar(
                color_mapper=mapper,
                major_label_text_font_size="8pt",
                ticker=BasicTicker(desired_num_ticks=10),
                formatter=PrintfTickFormatter(format="%.2f"),
                label_standoff=10,
                border_line_color=None,
                location=(0, 0)
            )
            
            p.add_layout(color_bar, 'right')
            
            # Customize
            p.axis.axis_line_color = None
            p.axis.major_tick_line_color = None
            p.axis.major_label_text_font_size = "8pt"
            p.axis.major_label_standoff = 0
            p.xaxis.major_label_orientation = np.pi/3
            
            # Add hover tool
            hover = HoverTool(
                tooltips=[
                    ("Symbol 1", "@x"),
                    ("Symbol 2", "@y"),
                    ("Correlation", "@value{0.00}")
                ]
            )
            p.add_tools(hover)
            
            viz_obj = p
            
        elif self.active_backend == "holoviews" and HOLOVIEWS_AVAILABLE and interactive:
            # Create HoloViews heatmap
            heatmap = hv.HeatMap((
                correlation_matrix.columns,
                correlation_matrix.index,
                correlation_matrix.values
            ))
            
            # Apply options
            heatmap = heatmap.opts(
                opts.HeatMap(
                    width=self.config["default_width"],
                    height=self.config["default_height"],
                    title=title,
                    tools=['hover'],
                    colorbar=True,
                    cmap=self.config["default_color_palette"],
                    xrotation=45,
                    fontsize={'title': self.config["title_font_size"],
                           'labels': self.config["axis_font_size"],
                           'xticks': self.config["tick_font_size"],
                           'yticks': self.config["tick_font_size"]},
                    clim=(-1, 1)
                )
            )
            
            viz_obj = heatmap
            
        else:
            # Fallback to Matplotlib (static)
            if not MATPLOTLIB_AVAILABLE:
                self.logger.error("No visualization backends available")
                return None
                
            # Create figure and axes
            fig, ax = plt.subplots(figsize=(
                self.config["default_width"] / 100,
                self.config["default_height"] / 100
            ))
            
            # Create heatmap
            if SEABORN_AVAILABLE:
                # Use Seaborn for better heatmap
                heatmap = sns.heatmap(
                    correlation_matrix,
                    annot=True,
                    fmt=".2f",
                    cmap=self.config["default_color_palette"],
                    vmin=-1, vmax=1,
                    ax=ax
                )
                
                # Rotate x-axis labels
                plt.xticks(rotation=45, ha='right')
                
            else:
                # Use Matplotlib
                heatmap = ax.imshow(
                    correlation_matrix.values,
                    cmap=self.config["default_color_palette"],
                    vmin=-1, vmax=1
                )
                
                # Add colorbar
                plt.colorbar(heatmap, ax=ax)
                
                # Set ticks
                ax.set_xticks(np.arange(len(correlation_matrix.columns)))
                ax.set_yticks(np.arange(len(correlation_matrix.index)))
                
                # Set tick labels
                ax.set_xticklabels(correlation_matrix.columns, rotation=45, ha='right')
                ax.set_yticklabels(correlation_matrix.index)
                
                # Add values to cells
                for i in range(len(correlation_matrix.index)):
                    for j in range(len(correlation_matrix.columns)):
                        text = ax.text(
                            j, i, f"{correlation_matrix.iloc[i, j]:.2f}",
                            ha="center", va="center",
                            color="white" if abs(correlation_matrix.iloc[i, j]) > 0.5 else "black"
                        )
                        
            # Set title
            ax.set_title(title)
            
            # Apply tight layout
            if self.config["use_tight_layout"]:
                fig.tight_layout()
                
            viz_obj = fig
            
        # Cache visualization
        self._cache_plot(cache_key, viz_obj)
        
        return viz_obj
    
    def create_network_visualization(self, nodes: List[str], edges: List[Tuple[str, str, float]],
                                  node_properties: Optional[Dict[str, Dict[str, Any]]] = None,
                                  title: Optional[str] = None,
                                  output_format: Optional[Union[str, OutputFormat]] = None) -> Any:
        """
        Create network visualization for correlation or MST graph.
        
        Args:
            nodes: List of node names
            edges: List of (source, target, weight) tuples
            node_properties: Optional dictionary of node properties
            title: Optional plot title
            output_format: Output format
            
        Returns:
            Visualization object
        """
        # Determine output format
        if output_format is None:
            output_format = self.config["default_format"]
        elif isinstance(output_format, str):
            output_format = output_format.lower()
            
        # Get interactive preference
        interactive = self.config["default_interactive"]
        
        # Set default title if not provided
        if title is None:
            title = "Network Visualization"
            
        # Create cache key
        cache_key = ("network", tuple(nodes), tuple(edges), output_format, title)
        cached_viz = self._get_cached_plot(cache_key)
        if cached_viz is not None:
            return cached_viz
            
        # Check for NetworkX
        if not NETWORKX_AVAILABLE:
            self.logger.error("NetworkX not available, cannot create network visualization")
            return None
            
        # Create graph
        G = nx.Graph()
        
        # Add nodes with properties
        for node in nodes:
            if node_properties and node in node_properties:
                G.add_node(node, **node_properties[node])
            else:
                G.add_node(node)
                
        # Add edges with weights
        edge_weights = {}
        for source, target, weight in edges:
            G.add_edge(source, target, weight=weight)
            edge_weights[(source, target)] = weight
            edge_weights[(target, source)] = weight
            
        # Calculate node sizes based on centrality if not provided
        if not node_properties or not all('size' in props for node, props in node_properties.items()):
            # Use degree centrality
            centrality = nx.degree_centrality(G)
            
            # Normalize to reasonable size range (5-30)
            min_cent = min(centrality.values()) if centrality else 0
            max_cent = max(centrality.values()) if centrality else 1
            
            range_cent = max_cent - min_cent
            if range_cent < 1e-6:
                range_cent = 1.0
                
            node_sizes = {
                node: 5 + 25 * (centrality[node] - min_cent) / range_cent
                for node in G.nodes()
            }
        else:
            # Use provided sizes
            node_sizes = {
                node: props.get('size', 10)
                for node, props in node_properties.items()
            }
            
        # Get node colors if provided
        if node_properties and all('color' in props for node, props in node_properties.items()):
            node_colors = {
                node: props.get('color', '#1f77b4')
                for node, props in node_properties.items()
            }
        else:
            # Use default color
            node_colors = {node: '#1f77b4' for node in G.nodes()}
            
        # Calculate layout
        layout_algo = self.config.get("graph_layout", "spring")
        
        if layout_algo == "spring":
            pos = nx.spring_layout(G, weight='weight', seed=42)
        elif layout_algo == "kamada_kawai":
            pos = nx.kamada_kawai_layout(G, weight='weight')
        elif layout_algo == "spectral":
            pos = nx.spectral_layout(G)
        elif layout_algo == "circular":
            pos = nx.circular_layout(G)
        else:
            # Default to spring
            pos = nx.spring_layout(G, weight='weight', seed=42)
            
        # Create visualization based on backend and interactivity
        viz_obj = None
        
        if self.active_backend == "plotly" and PLOTLY_AVAILABLE and interactive:
            # Create plotly network visualization
            edge_trace = []
            
            # Create edges
            for source, target in G.edges():
                x0, y0 = pos[source]
                x1, y1 = pos[target]
                weight = edge_weights.get((source, target), 1.0)
                
                # Scale width by weight
                width = max(1, min(8, weight * 5))
                
                # Add edge
                edge_trace.append(go.Scatter(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    mode='lines',
                    line=dict(width=width, color='rgba(150,150,150,0.7)'),
                    hoverinfo='none'
                ))
                
            # Create nodes
            node_x = []
            node_y = []
            node_text = []
            node_size = []
            node_color = []
            
            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                node_text.append(node)
                node_size.append(node_sizes.get(node, 10))
                node_color.append(node_colors.get(node, '#1f77b4'))
                
            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers',
                hoverinfo='text',
                text=node_text,
                marker=dict(
                    showscale=False,
                    color=node_color,
                    size=node_size,
                    line=dict(width=1, color='rgb(50,50,50)')
                )
            )
            
            # Create figure
            fig = go.Figure(data=edge_trace + [node_trace])
            
            # Customize layout
            fig.update_layout(
                title=title,
                titlefont_size=self.config["title_font_size"],
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=self.config["default_height"],
                width=self.config["default_width"],
                template="plotly_dark" if self.config["default_theme"] == "dark" else "plotly_white"
            )
            
            viz_obj = fig
            
        elif self.active_backend == "bokeh" and BOKEH_AVAILABLE and interactive:
            # Create bokeh network visualization
            # Prepare data
            node_data = {
                'x': [],
                'y': [],
                'name': [],
                'size': [],
                'color': []
            }
            
            edge_data = {
                'x0': [],
                'y0': [],
                'x1': [],
                'y1': [],
                'weight': []
            }
            
            # Add node data
            for node in G.nodes():
                x, y = pos[node]
                node_data['x'].append(x)
                node_data['y'].append(y)
                node_data['name'].append(node)
                node_data['size'].append(node_sizes.get(node, 10))
                node_data['color'].append(node_colors.get(node, '#1f77b4'))
                
            # Add edge data
            for source, target in G.edges():
                x0, y0 = pos[source]
                x1, y1 = pos[target]
                weight = edge_weights.get((source, target), 1.0)
                
                edge_data['x0'].append(x0)
                edge_data['y0'].append(y0)
                edge_data['x1'].append(x1)
                edge_data['y1'].append(y1)
                edge_data['weight'].append(weight)
                
            # Create data sources
            node_source = ColumnDataSource(node_data)
            edge_source = ColumnDataSource(edge_data)
            
            # Create figure
            p = bokeh_figure(
                width=self.config["default_width"],
                height=self.config["default_height"],
                title=title,
                tools="pan,wheel_zoom,box_zoom,reset,save",
                toolbar_location="above",
                x_range=(-1.1, 1.1),
                y_range=(-1.1, 1.1)
            )
            
            # Add edges
            for i in range(len(edge_data['x0'])):
                p.line(
                    x=[edge_data['x0'][i], edge_data['x1'][i]],
                    y=[edge_data['y0'][i], edge_data['y1'][i]],
                    line_width=max(1, min(8, edge_data['weight'][i] * 5)),
                    line_alpha=0.7,
                    line_color='gray'
                )
                
            # Add nodes
            p.circle(
                x='x',
                y='y',
                size='size',
                fill_color='color',
                line_color='black',
                line_width=1,
                source=node_source
            )
            
            # Add hover tool
            hover = HoverTool(
                tooltips=[
                    ("Node", "@name")
                ]
            )
            p.add_tools(hover)
            
            # Customize
            p.axis.visible = False
            p.grid.visible = False
            
            viz_obj = p
            
        elif self.active_backend == "holoviews" and HOLOVIEWS_AVAILABLE and interactive:
            # Create HoloViews network visualization
            # Prepare node data
            node_data = []
            for node in G.nodes():
                x, y = pos[node]
                node_data.append((x, y, node, node_sizes.get(node, 10), node_colors.get(node, '#1f77b4')))
                
            # Prepare edge data
            edge_data = []
            for source, target in G.edges():
                x0, y0 = pos[source]
                x1, y1 = pos[target]
                weight = edge_weights.get((source, target), 1.0)
                edge_data.append((x0, y0, x1, y1, weight))
                
            # Create nodes and edges
            nodes = hv.Points(
                node_data, 
                kdims=['x', 'y'], 
                vdims=['name', 'size', 'color']
            )
            
            edges = hv.Segments(
                edge_data,
                kdims=['x0', 'y0', 'x1', 'y1'],
                vdims=['weight']
            )
            
            # Style nodes
            nodes = nodes.opts(
                opts.Points(
                    color='color',
                    size='size',
                    tools=['hover'],
                    nonselection_alpha=0.2
                )
            )
            
            # Style edges
            edges = edges.opts(
                opts.Segments(
                    line_width='weight',
                    line_alpha=0.7,
                    line_color='gray',
                    nonselection_alpha=0.1
                )
            )
            
            # Combine into overlay
            network = edges * nodes
            
            # Apply options
            network = network.opts(
                opts.NdOverlay(
                    width=self.config["default_width"],
                    height=self.config["default_height"],
                    title=title
                )
            )
            
            viz_obj = network
            
        else:
            # Fallback to Matplotlib (static)
            if not MATPLOTLIB_AVAILABLE:
                self.logger.error("No visualization backends available")
                return None
                
            # Create figure
            fig, ax = plt.subplots(figsize=(
                self.config["default_width"] / 100,
                self.config["default_height"] / 100
            ))
            
            # Create a colormap for edge colors based on weight
            edge_colors = []
            edge_widths = []
            
            min_weight = min([w for _, _, w in edges])
            max_weight = max([w for _, _, w in edges])
            weight_range = max_weight - min_weight
            
            if weight_range < 1e-6:
                weight_range = 1.0
            
            for _, _, weight in edges:
                # Normalize weight to [0, 1]
                norm_weight = (weight - min_weight) / weight_range
                
                # Set edge width based on weight
                edge_widths.append(1 + 3 * norm_weight)
                
                # Set edge color based on weight
                edge_colors.append((0.5, 0.5, 0.5, 0.5 + 0.5 * norm_weight))
                
            # Draw network
            nx.draw_networkx(
                G,
                pos=pos,
                ax=ax,
                with_labels=True,
                node_color=[node_colors.get(node, '#1f77b4') for node in G.nodes()],
                node_size=[10 * node_sizes.get(node, 10) for node in G.nodes()],
                edge_color=edge_colors,
                width=edge_widths,
                font_size=self.config["tick_font_size"],
                alpha=0.8
            )
            
            # Remove axis
            ax.axis('off')
            
            # Set title
            ax.set_title(title)
            
            # Apply tight layout
            if self.config["use_tight_layout"]:
                fig.tight_layout()
                
            viz_obj = fig
            
        # Cache visualization
        self._cache_plot(cache_key, viz_obj)
        
        return viz_obj
    
    def create_regime_visualization(self, timestamps: List[Union[str, datetime, float]], 
                                 regimes: List[str], signals: Optional[Dict[str, List[float]]] = None,
                                 title: Optional[str] = None,
                                 regime_colors: Optional[Dict[str, str]] = None,
                                 output_format: Optional[Union[str, OutputFormat]] = None) -> Any:
        """
        Create visualization of market regimes over time.
        
        Args:
            timestamps: List of timestamps
            regimes: List of regime labels
            signals: Optional dictionary of signal name to values
            title: Optional plot title
            regime_colors: Optional dictionary of regime label to color
            output_format: Output format
            
        Returns:
            Visualization object
        """
        # Determine output format
        if output_format is None:
            output_format = self.config["default_format"]
        elif isinstance(output_format, str):
            output_format = output_format.lower()
            
        # Get interactive preference
        interactive = self.config["default_interactive"]
        
        # Set default title if not provided
        if title is None:
            title = "Market Regime Visualization"
            
        # Set default colors if not provided
        if regime_colors is None:
            regime_colors = {
                'growth': self.config["positive_color"],
                'conservation': self.config["neutral_color"],
                'release': self.config["negative_color"],
                'reorganization': self.config["highlight_color"],
                # Add more defaults
                'high_vol_uptrend': '#ff7f0e',
                'high_vol_downtrend': '#d62728',
                'low_vol_uptrend': '#2ca02c',
                'low_vol_downtrend': '#9467bd',
                'medium_vol_uptrend': '#8c564b',
                'medium_vol_downtrend': '#e377c2',
                'unknown': '#7f7f7f'
            }
            # Add fallback color for unknown regimes
            regime_colors['default'] = '#7f7f7f'
            
        # Create cache key
        cache_key = ("regime", tuple(timestamps), tuple(regimes), 
                   tuple(signals.keys()) if signals else None, 
                   output_format, title)
        cached_viz = self._get_cached_plot(cache_key)
        if cached_viz is not None:
            return cached_viz
            
        # Convert timestamps to datetime if necessary
        if timestamps and not isinstance(timestamps[0], datetime):
            try:
                if isinstance(timestamps[0], str):
                    timestamps = [pd.to_datetime(ts) for ts in timestamps]
                elif isinstance(timestamps[0], (int, float)):
                    timestamps = [pd.to_datetime(ts, unit='s') for ts in timestamps]
            except Exception as e:
                self.logger.warning(f"Failed to convert timestamps to datetime: {e}")
                # Use integers as fallback
                timestamps = list(range(len(timestamps)))
                
        # Create a DataFrame for easier manipulation
        regime_df = pd.DataFrame({
            'timestamp': timestamps,
            'regime': regimes
        })
        
        # Check if there's a signal dataframe to overlay
        signal_df = None
        if signals and len(signals) > 0:
            # Create signal dataframe
            signal_data = {}
            
            for name, values in signals.items():
                if len(values) > len(timestamps):
                    # Trim values
                    signal_data[name] = values[:len(timestamps)]
                elif len(values) < len(timestamps):
                    # Pad with NaN
                    signal_data[name] = list(values) + [float('nan')] * (len(timestamps) - len(values))
                else:
                    signal_data[name] = values
                    
            signal_df = pd.DataFrame(signal_data)
            signal_df['timestamp'] = timestamps
            
        # Create visualization based on backend and interactivity
        viz_obj = None
        
        if self.active_backend == "plotly" and PLOTLY_AVAILABLE and interactive:
            # Create plotly figure
            if signal_df is not None:
                # Create figure with two y-axes
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                
                # Add regime color bands
                unique_regimes = regime_df['regime'].unique()
                
                for regime in unique_regimes:
                    # Get segments of this regime
                    regime_segments = regime_df[regime_df['regime'] == regime]
                    
                    for i in range(len(regime_segments)):
                        start_idx = regime_segments.index[i]
                        
                        # Find end of segment
                        if i < len(regime_segments) - 1:
                            if regime_segments.index[i+1] == start_idx + 1:
                                continue
                                
                        end_idx = start_idx
                        while end_idx + 1 < len(regime_df) and regime_df['regime'].iloc[end_idx+1] == regime:
                            end_idx += 1
                            
                        # Get start and end times
                        start_time = regime_df['timestamp'].iloc[start_idx]
                        end_time = regime_df['timestamp'].iloc[end_idx]
                        
                        # Add colored rectangle
                        fig.add_shape(
                            type="rect",
                            x0=start_time,
                            x1=end_time,
                            y0=0,
                            y1=1,
                            yref="paper",
                            fillcolor=regime_colors.get(regime, regime_colors['default']),
                            opacity=0.3,
                            layer="below",
                            line_width=0
                        )
                        
                        # Add regime label
                        fig.add_annotation(
                            x=(start_time + end_time) / 2,
                            y=0.95,
                            yref="paper",
                            text=regime,
                            showarrow=False,
                            bgcolor="rgba(255,255,255,0.7)",
                            bordercolor="black",
                            borderwidth=1,
                            font=dict(size=10)
                        )
                        
                # Add signal lines
                for name in signal_df.columns:
                    if name != 'timestamp':
                        fig.add_trace(
                            go.Scatter(
                                x=signal_df['timestamp'],
                                y=signal_df[name],
                                mode='lines',
                                name=name,
                                line=dict(width=2),
                                hovertemplate="%{x}<br>%{y:.4f}"
                            ),
                            secondary_y=False
                        )
                        
                # Customize layout
                fig.update_layout(
                    title=title,
                    xaxis_title="Time",
                    yaxis_title="Signal Value",
                    height=self.config["default_height"],
                    width=self.config["default_width"],
                    template="plotly_dark" if self.config["default_theme"] == "dark" else "plotly_white",
                    hovermode="closest",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
            else:
                # Create figure with just regimes
                fig = go.Figure()
                
                # Add regime color bands
                unique_regimes = regime_df['regime'].unique()
                
                for regime in unique_regimes:
                    # Get segments of this regime
                    regime_segments = regime_df[regime_df['regime'] == regime]
                    
                    for i in range(len(regime_segments)):
                        start_idx = regime_segments.index[i]
                        
                        # Find end of segment
                        if i < len(regime_segments) - 1:
                            if regime_segments.index[i+1] == start_idx + 1:
                                continue
                                
                        end_idx = start_idx
                        while end_idx + 1 < len(regime_df) and regime_df['regime'].iloc[end_idx+1] == regime:
                            end_idx += 1
                            
                        # Get start and end times
                        start_time = regime_df['timestamp'].iloc[start_idx]
                        end_time = regime_df['timestamp'].iloc[end_idx]
                        
                        # Add colored rectangle
                        fig.add_shape(
                            type="rect",
                            x0=start_time,
                            x1=end_time,
                            y0=0,
                            y1=1,
                            yref="paper",
                            fillcolor=regime_colors.get(regime, regime_colors['default']),
                            opacity=0.6,
                            layer="below",
                            line_width=0
                        )
                        
                        # Add regime label
                        fig.add_annotation(
                            x=(start_time + end_time) / 2,
                            y=0.5,
                            yref="paper",
                            text=regime,
                            showarrow=False,
                            font=dict(size=12, color="black"),
                            bgcolor="rgba(255,255,255,0.7)"
                        )
                        
                # Customize layout
                fig.update_layout(
                    title=title,
                    xaxis_title="Time",
                    yaxis_title="",
                    height=self.config["default_height"],
                    width=self.config["default_width"],
                    template="plotly_dark" if self.config["default_theme"] == "dark" else "plotly_white",
                    showlegend=False
                )
                
            viz_obj = fig
            
        elif self.active_backend == "bokeh" and BOKEH_AVAILABLE and interactive:
            # Create bokeh figure
            p = bokeh_figure(
                width=self.config["default_width"],
                height=self.config["default_height"],
                title=title,
                x_axis_type="datetime" if isinstance(timestamps[0], datetime) else "auto",
                tools="pan,wheel_zoom,box_zoom,reset,save",
                toolbar_location="above"
            )
            
            # Get unique regimes
            unique_regimes = set(regimes)
            
            # Create color map
            colors = {}
            for regime in unique_regimes:
                colors[regime] = regime_colors.get(regime, regime_colors.get('default', '#7f7f7f'))
                
            # Add colored background for regimes
            prev_regime = None
            start_idx = 0
            
            for i, regime in enumerate(regimes):
                if regime != prev_regime:
                    if prev_regime is not None:
                        # Add background for previous regime
                        p.quad(
                            top=1, bottom=0,
                            left=timestamps[start_idx],
                            right=timestamps[i],
                            color=colors[prev_regime],
                            alpha=0.3,
                            line_color=None
                        )
                        
                        # Add label
                        if i - start_idx > 5:  # Only add label if segment is wide enough
                            p.text(
                                x=(timestamps[start_idx] + timestamps[i]) / 2,
                                y=0.5,
                                text=prev_regime,
                                text_font_size="10pt",
                                text_color="black",
                                text_align="center"
                            )
                            
                    # Start new segment
                    start_idx = i
                    
                prev_regime = regime
                
            # Add the last segment
            if prev_regime is not None:
                p.quad(
                    top=1, bottom=0,
                    left=timestamps[start_idx],
                    right=timestamps[-1],
                    color=colors[prev_regime],
                    alpha=0.3,
                    line_color=None
                )
                
                # Add label
                if len(timestamps) - start_idx > 5:
                    p.text(
                        x=(timestamps[start_idx] + timestamps[-1]) / 2,
                        y=0.5,
                        text=prev_regime,
                        text_font_size="10pt",
                        text_color="black",
                        text_align="center"
                    )
                    
            # Add signal lines if provided
            if signal_df is not None:
                for name in signal_df.columns:
                    if name != 'timestamp':
                        p.line(
                            signal_df['timestamp'],
                            signal_df[name],
                            line_width=2,
                            legend_label=name,
                            color=self.config["line_color"]
                        )
                        
                # Configure legend
                p.legend.location = "top_right"
                p.legend.click_policy = "hide"
                
            # Add hover tool
            hover = HoverTool(
                tooltips=[
                    ("Time", "@x{%F %T}"),
                    ("Regime", "$name")
                ],
                formatters={"@x": "datetime"} if isinstance(timestamps[0], datetime) else {}
            )
            p.add_tools(hover)
            
            viz_obj = p
            
        elif self.active_backend == "holoviews" and HOLOVIEWS_AVAILABLE and interactive:
            # Create HoloViews visualization
            # Convert regimes to categorical areas
            regime_areas = []
            prev_regime = None
            start_idx = 0
            
            for i, regime in enumerate(regimes):
                if regime != prev_regime:
                    if prev_regime is not None:
                        # Add area for previous regime
                        area = hv.Area((timestamps[start_idx:i+1], [0]*len(timestamps[start_idx:i+1]), [1]*len(timestamps[start_idx:i+1]))).opts(
                            color=regime_colors.get(prev_regime, regime_colors.get('default', '#7f7f7f')),
                            alpha=0.3,
                            line_alpha=0
                        )
                        regime_areas.append(area)
                        
                        # Add label
                        if i - start_idx > 5:
                            text = hv.Text((timestamps[start_idx] + timestamps[i]) / 2, 0.5, prev_regime)
                            regime_areas.append(text)
                            
                    # Start new segment
                    start_idx = i
                    
                prev_regime = regime
                
            # Add the last segment
            if prev_regime is not None:
                area = hv.Area((timestamps[start_idx:], [0]*len(timestamps[start_idx:]), [1]*len(timestamps[start_idx:]))).opts(
                    color=regime_colors.get(prev_regime, regime_colors.get('default', '#7f7f7f')),
                    alpha=0.3,
                    line_alpha=0
                )
                regime_areas.append(area)
                
                # Add label
                if len(timestamps) - start_idx > 5:
                    text = hv.Text((timestamps[start_idx] + timestamps[-1]) / 2, 0.5, prev_regime)
                    regime_areas.append(text)
                    
            # Combine regime areas
            regimes_overlay = hv.Overlay(regime_areas)
            
            # Add signals if provided
            if signal_df is not None:
                curves = []
                
                for name in signal_df.columns:
                    if name != 'timestamp':
                        curve = hv.Curve((signal_df['timestamp'], signal_df[name])).opts(
                            line_width=2
                        ).relabel(name)
                        curves.append(curve)
                        
                # Combine curves with regimes
                viz_obj = regimes_overlay * hv.Overlay(curves)
            else:
                viz_obj = regimes_overlay
                
            # Apply options
            viz_obj = viz_obj.opts(
                opts.Overlay(
                    width=self.config["default_width"],
                    height=self.config["default_height"],
                    title=title,
                    tools=['hover'],
                    show_grid=True
                )
            )
            
        else:
            # Fallback to Matplotlib (static)
            if not MATPLOTLIB_AVAILABLE:
                self.logger.error("No visualization backends available")
                return None
                
            # Create figure
            if signal_df is not None:
                # Create figure with two subplots (shared x-axis)
                fig, (ax1, ax2) = plt.subplots(
                    2, 1, 
                    figsize=(self.config["default_width"] / 100, self.config["default_height"] / 100),
                    sharex=True,
                    gridspec_kw={'height_ratios': [1, 4]}  # Regime plot is smaller than signal plot
                )
                
                # Plot regimes in top subplot
                prev_regime = None
                start_idx = 0
                
                for i, regime in enumerate(regimes):
                    if regime != prev_regime:
                        if prev_regime is not None:
                            # Add colored region for previous regime
                            ax1.axvspan(
                                timestamps[start_idx], timestamps[i],
                                facecolor=regime_colors.get(prev_regime, regime_colors.get('default', '#7f7f7f')),
                                alpha=0.6
                            )
                            
                            # Add label
                            if i - start_idx > 5:
                                ax1.text(
                                    (timestamps[start_idx] + timestamps[i]) / 2,
                                    0.5,
                                    prev_regime,
                                    horizontalalignment='center',
                                    verticalalignment='center',
                                    fontsize=8,
                                    bbox=dict(facecolor='white', alpha=0.7, boxstyle='round')
                                )
                                
                        # Start new segment
                        start_idx = i
                        
                    prev_regime = regime
                    
                # Add the last segment
                if prev_regime is not None:
                    ax1.axvspan(
                        timestamps[start_idx], timestamps[-1],
                        facecolor=regime_colors.get(prev_regime, regime_colors.get('default', '#7f7f7f')),
                        alpha=0.6
                    )
                    
                    # Add label
                    if len(timestamps) - start_idx > 5:
                        ax1.text(
                            (timestamps[start_idx] + timestamps[-1]) / 2,
                            0.5,
                            prev_regime,
                            horizontalalignment='center',
                            verticalalignment='center',
                            fontsize=8,
                            bbox=dict(facecolor='white', alpha=0.7, boxstyle='round')
                        )
                        
                # Remove y-axis for regime plot
                ax1.set_yticks([])
                ax1.set_yticklabels([])
                
                # Plot signals in bottom subplot
                for name in signal_df.columns:
                    if name != 'timestamp':
                        ax2.plot(timestamps, signal_df[name], label=name)
                        
                # Add legend to signal plot
                ax2.legend()
                
                # Set grid for signal plot
                ax2.grid(self.config["use_gridlines"], alpha=0.3)
                
                # Format x-axis for datetime
                if isinstance(timestamps[0], datetime):
                    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
                    fig.autofmt_xdate()
                    
            else:
                # Create figure with just regimes
                fig, ax = plt.subplots(figsize=(
                    self.config["default_width"] / 100,
                    self.config["default_height"] / 100
                ))
                
                # Plot regimes
                prev_regime = None
                start_idx = 0
                
                for i, regime in enumerate(regimes):
                    if regime != prev_regime:
                        if prev_regime is not None:
                            # Add colored region for previous regime
                            ax.axvspan(
                                timestamps[start_idx], timestamps[i],
                                facecolor=regime_colors.get(prev_regime, regime_colors.get('default', '#7f7f7f')),
                                alpha=0.6
                            )
                            
                            # Add label
                            if i - start_idx > 5:
                                ax.text(
                                    (timestamps[start_idx] + timestamps[i]) / 2,
                                    0.5,
                                    prev_regime,
                                    horizontalalignment='center',
                                    verticalalignment='center',
                                    transform=ax.get_xaxis_transform(),
                                    bbox=dict(facecolor='white', alpha=0.7, boxstyle='round')
                                )
                                
                        # Start new segment
                        start_idx = i
                        
                    prev_regime = regime
                    
                # Add the last segment
                if prev_regime is not None:
                    ax.axvspan(
                        timestamps[start_idx], timestamps[-1],
                        facecolor=regime_colors.get(prev_regime, regime_colors.get('default', '#7f7f7f')),
                        alpha=0.6
                    )
                    
                    # Add label
                    if len(timestamps) - start_idx > 5:
                        ax.text(
                            (timestamps[start_idx] + timestamps[-1]) / 2,
                            0.5,
                            prev_regime,
                            horizontalalignment='center',
                            verticalalignment='center',
                            transform=ax.get_xaxis_transform(),
                            bbox=dict(facecolor='white', alpha=0.7, boxstyle='round')
                        )
                        
                # Format x-axis for datetime
                if isinstance(timestamps[0], datetime):
                    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
                    fig.autofmt_xdate()
                    
                # Remove y-axis
                ax.set_yticks([])
                ax.set_yticklabels([])
                
            # Set title
            fig.suptitle(title)
            
            # Apply tight layout
            if self.config["use_tight_layout"]:
                fig.tight_layout()
                
            viz_obj = fig
            
        # Cache visualization
        self._cache_plot(cache_key, viz_obj)
        
        return viz_obj
    
    def create_pattern_visualization(self, pattern_data: Dict[str, Any], prices: Optional[List[float]] = None,
                                  timestamps: Optional[List[Union[str, datetime, float]]] = None,
                                  title: Optional[str] = None,
                                  output_format: Optional[Union[str, OutputFormat]] = None) -> Any:
        """
        Create visualization for pattern recognition results.
        
        Args:
            pattern_data: Dictionary with pattern recognition data
            prices: Optional price data to overlay
            timestamps: Optional list of timestamps
            title: Optional plot title
            output_format: Output format
            
        Returns:
            Visualization object
        """
        # Determine output format
        if output_format is None:
            output_format = self.config["default_format"]
        elif isinstance(output_format, str):
            output_format = output_format.lower()
            
        # Get interactive preference
        interactive = self.config["default_interactive"]
        
        # Set default title if not provided
        if title is None:
            title = "Pattern Recognition Visualization"
            
        # Create cache key
        cache_key = ("pattern", hash(str(pattern_data)), hash(str(prices)) if prices is not None else None,
                   output_format, title)
        cached_viz = self._get_cached_plot(cache_key)
        if cached_viz is not None:
            return cached_viz
            
        # Extract pattern data
        patterns = pattern_data.get("patterns", [])
        pattern_info = pattern_data.get("info", {})
        
        # Handle missing timestamps
        if timestamps is None and prices is not None:
            # Create default timestamps (ascending integers)
            timestamps = list(range(len(prices)))
            
        # Convert timestamps to datetime if necessary
        if timestamps and not isinstance(timestamps[0], datetime):
            try:
                if isinstance(timestamps[0], str):
                    timestamps = [pd.to_datetime(ts) for ts in timestamps]
                elif isinstance(timestamps[0], (int, float)):
                    timestamps = [pd.to_datetime(ts, unit='s') for ts in timestamps]
            except Exception as e:
                self.logger.warning(f"Failed to convert timestamps to datetime: {e}")
                # Use integers as fallback
                timestamps = list(range(len(timestamps)))
                
        # Create visualization based on backend and interactivity
        viz_obj = None
        
        if self.active_backend == "plotly" and PLOTLY_AVAILABLE and interactive:
            # Create plotly figure
            fig = go.Figure()
            
            # Add price data if provided
            if prices is not None and timestamps is not None:
                fig.add_trace(go.Scatter(
                    x=timestamps,
                    y=prices,
                    mode='lines',
                    name='Price',
                    line=dict(color=self.config["line_color"], width=1)
                ))
                
            # Add patterns
            for pattern in patterns:
                pattern_type = pattern.get("type", "unknown")
                start_idx = pattern.get("start_idx", 0)
                end_idx = pattern.get("end_idx", len(prices)-1 if prices is not None else 0)
                score = pattern.get("score", 0)
                points = pattern.get("points", [])
                
                # Add pattern area
                if timestamps is not None and start_idx < len(timestamps) and end_idx < len(timestamps):
                    fig.add_trace(go.Scatter(
                        x=timestamps[start_idx:end_idx+1],
                        y=prices[start_idx:end_idx+1] if prices is not None else [],
                        fill='toself',
                        mode='none',
                        name=f"{pattern_type} (Score: {score:.2f})",
                        fillcolor=self.config["highlight_color"],
                        opacity=0.3
                    ))
                    
                # Add pattern points
                if points and prices is not None:
                    point_x = []
                    point_y = []
                    point_text = []
                    
                    for point in points:
                        idx = point.get("idx", 0)
                        label = point.get("label", "")
                        
                        if idx < len(timestamps) and idx < len(prices):
                            point_x.append(timestamps[idx])
                            point_y.append(prices[idx])
                            point_text.append(label)
                            
                    fig.add_trace(go.Scatter(
                        x=point_x,
                        y=point_y,
                        mode='markers+text',
                        name=f"{pattern_type} Points",
                        marker=dict(size=10, color=self.config["highlight_color"]),
                        text=point_text,
                        textposition="top center"
                    ))
                    
            # Customize layout
            fig.update_layout(
                title=title,
                xaxis_title="Time",
                yaxis_title="Price",
                height=self.config["default_height"],
                width=self.config["default_width"],
                template="plotly_dark" if self.config["default_theme"] == "dark" else "plotly_white",
                hovermode="closest",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            viz_obj = fig
            
        elif self.active_backend == "bokeh" and BOKEH_AVAILABLE and interactive:
            # Create bokeh figure
            p = bokeh_figure(
                width=self.config["default_width"],
                height=self.config["default_height"],
                title=title,
                x_axis_type="datetime" if isinstance(timestamps[0], datetime) else "auto",
                tools="pan,wheel_zoom,box_zoom,reset,save",
                toolbar_location="above"
            )
            
            # Add price data if provided
            if prices is not None and timestamps is not None:
                p.line(
                    x=timestamps,
                    y=prices,
                    line_width=1,
                    legend_label="Price",
                    color=self.config["line_color"]
                )
                
            # Add patterns
            for pattern in patterns:
                pattern_type = pattern.get("type", "unknown")
                start_idx = pattern.get("start_idx", 0)
                end_idx = pattern.get("end_idx", len(prices)-1 if prices is not None else 0)
                score = pattern.get("score", 0)
                points = pattern.get("points", [])
                
                # Add pattern area
                if timestamps is not None and start_idx < len(timestamps) and end_idx < len(timestamps):
                    x_range = timestamps[start_idx:end_idx+1]
                    y_range = prices[start_idx:end_idx+1] if prices is not None else []
                    
                    if len(x_range) > 0 and len(y_range) > 0:
                        # Highlight area
                        p.patch(
                            x=x_range,
                            y=y_range,
                            fill_color=self.config["highlight_color"],
                            fill_alpha=0.3,
                            legend_label=f"{pattern_type} (Score: {score:.2f})"
                        )
                        
                # Add pattern points
                if points and prices is not None:
                    point_x = []
                    point_y = []
                    point_text = []
                    
                    for point in points:
                        idx = point.get("idx", 0)
                        label = point.get("label", "")
                        
                        if idx < len(timestamps) and idx < len(prices):
                            point_x.append(timestamps[idx])
                            point_y.append(prices[idx])
                            point_text.append(label)
                            
                    # Add markers
                    p.circle(
                        x=point_x,
                        y=point_y,
                        size=8,
                        color=self.config["highlight_color"],
                        legend_label=f"{pattern_type} Points"
                    )
                    
                    # Add labels
                    for i in range(len(point_x)):
                        p.text(
                            x=point_x[i],
                            y=point_y[i],
                            text=point_text[i],
                            text_font_size="8pt",
                            text_align="center",
                            text_baseline="bottom"
                        )
                        
            # Configure legend
            p.legend.location = "top_right"
            p.legend.click_policy = "hide"
            
            # Add hover tool
            hover = HoverTool(
                tooltips=[
                    ("Time", "@x{%F %T}"),
                    ("Price", "@y{0.0000}")
                ],
                formatters={"@x": "datetime"} if isinstance(timestamps[0], datetime) else {}
            )
            p.add_tools(hover)
            
            viz_obj = p
            
        else:
            # Fallback to Matplotlib (static)
            if not MATPLOTLIB_AVAILABLE:
                self.logger.error("No visualization backends available")
                return None
                
            # Create figure
            fig, ax = plt.subplots(figsize=(
                self.config["default_width"] / 100,
                self.config["default_height"] / 100
            ))
            
            # Add price data if provided
            if prices is not None and timestamps is not None:
                ax.plot(timestamps, prices, color=self.config["line_color"], linewidth=1, label="Price")
                
            # Add patterns
            for pattern in patterns:
                pattern_type = pattern.get("type", "unknown")
                start_idx = pattern.get("start_idx", 0)
                end_idx = pattern.get("end_idx", len(prices)-1 if prices is not None else 0)
                score = pattern.get("score", 0)
                points = pattern.get("points", [])
                
                # Add pattern area
                if timestamps is not None and start_idx < len(timestamps) and end_idx < len(timestamps):
                    x_range = timestamps[start_idx:end_idx+1]
                    y_range = prices[start_idx:end_idx+1] if prices is not None else []
                    
                    if len(x_range) > 0 and len(y_range) > 0:
                        # Highlight area
                        ax.fill_between(
                            x_range,
                            y_range,
                            alpha=0.3,
                            color=self.config["highlight_color"],
                            label=f"{pattern_type} (Score: {score:.2f})"
                        )
                        
                # Add pattern points
                if points and prices is not None:
                    point_x = []
                    point_y = []
                    point_text = []
                    
                    for point in points:
                        idx = point.get("idx", 0)
                        label = point.get("label", "")
                        
                        if idx < len(timestamps) and idx < len(prices):
                            point_x.append(timestamps[idx])
                            point_y.append(prices[idx])
                            point_text.append(label)
                            
                    # Add markers
                    ax.scatter(
                        point_x,
                        point_y,
                        color=self.config["highlight_color"],
                        s=50,
                        label=f"{pattern_type} Points"
                    )
                    
                    # Add labels
                    for i in range(len(point_x)):
                        ax.annotate(
                            point_text[i],
                            (point_x[i], point_y[i]),
                            textcoords="offset points",
                            xytext=(0, 10),
                            ha='center',
                            fontsize=self.config["tick_font_size"]
                        )
                        
            # Format x-axis for datetime
            if timestamps and isinstance(timestamps[0], datetime):
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
                fig.autofmt_xdate()
                
            # Set grid
            ax.grid(self.config["use_gridlines"], alpha=0.3)
            
            # Add legend
            ax.legend()
            
            # Set title
            ax.set_title(title)
            
            # Apply tight layout
            if self.config["use_tight_layout"]:
                fig.tight_layout()
                
            viz_obj = fig
            
        # Cache visualization
        self._cache_plot(cache_key, viz_obj)
        
        return viz_obj
    
    def create_contagion_visualization(self, contagion_data: Dict[str, Any], 
                                    title: Optional[str] = None,
                                    output_format: Optional[Union[str, OutputFormat]] = None) -> Any:
        """
        Create visualization for contagion risk analysis.
        
        Args:
            contagion_data: Dictionary with contagion risk data
            title: Optional plot title
            output_format: Output format
            
        Returns:
            Visualization object
        """
        # Determine output format
        if output_format is None:
            output_format = self.config["default_format"]
        elif isinstance(output_format, str):
            output_format = output_format.lower()
            
        # Get interactive preference
        interactive = self.config["default_interactive"]
        
        # Set default title if not provided
        if title is None:
            source_symbol = contagion_data.get("symbol", "")
            title = f"Contagion Risk Analysis for {source_symbol}" if source_symbol else "Contagion Risk Analysis"
            
        # Create cache key
        cache_key = ("contagion", hash(str(contagion_data)), output_format, title)
        cached_viz = self._get_cached_plot(cache_key)
        if cached_viz is not None:
            return cached_viz
            
        # Extract contagion data
        symbol = contagion_data.get("symbol", "")
        risk_score = contagion_data.get("risk_score", 0)
        impact_symbols = contagion_data.get("impact_symbols", [])
        systemic_impact = contagion_data.get("systemic_impact", 0)
        
        # Need NetworkX for graph visualization
        if not NETWORKX_AVAILABLE:
            self.logger.error("NetworkX not available, cannot create contagion visualization")
            return None
            
        # Create graph
        G = nx.Graph()
        
        # Add nodes
        G.add_node(symbol, type="source", risk=risk_score)
        
        # Add impact symbols as nodes
        for target, impact in impact_symbols:
            G.add_node(target, type="target", impact=impact)
            G.add_edge(symbol, target, weight=impact)
            
        # Calculate layout
        layout_algo = self.config.get("graph_layout", "spring")
        
        if layout_algo == "spring":
            pos = nx.spring_layout(G, weight='weight', seed=42)
        elif layout_algo == "kamada_kawai":
            pos = nx.kamada_kawai_layout(G, weight='weight')
        elif layout_algo == "spectral":
            pos = nx.spectral_layout(G)
        elif layout_algo == "circular":
            pos = nx.circular_layout(G)
        else:
            # Default to spring
            pos = nx.spring_layout(G, weight='weight', seed=42)
            
        # Create visualization based on backend and interactivity
        viz_obj = None
        
        if self.active_backend == "plotly" and PLOTLY_AVAILABLE and interactive:
            # Create plotly network visualization
            # Create edges
            edge_x = []
            edge_y = []
            edge_weights = []
            
            for source, target, data in G.edges(data=True):
                x0, y0 = pos[source]
                x1, y1 = pos[target]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                edge_weights.append(data.get('weight', 1.0))
                
            # Create edge trace
            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=1, color='rgba(150,150,150,0.5)'),
                hoverinfo='none',
                mode='lines'
            )
            
            # Create nodes
            node_x = []
            node_y = []
            node_text = []
            node_size = []
            node_color = []
            
            for node, data in G.nodes(data=True):
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                
                # Node text
                if data.get('type') == 'source':
                    node_text.append(f"{node}<br>Risk Score: {data.get('risk', 0):.2f}<br>Systemic Impact: {systemic_impact:.2f}")
                    node_size.append(20)
                    node_color.append('#ff7f0e')  # Orange for source
                else:
                    impact = data.get('impact', 0)
                    node_text.append(f"{node}<br>Impact: {impact:.2f}")
                    node_size.append(10 + 20 * impact)
                    # Color gradient based on impact
                    r = min(255, int(255 * impact))
                    g = min(255, int(255 * (1 - impact)))
                    b = 50
                    node_color.append(f'rgb({r},{g},{b})')
                    
            # Create node trace
            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers',
                hoverinfo='text',
                text=node_text,
                marker=dict(
                    showscale=False,
                    color=node_color,
                    size=node_size,
                    line=dict(width=1, color='rgb(50,50,50)')
                )
            )
            
            # Create figure
            fig = go.Figure(data=[edge_trace, node_trace])
            
            # Add source node label
            fig.add_annotation(
                x=pos[symbol][0],
                y=pos[symbol][1],
                text=symbol,
                showarrow=True,
                arrowhead=1,
                ax=0,
                ay=-30
            )
            
            # Customize layout
            fig.update_layout(
                title=title,
                titlefont_size=self.config["title_font_size"],
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=self.config["default_height"],
                width=self.config["default_width"],
                template="plotly_dark" if self.config["default_theme"] == "dark" else "plotly_white",
                annotations=[
                    dict(
                        text=f"Risk Score: {risk_score:.2f}<br>Systemic Impact: {systemic_impact:.2f}",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.01, y=0.01
                    )
                ]
            )
            
            viz_obj = fig
            
        elif self.active_backend == "bokeh" and BOKEH_AVAILABLE and interactive:
            # Create bokeh network visualization
            # Prepare data
            node_data = {
                'x': [],
                'y': [],
                'name': [],
                'size': [],
                'color': [],
                'is_source': [],
                'tooltip': []
            }
            
            edge_data = {
                'x0': [],
                'y0': [],
                'x1': [],
                'y1': [],
                'weight': []
            }
            
            # Add node data
            for node, data in G.nodes(data=True):
                x, y = pos[node]
                node_data['x'].append(x)
                node_data['y'].append(y)
                node_data['name'].append(node)
                
                if data.get('type') == 'source':
                    node_data['is_source'].append(True)
                    node_data['size'].append(15)
                    node_data['color'].append('#ff7f0e')  # Orange for source
                    node_data['tooltip'].append(f"{node}\nRisk Score: {data.get('risk', 0):.2f}\nSystemic Impact: {systemic_impact:.2f}")
                else:
                    node_data['is_source'].append(False)
                    impact = data.get('impact', 0)
                    node_data['size'].append(8 + 15 * impact)
                    # Color gradient based on impact
                    r = min(255, int(255 * impact))
                    g = min(255, int(255 * (1 - impact)))
                    b = 50
                    node_data['color'].append(f'#{r:02x}{g:02x}{b:02x}')
                    node_data['tooltip'].append(f"{node}\nImpact: {impact:.2f}")
                    
            # Add edge data
            for source, target, data in G.edges(data=True):
                x0, y0 = pos[source]
                x1, y1 = pos[target]
                weight = data.get('weight', 1.0)
                
                edge_data['x0'].append(x0)
                edge_data['y0'].append(y0)
                edge_data['x1'].append(x1)
                edge_data['y1'].append(y1)
                edge_data['weight'].append(weight)
                
            # Create data sources
            node_source = ColumnDataSource(node_data)
            edge_source = ColumnDataSource(edge_data)
            
            # Create figure
            p = bokeh_figure(
                width=self.config["default_width"],
                height=self.config["default_height"],
                title=title,
                tools="pan,wheel_zoom,box_zoom,reset,save",
                toolbar_location="above",
                x_range=(-1.1, 1.1),
                y_range=(-1.1, 1.1)
            )
            
            # Add edges
            for i in range(len(edge_data['x0'])):
                width = 1 + 5 * edge_data['weight'][i]
                p.line(
                    x=[edge_data['x0'][i], edge_data['x1'][i]],
                    y=[edge_data['y0'][i], edge_data['y1'][i]],
                    line_width=width,
                    line_alpha=0.6,
                    line_color='gray'
                )
                
            # Add nodes
            node_renderer = p.circle(
                x='x',
                y='y',
                size='size',
                fill_color='color',
                line_color='black',
                line_width=1,
                source=node_source
            )
            
            # Add node labels
            labels = LabelSet(
                x='x',
                y='y',
                text='name',
                text_font_size='8pt',
                x_offset=5,
                y_offset=5,
                source=node_source,
                render_mode='canvas'
            )
            p.add_layout(labels)
            
            # Add hover tool
            hover = HoverTool(
                tooltips=[
                    ("Info", "@tooltip")
                ],
                renderers=[node_renderer]
            )
            p.add_tools(hover)
            
            # Add info text
            info_text = f"Risk Score: {risk_score:.2f}     Systemic Impact: {systemic_impact:.2f}"
            p.add_layout(Title(text=info_text, align="left"), "below")
            
            # Customize
            p.axis.visible = False
            p.grid.visible = False
            
            viz_obj = p
            
        else:
            # Fallback to Matplotlib (static)
            if not MATPLOTLIB_AVAILABLE:
                self.logger.error("No visualization backends available")
                return None
                
            # Create figure
            fig, ax = plt.subplots(figsize=(
                self.config["default_width"] / 100,
                self.config["default_height"] / 100
            ))
            
            # Create node colors and sizes
            node_colors = []
            node_sizes = []
            
            for node, data in G.nodes(data=True):
                if data.get('type') == 'source':
                    node_colors.append('#ff7f0e')  # Orange for source
                    node_sizes.append(500)
                else:
                    impact = data.get('impact', 0)
                    # Color gradient based on impact
                    r = min(1.0, impact)
                    g = min(1.0, 1 - impact)
                    b = 0.2
                    node_colors.append((r, g, b))
                    node_sizes.append(100 + 400 * impact)
                    
            # Create edge weights
            edge_widths = []
            for _, _, data in G.edges(data=True):
                weight = data.get('weight', 1.0)
                edge_widths.append(1 + 3 * weight)
                
            # Draw network
            nx.draw_networkx(
                G,
                pos=pos,
                ax=ax,
                with_labels=True,
                node_color=node_colors,
                node_size=node_sizes,
                width=edge_widths,
                edge_color='gray',
                alpha=0.7,
                font_size=self.config["tick_font_size"],
                font_weight='bold'
            )
            
            # Add info text
            info_text = f"Risk Score: {risk_score:.2f}     Systemic Impact: {systemic_impact:.2f}"
            ax.text(
                0.02, 0.02, info_text,
                transform=ax.transAxes,
                fontsize=self.config["axis_font_size"],
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round')
            )
            
            # Remove axis
            ax.axis('off')
            
            # Set title
            ax.set_title(title)
            
            # Apply tight layout
            if self.config["use_tight_layout"]:
                fig.tight_layout()
                
            viz_obj = fig
            
        # Cache visualization
        self._cache_plot(cache_key, viz_obj)
        
        return viz_obj
    
    def create_performance_dashboard(self, performance_metrics: Dict[str, Dict[str, float]],
                                  title: Optional[str] = None,
                                  output_format: Optional[Union[str, OutputFormat]] = None) -> Any:
        """
        Create performance dashboard visualization.
        
        Args:
            performance_metrics: Dictionary of component name to metrics
            title: Optional plot title
            output_format: Output format
            
        Returns:
            Visualization object
        """
        # Determine output format
        if output_format is None:
            output_format = self.config["default_format"]
        elif isinstance(output_format, str):
            output_format = output_format.lower()
            
        # Get interactive preference
        interactive = self.config["default_interactive"]
        
        # Set default title if not provided
        if title is None:
            title = "CDFA Performance Dashboard"
            
        # Create cache key
        cache_key = ("dashboard", hash(str(performance_metrics)), output_format, title)
        cached_viz = self._get_cached_plot(cache_key)
        if cached_viz is not None:
            return cached_viz
            
        # Extract metrics
        components = list(performance_metrics.keys())
        metrics = {}
        
        # Get all available metrics
        for component, metrics_dict in performance_metrics.items():
            for metric, value in metrics_dict.items():
                if metric not in metrics:
                    metrics[metric] = []
                    
        # Fill metrics
        for metric in metrics:
            for component in components:
                metrics[metric].append(
                    performance_metrics[component].get(metric, 0.0)
                )
                
        # Create visualization based on backend and interactivity
        viz_obj = None
        
        if self.active_backend == "plotly" and PLOTLY_AVAILABLE and interactive:
            # Create plotly dashboard
            # Determine optimal grid layout
            n_metrics = len(metrics)
            cols = min(3, n_metrics)
            rows = (n_metrics + cols - 1) // cols  # Ceiling division
            
            # Create subplot grid
            fig = make_subplots(
                rows=rows, cols=cols,
                subplot_titles=list(metrics.keys())
            )
            
            # Add bar charts
            i = 0
            for metric, values in metrics.items():
                row = i // cols + 1
                col = i % cols + 1
                
                fig.add_trace(
                    go.Bar(
                        x=components,
                        y=values,
                        name=metric
                    ),
                    row=row, col=col
                )
                
                i += 1
                
            # Customize layout
            fig.update_layout(
                title=title,
                height=self.config["default_height"],
                width=self.config["default_width"],
                template="plotly_dark" if self.config["default_theme"] == "dark" else "plotly_white",
                showlegend=False
            )
            
            # Update axes
            fig.update_xaxes(tickangle=45)
            
            viz_obj = fig
            
        elif self.active_backend == "bokeh" and BOKEH_AVAILABLE and interactive:
            # Create bokeh dashboard
            # Determine optimal grid layout
            n_metrics = len(metrics)
            cols = min(3, n_metrics)
            rows = (n_metrics + cols - 1) // cols  # Ceiling division
            
            # Create subplots
            plots = []
            
            for metric, values in metrics.items():
                # Create data source
                source = ColumnDataSource({
                    'components': components,
                    'values': values
                })
                
                # Create figure
                p = bokeh_figure(
                    title=metric,
                    x_range=components,
                    height=self.config["default_height"] // rows,
                    width=self.config["default_width"] // cols,
                    toolbar_location=None
                )
                
                # Add bar chart
                p.vbar(
                    x='components',
                    top='values',
                    width=0.9,
                    source=source,
                    fill_color=self.config["line_color"],
                    line_color=None
                )
                
                # Customize
                p.xaxis.major_label_orientation = np.pi/4
                p.y_range.start = 0
                
                # Add hover tool
                hover = HoverTool(
                    tooltips=[
                        ("Component", "@components"),
                        (metric, "@values{0.000}")
                    ]
                )
                p.add_tools(hover)
                
                plots.append(p)
                
            # Create grid
            grid = gridplot(
                [plots[i:i+cols] for i in range(0, len(plots), cols)]
            )
            
            viz_obj = grid
            
        else:
            # Fallback to Matplotlib (static)
            if not MATPLOTLIB_AVAILABLE:
                self.logger.error("No visualization backends available")
                return None
                
            # Determine optimal grid layout
            n_metrics = len(metrics)
            cols = min(3, n_metrics)
            rows = (n_metrics + cols - 1) // cols  # Ceiling division
            
            # Create figure
            fig, axs = plt.subplots(
                rows, cols,
                figsize=(
                    self.config["default_width"] / 100,
                    self.config["default_height"] / 100
                )
            )
            
            # Handle single row/column case
            if n_metrics == 1:
                axs = np.array([axs])
            
            # Flatten axs array for easier indexing
            axs = axs.flatten() if hasattr(axs, 'flatten') else [axs]
            
            # Create bar charts
            for i, (metric, values) in enumerate(metrics.items()):
                if i < len(axs):
                    ax = axs[i]
                    
                    # Create bar chart
                    ax.bar(components, values)
                    
                    # Set title
                    ax.set_title(metric)
                    
                    # Rotate x-axis labels
                    ax.set_xticklabels(components, rotation=45, ha='right')
                    
                    # Set grid
                    ax.grid(self.config["use_gridlines"], axis='y', alpha=0.3)
                    
            # Hide unused subplots
            for i in range(n_metrics, len(axs)):
                axs[i].axis('off')
                
            # Set title
            fig.suptitle(title)
            
            # Apply tight layout
            if self.config["use_tight_layout"]:
                fig.tight_layout(rect=[0, 0, 1, 0.95])
                
            viz_obj = fig
            
        # Cache visualization
        self._cache_plot(cache_key, viz_obj)
        
        return viz_obj
