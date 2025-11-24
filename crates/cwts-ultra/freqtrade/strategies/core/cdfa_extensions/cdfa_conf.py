#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CDFA Integration with FreqTrade

This module integrates the Adaptive Market Data Fetcher, CDFA Analyzers,
and Holoviews Visualization into a cohesive system that interfaces with
FreqTrade for algorithmic trading.

Author: Created on May 8, 2025
"""

import logging
import os
import json
import time
import threading
import datetime
import signal
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple, Set, Callable

# Configuration and settings handler
class CDFAConfigManager:
    """Configuration manager for CDFA components with frontend integration capabilities."""
    
    def __init__(self, config_dir: str = "~/.cdfa"):
        """
        Initialize configuration manager.
        
        Args:
            config_dir: Directory for configuration files
        """
        self.logger = logging.getLogger(f"{__name__}.CDFAConfigManager")
        
        # Expand config directory path
        self.config_dir = os.path.expanduser(config_dir)
        
        # Create config directory if needed
        os.makedirs(self.config_dir, exist_ok=True)
        
        # Default configurations
        self.default_configs = {
            "market_data_fetcher": None,  # Will be loaded from AdaptiveMarketDataFetcher
            "visualizer": None,           # Will be loaded from HoloviewsVisualizer
            "freqtrade": {
                "enabled": True,
                "pairlist_path": "~/freqtrade/user_data/data/cdfa_pairs.json",
                "export_interval": 3600,  # 1 hour
                "include_metadata": True,
                "metadata_path": "~/freqtrade/user_data/data/cdfa_metadata.json",
                "min_active_pairs": 20,
                "max_active_pairs": 50,
                "auto_reload": True,
                "quote_currencies": ["USDT"],
                "user_interface": {
                    "webhook_url": None,
                    "telegram_bot_token": None,
                    "telegram_chat_id": None
                }
            },
            "dashboard": {
                "enabled": True,
                "port": 8501,
                "auto_refresh": 300,  # 5 minutes
                "default_visualizations": [
                    "opportunity", "regime", "correlation", "whale"
                ],
                "export_path": "~/cdfa_dashboard",
                "auto_export_html": True,
                "auto_export_interval": 86400  # 1 day
            },
            "system": {
                "log_level": "INFO",
                "data_dir": "~/.cdfa/data",
                "cache_ttl": 3600,
                "update_interval": 3600,
                "discovery_interval": 86400
            }
        }
        
        # Load configurations
        self.configs = {}
        self._load_configs()
        
        # Initialize component references
        self.market_data_fetcher = None
        self.visualizer = None
        
        # Initialize frontend integration
        self.frontend_config = self._prepare_frontend_config()
        
        self.logger.info("Configuration manager initialized")
    
    def _load_configs(self):
        """Load configurations from files."""
        config_files = {
            "market_data_fetcher": os.path.join(self.config_dir, "market_data_fetcher.json"),
            "visualizer": os.path.join(self.config_dir, "visualizer.json"),
            "freqtrade": os.path.join(self.config_dir, "freqtrade.json"),
            "dashboard": os.path.join(self.config_dir, "dashboard.json"),
            "system": os.path.join(self.config_dir, "system.json")
        }
        
        for config_name, config_file in config_files.items():
            try:
                if os.path.exists(config_file):
                    with open(config_file, 'r') as f:
                        self.configs[config_name] = json.load(f)
                    
                    self.logger.info(f"Loaded configuration from {config_file}")
                else:
                    # Use default config
                    self.configs[config_name] = self.default_configs.get(config_name, {})
                    
                    # Create config file with default values
                    with open(config_file, 'w') as f:
                        json.dump(self.configs[config_name], f, indent=2)
                    
                    self.logger.info(f"Created default configuration at {config_file}")
                    
            except Exception as e:
                self.logger.error(f"Error loading configuration from {config_file}: {e}")
                # Use default config
                self.configs[config_name] = self.default_configs.get(config_name, {})
    
    def _save_configs(self):
        """Save configurations to files."""
        for config_name, config_data in self.configs.items():
            config_file = os.path.join(self.config_dir, f"{config_name}.json")
            
            try:
                with open(config_file, 'w') as f:
                    json.dump(config_data, f, indent=2)
                
                self.logger.info(f"Saved configuration to {config_file}")
                
            except Exception as e:
                self.logger.error(f"Error saving configuration to {config_file}: {e}")
    
    def get_config(self, component_name: str) -> Dict[str, Any]:
        """
        Get configuration for a component.
        
        Args:
            component_name: Component name
            
        Returns:
            Configuration dictionary
        """
        return self.configs.get(component_name, {})
    
    def set_config(self, component_name: str, config: Dict[str, Any], save: bool = True):
        """
        Set configuration for a component.
        
        Args:
            component_name: Component name
            config: Configuration dictionary
            save: Whether to save to file
        """
        self.configs[component_name] = config
        
        # Update frontend config
        self.frontend_config = self._prepare_frontend_config()
        
        if save:
            self._save_configs()
    
    def update_config(self, component_name: str, param_name: str, param_value: Any, save: bool = True):
        """
        Update a specific configuration parameter.
        
        Args:
            component_name: Component name
            param_name: Parameter name
            param_value: Parameter value
            save: Whether to save to file
        """
        if component_name not in self.configs:
            self.configs[component_name] = {}
            
        # Handle nested parameters (e.g., "system.log_level")
        if "." in param_name:
            parts = param_name.split(".")
            config = self.configs[component_name]
            
            # Navigate to the nested dict
            for part in parts[:-1]:
                if part not in config:
                    config[part] = {}
                config = config[part]
                
            # Set the value
            config[parts[-1]] = param_value
        else:
            # Set top-level parameter
            self.configs[component_name][param_name] = param_value
        
        # Update frontend config
        self.frontend_config = self._prepare_frontend_config()
        
        # Update component if initialized
        if component_name == "market_data_fetcher" and self.market_data_fetcher:
            self.market_data_fetcher.update_config_parameter(param_name, param_value)
        elif component_name == "visualizer" and self.visualizer:
            self.visualizer.update_config_parameter(param_name, param_value)
        
        if save:
            self._save_configs()
    
    def _prepare_frontend_config(self) -> Dict[str, Any]:
        """
        Prepare configuration data for frontend integration.
        
        Returns:
            Frontend-friendly configuration dictionary
        """
        # Create a frontend-friendly version of the configuration
        frontend_config = {
            "market_data_fetcher": {
                "parameters": {},
                "metadata": {}
            },
            "visualizer": {
                "parameters": {},
                "metadata": {}
            },
            "freqtrade": self.configs.get("freqtrade", {}),
            "dashboard": self.configs.get("dashboard", {}),
            "system": self.configs.get("system", {})
        }
        
        # Add parameter metadata from AdaptiveMarketDataFetcher
        try:
            from cdfa_extensions.adaptive_market_data_fetcher import AdaptiveMarketDataFetcher
            mdf_params = AdaptiveMarketDataFetcher.DEFAULT_CONFIG
            
            for param, value in mdf_params.items():
                frontend_config["market_data_fetcher"]["parameters"][param] = value
                
            # Add parameter metadata
            if hasattr(AdaptiveMarketDataFetcher, "get_config_parameters"):
                frontend_config["market_data_fetcher"]["metadata"] = (
                    AdaptiveMarketDataFetcher.get_config_parameters({})
                )
        except (ImportError, AttributeError):
            self.logger.warning("Unable to load AdaptiveMarketDataFetcher defaults")
            
        # Add parameter metadata from HoloviewsVisualizer
        try:
            from cdfa_extensions.holoviews_visualizer import HoloviewsVisualizer
            viz_params = HoloviewsVisualizer.DEFAULT_CONFIG
            
            for param, value in viz_params.items():
                frontend_config["visualizer"]["parameters"][param] = value
                
            # Add parameter metadata
            if hasattr(HoloviewsVisualizer, "get_config_parameters"):
                frontend_config["visualizer"]["metadata"] = (
                    HoloviewsVisualizer.get_config_parameters({})
                )
        except (ImportError, AttributeError):
            self.logger.warning("Unable to load HoloviewsVisualizer defaults")
            
        return frontend_config
    
    def get_frontend_config(self) -> Dict[str, Any]:
        """
        Get frontend-friendly configuration.
        
        Returns:
            Frontend configuration
        """
        return self.frontend_config
    
    def set_component(self, name: str, component: Any):
        """
        Set component reference.
        
        Args:
            name: Component name
            component: Component instance
        """
        if name == "market_data_fetcher":
            self.market_data_fetcher = component
        elif name == "visualizer":
            self.visualizer = component
        else:
            self.logger.warning(f"Unknown component type: {name}")
    
    def get_log_level(self) -> int:
        """
        Get system log level.
        
        Returns:
            Log level as integer
        """
        log_level_str = self.configs.get("system", {}).get("log_level", "INFO")
        log_level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL
        }
        
        return log_level_map.get(log_level_str, logging.INFO)
    
    def export_to_json(self, path: str = None) -> str:
        """
        Export configurations to JSON file.
        
        Args:
            path: Output file path
            
        Returns:
            Path to saved file
        """
        if path is None:
            path = os.path.join(self.config_dir, "cdfa_config_export.json")
            
        path = os.path.expanduser(path)
        
        try:
            with open(path, 'w') as f:
                json.dump(self.configs, f, indent=2)
                
            self.logger.info(f"Exported configurations to {path}")
            return path
            
        except Exception as e:
            self.logger.error(f"Error exporting configurations: {e}")
            raise
    
    def import_from_json(self, path: str, save: bool = True):
        """
        Import configurations from JSON file.
        
        Args:
            path: Input file path
            save: Whether to save to files
        """
        path = os.path.expanduser(path)
        
        try:
            with open(path, 'r') as f:
                imported_configs = json.load(f)
                
            # Update configurations
            for component_name, config in imported_configs.items():
                self.configs[component_name] = config
                
            # Update frontend config
            self.frontend_config = self._prepare_frontend_config()
            
            self.logger.info(f"Imported configurations from {path}")
            
            if save:
                self._save_configs()
                
        except Exception as e:
            self.logger.error(f"Error importing configurations: {e}")
            raise

# Main CDFA FreqTrade Integration
class CDFAFreqTradeIntegration:
    """
    Integrates CDFA components with FreqTrade for algorithmic trading.
    
    Provides:
    - Dynamic pair selection based on CDFA analysis
    - Dashboard for market visualization
    - Data pipeline between CDFA and FreqTrade
    """
    
    def __init__(self, config_manager: Optional[CDFAConfigManager] = None):
        """
        Initialize integration.
        
        Args:
            config_manager: Optional configuration manager
        """
        # Set up logging
        self.logger = logging.getLogger(f"{__name__}.CDFAFreqTradeIntegration")
        
        # Create config manager if not provided
        if config_manager is None:
            self.config_manager = CDFAConfigManager()
        else:
            self.config_manager = config_manager
            
        # Set log level
        self.logger.setLevel(self.config_manager.get_log_level())
        
        # Initialize components
        self.market_data_fetcher = None
        self.visualizer = None
        
        # Initialize threads
        self.running = True
        self.export_thread = None
        self.dashboard_thread = None
        
        # Initialize components
        self._initialize_components()
        
        # Start background threads
        if self.config_manager.get_config("freqtrade").get("enabled", True):
            self._start_freqtrade_export_thread()
            
        if self.config_manager.get_config("dashboard").get("enabled", True):
            self._start_dashboard_thread()
            
        self.logger.info("CDFA FreqTrade Integration initialized")
    
    def _initialize_components(self):
        """Initialize components."""
        try:
            # Initialize market data fetcher
            from cdfa_extensions.adaptive_market_data_fetcher import AdaptiveMarketDataFetcher
            
            mdf_config = self.config_manager.get_config("market_data_fetcher")
            self.market_data_fetcher = AdaptiveMarketDataFetcher(mdf_config)
            self.config_manager.set_component("market_data_fetcher", self.market_data_fetcher)
            
            self.logger.info("Initialized AdaptiveMarketDataFetcher")
            
        except (ImportError, Exception) as e:
            self.logger.error(f"Error initializing MarketDataFetcher: {e}")
            self.market_data_fetcher = None
            
        try:
            # Initialize visualizer
            from cdfa_extensions.holoviews_visualizer import HoloviewsVisualizer
            
            viz_config = self.config_manager.get_config("visualizer")
            self.visualizer = HoloviewsVisualizer(
                market_data_fetcher=self.market_data_fetcher,
                config=viz_config
            )
            self.config_manager.set_component("visualizer", self.visualizer)
            
            self.logger.info("Initialized HoloviewsVisualizer")
            
        except (ImportError, Exception) as e:
            self.logger.error(f"Error initializing HoloviewsVisualizer: {e}")
            self.visualizer = None
    
    def _start_freqtrade_export_thread(self):
        """Start FreqTrade export thread."""
        def export_loop():
            """Background thread for exporting to FreqTrade."""
            freqtrade_config = self.config_manager.get_config("freqtrade")
            
            pairlist_path = os.path.expanduser(freqtrade_config.get("pairlist_path", ""))
            metadata_path = os.path.expanduser(freqtrade_config.get("metadata_path", ""))
            export_interval = freqtrade_config.get("export_interval", 3600)
            include_metadata = freqtrade_config.get("include_metadata", True)
            min_pairs = freqtrade_config.get("min_active_pairs", 20)
            max_pairs = freqtrade_config.get("max_active_pairs", 50)
            quote_currencies = freqtrade_config.get("quote_currencies", ["USDT"])
            
            while self.running:
                try:
                    if self.market_data_fetcher:
                        # Export pairlist
                        self.market_data_fetcher.export_to_freqtrade_pairlist(
                            limit=max_pairs,
                            quote_currencies=quote_currencies,
                            filename=pairlist_path
                        )
                        
                        # Export metadata if enabled
                        if include_metadata and metadata_path:
                            self.market_data_fetcher.export_extended_freqtrade_data(
                                limit=max_pairs,
                                filename=metadata_path
                            )
                            
                        self.logger.info(f"Exported FreqTrade data to {pairlist_path}")
                    
                    # Sleep until next export
                    for _ in range(export_interval):
                        if not self.running:
                            break
                        time.sleep(1)
                        
                except Exception as e:
                    self.logger.error(f"Error in FreqTrade export loop: {e}")
                    time.sleep(60)  # Sleep on error
        
        # Start thread
        self.export_thread = threading.Thread(
            target=export_loop,
            daemon=True,
            name="CDFA-FreqTrade-Export"
        )
        self.export_thread.start()
        
        self.logger.info("Started FreqTrade export thread")
    
    def _start_dashboard_thread(self):
        """Start dashboard thread."""
        def dashboard_loop():
            """Background thread for dashboard management."""
            dashboard_config = self.config_manager.get_config("dashboard")
            
            port = dashboard_config.get("port", 8501)
            auto_refresh = dashboard_config.get("auto_refresh", 300)
            export_path = os.path.expanduser(dashboard_config.get("export_path", ""))
            auto_export_html = dashboard_config.get("auto_export_html", True)
            auto_export_interval = dashboard_config.get("auto_export_interval", 86400)
            
            # Create export directory if needed
            if export_path:
                os.makedirs(export_path, exist_ok=True)
            
            # Initialize dashboard server
            if self.visualizer:
                try:
                    # Start server if needed
                    server = self.visualizer.serve_dashboard(port=port)
                    self.logger.info(f"Started dashboard server at http://localhost:{port}")
                    
                    # Start auto-export loop if enabled
                    last_export = 0
                    
                    while self.running:
                        try:
                            current_time = time.time()
                            
                            # Export dashboard if needed
                            if auto_export_html and export_path and current_time - last_export >= auto_export_interval:
                                export_file = os.path.join(
                                    export_path, 
                                    f"cdfa_dashboard_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                                )
                                
                                self.visualizer.save_dashboard(
                                    filename=export_file,
                                    title="CDFA Market Intelligence Dashboard"
                                )
                                
                                # Also create latest.html
                                latest_file = os.path.join(export_path, "latest.html")
                                self.visualizer.save_dashboard(
                                    filename=latest_file,
                                    title="CDFA Market Intelligence Dashboard"
                                )
                                
                                last_export = current_time
                                
                                self.logger.info(f"Exported dashboard to {export_file}")
                            
                            # Sleep for a bit
                            time.sleep(10)
                            
                        except Exception as e:
                            self.logger.error(f"Error in dashboard management: {e}")
                            time.sleep(60)  # Sleep on error
                    
                except Exception as e:
                    self.logger.error(f"Error starting dashboard server: {e}")
                
            else:
                self.logger.warning("Visualizer not available, dashboard not started")
        
        # Start thread
        self.dashboard_thread = threading.Thread(
            target=dashboard_loop,
            daemon=True,
            name="CDFA-Dashboard"
        )
        self.dashboard_thread.start()
        
        self.logger.info("Started dashboard thread")
    
    def register_freqtrade_feedback(self, symbol: str, 
                                   signal_type: str, 
                                   result: bool, 
                                   profit_pct: float = None):
        """
        Register feedback from FreqTrade for a trading signal.
        
        Args:
            symbol: Trading pair symbol
            signal_type: Signal type
            result: Whether the signal was successful
            profit_pct: Profit percentage
        """
        if self.market_data_fetcher:
            try:
                # Create signal data
                signal = {
                    "symbol": symbol,
                    "type": signal_type,
                    "strength": 0.5,
                    "confidence": 0.5,
                    "timestamp": time.time()
                }
                
                # Add profit data if available
                if profit_pct is not None:
                    signal["profit_pct"] = profit_pct
                    
                    # Adjust strength based on profit
                    if profit_pct > 5.0:
                        signal["strength"] = 0.9
                    elif profit_pct > 2.0:
                        signal["strength"] = 0.7
                    elif profit_pct > 0.0:
                        signal["strength"] = 0.6
                    elif profit_pct > -2.0:
                        signal["strength"] = 0.4
                    else:
                        signal["strength"] = 0.2
                
                # Register feedback
                self.market_data_fetcher.register_signal_feedback(signal, result)
                
                self.logger.info(
                    f"Registered feedback for {symbol}: {signal_type}, "
                    f"result={result}, profit={profit_pct}%"
                )
                
            except Exception as e:
                self.logger.error(f"Error registering feedback: {e}")
    
    def analyze_market(self, symbols: Optional[List[str]] = None):
        """
        Perform market analysis on specified symbols or active pairs.
        
        Args:
            symbols: List of symbols to analyze (None for active pairs)
        """
        if self.market_data_fetcher:
            try:
                self.market_data_fetcher.analyze_all_pairs(symbols)
                self.logger.info(f"Analyzed markets: {len(symbols) if symbols else 'all active pairs'}")
            except Exception as e:
                self.logger.error(f"Error analyzing market: {e}")
    
    def get_profitable_pairs(self, limit: int = 10) -> List[Tuple[str, float]]:
        """
        Get most profitable trading pairs based on analysis.
        
        Args:
            limit: Maximum number of pairs to return
            
        Returns:
            List of (symbol, score) tuples
        """
        if self.market_data_fetcher:
            try:
                return self.market_data_fetcher.get_pair_rankings(limit)
            except Exception as e:
                self.logger.error(f"Error getting profitable pairs: {e}")
        
        return []
    
    def export_freqtrade_data(self, pairlist_path: Optional[str] = None, 
                            metadata_path: Optional[str] = None):
        """
        Export data for FreqTrade.
        
        Args:
            pairlist_path: Path for pairlist file
            metadata_path: Path for metadata file
        """
        if self.market_data_fetcher:
            try:
                # Get paths from config if not provided
                freqtrade_config = self.config_manager.get_config("freqtrade")
                
                if pairlist_path is None:
                    pairlist_path = os.path.expanduser(freqtrade_config.get("pairlist_path", ""))
                else:
                    pairlist_path = os.path.expanduser(pairlist_path)
                    
                if metadata_path is None:
                    metadata_path = os.path.expanduser(freqtrade_config.get("metadata_path", ""))
                else:
                    metadata_path = os.path.expanduser(metadata_path)
                
                # Export pairlist
                if pairlist_path:
                    result = self.market_data_fetcher.export_to_freqtrade_pairlist(
                        limit=freqtrade_config.get("max_active_pairs", 50),
                        quote_currencies=freqtrade_config.get("quote_currencies", ["USDT"]),
                        filename=pairlist_path
                    )
                    
                    self.logger.info(f"Exported {len(result['pairs'])} pairs to {pairlist_path}")
                
                # Export metadata
                if metadata_path:
                    result = self.market_data_fetcher.export_extended_freqtrade_data(
                        limit=freqtrade_config.get("max_active_pairs", 50),
                        filename=metadata_path
                    )
                    
                    self.logger.info(f"Exported metadata for {len(result['pairs'])} pairs to {metadata_path}")
                    
            except Exception as e:
                self.logger.error(f"Error exporting FreqTrade data: {e}")
    
    def stop(self):
        """Stop background threads and clean up resources."""
        self.logger.info("Stopping CDFA FreqTrade Integration...")
        self.running = False
        
        # Stop components
        if self.market_data_fetcher:
            self.market_data_fetcher.stop()
            
        if self.visualizer:
            self.visualizer.stop()
        
        # Wait for threads to terminate
        if self.export_thread and self.export_thread.is_alive():
            self.export_thread.join(timeout=5.0)
            
        if self.dashboard_thread and self.dashboard_thread.is_alive():
            self.dashboard_thread.join(timeout=5.0)
            
        self.logger.info("CDFA FreqTrade Integration stopped")
    
    def __del__(self):
        """Destructor to ensure clean shutdown."""
        if hasattr(self, 'running') and self.running:
            self.stop()

# FreqTrade Strategy with CDFA Integration
class CDFAFreqTradeStrategy:
    """
    Utility for integrating CDFA with a FreqTrade strategy.
    
    This class is intended to be used within a FreqTrade strategy to leverage
    CDFA's analysis capabilities, especially the metadata from the extended
    pairlist export.
    """
    
    def __init__(self, metadata_path: Optional[str] = None):
        """
        Initialize integration.
        
        Args:
            metadata_path: Path to CDFA metadata file
        """
        self.logger = logging.getLogger(f"{__name__}.CDFAFreqTradeStrategy")
        
        # Default metadata path
        if metadata_path is None:
            metadata_path = os.path.expanduser("~/freqtrade/user_data/data/cdfa_metadata.json")
        else:
            metadata_path = os.path.expanduser(metadata_path)
            
        self.metadata_path = metadata_path
        
        # Metadata cache
        self.metadata = {}
        self.last_reload = 0
        self.reload_interval = 300  # 5 minutes
        
        # Load initial metadata
        self.reload_metadata()
        
        self.logger.info("CDFA FreqTrade Strategy initialized")
    
    def reload_metadata(self, force: bool = False) -> bool:
        """
        Reload metadata from file if needed.
        
        Args:
            force: Force reload regardless of time
            
        Returns:
            Success flag
        """
        current_time = time.time()
        
        # Check if reload is needed
        if not force and current_time - self.last_reload < self.reload_interval:
            return True
            
        try:
            if os.path.exists(self.metadata_path):
                with open(self.metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                    
                self.last_reload = current_time
                
                self.logger.info(f"Reloaded metadata from {self.metadata_path}")
                return True
            else:
                self.logger.warning(f"Metadata file not found: {self.metadata_path}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error reloading metadata: {e}")
            return False
    
    def get_pair_metadata(self, pair: str) -> Dict[str, Any]:
        """
        Get metadata for a specific pair.
        
        Args:
            pair: Trading pair
            
        Returns:
            Metadata dictionary
        """
        # Reload if needed
        self.reload_metadata()
        
        # Get metadata
        if "metadata" in self.metadata and pair in self.metadata["metadata"]:
            return self.metadata["metadata"][pair]
        
        return {}
    
    def get_pair_regime(self, pair: str) -> Dict[str, Any]:
        """
        Get regime data for a specific pair.
        
        Args:
            pair: Trading pair
            
        Returns:
            Regime data dictionary
        """
        # Reload if needed
        self.reload_metadata()
        
        # Get regime data
        if "regime_data" in self.metadata and pair in self.metadata["regime_data"]:
            return self.metadata["regime_data"][pair]
        
        return {}
    
    def get_pair_score(self, pair: str) -> float:
        """
        Get opportunity score for a specific pair.
        
        Args:
            pair: Trading pair
            
        Returns:
            Opportunity score (0-100)
        """
        # Reload if needed
        self.reload_metadata()
        
        # Get score
        if "opportunity_scores" in self.metadata and pair in self.metadata["opportunity_scores"]:
            return self.metadata["opportunity_scores"][pair]
        
        return 50.0  # Default score
    
    def is_pair_tradable(self, pair: str, min_score: float = 60.0, 
                       check_regime: bool = True) -> bool:
        """
        Check if a pair is tradable based on CDFA analysis.
        
        Args:
            pair: Trading pair
            min_score: Minimum opportunity score
            check_regime: Whether to check regime
            
        Returns:
            Whether the pair is tradable
        """
        # Get score
        score = self.get_pair_score(pair)
        
        # Check score
        if score < min_score:
            return False
            
        # Check regime if needed
        if check_regime:
            regime_data = self.get_pair_regime(pair)
            
            if regime_data:
                regime = regime_data.get("regime", "unknown")
                
                # Only trade in favorable regimes
                if regime.lower() in ["bearish", "distribution"]:
                    return False
                    
                # Check confidence
                confidence = regime_data.get("confidence", 0.0)
                if confidence < 0.5:
                    return False
        
        return True
    
    def adjust_position_size(self, pair: str, default_size: float, 
                           min_size: float = 0.1, max_size: float = 1.0) -> float:
        """
        Adjust position size based on CDFA analysis.
        
        Args:
            pair: Trading pair
            default_size: Default position size
            min_size: Minimum position size
            max_size: Maximum position size
            
        Returns:
            Adjusted position size
        """
        # Get score and regime data
        score = self.get_pair_score(pair)
        regime_data = self.get_pair_regime(pair)
        metadata = self.get_pair_metadata(pair)
        
        # Base size on score
        size_factor = score / 100.0
        
        # Adjust based on regime
        if regime_data:
            regime = regime_data.get("regime", "unknown")
            confidence = regime_data.get("confidence", 0.5)
            
            if regime.lower() == "bullish":
                size_factor *= 1.2
            elif regime.lower() == "accumulation":
                size_factor *= 1.1
            elif regime.lower() == "distribution":
                size_factor *= 0.8
            elif regime.lower() == "bearish":
                size_factor *= 0.7
                
            # Adjust based on confidence
            size_factor *= (0.5 + confidence * 0.5)
        
        # Adjust based on volatility if available
        if metadata and "volatility" in metadata:
            volatility = metadata["volatility"]
            
            # Reduce size for high volatility
            if volatility > 0.5:
                size_factor /= (volatility * 2)
        
        # Calculate adjusted size
        adjusted_size = default_size * size_factor
        
        # Ensure within limits
        adjusted_size = max(min_size, min(adjusted_size, max_size))
        
        return adjusted_size
    
    def get_market_state(self) -> Dict[str, Any]:
        """
        Get overall market state.
        
        Returns:
            Market state dictionary
        """
        # Reload if needed
        self.reload_metadata()
        
        # Get market state
        if "market_state" in self.metadata:
            return self.metadata["market_state"]
        
        return {}
    
    def log_trade_result(self, pair: str, profit_pct: float, trade_duration: int):
        """
        Log trade result for feedback.
        
        Args:
            pair: Trading pair
            profit_pct: Profit percentage
            trade_duration: Trade duration in minutes
        """
        success = profit_pct > 0
        
        self.logger.info(
            f"Trade result for {pair}: profit={profit_pct:.2f}%, "
            f"duration={trade_duration} min, success={success}"
        )
        
        # TODO: Implement feedback mechanism to report to CDFA

# Command-line interface
def main():
    """Command-line interface for CDFA FreqTrade Integration."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description="CDFA FreqTrade Integration")
    
    # Add commands
    subparsers = parser.add_subparsers(dest="command", help="Command")
    
    # Start command
    start_parser = subparsers.add_parser("start", help="Start integration")
    start_parser.add_argument("--config-dir", help="Configuration directory")
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export data")
    export_parser.add_argument("--pairlist", help="Path for pairlist file")
    export_parser.add_argument("--metadata", help="Path for metadata file")
    
    # Config command
    config_parser = subparsers.add_parser("config", help="Manage configuration")
    config_parser.add_argument("--export", help="Export configuration to file")
    config_parser.add_argument("--import", dest="import_file", help="Import configuration from file")
    
    # Dashboard command
    dashboard_parser = subparsers.add_parser("dashboard", help="Manage dashboard")
    dashboard_parser.add_argument("--port", type=int, help="Dashboard port")
    dashboard_parser.add_argument("--export", help="Export dashboard to file")
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze market")
    analyze_parser.add_argument("--symbols", help="Symbols to analyze (comma-separated)")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    logger = logging.getLogger("cdfa_freqtrade")
    
    # Handle commands
    try:
        if args.command == "start":
            # Create config manager
            config_dir = args.config_dir or "~/.cdfa"
            config_manager = CDFAConfigManager(config_dir=config_dir)
            
            # Create integration
            integration = CDFAFreqTradeIntegration(config_manager=config_manager)
            
            # Handle signals
            def signal_handler(sig, frame):
                logger.info("Stopping integration...")
                integration.stop()
                logger.info("Integration stopped")
                
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
            
            logger.info("CDFA FreqTrade Integration started")
            logger.info("Press Ctrl+C to stop")
            
            # Keep running
            while True:
                time.sleep(1)
                
        elif args.command == "export":
            # Create config manager
            config_manager = CDFAConfigManager()
            
            # Create integration
            integration = CDFAFreqTradeIntegration(config_manager=config_manager)
            
            # Export data
            integration.export_freqtrade_data(
                pairlist_path=args.pairlist,
                metadata_path=args.metadata
            )
            
            logger.info("Export completed")
            
        elif args.command == "config":
            # Create config manager
            config_manager = CDFAConfigManager()
            
            if args.export:
                # Export configuration
                path = config_manager.export_to_json(args.export)
                logger.info(f"Configuration exported to {path}")
                
            elif args.import_file:
                # Import configuration
                config_manager.import_from_json(args.import_file)
                logger.info(f"Configuration imported from {args.import_file}")
                
            else:
                # Show configuration
                frontend_config = config_manager.get_frontend_config()
                print(json.dumps(frontend_config, indent=2))
                
        elif args.command == "dashboard":
            # Create config manager
            config_manager = CDFAConfigManager()
            
            # Create integration
            integration = CDFAFreqTradeIntegration(config_manager=config_manager)
            
            if args.port:
                # Update port
                config_manager.update_config("dashboard", "port", args.port)
                
            if args.export:
                # Export dashboard
                if integration.visualizer:
                    path = integration.visualizer.save_dashboard(
                        filename=args.export,
                        title="CDFA Market Intelligence Dashboard"
                    )
                    logger.info(f"Dashboard exported to {path}")
                else:
                    logger.error("Visualizer not available")
                    
            else:
                # Show dashboard info
                dashboard_config = config_manager.get_config("dashboard")
                port = dashboard_config.get("port", 8501)
                logger.info(f"Dashboard available at http://localhost:{port}")
                
                # Keep running until Ctrl+C
                logger.info("Press Ctrl+C to stop")
                
                # Handle signals
                def signal_handler(sig, frame):
                    logger.info("Stopping integration...")
                    integration.stop()
                    logger.info("Integration stopped")
                    
                signal.signal(signal.SIGINT, signal_handler)
                signal.signal(signal.SIGTERM, signal_handler)
                
                # Keep running
                while True:
                    time.sleep(1)
                    
        elif args.command == "analyze":
            # Create config manager
            config_manager = CDFAConfigManager()
            
            # Create integration
            integration = CDFAFreqTradeIntegration(config_manager=config_manager)
            
            # Analyze market
            symbols = None
            if args.symbols:
                symbols = args.symbols.split(",")
                
            integration.analyze_market(symbols)
            
            # Show profitable pairs
            profitable_pairs = integration.get_profitable_pairs(limit=20)
            
            if profitable_pairs:
                logger.info("Profitable pairs:")
                for symbol, score in profitable_pairs:
                    logger.info(f"{symbol}: {score:.2f}")
            else:
                logger.info("No profitable pairs found")
                
        else:
            # Show help
            parser.print_help()
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        
    finally:
        # Clean up
        if "integration" in locals():
            integration.stop()

if __name__ == "__main__":
    main()
