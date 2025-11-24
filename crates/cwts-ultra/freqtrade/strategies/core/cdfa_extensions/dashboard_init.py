#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dashboard Initialization Module

Provides a simple interface for setting up and launching the CDFA
visualization dashboard with real market data.

Created on May 20, 2025
"""

import logging
import os
import argparse
import json
import time
from typing import Dict, Any, Optional, List

# Import local modules
from cdfa_extensions.realtime_market_analyzer import RealtimeMarketAnalyzer
from cdfa_extensions.analyzer_connector import AnalyzerConnector
from cdfa_extensions.visualization_adapter import VisualizationAdapter

class DashboardInitializer:
    """
    Initializes and manages the CDFA visualization dashboard.
    
    This class:
    1. Handles configuration loading and validation
    2. Sets up the required components (analyzers, connectors, visualizers)
    3. Provides a simple interface for launching the dashboard
    4. Implements command-line argument parsing for dashboard launching
    """
    
    def __init__(self, config_path: Optional[str] = None, log_level: str = "INFO"):
        """
        Initialize dashboard initializer.
        
        Args:
            config_path: Path to configuration file or None to use default
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
        
        # Load configuration
        self.config = self._load_configuration(config_path)
        
        # Set up components
        self.market_analyzer = None
        self.analyzer_connector = None
        self.visualization_adapter = None
        
        self.logger.info("DashboardInitializer initialized")
    
    def _load_configuration(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file or None to use default
            
        Returns:
            Configuration dictionary
        """
        # Default configuration
        default_config = {
            "exchange": {
                "name": "binance",
                "api_key": "",
                "api_secret": "",
                "use_testnet": True
            },
            "market_data": {
                "base_currencies": ["BTC", "ETH", "SOL", "AVAX", "ADA"],
                "quote_currencies": ["USDT", "BUSD", "USDC"],
                "timeframes": ["1h", "4h", "1d"],
                "default_timeframe": "1d",
                "update_interval": 300,
                "cache_ttl": 3600,
                "max_pairs": 100
            },
            "visualization": {
                "auto_update": True,
                "default_width": 1000,
                "default_height": 800,
                "default_cmap": "fire",
                "show_toolbar": True,
                "port": 5006
            },
            "system": {
                "cache_dir": "~/cdfa_cache",
                "log_level": "INFO",
                "max_workers": 4
            }
        }
        
        # If no config path provided, use default
        if not config_path:
            self.logger.info("Using default configuration")
            return default_config
        
        # Load configuration from file
        try:
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)
            
            # Merge with default configuration
            config = default_config.copy()
            self._merge_configs(config, loaded_config)
            
            self.logger.info(f"Loaded configuration from {config_path}")
            return config
            
        except Exception as e:
            self.logger.error(f"Error loading configuration from {config_path}: {e}")
            self.logger.info("Using default configuration")
            return default_config
    
    def _merge_configs(self, base_config: Dict[str, Any], override_config: Dict[str, Any]):
        """
        Recursively merge configurations.
        
        Args:
            base_config: Base configuration to update
            override_config: Override configuration
        """
        for key, value in override_config.items():
            if key in base_config and isinstance(base_config[key], dict) and isinstance(value, dict):
                self._merge_configs(base_config[key], value)
            else:
                base_config[key] = value
    
    def initialize_components(self):
        """Initialize all required components"""
        try:
            # Create cache directory
            cache_dir = os.path.expanduser(self.config["system"]["cache_dir"])
            os.makedirs(cache_dir, exist_ok=True)
            
            # Initialize market analyzer
            market_config = {
                "exchange": self.config["exchange"]["name"],
                "api_key": self.config["exchange"]["api_key"],
                "api_secret": self.config["exchange"]["api_secret"],
                "base_currencies": self.config["market_data"]["base_currencies"],
                "quote_currencies": self.config["market_data"]["quote_currencies"],
                "timeframes": self.config["market_data"]["timeframes"],
                "default_timeframe": self.config["market_data"]["default_timeframe"],
                "update_interval": self.config["market_data"]["update_interval"],
                "cache_ttl": self.config["market_data"]["cache_ttl"],
                "max_pairs": self.config["market_data"]["max_pairs"],
                "auto_update": self.config["visualization"]["auto_update"],
                "use_testnet": self.config["exchange"]["use_testnet"]
            }
            
            self.market_analyzer = RealtimeMarketAnalyzer(
                config=market_config,
                cache_dir=os.path.join(cache_dir, "market_data"),
                log_level=self.config["system"]["log_level"]
            )
            
            # Initialize analyzer connector
            self.analyzer_connector = AnalyzerConnector(
                market_data_analyzer=self.market_analyzer,
                max_workers=self.config["system"]["max_workers"],
                log_level=self.config["system"]["log_level"]
            )
            
            # Initialize visualization adapter
            viz_config = {
                "default_width": self.config["visualization"]["default_width"],
                "default_height": self.config["visualization"]["default_height"],
                "default_cmap": self.config["visualization"]["default_cmap"],
                "show_toolbar": self.config["visualization"]["show_toolbar"],
                "auto_update_interval": self.config["market_data"]["update_interval"] if self.config["visualization"]["auto_update"] else None
            }
            
            self.visualization_adapter = VisualizationAdapter(
                analyzer_connector=self.analyzer_connector,
                cache_dir=os.path.join(cache_dir, "visualization"),
                cache_ttl=self.config["market_data"]["cache_ttl"],
                auto_update=self.config["visualization"]["auto_update"],
                log_level=self.config["system"]["log_level"]
            )
            
            # Start auto-updates if configured
            if self.config["visualization"]["auto_update"]:
                self.analyzer_connector.start_auto_updates(
                    interval_seconds=self.config["market_data"]["update_interval"]
                )
            
            self.logger.info("All components initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing components: {e}")
            return False
    
    def create_dashboard(self):
        """
        Create dashboard with visualizations.
        
        Returns:
            Dashboard object
        """
        if not self.visualization_adapter:
            if not self.initialize_components():
                self.logger.error("Failed to initialize components")
                return None
        
        try:
            # Create dashboard
            dashboard = self.visualization_adapter.create_dashboard(
                width=self.config["visualization"]["default_width"],
                height=self.config["visualization"]["default_height"]
            )
            
            self.logger.info("Dashboard created")
            return dashboard
            
        except Exception as e:
            self.logger.error(f"Error creating dashboard: {e}")
            return None
    
    def serve_dashboard(self, port: Optional[int] = None):
        """
        Serve dashboard using Panel server.
        
        Args:
            port: Server port or None to use configured port
            
        Returns:
            Server instance
        """
        if not self.visualization_adapter:
            if not self.initialize_components():
                self.logger.error("Failed to initialize components")
                return None
        
        try:
            # Get port from config if not provided
            if port is None:
                port = self.config["visualization"]["port"]
            
            # Serve dashboard
            server = self.visualization_adapter.serve_dashboard(port=port)
            
            self.logger.info(f"Dashboard served at http://localhost:{port}")
            return server
            
        except Exception as e:
            self.logger.error(f"Error serving dashboard: {e}")
            return None
    
    def save_dashboard(self, filename: Optional[str] = None):
        """
        Save dashboard to HTML file.
        
        Args:
            filename: Output filename or None to use default
            
        Returns:
            Path to saved file
        """
        if not self.visualization_adapter:
            if not self.initialize_components():
                self.logger.error("Failed to initialize components")
                return None
        
        try:
            # Use default filename if not provided
            if filename is None:
                filename = os.path.expanduser("~/cdfa_dashboard.html")
            
            # Save dashboard
            path = self.visualization_adapter.save_dashboard(
                filename=filename,
                title="CDFA Market Intelligence Dashboard"
            )
            
            self.logger.info(f"Dashboard saved to {path}")
            return path
            
        except Exception as e:
            self.logger.error(f"Error saving dashboard: {e}")
            return None
    
    def create_tradingview_dashboard(self, symbols: Optional[List[str]] = None):
        """
        Create TradingView-style dashboard.
        
        Args:
            symbols: List of symbols to display or None for top pairs
            
        Returns:
            TradingView-style dashboard
        """
        if not self.visualization_adapter:
            if not self.initialize_components():
                self.logger.error("Failed to initialize components")
                return None
        
        try:
            # Get default timeframe from config
            timeframe = self.config["market_data"]["default_timeframe"]
            
            # Create dashboard
            dashboard = self.visualization_adapter.create_tradingview_dashboard(
                symbols=symbols,
                timeframe=timeframe
            )
            
            self.logger.info("TradingView dashboard created")
            return dashboard
            
        except Exception as e:
            self.logger.error(f"Error creating TradingView dashboard: {e}")
            return None
    
    def create_market_dashboard(self):
        """
        Create comprehensive market dashboard.
        
        Returns:
            Market dashboard
        """
        if not self.visualization_adapter:
            if not self.initialize_components():
                self.logger.error("Failed to initialize components")
                return None
        
        try:
            # Create dashboard
            dashboard = self.visualization_adapter.create_market_dashboard()
            
            self.logger.info("Market dashboard created")
            return dashboard
            
        except Exception as e:
            self.logger.error(f"Error creating market dashboard: {e}")
            return None
    
    def update_dashboard(self):
        """Update dashboard data"""
        if not self.visualization_adapter:
            if not self.initialize_components():
                self.logger.error("Failed to initialize components")
                return False
        
        try:
            # Update data
            self.visualization_adapter.update_all()
            
            self.logger.info("Dashboard data updated")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating dashboard data: {e}")
            return False
    
    def stop(self):
        """Stop all components and clean up resources"""
        self.logger.info("Stopping DashboardInitializer...")
        
        if hasattr(self, 'visualization_adapter') and self.visualization_adapter:
            self.visualization_adapter.stop()
        
        if hasattr(self, 'analyzer_connector') and self.analyzer_connector:
            self.analyzer_connector.stop()
        
        if hasattr(self, 'market_analyzer') and self.market_analyzer:
            self.market_analyzer.stop()
        
        self.logger.info("DashboardInitializer stopped")

def main():
    """Main function for command-line usage"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='CDFA Dashboard Initializer')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--port', type=int, help='Server port')
    parser.add_argument('--save', type=str, help='Save dashboard to HTML file')
    parser.add_argument('--tradingview', action='store_true', help='Create TradingView-style dashboard')
    parser.add_argument('--market', action='store_true', help='Create market dashboard')
    parser.add_argument('--log-level', type=str, default='INFO', help='Logging level')
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create initializer
    initializer = DashboardInitializer(config_path=args.config, log_level=args.log_level)
    
    try:
        # Initialize components
        if not initializer.initialize_components():
            logging.error("Failed to initialize components")
            return 1
        
        # Update data
        initializer.update_dashboard()
        
        # Save dashboard if requested
        if args.save:
            initializer.save_dashboard(filename=args.save)
            return 0
        
        # Create and serve dashboard
        if args.tradingview:
            dashboard = initializer.create_tradingview_dashboard()
        elif args.market:
            dashboard = initializer.create_market_dashboard()
        else:
            dashboard = initializer.create_dashboard()
        
        # Serve dashboard
        server = initializer.serve_dashboard(port=args.port)
        
        # Keep process running
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logging.info("Interrupted by user")
        finally:
            initializer.stop()
        
        return 0
        
    except Exception as e:
        logging.error(f"Error: {e}")
        return 1
    finally:
        initializer.stop()

if __name__ == '__main__':
    exit(main())