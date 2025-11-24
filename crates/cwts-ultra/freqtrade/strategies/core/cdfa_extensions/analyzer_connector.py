#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyzer Connector Module

Connects existing analyzers and detectors to real-time market data,
providing a unified interface for the frontend visualization system.

Created on May 20, 2025
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple, Set, Callable
import time
import os
import importlib
import inspect
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import from local modules
from realtime_market_analyzer import RealtimeMarketAnalyzer, PairMetadata, MarketRegime, OpportunityScore

class AnalyzerConnector:
    """
    Connector class that integrates existing analyzers and detectors 
    with real-time market data for frontend visualization.
    
    This class:
    1. Dynamically loads analyzers and detectors from their respective directories
    2. Connects them to the real-time market data
    3. Provides a unified interface for the frontend
    4. Handles parallel execution for improved performance
    """
    
    def __init__(self, 
                 market_data_analyzer: Optional[RealtimeMarketAnalyzer] = None,
                 analyzers_dir: str = 'cdfa_extensions/analyzers',
                 detectors_dir: str = 'cdfa_extensions/detectors',
                 max_workers: int = 4,
                 log_level: str = "INFO"):
        """
        Initialize analyzer connector.
        
        Args:
            market_data_analyzer: RealtimeMarketAnalyzer instance or None to create new
            analyzers_dir: Directory containing analyzer modules
            detectors_dir: Directory containing detector modules
            max_workers: Maximum number of parallel workers
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
        
        # Initialize market data analyzer
        self.market_data_analyzer = market_data_analyzer or RealtimeMarketAnalyzer(log_level=log_level)
        
        # Directories
        self.analyzers_dir = analyzers_dir
        self.detectors_dir = detectors_dir
        
        # Component storage
        self.analyzers = {}
        self.detectors = {}
        
        # Parallel processing
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Status flags
        self.is_initialized = False
        self.running = False
        self.update_thread = None
        
        # Load components
        self._load_analyzers()
        self._load_detectors()
        
        # Register with market data analyzer
        self._register_analyzers()
        
        self.is_initialized = True
        self.logger.info("AnalyzerConnector initialized")
    
    def _load_analyzers(self):
        """Load analyzer modules from directory"""
        self.logger.info(f"Loading analyzers from {self.analyzers_dir}")
        
        try:
            # Get analyzer module files
            analyzer_files = []
            for root, _, files in os.walk(self.analyzers_dir.replace('/', os.sep)):
                for file in files:
                    if file.endswith('_analyzer.py') and not file.startswith('__'):
                        analyzer_files.append(os.path.join(root, file))
            
            # Import modules
            for file_path in analyzer_files:
                try:
                    # Convert file path to module path
                    rel_path = os.path.relpath(file_path)
                    module_path = rel_path.replace(os.sep, '.').replace('.py', '')
                    
                    # Import module
                    module = importlib.import_module(module_path)
                    
                    # Find analyzer classes
                    for name, obj in inspect.getmembers(module):
                        if (inspect.isclass(obj) and 
                            name.endswith('Analyzer') and 
                            hasattr(obj, 'analyze')):
                            
                            # Create instance
                            analyzer_name = name.lower()
                            analyzer = obj()
                            
                            # Store in dict
                            self.analyzers[analyzer_name] = analyzer
                            self.logger.info(f"Loaded analyzer: {analyzer_name}")
                
                except Exception as e:
                    self.logger.error(f"Error loading analyzer from {file_path}: {e}")
            
            self.logger.info(f"Loaded {len(self.analyzers)} analyzers")
            
        except Exception as e:
            self.logger.error(f"Error loading analyzers: {e}")
    
    def _load_detectors(self):
        """Load detector modules from directory"""
        self.logger.info(f"Loading detectors from {self.detectors_dir}")
        
        try:
            # Get detector module files
            detector_files = []
            for root, _, files in os.walk(self.detectors_dir.replace('/', os.sep)):
                for file in files:
                    if (file.endswith('_detector.py') or file.endswith('_recognizer.py')) and not file.startswith('__'):
                        detector_files.append(os.path.join(root, file))
            
            # Import modules
            for file_path in detector_files:
                try:
                    # Convert file path to module path
                    rel_path = os.path.relpath(file_path)
                    module_path = rel_path.replace(os.sep, '.').replace('.py', '')
                    
                    # Import module
                    module = importlib.import_module(module_path)
                    
                    # Find detector classes
                    for name, obj in inspect.getmembers(module):
                        if (inspect.isclass(obj) and 
                            (name.endswith('Detector') or name.endswith('Recognizer')) and 
                            hasattr(obj, 'detect')):
                            
                            # Create instance
                            detector_name = name.lower()
                            detector = obj()
                            
                            # Store in dict
                            self.detectors[detector_name] = detector
                            self.logger.info(f"Loaded detector: {detector_name}")
                
                except Exception as e:
                    self.logger.error(f"Error loading detector from {file_path}: {e}")
            
            self.logger.info(f"Loaded {len(self.detectors)} detectors")
            
        except Exception as e:
            self.logger.error(f"Error loading detectors: {e}")
    
    def _register_analyzers(self):
        """Register analyzers with market data analyzer"""
        for name, analyzer in self.analyzers.items():
            self.market_data_analyzer.add_analyzer(name, analyzer)
        
        for name, detector in self.detectors.items():
            # Create adapter for detector
            detector_adapter = self._create_detector_adapter(detector)
            self.market_data_analyzer.add_analyzer(name, detector_adapter)
    
    def _create_detector_adapter(self, detector):
        """
        Create adapter for detector to match analyzer interface.
        
        Args:
            detector: Detector instance
            
        Returns:
            Adapter object with analyze method
        """
        class DetectorAdapter:
            def __init__(self, detector):
                self.detector = detector
            
            def analyze(self, dataframe, metadata):
                try:
                    # Call detect method
                    signals = self.detector.detect(dataframe, metadata)
                    
                    if signals:
                        return {
                            'signals': signals,
                            'score': self._calculate_score(signals)
                        }
                    return None
                except Exception as e:
                    logging.error(f"Error in detector adapter: {e}")
                    return None
            
            def _calculate_score(self, signals):
                """Calculate score from signals (0-1)"""
                if not signals:
                    return 0.0
                
                # Calculate based on strength and recency
                current_time = time.time()
                total_score = 0.0
                
                for signal in signals:
                    # Get signal strength
                    strength = signal.get('strength', 0.5)
                    
                    # Adjust by recency (decay over time)
                    timestamp = signal.get('timestamp', current_time)
                    age_days = (current_time - timestamp) / 86400
                    recency_factor = max(0.0, 1.0 - 0.1 * age_days)  # 10% decay per day
                    
                    # Combine
                    signal_score = strength * recency_factor
                    total_score += signal_score
                
                # Normalize to 0-1 range
                return min(1.0, total_score / len(signals))
        
        return DetectorAdapter(detector)
    
    def start_auto_updates(self, interval_seconds: int = 300):
        """
        Start automatic updates in background thread.
        
        Args:
            interval_seconds: Update interval in seconds
        """
        if self.running:
            self.logger.warning("Auto-updates already running")
            return
        
        self.running = True
        
        def update_loop():
            while self.running:
                try:
                    self.update_all()
                    
                    # Sleep until next update
                    for _ in range(interval_seconds):
                        if not self.running:
                            break
                        time.sleep(1)
                        
                except Exception as e:
                    self.logger.error(f"Error in update loop: {e}")
                    time.sleep(10)  # Sleep longer on error
        
        # Start thread
        self.update_thread = threading.Thread(
            target=update_loop,
            daemon=True,
            name="AnalyzerConnector-Update"
        )
        self.update_thread.start()
        
        self.logger.info(f"Auto-updates started with interval {interval_seconds}s")
    
    def stop_auto_updates(self):
        """Stop automatic updates"""
        if not self.running:
            return
        
        self.running = False
        
        if self.update_thread and self.update_thread.is_alive():
            self.update_thread.join(timeout=5.0)
        
        self.logger.info("Auto-updates stopped")
    
    def update_all(self):
        """Update all market data and run analysis"""
        if not self.is_initialized:
            self.logger.error("AnalyzerConnector not initialized")
            return False
        
        try:
            # Update market data
            success = self.market_data_analyzer.update_market_data()
            
            if not success:
                self.logger.warning("Market data update failed")
                return False
            
            self.logger.info("All market data updated")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating all data: {e}")
            return False
    
    def get_active_pairs(self) -> List[str]:
        """
        Get list of active trading pairs.
        
        Returns:
            List of symbol strings
        """
        return self.market_data_analyzer.get_active_pairs()
    
    def get_pair_rankings(self, limit: Optional[int] = None) -> List[Tuple[str, float]]:
        """
        Get ranked list of trading pairs by opportunity score.
        
        Args:
            limit: Maximum number of pairs to return
            
        Returns:
            List of (symbol, score) tuples
        """
        return self.market_data_analyzer.get_pair_rankings(limit=limit)
    
    def get_pair_metadata(self, symbol: str) -> Optional[PairMetadata]:
        """
        Get metadata for a specific trading pair.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            PairMetadata object or None if not found
        """
        return self.market_data_analyzer.get_pair_metadata(symbol)
    
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
        return self.market_data_analyzer.fetch_data(symbols, timeframe, lookback)
    
    def run_analyzer(self, analyzer_name: str, symbol: str, timeframe: str = "1d", 
                  lookback: str = "30d") -> Optional[Dict[str, Any]]:
        """
        Run specific analyzer on a symbol.
        
        Args:
            analyzer_name: Name of analyzer to run
            symbol: Trading pair symbol
            timeframe: Timeframe for data
            lookback: Lookback period
            
        Returns:
            Analysis result or None if error
        """
        try:
            # Get analyzer
            analyzer = None
            if analyzer_name in self.analyzers:
                analyzer = self.analyzers[analyzer_name]
            elif analyzer_name in self.detectors:
                analyzer = self._create_detector_adapter(self.detectors[analyzer_name])
            
            if not analyzer:
                self.logger.error(f"Analyzer not found: {analyzer_name}")
                return None
            
            # Fetch data
            data_dict = self.fetch_data([symbol], timeframe, lookback)
            
            if not data_dict or symbol not in data_dict:
                self.logger.warning(f"No data available for {symbol}")
                return None
            
            # Get metadata
            meta = self.get_pair_metadata(symbol)
            
            if not meta:
                self.logger.warning(f"No metadata available for {symbol}")
                metadata = {'symbol': symbol}
            else:
                metadata = {
                    'symbol': symbol,
                    'exchange': meta.exchange,
                    'base': meta.base_currency,
                    'quote': meta.quote_currency
                }
            
            # Run analysis
            result = analyzer.analyze(data_dict[symbol], metadata)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error running analyzer {analyzer_name} on {symbol}: {e}")
            return None
    
    def run_all_analyzers(self, symbol: str, timeframe: str = "1d", 
                       lookback: str = "30d") -> Dict[str, Dict[str, Any]]:
        """
        Run all analyzers on a symbol.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe for data
            lookback: Lookback period
            
        Returns:
            Dictionary of analyzer_name -> result
        """
        results = {}
        
        try:
            # Fetch data
            data_dict = self.fetch_data([symbol], timeframe, lookback)
            
            if not data_dict or symbol not in data_dict:
                self.logger.warning(f"No data available for {symbol}")
                return results
            
            df = data_dict[symbol]
            
            # Get metadata
            meta = self.get_pair_metadata(symbol)
            
            if not meta:
                self.logger.warning(f"No metadata available for {symbol}")
                metadata = {'symbol': symbol}
            else:
                metadata = {
                    'symbol': symbol,
                    'exchange': meta.exchange,
                    'base': meta.base_currency,
                    'quote': meta.quote_currency
                }
            
            # Run analyzers in parallel
            futures = {}
            
            # Add analyzer tasks
            for name, analyzer in self.analyzers.items():
                future = self.executor.submit(self._run_analysis_task, analyzer, df, metadata, name)
                futures[future] = name
            
            # Add detector tasks
            for name, detector in self.detectors.items():
                adapter = self._create_detector_adapter(detector)
                future = self.executor.submit(self._run_analysis_task, adapter, df, metadata, name)
                futures[future] = name
            
            # Collect results
            for future in as_completed(futures):
                name = futures[future]
                try:
                    result = future.result()
                    if result:
                        results[name] = result
                except Exception as e:
                    self.logger.error(f"Error in analyzer {name}: {e}")
            
            self.logger.info(f"Ran all analyzers on {symbol} - got {len(results)} results")
            return results
            
        except Exception as e:
            self.logger.error(f"Error running all analyzers on {symbol}: {e}")
            return results
    
    def _run_analysis_task(self, analyzer, dataframe, metadata, name):
        """Helper method for parallel analysis execution"""
        try:
            return analyzer.analyze(dataframe, metadata)
        except Exception as e:
            self.logger.error(f"Error in analyzer {name}: {e}")
            return None
    
    def get_available_analyzers(self) -> List[str]:
        """
        Get list of available analyzers.
        
        Returns:
            List of analyzer names
        """
        return list(self.analyzers.keys())
    
    def get_available_detectors(self) -> List[str]:
        """
        Get list of available detectors.
        
        Returns:
            List of detector names
        """
        return list(self.detectors.keys())
    
    def stop(self):
        """Stop background threads and clean up resources"""
        self.stop_auto_updates()
        
        if hasattr(self, 'executor') and self.executor:
            self.executor.shutdown(wait=False)
        
        if hasattr(self, 'market_data_analyzer') and self.market_data_analyzer:
            self.market_data_analyzer.stop()
        
        self.logger.info("AnalyzerConnector stopped")