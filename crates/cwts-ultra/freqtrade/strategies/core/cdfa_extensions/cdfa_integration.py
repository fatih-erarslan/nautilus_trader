#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CDFA Integration Module

Provides a unified interface to all CDFA extensions:
- Hardware-accelerated processing
- Cross-asset analysis
- Neuromorphic computing
- Advanced visualization
- Wavelet processing
- Pulsar communication
- PADS reporting

Author: Created on May 6, 2025
"""

import logging
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import warnings
import os
import threading
from collections import defaultdict
import tempfile
from datetime import datetime

# Import core extensions
from .advanced_cdfa import AdvancedCDFA, AdvancedCDFAConfig, SignalType, FusionType, NumbaDict, NumbaList
from .hw_acceleration import HardwareAccelerator
#from .hw_acceleration import HardwareAccelerator as NumbaAccelerator
from .cross_asset_analyzer import CrossAssetAnalyzer
from .advanced_visualization import VisualizationEngine, VisualizationType, OutputFormat
from .pulsar_connector import PulsarConnector, PulsarMessage
from .pads_reporter import PADSReporter, SignalType, PADSSignal, SignalTimeframe


# Import analyzers from the new subdirectory

try:
    from .wavelet_processor import WaveletAnalysisResult, WaveletDenoiseResult, WaveletFamily, WaveletProcessor, WaveletDecompResult
    WAVELET_AVAILABLE = True
except ImportError:
    WAVELET_AVAILABLE = False
    warnings.warn("Wavelet Module is not available")
    
try:
    from .neuromorphic_analyzer import NeuromorphicAnalyzer, NeuromorphicEngine
    NEUROMORPHIC_AVAILABLE = True
except ImportError:
    NEUROMORPHIC_AVAILABLE = False
    warnings.warn("Neuroporphic Module is not available")
    
try:
    from .mra_analyzer import MultiResolutionAnalyzer, MRADecomposition, MRAMode
    MRA_AVAILABLE = True
except ImportError:
    MRA_AVAILABLE = False
    warnings.warn("MRA Module is not available")
    
try:
    from .stdp_optimizer import STDPOptimizer, STDPMode, STDPParameters, STDPResult
    STDP_AVAILABLE = True
except ImportError:
    STDP_AVAILABLE = False
    warnings.warn("STDP Module is not available")
    
try:
    from .analyzers import antifragility_analyzer, fibonacci_analyzer, panarchy_analyzer, soc_analyzer
    from .analyzers import   ( AntifragilityAnalyzer, AntifragilityParameters,
                           FibonacciAnalyzer, FibonacciParameters, 
                           PanarchyAnalyzer, PanarchyParameters,
                           SOCAnalyzer, SOCParameters
        )
    ANALYZERS_AVAILABLE = True
except ImportError:
    ANALYZERS_AVAILABLE = False
    warnings.warn("Analyzers not available from cdfa_extensions/analyzers")


# Import detectors from the new subdirectory
try:
    from .detectors import fibonacci_pattern_detector, black_swan_detector, pattern_recognizer, whale_detector
    from .detectors import ( FibonacciPatternDetector, FibonacciPatternAnalyzer, 
                            FibonacciParameters,PatternConfig, PatternDetectionConfig, 
                            PatternPoint, PatternState, HarmonicPattern,
                           BlackSwanDetector, BlackSwanParameters, 
                           PatternRecognizer, PatternRecWindow,
                           WhaleDetector, WhaleParameters,
        )
    DETECTORS_AVAILABLE = True
except ImportError:
    DETECTORS_AVAILABLE = False
    warnings.warn("Detectors not available from cdfa_extensions/detectors")

# Import optional extensions

# Try to import enhanced CDFA
try:
    from enhanced_cdfa import CognitiveDiversityFusionAnalysis
    ENHANCED_CDFA_AVAILABLE = True
except ImportError:
    try:
        from cdfa import CognitiveDiversityFusionAnalysis
        ENHANCED_CDFA_AVAILABLE = True
    except ImportError:
        ENHANCED_CDFA_AVAILABLE = False
        warnings.warn("CDFA not available - integration will have limited functionality")

class CDFAIntegration:
    """
    Integration layer for all CDFA extensions.
    
    Provides a unified interface for the enhanced CDFA system with all
    extensions, handling coordination between components and providing
    high-level workflows for common use cases.
    """
    
    def _load_config_manager(self):
        """Lazy load the config manager to avoid circular imports"""
        from .cdfa_conf import CDFAConfigManager
        return CDFAConfigManager()
    
    def _load_market_data_fetcher(self):
        """Lazy load the market data fetcher to avoid circular imports"""
        from .adaptive_market_data_fetcher import AdaptiveMarketDataFetcher
        return AdaptiveMarketDataFetcher()
    
    def _load_cross_asset_analyzer(self):
        """Lazy load the cross asset analyzer to avoid circular imports"""
        from .cross_asset_analyzer import CrossAssetAnalyzer
        return CrossAssetAnalyzer()
    
    def _load_hardware_accelerator(self):
        """Lazy load the hardware accelerator to avoid circular imports"""
        from .hw_acceleration import HardwareAccelerator
        return HardwareAccelerator()
    
    def _load_visualizer(self):
        """Lazy load the visualizer to avoid circular imports"""
        from .holoviews_visualizer import HoloviewsVisualizer
        return HoloviewsVisualizer()
    
    def _load_pads(self):
        """Lazy load PADS to avoid circular imports"""
        from .pads import PulsarAdaptiveDataSystem
        return PulsarAdaptiveDataSystem()
    
    def __init__(self, config: Optional[Dict[str, Any]] = None,
               log_level: int = logging.INFO):
        """
        Initialize the CDFA integration layer.
        
        Args:
            config: Configuration dictionary
            log_level: Logging level
        """
        # Setup logging
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.setLevel(log_level)
        
        # Default configuration
        self.default_config = {
            # Hardware acceleration config
            "enable_gpu": True,
            "prefer_cuda": True,
            "device": None,  # Auto-detect
            
            # Cross-asset analysis config
            "default_correlation_method": "wavelet",
            "default_timeframe": "1d",
            "correlation_threshold": 0.5,
            
            # Visualization config
            "default_theme": "dark",
            "default_interactive": True,
            "output_dir": "./visualizations",
            
            # Neural config
            "default_neuron_type": "lif",
            "default_learning_rule": "stdp",
            
            # Communication config
            "redis_host": "localhost",
            "redis_port": 6379,
            "redis_db": 0,
            
            # Analysis config
            "analysis_window": 100,
            "analysis_overlap": 50,
            "regime_detection_method": "wavelet",
            "cycle_detection_method": "wavelet",
            
            # Integration config
            "auto_cross_asset": True,
            "auto_visualization": True,
            "auto_wavelet": True,
            "auto_reporting": False,
            "cross_asset_limit": 10,
            "report_frequency": 20,  # Report every N periods
            "advanced_regime_detection": True,
            "thread_reporting": True,
            "use_pulsar_indicators": True,
            "use_neuromorphic_detection": True,
            "max_threads": 8,
            "cache_results": True,
            "cache_ttl": 3600  # 1 hour
        }
        
        # Merge with provided config
        self.config = {**self.default_config, **(config or {})}
        
        # Initialize hardware accelerator
        self.hw_accelerator = HardwareAccelerator(
            enable_gpu=self.config["enable_gpu"],
            prefer_cuda=self.config["prefer_cuda"],
            device=self.config["device"],
            log_level=log_level
        )
        
        # Initialize CDFA (if available)
        self.cdfa = None
        if ENHANCED_CDFA_AVAILABLE:
            try:
                self.cdfa = CognitiveDiversityFusionAnalysis(log_level=log_level)
                self.logger.info("Enhanced CDFA initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize CDFA: {e}")
                
        # Initialize extensions
        self.cross_asset = CrossAssetAnalyzer(
            hw_accelerator=self.hw_accelerator,
            config={
                "default_correlation_method": self.config["default_correlation_method"],
                "default_timeframe": self.config["default_timeframe"],
                "correlation_threshold": self.config["correlation_threshold"]
            },
            log_level=log_level
        )
        
        self.visualization = VisualizationEngine(
            hw_accelerator=self.hw_accelerator,
            config={
                "default_theme": self.config["default_theme"],
                "default_interactive": self.config["default_interactive"],
                "output_dir": self.config["output_dir"]
            },
            log_level=log_level
        )
        
        self.pulsar = PulsarConnector(
            mode="redis",
            config={
                "redis_host": self.config["redis_host"],
                "redis_port": self.config["redis_port"],
                "redis_db": self.config["redis_db"]
            },
            log_level=log_level
        )
        
        self.pads = PADSReporter(
            mode="redis",
            config={
                "redis_host": self.config["redis_host"],
                "redis_port": self.config["redis_port"],
                "redis_db": self.config["redis_db"]
            },
            log_level=log_level
        )
        
        # Initialize optional extensions
        self.neuromorphic = None
        if NEUROMORPHIC_AVAILABLE:
            try:
                self.neuromorphic = NeuromorphicAnalyzer(
                    hw_accelerator=self.hw_accelerator,
                    config={
                        "default_neuron_type": self.config["default_neuron_type"],
                        "default_learning_rule": self.config["default_learning_rule"]
                    },
                    log_level=log_level
                )
                self.logger.info("Neuromorphic Analyzer initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize Neuromorphic Analyzer: {e}")
                
        self.wavelets = None
        if WAVELET_AVAILABLE:
            try:
                self.wavelets = WaveletProcessor(
                    hw_accelerator=self.hw_accelerator,
                    log_level=log_level
                )
                self.logger.info("Wavelet Processor initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize Wavelet Processor: {e}")
                
        # Initialize state
        self._lock = threading.RLock()
        self._initialized_analyzers = set()
        self._initialized_detectors = set()
        self._trained_models = {}
        self._analysis_cache = {}
        self._cross_asset_cache = {}
        self._status = {
            "last_analysis_time": None,
            "connected_components": [],
            "active_models": [],
            "pending_reports": 0,
            "threads_active": 0
        }
        
        # Update connected components list
        self._update_connected_components()
        
        self.logger.info("CDFA Integration initialized")
        
    def _update_connected_components(self):
        """Update the list of connected components."""
        connected = []
        
        if self.cdfa is not None:
            connected.append("cdfa")
            
        if self.cross_asset is not None:
            connected.append("cross_asset")
            
        if self.visualization is not None:
            connected.append("visualization")
            
        if self.pulsar is not None and self.pulsar.is_connected():
            connected.append("pulsar")
            
        if self.pads is not None and self.pads.is_connected():
            connected.append("pads")
            
        if self.neuromorphic is not None:
            connected.append("neuromorphic")
            
        if self.wavelets is not None:
            connected.append("wavelets")
            
        with self._lock:
            self._status["connected_components"] = connected
            
    def connect_analyzers(self):
        """
        Connect all available analyzers to the CDFA.
        
        Returns:
            List of connected analyzers
        """
        if self.cdfa is None:
            self.logger.error("CDFA not available, cannot connect analyzers")
            return []
            
        connected = []
        
        # Connect core analyzers if not already connected
        try:
            # Instantiate and connect each analyzer
            if ANALYZERS_AVAILABLE:
                if hasattr(self.cdfa, "connect_soc_analyzer") and "soc_analyzer" not in self._initialized_analyzers:
                    soc_analyzer_instance = soc_analyzer.SOCAnalyzer(log_level=self.logger.level)
                    self.cdfa.connect_soc_analyzer(soc_analyzer_instance)
                    self._initialized_analyzers.add("soc_analyzer")
                    connected.append("soc_analyzer")
                    self.logger.info("Connected soc_analyzer")

                if hasattr(self.cdfa, "connect_panarchy_analyzer") and "panarchy_analyzer" not in self._initialized_analyzers:
                    panarchy_analyzer_instance = panarchy_analyzer.PanarchyAnalyzer(log_level=self.logger.level)
                    self.cdfa.connect_panarchy_analyzer(panarchy_analyzer_instance)
                    self._initialized_analyzers.add("panarchy_analyzer")
                    connected.append("panarchy_analyzer")
                    self.logger.info("Connected panarchy_analyzer")

                if hasattr(self.cdfa, "connect_fibonacci_analyzer") and "fibonacci_analyzer" not in self._initialized_analyzers:
                    fibonacci_analyzer_instance = fibonacci_analyzer.FibonacciAnalyzer(log_level=self.logger.level)
                    self.cdfa.connect_fibonacci_analyzer(fibonacci_analyzer_instance)
                    self._initialized_analyzers.add("fibonacci_analyzer")
                    connected.append("fibonacci_analyzer")
                    self.logger.info("Connected fibonacci_analyzer")

                if hasattr(self.cdfa, "connect_antifragility_analyzer") and "antifragility_analyzer" not in self._initialized_analyzers:
                    antifragility_analyzer_instance = antifragility_analyzer.AntifragilityAnalyzer(log_level=self.logger.level)
                    self.cdfa.connect_antifragility_analyzer(antifragility_analyzer_instance)
                    self._initialized_analyzers.add("antifragility_analyzer")
                    connected.append("antifragility_analyzer")
                    self.logger.info("Connected antifragility_analyzer")

            # Pattern recognizer is in detectors now, handle in connect_detectors
            # if hasattr(self.cdfa, "connect_pattern_recognizer") and "pattern_recognizer" not in self._initialized_analyzers:
            #     pattern_recognizer_instance = pattern_recognizer.PatternRecognizer()
            #     self.cdfa.connect_pattern_recognizer(pattern_recognizer_instance)
            #     self._initialized_analyzers.add("pattern_recognizer")
            #     connected.append("pattern_recognizer")
            #     self.logger.info("Connected pattern_recognizer")

        except Exception as e:
            self.logger.error(f"Error connecting analyzers: {e}")
            
        # Connect advanced analyzers if available
        if self.neuromorphic is not None and "neuromorphic_analyzer" not in self._initialized_analyzers:
            try:
                # Create method to integrate neuromorphic analyzer
                def _handle_neuromorphic_signal(signal_data, source="unknown"):
                    if self.neuromorphic is not None:
                        # Process data with neuromorphic analyzer
                        return self.neuromorphic.analyze_market_regime_with_snn(signal_data, source=source)
                    return None
                    
                # Register the method with CDFA
                if hasattr(self.cdfa, "register_custom_analyzer"):
                    self.cdfa.register_custom_analyzer("neuromorphic", _handle_neuromorphic_signal)
                    self._initialized_analyzers.add("neuromorphic_analyzer")
                    connected.append("neuromorphic_analyzer")
                    self.logger.info("Connected neuromorphic analyzer")
            except Exception as e:
                self.logger.error(f"Error connecting neuromorphic analyzer: {e}")
                
        # Connect wavelet analyzer if available
        if self.wavelets is not None and "wavelet_analyzer" not in self._initialized_analyzers:
            try:
                # Create method to integrate wavelet analyzer
                def _handle_wavelet_signal(signal_data, source="unknown"):
                    if self.wavelets is not None:
                        # Process data with wavelet analyzer
                        if isinstance(signal_data, pd.DataFrame) and 'close' in signal_data.columns:
                            # Extract close prices
                            data = signal_data['close'].values
                        else:
                            # Use raw data
                            data = signal_data
                            
                        # Run wavelet analysis
                        return self.wavelets.analyze_market_regime(data)
                    return None
                    
                # Register the method with CDFA
                if hasattr(self.cdfa, "register_custom_analyzer"):
                    self.cdfa.register_custom_analyzer("wavelet", _handle_wavelet_signal)
                    self._initialized_analyzers.add("wavelet_analyzer")
                    connected.append("wavelet_analyzer")
                    self.logger.info("Connected wavelet analyzer")
            except Exception as e:
                self.logger.error(f"Error connecting wavelet analyzer: {e}")
                
        return connected
    
    def connect_detectors(self):
        """
        Connect all available detectors to the CDFA.
        
        Returns:
            List of connected detectors
        """
        if self.cdfa is None:
            self.logger.error("CDFA not available, cannot connect detectors")
            return []
            
        connected = []
        
        # Connect core detectors if not already connected
        try:
            # Instantiate and connect each detector
            if DETECTORS_AVAILABLE:
                if hasattr(self.cdfa, "integrate_whale_detector") and "whale_detector" not in self._initialized_detectors:
                    whale_detector_instance = whale_detector.WhaleDetector(log_level=self.logger.level)
                    self.cdfa.integrate_whale_detector(whale_detector_instance)
                    self._initialized_detectors.add("whale_detector")
                    connected.append("whale_detector")
                    self.logger.info("Connected whale_detector")

                if hasattr(self.cdfa, "integrate_black_swan_detector") and "black_swan_detector" not in self._initialized_detectors:
                    black_swan_detector_instance = black_swan_detector.BlackSwanDetector(log_level=self.logger.level)
                    self.cdfa.integrate_black_swan_detector(black_swan_detector_instance)
                    self._initialized_detectors.add("black_swan_detector")
                    connected.append("black_swan_detector")
                    self.logger.info("Connected black_swan_detector")

                if hasattr(self.cdfa, "integrate_fibonacci_detector") and "fibonacci_detector" not in self._initialized_detectors:
                    fibonacci_detector_instance = fibonacci_pattern_detector.FibonacciPatternDetector(log_level=self.logger.level)
                    self.cdfa.integrate_fibonacci_detector(fibonacci_detector_instance)
                    self._initialized_detectors.add("fibonacci_detector")
                    connected.append("fibonacci_detector")
                    self.logger.info("Connected fibonacci_detector")

                # Pattern recognizer is now a detector
                if hasattr(self.cdfa, "integrate_pattern_recognizer") and "pattern_recognizer" not in self._initialized_detectors:
                    pattern_recognizer_instance = pattern_recognizer.PatternRecognizer()
                    self.cdfa.integrate_pattern_recognizer(pattern_recognizer_instance)
                    self._initialized_detectors.add("pattern_recognizer")
                    connected.append("pattern_recognizer")
                    self.logger.info("Connected pattern_recognizer")

        except Exception as e:
            self.logger.error(f"Error connecting detectors: {e}")
            
        # Connect advanced detectors if available
        if self.wavelets is not None and "singularity_detector" not in self._initialized_detectors:
            try:
                # Create method to integrate singularity detector
                def _detect_singularities(data, metadata=None):
                    if self.wavelets is not None:
                        # Extract price data if DataFrame
                        if isinstance(data, pd.DataFrame) and 'close' in data.columns:
                            price_data = data['close'].values
                        else:
                            price_data = data
                            
                        # Detect singularities
                        singularities = self.wavelets.find_singularities(price_data)
                        
                        # Convert to detector output format
                        if singularities:
                            # Find strongest singularity
                            strongest = max(singularities, key=lambda x: x["strength"])
                            
                            # Create signal
                            signal = {
                                "detected": len(singularities) > 0,
                                "strength": strongest["strength"],
                                "position": strongest["position"],
                                "count": len(singularities),
                                "exponent": strongest["holder_exponent"]
                            }
                            
                            return signal
                        else:
                            return {"detected": False, "count": 0}
                    return {"detected": False}
                    
                # Register the method with CDFA
                if hasattr(self.cdfa, "register_custom_detector"):
                    self.cdfa.register_custom_detector("singularity", _detect_singularities)
                    self._initialized_detectors.add("singularity_detector")
                    connected.append("singularity_detector")
                    self.logger.info("Connected singularity detector")
            except Exception as e:
                self.logger.error(f"Error connecting singularity detector: {e}")
                
        # Connect neuromorphic pattern detector if available
        if self.neuromorphic is not None and "pattern_detector" not in self._initialized_detectors:
            try:
                # Create method to integrate pattern detector
                def _detect_patterns(data, metadata=None):
                    if self.neuromorphic is not None and hasattr(self.neuromorphic, "detect_pattern_with_snn"):
                        # Check if we have trained models
                        model_ids = self._trained_models.keys()
                        
                        if not model_ids:
                            return {"detected": False, "reason": "no_models"}
                            
                        # Extract features
                        if isinstance(data, pd.DataFrame):
                            features = self.neuromorphic.extract_features(data)
                            
                            # Combine features into a single vector
                            feature_vector = np.concatenate([
                                f for f in features.values() if isinstance(f, np.ndarray)
                            ])
                        else:
                            feature_vector = data
                            
                        # Use first available model
                        model_id = list(model_ids)[0]
                        
                        # Detect pattern
                        result = self.neuromorphic.detect_pattern_with_snn(model_id, feature_vector)
                        
                        if result:
                            return {
                                "detected": True,
                                "pattern": result["predicted_class"],
                                "confidence": result["confidence"],
                                "probabilities": result["probabilities"]
                            }
                        else:
                            return {"detected": False, "reason": "detection_failed"}
                    return {"detected": False}
                    
                # Register the method with CDFA
                if hasattr(self.cdfa, "register_custom_detector"):
                    self.cdfa.register_custom_detector("pattern", _detect_patterns)
                    self._initialized_detectors.add("pattern_detector")
                    connected.append("pattern_detector")
                    self.logger.info("Connected pattern detector")
            except Exception as e:
                self.logger.error(f"Error connecting pattern detector: {e}")
                
        return connected
    
    def connect_to_panarchy(self):
        """
        Connect CDFA to Panarchy Adaptive Decision System.
        
        Returns:
            Success status
        """
        if self.cdfa is None:
            self.logger.error("CDFA not available, cannot connect to Panarchy")
            return False
            
        try:
            # Check if the method exists
            if hasattr(self.cdfa, "publish_to_panarchy"):
                # Set up the reporting connection
                self.cdfa.publish_to_panarchy()
                self.logger.info("Connected to Panarchy Adaptive Decision System")
                return True
            else:
                self.logger.warning("CDFA does not support Panarchy connection")
                return False
        except Exception as e:
            self.logger.error(f"Error connecting to Panarchy: {e}")
            return False
    
    def connect_redis(self):
        """
        Connect all components to Redis.
        
        Returns:
            Success status
        """
        success = True
        
        # Connect Pulsar
        if self.pulsar is not None:
            try:
                pulsar_success = self.pulsar.connect()
                if not pulsar_success:
                    self.logger.warning("Failed to connect Pulsar to Redis")
                    success = False
            except Exception as e:
                self.logger.error(f"Error connecting Pulsar to Redis: {e}")
                success = False
                
        # Connect PADS
        if self.pads is not None:
            try:
                pads_success = self.pads.connect()
                if not pads_success:
                    self.logger.warning("Failed to connect PADS to Redis")
                    success = False
            except Exception as e:
                self.logger.error(f"Error connecting PADS to Redis: {e}")
                success = False
                
        # Update connected components
        self._update_connected_components()
        
        return success
    
    def analyze_symbol(self, symbol: str, data: pd.DataFrame, 
                    calculate_fusion: bool = True) -> Dict[str, Any]:
        """
        Perform comprehensive analysis on a symbol.
        
        Args:
            symbol: Symbol to analyze
            data: OHLCV data for the symbol
            calculate_fusion: Whether to calculate fusion result
            
        Returns:
            Dictionary with analysis results
        """
        # Start timing
        start_time = time.time()
        
        # Store the data in cross-asset analyzer
        self.cross_asset.add_price_data(symbol, data)
        
        # Initialize result
        result = {
            "symbol": symbol,
            "timestamp": start_time,
            "data_length": len(data)
        }
        
        # Process with CDFA if available
        if self.cdfa is not None:
            # Connect analyzers and detectors if needed
            if not self._initialized_analyzers:
                self.connect_analyzers()
                
            if not self._initialized_detectors:
                self.connect_detectors()
                
            # Process with CDFA
            cdfa_result = self.cdfa.process_signals_from_dataframe(
                data, symbol, calculate_fusion=calculate_fusion
            )
            
            # Add CDFA results
            if cdfa_result:
                result["cdfa"] = cdfa_result
                
                # Store signals
                if "signals" in cdfa_result:
                    result["signals"] = cdfa_result["signals"]
                    
                # Store fusion result
                if "fusion_result" in cdfa_result:
                    result["fusion"] = cdfa_result["fusion_result"]
                    
                # Store market regime
                if "market_regime" in cdfa_result:
                    result["market_regime"] = cdfa_result["market_regime"]
                    
        # Add cross-asset analysis if enabled
        if self.config["auto_cross_asset"]:
            # Get correlated assets
            correlated = self.cross_asset.find_correlated_assets(
                symbol, 
                limit=self.config["cross_asset_limit"]
            )
            
            if correlated:
                result["correlated_assets"] = correlated
                
            # Check for systemic importance
            contagion = self.cross_asset.calculate_contagion_risk(symbol)
            if contagion:
                result["contagion_risk"] = contagion
                
        # Add wavelet analysis if enabled
        if self.config["auto_wavelet"] and self.wavelets is not None:
            try:
                # Extract price data
                if "close" in data.columns:
                    price_data = data["close"].values
                    
                    # Perform wavelet analysis
                    wavelet_result = self.wavelets.analyze_market_regime(price_data)
                    
                    if wavelet_result:
                        result["wavelet"] = {
                            "regime_indicators": wavelet_result.regime_indicators,
                            "trend_strength": wavelet_result.trend_strength,
                            "dominant_scales": wavelet_result.dominant_scales,
                            "noise_level": wavelet_result.noise_level
                        }
                        
                    # Detect cycles
                    cycles = self.wavelets.detect_cycles(price_data)
                    if cycles:
                        result["cycles"] = cycles
                        
                    # Find singularities (potential regime change points)
                    singularities = self.wavelets.find_singularities(price_data)
                    if singularities:
                        result["singularities"] = singularities
            except Exception as e:
                self.logger.error(f"Error in wavelet analysis: {e}")
                
        # Add neuromorphic analysis if enabled
        if (self.config["use_neuromorphic_detection"] and 
            self.neuromorphic is not None and 
            len(self._trained_models) > 0):
            try:
                # Analyze with trained models
                regime_result = self.neuromorphic.analyze_market_regime_with_snn(
                    data, model_id=list(self._trained_models.keys())[0]
                )
                
                if regime_result:
                    result["neuromorphic"] = regime_result
            except Exception as e:
                self.logger.error(f"Error in neuromorphic analysis: {e}")
                
        # Create visualizations if enabled
        if self.config["auto_visualization"] and "signals" in result:
            try:
                # Create signal visualization
                viz = self.visualization.create_signal_visualization(
                    symbol, result["signals"],
                    timestamps=data.index if isinstance(data.index, pd.DatetimeIndex) else None,
                    title=f"Signal Analysis for {symbol}"
                )
                
                if viz is not None:
                    # Save visualization
                    viz_path = self.visualization.save_visualization(
                        viz, f"{symbol}_signals", "html"
                    )
                    
                    # Add to result
                    result["visualizations"] = {
                        "signals": viz_path
                    }
                    
                    # Create fusion heatmap if available
                    if "fusion" in result and "correlation_matrix" in result["fusion"]:
                        fusion_viz = self.visualization.create_fusion_heatmap(
                            result["fusion"]["correlation_matrix"],
                            title=f"Correlation Matrix for {symbol}"
                        )
                        
                        if fusion_viz is not None:
                            # Save visualization
                            fusion_path = self.visualization.save_visualization(
                                fusion_viz, f"{symbol}_correlation", "html"
                            )
                            
                            # Add to result
                            result["visualizations"]["correlation"] = fusion_path
            except Exception as e:
                self.logger.error(f"Error creating visualizations: {e}")
                
        # Send report to PADS if enabled
        if self.config["auto_reporting"] and "signals" in result:
            try:
                self._report_to_pads(symbol, result)
            except Exception as e:
                self.logger.error(f"Error reporting to PADS: {e}")
                
        # Query Pulsar if enabled
        if self.config["use_pulsar_indicators"] and self.pulsar.is_connected():
            try:
                pulsar_result = self._query_pulsar(symbol, result)
                if pulsar_result:
                    result["pulsar"] = pulsar_result
            except Exception as e:
                self.logger.error(f"Error querying Pulsar: {e}")
                
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        result["analysis_time"] = elapsed_time
        
        # Update status
        with self._lock:
            self._status["last_analysis_time"] = time.time()
            
        # Cache result
        self._cache_analysis(symbol, result)
        
        return result
    
    def _cache_analysis(self, symbol: str, result: Dict[str, Any]):
        """
        Cache analysis result.
        
        Args:
            symbol: Symbol analyzed
            result: Analysis result
        """
        if not self.config["cache_results"]:
            return
            
        with self._lock:
            self._analysis_cache[symbol] = (result, time.time())
            
    def _get_cached_analysis(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get cached analysis if valid.
        
        Args:
            symbol: Symbol to retrieve
            
        Returns:
            Cached analysis or None if not found or expired
        """
        if not self.config["cache_results"]:
            return None
            
        with self._lock:
            # Check if analysis is in cache
            cache_entry = self._analysis_cache.get(symbol)
            
            if cache_entry is None:
                return None
                
            analysis, timestamp = cache_entry
            
            # Check if expired
            current_time = time.time()
            if current_time - timestamp > self.config["cache_ttl"]:
                # Remove from cache
                self._analysis_cache.pop(symbol, None)
                return None
                
            return analysis
            
    def _report_to_pads(self, symbol: str, result: Dict[str, Any]):
        """
        Report analysis results to PADS.
        
        Args:
            symbol: Symbol analyzed
            result: Analysis result
        """
        if self.pads is None or not self.pads.is_connected():
            return
            
        # Create and send signals
        signals_sent = 0
        
        # Report market regime if available
        if "market_regime" in result:
            regime = result["market_regime"]
            
            # Create regime signal
            regime_signal = self.pads.create_market_regime_signal(
                symbol=symbol,
                regime=regime,
                timeframe=SignalTimeframe.MEDIUM,
                source_component="cdfa"
            )
            
            # Send to PADS
            self.pads.report_signal(regime_signal)
            signals_sent += 1
            
        # Report fusion result if available
        if "fusion" in result and "fused_signal" in result["fusion"]:
            fused_value = result["fusion"]["fused_signal"][-1]
            confidence = result["fusion"].get("confidence", 0.7)
            
            # Create trading signal
            if fused_value > 0.6:
                action = "buy"
                strength = (fused_value - 0.5) * 2
            elif fused_value < 0.4:
                action = "sell"
                strength = (0.5 - fused_value) * 2
            else:
                action = "hold"
                strength = 0.5
                
            # Create and send signal
            trading_signal = self.pads.create_trading_signal(
                symbol=symbol,
                action=action,
                strength=strength,
                confidence=confidence,
                timeframe=SignalTimeframe.MEDIUM,
                source_component="cdfa"
            )
            
            # Send to PADS
            self.pads.report_signal(trading_signal)
            signals_sent += 1
            
        # Report wavelet analysis if available
        if "wavelet" in result and "regime_indicators" in result["wavelet"]:
            indicators = result["wavelet"]["regime_indicators"]
            
            # Determine trend strength
            trend = indicators.get("trend", 0.5)
            
            # Create signal
            signal = self.pads.create_market_signal(
                symbol=symbol,
                value=trend,
                confidence=0.7,
                timeframe=SignalTimeframe.MEDIUM,
                source_component="wavelet",
                data=indicators
            )
            
            # Send to PADS
            self.pads.report_signal(signal)
            signals_sent += 1
            
        # Report volatility if available
        if "signals" in result and "volatility" in result:
            vol = result["volatility"]
            
            # Create volatility signal
            vol_signal = self.pads.create_volatility_signal(
                symbol=symbol,
                volatility=vol,
                vol_regime="high" if vol > 0.7 else "medium" if vol > 0.4 else "low",
                timeframe=SignalTimeframe.MEDIUM,
                source_component="cdfa"
            )
            
            # Send to PADS
            self.pads.report_signal(vol_signal)
            signals_sent += 1
            
        # Update status
        with self._lock:
            self._status["pending_reports"] = max(0, self._status["pending_reports"] - signals_sent)
            
    def _query_pulsar(self, symbol: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Query Pulsar for additional analysis.
        
        Args:
            symbol: Symbol analyzed
            result: Analysis result so far
            
        Returns:
            Pulsar analysis results
        """
        if self.pulsar is None or not self.pulsar.is_connected():
            return {}
            
        pulsar_results = {}
        
        # Query Q* for decision recommendation
        if "fusion" in result:
            try:
                # Convert CDFA state to Q* state
                state = {
                    "symbol": symbol,
                    "fusion_value": result["fusion"]["fused_signal"][-1] if "fused_signal" in result["fusion"] else 0.5,
                    "confidence": result["fusion"].get("confidence", 0.5),
                    "market_regime": result.get("market_regime", "unknown"),
                    "volatility": result.get("volatility", 0.5),
                    "trend_strength": result.get("wavelet", {}).get("trend_strength", 0.5) if "wavelet" in result else 0.5
                }
                
                # Add available signals
                if "signals" in result:
                    for name, values in result["signals"].items():
                        if values and len(values) > 0:
                            state[name] = values[-1]
                            
                # Query Q*
                q_star_result = self.pulsar.query_q_star(
                    state=state,
                    action_space=["buy", "sell", "hold"]
                )
                
                if q_star_result:
                    pulsar_results["q_star"] = q_star_result
            except Exception as e:
                self.logger.error(f"Error querying Q*: {e}")
                
        # Query River ML for pattern probability
        try:
            # Extract features
            features = {}
            
            if "signals" in result:
                for name, values in result["signals"].items():
                    if values and len(values) > 0:
                        features[name] = values[-1]
                        
            # Add market regime
            if "market_regime" in result:
                features["market_regime"] = result["market_regime"]
                
            # Query River ML
            river_result = self.pulsar.query_river_ml(features=features)
            
            if river_result:
                pulsar_results["river"] = river_result
        except Exception as e:
            self.logger.error(f"Error querying River ML: {e}")
            
        # Analyze any available news with Narrative Forecaster
        if "signals" in result and "news" in result["signals"]:
            try:
                news_text = result["signals"]["news"]
                
                if isinstance(news_text, list) and news_text and isinstance(news_text[-1], str):
                    # Get the latest news
                    latest_news = news_text[-1]
                    
                    # Context with symbol
                    context = {"symbol": symbol}
                    
                    # Query Narrative Forecaster asynchronously
                    self.pulsar.async_query_narrative_forecaster(
                        text=latest_news,
                        context=context,
                        callback=self._handle_narrative_result
                    )
                    
                    # Note that we sent a query
                    pulsar_results["narrative"] = {"status": "pending"}
            except Exception as e:
                self.logger.error(f"Error querying Narrative Forecaster: {e}")
                
        return pulsar_results
    
    def _handle_narrative_result(self, result: Dict[str, Any]):
        """
        Handle asynchronous result from Narrative Forecaster.
        
        Args:
            result: Analysis result
        """
        if not result:
            return
            
        try:
            # Extract symbol from context
            symbol = result.get("context", {}).get("symbol")
            
            if not symbol:
                return
                
            # Create PADS signal if connected
            if self.pads is not None and self.pads.is_connected():
                # Extract sentiment
                sentiment = result.get("sentiment", 0)
                
                # Extract magnitude
                magnitude = result.get("magnitude", 0)
                
                # Create narrative shift signal
                signal = self.pads.create_narrative_shift_signal(
                    symbol=symbol,
                    sentiment=sentiment,
                    shift_magnitude=magnitude,
                    timeframe=SignalTimeframe.LONG,
                    source_component="pulsar_narrative",
                    data=result
                )
                
                # Send to PADS
                self.pads.report_signal(signal)
        except Exception as e:
            self.logger.error(f"Error handling narrative result: {e}")
            
    def analyze_multi_asset(self, symbols: List[str], data_dict: Dict[str, pd.DataFrame],
                         analyze_correlation: bool = True) -> Dict[str, Any]:
        """
        Analyze multiple assets together.
        
        Args:
            symbols: List of symbols to analyze
            data_dict: Dictionary of symbol to OHLCV data
            analyze_correlation: Whether to analyze correlations
            
        Returns:
            Dictionary with multi-asset analysis results
        """
        # Start timing
        start_time = time.time()
        
        # Initialize result
        result = {
            "timestamp": start_time,
            "symbols": symbols,
            "individual_results": {}
        }
        
        # Store data in cross-asset analyzer
        for symbol, data in data_dict.items():
            self.cross_asset.add_price_data(symbol, data)
            
        # Analyze each symbol
        for symbol in symbols:
            if symbol in data_dict:
                # Check cache first
                cached_result = self._get_cached_analysis(symbol)
                
                if cached_result is not None:
                    # Use cached result
                    result["individual_results"][symbol] = cached_result
                else:
                    # Analyze symbol
                    analysis = self.analyze_symbol(symbol, data_dict[symbol])
                    
                    # Add to result
                    result["individual_results"][symbol] = analysis
                    
        # Analyze correlation structure if requested
        if analyze_correlation and len(symbols) > 1:
            try:
                # Calculate correlation matrix
                corr_matrix = self.cross_asset.calculate_correlation_matrix(
                    symbols, method=self.config["default_correlation_method"]
                )
                
                if not corr_matrix.empty:
                    result["correlation_matrix"] = corr_matrix
                    
                    # Create visualization if enabled
                    if self.config["auto_visualization"]:
                        try:
                            # Create heatmap
                            viz = self.visualization.create_fusion_heatmap(
                                corr_matrix,
                                title="Cross-Asset Correlation Matrix"
                            )
                            
                            if viz is not None:
                                # Save visualization
                                viz_path = self.visualization.save_visualization(
                                    viz, "cross_asset_correlation", "html"
                                )
                                
                                # Add to result
                                if "visualizations" not in result:
                                    result["visualizations"] = {}
                                    
                                result["visualizations"]["correlation"] = viz_path
                        except Exception as e:
                            self.logger.error(f"Error creating correlation visualization: {e}")
                            
                # Analyze market structure
                structure = self.cross_asset.analyze_market_correlation_structure(symbols)
                
                if structure:
                    result["market_structure"] = structure
                    
                    # Create visualization if enabled
                    if self.config["auto_visualization"] and "mst_edges" in structure:
                        try:
                            # Create network visualization
                            network_viz = self.visualization.create_network_visualization(
                                nodes=symbols,
                                edges=[(e["source"], e["target"], e["correlation"]) 
                                      for e in structure["mst_edges"]],
                                title="Market Structure Network"
                            )
                            
                            if network_viz is not None:
                                # Save visualization
                                network_path = self.visualization.save_visualization(
                                    network_viz, "market_structure", "html"
                                )
                                
                                # Add to result
                                if "visualizations" not in result:
                                    result["visualizations"] = {}
                                    
                                result["visualizations"]["network"] = network_path
                        except Exception as e:
                            self.logger.error(f"Error creating network visualization: {e}")
                            
                # Find systemic assets
                systemic = self.cross_asset.find_systemic_assets()
                
                if systemic:
                    result["systemic_assets"] = systemic
                    
                # Analyze market flows between asset classes
                flows = self.cross_asset.analyze_market_flows()
                
                if flows:
                    result["market_flows"] = flows
            except Exception as e:
                self.logger.error(f"Error in cross-asset analysis: {e}")
                
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        result["analysis_time"] = elapsed_time
        
        # Cache result
        self._cache_cross_asset(symbols, result)
        
        return result
    
    def _cache_cross_asset(self, symbols: List[str], result: Dict[str, Any]):
        """
        Cache cross-asset analysis result.
        
        Args:
            symbols: Symbols analyzed
            result: Analysis result
        """
        if not self.config["cache_results"]:
            return
            
        key = tuple(sorted(symbols))
        
        with self._lock:
            self._cross_asset_cache[key] = (result, time.time())
            
    def _get_cached_cross_asset(self, symbols: List[str]) -> Optional[Dict[str, Any]]:
        """
        Get cached cross-asset analysis if valid.
        
        Args:
            symbols: Symbols to retrieve
            
        Returns:
            Cached analysis or None if not found or expired
        """
        if not self.config["cache_results"]:
            return None
            
        key = tuple(sorted(symbols))
        
        with self._lock:
            # Check if analysis is in cache
            cache_entry = self._cross_asset_cache.get(key)
            
            if cache_entry is None:
                return None
                
            analysis, timestamp = cache_entry
            
            # Check if expired
            current_time = time.time()
            if current_time - timestamp > self.config["cache_ttl"]:
                # Remove from cache
                self._cross_asset_cache.pop(key, None)
                return None
                
            return analysis
            
    def train_regime_model(self, training_data: Dict[str, List[pd.DataFrame]]) -> Dict[str, Any]:
        """
        Train a regime detection model using neuromorphic SNN.
        
        Args:
            training_data: Dictionary of regime label to list of example dataframes
            
        Returns:
            Training results
        """
        if self.neuromorphic is None:
            self.logger.error("Neuromorphic analyzer not available, cannot train regime model")
            return {"success": False, "error": "neuromorphic_not_available"}
            
        try:
            # Extract features from training data
            patterns = {}
            
            for regime, examples in training_data.items():
                regime_patterns = []
                
                for df in examples:
                    # Extract features
                    features = self.neuromorphic.extract_features(df)
                    
                    if features:
                        # Combine features into a single vector
                        feature_vector = np.concatenate([
                            f for f in features.values() if isinstance(f, np.ndarray)
                        ])
                        
                        regime_patterns.append(feature_vector)
                        
                if regime_patterns:
                    patterns[regime] = regime_patterns
                    
            if not patterns:
                return {"success": False, "error": "no_valid_patterns_extracted"}
                
            # Train SNN model
            training_result = self.neuromorphic.train_pattern_recognition_snn(patterns)
            
            if training_result and "model_id" in training_result:
                # Store model
                model_id = training_result["model_id"]
                self._trained_models[model_id] = {
                    "classes": training_result.get("classes", []),
                    "accuracy": training_result.get("final_accuracy", 0.0),
                    "trained_at": time.time()
                }
                
                # Update status
                with self._lock:
                    self._status["active_models"] = list(self._trained_models.keys())
                    
                # Connect pattern detector if not already connected
                if "pattern_detector" not in self._initialized_detectors and self.cdfa is not None:
                    self.connect_detectors()
                    
                return {
                    "success": True,
                    "model_id": model_id,
                    "classes": training_result.get("classes", []),
                    "accuracy": training_result.get("final_accuracy", 0.0)
                }
            else:
                return {"success": False, "error": "training_failed"}
                
        except Exception as e:
            self.logger.error(f"Error training regime model: {e}")
            return {"success": False, "error": str(e)}
            
    def detect_regime(self, data: pd.DataFrame, method: Optional[str] = None) -> Dict[str, Any]:
        """
        Detect market regime using multiple methods.
        
        Args:
            data: OHLCV data
            method: Detection method (wavelet, neuromorphic, or cdfa)
            
        Returns:
            Regime detection results
        """
        # Get default method if not provided
        method = method or self.config["regime_detection_method"]
        
        # Initialize results
        results = {
            "timestamp": time.time(),
            "methods_used": []
        }
        
        # Detect using wavelet analysis
        if (method == "all" or method == "wavelet") and self.wavelets is not None:
            try:
                # Extract price data
                if "close" in data.columns:
                    price_data = data["close"].values
                    
                    # Analyze regime
                    wavelet_result = self.wavelets.analyze_market_regime(price_data)
                    
                    if wavelet_result:
                        results["wavelet"] = {
                            "regime_indicators": wavelet_result.regime_indicators,
                            "trend_strength": wavelet_result.trend_strength,
                            "dominant_scales": wavelet_result.dominant_scales,
                            "noise_level": wavelet_result.noise_level
                        }
                        
                        # Determine regime based on indicators
                        indicators = wavelet_result.regime_indicators
                        trend = indicators.get("trend", 0)
                        noise = indicators.get("noise", 0)
                        cyclical = indicators.get("cyclical", 0)
                        
                        if trend > 0.6:
                            if noise < 0.3:
                                regime = "conservation"
                            else:
                                regime = "growth"
                        elif cyclical > 0.6:
                            regime = "reorganization"
                        elif noise > 0.6:
                            regime = "release"
                        else:
                            regime = "transition"
                            
                        results["wavelet"]["regime"] = regime
                        results["methods_used"].append("wavelet")
            except Exception as e:
                self.logger.error(f"Error in wavelet regime detection: {e}")
                
        # Detect using neuromorphic SNN
        if (method == "all" or method == "neuromorphic") and self.neuromorphic is not None:
            try:
                # Check if we have trained models
                if self._trained_models:
                    model_id = list(self._trained_models.keys())[0]
                    
                    # Analyze regime
                    neuro_result = self.neuromorphic.analyze_market_regime_with_snn(
                        data, model_id=model_id
                    )
                    
                    if neuro_result:
                        results["neuromorphic"] = neuro_result
                        results["methods_used"].append("neuromorphic")
            except Exception as e:
                self.logger.error(f"Error in neuromorphic regime detection: {e}")
                
        # Detect using CDFA
        if (method == "all" or method == "cdfa") and self.cdfa is not None:
            try:
                # Connect analyzers if needed
                if not self._initialized_analyzers:
                    self.connect_analyzers()
                    
                # Process with CDFA
                cdfa_result = self.cdfa.process_signals_from_dataframe(
                    data, "temp", calculate_fusion=False
                )
                
                if cdfa_result and "market_regime" in cdfa_result:
                    results["cdfa"] = {
                        "regime": cdfa_result["market_regime"]
                    }
                    results["methods_used"].append("cdfa")
            except Exception as e:
                self.logger.error(f"Error in CDFA regime detection: {e}")
                
        # Determine consensus regime if multiple methods
        if len(results["methods_used"]) > 1:
            regimes = []
            
            if "wavelet" in results and "regime" in results["wavelet"]:
                regimes.append(results["wavelet"]["regime"])
                
            if "neuromorphic" in results and "current_regime" in results["neuromorphic"]:
                regimes.append(results["neuromorphic"]["current_regime"])
                
            if "cdfa" in results and "regime" in results["cdfa"]:
                regimes.append(results["cdfa"]["regime"])
                
            if regimes:
                # Count regimes
                regime_counts = {}
                for r in regimes:
                    if r not in regime_counts:
                        regime_counts[r] = 0
                    regime_counts[r] += 1
                    
                # Find most common
                consensus = max(regime_counts.items(), key=lambda x: x[1])[0]
                results["consensus_regime"] = consensus
                
        # Return results
        return results
    
    def detect_cycles(self, data: Union[pd.DataFrame, np.ndarray], 
                   method: Optional[str] = None) -> Dict[str, Any]:
        """
        Detect cycles in time series data.
        
        Args:
            data: OHLCV data or raw time series
            method: Detection method (wavelet or fft)
            
        Returns:
            Cycle detection results
        """
        # Get default method if not provided
        method = method or self.config["cycle_detection_method"]
        
        # Extract price data if DataFrame
        if isinstance(data, pd.DataFrame) and "close" in data.columns:
            series = data["close"].values
        else:
            series = data
            
        # Initialize results
        results = {
            "timestamp": time.time(),
            "method": method
        }
        
        # Detect using wavelet analysis
        if method == "wavelet" and self.wavelets is not None:
            try:
                # Detect cycles
                wavelet_result = self.wavelets.detect_cycles(series)
                
                if wavelet_result:
                    results["cycles"] = wavelet_result["periods"]
                    results["powers"] = wavelet_result["powers"]
                    results["dominant_period"] = wavelet_result["dominant_period"]
                    results["power_spectrum"] = wavelet_result["power_spectrum"]
            except Exception as e:
                self.logger.error(f"Error in wavelet cycle detection: {e}")
                
        # Detect using FFT
        elif method == "fft":
            try:
                if not self.has_scipy:
                    self.logger.error("SciPy not available for FFT cycle detection")
                    return results
                    
                from scipy import signal
                
                # Detrend series
                detrended = signal.detrend(series)
                
                # Calculate FFT
                n = len(detrended)
                freq = np.fft.fftfreq(n)
                fft_vals = np.fft.fft(detrended)
                
                # Get power spectrum
                power = np.abs(fft_vals) ** 2
                
                # Filter positive frequencies only
                pos_mask = freq > 0
                freqs = freq[pos_mask]
                power = power[pos_mask]
                
                # Sort by power
                sort_idx = np.argsort(power)[::-1]  # Descending
                sorted_freqs = freqs[sort_idx]
                sorted_power = power[sort_idx]
                
                # Calculate periods
                periods = 1 / sorted_freqs
                
                # Find peaks (exclude very low frequencies)
                min_freq = 1 / (n // 2)  # Minimum meaningful frequency
                valid_idx = sorted_freqs >= min_freq
                
                periods = periods[valid_idx]
                powers = sorted_power[valid_idx]
                
                # Limit to top 10
                if len(periods) > 10:
                    periods = periods[:10]
                    powers = powers[:10]
                    
                results["cycles"] = periods.tolist()
                results["powers"] = powers.tolist()
                results["dominant_period"] = float(periods[0]) if len(periods) > 0 else None
                results["power_spectrum"] = {
                    "freqs": freqs.tolist(),
                    "power": power.tolist()
                }
            except Exception as e:
                self.logger.error(f"Error in FFT cycle detection: {e}")
                
        # Return results
        return results
                
    def get_status(self) -> Dict[str, Any]:
        """
        Get current system status.
        
        Returns:
            Status dictionary
        """
        with self._lock:
            # Update real-time status
            status = self._status.copy()
            
            # Add available components
            self._update_connected_components()
            status["connected_components"] = self._status["connected_components"]
            
            # Add cache statistics
            status["cache"] = {
                "analysis_entries": len(self._analysis_cache),
                "cross_asset_entries": len(self._cross_asset_cache),
                "trained_models": len(self._trained_models)
            }
            
            # Add connection status
            status["connections"] = {
                "pulsar": self.pulsar.is_connected() if self.pulsar is not None else False,
                "pads": self.pads.is_connected() if self.pads is not None else False
            }
            
            return status
            
    def shutdown(self):
        """Shutdown all components and release resources."""
        # Disconnect from services
        if self.pulsar is not None:
            try:
                self.pulsar.disconnect()
            except:
                pass
                
        if self.pads is not None:
            try:
                self.pads.disconnect()
            except:
                pass
                
        # Stop background threads
        self._is_running = False
        
        # Close any open resources
        self.logger.info("Shutdown complete")
