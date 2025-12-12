#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced CDFA with Dependency Injection
=======================================

This module provides the AdvancedCDFA implementation using dependency injection
to break circular import dependencies while maintaining full functionality.

Author: Agent 6 - Circular Import Resolution Specialist
Date: 2025-06-29
"""

import os
import time
import uuid
import json
import logging
import warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
from dataclasses import dataclass, field, asdict
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache, partial
from enum import Enum

# Import interfaces and factory
from cdfa_interfaces import (
    ICDFACore, FusionType, SignalType, DiversityMethod, CDFAConfig, CDFAResult, MarketRegime,
    runtime_import, cdfa_container
)
from cdfa_factory import cdfa_factory


@dataclass
class AdvancedCDFAConfig:
    """Configuration for the Advanced CDFA module with dependency injection"""
    # Base CDFA configuration
    base_config: CDFAConfig = field(default_factory=CDFAConfig)
    
    # Hardware acceleration
    use_gpu: bool = True
    gpu_vendor: str = "auto"  # "auto", "nvidia", "amd", "apple"
    torch_device: str = "auto"  # "auto", "cuda", "rocm", "mps", "cpu"
    
    # TorchScript
    use_torchscript: bool = True
    enable_quantization: bool = True
    
    # PyWavelets
    wavelet_family: str = "sym8"
    wavelet_mode: str = "symmetric"
    wavelet_level: int = 4
    
    # Numba
    use_numba: bool = True
    parallel_threshold: int = 1000
    
    # Neuromorphic
    use_snn: bool = True
    stdp_learning_rate: float = 0.01
    snn_timesteps: int = 100
    snn_hidden_size: int = 128
    
    # Cross-asset
    max_assets: int = 50
    correlation_window: int = 60
    
    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: str = ""
    redis_channel_prefix: str = "adv_cdfa:"
    pulsar_channel: str = "pulsar:"
    pads_channel: str = "pads:"
    message_ttl: int = 3600
    
    # Performance
    num_threads: int = 8
    cache_size: int = 1000
    
    # Logging
    log_level: int = logging.INFO


class AdvancedCDFA:
    """
    Advanced Cognitive Diversity Fusion Analysis with Dependency Injection
    
    This class provides all advanced CDFA functionality while avoiding circular imports
    through dependency injection and interface-based design.
    """
    
    def __init__(self, config: AdvancedCDFAConfig = None):
        """
        Initialize the Advanced CDFA module with dependency injection
        
        Args:
            config: Configuration for the module (optional)
        """
        # Use default config if not provided
        self.config = config or AdvancedCDFAConfig()
        
        # Set up logger
        self.logger = logging.getLogger("AdvancedCDFA")
        self.logger.setLevel(self.config.log_level)
        
        # Initialize base CDFA core using runtime import
        self._base_cdfa = None
        self._initialize_base_cdfa()
        
        # Initialize dependency injection container
        self._setup_dependency_injection()
        
        # Initialize components using dependency injection
        self._initialize_components()
        
        # Set up communication
        self._setup_communication()
        
        self.logger.info("Advanced CDFA with dependency injection initialized")
    
    def _initialize_base_cdfa(self):
        """Initialize base CDFA using runtime import"""
        try:
            # Try to import the enhanced CDFA class
            enhanced_cdfa_module = runtime_import("enhanced_cdfa")
            if enhanced_cdfa_module and hasattr(enhanced_cdfa_module, "CognitiveDiversityFusionAnalysis"):
                CognitiveDiversityFusionAnalysis = enhanced_cdfa_module.CognitiveDiversityFusionAnalysis
                self._base_cdfa = CognitiveDiversityFusionAnalysis(self.config.base_config)
                self.logger.info("Successfully initialized enhanced CDFA base")
            else:
                self.logger.warning("Enhanced CDFA not available, using basic implementation")
                self._base_cdfa = BasicCDFAImplementation(self.config.base_config)
        except Exception as e:
            self.logger.warning(f"Failed to initialize enhanced CDFA: {e}, using basic implementation")
            self._base_cdfa = BasicCDFAImplementation(self.config.base_config)
    
    def _setup_dependency_injection(self):
        """Setup dependency injection container with configuration"""
        config_dict = asdict(self.config)
        
        # Register configuration
        cdfa_container.register_config(config_dict)
        
        # Register component factories if not already registered
        if not hasattr(cdfa_container, '_factories_registered'):
            self._register_component_factories()
            cdfa_container._factories_registered = True
    
    def _register_component_factories(self):
        """Register component factories with the DI container"""
        # Register each component factory
        cdfa_container.register_factory("hardware_accelerator", 
                                       lambda config: cdfa_factory.create_hardware_accelerator(config))
        
        cdfa_container.register_factory("wavelet_processor",
                                       lambda config: cdfa_factory.create_wavelet_processor(config))
        
        cdfa_container.register_factory("neuromorphic_analyzer",
                                       lambda config: cdfa_factory.create_neuromorphic_analyzer(
                                           cdfa_container.get("hardware_accelerator"), config))
        
        cdfa_container.register_factory("cross_asset_analyzer",
                                       lambda config: cdfa_factory.create_cross_asset_analyzer(config))
        
        cdfa_container.register_factory("visualization_engine",
                                       lambda config: cdfa_factory.create_visualization_engine(config))
        
        cdfa_container.register_factory("redis_connector",
                                       lambda config: cdfa_factory.create_redis_connector(config))
        
        cdfa_container.register_factory("torchscript_fusion",
                                       lambda config: cdfa_factory.create_torchscript_fusion(
                                           cdfa_container.get("hardware_accelerator"), config))
    
    def _initialize_components(self):
        """Initialize components using dependency injection"""
        try:
            # Initialize components through DI container
            self.hardware = cdfa_container.get("hardware_accelerator")
            self.wavelet = cdfa_container.get("wavelet_processor")
            self.neuromorphic = cdfa_container.get("neuromorphic_analyzer")
            self.cross_asset = cdfa_container.get("cross_asset_analyzer")
            self.visualization = cdfa_container.get("visualization_engine")
            self.redis = cdfa_container.get("redis_connector")
            self.torch_optimizer = cdfa_container.get("torchscript_fusion")
            
            self.logger.info("All components initialized successfully via dependency injection")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            # Initialize fallback components
            self._initialize_fallback_components()
    
    def _initialize_fallback_components(self):
        """Initialize fallback components if DI fails"""
        from cdfa_factory import (
            FallbackHardwareAccelerator, FallbackWaveletProcessor,
            FallbackNeuromorphicAnalyzer, FallbackCrossAssetAnalyzer,
            FallbackVisualizationEngine, FallbackRedisConnector,
            FallbackTorchScriptFusion
        )
        
        config_dict = asdict(self.config)
        
        self.hardware = FallbackHardwareAccelerator(config_dict)
        self.wavelet = FallbackWaveletProcessor(config_dict)
        self.neuromorphic = FallbackNeuromorphicAnalyzer(self.hardware, config_dict)
        self.cross_asset = FallbackCrossAssetAnalyzer(config_dict)
        self.visualization = FallbackVisualizationEngine(config_dict)
        self.redis = FallbackRedisConnector(config_dict)
        self.torch_optimizer = FallbackTorchScriptFusion(self.hardware, config_dict)
        
        self.logger.warning("Using fallback component implementations")
    
    def _setup_communication(self):
        """Set up communication channels with other systems"""
        try:
            if hasattr(self.redis, 'redis') and self.redis.redis is not None:
                # Subscribe to relevant channels
                pads_channel = self.config.pads_channel
                pulsar_channel = self.config.pulsar_channel
                
                self.redis.subscribe(f"{pads_channel}feedback", self._handle_pads_feedback)
                self.redis.subscribe(f"{pulsar_channel}notification", self._handle_pulsar_notification)
                
                self.logger.info("Communication channels set up successfully")
            else:
                self.logger.warning("Redis not connected, communication disabled")
        except Exception as e:
            self.logger.error(f"Error setting up communication: {e}")
    
    def _handle_pads_feedback(self, data):
        """Handle feedback from PADS"""
        try:
            self.logger.debug(f"Received PADS feedback: {data}")
            signal_id = data.get("signal_id")
            performance = data.get("performance")
            
            if signal_id and performance is not None:
                self.logger.info(f"Updating performance for signal {signal_id}: {performance}")
                
        except Exception as e:
            self.logger.error(f"Error handling PADS feedback: {e}")
    
    def _handle_pulsar_notification(self, data):
        """Handle notifications from Pulsar"""
        try:
            self.logger.debug(f"Received Pulsar notification: {data}")
            notification_type = data.get("type")
            
            if notification_type == "q_star_prediction":
                self._process_q_star_prediction(data)
            elif notification_type == "narrative_forecast":
                self._process_narrative_forecast(data)
                
        except Exception as e:
            self.logger.error(f"Error handling Pulsar notification: {e}")
    
    def _process_q_star_prediction(self, data):
        """Process Q* prediction result from Pulsar"""
        prediction = data.get("prediction")
        confidence = data.get("confidence")
        horizon = data.get("horizon")
        
        self.logger.info(f"Received Q* prediction for horizon {horizon} with confidence {confidence}")
    
    def _process_narrative_forecast(self, data):
        """Process narrative forecast result from Pulsar"""
        forecast = data.get("forecast")
        sentiment = data.get("sentiment")
        topics = data.get("topics")
        
        self.logger.info(f"Received narrative forecast with sentiment {sentiment}")
    
    # ============================================================================
    # Core CDFA Methods (Delegated to base implementation)
    # ============================================================================
    
    def process_signals_from_dataframe(self, dataframe: pd.DataFrame, symbol: str = None,
                                     calculate_fusion: bool = True, 
                                     use_advanced: bool = True) -> Dict[str, Any]:
        """
        Process signals from a DataFrame with advanced capabilities
        
        Args:
            dataframe: Input DataFrame with OHLCV data
            symbol: Symbol for the data (optional)
            calculate_fusion: Whether to calculate fusion result
            use_advanced: Whether to use advanced capabilities
            
        Returns:
            Dictionary with processing results
        """
        if not use_advanced or self._base_cdfa is None:
            # Fall back to basic processing
            return self._basic_signal_processing(dataframe, symbol, calculate_fusion)
        
        try:
            # Preprocess data with wavelet denoising
            preprocessed_df = self._preprocess_with_wavelets(dataframe)
            
            # Get base signals using base implementation
            base_result = self._base_cdfa.process_signals_from_dataframe(preprocessed_df, symbol, False)
            
            # Extract signals
            signals = base_result.get("signals", {})
            
            # Add wavelet-based features
            wavelet_signals = self._extract_wavelet_signals(preprocessed_df)
            signals.update(wavelet_signals)
            
            # Add neuromorphic signals if enabled
            if self.config.use_snn:
                try:
                    neuromorphic_signals = self._extract_neuromorphic_signals(preprocessed_df)
                    signals.update(neuromorphic_signals)
                except Exception as e:
                    self.logger.warning(f"Neuromorphic signal extraction failed: {e}")
            
            # Detect market regime
            market_regime = self.wavelet.analyze_market_regime(preprocessed_df)
            
            # Calculate fusion if requested
            result = {
                "signals": signals,
                "market_regime": market_regime["regime"],
                "regime_details": market_regime
            }
            
            if calculate_fusion:
                fusion_result = self._calculate_advanced_fusion(signals, market_regime)
                result["fusion_result"] = fusion_result
            
            return result
            
        except Exception as e:
            self.logger.error(f"Advanced signal processing failed: {e}")
            return self._basic_signal_processing(dataframe, symbol, calculate_fusion)
    
    def _basic_signal_processing(self, dataframe: pd.DataFrame, symbol: str = None,
                               calculate_fusion: bool = True) -> Dict[str, Any]:
        """Basic signal processing fallback"""
        # Simple signal extraction
        signals = {}
        
        if 'close' in dataframe.columns:
            close_prices = dataframe['close'].values
            
            # Calculate basic signals
            if len(close_prices) > 1:
                returns = np.diff(np.log(close_prices))
                signals['momentum'] = np.append(0, returns)
                
                # Simple trend signal
                if len(close_prices) > 5:
                    trend = np.zeros_like(close_prices)
                    for i in range(5, len(close_prices)):
                        trend[i] = (close_prices[i] - close_prices[i-5]) / close_prices[i-5]
                    signals['trend'] = trend
        
        # Basic market regime
        regime = {"regime": "mixed", "trend_strength": 0.5, "volatility": 0.0}
        
        result = {
            "signals": signals,
            "market_regime": regime["regime"],
            "regime_details": regime
        }
        
        if calculate_fusion and signals:
            # Simple fusion
            signal_df = pd.DataFrame(signals)
            fused = signal_df.mean(axis=1).values
            result["fusion_result"] = {
                "fused_signal": fused,
                "confidence": 0.5,
                "weights": {name: 1.0/len(signals) for name in signals.keys()}
            }
        
        return result
    
    def fuse_signals(self, signals_df: pd.DataFrame) -> pd.Series:
        """
        Fuse multiple signals into a single signal
        
        Args:
            signals_df: DataFrame with signals as columns
            
        Returns:
            Fused signal as pandas Series
        """
        if self._base_cdfa and hasattr(self._base_cdfa, 'fuse_signals'):
            return self._base_cdfa.fuse_signals(signals_df)
        else:
            # Simple fallback fusion
            return signals_df.mean(axis=1)
    
    def adaptive_fusion(self, system_scores: Dict[str, List[float]], 
                       performance_metrics: Dict[str, float],
                       market_regime: str = "mixed",
                       volatility: float = 0.5) -> List[float]:
        """
        Perform adaptive signal fusion
        
        Args:
            system_scores: Dictionary of signal scores
            performance_metrics: Performance metrics for each signal
            market_regime: Current market regime
            volatility: Market volatility
            
        Returns:
            Fused signal as list
        """
        if self._base_cdfa and hasattr(self._base_cdfa, 'adaptive_fusion'):
            return self._base_cdfa.adaptive_fusion(system_scores, performance_metrics, market_regime, volatility)
        else:
            # Simple fallback adaptive fusion
            if not system_scores:
                return []
            
            # Calculate weighted average based on performance metrics
            signals_array = []
            weights = []
            
            for signal_name, scores in system_scores.items():
                if scores:
                    signals_array.append(scores)
                    weights.append(performance_metrics.get(signal_name, 1.0))
            
            if not signals_array:
                return []
            
            # Normalize weights
            weights = np.array(weights)
            weights = weights / np.sum(weights) if np.sum(weights) > 0 else np.ones_like(weights) / len(weights)
            
            # Calculate weighted fusion
            min_length = min(len(signal) for signal in signals_array)
            fused = np.zeros(min_length)
            
            for i, (signal, weight) in enumerate(zip(signals_array, weights)):
                fused += weight * np.array(signal[:min_length])
            
            return fused.tolist()
    
    def analyze_signals(self, signals_list: List[np.ndarray]) -> Dict[str, Any]:
        """
        Analyze list of signal arrays for CDFA server compatibility
        
        Args:
            signals_list: List of numpy arrays containing signal data
            
        Returns:
            Dictionary with fused_signal, confidence, components, processing_time
        """
        try:
            start_time = time.time()
            
            if len(signals_list) < 1:
                return {"fused_signal": 0.5, "confidence": 0.0, "components": {}, "processing_time": 0.0}
            
            # Convert arrays to DataFrame format
            prices = signals_list[0]
            volumes = signals_list[1] if len(signals_list) > 1 else np.ones_like(prices)
            
            if len(prices) < 10:
                return {"fused_signal": 0.5, "confidence": 0.0, "components": {}, "processing_time": 0.0}
            
            # Create temporary DataFrame
            df = pd.DataFrame({
                'close': prices,
                'volume': volumes,
                'open': prices,
                'high': prices * 1.001,
                'low': prices * 0.999,
                'timestamp': pd.date_range('2024-01-01', periods=len(prices), freq='1H')
            })
            
            # Process using advanced capabilities
            result = self.process_signals_from_dataframe(df, use_advanced=True)
            
            # Extract results
            if "fusion_result" in result:
                fusion_result = result["fusion_result"]
                fused_signal = fusion_result.get("fused_signal", [0.5])
                confidence = fusion_result.get("confidence", 0.0)
                components = fusion_result.get("weights", {})
            else:
                # Fallback calculation
                price_change = (prices[-1] - prices[0]) / prices[0] if prices[0] != 0 else 0.0
                fused_signal = 0.5 + np.tanh(price_change) * 0.3
                confidence = min(0.5, abs(price_change) * 10)
                components = {"price_change": price_change}
            
            processing_time = time.time() - start_time
            
            return {
                "fused_signal": float(np.clip(fused_signal[-1] if isinstance(fused_signal, (list, np.ndarray)) else fused_signal, 0.0, 1.0)),
                "confidence": float(np.clip(confidence, 0.0, 1.0)),
                "components": components,
                "processing_time": processing_time
            }
            
        except Exception as e:
            self.logger.error(f"Error in analyze_signals: {e}")
            return {"fused_signal": 0.5, "confidence": 0.0, "components": {}, "processing_time": 0.0}
    
    # ============================================================================
    # Advanced Processing Methods
    # ============================================================================
    
    def _preprocess_with_wavelets(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Preprocess data with wavelet denoising"""
        try:
            result_df = dataframe.copy()
            
            # Apply wavelet denoising to OHLC data
            for col in ['open', 'high', 'low', 'close']:
                if col in result_df.columns:
                    result_df[col] = self.wavelet.denoise_signal(result_df[col].values)
            
            return result_df
        except Exception as e:
            self.logger.warning(f"Wavelet preprocessing failed: {e}")
            return dataframe
    
    def _extract_wavelet_signals(self, dataframe: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Extract signals using wavelet analysis"""
        signals = {}
        
        try:
            if 'close' not in dataframe.columns:
                return signals
            
            close_prices = dataframe['close'].values
            cycles = self.wavelet.detect_cycles(close_prices)
            
            # Add cycle-based signals
            signals['wavelet_dominant_cycle'] = np.full(len(close_prices), cycles['dominant_cycle']['period'])
            
            # Calculate phase alignment
            current_phase = cycles['current_phase']
            phase_signal = np.full(len(close_prices), 0.5)
            
            # Convert phase to signal (0-1 range)
            normalized_phase = (current_phase + np.pi) / (2 * np.pi)
            phase_signal[-1] = normalized_phase
            
            signals['wavelet_cycle_phase'] = phase_signal
            
        except Exception as e:
            self.logger.warning(f"Wavelet signal extraction failed: {e}")
        
        return signals
    
    def _extract_neuromorphic_signals(self, dataframe: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Extract signals using neuromorphic processing"""
        signals = {}
        
        try:
            # Extract features for SNN processing
            features = self._extract_snn_features(dataframe)
            
            # Process with SNN
            snn_result = self.neuromorphic.process_with_snn(features)
            
            if "error" in snn_result:
                self.logger.warning(f"SNN processing error: {snn_result['error']}")
                return signals
            
            # Extract output signal
            output = snn_result["output"]
            
            # Normalize to [0, 1] range
            min_val = np.min(output)
            max_val = np.max(output)
            range_val = max_val - min_val
            
            if range_val > 1e-10:
                normalized = (output - min_val) / range_val
            else:
                normalized = np.full_like(output, 0.5)
            
            # Create signal with the same length as input data
            signal = np.full(len(dataframe), np.nan)
            signal[-len(normalized):] = normalized
            
            signals['neuromorphic_signal'] = signal
            
            # Add synchrony measure if available
            if "synchrony" in snn_result:
                synchrony = snn_result["synchrony"]
                sync_signal = np.full(len(dataframe), np.nan)
                sync_signal[-len(synchrony):] = synchrony
                signals['neuromorphic_synchrony'] = sync_signal
            
        except Exception as e:
            self.logger.warning(f"Neuromorphic signal extraction failed: {e}")
        
        return signals
    
    def _extract_snn_features(self, dataframe: pd.DataFrame) -> np.ndarray:
        """Extract features for SNN processing"""
        features = []
        
        # Use OHLCV data
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in dataframe.columns:
                if col != 'volume':
                    values = dataframe[col].values
                    returns = np.diff(np.log(values))
                    returns = np.append(0, returns)
                    features.append(returns)
                else:
                    values = dataframe[col].values
                    if np.max(values) > 0:
                        norm_volume = values / np.max(values)
                    else:
                        norm_volume = np.zeros_like(values)
                    features.append(norm_volume)
        
        # Calculate additional features
        if 'close' in dataframe.columns:
            close = dataframe['close'].values
            
            # Add moving averages
            for window in [5, 20, 50]:
                if len(close) >= window:
                    ma = np.full_like(close, np.nan)
                    for i in range(window - 1, len(close)):
                        ma[i] = np.mean(close[i-window+1:i+1])
                    
                    # Calculate relative position
                    rel_pos = (close - ma) / ma
                    features.append(rel_pos)
        
        # Stack features
        feature_array = np.column_stack(features)
        
        # Replace NaN values
        feature_array = np.nan_to_num(feature_array, nan=0.0)
        
        return feature_array
    
    def _calculate_advanced_fusion(self, signals: Dict[str, np.ndarray], 
                                  market_regime: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate fusion result with advanced methods"""
        try:
            # Get regime-specific parameters
            fusion_type, alpha = self._get_regime_parameters(market_regime["regime"])
            
            # Normalize signals
            normalized_signals = {}
            for name, values in signals.items():
                normalized = self.hardware.normalize_scores(values)
                normalized_signals[name] = normalized
            
            # Calculate pairwise diversity
            try:
                diversity_matrix = self._calculate_diversity_matrix(normalized_signals)
            except Exception as e:
                self.logger.warning(f"Diversity calculation failed: {e}")
                diversity_matrix = None
            
            # Apply regime-specific weighting
            weighted_signals, weights = self._apply_regime_weighting(normalized_signals, 
                                                                   diversity_matrix, 
                                                                   market_regime)
            
            # Create signal DataFrame
            signal_df = pd.DataFrame(weighted_signals)
            
            if len(signal_df) == 0:
                self.logger.warning("No signals available for fusion")
                return {
                    "fused_signal": np.array([]),
                    "confidence": 0.0,
                    "weights": weights
                }
            
            # Calculate fusion
            fused_signal = self._calculate_base_fusion(signal_df, fusion_type, alpha)
            
            # Calculate confidence
            confidence = self._calculate_confidence(normalized_signals, fused_signal, diversity_matrix)
            
            return {
                "fused_signal": fused_signal,
                "confidence": confidence,
                "weights": weights,
                "fusion_type": str(fusion_type),
                "alpha": alpha
            }
            
        except Exception as e:
            self.logger.error(f"Advanced fusion calculation failed: {e}")
            # Return fallback result
            return {
                "fused_signal": np.array([0.5]),
                "confidence": 0.0,
                "weights": {}
            }
    
    def _get_regime_parameters(self, regime: str) -> Tuple[FusionType, float]:
        """Get fusion parameters based on market regime"""
        regime_params = {
            "strong_trend": (FusionType.SCORE, 0.8),
            "trending": (FusionType.HYBRID, 0.7),
            "mixed": (FusionType.HYBRID, 0.5),
            "mean_reverting": (FusionType.RANK, 0.4),
            "choppy": (FusionType.LAYERED, 0.3)
        }
        
        return regime_params.get(regime, (FusionType.HYBRID, 0.5))
    
    def _calculate_diversity_matrix(self, signals: Dict[str, np.ndarray]) -> np.ndarray:
        """Calculate diversity matrix between signals"""
        names = list(signals.keys())
        values = np.array([signals[name] for name in names])
        
        return self.hardware.calculate_diversity_matrix(values)
    
    def _apply_regime_weighting(self, signals: Dict[str, np.ndarray],
                               diversity_matrix: np.ndarray,
                               market_regime: Dict[str, Any]) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
        """Apply regime-specific weighting to signals"""
        regime = market_regime["regime"]
        
        # Define regime-specific weights
        signal_type_weights = {
            "strong_trend": {"trend": 2.0, "momentum": 1.5, "wavelet": 1.2, "neuromorphic": 1.0, "mean_reversion": 0.5, "oscillator": 0.5},
            "trending": {"trend": 1.5, "momentum": 1.5, "wavelet": 1.2, "neuromorphic": 1.0, "mean_reversion": 0.7, "oscillator": 0.8},
            "mean_reverting": {"trend": 0.7, "momentum": 0.8, "wavelet": 1.2, "neuromorphic": 1.0, "mean_reversion": 1.5, "oscillator": 1.5},
            "choppy": {"trend": 0.5, "momentum": 0.6, "wavelet": 1.5, "neuromorphic": 1.5, "mean_reversion": 1.0, "oscillator": 1.0},
        }
        
        weights_map = signal_type_weights.get(regime, {
            "trend": 1.0, "momentum": 1.0, "wavelet": 1.0, "neuromorphic": 1.0, "mean_reversion": 1.0, "oscillator": 1.0
        })
        
        # Apply weights
        weights = {}
        weighted_signals = {}
        
        for name, signal in signals.items():
            # Determine signal type from name
            signal_type = "other"
            for type_name in weights_map.keys():
                if type_name in name.lower():
                    signal_type = type_name
                    break
            
            weight = weights_map.get(signal_type, 1.0)
            
            # Apply diversity adjustment
            if diversity_matrix is not None:
                signal_names = list(signals.keys())
                try:
                    idx = signal_names.index(name)
                    avg_diversity = np.mean(diversity_matrix[idx, :])
                    weight *= (0.5 + 0.5 * avg_diversity)
                except ValueError:
                    pass
            
            weights[name] = weight
            weighted_signals[name] = signal
        
        return weighted_signals, weights
    
    def _calculate_base_fusion(self, signal_df: pd.DataFrame,
                              fusion_type: FusionType, alpha: float) -> np.ndarray:
        """Calculate fusion using base methods"""
        if fusion_type == FusionType.SCORE:
            return signal_df.mean(axis=1).values
        elif fusion_type == FusionType.RANK:
            ranks = signal_df.rank(axis=1)
            return ranks.mean(axis=1).values / len(signal_df.columns)
        elif fusion_type == FusionType.HYBRID:
            score_result = signal_df.mean(axis=1).values
            ranks = signal_df.rank(axis=1)
            rank_result = ranks.mean(axis=1).values / len(signal_df.columns)
            return alpha * score_result + (1 - alpha) * rank_result
        else:  # FusionType.LAYERED
            weighted_sum = np.zeros(len(signal_df))
            weight_sum = 0
            
            for i, col in enumerate(signal_df.columns):
                weight = 1.0 / (i + 1)
                weighted_sum += weight * signal_df[col].values
                weight_sum += weight
            
            return weighted_sum / weight_sum if weight_sum > 0 else np.zeros(len(signal_df))
    
    def _calculate_confidence(self, signals: Dict[str, np.ndarray],
                             fused_signal: np.ndarray,
                             diversity_matrix: np.ndarray = None) -> float:
        """Calculate confidence score for the fusion result"""
        if len(signals) < 2 or len(fused_signal) == 0:
            return 0.5
        
        # Use the last value for confidence calculation
        last_idx = len(fused_signal) - 1
        signal_values = np.array([s[last_idx] for s in signals.values() if last_idx < len(s)])
        
        if len(signal_values) < 2:
            return 0.5
        
        # Calculate agreement between signals
        signal_std = np.std(signal_values)
        agreement = 1.0 - min(1.0, signal_std * 2)
        
        # Diversity factor
        diversity = 0.5
        if diversity_matrix is not None:
            non_diagonal = diversity_matrix[~np.eye(diversity_matrix.shape[0], dtype=bool)]
            diversity = np.mean(non_diagonal) if len(non_diagonal) > 0 else 0.5
        
        # Combine factors
        confidence = 0.5 * agreement + 0.5 * diversity
        
        return confidence
    
    # ============================================================================
    # Cross-Asset Analysis Methods
    # ============================================================================
    
    def analyze_cross_asset(self, symbols_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Perform cross-asset analysis on multiple assets"""
        try:
            # Clear previous data
            if hasattr(self.cross_asset, 'clear_asset_data'):
                self.cross_asset.clear_asset_data()
            
            # Add each asset's data
            for symbol, data in symbols_data.items():
                self.cross_asset.add_asset_data(symbol, data)
            
            # Calculate correlations
            correlations = {}
            for method in ['pearson']:  # Start with just pearson
                try:
                    correlations[method] = self.cross_asset.calculate_correlation_matrix(method)
                except Exception as e:
                    self.logger.warning(f"Correlation calculation failed for {method}: {e}")
            
            # Calculate lead-lag relationships
            try:
                lead_lag = self.cross_asset.calculate_lead_lag_relationships()
            except Exception as e:
                self.logger.warning(f"Lead-lag calculation failed: {e}")
                lead_lag = pd.DataFrame()
            
            return {
                "correlations": correlations,
                "lead_lag": lead_lag,
                "contagion_risk": pd.DataFrame(),  # Placeholder
                "regimes": {},  # Placeholder
                "market_structure": {}  # Placeholder
            }
            
        except Exception as e:
            self.logger.error(f"Cross-asset analysis failed: {e}")
            return {}
    
    # ============================================================================
    # Utility Methods
    # ============================================================================
    
    def get_version_info(self) -> Dict[str, Any]:
        """Get version information about the module and its components"""
        try:
            import platform
            import numpy as np
            import pandas as pd
            
            # Hardware info
            hw_info = {
                "device": "cpu",
                "gpu_available": self.hardware.is_gpu_available(),
                "compute_capability": self.hardware.get_compute_capability()
            }
            
            # Software versions
            sw_info = {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__
            }
            
            # Configuration
            config_info = {
                "use_gpu": self.config.use_gpu,
                "use_torchscript": self.config.use_torchscript,
                "use_numba": self.config.use_numba,
                "use_snn": self.config.use_snn,
                "dependency_injection": True
            }
            
            return {
                "version": "1.0.0-injected",
                "hardware": hw_info,
                "software": sw_info,
                "config": config_info
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get version info: {e}")
            return {"version": "1.0.0-injected", "error": str(e)}


# ============================================================================
# Basic CDFA Implementation for Fallback
# ============================================================================

class BasicCDFAImplementation:
    """Basic CDFA implementation for fallback when enhanced_cdfa is not available"""
    
    def __init__(self, config: CDFAConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def process_signals_from_dataframe(self, dataframe: pd.DataFrame, symbol: str = None,
                                     calculate_fusion: bool = True) -> Dict[str, Any]:
        """Basic signal processing implementation"""
        signals = {}
        
        if 'close' in dataframe.columns:
            close_prices = dataframe['close'].values
            
            # Basic momentum signal
            if len(close_prices) > 1:
                returns = np.diff(np.log(close_prices))
                signals['momentum'] = np.append(0, returns)
            
            # Basic trend signal
            if len(close_prices) > 10:
                trend = np.zeros_like(close_prices)
                for i in range(10, len(close_prices)):
                    trend[i] = (close_prices[i] - close_prices[i-10]) / close_prices[i-10]
                signals['trend'] = trend
        
        return {"signals": signals}
    
    def fuse_signals(self, signals_df: pd.DataFrame) -> pd.Series:
        """Basic signal fusion"""
        return signals_df.mean(axis=1)
    
    def adaptive_fusion(self, system_scores: Dict[str, List[float]], 
                       performance_metrics: Dict[str, float],
                       market_regime: str = "mixed",
                       volatility: float = 0.5) -> List[float]:
        """Basic adaptive fusion"""
        if not system_scores:
            return []
        
        # Simple weighted average
        signals_array = []
        weights = []
        
        for signal_name, scores in system_scores.items():
            if scores:
                signals_array.append(scores)
                weights.append(performance_metrics.get(signal_name, 1.0))
        
        if not signals_array:
            return []
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / np.sum(weights) if np.sum(weights) > 0 else np.ones_like(weights) / len(weights)
        
        # Calculate weighted fusion
        min_length = min(len(signal) for signal in signals_array)
        fused = np.zeros(min_length)
        
        for i, (signal, weight) in enumerate(zip(signals_array, weights)):
            fused += weight * np.array(signal[:min_length])
        
        return fused.tolist()