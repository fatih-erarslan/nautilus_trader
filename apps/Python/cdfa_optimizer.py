#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  6 11:21:13 2025

@author: ashina
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CDFA Optimizer Module

A unified optimization module that integrates TorchScript, PyWavelets, Numba, 
Norse and Rockpool with STDP neuroplasticity for CDFA system enhancement.

This module provides high-performance computing abstractions and optimized 
models for financial analysis and signal processing.

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
import json
from datetime import datetime, timedelta
import uuid
import tempfile

# Import from cdfa_extensions
from .hw_acceleration import HardwareAccelerator
from .wavelet_processor import WaveletProcessor
from .neuromorphic_analyzer import NeuromorphicAnalyzer
from .cross_asset_analyzer import CrossAssetAnalyzer
from .advanced_visualization import VisualizationEngine
from .pulsar_connector import PulsarConnector
from .pads_reporter import PADSReporter

# Optional component imports - handle gracefully
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. Model optimization will be limited.")

try:
    import pywt
    PYWAVELETS_AVAILABLE = True
except ImportError:
    PYWAVELETS_AVAILABLE = False
    warnings.warn("PyWavelets not available. Wavelet processing will be limited.")

try:
    import numba
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    warnings.warn("Numba not available. Acceleration will be limited.")

try:
    import norse
    NORSE_AVAILABLE = True
except ImportError:
    NORSE_AVAILABLE = False
    warnings.warn("Norse not available. Neuromorphic computing will be limited.")

try:
    import rockpool
    ROCKPOOL_AVAILABLE = True
except ImportError:
    ROCKPOOL_AVAILABLE = False
    warnings.warn("Rockpool not available. Neuromorphic computing will be limited.")

class OptimizationLevel(Enum):
    """Optimization levels for CDFA model processing."""
    NONE = 0       # No optimization
    BASIC = 1      # Basic optimizations (vectorization, JIT)
    STANDARD = 2   # Standard optimizations (TorchScript, Numba)
    ADVANCED = 3   # Advanced optimizations (GPU, quantization)
    EXPERIMENTAL = 4  # Experimental features (mixed precision, SNN)

class ModelFormat(Enum):
    """Model formats for CDFA optimizer."""
    PYTORCH = auto()      # PyTorch model
    TORCHSCRIPT = auto()  # TorchScript model
    ONNX = auto()         # ONNX model
    SNN = auto()          # Spiking Neural Network
    CUSTOM = auto()       # Custom model format

class CDFAOptimizer:
    """
    Unified optimization manager for CDFA system.
    
    Integrates various acceleration and optimization techniques:
    - TorchScript compilation and optimization
    - PyWavelets signal processing
    - Numba JIT compilation
    - Norse/Rockpool neuromorphic computing
    - Cross-asset optimization
    
    Provides a unified interface for all optimization needs.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, 
               log_level: int = logging.INFO):
        """
        Initialize the CDFA optimizer.
        
        Args:
            config: Configuration options
            log_level: Logging level
        """
        # Setup logging
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.setLevel(log_level)
        
        # Default configuration
        self.default_config = {
            # Hardware acceleration
            "enable_gpu": True,
            "prefer_cuda": True,
            "use_torch_compile": True,  # PyTorch 2.0+ feature
            "use_amp": True,            # Automatic Mixed Precision
            "use_cuda_graphs": True,    # CUDA Graphs for repeated computation
            
            # Wavelet processing
            "default_wavelet": "db4",
            "wavelet_denoising": True,
            "wavelet_feature_extraction": True,
            
            # Neuromorphic computing
            "enable_snn": True,
            "use_stdp": True,
            "default_neuron_type": "lif",
            
            # Model optimization
            "optimization_level": OptimizationLevel.STANDARD.value,
            "quantize_models": False,
            "model_pruning": False,
            "batch_inference": True,
            
            # Caching
            "enable_caching": True,
            "cache_ttl": 3600,  # 1 hour
            "cache_models": True,
            
            # Cross-asset optimization
            "parallel_asset_processing": True,
            "batch_size": 32,
            
            # External connections
            "connect_to_pulsar": True,
            "connect_to_pads": True,
            
            # Visualization
            "enable_visualization": True,
            "interactive_visualization": True
        }
        
        # Merge with provided config
        self.config = {**self.default_config, **(config or {})}
        
        # Initialize state
        self._lock = threading.RLock()
        self._compiled_models = {}  # name -> model
        self._optimized_pipelines = {}  # name -> pipeline
        
        # Initialize components with shared hardware accelerator
        self.hw_accelerator = HardwareAccelerator(
            enable_gpu=self.config["enable_gpu"],
            prefer_cuda=self.config["prefer_cuda"]
        )
        
        # Initialize wavelet processor
        self.wavelet_processor = WaveletProcessor(
            hw_accelerator=self.hw_accelerator,
            config={
                "default_wavelet": self.config["default_wavelet"],
                "cache_results": self.config["enable_caching"],
                "cache_ttl": self.config["cache_ttl"]
            },
            log_level=log_level
        )
        
        # Initialize neuromorphic analyzer if available
        if NORSE_AVAILABLE or ROCKPOOL_AVAILABLE:
            self.neuromorphic_analyzer = NeuromorphicAnalyzer(
                hw_accelerator=self.hw_accelerator,
                config={
                    "default_neuron_type": self.config["default_neuron_type"],
                    "cache_models": self.config["cache_models"],
                    "cache_ttl": self.config["cache_ttl"]
                },
                log_level=log_level
            )
        else:
            self.neuromorphic_analyzer = None
            
        # Initialize cross-asset analyzer
        self.cross_asset_analyzer = CrossAssetAnalyzer(
            hw_accelerator=self.hw_accelerator,
            config={
                "use_parallel": self.config["parallel_asset_processing"],
                "cache_results": self.config["enable_caching"],
                "cache_ttl": self.config["cache_ttl"]
            },
            log_level=log_level
        )
        
        # Initialize visualization engine if enabled
        if self.config["enable_visualization"]:
            self.visualization_engine = VisualizationEngine(
                hw_accelerator=self.hw_accelerator,
                config={
                    "default_interactive": self.config["interactive_visualization"],
                    "cache_plots": self.config["enable_caching"],
                    "cache_ttl": self.config["cache_ttl"]
                },
                log_level=log_level
            )
        else:
            self.visualization_engine = None
            
        # Initialize external connections if enabled
        if self.config["connect_to_pulsar"]:
            self.pulsar_connector = PulsarConnector(
                config={
                    "keep_alive": True,
                    "background_processing": True
                },
                log_level=log_level
            )
        else:
            self.pulsar_connector = None
            
        if self.config["connect_to_pads"]:
            self.pads_reporter = PADSReporter(
                config={
                    "feedback_enabled": True
                },
                log_level=log_level
            )
        else:
            self.pads_reporter = None
            
        self.logger.info("CDFA Optimizer initialized")
    
    # ----- Model Optimization Methods -----
    
    def optimize_model(self, model: Any, model_format: Union[str, ModelFormat] = "pytorch",
                    optimization_level: Optional[Union[int, OptimizationLevel]] = None,
                    example_input: Optional[Any] = None,
                    quantize: Optional[bool] = None,
                    name: Optional[str] = None) -> Any:
        """
        Optimize a model for inference performance.
        
        Args:
            model: Model to optimize
            model_format: Format of the input model
            optimization_level: Optimization level (default from config)
            example_input: Example input for tracing (required for some optimizations)
            quantize: Whether to quantize the model (default from config)
            name: Optional name for the optimized model
            
        Returns:
            Optimized model
        """
        # Handle string or enum for model format
        if isinstance(model_format, str):
            try:
                model_format = ModelFormat[model_format.upper()]
            except KeyError:
                self.logger.warning(f"Unknown model format: {model_format}, defaulting to PYTORCH")
                model_format = ModelFormat.PYTORCH
                
        # Get defaults from config if not provided
        if optimization_level is None:
            optimization_level = self.config["optimization_level"]
            
        if isinstance(optimization_level, int):
            try:
                optimization_level = OptimizationLevel(optimization_level)
            except ValueError:
                self.logger.warning(f"Unknown optimization level: {optimization_level}, defaulting to STANDARD")
                optimization_level = OptimizationLevel.STANDARD
                
        if quantize is None:
            quantize = self.config["quantize_models"]
            
        # Generate name if not provided
        if name is None:
            name = f"model_{uuid.uuid4().hex[:8]}"
            
        # Optimize based on model format
        if model_format == ModelFormat.PYTORCH:
            optimized_model = self._optimize_pytorch_model(
                model, optimization_level, example_input, quantize
            )
        elif model_format == ModelFormat.TORCHSCRIPT:
            # Already TorchScript, apply additional optimizations
            optimized_model = self._optimize_torchscript_model(
                model, optimization_level, quantize
            )
        elif model_format == ModelFormat.ONNX:
            self.logger.warning("ONNX optimization not implemented yet")
            optimized_model = model
        elif model_format == ModelFormat.SNN:
            # Optimize SNN model
            optimized_model = self._optimize_snn_model(
                model, optimization_level
            )
        else:
            self.logger.warning(f"Unsupported model format: {model_format}")
            optimized_model = model
            
        # Store optimized model
        with self._lock:
            self._compiled_models[name] = optimized_model
            
        return optimized_model
    
    def _optimize_pytorch_model(self, model: Any, optimization_level: OptimizationLevel,
                             example_input: Optional[Any] = None,
                             quantize: bool = False) -> Any:
        """
        Optimize a PyTorch model.
        
        Args:
            model: PyTorch model
            optimization_level: Optimization level
            example_input: Example input for tracing
            quantize: Whether to quantize the model
            
        Returns:
            Optimized model
        """
        if not TORCH_AVAILABLE:
            self.logger.warning("PyTorch not available, returning original model")
            return model
            
        # Set model to evaluation mode
        if hasattr(model, 'eval'):
            model.eval()
            
        # Apply optimization based on level
        if optimization_level == OptimizationLevel.NONE:
            # No optimization
            return model
            
        elif optimization_level == OptimizationLevel.BASIC:
            # Basic optimization: TorchScript with minimal settings
            if example_input is not None:
                return self.hw_accelerator.create_torchscript_model(
                    model, example_input, optimization_level=1, quantize=False
                )
            else:
                self.logger.warning("Example input required for basic optimization")
                return model
                
        elif optimization_level == OptimizationLevel.STANDARD:
            # Standard optimization: TorchScript with inference optimization
            if example_input is not None:
                return self.hw_accelerator.create_torchscript_model(
                    model, example_input, optimization_level=2, quantize=quantize
                )
            else:
                self.logger.warning("Example input required for standard optimization")
                return model
                
        elif optimization_level == OptimizationLevel.ADVANCED:
            # Advanced optimization: TorchScript with full optimization
            if example_input is not None:
                return self.hw_accelerator.create_torchscript_model(
                    model, example_input, optimization_level=3, quantize=quantize,
                    freeze=True
                )
            else:
                self.logger.warning("Example input required for advanced optimization")
                return model
                
        elif optimization_level == OptimizationLevel.EXPERIMENTAL:
            # Experimental optimization: Try torch.compile() (PyTorch 2.0+)
            if self.config["use_torch_compile"] and hasattr(torch, 'compile'):
                try:
                    # Try to use torch.compile
                    compiled_model = torch.compile(model)
                    self.logger.info("Model optimized with torch.compile()")
                    return compiled_model
                except Exception as e:
                    self.logger.warning(f"torch.compile() failed: {e}, falling back to TorchScript")
                    
            # Fallback to TorchScript with maximum optimization
            if example_input is not None:
                return self.hw_accelerator.create_torchscript_model(
                    model, example_input, optimization_level=3, quantize=quantize,
                    freeze=True, dynamic=True
                )
            else:
                self.logger.warning("Example input required for experimental optimization")
                return model
        else:
            self.logger.warning(f"Unknown optimization level: {optimization_level}")
            return model
    
    def _optimize_torchscript_model(self, model: Any, optimization_level: OptimizationLevel,
                                 quantize: bool = False) -> Any:
        """
        Apply additional optimizations to a TorchScript model.
        
        Args:
            model: TorchScript model
            optimization_level: Optimization level
            quantize: Whether to quantize the model
            
        Returns:
            Optimized TorchScript model
        """
        if not TORCH_AVAILABLE:
            self.logger.warning("PyTorch not available, returning original model")
            return model
            
        if optimization_level == OptimizationLevel.NONE:
            # No additional optimization
            return model
            
        try:
            # Apply JIT optimization passes
            optimized_model = self.hw_accelerator._apply_jit_passes(
                model, optimization_level.value
            )
            
            # Apply quantization if requested
            if quantize and optimization_level.value >= 2:
                try:
                    # Import quantization modules
                    from torch.quantization import quantize_dynamic
                    
                    # Apply dynamic quantization
                    quantized_model = quantize_dynamic(
                        optimized_model,
                        dtype=torch.qint8
                    )
                    
                    self.logger.info("Applied quantization to TorchScript model")
                    return quantized_model
                    
                except Exception as e:
                    self.logger.warning(f"Quantization failed: {e}, returning unquantized model")
                    
            return optimized_model
            
        except Exception as e:
            self.logger.error(f"Error optimizing TorchScript model: {e}")
            return model
    
    def _optimize_snn_model(self, model: Any, optimization_level: OptimizationLevel) -> Any:
        """
        Optimize a Spiking Neural Network model.
        
        Args:
            model: SNN model
            optimization_level: Optimization level
            
        Returns:
            Optimized SNN model
        """
        if self.neuromorphic_analyzer is None:
            self.logger.warning("Neuromorphic analyzer not available, returning original model")
            return model
            
        # Basic sanity check
        if not NORSE_AVAILABLE and not ROCKPOOL_AVAILABLE:
            self.logger.warning("Neither Norse nor Rockpool available, returning original model")
            return model
            
        # For now, just return the original model
        # TODO: Implement SNN-specific optimizations
        return model
    
    def optimize_pipeline(self, pipeline_config: Dict[str, Any], name: Optional[str] = None) -> Dict[str, Any]:
        """
        Create an optimized processing pipeline based on configuration.
        
        Args:
            pipeline_config: Pipeline configuration
            name: Optional name for the pipeline
            
        Returns:
            Optimized pipeline configuration
        """
        # Generate name if not provided
        if name is None:
            name = f"pipeline_{uuid.uuid4().hex[:8]}"
            
        # Extract pipeline components
        components = pipeline_config.get("components", [])
        optimization_level = pipeline_config.get("optimization_level", self.config["optimization_level"])
        
        # Convert optimization level to enum if needed
        if isinstance(optimization_level, int):
            try:
                optimization_level = OptimizationLevel(optimization_level)
            except ValueError:
                optimization_level = OptimizationLevel.STANDARD
                
        # Create optimized pipeline configuration
        optimized_pipeline = {
            "name": name,
            "optimization_level": optimization_level,
            "components": []
        }
        
        # Optimize each component
        for component in components:
            component_type = component.get("type")
            
            if component_type == "model":
                # Optimize model component
                model = component.get("model")
                model_format = component.get("format", "pytorch")
                example_input = component.get("example_input")
                quantize = component.get("quantize", self.config["quantize_models"])
                
                if model is not None:
                    # Generate component name
                    component_name = component.get("name", f"{name}_model_{len(optimized_pipeline['components'])}")
                    
                    # Optimize model
                    optimized_model = self.optimize_model(
                        model, model_format, optimization_level, example_input, quantize, component_name
                    )
                    
                    # Add to pipeline
                    optimized_pipeline["components"].append({
                        "type": "model",
                        "name": component_name,
                        "optimized_model": optimized_model,
                        "format": "torchscript" if model_format == "pytorch" else model_format,
                        "config": component.get("config", {})
                    })
                    
            elif component_type == "wavelet":
                # Add wavelet processing component
                optimized_pipeline["components"].append({
                    "type": "wavelet",
                    "name": component.get("name", f"{name}_wavelet_{len(optimized_pipeline['components'])}"),
                    "processor": self.wavelet_processor,
                    "config": component.get("config", {})
                })
                
            elif component_type == "neuromorphic":
                # Add neuromorphic component if available
                if self.neuromorphic_analyzer is not None:
                    optimized_pipeline["components"].append({
                        "type": "neuromorphic",
                        "name": component.get("name", f"{name}_neuromorphic_{len(optimized_pipeline['components'])}"),
                        "analyzer": self.neuromorphic_analyzer,
                        "config": component.get("config", {})
                    })
                else:
                    self.logger.warning(f"Neuromorphic analyzer not available, skipping component")
                    
            elif component_type == "cross_asset":
                # Add cross-asset component
                optimized_pipeline["components"].append({
                    "type": "cross_asset",
                    "name": component.get("name", f"{name}_cross_asset_{len(optimized_pipeline['components'])}"),
                    "analyzer": self.cross_asset_analyzer,
                    "config": component.get("config", {})
                })
                
            elif component_type == "visualization":
                # Add visualization component if enabled
                if self.visualization_engine is not None:
                    optimized_pipeline["components"].append({
                        "type": "visualization",
                        "name": component.get("name", f"{name}_viz_{len(optimized_pipeline['components'])}"),
                        "engine": self.visualization_engine,
                        "config": component.get("config", {})
                    })
                else:
                    self.logger.warning(f"Visualization engine not available, skipping component")
                    
            elif component_type == "external":
                # Add external communication component
                protocol = component.get("protocol", "")
                
                if protocol == "pulsar" and self.pulsar_connector is not None:
                    optimized_pipeline["components"].append({
                        "type": "external",
                        "protocol": "pulsar",
                        "name": component.get("name", f"{name}_pulsar_{len(optimized_pipeline['components'])}"),
                        "connector": self.pulsar_connector,
                        "config": component.get("config", {})
                    })
                elif protocol == "pads" and self.pads_reporter is not None:
                    optimized_pipeline["components"].append({
                        "type": "external",
                        "protocol": "pads",
                        "name": component.get("name", f"{name}_pads_{len(optimized_pipeline['components'])}"),
                        "reporter": self.pads_reporter,
                        "config": component.get("config", {})
                    })
                else:
                    self.logger.warning(f"Unsupported external protocol: {protocol}, skipping component")
                    
            elif component_type == "custom":
                # Add custom component as-is
                optimized_pipeline["components"].append(component)
                
            else:
                self.logger.warning(f"Unknown component type: {component_type}, skipping")
                
        # Store optimized pipeline
        with self._lock:
            self._optimized_pipelines[name] = optimized_pipeline
            
        return optimized_pipeline
    
    def run_pipeline(self, pipeline_name: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run an optimized pipeline with the given inputs.
        
        Args:
            pipeline_name: Name of the pipeline
            inputs: Input data for the pipeline
            
        Returns:
            Pipeline outputs
        """
        # Get pipeline configuration
        with self._lock:
            pipeline = self._optimized_pipelines.get(pipeline_name)
            
        if pipeline is None:
            self.logger.error(f"Pipeline '{pipeline_name}' not found")
            return {}
            
        # Initialize pipeline state
        state = {
            "inputs": inputs,
            "outputs": {},
            "intermediate": {}
        }
        
        # Process each component
        for component in pipeline["components"]:
            component_type = component["type"]
            component_name = component["name"]
            
            try:
                if component_type == "model":
                    # Run model component
                    model = component["optimized_model"]
                    config = component["config"]
                    
                    # Get model input
                    input_name = config.get("input", "default")
                    model_input = state["inputs"].get(input_name)
                    
                    if model_input is None:
                        # Try to get from intermediate outputs
                        model_input = state["intermediate"].get(input_name)
                        
                    if model_input is None:
                        self.logger.warning(f"Input '{input_name}' not found for model component '{component_name}'")
                        continue
                        
                    # Run model
                    if component["format"] == "torchscript":
                        # Run TorchScript model
                        output = self.hw_accelerator.run_torchscript_model(
                            model, model_input,
                            optimize_execution=True,
                            warm_up=True,
                            batch_size=self.config["batch_size"] if config.get("use_batching", True) else None
                        )
                    else:
                        # Run generic model
                        output = model(model_input)
                        
                    # Store output
                    output_name = config.get("output", component_name)
                    state["intermediate"][output_name] = output
                    
                    # Add to final outputs if specified
                    if config.get("is_output", False):
                        state["outputs"][output_name] = output
                        
                elif component_type == "wavelet":
                    # Run wavelet processing component
                    processor = component["processor"]
                    config = component["config"]
                    
                    # Get processor input
                    input_name = config.get("input", "default")
                    wavelet_input = state["inputs"].get(input_name)
                    
                    if wavelet_input is None:
                        # Try to get from intermediate outputs
                        wavelet_input = state["intermediate"].get(input_name)
                        
                    if wavelet_input is None:
                        self.logger.warning(f"Input '{input_name}' not found for wavelet component '{component_name}'")
                        continue
                        
                    # Get operation type
                    operation = config.get("operation", "denoise")
                    
                    if operation == "denoise":
                        # Denoise signal
                        result = processor.denoise_signal(
                            wavelet_input,
                            wavelet=config.get("wavelet"),
                            level=config.get("level"),
                            method=config.get("method")
                        )
                        output = result.denoised if result else wavelet_input
                        
                    elif operation == "extract_features":
                        # Extract wavelet features
                        output = processor.extract_wavelet_features(
                            wavelet_input,
                            wavelet=config.get("wavelet"),
                            level=config.get("level")
                        )
                        
                    elif operation == "analyze_regime":
                        # Analyze market regime
                        output = processor.analyze_market_regime(
                            wavelet_input,
                            wavelet=config.get("wavelet"),
                            level=config.get("level")
                        )
                        
                    else:
                        self.logger.warning(f"Unknown wavelet operation: {operation}")
                        output = wavelet_input
                        
                    # Store output
                    output_name = config.get("output", component_name)
                    state["intermediate"][output_name] = output
                    
                    # Add to final outputs if specified
                    if config.get("is_output", False):
                        state["outputs"][output_name] = output
                        
                elif component_type == "neuromorphic":
                    # Run neuromorphic component
                    analyzer = component["analyzer"]
                    config = component["config"]
                    
                    # Get analyzer input
                    input_name = config.get("input", "default")
                    neuro_input = state["inputs"].get(input_name)
                    
                    if neuro_input is None:
                        # Try to get from intermediate outputs
                        neuro_input = state["intermediate"].get(input_name)
                        
                    if neuro_input is None:
                        self.logger.warning(f"Input '{input_name}' not found for neuromorphic component '{component_name}'")
                        continue
                        
                    # Get operation type
                    operation = config.get("operation", "encode")
                    
                    if operation == "encode":
                        # Encode data to spikes
                        output = analyzer.encode_data(
                            neuro_input,
                            method=config.get("encoding_method"),
                            time_window=config.get("time_window"),
                            feature_name=config.get("feature_name", "generic"),
                            source=config.get("source", "unknown")
                        )
                        
                    elif operation == "detect_pattern":
                        # Detect pattern with SNN
                        output = analyzer.detect_pattern_with_snn(
                            config.get("model_id"),
                            neuro_input
                        )
                        
                    else:
                        self.logger.warning(f"Unknown neuromorphic operation: {operation}")
                        output = neuro_input
                        
                    # Store output
                    output_name = config.get("output", component_name)
                    state["intermediate"][output_name] = output
                    
                    # Add to final outputs if specified
                    if config.get("is_output", False):
                        state["outputs"][output_name] = output
                        
                elif component_type == "cross_asset":
                    # Run cross-asset component
                    analyzer = component["analyzer"]
                    config = component["config"]
                    
                    # Get operation type
                    operation = config.get("operation", "correlation")
                    
                    if operation == "correlation":
                        # Get symbols
                        symbols = config.get("symbols", [])
                        
                        # Calculate correlation matrix
                        output = analyzer.calculate_correlation_matrix(
                            symbols,
                            method=config.get("method"),
                            timeframe=config.get("timeframe")
                        )
                        
                    elif operation == "find_correlated":
                        # Find correlated assets
                        symbol = config.get("symbol")
                        
                        if symbol:
                            output = analyzer.find_correlated_assets(
                                symbol,
                                threshold=config.get("threshold"),
                                method=config.get("method"),
                                timeframe=config.get("timeframe"),
                                asset_classes=config.get("asset_classes"),
                                limit=config.get("limit", 10)
                            )
                        else:
                            self.logger.warning(f"Symbol not specified for find_correlated operation")
                            output = []
                            
                    elif operation == "lead_lag":
                        # Analyze lead-lag relationship
                        symbol1 = config.get("symbol1")
                        symbol2 = config.get("symbol2")
                        
                        if symbol1 and symbol2:
                            output = analyzer.analyze_lead_lag_relationship(
                                symbol1, symbol2,
                                method=config.get("method"),
                                max_lag=config.get("max_lag"),
                                timeframe=config.get("timeframe")
                            )
                        else:
                            self.logger.warning(f"Symbols not specified for lead_lag operation")
                            output = {}
                            
                    else:
                        self.logger.warning(f"Unknown cross-asset operation: {operation}")
                        output = {}
                        
                    # Store output
                    output_name = config.get("output", component_name)
                    state["intermediate"][output_name] = output
                    
                    # Add to final outputs if specified
                    if config.get("is_output", False):
                        state["outputs"][output_name] = output
                        
                elif component_type == "visualization":
                    # Run visualization component
                    engine = component["engine"]
                    config = component["config"]
                    
                    # Get input
                    input_name = config.get("input", "default")
                    viz_input = state["inputs"].get(input_name)
                    
                    if viz_input is None:
                        # Try to get from intermediate outputs
                        viz_input = state["intermediate"].get(input_name)
                        
                    if viz_input is None:
                        self.logger.warning(f"Input '{input_name}' not found for visualization component '{component_name}'")
                        continue
                        
                    # Get visualization type
                    viz_type = config.get("type", "default")
                    
                    # Create visualization based on type
                    if viz_type == "signal":
                        # Signal visualization
                        output = engine.create_signal_visualization(
                            config.get("symbols", ["unknown"]),
                            viz_input,
                            title=config.get("title"),
                            output_format=config.get("format")
                        )
                        
                    elif viz_type == "heatmap":
                        # Heatmap visualization
                        output = engine.create_fusion_heatmap(
                            viz_input,
                            title=config.get("title"),
                            output_format=config.get("format")
                        )
                        
                    elif viz_type == "network":
                        # Network visualization
                        nodes = config.get("nodes", [])
                        edges = config.get("edges", [])
                        
                        # Extract from input if not provided
                        if not nodes and hasattr(viz_input, 'nodes'):
                            nodes = viz_input.nodes
                        if not edges and hasattr(viz_input, 'edges'):
                            edges = viz_input.edges
                            
                        output = engine.create_network_visualization(
                            nodes, edges,
                            node_properties=config.get("node_properties"),
                            title=config.get("title"),
                            output_format=config.get("format")
                        )
                        
                    else:
                        self.logger.warning(f"Unknown visualization type: {viz_type}")
                        output = None
                        
                    # Store output
                    output_name = config.get("output", component_name)
                    state["intermediate"][output_name] = output
                    
                    # Add to final outputs if specified
                    if config.get("is_output", False):
                        state["outputs"][output_name] = output
                        
                    # Save visualization if path provided
                    save_path = config.get("save_path")
                    if save_path and output is not None:
                        engine.save_visualization(output, save_path, config.get("format"))
                        
                elif component_type == "external":
                    # Run external communication component
                    protocol = component["protocol"]
                    config = component["config"]
                    
                    if protocol == "pulsar":
                        # Pulsar communication
                        connector = component["connector"]
                        
                        # Get input
                        input_name = config.get("input", "default")
                        message_data = state["inputs"].get(input_name)
                        
                        if message_data is None:
                            # Try to get from intermediate outputs
                            message_data = state["intermediate"].get(input_name)
                            
                        if message_data is None:
                            self.logger.warning(f"Input '{input_name}' not found for Pulsar component '{component_name}'")
                            continue
                            
                        # Get operation
                        operation = config.get("operation", "query")
                        
                        if operation == "query_q_star":
                            output = connector.query_q_star(
                                message_data,
                                action_space=config.get("action_space"),
                                timeout=config.get("timeout")
                            )
                            
                        elif operation == "query_river_ml":
                            output = connector.query_river_ml(
                                message_data,
                                model_name=config.get("model_name"),
                                timeout=config.get("timeout")
                            )
                            
                        elif operation == "query_cerebellum":
                            output = connector.query_cerebellum(
                                message_data,
                                pattern_type=config.get("pattern_type"),
                                timeout=config.get("timeout")
                            )
                            
                        elif operation == "query_narrative":
                            output = connector.query_narrative_forecaster(
                                message_data,
                                context=config.get("context"),
                                timeout=config.get("timeout")
                            )
                            
                        else:
                            self.logger.warning(f"Unknown Pulsar operation: {operation}")
                            output = None
                            
                    elif protocol == "pads":
                        # PADS communication
                        reporter = component["reporter"]
                        
                        # Get operation
                        operation = config.get("operation", "report")
                        
                        if operation == "report_market_signal":
                            # Get inputs
                            symbol = config.get("symbol", "unknown")
                            value = config.get("value", 0.5)
                            
                            # Get from state if not provided
                            if config.get("use_input", False):
                                input_name = config.get("input", "default")
                                input_data = state["inputs"].get(input_name)
                                
                                if input_data is None:
                                    # Try to get from intermediate outputs
                                    input_data = state["intermediate"].get(input_name)
                                    
                                if input_data is not None:
                                    if isinstance(input_data, dict):
                                        symbol = input_data.get("symbol", symbol)
                                        value = input_data.get("value", value)
                                        
                            # Create and report signal
                            signal = reporter.create_market_signal(
                                symbol, value,
                                confidence=config.get("confidence", 0.7),
                                timeframe=config.get("timeframe"),
                                source_component=config.get("source_component", "cdfa"),
                                data=config.get("data"),
                                tags=config.get("tags")
                            )
                            
                            reporter.report_signal(signal)
                            output = {"signal_id": signal.id}
                            
                        elif operation == "report_trading_signal":
                            # Create and report trading signal
                            signal = reporter.create_trading_signal(
                                config.get("symbol", "unknown"),
                                config.get("action", "hold"),
                                config.get("strength", 0.5),
                                confidence=config.get("confidence", 0.7),
                                timeframe=config.get("timeframe"),
                                source_component=config.get("source_component", "cdfa"),
                                data=config.get("data"),
                                tags=config.get("tags")
                            )
                            
                            reporter.report_signal(signal)
                            output = {"signal_id": signal.id}
                            
                        elif operation == "report_regime_signal":
                            # Create and report regime signal
                            signal = reporter.create_market_regime_signal(
                                config.get("symbol", "unknown"),
                                config.get("regime", "unknown"),
                                transition_probability=config.get("transition_probability", 0.0),
                                timeframe=config.get("timeframe"),
                                source_component=config.get("source_component", "cdfa"),
                                data=config.get("data"),
                                tags=config.get("tags")
                            )
                            
                            reporter.report_signal(signal)
                            output = {"signal_id": signal.id}
                            
                        else:
                            self.logger.warning(f"Unknown PADS operation: {operation}")
                            output = None
                            
                    else:
                        self.logger.warning(f"Unknown external protocol: {protocol}")
                        output = None
                        
                    # Store output if any
                    if output is not None:
                        output_name = config.get("output", component_name)
                        state["intermediate"][output_name] = output
                        
                        # Add to final outputs if specified
                        if config.get("is_output", False):
                            state["outputs"][output_name] = output
                            
                elif component_type == "custom":
                    # Run custom component
                    process_func = component.get("process_function")
                    config = component["config"]
                    
                    if callable(process_func):
                        # Get input
                        input_name = config.get("input", "default")
                        custom_input = state["inputs"].get(input_name)
                        
                        if custom_input is None:
                            # Try to get from intermediate outputs
                            custom_input = state["intermediate"].get(input_name)
                            
                        # Pass state to function
                        output = process_func(custom_input, state, config)
                        
                        # Store output
                        output_name = config.get("output", component_name)
                        state["intermediate"][output_name] = output
                        
                        # Add to final outputs if specified
                        if config.get("is_output", False):
                            state["outputs"][output_name] = output
                    else:
                        self.logger.warning(f"No process function for custom component '{component_name}'")
                        
            except Exception as e:
                self.logger.error(f"Error processing component '{component_name}': {e}")
                
        return state["outputs"]
    
    # ----- Helper Methods -----
    
    def create_pipeline_config(self, components: List[Dict[str, Any]], 
                            name: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a pipeline configuration from component configurations.
        
        Args:
            components: List of component configurations
            name: Optional pipeline name
            
        Returns:
            Pipeline configuration
        """
        # Generate name if not provided
        if name is None:
            name = f"pipeline_{uuid.uuid4().hex[:8]}"
            
        # Create pipeline configuration
        pipeline_config = {
            "name": name,
            "optimization_level": self.config["optimization_level"],
            "components": components
        }
        
        return pipeline_config
    
    def save_pipeline(self, pipeline_name: str, path: str) -> str:
        """
        Save a pipeline configuration to disk.
        
        Args:
            pipeline_name: Name of the pipeline
            path: Save path
            
        Returns:
            Full save path
        """
        # Get pipeline configuration
        with self._lock:
            pipeline = self._optimized_pipelines.get(pipeline_name)
            
        if pipeline is None:
            self.logger.error(f"Pipeline '{pipeline_name}' not found")
            return ""
            
        # Serialize pipeline (excluding non-serializable objects)
        serializable_pipeline = {
            "name": pipeline["name"],
            "optimization_level": pipeline["optimization_level"].value 
                if isinstance(pipeline["optimization_level"], OptimizationLevel) 
                else pipeline["optimization_level"],
            "components": []
        }
        
        # Extract serializable component info
        for component in pipeline["components"]:
            serializable_component = {
                "type": component["type"],
                "name": component["name"],
                "config": component.get("config", {})
            }
            
            # Add component-specific info
            if component["type"] == "model":
                serializable_component["format"] = component["format"]
                
                # Save model to file
                if component.get("optimized_model") is not None and TORCH_AVAILABLE:
                    try:
                        model_dir = os.path.join(os.path.dirname(path), "models")
                        os.makedirs(model_dir, exist_ok=True)
                        
                        model_filename = f"{component['name']}.pt"
                        model_path = os.path.join(model_dir, model_filename)
                        
                        if component["format"] == "torchscript":
                            # Save TorchScript model
                            self.hw_accelerator.export_torchscript_model(
                                component["optimized_model"],
                                model_path
                            )
                            
                            serializable_component["model_file"] = os.path.join("models", model_filename)
                    except Exception as e:
                        self.logger.error(f"Error saving model: {e}")
                        
            serializable_pipeline["components"].append(serializable_component)
            
        # Create directory if needed
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        
        # Save to file
        with open(path, 'w') as f:
            json.dump(serializable_pipeline, f, indent=2)
            
        self.logger.info(f"Saved pipeline '{pipeline_name}' to {path}")
        return path
    
    def load_pipeline(self, path: str) -> str:
        """
        Load a pipeline configuration from disk.
        
        Args:
            path: Load path
            
        Returns:
            Name of the loaded pipeline
        """
        try:
            # Load pipeline configuration
            with open(path, 'r') as f:
                pipeline = json.load(f)
                
            # Get pipeline name
            name = pipeline.get("name", f"pipeline_{uuid.uuid4().hex[:8]}")
            
            # Convert optimization level to enum
            optimization_level = pipeline.get("optimization_level", self.config["optimization_level"])
            try:
                optimization_level = OptimizationLevel(optimization_level)
            except ValueError:
                optimization_level = OptimizationLevel.STANDARD
                
            # Create reconstructed pipeline
            reconstructed_pipeline = {
                "name": name,
                "optimization_level": optimization_level,
                "components": []
            }
            
            # Reconstruct components
            for component in pipeline.get("components", []):
                reconstructed_component = {
                    "type": component["type"],
                    "name": component["name"],
                    "config": component.get("config", {})
                }
                
                # Handle component-specific reconstruction
                if component["type"] == "model" and TORCH_AVAILABLE:
                    # Try to load model
                    model_file = component.get("model_file")
                    
                    if model_file:
                        try:
                            model_path = os.path.join(os.path.dirname(path), model_file)
                            
                            if component["format"] == "torchscript":
                                # Load TorchScript model
                                model = self.hw_accelerator.load_torchscript_model(model_path)
                                
                                # Add to component
                                reconstructed_component["optimized_model"] = model
                                reconstructed_component["format"] = "torchscript"
                        except Exception as e:
                            self.logger.error(f"Error loading model: {e}")
                            
                elif component["type"] == "wavelet":
                    # Add wavelet processor
                    reconstructed_component["processor"] = self.wavelet_processor
                    
                elif component["type"] == "neuromorphic" and self.neuromorphic_analyzer is not None:
                    # Add neuromorphic analyzer
                    reconstructed_component["analyzer"] = self.neuromorphic_analyzer
                    
                elif component["type"] == "cross_asset":
                    # Add cross-asset analyzer
                    reconstructed_component["analyzer"] = self.cross_asset_analyzer
                    
                elif component["type"] == "visualization" and self.visualization_engine is not None:
                    # Add visualization engine
                    reconstructed_component["engine"] = self.visualization_engine
                    
                elif component["type"] == "external":
                    # Add external connector
                    protocol = component.get("protocol", "")
                    
                    if protocol == "pulsar" and self.pulsar_connector is not None:
                        reconstructed_component["connector"] = self.pulsar_connector
                        reconstructed_component["protocol"] = "pulsar"
                    elif protocol == "pads" and self.pads_reporter is not None:
                        reconstructed_component["reporter"] = self.pads_reporter
                        reconstructed_component["protocol"] = "pads"
                        
                reconstructed_pipeline["components"].append(reconstructed_component)
                
            # Store reconstructed pipeline
            with self._lock:
                self._optimized_pipelines[name] = reconstructed_pipeline
                
            self.logger.info(f"Loaded pipeline '{name}' from {path}")
            return name
            
        except Exception as e:
            self.logger.error(f"Error loading pipeline: {e}")
            return ""
    
    def cleanup(self):
        """Clean up resources and connections."""
        # Stop background processing threads
        self._is_running = False
        
        # Clean up hardware accelerator
        if self.hw_accelerator is not None:
            pass  # No specific cleanup needed
            
        # Clean up external connections
        if self.pulsar_connector is not None:
            self.pulsar_connector.disconnect()
            
        if self.pads_reporter is not None:
            self.pads_reporter.disconnect()
            
        # Clean up analyzers
        if self.neuromorphic_analyzer is not None:
            pass  # No specific cleanup needed
            
        if self.cross_asset_analyzer is not None:
            if hasattr(self.cross_asset_analyzer, 'stop'):
                self.cross_asset_analyzer.stop()
                
        self.logger.info("CDFA Optimizer cleaned up")