#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TorchScript Fusion Module for CDFA Extensions

Provides hardware-agnostic acceleration for the CDFA fusion algorithms using TorchScript:
- GPU-accelerated fusion operations for AMD, NVIDIA, and Apple Silicon
- Optimized models for different fusion types (score, rank, hybrid, layered)
- JIT compilation for fusion operations
- Quantized models for reduced memory footprint and faster inference
- Inference-only optimized models for deployment

Author: Created on May 6, 2025
"""

import logging
import time
import numpy as np
import json
import threading
from typing import Dict, List, Tuple, Optional, Union, Any, Callable, Set
from enum import Enum, auto
from dataclasses import dataclass, field
import warnings
import os
from datetime import datetime, timedelta
import uuid

# Import from cdfa_extensions
from .hw_acceleration import HardwareAccelerator

# ---- Optional dependencies with graceful fallbacks ----

# PyTorch for TorchScript
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. TorchScript fusion will be disabled.", DeprecationWarning, DeprecationWarning)

class FusionType(Enum):
    """Types of fusion algorithms."""
    SCORE = auto()        # Score-based fusion
    RANK = auto()         # Rank-based fusion
    HYBRID = auto()       # Hybrid score/rank fusion
    WEIGHTED = auto()     # Weighted fusion
    LAYERED = auto()      # Layered fusion
    ADAPTIVE = auto()     # Adaptive fusion
    
    def __str__(self) -> str:
        return self.name.lower()
    
    @classmethod
    def from_string(cls, s: str) -> 'FusionType':
        """Create enum from string representation."""
        s_upper = s.upper()
        for item in cls:
            if item.name == s_upper:
                return item
        for item in cls:
            if item.name.startswith(s_upper):
                return item
        raise ValueError(f"Unknown FusionType: {s}")

@dataclass
class FusionResult:
    """Result of signal fusion operation."""
    fused_signal: np.ndarray           # Fused signal values
    confidence: np.ndarray             # Confidence values
    individual_signals: Dict[str, np.ndarray]  # Original signals
    weights: Dict[str, np.ndarray]     # Weights applied to each signal
    fusion_type: FusionType            # Type of fusion used
    metadata: Dict[str, Any] = field(default_factory=dict)

class TorchScriptFusion:
    """
    TorchScript implementation of CDFA fusion algorithms.
    
    Provides hardware-accelerated implementations of various signal fusion 
    methods using TorchScript for CPU, NVIDIA CUDA, AMD ROCm, and Apple Metal.
    """
    
    def __init__(self, hw_accelerator: Optional[HardwareAccelerator] = None,
                config: Optional[Dict[str, Any]] = None,
                log_level: int = logging.INFO):
        """
        Initialize the TorchScript fusion module.
        
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
            # Fusion parameters
            "default_fusion_type": "hybrid",
            "score_alpha": 0.5,  # Weight between score and rank for hybrid fusion
            "min_weight": 0.01,  # Minimum weight for signals
            "confidence_factor": 0.7,  # How much to factor in confidence
            "diversity_factor": 0.3,  # How much to factor in diversity
            "use_nonlinear_weighting": True,  # Use nonlinear weighting
            "nonlinear_exponent": 2.0,  # Exponent for nonlinear weighting
            
            # Optimization parameters
            "use_jit": True,  # Use JIT compilation
            "use_script": True,  # Use script instead of trace when possible
            "use_quantization": False,  # Use quantized models
            "quantization_dtype": "qint8",  # Quantization data type
            "quantization_scheme": "per_tensor_affine",  # Quantization scheme
            "optimization_level": 3,  # Optimization level (0-3)
            "compile_with_backend": True,  # Use torch.compile with appropriate backend
            "fusion_chunk_size": 100,  # Chunk size for processing large inputs
            
            # Caching parameters
            "cache_models": True,  # Cache compiled models
            "cache_ttl": 3600,  # Cache TTL in seconds
            "reuse_tensors": True,  # Reuse tensors to reduce memory allocation
            
            # Other parameters
            "device": None,  # None for auto-detection
            "default_dtype": "float32",  # Default data type
            "log_compilation": True,  # Log compilation time
            "log_inference": False  # Log inference time
        }
        
        # Merge with provided config
        self.config = {**self.default_config, **(config or {})}
        
        # Check if PyTorch is available
        self.has_torch = TORCH_AVAILABLE
        if not self.has_torch:
            self.logger.warning("PyTorch not available! All methods will return None.")
            return
            
        # Initialize state
        self._lock = threading.RLock()
        self._models = {}  # {fusion_type: model}
        self._cached_tensors = {}  # {shape: tensor}
        self._compilation_times = {}  # {fusion_type: time}
        self._inference_times = {}  # {fusion_type: [times]}
        
        # Detect torch.compile backend
        self.compile_backend = self._detect_compile_backend()
        
        # Initialize models
        self._initialize_models()
        
        self.logger.info(f"TorchScriptFusion initialized with device: {self.hw_accelerator.get_torch_device()}")
        if self.compile_backend:
            self.logger.info(f"Using torch.compile with backend: {self.compile_backend}")
    
    def _detect_compile_backend(self) -> Optional[str]:
        """Detect the best backend for torch.compile."""
        if not self.has_torch:
            return None
            
        # Check if torch.compile is available (PyTorch 2.0+)
        if not hasattr(torch, 'compile'):
            return None
            
        # Detect appropriate backend
        if torch.cuda.is_available():
            return "inductor"  # Best for CUDA
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "aot_eager"  # For MPS (Apple Silicon)
        else:
            # Check for ROCm
            if hasattr(torch.version, 'hip') and torch.version.hip is not None:
                return "inductor"  # Should work for ROCm
                
        # Default backend for CPU
        return "inductor"
    
    def _initialize_models(self):
        """Initialize TorchScript fusion models."""
        if not self.has_torch:
            return
            
        # Initialize models for each fusion type
        for fusion_type in FusionType:
            self._create_fusion_model(fusion_type)
    
    def _create_fusion_model(self, fusion_type: FusionType) -> Optional[torch.jit.ScriptModule]:
        """
        Create and compile a TorchScript fusion model.
        
        Args:
            fusion_type: Type of fusion model to create
            
        Returns:
            Compiled TorchScript model or None if failed
        """
        if not self.has_torch:
            return None
            
        # Check if model already exists
        with self._lock:
            existing_model = self._models.get(fusion_type)
            if existing_model is not None:
                return existing_model
                
        try:
            # Start compilation timer
            start_time = time.time()
            
            # Create appropriate model class based on fusion type
            if fusion_type == FusionType.SCORE:
                model = self._create_score_fusion_model()
            elif fusion_type == FusionType.RANK:
                model = self._create_rank_fusion_model()
            elif fusion_type == FusionType.HYBRID:
                model = self._create_hybrid_fusion_model()
            elif fusion_type == FusionType.WEIGHTED:
                model = self._create_weighted_fusion_model()
            elif fusion_type == FusionType.LAYERED:
                model = self._create_layered_fusion_model()
            elif fusion_type == FusionType.ADAPTIVE:
                model = self._create_adaptive_fusion_model()
            else:
                self.logger.error(f"Unknown fusion type: {fusion_type}")
                return None
                
            # Move model to device
            device = self.hw_accelerator.get_torch_device()
            model = model.to(device)
            
            # Apply optimization
            if self.config["use_jit"]:
                # Create example inputs for JIT compilation
                # Assume 5 signals, each with 100 values and confidence
                example_signals = torch.randn(5, 100, device=device)
                example_confidences = torch.rand(5, 100, device=device)
                
                if self.config["use_script"]:
                    # Use script if possible (better for control flow)
                    try:
                        scripted_model = torch.jit.script(model)
                        model = scripted_model
                    except Exception as e:
                        self.logger.warning(f"Failed to script model: {e}")
                        # Fallback to tracing
                        traced_model = torch.jit.trace(model, (example_signals, example_confidences))
                        model = traced_model
                else:
                    # Use trace directly
                    traced_model = torch.jit.trace(model, (example_signals, example_confidences))
                    model = traced_model
                    
            # Apply torch.compile if available and enabled
            if self.config["compile_with_backend"] and self.compile_backend and hasattr(torch, 'compile'):
                try:
                    model = torch.compile(
                        model, 
                        backend=self.compile_backend,
                        mode="reduce-overhead" if self.config["optimization_level"] >= 2 else "default"
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to compile model with {self.compile_backend}: {e}")
                    
            # Apply quantization if enabled
            if self.config["use_quantization"]:
                try:
                    # Get quantization parameters
                    dtype = self.config["quantization_dtype"]
                    scheme = self.config["quantization_scheme"]
                    
                    # Prepare for static quantization
                    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
                    torch.quantization.prepare(model, inplace=True)
                    
                    # Calibrate with example data (TODO: Add real calibration)
                    with torch.no_grad():
                        model(example_signals, example_confidences)
                        
                    # Convert to quantized model
                    quantized_model = torch.quantization.convert(model, inplace=False)
                    model = quantized_model
                    
                except Exception as e:
                    self.logger.warning(f"Failed to quantize model: {e}")
                    
            # Log compilation time
            end_time = time.time()
            compilation_time = end_time - start_time
            self._compilation_times[fusion_type] = compilation_time
            
            if self.config["log_compilation"]:
                self.logger.info(f"Compiled {fusion_type} fusion model in {compilation_time:.2f} seconds")
                
            # Cache model
            with self._lock:
                self._models[fusion_type] = model
                
            return model
            
        except Exception as e:
            self.logger.error(f"Error creating fusion model: {e}")
            return None
    
    def _create_score_fusion_model(self) -> nn.Module:
        """
        Create a score-based fusion model.
        
        Returns:
            PyTorch module for score fusion
        """
        # Define model class for score-based fusion
        class ScoreFusionModel(nn.Module):
            def __init__(self, min_weight=0.01, use_nonlinear=True, nonlinear_exponent=2.0):
                super(ScoreFusionModel, self).__init__()
                self.min_weight = min_weight
                self.use_nonlinear = use_nonlinear
                self.nonlinear_exponent = nonlinear_exponent
                
            def forward(self, signals, confidences):
                """
                Perform score-based fusion.
                
                Args:
                    signals: Tensor of shape [num_signals, sequence_length]
                    confidences: Tensor of shape [num_signals, sequence_length]
                    
                Returns:
                    Tuple of (fused_signal, fused_confidence, weights)
                """
                # Get shape information
                num_signals, seq_length = signals.shape
                
                # Calculate weights based on confidences
                weights = confidences.clone()
                
                # Apply nonlinear weighting if enabled
                if self.use_nonlinear:
                    weights = weights ** self.nonlinear_exponent
                    
                # Ensure minimum weight
                weights = torch.clamp(weights, min=self.min_weight)
                
                # Normalize weights to sum to 1 for each time step
                weights_sum = torch.sum(weights, dim=0, keepdim=True)
                weights = weights / torch.clamp(weights_sum, min=1e-8)
                
                # Apply weighted combination of signals
                weighted_signals = signals * weights
                fused_signal = torch.sum(weighted_signals, dim=0)
                
                # Calculate fused confidence
                # Weighted average of confidences
                fused_confidence = torch.sum(confidences * weights, dim=0)
                
                return fused_signal, fused_confidence, weights
                
        # Create and return model instance
        return ScoreFusionModel(
            min_weight=self.config["min_weight"],
            use_nonlinear=self.config["use_nonlinear_weighting"],
            nonlinear_exponent=self.config["nonlinear_exponent"]
        )
    
    def _create_rank_fusion_model(self) -> nn.Module:
        """
        Create a rank-based fusion model.
        
        Returns:
            PyTorch module for rank fusion
        """
        # Define model class for rank-based fusion
        class RankFusionModel(nn.Module):
            def __init__(self, min_weight=0.01):
                super(RankFusionModel, self).__init__()
                self.min_weight = min_weight
                
            def forward(self, signals, confidences):
                """
                Perform rank-based fusion.
                
                Args:
                    signals: Tensor of shape [num_signals, sequence_length]
                    confidences: Tensor of shape [num_signals, sequence_length]
                    
                Returns:
                    Tuple of (fused_signal, fused_confidence, weights)
                """
                # Get shape information
                num_signals, seq_length = signals.shape
                
                # Calculate ranks for each signal at each time step
                ranks = torch.zeros_like(signals)
                
                # For each time step
                for t in range(seq_length):
                    # Get signals at this time step
                    signals_t = signals[:, t]
                    
                    # Calculate ranks (using argsort of argsort)
                    sorted_indices = torch.argsort(signals_t)
                    ranks_t = torch.argsort(sorted_indices).float() / max(1, (num_signals - 1))
                    
                    # Store ranks
                    ranks[:, t] = ranks_t
                    
                # Combine ranks using confidence-weighted average
                weights = torch.clamp(confidences, min=self.min_weight)
                
                # Normalize weights to sum to 1 for each time step
                weights_sum = torch.sum(weights, dim=0, keepdim=True)
                weights = weights / torch.clamp(weights_sum, min=1e-8)
                
                # Apply weighted combination of ranks
                weighted_ranks = ranks * weights
                fused_rank = torch.sum(weighted_ranks, dim=0)
                
                # Calculate fused confidence
                # Weighted average of confidences
                fused_confidence = torch.sum(confidences * weights, dim=0)
                
                return fused_rank, fused_confidence, weights
                
        # Create and return model instance
        return RankFusionModel(min_weight=self.config["min_weight"])
    
    def _create_hybrid_fusion_model(self) -> nn.Module:
        """
        Create a hybrid score/rank fusion model.
        
        Returns:
            PyTorch module for hybrid fusion
        """
        # Define model class for hybrid fusion
        class HybridFusionModel(nn.Module):
            def __init__(self, alpha=0.5, min_weight=0.01, use_nonlinear=True, nonlinear_exponent=2.0):
                super(HybridFusionModel, self).__init__()
                self.alpha = alpha  # Weight between score and rank
                self.min_weight = min_weight
                self.use_nonlinear = use_nonlinear
                self.nonlinear_exponent = nonlinear_exponent
                
                # Create submodels
                self.score_model = ScoreFusionModel(min_weight, use_nonlinear, nonlinear_exponent)
                self.rank_model = RankFusionModel(min_weight)
                
            def forward(self, signals, confidences):
                """
                Perform hybrid score/rank fusion.
                
                Args:
                    signals: Tensor of shape [num_signals, sequence_length]
                    confidences: Tensor of shape [num_signals, sequence_length]
                    
                Returns:
                    Tuple of (fused_signal, fused_confidence, weights)
                """
                # Perform score-based fusion
                score_signal, score_confidence, score_weights = self.score_model(signals, confidences)
                
                # Perform rank-based fusion
                rank_signal, rank_confidence, rank_weights = self.rank_model(signals, confidences)
                
                # Combine score and rank fusion results
                fused_signal = self.alpha * score_signal + (1 - self.alpha) * rank_signal
                fused_confidence = self.alpha * score_confidence + (1 - self.alpha) * rank_confidence
                
                # Combine weights (use score weights for now)
                weights = score_weights
                
                return fused_signal, fused_confidence, weights
                
        # Create Score and Rank fusion model classes inside the scope
        class ScoreFusionModel(nn.Module):
            def __init__(self, min_weight, use_nonlinear, nonlinear_exponent):
                super(ScoreFusionModel, self).__init__()
                self.min_weight = min_weight
                self.use_nonlinear = use_nonlinear
                self.nonlinear_exponent = nonlinear_exponent
                
            def forward(self, signals, confidences):
                # Get shape information
                num_signals, seq_length = signals.shape
                
                # Calculate weights based on confidences
                weights = confidences.clone()
                
                # Apply nonlinear weighting if enabled
                if self.use_nonlinear:
                    weights = weights ** self.nonlinear_exponent
                    
                # Ensure minimum weight
                weights = torch.clamp(weights, min=self.min_weight)
                
                # Normalize weights to sum to 1 for each time step
                weights_sum = torch.sum(weights, dim=0, keepdim=True)
                weights = weights / torch.clamp(weights_sum, min=1e-8)
                
                # Apply weighted combination of signals
                weighted_signals = signals * weights
                fused_signal = torch.sum(weighted_signals, dim=0)
                
                # Calculate fused confidence
                fused_confidence = torch.sum(confidences * weights, dim=0)
                
                return fused_signal, fused_confidence, weights
                
        class RankFusionModel(nn.Module):
            def __init__(self, min_weight):
                super(RankFusionModel, self).__init__()
                self.min_weight = min_weight
                
            def forward(self, signals, confidences):
                # Get shape information
                num_signals, seq_length = signals.shape
                
                # Calculate ranks for each signal at each time step
                ranks = torch.zeros_like(signals)
                
                # For each time step
                for t in range(seq_length):
                    # Get signals at this time step
                    signals_t = signals[:, t]
                    
                    # Calculate ranks (using argsort of argsort)
                    sorted_indices = torch.argsort(signals_t)
                    ranks_t = torch.argsort(sorted_indices).float() / max(1, (num_signals - 1))
                    
                    # Store ranks
                    ranks[:, t] = ranks_t
                    
                # Combine ranks using confidence-weighted average
                weights = torch.clamp(confidences, min=self.min_weight)
                
                # Normalize weights to sum to 1 for each time step
                weights_sum = torch.sum(weights, dim=0, keepdim=True)
                weights = weights / torch.clamp(weights_sum, min=1e-8)
                
                # Apply weighted combination of ranks
                weighted_ranks = ranks * weights
                fused_rank = torch.sum(weighted_ranks, dim=0)
                
                # Calculate fused confidence
                fused_confidence = torch.sum(confidences * weights, dim=0)
                
                return fused_rank, fused_confidence, weights
                
        # Create and return model instance
        return HybridFusionModel(
            alpha=self.config["score_alpha"],
            min_weight=self.config["min_weight"],
            use_nonlinear=self.config["use_nonlinear_weighting"],
            nonlinear_exponent=self.config["nonlinear_exponent"]
        )
    
    def _create_weighted_fusion_model(self) -> nn.Module:
        """
        Create a weighted fusion model.
        
        Returns:
            PyTorch module for weighted fusion
        """
        # Define model class for weighted fusion
        class WeightedFusionModel(nn.Module):
            def __init__(self, confidence_factor=0.7, diversity_factor=0.3, min_weight=0.01):
                super(WeightedFusionModel, self).__init__()
                self.confidence_factor = confidence_factor
                self.diversity_factor = diversity_factor
                self.min_weight = min_weight
                
            def forward(self, signals, confidences):
                """
                Perform weighted fusion using confidence and diversity.
                
                Args:
                    signals: Tensor of shape [num_signals, sequence_length]
                    confidences: Tensor of shape [num_signals, sequence_length]
                    
                Returns:
                    Tuple of (fused_signal, fused_confidence, weights)
                """
                # Get shape information
                num_signals, seq_length = signals.shape
                
                # Calculate weights based on confidences
                conf_weights = confidences.clone()
                
                # Calculate diversity weights
                div_weights = torch.ones_like(confidences)
                
                # For each signal
                for i in range(num_signals):
                    # Calculate average correlation with other signals
                    corr_sum = torch.zeros(seq_length, device=signals.device)
                    count = 0
                    
                    for j in range(num_signals):
                        if i != j:
                            # Simplified correlation calculation
                            # Normalize signals for more stable correlation
                            si_norm = signals[i] - torch.mean(signals[i])
                            sj_norm = signals[j] - torch.mean(signals[j])
                            
                            # Avoid division by zero
                            si_std = torch.std(si_norm) + 1e-8
                            sj_std = torch.std(sj_norm) + 1e-8
                            
                            si_norm = si_norm / si_std
                            sj_norm = sj_norm / sj_std
                            
                            # Calculate correlation
                            corr = torch.sum(si_norm * sj_norm) / seq_length
                            
                            # Accumulate correlation
                            corr_sum += corr
                            count += 1
                            
                    # Calculate average correlation
                    avg_corr = corr_sum / max(1, count)
                    
                    # Convert to diversity (1 - |correlation|)
                    diversity = 1.0 - torch.abs(avg_corr)
                    
                    # Set diversity weight
                    div_weights[i] = diversity
                    
                # Combine confidence and diversity weights
                weights = (self.confidence_factor * conf_weights + 
                         self.diversity_factor * div_weights)
                
                # Ensure minimum weight
                weights = torch.clamp(weights, min=self.min_weight)
                
                # Normalize weights to sum to 1 for each time step
                weights_sum = torch.sum(weights, dim=0, keepdim=True)
                weights = weights / torch.clamp(weights_sum, min=1e-8)
                
                # Apply weighted combination of signals
                weighted_signals = signals * weights
                fused_signal = torch.sum(weighted_signals, dim=0)
                
                # Calculate fused confidence
                # Weighted average of confidences
                fused_confidence = torch.sum(confidences * weights, dim=0)
                
                return fused_signal, fused_confidence, weights
                
        # Create and return model instance
        return WeightedFusionModel(
            confidence_factor=self.config["confidence_factor"],
            diversity_factor=self.config["diversity_factor"],
            min_weight=self.config["min_weight"]
        )
    
    def _create_layered_fusion_model(self) -> nn.Module:
        """
        Create a layered fusion model.
        
        Returns:
            PyTorch module for layered fusion
        """
        # Define model class for layered fusion
        class LayeredFusionModel(nn.Module):
            def __init__(self, min_weight=0.01, use_nonlinear=True, nonlinear_exponent=2.0):
                super(LayeredFusionModel, self).__init__()
                self.min_weight = min_weight
                self.use_nonlinear = use_nonlinear
                self.nonlinear_exponent = nonlinear_exponent
                
            def forward(self, signals, confidences):
                """
                Perform layered fusion (sub-groups then final fusion).
                
                Args:
                    signals: Tensor of shape [num_signals, sequence_length]
                    confidences: Tensor of shape [num_signals, sequence_length]
                    
                Returns:
                    Tuple of (fused_signal, fused_confidence, weights)
                """
                # Get shape information
                num_signals, seq_length = signals.shape
                
                # If only 1 or 2 signals, just do simple score fusion
                if num_signals <= 2:
                    return self._score_fusion(signals, confidences)
                    
                # Split signals into two groups
                group_size = num_signals // 2
                
                # Group 1
                signals1 = signals[:group_size]
                confidences1 = confidences[:group_size]
                
                # Group 2
                signals2 = signals[group_size:]
                confidences2 = confidences[group_size:]
                
                # Fuse each group
                fused1, conf1, weights1 = self._score_fusion(signals1, confidences1)
                fused2, conf2, weights2 = self._score_fusion(signals2, confidences2)
                
                # Combine group results
                group_signals = torch.stack([fused1, fused2])
                group_confidences = torch.stack([conf1, conf2])
                
                # Final fusion
                fused_signal, fused_confidence, group_weights = self._score_fusion(
                    group_signals, group_confidences
                )
                
                # Combine weights for all signals
                # (this is approximation as we lose the exact mapping)
                combined_weights = torch.zeros_like(signals)
                
                # Group 1 weights
                w1 = weights1 * group_weights[0].unsqueeze(0)
                combined_weights[:group_size] = w1
                
                # Group 2 weights
                w2 = weights2 * group_weights[1].unsqueeze(0)
                combined_weights[group_size:] = w2
                
                return fused_signal, fused_confidence, combined_weights
                
            def _score_fusion(self, signals, confidences):
                """Simple score fusion helper."""
                # Get shape information
                num_signals, seq_length = signals.shape
                
                # Calculate weights based on confidences
                weights = confidences.clone()
                
                # Apply nonlinear weighting if enabled
                if self.use_nonlinear:
                    weights = weights ** self.nonlinear_exponent
                    
                # Ensure minimum weight
                weights = torch.clamp(weights, min=self.min_weight)
                
                # Normalize weights to sum to 1 for each time step
                weights_sum = torch.sum(weights, dim=0, keepdim=True)
                weights = weights / torch.clamp(weights_sum, min=1e-8)
                
                # Apply weighted combination of signals
                weighted_signals = signals * weights
                fused_signal = torch.sum(weighted_signals, dim=0)
                
                # Calculate fused confidence
                fused_confidence = torch.sum(confidences * weights, dim=0)
                
                return fused_signal, fused_confidence, weights
                
        # Create and return model instance
        return LayeredFusionModel(
            min_weight=self.config["min_weight"],
            use_nonlinear=self.config["use_nonlinear_weighting"],
            nonlinear_exponent=self.config["nonlinear_exponent"]
        )
    
    def _create_adaptive_fusion_model(self) -> nn.Module:
        """
        Create an adaptive fusion model.
        
        Returns:
            PyTorch module for adaptive fusion
        """
        # Define model class for adaptive fusion
        class AdaptiveFusionModel(nn.Module):
            def __init__(self, min_weight=0.01, use_nonlinear=True, nonlinear_exponent=2.0,
                       confidence_factor=0.7, diversity_factor=0.3):
                super(AdaptiveFusionModel, self).__init__()
                self.min_weight = min_weight
                self.use_nonlinear = use_nonlinear
                self.nonlinear_exponent = nonlinear_exponent
                self.confidence_factor = confidence_factor
                self.diversity_factor = diversity_factor
                
                # Create submodels for different fusion methods
                self.score_model = ScoreFusionModel(min_weight, use_nonlinear, nonlinear_exponent)
                self.rank_model = RankFusionModel(min_weight)
                self.weighted_model = WeightedFusionModel(confidence_factor, diversity_factor, min_weight)
                
            def forward(self, signals, confidences):
                """
                Perform adaptive fusion based on signal properties.
                
                Args:
                    signals: Tensor of shape [num_signals, sequence_length]
                    confidences: Tensor of shape [num_signals, sequence_length]
                    
                Returns:
                    Tuple of (fused_signal, fused_confidence, weights)
                """
                # Get shape information
                num_signals, seq_length = signals.shape
                
                # Calculate signal properties for adaptation
                # 1. Average confidence
                avg_confidence = torch.mean(confidences)
                
                # 2. Signal agreement (inverse of variance)
                signal_mean = torch.mean(signals, dim=0, keepdim=True)
                signal_var = torch.mean((signals - signal_mean) ** 2)
                signal_agreement = 1.0 / (1.0 + signal_var)
                
                # Adapt fusion method based on properties
                # If high confidence and high agreement, use score fusion
                # If low confidence or low agreement, use rank fusion
                # Otherwise, use weighted fusion
                
                # Calculate method weights
                score_weight = avg_confidence * signal_agreement
                rank_weight = (1.0 - avg_confidence) * (1.0 - signal_agreement)
                weighted_weight = 1.0 - score_weight - rank_weight
                
                # Normalize weights
                method_weights = torch.tensor(
                    [score_weight, rank_weight, weighted_weight],
                    device=signals.device
                )
                method_weights = F.softmax(method_weights, dim=0)
                
                # Apply each fusion method
                score_signal, score_conf, score_w = self.score_model(signals, confidences)
                rank_signal, rank_conf, rank_w = self.rank_model(signals, confidences)
                weighted_signal, weighted_conf, weighted_w = self.weighted_model(signals, confidences)
                
                # Combine results with adaptive weights
                fused_signal = (
                    method_weights[0] * score_signal +
                    method_weights[1] * rank_signal +
                    method_weights[2] * weighted_signal
                )
                
                fused_confidence = (
                    method_weights[0] * score_conf +
                    method_weights[1] * rank_conf +
                    method_weights[2] * weighted_conf
                )
                
                # Combine weights (weighted average of all methods)
                weights = (
                    method_weights[0].unsqueeze(-1).unsqueeze(-1) * score_w +
                    method_weights[1].unsqueeze(-1).unsqueeze(-1) * rank_w +
                    method_weights[2].unsqueeze(-1).unsqueeze(-1) * weighted_w
                )
                
                return fused_signal, fused_confidence, weights
                
        # Create Score, Rank, and Weighted fusion model classes inside the scope
        class ScoreFusionModel(nn.Module):
            def __init__(self, min_weight, use_nonlinear, nonlinear_exponent):
                super(ScoreFusionModel, self).__init__()
                self.min_weight = min_weight
                self.use_nonlinear = use_nonlinear
                self.nonlinear_exponent = nonlinear_exponent
                
            def forward(self, signals, confidences):
                # Get shape information
                num_signals, seq_length = signals.shape
                
                # Calculate weights based on confidences
                weights = confidences.clone()
                
                # Apply nonlinear weighting if enabled
                if self.use_nonlinear:
                    weights = weights ** self.nonlinear_exponent
                    
                # Ensure minimum weight
                weights = torch.clamp(weights, min=self.min_weight)
                
                # Normalize weights to sum to 1 for each time step
                weights_sum = torch.sum(weights, dim=0, keepdim=True)
                weights = weights / torch.clamp(weights_sum, min=1e-8)
                
                # Apply weighted combination of signals
                weighted_signals = signals * weights
                fused_signal = torch.sum(weighted_signals, dim=0)
                
                # Calculate fused confidence
                fused_confidence = torch.sum(confidences * weights, dim=0)
                
                return fused_signal, fused_confidence, weights
                
        class RankFusionModel(nn.Module):
            def __init__(self, min_weight):
                super(RankFusionModel, self).__init__()
                self.min_weight = min_weight
                
            def forward(self, signals, confidences):
                # Get shape information
                num_signals, seq_length = signals.shape
                
                # Calculate ranks for each signal at each time step
                ranks = torch.zeros_like(signals)
                
                # For each time step
                for t in range(seq_length):
                    # Get signals at this time step
                    signals_t = signals[:, t]
                    
                    # Calculate ranks (using argsort of argsort)
                    sorted_indices = torch.argsort(signals_t)
                    ranks_t = torch.argsort(sorted_indices).float() / max(1, (num_signals - 1))
                    
                    # Store ranks
                    ranks[:, t] = ranks_t
                    
                # Combine ranks using confidence-weighted average
                weights = torch.clamp(confidences, min=self.min_weight)
                
                # Normalize weights to sum to 1 for each time step
                weights_sum = torch.sum(weights, dim=0, keepdim=True)
                weights = weights / torch.clamp(weights_sum, min=1e-8)
                
                # Apply weighted combination of ranks
                weighted_ranks = ranks * weights
                fused_rank = torch.sum(weighted_ranks, dim=0)
                
                # Calculate fused confidence
                fused_confidence = torch.sum(confidences * weights, dim=0)
                
                return fused_rank, fused_confidence, weights
                
        class WeightedFusionModel(nn.Module):
            def __init__(self, confidence_factor, diversity_factor, min_weight):
                super(WeightedFusionModel, self).__init__()
                self.confidence_factor = confidence_factor
                self.diversity_factor = diversity_factor
                self.min_weight = min_weight
                
            def forward(self, signals, confidences):
                # Get shape information
                num_signals, seq_length = signals.shape
                
                # Calculate weights based on confidences
                conf_weights = confidences.clone()
                
                # Calculate diversity weights
                div_weights = torch.ones_like(confidences)
                
                # For each signal
                for i in range(num_signals):
                    # Calculate average correlation with other signals
                    corr_sum = torch.zeros(seq_length, device=signals.device)
                    count = 0
                    
                    for j in range(num_signals):
                        if i != j:
                            # Simplified correlation calculation
                            # Normalize signals for correlation
                            si_norm = signals[i] - torch.mean(signals[i])
                            sj_norm = signals[j] - torch.mean(signals[j])
                            
                            # Avoid division by zero
                            si_std = torch.std(si_norm) + 1e-8
                            sj_std = torch.std(sj_norm) + 1e-8
                            
                            si_norm = si_norm / si_std
                            sj_norm = sj_norm / sj_std
                            
                            # Calculate correlation
                            corr = torch.sum(si_norm * sj_norm) / seq_length
                            
                            # Accumulate correlation
                            corr_sum += corr
                            count += 1
                            
                    # Calculate average correlation
                    avg_corr = corr_sum / max(1, count)
                    
                    # Convert to diversity (1 - |correlation|)
                    diversity = 1.0 - torch.abs(avg_corr)
                    
                    # Set diversity weight
                    div_weights[i] = diversity
                    
                # Combine confidence and diversity weights
                weights = (self.confidence_factor * conf_weights + 
                         self.diversity_factor * div_weights)
                
                # Ensure minimum weight
                weights = torch.clamp(weights, min=self.min_weight)
                
                # Normalize weights to sum to 1 for each time step
                weights_sum = torch.sum(weights, dim=0, keepdim=True)
                weights = weights / torch.clamp(weights_sum, min=1e-8)
                
                # Apply weighted combination of signals
                weighted_signals = signals * weights
                fused_signal = torch.sum(weighted_signals, dim=0)
                
                # Calculate fused confidence
                fused_confidence = torch.sum(confidences * weights, dim=0)
                
                return fused_signal, fused_confidence, weights
                
        # Create and return model instance
        return AdaptiveFusionModel(
            min_weight=self.config["min_weight"],
            use_nonlinear=self.config["use_nonlinear_weighting"],
            nonlinear_exponent=self.config["nonlinear_exponent"],
            confidence_factor=self.config["confidence_factor"],
            diversity_factor=self.config["diversity_factor"]
        )

    def create_optimized_model(self, model: 'torch.nn.Module', example_input: Any, 
                              optimization_level: Optional[Union[str, int]] = None,
                              use_tracing: Optional[bool] = None,
                              inference_mode: Optional[bool] = True) -> Any:
        """
        Create an optimized TorchScript model from a PyTorch model.
        
        The key insight here is that we need to choose between TorchScript and torch.compile,
        not try to use both. TorchScript creates a serializable, standalone model, while 
        torch.compile is a newer compilation approach that optimizes execution but doesn't
        create a standalone model.
        """
        if not self.has_torch:
            self.logger.error("PyTorch not available, cannot optimize model")
            return None
            
        try:
            # Start compilation timer
            start_time = time.time()
            
            # Get configuration values
            use_tracing = use_tracing if use_tracing is not None else not self.config.get("use_script", False)
            inference_mode = inference_mode if inference_mode is not None else True
            
            # Determine optimization level
            if optimization_level is None:
                optimization_level = self.config.get("optimization_level", 2)
            elif isinstance(optimization_level, str):
                if optimization_level.lower() == "aggressive":
                    optimization_level = 3
                elif optimization_level.lower() == "conservative":
                    optimization_level = 1
                else:
                    try:
                        optimization_level = int(optimization_level)
                    except ValueError:
                        optimization_level = 2
            
            # Move model to device
            device = self.hw_accelerator.get_torch_device()
            model = model.to(device)
            
            # Move example inputs to device
            if isinstance(example_input, tuple) or isinstance(example_input, list):
                example_input_device = tuple(
                    x.to(device) if isinstance(x, torch.Tensor) else x
                    for x in example_input
                )
            elif isinstance(example_input, torch.Tensor):
                example_input_device = example_input.to(device)
            else:
                example_input_device = example_input
            
            # Put model in eval mode for inference
            if inference_mode:
                model.eval()
            
            # Here's the key fix: Choose between TorchScript and torch.compile
            # based on the configuration and requirements
            
            # If we want TorchScript (for serialization, deployment, etc.)
            if self.config.get("use_jit", True):
                if use_tracing:
                    # Use tracing for models without control flow
                    with torch.no_grad():
                        traced_model = torch.jit.trace(model, example_input_device)
                    optimized_model = traced_model
                else:
                    # Use scripting for models with control flow
                    try:
                        scripted_model = torch.jit.script(model)
                        optimized_model = scripted_model
                    except Exception as e:
                        self.logger.warning(f"Scripting failed, falling back to tracing: {e}")
                        # Fallback to tracing
                        with torch.no_grad():
                            traced_model = torch.jit.trace(model, example_input_device)
                        optimized_model = traced_model
                
                # Apply TorchScript optimizations
                if optimization_level >= 2 and hasattr(torch.jit, 'optimize_for_inference'):
                    optimized_model = torch.jit.optimize_for_inference(optimized_model)
                    
            # If we want torch.compile instead (for better performance but no serialization)
            elif optimization_level >= 3 and hasattr(torch, 'compile') and self.config.get("compile_with_backend", False):
                try:
                    backend = self.compile_backend or "inductor"
                    optimized_model = torch.compile(
                        model,
                        backend=backend,
                        mode="reduce-overhead" if optimization_level >= 3 else "default"
                    )
                    self.logger.info(f"Applied torch.compile with backend: {backend}")
                except Exception as e:
                    self.logger.warning(f"Failed to apply torch.compile: {e}")
                    # Fallback to regular model
                    optimized_model = model
            else:
                # No special optimization, just return the model as-is
                optimized_model = model
            
            # Log compilation time
            end_time = time.time()
            compilation_time = end_time - start_time
            
            if self.config.get("log_compilation", True):
                self.logger.info(f"Compiled model in {compilation_time:.2f} seconds (optimization level: {optimization_level})")
            
            return optimized_model
            
        except Exception as e:
            self.logger.error(f"Error optimizing model: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None


    def fuse_signals(self, signals: Dict[str, np.ndarray], confidences: Optional[Dict[str, np.ndarray]] = None,
                  fusion_type: Optional[Union[str, FusionType]] = None) -> FusionResult:
        """
        Fuse multiple signals using TorchScript acceleration.
        
        Args:
            signals: Dictionary of signal name to value array
            confidences: Dictionary of signal name to confidence array (optional)
            fusion_type: Type of fusion to use (default from config)
            
        Returns:
            Fusion result
        """
        if not self.has_torch:
            self.logger.error("PyTorch not available, cannot perform fusion")
            return None
            
        # Get defaults from config if not provided
        if fusion_type is None:
            fusion_type = self.config["default_fusion_type"]
            
        if isinstance(fusion_type, str):
            fusion_type = FusionType.from_string(fusion_type)
            
        # Handle confidences if not provided
        if confidences is None:
            confidences = {name: np.ones_like(values) for name, values in signals.items()}
            
        try:
            # Start inference timer
            start_time = time.time()
            
            # Get model for this fusion type
            model = self._models.get(fusion_type)
            
            # Create or reload model if needed
            if model is None:
                model = self._create_fusion_model(fusion_type)
                
                if model is None:
                    self.logger.error(f"Failed to create model for {fusion_type}")
                    return None
                    
            # Prepare input data
            signal_names = list(signals.keys())
            
            # Check if all signals have the same length
            signal_lengths = [len(signals[name]) for name in signal_names]
            if len(set(signal_lengths)) > 1:
                self.logger.error("All signals must have the same length")
                return None
                
            seq_length = signal_lengths[0]
            num_signals = len(signal_names)
            
            # Convert signals and confidences to tensors
            signals_tensor = torch.zeros((num_signals, seq_length), dtype=torch.float32)
            confidences_tensor = torch.zeros((num_signals, seq_length), dtype=torch.float32)
            
            for i, name in enumerate(signal_names):
                signals_tensor[i] = torch.tensor(signals[name], dtype=torch.float32)
                confidences_tensor[i] = torch.tensor(confidences.get(name, np.ones_like(signals[name])), 
                                                  dtype=torch.float32)
                
            # Move tensors to device
            device = self.hw_accelerator.get_torch_device()
            signals_tensor = signals_tensor.to(device)
            confidences_tensor = confidences_tensor.to(device)
            
            # Process in chunks if data is too large
            chunk_size = self.config["fusion_chunk_size"]
            
            if seq_length <= chunk_size:
                # Process all data at once
                fused_signal, fused_confidence, weights = model(signals_tensor, confidences_tensor)
            else:
                # Process in chunks
                fused_signal_chunks = []
                fused_confidence_chunks = []
                weights_chunks = []
                
                for i in range(0, seq_length, chunk_size):
                    end_idx = min(i + chunk_size, seq_length)
                    chunk_signals = signals_tensor[:, i:end_idx]
                    chunk_confidences = confidences_tensor[:, i:end_idx]
                    
                    chunk_fused, chunk_conf, chunk_weights = model(chunk_signals, chunk_confidences)
                    
                    fused_signal_chunks.append(chunk_fused)
                    fused_confidence_chunks.append(chunk_conf)
                    weights_chunks.append(chunk_weights)
                    
                # Combine chunks
                fused_signal = torch.cat(fused_signal_chunks)
                fused_confidence = torch.cat(fused_confidence_chunks)
                weights = torch.cat(weights_chunks, dim=1)
                
            # Convert back to numpy
            fused_signal_np = fused_signal.cpu().numpy()
            fused_confidence_np = fused_confidence.cpu().numpy()
            weights_np = weights.cpu().numpy()
            
            # End inference timer
            end_time = time.time()
            inference_time = end_time - start_time
            
            # Log inference time if configured
            if self.config["log_inference"]:
                self.logger.info(f"Fusion inference time: {inference_time:.4f} seconds")
                
            # Store inference time history
            with self._lock:
                if fusion_type not in self._inference_times:
                    self._inference_times[fusion_type] = []
                self._inference_times[fusion_type].append(inference_time)
                
            # Create weights dictionary
            weights_dict = {name: weights_np[i] for i, name in enumerate(signal_names)}
            
            # Create result
            result = FusionResult(
                fused_signal=fused_signal_np,
                confidence=fused_confidence_np,
                individual_signals=signals,
                weights=weights_dict,
                fusion_type=fusion_type,
                metadata={
                    "inference_time": inference_time,
                    "fusion_type": str(fusion_type),
                    "num_signals": num_signals,
                    "sequence_length": seq_length,
                    "timestamp": time.time()
                }
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error performing fusion: {e}")
            return None
    
    def get_compilation_stats(self) -> Dict[str, float]:
        """
        Get model compilation statistics.
        
        Returns:
            Dictionary of fusion type to compilation time
        """
        return {str(k): v for k, v in self._compilation_times.items()}
    
    def get_inference_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Get inference statistics.
        
        Returns:
            Dictionary of fusion type to statistics
        """
        stats = {}
        
        for fusion_type, times in self._inference_times.items():
            if times:
                stats[str(fusion_type)] = {
                    "mean": np.mean(times),
                    "min": np.min(times),
                    "max": np.max(times),
                    "median": np.median(times),
                    "count": len(times)
                }
                
        return stats

    def run_model(self, model: Any, input_data: Any) -> Any:
        """
        Run an optimized model with the provided input data.
        
        Args:
            model: Optimized TorchScript model
            input_data: Model input(s) (can be tensor, numpy array, or tuple)
            
        Returns:
            Model output
        """
        if not self.has_torch:
            self.logger.error("PyTorch not available, cannot run model")
            return None
            
        try:
            # Start inference timer
            start_time = time.time()
            
            # Get device
            device = self.hw_accelerator.get_torch_device()
            
            # Convert input data to proper format
            if isinstance(input_data, tuple) or isinstance(input_data, list):
                # Handle tuple of inputs
                device_inputs = []
                for x in input_data:
                    if isinstance(x, np.ndarray):
                        # Convert numpy array to tensor
                        device_inputs.append(torch.tensor(x, device=device))
                    elif isinstance(x, torch.Tensor):
                        # Move tensor to device
                        device_inputs.append(x.to(device))
                    else:
                        # Keep as is
                        device_inputs.append(x)
                model_input = tuple(device_inputs)
            elif isinstance(input_data, np.ndarray):
                # Convert numpy array to tensor
                model_input = torch.tensor(input_data, device=device)
            elif isinstance(input_data, torch.Tensor):
                # Move tensor to device
                model_input = input_data.to(device)
            else:
                # Keep as is
                model_input = input_data
            
            # Run model
            with torch.no_grad():
                try:
                    # Check if the model's forward method expects unpacked arguments
                    # First attempt: try with the tuple directly
                    if isinstance(model_input, tuple):
                        try:
                            output = model(model_input)
                        except (RuntimeError, TypeError) as e:
                            error_msg = str(e)
                            # Check if error indicates argument mismatch
                            if ("argument" in error_msg and "tuple" in error_msg) or \
                               ("expects" in error_msg and "got" in error_msg) or \
                               ("Expected" in error_msg and "found type 'tuple'" in error_msg):
                                # Try unpacking the tuple
                                self.logger.debug("Model expects unpacked arguments, trying to unpack tuple")
                                output = model(*model_input)
                            else:
                                raise
                    else:
                        # Not a tuple, just call directly
                        output = model(model_input)
                except Exception as e:
                    self.logger.error(f"Error running model: {e}")
                    import traceback
                    self.logger.error(traceback.format_exc())
                    return None
            
            # Convert output to numpy if needed
            if isinstance(output, torch.Tensor):
                output = output.cpu().numpy()
            elif isinstance(output, tuple):
                output = tuple(
                    x.cpu().numpy() if isinstance(x, torch.Tensor) else x
                    for x in output
                )
            
            # Log inference time if configured
            end_time = time.time()
            inference_time = end_time - start_time
            
            if self.config.get("log_inference", False):
                self.logger.debug(f"Model inference time: {inference_time:.4f} seconds")
            
            return output
            
        except Exception as e:
            self.logger.error(f"Error running model: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    
    def export_model(self, fusion_type: Union[str, FusionType], path: str) -> bool:
        """
        Export a compiled model to disk.
        
        Args:
            fusion_type: Type of fusion model to export
            path: Path to save the model
            
        Returns:
            Success flag
        """
        if not self.has_torch:
            self.logger.error("PyTorch not available, cannot export model")
            return False
            
        # Convert fusion type if needed
        if isinstance(fusion_type, str):
            fusion_type = FusionType.from_string(fusion_type)
            
        # Get model for this fusion type
        model = self._models.get(fusion_type)
        
        # Create or reload model if needed
        if model is None:
            model = self._create_fusion_model(fusion_type)
            
            if model is None:
                self.logger.error(f"Failed to create model for {fusion_type}")
                return False
                
        try:
            # Save model
            torch.jit.save(model, path)
            self.logger.info(f"Exported {fusion_type} model to {path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting model: {e}")
            return False
    
    def load_model(self, fusion_type: Union[str, FusionType], path: str) -> bool:
        """
        Load a compiled model from disk.
        
        Args:
            fusion_type: Type of fusion model to load
            path: Path to the saved model
            
        Returns:
            Success flag
        """
        if not self.has_torch:
            self.logger.error("PyTorch not available, cannot load model")
            return False
            
        # Convert fusion type if needed
        if isinstance(fusion_type, str):
            fusion_type = FusionType.from_string(fusion_type)
            
        try:
            # Load model
            model = torch.jit.load(path, map_location=self.hw_accelerator.get_torch_device())
            
            # Store model
            with self._lock:
                self._models[fusion_type] = model
                
            self.logger.info(f"Loaded {fusion_type} model from {path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            return False
            
    def clear_cache(self):
        """Clear compiled model cache."""
        with self._lock:
            self._models.clear()
            self._cached_tensors.clear()
            self.logger.info("Cleared model cache")
