#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neuromorphic Analyzer for CDFA Extensions

Provides spiking neural network capabilities for market analysis using:
- Norse for PyTorch-based SNN implementations
- Rockpool for efficient neuromorphic computing
- STDP (Spike-Timing-Dependent Plasticity) for adaptive learning
- Event-driven processing for efficient computation
- Temporal pattern recognition for market regimes

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
import queue
import os
import json
from datetime import datetime, timedelta
import uuid
# Optional neuromorphic libraries
try:
    import norse.torch as norse
    NORSE_AVAILABLE = True
except ImportError:
    norse = None
    NORSE_AVAILABLE = False

try:
    import rockpool as rp
    import rockpool.nn as rnn
    import rockpool.training as rpt
    import rockpool.timeseries as rpts
    ROCKPOOL_AVAILABLE = True
except ImportError:
    rp = None
    rnn = None
    rpt = None
    rpts = None
    ROCKPOOL_AVAILABLE = False
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.dlpack import to_dlpack, from_dlpack

# Import from cdfa_extensions
from .hw_acceleration import HardwareAccelerator

# ---- Optional dependencies with graceful fallbacks ----

# PyTorch for tensor operations
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. Neuromorphic computing will be disabled.", DeprecationWarning, DeprecationWarning)

# Norse for PyTorch-based SNNs
try:
    import norse.torch as norse
    from norse.torch.functional.lif import LIFParameters
    from norse.torch.module.lif import LIFCell, LIFRecurrentCell
    from norse.torch.module.leaky_integrator import LI
    NORSE_AVAILABLE = True
except ImportError:
    NORSE_AVAILABLE = False
    warnings.warn("Norse not available. PyTorch-based SNN will be limited.", DeprecationWarning, DeprecationWarning)

# Rockpool for neuromorphic modeling
try:
    import rockpool
    import rockpool.nn as rnn
    import rockpool.parameters as rp
    ROCKPOOL_AVAILABLE = True
    # Verify Rockpool is working correctly by checking its version
    rockpool_version = getattr(rockpool, "__version__", "unknown")
    if rockpool_version == "unknown":
        # Additional check if version attribute is missing
        try:
            # Try to access a core module or function
            _ = rp.Parameter(1.0)
            ROCKPOOL_AVAILABLE = True
        except Exception as e:
            ROCKPOOL_AVAILABLE = False
            warnings.warn(f"Rockpool found but may be incompatible: {e}", DeprecationWarning, DeprecationWarning)
except ImportError as e:
    ROCKPOOL_AVAILABLE = False
    warnings.warn(f"Rockpool not available: {e}. Advanced neuromorphic capabilities will be limited.", DeprecationWarning, DeprecationWarning)
except Exception as e:
    ROCKPOOL_AVAILABLE = False
    warnings.warn(f"Error loading Rockpool: {e}. Advanced neuromorphic capabilities will be limited.", DeprecationWarning, DeprecationWarning)

# PyWavelets for signal processing
try:
    import pywt
    PYWAVELETS_AVAILABLE = True
except ImportError:
    PYWAVELETS_AVAILABLE = False
    warnings.warn("PyWavelets not available. Wavelet feature extraction will be limited.", DeprecationWarning, DeprecationWarning)

class EncodingMethod(Enum):
    """Methods for encoding market data into spike trains."""
    RATE = auto()      # Rate coding (frequency)
    TEMPORAL = auto()  # Temporal coding (timing)
    POPULATION = auto() # Population coding
    PHASE = auto()     # Phase coding
    BURST = auto()     # Burst coding
    
    def __str__(self) -> str:
        return self.name.lower()
    
    @classmethod
    def from_string(cls, s: str) -> 'EncodingMethod':
        """Create enum from string representation."""
        s_upper = s.upper()
        for item in cls:
            if item.name == s_upper:
                return item
        for item in cls:
            if item.name.startswith(s_upper):
                return item
        raise ValueError(f"Unknown EncodingMethod: {s}")

class NeuronType(Enum):
    """Types of neurons for SNN models."""
    LIF = auto()       # Leaky Integrate-and-Fire
    ALIF = auto()      # Adaptive LIF
    QLIF = auto()      # Quantized LIF
    IF = auto()        # Integrate-and-Fire
    LSTM = auto()      # Long Short-Term Memory (as SNN)
    IAF = auto()       # Integrate-and-Fire
    
    def __str__(self) -> str:
        return self.name.lower()
    
    @classmethod
    def from_string(cls, s: str) -> 'NeuronType':
        """Create enum from string representation."""
        s_upper = s.upper()
        for item in cls:
            if item.name == s_upper:
                return item
        for item in cls:
            if item.name.startswith(s_upper):
                return item
        raise ValueError(f"Unknown NeuronType: {s}")

class LearningRule(Enum):
    """Learning rules for SNN training."""
    STDP = auto()      # Spike-Timing-Dependent Plasticity
    R_STDP = auto()    # Reward-modulated STDP
    STDP_SYMMETRIC = auto() # Symmetric STDP
    HEBBIAN = auto()   # Hebbian learning
    RSTDP_DOPAMINE = auto() # Dopamine-modulated RSTDP
    
    def __str__(self) -> str:
        return self.name.lower()
    
    @classmethod
    def from_string(cls, s: str) -> 'LearningRule':
        """Create enum from string representation."""
        s_upper = s.upper()
        for item in cls:
            if item.name == s_upper:
                return item
        for item in cls:
            if item.name.startswith(s_upper):
                return item
        raise ValueError(f"Unknown LearningRule: {s}")

@dataclass
class SpikeTrainData:
    """Container for spike train data."""
    spikes: torch.Tensor  # Shape: [batch, time, neurons] or [time, neurons]
    times: torch.Tensor   # Time values for spikes
    source: str           # Data source (symbol, etc.)
    feature: str          # Feature name
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SNNModelConfig:
    """Configuration for SNN models."""
    input_size: int
    hidden_size: int
    output_size: int
    neuron_type: NeuronType = NeuronType.LIF
    learning_rule: LearningRule = LearningRule.STDP
    batch_size: int = 32
    seq_length: int = 100
    dt: float = 0.001  # Time step in seconds
    threshold: float = 1.0  # Firing threshold
    reset_voltage: float = 0.0  # Reset voltage
    tau_mem: float = 20.0  # Membrane time constant
    tau_syn: float = 10.0  # Synaptic time constant
    learning_rate: float = 0.001
    use_recurrent: bool = True
    use_inhibition: bool = True
    dropout: float = 0.0  # Missing attribute in original code
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Ensure all required attributes have valid values
        if not hasattr(self, 'dropout'):
            self.dropout = 0.0
            
class NeuromorphicAnalyzer:
    """
    Neuromorphic computing module for CDFA using spiking neural networks.
    
    Leverages bio-inspired computing models for efficient pattern recognition,
    anomaly detection, and adaptive learning using STDP neuroplasticity.
    """
    
    def __init__(self, hw_accelerator: Optional[HardwareAccelerator] = None,
                config: Optional[Dict[str, Any]] = None,
                log_level: int = logging.INFO):
        """
        Initialize the neuromorphic analyzer.
        
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
            # Encoding parameters
            "default_encoding": "rate",
            "encoding_time_window": 100,  # ms
            "rate_scale_factor": 1.0,
            "temporal_precision": 0.1,    # ms
            "population_size": 32,
            "use_wavelet_features": True,
            
            # Neuron parameters
            "default_neuron_type": "lif",
            "membrane_threshold": 1.0,
            "membrane_reset": 0.0,
            "tau_mem": 20.0,  # ms
            "tau_syn": 10.0,  # ms
            "refractory_period": 5.0,  # ms
            "adaptation_time_constant": 100.0,  # ms
            
            # Network parameters
            "default_network_size": 128,
            "recurrent_connections": True,
            "inhibitory_connections": True,
            "connection_density": 0.1,
            "inhibitory_ratio": 0.2,
            
            # STDP parameters
            "stdp_learning_rate": 0.01,
            "stdp_a_plus": 0.05,
            "stdp_a_minus": 0.05,
            "stdp_tau_plus": 20.0,  # ms
            "stdp_tau_minus": 20.0,  # ms
            "stdp_nearest_neighbor": True,
            "homeostasis_target_rate": 0.1,  # Hz
            "homeostasis_strength": 0.01,
            
            # Training parameters
            "batch_size": 32,
            "training_epochs": 100,
            "learning_rate": 0.001,
            "reward_modulation": True,
            "eligibility_trace_decay": 0.95,
            
            # Performance parameters
            "use_jit": True,
            "use_cuda": torch.cuda.is_available(),
            "dtype": "float32",
            "cache_models": True,
            "cache_spikes": True,
            "cache_ttl": 3600,  # 1 hour
            
            # Feature extraction
            "default_features": ["price", "volume", "volatility"],
            "wavelet_family": "db4",
            "wavelet_levels": 4,
            "feature_normalization": True
        }
        
        # Merge with provided config
        self.config = {**self.default_config, **(config or {})}
        
        # Initialize state
        self._lock = threading.RLock()
        self._models = {}  # name -> model
        self._encoders = {}  # name -> encoder
        self._spike_cache = {}  # key -> (spikes, timestamp)
        self._feature_extractors = {}  # name -> extractor
        self._training_state = {}  # model_name -> training state
        
        # Check available backends
        self.has_pytorch = TORCH_AVAILABLE
        self.has_norse = NORSE_AVAILABLE
        self.has_rockpool = ROCKPOOL_AVAILABLE
        self.has_wavelets = PYWAVELETS_AVAILABLE
        
        self.available_backends = []
        if self.has_norse:
            self.available_backends.append("norse")
        if self.has_rockpool:
            self.available_backends.append("rockpool")
        if self.has_pytorch:
            self.available_backends.append("pytorch")
            
        if not self.available_backends:
            self.logger.warning("No neuromorphic backends available!")
        else:
            self.logger.info(f"Available neuromorphic backends: {', '.join(self.available_backends)}")
            
        # Initialize encoders and feature extractors
        self._initialize_encoders()
        self._initialize_feature_extractors()
        
        self.logger.info("NeuromorphicAnalyzer initialized")
        
    def _initialize_encoders(self):
        """Initialize spike encoders."""
        if not self.has_pytorch:
            return
            
        # Rate coding encoder
        self._encoders["rate"] = self._rate_encoder
        
        # Temporal coding encoder
        self._encoders["temporal"] = self._temporal_encoder
        
        # Population coding encoder
        self._encoders["population"] = self._population_encoder
        
        # Phase coding encoder
        self._encoders["phase"] = self._phase_encoder
        
        # Burst coding encoder
        self._encoders["burst"] = self._burst_encoder
        
    def _initialize_feature_extractors(self):
        """Initialize feature extractors."""
        # Basic price features
        self._feature_extractors["price"] = self._extract_price_features
        
        # Volatility features
        self._feature_extractors["volatility"] = self._extract_volatility_features
        
        # Volume features
        self._feature_extractors["volume"] = self._extract_volume_features
        
        # Technical indicators
        self._feature_extractors["technical"] = self._extract_technical_features
        
        # Wavelet features
        if self.has_wavelets:
            self._feature_extractors["wavelet"] = self._extract_wavelet_features
            
    def _get_cached_spikes(self, key: Any) -> Optional[SpikeTrainData]:
        """
        Get cached spike train if valid.
        
        Args:
            key: Cache key
            
        Returns:
            Cached spike train or None if not found or expired
        """
        if not self.config["cache_spikes"]:
            return None
            
        with self._lock:
            # Check if spike train is in cache
            cache_entry = self._spike_cache.get(key)
            
            if cache_entry is None:
                return None
                
            spike_train, timestamp = cache_entry
            
            # Check if expired
            current_time = time.time()
            if current_time - timestamp > self.config["cache_ttl"]:
                # Remove from cache
                self._spike_cache.pop(key, None)
                return None
                
            return spike_train
            
    def _cache_spikes(self, key: Any, spike_train: SpikeTrainData):
        """
        Cache spike train for future use.
        
        Args:
            key: Cache key
            spike_train: Spike train data
        """
        if not self.config["cache_spikes"]:
            return
            
        with self._lock:
            self._spike_cache[key] = (spike_train, time.time())
            
    # ----- Spike Encoding Methods -----
    
    def _rate_encoder(self, data: np.ndarray, time_window: int = None, scale_factor: float = None) -> torch.Tensor:
        """
        Encode data using rate coding.
        
        Args:
            data: Input data [batch_size, input_size] or [input_size]
            time_window: Length of time window (default from config)
            scale_factor: Scaling factor (default from config)
            
        Returns:
            Spike tensor [batch_size, time_window, input_size] or [time_window, input_size]
        """
        if not self.has_pytorch:
            return None
            
        # Get defaults from config if not provided
        time_window = time_window or self.config["encoding_time_window"]
        scale_factor = scale_factor or self.config["rate_scale_factor"]
        
        # Handle 1D data
        is_1d = len(data.shape) == 1
        if is_1d:
            data = data.reshape(1, -1)
            
        # Convert to tensor if needed
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)
            
        # Move to device
        device = torch.device("cuda" if self.config.get("use_cuda", False) and torch.cuda.is_available() else "cpu")
        data = data.to(device)
        
        # Scale data to [0, 1] for rate coding
        data_min = data.min(dim=1, keepdim=True)[0]
        data_max = data.max(dim=1, keepdim=True)[0]
        data_range = data_max - data_min
        data_range[data_range == 0] = 1.0  # Prevent division by zero
        scaled_data = (data - data_min) / data_range
        
        # Apply rate scaling
        rates = scaled_data * scale_factor
        
        # Generate Poisson spike trains
        spikes = torch.rand(data.shape[0], time_window, data.shape[1], device=device) < rates.unsqueeze(1)
        
        # Return to original shape if 1D
        if is_1d:
            spikes = spikes.squeeze(0)
            
        return spikes.float()
    
    def _temporal_encoder(self, data: np.ndarray, time_window: int = None, precision: float = None) -> torch.Tensor:
        """
        Encode data using temporal coding (time-to-first-spike).
        
        Args:
            data: Input data [batch_size, input_size] or [input_size]
            time_window: Length of time window (default from config)
            precision: Temporal precision (default from config)
            
        Returns:
            Spike tensor [batch_size, time_window, input_size] or [time_window, input_size]
        """
        if not self.has_pytorch:
            return None
            
        # Get defaults from config if not provided
        time_window = time_window or self.config["encoding_time_window"]
        precision = precision or self.config["temporal_precision"]
        
        # Handle 1D data
        is_1d = len(data.shape) == 1
        if is_1d:
            data = data.reshape(1, -1)
            
        # Convert to tensor if needed
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)
            
        # Move to device
        device = torch.device("cuda" if self.config.get("use_cuda", False) and torch.cuda.is_available() else "cpu")
        data = data.to(device)
        
        # Scale data to [0, 1] for encoding
        data_min = data.min(dim=1, keepdim=True)[0]
        data_max = data.max(dim=1, keepdim=True)[0]
        data_range = data_max - data_min
        data_range[data_range == 0] = 1.0  # Prevent division by zero
        scaled_data = (data - data_min) / data_range
        
        # Calculate spike times (inverse relationship - higher values spike earlier)
        spike_times = (1.0 - scaled_data) * time_window
        
        # Create time matrix
        time_array = torch.arange(time_window, device=device).unsqueeze(0).unsqueeze(2).expand(
            data.shape[0], -1, data.shape[1]
        )
        
        # Create spike matrix (time >= spike_time)
        spike_times = spike_times.unsqueeze(1).expand(-1, time_window, -1)
        spikes = (time_array >= spike_times).float()
        
        # Each neuron fires only once - ensure only first spike is kept
        # Calculate cumulative sum along time dimension
        cumsum = torch.cumsum(spikes, dim=1)
        
        # Only keep first spike (where cumsum == 1)
        spikes = (cumsum == 1).float() * spikes
        
        # Return to original shape if 1D
        if is_1d:
            spikes = spikes.squeeze(0)
            
        return spikes
    
    def _population_encoder(self, data: np.ndarray, population_size: int = None, time_window: int = None) -> torch.Tensor:
        """
        Encode data using population coding.
        
        Args:
            data: Input data [batch_size, input_size] or [input_size]
            population_size: Number of neurons per input dimension (default from config)
            time_window: Length of time window (default from config)
            
        Returns:
            Spike tensor [batch_size, time_window, input_size * population_size] or 
                       [time_window, input_size * population_size]
        """
        if not self.has_pytorch:
            return None
            
        # Get defaults from config if not provided
        population_size = population_size or self.config["population_size"]
        time_window = time_window or self.config["encoding_time_window"]
        
        # Handle 1D data
        is_1d = len(data.shape) == 1
        if is_1d:
            data = data.reshape(1, -1)
            
        # Convert to tensor if needed
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)
            
        # Move to device
        device = torch.device("cuda" if self.config.get("use_cuda", False) and torch.cuda.is_available() else "cpu")
        data = data.to(device)
        
        batch_size, input_size = data.shape
        
        # Scale data to [0, 1]
        data_min = data.min(dim=1, keepdim=True)[0]
        data_max = data.max(dim=1, keepdim=True)[0]
        data_range = data_max - data_min
        data_range[data_range == 0] = 1.0  # Prevent division by zero
        scaled_data = (data - data_min) / data_range
        
        # Create population centers (receptive fields)
        centers = torch.linspace(0, 1, population_size, device=device)
        
        # Expand data and centers for broadcasting
        scaled_data_expanded = scaled_data.unsqueeze(-1).expand(-1, -1, population_size)
        centers_expanded = centers.unsqueeze(0).unsqueeze(1).expand(batch_size, input_size, -1)
        
        # Calculate responses based on distance to centers (Gaussian receptive fields)
        sigma = 1.0 / (population_size * 0.5)  # Width of Gaussian
        responses = torch.exp(-((scaled_data_expanded - centers_expanded) ** 2) / (2 * sigma ** 2))
        
        # Reshape to combine input dimensions and populations
        responses = responses.reshape(batch_size, -1)
        
        # Generate spikes using rate coding for the expanded population
        # Expand to time dimension
        rates = responses.unsqueeze(1).expand(-1, time_window, -1)
        
        # Generate Poisson spike trains
        spikes = torch.rand(rates.shape, device=device) < rates
        
        # Return to original shape if 1D
        if is_1d:
            spikes = spikes.squeeze(0)
            
        return spikes.float()
    
    def _phase_encoder(self, data: np.ndarray, time_window: int = None, freq_factor: float = 10.0) -> torch.Tensor:
        """
        Encode data using phase coding.
        
        Args:
            data: Input data [batch_size, input_size] or [input_size]
            time_window: Length of time window (default from config)
            freq_factor: Base frequency factor
            
        Returns:
            Spike tensor [batch_size, time_window, input_size] or [time_window, input_size]
        """
        if not self.has_pytorch:
            return None
            
        # Get defaults from config if not provided
        time_window = time_window or self.config["encoding_time_window"]
        
        # Handle 1D data
        is_1d = len(data.shape) == 1
        if is_1d:
            data = data.reshape(1, -1)
            
        # Convert to tensor if needed
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)
            
        # Move to device
        device = torch.device("cuda" if self.config.get("use_cuda", False) and torch.cuda.is_available() else "cpu")
        data = data.to(device)
        
        # Scale data to [0, 2π] for phase encoding
        data_min = data.min(dim=1, keepdim=True)[0]
        data_max = data.max(dim=1, keepdim=True)[0]
        data_range = data_max - data_min
        data_range[data_range == 0] = 1.0  # Prevent division by zero
        scaled_data = (data - data_min) / data_range * (2 * np.pi)
        
        # Create time array
        t = torch.linspace(0, 1, time_window, device=device).unsqueeze(0).unsqueeze(-1)
        
        # Base frequency
        base_freq = freq_factor * 2 * np.pi
        
        # Calculate phases
        phases = torch.sin(base_freq * t + scaled_data.unsqueeze(1))
        
        # Generate spikes when phase crosses threshold
        threshold = 0.9  # Threshold for spike generation
        spikes = (phases > threshold).float()
        
        # Return to original shape if 1D
        if is_1d:
            spikes = spikes.squeeze(0)
            
        return spikes
    
    def _burst_encoder(self, data: np.ndarray, time_window: int = None, 
                      burst_length: int = 5, max_bursts: int = 3) -> torch.Tensor:
        """
        Encode data using burst coding (sequences of spikes).
        
        Args:
            data: Input data [batch_size, input_size] or [input_size]
            time_window: Length of time window (default from config)
            burst_length: Length of each burst
            max_bursts: Maximum number of bursts
            
        Returns:
            Spike tensor [batch_size, time_window, input_size] or [time_window, input_size]
        """
        if not self.has_pytorch:
            return None
            
        # Get defaults from config if not provided
        time_window = time_window or self.config["encoding_time_window"]
        
        # Handle 1D data
        is_1d = len(data.shape) == 1
        if is_1d:
            data = data.reshape(1, -1)
            
        # Convert to tensor if needed
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)
            
        # Move to device
        device = torch.device("cuda" if self.config.get("use_cuda", False) and torch.cuda.is_available() else "cpu")
        data = data.to(device)
        
        batch_size, input_size = data.shape
        
        # Scale data to [0, 1]
        data_min = data.min(dim=1, keepdim=True)[0]
        data_max = data.max(dim=1, keepdim=True)[0]
        data_range = data_max - data_min
        data_range[data_range == 0] = 1.0  # Prevent division by zero
        scaled_data = (data - data_min) / data_range
        
        # Calculate number of bursts based on data value
        num_bursts = torch.floor(scaled_data * (max_bursts + 1)).long()
        
        # Initialize spike tensor
        spikes = torch.zeros(batch_size, time_window, input_size, device=device)
        
        # Maximum time for last burst to complete
        max_start_time = time_window - burst_length
        
        # Generate bursts
        for b in range(batch_size):
            for i in range(input_size):
                n_burst = num_bursts[b, i].item()
                
                if n_burst > 0:
                    # Distribute bursts evenly throughout time window
                    for j in range(n_burst):
                        start_time = min(max_start_time, int(j * max_start_time / n_burst))
                        end_time = min(time_window, start_time + burst_length)
                        
                        # Create burst
                        spikes[b, start_time:end_time, i] = 1
                        
        # Return to original shape if 1D
        if is_1d:
            spikes = spikes.squeeze(0)
            
        return spikes
    
    def encode_data(self, data: np.ndarray, method: Optional[Union[str, EncodingMethod]] = None,
                  time_window: Optional[int] = None, feature_name: str = "generic",
                  source: str = "unknown") -> SpikeTrainData:
        """
        Encode data into spike trains using specified method.
        
        Args:
            data: Input data to encode
            method: Encoding method (default from config)
            time_window: Time window length (default from config)
            feature_name: Feature name for metadata
            source: Data source (symbol, etc.)
            
        Returns:
            Spike train data
        """
        if not self.has_pytorch:
            self.logger.error("PyTorch not available, cannot encode data")
            return None
            
        # Get defaults from config if not provided
        if method is None:
            method = self.config["default_encoding"]
            
        if isinstance(method, EncodingMethod):
            method = str(method)
            
        method = method.lower()
        
        time_window = time_window or self.config["encoding_time_window"]
        
        # Get encoder function
        encoder_func = self._encoders.get(method)
        
        if encoder_func is None:
            self.logger.error(f"Unknown encoding method: {method}")
            return None
            
        # Create cache key
        cache_key = (str(data.tobytes()), method, time_window, feature_name, source)
        cached_spikes = self._get_cached_spikes(cache_key)
        if cached_spikes is not None:
            return cached_spikes
            
        # Encode data
        spikes = encoder_func(data, time_window)
        
        # Create time values tensor
        device = spikes.device
        if len(spikes.shape) == 3:  # [batch, time, neurons]
            times = torch.arange(0, time_window, device=device).unsqueeze(0).expand(spikes.shape[0], -1)
        else:  # [time, neurons]
            times = torch.arange(0, time_window, device=device)
            
        # Create spike train data
        spike_train = SpikeTrainData(
            spikes=spikes,
            times=times,
            source=source,
            feature=feature_name,
            metadata={
                "encoding_method": method,
                "time_window": time_window,
                "shape": spikes.shape,
                "timestamp": time.time()
            }
        )
        
        # Cache spike train
        self._cache_spikes(cache_key, spike_train)
        
        return spike_train
    
    # ----- Feature Extraction Methods -----
    
    def _extract_price_features(self, df: pd.DataFrame, window_sizes: List[int] = [5, 10, 20]) -> Dict[str, np.ndarray]:
        """
        Extract price-based features from OHLCV data with safe log operations.
        """
        features = {}
        
        # Check required columns
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            self.logger.error(f"Missing required columns in dataframe: {required_cols}")
            return features
            
        # Calculate returns
        df['returns'] = df['close'].pct_change()
        features['returns'] = df['returns'].fillna(0).values
        
        # Calculate log returns with explicit safe handling for log operations
        df['log_returns'] = np.zeros(len(df))
        valid_mask = (df['close'] > 0) & (df['close'].shift(1) > 0)
        if valid_mask.any():
            # Add small epsilon to avoid log(0)
            epsilon = 1e-10
            df.loc[valid_mask, 'log_returns'] = np.log(
                df.loc[valid_mask, 'close'] / (df.loc[valid_mask, 'close'].shift(1) + epsilon) + epsilon
            )
        features['log_returns'] = df['log_returns'].fillna(0).values
        
        # Calculate price momentum at different windows
        for window in window_sizes:
            # Simple momentum (current price / price n periods ago - 1)
            df[f'momentum_{window}'] = np.zeros(len(df))
            valid_mask = (df['close'] > 0) & (df['close'].shift(window) > 0)
            if valid_mask.any():
                df.loc[valid_mask, f'momentum_{window}'] = (
                    df.loc[valid_mask, 'close'] / df.loc[valid_mask, 'close'].shift(window) - 1
                )
            features[f'momentum_{window}'] = df[f'momentum_{window}'].fillna(0).values
            
            # Moving averages
            df[f'ma_{window}'] = df['close'].rolling(window).mean()
            # Use bfill() instead of fillna(method='bfill')
            features[f'ma_{window}'] = df[f'ma_{window}'].bfill().values
            
            # Relative position to moving average - handle division by zero
            df[f'rel_ma_{window}'] = np.zeros(len(df))
            valid_mask = df[f'ma_{window}'] > 0
            if valid_mask.any():
                df.loc[valid_mask, f'rel_ma_{window}'] = (
                    df.loc[valid_mask, 'close'] / df.loc[valid_mask, f'ma_{window}'] - 1
                )
            features[f'rel_ma_{window}'] = df[f'rel_ma_{window}'].fillna(0).values
            
        # Calculate candle features
        df['body_size'] = abs(df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8)
        features['body_size'] = df['body_size'].fillna(0).values
        
        df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / (df['high'] - df['low'] + 1e-8)
        features['upper_shadow'] = df['upper_shadow'].fillna(0).values
        
        df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / (df['high'] - df['low'] + 1e-8)
        features['lower_shadow'] = df['lower_shadow'].fillna(0).values
        
        return features
    
    def _extract_volatility_features(self, df: pd.DataFrame, window_sizes: List[int] = [5, 10, 20]) -> Dict[str, np.ndarray]:
        """
        Extract volatility-based features from OHLCV data.
        
        Args:
            df: OHLCV dataframe
            window_sizes: Window sizes for feature extraction
            
        Returns:
            Dictionary of feature name to feature array
        """
        features = {}
        
        # Check required columns
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            self.logger.error(f"Missing required columns in dataframe: {required_cols}")
            return features
            
        # Calculate returns if not already present
        if 'returns' not in df.columns:
            df['returns'] = df['close'].pct_change()
            
        # Calculate log returns if not already present
        if 'log_returns' not in df.columns:
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            
        # Calculate volatility (standard deviation of returns) at different windows
        for window in window_sizes:
            # Standard volatility (standard deviation of returns)
            df[f'volatility_{window}'] = df['returns'].rolling(window).std()
            features[f'volatility_{window}'] = df[f'volatility_{window}'].fillna(0).values
            
            # Log volatility
            df[f'log_volatility_{window}'] = df['log_returns'].rolling(window).std()
            features[f'log_volatility_{window}'] = df[f'log_volatility_{window}'].fillna(0).values
            
            # Realized volatility (Parkinson estimator using high-low range)
            df[f'realized_vol_{window}'] = (
                np.sqrt(1.0 / (4.0 * np.log(2.0)) * 
                      (np.log(df['high'] / df['low']) ** 2).rolling(window).mean()) * 
                np.sqrt(252)  # Annualize
            )
            features[f'realized_vol_{window}'] = df[f'realized_vol_{window}'].fillna(0).values
            
            # Volatility of volatility
            df[f'vol_of_vol_{window}'] = df[f'volatility_{window}'].rolling(window).std()
            features[f'vol_of_vol_{window}'] = df[f'vol_of_vol_{window}'].fillna(0).values
            
        # Calculate high-low range volatility
        df['daily_range'] = (df['high'] - df['low']) / df['close'].shift(1)
        features['daily_range'] = df['daily_range'].fillna(0).values
        
        # Daily gap (open to previous close)
        df['daily_gap'] = abs(df['open'] - df['close'].shift(1)) / df['close'].shift(1)
        features['daily_gap'] = df['daily_gap'].fillna(0).values
        
        return features
    
    def _extract_volume_features(self, df: pd.DataFrame, window_sizes: List[int] = [5, 10, 20]) -> Dict[str, np.ndarray]:
        """
        Extract volume-based features from OHLCV data.
        
        Args:
            df: OHLCV dataframe
            window_sizes: Window sizes for feature extraction
            
        Returns:
            Dictionary of feature name to feature array
        """
        features = {}
        
        # Check required columns
        if 'volume' not in df.columns:
            self.logger.error("Volume column not found in dataframe")
            return features
            
        # Normalize volume (divide by mean volume)
        mean_volume = df['volume'].mean()
        if mean_volume > 0:
            df['norm_volume'] = df['volume'] / mean_volume
        else:
            df['norm_volume'] = df['volume']
            
        features['norm_volume'] = df['norm_volume'].fillna(0).values
        
        # Calculate log volume
        df['log_volume'] = np.log1p(df['volume'])
        features['log_volume'] = df['log_volume'].fillna(0).values
        
        # Calculate volume momentum at different windows
        for window in window_sizes:
            # Volume moving average
            df[f'volume_ma_{window}'] = df['volume'].rolling(window).mean()
            
            # Relative volume (current volume / average volume)
            df[f'rel_volume_{window}'] = df['volume'] / df[f'volume_ma_{window}']
            features[f'rel_volume_{window}'] = df[f'rel_volume_{window}'].fillna(1).values
            
            # Volume momentum
            df[f'volume_momentum_{window}'] = df['volume'] / df['volume'].shift(window)
            features[f'volume_momentum_{window}'] = df[f'volume_momentum_{window}'].fillna(1).values
            
            # On-balance volume (OBV)
            if 'close' in df.columns and f'obv_{window}' not in df.columns:
                # Calculate daily OBV direction
                df['obv_direction'] = np.where(df['close'] > df['close'].shift(1), 1,
                                            np.where(df['close'] < df['close'].shift(1), -1, 0))
                df['obv_daily'] = df['volume'] * df['obv_direction']
                df['obv'] = df['obv_daily'].cumsum()
                
                # Calculate OBV momentum
                df[f'obv_{window}'] = df['obv'] - df['obv'].shift(window)
                features[f'obv_{window}'] = df[f'obv_{window}'].fillna(0).values
                
        # Calculate up/down volume ratio (if we have price data)
        if 'close' in df.columns and 'up_down_vol_ratio' not in df.columns:
            df['up_volume'] = np.where(df['close'] > df['close'].shift(1), df['volume'], 0)
            df['down_volume'] = np.where(df['close'] < df['close'].shift(1), df['volume'], 0)
            
            # Calculate ratio with smoothing to avoid division by zero
            df['up_volume_ma'] = df['up_volume'].rolling(5).mean()
            df['down_volume_ma'] = df['down_volume'].rolling(5).mean()
            df['up_down_vol_ratio'] = df['up_volume_ma'] / (df['down_volume_ma'] + 1e-8)
            
            features['up_down_vol_ratio'] = df['up_down_vol_ratio'].fillna(1).values
            
        return features
    
    def _extract_technical_features(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Extract technical indicators from OHLCV data.
        
        Args:
            df: OHLCV dataframe
            
        Returns:
            Dictionary of feature name to feature array
        """
        features = {}
        
        # Check required columns
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            self.logger.error(f"Missing required columns in dataframe: {required_cols}")
            return features
            
        # Calculate RSI
        def calculate_rsi(prices, period=14):
            delta = prices.diff()
            gain = delta.where(delta > 0, 0).fillna(0)
            loss = -delta.where(delta < 0, 0).fillna(0)
            
            avg_gain = gain.rolling(period).mean()
            avg_loss = loss.rolling(period).mean()
            
            rs = avg_gain / (avg_loss + 1e-8)
            rsi = 100 - (100 / (1 + rs))
            return rsi
            
        df['rsi_14'] = calculate_rsi(df['close'], 14)
        features['rsi_14'] = df['rsi_14'].fillna(50).values
        
        # MACD
        df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Normalize MACD by price
        df['macd_norm'] = df['macd'] / df['close']
        features['macd_norm'] = df['macd_norm'].fillna(0).values
        
        df['macd_hist_norm'] = df['macd_hist'] / df['close']
        features['macd_hist_norm'] = df['macd_hist_norm'].fillna(0).values
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(20).mean()
        df['bb_std'] = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
        
        # BB width (volatility measure)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        features['bb_width'] = df['bb_width'].fillna(0).values
        
        # BB position (where price is within the bands, 0 = lower, 1 = upper)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-8)
        features['bb_position'] = df['bb_position'].fillna(0.5).values
        
        # Stochastic Oscillator
        df['stoch_k'] = 100 * ((df['close'] - df['low'].rolling(14).min()) / 
                             (df['high'].rolling(14).max() - df['low'].rolling(14).min() + 1e-8))
        df['stoch_d'] = df['stoch_k'].rolling(3).mean()
        
        features['stoch_k'] = df['stoch_k'].fillna(50).values
        features['stoch_d'] = df['stoch_d'].fillna(50).values
        
        # Average True Range (ATR)
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = abs(df['high'] - df['close'].shift(1))
        df['tr3'] = abs(df['low'] - df['close'].shift(1))
        df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        df['atr_14'] = df['true_range'].rolling(14).mean()
        
        # Normalize ATR by price
        df['atr_14_norm'] = df['atr_14'] / df['close']
        features['atr_14_norm'] = df['atr_14_norm'].fillna(0).values
        
        return features
    
    def _extract_wavelet_features(self, df: pd.DataFrame,
                                wavelet: Optional[str] = None,
                                levels: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Extract wavelet-based features from OHLCV data.
        
        Args:
            df: OHLCV dataframe
            wavelet: Wavelet family (default from config)
            levels: Number of decomposition levels (default from config)
            
        Returns:
            Dictionary of feature name to feature array
        """
        features = {}
        
        if not self.has_wavelets:
            self.logger.error("PyWavelets not available, cannot extract wavelet features")
            return features
            
        # Get defaults from config if not provided
        wavelet = wavelet or self.config["wavelet_family"]
        levels = levels or self.config["wavelet_levels"]
        
        # Check required columns
        if 'close' not in df.columns:
            self.logger.error("Close price column not found in dataframe")
            return features
            
        # Get price series
        price_series = df['close'].values
        
        # Calculate returns if there are enough data points
        if len(price_series) > 1:
            returns = np.diff(price_series) / price_series[:-1]
            returns = np.append(0, returns)  # Add zero for first point
        else:
            returns = np.zeros_like(price_series)
            
        try:
            # Multi-resolution analysis with wavelets
            # Set max level based on data length
            max_allowed_level = pywt.dwt_max_level(len(price_series), wavelet)
            actual_levels = min(levels, max_allowed_level)
            
            # Decompose returns
            coeffs = pywt.wavedec(returns, wavelet, level=actual_levels)
            
            # Extract approximation coefficient (low frequency component)
            features['wavelet_approx'] = coeffs[0]
            
            # Extract detail coefficients (high frequency components)
            for i, detail in enumerate(coeffs[1:]):
                features[f'wavelet_detail_{i+1}'] = detail
                
            # Calculate energy of wavelet coefficients
            features['wavelet_energy_approx'] = np.sum(coeffs[0]**2)
            
            energy_features = []
            for i, detail in enumerate(coeffs[1:]):
                energy = np.sum(detail**2)
                features[f'wavelet_energy_detail_{i+1}'] = energy
                energy_features.append(energy)
                
            # Energy distribution ratio (detail energy / total energy)
            total_energy = features['wavelet_energy_approx'] + sum(energy_features)
            if total_energy > 0:
                features['wavelet_energy_ratio_approx'] = features['wavelet_energy_approx'] / total_energy
                
                for i in range(len(energy_features)):
                    features[f'wavelet_energy_ratio_detail_{i+1}'] = energy_features[i] / total_energy
                    
            # Calculate instantaneous variance at different scales
            # 1. Reconstruct each detail level separately
            for i in range(1, actual_levels + 1):
                # Create coefficient list with zeros except for the detail at level i
                zero_coeffs = [np.zeros_like(c) for c in coeffs]
                zero_coeffs[i] = coeffs[i]
                
                # Reconstruct
                reconstructed = pywt.waverec(zero_coeffs, wavelet)
                
                # Trim to match original length
                reconstructed = reconstructed[:len(returns)]
                
                # Calculate instantaneous variance (squared amplitude)
                features[f'wavelet_inst_var_detail_{i}'] = reconstructed**2
                
            # Continuous wavelet transform for selected scales
            scales = np.arange(1, 32)
            
            # Use a subset of scales for efficiency
            selected_scales = np.array([1, 2, 4, 8, 16, 32])
            selected_scales = selected_scales[selected_scales < len(returns) // 2]
            
            if len(selected_scales) > 0:
                # Perform CWT
                coef, _ = pywt.cwt(returns, selected_scales, wavelet)
                
                # Store CWT coefficients for selected scales
                for i, scale in enumerate(selected_scales):
                    features[f'wavelet_cwt_scale_{scale}'] = np.abs(coef[i])
                    
        except Exception as e:
            self.logger.error(f"Error extracting wavelet features: {e}")
            
        return features
    
    def extract_features(self, df: pd.DataFrame, feature_types: Optional[List[str]] = None,
                      source: str = "unknown") -> Dict[str, np.ndarray]:
        """
        Extract multiple feature types from dataframe.
        
        Args:
            df: OHLCV dataframe
            feature_types: List of feature types (default: from config)
            source: Data source (symbol, etc.)
            
        Returns:
            Dictionary of feature name to feature array
        """
        # Get defaults from config if not provided
        if feature_types is None:
            feature_types = self.config["default_features"]
            
        # Initialize features dictionary
        all_features = {}
        
        # Extract each feature type
        for feature_type in feature_types:
            if feature_type in self._feature_extractors:
                features = self._feature_extractors[feature_type](df)
                
                # Add to all features
                all_features.update(features)
            else:
                self.logger.warning(f"Unknown feature type: {feature_type}")
                
        # Apply normalization if enabled
        if self.config["feature_normalization"]:
            normalized_features = {}
            
            for name, feature in all_features.items():
                # Skip features that are already normalized (0-1 range)
                if name.startswith('rsi_') or name.startswith('stoch_') or name.startswith('bb_position'):
                    normalized_features[name] = feature
                    continue
                    
                # Normalize using min-max scaling with outlier handling
                if len(feature) > 0:
                    # Find non-NaN and non-infinite values
                    valid_mask = np.isfinite(feature)
                    
                    if np.any(valid_mask):
                        valid_values = feature[valid_mask]
                        
                        # Calculate min/max with outlier removal
                        q1, q99 = np.percentile(valid_values, [1, 99])
                        value_range = q99 - q1
                        
                        if value_range > 0:
                            # Apply normalization
                            normalized = np.zeros_like(feature)
                            normalized[valid_mask] = (valid_values - q1) / value_range
                            
                            # Clip to [0, 1]
                            normalized = np.clip(normalized, 0, 1)
                            
                            normalized_features[name] = normalized
                        else:
                            normalized_features[name] = feature
                    else:
                        normalized_features[name] = feature
                else:
                    normalized_features[name] = feature
                    
            all_features = normalized_features
            
        return all_features
    
    # ----- SNN Model Creation Methods -----
    
    def create_norse_lif_model(self, config: SNNModelConfig) -> nn.Module:
        """
        Create a Norse LIF neural network model.
        
        Args:
            config: Model configuration
            
        Returns:
            PyTorch module implementing SNN
        """
        if not self.has_pytorch or not self.has_norse:
            self.logger.error("PyTorch or Norse not available, cannot create model")
            return None
            
        # Create LIF parameters
        lif_params = LIFParameters(
            tau_mem_inv=1.0 / config.tau_mem,
            tau_syn_inv=1.0 / config.tau_syn,
            v_leak=0.0,
            v_th=config.threshold,
            v_reset=config.reset_voltage,
            method="super",
            alpha=100.0
        )
        
        # Create model class
        class NorseLIFNetwork(nn.Module):
            def __init__(self, input_size, hidden_size, output_size, params, recurrent=True, dropout=0.0):
                super(NorseLIFNetwork, self).__init__()
                
                self.input_size = input_size
                self.hidden_size = hidden_size
                self.output_size = output_size
                self.params = params
                self.recurrent = recurrent
                
                # Input layers
                self.fc_input = nn.Linear(input_size, hidden_size)
                
                # Recurrent layer (if enabled)
                if recurrent:
                    self.lif_recurrent = LIFRecurrentCell(
                        hidden_size,
                        hidden_size,
                        p=lif_params
                    )
                else:
                    self.lif1 = LIFCell(p=lif_params)
                    
                # Dropout
                self.dropout = nn.Dropout(dropout) if dropout > 0 else None
                
                # Output layer
                self.fc_output = nn.Linear(hidden_size, output_size)
                self.lif_output = LIFCell(p=lif_params)
                
                # Readout layer
                self.readout = LI()
                
            def forward(self, x):
                # Input shape: [batch_size, seq_length, input_size]
                batch_size, seq_length, _ = x.shape
                
                # Initialize states
                if self.recurrent:
                    s1 = self.lif_recurrent.initial_state(batch_size, device=x.device)
                else:
                    s1 = self.lif1.initial_state(batch_size, device=x.device)
                    
                so = self.lif_output.initial_state(batch_size, device=x.device)
                state = self.readout.initial_state(batch_size, device=x.device)
                
                # Collect outputs for each timestep
                outputs = []
                spikes = []
                
                # Process each timestep
                for ts in range(seq_length):
                    z = x[:, ts, :]
                    
                    # Input layer
                    z = self.fc_input(z)
                    
                    # Hidden layer
                    if self.recurrent:
                        z, s1 = self.lif_recurrent(z, s1)
                    else:
                        z, s1 = self.lif1(z, s1)
                        
                    # Apply dropout if enabled
                    if self.dropout is not None:
                        z = self.dropout(z)
                        
                    # Store spikes from hidden layer
                    spikes.append(z)
                    
                    # Output layer
                    z = self.fc_output(z)
                    z, so = self.lif_output(z, so)
                    
                    # Leaky integrator readout
                    v, state = self.readout(z, state)
                    outputs.append(v)
                    
                # Stack outputs
                outputs = torch.stack(outputs, dim=1)
                spikes = torch.stack(spikes, dim=1)
                
                return outputs, spikes
                
        # Create model instance
        model = NorseLIFNetwork(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            output_size=config.output_size,
            params=lif_params,
            recurrent=config.use_recurrent,
            dropout=config.dropout
        )
        
        # Move to device
        device = torch.device(config.device)
        model = model.to(device)
        
        # Apply TorchScript optimization if enabled
        if self.config["use_jit"]:
            try:
                # Create example input for tracing
                example_input = torch.zeros(
                    (config.batch_size, config.seq_length, config.input_size),
                    device=device
                )
                
                # Use script (supports dynamic behavior) rather than trace
                model = torch.jit.script(model)
                
                self.logger.info("Created TorchScript-optimized Norse model")
            except Exception as e:
                self.logger.warning(f"Failed to apply TorchScript optimization: {e}")
                
        return model
    

    
    def create_rockpool_model(self, config: SNNModelConfig) -> Any:
        """
        Create a Rockpool neural network model with the *strictly* specified
        neuron type from the configuration. No fallbacks are used.
    
        Args:
            config: Model configuration. The `neuron_type` must match an
                    available Rockpool module name directly or indirectly
                    via its .name attribute.
    
        Returns:
            Rockpool model or None if creation of the specified type fails
            or the type is not available.
        """
        if not self.has_rockpool:
            self.logger.error("Rockpool not available, cannot create model")
            return None
    
        model = None # Initialize model to None
    
        # Safely get the string representation of the requested neuron type
        # This handles both Enum members (NeuronType.LIF) and potentially string inputs
        try:
            requested_neuron_type_str = str(config.neuron_type.name)
        except AttributeError:
            requested_neuron_type_str = str(config.neuron_type)
    
        self.logger.info(f"Attempting to create Rockpool model with requested neuron type: {requested_neuron_type_str}")
    
        try:
            # Import necessary components with better error handling
            try:
                from rockpool.nn import modules as rp_modules
                from rockpool.parameters import Constant
            except (ImportError, AttributeError) as e:
                self.logger.error(f"Error importing essential Rockpool components: {e}")
                return None
    
            # Get the specific module class based on the requested neuron type string
            model_class = None
            try:
                model_class = getattr(rp_modules, requested_neuron_type_str)
                self.logger.debug(f"Found Rockpool module class: {requested_neuron_type_str}")
            except AttributeError:
                self.logger.error(f"Requested Rockpool neuron type '{requested_neuron_type_str}' not found in rockpool.nn.modules")
                return None # Return None if the requested module class doesn't exist
    
            # Attempt to instantiate the specific model class
            try:
                # --- Instantiate based on the SPECIFIC requested type ---
                # Need to handle different constructors based on type, WITHOUT fallback
                if requested_neuron_type_str == 'LIFBitshiftRecurrent':
                     self.logger.debug("Instantiating LIFBitshiftRecurrent")
                     model = model_class( # model_class is rp_modules.LIFBitshiftRecurrent
                         input_size=config.input_size,
                         hidden_size=config.hidden_size,
                         output_size=config.output_size,
                         threshold=Constant(config.threshold),
                         tau_mem=Constant(config.tau_mem),
                         tau_syn=Constant(config.tau_syn),
                         dash_dtype=np.float32,
                         bias=True,
                         has_recurrent=config.use_recurrent
                     )
                     self.logger.info(f"Successfully created Rockpool model with {requested_neuron_type_str} neurons")
    
                elif requested_neuron_type_str == 'LIFAdaptive':
                     self.logger.debug("Instantiating LIFAdaptive")
                     model = model_class( # model_class is rp_modules.LIFAdaptive
                         input_size=config.input_size,
                         hidden_size=config.hidden_size,
                         output_size=config.output_size,
                         threshold=Constant(config.threshold),
                         tau_mem=Constant(config.tau_mem),
                         tau_syn=Constant(config.tau_syn),
                         tau_adapt=Constant(config.metadata.get("tau_adapt", 100.0)),
                         bias=True
                     )
                     self.logger.info(f"Successfully created Rockpool model with {requested_neuron_type_str} neurons")
    
                elif requested_neuron_type_str == 'IAF':
                     self.logger.debug("Instantiating IAF")
                     model = model_class( # model_class is rp_modules.IAF
                         input_size=config.input_size,
                         hidden_size=config.hidden_size,
                         output_size=config.output_size,
                         threshold=Constant(config.threshold),
                         bias=True
                     )
                     self.logger.info(f"Successfully created Rockpool model with {requested_neuron_type_str} neurons")
    
                elif requested_neuron_type_str == 'LIF':
                     self.logger.debug("Instantiating standard LIF")
                     # Standard LIF takes 'shape', not structural sizes directly like multi-layer modules
                     model = model_class( # model_class is rp_modules.LIF
                         shape=(config.hidden_size,), # Assuming hidden_size dictates the layer size
                         threshold=Constant(config.threshold),
                         tau_mem=Constant(config.tau_mem),
                         tau_syn=Constant(config.tau_syn),
                         bias=True
                     )
                     self.logger.info(f"Successfully created Rockpool model with {requested_neuron_type_str} neurons (shape: {config.hidden_size})")
    
                # Add more elif blocks here for other specific Rockpool module types you need to support
                # if requested_neuron_type_str == 'SomeOtherType':
                #    model = model_class(...) # Instantiate with correct arguments for SomeOtherType
                #    self.logger.info(...)
    
                else:
                    # If the requested type was found via getattr but isn't explicitly handled
                    # above with correct arguments, this means we don't know how to instantiate it.
                    # This acts as a safeguard against trying to instantiate with incorrect
                    # default arguments.
                     self.logger.error(f"Rockpool neuron type '{requested_neuron_type_str}' is available "
                                       f"but its instantiation arguments are not explicitly handled in this method.")
                     return None
    
    
                # If model is successfully created at this point, return it.
                # If model is still None, it means the requested_neuron_type_str matched a module
                # but wasn't one of the explicitly handled types for instantiation, which
                # triggered the 'else' block above.
                if model is not None:
                    return model
                else:
                    # This case should ideally be caught by the 'else' block above, but added for safety
                    self.logger.error(f"Failed to create Rockpool model for requested type '{requested_neuron_type_str}' "
                                      f"due to unhandled instantiation logic.")
                    return None
    
    
            except Exception as e:
                # This catches errors during the instantiation of the specific module
                self.logger.error(f"Error creating Rockpool model with requested type '{requested_neuron_type_str}': {e}")
                return None # Return None if instantiation fails
    
        except Exception as e:
            # This outer exception catches any other unexpected errors
            self.logger.error(f"An unexpected error occurred during Rockpool model creation: {e}")
            return None
        
    def create_snn_model(self, config: Optional[SNNModelConfig] = None, backend: Optional[str] = None) -> Any:
        """
        Create a spiking neural network model with improved error handling and fallbacks.
        
        Args:
            config: Model configuration (default simple config created if None)
            backend: Backend engine ('norse', 'rockpool', 'pytorch')
            
        Returns:
            SNN model
        """
        # Handle None config for testing
        if config is None:
            # Create default config
            config = SNNModelConfig(
                input_size=10,
                hidden_size=20,
                output_size=2,
                dropout=0.0  # Ensure dropout is set
            )
        
        # Select backend if not specified
        if backend is None:
            if self.has_norse:
                backend = "norse"
            elif self.has_rockpool:
                backend = "rockpool"
            elif self.has_pytorch:
                backend = "pytorch"
            else:
                self.logger.error("No backend available for SNN model creation")
                return None
        
        # Create model with the selected backend, with fallback chain
        backend = backend.lower()
        
        # Try Norse backend
        if backend == "norse" and self.has_norse:
            try:
                model = self.create_norse_lif_model(config)
                if model is not None:
                    return model
                self.logger.warning("Failed to create Norse model, trying alternative backends")
            except Exception as e:
                self.logger.warning(f"Error with Norse backend: {e}, trying alternatives")
            
            # Norse failed, try next backend
            if self.has_rockpool:
                backend = "rockpool"
            elif self.has_pytorch:
                backend = "pytorch"
        
        # Try Rockpool backend
        if backend == "rockpool" and self.has_rockpool:
            try:
                model = self.create_rockpool_model(config)
                if model is not None:
                    return model
                self.logger.warning("Failed to create Rockpool model, trying alternative backends")
            except Exception as e:
                self.logger.warning(f"Error with Rockpool backend: {e}, trying alternatives")
            
            # Rockpool failed, try PyTorch
            if self.has_pytorch:
                backend = "pytorch"
        
        # Try PyTorch backend
        if backend == "pytorch" and self.has_pytorch:
            try:
                return self.create_pytorch_snn_model(config)
            except Exception as e:
                self.logger.error(f"Error creating PyTorch SNN model: {e}")
        
        # All backends failed or none available
        self.logger.error(f"All available SNN backends failed")
        return None

    def create_pytorch_snn_model(self, config: Optional[SNNModelConfig] = None) -> Optional[nn.Module]:
        """
        Create a basic PyTorch SNN implementation.
        
        Args:
            config: Model configuration
            
        Returns:
            PyTorch SNN model
        """
        if not self.has_pytorch:
            self.logger.error("PyTorch not available, cannot create model")
            return None
        
        # Handle case where config is None (for testing)
        if config is None:
            # Create default config
            config = SNNModelConfig(
                input_size=10,
                hidden_size=20,
                output_size=2,
                dropout=0.0  # Ensure dropout is set
            )
            
        try:
            # Define a simple PyTorch SNN model
            class PyTorchSNN(nn.Module):
                def __init__(self, input_size, hidden_size, output_size, threshold=1.0, 
                           tau=20.0, dropout=0.0):
                    super(PyTorchSNN, self).__init__()
                    
                    self.input_size = input_size
                    self.hidden_size = hidden_size
                    self.output_size = output_size
                    self.threshold = threshold
                    self.tau = tau
                    
                    # Layers
                    self.fc1 = nn.Linear(input_size, hidden_size)
                    self.fc2 = nn.Linear(hidden_size, output_size)
                    self.dropout = nn.Dropout(dropout) if dropout > 0 else None
                    
                    # Initialize membrane potential
                    self.register_buffer('mem_hidden', torch.zeros(1, hidden_size))
                    self.register_buffer('mem_output', torch.zeros(1, output_size))
                    
                def forward(self, x):
                    # Handle input dimensions - reshape if needed
                    if len(x.shape) == 2:  # [batch_size, input_size]
                        x = x.unsqueeze(1)  # Add time dimension [batch_size, 1, input_size]
                    
                    # Get dimensions
                    batch_size, time_steps, _ = x.shape
                    
                    # Initialize membrane potentials
                    mem_hidden = torch.zeros(batch_size, self.hidden_size, device=x.device)
                    mem_output = torch.zeros(batch_size, self.output_size, device=x.device)
                    
                    # Initialize outputs for each timestep
                    outputs = []
                    spikes_hidden = []
                    
                    # Decay factor
                    decay = torch.exp(torch.tensor(-1.0 / self.tau))
                    
                    # Process each timestep
                    for t in range(time_steps):
                        # Get input at current timestep
                        x_t = x[:, t, :]
                        
                        # First layer
                        current = self.fc1(x_t)
                        mem_hidden = mem_hidden * decay + current
                        
                        # Spike if membrane potential exceeds threshold
                        spike = (mem_hidden > self.threshold).float()
                        spikes_hidden.append(spike)
                        
                        # Reset membrane potential after spike
                        mem_hidden = mem_hidden * (1 - spike)
                        
                        # Apply dropout if enabled
                        if self.dropout is not None:
                            spike = self.dropout(spike)
                        
                        # Second layer
                        current = self.fc2(spike)
                        mem_output = mem_output * decay + current
                        
                        # Output is membrane potential
                        outputs.append(mem_output)
                    
                    # Stack outputs for all timesteps
                    return torch.stack(outputs, dim=1), torch.stack(spikes_hidden, dim=1)
            
            # Create model instance with safe attribute access
            model = PyTorchSNN(
                input_size=getattr(config, 'input_size', 10),
                hidden_size=getattr(config, 'hidden_size', 20),
                output_size=getattr(config, 'output_size', 2),
                threshold=getattr(config, 'threshold', 1.0),
                tau=getattr(config, 'tau_mem', 20.0),
                dropout=getattr(config, 'dropout', 0.0)  # Safe access to dropout
            )
            
            # Move to device if available
            if hasattr(config, 'device') and torch.cuda.is_available():
                device = torch.device(config.device)
                model = model.to(device)
            
            self.logger.info("Created PyTorch SNN model")
            return model
            
        except Exception as e:
            self.logger.error(f"Error creating PyTorch SNN model: {e}")
            return None
    
    # ----- STDP Learning Implementation -----
    
    def apply_stdp(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, 
                weights: torch.Tensor, learning_rate: Optional[float] = None,
                a_plus: Optional[float] = None, a_minus: Optional[float] = None,
                tau_plus: Optional[float] = None, tau_minus: Optional[float] = None) -> torch.Tensor:
        """
        Apply STDP learning rule to modify connection weights.
        
        Args:
            pre_spikes: Presynaptic spike trains [batch, time, n_pre]
            post_spikes: Postsynaptic spike trains [batch, time, n_post]
            weights: Connection weights [n_pre, n_post]
            learning_rate: Learning rate (default from config)
            a_plus: STDP potentiation factor (default from config)
            a_minus: STDP depression factor (default from config)
            tau_plus: STDP potentiation time constant (default from config)
            tau_minus: STDP depression time constant (default from config)
            
        Returns:
            Updated weights
        """
        if not self.has_pytorch:
            self.logger.error("PyTorch not available, cannot apply STDP")
            return weights
            
        # Get defaults from config if not provided
        learning_rate = learning_rate or self.config["stdp_learning_rate"]
        a_plus = a_plus or self.config["stdp_a_plus"]
        a_minus = a_minus or self.config["stdp_a_minus"]
        tau_plus = tau_plus or self.config["stdp_tau_plus"]
        tau_minus = tau_minus or self.config["stdp_tau_minus"]
        
        # Convert to torch tensors if needed
        if not isinstance(pre_spikes, torch.Tensor):
            pre_spikes = torch.tensor(pre_spikes, dtype=torch.float32)
            
        if not isinstance(post_spikes, torch.Tensor):
            post_spikes = torch.tensor(post_spikes, dtype=torch.float32)
            
        if not isinstance(weights, torch.Tensor):
            weights = torch.tensor(weights, dtype=torch.float32)
            
        # Move to device
        device = pre_spikes.device
        weights = weights.to(device)
        
        # Check dimensions
        batch_size, seq_length, n_pre = pre_spikes.shape
        _, _, n_post = post_spikes.shape
        
        # Create copy of weights for update
        new_weights = weights.clone()
        
        # Step through time for STDP
        for b in range(batch_size):
            # Calculate trace of pre-synaptic spikes
            pre_trace = torch.zeros(n_pre, device=device)
            
            # Calculate trace of post-synaptic spikes
            post_trace = torch.zeros(n_post, device=device)
            
            for t in range(seq_length):
                # Get spikes at this timestep
                pre_spike = pre_spikes[b, t]
                post_spike = post_spikes[b, t]
                
                # Update pre-synaptic trace
                pre_trace = pre_trace * torch.exp(torch.tensor(-1.0 / tau_plus, device=device))
                pre_trace += pre_spike
                
                # Update post-synaptic trace
                post_trace = post_trace * torch.exp(torch.tensor(-1.0 / tau_minus, device=device))
                post_trace += post_spike
                
                # Calculate weight updates
                # LTP: pre -> post (potentiation when pre spikes before post)
                ltp = a_plus * torch.outer(pre_trace, post_spike)
                
                # LTD: post -> pre (depression when post spikes before pre)
                ltd = a_minus * torch.outer(pre_spike, post_trace)
                
                # Apply weight updates
                dw = learning_rate * (ltp - ltd)
                new_weights += dw
                
        # Apply weight constraints (normalize to [0, 1])
        new_weights = torch.clamp(new_weights, 0.0, 1.0)
        
        return new_weights
    
    def apply_rstdp(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, 
                 weights: torch.Tensor, reward: float,
                 learning_rate: Optional[float] = None,
                 a_plus: Optional[float] = None, a_minus: Optional[float] = None,
                 tau_plus: Optional[float] = None, tau_minus: Optional[float] = None,
                 eligibility_trace_decay: Optional[float] = None) -> torch.Tensor:
        """
        Apply reward-modulated STDP learning rule.
        
        Args:
            pre_spikes: Presynaptic spike trains [batch, time, n_pre]
            post_spikes: Postsynaptic spike trains [batch, time, n_post]
            weights: Connection weights [n_pre, n_post]
            reward: Reward signal (scalar)
            learning_rate: Learning rate (default from config)
            a_plus: STDP potentiation factor (default from config)
            a_minus: STDP depression factor (default from config)
            tau_plus: STDP potentiation time constant (default from config)
            tau_minus: STDP depression time constant (default from config)
            eligibility_trace_decay: Decay rate for eligibility trace (default from config)
            
        Returns:
            Updated weights
        """
        if not self.has_pytorch:
            self.logger.error("PyTorch not available, cannot apply R-STDP")
            return weights
            
        # Get defaults from config if not provided
        learning_rate = learning_rate or self.config["stdp_learning_rate"]
        a_plus = a_plus or self.config["stdp_a_plus"]
        a_minus = a_minus or self.config["stdp_a_minus"]
        tau_plus = tau_plus or self.config["stdp_tau_plus"]
        tau_minus = tau_minus or self.config["stdp_tau_minus"]
        eligibility_trace_decay = eligibility_trace_decay or self.config["eligibility_trace_decay"]
        
        # Convert to torch tensors if needed
        if not isinstance(pre_spikes, torch.Tensor):
            pre_spikes = torch.tensor(pre_spikes, dtype=torch.float32)
            
        if not isinstance(post_spikes, torch.Tensor):
            post_spikes = torch.tensor(post_spikes, dtype=torch.float32)
            
        if not isinstance(weights, torch.Tensor):
            weights = torch.tensor(weights, dtype=torch.float32)
            
        # Move to device
        device = pre_spikes.device
        weights = weights.to(device)
        
        # Check dimensions
        batch_size, seq_length, n_pre = pre_spikes.shape
        _, _, n_post = post_spikes.shape
        
        # Create copy of weights for update
        new_weights = weights.clone()
        
        # Create eligibility trace matrix
        eligibility_trace = torch.zeros((n_pre, n_post), device=device)
        
        # Step through time for R-STDP
        for b in range(batch_size):
            # Calculate trace of pre-synaptic spikes
            pre_trace = torch.zeros(n_pre, device=device)
            
            # Calculate trace of post-synaptic spikes
            post_trace = torch.zeros(n_post, device=device)
            
            for t in range(seq_length):
                # Get spikes at this timestep
                pre_spike = pre_spikes[b, t]
                post_spike = post_spikes[b, t]
                
                # Update pre-synaptic trace
                pre_trace = pre_trace * torch.exp(torch.tensor(-1.0 / tau_plus, device=device))
                pre_trace += pre_spike
                
                # Update post-synaptic trace
                post_trace = post_trace * torch.exp(torch.tensor(-1.0 / tau_minus, device=device))
                post_trace += post_spike
                
                # Calculate eligibility trace updates
                # LTP: pre -> post (potentiation when pre spikes before post)
                ltp = a_plus * torch.outer(pre_trace, post_spike)
                
                # LTD: post -> pre (depression when post spikes before pre)
                ltd = a_minus * torch.outer(pre_spike, post_trace)
                
                # Update eligibility trace
                eligibility_trace = eligibility_trace * eligibility_trace_decay + (ltp - ltd)
                
        # Apply reward modulation to weights using eligibility trace
        new_weights += learning_rate * reward * eligibility_trace
        
        # Apply weight constraints (normalize to [0, 1])
        new_weights = torch.clamp(new_weights, 0.0, 1.0)
        
        return new_weights
    
    # ----- Model Training and Pattern Recognition Methods -----
    
    def train_pattern_recognition_snn(self, patterns: Dict[str, List[np.ndarray]], 
                                   config: Optional[SNNModelConfig] = None,
                                   learning_rule: Optional[Union[str, LearningRule]] = None,
                                   epochs: Optional[int] = None,
                                   batch_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Train an SNN for pattern recognition using STDP/R-STDP.
        
        Args:
            patterns: Dictionary of pattern label to list of examples
            config: Model configuration (or use defaults)
            learning_rule: Learning rule (STDP, R-STDP, etc.)
            epochs: Number of training epochs
            batch_size: Training batch size
            
        Returns:
            Dictionary with model and training metrics
        """
        if not self.has_pytorch:
            self.logger.error("PyTorch not available, cannot train SNN")
            return None
            
        # Get defaults from config if not provided
        if learning_rule is None:
            learning_rule = self.config["default_learning_rule"]
            
        if isinstance(learning_rule, LearningRule):
            learning_rule = str(learning_rule)
            
        # Get defaults from config
        epochs = epochs or self.config["training_epochs"]
        batch_size = batch_size or self.config["batch_size"]
        
        # Create default config if not provided
        if config is None:
            # Determine input size from patterns
            example_pattern = next(iter(patterns.values()))[0]
            input_size = len(example_pattern)
            
            # Create configuration
            config = SNNModelConfig(
                input_size=input_size,
                hidden_size=self.config["default_network_size"],
                output_size=len(patterns),  # One output per pattern class
                neuron_type=NeuronType.from_string(self.config["default_neuron_type"]),
                learning_rule=LearningRule.from_string(learning_rule),
                batch_size=batch_size,
                seq_length=self.config["encoding_time_window"],
                dt=0.001,  # 1ms time step
                threshold=self.config["membrane_threshold"],
                reset_voltage=self.config["membrane_reset"],
                tau_mem=self.config["tau_mem"],
                tau_syn=self.config["tau_syn"],
                learning_rate=self.config["stdp_learning_rate"],
                use_recurrent=self.config["recurrent_connections"],
                use_inhibition=self.config["inhibitory_connections"],
                dropout=0.0
            )
            
        # Create SNN model
        model = self.create_snn_model(config, backend="norse")
        
        if model is None:
            self.logger.error("Failed to create SNN model")
            return None
            
        # Prepare data
        encoded_patterns = {}
        for label, examples in patterns.items():
            encoded_examples = []
            
            for example in examples:
                # Encode pattern into spike train
                spike_train = self.encode_data(
                    example,
                    method=self.config["default_encoding"],
                    time_window=config.seq_length,
                    feature_name=label
                )
                
                if spike_train is not None:
                    encoded_examples.append(spike_train.spikes)
                    
            if encoded_examples:
                encoded_patterns[label] = encoded_examples
                
        if not encoded_patterns:
            self.logger.error("Failed to encode any patterns")
            return None
            
        # Create target tensors (one-hot encoded)
        label_to_idx = {label: i for i, label in enumerate(encoded_patterns.keys())}
        
        # Create dataset with all examples
        all_inputs = []
        all_targets = []
        
        for label, examples in encoded_patterns.items():
            for example in examples:
                all_inputs.append(example)
                
                # Create one-hot target
                target = torch.zeros(config.output_size)
                target[label_to_idx[label]] = 1.0
                all_targets.append(target)
                
        # Convert to tensors
        all_inputs = torch.stack(all_inputs)
        all_targets = torch.stack(all_targets)
        
        # Move to device
        device = torch.device(config.device)
        all_inputs = all_inputs.to(device)
        all_targets = all_targets.to(device)
        
        # Create dataloader
        dataset = torch.utils.data.TensorDataset(all_inputs, all_targets)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True
        )
        
        # Initialize training metrics
        metrics = {
            "loss_history": [],
            "accuracy_history": []
        }
        
        # Initialize weights (stored explicitly for STDP)
        input_weights = torch.rand((config.input_size, config.hidden_size), device=device) * 0.1
        hidden_weights = torch.rand((config.hidden_size, config.output_size), device=device) * 0.1
        
        # Initialize accumulation variables for rate learning
        rate_scaling = 10.0  # Scaling factor for rate-based learning
        
        # Training loop
        for epoch in range(epochs):
            total_loss = 0.0
            correct = 0
            total = 0
            
            for inputs, targets in dataloader:
                batch_size = inputs.size(0)
                
                # Forward pass
                outputs, hidden_spikes = model(inputs)
                
                # Get output and target spike rates
                output_rates = outputs[:, -1, :]  # Use final timestep output
                
                # Calculate classification
                _, predicted = torch.max(output_rates, 1)
                _, target_class = torch.max(targets, 1)
                
                # Update metrics
                correct += (predicted == target_class).sum().item()
                total += batch_size
                
                # Calculate loss
                loss = F.mse_loss(output_rates, targets)
                total_loss += loss.item() * batch_size
                
                # Apply learning rule
                if learning_rule.lower() == "stdp":
                    # Extract hidden layer spikes for STDP
                    # We need to use the inputs as pre-synaptic spikes for input->hidden
                    input_spikes = inputs
                    
                    # Use hidden spikes as pre-synaptic spikes for hidden->output
                    # Create output spikes based on targets (teacher forcing)
                    target_spikes = torch.zeros_like(outputs)
                    for b in range(batch_size):
                        class_idx = target_class[b].item()
                        # Set target spikes for correct class
                        target_spikes[b, -5:, class_idx] = 1.0  # Last 5 timesteps
                    
                    # Apply STDP to input -> hidden weights
                    input_weights = self.apply_stdp(
                        input_spikes,
                        hidden_spikes,
                        input_weights
                    )
                    
                    # Apply STDP to hidden -> output weights
                    hidden_weights = self.apply_stdp(
                        hidden_spikes,
                        target_spikes,
                        hidden_weights
                    )
                    
                    # Update model weights (need to map between our matrices and model)
                    with torch.no_grad():
                        model.fc_input.weight.data = input_weights.t()  # Transpose for PyTorch convention
                        model.fc_output.weight.data = hidden_weights.t()
                        
                elif learning_rule.lower() == "r_stdp":
                    # Apply reward-based STDP
                    # Calculate reward based on accuracy
                    reward = (predicted == target_class).float() * 2.0 - 1.0  # 1 for correct, -1 for incorrect
                    
                    # Apply R-STDP to input -> hidden weights
                    for b in range(batch_size):
                        b_reward = reward[b].item()
                        
                        # Apply R-STDP to batch element
                        input_weights = self.apply_rstdp(
                            inputs[b].unsqueeze(0),
                            hidden_spikes[b].unsqueeze(0),
                            input_weights,
                            b_reward
                        )
                        
                        # Apply R-STDP to hidden -> output
                        output_spikes = torch.zeros_like(outputs[b]).unsqueeze(0)
                        output_spikes[0, -5:, target_class[b].item()] = 1.0  # Teacher forcing
                        
                        hidden_weights = self.apply_rstdp(
                            hidden_spikes[b].unsqueeze(0),
                            output_spikes,
                            hidden_weights,
                            b_reward
                        )
                        
                    # Update model weights
                    with torch.no_grad():
                        model.fc_input.weight.data = input_weights.t()
                        model.fc_output.weight.data = hidden_weights.t()
                        
                else:
                    # Default: Use rate-based learning with PyTorch optimizers
                    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
            # Calculate epoch metrics
            epoch_loss = total_loss / total
            epoch_acc = correct / total
            
            # Store metrics
            metrics["loss_history"].append(epoch_loss)
            metrics["accuracy_history"].append(epoch_acc)
            
            # Log progress
            if (epoch + 1) % 10 == 0:
                self.logger.info(f"Epoch {epoch+1}/{epochs}: Loss={epoch_loss:.4f}, Acc={epoch_acc:.4f}")
                
        # Store final model and metrics
        result = {
            "model": model,
            "config": config,
            "metrics": metrics,
            "class_mapping": label_to_idx,
            "input_weights": input_weights,
            "hidden_weights": hidden_weights
        }
        
        # Store in models dictionary
        model_id = str(uuid.uuid4())
        with self._lock:
            self._models[model_id] = result
            
        self.logger.info(f"Trained SNN model with final accuracy: {metrics['accuracy_history'][-1]:.4f}")
        
        return {
            "model_id": model_id,
            "metrics": metrics,
            "classes": list(label_to_idx.keys()),
            "final_accuracy": metrics["accuracy_history"][-1]
        }
    
    def detect_pattern_with_snn(self, model_id: str, data: np.ndarray) -> Dict[str, Any]:
        """
        Detect patterns using a trained SNN model.
        
        Args:
            model_id: ID of the trained model
            data: Input data for detection
            
        Returns:
            Detection results
        """
        if not self.has_pytorch:
            self.logger.error("PyTorch not available, cannot detect patterns")
            return None
            
        # Get model from storage
        with self._lock:
            model_data = self._models.get(model_id)
            
        if model_data is None:
            self.logger.error(f"Model with ID {model_id} not found")
            return None
            
        model = model_data["model"]
        config = model_data["config"]
        class_mapping = model_data["class_mapping"]
        
        # Inverse class mapping
        idx_to_class = {idx: label for label, idx in class_mapping.items()}
        
        # Encode input data
        spike_train = self.encode_data(
            data,
            method=self.config["default_encoding"],
            time_window=config.seq_length,
            feature_name="detection"
        )
        
        if spike_train is None:
            self.logger.error("Failed to encode input data")
            return None
            
        # Move to device
        device = torch.device(config.device)
        inputs = spike_train.spikes.unsqueeze(0).to(device)  # Add batch dimension
        
        # Run model
        with torch.no_grad():
            outputs, _ = model(inputs)
            
        # Get output rates
        output_rates = outputs[0, -1, :].cpu().numpy()  # Remove batch dimension
        
        # Get prediction
        predicted_idx = np.argmax(output_rates)
        predicted_class = idx_to_class.get(predicted_idx, "unknown")
        
        # Calculate confidence
        confidence = output_rates[predicted_idx]
        
        # Normalize output rates to probabilities
        probabilities = output_rates / np.sum(output_rates)
        
        # Create results
        results = {
            "predicted_class": predicted_class,
            "confidence": float(confidence),
            "probabilities": {idx_to_class.get(i, f"class_{i}"): float(p) for i, p in enumerate(probabilities)}
        }
        
        return results
    
    # ----- Market Regime Analysis Methods -----
    
    def analyze_market_regime_with_snn(self, df: pd.DataFrame, feature_types: Optional[List[str]] = None,
                                    model: Optional[Any] = None, model_id: Optional[str] = None,
                                    sequence_length: Optional[int] = None) -> Dict[str, Any]:
        """
        Analyze market regime using SNN-based pattern recognition.
        
        Args:
            df: OHLCV dataframe
            feature_types: Types of features to extract
            model: Pre-trained SNN model (or use model_id)
            model_id: ID of the trained model (or use model)
            sequence_length: Length of sequence for analysis
            
        Returns:
            Market regime analysis results
        """
        if not self.has_pytorch:
            self.logger.error("PyTorch not available, cannot analyze market regime")
            return {}
            
        # Get model if model_id provided
        if model is None and model_id is not None:
            with self._lock:
                model_data = self._models.get(model_id)
                
            if model_data is None:
                self.logger.error(f"Model with ID {model_id} not found")
                return {}
                
            model = model_data["model"]
            
        # If still no model, need to create one
        if model is None:
            self.logger.warning("No model provided for regime analysis, using default model")
            # TODO: Implement default regime analysis model
            return {}
            
        # Get defaults from config if not provided
        if feature_types is None:
            feature_types = self.config["default_features"]
            
        if sequence_length is None:
            sequence_length = 50  # Default sequence length
            
        # Extract features
        features = self.extract_features(df, feature_types)
        
        if not features:
            self.logger.error("Failed to extract features from dataframe")
            return {}
            
        # Prepare sequences for analysis
        sequences = []
        
        # Combine all features into a single array
        combined_features = []
        feature_names = []
        
        for name, feature_array in features.items():
            combined_features.append(feature_array)
            feature_names.append(name)
            
        # Stack features
        feature_matrix = np.column_stack(combined_features)
        
        # Create sequences (sliding window)
        total_samples = len(feature_matrix)
        
        for i in range(total_samples - sequence_length + 1):
            seq = feature_matrix[i:i+sequence_length, :]
            sequences.append(seq)
            
        # Process each sequence
        regimes = []
        confidences = []
        
        for seq in sequences:
            # Encode sequence
            # Flatten sequence for processing
            flat_seq = seq.flatten()
            
            # Detect pattern
            result = self.detect_pattern_with_snn(model_id, flat_seq)
            
            if result:
                regime = result["predicted_class"]
                confidence = result["confidence"]
                
                regimes.append(regime)
                confidences.append(confidence)
            else:
                regimes.append("unknown")
                confidences.append(0.0)
                
        # Create result dataframe
        result_df = pd.DataFrame({
            "regime": regimes,
            "confidence": confidences
        })
        
        # Pad with NaN for initial sequence_length-1 rows
        pad_df = pd.DataFrame({
            "regime": ["unknown"] * (sequence_length - 1),
            "confidence": [0.0] * (sequence_length - 1)
        })
        
        result_df = pd.concat([pad_df, result_df]).reset_index(drop=True)
        
        # Calculate summary statistics
        regime_counts = result_df["regime"].value_counts().to_dict()
        
        # Detect regime transitions
        transitions = []
        current_regime = "unknown"
        
        for i, regime in enumerate(result_df["regime"]):
            if regime != current_regime and regime != "unknown":
                transitions.append({
                    "from": current_regime,
                    "to": regime,
                    "index": i,
                    "confidence": result_df["confidence"].iloc[i]
                })
                current_regime = regime
                
        # Calculate average regime duration
        durations = {}
        for regime in set(result_df["regime"]):
            if regime == "unknown":
                continue
                
            # Find all segments
            in_segment = False
            segment_start = 0
            segment_durations = []
            
            for i, r in enumerate(result_df["regime"]):
                if r == regime and not in_segment:
                    # Start of segment
                    in_segment = True
                    segment_start = i
                elif r != regime and in_segment:
                    # End of segment
                    in_segment = False
                    segment_durations.append(i - segment_start)
                    
            # Handle last segment
            if in_segment:
                segment_durations.append(len(result_df["regime"]) - segment_start)
                
            if segment_durations:
                durations[regime] = {
                    "mean": np.mean(segment_durations),
                    "median": np.median(segment_durations),
                    "min": np.min(segment_durations),
                    "max": np.max(segment_durations)
                }
                
        # Create result
        result = {
            "regimes": result_df["regime"].tolist(),
            "confidences": result_df["confidence"].tolist(),
            "current_regime": result_df["regime"].iloc[-1],
            "current_confidence": result_df["confidence"].iloc[-1],
            "regime_counts": regime_counts,
            "transitions": transitions,
            "durations": durations
        }
        
        return result

# Neuromorphic engine with Norse and Rockpool
class NeuromorphicEngine:
    """
    Neuromorphic computing engine using Norse for Spiking Neural Networks
    and Rockpool for STDP learning.
    """
    
    def __init__(self, learning_rate=0.01, stdp_enabled=True):
        """
        Initialize the neuromorphic engine.
        
        Args:
            learning_rate: Learning rate for STDP
            stdp_enabled: Whether to enable STDP learning
        """
        self.learning_rate = learning_rate
        self.stdp_enabled = stdp_enabled
        self.logger = logging.getLogger(__name__ + ".NeuromorphicEngine")
        
        # Check if required libraries are available
        if not NORSE_AVAILABLE:
            self.logger.warning("Norse library not available. Neuromorphic features limited.")
        
        if not ROCKPOOL_AVAILABLE:
            self.logger.warning("Rockpool library not available. STDP features disabled.")
            self.stdp_enabled = False
        
        # Model storage
        self.models = {}
        self.encoders = {}
        self.optimizers = {}
        
        # Lock for thread safety
        self.lock = threading.RLock()
        
        # Initialize components if available
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize neuromorphic components if available."""
        if not NORSE_AVAILABLE or not ROCKPOOL_AVAILABLE:
            return
        
        try:
            # Initialize spike encoders
            self._initialize_encoders()
            
            # Initialize default SNN models
            self._initialize_models()
            
            self.logger.info("Neuromorphic components initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize neuromorphic components: {e}")
    
    def _initialize_encoders(self):
        """Initialize spike encoders."""
        if not NORSE_AVAILABLE:
            return
        
        try:
            import norse.torch as norse
            
            # Initialize different encoder types
            
            # Constant rate encoder
            self.encoders['constant_rate'] = norse.encoder.ConstantCurrentLIFEncoder(
                sequence_length=100,
                current_amplitude=0.5
            )
            
            # Poisson encoder
            self.encoders['poisson'] = norse.encoder.PoissonEncoder(
                sequence_length=100
            )
            
            # Population encoder
            self.encoders['population'] = norse.encoder.PopulationEncoder(
                sequence_length=100,
                population_size=10,
                tau=5.0
            )
            
            # Spike latency encoder
            self.encoders['latency'] = torch.nn.Sequential(
                torch.nn.Linear(1, 10),  # Input dimensionality to population size
                torch.nn.ReLU(),
                norse.torch.LIFParameters(method="super", alpha=100.0)
            )
            
            self.logger.debug("Spike encoders initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize spike encoders: {e}")
    
    def _initialize_models(self):
        """Initialize SNN models."""
        if not NORSE_AVAILABLE:
            return
        
        try:
            import norse.torch as norse
            
            # Basic LIF network
            class LIFNetwork(torch.nn.Module):
                def __init__(self, input_size=10, hidden_size=20, output_size=1):
                    super(LIFNetwork, self).__init__()
                    self.input_size = input_size
                    self.hidden_size = hidden_size
                    self.output_size = output_size
                    
                    self.fc1 = torch.nn.Linear(input_size, hidden_size)
                    self.lif1 = norse.LIFCell(p=norse.LIFParameters(method="super", alpha=100.0))
                    self.fc2 = torch.nn.Linear(hidden_size, output_size)
                    self.lif2 = norse.LIFCell(p=norse.LIFParameters(method="super", alpha=100.0))
                
                def forward(self, x):
                    # Initial state
                    s1 = self.lif1.initial_state(x.shape[0], device=x.device)
                    s2 = self.lif2.initial_state(x.shape[0], device=x.device)
                    
                    # Iterate through time
                    seq_length = x.shape[1]
                    outputs = []
                    
                    for t in range(seq_length):
                        z = self.fc1(x[:, t, :])
                        z, s1 = self.lif1(z, s1)
                        z = self.fc2(z)
                        z, s2 = self.lif2(z, s2)
                        outputs.append(z)
                    
                    return torch.stack(outputs, dim=1)
            
            # Recurrent network
            class LIFRecurrent(torch.nn.Module):
                def __init__(self, input_size=10, hidden_size=20, output_size=1):
                    super(LIFRecurrent, self).__init__()
                    self.input_size = input_size
                    self.hidden_size = hidden_size
                    self.output_size = output_size
                    
                    self.recurrent = norse.LIFRecurrent(
                        input_size, 
                        hidden_size,
                        p=norse.LIFParameters(method="super", alpha=100.0)
                    )
                    self.fc = torch.nn.Linear(hidden_size, output_size)
                    self.lif_out = norse.LIFCell(p=norse.LIFParameters(method="super", alpha=100.0))
                
                def forward(self, x):
                    # Initial state
                    state = None
                    s_out = self.lif_out.initial_state(x.shape[0], device=x.device)
                    
                    # Iterate through time
                    seq_length = x.shape[1]
                    outputs = []
                    
                    for t in range(seq_length):
                        z, state = self.recurrent(x[:, t, :], state)
                        z = self.fc(z)
                        z, s_out = self.lif_out(z, s_out)
                        outputs.append(z)
                    
                    return torch.stack(outputs, dim=1)
            
            # Store models
            self.models['lif_network'] = LIFNetwork()
            self.models['lif_recurrent'] = LIFRecurrent()
            
            # STDP model using Rockpool (if available)
            if ROCKPOOL_AVAILABLE:
                self._initialize_stdp_model()
            
            self.logger.debug("SNN models initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize SNN models: {e}")
    
    def _initialize_stdp_model(self):
        """Initialize STDP model using Rockpool."""
        if not ROCKPOOL_AVAILABLE:
            return
        
        try:
            import rockpool.nn.modules as rpm
            import rockpool.parameters as rp
            
            # Create a simple STDP network
            stdp_model = rpm.RecurrentJAXSpikingNet(
                N_in=10,           # Input neurons
                N_rec=20,          # Recurrent neurons
                N_out=1,           # Output neurons
                has_rec=True,      # Use recurrent connections
                tau_mem=0.02,      # Membrane time constant
                tau_syn=0.01,      # Synaptic time constant
                threshold=1.0,     # Firing threshold
                dt=0.001,          # Timestep
                weight_variance=0.1,  # Weight initialization variance
                # STDP parameters
                learning_rule="exponential",
                update_output=True,
                plasticity_rule="stdp",
                stdp_learning_rate=self.learning_rate,
                stdp_energy=0.1
            )
            
            # Store model
            self.models['stdp_network'] = stdp_model
            
            self.logger.debug("STDP model initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize STDP model: {e}")
    
    def process_data(self, dataframe):
        """
        Process data with neuromorphic models.
        
        Args:
            dataframe: DataFrame with financial data
            
        Returns:
            dict: Processing results
        """
        if not NORSE_AVAILABLE:
            self.logger.warning("Norse not available. Cannot process data.")
            return {}
        
        try:
            # Extract features from dataframe
            features = self._extract_features(dataframe)
            
            # Encode features as spikes
            encoded_data = self._encode_features(features)
            
            # Process with SNN models
            results = {}
            for model_name, model in self.models.items():
                if model_name in ['lif_network', 'lif_recurrent']:
                    # Process with Norse models
                    output = self._process_with_norse_model(encoded_data, model_name)
                    results[model_name] = output
                elif model_name == 'stdp_network' and ROCKPOOL_AVAILABLE:
                    # Process with Rockpool STDP model
                    output = self._process_with_stdp_model(features)
                    results[model_name] = output
            
            # Generate signals from results
            signals = self._generate_signals(results, len(dataframe))
            
            return signals
        except Exception as e:
            self.logger.error(f"Error processing data with neuromorphic models: {e}")
            return {}
    
    def _extract_features(self, dataframe):
        """
        Extract features from financial data.
        
        Args:
            dataframe: DataFrame with financial data
            
        Returns:
            dict: Extracted features
        """
        features = {}
        
        try:
            # Check if required columns exist
            ohlc_columns = ['open', 'high', 'low', 'close']
            if not all(col in dataframe.columns for col in ohlc_columns):
                self.logger.warning("OHLC columns missing in dataframe")
                return features
            
            # Calculate returns
            if 'close' in dataframe.columns:
                close = dataframe['close'].values
                returns = np.zeros_like(close)
                returns[1:] = (close[1:] - close[:-1]) / close[:-1]
                features['returns'] = returns
            
            # Calculate volatility
            if 'close' in dataframe.columns:
                window = min(20, len(dataframe) // 4)
                if window > 1:
                    volatility = np.zeros_like(close)
                    for i in range(window, len(close)):
                        volatility[i] = np.std(returns[i-window:i])
                    features['volatility'] = volatility
            
            # Calculate price momentum
            if 'close' in dataframe.columns:
                momentum = np.zeros_like(close)
                momentum[1:] = close[1:] - close[:-1]
                features['momentum'] = momentum
            
            # Calculate volume momentum if volume is available
            if 'volume' in dataframe.columns:
                volume = dataframe['volume'].values
                vol_momentum = np.zeros_like(volume)
                vol_momentum[1:] = volume[1:] - volume[:-1]
                features['volume_momentum'] = vol_momentum
            
            # Calculate price position relative to recent range
            if all(col in dataframe.columns for col in ['high', 'low', 'close']):
                window = min(20, len(dataframe) // 4)
                if window > 1:
                    pos_in_range = np.zeros_like(close)
                    for i in range(window, len(close)):
                        recent_high = np.max(dataframe['high'].values[i-window:i])
                        recent_low = np.min(dataframe['low'].values[i-window:i])
                        range_size = recent_high - recent_low
                        if range_size > 0:
                            pos_in_range[i] = (close[i] - recent_low) / range_size
                        else:
                            pos_in_range[i] = 0.5
                    features['price_position'] = pos_in_range
            
            return features
        except Exception as e:
            self.logger.error(f"Error extracting features: {e}")
            return {}
    
    def _encode_features(self, features):
        """
        Encode features as spike trains.
        
        Args:
            features: Dictionary of features
            
        Returns:
            dict: Encoded spike trains
        """
        if not NORSE_AVAILABLE or not self.encoders:
            return {}
        
        encoded_data = {}
        
        try:
            for feature_name, feature_values in features.items():
                # Normalize feature to [0, 1] range
                normalized = self._normalize_feature(feature_values)
                
                # Convert to tensor
                feature_tensor = torch.tensor(normalized, dtype=torch.float32).unsqueeze(1)
                
                # Encode with different encoders
                for encoder_name, encoder in self.encoders.items():
                    try:
                        # Skip latency encoder for now (needs special handling)
                        if encoder_name == 'latency':
                            continue
                        
                        # Encode feature values into spikes
                        spikes = encoder(feature_tensor)
                        
                        # Store encoded data
                        encoded_data[f"{feature_name}_{encoder_name}"] = spikes
                    except Exception as enc_error:
                        self.logger.debug(f"Error encoding {feature_name} with {encoder_name}: {enc_error}")
            
            return encoded_data
        except Exception as e:
            self.logger.error(f"Error encoding features: {e}")
            return {}
    
    def _normalize_feature(self, feature_values):
        """
        Normalize feature values to [0, 1] range.
        
        Args:
            feature_values: Feature values array
            
        Returns:
            array: Normalized feature values
        """
        values = np.array(feature_values)
        
        # Handle flat values
        if np.max(values) == np.min(values):
            return np.ones_like(values) * 0.5
        
        # Normalize to [0, 1]
        normalized = (values - np.min(values)) / (np.max(values) - np.min(values))
        
        return normalized
    
    def _process_with_norse_model(self, encoded_data, model_name):
        """
        Process encoded data with Norse SNN model.
        
        Args:
            encoded_data: Dictionary of encoded spike trains
            model_name: Name of the model to use
            
        Returns:
            dict: Processing results
        """
        if not NORSE_AVAILABLE:
            return {}
        
        try:
            model = self.models.get(model_name)
            if model is None:
                self.logger.warning(f"Model {model_name} not found")
                return {}
            
            # Process each feature
            results = {}
            for feature_name, spikes in encoded_data.items():
                # Skip if this isn't a valid tensor
                if not isinstance(spikes, torch.Tensor):
                    continue
                
                # Prepare input for model
                # Reshape if necessary [batch, time, features]
                if len(spikes.shape) == 2:  # [time, features]
                    x = spikes.unsqueeze(0)  # Add batch dimension
                elif len(spikes.shape) == 3:  # [batch, time, features]
                    x = spikes
                else:
                    continue  # Skip invalid shape
                
                # Process with model
                with torch.no_grad():
                    try:
                        output = model(x)
                        
                        # Extract output spikes
                        if isinstance(output, torch.Tensor):
                            # Convert to numpy for storage
                            output_np = output.squeeze().cpu().numpy()
                            results[feature_name] = output_np
                    except Exception as model_error:
                        self.logger.debug(f"Error processing {feature_name} with {model_name}: {model_error}")
            
            return results
        except Exception as e:
            self.logger.error(f"Error processing with Norse model: {e}")
            return {}
    
    def _process_with_stdp_model(self, features):
        """
        Process features with Rockpool STDP model.
        
        Args:
            features: Dictionary of features
            
        Returns:
            dict: Processing results
        """
        if not ROCKPOOL_AVAILABLE:
            return {}
        
        try:
            model = self.models.get('stdp_network')
            if model is None:
                self.logger.warning("STDP model not found")
                return {}
            
            # Combine features into input array
            feature_names = list(features.keys())
            
            if not feature_names:
                return {}
            
            # Get the length of the first feature
            first_feature = features[feature_names[0]]
            data_length = len(first_feature)
            
            # Prepare input matrix [time, features]
            input_data = np.zeros((data_length, len(feature_names)))
            
            for i, name in enumerate(feature_names):
                # Get feature values and ensure correct length
                values = features[name]
                if len(values) == data_length:
                    input_data[:, i] = self._normalize_feature(values)
            
            # Ensure the expected input size
            expected_size = model.N_in
            if input_data.shape[1] < expected_size:
                # Pad with zeros
                padding = np.zeros((data_length, expected_size - input_data.shape[1]))
                input_data = np.concatenate([input_data, padding], axis=1)
            elif input_data.shape[1] > expected_size:
                # Truncate
                input_data = input_data[:, :expected_size]
            
            # Process with model
            output, state, recordings = model(input_data, record=True)
            
            # Extract results
            results = {
                'output_spikes': output,
                'neuron_states': recordings.get('Vmem', []),
                'synapse_weights': state.get('weights_out', [])
            }
            
            # Update model if STDP is enabled
            if self.stdp_enabled:
                # Learning happens automatically when using the model with `update_output=True`
                pass
            
            return results
        except Exception as e:
            self.logger.error(f"Error processing with STDP model: {e}")
            return {}
    
    def _generate_signals(self, model_results, data_length):
        """
        Generate trading signals from model results.
        
        Args:
            model_results: Dictionary of model outputs
            data_length: Length of original data
            
        Returns:
            dict: Trading signals
        """
        signals = {}
        
        try:
            # Combine results from different models
            
            # 1. Extract outputs from LIF models
            lif_outputs = {}
            if 'lif_network' in model_results:
                lif_outputs.update(model_results['lif_network'])
            
            if 'lif_recurrent' in model_results:
                lif_outputs.update(model_results['lif_recurrent'])
            
            # 2. Generate signal from LIF outputs
            if lif_outputs:
                # Combine signals from different features
                combined_signal = np.zeros(data_length)
                count = 0
                
                for feature_name, output in lif_outputs.items():
                    if len(output) == data_length:
                        # Sum positive spike outputs
                        if output.ndim > 1:
                            # Take the mean across neurons if multiple
                            feature_signal = np.mean(output, axis=-1)
                        else:
                            feature_signal = output
                        
                        combined_signal += feature_signal
                        count += 1
                
                if count > 0:
                    combined_signal /= count
                    
                    # Normalize to [-1, 1] range
                    max_abs = np.max(np.abs(combined_signal))
                    if max_abs > 0:
                        combined_signal /= max_abs
                    
                    signals['snn_signal'] = combined_signal.tolist()
            
            # 3. Generate signal from STDP model
            if 'stdp_network' in model_results:
                stdp_results = model_results['stdp_network']
                
                if 'output_spikes' in stdp_results:
                    output_spikes = stdp_results['output_spikes']
                    
                    if len(output_spikes) == data_length:
                        # Convert binary spikes to signal between [-1, 1]
                        if output_spikes.ndim > 1:
                            # Take the mean across neurons if multiple
                            stdp_signal = np.mean(output_spikes, axis=-1)
                        else:
                            stdp_signal = output_spikes
                        
                        # Apply rolling window to smooth
                        window = min(10, data_length // 10)
                        if window > 1:
                            smoothed = np.zeros_like(stdp_signal)
                            for i in range(window, len(stdp_signal)):
                                smoothed[i] = np.mean(stdp_signal[i-window:i])
                            stdp_signal = smoothed
                        
                        # Scale to [-1, 1]
                        max_abs = np.max(np.abs(stdp_signal))
                        if max_abs > 0:
                            stdp_signal /= max_abs
                        
                        signals['stdp_signal'] = stdp_signal.tolist()
            
            return signals
        except Exception as e:
            self.logger.error(f"Error generating signals: {e}")
            return {}
    
    def train(self, features, target=None, epochs=100):
        """
        Train neuromorphic models with data.
        
        Args:
            features: Input features (array or DataFrame)
            target: Target values (optional)
            epochs: Number of training epochs
            
        Returns:
            dict: Training results and metrics
        """
        if not NORSE_AVAILABLE:
            self.logger.warning("Norse not available. Cannot train models.")
            return {"status": "error", "message": "Norse not available"}
        
        try:
            # Extract features if DataFrame is provided
            if isinstance(features, pd.DataFrame):
                feature_dict = self._extract_features(features)
                
                # Convert to array format
                feature_names = list(feature_dict.keys())
                if not feature_names:
                    return {"status": "error", "message": "No features extracted"}
                
                # Get the length of the first feature
                first_feature = feature_dict[feature_names[0]]
                data_length = len(first_feature)
                
                # Prepare input matrix [samples, features]
                X = np.zeros((data_length, len(feature_names)))
                
                for i, name in enumerate(feature_names):
                    values = feature_dict[name]
                    if len(values) == data_length:
                        X[:, i] = self._normalize_feature(values)
            else:
                # Use provided array
                X = np.array(features)
                
                # Normalize features
                for i in range(X.shape[1]):
                    X[:, i] = self._normalize_feature(X[:, i])
            
            # Prepare target if provided
            if target is not None:
                Y = np.array(target)
                
                # Normalize target
                Y = self._normalize_feature(Y)
            else:
                # Use next timestep features as target (simple prediction task)
                Y = np.zeros((X.shape[0], X.shape[1]))
                Y[:-1, :] = X[1:, :]
                Y[-1, :] = X[-1, :]
            
            # Train different models
            results = {}
            
            # Train STDP model if available
            if ROCKPOOL_AVAILABLE and self.stdp_enabled:
                stdp_results = self._train_stdp_model(X, Y, epochs)
                results['stdp_model'] = stdp_results
            
            # Train Norse models
            if NORSE_AVAILABLE:
                norse_results = self._train_norse_models(X, Y, epochs)
                results.update(norse_results)
            
            return {
                "status": "success", 
                "results": results,
                "message": "Training completed"
            }
        except Exception as e:
            self.logger.error(f"Error training models: {e}")
            return {"status": "error", "message": str(e)}
    
    def _train_stdp_model(self, X, Y, epochs):
        """
        Train Rockpool STDP model.
        
        Args:
            X: Input features
            Y: Target values
            epochs: Number of training epochs
            
        Returns:
            dict: Training results
        """
        if not ROCKPOOL_AVAILABLE:
            return {"status": "error", "message": "Rockpool not available"}
        
        try:
            model = self.models.get('stdp_network')
            if model is None:
                self.logger.warning("STDP model not found")
                return {"status": "error", "message": "STDP model not found"}
            
            # Ensure the expected input size
            expected_size = model.N_in
            if X.shape[1] < expected_size:
                # Pad with zeros
                padding = np.zeros((X.shape[0], expected_size - X.shape[1]))
                X = np.concatenate([X, padding], axis=1)
            elif X.shape[1] > expected_size:
                # Truncate
                X = X[:, :expected_size]
            
            # Training loop
            losses = []
            
            for epoch in range(epochs):
                epoch_loss = 0
                
                # Process each sample
                for i in range(X.shape[0]):
                    # Get input sample
                    x = X[i:i+1, :]
                    y = Y[i:i+1, :]
                    
                    # Forward pass with learning
                    output, new_state, recordings = model(x, record=True)
                    
                    # Compute loss (simple MSE)
                    loss = np.mean((output - y)**2)
                    epoch_loss += loss
                
                # Average loss for epoch
                epoch_loss /= X.shape[0]
                losses.append(epoch_loss)
                
                # Log progress
                if epoch % 10 == 0:
                    self.logger.debug(f"STDP Epoch {epoch}/{epochs}, Loss: {epoch_loss:.6f}")
            
            return {
                "status": "success",
                "losses": losses,
                "final_loss": losses[-1] if losses else float('inf')
            }
        except Exception as e:
            self.logger.error(f"Error training STDP model: {e}")
            return {"status": "error", "message": str(e)}
    
    def _train_norse_models(self, X, Y, epochs):
        """
        Train Norse SNN models.
        
        Args:
            X: Input features
            Y: Target values
            epochs: Number of training epochs
            
        Returns:
            dict: Training results
        """
        if not NORSE_AVAILABLE:
            return {}
        
        try:
            results = {}
            
            # Train LIF network
            lif_network = self.models.get('lif_network')
            if lif_network is not None:
                # Create a new instance for training
                input_size = min(X.shape[1], lif_network.input_size)
                model = type(lif_network)(input_size=input_size)
                
                # Create optimizer
                optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
                
                # Convert data to tensors
                X_tensor = torch.tensor(X[:, :input_size], dtype=torch.float32)
                Y_tensor = torch.tensor(Y, dtype=torch.float32)
                
                # Prepare dataset
                dataset = TensorDataset(X_tensor, Y_tensor)
                dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
                
                # Training loop
                losses = []
                
                for epoch in range(epochs):
                    epoch_loss = 0
                    
                    for batch_x, batch_y in dataloader:
                        # Convert to sequential format
                        batch_x = batch_x.unsqueeze(1)  # [batch, time=1, features]
                        batch_y = batch_y.unsqueeze(1)  # [batch, time=1, target]
                        
                        # Forward pass
                        optimizer.zero_grad()
                        output = model(batch_x)
                        
                        # Loss calculation
                        loss = torch.nn.functional.mse_loss(output, batch_y)
                        
                        # Backward pass
                        loss.backward()
                        optimizer.step()
                        
                        epoch_loss += loss.item()
                    
                    # Log progress
                    if epoch % 10 == 0:
                        self.logger.debug(f"LIF Network Epoch {epoch}/{epochs}, Loss: {epoch_loss:.6f}")
                    
                    losses.append(epoch_loss)
                
                # Save trained model
                with self.lock:
                    self.models['lif_network_trained'] = model
                
                results['lif_network'] = {
                    "status": "success",
                    "losses": losses,
                    "final_loss": losses[-1] if losses else float('inf')
                }
            
            return results
        except Exception as e:
            self.logger.error(f"Error training Norse models: {e}")
            return {}
