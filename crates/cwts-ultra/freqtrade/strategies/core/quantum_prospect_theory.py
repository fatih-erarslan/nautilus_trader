#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 10 14:50:47 2025

@author: ashina
"""

"""
Quantum-Enhanced Prospect Theory Implementation
----------------------------------------------
Implements hardware-accelerated Prospect Theory with quantum computing enhancements.
Provides 7 distinct quantum applications of Prospect Theory for trading.
"""

import numpy as np
import pennylane as qml

# Create a compatibility layer for PennyLane 0.41.0 where qml.math structure has changed
if not hasattr(qml, 'math'):
    # Create a minimal math namespace with required functions to avoid import errors
    class MathCompatLayer:
        def __init__(self):
            pass
            
        def get_interface(self, tensor):
            # Simple implementation to handle basic tensor type checks
            if hasattr(tensor, 'dtype') and hasattr(tensor, 'shape'):
                return "numpy"
            return None
            
        def is_abstract(self, tensor):
            # Simple implementation that should work for most cases
            return False
    
    # Attach the compatibility layer to qml
    qml.math = MathCompatLayer()

from enum import Enum
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from functools import wraps, lru_cache
import logging
import time
import json
import argparse
import multiprocessing as mp
import threading
from concurrent.futures import ThreadPoolExecutor
from numba import njit, prange, vectorize, float32, float64, cuda, jit
import math
from collections import deque
import warnings
import os
import psutil
import json
from datetime import datetime


# Standard factor definitions for system-wide consistency
class StandardFactors(Enum):
    """Standard factors to ensure alignment across all quantum components"""
    TREND = "trend"
    VOLATILITY = "volatility"
    MOMENTUM = "momentum"
    SENTIMENT = "sentiment"
    LIQUIDITY = "liquidity"
    CORRELATION = "correlation"
    CYCLE = "cycle"
    ANOMALY = "anomaly"
    
    @classmethod
    def get_ordered_list(cls) -> List[str]:
        """Get standard ordered list of factor names"""
        return [factor.value for factor in cls]
    
    @classmethod
    def get_default_weights(cls) -> Dict[str, float]:
        """Get initial default weights for standard factors"""
        return {
            cls.TREND.value: 0.600,        # Higher weight for trend
            cls.VOLATILITY.value: 0.500,   # Medium-high weight for volatility
            cls.MOMENTUM.value: 0.550,     # Medium-high weight for momentum
            cls.SENTIMENT.value: 0.450,    # Medium weight for sentiment
            cls.LIQUIDITY.value: 0.350,    # Lower weight for liquidity
            cls.CORRELATION.value: 0.400,  # Medium-low weight for correlation
            cls.CYCLE.value: 0.500,        # Medium-high weight for cycle
            cls.ANOMALY.value: 0.300       # Lower weight for anomaly (rare events)
        }
    
    @classmethod
    def get_factor_index(cls, factor_name: str) -> int:
        """Get index of a factor in the standard ordering"""
        try:
            return cls.get_ordered_list().index(factor_name)
        except ValueError:
            return -1  # Factor not found


# Import Numba warning classes
try:
    from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning, NumbaWarning
except ImportError:
    try:
        # For older versions of Numba
        from numba.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning, NumbaWarning
    except ImportError:
        # Define dummy classes if Numba not available
        class NumbaDeprecationWarning(Warning): pass
        class NumbaPendingDeprecationWarning(Warning): pass
        class NumbaWarning(Warning): pass

# Suppress warnings from Numba
warnings.filterwarnings('ignore', category=NumbaDeprecationWarning)
warnings.filterwarnings('ignore', category=NumbaPendingDeprecationWarning)
warnings.filterwarnings('ignore', category=NumbaWarning)

# Constants
DEFAULT_ALPHA = 0.88  # Value function exponent for gains
DEFAULT_BETA = 0.88   # Value function exponent for losses
DEFAULT_LAMBDA = 2.25  # Loss aversion parameter
DEFAULT_GAMMA = 0.61  # Probability weighting parameter
DEFAULT_DELTA = 0.69  # Probability weighting parameter (rarely used)
DEFAULT_QUBITS = 4   # Default number of qubits to use
DEFAULT_SHOTS = None  # Default to exact computation (no sampling)
DEFAULT_LAYERS = 2    # Default number of variational layers
DEFAULT_CACHE_SIZE = 1024  # LRU cache size
DEFAULT_BATCH_SIZE = 32    # Default batch size for processing

# Check if we can import the hardware management modules
# If not available, we'll create stubs for testing
HARDWARE_ACCEL_AVAILABLE = False
try:
    from cdfa_extensions.hw_acceleration import HardwareAccelerator
    from hardware_manager import HardwareManager
    HARDWARE_ACCEL_AVAILABLE = True
except ImportError:
    class HardwareManager:
        @staticmethod
        def get_manager():
            return HardwareManager()
        
        def __init__(self):
            self.quantum_available = False
            self.gpu_available = False
            
        def initialize_hardware(self):
            pass
    
    class HardwareAccelerator:
        def __init__(self, enable_gpu=False):
            self.enable_gpu = enable_gpu
            
        def get_accelerator_type(self):
            return "CPU"


# Enums for configuring the quantum PT implementation
class ProcessingMode(Enum):
    """Specifies which processing path to use for computation."""
    QUANTUM = "quantum"  # Use quantum computing
    CLASSICAL = "classical"  # Use classical computing only
    HYBRID = "hybrid"  # Use a mix of quantum and classical
    AUTO = "auto"  # Automatically determine the best mode
    
class PrecisionMode(Enum):
    """Specifies the floating point precision to use."""
    SINGLE = "single"  # Use single precision (float32)
    DOUBLE = "double"  # Use double precision (float64)
    MIXED = "mixed"  # Use mixed precision (compute in f32, store in f64)
    AUTO = "auto"  # Automatically determine based on hardware


# Utility decorator for timing function execution
def time_execution(func):
    """Decorator to track execution time of methods."""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, '_timing_stats'):
            self._timing_stats = {}
        
        start_time = time.time()
        result = func(self, *args, **kwargs)
        execution_time = time.time() - start_time
        
        # Update timing statistics
        func_name = func.__name__
        if func_name not in self._timing_stats:
            self._timing_stats[func_name] = {
                'count': 0,
                'total_time': 0.0,
                'min_time': float('inf'),
                'max_time': 0.0
            }
        
        stats = self._timing_stats[func_name]
        stats['count'] += 1
        stats['total_time'] += execution_time
        stats['min_time'] = min(stats['min_time'], execution_time)
        stats['max_time'] = max(stats['max_time'], execution_time)
        
        # Log if debug is enabled
        if hasattr(self, '_debug') and self._debug:
            avg_time = stats['total_time'] / stats['count']
            self._logger.debug(f"{func_name} executed in {execution_time:.6f}s "
                              f"(avg: {avg_time:.6f}s, min: {stats['min_time']:.6f}s, "
                              f"max: {stats['max_time']:.6f}s, count: {stats['count']})")
        
        return result
    return wrapper


# Numba-accelerated core PT functions
@vectorize([float64(float64, float64, float64, float64)], fastmath=True)
def _value_function_numba(x, alpha, beta, lambda_):
    """Vectorized Prospect Theory value function."""
    if x >= 0:
        return x ** alpha
    else:
        return -lambda_ * ((-x) ** beta)

@vectorize([float64(float64, float64)], fastmath=True)
def _probability_weighting_numba(p, gamma):
    """Vectorized Prospect Theory probability weighting function."""
    # Handle edge cases to prevent numerical issues
    if p <= 0.0:
        return 0.0
    elif p >= 1.0:
        return 1.0
    
    # Prelec weighting function
    try:
        w = np.exp(-(-np.log(p)) ** gamma)
        return w
    except:
        # Fallback for numerical issues
        if p < 0.01:
            return 0.01  # Minimum weight
        elif p > 0.99:
            return 0.99  # Maximum weight
        else:
            return p

# CUDA kernels for GPU acceleration
@cuda.jit
def _cuda_value_function_kernel(x_values, alpha, beta, lambda_, results):
    """CUDA kernel for PT value function."""
    i = cuda.grid(1)
    if i < x_values.shape[0]:
        x = x_values[i]
        if x >= 0:
            results[i] = x ** alpha
        else:
            results[i] = -lambda_ * ((-x) ** beta)

@cuda.jit
def _cuda_probability_weighting_kernel(probabilities, gamma, results):
    """CUDA kernel for PT probability weighting function."""
    i = cuda.grid(1)
    if i < probabilities.shape[0]:
        p = probabilities[i]
        # Handle edge cases
        if p <= 0.0:
            results[i] = 0.0
        elif p >= 1.0:
            results[i] = 1.0
        else:
            # Note: CUDA doesn't have a built-in log function for complex numbers
            # So we'll use the real log function
            try:
                log_p = math.log(p)
                neg_log_p = -log_p
                pow_result = math.pow(neg_log_p, gamma)
                neg_pow_result = -pow_result
                results[i] = math.exp(neg_pow_result)
            except:
                # Fallback
                if p < 0.01:
                    results[i] = 0.01
                elif p > 0.99:
                    results[i] = 0.99
                else:
                    results[i] = p

# Logger setup
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("qar.log"), logging.StreamHandler()],
)
logger = logging.getLogger("Quantum PT")

class QuantumProspectTheory:
    """
    High-Performance Quantum-Enhanced Prospect Theory
    
    Implements all 7 quantum PT applications with hardware acceleration
    leveraging quantum computing via PennyLane with lightning.kokkos backend
    for significant speedups over classical implementations.
    """
    
    def __init__(self, 
                 alpha: float = DEFAULT_ALPHA,
                 beta: float = DEFAULT_BETA,
                 lambda_: float = DEFAULT_LAMBDA,
                 gamma: float = DEFAULT_GAMMA,
                 delta: float = DEFAULT_DELTA,                 
                 qubits: int = 8,  # Standardized to 8 for the 8-factor model
                 layers: int = DEFAULT_LAYERS, 
                 shots: Optional[int] = DEFAULT_SHOTS,
                 precision: PrecisionMode = PrecisionMode.AUTO,
                 mode: ProcessingMode = ProcessingMode.AUTO,
                 enable_caching: bool = True,
                 cache_size: int = DEFAULT_CACHE_SIZE,
                 batch_size: int = DEFAULT_BATCH_SIZE,
                 device_name: Optional[str] = None,
                 hw_manager: Optional[Any] = None,
                 hw_accelerator: Optional[Any] = None,
                 debug: bool = False,
                 use_standard_factors: bool = True,  # Whether to use standard 8-factor model
                 factor_names: Optional[List[str]] = None,  # Custom factor names if not using standard model
                 initial_weights: Optional[Dict[str, float]] = None):  # Initial weights for factors
        """
        Initialize QuantumProspectTheory with hardware-aware configuration.
        
        Args:
            alpha: Value function exponent for gains (0 < alpha < 1)
            beta: Value function exponent for losses (0 < beta < 1)
            lambda_: Loss aversion parameter (lambda > 1)
            gamma: Probability weighting parameter (0 < gamma < 1)
            qubits: Number of qubits to use for quantum circuits
            layers: Number of variational layers for quantum circuits
            shots: Number of measurement shots (None for analytic)
            precision: Floating-point precision mode
            mode: Processing mode (quantum/classical/hybrid/auto)
            enable_caching: Whether to enable LRU caching
            cache_size: Size of the LRU cache
            batch_size: Size of batches for bulk processing
            device_name: Name of the quantum device to use
            hw_manager: Optional HardwareManager instance
            hw_accelerator: Optional HardwareAccelerator instance
            debug: Enable debug logging
        """
        # Setup logging
        self._debug = debug
        self._logger = logging.getLogger("QuantumPT")
        log_level = logging.DEBUG if debug else logging.INFO
        self._logger.setLevel(log_level)
        
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self._logger.addHandler(handler)
        
        # Store initialization parameters
        self.alpha = alpha
        self.beta = beta
        self.lambda_ = lambda_
        self.gamma = gamma
        self.delta = delta
        self.qubits = qubits
        self.layers = layers
        self.shots = shots
        self.precision_mode = precision
        self.processing_mode = mode
        self.cache_size = cache_size
        self.batch_size = batch_size
        self.debug = debug
        self.enable_caching = enable_caching
        
        # Initialize factor support
        self.use_standard_factors = use_standard_factors
        
        # Configure factor names and weights for standardized 8-factor model
        if self.use_standard_factors and qubits == 8:
            # Use the standardized 8-factor model
            self.factor_names = StandardFactors.get_ordered_list()
            
            # Use provided weights or defaults
            if initial_weights is None:
                self.factor_weights = StandardFactors.get_default_weights()
            else:
                # Merge provided weights with defaults for any missing factors
                self.factor_weights = {}
                default_weights = StandardFactors.get_default_weights()
                for factor in self.factor_names:
                    self.factor_weights[factor] = initial_weights.get(factor, default_weights.get(factor, 0.5))
        else:
            # Use custom factors or generate generic ones
            if factor_names is not None and len(factor_names) == qubits:
                self.factor_names = factor_names
            else:
                self.factor_names = [f"factor_{i}" for i in range(qubits)]
                
            # Initialize factor weights
            self.factor_weights = {}
            if initial_weights is not None:
                # Use provided weights for known factors
                for factor in self.factor_names:
                    self.factor_weights[factor] = initial_weights.get(factor, 0.5)
            else:
                # Use uniform weights
                for factor in self.factor_names:
                    self.factor_weights[factor] = 1.0 / len(self.factor_names)
        
        # Ensure qubits matches number of factors
        if qubits != len(self.factor_names) and self.use_standard_factors:
            self.logger.warning(f"Adjusting qubits from {qubits} to {len(self.factor_names)} to match number of factors")
            self.qubits = len(self.factor_names)
        
        # Determine floating point type based on precision
        self.float_type = np.float32 if self.precision_mode == PrecisionMode.SINGLE else np.float64
        
        # Initialize hardware components
        self._init_hardware(hw_manager, hw_accelerator)
        
        # Determine processing mode first
        self.processing_mode = self._determine_processing_mode(mode)
        
        # Initialize quantum device
        self.device, self.quantum_available = self._initialize_quantum_device(device_name)
        
        # Setup LRU caching if enabled
        if enable_caching:
            self._setup_caching(cache_size)
        else:
            # Create dummy cache functions
            self._value_function_cached = self.value_function
            self._probability_weighting_cached = self.probability_weighting
        
        # Thread lock for device switching
        self._device_lock = threading.RLock()
        
        # Create quantum circuits
        self._circuits = {}
        self._init_quantum_circuits()
        
        self._logger.info(f"Initialized QuantumProspectTheory - "
                         f"Mode: {self.processing_mode.value}, "
                         f"Precision: {self.precision_mode.value}, "
                         f"Quantum: {'Available' if self.quantum_available else 'Not Available'}, "
                         f"GPU: {'Available' if self.gpu_available else 'Not Available'}")

        
    
    def _init_hardware(self, hw_manager, hw_accelerator):
        """Initialize hardware management components."""
        # Initialize hardware manager if not provided
        if hw_manager is not None:
            self.hw_manager = hw_manager
        elif HARDWARE_ACCEL_AVAILABLE:
            self.hw_manager = HardwareManager.get_manager()
            self.hw_manager.initialize_hardware()
        else:
            self.hw_manager = HardwareManager()
            
        # Initialize hardware accelerator if not provided
        if hw_accelerator is not None:
            self.hw_accelerator = hw_accelerator
        elif HARDWARE_ACCEL_AVAILABLE:
            self.hw_accelerator = HardwareAccelerator(enable_gpu=True)
        else:
            self.hw_accelerator = HardwareAccelerator()
            
        # Determine hardware capabilities
        self.quantum_available = getattr(self.hw_manager, 'quantum_available', False)
        self.gpu_available = getattr(self.hw_manager, 'gpu_available', False)
        
        # For CUDA operations
        if self.gpu_available:
            try:
                cuda.detect()
                self.cuda_available = True
                self.cuda_device = cuda.get_current_device()
                self.cuda_threads_per_block = min(1024, self.cuda_device.MAX_THREADS_PER_BLOCK)
            except:
                self.cuda_available = False
                self._logger.warning("CUDA detected but failed to initialize")
        else:
            self.cuda_available = False
    
    def _determine_precision(self, precision: PrecisionMode) -> PrecisionMode:
        """Determine the appropriate precision to use based on hardware."""
        if precision != PrecisionMode.AUTO:
            return precision
        
        # Check available memory to determine if we can use double precision
        try:
            mem_info = psutil.virtual_memory()
            if mem_info.available > 8 * 1024 * 1024 * 1024:  # 8 GB
                return PrecisionMode.DOUBLE
            else:
                return PrecisionMode.SINGLE
        except:
            # Default to double precision if we can't check memory
            return PrecisionMode.DOUBLE
    
    def _determine_processing_mode(self, mode: ProcessingMode) -> ProcessingMode:
        """Determine the appropriate processing mode based on hardware."""
        if mode != ProcessingMode.AUTO:
            return mode
        
        # Use quantum if available
        if self.quantum_available:
            return ProcessingMode.HYBRID
        
        # Otherwise use classical with GPU if available
        return ProcessingMode.CLASSICAL
    
    def _initialize_quantum_device(self, device_name: Optional[str]) -> Tuple[Any, bool]:
        """Initialize the quantum device with optimal settings for the hardware."""
        # First check hardware availability
        if not self.quantum_available:
            return None, False
            
        # Check processing mode if it's already set
        if hasattr(self, 'processing_mode') and self.processing_mode == ProcessingMode.CLASSICAL:
            return None, False
        
        # Check if PennyLane is available
        try:
            import pennylane as qml
        except ImportError:
            self._logger.warning("PennyLane not available. Quantum features disabled.")
            return None, False
        
        # Try to initialize the device
        try:
            # Determine the device to use
            if device_name is None:
                # Try to use lightning.kokkos if available for best performance
                try:
                    device = qml.device("lightning.kokkos", wires=self.qubits, shots=self.shots)
                    self._logger.info("Using lightning.kokkos device")
                    return device, True
                except:
                    # Try lightning.gpu
                    try:
                        device = qml.device("lightning.gpu", wires=self.qubits, shots=self.shots)
                        self._logger.info("Using lightning.gpu device")
                        return device, True
                    except:
                        # Fallback to default
                        device = qml.device("default.qubit", wires=self.qubits, shots=self.shots)
                        self._logger.info("Using default.qubit device")
                        return device, True
            else:
                # Use the specified device
                device = qml.device(device_name, wires=self.qubits, shots=self.shots)
                self._logger.info(f"Using {device_name} device")
                return device, True
        except Exception as e:
            self._logger.warning(f"Failed to initialize quantum device: {str(e)}")
            return None, False
    
    def _setup_caching(self, cache_size: int):
        """Setup LRU caching for expensive functions."""
        # Create cached versions of the core functions
        self._value_function_cached = lru_cache(maxsize=cache_size)(self.value_function)
        self._probability_weighting_cached = lru_cache(maxsize=cache_size)(self.probability_weighting)
    
    def _init_quantum_circuits(self):
        """Initialize all quantum circuits."""
        if not self.quantum_available or self.device is None:
            return
        
        # Create circuits
        with self._device_lock:
            self._create_value_function_circuit()
            self._create_probability_weighting_circuit()
            self._create_reference_points_circuit()
            self._create_entangled_assets_circuit()
            self._create_feature_selection_circuit()
            self._create_mental_accounting_circuit()
            self._create_framing_effects_circuit()
            self._create_ambiguity_aversion_circuit()
    
    def _create_value_function_circuit(self):
        """Create quantum circuit for value function evaluation."""
        if not self.quantum_available or self.device is None:
            return
        
        @qml.qnode(self.device, interface="numpy")
        def value_function_circuit(x, alpha, beta, lambda_):
            """Quantum circuit that implements the PT value function."""
            # Encode input value into amplitude
            qml.AmplitudeEmbedding([abs(x)], [0], normalize=False)
            
            # Apply the appropriate power transformation
            if x >= 0:
                # For gains: Apply rotation based on alpha
                qml.RY(np.arcsin(alpha) * np.pi, wires=0)
                
                # Apply appropriate transformation to represent power function
                for _ in range(self.layers):
                    qml.RY(np.pi/4, wires=0)
                    qml.RZ(np.pi/8, wires=0)
            else:
                # For losses: Apply rotation based on beta and lambda
                qml.RY(np.arcsin(beta) * np.pi, wires=0)
                qml.RZ(np.arctan(lambda_) * np.pi, wires=0)
                
                # Apply appropriate transformation to represent power function
                for _ in range(self.layers):
                    qml.RY(np.pi/4, wires=0)
                    qml.RZ(np.pi/8, wires=0)
            
            # Return the expectation value
            return qml.expval(qml.PauliZ(0))
        
        self._circuits['value_function'] = value_function_circuit
    
    def _create_probability_weighting_circuit(self):
        """Create quantum circuit for probability weighting."""
        if not self.quantum_available or self.device is None:
            return
        
        @qml.qnode(self.device, interface="numpy")
        def probability_weighting_circuit(p, gamma):
            """Quantum circuit that implements PT probability weighting."""
            # Encode probability into amplitude
            qml.AmplitudeEmbedding([p], [0], normalize=False)
            
            # Apply transformation based on gamma parameter
            qml.RY(np.arcsin(gamma) * np.pi, wires=0)
            
            # Apply nonlinear transformation to simulate probability weighting function
            for _ in range(self.layers):
                qml.RZ(np.pi/4, wires=0)
                qml.RY(np.pi/2, wires=0)
            
            # Return the weighted probability
            return qml.expval(qml.PauliZ(0))
        
        self._circuits['probability_weighting'] = probability_weighting_circuit
    
    def _create_reference_points_circuit(self):
        """Create quantum circuit for superposition of reference points."""
        if not self.quantum_available or self.device is None:
            return
        
        @qml.qnode(self.device, interface="numpy")
        def reference_points_circuit(price, ref_points, weights):
            """
            Quantum circuit for evaluating multiple reference points in superposition.
            
            Args:
                price: Current price
                ref_points: List of reference points
                weights: List of weights for each reference point
            """
            n_points = len(ref_points)
            n_qubits_needed = max(1, math.ceil(math.log2(n_points)))
            
            # Prepare superposition of reference points
            qml.MottonenStatePreparation(np.sqrt(weights), wires=range(n_qubits_needed))
            
            # Embed the current price in an ancilla qubit
            qml.AmplitudeEmbedding([price], [n_qubits_needed], normalize=False)
            
            # Apply controlled operations to evaluate against each reference point
            for i in range(n_points):
                # Calculate binary representation of i
                binary = format(i, f'0{n_qubits_needed}b')
                
                # Apply controlled operations based on the binary representation
                control_wires = []
                for j, bit in enumerate(binary):
                    if bit == '1':
                        control_wires.append(j)
                    else:
                        # Apply X gate for 0 bit to use as control
                        qml.PauliX(wires=j)
                        control_wires.append(j)
                
                # Calculate relative value
                relative_val = price - ref_points[i]
                sign = 1 if relative_val >= 0 else -1
                
                # Apply controlled rotation based on value function
                with qml.ctrl(control_wires):
                    if sign > 0:
                        # Gain case
                        qml.RY(np.arcsin(abs(relative_val)**self.alpha) * sign, wires=n_qubits_needed)
                    else:
                        # Loss case
                        qml.RY(np.arcsin(self.lambda_ * abs(relative_val)**self.beta) * sign, wires=n_qubits_needed)
                
                # Restore qubits by applying X gates again where needed
                for j, bit in enumerate(binary):
                    if bit == '0':
                        qml.PauliX(wires=j)
            
            # Measure the ancilla qubit
            return qml.expval(qml.PauliZ(n_qubits_needed))
        
        self._circuits['reference_points'] = reference_points_circuit
    
    def _create_entangled_assets_circuit(self):
        """Create quantum circuit for entangled multi-asset evaluation."""
        if not self.quantum_available or self.device is None:
            return
        
        @qml.qnode(self.device, interface="numpy")
        def entangled_assets_circuit(prices, ref_prices, correlations, weights):
            """
            Quantum circuit for evaluating multiple correlated assets.
            
            Args:
                prices: Current prices for each asset
                ref_prices: Reference prices for each asset
                correlations: Correlation matrix between assets
                weights: Portfolio weights for each asset
            """
            n_assets = len(prices)
            
            # Initial state preparation based on portfolio weights
            qml.MottonenStatePreparation(np.sqrt(weights), wires=range(n_assets))
            
            # Apply entangling operations based on correlations
            for i in range(n_assets):
                for j in range(i+1, n_assets):
                    if abs(correlations[i][j]) > 0.05:  # Only entangle if correlation is significant
                        # Use correlation strength to determine entanglement
                        qml.CRZ(correlations[i][j] * np.pi, wires=[i, j])
            
            # Calculate PT values for each asset
            for i in range(n_assets):
                # Calculate relative performance
                relative_val = (prices[i] - ref_prices[i]) / ref_prices[i]  # Percentage change
                sign = 1 if relative_val >= 0 else -1
                
                # Apply rotation based on PT value function
                if sign > 0:
                    # Gain case
                    qml.RY(np.arcsin(abs(relative_val)**self.alpha) * sign, wires=i)
                else:
                    # Loss case - with loss aversion
                    qml.RY(np.arcsin(self.lambda_ * abs(relative_val)**self.beta) * sign, wires=i)
            
            # Measure all qubits
            return [qml.expval(qml.PauliZ(i)) for i in range(n_assets)]
        
        self._circuits['entangled_assets'] = entangled_assets_circuit
    
    def _create_feature_selection_circuit(self):
        """Create quantum circuit for PT-based feature selection."""
        if not self.quantum_available or self.device is None:
            return
        
        @qml.qnode(self.device, interface="numpy")
        def feature_selection_circuit(features, importance_scores, risk_measures):
            """
            Quantum circuit for feature selection with PT risk weighting.
            
            Args:
                features: Feature values (normalized)
                importance_scores: Base importance scores for each feature
                risk_measures: Risk measure for each feature (positive = gain indicator, negative = loss indicator)
            """
            n_features = len(features)
            
            # Prepare initial state based on feature importance
            qml.MottonenStatePreparation(np.sqrt(importance_scores), wires=range(n_features))
            
            # Apply PT-based transformations to each feature
            for i in range(n_features):
                # For loss indicators, increase importance using loss aversion
                if risk_measures[i] < 0:
                    qml.RY(np.arcsin(self.lambda_ * abs(risk_measures[i])), wires=i)
                else:
                    qml.RY(np.arcsin(risk_measures[i]), wires=i)
                
                # Apply feature value as amplitude
                qml.RZ(features[i] * np.pi, wires=i)
            
            # Apply mixing to allow for interference
            qml.broadcast(qml.Hadamard, wires=range(n_features), pattern="single")
            
            # Measure all qubits to get modified importance
            return [qml.expval(qml.PauliZ(i)) for i in range(n_features)]
        
        self._circuits['feature_selection'] = feature_selection_circuit
    
    def _create_mental_accounting_circuit(self):
        """Create quantum circuit for quantum mental accounting."""
        if not self.quantum_available or self.device is None:
            return
        
        @qml.qnode(self.device, interface="numpy")
        def mental_accounting_circuit(values, account_types, account_weights):
            """
            Quantum circuit for mental accounting, treating different accounts distinctly.
            
            Args:
                values: Values for each account
                account_types: Type of each account (0=gain-oriented, 1=loss-oriented)
                account_weights: Importance weights for each account
            """
            n_accounts = len(values)
            q_register_size = math.ceil(math.log2(n_accounts))
            
            # Setup account superposition based on weights
            qml.MottonenStatePreparation(np.sqrt(account_weights), wires=range(q_register_size))
            
            # Ancilla qubit for value function
            value_qubit = q_register_size
            
            # For each account, apply PT value function with controlled operations
            for i in range(n_accounts):
                # Binary representation of account index
                binary = format(i, f'0{q_register_size}b')
                control_wires = []
                
                # Setup control qubits
                for j, bit in enumerate(binary):
                    if bit == '0':
                        qml.PauliX(wires=j)
                    control_wires.append(j)
                
                # Apply controlled operation for this account
                with qml.ctrl(control_wires):
                    x = values[i]
                    if x >= 0:
                        # Gain case - account type affects alpha
                        alpha_mod = self.alpha * (1.1 if account_types[i] == 0 else 0.9)
                        qml.RY(np.arcsin(abs(x)**alpha_mod), wires=value_qubit)
                    else:
                        # Loss case - account type affects beta and lambda
                        beta_mod = self.beta * (0.9 if account_types[i] == 1 else 1.1)
                        lambda_mod = self.lambda_ * (0.9 if account_types[i] == 1 else 1.1)
                        qml.RY(-np.arcsin(lambda_mod * abs(x)**beta_mod), wires=value_qubit)
                
                # Reset control qubits
                for j, bit in enumerate(binary):
                    if bit == '0':
                        qml.PauliX(wires=j)
            
            # Measure value qubit
            return qml.expval(qml.PauliZ(value_qubit))
        
        self._circuits['mental_accounting'] = mental_accounting_circuit
    
    def _create_framing_effects_circuit(self):
        """Create quantum circuit for quantum framing effects."""
        if not self.quantum_available or self.device is None:
            return
        
        @qml.qnode(self.device, interface="numpy")
        def framing_effects_circuit(value, is_loss_frame):
            """
            Quantum circuit modeling framing effects in PT.
            
            Args:
                value: The objective value 
                is_loss_frame: Boolean, True if presented as loss frame
            """
            # Use 2 qubits: one for value, one for frame
            
            # Prepare value qubit
            qml.RY(np.arcsin(abs(value)), wires=0)
            
            # Prepare frame qubit: |0⟩ for gain frame, |1⟩ for loss frame
            if is_loss_frame:
                qml.PauliX(wires=1)
                
            # Entangle value with frame
            qml.CNOT(wires=[1, 0])
            
            # Apply frame-dependent PT parameters
            with qml.ctrl(1):
                # Loss frame: Amplify loss aversion
                qml.RZ(np.arcsin(self.lambda_ * 1.2), wires=0)
            
            with qml.ctrl(1, ctrl_values=0):
                # Gain frame: Standard PT parameters
                qml.RZ(np.arcsin(self.lambda_), wires=0)
            
            # Measure value qubit
            return qml.expval(qml.PauliZ(0))
        
        self._circuits['framing_effects'] = framing_effects_circuit
    
    def _create_ambiguity_aversion_circuit(self):
        """Create quantum circuit for ambiguity aversion via interference."""
        if not self.quantum_available or self.device is None:
            return
        
        @qml.qnode(self.device, interface="numpy")
        def ambiguity_aversion_circuit(known_prob, ambiguity_level, value):
            """
            Quantum circuit for modeling ambiguity aversion through interference.
            
            Args:
                known_prob: Known probability component
                ambiguity_level: Level of ambiguity (0-1)
                value: Value of the outcome
            """
            # Encode known probability in qubit 0
            qml.RY(2 * np.arcsin(np.sqrt(known_prob)), wires=0)
            
            # Encode ambiguity level in qubit 1
            qml.RY(2 * np.arcsin(np.sqrt(ambiguity_level)), wires=1)
            
            # Create interference with Hadamard
            qml.Hadamard(wires=1)
            
            # Controlled value function application
            if value >= 0:
                # Gain case
                with qml.ctrl(0):
                    qml.RY(np.arcsin(value**self.alpha), wires=2)
            else:
                # Loss case
                with qml.ctrl(0):
                    qml.RY(-np.arcsin(self.lambda_ * abs(value)**self.beta), wires=2)
            
            # Ambiguity aversion: amplify negative outcomes under ambiguity
            with qml.ctrl([0, 1]):
                if value < 0:
                    # Additional aversion to ambiguous losses
                    qml.RZ(1.5 * np.pi, wires=2)
            
            # Measure result
            return qml.expval(qml.PauliZ(2))
        
        self._circuits['ambiguity_aversion'] = ambiguity_aversion_circuit
    
    @time_execution
    def value_function(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate the PT value function with hardware acceleration.
        
        Args:
            x: Input values (gains/losses relative to reference point)
            
        Returns:
            PT subjective values
        """
        # Convert input to appropriate numpy array
        scalar_input = np.isscalar(x)
        x_array = np.asarray(x, dtype=self.float_type)
        if scalar_input:
            x_array = np.array([x_array], dtype=self.float_type)
        
        # Select appropriate implementation based on processing mode and input size
        if self.processing_mode == ProcessingMode.QUANTUM and self.quantum_available:
            # For small inputs, use quantum implementation
            if len(x_array) == 1:
                try:
                    with self._device_lock:
                        circuit = self._circuits.get('value_function')
                        if circuit is not None:
                            result = circuit(x_array[0], self.alpha, self.beta, self.lambda_)
                            # Scale the result from [-1,1] to actual value
                            if x_array[0] >= 0:
                                result = (result + 1) / 2 * x_array[0]**self.alpha
                            else:
                                result = (result + 1) / 2 * (-self.lambda_ * ((-x_array[0])**self.beta))
                            return result if not scalar_input else result[0]
                except Exception as e:
                    self._logger.warning(f"Quantum value_function failed: {str(e)}. Falling back to classical.")
        
        # For larger inputs or if quantum failed, try GPU
        if self.gpu_available and self.cuda_available and len(x_array) > 100:
            try:
                # Allocate memory on device
                d_x = cuda.to_device(x_array)
                d_result = cuda.device_array_like(d_x)
                
                # Calculate grid dimensions
                threads_per_block = self.cuda_threads_per_block
                blocks_per_grid = (len(x_array) + threads_per_block - 1) // threads_per_block
                
                # Launch kernel
                _cuda_value_function_kernel[blocks_per_grid, threads_per_block](
                    d_x, self.alpha, self.beta, self.lambda_, d_result
                )
                
                # Copy result back
                result = d_result.copy_to_host()
                return result if not scalar_input else result[0]
            except Exception as e:
                self._logger.warning(f"CUDA value_function failed: {str(e)}. Falling back to Numba.")
        
        # CPU implementation with Numba
        result = _value_function_numba(x_array, self.alpha, self.beta, self.lambda_)
        return result if not scalar_input else result[0]
    
    @time_execution
    def probability_weighting(self, p: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate probability weighting function with hardware acceleration.
        
        Args:
            p: Input probabilities
            
        Returns:
            Weighted probabilities
        """
        # Convert input to appropriate numpy array
        scalar_input = np.isscalar(p)
        p_array = np.asarray(p, dtype=self.float_type)
        if scalar_input:
            p_array = np.array([p_array], dtype=self.float_type)
        
        # Ensure probabilities are in [0,1]
        p_array = np.clip(p_array, 0.0, 1.0)
        
        # Select appropriate implementation based on processing mode and input size
        if self.processing_mode == ProcessingMode.QUANTUM and self.quantum_available:
            # For small inputs, use quantum implementation
            if len(p_array) == 1:
                try:
                    with self._device_lock:
                        circuit = self._circuits.get('probability_weighting')
                        if circuit is not None:
                            result = circuit(p_array[0], self.gamma)
                            # Scale the result from [-1,1] to [0,1]
                            result = (result + 1) / 2
                            return result if not scalar_input else result[0]
                except Exception as e:
                    self._logger.warning(f"Quantum probability_weighting failed: {str(e)}. Falling back to classical.")
        
        # For larger inputs or if quantum failed, try GPU
        if self.gpu_available and self.cuda_available and len(p_array) > 100:
            try:
                # Allocate memory on device
                d_p = cuda.to_device(p_array)
                d_result = cuda.device_array_like(d_p)
                
                # Calculate grid dimensions
                threads_per_block = self.cuda_threads_per_block
                blocks_per_grid = (len(p_array) + threads_per_block - 1) // threads_per_block
                
                # Launch kernel
                _cuda_probability_weighting_kernel[blocks_per_grid, threads_per_block](
                    d_p, self.gamma, d_result
                )
                
                # Copy result back
                result = d_result.copy_to_host()
                return result if not scalar_input else result[0]
            except Exception as e:
                self._logger.warning(f"CUDA probability_weighting failed: {str(e)}. Falling back to Numba.")
        
        # CPU implementation with Numba
        result = _probability_weighting_numba(p_array, self.gamma)
        return result if not scalar_input else result[0]
    
    @time_execution
    def evaluate_superposed_references(self, 
                                      current_price: float, 
                                      reference_points: List[float],
                                      weights: Optional[List[float]] = None) -> float:
        """
        Evaluate a price against multiple reference points in quantum superposition.
        
        Args:
            current_price: Current price to evaluate
            reference_points: List of reference points
            weights: Optional weights for each reference point (defaults to equal)
            
        Returns:
            PT value considering all reference points simultaneously
        """
        if len(reference_points) == 0:
            return 0.0
        
        # Normalize weights if provided, otherwise use equal weights
        if weights is None:
            weights = [1.0 / len(reference_points)] * len(reference_points)
        else:
            # Ensure weights match reference points
            if len(weights) != len(reference_points):
                raise ValueError(f"Number of weights ({len(weights)}) must match reference points ({len(reference_points)})")
            # Normalize
            total = sum(weights)
            weights = [w / total for w in weights]
        
        # Try quantum implementation if available
        if self.processing_mode == ProcessingMode.QUANTUM and self.quantum_available:
            try:
                with self._device_lock:
                    circuit = self._circuits.get('reference_points')
                    if circuit is not None:
                        return circuit(current_price, reference_points, weights)
            except Exception as e:
                self._logger.warning(f"Quantum reference points failed: {str(e)}. Falling back to classical.")
        
        # Classical implementation (weighted average of individual PT values)
        values = []
        for i, ref_point in enumerate(reference_points):
            # Calculate relative value
            rel_value = current_price - ref_point
            # Apply PT value function
            pt_value = self.value_function(rel_value)
            # Apply weight
            values.append(weights[i] * pt_value)
        
        # Return weighted sum
        return sum(values)
    
    @time_execution
    def evaluate_entangled_portfolio(self,
                                    current_prices: List[float],
                                    reference_prices: List[float],
                                    correlations: List[List[float]],
                                    weights: Optional[List[float]] = None) -> float:
        """
        Evaluate a portfolio with entangled assets using PT.
        
        Args:
            current_prices: Current prices for each asset
            reference_prices: Reference prices for each asset
            correlations: Correlation matrix between assets
            weights: Optional portfolio weights (defaults to equal)
            
        Returns:
            PT value of the entangled portfolio
        """
        n_assets = len(current_prices)
        
        if len(reference_prices) != n_assets:
            raise ValueError(f"Number of reference prices ({len(reference_prices)}) must match current prices ({n_assets})")
        
        if len(correlations) != n_assets or any(len(row) != n_assets for row in correlations):
            raise ValueError("Correlation matrix must be square and match the number of assets")
        
        # Normalize weights if provided, otherwise use equal weights
        if weights is None:
            weights = [1.0 / n_assets] * n_assets
        else:
            # Ensure weights match assets
            if len(weights) != n_assets:
                raise ValueError(f"Number of weights ({len(weights)}) must match assets ({n_assets})")
            # Normalize
            total = sum(weights)
            weights = [w / total for w in weights]
        
        # Try quantum implementation if available
        if self.processing_mode == ProcessingMode.QUANTUM and self.quantum_available and n_assets <= self.qubits:
            try:
                with self._device_lock:
                    circuit = self._circuits.get('entangled_assets')
                    if circuit is not None:
                        results = circuit(current_prices, reference_prices, correlations, weights)
                        # Weight the results
                        weighted_result = sum(w * r for w, r in zip(weights, results))
                        return weighted_result
            except Exception as e:
                self._logger.warning(f"Quantum entangled portfolio failed: {str(e)}. Falling back to classical.")
        
        # Classical implementation (correlation-weighted PT values)
        pt_values = []
        
        for i in range(n_assets):
            # Calculate relative value
            rel_value = (current_prices[i] - reference_prices[i]) / reference_prices[i]
            
            # Apply PT value function
            base_pt_value = self.value_function(rel_value)
            
            # Apply correlation-based adjustment
            corr_factor = 1.0
            for j in range(n_assets):
                if i != j:
                    # Increase value if correlated asset is also performing well
                    other_rel_value = (current_prices[j] - reference_prices[j]) / reference_prices[j]
                    other_sign = 1 if other_rel_value >= 0 else -1
                    this_sign = 1 if rel_value >= 0 else -1
                    
                    # If correlation is positive and both assets have the same sign, amplify
                    # If correlation is negative and assets have opposite signs, amplify
                    if (correlations[i][j] > 0 and this_sign == other_sign) or \
                       (correlations[i][j] < 0 and this_sign != other_sign):
                        corr_factor += abs(correlations[i][j]) * 0.1
                    else:
                        corr_factor -= abs(correlations[i][j]) * 0.1
            
            # Apply correlation adjustment
            pt_values.append(base_pt_value * corr_factor)
        
        # Return weighted sum
        return sum(weights[i] * pt_values[i] for i in range(n_assets))
    
    @time_execution
    def feature_selection(self,
                         features: List[float],
                         base_importance: List[float],
                         risk_indicators: List[float]) -> List[float]:
        """
        Perform feature selection with PT-based risk weighting.
        
        Args:
            features: Normalized feature values
            base_importance: Base importance of each feature
            risk_indicators: Risk indicator for each feature (+1 = gain, -1 = loss)
            
        Returns:
            PT-weighted feature importance scores
        """
        n_features = len(features)
        
        if len(base_importance) != n_features:
            raise ValueError(f"Number of importance scores ({len(base_importance)}) must match features ({n_features})")
        
        if len(risk_indicators) != n_features:
            raise ValueError(f"Number of risk indicators ({len(risk_indicators)}) must match features ({n_features})")
        
        # Normalize base importance
        total_importance = sum(base_importance)
        normalized_importance = [imp / total_importance for imp in base_importance]
        
        # Try quantum implementation if available
        if self.processing_mode == ProcessingMode.QUANTUM and self.quantum_available and n_features <= self.qubits:
            try:
                with self._device_lock:
                    circuit = self._circuits.get('feature_selection')
                    if circuit is not None:
                        results = circuit(features, normalized_importance, risk_indicators)
                        # Convert from [-1,1] to [0,1] range
                        results = [(r + 1) / 2 for r in results]
                        # Normalize results
                        total = sum(results)
                        if total > 0:
                            return [r / total for r in results]
                        return normalized_importance
            except Exception as e:
                self._logger.warning(f"Quantum feature selection failed: {str(e)}. Falling back to classical.")
        
        # Classical implementation
        pt_importance = []
        
        for i in range(n_features):
            # Base importance
            base_imp = normalized_importance[i]
            
            # Apply PT-based adjustment based on risk indicator
            if risk_indicators[i] < 0:
                # Loss indicator - amplify importance using loss aversion
                pt_factor = self.lambda_ * abs(risk_indicators[i])
            else:
                # Gain indicator - standard weighting
                pt_factor = risk_indicators[i]
            
            # Feature value contribution
            value_contribution = abs(features[i])
            
            # Combined importance
            pt_importance.append(base_imp * (1 + pt_factor * value_contribution))
        
        # Normalize results
        total = sum(pt_importance)
        return [imp / total for imp in pt_importance]
    
    @time_execution
    def evaluate_mental_accounting(self,
                                  account_values: List[float],
                                  account_types: List[int],
                                  account_weights: Optional[List[float]] = None) -> float:
        """
        Evaluate values with mental accounting effects.
        
        Args:
            account_values: Values for each mental account
            account_types: Type of each account (0=gain-oriented, 1=loss-oriented)
            account_weights: Optional importance weights for each account
            
        Returns:
            PT value with mental accounting effects
        """
        n_accounts = len(account_values)
        
        if len(account_types) != n_accounts:
            raise ValueError(f"Number of account types ({len(account_types)}) must match values ({n_accounts})")
        
        # Normalize weights if provided, otherwise use equal weights
        if account_weights is None:
            account_weights = [1.0 / n_accounts] * n_accounts
        else:
            # Ensure weights match accounts
            if len(account_weights) != n_accounts:
                raise ValueError(f"Number of weights ({len(account_weights)}) must match accounts ({n_accounts})")
            # Normalize
            total = sum(account_weights)
            account_weights = [w / total for w in account_weights]
        
        # Try quantum implementation if available
        if self.processing_mode == ProcessingMode.QUANTUM and self.quantum_available and n_accounts <= 2**(self.qubits-1):
            try:
                with self._device_lock:
                    circuit = self._circuits.get('mental_accounting')
                    if circuit is not None:
                        result = circuit(account_values, account_types, account_weights)
                        return result
            except Exception as e:
                self._logger.warning(f"Quantum mental accounting failed: {str(e)}. Falling back to classical.")
        
        # Classical implementation
        mental_values = []
        
        for i in range(n_accounts):
            x = account_values[i]
            
            # Apply account-specific PT parameters
            if x >= 0:
                # Gain case - account type affects alpha
                alpha_mod = self.alpha * (1.1 if account_types[i] == 0 else 0.9)
                value = x ** alpha_mod
            else:
                # Loss case - account type affects beta and lambda
                beta_mod = self.beta * (0.9 if account_types[i] == 1 else 1.1)
                lambda_mod = self.lambda_ * (0.9 if account_types[i] == 1 else 1.1)
                value = -lambda_mod * ((-x) ** beta_mod)
            
            mental_values.append(account_weights[i] * value)
        
        # Return weighted sum
        return sum(mental_values)
    
    @time_execution
    def evaluate_framing_effects(self, 
                               value: float, 
                               gain_frame: bool = True) -> float:
        """
        Evaluate a value with framing effects.
        
        Args:
            value: The objective value
            gain_frame: True if presented as a gain frame, False for loss frame
            
        Returns:
            PT value with framing effects
        """
        # Try quantum implementation if available
        if self.processing_mode == ProcessingMode.QUANTUM and self.quantum_available:
            try:
                with self._device_lock:
                    circuit = self._circuits.get('framing_effects')
                    if circuit is not None:
                        result = circuit(value, not gain_frame)  # Invert for is_loss_frame parameter
                        return result
            except Exception as e:
                self._logger.warning(f"Quantum framing effects failed: {str(e)}. Falling back to classical.")
        
        # Classical implementation
        # Define a scaling factor to handle extreme values - use log scaling for better handling
        raw_value = value
        
        # Log scaling for extreme values to prevent explosive growth and maintain sensitivity
        if abs(value) > 100.0:
            # Log scaling with sign preservation
            scaled_value = math.copysign(math.log(abs(value) + 1.0), value)
        else:
            scaled_value = value
        
        if gain_frame:
            # Gain frame: Standard PT parameters with smoothed response
            if scaled_value >= 0:
                base_effect = scaled_value ** self.alpha
            else:
                base_effect = -self.lambda_ * ((-scaled_value) ** self.beta)
        else:
            # Loss frame: Dynamic lambda based on value magnitude
            # Higher loss values increase risk aversion more pronouncedly
            base_lambda = self.lambda_
            
            # More aggressive scaling for loss frame to differentiate it from gain frame
            if abs(raw_value) > 5000.0:
                framing_lambda = base_lambda * (1.5 + 0.5 * math.tanh(abs(raw_value)/10000.0))
            else:
                # Smoother transition for moderate values
                framing_lambda = base_lambda * (1.2 + 0.3 * abs(scaled_value)/10.0)
            
            if scaled_value >= 0:
                base_effect = scaled_value ** self.alpha
            else:
                base_effect = -framing_lambda * ((-scaled_value) ** self.beta)
        
        # Apply progressive capping using tanh for smoother boundary behavior
        MAX_EFFECT_MAGNITUDE = 30.0  # Increased to allow more differentiation
        return MAX_EFFECT_MAGNITUDE * math.tanh(base_effect / MAX_EFFECT_MAGNITUDE)
    
    @time_execution
    def evaluate_ambiguity(self,
                          known_prob: float,
                          ambiguity_level: float,
                          value: float) -> float:
        """
        Evaluate a prospect with ambiguity aversion.
        
        Args:
            known_prob: Known probability component
            ambiguity_level: Level of ambiguity (0-1)
            value: Value of the outcome
            
        Returns:
            PT value with ambiguity aversion effects
        """
        # Check parameter ranges
        known_prob = np.clip(known_prob, 0.0, 1.0)
        ambiguity_level = np.clip(ambiguity_level, 0.0, 1.0)
        
        # Try quantum implementation if available
        if self.processing_mode == ProcessingMode.QUANTUM and self.quantum_available:
            try:
                with self._device_lock:
                    circuit = self._circuits.get('ambiguity_aversion')
                    if circuit is not None:
                        result = circuit(known_prob, ambiguity_level, value)
                        return result
            except Exception as e:
                self._logger.warning(f"Quantum ambiguity aversion failed: {str(e)}. Falling back to classical.")
        
        # Classical implementation
        
        # Calculate base PT value
        if value >= 0:
            base_value = value ** self.alpha
        else:
            base_value = -self.lambda_ * ((-value) ** self.beta)
        
        # Calculate weighted probability
        weighted_prob = self.probability_weighting(known_prob)
        
        # Apply ambiguity aversion
        if value < 0:
            # For losses, ambiguity increases the effect (pessimism)
            ambiguity_factor = 1 + ambiguity_level * 0.5
            return weighted_prob * base_value * ambiguity_factor
        else:
            # For gains, ambiguity decreases the effect (pessimism)
            ambiguity_factor = 1 - ambiguity_level * 0.3
            return weighted_prob * base_value * ambiguity_factor


    def save_state(self, filepath: str) -> bool:
        """
        Saves the current state of the QuantumProspectTheory instance to a file.

        Args:
            filepath: The path to the file where the state will be saved.

        Returns:
            True if saving was successful, False otherwise.
        """
        current_logger = getattr(self, '_logger', logger) # Use instance logger or fallback
        try:
            # Using self._device_lock assuming it exists from your full QPT code for consistency
            # If not, remove the lock or use a local threading.Lock() for this method.
            with getattr(self, '_device_lock', threading.RLock()): # Fallback to a new lock if not present
                state_to_save = {
                    'alpha': self.alpha,
                    'beta': self.beta,
                    'lambda_': self.lambda_, # Underscore to match attribute
                    'gamma': self.gamma,
                    # Quantum configuration that was set
                    'qubits': self.qubits,
                    'layers': self.layers,
                    'shots': self.shots,
                    # Save Enum values
                    'precision_mode': self.precision_mode.value if hasattr(self.precision_mode, 'value') else str(self.precision_mode),
                    'processing_mode': self.processing_mode.value if hasattr(self.processing_mode, 'value') else str(self.processing_mode),
                    'batch_size': self.batch_size,
                    'enable_caching': hasattr(self, '_value_function_cached'), # Check if caching was set up
                    'cache_size': getattr(self, '_cache_size_used_for_setup', DEFAULT_CACHE_SIZE), # If you store it
                    # Add any other adaptable parameters specific to QPT here
                    # e.g., if reference point adaptation logic had internal state in QPT:
                    # 'internal_reference_model_state': self.internal_reference_model.get_state() if hasattr...
                    'saved_timestamp': datetime.now().isoformat()
                }

                with open(filepath, 'w') as f:
                    json.dump(state_to_save, f, indent=2)
                
                current_logger.info(f"QuantumProspectTheory state saved to {filepath}")
                return True
        except Exception as e:
            current_logger.error(f"Error saving QuantumProspectTheory state to {filepath}: {e}", exc_info=True)
            return False

    def load_state(self, filepath: str) -> bool:
        """
        Loads the state of the QuantumProspectTheory instance from a file.

        Args:
            filepath: The path to the file from which the state will be loaded.

        Returns:
            True if loading was successful, False otherwise.
        """
        current_logger = getattr(self, '_logger', logger)
        try:
            if not os.path.exists(filepath):
                current_logger.error(f"QuantumProspectTheory state file not found: {filepath}")
                return False

            with getattr(self, '_device_lock', threading.RLock()):
                with open(filepath, 'r') as f:
                    loaded_state = json.load(f)

                # Restore PT parameters
                self.alpha = loaded_state.get('alpha', self.alpha)
                self.beta = loaded_state.get('beta', self.beta)
                self.lambda_ = loaded_state.get('lambda_', self.lambda_)
                self.gamma = loaded_state.get('gamma', self.gamma)

                # Restore quantum configuration
                loaded_qubits = loaded_state.get('qubits', self.qubits)
                if self.qubits != loaded_qubits:
                     current_logger.warning(f"Loading QPT state with different qubit count ({loaded_qubits}) than current ({self.qubits}).")
                self.qubits = loaded_qubits
                self.layers = loaded_state.get('layers', self.layers)
                self.shots = loaded_state.get('shots', self.shots)
                
                self.precision_mode = PrecisionMode(loaded_state.get('precision_mode', self.precision_mode.value if hasattr(self.precision_mode, 'value') else str(self.precision_mode)))
                # Determine processing mode - might need to re-evaluate based on current hardware vs saved mode
                saved_processing_mode_val = loaded_state.get('processing_mode', self.processing_mode.value if hasattr(self.processing_mode, 'value') else str(self.processing_mode))
                self.selected_mode = ProcessingMode(saved_processing_mode_val) # Store what was saved
                # self.processing_mode = self._determine_processing_mode(self.selected_mode) # Re-evaluate based on current hardware

                self.batch_size = loaded_state.get('batch_size', self.batch_size)
                
                # Caching setup (re-apply based on loaded setting)
                enable_caching_loaded = loaded_state.get('enable_caching', True) # Default to True if not saved
                cache_size_loaded = loaded_state.get('cache_size', DEFAULT_CACHE_SIZE)
                if enable_caching_loaded:
                    self._setup_caching(cache_size_loaded)
                else: # Ensure caching is off if loaded state says so
                    self._value_function_cached = self.value_function
                    self._probability_weighting_cached = self.probability_weighting


                # Re-initialize components that depend on loaded parameters
                # For QPT, this might include re-initializing the quantum device and circuits
                # if critical parameters like 'qubits' changed.
                # self.device, self.quantum_available = self._initialize_quantum_device(None) # Or a saved device_name
                if self.quantum_available and self.processing_mode != ProcessingMode.CLASSICAL: # Use current processing_mode
                    self._init_quantum_circuits() # Re-create circuits based on potentially new params

                current_logger.info(f"QuantumProspectTheory state loaded from {filepath} (saved at {loaded_state.get('saved_timestamp', 'unknown')})")
                return True
        except Exception as e:
            current_logger.error(f"Error loading QuantumProspectTheory state from {filepath}: {e}", exc_info=True)
            return False
        
    
    def get_timing_stats(self) -> Dict[str, Dict[str, float]]:
        """Get execution timing statistics."""
        if hasattr(self, '_timing_stats'):
            # Add average time
            for func_name, stats in self._timing_stats.items():
                if stats['count'] > 0:
                    stats['avg_time'] = stats['total_time'] / stats['count']
            return self._timing_stats
        return {}
    
    def __str__(self) -> str:
        """String representation."""
        mode_str = f"Mode: {self.processing_mode.value}"
        precision_str = f"Precision: {self.precision_mode.value}"
        hw_str = f"Hardware: Quantum={'Available' if self.quantum_available else 'Not Available'}, " \
                f"GPU={'Available' if self.gpu_available else 'Not Available'}"
        params_str = f"PT Parameters: α={self.alpha:.2f}, β={self.beta:.2f}, λ={self.lambda_:.2f}, γ={self.gamma:.2f}"
        return f"QuantumProspectTheory({mode_str}, {precision_str}, {hw_str}, {params_str})"
