#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 12 01:23:48 2025

@author: ashina
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import logging
import time
import threading
from typing import List, Dict, Tuple, Optional, Callable, Any, Union
from enum import Enum, auto
from functools import lru_cache, wraps
import warnings
import random
import math
from numba import njit, prange
import json

from adaptive_market_data_fetcher import AdaptiveMarketDataFetcher

try:
    import pennylane as qml
    import pennylane.numpy as qnp
    QUANTUM_AVAILABLE = True
except ImportError:
    QUANTUM_AVAILABLE = False
    logging.warning("PennyLane not installed; quantum features will be disabled")

# Optional imports for hardware acceleration
try:
    from hardware_manager import HardwareManager
    from cdfa_extensions.hw_acceleration import HardwareAccelerator, AcceleratorType
    HARDWARE_ACCEL_AVAILABLE = True
except ImportError:
    HARDWARE_ACCEL_AVAILABLE = False
    warnings.warn("Hardware acceleration modules not available. Using fallback implementation.")

    # Define dummy classes if the imports fail
    class HardwareManager:
        @classmethod
        def get_manager(cls, **kwargs):
            return cls(**kwargs)

        def __init__(self, **kwargs):
            self.quantum_available = False
            self.gpu_available = False

        def initialize_hardware(self):
            return False

        def _get_quantum_device(self, qubits):
            return {"device": "lightning.kokkos", "wires": qubits, "shots": None}

        def get_optimal_device(self, **kwargs):
            return {"type": "gpu", "available": True}

    class HardwareAccelerator:
        def __init__(self, **kwargs):
            self.gpu_available = False

        def get_accelerator_type(self):
            return "cpu"

        def get_torch_device(self):
            return None

    class AcceleratorType(Enum):
        CPU = auto()
        CUDA = auto()
        ROCM = auto()
        MPS = auto()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("QuantumLMSR")

# Enums for configuration
class ProcessingMode(Enum):
    """Modes for quantum circuit execution."""
    QUANTUM = "quantum"  # Use quantum circuits when available
    CLASSICAL = "classical"  # Always use classical methods
    HYBRID = "hybrid"  # Use quantum for some operations, classical for others
    AUTO = "auto"  # Automatically select the best method

class PrecisionMode(Enum):
    """Numeric precision modes."""
    SINGLE = "single"  # Use single precision (float32/complex64)
    DOUBLE = "double"  # Use double precision (float64/complex128)
    MIXED = "mixed"  # Use mixed precision (training in fp32, inference in fp16)
    AUTO = "auto"  # Automatically select based on hardware
    
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

# Utility decorator for timing function execution
def time_execution(func):
    """Decorator to track execution time of methods."""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        start_time = time.time()
        result = func(self, *args, **kwargs)
        execution_time = time.time() - start_time

        # Update metrics
        func_name = func.__name__
        if not hasattr(self, '_execution_times'):
            self._execution_times = {}
        if not hasattr(self, '_call_counts'):
            self._call_counts = {}

        if func_name not in self._execution_times:
            self._execution_times[func_name] = []
        if func_name not in self._call_counts:
            self._call_counts[func_name] = 0

        self._execution_times[func_name].append(execution_time)
        self._call_counts[func_name] += 1

        # Limit stored times to prevent memory growth
        if len(self._execution_times[func_name]) > 100:
            self._execution_times[func_name].pop(0)

        return result
    return wrapper

try:
    # Try to import CircularBuffer from cache_manager
    from cache_manager import CircularBuffer
    CIRCULAR_BUFFER_AVAILABLE = True
except ImportError:
    # Define a simple CircularBuffer as fallback
    CIRCULAR_BUFFER_AVAILABLE = False
    class CircularBuffer:
        def __init__(self, max_size=100):
            self.max_size = max_size
            self.buffer = {}

        def __setitem__(self, key, value):
            self.buffer[key] = value
            if len(self.buffer) > self.max_size:
                # Remove oldest item (first key)
                self.buffer.pop(next(iter(self.buffer)))

        def __getitem__(self, key):
            return self.buffer.get(key)

        def get(self, key, default=None):
            return self.buffer.get(key, default)

        def __contains__(self, key):
            return key in self.buffer

        def clear(self):
            self.buffer.clear()

try:
    # Optional fault tolerance manager import
    from fault_manager import FaultToleranceManager, get_fault_tolerance_manager
    FAULT_TOLERANCE_AVAILABLE = True
except ImportError:
    FAULT_TOLERANCE_AVAILABLE = False
    FaultToleranceManager = None
    def get_fault_tolerance_manager(*args, **kwargs):
        logging.debug("fault_manager not available; fault tolerance disabled")
        return None


# --- Numba-optimized stat functions ---
@njit
def compute_zscore(spread):
    mean = spread.mean()
    std = spread.std()
    return (spread[-1] - mean) / std

@njit
def softmax_numba(vals):
    max_val = np.max(vals)
    exps = np.exp(vals - max_val)
    return exps / exps.sum()

@njit(parallel=True)
def vector_levels(base, spacing, levels, scale, base_size):
    prices_buy = np.empty(levels)
    prices_sell = np.empty(levels)
    sizes = np.empty(levels)
    for i in prange(levels):
        lvl = i + 1
        prices_buy[i] = base - lvl * spacing
        prices_sell[i] = base + lvl * spacing
        sizes[i] = base_size * (1 + scale * lvl)
    return prices_buy, prices_sell, sizes




# Configure logging
logger = logging.getLogger("QuantumHedge")
# Ensure logger has handlers to prevent "No handlers could be found" errors
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class QuantumHedgeAlgorithm:
    """
    A sophisticated implementation of the Quantum Hedge Algorithm for adaptive expert weighting
    in financial markets. This algorithm combines classical adaptive experts learning with
    quantum computing enhancements for probability distribution processing.

    The algorithm maintains weights for multiple experts, updates them based on performance,
    and can leverage quantum computing for potential advantage in processing market features
    and enhancing the weight distribution.
    """



    # Now update the QuantumHedgeAlgorithm class methods:

    def __init__(
        self, *args, # added
        market_data = None,
        options_model = None,  # added
        risk_params = None, # added
        num_experts: int = 8,  # Default to 8 experts for the standardized factors
        initial_weights: Optional[Union[np.ndarray, Dict[str, float]]] = None,
        use_quantum: bool = True,
        learning_rate: float = 0.03,
        regret_type: str = "external",
        quantum_backend: str = "pennylane",
        quantum_enhancement: float = 0.2,
        feature_map_type: str = "amplitude",
        feature_dim: int = 8,  # Default to 8 dimensions for standardized factors
        num_qubits: Optional[int] = None,
        market_adaptive_learning: bool = True,
        weight_decay: float = 0.01,
        min_weight: float = 0.001,  # Lower minimum weight to allow for better differentiation
        meta_learning_rate: float = 0.003,
        precision: PrecisionMode = PrecisionMode.AUTO, # Added
        mode: ProcessingMode = ProcessingMode.AUTO, # Added
        hw_manager: Optional[Any] = None, # Added
        hw_accelerator: Optional[Any] = None, # Added
        log_level: int = logging.INFO, # Added
        use_standard_factors: bool = True  # Whether to use the standard 8-factor model
    ):
        """
        Initialize the Quantum Hedge Algorithm.

        Args:
            num_experts: Number of experts to track
            initial_weights: Optional initial weights for experts (default: equal weights)
            use_quantum: Whether to use quantum computing enhancements
            learning_rate: Base learning rate for weight updates
            regret_type: Type of regret calculation ("external" or "internal")
            quantum_backend: PennyLane backend to use for quantum computations
            feature_dim: Dimension of market feature vectors for quantum circuit
            num_qubits: Number of qubits to use (defaults to feature_dim)
            market_adaptive_learning: Whether to adapt learning rate based on market conditions
            weight_decay: Rate at which weights decay toward uniformity (regularization)
            min_weight: Minimum weight floor for any expert
            meta_learning_rate: Learning rate for adaptive learning rate adjustment
            precision: Numeric precision mode (Added)
            mode: Processing mode (quantum, classical, hybrid, auto) (Added)
            hw_manager: HardwareManager instance or None to create new one (Added)
            hw_accelerator: HardwareAccelerator instance or None to create new one (Added)
            :param market_data: instance of AdaptiveMarketDataFetcher or any object
            providing the required market data and execution methods:
            get_spot, history, atr, correlation, realized_vol,
            compute_portfolio_delta, recent_trades, get_order_book_spread,
            place_order, place_limit_order, place_market_order,
            place_trailing_stop, onchain_whale.
            :param options_model: model for option pricing (get_option_chain, create_order)
            :param risk_params: dict of strategy weights and thresholds (see docs)
            log_level: Logging level (Added)
        """
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        
        # Store hardware components
        self.hw_manager = hw_manager
        self.hw_accelerator = hw_accelerator

        # Initialize hardware components if not provided
        if self.hw_manager is None and HARDWARE_ACCEL_AVAILABLE:
            self.hw_manager = HardwareManager.get_manager()
        if self.hw_accelerator is None and HARDWARE_ACCEL_AVAILABLE:
            self.hw_accelerator = HardwareAccelerator(enable_gpu=True)

        # Ensure hardware is initialized
        if self.hw_manager is not None and hasattr(self.hw_manager, 'initialize_hardware'):
             if not getattr(self.hw_manager, '_is_initialized', False):
                 self.hw_manager.initialize_hardware()
                 
        self.options_model = options_model
        self.risk_params = risk_params

        self._init_hardware_components() # Call the new method
        self.context_signals = {
            'whale_alert': 0.0,
            'black_swan_risk': 0.0,
            'fusion_confidence': 0.0
        }
        self._last_grid = 0


        self.options_model = options_model
        self.risk_params = risk_params
        self.context_signals = {
            'whale_alert': 0.0,
            'black_swan_risk': 0.0,
            'fusion_confidence': 0.0
        }
        self._last_grid = 0
        # Store core parameters
        self.market_data = market_data
        self.num_experts = num_experts
        self.use_quantum_preference = use_quantum # Renamed to clarify it's a preference
        self.base_learning_rate = learning_rate
        self.learning_rate = learning_rate  # Current effective learning rate
        self.regret_type = regret_type
        self.quantum_backend = quantum_backend # Keep for PennyLane device creation
        self.feature_dim = feature_dim
        self.num_qubits = num_qubits if num_qubits is not None else feature_dim
        self.market_adaptive_learning = market_adaptive_learning # Added back
        self.weight_decay = weight_decay # Added back
        self.min_weight = min_weight # Added back
        self.meta_learning_rate = meta_learning_rate # Added back
        self.quantum_enhancement = quantum_enhancement
        # Store processing parameters
        self.precision_mode = precision
        self.processing_mode = mode

        # Determine optimal dtype based on precision mode
        self.dtype, self.c_dtype = self._get_optimal_dtypes()

        # Set processing mode based on hardware availability and user preference
        if self.processing_mode == ProcessingMode.AUTO:
            if self.use_quantum_preference and self.quantum_available:
                self.processing_mode = ProcessingMode.QUANTUM
            else:
                self.processing_mode = ProcessingMode.CLASSICAL
        elif self.processing_mode == ProcessingMode.QUANTUM and not (self.use_quantum_preference and self.quantum_available):
             self.logger.warning("Quantum mode requested but not available or preferred. Falling back to classical.")
             self.processing_mode = ProcessingMode.CLASSICAL

        # Flag for quantum component initialization state
        self._quantum_components_initialized = False # Added

        # Enhanced initialization for standardized factors
        self.factor_names = []
        self.use_standard_factors = use_standard_factors
        
        # Initialize weights with either standard factors or generic experts
        if self.use_standard_factors and num_experts == 8:
            # Use standard 8-factor model
            self.factor_names = StandardFactors.get_ordered_list()
            
            # Handle dictionary of named weights if provided
            if isinstance(initial_weights, dict):
                # Ensure all factors are present and in correct order
                self.weights = np.zeros(num_experts, dtype=np.float64)
                for i, factor_name in enumerate(self.factor_names):
                    self.weights[i] = initial_weights.get(factor_name, StandardFactors.get_default_weights().get(factor_name, 1.0/num_experts))
            elif initial_weights is None:
                # Use default weights for standard factors
                default_weights = StandardFactors.get_default_weights()
                self.weights = np.array([default_weights[factor] for factor in self.factor_names], dtype=np.float64)
            else:
                # Use provided weights array
                if len(initial_weights) != num_experts:
                    raise ValueError(f"Expected {num_experts} initial weights, got {len(initial_weights)}")
                self.weights = np.array(initial_weights, dtype=np.float64)
        else:
            # Generic expert weights (non-standard factors or different number)
            if initial_weights is None:
                self.weights = np.ones(num_experts, dtype=np.float64) / num_experts
                self.factor_names = [f"expert_{i}" for i in range(num_experts)]
            else:
                if isinstance(initial_weights, dict):
                    # Convert dict to array
                    self.factor_names = list(initial_weights.keys())
                    self.weights = np.array(list(initial_weights.values()), dtype=np.float64)
                    if len(self.weights) != num_experts:
                        raise ValueError(f"Expected {num_experts} initial weights, got {len(self.weights)}")
                else:
                    # Use provided weights array
                    if len(initial_weights) != num_experts:
                        raise ValueError(f"Expected {num_experts} initial weights, got {len(initial_weights)}")
                    self.weights = np.array(initial_weights, dtype=np.float64)
                    self.factor_names = [f"expert_{i}" for i in range(num_experts)]
        
        # Ensure weights are properly normalized and non-negative
        self.weights = np.maximum(self.weights, self.min_weight)
        self.weights /= np.sum(self.weights)
        
        # Create factor weight mapping for easier access
        self.factor_weight_map = {name: self.weights[i] for i, name in enumerate(self.factor_names)}

        # History tracking
        self.weights_history = [self.weights.copy()]
        self.rewards_history = []
        self.regret_history = [0.0]
        self.cumulative_regret = 0.0
        self.weight_history = [self.weights.copy()]

        self.cumulative_reward = 0
        self.cumulative_rewards = [0]
        self.best_expert_reward = 0
        self.best_expert_rewards = [0]
        self.regret = 0


        self.feature_map_type = feature_map_type
        self.iterations = 0

        # Market sensitivity parameters
        self.market_volatility_sensitivity = 0.2
        self.market_trend_sensitivity = 0.1
        self.market_volume_sensitivity = 0.1

        # Adaptive learning rate parameters
        self.adaptive_lr_weights = {
            'volatility': 0.3,
            'trend': 0.3,
            'volume': 0.2,
            'expert_disagreement': 0.2
        }

        # Expert performance tracking
        self.expert_performance = np.zeros(num_experts)
        self.expert_consistency = np.zeros(num_experts)

        # Initialize quantum circuit components later, on first use
        # if self.use_quantum: # Original check, now handled by processing_mode and lazy init
        #     self._initialize_quantum_components() # Removed

        self.logger.info(f"Initialized QuantumHedgeAlgorithm with {num_experts} experts, mode={self.processing_mode.value}")

    def _init_hardware_components(self):
        """Initialize hardware components and set availability flags."""
        # Initialize hardware components if not provided
        if self.hw_manager is None and HARDWARE_ACCEL_AVAILABLE:
            self.hw_manager = HardwareManager.get_manager()
        if self.hw_accelerator is None and HARDWARE_ACCEL_AVAILABLE:
            self.hw_accelerator = HardwareAccelerator(enable_gpu=True)

        # Ensure hardware is initialized
        if self.hw_manager is not None and hasattr(self.hw_manager, 'initialize_hardware'):
             if not getattr(self.hw_manager, '_is_initialized', False):
                 self.hw_manager.initialize_hardware()

        # Determine hardware capabilities
        self.quantum_available = getattr(self.hw_manager, 'quantum_available', False)
        self.gpu_available = getattr(self.hw_accelerator, 'gpu_available', False) if self.hw_accelerator else False # Use hw_accelerator for GPU check

    def _initialize_quantum_components(self):
        """Initialize the quantum computing components for the algorithm using PennyLane."""
        if self._quantum_components_initialized: # Check if already initialized
            return

        self.logger.info("Initializing quantum components...")

        # Get and initialize the quantum device using the manager or fallback
        # Use self.quantum_available determined in _init_hardware_components
        if not self.quantum_available:
             self.logger.warning("Quantum hardware not available. Skipping quantum component initialization.")
             self._quantum_components_initialized = True
             return

        self.device, quantum_device_initialized = self._get_quantum_device_from_manager()

        # If device initialization failed, set flags and return
        if self.device is None or not quantum_device_initialized:
            self.use_quantum_preference = False
            self.processing_mode = ProcessingMode.CLASSICAL
            self._quantum_components_initialized = True
            self.logger.warning("Quantum components could not be initialized. Falling back to classical.")
            return

        try:
            # Define the quantum circuit as a QNode using the initialized device
            @qml.qnode(self.device)
            def quantum_circuit(features, weights):
                # This function will be filled by the specific circuit definition
                # when called by _process_with_quantum
                return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]

            self.quantum_circuit = quantum_circuit
            self.logger.info(f"Successfully initialized PennyLane quantum device: {self.device.name}")

        except Exception as e:
            self.logger.error(f"Failed to define quantum circuits: {e}")
            self.use_quantum_preference = False
            self.processing_mode = ProcessingMode.CLASSICAL
            self.logger.warning("Falling back to classical computation")

        self._quantum_components_initialized = True # Mark as initialized (even if failed)


    def _get_optimal_dtypes(self) -> Tuple[np.dtype, np.dtype]:
        """Determine optimal dtypes based on precision mode and hardware."""
        if self.precision_mode == PrecisionMode.SINGLE:
            return np.float32, np.complex64
        elif self.precision_mode == PrecisionMode.DOUBLE:
            return np.float64, np.complex128
        elif self.precision_mode == PrecisionMode.AUTO:
            # Use hardware accelerator to determine precision
            if self.hw_accelerator is not None and hasattr(self.hw_accelerator, 'get_accelerator_type'):
                accel_type = self.hw_accelerator.get_accelerator_type()
                # AMD GPUs often prefer single precision
                if accel_type == AcceleratorType.ROCM:
                    self.logger.debug("Auto-precision: ROCm detected, using float32.")
                    return np.float32, np.complex64
                # NVIDIA GPUs often perform well with FP32, support FP64
                elif accel_type == AcceleratorType.CUDA:
                     self.logger.debug("Auto-precision: CUDA detected, using float32.")
                     return np.float32, np.complex64
                # MPS and IPEX also typically prefer FP32
                elif accel_type in (AcceleratorType.MPS, AcceleratorType.IPEX):
                     self.logger.debug(f"Auto-precision: {accel_type} detected, using float32.")
                     return np.float32, np.complex64

            # Fallback to double precision if no specific hardware preference or detection failed
            self.logger.debug("Auto-precision: No specific hardware preference, using float64.")
            return np.float64, np.complex128
        else: # Should not happen if Enum conversion is correct
            self.logger.warning(f"Unknown precision mode: {self.precision_mode}. Defaulting to float32.")
            return np.float32, np.complex64

    def _get_quantum_device_from_manager(self) -> Tuple[Any, bool]:
        """
        Initialize the quantum device with optimal settings using HardwareManager.

        Returns:
            Tuple of (device, quantum_available)
        """
        if not QUANTUM_AVAILABLE:
            self.logger.warning("PennyLane not available. Quantum features disabled.")
            return None, False

        if self.hw_manager is not None and hasattr(self.hw_manager, 'get_optimal_device'):
            try:
                # Get quantum device configuration from hardware manager
                device_config = self.hw_manager.get_optimal_device(
                    quantum_required=True,
                    qubits_required=self.num_qubits
                )

                # Check if the hardware manager returned a quantum device
                if device_config.get('type') == 'quantum':
                    device_name = device_config.get('device', 'default.qubit')
                    wires = device_config.get('wires', self.num_qubits)
                    shots = device_config.get('shots', None) # Use manager's shots if provided

                    # Try to create PennyLane device
                    try:
                        device = qml.device(
                            device_name,
                            wires=wires,
                            shots=shots,
                            c_dtype=self.c_dtype # Use determined complex dtype
                        )
                        self.logger.info(f"Using HardwareManager-recommended quantum device: {device.name}")
                        return device, True
                    except Exception as e:
                        self.logger.warning(f"Failed to initialize HardwareManager-recommended device {device_name}: {e}")
                        # Fallback to default.qubit if recommended device fails
                        try:
                            device = qml.device("default.qubit", wires=self.num_qubits, shots=None)
                            self.logger.warning("Falling back to default.qubit.")
                            return device, True
                        except Exception as e_fallback:
                            self.logger.error(f"Failed to initialize default.qubit fallback: {e_fallback}")
                            return None, False
                else:
                    self.logger.warning("HardwareManager did not recommend a quantum device.")
                    return None, False

            except Exception as e:
                self.logger.warning(f"Error getting optimal quantum device from HardwareManager: {e}")
                # Fallback to default.qubit if manager call fails
                try:
                    device = qml.device("default.qubit", wires=self.num_qubits, shots=None)
                    self.logger.warning("Falling back to default.qubit.")
                    return device, True
                except Exception as e_fallback:
                    self.logger.error(f"Failed to initialize default.qubit fallback: {e_fallback}")
                    return None, False

        # Fallback if HardwareManager is not available or doesn't have get_optimal_device
        self.logger.info("HardwareManager not available or does not support optimal device selection. Trying default PennyLane devices.")
        try:
            # Try lightning.kokkos first (often good for AMD)
            try:
                device = qml.device("lightning.kokkos", wires=self.num_qubits, shots=self.shots, c_dtype=self.c_dtype)
                self.logger.info("Using lightning.kokkos device.")
                return device, True
            except Exception as e_kokkos:
                self.logger.debug(f"lightning.kokkos not available: {e_kokkos}. Trying lightning.gpu.")
                # Try lightning.gpu (often good for NVIDIA)
                try:
                    device = qml.device("lightning.gpu", wires=self.num_qubits, shots=self.shots, c_dtype=self.c_dtype)
                    self.logger.info("Using lightning.gpu device.")
                    return device, True
                except Exception as e_gpu:
                    self.logger.debug(f"lightning.gpu not available: {e_gpu}. Trying lightning.qubit.")
                    # Try lightning.qubit (CPU simulator)
                    try:
                        device = qml.device("lightning.qubit", wires=self.num_qubits, shots=self.shots, c_dtype=self.c_dtype)
                        self.logger.info("Using lightning.qubit device.")
                        return device, True
                    except Exception as e_qubit:
                        self.logger.debug(f"lightning.qubit not available: {e_qubit}. Trying default.qubit.")
                        # Fallback to default.qubit
                        try:
                            device = qml.device("default.qubit", wires=self.num_qubits, shots=None)
                            self.logger.info("Using default.qubit device.")
                            return device, True
                        except Exception as e_default:
                            self.logger.error(f"Failed to initialize default.qubit fallback: {e_default}")
                            return None, False
        except Exception as e:
            self.logger.error(f"Unexpected error during fallback device initialization: {e}")
            return None, False


    def _quantum_circuit(self, features: np.ndarray, weights: np.ndarray):
        """
        Define the complete quantum circuit using PennyLane.
        
        Args:
            features: Normalized market feature vector
            weights: Current expert weights
            
        Returns:
            PennyLane quantum circuit function
        """
        @qml.qnode(self.device)
        def circuit(features_input, weights_input):
            # Apply the enhancement circuit which includes the feature mapping
            self._create_enhancement_circuit(features_input, weights_input)()
            
            # Return the expectations of Z measurements for each qubit
            return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]
        
        return circuit

    def _create_quantum_circuit(self) -> Callable:
        """
        Create the quantum circuit for enhancing probability distributions.
        
        Returns:
            A PennyLane QNode function implementing the quantum circuit
        """
        @qml.qnode(self.device)
        def quantum_circuit(weights, market_features=None):
            """
            Quantum circuit that enhances the probability distribution using
            principles from quantum amplitude amplification.
            
            Args:
                weights: Classical probability weights for each expert
                market_features: Features from market data for encoding
                
            Returns:
                Measurement probabilities representing enhanced distribution
            """
            # Normalize weights to ensure valid quantum state
            norm_weights = np.sqrt(weights / np.sum(weights))
            
            # Prepare initial state based on weights
            for i in range(self.num_experts):
                if i < 2**self.n_qubits:  # Ensure we don't exceed quantum state space
                    # Convert expert index to binary for qubit addressing
                    binary_rep = format(i, f'0{self.n_qubits}b')
                    # Apply controlled rotations to encode weights
                    for j, bit in enumerate(binary_rep):
                        if bit == '0':
                            qml.PauliX(wires=j)
                    
                    # Apply controlled rotation to encode amplitude
                    angle = 2 * np.arcsin(norm_weights[i])
                    qml.RY(angle, wires=0)
                    
                    # Undo the bit flips
                    for j, bit in enumerate(binary_rep):
                        if bit == '0':
                            qml.PauliX(wires=j)
            
            # Apply feature map if market features are provided
            if market_features is not None:
                self._apply_feature_map(market_features)
            
            # Apply quantum amplitude amplification
            self._apply_amplitude_amplification(weights)
            
            # Measure all qubits to get the enhanced distribution
            return qml.probs(wires=range(self.n_qubits))
        
        return quantum_circuit
    
    def _apply_feature_map(self, features: np.ndarray) -> None:
        """
        Apply a quantum feature map to encode market features into the quantum state.
        
        Args:
            features: Array of market features to encode
        """
        # Ensure features are scaled appropriately
        scaled_features = features.copy()
        if np.max(np.abs(scaled_features)) > 0:
            scaled_features = scaled_features / np.max(np.abs(scaled_features)) * np.pi
        
        if self.feature_map_type == "amplitude":
            # Amplitude encoding
            for i in range(self.num_qubits):
                if i < len(scaled_features):
                    qml.RY(scaled_features[i], wires=i)
                    qml.RZ(scaled_features[i] * 0.5, wires=i)
                    
        elif self.feature_map_type == "angle":
            # Angle encoding
            for i in range(self.num_qubits):
                if i < len(scaled_features):
                    qml.RX(scaled_features[i], wires=i)
                    qml.RZ(scaled_features[i], wires=i)
                    
        elif self.feature_map_type == "iqp":
            # IQP-inspired feature map (more complex entanglement)
            # First layer - single qubit rotations
            for i in range(self.num_qubits):
                if i < len(scaled_features):
                    qml.RZ(scaled_features[i], wires=i)
            
            # Entangling layer
            for i in range(self.num_qubits - 1):
                qml.CNOT(wires=[i, i+1])
            
            # Second rotation layer
            for i in range(self.num_qubits):
                if i < len(scaled_features):
                    qml.RY(scaled_features[i] * 0.5, wires=i)
    
    def _apply_amplitude_amplification(self, weights: np.ndarray) -> None:
        """
        Apply quantum amplitude amplification to enhance the probabilities
        of the most promising experts.
        
        Args:
            weights: Current weights of experts
        """
        # Create a phase oracle that marks the "good" states
        # In this case, states corresponding to experts with higher weights
        weight_threshold = np.mean(weights) * (1 + self.quantum_enhancement)
        
        # Apply phase shifts proportional to weights
        for i in range(self.num_experts):
            if i < 2**self.num_qubits: # Use num_qubits here
                binary_rep = format(i, f'0{self.num_qubits}b') # Use num_qubits here
                
                # Create projector for this expert state
                for j, bit in enumerate(binary_rep):
                    if bit == '0':
                        qml.PauliX(wires=j)
                
                # Apply phase shift if weight is above threshold
                if weights[i] > weight_threshold:
                    qml.PhaseShift(np.pi, wires=0)
                
                # Undo the projector
                for j, bit in enumerate(binary_rep):
                    if bit == '0':
                        qml.PauliX(wires=j)
        
        # Apply diffusion operator (Grover diffusion)
        for i in range(self.num_qubits):
            qml.Hadamard(wires=i)
            
        # Apply zero-state phase shift
        for i in range(self.num_qubits):
            qml.PauliX(wires=i)
            
        # Multi-controlled phase shift
        qml.PhaseShift(np.pi, wires=0)
        
        # Undo X gates
        for i in range(self.num_qubits):
            qml.PauliX(wires=i)
        
        # Apply final Hadamard gates
        for i in range(self.num_qubits):
            qml.Hadamard(wires=i)
    
    
    def _process_with_quantum(self, market_features: np.ndarray) -> np.ndarray:
        """
        Process market features with quantum computation using PennyLane to enhance expert weights.

        Args:
            market_features: Vector of market features (normalized)

        Returns:
            Enhanced weight distribution
        """
        # Ensure quantum components are initialized
        if not self._quantum_components_initialized:
            self._initialize_quantum_components()

        # Check if quantum processing is enabled and available
        if self.processing_mode == ProcessingMode.CLASSICAL or not self.quantum_available or self.device is None:
            self.logger.debug("Quantum processing not enabled or available. Using classical weights.")
            return self.weights.copy()

        try:
            # Normalize market features to range [0, 2π] for quantum circuit
            # Handle case where market_features might be all zeros or have zero max abs value
            max_abs_features = np.max(np.abs(market_features))
            if max_abs_features > 1e-9: # Use a small epsilon to avoid division by zero
                normalized_features = market_features / max_abs_features * np.pi
            else:
                normalized_features = np.zeros_like(market_features) # Use zeros if features are all zeros

            # Execute the quantum circuit using PennyLane
            circuit = self._quantum_circuit(normalized_features, self.weights)
            results = circuit(normalized_features, self.weights)

            # Convert the measurement results (expectations in [-1,1]) to probabilities [0,1]
            expectations = np.array(results)
            probabilities = (expectations + 1) / 2  # Map from [-1,1] to [0,1]

            # Map these probabilities to expert weights
            # We need to ensure we have enough values for all experts
            enhanced_weights = np.zeros(self.num_experts)

            for i in range(self.num_experts):
                # Map the quantum results to expert indices
                # For example, use combinations of qubits or specific qubits for each expert
                if i < len(probabilities):
                    # Direct mapping if we have enough qubits
                    enhanced_weights[i] = probabilities[i]
                else:
                    # For more experts than qubits, use modulo mapping
                    idx = i % len(probabilities)
                    enhanced_weights[i] = probabilities[idx]

            # Alternative mapping method: use binary representation
            # Check if we have enough qubits to represent expert indices
            if self.num_qubits >= np.ceil(np.log2(self.num_experts)) if self.num_experts > 0 else True:
                # If we have enough qubits, we can use them to represent binary patterns
                # This creates more complex weight distributions
                # Ensure we don't exceed the number of experts or the number of possible qubit states
                num_patterns = min(2**self.num_qubits, self.num_experts)
                enhanced_weights_binary = np.zeros(self.num_experts)

                for bit_pattern in range(num_patterns):
                    weight_value = 1.0

                    # Calculate weight based on binary representation and probabilities
                    for bit_idx in range(self.num_qubits):
                        if (bit_pattern & (1 << bit_idx)) > 0:
                            # If the bit is 1, use the probability of the |1> state for this qubit
                            if bit_idx < len(probabilities):
                                weight_value *= probabilities[bit_idx]
                            else:
                                # If not enough probabilities, assume neutral influence (0.5)
                                weight_value *= 0.5
                        else:
                            # If the bit is 0, use the probability of the |0> state (1 - probability of |1>)
                            if bit_idx < len(probabilities):
                                weight_value *= (1 - probabilities[bit_idx])
                            else:
                                # If not enough probabilities, assume neutral influence (0.5)
                                weight_value *= 0.5

                    if bit_pattern < self.num_experts:
                        enhanced_weights_binary[bit_pattern] = weight_value

                # Use the binary mapping result if it's available and seems reasonable
                if np.sum(enhanced_weights_binary) > 0:
                     enhanced_weights = enhanced_weights_binary
                else:
                     # Fallback to the direct mapping if binary mapping results in zero sum
                     self.logger.warning("Binary mapping resulted in zero sum, falling back to direct mapping.")


            # Normalize the enhanced weights
            if np.sum(enhanced_weights) > 0:
                enhanced_weights = enhanced_weights / np.sum(enhanced_weights)
            else:
                # Fallback to original weights if quantum computation fails or results in zero sum
                self.logger.warning("Enhanced weights sum to zero after quantum processing. Falling back to original weights.")
                enhanced_weights = self.weights.copy()

            # Blend original weights with quantum-enhanced weights
            # The quantum influence increases as we gain confidence in its effectiveness
            quantum_confidence = min(0.5, self.iterations / 1000)  # Gradually increase quantum influence
            blended_weights = (1 - quantum_confidence) * self.weights + quantum_confidence * enhanced_weights

            # Ensure weights are normalized
            if np.sum(blended_weights) > 0:
                 blended_weights = blended_weights / np.sum(blended_weights)
            else:
                 # Fallback to original weights if blending results in zero sum
                 self.logger.warning("Blended weights sum to zero. Falling back to original weights.")
                 blended_weights = self.weights.copy()


            self.logger.debug(f"Generated quantum-enhanced weights: {blended_weights}")
            return blended_weights

        except Exception as e:
            self.logger.error(f"Quantum processing error: {e}")
            self.logger.warning("Falling back to classical weights")
            return self.weights.copy()

    def _map_quantum_to_weights(self, quantum_output):
        """
        Fallback method to map quantum circuit output to weights when
        the accelerator doesn't provide this functionality.
        
        Args:
            quantum_output: Output from quantum circuit
            
        Returns:
            Mapped weights for experts
        """
        # Convert to numpy array if not already
        quantum_output = np.array(quantum_output)
        
        # Basic mapping: ensure positive values and normalize
        weights = np.abs(quantum_output)
        
        # Handle case where we have more experts than quantum outputs
        if len(weights) < self.num_experts:
            # Extend by cycling the values
            weights = np.tile(weights, (self.num_experts // len(weights) + 1))[:self.num_experts]
        
        # Handle case where we have more quantum outputs than experts
        if len(weights) > self.num_experts:
            # Use first values or aggregate
            weights = weights[:self.num_experts]
        
        # Normalize
        if np.sum(weights) > 0:
            return weights / np.sum(weights)
        else:
            return np.ones(self.num_experts) / self.num_experts        
            
    def _create_feature_map_circuit(self, features: np.ndarray):
        """
        Create a PennyLane quantum circuit that implements a feature map for data encoding.
        
        Args:
            features: Normalized market feature vector
            
        Returns:
            A function that applies the feature map to a PennyLane QNode
        """
        # Ensure features are appropriately sized and normalized
        if len(features) < self.feature_dim:
            # Pad with zeros if needed
            features = np.pad(features, (0, self.feature_dim - len(features)), mode='constant') # Use constant mode
        elif len(features) > self.feature_dim:
            # Truncate if too many features
            features = features[:self.feature_dim]
            
        # Define the ZZ feature map similar to Qiskit's ZZFeatureMap
        def zz_feature_map():
            # First layer of Hadamards
            for i in range(self.num_qubits):
                qml.Hadamard(wires=i)
                
            # First layer of rotations based on features
            for i in range(self.num_qubits):
                if i < len(features):
                    qml.RZ(features[i], wires=i)
                
            # Entangling layer using ZZ rotations
            for i in range(self.num_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
                if i < len(features) and i + 1 < len(features):
                    qml.RZ(features[i] * features[i + 1], wires=i + 1)
                qml.CNOT(wires=[i, i + 1])
                
            # Second layer of rotations
            for i in range(self.num_qubits):
                if i < len(features):
                    qml.RZ(features[i], wires=i)
                
            # Second layer of Hadamards
            for i in range(self.num_qubits):
                qml.Hadamard(wires=i)
        
        return zz_feature_map
    
    def _create_enhancement_circuit(self, features: np.ndarray, weights: np.ndarray):
        """
        Create a PennyLane circuit that enhances the probability distribution
        based on the current weights and market features.
        
        Args:
            features: Normalized market feature vector
            weights: Current expert weights
            
        Returns:
            A function that applies the enhancement to a PennyLane QNode
        """
        def enhancement_circuit():
            # Apply feature map first
            self._create_feature_map_circuit(features)()
            
            # Additional enhancement operations
            # Apply controlled rotations for useful interference patterns
            for i in range(self.num_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
                qml.RZ(np.pi / 4, wires=i + 1)
                qml.CNOT(wires=[i, i + 1])
            
            # Apply Hadamard gates to transform phase differences into amplitude differences
            for i in range(self.num_qubits):
                qml.Hadamard(wires=i)
            
            # Apply additional phase rotations based on current weights
            # This creates a feedback mechanism where current weights influence quantum enhancement
            for i in range(min(self.num_qubits, self.num_experts)):
                # Ensure weight index is within bounds
                weight_index = i % len(weights)
                phase = np.arcsin(np.sqrt(weights[weight_index]))
                qml.RZ(phase * 2, wires=i)
        
        return enhancement_circuit


    def get_weights(self, market_features: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Get the current expert weights, optionally enhanced by quantum processing.

        Args:
            market_features: Optional vector of market features for quantum enhancement

        Returns:
            Current expert weights as a normalized array
        """
        # Ensure quantum components are initialized if quantum processing is preferred
        if self.use_quantum_preference and not self._quantum_components_initialized:
             self._initialize_quantum_components()

        # Check if quantum processing is enabled and available
        if self.processing_mode != ProcessingMode.CLASSICAL and self.quantum_available and market_features is not None:
            try:
                # Process weights with quantum enhancement
                return self._process_with_quantum(market_features)
            except Exception as e:
                self.logger.error(f"Error in quantum weight processing: {e}")
                return self.weights.copy()
        else:
            # Return classical weights
            return self.weights.copy()

    def _calculate_adaptive_learning_rate(
        self,
        market_features: Optional[Dict[str, float]] = None,
        expert_signals: Optional[np.ndarray] = None
    ) -> float:
        """
        Calculate an adaptive learning rate based on market conditions and expert consensus.
        
        Args:
            market_features: Dictionary of market metrics
            expert_signals: Array of expert signals/predictions
            
        Returns:
            Adjusted learning rate
        """
        if not self.market_adaptive_learning:
            return self.base_learning_rate

        # Start with base learning rate
        adaptive_lr = self.base_learning_rate

        # Initialize factors to 1.0 to ensure they are always defined
        caution_factor = 1.0
        trend_factor = 1.0
        vol_factor = 1.0
        disagreement_factor = 1.0

        # Adjust based on iteration count (decreasing over time for stability)
        iteration_factor = max(0.5, 1.0 / np.sqrt(1 + self.iterations * 0.01))
        adaptive_lr *= iteration_factor

        # Adjust based on market features if provided
        if market_features is not None:
            # Volatility adjustment - higher volatility = higher learning rate
            if 'volatility' in market_features:
                volatility = market_features['volatility']
                caution_factor = 1.0 + self.market_volatility_sensitivity * volatility

            # Trend strength adjustment - stronger trend = lower learning rate
            if 'trend_strength' in market_features:
                trend_factor = 1.0 - self.market_trend_sensitivity * market_features['trend_strength']

            # Volume adjustment - higher volume = higher learning rate
            if 'volume' in market_features:
                vol_factor = 1.0 + self.market_volume_sensitivity * market_features['volume']

        # Adjust based on expert disagreement if signals provided
        if expert_signals is not None:
            # Calculate variance of expert signals as measure of disagreement
            signal_variance = np.var(expert_signals)
            disagreement_factor = 1.0 + signal_variance * 2  # More disagreement = higher learning rate

        # Apply factors to adaptive learning rate
        adaptive_lr *= caution_factor
        adaptive_lr *= max(0.5, trend_factor) # Ensure trend_factor doesn't reduce by more than half
        adaptive_lr *= vol_factor
        adaptive_lr *= disagreement_factor

        # Ensure learning rate is within reasonable bounds
        adaptive_lr = max(0.001, min(0.5, adaptive_lr))

        return adaptive_lr
    
    def _calculate_regret(self, rewards: np.ndarray) -> float:
        """
        Calculate regret based on rewards and current weights.
        
        Args:
            rewards: Array of rewards for each expert
            
        Returns:
            Calculated regret value
        """
        if self.regret_type == "external":
            # External regret: difference between reward of best expert and our weighted reward
            best_expert_reward = np.max(rewards)
            weighted_reward = np.sum(self.weights * rewards)
            regret = best_expert_reward - weighted_reward
        else:  # "internal" regret
            # Internal regret: more sophisticated pairwise comparison
            weighted_reward = np.sum(self.weights * rewards)
            regrets = np.zeros(self.num_experts)
            
            for i in range(self.num_experts):
                # Calculate regret of not putting all weight on expert i
                regrets[i] = max(0, rewards[i] - weighted_reward)
            
            regret = np.sum(self.weights * regrets)
        
        return float(regret)

    def _apply_multiplicative_weights_update(
        self, 
        rewards: np.ndarray, 
        learning_rate: float,
        market_features: Optional[Dict[str, float]] = None # Added market_features parameter
    ) -> np.ndarray:
        """
        Apply the multiplicative weights update rule to adjust expert weights.
        
        Args:
            rewards: Array of rewards for each expert
            learning_rate: Current learning rate
            market_features: Optional dictionary of market metrics for adaptive learning (Added)
            
        Returns:
            Updated weight array
        """
        # Note: Iteration count increment and adaptive learning rate calculation
        # have been moved to the `update` method where market_features is available.
        # This method now just applies the update rule with the provided learning_rate.
    
        # Normalize rewards to [0, 1] range to avoid numerical issues
        min_reward = np.min(rewards)
        max_reward = np.max(rewards)
        
        if max_reward > min_reward:
            normalized_rewards = (rewards - min_reward) / (max_reward - min_reward)
        else:
            normalized_rewards = np.zeros_like(rewards)
        
        # Apply multiplicative weights update formula
        updated_weights = self.weights * np.exp(learning_rate * normalized_rewards)
        
        # Apply weight decay toward uniform distribution (regularization)
        uniform_weights = np.ones(self.num_experts) / self.num_experts
        updated_weights = (1 - self.weight_decay) * updated_weights + self.weight_decay * uniform_weights
        
        # Ensure minimum weight for each expert (prevents experts from being completely ignored)
        updated_weights = np.maximum(updated_weights, self.min_weight)
        
        # Normalize weights to sum to 1
        updated_weights = updated_weights / np.sum(updated_weights)
        
        return updated_weights
        

    def update(
        self,
        rewards: np.ndarray,
        market_features: Optional[Dict[str, float]] = None, # Expects DICT from agent
        expert_signals: Optional[np.ndarray] = None
    ) -> float:
        if len(rewards) != self.num_experts:
            raise ValueError(f"Expected {self.num_experts} rewards, got {len(rewards)}")

        # Calculate adaptive learning rate using the market_features DICT
        self.learning_rate = self._calculate_adaptive_learning_rate(market_features, expert_signals)

        regret = self._calculate_regret(rewards)
        self.cumulative_regret += regret

        self.expert_performance = 0.9 * self.expert_performance + 0.1 * rewards
        if len(self.rewards_history) > 0:
            recent_rewards = np.array(self.rewards_history[-5:] + [rewards])
            expert_variances = np.var(recent_rewards, axis=0)
            self.expert_consistency = 1.0 / (1.0 + expert_variances + 1e-9) # Added epsilon

        # _apply_multiplicative_weights_update currently doesn't use its market_features param,
        # but if it did, it would also expect a dict based on this flow.
        self.weights = self._apply_multiplicative_weights_update(rewards, self.learning_rate, market_features)

        self.weights_history.append(self.weights.copy())
        self.rewards_history.append(rewards.copy())
        self.regret_history.append(regret)
        self.iterations += 1

        # --- Meta-learning part: Needs an ARRAY for self.quantum_circuit ---
        if market_features is not None and self.iterations > 10 and \
           self.processing_mode != ProcessingMode.CLASSICAL and self.quantum_available:

            market_features_array_for_quantum = None
            if isinstance(market_features, dict):
                # Attempt to construct the array QHA's quantum circuit expects.
                # This mapping is an assumption and might need to be more robust or configurable.
                # Defaulting to specific keys or requiring QHA to be told how to map.
                # For now, let's assume QHA's feature_dim corresponds to a known order of these keys.
                # Example: if feature_dim = 4, it expects ['volatility', 'trend_strength', 'trend_direction', 'volume']
                # This is fragile and ideally QHA would have a defined mapping.
                default_keys_for_quantum_array = ['volatility', 'trend_strength', 'trend_direction', 'volume']
                temp_array = []
                for i in range(self.feature_dim): # self.feature_dim is from QHA's __init__
                    key_to_use = default_keys_for_quantum_array[i] if i < len(default_keys_for_quantum_array) else f"feature_{i}"
                    temp_array.append(float(market_features.get(key_to_use, 0.0))) # Ensure float, default to 0.0
                market_features_array_for_quantum = np.array(temp_array, dtype=self.dtype)

                # Ensure it strictly matches self.feature_dim (already handled by padding/truncation in _process_with_quantum, but good to be explicit)
                if len(market_features_array_for_quantum) < self.feature_dim:
                    market_features_array_for_quantum = np.pad(market_features_array_for_quantum, (0, self.feature_dim - len(market_features_array_for_quantum)), mode='constant')
                elif len(market_features_array_for_quantum) > self.feature_dim:
                    market_features_array_for_quantum = market_features_array_for_quantum[:self.feature_dim]
            else:
                self.logger.warning("Meta-learning in QHA.update expects market_features as a dict to convert to array for quantum circuit, but received non-dict. Skipping quantum meta-learning.")

            if market_features_array_for_quantum is not None:
                try:
                    if not self._quantum_components_initialized:
                        self._initialize_quantum_components()

                    if self.quantum_available and self.device is not None and self.processing_mode != ProcessingMode.CLASSICAL:
                        # self.quantum_circuit is the QNode that takes (weights, market_features_array)                        # However, the original QHA update meta-learning uses self.quantum_circuit(self.weights, market_features)
                        # which is a QNode returned by self._quantum_circuit(features, weights)
                        # Let's use the _process_with_quantum method as it encapsulates the logic better
                        # and handles normalization, etc. _process_with_quantum uses self.weights internally.

                        # The meta-learning aims to get a quantum-enhanced distribution and blend it.
                        # Re-using _process_with_quantum makes sense here.
                        enhanced_weights_from_meta = self._process_with_quantum(market_features_array_for_quantum)

                        # Blend classical and quantum weights (original logic from the meta-learning part)
                        self.weights = (1 - self.quantum_enhancement) * self.weights + \
                                     self.quantum_enhancement * enhanced_weights_from_meta
                        
                        # Normalize final weights
                        if np.sum(self.weights) > 0:
                            self.weights /= np.sum(self.weights)
                        else: # Should not happen if enhanced_weights_from_meta is valid
                            self.logger.warning("Weights sum to zero after quantum meta-learning blend. Resetting to uniform.")
                            self.weights = np.ones_like(self.weights) / self.num_experts
                        
                        self.logger.debug("Applied quantum meta-learning adjustment to weights in QHA.update.")
                except Exception as e_meta:
                    self.logger.error(f"Error during quantum meta-learning in QHA.update: {e_meta}")
        
        logger.debug(f"Update: learning_rate={self.learning_rate:.4f}, regret={regret:.4f}")
        return regret

    def hedge_with_options(self, asset, target_delta, expiry):
        # Current delta exposure from portfolio
        current_delta = self.market_data.compute_portfolio_delta(asset)
        delta_shortfall = target_delta - current_delta
        # Fetch option Greeks
        strikes, greeks = self.options_model.get_option_chain(asset, expiry)
        # Choose strike minimizing gamma risk vs delta need
        best = min(greeks, key=lambda g: abs(delta_shortfall / g['gamma']))
        qty = delta_shortfall / best['delta']
        return self.options_model.create_order(
            asset, best['strike'], expiry, abs(qty), 'buy' if qty>0 else 'sell', best['type']
        )

    # --- Statistical pair trading ---
    def hedge_with_pair_trading(self, asset_x, asset_y):
        window = self.risk_params['coint_window']
        # fetch and vectorize
        px = np.array(self.market_data.history(asset_x, window))
        py = np.array(self.market_data.history(asset_y, window))
        pval, spread = self.market_data.cointegration_test(px, py)
        if pval < self.risk_params['coint_p']:
            z = compute_zscore(spread)
            if abs(z) > self.risk_params['pair_z']:
                beta = self.market_data.rolling_beta(px, py)
                size_x = self.risk_params['pair_base_size']
                size_y = size_x * beta
                if z > 0:
                    return (
                        self.market_data.place_order(asset_x, 'short', size_x),
                        self.market_data.place_order(asset_y, 'long', size_y)
                    )
                else:
                    return (
                        self.market_data.place_order(asset_x, 'long', size_x),
                        self.market_data.place_order(asset_y, 'short', size_y)
                    )
        return None, None

    # --- Correlation-weighted cross-short ---
    def hedge_with_cross_short(self, primary, hedge_asset):
        corr = self.market_data.correlation(primary, hedge_asset, self.risk_params['corr_window'])
        if corr >= self.risk_params['corr_thresh']:
            vol_p = self.market_data.realized_vol(primary)
            vol_h = self.market_data.realized_vol(hedge_asset)
            size = self.risk_params['cross_base_size'] * (vol_p / vol_h)
            return self.market_data.place_order(hedge_asset, 'short', size)
        return None

    # --- Volatility-adaptive grid hedging ---
    def grid_hedge(self, asset):
        now = time.time()
        if now - self._last_grid < self.risk_params['grid_cooldown']:
            return []
        self._last_grid = now
        levels = self.risk_params['grid_levels']
        atr = self.market_data.atr(asset, self.risk_params['atr_window'])
        spacing = atr * self.risk_params['grid_spacing_mult']
        base = self.market_data.get_spot(asset)
        p_buy, p_sell, sizes = vector_levels(
            base, spacing, levels,
            self.risk_params['grid_size_scale'],
            self.risk_params['grid_base_size']
        )
        orders = []
        for price, size in zip(p_buy, sizes):
            orders.append(self.market_data.place_limit_order(asset, 'buy', size, price))
        for price, size in zip(p_sell, sizes):
            orders.append(self.market_data.place_limit_order(asset, 'sell', size, price))
        return orders

    # --- Whale-momentum detection & rapid entry ---
    def detect_whale_momentum(self, asset):
        trades = self.market_data.recent_trades(asset, window_secs=self.risk_params['whale_window'])
        sizes = np.array([t.size for t in trades])
        # detect clusters of large trades
        if np.count_nonzero(sizes > self.risk_params['whale_size']) >= self.risk_params['whale_count']:
            return True
        # on-chain detection fallback
        return self.market_data.onchain_whale(asset, self.risk_params['onchain_size'])

    def enter_momentum_trade(self, asset):
        if self.detect_whale_momentum(asset):
            size = self.risk_params['momentum_size']
            price = self.market_data.get_spot(asset)
            order = self.market_data.place_market_order(asset, 'buy', size)
            # attach trailing stop
            self.market_data.place_trailing_stop(asset, size, self.risk_params['trail_pct'])
            return order
        return None

    # --- Context signal ingestion ---
    def update_context_signals(self, whale, swan, fusion):
        self.context_signals['whale_alert'] = whale
        self.context_signals['black_swan_risk'] = swan
        self.context_signals['fusion_confidence'] = fusion

    # --- Strategy selection & orchestration ---
    def _select_strategy(self):
        weights = np.array(list(self.risk_params['strategy_weights'].values()), dtype=np.float64)
        probs = softmax_numba(weights)
        keys = list(self.risk_params['strategy_weights'].keys())
        return random.choices(keys, weights=probs.tolist(), k=1)[0]

    def manage_hedge(self, positions, features):
        # integrate QAR context
        if self.qar:
            self.qar.context_signals = self.context_signals
            probs = self.qar.make_decision()
            self.risk_params['strategy_weights'] = probs
        # pick and execute
        strat = self._select_strategy()
        return {
            'options': lambda: self.hedge_with_options(features['asset'], features['target_delta'], features['expiry']),
            'pair': lambda: self.hedge_with_pair_trading(features['asset'], features['hedge_asset']),
            'cross': lambda: self.hedge_with_cross_short(features['asset'], features['hedge_asset']),
            'grid': lambda: self.grid_hedge(features['asset']),
            'momentum': lambda: self.enter_momentum_trade(features['asset']),
        }.get(strat, lambda: None)()


    def predict(self, market_features: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Predict the best action distribution based on current weights,
        potentially enhanced by quantum processing of market features.

        Args:
            market_features: Optional market features for context-aware prediction

        Returns:
            Probability distribution over experts representing the algorithm's
            recommendation for the next action
        """
        # If quantum is available and features is not None, use the quantum circuit for prediction
        if self.quantum_available and self.processing_mode != ProcessingMode.CLASSICAL and market_features is not None and self.hw_accelerator is not None:
            try:
                # Ensure quantum components are initialized
                if not self._quantum_components_initialized:
                    self._initialize_quantum_components()

                # Check again if quantum is available after initialization attempt
                if self.processing_mode != ProcessingMode.CLASSICAL and self.quantum_available and self.device is not None:
                    # Process weights with quantum enhancement using the internal method
                    return self._process_with_quantum(market_features)
                else:
                    self.logger.debug("Quantum prediction not enabled or available after initialization. Using classical weights.")
                    return self.weights.copy()

                if market_features is not None:
                    # Use quantum circuit to get enhanced prediction
                    quantum_probs = self.quantum_circuit(self.weights, market_features)
                    
                    # Convert to expert weights
                    prediction = np.zeros_like(self.weights)
                    for i in range(min(len(quantum_probs), self.num_experts)):
                        prediction[i] = quantum_probs[i]
                        
                    # Handle any remaining experts
                    if self.num_experts > len(quantum_probs):
                        remaining_weight = 1.0 - np.sum(prediction)
                        remaining_count = self.num_experts - len(quantum_probs)
                        prediction[len(quantum_probs):] = remaining_weight / remaining_count
            
                # Normalize
                prediction = prediction / np.sum(prediction)
                return prediction
        
            except Exception as e:
                self.logger.warning(f"Quantum prediction failed: {e}. Falling back to classical weights.")
                # Fallback to classical weights if quantum prediction fails
                return self.weights.copy()

        else:
            # Without quantum or features, return current classical weight distribution
            return self.weights.copy()

    def select_action(self, market_features: Optional[np.ndarray] = None) -> int:
        """
        Select a single expert/action based on the current probability distribution.

        Args:
            market_features: Optional market features for context-aware selection

        Returns:
            Index of the selected expert/action
        """
        probabilities = self.predict(market_features)

        # Handle potential issues with probabilities (e.g., sum not exactly 1 due to float errors)
        probabilities = np.array(probabilities, dtype=np.float64) # Ensure float64 for np.random.choice
        probabilities = np.clip(probabilities, 0, 1) # Clip to [0, 1]
        prob_sum = np.sum(probabilities)
        if prob_sum > 0:
             probabilities = probabilities / prob_sum # Re-normalize
        else:
             # If sum is zero, assign equal probability to all experts
             probabilities = np.ones_like(probabilities) / self.num_experts
             self.logger.warning("Probabilities sum to zero, using uniform distribution for action selection.")

        # Select action based on the probability distribution
        return np.random.choice(self.num_experts, p=probabilities)
        
    def calculate_optimal_rewards(
        self, 
        expert_signals: np.ndarray,
        market_outcome: float,
        market_features: Optional[Dict[str, float]] = None,
        time_horizon: str = "short"
    ) -> np.ndarray:
        """
        Calculate sophisticated rewards for experts based on their signals and market outcomes.
        
        Args:
            expert_signals: Array of signals/predictions from each expert (-1 to 1 range)
            market_outcome: Actual market return or outcome
            market_features: Optional market metrics for context-sensitive rewards
            time_horizon: Time horizon of prediction ("short", "medium", or "long")
            
        Returns:
            Array of calculated rewards for each expert
        """
        # Initialize rewards array
        rewards = np.zeros(self.num_experts)
        
        # Base directional accuracy reward
        # Higher reward for correctly predicting direction, penalty for wrong direction
        directional_accuracy = np.sign(expert_signals) * np.sign(market_outcome)
        
        # Adjust base reward scale based on time horizon
        if time_horizon == "short":
            base_scale = 1.0
        elif time_horizon == "medium":
            base_scale = 1.5  # Medium-term predictions are harder, reward more
        else:  # "long"
            base_scale = 2.0  # Long-term predictions are hardest, reward most
        
        # Base reward component: directional accuracy * magnitude accuracy
        magnitude_factor = 1.0 - np.minimum(1.0, np.abs(np.abs(expert_signals) - np.abs(market_outcome)))
        
        # Combine directional and magnitude components
        base_rewards = base_scale * (directional_accuracy * 0.6 + magnitude_factor * 0.4)
        
        # Market context adjustments
        context_factor = np.ones(self.num_experts)
        
        if market_features is not None:
            # In high volatility, reward cautious experts (lower signal magnitude) more
            if 'volatility' in market_features:
                volatility = market_features['volatility']
                caution_factor = 1.0 + volatility * (1.0 - np.abs(expert_signals))
                context_factor *= caution_factor
            
            # In trending markets, reward trend-following experts more
            if 'trend_strength' in market_features and 'trend_direction' in market_features:
                trend_strength = market_features['trend_strength']
                trend_direction = market_features['trend_direction']
                trend_alignment = np.sign(expert_signals) * trend_direction
                trend_factor = 1.0 + trend_strength * trend_alignment * 0.5
                context_factor *= trend_factor
        
        # Risk-adjusted reward component
        # Experts that signaled higher confidence (larger magnitude) get higher reward when right
        # but larger penalty when wrong
        risk_factor = 1.0 + 0.5 * np.abs(expert_signals) * np.sign(directional_accuracy)
        
        # Combine all factors into final rewards
        rewards = base_rewards * context_factor * risk_factor
        
        # Scale rewards to reasonable range
        rewards = np.clip(rewards, -2.0, 2.0)
        
        return rewards
    
    def visualize_weights(self, expert_names: Optional[List[str]] = None, figsize: Tuple[int, int] = (10, 6), save_path: Optional[str] = None):
        """
        Visualize the evolution of expert weights over time.
        
        Args:
            expert_names: Optional list of expert names for the legend
            figsize: Figure size as (width, height)
            save_path: Path to save the figure instead of showing it
        """
        plt.figure(figsize=figsize)
        
        # Create array of weights history
        weights_array = np.array(self.weights_history)
        
        # Create x-axis (iterations)
        iterations = np.arange(len(self.weights_history))
        
        # Plot each expert's weight over time
        for i in range(self.num_experts):
            if expert_names and i < len(expert_names):
                label = expert_names[i]
            else:
                label = f"Expert {i+1}"
            plt.plot(iterations, weights_array[:, i], label=label)
        
        plt.xlabel("Iteration")
        plt.ylabel("Weight")
        plt.title("Evolution of Expert Weights")
        plt.legend(loc="best")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Always save the figure, use a default name if save_path is not provided
        actual_save_path = save_path or "quantum_hedge_weights.png"
        plt.savefig(actual_save_path)
        plt.close()
        logger.info(f"Weights visualization saved to {actual_save_path}")
    
    def visualize_regret(self, figsize: Tuple[int, int] = (10, 6), save_path: Optional[str] = None):
        """
        Visualize the regret over time.
        
        Args:
            figsize: Figure size as (width, height)
            save_path: Path to save the figure instead of showing it
        """
        plt.figure(figsize=figsize)
        
        # Create x-axis (iterations)
        iterations = np.arange(len(self.regret_history))
        
        # Plot regret over time
        plt.subplot(2, 1, 1)
        plt.plot(iterations, self.regret_history, label="Per-iteration Regret")
        plt.xlabel("Iteration")
        plt.ylabel("Regret")
        plt.title("Per-iteration Regret")
        plt.grid(True, alpha=0.3)
        
        # Plot cumulative regret
        plt.subplot(2, 1, 2)
        cumulative_regret = np.cumsum(self.regret_history)
        plt.plot(iterations, cumulative_regret, label="Cumulative Regret", color="orange")
        plt.xlabel("Iteration")
        plt.ylabel("Cumulative Regret")
        plt.title("Cumulative Regret")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Always save the figure, use a default name if save_path is not provided
        actual_save_path = save_path or "quantum_hedge_regret.png"
        plt.savefig(actual_save_path)
        plt.close()
        logger.info(f"Regret visualization saved to {actual_save_path}")
    
    def visualize_performance(self, figsize: Tuple[int, int] = (10, 8), save_path: Optional[str] = None):
        """
        Visualize the performance of experts over time.
        
        Args:
            figsize: Figure size as (width, height)
            save_path: Path to save the figure instead of showing it
        """
        if len(self.rewards_history) < 2:
            logger.warning("Not enough data to visualize performance")
            return
            
        plt.figure(figsize=figsize)
        
        # Create array of rewards history
        rewards_array = np.array(self.rewards_history)
        
        # Create x-axis (iterations)
        iterations = np.arange(len(self.rewards_history))
        
        # Plot each expert's reward over time
        plt.subplot(3, 1, 1)
        for i in range(self.num_experts):
            plt.plot(iterations, rewards_array[:, i], label=f"Expert {i+1}")
        plt.xlabel("Iteration")
        plt.ylabel("Reward")
        plt.title("Expert Rewards Over Time")
        plt.legend(loc="best")
        plt.grid(True, alpha=0.3)
        
        # Plot cumulative rewards for each expert
        plt.subplot(3, 1, 2)
        cumulative_rewards = np.cumsum(rewards_array, axis=0)
        for i in range(self.num_experts):
            plt.plot(iterations, cumulative_rewards[:, i], label=f"Expert {i+1}")
        plt.xlabel("Iteration")
        plt.ylabel("Cumulative Reward")
        plt.title("Cumulative Expert Rewards")
        plt.legend(loc="best")
        plt.grid(True, alpha=0.3)
        
        # Plot algorithm's weighted reward vs best expert
        plt.subplot(3, 1, 3)
        weighted_rewards = np.sum(np.array(self.weights_history[:-1]) * rewards_array, axis=1)
        best_expert_rewards = np.max(rewards_array, axis=1)
        plt.plot(iterations, weighted_rewards, label="Algorithm (Weighted)", linewidth=2)
        plt.plot(iterations, best_expert_rewards, label="Best Expert", linestyle='--')
        plt.xlabel("Iteration")
        plt.ylabel("Reward")
        plt.title("Algorithm vs Best Expert Performance")
        plt.legend(loc="best")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Always save the figure, use a default name if save_path is not provided
        actual_save_path = save_path or "quantum_hedge_performance.png"
        plt.savefig(actual_save_path)
        plt.close()
        logger.info(f"Performance visualization saved to {actual_save_path}")  
        
    def save_state(self, filepath: str):
        """
        Save the full adaptive state of the QuantumHedgeAlgorithm, including all relevant variables for persistent learning.
        """
        def safe_tolist(val):
            # Only call .tolist() if it's a numpy array, else return as-is
            if hasattr(val, 'tolist'):
                return val.tolist()
            return val

        state = {
            'weights': safe_tolist(self.weights) if hasattr(self, 'weights') else None,
            'regret': safe_tolist(self.regret) if hasattr(self, 'regret') else None,
            'learning_rate': self.learning_rate if hasattr(self, 'learning_rate') else None,
            'meta_learning_rate': self.meta_learning_rate if hasattr(self, 'meta_learning_rate') else None,
            'history': self.history if hasattr(self, 'history') else None,
            'iterations': self.iterations if hasattr(self, 'iterations') else None,
            'options_model': None,  # Not serializable, placeholder
            'risk_params': self.risk_params if hasattr(self, 'risk_params') else None,
            'expert_performance': safe_tolist(self.expert_performance) if hasattr(self, 'expert_performance') else None,
            'expert_consistency': safe_tolist(self.expert_consistency) if hasattr(self, 'expert_consistency') else None,
            'weights_history': [safe_tolist(w) for w in self.weights_history] if hasattr(self, 'weights_history') else None,
            'rewards_history': [safe_tolist(r) for r in self.rewards_history] if hasattr(self, 'rewards_history') else None,
            'regret_history': self.regret_history if hasattr(self, 'regret_history') else None,
            'cumulative_regret': self.cumulative_regret if hasattr(self, 'cumulative_regret') else None,
            'processing_mode': str(self.processing_mode) if hasattr(self, 'processing_mode') else None,
            'quantum_available': self.quantum_available if hasattr(self, 'quantum_available') else None,
            'device': None,  # Not serializable, placeholder
            'use_quantum_preference': self.use_quantum_preference if hasattr(self, 'use_quantum_preference') else None,
            'market_adaptive_learning': self.market_adaptive_learning if hasattr(self, 'market_adaptive_learning') else None,
            'num_experts': self.num_experts if hasattr(self, 'num_experts') else None,
            'num_qubits': self.num_qubits if hasattr(self, 'num_qubits') else None,
            'feature_dim': self.feature_dim if hasattr(self, 'feature_dim') else None,
            'regret_type': self.regret_type if hasattr(self, 'regret_type') else None,
            'base_learning_rate': self.base_learning_rate if hasattr(self, 'base_learning_rate') else None,
            'weight_decay': self.weight_decay if hasattr(self, 'weight_decay') else None,
            'min_weight': self.min_weight if hasattr(self, 'min_weight') else None,
            'quantum_enhancement': self.quantum_enhancement if hasattr(self, 'quantum_enhancement') else None,
            'market_data': None,  # Not serializable, placeholder
            'context_signals': self.context_signals if hasattr(self, 'context_signals') else None,
            'qar': None,  # Not serializable, placeholder
            '_last_grid': self._last_grid if hasattr(self, '_last_grid') else None,
            '_quantum_components_initialized': self._quantum_components_initialized if hasattr(self, '_quantum_components_initialized') else None,
            'hw_accelerator': None,  # Not serializable, placeholder
        }
        try:
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=4, sort_keys=True)
            logger.info(f"QHA state saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save QHA state to {filepath}: {e}")

    def load_state(self, filepath: str):
        """
        Load the full adaptive state of the QuantumHedgeAlgorithm, restoring all relevant variables for persistent learning.
        """
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            if 'weights' in state and state['weights'] is not None:
                self.weights = np.array(state['weights'])
            if 'regret' in state and state['regret'] is not None:
                self.regret = np.array(state['regret'])
            if 'learning_rate' in state and state['learning_rate'] is not None:
                self.learning_rate = state['learning_rate']
            if 'meta_learning_rate' in state and state['meta_learning_rate'] is not None:
                self.meta_learning_rate = state['meta_learning_rate']
            if 'history' in state and state['history'] is not None:
                self.history = state['history']
            if 'iterations' in state and state['iterations'] is not None:
                self.iterations = state['iterations']
            if 'risk_params' in state and state['risk_params'] is not None:
                self.risk_params = state['risk_params']
            if 'weights_history' in state and state['weights_history'] is not None:
                self.weights_history = [np.array(w) for w in state['weights_history']]
            if 'rewards_history' in state and state['rewards_history'] is not None:
                self.rewards_history = [np.array(r) for r in state['rewards_history']]
            if 'regret_history' in state and state['regret_history'] is not None:
                self.regret_history = state['regret_history']
            if 'cumulative_regret' in state and state['cumulative_regret'] is not None:
                self.cumulative_regret = state['cumulative_regret']
            if 'expert_performance' in state and state['expert_performance'] is not None:
                self.expert_performance = np.array(state['expert_performance'])
            if 'expert_consistency' in state and state['expert_consistency'] is not None:
                self.expert_consistency = np.array(state['expert_consistency'])
            if 'processing_mode' in state and state['processing_mode'] is not None:
                self.processing_mode = state['processing_mode']
            if 'quantum_available' in state and state['quantum_available'] is not None:
                self.quantum_available = state['quantum_available']
            if 'use_quantum_preference' in state and state['use_quantum_preference'] is not None:
                self.use_quantum_preference = state['use_quantum_preference']
            if 'market_adaptive_learning' in state and state['market_adaptive_learning'] is not None:
                self.market_adaptive_learning = state['market_adaptive_learning']
            if 'base_learning_rate' in state and state['base_learning_rate'] is not None:
                self.base_learning_rate = state['base_learning_rate']
            if 'weight_decay' in state and state['weight_decay'] is not None:
                self.weight_decay = state['weight_decay']
            if 'min_weight' in state and state['min_weight'] is not None:
                self.min_weight = state['min_weight']
            if 'regret_type' in state and state['regret_type'] is not None:
                self.regret_type = state['regret_type']
            if 'num_experts' in state and state['num_experts'] is not None:
                self.num_experts = state['num_experts']
            if 'feature_dim' in state and state['feature_dim'] is not None:
                self.feature_dim = state['feature_dim']
            if 'num_qubits' in state and state['num_qubits'] is not None:
                self.num_qubits = state['num_qubits']
            if 'quantum_enhancement' in state and state['quantum_enhancement'] is not None:
                self.quantum_enhancement = state['quantum_enhancement']
            if 'context_signals' in state and state['context_signals'] is not None:
                self.context_signals = state['context_signals']
            if '_last_grid' in state and state['_last_grid'] is not None:
                self._last_grid = state['_last_grid']
            # Non-serializable objects (qar, device, market_data, hw_accelerator, options_model) must be re-initialized or set externally.
            logger.info(f"QHA state loaded from {filepath}")
        except Exception as e:
            logger.error(f"Failed to load QHA state from {filepath}: {e}")

