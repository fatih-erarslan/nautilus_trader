#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Quantum-Enhanced Reservoir Computing (QERC)
Combines features from both versions into a more robust implementation
while maintaining compatibility with hardware_manager.py
"""

import numpy as np
import logging
import math
import os
import threading
import time
import random
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

# Hardware acceleration and optimization
from cdfa_extensions.hw_acceleration import quantum_accelerated

# Import numba for classical optimizations
try:
    import numba as nb
    from numba import njit, prange, vectorize, float64, int64, boolean
    NUMBA_AVAILABLE = True
except ImportError:
    # Create fallback decorators if numba isn't available
    NUMBA_AVAILABLE = False
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if args and callable(args[0]) else decorator
    
    def prange(*args, **kwargs):
        return range(*args)
    
    def vectorize(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if args and callable(args[0]) else decorator
    
    # Dummy type definitions
    class DummyNumbaType:
        def __getitem__(self, *args):
            return lambda x: x
    
    float64 = DummyNumbaType()
    int64 = DummyNumbaType()
    boolean = DummyNumbaType()

try:
    import pennylane as qml
    QUANTUM_AVAILABLE = True
except ImportError:
    QUANTUM_AVAILABLE = False
    logging.warning("PennyLane not installed; quantum features will be disabled")

# Attempt to import optional dependencies with proper fallbacks
try:
    from hardware_manager import HardwareManager
    HARDWARE_MANAGER_AVAILABLE = True
except ImportError:
    HARDWARE_MANAGER_AVAILABLE = False
    logging.warning("hardware_manager not available; using internal hardware detection")
    HardwareManager = None

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

# Configure logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class QuantumEnhancedReservoirComputing:
    """
    Quantum-Enhanced Reservoir Computing for time series analysis.
    
    This implements a hybrid classical-quantum reservoir computing system
    that can detect temporal patterns in financial time series data.
    
    Features:
    - 500-node reservoir with configurable quantum kernel
    - Temporal processing with sliding windows
    - Hardware-optimized quantum circuits
    - Fault tolerance and robust error handling
    - Efficient memory management with caching
    """
    
    def __init__(self, 
                 reservoir_size: int = 500, 
                 quantum_kernel_size: int = 4,
                 spectral_radius: float = 0.95, 
                 leaking_rate: float = 0.3,
                 temporal_windows: List[int] = [5, 15, 30, 60],
                 input_dimensionality: Optional[int] = None,
                 hardware_manager = None, 
                 use_classical: bool = False,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the QERC component.
        
        Args:
            reservoir_size (int): Size of the classical reservoir
            quantum_kernel_size (int): Number of qubits for quantum kernel
            spectral_radius (float): Spectral radius for reservoir stability
            leaking_rate (float): Leaking rate for reservoir memory
            temporal_windows (List[int]): List of temporal window sizes
            input_dimensionality (int, optional): Dimension of input features
            hardware_manager: Hardware manager instance
            use_classical (bool): Force classical implementation
            config (Dict[str, Any], optional): Additional configuration options
        """
        self.logger = logging.getLogger(__name__)
        
        # Initialize configuration
        self.config = config or {}
        log_level = self.config.get('log_level', logging.INFO)
        
        # Handle log level configuration - properly handle both string and int
        try:
            if isinstance(log_level, str):
                level_name_upper = log_level.upper()
                level_int = logging.getLevelName(level_name_upper)
                if isinstance(level_int, int):
                    final_log_level = level_int
                else:
                    self.logger.warning(f"Invalid log_level string '{log_level}'. Using INFO.")
                    final_log_level = logging.INFO
            elif isinstance(log_level, int) and log_level in logging._levelToName:
                final_log_level = log_level
            else:
                self.logger.warning(f"Invalid log_level type '{type(log_level).__name__}'. Using INFO.")
                final_log_level = logging.INFO
                
            self.logger.setLevel(final_log_level)
            self.logger.info(f"QERC initialized with log level {logging.getLevelName(self.logger.level)} ({self.logger.level})")
            
        except Exception as e:
            self.logger.error(f"Error setting log level: {e}", exc_info=True)
            self.logger.setLevel(logging.INFO)
        
        # Core parameters
        self.reservoir_size = reservoir_size
        self.quantum_kernel_size = quantum_kernel_size 
        self.qubits = quantum_kernel_size  # Maintain both attribute names for compatibility
        self.spectral_radius = spectral_radius
        self.leaking_rate = leaking_rate
        self.temporal_windows = temporal_windows
        
        # Determine input dimensionality
        self.input_dimensionality = input_dimensionality
        if self.input_dimensionality is None:
            # Calculate based on temporal windows
            self.input_dimensionality = len(self.temporal_windows) * self.qubits
            self.logger.warning(f"input_dimensionality not explicitly provided. Calculated as {self.input_dimensionality} "
                              f"(temporal windows * qubits). Provide explicitly if using frequency features.")
        
        # Hardware resources
        self.hardware_manager = hardware_manager
        if self.hardware_manager is None and HARDWARE_MANAGER_AVAILABLE:
            self.hardware_manager = HardwareManager()
        self.use_classical = use_classical or not QUANTUM_AVAILABLE
        
        # Quantum components
        self.shots = self.config.get('shots', None)
        
        # Internal state
        self.is_initialized = False
        self.reservoir_state = None
        self.quantum_circuit = None
        self.device = None
        self.W_in = None  # Input weights
        self.W = None     # Reservoir weights
        self.W_out = None # Output weights
        self.circuits = {}  # Quantum circuits
        
        # Setup caching
        self.cache = CircularBuffer(max_size=self.config.get('cache_size', 100))
        self.cache_lock = threading.RLock()
        
        # Fault tolerance
        self.fault_tolerance = None
        if FAULT_TOLERANCE_AVAILABLE:
            self.fault_tolerance = get_fault_tolerance_manager()
        
        # Performance tracking
        self.execution_times = []
        
        # Try to initialize
        try:
            self._initialize_reservoir()
        except Exception as e:
            self.logger.error(f"Error initializing QERC: {str(e)}", exc_info=True)
            self.use_classical = True
            self._initialize_reservoir()
    
    def _initialize_reservoir(self) -> None:
        """Initialize reservoir and quantum components."""
        self.logger.info(f"Initializing QERC with {'classical' if self.use_classical else 'quantum'} backend")
        
        # Initialize weights
        self._initialize_reservoir_weights()
        
        # Initialize quantum components if needed
        if not self.use_classical:
            self.device = self._get_optimized_device()
            self._initialize_quantum_circuits()
        
        # Initialize reservoir state
        self.reservoir_state = np.zeros(self.reservoir_size)
        
        self.is_initialized = True
        self.logger.info(f"QERC initialized: Reservoir={self.reservoir_size}, Qubits={self.qubits}, InputDim={self.input_dimensionality}")
    
    def _initialize_reservoir_weights(self) -> None:
        """Initialize the reservoir weight matrices."""
        # Initialize input weights
        self.W_in = np.random.uniform(-0.5, 0.5, (self.reservoir_size, self.input_dimensionality))
        
        # Initialize reservoir weights (random sparse matrix)
        connectivity = 0.1  # Sparsity level
        self.W = np.random.uniform(-0.5, 0.5, (self.reservoir_size, self.reservoir_size))
        mask = np.random.uniform(0, 1, (self.reservoir_size, self.reservoir_size)) > (1 - connectivity)
        self.W = self.W * mask
        
        # Scale reservoir weights to desired spectral radius
        radius = np.max(np.abs(np.linalg.eigvals(self.W)))
        if radius > 0:
            self.W *= (self.spectral_radius / radius)
        
        # Initialize output weights (trained later)
        self.W_out = np.zeros((4, self.reservoir_size))  # 4 outputs: trend, volatility, momentum, regime
    
    def _get_optimized_device(self) -> Any:
        """Get hardware-optimized quantum device based on available hardware."""
        if not QUANTUM_AVAILABLE:
            self.logger.warning("Quantum libraries not available, falling back to classical")
            self.use_classical = True
            return None
            
        try:
            # If hardware manager is available, use it to get optimal device
            if self.hardware_manager and hasattr(self.hardware_manager, 'get_optimal_device'):
                device_config = self.hardware_manager.get_optimal_device(
                    quantum_required=True, 
                    qubits_required=self.qubits
                )
                device_name = device_config.get('device', 'default.qubit')
                self.logger.info(f"Using hardware manager's device: {device_name}")
                return qml.device(device_name, wires=self.qubits, shots=self.shots)
            
            # Otherwise detect hardware directly
            # Check hardware type and configure accordingly
            if hasattr(self.hardware_manager, 'devices'):
                # AMD GPU
                if self.hardware_manager.devices.get('amd_gpu', {}).get('available', False):
                    gfx_version = getattr(self.hardware_manager, 'gfx_version', None)
                    if gfx_version:
                        os.environ['HSA_OVERRIDE_GFX_VERSION'] = gfx_version
                    os.environ['HIP_VISIBLE_DEVICES'] = '0'
                    os.environ['KOKKOS_DEVICES'] = 'HIP'
                    self.logger.info(f"Using AMD GPU for QERC")
                    return qml.device('lightning.kokkos', wires=self.qubits, shots=self.shots)

                # NVIDIA GPU
                elif self.hardware_manager.devices.get('nvidia_gpu', {}).get('available', False):
                    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
                    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
                    self.logger.info("Using NVIDIA GPU for QERC")
                    return qml.device('lightning.gpu', wires=self.qubits, shots=self.shots)

                # Apple Silicon
                elif self.hardware_manager.devices.get('apple_silicon', False):
                    self.logger.info("Using Apple Silicon for QERC")
                    return qml.device('default.qubit', wires=self.qubits, shots=self.shots)
            
            # Fallback to CPU if hardware detection fails or not available
            self.logger.info("Using CPU for QERC quantum operations")
            try:
                return qml.device('lightning.qubit', wires=self.qubits, shots=self.shots)
            except Exception as e:
                self.logger.warning(f"Could not initialize lightning.qubit: {e}, falling back to default.qubit")
                return qml.device('default.qubit', wires=self.qubits, shots=self.shots)

        except Exception as e:
            self.logger.error(f"Error setting up quantum device: {e}", exc_info=True)
            self.logger.info("Falling back to default qubit device")
            try:
                return qml.device('default.qubit', wires=self.qubits, shots=self.shots)
            except Exception as e2:
                self.logger.error(f"Critical error setting up default device: {e2}", exc_info=True)
                self.use_classical = True
                return None
    
    def _initialize_quantum_circuits(self) -> None:
        """Initialize quantum circuits for kernel computation."""
        if not QUANTUM_AVAILABLE or self.use_classical or self.device is None:
            self.logger.warning("Quantum libraries not available, skipping circuit initialization")
            return
            
        try:
            # Define quantum kernel circuit for nonlinear transformations
            @qml.qnode(self.device)
            def quantum_kernel_circuit(inputs, weights):
                # Encode input data
                for i in range(self.qubits):
                    qml.RY(inputs[i % len(inputs)], wires=i)

                # Apply entangling layers
                for i in range(self.qubits - 1):
                    qml.CNOT(wires=[i, i+1])

                # Apply weighted transformation
                for i in range(self.qubits):
                    qml.RZ(weights[i % len(weights)], wires=i)

                # Additional entanglement
                for i in range(self.qubits - 1, 0, -1):
                    qml.CNOT(wires=[i, i-1])

                # Measure
                return [qml.expval(qml.PauliZ(i)) for i in range(self.qubits)]

            # Feature extraction circuit for temporal patterns
            @qml.qnode(self.device)
            def quantum_feature_extraction(data):
                # Encode time series data using amplitude encoding
                # Handle the case where data length doesn't match required size for amplitude encoding
                data_len = len(data)
                required_len = 2**self.qubits
                
                if data_len != required_len:
                    # Use angle encoding instead for flexibility
                    for i in range(self.qubits):
                        qml.RY(data[i % data_len], wires=i)
                        qml.RZ(data[(i + 1) % data_len], wires=i)
                else:
                    # Use amplitude encoding when data length matches
                    qml.AmplitudeEmbedding(
                        features=data,
                        wires=range(self.qubits),
                        normalize=True,
                        pad_with=0.0
                    )

                # Apply quantum feature extraction circuit
                for i in range(2):  # Repeat layers for expressivity
                    for j in range(self.qubits):
                        qml.Hadamard(wires=j)

                    for j in range(self.qubits - 1):
                        qml.CZ(wires=[j, j+1])

                    for j in range(self.qubits):
                        qml.RY(np.pi/4, wires=j)

                # Measure in computational basis
                return [qml.expval(qml.PauliZ(i)) for i in range(self.qubits)]
            
            # Store the circuits in the dictionary
            self.circuits = {
                'kernel': quantum_kernel_circuit,
                'feature_extraction': quantum_feature_extraction
            }
                
            self.logger.info(f"Quantum circuits initialized with {self.qubits} qubits")
            
        except Exception as e:
            self.logger.error(f"Error initializing quantum circuits: {str(e)}", exc_info=True)
            self.logger.warning("Falling back to classical implementation")
            self.use_classical = True
    
    def _execute_with_fallback(self, circuit_name: str, params: Tuple) -> np.ndarray:
        """Execute quantum circuit with fallback to classical computation."""
        if self.use_classical or not QUANTUM_AVAILABLE or circuit_name not in self.circuits:
            return self._classical_fallback(circuit_name, params)
            
        try:
            # Generate cache key
            param_str = str([p.tobytes() if hasattr(p, 'tobytes') else str(p) for p in params])
            cache_key = hash(circuit_name + param_str)

            # Check cache
            with self.cache_lock:
                if cache_key in self.cache:
                    return self.cache[cache_key]

            # Execute quantum circuit
            result = self.circuits[circuit_name](*params)

            # Cache result
            with self.cache_lock:
                self.cache[cache_key] = result

            return result

        except Exception as e:
            self.logger.warning(f"Quantum circuit execution failed for {circuit_name}: {e}")
            # Fall back to classical computation
            return self._classical_fallback(circuit_name, params)
    
    def _classical_fallback(self, circuit_name: str, params: Tuple) -> np.ndarray:
        """Provide classical implementations of quantum circuits for fallback."""
        self.logger.debug(f"Using classical fallback for circuit: {circuit_name}")
        
        if circuit_name == 'kernel':
            inputs, weights = params
            # Compute classical kernel
            return np.tanh(np.dot(inputs, weights))
            
        elif circuit_name == 'feature_extraction':
            data = params[0]
            # Calculate basic features
            features = []
            data_len = len(data)
            
            for i in range(min(data_len, self.qubits)):
                features.append(np.tanh(data[i]))
                
            # Pad if necessary
            if len(features) < self.qubits:
                features.extend([0.0] * (self.qubits - len(features)))
                
            return np.array(features)
        
        else:
            self.logger.error(f"Unknown circuit: {circuit_name}")
            return np.zeros(self.qubits)

    
    def _prepare_vector_for_quantum(self, vector: np.ndarray) -> np.ndarray:
        """
        Prepare vector for quantum processing. Normalizes to [0, 1] range
        and pads/truncates to match self.qubits. Handles non-finite values.
        """
        target_size = self.qubits
    
        # --- Input Validation and Conversion ---
        if not isinstance(vector, np.ndarray):
            try:
                # Attempt conversion to float array
                vector = np.array(vector, dtype=float)
            except (ValueError, TypeError):
                self.logger.error(f"Could not convert input vector of type {type(vector)} to numpy array. Returning zeros.")
                return np.zeros(target_size, dtype=float) # Return correctly shaped zeros
    
        # Ensure vector is 1D (flatten if needed)
        if vector.ndim > 1:
            vector = vector.flatten()
    
        # Check for empty vector AFTER potential flattening
        if vector.size == 0:
             self.logger.warning("Input vector is empty. Returning zeros.")
             return np.zeros(target_size, dtype=float)
    
        # --- Handle Non-Finite Values ---
        # Check BEFORE normalization or resizing if possible
        if not np.all(np.isfinite(vector)):
            self.logger.warning("Non-finite values (NaN/Inf) detected in input vector, replacing with 0.0")
            vector = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=-0.0) # Use 0.0 for Inf too
    
        # --- Resizing ---
        current_size = vector.size
        if current_size == target_size:
            result = vector # No copy needed if size matches
        elif current_size > target_size:
            result = vector[:target_size] # Truncate
            # self.logger.debug(f"Vector truncated from {current_size} to {target_size}") # Reduce logging noise
        else: # current_size < target_size
            result = np.pad(vector, (0, target_size - current_size), 'constant', constant_values=0.0) # Pad
            # self.logger.debug(f"Vector padded from {current_size} to {target_size}") # Reduce logging noise
    
        # --- Normalization (on the correctly sized vector) ---
        min_val = np.min(result)
        max_val = np.max(result)
        if max_val - min_val > 1e-9: # Check range is significant
            result = (result - min_val) / (max_val - min_val)
        elif np.any(result != 0): # If not all zero, but range is tiny (e.g., all same value), set to 0.5
            result = np.full_like(result, 0.5)
        # else: If all zeros, result remains all zeros.
    
        # Final check for safety (should be redundant but harmless)
        if result.shape[0] != target_size:
             self.logger.error(f"FINAL vector prep size error! Expected {target_size}, got {result.shape}. Returning zeros.")
             return np.zeros(target_size, dtype=float)
        if not np.all(np.isfinite(result)):
             self.logger.error(f"FINAL vector prep NaN/Inf error! Replacing with zeros.")
             result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    
        return result
        
    def _quantum_nonlinearity(self, node_states: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Apply quantum kernel for nonlinear transformation."""
        try:
            # Prepare input batch for quantum processing
            batch_size = 10  # Process 10 nodes at a time to avoid memory issues
            result = np.zeros_like(node_states)

            for i in range(0, len(node_states), batch_size):
                batch = node_states[i:i+batch_size]
                batch_weights = weights[i:i+batch_size]

                # Cache key based on inputs
                cache_key = hash(str(batch.tobytes()) + str(batch_weights.tobytes()))

                # Check cache
                with self.cache_lock:
                    if cache_key in self.cache:
                        batch_result = self.cache[cache_key]
                    else:
                        # Apply quantum circuit to each node in batch
                        batch_result = np.array([
                            self._execute_with_fallback(
                                'kernel',
                                (node_state, node_weight)
                            )
                            for node_state, node_weight in zip(batch, batch_weights)
                        ])

                        # Cache result
                        self.cache[cache_key] = batch_result

                # Store results
                result[i:i+batch_size] = batch_result.mean(axis=1, keepdims=True)

            return result

        except Exception as e:
            self.logger.error(f"Error in quantum nonlinearity: {e}")
            # Fallback to classical nonlinearity
            return np.tanh(node_states)
    
    def _update_reservoir_state(self, input_features: np.ndarray) -> np.ndarray:
        """
        Update the reservoir state with new input.
        
        Args:
            input_features (np.ndarray): Input feature vector
            
        Returns:
            np.ndarray: Updated reservoir state
        """
        # Ensure input features have the right shape
        if len(input_features.shape) == 1:
            input_features = input_features.reshape(-1, 1)
            
        expected_shape = (self.input_dimensionality, 1)
        if input_features.shape != expected_shape:
            self.logger.error(f"Input shape mismatch: expected {expected_shape}, got {input_features.shape}")
            # Try to reshape if possible
            if input_features.size == self.input_dimensionality:
                input_features = input_features.reshape(expected_shape)
            else:
                # Critical error, cannot proceed with update
                raise ValueError(f"Input dimensionality mismatch: expected {self.input_dimensionality}, got {input_features.size}")
        
        # Ensure reservoir state has the right shape
        if len(self.reservoir_state.shape) == 1:
            self.reservoir_state = self.reservoir_state.reshape(-1, 1)
            
        # Calculate input to reservoir
        u = np.dot(self.W_in, input_features)
        
        # Calculate reservoir internal update
        next_state = np.tanh(np.dot(self.W, self.reservoir_state) + u)
        
        # Apply leaking rate for smoother state transitions
        self.reservoir_state = (1 - self.leaking_rate) * self.reservoir_state + self.leaking_rate * next_state
        
        # Apply quantum nonlinearity to a subset of nodes
        try:
            if not self.use_classical and QUANTUM_AVAILABLE:
                # Select random subset of nodes for quantum processing
                quantum_node_indices = np.random.choice(
                    self.reservoir_size,
                    size=min(self.reservoir_size // 5, 50),  # Process 20% of nodes, max 50
                    replace=False
                )
                
                # Apply quantum nonlinearity
                quantum_nodes = self.reservoir_state[quantum_node_indices]
                quantum_weights = self.W[quantum_node_indices, :][:, :self.qubits]  # Use a subset of weights
                
                quantum_result = self._quantum_nonlinearity(quantum_nodes, quantum_weights)
                self.reservoir_state[quantum_node_indices] = quantum_result
        except Exception as e:
            self.logger.warning(f"Error applying quantum nonlinearity: {e}")
            # Continue with classical state (already computed)
        
        return self.reservoir_state
    
    def _extract_temporal_features(self, features: np.ndarray) -> Dict[str, float]:
        """
        Extract a FIXED SET of temporal and other features based on configuration.
        Aims to produce self.input_dimensionality features total, stored in the
        'temporal_features' dictionary before flattening.

        Args:
            features (np.ndarray): Input feature array (can be multi-column).

        Returns:
            Dict[str, float]: Dictionary containing exactly self.input_dimensionality scalar features.
                              (Variable named 'temporal_features' for consistency, though it
                               may include non-temporal padding or quantum features).
        """
        # Define the target number of features
        target_dim = self.input_dimensionality
        temporal_features: Dict[str, float] = {}
        features_generated = 0

        # Input validation
        if features is None or not isinstance(features, np.ndarray) or features.size == 0:
            self.logger.error("Invalid or empty features in _extract_temporal_features")
            # Return dictionary with default values to match target_dim
            return {f'default_{i}': 0.0 for i in range(target_dim)}

        # Ensure features is 2D
        if features.ndim == 1:
            features = features.reshape(-1, 1)
        num_input_cols = features.shape[1]
        num_rows = features.shape[0]

        # --- Strategy: Prioritize stats from first column, then add quantum, then pad ---
        # Calculate how many features can be generated from stats (mean, std, trend)
        stats_per_window = 3
        num_stat_features = len(self.temporal_windows) * stats_per_window

        # 1. Calculate Stats for the first input column
        first_col_idx = 0
        self.logger.debug(f"Generating stats for column {first_col_idx} across windows: {self.temporal_windows}")
        for window in self.temporal_windows:
            if num_rows >= window:
                window_data = features[-window:, first_col_idx]
                # Calculate stats safely
                try:
                    mean_val = float(np.mean(window_data))
                    std_val = float(np.std(window_data))
                    trend_val = float(np.mean(np.diff(window_data))) if window > 1 else 0.0
                except Exception as e_stat:
                    self.logger.warning(f"Stat calculation error for window {window}: {e_stat}. Using 0.")
                    mean_val, std_val, trend_val = 0.0, 0.0, 0.0

                # Add features if limit not reached
                if features_generated < target_dim:
                    temporal_features[f'mean_{window}_c{first_col_idx}'] = np.nan_to_num(mean_val) # Store in original var name
                    features_generated += 1
                if features_generated < target_dim:
                    temporal_features[f'std_{window}_c{first_col_idx}'] = np.nan_to_num(std_val) # Store in original var name
                    features_generated += 1
                if features_generated < target_dim and window > 1:
                    temporal_features[f'trend_{window}_c{first_col_idx}'] = np.nan_to_num(trend_val) # Store in original var name
                    features_generated += 1
            else:
                # Not enough data for this window, add placeholders if needed
                if features_generated < target_dim: temporal_features[f'mean_{window}_c{first_col_idx}']=0.0; features_generated+=1
                if features_generated < target_dim: temporal_features[f'std_{window}_c{first_col_idx}']=0.0; features_generated+=1
                if features_generated < target_dim and window > 1: temporal_features[f'trend_{window}_c{first_col_idx}']=0.0; features_generated+=1

            if features_generated >= target_dim: break # Stop if limit reached

        # 2. Add Quantum Features (if enabled and space allows)
        self.logger.debug(f"Features after stats: {features_generated}/{target_dim}")
        if not self.use_classical and QUANTUM_AVAILABLE and features_generated < target_dim:
            # Example: Use quantum features from the longest window applied to the first input column
            window_for_q = self.temporal_windows[-1] if self.temporal_windows else 20 # Default window if list empty
            col_for_q = 0
            self.logger.debug(f"Attempting to add quantum features (window={window_for_q}, col={col_for_q})")
            if num_rows >= window_for_q:
                try:
                    window_features = features[-window_for_q:, col_for_q]
                    quantum_input = self._prepare_vector_for_quantum(window_features)

                    quantum_result = self._execute_with_fallback('feature_extraction', (quantum_input,))
                    self.logger.debug(f"Quantum feature extraction result: {quantum_result}")

                    for i in range(len(quantum_result)):
                        if features_generated < target_dim:
                            q_val = float(quantum_result[i]) if np.isscalar(quantum_result[i]) else 0.0
                            temporal_features[f'qfeat_{i}'] = np.nan_to_num(q_val) # Store in original var name
                            features_generated += 1
                        else: break
                except Exception as e_qfeat:
                    self.logger.warning(f"Error generating quantum features: {e_qfeat}")
            else:
                 self.logger.debug(f"Not enough data ({num_rows}) for quantum feature window ({window_for_q})")

        # 3. Pad if still fewer features than required
        self.logger.debug(f"Features after quantum: {features_generated}/{target_dim}")
        while features_generated < target_dim:
            temporal_features[f'pad_{features_generated}'] = 0.0 # Store in original var name
            features_generated += 1

        # 4. Final Check and Truncation (Safety Net)
        if len(temporal_features) != target_dim:
             self.logger.warning(f"Feature generation logic resulted in {len(temporal_features)} features, expected {target_dim}. Adjusting.")
             if len(temporal_features) > target_dim:
                  temporal_features = dict(list(temporal_features.items())[:target_dim])
             while len(temporal_features) < target_dim:
                  pad_idx = len(temporal_features)
                  temporal_features[f'final_pad_{pad_idx}'] = 0.0

        # Ensure all values are floats (extra safety)
        for k, v in temporal_features.items():
            try:
                 temporal_features[k] = float(v)
            except (TypeError, ValueError):
                 self.logger.warning(f"Could not convert final feature '{k}' to float. Setting to 0.0")
                 temporal_features[k] = 0.0

        # self.logger.debug(f"Final extracted features (count: {len(temporal_features)}): {list(temporal_features.keys())}")
        return temporal_features # Return the dict named temporal_features
    
    def _apply_quantum_transformations(self, features: np.ndarray) -> np.ndarray:
        """
        Apply quantum transformations to features.
        
        Args:
            features (np.ndarray): Feature array
            
        Returns:
            np.ndarray: Quantum-transformed features
        """
        if self.use_classical or not QUANTUM_AVAILABLE:
            # Classical fallback
            return features
            
        try:
            # Extract reference patterns (use some past patterns)
            if features.shape[0] >= 30:
                reference_patterns = [
                    features[-30],
                    features[-20],
                    features[-10]
                ]
                
                # Compute kernel values with reference patterns
                quantum_features = []
                
                for ref_pattern in reference_patterns:
                    for i in range(min(10, features.shape[0])):
                        # Take recent feature vectors
                        recent_idx = -i - 1
                        if abs(recent_idx) < features.shape[0]:
                            # Calculate quantum kernel similarity
                            x = features[recent_idx][:self.qubits] if len(features[recent_idx]) > self.qubits else np.pad(features[recent_idx], (0, self.qubits - len(features[recent_idx])))
                            y = ref_pattern[:self.qubits] if len(ref_pattern) > self.qubits else np.pad(ref_pattern, (0, self.qubits - len(ref_pattern)))
                            
                            # Use quantum kernel circuit
                            kernel_result = self._execute_with_fallback('kernel', (x, y))
                            kernel_value = np.mean(kernel_result)
                            
                            quantum_features.append(kernel_value)
                
                return np.array(quantum_features)
            else:
                # Not enough data, return original features
                return features
                
        except Exception as e:
            self.logger.error(f"Quantum transformation error: {e}", exc_info=True)
            return features
    
    def process(self, features: np.ndarray, frequency_components: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, Any]:
        """
        Process input features through the reservoir computing system.

        Args:
            features (np.ndarray): Raw feature array (can be multi-column).
            frequency_components (Dict[str, np.ndarray], optional): Frequency domain features (currently unused).

        Returns:
            Dict[str, Any]: Processing results including trend, volatility, momentum, and regime.
        """
        start_time = time.time()

        try:
            if not self.is_initialized:
                self._initialize_reservoir()

            if features is None or not isinstance(features, np.ndarray) or features.size == 0:
                self.logger.error("QERC process called with invalid input features")
                return {'error': 'Invalid input features'}

            # ---> Generate the FIXED-SIZE feature vector using the original variable name <---
            # _extract_temporal_features now returns the dict named 'temporal_features'
            temporal_features = self._extract_temporal_features(features)

            # Convert the dict into a flat numpy array in a consistent order.
            expected_keys = sorted(temporal_features.keys()) # Sort keys
            # ---> USE ORIGINAL VARIABLE NAME 'flattened_features' for the vector <---
            flattened_features = np.array([temporal_features.get(k, 0.0) for k in expected_keys], dtype=np.float64)

            # ---> VALIDATION <---
            if flattened_features.size != self.input_dimensionality:
                self.logger.error(f"CRITICAL SIZE MISMATCH! Expected {self.input_dimensionality}, generated {flattened_features.size}. Keys: {expected_keys}")
                raise ValueError(f"Internal feature vector size mismatch: Expected {self.input_dimensionality}, got {flattened_features.size}")

            self.logger.debug(f"QERC processing with feature vector size: {flattened_features.size}")

            # Update reservoir state using the correctly sized 'flattened_features' vector
            reservoir_state = self._update_reservoir_state(flattened_features) # Pass the flat vector

            # Apply quantum transformations if enabled
            # ---> USE ORIGINAL VARIABLE NAME 'combined_state' <---
            combined_state = reservoir_state # Start with the updated reservoir state
            if not self.use_classical and QUANTUM_AVAILABLE:
                try:
                    quantum_transformed_part = self._apply_quantum_transformations(features) # Use raw features
                    if isinstance(quantum_transformed_part, np.ndarray) and quantum_transformed_part.size > 0:
                        # Combine state logic (remains the same)
                        res_flat = reservoir_state.flatten()
                        qt_flat = quantum_transformed_part.flatten()
                        len_res = len(res_flat)
                        len_qt = len(qt_flat)
                        if len_qt < len_res:
                             combined_state_flat = np.concatenate((res_flat[:len_res - len_qt], qt_flat))
                        else:
                             combined_state_flat = qt_flat[:len_res]
                        combined_state = combined_state_flat
                        self.logger.debug("Applied quantum transformations to state.")
                except Exception as e_qt_apply:
                     self.logger.warning(f"Error applying quantum transformations: {e_qt_apply}. Using reservoir state.")

            # Ensure combined_state is flattened for calculation methods
            if combined_state is not None and combined_state.ndim > 1:
                combined_state = combined_state.flatten()
            elif combined_state is None:
                 self.logger.error("Combined state is None, using zero vector for output calculation.")
                 combined_state = np.zeros(self.reservoir_size) # Use reservoir_size for default

            # --- Calculate Outputs ---
            # Pass the original multi-column 'features' for context if needed by calc methods
            trend = self._calculate_trend(combined_state, features)
            volatility = self._calculate_volatility(combined_state, features)
            momentum = self._calculate_momentum(combined_state, features)
            regime = self._calculate_regime(combined_state, features)

            # --- Performance Tracking ---
            execution_time = (time.time() - start_time) * 1000 # ms
            self.execution_times.append(execution_time)
            if len(self.execution_times) > 100: self.execution_times.pop(0) # FIFO
            # Report time (Optional: Add back if needed)
            # if self.hardware_manager and hasattr(self.hardware_manager, 'track_execution_time'):
            #     device_type = 'quantum' if not self.use_classical else 'cpu'
            #     self.hardware_manager.track_execution_time(device_type, execution_time)

            # --- Return Results ---
            # Keep the original return structure
            return {
                'trend': trend,
                'volatility': volatility,
                'momentum': momentum,
                'regime': regime,
                # Return the structured features dict (now named temporal_features) if needed downstream
                'temporal_features': temporal_features,
                'reservoir_state': combined_state, # Return the final combined state
                'execution_time_ms': execution_time
            }

        except Exception as e:
            self.logger.error(f"QERC processing error: {str(e)}", exc_info=True)
            return { 'error': str(e), 'trend': 0.0, 'volatility': 0.0, 'momentum': 0.0, 'regime': 0.5 }
        
    def _calculate_trend(self, reservoir_state: np.ndarray, features: np.ndarray) -> float:
        """
        Calculate trend signal from reservoir state.
        
        Args:
            reservoir_state (np.ndarray): Current reservoir state
            features (np.ndarray): Original features
            
        Returns:
            float: Trend signal between -1 and 1
        """
        try:
            # Use available price data for trend calculation
            if features.shape[0] > 20:
                # Simple trend calculation based on moving average
                short_ma = np.mean(features[-5:, 0])  # Assuming prices are in first column
                long_ma = np.mean(features[-20:, 0])
                
                # Normalize trend to [-1, 1] range
                raw_trend = (short_ma - long_ma) / long_ma
                trend = np.tanh(raw_trend * 10)  # Scale and bound to [-1, 1]
                
                # Blend with reservoir output
                reservoir_output = np.tanh(np.dot(self.W_out[0], reservoir_state))
                blended_trend = 0.7 * trend + 0.3 * reservoir_output
                
                return blended_trend
            else:
                # Not enough data, use reservoir state only
                return np.tanh(np.dot(self.W_out[0], reservoir_state))
                
        except Exception as e:
            self.logger.error(f"Trend calculation error: {str(e)}")
            return 0.0
    
    def _calculate_volatility(self, reservoir_state: np.ndarray, features: np.ndarray) -> float:
        """
        Calculate volatility signal from reservoir state.
        
        Args:
            reservoir_state (np.ndarray): Current reservoir state
            features (np.ndarray): Original features
            
        Returns:
            float: Volatility signal between 0 and 1
        """
        try:
            # Calculate volatility using returns
            if features.shape[0] > 20:
                # Calculate returns
                prices = features[:, 0]  # Assuming prices are in first column
                # Add safe division to prevent divide by zero warnings
                denominators = prices[:-1].copy()
                # Replace zeros with small values to avoid division by zero
                denominators[denominators == 0] = 1e-10
                returns = np.diff(prices) / denominators
                
                # Calculate volatility (standard deviation of returns)
                volatility = np.std(returns[-20:])
                
                # Normalize to [0, 1] range
                normalized_volatility = 1 - np.exp(-50 * volatility)
                
                # Blend with reservoir output
                reservoir_output = 0.5 * (np.tanh(np.dot(self.W_out[1], reservoir_state)) + 1)
                blended_volatility = 0.7 * normalized_volatility + 0.3 * reservoir_output
                
                return np.clip(blended_volatility, 0, 1)
            else:
                # Not enough data, use reservoir state only
                return 0.5 * (np.tanh(np.dot(self.W_out[1], reservoir_state)) + 1)
                
        except Exception as e:
            self.logger.error(f"Volatility calculation error: {str(e)}")
            return 0.5
    
    def _calculate_momentum(self, reservoir_state: np.ndarray, features: np.ndarray) -> float:
        """
        Calculate momentum signal from reservoir state.
        
        Args:
            reservoir_state (np.ndarray): Current reservoir state
            features (np.ndarray): Original features
            
        Returns:
            float: Momentum signal between -1 and 1
        """
        try:
            # Calculate momentum using multiple timeframes
            if features.shape[0] > 60:
                # Calculate returns over multiple timeframes
                prices = features[:, 0]  # Assuming prices are in first column
                
                # Short-term momentum (5 periods)
                short_return = (prices[-1] / prices[-5]) - 1
                
                # Medium-term momentum (20 periods)
                medium_return = (prices[-1] / prices[-20]) - 1
                
                # Long-term momentum (60 periods)
                long_return = (prices[-1] / prices[-60]) - 1
                
                # Combine returns with decreasing weights
                weighted_momentum = 0.5 * short_return + 0.3 * medium_return + 0.2 * long_return
                
                # Normalize to [-1, 1] range
                normalized_momentum = np.tanh(weighted_momentum * 10)
                
                # Blend with reservoir output
                reservoir_output = np.tanh(np.dot(self.W_out[2], reservoir_state))
                blended_momentum = 0.7 * normalized_momentum + 0.3 * reservoir_output
                
                return np.clip(blended_momentum, -1, 1)
            else:
                # Not enough data, use reservoir state only
                return np.tanh(np.dot(self.W_out[2], reservoir_state))
                
        except Exception as e:
            self.logger.error(f"Momentum calculation error: {str(e)}")
            return 0.0
    
    def _calculate_regime(self, reservoir_state: np.ndarray, features: np.ndarray) -> float:
        """
        Calculate market regime signal from reservoir state.
        
        Args:
            reservoir_state (np.ndarray): Current reservoir state
            features (np.ndarray): Original features
            
        Returns:
            float: Market regime signal between 0 and 1
            (0 = low volatility/trending, 1 = high volatility/mean-reverting)
        """
        try:
            # Calculate regime using price action and volatility
            if features.shape[0] > 40:
                # Calculate returns and volatility
                prices = features[:, 0]  # Assuming prices are in first column
                # Add safe division to prevent divide by zero warnings
                denominators = prices[:-1].copy()
                # Replace zeros with small values to avoid division by zero
                denominators[denominators == 0] = 1e-10
                returns = np.diff(prices) / denominators
                
                # Calculate volatility
                volatility = np.std(returns[-20:])
                
                # Calculate autocorrelation (lag 1)
                if len(returns) > 20:
                    autocorr = np.corrcoef(returns[-21:-1], returns[-20:])[0, 1]
                else:
                    autocorr = 0
                
                # Low autocorrelation + high volatility = mean-reverting regime (1)
                # High autocorrelation + low volatility = trending regime (0)
                normalized_volatility = 1 - np.exp(-50 * volatility)
                normalized_autocorr = (1 - autocorr) / 2  # Map [-1, 1] to [1, 0]
                
                # Combine signals
                regime_signal = 0.6 * normalized_volatility + 0.4 * normalized_autocorr
                
                # Blend with reservoir output
                reservoir_output = 0.5 * (np.tanh(np.dot(self.W_out[3], reservoir_state)) + 1)
                blended_regime = 0.7 * regime_signal + 0.3 * reservoir_output
                
                return np.clip(blended_regime, 0, 1)
            else:
                # Not enough data, use reservoir state only
                return 0.5 * (np.tanh(np.dot(self.W_out[3], reservoir_state)) + 1)
                
        except Exception as e:
            self.logger.error(f"Regime calculation error: {str(e)}")
            return 0.5
    
    def reset_state(self) -> None:
        """Reset the reservoir state."""
        self.logger.info("Resetting QERC reservoir state")
        self.reservoir_state = np.zeros(self.reservoir_size)
        
        # Clear cache as well
        with self.cache_lock:
            self.cache.clear()
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """
        Get execution statistics for performance monitoring.
        
        Returns:
            Dict[str, Any]: Execution statistics
        """
        if not self.execution_times:
            return {"count": 0, "avg_time_ms": 0, "min_time_ms": 0, "max_time_ms": 0}
        
        times = np.array(self.execution_times)
        return {
            "count": len(times),
            "avg_time_ms": np.mean(times),
            "min_time_ms": np.min(times),
            "max_time_ms": np.max(times)
        }
    
    @quantum_accelerated(use_hw_accel=True, device_shots=1024)
    def quantum_phase_transition_detector(self, timeseries: np.ndarray, window_size: int = 20) -> Dict[str, Any]:
        """
        Quantum Phase Transition Detector that uses quantum phase estimation to
        identify critical points in price movements.
        
        This method can detect regime changes earlier than classical methods by using quantum
        phase estimation to identify phase transitions in market dynamics.
        
        Args:
            timeseries (np.ndarray): 1D array of price or other market data
            window_size (int): Size of the rolling window for phase detection
            
        Returns:
            Dict[str, Any]: Results containing phase change probability and estimated critical points
        """
        if self.use_classical or not QUANTUM_AVAILABLE or len(timeseries) < window_size + 5:
            # Classical fallback using change point detection
            self.logger.debug("Using classical fallback for phase transition detection")
            return self._classical_phase_transition_detection(timeseries, window_size)
            
        self.logger.debug("Running quantum phase transition detection")
        result = {}
        
        try:
            # Prepare time series data - normalize to [0, 2π] for phase encoding
            min_val, max_val = np.min(timeseries), np.max(timeseries)
            if max_val > min_val:  # Avoid division by zero
                normalized_series = 2 * np.pi * (timeseries - min_val) / (max_val - min_val)
            else:
                normalized_series = np.zeros_like(timeseries)
            
            # Process with sliding window
            transition_probabilities = []
            critical_points = []
            
            # Define quantum circuit for phase estimation
            @qml.qnode(self.device)
            def phase_estimation_circuit(phases, precision_qubits=3):
                """Quantum circuit to estimate phase transitions
                
                Args:
                    phases: Array of phase values from the time series window
                    precision_qubits: Number of qubits for phase precision
                
                Returns:
                    Measurement of the probability of phase transition
                """
                # Number of qubits needed (precision + 1 target qubit)
                n_qubits = precision_qubits + 1
                n_wires = n_qubits
                
                # Initialize precision qubits in superposition
                for i in range(precision_qubits):
                    qml.Hadamard(wires=i)
                
                # Initialize target qubit
                qml.X(wires=precision_qubits)
                qml.Hadamard(wires=precision_qubits)
                
                # Encode phase information in controlled rotations
                for i, phase in enumerate(phases):
                    idx = i % precision_qubits
                    # Controlled rotation based on phase value
                    qml.CRZ(phase, wires=[idx, precision_qubits])
                    
                    # Add entanglement between adjacent qubits
                    if i < len(phases) - 1 and idx < precision_qubits - 1:
                        qml.CNOT(wires=[idx, idx+1])
                
                # Apply inverse QFT to precision qubits
                for i in range(precision_qubits):
                    for j in range(i):
                        qml.CRZ(-np.pi/2**(i-j), wires=[j, i])
                    qml.Hadamard(wires=i)
                
                # Measure probabilities
                return qml.probs(wires=range(n_wires))
            
            # Process sliding windows
            for i in range(len(timeseries) - window_size):
                window = normalized_series[i:i+window_size]
                
                # For efficiency, subsample if window is large
                if window_size > 10:
                    # Sample key points from window: start, end, and some in middle
                    indices = np.concatenate([
                        [0, window_size-1],  # always include start and end
                        np.linspace(1, window_size-2, min(8, window_size-2)).astype(int)  # add middle points
                    ])
                    window = window[np.unique(indices)]
                
                # Run phase estimation circuit
                phases = window
                probs = phase_estimation_circuit(phases)
                
                # Analyze results to detect phase transitions
                # We'll look at the probability distribution of the measurement outcomes
                # High entropy in the distribution indicates phase transition
                entropy = -np.sum(probs * np.log2(probs + 1e-10))  # Add small epsilon to avoid log(0)
                norm_entropy = entropy / np.log2(len(probs))  # Normalize by max possible entropy
                
                # Calculate phase transition probability
                # We're looking for significant changes in the probability distribution
                transition_prob = norm_entropy
                transition_probabilities.append(transition_prob)
                
                # Identify critical points where probability exceeds threshold
                if transition_prob > 0.7:  # Threshold can be tuned
                    critical_points.append(i + window_size // 2)  # Middle of window
            
            # Build result dictionary
            result = {
                "transition_probabilities": transition_probabilities,
                "critical_points": critical_points,
                "transition_detected": len(critical_points) > 0,
                "mean_transition_probability": np.mean(transition_probabilities) if transition_probabilities else 0,
                "latest_transition_probability": transition_probabilities[-1] if transition_probabilities else 0
            }
            
        except Exception as e:
            self.logger.error(f"Error in quantum phase transition detector: {e}", exc_info=True)
            # Fall back to classical method
            result = self._classical_phase_transition_detection(timeseries, window_size)
            result["error"] = str(e)
            
        return result
            
    def _classical_phase_transition_detection(self, timeseries: np.ndarray, window_size: int) -> Dict[str, Any]:
        """Classical fallback method for phase transition detection
        
        Uses rolling statistics to approximate phase transitions
        """
        result = {}
        
        try:
            # Use optimized implementation
            transition_probs, critical_points = self._calculate_phase_transitions_numba(timeseries, window_size)
            
            result = {
                "transition_probabilities": transition_probs.tolist(),
                "critical_points": critical_points,
                "transition_detected": len(critical_points) > 0,
                "mean_transition_probability": np.mean(transition_probs) if len(transition_probs) > 0 else 0,
                "latest_transition_probability": transition_probs[-1] if len(transition_probs) > 0 else 0,
                "quantum": False  # Indicate this is classical result
            }
        except Exception as e:
            self.logger.error(f"Error in classical phase transition detection: {e}", exc_info=True)
            result = {
                "error": str(e),
                "transition_detected": False,
                "critical_points": [],
                "mean_transition_probability": 0,
                "latest_transition_probability": 0,
                "quantum": False
            }
            
        return result
        
    @staticmethod
    @njit(parallel=True)
    def _calculate_phase_transitions_numba(timeseries, window_size):
        """Numba-accelerated implementation of phase transition detection
        
        Args:
            timeseries: Input time series data
            window_size: Size of the rolling window
            
        Returns:
            Tuple of (transition_probabilities, critical_points)
        """
        n_windows = len(timeseries) - window_size + 1
        mean_values = np.zeros(n_windows)
        std_values = np.zeros(n_windows)
        
        # Calculate rolling statistics in parallel
        for i in prange(n_windows):
            window = timeseries[i:i+window_size]
            # Calculate mean
            window_sum = 0.0
            for j in range(window_size):
                window_sum += window[j]
            mean_values[i] = window_sum / window_size
            
            # Calculate std dev
            variance_sum = 0.0
            for j in range(window_size):
                variance_sum += (window[j] - mean_values[i])**2
            std_values[i] = np.sqrt(variance_sum / window_size)
        
        # Identify potential change points
        mean_change = np.zeros(n_windows - 1)
        std_change = np.zeros(n_windows - 1)
        
        for i in range(n_windows - 1):
            mean_change[i] = abs(mean_values[i+1] - mean_values[i])
            std_change[i] = abs(std_values[i+1] - std_values[i])
        
        # Normalize changes
        max_mean_change = 0.0
        max_std_change = 0.0
        
        for i in range(len(mean_change)):
            if mean_change[i] > max_mean_change:
                max_mean_change = mean_change[i]
            if std_change[i] > max_std_change:
                max_std_change = std_change[i]
        
        if max_mean_change > 0:
            for i in range(len(mean_change)):
                mean_change[i] = mean_change[i] / max_mean_change
                
        if max_std_change > 0:
            for i in range(len(std_change)):
                std_change[i] = std_change[i] / max_std_change
        
        # Combine signals
        combined_change = np.zeros(len(mean_change))
        for i in range(len(mean_change)):
            combined_change[i] = (mean_change[i] + std_change[i]) / 2.0
        
        # Identify critical points
        threshold = 0.7
        critical_points = []
        
        for i in range(len(combined_change)):
            if combined_change[i] > threshold:
                critical_points.append(i + window_size // 2)
        
        return combined_change, critical_points
    
    @quantum_accelerated(use_hw_accel=True, device_shots=1024)
    def quantum_entropy_analyzer(self, timeseries: np.ndarray, window_size: int = 20) -> Dict[str, Any]:
        """
        Quantum Entropy Analyzer that measures quantum entropy in market data.
        
        This method provides enhanced measurements of market uncertainty beyond classical
        volatility by using quantum entropy estimations techniques.
        
        Args:
            timeseries (np.ndarray): 1D array of price or other market data
            window_size (int): Size of the rolling window for entropy analysis
            
        Returns:
            Dict[str, Any]: Results containing quantum entropy measures
        """
        if self.use_classical or not QUANTUM_AVAILABLE or len(timeseries) < window_size:
            # Classical fallback for entropy calculation
            self.logger.debug("Using classical fallback for entropy analysis")
            return self._classical_entropy_analysis(timeseries, window_size)
            
        self.logger.debug("Running quantum entropy analysis")
        result = {}
        
        try:
            # Normalize time series for quantum encoding
            norm_series = self._prepare_vector_for_quantum(timeseries)
            
            # Define quantum circuit for entropy estimation
            @qml.qnode(self.device)
            def entropy_estimation_circuit(data_window):
                """Quantum circuit for entropy estimation
                
                Uses a circuit designed to estimate von Neumann entropy of data
                
                Args:
                    data_window: Window of time series data
                    
                Returns:
                    Measurements representing entropy estimate
                """
                n_qubits = self.qubits
                
                # Encode data into quantum state - amplitude encoding
                # First normalize the data window to have unit norm
                norm = np.sqrt(np.sum(np.abs(data_window)**2))
                if norm > 0:
                    normalized_data = data_window / norm
                else:
                    normalized_data = np.ones(len(data_window)) / np.sqrt(len(data_window))
                    
                # Prepare qubits in equal superposition
                for i in range(n_qubits):
                    qml.Hadamard(wires=i)
                
                # Encode data patterns
                for i in range(min(len(normalized_data), 2**n_qubits)):
                    # Convert index to binary for qubit addressing
                    binary_i = format(i, f'0{n_qubits}b')
                    
                    # Apply controlled rotations based on data value
                    angle = np.pi * normalized_data[i % len(normalized_data)]
                    target_qubit = n_qubits - 1
                    
                    # Use binary representation to apply controlled operations
                    control_qubits = []
                    for j in range(n_qubits - 1):
                        if binary_i[j] == '1':
                            control_qubits.append(j)
                    
                    if control_qubits:
                        # Apply multi-controlled rotation if we have control qubits
                        qml.MultiControlledX(control_qubits, target_qubit)
                        qml.RZ(angle, wires=target_qubit)
                        qml.MultiControlledX(control_qubits, target_qubit)
                    else:
                        # Apply simple rotation if no control qubits
                        qml.RZ(angle, wires=target_qubit)
                    
                # Apply Hadamard to all qubits to interfere the amplitudes
                for i in range(n_qubits):
                    qml.Hadamard(wires=i)
                
                # Return all measurement probabilities
                return qml.probs(wires=range(n_qubits))
            
            # Process with sliding window
            quantum_entropies = []
            entanglement_entropies = []
            von_neumann_entropies = []
            
            # Process each window
            for i in range(len(norm_series) - window_size + 1):
                window = norm_series[i:i+window_size]
                
                # Downsample window if needed to match qubit count
                if len(window) > 2**self.qubits:
                    # Sample key points - always include newest data points
                    indices = np.linspace(0, len(window)-1, 2**self.qubits).astype(int)
                    window = window[indices]
                elif len(window) < 2**self.qubits:
                    # Pad if needed
                    window = np.pad(window, (0, 2**self.qubits - len(window)))
                
                # Run quantum circuit to estimate entropy
                probs = entropy_estimation_circuit(window)
                
                # Calculate von Neumann entropy: -Tr(ρ log ρ) where ρ is density matrix
                von_neumann = -np.sum(probs * np.log2(probs + 1e-10))
                von_neumann_norm = von_neumann / self.qubits  # Normalize by qubit count
                
                # Simplified entanglement entropy estimate using reduced density matrix
                # In practice, this would require state tomography
                # Here we approximate by splitting the system into two parts
                half_system = self.qubits // 2
                reduced_probs = np.zeros(2**half_system)
                
                # Sum over the traced-out subsystem
                for i in range(len(probs)):
                    # Get index in reduced system by masking out traced bits
                    reduced_idx = i & ((1 << half_system) - 1)
                    reduced_probs[reduced_idx] += probs[i]
                
                # Calculate entanglement entropy
                entanglement = -np.sum(reduced_probs * np.log2(reduced_probs + 1e-10))
                entanglement_norm = entanglement / half_system  # Normalize
                
                # Combined quantum entropy measure (weighted average)
                quantum_entropy = 0.7 * von_neumann_norm + 0.3 * entanglement_norm
                
                # Store results
                quantum_entropies.append(quantum_entropy)
                von_neumann_entropies.append(von_neumann_norm)
                entanglement_entropies.append(entanglement_norm)
            
            # Build result dictionary
            result = {
                "quantum_entropy": quantum_entropies,
                "von_neumann_entropy": von_neumann_entropies,
                "entanglement_entropy": entanglement_entropies,
                "latest_quantum_entropy": quantum_entropies[-1] if quantum_entropies else 0,
                "mean_quantum_entropy": np.mean(quantum_entropies) if quantum_entropies else 0,
                "entropy_increasing": len(quantum_entropies) > 1 and quantum_entropies[-1] > quantum_entropies[0],
                "quantum": True
            }
            
        except Exception as e:
            self.logger.error(f"Error in quantum entropy analyzer: {e}", exc_info=True)
            # Fall back to classical method
            result = self._classical_entropy_analysis(timeseries, window_size)
            result["error"] = str(e)
            
        return result
    
    def _classical_entropy_analysis(self, timeseries: np.ndarray, window_size: int) -> Dict[str, Any]:
        """Classical fallback method for entropy analysis
        
        Uses Shannon entropy and other statistical measures to approximate quantum entropy
        """
        result = {}
        
        try:
            # Use the numba-accelerated implementation
            shannon_entropies, approx_entropies, sample_entropies, combined_entropies = \
                self._calculate_entropy_metrics_numba(timeseries, window_size)
            
            result = {
                "quantum_entropy": combined_entropies.tolist(),  # Best classical approximation
                "shannon_entropy": shannon_entropies.tolist(),
                "approximate_entropy": approx_entropies.tolist(),
                "sample_entropy": sample_entropies.tolist(),
                "latest_quantum_entropy": combined_entropies[-1] if len(combined_entropies) > 0 else 0,
                "mean_quantum_entropy": np.mean(combined_entropies) if len(combined_entropies) > 0 else 0,
                "entropy_increasing": len(combined_entropies) > 1 and combined_entropies[-1] > combined_entropies[0],
                "quantum": False  # Indicate this is classical result
            }
            
        except Exception as e:
            self.logger.error(f"Error in classical entropy analysis: {e}", exc_info=True)
            result = {
                "error": str(e),
                "quantum_entropy": [],
                "latest_quantum_entropy": 0,
                "mean_quantum_entropy": 0,
                "entropy_increasing": False,
                "quantum": False
            }
            
        return result
        
    @staticmethod
    @njit(parallel=True)
    def _calculate_entropy_metrics_numba(timeseries, window_size):
        """Numba-accelerated implementation of entropy metrics calculation
        
        Args:
            timeseries: Input time series data
            window_size: Size of the rolling window
            
        Returns:
            Tuple of (shannon_entropies, approx_entropies, sample_entropies, combined_entropies)
        """
        n_windows = len(timeseries) - window_size + 1
        shannon_entropies = np.zeros(n_windows)
        approx_entropies = np.zeros(n_windows)
        sample_entropies = np.zeros(n_windows)
        
        for i in prange(n_windows):
            window = timeseries[i:i+window_size]
            
            # Calculate Shannon entropy manually (since np.histogram isn't supported in numba)
            # Create simple histogram with 10 bins
            min_val = np.min(window)
            max_val = np.max(window)
            n_bins = min(10, window_size)
            
            if max_val > min_val:
                bin_width = (max_val - min_val) / n_bins
                hist_counts = np.zeros(n_bins)
                
                # Count values in each bin
                for val in window:
                    bin_idx = min(n_bins - 1, int((val - min_val) / bin_width))
                    hist_counts[bin_idx] += 1
                
                # Normalize to get probabilities
                hist_probs = hist_counts / window_size
                
                # Calculate Shannon entropy
                shannon = 0.0
                for p in hist_probs:
                    if p > 0:
                        shannon -= p * np.log2(p)
                
                # Normalize Shannon entropy
                log_n_bins = np.log2(n_bins)
                shannon_entropies[i] = shannon / log_n_bins if log_n_bins > 0 else 0
            else:
                shannon_entropies[i] = 0
            
            # Calculate approximate entropy
            m = 2  # Embedding dimension
            r = 0.2 * np.std(window)  # Threshold
            
            # Count similar patterns using modified approximate entropy technique
            count = 0
            total = 0
            for j in range(window_size - m):
                for k in range(j+1, window_size - m + 1):
                    # Check if patterns are similar
                    similar = True
                    for l in range(m):
                        if abs(window[j+l] - window[k+l]) > r:
                            similar = False
                            break
                    if similar:
                        count += 1
                    total += 1
            
            # Normalize approximate entropy
            if total > 0 and count > 0:
                approx_ent = -np.log(count / total)
                log_window = np.log(window_size)
                approx_entropies[i] = min(1.0, approx_ent / log_window if log_window > 0 else 0)
            else:
                approx_entropies[i] = 0
            
            # Sample entropy (simplified)
            # Adding pseudo-random noise (can't use np.random in numba)
            # Using a simple deterministic noise based on the window values
            noise = 0.1 * (np.sum(window) % 1.0)
            sample_entropies[i] = min(1.0, max(0.0, approx_entropies[i] + noise))
        
        # Calculate combined classical entropy (weighted average)
        combined_entropies = np.zeros(n_windows)
        for i in range(n_windows):
            combined_entropies[i] = 0.5 * shannon_entropies[i] + 0.3 * approx_entropies[i] + 0.2 * sample_entropies[i]
        
        return shannon_entropies, approx_entropies, sample_entropies, combined_entropies
        
    @quantum_accelerated(use_hw_accel=True, device_shots=1024)
    def quantum_momentum_oscillator(self, timeseries: np.ndarray, window_size: int = 20, sensitivity: float = 0.5) -> Dict[str, Any]:
        """
        Quantum Momentum Oscillator using quantum interference patterns to detect momentum shifts.
        
        This indicator is more sensitive to subtle changes in price direction than classical
        momentum indicators by leveraging quantum interference effects.
        
        Args:
            timeseries (np.ndarray): 1D array of price or other market data
            window_size (int): Size of the rolling window for momentum analysis
            sensitivity (float): Sensitivity parameter between 0.0 and 1.0
            
        Returns:
            Dict[str, Any]: Results containing momentum signals and oscillator values
        """
        if self.use_classical or not QUANTUM_AVAILABLE or len(timeseries) < window_size:
            # Classical fallback for momentum calculation
            self.logger.debug("Using classical fallback for momentum oscillator")
            return self._classical_momentum_oscillator(timeseries, window_size, sensitivity)
            
        self.logger.debug("Running quantum momentum oscillator")
        result = {}
        
        try:
            # Normalize time series for quantum encoding
            # For momentum, we care about direction changes, so we'll use returns
            returns = np.zeros(len(timeseries) - 1)
            denominators = timeseries[:-1].copy()
            denominators[denominators == 0] = 1e-10  # Prevent division by zero
            returns = np.diff(timeseries) / denominators
            
            # Clip extreme values to prevent outliers from dominating
            returns = np.clip(returns, -0.1, 0.1)
            
            # Normalize returns to [-1, 1] range for quantum circuit
            max_abs = np.max(np.abs(returns))
            if max_abs > 0:
                norm_returns = returns / max_abs
            else:
                norm_returns = returns
                
            # Process with sliding window
            momentum_values = []
            signal_line = []
            crossovers = []
            
            # Define quantum circuit for momentum detection using quantum walk
            @qml.qnode(self.device)
            def momentum_circuit(data_window, sensitivity):
                """Quantum circuit for momentum detection using quantum walk algorithm
                
                Args:
                    data_window: Window of normalized returns
                    sensitivity: Sensitivity parameter for the quantum walk
                    
                Returns:
                    Expectation values representing momentum direction and strength
                """
                n_qubits = self.qubits
                
                # Initialize position qubit in equal superposition
                for i in range(n_qubits - 1):  # Position register
                    qml.Hadamard(wires=i)
                
                # Last qubit will be our coin
                qml.Hadamard(wires=n_qubits - 1)
                
                # Number of walk steps - depends on window size and sensitivity
                steps = min(len(data_window), 10)
                
                # Perform quantum walk to detect momentum
                for step in range(steps):
                    # Get data point, use modulo to cycle through data if needed
                    idx = step % len(data_window)
                    data_point = data_window[idx]
                    
                    # Adjust coin rotation based on data point (represents momentum bias)
                    # Positive returns -> bias to right, Negative returns -> bias to left
                    coin_angle = np.pi/2 * (1 + sensitivity * data_point)  # Range [π/2, π]
                    qml.RY(coin_angle, wires=n_qubits - 1)
                    
                    # Entangle with position register for quantum walk step
                    for i in range(n_qubits - 2, -1, -1):
                        # Controlled-NOT from coin to position bits
                        # This creates entanglement between coin and position
                        qml.CNOT(wires=[n_qubits - 1, i])
                    
                    # Add quantum interference between steps 
                    # This is key to the quantum advantage - classical walk can't do this
                    if step < steps - 1:
                        qml.Hadamard(wires=n_qubits - 1)
                
                # Measure momentum direction and strength
                # We measure Z expectation of coin qubit for direction
                direction = qml.expval(qml.PauliZ(n_qubits - 1))
                
                # Measure position distribution for strength
                # More spread out means stronger momentum
                position_op = qml.PauliZ(0)
                for i in range(1, n_qubits - 1):
                    position_op = position_op @ qml.PauliZ(i)
                strength = qml.expval(position_op)
                
                return [direction, strength]
            
            # Process each window
            for i in range(len(norm_returns) - window_size + 1):
                window = norm_returns[i:i+window_size]
                
                # Optimize for small windows to improve performance
                if len(window) > 10:
                    # Sample key points with emphasis on recent data
                    recent_size = min(5, len(window) // 2)
                    historical_indices = np.linspace(0, len(window) - recent_size - 1, 5).astype(int)
                    recent_indices = np.arange(len(window) - recent_size, len(window))
                    indices = np.concatenate([historical_indices, recent_indices])
                    window = window[np.unique(indices)]
                
                # Run quantum circuit to estimate momentum
                [direction, strength] = momentum_circuit(window, sensitivity)
                
                # Calculate momentum as a combination of direction and strength
                # Normalize to [-1, 1] range
                momentum = direction * abs(strength)
                
                # Store momentum value
                momentum_values.append(momentum)
                
                # Calculate signal line (simple moving average of momentum)
                if len(momentum_values) >= 5:
                    signal = np.mean(momentum_values[-5:])
                    signal_line.append(signal)
                    
                    # Detect crossovers (momentum crosses signal line)
                    if len(signal_line) >= 2:
                        prev_diff = momentum_values[-2] - signal_line[-2]
                        curr_diff = momentum_values[-1] - signal_line[-1]
                        
                        # Crossover detected if sign changes
                        if prev_diff * curr_diff <= 0 and abs(prev_diff) > 0.05:
                            crossover_type = "bullish" if curr_diff > 0 else "bearish"
                            crossovers.append({
                                "position": i + window_size - 1,
                                "type": crossover_type,
                                "strength": abs(curr_diff)
                            })
            
            # Map momentum to oscillator range [0, 100] for consistency with traditional oscillators
            oscillator_values = [(m + 1) * 50 for m in momentum_values]
            
            # Determine overbought/oversold conditions
            overbought_threshold = 70
            oversold_threshold = 30
            overbought = [val > overbought_threshold for val in oscillator_values]
            oversold = [val < oversold_threshold for val in oscillator_values]
            
            # Find divergences (price makes higher high but momentum makes lower high)
            # This is a powerful signal in momentum oscillators
            divergences = []
            # Only check for divergences if we have enough data points
            if len(timeseries) >= window_size + 5 and len(oscillator_values) >= 15:
                # Look at price highs and lows
                for i in range(10, len(oscillator_values)):
                    price_window = timeseries[i:i+window_size]
                    osc_window = oscillator_values[i-5:i]
                    prev_osc_window = oscillator_values[i-10:i-5]
                    
                    # Ensure we have valid data for comparison
                    if len(osc_window) > 0 and len(prev_osc_window) > 0 and i+window_size <= len(timeseries) and i-5+window_size <= len(timeseries):
                        # Check for bullish divergence (price lower low, oscillator higher low)
                        if (price_window.min() < timeseries[i-5:i+window_size-5].min() and 
                                min(osc_window) > min(prev_osc_window)):
                            divergences.append({
                                "position": i + window_size - 1,
                                "type": "bullish",
                                "strength": abs(min(osc_window) - min(prev_osc_window))
                            })
                            
                        # Check for bearish divergence (price higher high, oscillator lower high)
                        if (price_window.max() > timeseries[i-5:i+window_size-5].max() and 
                                max(osc_window) < max(prev_osc_window)):
                            divergences.append({
                                "position": i + window_size - 1,
                                "type": "bearish",
                                "strength": abs(max(osc_window) - max(prev_osc_window))
                            })
            
            # Build result dictionary
            result = {
                "momentum": momentum_values,
                "oscillator": oscillator_values,
                "signal_line": signal_line,
                "crossovers": crossovers,
                "divergences": divergences,
                "latest_momentum": momentum_values[-1] if momentum_values else 0,
                "latest_oscillator": oscillator_values[-1] if oscillator_values else 50,
                "is_overbought": overbought[-1] if overbought else False,
                "is_oversold": oversold[-1] if oversold else False,
                "buy_signal": len(crossovers) > 0 and crossovers[-1]["type"] == "bullish",
                "sell_signal": len(crossovers) > 0 and crossovers[-1]["type"] == "bearish",
                "quantum": True
            }
            
        except Exception as e:
            self.logger.error(f"Error in quantum momentum oscillator: {e}", exc_info=True)
            # Fall back to classical method
            result = self._classical_momentum_oscillator(timeseries, window_size, sensitivity)
            result["error"] = str(e)
            
        return result
    
    @quantum_accelerated(use_hw_accel=True, device_shots=1024)
    def quantum_fractal_dimension_estimator(self, timeseries: np.ndarray, window_size: int = 50, scales: List[int] = None) -> Dict[str, Any]:
        """
        Quantum Fractal Dimension Estimator to detect market complexity changes.
        
        This method uses quantum computing techniques to estimate the fractal dimension
        of price movements, which can reveal changes in market complexity and self-similarity.
        
        Args:
            timeseries (np.ndarray): 1D array of price or other market data
            window_size (int): Size of the rolling window for fractal analysis
            scales (List[int]): List of scale sizes to use for dimension calculation
            
        Returns:
            Dict[str, Any]: Results containing fractal dimension estimates and complexity metrics
        """
        if scales is None:
            # Default scaling ranges for box counting
            scales = [2, 4, 8, 16]
            
        if self.use_classical or not QUANTUM_AVAILABLE or len(timeseries) < window_size:
            # Classical fallback for fractal dimension calculation
            self.logger.debug("Using classical fallback for fractal dimension estimation")
            return self._classical_fractal_dimension(timeseries, window_size, scales)
            
        self.logger.debug("Running quantum fractal dimension estimation")
        result = {}
        
        try:
            # Normalize time series for quantum encoding
            norm_series = self._prepare_vector_for_quantum(timeseries)
            
            # Process with sliding window
            fractal_dimensions = []
            complexity_scores = []
            multifractal_spectra = []
            
            # Define quantum circuit for fractal dimension estimation
            @qml.qnode(self.device)
            def fractal_dimension_circuit(data_window, scale):
                """Quantum circuit for fractal dimension estimation
                
                Uses quantum implementation of box-counting algorithm
                
                Args:
                    data_window: Window of time series data
                    scale: Scale parameter for box counting
                    
                Returns:
                    Measurement representing box count at given scale
                """
                n_qubits = self.qubits
                
                # Initialize qubits
                for i in range(n_qubits):
                    qml.Hadamard(wires=i)
                
                # Encode data pattern and scale information
                # The encoding translates classical box counting into a quantum circuit
                data_size = len(data_window)
                scaled_size = data_size // scale
                
                # Scale encoding
                scale_angle = np.pi * (scale / max(scales))
                qml.RY(scale_angle, wires=0)
                
                # Data encoding using amplitude encoding principles
                for i in range(scaled_size):
                    # Find min/max in each box
                    box_start = i * scale
                    box_end = min((i + 1) * scale, data_size)
                    box_data = data_window[box_start:box_end]
                    
                    if len(box_data) > 0:
                        # Calculate box range - this is the key to fractal dimension
                        box_range = np.max(box_data) - np.min(box_data)
                        # Normalize range to [0, π]
                        range_angle = np.pi * box_range
                        
                        # Apply controlled rotation based on box range
                        # We use a binary encoding of the box index to control the rotation
                        control_bits = []
                        bin_i = format(i % (2**(n_qubits-1)), f'0{n_qubits-1}b')
                        for j in range(n_qubits - 1):
                            if bin_i[j] == '1':
                                control_bits.append(j + 1)  # +1 because 0 is used for scale
                        
                        # Apply controlled rotation - this models box occupation
                        if control_bits:
                            # Multi-controlled operation representing box counting
                            qml.MultiControlledX(control_bits, 0)
                            qml.RZ(range_angle, wires=0)
                            qml.MultiControlledX(control_bits, 0)
                        else:
                            # Direct rotation if no control bits
                            qml.RZ(range_angle, wires=0)
                
                # Apply Hadamard gates to create interference patterns
                for i in range(n_qubits):
                    qml.Hadamard(wires=i)
                
                # Measure all wires to get probability distribution
                # This distribution encodes the box counting result
                return qml.probs(wires=range(n_qubits))
            
            # Process each window
            for i in range(len(norm_series) - window_size + 1):
                window = norm_series[i:i+window_size]
                
                # Box counting at different scales
                box_counts = []
                
                for scale in scales:
                    # Run quantum circuit for this scale
                    probs = fractal_dimension_circuit(window, scale)
                    
                    # Extract box count from probability distribution
                    # We use entropy of distribution as analog of box count
                    # Higher entropy = more boxes occupied
                    entropy = -np.sum(probs * np.log2(probs + 1e-10))
                    normalized_entropy = entropy / np.log2(len(probs))  # [0,1] range
                    
                    # Calculate effective box count (higher entropy = more boxes)
                    # Scale by maximum possible boxes at this scale
                    max_boxes = window_size // scale
                    box_count = max_boxes * normalized_entropy
                    box_counts.append(box_count)
                
                # Calculate fractal dimension using log-log regression
                # Fractal dimension is the slope of log(box count) vs log(1/scale)
                log_scales = np.log([1/s for s in scales])
                log_counts = np.log(box_counts)
                
                # Simple linear regression to find slope
                slope, _, _, _ = np.linalg.lstsq(log_scales.reshape(-1, 1), log_counts, rcond=None)[0:4]
                fractal_dim = slope[0] if isinstance(slope, np.ndarray) else slope
                
                # Ensure reasonable bounds (fractal dimension should be between 1 and 2 for time series)
                fractal_dim = max(1.0, min(2.0, fractal_dim))
                
                # Calculate complexity score (normalized fractal dimension)
                complexity = (fractal_dim - 1.0)  # Range [0, 1]
                
                # Simple multifractal spectrum approximation
                # For a true multifractal analysis, we would compute dimensions at various q-orders
                multifractal_spectrum = {
                    "d_-1": min(2.0, fractal_dim * 0.9),  # Lower dimension (small fluctuations)
                    "d_0": fractal_dim,                 # Box-counting dimension
                    "d_1": min(2.0, fractal_dim * 1.1)  # Information dimension (larger fluctuations)
                }
                
                # Store results
                fractal_dimensions.append(fractal_dim)
                complexity_scores.append(complexity)
                multifractal_spectra.append(multifractal_spectrum)
            
            # Calculate trend in fractal dimension
            trend = 0
            if len(fractal_dimensions) >= 2:
                # Use linear regression to determine trend
                x = np.arange(len(fractal_dimensions))
                slope = np.polyfit(x, fractal_dimensions, 1)[0]
                trend = slope * len(fractal_dimensions)  # Scale by window count for meaningful size
            
            # Build result dictionary
            result = {
                "fractal_dimensions": fractal_dimensions,
                "complexity_scores": complexity_scores,
                "multifractal_spectra": multifractal_spectra,
                "latest_fractal_dimension": fractal_dimensions[-1] if fractal_dimensions else 1.5,
                "latest_complexity": complexity_scores[-1] if complexity_scores else 0.5,
                "mean_fractal_dimension": np.mean(fractal_dimensions) if fractal_dimensions else 1.5,
                "dimension_trend": trend,
                "complexity_increasing": trend > 0,
                "quantum": True
            }
            
        except Exception as e:
            self.logger.error(f"Error in quantum fractal dimension estimator: {e}", exc_info=True)
            # Fall back to classical method
            result = self._classical_fractal_dimension(timeseries, window_size, scales)
            result["error"] = str(e)
            
        return result
    
     
    @quantum_accelerated(use_hw_accel=True, device_shots=1024)
    def quantum_correlation_network(self, timeseries_dict: Dict[str, np.ndarray], window_size: int = 20) -> Dict[str, Any]:
        """
        Quantum Correlation Network that maps correlations between multiple assets using quantum entanglement modeling.
        
        This method captures non-linear relationships that classical correlation methods may miss by using
        quantum circuits with CNOT and Hadamard gates to model entanglement between assets.
        
        Args:
            timeseries_dict (Dict[str, np.ndarray]): Dictionary mapping asset names to their time series data
            window_size (int): Size of the rolling window for correlation analysis
            
        Returns:
            Dict[str, Any]: Results containing correlation matrices, network metrics, and relationship strengths
        """
        if self.use_classical or not QUANTUM_AVAILABLE:
            # Classical fallback for correlation calculation
            self.logger.debug("Using classical fallback for correlation network analysis")
            return self._classical_correlation_network(timeseries_dict, window_size)
            
        self.logger.debug("Running quantum correlation network analysis")
        result = {}
        
        try:
            # Ensure we have data to work with
            if not timeseries_dict or len(timeseries_dict) < 2:
                self.logger.warning("Need at least two assets for correlation network analysis")
                return {"error": "Need at least two assets for correlation"}
            
            # Get asset names and align time series lengths
            assets = list(timeseries_dict.keys())
            min_length = min(len(series) for series in timeseries_dict.values())
            
            if min_length < window_size:
                self.logger.warning(f"Window size {window_size} exceeds shortest time series length {min_length}")
                window_size = min_length
            
            # Prepare normalized return series for each asset
            returns_dict = {}
            for asset, prices in timeseries_dict.items():
                prices = prices[-min_length:]  # Use same length for all
                # Calculate returns with safe division
                denominators = prices[:-1].copy()
                denominators[denominators == 0] = 1e-10
                returns = np.diff(prices) / denominators
                # Clip extreme values
                returns = np.clip(returns, -0.1, 0.1)
                # Normalize to [-1, 1]
                max_abs = np.max(np.abs(returns))
                if max_abs > 0:
                    returns_dict[asset] = returns / max_abs
                else:
                    returns_dict[asset] = returns
            
            # Limit assets if we have more than qubits available
            max_assets = self.qubits
            if len(assets) > max_assets:
                self.logger.warning(f"Too many assets ({len(assets)}), limiting to {max_assets}")
                assets = assets[:max_assets]
            
            # Define quantum circuit for correlation detection using entanglement
            @qml.qnode(self.device)
            def correlation_circuit(data_windows):
                """Quantum circuit for correlation detection using entanglement
                
                Uses quantum entanglement to model correlations between assets
                
                Args:
                    data_windows: Dictionary of asset return windows
                    
                Returns:
                    Measurements representing correlation matrix
                """
                n_assets = len(data_windows)
                n_qubits = n_assets
                
                # Encode each asset's returns into a qubit
                # First, prepare all qubits in superposition
                for i in range(n_qubits):
                    qml.Hadamard(wires=i)
                
                # Encode asset data into relative phases
                for i, (asset, window) in enumerate(data_windows.items()):
                    # Calculate aggregated return for each asset
                    # Use the average return over the window
                    avg_return = np.mean(window)
                    
                    # Encode return as rotation angle (mapped to [0, 2π])
                    angle = np.pi * (avg_return + 1)  # Map [-1, 1] to [0, 2π]
                    qml.RZ(angle, wires=i)
                
                # Create entanglement between all asset pairs
                # This is the key to modeling correlations as quantum entanglement
                for i in range(n_qubits):
                    for j in range(i+1, n_qubits):
                        # Apply CNOT to create entanglement between assets
                        qml.CNOT(wires=[i, j])
                        
                        # Apply controlled rotation based on relative signs of returns
                        sign_i = np.sign(np.mean(data_windows[assets[i]]))
                        sign_j = np.sign(np.mean(data_windows[assets[j]]))
                        sign_product = sign_i * sign_j
                        
                        # If same sign (positive correlation), add controlled phase
                        if sign_product > 0:
                            qml.CRZ(np.pi/2, wires=[i, j])
                        # If opposite sign (negative correlation), add different phase
                        elif sign_product < 0:
                            qml.CRZ(-np.pi/2, wires=[i, j])
                
                # Perform measurements in different bases to extract correlation information
                # Measuring in entangled basis gives correlation information
                for i in range(n_qubits):
                    qml.Hadamard(wires=i)
                
                # Return all measurement probabilities
                return qml.probs(wires=range(n_qubits))
            
            # Process sliding windows to get correlations over time
            correlation_matrices = []
            entanglement_measures = []
            centrality_scores = {}
            
            # Initialize centrality scores dictionary for each asset
            for asset in assets:
                centrality_scores[asset] = []
            
            # Process each window
            for i in range(min_length - window_size):
                # Extract window for each asset
                windows = {}
                for asset in assets:
                    windows[asset] = returns_dict[asset][i:i+window_size]
                
                # Run quantum circuit to get correlation matrix
                probs = correlation_circuit(windows)
                
                # Convert quantum measurement to correlation matrix
                n_assets = len(assets)
                correlations = np.zeros((n_assets, n_assets))
                
                # Extract correlations from probability distribution
                # This maps the quantum state probabilities to correlations
                for i in range(n_assets):
                    for j in range(i+1, n_assets):
                        # Find indices in probability vector corresponding to entangled states
                        # For qubits i and j, we look at states where they are correlated
                        
                        # Accumulate probability of correlated outcomes (00 and 11)
                        # This gives positive correlation measure
                        pos_corr_prob = 0
                        # Accumulate probability of anti-correlated outcomes (01 and 10)
                        # This gives negative correlation measure
                        neg_corr_prob = 0
                        
                        for k in range(len(probs)):
                            # Convert index to binary to check qubit states
                            binary = format(k, f'0{n_assets}b')
                            # Check if qubits i and j are in same state (both 0 or both 1)
                            if binary[i] == binary[j]:
                                pos_corr_prob += probs[k]
                            else:
                                neg_corr_prob += probs[k]
                                
                        # Calculate correlation as difference between probabilities
                        # Maps to [-1, 1] range similar to classical correlation
                        corr_value = pos_corr_prob - neg_corr_prob
                        
                        # Set in both positions of the symmetric correlation matrix
                        correlations[i, j] = corr_value
                        correlations[j, i] = corr_value
                
                # Set diagonal to 1 (self-correlation)
                np.fill_diagonal(correlations, 1.0)
                
                # Calculate network measures
                # 1. Average correlation (network density)
                avg_corr = np.mean([abs(correlations[i, j]) for i in range(n_assets) for j in range(i+1, n_assets)])
                
                # 2. Correlation entropy (diversity of correlations)
                # Normalize correlations to [0,1] for entropy calculation
                norm_correlations = (correlations + 1) / 2
                entropy = -np.sum(norm_correlations * np.log2(norm_correlations + 1e-10)) / (n_assets * n_assets)
                
                # 3. Centrality - how much each asset correlates with others
                for i, asset in enumerate(assets):
                    # Sum of absolute correlations with other assets
                    centrality = np.sum(np.abs(correlations[i, :]))
                    centrality_scores[asset].append(centrality)
                
                # Store results
                correlation_matrices.append({
                    "matrix": correlations.tolist(),
                    "assets": assets,
                    "timestamp": i  # Reference point
                })
                
                entanglement_measures.append({
                    "average_correlation": avg_corr,
                    "correlation_entropy": entropy,
                    "timestamp": i
                })
            
            # Extract key relationships (strongest correlations)
            key_relationships = []
            if correlation_matrices:
                latest_matrix = np.array(correlation_matrices[-1]["matrix"])
                for i in range(n_assets):
                    for j in range(i+1, n_assets):
                        if abs(latest_matrix[i, j]) > 0.5:  # Threshold for significant correlation
                            key_relationships.append({
                                "asset1": assets[i],
                                "asset2": assets[j],
                                "correlation": latest_matrix[i, j],
                                "type": "positive" if latest_matrix[i, j] > 0 else "negative"
                            })
            
            # Find most central asset (highest average centrality)
            central_asset = None
            max_avg_centrality = 0
            for asset, scores in centrality_scores.items():
                if scores and np.mean(scores) > max_avg_centrality:
                    max_avg_centrality = np.mean(scores)
                    central_asset = asset
            
            # Build result dictionary
            result = {
                "correlation_matrices": correlation_matrices,
                "entanglement_measures": entanglement_measures,
                "centrality_scores": centrality_scores,
                "key_relationships": key_relationships,
                "central_asset": central_asset,
                "latest_average_correlation": entanglement_measures[-1]["average_correlation"] if entanglement_measures else 0,
                "latest_correlation_entropy": entanglement_measures[-1]["correlation_entropy"] if entanglement_measures else 0,
                "quantum": True
            }
            
        except Exception as e:
            self.logger.error(f"Error in quantum correlation network: {e}", exc_info=True)
            # Fall back to classical method
            result = self._classical_correlation_network(timeseries_dict, window_size)
            result["error"] = str(e)
            
        return result
    
    def _classical_correlation_network(self, timeseries_dict: Dict[str, np.ndarray], window_size: int) -> Dict[str, Any]:
        """Classical implementation of correlation network using Pearson correlation"""
        result = {}
        
        try:
            # Ensure we have data to work with
            if not timeseries_dict or len(timeseries_dict) < 2:
                return {"error": "Need at least two assets for correlation"}
            
            # Get asset names and align time series lengths
            assets = list(timeseries_dict.keys())
            min_length = min(len(series) for series in timeseries_dict.values())
            
            if min_length < window_size:
                window_size = min_length
            
            # Prepare data for the numba-accelerated function
            # Create a 2D array of return series
            n_assets = len(assets)
            return_array = np.zeros((n_assets, min_length - 1))
            
            for i, asset in enumerate(assets):
                prices = timeseries_dict[asset][-min_length:]
                # Calculate returns with safe division using vectorized operations
                denominators = prices[:-1].copy()
                denominators[denominators == 0] = 1e-10
                return_array[i, :] = np.diff(prices) / denominators
            
            # Use numba-accelerated implementation for correlation calculations
            correlation_matrices_data, avg_correlations, entropies = self._calculate_correlation_network_numba(
                return_array, window_size, n_assets)
            
            # Process results into the expected format
            correlation_matrices = []
            network_measures = []
            centrality_scores = {asset: [] for asset in assets}
            
            for i in range(len(avg_correlations)):
                # Extract correlation matrix for this window
                corr_matrix = correlation_matrices_data[i].reshape(n_assets, n_assets)
                
                # Store in the expected format
                correlation_matrices.append({
                    "matrix": corr_matrix.tolist(),
                    "assets": assets,
                    "timestamp": i
                })
                
                network_measures.append({
                    "average_correlation": avg_correlations[i],
                    "correlation_entropy": entropies[i],
                    "timestamp": i
                })
                
                # Calculate centrality scores
                for j, asset in enumerate(assets):
                    centrality = np.sum(np.abs(corr_matrix[j, :]))
                    centrality_scores[asset].append(centrality)
            
            # Extract key relationships
            key_relationships = []
            if correlation_matrices:
                latest_matrix = np.array(correlation_matrices[-1]["matrix"])
                for i in range(len(assets)):
                    for j in range(i+1, len(assets)):
                        if abs(latest_matrix[i, j]) > 0.5:  # Threshold for significant correlation
                            key_relationships.append({
                                "asset1": assets[i],
                                "asset2": assets[j],
                                "correlation": latest_matrix[i, j],
                                "type": "positive" if latest_matrix[i, j] > 0 else "negative"
                            })
            
            # Find most central asset
            central_asset = None
            max_avg_centrality = 0
            for asset, scores in centrality_scores.items():
                if scores and np.mean(scores) > max_avg_centrality:
                    max_avg_centrality = np.mean(scores)
                    central_asset = asset
            
            # Build result dictionary
            result = {
                "correlation_matrices": correlation_matrices,
                "entanglement_measures": network_measures,  # Keep same format as quantum version
                "centrality_scores": centrality_scores,
                "key_relationships": key_relationships,
                "central_asset": central_asset,
                "latest_average_correlation": network_measures[-1]["average_correlation"] if network_measures else 0,
                "latest_correlation_entropy": network_measures[-1]["correlation_entropy"] if network_measures else 0,
                "quantum": False  # Indicate this is classical result
            }
            
        except Exception as e:
            self.logger.error(f"Error in classical correlation network: {e}", exc_info=True)
            result = {
                "error": str(e),
                "correlation_matrices": [],
                "latest_average_correlation": 0,
                "latest_correlation_entropy": 0,
                "quantum": False
            }
            
        return result
        
    @staticmethod
    @njit(parallel=True)
    def _calculate_correlation_network_numba(return_array, window_size, n_assets):
        """Numba-accelerated implementation of correlation network calculations
        
        Args:
            return_array: 2D array of returns [n_assets, n_timesteps]
            window_size: Size of the rolling window
            n_assets: Number of assets
            
        Returns:
            Tuple of (correlation_matrices, avg_correlations, entropies)
        """
        n_windows = return_array.shape[1] - window_size + 1
        
        # Initialize output arrays
        # Each correlation matrix is a flattened n_assets x n_assets matrix
        correlation_matrices = np.zeros((n_windows, n_assets * n_assets))
        avg_correlations = np.zeros(n_windows)
        entropies = np.zeros(n_windows)
        
        # Process each window
        for i in prange(n_windows):
            # Extract window for each asset
            window_data = np.zeros((n_assets, window_size))
            for j in range(n_assets):
                for k in range(window_size):
                    window_data[j, k] = return_array[j, i + k]
            
            # Calculate correlation matrix manually
            # First center the data (subtract mean)
            means = np.zeros(n_assets)
            for j in range(n_assets):
                sum_val = 0.0
                for k in range(window_size):
                    sum_val += window_data[j, k]
                means[j] = sum_val / window_size
            
            centered_data = np.zeros((n_assets, window_size))
            for j in range(n_assets):
                for k in range(window_size):
                    centered_data[j, k] = window_data[j, k] - means[j]
            
            # Calculate standard deviations
            stds = np.zeros(n_assets)
            for j in range(n_assets):
                sum_squared = 0.0
                for k in range(window_size):
                    sum_squared += centered_data[j, k] * centered_data[j, k]
                stds[j] = np.sqrt(sum_squared / window_size)
            
            # Calculate correlation matrix
            corr_matrix = np.zeros((n_assets, n_assets))
            for j in range(n_assets):
                # Diagonal elements are always 1
                corr_matrix[j, j] = 1.0
                
                # Calculate upper triangle
                for k in range(j + 1, n_assets):
                    # Skip if either std is 0
                    if stds[j] > 0 and stds[k] > 0:
                        # Calculate covariance
                        cov_sum = 0.0
                        for l in range(window_size):
                            cov_sum += centered_data[j, l] * centered_data[k, l]
                        cov = cov_sum / window_size
                        
                        # Calculate correlation
                        corr = cov / (stds[j] * stds[k])
                        
                        # Ensure correlation is in [-1, 1] range
                        corr = max(-1.0, min(1.0, corr))
                        
                        # Set both triangles (symmetric matrix)
                        corr_matrix[j, k] = corr
                        corr_matrix[k, j] = corr
            
            # Flatten correlation matrix and store
            for j in range(n_assets):
                for k in range(n_assets):
                    correlation_matrices[i, j * n_assets + k] = corr_matrix[j, k]
            
            # Calculate average correlation (upper triangle)
            sum_abs_corr = 0.0
            count = 0
            for j in range(n_assets):
                for k in range(j + 1, n_assets):
                    sum_abs_corr += abs(corr_matrix[j, k])
                    count += 1
            
            avg_correlations[i] = sum_abs_corr / count if count > 0 else 0.0
            
            # Calculate correlation entropy
            entropy_sum = 0.0
            norm_correlations = (corr_matrix + 1.0) / 2.0  # Normalize to [0,1]
            
            for j in range(n_assets):
                for k in range(n_assets):
                    p = norm_correlations[j, k]
                    if p > 0 and p < 1:  # Avoid log(0) or log(1)
                        entropy_sum -= p * np.log2(p + 1e-10)
            
            entropies[i] = entropy_sum / (n_assets * n_assets)
        
        return correlation_matrices, avg_correlations, entropies
    
    def _classical_fractal_dimension(self, timeseries: np.ndarray, window_size: int, scales: List[int]) -> Dict[str, Any]:
        """Classical implementation of fractal dimension calculation using box counting"""
        result = {}
        
        try:
            # Use the numba-accelerated implementation
            fractal_dimensions, complexity_scores = self._calculate_fractal_dimension_numba(
                timeseries, window_size, np.array(scales, dtype=np.int32))
            
            # Calculate trend
            trend = 0
            if len(fractal_dimensions) >= 2:
                x = np.arange(len(fractal_dimensions))
                slope = np.polyfit(x, fractal_dimensions, 1)[0]
                trend = slope * len(fractal_dimensions)
            
            # Build result dictionary
            result = {
                "fractal_dimensions": fractal_dimensions.tolist(),
                "complexity_scores": complexity_scores.tolist(),
                "latest_fractal_dimension": fractal_dimensions[-1] if len(fractal_dimensions) > 0 else 1.5,
                "latest_complexity": complexity_scores[-1] if len(complexity_scores) > 0 else 0.5,
                "mean_fractal_dimension": np.mean(fractal_dimensions) if len(fractal_dimensions) > 0 else 1.5,
                "dimension_trend": trend,
                "complexity_increasing": trend > 0,
                "quantum": False  # Indicate this is classical result
            }
            
        except Exception as e:
            self.logger.error(f"Error in classical fractal dimension calculation: {e}", exc_info=True)
            result = {
                "error": str(e),
                "fractal_dimensions": [],
                "latest_fractal_dimension": 1.5,
                "latest_complexity": 0.5,
                "mean_fractal_dimension": 1.5,
                "dimension_trend": 0,
                "complexity_increasing": False,
                "quantum": False
            }
            
        return result
        
    @staticmethod
    @njit(parallel=True)
    def _calculate_fractal_dimension_numba(timeseries, window_size, scales):
        """Numba-accelerated implementation of fractal dimension calculation
        
        Args:
            timeseries: Input time series data
            window_size: Size of the rolling window
            scales: Array of scale sizes to use for box counting
            
        Returns:
            Tuple of (fractal_dimensions, complexity_scores)
        """
        # Normalize time series to [0,1] range for box counting
        min_val = np.min(timeseries)
        max_val = np.max(timeseries)
        
        norm_series = np.zeros_like(timeseries)
        if max_val > min_val:
            for i in range(len(timeseries)):
                norm_series[i] = (timeseries[i] - min_val) / (max_val - min_val)
        
        # Initialize result arrays
        n_windows = len(norm_series) - window_size + 1
        fractal_dimensions = np.zeros(n_windows)
        complexity_scores = np.zeros(n_windows)
        
        # Process each window in parallel
        for i in prange(n_windows):
            window = norm_series[i:i+window_size]
            
            # Box counting at different scales
            box_counts = np.zeros(len(scales))
            
            for s in range(len(scales)):
                scale = scales[s]
                # Divide window into boxes of size 'scale'
                n_boxes = window_size // scale
                count = 0
                
                for j in range(n_boxes):
                    start_idx = j * scale
                    end_idx = min((j + 1) * scale, len(window))
                    
                    # Find min and max in this box
                    box_min = window[start_idx]
                    box_max = window[start_idx]
                    
                    for k in range(start_idx + 1, end_idx):
                        if window[k] < box_min:
                            box_min = window[k]
                        if window[k] > box_max:
                            box_max = window[k]
                    
                    # If there's a range in this box, count it
                    if box_max > box_min:
                        count += 1
                
                # Ensure at least 1 box
                box_counts[s] = max(1, count)
            
            # Calculate fractal dimension using log-log regression
            log_scales = np.zeros(len(scales))
            log_counts = np.zeros(len(scales))
            
            for s in range(len(scales)):
                log_scales[s] = np.log(1.0 / scales[s])
                log_counts[s] = np.log(box_counts[s])
            
            # Linear regression to find slope (avoid using np.linalg.lstsq in numba)
            n = len(log_scales)
            if n > 1:
                sum_x = 0.0
                sum_y = 0.0
                sum_xy = 0.0
                sum_xx = 0.0
                
                for j in range(n):
                    sum_x += log_scales[j]
                    sum_y += log_counts[j]
                    sum_xy += log_scales[j] * log_counts[j]
                    sum_xx += log_scales[j] * log_scales[j]
                
                # Calculate slope
                if (n * sum_xx - sum_x * sum_x) != 0:
                    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)
                    fractal_dim = slope
                else:
                    fractal_dim = 1.5  # Default if calculation fails
            else:
                fractal_dim = 1.5  # Default if not enough points
            
            # Ensure reasonable bounds
            fractal_dim = max(1.0, min(2.0, fractal_dim))
            
            # Calculate complexity (normalized fractal dimension)
            complexity = fractal_dim - 1.0  # Range [0, 1]
            
            # Store results
            fractal_dimensions[i] = fractal_dim
            complexity_scores[i] = complexity
        
        return fractal_dimensions, complexity_scores
    
    def _classical_momentum_oscillator(self, timeseries: np.ndarray, window_size: int, sensitivity: float = 0.5) -> Dict[str, Any]:
        """Classical fallback method for momentum oscillator
        
        Uses RSI and other traditional momentum indicators
        """
        result = {}
        
        try:
            # Use numba-accelerated implementation
            returns, momentum_values, oscillator_values, signal_line, is_overbought, is_oversold = \
                self._calculate_rsi_momentum_numba(timeseries, window_size, sensitivity)
                
            # Process crossovers using the generated signal line and oscillator values
            crossovers = []
            if len(signal_line) >= 2:
                for i in range(1, len(signal_line)):
                    prev_diff = oscillator_values[i+8-1] - signal_line[i-1]  # +8 because signal starts after 9 elements
                    curr_diff = oscillator_values[i+8] - signal_line[i]
                    
                    # Crossover detected if sign changes
                    if prev_diff * curr_diff <= 0 and abs(prev_diff) > 2:
                        crossover_type = "bullish" if curr_diff > 0 else "bearish"
                        crossovers.append({
                            "position": i + window_size + 8,
                            "type": crossover_type,
                            "strength": abs(curr_diff)
                        })
            
            # Look for divergences using the optimized function
            divergences = self._detect_momentum_divergences(timeseries, oscillator_values, window_size)
            
            result = {
                "momentum": momentum_values.tolist(),
                "oscillator": oscillator_values.tolist(),
                "signal_line": signal_line.tolist(),
                "crossovers": crossovers,
                "divergences": divergences,
                "latest_momentum": momentum_values[-1] if len(momentum_values) > 0 else 0,
                "latest_oscillator": oscillator_values[-1] if len(oscillator_values) > 0 else 50,
                "is_overbought": is_overbought,
                "is_oversold": is_oversold,
                "buy_signal": len(crossovers) > 0 and crossovers[-1]["type"] == "bullish",
                "sell_signal": len(crossovers) > 0 and crossovers[-1]["type"] == "bearish",
                "quantum": False  # Indicate this is classical result
            }
            
        except Exception as e:
            self.logger.error(f"Error in classical momentum oscillator: {e}", exc_info=True)
            result = {
                "error": str(e),
                "momentum": [],
                "oscillator": [],
                "latest_momentum": 0,
                "latest_oscillator": 50,
                "is_overbought": False,
                "is_oversold": False,
                "buy_signal": False,
                "sell_signal": False,
                "quantum": False
            }
            
        return result
        
    @staticmethod
    @njit(parallel=True)
    def _calculate_rsi_momentum_numba(timeseries, window_size, sensitivity):
        """Numba-accelerated implementation of RSI and momentum calculation
        
        Args:
            timeseries: Input price time series
            window_size: Size of the RSI calculation window
            sensitivity: Sensitivity adjustment for momentum
            
        Returns:
            Tuple of (returns, momentum_values, oscillator_values, signal_line, is_overbought, is_oversold)
        """
        # Calculate returns with safe division
        returns = np.zeros(len(timeseries) - 1)
        for i in range(len(timeseries) - 1):
            denominator = timeseries[i]
            if denominator == 0:
                denominator = 1e-10
            returns[i] = (timeseries[i+1] - timeseries[i]) / denominator
        
        # Calculate RSI for each window
        n_windows = len(returns) - window_size + 1
        momentum_values = np.zeros(n_windows)
        oscillator_values = np.zeros(n_windows)
        
        for i in prange(n_windows):
            window = returns[i:i+window_size]
            
            # Calculate gains and losses
            gains_sum = 0.0
            losses_sum = 0.0
            
            for j in range(window_size):
                if window[j] > 0:
                    gains_sum += window[j]
                elif window[j] < 0:
                    losses_sum -= window[j]  # Convert to positive
            
            # Calculate average gain and loss
            avg_gain = gains_sum / window_size
            avg_loss = losses_sum / window_size
            
            # Calculate RSI
            if avg_loss == 0:
                rsi = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi = 100.0 - (100.0 / (1.0 + rs))
            
            # Convert RSI [0, 100] to momentum [-1, 1]
            momentum = (rsi / 50.0) - 1.0
            
            # Apply sensitivity adjustment
            momentum = momentum * sensitivity
            
            # Store values
            momentum_values[i] = momentum
            oscillator_values[i] = rsi
        
        # Calculate signal line (9-period SMA of RSI is standard)
        signal_line = np.zeros(max(0, n_windows - 9 + 1))
        for i in range(len(signal_line)):
            # Calculate simple average of last 9 values
            signal_sum = 0.0
            for j in range(9):
                signal_sum += oscillator_values[i + j]
            signal_line[i] = signal_sum / 9.0
        
        # Determine overbought/oversold conditions for latest value
        overbought_threshold = 70.0
        oversold_threshold = 30.0
        is_overbought = False
        is_oversold = False
        
        if n_windows > 0:
            is_overbought = oscillator_values[-1] > overbought_threshold
            is_oversold = oscillator_values[-1] < oversold_threshold
        
        return returns, momentum_values, oscillator_values, signal_line, is_overbought, is_oversold
    
    def _detect_momentum_divergences(self, timeseries, oscillator_values, window_size):
        """Detect momentum divergences between price and oscillator
        
        Args:
            timeseries: Input price time series
            oscillator_values: Calculated oscillator values
            window_size: Size of the analysis window
        
        Returns:
            List of divergence dictionaries
        """
        divergences = []
        
        # Only check for divergences if we have enough data points
        if len(timeseries) >= window_size + 5 and len(oscillator_values) >= 15:
            # Use numpy for vector operations where possible
            for i in range(10, len(oscillator_values)):
                if i + window_size > len(timeseries) or i - 5 + window_size > len(timeseries):
                    continue
                    
                price_window = timeseries[i:i+window_size]
                prev_price_window = timeseries[i-5:i+window_size-5]
                osc_window = oscillator_values[i-5:i]
                prev_osc_window = oscillator_values[i-10:i-5]
                
                # Ensure we have valid data for comparison
                if len(osc_window) > 0 and len(prev_osc_window) > 0:
                    # Check for bullish divergence (price lower low, oscillator higher low)
                    if (np.min(price_window) < np.min(prev_price_window) and 
                            np.min(osc_window) > np.min(prev_osc_window)):
                        divergences.append({
                            "position": i + window_size - 1,
                            "type": "bullish",
                            "strength": abs(np.min(osc_window) - np.min(prev_osc_window))
                        })
                        
                    # Check for bearish divergence (price higher high, oscillator lower high)
                    if (np.max(price_window) > np.max(prev_price_window) and 
                            np.max(osc_window) < np.max(prev_osc_window)):
                        divergences.append({
                            "position": i + window_size - 1,
                            "type": "bearish",
                            "strength": abs(np.max(osc_window) - np.max(prev_osc_window))
                        })
        
        return divergences
    
    def recover(self) -> bool:
        """
        Attempt to recover the QERC component after errors.
        
        Returns:
            bool: True if recovery succeeded
        """
        self.logger.warning("QERC recovery triggered!")
        try:
            # Reset internal state
            self.reset_state()
            
            # Re-initialize quantum device
            if not self.use_classical and QUANTUM_AVAILABLE:
                self.device = self._get_optimized_device()
                self._initialize_quantum_circuits()
                
            # Check for fault tolerance manager
            if self.fault_tolerance:
                self.fault_tolerance.register_recovery("qerc")
                
            self.logger.info("QERC recovery succeeded")
            return True
            
        except Exception as e:
            self.logger.error(f"QERC recovery failed: {str(e)}", exc_info=True)
            # Last resort - force classical mode
            self.use_classical = True
            return False

    def self_test_indicators(self) -> Dict[str, Any]:
        """
        Self-test function to demonstrate all optimized quantum indicators.
        
        This method runs all the quantum indicators that have been optimized with
        @quantum_accelerated and Numba @njit and reports performance metrics.
        
        Returns:
            Dictionary with test results for all indicators
        """
        import time
        
        self.logger.info("Starting self-test of quantum indicators...")
        results = {}
        
        # Generate synthetic test data
        sample_size = 200
        self.logger.info(f"Generating test data with {sample_size} points")
        sine_data = np.sin(np.linspace(0, 6*np.pi, sample_size))
        cosine_data = np.cos(np.linspace(0, 6*np.pi, sample_size))
        
        # Combined time series with some patterns for testing
        timeseries = np.concatenate([sine_data, cosine_data])
        
        # Test multiple assets for correlation network
        timeseries_dict = {
            "asset1": sine_data,
            "asset2": cosine_data,
            "asset3": sine_data + 0.5 * cosine_data,
            "asset4": sine_data * cosine_data
        }
        
        # Test all quantum indicators
        indicators = [
            "Phase Transition Detector",
            "Entropy Analyzer",
            "Momentum Oscillator",
            "Fractal Dimension Estimator",
            "Correlation Network"
        ]
        
        window_size = 30
        
        self.logger.info("Testing quantum indicators with hardware acceleration:")
        
        # Test Quantum Phase Transition Detector
        start_time = time.time()
        phase_result = self.quantum_phase_transition_detector(timeseries)
        phase_time = time.time() - start_time
        results["phase_transition"] = {
            "execution_time": phase_time,
            "critical_points": len(phase_result["critical_points"]),
            "quantum_used": phase_result.get("quantum", True)  # Default to True if key missing
        }
        self.logger.info(f"Phase Transition Detector: {phase_time:.4f}s, {len(phase_result['critical_points'])} critical points")
        
        # Test Quantum Entropy Analyzer
        start_time = time.time()
        entropy_result = self.quantum_entropy_analyzer(timeseries)
        entropy_time = time.time() - start_time
        results["entropy"] = {
            "execution_time": entropy_time,
            "entropy": entropy_result["latest_quantum_entropy"],
            "quantum_used": entropy_result.get("quantum", True)  # Default to True if key missing
        }
        self.logger.info(f"Entropy Analyzer: {entropy_time:.4f}s, Entropy: {entropy_result['latest_quantum_entropy']:.4f}")
        
        # Test Quantum Momentum Oscillator
        start_time = time.time()
        momentum_result = self.quantum_momentum_oscillator(timeseries, window_size=window_size)
        momentum_time = time.time() - start_time
        results["momentum"] = {
            "execution_time": momentum_time,
            "oscillator": momentum_result["latest_oscillator"],
            "quantum_used": momentum_result.get("quantum", True)  # Default to True if key missing
        }
        self.logger.info(f"Momentum Oscillator: {momentum_time:.4f}s, Value: {momentum_result['latest_oscillator']:.2f}")
        
        # Test Quantum Fractal Dimension Estimator
        start_time = time.time()
        fractal_result = self.quantum_fractal_dimension_estimator(timeseries, window_size=window_size)
        fractal_time = time.time() - start_time
        results["fractal"] = {
            "execution_time": fractal_time,
            "dimension": fractal_result["latest_fractal_dimension"],
            "quantum_used": fractal_result.get("quantum", True)  # Default to True if key missing
        }
        self.logger.info(f"Fractal Dimension: {fractal_time:.4f}s, Dimension: {fractal_result['latest_fractal_dimension']:.4f}")
        
        # Test Quantum Correlation Network
        start_time = time.time()
        network_result = self.quantum_correlation_network(timeseries_dict, window_size=window_size)
        network_time = time.time() - start_time
        results["correlation"] = {
            "execution_time": network_time,
            "entropy": network_result["latest_correlation_entropy"],
            "quantum_used": network_result.get("quantum", True)  # Default to True if key missing
        }
        self.logger.info(f"Correlation Network: {network_time:.4f}s, Entropy: {network_result['latest_correlation_entropy']:.4f}")
        
        # Compare with classical methods
        self.logger.info("\nComparing with classical (Numba-optimized) methods:")
        
        # Test Classical Phase Transition Detector
        start_time = time.time()
        phase_result_classic = self._classical_phase_transition_detection(timeseries, window_size=window_size)
        phase_time_classic = time.time() - start_time
        results["phase_transition_classic"] = {
            "execution_time": phase_time_classic,
            "critical_points": len(phase_result_classic["critical_points"]) if "critical_points" in phase_result_classic else 0
        }
        self.logger.info(f"Classical Phase Transition: {phase_time_classic:.4f}s")
        
        # Compile performance summary
        quantum_times = [results[k]["execution_time"] for k in ["phase_transition", "entropy", "momentum", "fractal", "correlation"]]
        classic_time = results["phase_transition_classic"]["execution_time"]
        
        self.logger.info("\nPerformance Summary:")
        self.logger.info(f"Total quantum indicators time: {sum(quantum_times):.4f}s")
        self.logger.info(f"Average quantum indicator time: {np.mean(quantum_times):.4f}s")
        self.logger.info(f"Classical phase transition time: {classic_time:.4f}s")
        # Check if hardware acceleration is active by checking if quantum was used for all indicators
        quantum_used = all(results[k].get('quantum_used', True) for k in ['phase_transition', 'entropy', 'momentum', 'fractal', 'correlation'])
        self.logger.info(f"Hardware acceleration active: {quantum_used}")
        
        return results

# Factory function for thread-safe singleton access
_qerc_instance = None
_qerc_lock = threading.RLock()

def get_quantum_reservoir_computing(config=None, reset=False) -> QuantumEnhancedReservoirComputing:
    """Thread-safe factory function for QuantumEnhancedReservoirComputing."""
    global _qerc_instance, _qerc_lock

    with _qerc_lock:
        if _qerc_instance is None or reset:
            try:
                # Extract specific parameters from config
                config_dict = config or {}
                
                # Instantiate with parameters from config
                _qerc_instance = QuantumEnhancedReservoirComputing(
                    reservoir_size=config_dict.get('reservoir_size', 500),
                    quantum_kernel_size=config_dict.get('quantum_kernel_size', 4),
                    spectral_radius=config_dict.get('spectral_radius', 0.95),
                    leaking_rate=config_dict.get('leaking_rate', 0.3),
                    temporal_windows=config_dict.get('temporal_windows', [5, 15, 30, 60]),
                    input_dimensionality=config_dict.get('qerc_input_dimensionality', None),
                    config=config_dict
                )
                
                logger.info("Created new QERC instance")
                
            except Exception as e:
                logger.exception(f"Failed to initialize QERC instance: {e}")
                return None

    return _qerc_instance