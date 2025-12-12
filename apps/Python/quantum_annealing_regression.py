#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quantum Annealing Regression

A sophisticated quantum annealing implementation for time series forecasting that:
- Leverages real quantum hardware via PennyLane when available
- Uses GPU acceleration for matrix operations
- Falls back to Hamiltonian simulation when quantum hardware is unavailable
- Supports multiple market regimes and adaptive parameters
- Provides comprehensive metrics and diagnostics
- Implements thread-safety and efficient caching

This implementation merges features from multiple quantum annealing approaches
and optimizes for hardware acceleration where possible.

Author: Merged implementation based on existing quantum annealing code
"""

import os
import time
import logging
import warnings
import threading
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Callable, Tuple, Set
import math
from functools import wraps
import json
from enum import Enum, auto


# Optional imports
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. GPU acceleration will be limited.")

# PennyLane for quantum computing
try:
    import pennylane as qml
    from pennylane import numpy as qnp
    
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
        
    PENNYLANE_AVAILABLE = True
except ImportError:
    qml = None
    qnp = np
    PENNYLANE_AVAILABLE = False
    warnings.warn("PennyLane not available. Quantum path disabled.")

# Local imports - using try/except to handle potential import errors
try:
    from hardware_manager import HardwareManager
    from cdfa_extensions.hw_acceleration import HardwareAccelerator, AcceleratorType
    HARDWARE_ACCEL_AVAILABLE = True
except ImportError:
    HardwareManager = object
    HardwareAccelerator = object
    AcceleratorType = object
    HARDWARE_ACCEL_AVAILABLE = False
    warnings.warn("Hardware acceleration modules not available. Using fallback implementation.")

try:
    from cache_manager import CircularBuffer
except ImportError:
    # Simplified CircularBuffer implementation if the import fails
    class CircularBuffer:
        def __init__(self, max_size=100):
            self.max_size = max_size
            self.buffer = {}
            self._keys = []
            
        def __setitem__(self, key, value):
            if key not in self.buffer and len(self._keys) >= self.max_size:
                # Remove oldest item
                old_key = self._keys.pop(0)
                del self.buffer[old_key]
            
            self.buffer[key] = value
            if key not in self._keys:
                self._keys.append(key)
                
        def __getitem__(self, key):
            return self.buffer[key]
            
        def __contains__(self, key):
            return key in self.buffer

try:
    from fault_manager import FaultToleranceManager, FaultSeverity, FaultCategory
except ImportError:
    # Simple FaultToleranceManager if the import fails
    class FaultToleranceManager:
        def __init__(self):
            self.error_count = 0
            self.max_retries = 3
            
        def report_fault(self, component_name, message, exception=None, severity=None, category=None):
            self.error_count += 1
            logging.warning(f"Error in component {component_name}: {message}")
            
        def should_retry(self):
            return self.error_count < self.max_retries
            
    # Define enum classes if needed for fallback
    
    class FaultSeverity(Enum):
        INFO = auto()
        WARNING = auto() 
        ERROR = auto()
        CRITICAL = auto()
        FATAL = auto()
        
    class FaultCategory(Enum):
        HARDWARE = auto()
        QUANTUM = auto()
        COMPUTATION = auto()
        UNKNOWN = auto()

try:
    from iqad import get_immune_quantum_anomaly_detector
except ImportError:
    # Dummy IQAD function if the import fails
    def get_immune_quantum_anomaly_detector():
        class DummyIQAD:
            def detect_anomalies(self, data):
                return {"detected": False, "score": 0.0}
        return DummyIQAD()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("quantum_annealing.log"), logging.StreamHandler()],
)
logger = logging.getLogger("QuantumAnnealingRegression")


def quantum_accelerated(wires: int = 8, shots: int = None):
    """
    Decorator to enable quantum acceleration for functions when available
    
    Args:
        wires: Number of qubits to use
        shots: Number of measurement shots (None for exact simulation)
        
    Returns:
        Decorated function with quantum acceleration
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Check if we can use PennyLane
            if not PENNYLANE_AVAILABLE:
                return func(*args, **kwargs)
                
            # Create device if not provided
            if "device" not in kwargs or kwargs["device"] is None:
                try:
                    # Try to get device from class if it's a method
                    if len(args) > 0 and hasattr(args[0], 'hw_manager'):
                        hw_manager = args[0].hw_manager
                        device_config = hw_manager._get_quantum_device(wires)
                        device = qml.device(
                            device_config['device'],
                            wires=wires,
                            shots=device_config.get('shots', shots)
                        )
                        kwargs["device"] = device
                    else:
                        # Fallback to default device
                        device = qml.device("lightning.qubit", wires=wires, shots=shots)
                        kwargs["device"] = device
                except Exception as e:
                    logger.warning(f"Could not create quantum device: {e}")
                    return func(*args, **kwargs)
                    
            return func(*args, **kwargs)
            
        return wrapper
    return decorator


class QuantumAnnealingRegression:
    """
    Quantum Annealing Regression for time-series forecasting with hardware acceleration.
    
    Features:
    - Hamiltonian encoding of price trajectories
    - Adiabatic quantum computation (via hardware or simulation)
    - Multi-timeframe quantum regression models
    - GPU/CPU acceleration for matrix operations
    - Market regime adaptation
    - Anomaly detection integration
    
    This implementation merges features from multiple quantum annealing approaches
    and optimizes for hardware acceleration using available GPUs or quantum processors.
    """
    
    def __init__(self,
                hw_manager: Optional[HardwareManager] = None,
                hw_accelerator: Optional[HardwareAccelerator] = None,
                window_size: int = 20,
                forecast_horizon: int = 5,
                n_features: int = 1,
                annealing_steps: int = 100,
                annealing_repeats: int = 5,
                learning_rate: float = 0.01,
                iqad=None,
                config: Optional[Dict[str, Any]] = None):
        """
        Initialize Quantum Annealing Regression with hardware management.
        
        Args:
            hw_manager: HardwareManager instance for quantum device management
            hw_accelerator: HardwareAccelerator for GPU acceleration
            window_size: Window size for time series analysis
            forecast_horizon: Number of steps to forecast
            n_features: Number of features to use
            annealing_steps: Number of steps in annealing schedule
            annealing_repeats: Number of annealing runs to perform
            learning_rate: Learning rate for gradient-based fitting
            iqad: Immune Quantum Anomaly Detector instance
            config: Additional configuration parameters
        """
        # Initialize configuration
        self.config = config or {}
        
        # Initialize hardware management
        self._init_hardware(hw_manager, hw_accelerator)
        
        # Initialize fault tolerance and anomaly detection
        self.fault_manager = FaultToleranceManager()
        # Add this to the end of the __init__ method:
        # Register with fault tolerance manager if possible
        if self.fault_manager is not None and hasattr(self.fault_manager, 'register_component'):
            try:
                # Check what parameters register_component accepts
                import inspect
                register_params = inspect.signature(self.fault_manager.register_component).parameters
                
                # Basic registration - component name and instance only
                if len(register_params) >= 2:
                    self.fault_manager.register_component("QuantumAnnealingRegression", self)
            except Exception as e:
                logger.debug(f"Could not register with fault manager: {e}")
        self.iqad = iqad or get_immune_quantum_anomaly_detector()
        
        # Initialize core parameters
        self.window_size = window_size
        self.forecast_horizon = forecast_horizon
        self.n_features = n_features
        self.annealing_steps = annealing_steps
        self.annealing_repeats = annealing_repeats
        self.learning_rate = learning_rate
        
        # Initialize market regime parameters
        self.regime_params = {
            'bull': {'temp_schedule': [2.0, 1.0, 0.5], 'kernel_size': 7},
            'bear': {'temp_schedule': [1.5, 0.8, 0.3], 'kernel_size': 5},
            'volatile': {'temp_schedule': [3.0, 1.5, 0.7], 'kernel_size': 3},
            'neutral': {'temp_schedule': [2.0, 1.0, 0.5], 'kernel_size': 5}
        }
        self.current_regime = "neutral"
        self.temp_scale = 1.0
        self.kernel_size = self.regime_params[self.current_regime]['kernel_size']
        
        # Initialize Hamiltonian components
        self.hamiltonians = self._initialize_hamiltonians()
        
        # Initialize timeframe-specific models
        self.models = self._initialize_regression_models()
        
        # Initialize states for fitting/training
        self.weights = None
        self.bias = None
        self.circuit_weights = None
        
        # Setup caching with thread safety
        self.cache = CircularBuffer(max_size=self.config.get('cache_size', 100))
        self.cache_lock = threading.RLock()
        
        # Initialize results tracking
        self.forecast_history = []
        self.prediction_errors = []
        self.training_history = []
        self.max_history = self.config.get('max_history', 100)
        
        # Initialize circuits for quantum path
        self._initialize_circuits()
        
        logger.info(
            f"Initialized QuantumAnnealingRegression with {self.window_size} window size, "
            f"{self.forecast_horizon} horizon, quantum: {self.quantum_available}, "
            f"GPU: {self.gpu_available}, max_qubits: {self.max_qubits}"
        )
    
    def _init_hardware(self, hw_manager, hw_accelerator):
        """
        Initialize hardware management components.
        """
        # Set up hardware management
        if hw_manager is None:
            try:
                from hardware_manager import HardwareManager
                self.hw_manager = HardwareManager()
            except ImportError:
                self.hw_manager = None
        else:
            self.hw_manager = hw_manager
            
        if hw_accelerator is None:
            try:
                from cdfa_extensions.hw_acceleration import HardwareAccelerator
                self.hw_accelerator = HardwareAccelerator()
            except ImportError:
                self.hw_accelerator = None
        else:
            self.hw_accelerator = hw_accelerator
            
        # Setup qubits and initial device attributes
        self.max_qubits = 8
        self.device = None
        self.device_name = 'none'
        self.quantum_available = False
        
        # Determine quantum availability
        try:
            import pennylane as qml
            # For PennyLane 0.41.0 compatibility, don't use plugin_devices or qml.devices
            # Instead, try to create a device directly and check if it works
            try:
                # Try to initialize a minimal device to check if quantum is available
                self.device = qml.device('default.qubit', wires=self.max_qubits)
                self.device_name = 'default.qubit'
                # If we get here, quantum is available
                self.quantum_available = True
            except Exception as e:
                # Try alternative devices that might be available
                try:
                    self.device = qml.device('lightning.qubit', wires=self.max_qubits)
                    self.device_name = 'lightning.qubit'
                    self.quantum_available = True
                except Exception as e2:
                    try:
                        self.device = qml.device('lightning.kokkos', wires=self.max_qubits)
                        self.device_name = 'lightning.kokkos'
                        self.quantum_available = True
                    except Exception as e3:
                        logger.warning(f"Could not create any quantum device: {e3}")
        except ImportError:
            logger.warning("PennyLane not installed; quantum capabilities will be disabled")
        
        # Initialize GPU information
        self.gpu_available = False
        if self.hw_accelerator is not None:
            try:
                accel_device = self.hw_accelerator.get_accelerator_device()
                self.gpu_available = accel_device.startswith('cuda') or accel_device.startswith('rocm') or accel_device.startswith('mps')
            except Exception as e:
                logger.debug(f"Error getting accelerator device: {e}")
                
        if not self.gpu_available and self.hw_manager is not None:
            # Fallback detection via hardware manager
            try:
                # Safe access to hardware manager attributes
                if hasattr(self.hw_manager, 'devices'):
                    self.gpu_available = (
                        self.hw_manager.devices.get('nvidia_gpu', {}).get('available', False) or
                        self.hw_manager.devices.get('amd_gpu', {}).get('available', False)
                    )
            except Exception as e:
                logger.debug(f"Error checking GPU via hardware manager: {e}")
    
    def _initialize_hamiltonians(self) -> Dict[str, Callable]:
        """
        Initialize Hamiltonian encodings for different regression types.
        Uses hardware acceleration if available.
        """
        def trend_hamiltonian(time_series: np.ndarray) -> np.ndarray:
            """Create Hamiltonian for trend regression."""
            # If GPU acceleration is available, use it
            if self.gpu_available and TORCH_AVAILABLE and self.hw_accelerator is not None:
                return self._trend_hamiltonian_gpu(time_series)
            
            # CPU fallback implementation
            n = len(time_series)
            # Create Ising Hamiltonian matrix
            h = np.zeros((n, n))
            
            # Add trend terms
            for i in range(n):
                for j in range(n):
                    if i == j:
                        # Diagonal terms based on values
                        h[i, j] = time_series[i] * 2
                    elif abs(i - j) == 1:
                        # Nearest neighbor coupling
                        h[i, j] = -1.0
            
            return h
        
        def momentum_hamiltonian(time_series: np.ndarray) -> np.ndarray:
            """Create Hamiltonian for momentum regression."""
            # If GPU acceleration is available, use it
            if self.gpu_available and TORCH_AVAILABLE and self.hw_accelerator is not None:
                return self._momentum_hamiltonian_gpu(time_series)
            
            # CPU fallback implementation
            n = len(time_series)
            # Create Hamiltonian matrix
            h = np.zeros((n, n))
            
            # Calculate momentum as differences
            diffs = np.diff(time_series)
            padded_diffs = np.pad(diffs, (0, 1), 'edge')
            
            # Add momentum terms
            for i in range(n):
                for j in range(n):
                    if i == j:
                        # Diagonal terms based on momentum
                        h[i, j] = padded_diffs[i] * 3
                    elif abs(i - j) == 1:
                        # Nearest neighbor coupling
                        h[i, j] = -1.5
                    elif abs(i - j) == 2:
                        # Next-nearest coupling for momentum
                        h[i, j] = -0.5
            
            return h
        
        def volatility_hamiltonian(time_series: np.ndarray) -> np.ndarray:
            """Create Hamiltonian for volatility regression."""
            # If GPU acceleration is available, use it
            if self.gpu_available and TORCH_AVAILABLE and self.hw_accelerator is not None:
                return self._volatility_hamiltonian_gpu(time_series)
            
            # CPU fallback implementation
            n = len(time_series)
            # Create Hamiltonian matrix
            h = np.zeros((n, n))
            
            # Calculate volatility as local standard deviation
            window = 3
            vols = np.zeros(n)
            for i in range(n):
                start = max(0, i - window)
                end = min(n, i + window + 1)
                vols[i] = np.std(time_series[start:end])
            
            # Add volatility terms
            for i in range(n):
                for j in range(n):
                    if i == j:
                        # Diagonal terms based on volatility
                        h[i, j] = vols[i] * 4
                    elif abs(i - j) == 1:
                        # Nearest neighbor coupling
                        h[i, j] = -vols[i] * 0.5
            
            return h
        
        def combined_hamiltonian(time_series: np.ndarray, weights: Dict[str, float]) -> np.ndarray:
            """Create combined Hamiltonian with weighted components."""
            # Get individual Hamiltonians
            h_trend = trend_hamiltonian(time_series)
            h_momentum = momentum_hamiltonian(time_series)
            h_volatility = volatility_hamiltonian(time_series)
            
            # GPU acceleration for the combination if available
            if self.gpu_available and TORCH_AVAILABLE and self.hw_accelerator is not None:
                trend_weight = weights.get('trend', 0.4)
                momentum_weight = weights.get('momentum', 0.4)
                volatility_weight = weights.get('volatility', 0.2)
                
                # Convert to tensors
                device = self.hw_accelerator.get_torch_device() if hasattr(self.hw_accelerator, 'get_torch_device') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                h_trend_t = torch.tensor(h_trend, device=device)
                h_momentum_t = torch.tensor(h_momentum, device=device)
                h_volatility_t = torch.tensor(h_volatility, device=device)
                
                # Combine with weights
                h_combined_t = (trend_weight * h_trend_t + 
                              momentum_weight * h_momentum_t + 
                              volatility_weight * h_volatility_t)
                
                # Convert back to numpy
                return h_combined_t.cpu().numpy()
            else:
                # CPU fallback implementation
                h_combined = (
                    weights.get('trend', 0.4) * h_trend + 
                    weights.get('momentum', 0.4) * h_momentum + 
                    weights.get('volatility', 0.2) * h_volatility
                )
                
                return h_combined
        
        return {
            'trend': trend_hamiltonian,
            'momentum': momentum_hamiltonian,
            'volatility': volatility_hamiltonian,
            'combined': combined_hamiltonian
        }
    
    def _trend_hamiltonian_gpu(self, time_series: np.ndarray) -> np.ndarray:
        """GPU-accelerated trend Hamiltonian calculation."""
        device = self.hw_accelerator.get_torch_device() if hasattr(self.hw_accelerator, 'get_torch_device') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        n = len(time_series)
        
        # Create tensor and move to device
        ts_tensor = torch.tensor(time_series, device=device)
        
        # Create diagonal matrix with time series values
        diag = torch.diag(ts_tensor * 2)
        
        # Create nearest-neighbor coupling
        nn_coupling = torch.zeros((n, n), device=device)
        idx = torch.arange(n-1, device=device)
        nn_coupling[idx, idx+1] = -1.0
        nn_coupling[idx+1, idx] = -1.0
        
        # Combine the matrices
        h = diag + nn_coupling
        
        # Move back to CPU and convert to numpy
        return h.cpu().numpy()
    
    def _momentum_hamiltonian_gpu(self, time_series: np.ndarray) -> np.ndarray:
        """GPU-accelerated momentum Hamiltonian calculation."""
        device = self.hw_accelerator.get_torch_device() if hasattr(self.hw_accelerator, 'get_torch_device') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        n = len(time_series)
        
        # Create tensor and move to device
        ts_tensor = torch.tensor(time_series, dtype=torch.float32, device=device)
        
        # Calculate momentum as differences
        diffs = torch.diff(ts_tensor)
        
        # Manual padding by duplicating the last element (equivalent to 'replicate' mode)
        if len(diffs) > 0:
            padded_diffs = torch.cat([diffs, diffs[-1:]], dim=0)
        else:
            padded_diffs = torch.zeros(1, device=device)
        
        # Create diagonal matrix with momentum values
        diag = torch.diag(padded_diffs * 3)
        
        # Create nearest-neighbor coupling
        nn_coupling = torch.zeros((n, n), device=device)
        
        # Safe indexing for nearest neighbors
        if n > 1:
            idx = torch.arange(n-1, device=device)
            nn_coupling[idx, idx+1] = -1.5
            nn_coupling[idx+1, idx] = -1.5
        
        # Create next-nearest coupling - only if we have enough elements
        nnn_coupling = torch.zeros((n, n), device=device)
        if n > 2:
            idx = torch.arange(n-2, device=device)
            nnn_coupling[idx, idx+2] = -0.5
            nnn_coupling[idx+2, idx] = -0.5
        
        # Combine the matrices
        h = diag + nn_coupling + nnn_coupling
        
        # Move back to CPU and convert to numpy
        return h.cpu().numpy()
        
    def _volatility_hamiltonian_gpu(self, time_series: np.ndarray) -> np.ndarray:
        """GPU-accelerated volatility Hamiltonian calculation."""
        device = self.hw_accelerator.get_torch_device() if hasattr(self.hw_accelerator, 'get_torch_device') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        n = len(time_series)
        
        # Create tensor and move to device
        ts_tensor = torch.tensor(time_series, dtype=torch.float32, device=device)
        
        # Calculate volatility using manual window operations instead of unfold
        window = 3
        vols = torch.zeros(n, dtype=torch.float32, device=device)
        
        # Manual padding - create padded tensor by repeating edge values
        left_padding = ts_tensor[0].repeat(window)
        right_padding = ts_tensor[-1].repeat(window)
        padded = torch.cat([left_padding, ts_tensor, right_padding])
        
        # Calculate volatility for each position
        for i in range(n):
            start_idx = i  # Offset by window due to padding
            window_vals = padded[start_idx:start_idx + 2*window+1]
            vols[i] = torch.std(window_vals)
        
        # Create diagonal matrix with volatility values
        diag = torch.diag(vols * 4)
        
        # Create nearest-neighbor coupling
        nn_coupling = torch.zeros((n, n), device=device)
        
        # Safe indexing for nearest neighbors
        if n > 1:
            idx = torch.arange(n-1, device=device)
            nn_coupling[idx, idx+1] = -vols[:-1] * 0.5  # Forward connections
            nn_coupling[idx+1, idx] = -vols[1:] * 0.5   # Backward connections
        
        # Combine the matrices
        h = diag + nn_coupling
        
        # Move back to CPU and convert to numpy
        return h.cpu().numpy()
    
    def _initialize_regression_models(self) -> Dict[str, Dict[str, Any]]:
        """Initialize regression models for different timeframes."""
        models = {}
        
        # Define timeframes
        timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']
        
        for tf in timeframes:
            # Define model parameters for each timeframe
            models[tf] = {
                'hamiltonian_weights': {
                    'trend': 0.4,
                    'momentum': 0.4,
                    'volatility': 0.2
                },
                'annealing_schedule': self._create_annealing_schedule(tf),
                'forecast_params': {
                    '1m': {'steps': 10, 'confidence_window': 0.2},
                    '5m': {'steps': 8, 'confidence_window': 0.25},
                    '15m': {'steps': 6, 'confidence_window': 0.3},
                    '1h': {'steps': 5, 'confidence_window': 0.35},
                    '4h': {'steps': 3, 'confidence_window': 0.4},
                    '1d': {'steps': 2, 'confidence_window': 0.5}
                }[tf]
            }
        
        return models
    
    def _create_annealing_schedule(self, timeframe: str = None) -> np.ndarray:
        """
        Create annealing schedule based on timeframe and market regime.
        
        Args:
            timeframe: Trading timeframe ('1m', '5m', '15m', '1h', '4h', '1d')
            
        Returns:
            Annealing schedule as numpy array
        """
        # Base schedule
        steps = self.annealing_steps
        
        # Adjust steps based on timeframe for performance
        if timeframe == '1m':
            steps = int(steps * 0.6)  # Faster for short timeframes
        elif timeframe == '1d':
            steps = int(steps * 1.2)  # More steps for longer timeframes
        
        # Check if we have a specific schedule for the current regime
        if self.current_regime in self.regime_params and 'temp_schedule' in self.regime_params[self.current_regime]:
            regime_schedule = self.regime_params[self.current_regime]['temp_schedule']
            
            # If it's just a list of key temperatures, interpolate
            if len(regime_schedule) < steps:
                temp_points = np.array(regime_schedule) * self.temp_scale
                idx_points = np.linspace(0, steps-1, len(temp_points), dtype=int)
                full_schedule = np.zeros(steps)
                
                # Linear interpolation between key points
                for i in range(len(temp_points)-1):
                    start_idx, end_idx = idx_points[i], idx_points[i+1]
                    start_temp, end_temp = temp_points[i], temp_points[i+1]
                    segment_steps = end_idx - start_idx
                    
                    if segment_steps > 0:
                        full_schedule[start_idx:end_idx+1] = np.linspace(
                            start_temp, end_temp, segment_steps+1
                        )
                
                # Ensure end matches
                if idx_points[-1] < steps-1:
                    full_schedule[idx_points[-1]+1:] = temp_points[-1]
                    
                return full_schedule
            else:
                # If it's already a full schedule
                return np.array(regime_schedule) * self.temp_scale
        
        # Default schedule (non-linear)
        s = np.linspace(0, 1, steps)
        schedule = s**2  # Quadratic schedule for slower initial annealing
        
        # Apply temperature scaling
        schedule = schedule * self.temp_scale
        
        return schedule
    
    def _initialize_circuits(self):
        """Initialize quantum circuit weights and parameters."""
        try:
            np.random.seed(42)  # For reproducibility
            
            # Initialize circuit weights for annealing
            # Shape: [annealing_steps, qubits, 3_rotation_params]
            self.circuit_weights = np.random.uniform(
                0, np.pi, (self.annealing_steps, self.max_qubits, 3)
            )
            
            # Temperature schedule for annealing
            self.temp_schedule = np.linspace(2.0, 0.5, self.annealing_steps)
            
            # For trainable circuits (used in quantum fitting)
            if self.quantum_available and PENNYLANE_AVAILABLE:
                # Create base parameters for multiple circuit types
                self.trainable_circuit_weights = None  # Will be initialized during first use
                
            logger.debug("Quantum circuit weights initialized")
        except Exception as e:
            logger.error(f"Error initializing quantum circuit weights: {e}")
            # Initialize with minimal fallback values
            self.circuit_weights = np.ones((2, min(4, self.max_qubits), 2))
            self.temp_schedule = np.array([1.0, 0.5])
    
    @quantum_accelerated(wires=8)
    def _quantum_annealing_circuit(self, time_series_data, temp, device=None):
        """
        Creates a quantum circuit for annealing using PennyLane.
        
        Args:
            time_series_data: Time series data to encode
            temp: Temperature for annealing
            device: Quantum device to use
            
        Returns:
            QNode circuit function
        """
        if not self.quantum_available:
            raise ValueError("Quantum device not available")
        
        @qml.qnode(device, interface="autograd")
        def circuit(data, temperature):
            n_qubits = min(len(data), self.max_qubits)
            
            # Embed data as angles
            qml.templates.AngleEmbedding(data[:n_qubits], wires=range(n_qubits))
            
            # Apply annealing schedule
            for step, temp_val in enumerate(temperature):
                # Apply rotations with temperature
                for i in range(n_qubits):
                    # Access circuit_weights safely
                    step_idx = min(step, self.circuit_weights.shape[0] - 1)
                    qubit_idx = min(i, self.circuit_weights.shape[1] - 1)
                    
                    # Apply parameterized rotations
                    qml.RX(temp_val * self.circuit_weights[step_idx, qubit_idx, 0], wires=i)
                    qml.RY(temp_val * self.circuit_weights[step_idx, qubit_idx, 1], wires=i)
                    qml.RZ(temp_val * self.circuit_weights[step_idx, qubit_idx, 2], wires=i)
                
                # Apply entangling layer
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                    qml.RZ(temp_val * np.pi / 2, wires=i + 1)
                    qml.CNOT(wires=[i, i + 1])
            
            # Measure each qubit
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        
        return circuit
    
    def _simulate_quantum_annealing(self, 
                                  hamiltonian: np.ndarray, 
                                  annealing_schedule: np.ndarray,
                                  initial_state: Optional[np.ndarray] = None,
                                  repeats: int = None) -> np.ndarray:
        """
        Simulate quantum annealing process with hardware acceleration if available.
        
        Args:
            hamiltonian: Hamiltonian matrix
            annealing_schedule: Annealing schedule
            initial_state: Initial quantum state (default: equal superposition)
            repeats: Number of annealing runs
            
        Returns:
            Final quantum state
        """
        try:
            # Get problem size
            n = hamiltonian.shape[0]
            
            # Set repeats
            if repeats is None:
                repeats = self.annealing_repeats
            
            # Initialize state if not provided
            if initial_state is None:
                # Start in superposition
                initial_state = np.ones(n) / np.sqrt(n)
            
            # If GPU acceleration is available, use it
            if self.gpu_available and TORCH_AVAILABLE:
                return self._simulate_quantum_annealing_gpu(
                    hamiltonian, annealing_schedule, initial_state, repeats
                )
            
            # CPU fallback implementation
            # Initialize results storage
            all_states = []
            all_energies = []
            
            # Perform multiple annealing runs
            for _ in range(repeats):
                # Start with initial state
                state = initial_state.copy()
                
                # Apply annealing schedule
                for s in annealing_schedule:
                    # Create annealing operator
                    # As s increases from 0 to 1, we transition from quantum to classical
                    quantum_term = np.eye(n) * (1 - s)
                    classical_term = hamiltonian * s
                    operator = quantum_term + classical_term
                    
                    # Apply operator (simplified simulation)
                    state = np.dot(operator, state)
                    
                    # Normalize state
                    state = state / np.linalg.norm(state)
                
                # Calculate final energy
                energy = np.dot(state, np.dot(hamiltonian, state))
                
                # Store results
                all_states.append(state)
                all_energies.append(energy)
            
            # Select best state (lowest energy)
            best_idx = np.argmin(all_energies)
            best_state = all_states[best_idx]
            
            return best_state
            
        except Exception as e:
            self.fault_manager.report_fault("QuantumAnnealingRegression", f"Error in forecast generation: {e}", 
                               exception=e, 
                               severity=FaultSeverity.ERROR if hasattr(FaultSeverity, 'ERROR') else None,
                               category=FaultCategory.COMPUTATION if hasattr(FaultCategory, 'COMPUTATION') else None)
            logger.error(f"Error in quantum annealing simulation: {e}")
            # Return fallback state
            return np.ones(hamiltonian.shape[0]) / np.sqrt(hamiltonian.shape[0])
    
    def _simulate_quantum_annealing_gpu(self, 
                                     hamiltonian: np.ndarray, 
                                     annealing_schedule: np.ndarray,
                                     initial_state: np.ndarray,
                                     repeats: int) -> np.ndarray:
        """
        GPU-accelerated quantum annealing simulation.
        
        Args:
            hamiltonian: Hamiltonian matrix
            annealing_schedule: Annealing schedule
            initial_state: Initial quantum state
            repeats: Number of annealing runs
            
        Returns:
            Final quantum state as numpy array
        """
        device = self.hw_accelerator.get_torch_device() if hasattr(self.hw_accelerator, 'get_torch_device') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Convert inputs to tensors
        h_tensor = torch.tensor(hamiltonian, dtype=torch.float32, device=device)
        schedule_tensor = torch.tensor(annealing_schedule, dtype=torch.float32, device=device)
        init_state_tensor = torch.tensor(initial_state, dtype=torch.float32, device=device)
        
        # Identity matrix for quantum term
        eye_tensor = torch.eye(h_tensor.shape[0], dtype=torch.float32, device=device)
        
        # Storage for results
        all_states = []
        all_energies = torch.zeros(repeats, dtype=torch.float32, device=device)
        
        # Run multiple annealing simulations in parallel if possible
        for r in range(repeats):
            # Start with initial state
            state = init_state_tensor.clone()
            
            # Apply annealing schedule
            for s in schedule_tensor:
                # Create annealing operator
                quantum_term = eye_tensor * (1 - s)
                classical_term = h_tensor * s
                operator = quantum_term + classical_term
                
                # Apply operator
                state = torch.matmul(operator, state)
                
                # Normalize state
                state = state / torch.norm(state)
            
            # Calculate final energy
            energy = torch.matmul(state, torch.matmul(h_tensor, state))
            
            # Store results
            all_states.append(state)
            all_energies[r] = energy
        
        # Select best state (lowest energy)
        best_idx = torch.argmin(all_energies).item()
        best_state = all_states[best_idx]
        
        # Convert back to numpy
        return best_state.cpu().numpy()
    
    def _execute_with_cache(self, 
                          func: Callable, 
                          args: Tuple, 
                          cache_key_prefix: str) -> Any:
        """
        Execute function with caching for performance optimization.
        
        Args:
            func: Function to execute
            args: Arguments for the function
            cache_key_prefix: Prefix for cache key
            
        Returns:
            Function result (from cache or freshly executed)
        """
        try:
            # Generate cache key
            cache_key = hash(cache_key_prefix + str([
                a.tobytes() if hasattr(a, 'tobytes') else str(a) for a in args
            ]))
            
            # Check cache
            with self.cache_lock:
                if cache_key in self.cache:
                    return self.cache[cache_key]
            
            # Execute function
            result = func(*args)
            
            # Cache result
            with self.cache_lock:
                self.cache[cache_key] = result
            
            return result
            
        except Exception as e:
            logger.warning(f"Cache execution failed for {cache_key_prefix}: {e}")
            # Execute without caching
            return func(*args)
    
    def _prepare_time_series(self, market_data: Dict[str, Any]) -> np.ndarray:
        """
        Extract and prepare time series data from market data.
        
        Args:
            market_data: Market data dictionary with price information
            
        Returns:
            Normalized time series data
        """
        # Extract price data
        if 'close' in market_data:
            if isinstance(market_data['close'], (list, np.ndarray)):
                # Already a list or array
                prices = np.array(market_data['close'])
            else:
                # Single value
                prices = np.array([market_data['close']])
        elif 'price_data' in market_data:
            prices = np.array(market_data['price_data'])
        else:
            # Try to extract from dictionary values
            try:
                prices = np.array(list(market_data.values()))
            except:
                logger.warning("Could not extract price data from market_data")
                # Generate dummy data
                prices = np.linspace(1, 2, self.window_size)
        
        # Ensure we have enough data
        if len(prices) < self.window_size:
            # Pad with duplicate values if needed
            prices = np.pad(prices, (self.window_size - len(prices), 0), 'edge')
        
        # Use only the most recent window_size elements
        time_series = prices[-self.window_size:]
        
        # Normalize time series
        normalized = (time_series - np.mean(time_series)) / (np.std(time_series) + 1e-8)
        
        return normalized
    
    def _prepare_market_data(self, data_chunk):
        """
        Prepare market data for quantum encoding.
        
        Args:
            data_chunk: Market data chunk
            
        Returns:
            Feature vector for quantum input
        """
        if len(data_chunk) < 2:
            return np.zeros(self.max_qubits)
            
        features = []
        
        # Calculate price changes
        changes = np.diff(data_chunk) / np.maximum(data_chunk[:-1], 1e-8)
        
        # Recent changes (most important)
        features.extend(changes[-min(4, len(changes)):])
        
        # Volatility
        if len(changes) > 3:
            features.append(np.std(changes))
        else:
            features.append(0)
            
        # Momentum
        if len(data_chunk) > 5:
            momentum = data_chunk[-1] / max(data_chunk[-5], 1e-8) - 1
            features.append(momentum)
        else:
            features.append(0)
            
        # Mean reversion potential
        if len(data_chunk) > 20:
            mean_20 = np.mean(data_chunk[-20:])
            mean_rev = (data_chunk[-1] / mean_20) - 1
            features.append(mean_rev)
        else:
            features.append(0)
            
        # Ensure we have the right number of features
        if len(features) > self.max_qubits:
            features = features[:self.max_qubits]
        elif len(features) < self.max_qubits:
            features.extend([0] * (self.max_qubits - len(features)))
            
        # Handle NaN/Inf values
        features = np.nan_to_num(np.array(features, dtype=np.float64), 
                               nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Normalize features to suitable range for quantum circuit
        feature_max = max(1e-8, np.max(np.abs(features)))
        features = features / feature_max * (np.pi / 2)
        
        return features
    
    def _create_hamiltonian_from_time_series(self, 
                                           time_series: np.ndarray, 
                                           timeframe: str) -> np.ndarray:
        """
        Create Hamiltonian encoding from time series data.
        
        Args:
            time_series: Time series data
            timeframe: Trading timeframe
            
        Returns:
            Hamiltonian matrix
        """
        # Get model parameters for timeframe
        if timeframe not in self.models:
            timeframe = '1h'  # Default
        
        model_params = self.models[timeframe]
        weights = model_params['hamiltonian_weights']
        
        # Create combined Hamiltonian
        hamiltonian_creator = self.hamiltonians['combined']
        
        # Execute with caching
        return self._execute_with_cache(
            hamiltonian_creator,
            (time_series, weights),
            f"hamiltonian_{timeframe}"
        )
    
    def _decode_forecast_from_state(self, 
                                  state: np.ndarray, 
                                  time_series: np.ndarray,
                                  steps: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decode forecast from quantum state.
        
        Args:
            state: Quantum state vector
            time_series: Original time series
            steps: Number of steps to forecast
            
        Returns:
            Tuple of (forecast_array, confidence_array)
        """
        try:
            # Get original scale - ensure local variables are properly defined
            if len(time_series) == 0:
                return np.zeros(steps), np.ones(steps)
                
            mean = np.mean(time_series)
            std = np.std(time_series) if len(time_series) > 1 else 0.01
            
            # Safety check for state - ensure it's a proper array with values
            if not isinstance(state, np.ndarray):
                try:
                    state = np.array(state, dtype=np.float64)
                except:
                    # If conversion fails, create a simple uniform state
                    state = np.ones(max(8, len(time_series))) / max(8, len(time_series))
            
            # Handle scalar or empty state
            if state.size == 0 or state.ndim == 0:
                logger.warning("Received empty quantum state. Using fallback forecasting.")
                # Create fallback forecast based on simple trend
                if len(time_series) >= 2:
                    last_diff = time_series[-1] - time_series[-2]
                    forecast = np.array([time_series[-1] + last_diff * i for i in range(1, steps + 1)])
                    confidence = np.array([std * 0.5 * i for i in range(1, steps + 1)])
                else:
                    # Very basic fallback if we don't have enough history
                    forecast = np.array([time_series[-1]] * steps)
                    confidence = np.array([std] * steps)
                    
                return forecast, confidence
                
            # Normalize state if needed
            if np.sum(state**2) > 0:
                state = state / np.sqrt(np.sum(state**2))
            else:
                # If state has zero norm, use uniform distribution
                state = np.ones(len(state)) / np.sqrt(len(state))
            
            # Ensure we have a minimum length for state
            if len(state) < 2:
                state = np.pad(state, (0, max(2, steps) - len(state)), mode='constant', constant_values=state[-1] if len(state) > 0 else 0)
            
            # Calculate trend from time series
            if len(time_series) >= 5:
                recent = time_series[-5:]
                recent_diffs = np.diff(recent)
                avg_diff = np.mean(recent_diffs) if len(recent_diffs) > 0 else 0
            else:
                recent_diffs = np.diff(time_series)
                avg_diff = np.mean(recent_diffs) if len(recent_diffs) > 0 else 0
            
            # Create containers for results
            forecast = np.zeros(steps)
            confidence_intervals = np.zeros(steps)
            
            # Use state vector to weight possible futures
            n = len(state)
            weights = state**2  # Use probability amplitudes as weights
            
            for step in range(steps):
                # Base forecast on trend
                base_forecast = time_series[-1] + avg_diff * (step + 1)
                
                # Add quantum adjustments
                forecast_distribution = np.zeros(n)
                for i in range(n):
                    # Create different trajectories based on position in state vector
                    adjustment = (i / n - 0.5) * std * (step + 1) * 0.2
                    forecast_distribution[i] = base_forecast + adjustment
                
                # Weight trajectories using quantum state
                weighted_forecast = np.sum(weights * forecast_distribution)
                
                # Calculate confidence interval
                ci_width = np.std(forecast_distribution) * 1.96  # 95% confidence
                
                # Store results
                forecast[step] = weighted_forecast
                confidence_intervals[step] = ci_width
            
            return forecast, confidence_intervals
            
        except Exception as e:
            logger.error(f"Error decoding forecast: {e}")
            # Return fallback forecast - with local variable definitions
            local_std = np.std(time_series) if len(time_series) > 1 else 0.01
            last_value = time_series[-1] if len(time_series) > 0 else 0.0
            zeros = np.zeros(steps)
            return last_value + zeros, zeros + local_std 
        
    def _decode_forecast_from_state_gpu(self, 
                                      state: np.ndarray, 
                                      time_series: np.ndarray,
                                      steps: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        GPU-accelerated forecast decoding from quantum state.
        
        Args:
            state: Quantum state vector
            time_series: Original time series
            steps: Number of steps to forecast
            
        Returns:
            Tuple of (forecast_array, confidence_array)
        """
        device = self.hw_accelerator.get_torch_device() if hasattr(self.hw_accelerator, 'get_torch_device') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Convert inputs to tensors
        state_t = torch.tensor(state, dtype=torch.float32, device=device)
        time_series_t = torch.tensor(time_series, dtype=torch.float32, device=device)
        
        # Get original scale
        mean = torch.mean(time_series_t)
        std = torch.std(time_series_t)
        
        # Create containers for results
        forecast = torch.zeros(steps, dtype=torch.float32, device=device)
        confidence_intervals = torch.zeros(steps, dtype=torch.float32, device=device)
        
        # Base forecast on recent trend
        recent = time_series_t[-5:]
        recent_diffs = torch.diff(recent)
        avg_diff = torch.mean(recent_diffs)
        
        # Use state vector to weight possible futures
        weights = state_t**2  # Use probability amplitudes as weights
        n = len(state_t)
        
        # Process each forecast step
        for step in range(1, steps + 1):
            # Base forecast
            base_forecast = time_series_t[-1] + avg_diff * step
            
            # Create all possible trajectories efficiently
            adjustments = torch.linspace(
                -0.5, 0.5, n, device=device) * std * step * 0.2
            forecast_distribution = base_forecast + adjustments
            
            # Weight trajectories using quantum state
            weighted_forecast = torch.sum(weights * forecast_distribution)
            
            # Calculate confidence interval
            ci_width = torch.std(forecast_distribution) * 1.96  # 95% confidence
            
            # Store results
            forecast[step-1] = weighted_forecast
            confidence_intervals[step-1] = ci_width
        
        # Denormalize forecast
        forecast_array = (forecast * std + mean).cpu().numpy()
        confidence_array = (confidence_intervals * std).cpu().numpy()
        
        return forecast_array, confidence_array
    
    def _adjust_for_anomalies(self, 
                            forecast: np.ndarray, 
                            confidence: np.ndarray,
                            anomaly_info: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Adjust forecast based on anomaly information.
        
        Args:
            forecast: Forecast array
            confidence: Confidence intervals array
            anomaly_info: Anomaly detection information
            
        Returns:
            Tuple of (adjusted_forecast, adjusted_confidence)
        """
        try:
            if not anomaly_info or not anomaly_info.get('detected', False):
                return forecast, confidence
            
            # Get anomaly score and confidence
            anomaly_score = anomaly_info.get('score', 0)
            
            if anomaly_score > 0.7:  # Strong anomaly
                # Widen confidence intervals
                confidence = confidence * (1 + anomaly_score)
                
                # Adjust forecast based on time_to_event
                if 'time_to_event' in anomaly_info:
                    if anomaly_info['time_to_event'] == 'imminent':
                        # Expect more extreme movement for imminent events
                        trend = forecast[-1] - forecast[0]
                        forecast = forecast + trend * anomaly_score * 0.5
                    elif anomaly_info['time_to_event'] == 'near-term':
                        # Expect movement in later part of forecast
                        forecast[-len(forecast)//2:] = forecast[-len(forecast)//2:] * (1 + 0.3 * anomaly_score)
            
            return forecast, confidence
            
        except Exception as e:
            logger.error(f"Error adjusting for anomalies: {e}")
            return forecast, confidence
    
    def _prepare_quantum_input(self, features: np.ndarray, num_qubits: int) -> np.ndarray:
        """
        Prepares and pads/truncates features for quantum circuit input.
        
        Args:
            features: Feature vector
            num_qubits: Number of qubits available
            
        Returns:
            Prepared feature vector for quantum input
        """
        if len(features) == num_qubits:
            return features
        elif len(features) > num_qubits:
            return features[:num_qubits]
        else:
            return np.pad(features, (0, num_qubits - len(features)), 'constant')
    
    def fit(self, X: np.ndarray, y: np.ndarray, learning_rate: Optional[float] = None,
          max_iterations: int = 100, verbose: bool = True) -> Dict[str, Any]:
        """
        Fit the model to training data, using quantum or classical methods.
        
        Args:
            X: Training features
            y: Target values
            learning_rate: Learning rate for optimization
            max_iterations: Maximum number of iterations
            verbose: Whether to log progress
            
        Returns:
            Dictionary with training metrics
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        n_samples, n_features = X.shape
        self.n_features = n_features  # Update n_features based on input
        
        if learning_rate is not None:
            self.learning_rate = learning_rate
            
        if self.weights is None or len(self.weights) != n_features:
            self.weights = np.random.randn(n_features) * 0.1
            
        if self.bias is None:
            self.bias = 0.0
        
        # Choose fit method
        if self.quantum_available:
            try:
                if verbose:
                    logger.info(f"Using quantum-enhanced training with {self.device_name}")
                metrics = self._quantum_fit(X, y, self.learning_rate, max_iterations)
            except Exception as e:
                if verbose:
                    logger.warning(f"Quantum training failed: {e}. Falling back to classical.")
                metrics = self._classical_fit(X, y, self.learning_rate, max_iterations, verbose)
        else:
            if verbose:
                logger.info("Using classical training (quantum not available/disabled)")
            metrics = self._classical_fit(X, y, self.learning_rate, max_iterations, verbose)
        
        # Track training history
        self.training_history.append(metrics)
        if len(self.training_history) > self.max_history:
            self.training_history.pop(0)
            
        return metrics
    
    def _quantum_fit(self, X: np.ndarray, y: np.ndarray, learning_rate: float, 
                   max_iterations: int) -> Dict[str, Any]:
        """
        Fit using quantum circuit with PennyLane.
        
        Args:
            X: Training features
            y: Target values
            learning_rate: Learning rate for optimization
            max_iterations: Maximum number of iterations
            
        Returns:
            Dictionary with training metrics
        """
        if not self.quantum_available:
            raise ValueError("Quantum not available")
            
        try:
            n_samples, n_features = X.shape
            
            # Determine qubits needed based on feature size
            num_qubits = min(n_features + 1, self.max_qubits)
            if num_qubits <= n_features:
                logger.warning(f"Feature count {n_features} exceeds available qubits {num_qubits}. Truncating features.")
            
            # Normalize data
            X_mean = np.mean(X, axis=0)
            X_std = np.std(X, axis=0) + 1e-8
            X_norm = (X - X_mean) / X_std
            
            y_mean = np.mean(y)
            y_std = np.std(y) + 1e-8
            y_norm = (y - y_mean) / y_std
            
            # Initialize weights and bias
            weights = self.weights.copy() if self.weights is not None else np.random.randn(n_features) * 0.1
            bias = self.bias if self.bias is not None else 0.0
            
            # Initialize trainable circuit weights if first time
            if self.trainable_circuit_weights is None and PENNYLANE_AVAILABLE:
                try:
                    # Initialize with 2 layers of strongly entangling circuit
                    self.trainable_circuit_weights = np.random.normal(
                        0, 0.1, qml.StronglyEntanglingLayers.shape(n_layers=2, n_wires=num_qubits-1)
                    )
                except Exception as e:
                    logger.warning(f"Failed to initialize circuit weights: {e}")
                    # Fallback initialization
                    self.trainable_circuit_weights = np.random.normal(0, 0.1, (2, num_qubits-1, 3))
            
            # Initialize device
            if self.device is None and self.hw_manager is not None:
                try:
                    device_config = self.hw_manager._get_quantum_device(num_qubits)
                    self.device = qml.device(
                        device_config['device'],
                        wires=num_qubits,
                        shots=device_config.get('shots', None)
                    )
                except Exception as e:
                    logger.warning(f"Failed to initialize quantum device: {e}")
                    # Fallback to default
                    self.device = qml.device("lightning.qubit", wires=num_qubits)
            
            # Accelerate with GPU if available
            if self.gpu_available and TORCH_AVAILABLE:
                return self._quantum_fit_gpu(X_norm, y_norm, weights, bias, learning_rate, max_iterations, num_qubits)
            
            # Track losses
            losses = []
            
            # Simplified gradient descent using quantum circuit
            for iteration in range(max_iterations):
                predictions_norm = np.zeros(n_samples)
                gradients = np.zeros_like(weights)
                bias_grad = 0.0
                
                for i in range(n_samples):
                    # Prepare input for quantum circuit
                    circuit_input = self._prepare_quantum_input(X_norm[i], num_qubits-1)
                    
                    try:
                        # Define prediction circuit
                        @qml.qnode(self.device)
                        def prediction_circuit(inputs, circuit_w):
                            # Embed inputs
                            qml.templates.AngleEmbedding(inputs, wires=range(len(inputs)))
                            
                            # Apply parameterized circuit
                            qml.templates.StronglyEntanglingLayers(circuit_w, wires=range(len(inputs)))
                            
                            # Measure first qubit expectation
                            return qml.expval(qml.PauliZ(0))
                        
                        # Get prediction
                        pred_norm_q = prediction_circuit(circuit_input, self.trainable_circuit_weights)
                        predictions_norm[i] = pred_norm_q
                        
                        # Calculate error
                        error_norm = pred_norm_q - y_norm[i]
                        
                        # Update gradients
                        gradients += error_norm * X_norm[i, :len(weights)] / n_samples
                        bias_grad += error_norm / n_samples
                        
                    except Exception as e:
                        logger.warning(f"Quantum circuit error: {e}")
                        # Fallback to classical prediction
                        pred_norm_classical = np.dot(X_norm[i, :len(weights)], weights) + bias
                        predictions_norm[i] = pred_norm_classical
                        
                        # Calculate error
                        error_norm = pred_norm_classical - y_norm[i]
                        
                        # Update gradients
                        gradients += error_norm * X_norm[i, :len(weights)] / n_samples
                        bias_grad += error_norm / n_samples
                
                # Update parameters
                weights -= learning_rate * gradients
                bias -= learning_rate * bias_grad
                
                # Calculate loss
                loss = np.mean((predictions_norm - y_norm)**2)
                losses.append(loss)
                
                # Log progress
                if (iteration + 1) % 10 == 0:
                    logger.info(f"Q-Fit Iter {iteration+1}, Loss: {loss:.6f}")
            
            # Save model parameters
            self.weights = weights
            self.bias = bias
            
            # Calculate denormalized predictions
            predictions = predictions_norm * y_std + y_mean
            mse = np.mean((predictions - y)**2)
            rmse = np.sqrt(mse)
            
            return {
                "iterations": max_iterations,
                "final_loss": losses[-1] if losses else float("inf"),
                "loss_history": losses,
                "weights": self.weights,
                "bias": self.bias,
                "mse": mse,
                "rmse": rmse
            }
            
        except Exception as e:
            logger.error(f"Error in quantum fitting: {e}")
            # Fallback to classical
            return self._classical_fit(X, y, learning_rate, max_iterations, True)
    
    def _quantum_fit_gpu(self, X_norm: np.ndarray, y_norm: np.ndarray, 
                       weights: np.ndarray, bias: float,
                       learning_rate: float, max_iterations: int, 
                       num_qubits: int) -> Dict[str, Any]:
        """
        GPU-accelerated quantum fitting.
        
        Args:
            X_norm: Normalized features
            y_norm: Normalized targets
            weights: Initial weights
            bias: Initial bias
            learning_rate: Learning rate
            max_iterations: Maximum iterations
            num_qubits: Number of qubits
            
        Returns:
            Dictionary with training metrics
        """
        try:
            device = self.hw_accelerator.get_torch_device() if hasattr(self.hw_accelerator, 'get_torch_device') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            n_samples = X_norm.shape[0]
            
            # Convert to PyTorch tensors
            X_tensor = torch.tensor(X_norm, dtype=torch.float32, device=device)
            y_tensor = torch.tensor(y_norm, dtype=torch.float32, device=device)
            weights_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
            bias_tensor = torch.tensor(bias, dtype=torch.float32, device=device)
            
            # Handle trainable circuit weights
            if not isinstance(self.trainable_circuit_weights, torch.Tensor):
                circuit_weights_tensor = torch.tensor(
                    self.trainable_circuit_weights, dtype=torch.float32, device=device, requires_grad=True
                )
            else:
                circuit_weights_tensor = self.trainable_circuit_weights.to(device)
            
            # Track losses
            losses = []
            
            # Optimization loop
            for iteration in range(max_iterations):
                # Storage for predictions and gradients
                predictions = torch.zeros(n_samples, dtype=torch.float32, device=device)
                
                # Process each sample
                for i in range(n_samples):
                    # Classical prediction component
                    classical_pred = torch.dot(X_tensor[i, :len(weights_tensor)], weights_tensor) + bias_tensor
                    
                    # Try quantum prediction
                    try:
                        # We'll only prepare data on GPU, but need to execute quantum circuit on CPU
                        # since PennyLane tensor interop is complex. This is a bottleneck that could
                        # be improved in future versions.
                        
                        # Prepare input
                        circuit_input = X_tensor[i, :num_qubits-1].cpu().numpy()
                        
                        # Run quantum circuit (on CPU)
                        @qml.qnode(self.device)
                        def prediction_circuit(inputs, circuit_w):
                            qml.templates.AngleEmbedding(inputs, wires=range(len(inputs)))
                            qml.templates.StronglyEntanglingLayers(circuit_w, wires=range(len(inputs)))
                            return qml.expval(qml.PauliZ(0))
                        
                        # Get quantum prediction
                        qc_weights = circuit_weights_tensor.cpu().numpy()
                        pred_q = prediction_circuit(circuit_input, qc_weights)
                        
                        # Combine classical and quantum components
                        predictions[i] = (classical_pred + torch.tensor(pred_q, device=device)) / 2
                    except Exception as e:
                        # Fallback to classical only
                        predictions[i] = classical_pred
                
                # Calculate loss
                loss = torch.mean((predictions - y_tensor)**2)
                losses.append(loss.item())
                
                # Calculate gradients
                dl_dp = 2 * (predictions - y_tensor) / n_samples
                gradients = torch.zeros_like(weights_tensor)
                for i in range(n_samples):
                    gradients += dl_dp[i] * X_tensor[i, :len(weights_tensor)]
                bias_grad = torch.sum(dl_dp)
                
                # Update parameters
                weights_tensor -= learning_rate * gradients
                bias_tensor -= learning_rate * bias_grad
                
                # Log progress
                if (iteration + 1) % 10 == 0:
                    logger.info(f"Q-GPU Iter {iteration+1}, Loss: {loss.item():.6f}")
            
            # Convert parameters back to NumPy
            self.weights = weights_tensor.cpu().numpy()
            self.bias = bias_tensor.cpu().numpy()
            
            return {
                "iterations": max_iterations,
                "final_loss": losses[-1] if losses else float("inf"),
                "loss_history": losses,
                "weights": self.weights,
                "bias": self.bias
            }
            
        except Exception as e:
            logger.error(f"Error in GPU quantum fitting: {e}")
            # Continue with CPU version
            return self._quantum_fit(X_norm, y_norm, weights, bias, learning_rate, max_iterations)
    
    def _classical_fit(self, X: np.ndarray, y: np.ndarray, learning_rate: float,
                     max_iterations: int, verbose: bool = True) -> Dict[str, Any]:
        """
        Fit using classical gradient descent with potential GPU acceleration.
        
        Args:
            X: Training features
            y: Target values
            learning_rate: Learning rate
            max_iterations: Maximum iterations
            verbose: Whether to log progress
            
        Returns:
            Dictionary with training metrics
        """
        # Use GPU acceleration if available
        if self.gpu_available and TORCH_AVAILABLE:
            return self._classical_fit_gpu(X, y, learning_rate, max_iterations, verbose)
        
        # Regular CPU implementation
        n_samples, n_features = X.shape
        
        # Normalize data
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0) + 1e-8
        X_norm = (X - X_mean) / X_std
        
        y_mean = np.mean(y)
        y_std = np.std(y) + 1e-8
        y_norm = (y - y_mean) / y_std
        
        # Initialize parameters if needed
        if self.weights is None or len(self.weights) != n_features:
            self.weights = np.random.randn(n_features) * 0.1
        if self.bias is None:
            self.bias = 0.0
        
        # Copy to avoid modifying original during training
        weights = self.weights.copy()
        bias = self.bias
        
        # Track losses
        losses = []
        
        # Optimization loop
        for iteration in range(max_iterations):
            # Forward pass
            predictions_norm = X_norm.dot(weights) + bias
            
            # Calculate error
            errors_norm = predictions_norm - y_norm
            loss = np.mean(errors_norm**2)
            losses.append(loss)
            
            # Calculate gradients
            grad_weights = (2/n_samples) * X_norm.T.dot(errors_norm)
            grad_bias = (2/n_samples) * np.sum(errors_norm)
            
            # Update parameters
            weights -= learning_rate * grad_weights
            bias -= learning_rate * grad_bias
            
            # Log progress
            if verbose and (iteration + 1) % 20 == 0:
                logger.info(f"C-Fit Iter {iteration+1}, Loss: {loss:.6f}")
        
        # Save parameters
        self.weights = weights
        self.bias = bias
        
        # Denormalize predictions for metrics
        predictions = predictions_norm * y_std + y_mean
        mse = np.mean((predictions - y)**2)
        rmse = np.sqrt(mse)
        
        return {
            "iterations": max_iterations,
            "final_loss": losses[-1] if losses else float("inf"),
            "loss_history": losses,
            "weights": self.weights,
            "bias": self.bias,
            "mse": mse,
            "rmse": rmse
        }
    
    def _classical_fit_gpu(self, X: np.ndarray, y: np.ndarray, learning_rate: float,
                         max_iterations: int, verbose: bool = True) -> Dict[str, Any]:
        """
        GPU-accelerated classical fitting using PyTorch.
        
        Args:
            X: Training features
            y: Target values
            learning_rate: Learning rate
            max_iterations: Maximum iterations
            verbose: Whether to log progress
            
        Returns:
            Dictionary with training metrics
        """
        try:
            device = self.hw_accelerator.get_torch_device() if hasattr(self.hw_accelerator, 'get_torch_device') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            n_samples, n_features = X.shape
            
            # Convert to PyTorch tensors
            X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
            y_tensor = torch.tensor(y, dtype=torch.float32, device=device)
            
            # Normalize data
            X_mean = torch.mean(X_tensor, dim=0)
            X_std = torch.std(X_tensor, dim=0) + 1e-8
            X_norm = (X_tensor - X_mean) / X_std
            
            y_mean = torch.mean(y_tensor)
            y_std = torch.std(y_tensor) + 1e-8
            y_norm = (y_tensor - y_mean) / y_std
            
            # Initialize parameters if needed
            if self.weights is None or len(self.weights) != n_features:
                weights_tensor = torch.randn(n_features, dtype=torch.float32, device=device) * 0.1
            else:
                weights_tensor = torch.tensor(self.weights, dtype=torch.float32, device=device)
                
            bias_tensor = torch.tensor(self.bias if self.bias is not None else 0.0, dtype=torch.float32, device=device)
            
            # Track losses
            losses = []
            
            # Optimization loop
            for iteration in range(max_iterations):
                # Forward pass
                predictions_norm = torch.matmul(X_norm, weights_tensor) + bias_tensor
                
                # Calculate error
                errors_norm = predictions_norm - y_norm
                loss = torch.mean(errors_norm**2)
                losses.append(loss.item())
                
                # Calculate gradients
                grad_weights = (2/n_samples) * torch.matmul(X_norm.T, errors_norm)
                grad_bias = (2/n_samples) * torch.sum(errors_norm)
                
                # Update parameters
                weights_tensor -= learning_rate * grad_weights
                bias_tensor -= learning_rate * grad_bias
                
                # Log progress
                if verbose and (iteration + 1) % 20 == 0:
                    logger.info(f"C-GPU Iter {iteration+1}, Loss: {loss.item():.6f}")
            
            # Save parameters
            self.weights = weights_tensor.cpu().numpy()
            self.bias = bias_tensor.cpu().numpy()
            
            # Calculate denormalized predictions for metrics
            predictions = predictions_norm * y_std + y_mean
            mse = torch.mean((predictions - y_tensor)**2).item()
            rmse = torch.sqrt(torch.tensor(mse)).item()
            
            return {
                "iterations": max_iterations,
                "final_loss": losses[-1] if losses else float("inf"),
                "loss_history": losses,
                "weights": self.weights,
                "bias": self.bias,
                "mse": mse,
                "rmse": rmse
            }
            
        except Exception as e:
            logger.error(f"Error in GPU classical fitting: {e}")
            # Fall back to CPU implementation
            return self._classical_fit(X, y, learning_rate, max_iterations, verbose)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the trained model.
        
        Args:
            X: Input features
            
        Returns:
            Predicted values
        """
        if self.weights is None:
            raise ValueError("Model not trained yet.")
            
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        if X.shape[1] != len(self.weights):
            raise ValueError(f"Input feature mismatch: expected {len(self.weights)}, got {X.shape[1]}")
        
        # Use GPU if available
        if self.gpu_available and TORCH_AVAILABLE:
            try:
                device = self.hw_accelerator.get_torch_device() if hasattr(self.hw_accelerator, 'get_torch_device') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
                weights_tensor = torch.tensor(self.weights, dtype=torch.float32, device=device)
                bias_tensor = torch.tensor(self.bias, dtype=torch.float32, device=device)
                
                predictions = torch.matmul(X_tensor, weights_tensor) + bias_tensor
                return predictions.cpu().numpy()
            except Exception as e:
                logger.warning(f"GPU prediction failed: {e}, falling back to CPU")
                
        # CPU fallback
        return np.dot(X, self.weights) + self.bias
    
    def forecast(self, market_data: Dict[str, Any],
               timeframe: str = '1h',
               steps: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate forecast using quantum annealing regression.
        
        Args:
            market_data: Market data dictionary with price information
            timeframe: Trading timeframe ('1m', '5m', '15m', '1h', '4h', '1d')
            steps: Number of steps to forecast (default: use timeframe setting)
            
        Returns:
            Dictionary containing forecast, confidence intervals, and metadata
        """
        start_time = time.time()
        
        try:
            # Prepare time series data
            time_series = self._prepare_time_series(market_data)
            
            # Default timeframe
            if timeframe not in self.models:
                timeframe = '1h'  # Default to 1-hour timeframe
            
            # Get model parameters
            model = self.models[timeframe]
            annealing_schedule = model['annealing_schedule']
            forecast_params = model['forecast_params']
            
            # Use provided steps or default from timeframe
            if steps is None:
                steps = forecast_params['steps']
            
            # Create Hamiltonian encoding
            hamiltonian = self._create_hamiltonian_from_time_series(time_series, timeframe)
            
            # Generate cache key for annealing
            # This allows reusing results for similar inputs
            cache_key = f"annealing_{timeframe}_{hash(time_series.tobytes())}"
            
            # Check if we can reuse a cached result
            annealing_result = None
            with self.cache_lock:
                if cache_key in self.cache:
                    annealing_result = self.cache[cache_key]
            
            # Perform quantum annealing if not cached
            if annealing_result is None:
                # Try using real quantum hardware if available
                if self.quantum_available:
                    try:
                        # Prepare quantum features
                        quantum_features = self._prepare_market_data(time_series)
                        
                        # Create quantum circuit
                        qc = self._quantum_annealing_circuit(
                            quantum_features, 
                            self.temp_schedule,
                            device=self.device
                        )
                        
                        # Run quantum circuit
                        annealing_result = qc(quantum_features, self.temp_schedule)
                        logger.info("Used real quantum hardware for annealing")
                    except Exception as e:
                        logger.warning(f"Quantum hardware execution failed: {e}. Falling back to simulation.")
                        annealing_result = None
                
                # Fall back to simulation if quantum hardware failed or unavailable
                if annealing_result is None:
                    annealing_result = self._simulate_quantum_annealing(
                        hamiltonian=hamiltonian,
                        annealing_schedule=annealing_schedule
                    )
                
                # Cache result
                with self.cache_lock:
                    self.cache[cache_key] = annealing_result
            
            # Decode forecast from quantum state
            forecast, confidence = self._decode_forecast_from_state(
                annealing_result, time_series, steps
            )
            
            # Detect anomalies if IQAD is available
            anomaly_info = None
            if self.iqad:
                # --- START OF MODIFICATION ---
                iqad_input_features = {}
                # IQAD expects features for a single point in time.
                # Extract latest scalar values from market_data.
                
                # Primary features IQAD's _feature_encoding method looks for:
                # 'close', 'volume', 'volatility', 'volatility_regime', 
                # 'rsi_14', 'rsi', 'adx', 'trend', 'momentum', 'regime'
                
                # Iterate through all keys that might be in market_data and are relevant to IQAD
                all_relevant_iqad_keys = [
                    'close', 'volume', 'volatility', 'volatility_regime', 
                    'rsi_14', 'rsi', 'adx', 'trend', 'momentum', 'regime'
                ] # This list includes all keys directly used by iqad._feature_encoding

                for key in all_relevant_iqad_keys:
                    if key in market_data:
                        val = market_data[key]
                        # If the value is a list or numpy array, take the last element
                        if isinstance(val, (np.ndarray, list)):
                            if len(val) > 0:
                                # Ensure the extracted element is a standard Python float
                                try:
                                    iqad_input_features[key] = float(val[-1])
                                except (TypeError, ValueError):
                                    logger.warning(f"Could not convert market_data['{key}'][-1] to float for IQAD. Skipping.")
                            # If val is an empty list/array, this key won't be added to iqad_input_features.
                            # IQAD will then use its default for this key.
                        # If the value is already a scalar (Python or NumPy type)
                        elif isinstance(val, (int, float, np.generic)): # np.generic covers all NumPy scalars
                            try:
                                iqad_input_features[key] = float(val)
                            except (TypeError, ValueError):
                                logger.warning(f"Could not convert market_data['{key}'] to float for IQAD. Skipping.")
                        # else: market_data[key] is of an unexpected type or structure for scalar extraction.
                        # This key won't be added, and IQAD will use its default.
                
                # If some features are still missing from iqad_input_features (e.g., 'volume' if not in market_data), 
                # IQAD's .get(key, default_value) in its _feature_encoding method will handle it gracefully.
                
                logger.debug(f"Features prepared for IQAD: {iqad_input_features}") # For debugging
                anomaly_info = self.iqad.detect_anomalies(iqad_input_features)
                # --- END OF MODIFICATION ---
                
                # Adjust forecast for anomalies
                forecast, confidence = self._adjust_for_anomalies(
                    forecast, confidence, anomaly_info
                )
            
            # Calculate direction and confidence
            direction = np.sign(forecast[-1] - time_series[-1])
            forecast_confidence = 1.0 - np.mean(confidence) / (np.std(time_series) * 4)
            forecast_confidence = max(0.1, min(0.95, forecast_confidence))
            
            # Prepare result
            result = {
                'forecast': forecast.tolist(),
                'confidence_intervals': confidence.tolist(),
                'direction': float(direction),
                'confidence': float(forecast_confidence),
                'timeframe': timeframe,
                'steps': steps,
                'execution_time_ms': (time.time() - start_time) * 1000
            }
            
            # Add anomaly information if available
            if anomaly_info:
                result['anomaly_detected'] = anomaly_info.get('detected', False)
                result['anomaly_score'] = anomaly_info.get('score', 0)
            
            # Store forecast in history
            self._store_forecast_history(result)
            
            return result
            
        except Exception as e:
            self.fault_manager.report_fault(
            "QuantumAnnealingRegression", 
            f"Error in forecast generation: {e}", 
            exception=e
            )
            logger.error(f"Error in forecast generation: {e}")
            execution_time = (time.time() - start_time) * 1000
            
            # Return fallback forecast
            if steps is None:
                steps = 5  # Default steps
                
            return {
                'forecast': [float(0)] * steps,
                'confidence_intervals': [float(1)] * steps,
                'direction': 0.0,
                'confidence': 0.1,
                'timeframe': timeframe,
                'steps': steps,
                'execution_time_ms': execution_time,
                'error': str(e)
            }
    
    def perform_regression(self, data: pd.DataFrame) -> pd.Series:
        """
        Perform regression on a DataFrame, with market regime adaptation.
        
        Args:
            data: DataFrame with time series data (must have 'close' column)
            
        Returns:
            Series with regression results
        """
        if not isinstance(data, pd.DataFrame) or data.empty or "close" not in data.columns:
            logger.warning("Invalid data for Quantum Annealing regression.")
            return pd.Series(dtype=float)
            
        try:
            prices = data["close"].values
            if len(prices) < self.window_size:
                return pd.Series(prices, index=data.index)  # Return original if too short
            
            # Prepare features (e.g., lagged prices)
            X = np.array([prices[i:i+self.window_size] for i in range(len(prices)-self.window_size)])
            y = prices[self.window_size:]
            
            if X.shape[0] == 0:  # Not enough data after lag prep
                return pd.Series(np.nan, index=data.index)
            
            # Fit the model (will choose quantum or classical internally)
            self.fit(X, y, verbose=False)  # Fit silently within perform_regression
            
            # Predict on the same lagged data used for training (for smoothing effect)
            predictions = self.predict(X)
            
            # Create output series, aligning with original index (add NaNs at start)
            output = pd.Series(np.nan, index=data.index)
            output.iloc[self.window_size:] = predictions
            output = output.bfill()  # Backfill initial NaNs
            
            # Error Tracking & Adaptation
            if len(output) > 1 and pd.notna(output.iloc[-1]) and pd.notna(prices[-1]):
                prediction_error = abs(output.iloc[-1] - prices[-1]) / (prices[-1] + 1e-9)
                self.prediction_errors.append(prediction_error)
                if len(self.prediction_errors) > self.max_history:
                    self.prediction_errors.pop(0)
                self._adapt_parameters()
            
            return output.rename("quareg_signal")
            
        except Exception as e:
            logger.error(f"Error in Quantum Annealing perform_regression: {e}", exc_info=True)
            # Fallback to simpler moving average
            try:
                return data["close"].rolling(window=max(3, self.kernel_size)).mean().rename("quareg_signal_fallback")
            except:
                return pd.Series(dtype=float)
    
    def _store_forecast_history(self, forecast: Dict[str, Any]) -> None:
        """Store forecast in history for tracking performance."""
        try:
            # Add timestamp
            forecast_record = forecast.copy()
            forecast_record['timestamp'] = time.time()
            
            # Add to history
            self.forecast_history.append(forecast_record)
            
            # Trim history if too long
            if len(self.forecast_history) > self.max_history:
                self.forecast_history = self.forecast_history[-self.max_history:]
                
        except Exception as e:
            logger.error(f"Error storing forecast history: {e}")
    
    def set_regime(self, regime: str) -> None:
        """
        Set market regime for adaptive annealing.
        
        Args:
            regime: Market regime ('bull', 'bear', 'volatile', 'neutral')
        """
        if regime in self.regime_params:
            self.current_regime = regime
            logger.info(f"Quantum Annealing regime set: {regime}")
        else:
            logger.warning(f"Unknown regime: {regime}. Using default: {self.current_regime}")
            
        # Update internal params based on new regime
        if "temp_schedule" in self.regime_params.get(self.current_regime, {}):
            self.temp_schedule = self.regime_params[self.current_regime]["temp_schedule"]
            
        if "kernel_size" in self.regime_params.get(self.current_regime, {}):
            self.kernel_size = self.regime_params[self.current_regime]["kernel_size"]
            
        # Adjust temperature scaling
        if self.current_regime == "volatile":
            self.temp_scale = 1.5
        elif self.current_regime == "bull":
            self.temp_scale = 1.2
        else:
            self.temp_scale = 1.0
    
    def _adapt_parameters(self):
        """Adapt parameters based on recent prediction error."""
        if len(self.prediction_errors) < 10:
            return
            
        try:
            recent_mae = np.mean(np.abs(self.prediction_errors[-10:]))
            
            if recent_mae > 0.05:  # High error -> adapt more
                self.learning_rate = min(0.05, self.learning_rate * 1.1)
                self.kernel_size = max(3, self.kernel_size-1) if self.current_regime == "volatile" else min(9, self.kernel_size+1)
            else:  # Low error -> stabilize
                self.learning_rate = max(0.005, self.learning_rate * 0.95)
                
            logger.debug(f"Adapted parameters: LR={self.learning_rate:.4f}, Kernel={self.kernel_size}")
        except Exception as e:
            logger.error(f"Error adapting parameters: {e}")
    
    def get_accuracy_metrics(self) -> Dict[str, float]:
        """Calculate accuracy metrics for past forecasts."""
        try:
            if not self.forecast_history:
                return {'mean_error': 0.0, 'direction_accuracy': 0.0, 'sample_size': 0}
            
            # Calculate metrics
            errors = []
            direction_correct = 0
            total_directions = 0
            
            for i in range(len(self.forecast_history) - 1):
                current = self.forecast_history[i]
                next_data = self.forecast_history[i + 1]
                
                # Skip if different timeframes
                if current.get('timeframe') != next_data.get('timeframe'):
                    continue
                
                # Get actual value (from next record's first value)
                if 'forecast' in next_data and len(next_data['forecast']) > 0:
                    actual = next_data['forecast'][0]
                    
                    # Get forecast for this timepoint
                    if 'forecast' in current and len(current['forecast']) > 0:
                        forecast = current['forecast'][0]
                        
                        # Calculate error
                        error = abs(forecast - actual) / (abs(actual) + 1e-8)
                        errors.append(error)
                        
                        # Check direction
                        forecast_direction = current.get('direction', 0)
                        actual_direction = np.sign(actual - forecast)
                        
                        if forecast_direction * actual_direction > 0:
                            direction_correct += 1
                        
                        total_directions += 1
            
            # Calculate average metrics
            mean_error = np.mean(errors) if errors else 0.0
            direction_accuracy = direction_correct / total_directions if total_directions > 0 else 0.0
            
            return {
                'mean_error': float(mean_error),
                'direction_accuracy': float(direction_accuracy),
                'sample_size': len(errors)
            }
            
        except Exception as e:
            logger.error(f"Error calculating accuracy metrics: {e}")
            return {'mean_error': 0.0, 'direction_accuracy': 0.0, 'sample_size': 0, 'error': str(e)}
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information including hardware details."""
        model_info = {
            "model_type": "QuantumAnnealingRegression",
            "hardware_device": self.device_name if hasattr(self, 'device_name') else "unknown",
            "quantum_available": self.quantum_available,
            "gpu_available": self.gpu_available,
            "window_size": self.window_size,
            "forecast_horizon": self.forecast_horizon,
            "n_features": self.n_features,
            "annealing_steps": self.annealing_steps,
            "kernel_size": self.kernel_size,
            "max_qubits": self.max_qubits,
            "learning_rate": self.learning_rate,
            "current_regime": self.current_regime,
            "weights_shape": None if self.weights is None else self.weights.shape,
            "recent_error": np.mean(np.abs(self.prediction_errors[-10:])) if len(self.prediction_errors) >= 10 else None,
            "training_history_entries": len(self.training_history)
        }
        
        # Add hardware accelerator info
        if self.hw_accelerator is not None:
            try:
                # Simplified, safe accelerator info
                model_info["accelerator"] = {
                    "type": "gpu" if self.gpu_available else "cpu",
                    "available": True
                }
                
                # Add device name if we can safely retrieve it
                if hasattr(self.hw_accelerator, 'get_accelerator_type'):
                    model_info["accelerator"]["backend"] = str(self.hw_accelerator.get_accelerator_type())
                    
            except Exception as e:
                logger.warning(f"Error getting accelerator info: {e}")
        
        # Add quantum device info
        if self.hw_manager is not None:
            try:
                # Add basic quantum info
                model_info["quantum_backend"] = self.device_name if hasattr(self, 'device_name') else "none"
                
                # Only add optimal device info if we can safely access it
                if hasattr(self.hw_manager, 'get_optimal_device'):
                    try:
                        # Get only basic device type information
                        device_info = {"type": "quantum_simulator"}
                        if hasattr(self.hw_manager, 'devices') and 'quantum' in self.hw_manager.devices:
                            if 'default_device' in self.hw_manager.devices['quantum']:
                                device_info["device"] = self.hw_manager.devices['quantum']['default_device']
                        
                        model_info["optimal_device"] = device_info
                    except Exception:
                        # Safe fallback
                        model_info["optimal_device"] = {"type": "quantum_simulator", "available": self.quantum_available}
            except Exception as e:
                logger.warning(f"Error getting hardware manager info: {e}")
        
        return model_info


# Factory function for thread-safe singleton access
_annealing_instance = None
_annealing_lock = threading.RLock()

def get_quantum_annealing_regression(
    hw_manager=None, hw_accelerator=None, config=None, reset=False
) -> QuantumAnnealingRegression:
    """
    Thread-safe factory function for QuantumAnnealingRegression.
    
    Args:
        hw_manager: HardwareManager instance or None to create new one
        hw_accelerator: HardwareAccelerator instance or None to create new one
        config: Configuration dictionary
        reset: Whether to reset existing instance
        
    Returns:
        QuantumAnnealingRegression instance
    """
    global _annealing_instance, _annealing_lock
    
    with _annealing_lock:
        if _annealing_instance is None or reset:
            # Create hardware manager if not provided
            if hw_manager is None and HARDWARE_ACCEL_AVAILABLE:
                try:
                    hw_manager = HardwareManager.get_manager()
                except Exception as e:
                    logger.warning(f"Failed to create HardwareManager: {e}")
            
            # Create hardware accelerator if not provided
            if hw_accelerator is None and HARDWARE_ACCEL_AVAILABLE:
                try:
                    hw_accelerator = HardwareAccelerator(enable_gpu=True)
                except Exception as e:
                    logger.warning(f"Failed to create HardwareAccelerator: {e}")
            
            # Create quantum annealing instance
            _annealing_instance = QuantumAnnealingRegression(
                hw_manager=hw_manager,
                hw_accelerator=hw_accelerator,
                config=config
            )
    
    return _annealing_instance


def main():
    """Test the Quantum Annealing Regression implementation."""
    print("Testing Quantum Annealing Regression...")
    
    # Set up logger for testing
    testing_logger = logging.getLogger("test_quantum_annealing")
    testing_logger.setLevel(logging.INFO)
    
    # Initialize hardware components
    hw_manager = None
    hw_accelerator = None
    
    try:
        print("Initializing hardware management...")
        if HARDWARE_ACCEL_AVAILABLE:
            hw_manager = HardwareManager.get_manager()
            hw_manager.initialize_hardware()
            
            hw_accelerator = HardwareAccelerator(enable_gpu=True)
            
            print(f"Hardware initialized: Quantum available: {hw_manager.quantum_available if hasattr(hw_manager, 'quantum_available') else False}")
            print(f"GPU acceleration: {hw_accelerator.get_accelerator_type() if hasattr(hw_accelerator, 'get_accelerator_type') else 'Unavailable'}")
        else:
            print("Hardware acceleration modules not available. Using fallback.")
            
        # Generate test data
        print("\nGenerating test data...")
        n_samples = 100
        window_size = 20
        forecast_horizon = 5
        
        # Create artificial price data
        np.random.seed(42)
        t = np.linspace(0, 4*np.pi, n_samples)
        prices = 100 + 10 * np.sin(t) + 2 * np.sin(5*t) + np.random.normal(0, 1, n_samples).cumsum()
        
        # Create market data dictionary
        market_data = {
            'close': prices,
            'timestamp': np.arange(n_samples)
        }
        
        # Create DataFrame for regression testing
        df = pd.DataFrame({
            'close': prices,
            'timestamp': np.arange(n_samples)
        })
        
        # Test initialization
        print("\nInitializing Quantum Annealing Regression...")
        model = QuantumAnnealingRegression(
            hw_manager=hw_manager,
            hw_accelerator=hw_accelerator,
            window_size=window_size,
            forecast_horizon=forecast_horizon
        )
        print(f"Model initialized successfully")
        
        # Print hardware capabilities
        print(f"Quantum available: {model.quantum_available}")
        print(f"GPU available: {model.gpu_available}")
        print(f"Max qubits: {model.max_qubits}")
        
        # Test forecasting
        print("\nTesting forecasting...")
        forecast_result = model.forecast(market_data, timeframe="1h", steps=forecast_horizon)
        print(f"Forecast generated in {forecast_result['execution_time_ms']:.2f} ms")
        print(f"Direction: {'Up' if forecast_result['direction'] > 0 else 'Down'} with confidence: {forecast_result['confidence']:.2f}")
        print(f"Forecast values: {forecast_result['forecast']}")
        
        # Test regression
        print("\nTesting regression...")
        regression_result = model.perform_regression(df)
        print(f"Regression completed with {len(regression_result)} values")
        
        # Test different market regimes
        print("\nTesting market regimes...")
        for regime in ["bull", "bear", "volatile", "neutral"]:
            model.set_regime(regime)
            forecast_regime = model.forecast(market_data, timeframe="1h", steps=3)
            print(f"{regime.capitalize()} regime forecast: {forecast_regime['forecast']} (Exec time: {forecast_regime['execution_time_ms']:.2f} ms)")
        
        # Get model info
        print("\nModel information:")
        model_info = model.get_model_info()
        for key, value in model_info.items():
            if not isinstance(value, dict):  # Skip nested dictionaries for cleaner output
                print(f"  {key}: {value}")
        
        print("\nAll tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

# Call main function if script is run directly
if __name__ == "__main__":
    main()