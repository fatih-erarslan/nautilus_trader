import os
import logging
import numpy as np
import pandas as pd
import time
from datetime import datetime
import threading
from typing import Dict, List, Any, Optional, Union, Tuple, Set, Callable
import math
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import warnings
import hashlib
from dataclasses import dataclass, field
from collections import deque
from functools import partial, lru_cache
from enum import Enum, auto
# Try to import numba for JIT compilation where applicable
try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # Create dummy decorators to avoid code changes
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if args and callable(args[0]) else decorator
    
    def prange(*args):
        return range(*args)

# Safe imports for PyTorch
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. Classical GPU processing disabled.")

# Safe imports for SciPy
try:
    import scipy.signal as signal
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("SciPy not available. Some filtering operations disabled.")

# Try to import H2O if needed
try:
    import h2o
    H2O_AVAILABLE = True
except ImportError:
    H2O_AVAILABLE = False
    warnings.warn("H2O not available. H2O models will be disabled.")

# Try to import quantum libraries with fallbacks
try:
    import pennylane as qml
    from pennylane import numpy as qnp  # Use PennyLane's numpy for QNodes
    HAS_PENNYLANE = True
    
    # Try to import PennyLane Catalyst for quantum JIT if available
    try:
        from pennylane import catalyst
        HAS_CATALYST = True
        # Enable Catalyst
        catalyst.compile = True
        catalyst.enable_queuing()
        
        def qjit(func):
            try:
                return catalyst.qjit(func)
            except Exception as e:
                warnings.warn(f"Catalyst qjit failed: {e}. Using original function.")
                return func
    except ImportError:
        HAS_CATALYST = False
        # Create dummy qjit decorator
        def qjit(func): 
            return func  # Dummy decorator
            
except ImportError:
    HAS_PENNYLANE = False
    HAS_CATALYST = False
    qnp = np  # Fallback to standard numpy
    warnings.warn("PennyLane not available. Quantum processing disabled.")
    
    # Create dummy qjit decorator
    def qjit(func): 
        return func  # Dummy decorator

# Import necessary hardware manager class if available
try:
    from hardware_manager import HardwareManager
except ImportError:
    # Create dummy class for type hints only
    class HardwareManager:
        pass
    warnings.warn("HardwareManager import failed. This is required for proper functioning.")
# Define enums for processing options
class ProcessingMode(Enum):
    CLASSICAL = auto()
    QUANTUM = auto()
    HYBRID = auto()
    AUTO = auto()

class QuantumBackend(Enum):
    PENNYLANE = auto()
    QISKIT = auto()
    CIRQ = auto()
    CUSTOM = auto()

class SignalMetadata:
    """Metadata container for signal processing"""
    def __init__(self, sample_rate=1.0, dimension=1, source="unknown"):
        self.sample_rate = sample_rate
        self.dimension = dimension
        self.source = source
        self.processing_history = []


class UniversalSignalProcessor:
    """
    Universal Signal Processor - A consolidated toolkit for signal processing operations
    combining quantum, classical, ML, and fuzzy logic approaches within a Complex Adaptive System framework.
    
    This class provides a comprehensive set of signal processing capabilities:
    1. Classical signal processing (filtering, FFT, noise reduction)
    2. Quantum signal processing (QFT, QPE, quantum filtering)
    3. Machine learning predictions (CatBoost, H2O, ensemble methods)
    4. Neuro-fuzzy evaluations (SANFIS/ANFIS)
    5. Critical state detection and analysis
    6. Adaptive processing mode selection
    7. Optional parallel processing for large datasets
    
    It acts as a toolkit of methods rather than orchestrating the full pipeline.
    Components are initialized selectively based on configuration.
    """
    
    def __init__(
        self,
        # Required hardware manager
        hw_manager,
        
        # Configuration options
        config: Optional[Dict] = None,
        log_level: int = logging.INFO,
        
        # Pre-trained models (direct instances)
        catboost_model: Optional[Any] = None,
        sanfis_model: Optional[Any] = None, 
        h2o_model: Optional[Any] = None,
        nn_scaler: Optional[Any] = None,
        
        # Pre-initialized quantum applications
        qaoa_optimizer: Optional[Any] = None,
        quareg_model: Optional[Any] = None,
        
        # Additional model collections
        pre_trained_models: Optional[Dict[str, Any]] = None,
        quantum_applications: Optional[Dict[str, Any]] = None,
        
        # System parameters
        adaptation_rate: float = 0.01,
        energy_budget: float = 1.0,
        max_memory_entries: int = 1000,
        
        # Processing settings
        default_processing_mode: str = "AUTO",
        n_qubits: Optional[int] = None,
        shots: Optional[int] = None,
        
        # Cache and parallel processing
        cache_size: Optional[int] = None,
        use_parallel: bool = False,
        num_parallel_workers: Optional[int] = None
    ):
        """
        Initialize the UniversalSignalProcessor with hardware manager and optional components.
        
        Args:
            hw_manager: Hardware abstraction layer for device access
            config: Configuration dictionary
            log_level: Logging verbosity
            
            # Pre-trained models
            catboost_model: Pre-trained CatBoost model
            sanfis_model: Pre-trained SANFIS model
            h2o_model: Pre-trained H2O model
            nn_scaler: Neural network scaler
            pre_trained_models: Dictionary of additional models
            
            # Quantum applications
            qaoa_optimizer: Pre-initialized QAOA optimizer
            quareg_model: Pre-initialized QUAREG model
            quantum_applications: Dictionary of additional quantum applications
            
            # System parameters
            adaptation_rate: Learning rate for adaptive components
            energy_budget: Energy budget for component activation
            max_memory_entries: Maximum memory entries
            
            # Processing settings
            default_processing_mode: Default mode (AUTO, CLASSICAL, QUANTUM, HYBRID)
            n_qubits: Override for number of qubits
            shots: Override for quantum shots
            
            # Cache and parallel 
            cache_size: Size of cache for results
            use_parallel: Enable parallel processing
            num_parallel_workers: Number of parallel workers
        """
        # Set up logging
        self.logger = logging.getLogger("universal.signal.processor")
        self.logger.setLevel(log_level)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(handler)
        
        # Store hardware manager
        if hw_manager is None:
            raise ValueError("Hardware manager is required")
        self.hw_manager = hw_manager
        
        # Load configuration
        self.config = config or {}
        
        # Initialize threading lock for thread safety
        self._lock = threading.RLock()
        
        # System parameters
        self.adaptation_rate = adaptation_rate
        self.energy_budget = energy_budget
        self.max_memory_entries = max_memory_entries
        
        # Safe access to hardware resources
        self.torch_device = getattr(self.hw_manager, 'torch_device', 
                                  torch.device('cpu') if TORCH_AVAILABLE else None)
        self.quantum_device = getattr(self.hw_manager, 'device', None)
        
        # Use override values if provided, otherwise get from hardware manager
        self.n_qubits = n_qubits or getattr(self.hw_manager, 'n_qubits', 8)
        self.shots = shots or getattr(getattr(self.hw_manager, 'config', {}), 'shots', 1000)
        
        # Default processing mode
        self.default_processing_mode = default_processing_mode
        
        # Initialize component status
        self.component_status = {}
        self.is_initialized = False
        
        # Initialize critical state detector
        self.current_regime = "unknown"
        self.system_entropy = 0.0
        self.critical_state_detector = self._initialize_critical_detector()
        
        # Initialize caches with specified or default size
        self.cache_size = cache_size or self.config.get('cache_size', 128)
        self._results_cache = {}      # For processed signals
        self._qnode_cache = {}        # For quantum circuits
        self._filter_cache = {}       # For filter results
        self._fft_cache = {}          # For FFT results
        self._prediction_cache = {}   # For ML predictions
        self._internal_cache = {}     # Generic cache
        self._tail_risk_cache = {}    # For extreme events
        
        # Parallel processing configuration
        self.use_parallel = use_parallel
        self.num_parallel_workers = num_parallel_workers or max(1, os.cpu_count() - 1)
        self.pool = None
        
        # Performance tracking
        self.metrics = {
            "total_processed_signals": 0,
            "classical_processing_time": 0.0,
            "quantum_processing_time": 0.0,
            "ml_processing_time": 0.0,
            "errors": 0,
            "cache_hits": 0
        }
        
        # Store model references directly
        self.catboost = catboost_model
        self.sanfis = sanfis_model
        self.h2o_model = h2o_model
        self.nn_scaler = nn_scaler
        self.qaoa = qaoa_optimizer
        self.quareg = quareg_model
        
        # Also store models from dictionary
        self.ml_models = {}
        if pre_trained_models:
            self.ml_models.update(pre_trained_models)
        if catboost_model and "catboost" not in self.ml_models:
            self.ml_models["catboost"] = catboost_model
        if sanfis_model and "sanfis" not in self.ml_models:
            self.ml_models["sanfis"] = sanfis_model
        if h2o_model and "h2o" not in self.ml_models:
            self.ml_models["h2o"] = h2o_model
        
        # Store quantum applications
        self.quantum_apps = {}
        if quantum_applications:
            self.quantum_apps.update(quantum_applications)
        if qaoa_optimizer and "qaoa" not in self.quantum_apps:
            self.quantum_apps["qaoa"] = qaoa_optimizer
        if quareg_model and "quareg" not in self.quantum_apps:
            self.quantum_apps["quareg"] = quareg_model
        
        # Decision memory for feedback systems
        self.decision_memory = deque(maxlen=max_memory_entries)
        
        # Required columns for feature generation
        self.required_columns = ["open", "high", "low", "close", "volume"]
        self.feature_lookback = self.config.get("feature_lookback", 20)
        
        # Update component status
        self._log_component_status() # Call the existing logging method

        # Mark as initialized
        self.is_initialized = True
        self.logger.info(f"UniversalSignalProcessor initialized. Torch: {self.torch_device}, "
                         f"PL Device: {self.quantum_device}, N-Qubits: {self.n_qubits}")
        #self._log_component_status()

    def _init_classical_processing(self):
        """Initialize classical processing components"""
        try:
            # Set up PyTorch device
            if torch.cuda.is_available() and self.hw_manager and hasattr(self.hw_manager, 'gpu_info'):
                device_id = 0
                if hasattr(self.hw_manager.gpu_info, 'gpus'):
                    # Use the first available GPU
                    available_gpus = list(self.hw_manager.gpu_info.gpus.keys())
                    if available_gpus:
                        device_id = available_gpus[0]
                
                self.torch_device = torch.device(f"cuda:{device_id}")
                self.logger.info(f"Using GPU acceleration on device {device_id}")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.torch_device = torch.device("mps")
                self.logger.info("Using Apple Metal Performance Shaders acceleration")
            else:
                self.torch_device = torch.device("cpu")
                self.logger.info("Using CPU for classical processing")
            
            self.component_status["classical_processing"] = True
        except Exception as e:
            self.logger.warning(f"Error initializing classical processing: {e}")
            self.torch_device = torch.device("cpu")
            self.component_status["classical_processing"] = False

    def _init_quantum_processing(self):
        """Initialize quantum processing components if available"""
        if not HAS_PENNYLANE:
            self.logger.warning("PennyLane not available. Quantum processing disabled.")
            self.component_status["quantum_processing"] = False
            self.quantum_device = None
            return
        
        try:
            # Try to use hardware acceleration if available
            if hasattr(self.hw_manager, 'gpu_info') and self.hw_manager.gpu_info.gpus:
                try:
                    # Check for Lightning GPU backend (NVIDIA)
                    if hasattr(qml, "device") and "lightning.gpu" in qml.device.device_list:
                        self.logger.info("Using PennyLane Lightning GPU backend")
                        self.quantum_device = qml.device(
                            "lightning.gpu", wires=self.n_qubits, shots=self.shots
                        )
                        if HAS_CATALYST:
                            self.logger.info("Catalyst JIT compilation enabled for quantum circuits")
                    # Check for Lightning Kokkos backend (AMD)
                    elif hasattr(qml, "device") and "lightning.kokkos" in qml.device.device_list:
                        self.logger.info("Using PennyLane Lightning Kokkos backend with ROCm/HIP")
                        self.quantum_device = qml.device(
                            "lightning.kokkos",
                            wires=self.n_qubits,
                            shots=self.shots,
                            kokkos_args={"threads": min(16, os.cpu_count() or 4), "use_gpu": True},
                        )
                        if HAS_CATALYST:
                            self.logger.info("Catalyst JIT compilation enabled for quantum circuits")
                    else:
                        self.logger.warning("No GPU-accelerated quantum device found, falling back to CPU")
                        self.quantum_device = qml.device(
                            "lightning.qubit", wires=self.n_qubits, shots=self.shots
                        )
                except (ImportError, RuntimeError) as e:
                    self.logger.warning(f"GPU quantum device unavailable: {e}")
                    self.quantum_device = qml.device(
                        "lightning.qubit", wires=self.n_qubits, shots=self.shots
                    )
            else:
                self.quantum_device = qml.device(
                    "lightning.qubit", wires=self.n_qubits, shots=self.shots
                )
                self.logger.info("Using CPU-based quantum simulation")

            # Test the quantum device with a simple circuit
            test_circuit = self._get_cached_qnode("test_circuit", 1, lambda: self._build_test_circuit())
            
            result = test_circuit()
            self.logger.info(f"Quantum processing initialized: {result}")
            self.component_status["quantum_processing"] = True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize quantum processing: {e}")
            self.quantum_device = None
            self.component_status["quantum_processing"] = False

    def _init_ml_components(self):
        """Initialize machine learning components"""
        try:
            # Import ML models if provided
            self.ml_models = {}
            if "catboost" in self.pre_trained_models:
                self.ml_models["catboost"] = self.pre_trained_models["catboost"]
                self.component_status["catboost"] = True
            else:
                self.component_status["catboost"] = False
            
            # H2O initialization if needed
            if "h2o" in self.pre_trained_models:
                try:
                    h2o.init()
                    self.ml_models["h2o"] = self.pre_trained_models["h2o"]
                    self.component_status["h2o"] = True
                except Exception as e:
                    self.logger.warning(f"Failed to initialize H2O: {e}")
                    self.component_status["h2o"] = False
            else:
                self.component_status["h2o"] = False
            
            # Feature generation capabilities
            self.feature_lookback = self.config.get("feature_lookback", 20)
            self.required_columns = ["open", "high", "low", "close", "volume"]
            self.component_status["feature_generation"] = True
            
            # Track performance metrics
            self.ml_metrics = {"accuracy": 0, "samples": 0}
            
            self.logger.info("ML components initialized")
        except Exception as e:
            self.logger.error(f"Error initializing ML components: {e}")
            self.component_status["ml_components"] = False

    def _init_fuzzy_components(self):
        """Initialize neuro-fuzzy components with proper error handling for SANFIS"""
        try:
            # Try to initialize SANFIS properly if pre-trained model provided
            if "sanfis" in self.pre_trained_models:
                self.sanfis = self.pre_trained_models["sanfis"]
                self.component_status["sanfis"] = True
            else:
                # Create a new SANFIS if config is provided
                if "sanfis_config" in self.config:
                    sanfis_config = self.config["sanfis_config"]
                    try:
                        # Import SANFIS safely
                        sanfis_cls = self.config.get("sanfis_class")
                        if sanfis_cls:
                            # Extract parameters safely with defaults
                            input_dim = sanfis_config.get("input_dim", 10)
                            output_dim = sanfis_config.get("output_dim", 1)
                            num_rules = sanfis_config.get("num_rules", 5)
                            learning_rate = sanfis_config.get("learning_rate", 0.01)
                            adaptive_rules = sanfis_config.get("adaptive_rules", True)
                            
                            # Initialize using explicit keyword arguments
                            self.sanfis = sanfis_cls(
                                input_dim=input_dim,
                                output_dim=output_dim,
                                num_rules=num_rules,
                                learning_rate=learning_rate,
                                adaptive_rules=adaptive_rules
                            )
                            self.component_status["sanfis"] = True
                            self.logger.info("SANFIS model initialized successfully")
                        else:
                            self.component_status["sanfis"] = False
                            self.logger.warning("SANFIS class not provided in config")
                    except Exception as e:
                        self.component_status["sanfis"] = False
                        self.logger.error(f"Failed to initialize SANFIS: {e}")
                else:
                    self.component_status["sanfis"] = False
            
            # ANFIS initialization if provided
            if "anfis" in self.pre_trained_models:
                self.anfis = self.pre_trained_models["anfis"]
                self.component_status["anfis"] = True
            else:
                self.component_status["anfis"] = False
            
            self.logger.info("Fuzzy components initialized")
        except Exception as e:
            self.logger.error(f"Error initializing fuzzy components: {e}")
            self.component_status["fuzzy_components"] = False

    def _initialize_critical_detector(self):
        """Initialize critical state detector for Self-Organized Criticality detection"""
        return {
            "power_law_exponent": 0.0,  # Avalanche size distribution exponent
            "fractal_dimension": 0.0,  # Fractal dimension of price movements
            "critical_point_estimate": 0.0,  # Estimate of distance to critical point
            "warning_level": "normal",  # Current warning level
            "realized_volatility": 0.0,  # Last realized volatility value
            "critical_indicators": {  # Indicators of criticality
                "synchronization": 0.0,  # Component synchronization level
                "long_range_correlations": 0.0,  # Long-range correlations in data
                "complexity_measures": 0.0,  # Algorithmic complexity measures
            },
        }

    # ===== QNode CACHE AND BUILDER METHODS =====
    
    def _get_cached_qnode(self, name, n_wires, circuit_builder_fn):
        """
        Get or create a cached QNode with optional Catalyst JIT compilation
        
        Args:
            name: Unique name for the QNode
            n_wires: Number of wires (qubits) required
            circuit_builder_fn: Function that returns the circuit definition
        
        Returns:
            QNode: The quantum circuit
        """
        cache_key = f"{name}_{n_wires}"
        if cache_key in self._qnode_cache:
            return self._qnode_cache[cache_key]
        
        if not hasattr(self, "quantum_device") or self.quantum_device is None:
            raise ValueError("Quantum device not initialized")
        
        # Build the circuit
        circuit_fn = circuit_builder_fn()
        
        # Apply Catalyst JIT if available
        if HAS_CATALYST:
            try:
                # Apply Catalyst JIT to the circuit function
                circuit_fn = qjit(circuit_fn)
            except Exception as e:
                self.logger.warning(f"Failed to apply Catalyst JIT to circuit {name}: {e}")
        
        # Create the QNode
        try:
            qnode = qml.QNode(circuit_fn, self.quantum_device)
            self._qnode_cache[cache_key] = qnode
            return qnode
        except Exception as e:
            self.logger.error(f"Failed to create QNode {name}: {e}")
            raise

    def _build_test_circuit(self):
        """Build a simple test circuit for device validation"""
        def circuit():
            qml.Hadamard(wires=0)
            return qml.expval(qml.PauliZ(0))
        return circuit

    def _build_qft_circuit(self, inverse=False):
        """Build a Quantum Fourier Transform circuit"""
        def circuit(data):
            # Encode normalized data as amplitudes
            qml.AmplitudeEmbedding(data, wires=range(len(data)), normalize=True)
            
            # Apply QFT or inverse QFT
            if not inverse:
                qml.QFT(wires=range(len(data)))
            else:
                qml.adjoint(qml.QFT)(wires=range(len(data)))
            
            # Return the quantum state
            return qml.state()
        return circuit

    def _build_qpe_circuit(self, precision):
        """Build a Quantum Phase Estimation circuit with specified precision"""
        def circuit(phase):
            # Initialize target qubit
            qml.Hadamard(wires=precision)
            
            # Initialize estimation qubits
            for qubit in range(precision):
                qml.Hadamard(wires=qubit)
            
            # Apply controlled rotations
            for qubit in range(precision):
                power = 2 ** (qubit)
                for _ in range(power):
                    qml.ctrl(qml.PhaseShift, control=qubit)(
                        phase * np.pi * 2, wires=precision
                    )
            
            # Apply inverse QFT to estimation qubits
            qml.adjoint(qml.QFT)(wires=range(precision))
            
            # Measure estimation qubits
            return [qml.expval(qml.PauliZ(j)) for j in range(precision)]
        return circuit

    def _build_quantum_filter_circuit(self):
        """Build a quantum filtering circuit"""
        def circuit(sample_value, threshold):
            # Encode value in qubit rotation
            qml.RY(np.pi * sample_value, wires=0)
            
            # Apply conditional filter based on threshold
            qml.cond(qml.expval(qml.PauliZ(0)) < threshold, qml.Hadamard)(wires=0)
            
            # Apply additional filtering layers
            qml.RZ(np.pi / 4, wires=0)
            qml.Hadamard(wires=0)
            
            # Measure to get filtered value
            return qml.expval(qml.PauliZ(0))
        return circuit

    # ===== CORE SIGNAL PROCESSING METHODS =====

    def _log_component_status(self):
        """Log status of internal component references"""
        status = {
            "PyTorch": TORCH_AVAILABLE and self.torch_device is not None,
            "SciPy": SCIPY_AVAILABLE,
            "PennyLane": HAS_PENNYLANE and self.quantum_device is not None,
            "Catalyst": HAS_CATALYST,
            "CatBoost Model": self.catboost is not None,
            "SANFIS Model": self.sanfis is not None,
            "H2O Model": self.h2o_model is not None,
            "QAOA Opt": self.qaoa is not None,
            "QUAREG Model": self.quareg is not None
        }
        self.logger.info(f"USP Component Availability: {status}")
    
    def preprocess_signal(self, signal_data: np.ndarray) -> Tuple[np.ndarray, SignalMetadata]:
        """
        Normalizes signal to [-1, 1], handles shape, returns metadata.
        Vectorized implementation with proper error handling.
        """
        metadata = SignalMetadata()
        start_time = time.time()
        notes = []
        
        # Generate cache key for possible caching
        cache_key = self._safe_cache_key("preprocess", signal_data.shape)
        if cache_key in self._internal_cache:
            self.metrics["cache_hits"] += 1
            return self._internal_cache[cache_key], metadata
            
        try:
            # Ensure numpy array (vectorized)
            if not isinstance(signal_data, np.ndarray):
                signal_data = np.array(signal_data, dtype=np.float64)
            
            # Store original shape
            original_shape = signal_data.shape
            metadata.original_shape = original_shape
            
            # Reshape 1D arrays to 2D (vectorized)
            if signal_data.ndim == 1:
                signal_data = signal_data[:, np.newaxis]
            
            metadata.dimension = signal_data.shape[1]
            
            # Handle NaN/Inf values (vectorized)
            if np.isnan(signal_data).any() or np.isinf(signal_data).any():
                signal_data = np.nan_to_num(signal_data, nan=0.0, posinf=1.0, neginf=-1.0)
                notes.append("NaN/Inf values replaced")
            
            # Normalize to [-1, 1] range (vectorized)
            signal_max = np.max(np.abs(signal_data))
            if signal_max > 1e-9:
                normalized_signal = signal_data / signal_max
            else:
                normalized_signal = np.zeros_like(signal_data)
                notes.append("Signal max amplitude near zero, normalization skipped")
            
            # Update metadata
            metadata.processing_history.append("normalization")
            metadata.notes = "; ".join(notes) if notes else None
            metadata.processing_time = time.time() - start_time
            
            # Restore original shape if needed
            if original_shape == () or (len(original_shape) == 1 and normalized_signal.ndim > 1):
                result = normalized_signal.squeeze()
            else:
                result = normalized_signal
                
            # Cache result for future use
            self._update_cache(self._internal_cache, cache_key, result, self.cache_size)
                
            return result, metadata
                
        except Exception as e:
            self.logger.error(f"Error in preprocess_signal: {e}")
            metadata.notes = f"Preprocess Error: {e}"
            metadata.processing_time = time.time() - start_time
            return signal_data, metadata
    
    def apply_classical_filter(
        self, 
        signal_data: np.ndarray, 
        filter_type: str = 'lowpass', 
        filter_order: int = 4, 
        cutoff_freq: Union[float, List[float]] = 0.2,
        sample_rate: Optional[float] = None
    ) -> Tuple[np.ndarray, SignalMetadata]:
        """
        Apply classical filter to signal data with vectorized operations.
        Uses caching for repeated filter operations.
        """
        metadata = SignalMetadata(sample_rate=sample_rate)
        start_time = time.time()
        notes = []
        
        if not SCIPY_AVAILABLE:
            notes.append("SciPy missing")
            metadata.notes = "; ".join(notes)
            metadata.processing_time = 0.0
            return signal_data, metadata
        
        # Generate cache key
        cache_key = self._safe_cache_key("filter", filter_type, filter_order, cutoff_freq, 
                                       sample_rate, signal_data.shape)
        
        # Check cache
        if cache_key in self._filter_cache:
            self.metrics["cache_hits"] += 1
            return self._filter_cache[cache_key], metadata
        
        try:
            # Store original shape/dimensionality
            is_1d = signal_data.ndim == 1
            data_2d = signal_data[:, np.newaxis] if is_1d else signal_data
            
            # Calculate normalized cutoff frequency
            nyquist = 0.5 * (sample_rate if sample_rate else 1.0)
            Wn = np.array(cutoff_freq) / nyquist if hasattr(cutoff_freq, '__len__') else cutoff_freq / nyquist
            
            # Validate cutoff frequency
            if np.any(Wn <= 0) or np.any(Wn >= 1):
                raise ValueError("Cutoff frequency invalid relative to Nyquist")
            
            # Design filter (vectorized)
            sos = signal.butter(filter_order, Wn, btype=filter_type, output='sos')
            
            # Apply filter to each channel (can't fully vectorize due to filtfilt)
            filtered_data = np.zeros_like(data_2d)
            for i in range(data_2d.shape[1]):
                filtered_data[:, i] = signal.sosfiltfilt(sos, data_2d[:, i])
            
            # Restore original shape
            output_data = filtered_data.squeeze() if is_1d else filtered_data
            
            metadata.processing_history.append(f"classical_filter_{filter_type}")
            
            # Cache the result
            self._update_cache(self._filter_cache, cache_key, output_data, self.cache_size)
            
        except ValueError as ve:
            notes.append(f"Filter Value Error: {ve}")
            output_data = signal_data
            self.logger.warning(f"Filter error: {ve}")
        except Exception as e:
            notes.append(f"Filter Unknown Error: {e}")
            output_data = signal_data
            self.logger.error(f"Filter error: {e}")
        
        metadata.processing_time = time.time() - start_time
        metadata.notes = "; ".join(notes) if notes else None
        
        return output_data, metadata
    
    def apply_fft_processing(
        self, 
        signal_data: np.ndarray, 
        method: str = 'spectral_gating',
        threshold_factor: float = 0.1
    ) -> Tuple[np.ndarray, SignalMetadata]:
        """
        Apply FFT-based processing to signal data using vectorized PyTorch operations.
        Implements caching for performance.
        """
        metadata = SignalMetadata()
        start_time = time.time()
        notes = []
        
        if not TORCH_AVAILABLE or not self.torch_device:
            notes.append("Torch/Device unavailable")
            metadata.notes = "; ".join(notes)
            metadata.processing_time = 0.0
            return signal_data, metadata
        
        # Generate cache key
        cache_key = self._safe_cache_key("fft", method, threshold_factor, signal_data.shape)
        
        # Check cache
        if cache_key in self._fft_cache:
            self.metrics["cache_hits"] += 1
            return self._fft_cache[cache_key], metadata
        
        try:
            # Convert to torch tensor for GPU acceleration
            signal_tensor = torch.tensor(signal_data, device=self.torch_device, dtype=torch.float32)
            original_shape = signal_tensor.shape
            original_ndim = signal_tensor.ndim
            
            # Ensure 2D shape for processing
            if original_ndim == 1:
                signal_tensor = signal_tensor.unsqueeze(-1)
            
            # Apply FFT (vectorized)
            signal_fft = torch.fft.rfft(signal_tensor, dim=0)
            
            # Process in frequency domain
            if method == 'spectral_gating':
                # Calculate magnitude (vectorized)
                magnitude = torch.abs(signal_fft)
                
                # Calculate threshold (vectorized)
                threshold = torch.mean(magnitude, dim=0, keepdim=True) * threshold_factor
                
                # Apply threshold mask (vectorized)
                mask = magnitude > threshold
                signal_fft_processed = signal_fft * mask
            else:
                signal_fft_processed = signal_fft
                notes.append(f"Unknown FFT Method {method}")
            
            # Apply inverse FFT (vectorized)
            processed_tensor = torch.fft.irfft(signal_fft_processed, dim=0, n=original_shape[0])
            
            # Convert back to numpy (vectorized)
            processed_data = processed_tensor.cpu().numpy()
            
            # Restore original shape
            if original_ndim == 1 and processed_data.ndim > 1:
                processed_data = processed_data.squeeze(-1)
            
            metadata.processing_history.append(f"fft_{method}")
            
            # Cache the result
            self._update_cache(self._fft_cache, cache_key, processed_data, self.cache_size)
            
        except Exception as e:
            notes.append(f"FFT Error: {e}")
            processed_data = signal_data
            self.logger.error(f"FFT processing error: {e}")
        
        metadata.processing_time = time.time() - start_time
        metadata.notes = "; ".join(notes) if notes else None
        
        return processed_data, metadata
    
    
    def get_ml_prediction(self, features: pd.DataFrame, model_type: str='ensemble') -> Tuple[pd.Series, SignalMetadata]:
        """
        Uses pre-trained ML models to predict. Implements caching and vectorization.
        """
        metadata = SignalMetadata(processing_mode_used=f"ml_predict_{model_type}")
        start_time = time.time()
        notes = []
        
        # Default output for errors
        default_output = pd.Series(0.5, index=features.index)
        
        # Generate cache key for results
        cache_key = self._generate_prediction_cache_key(features, model_type)
        
        # Check cache first
        if cache_key in self._prediction_cache:
            self.metrics["cache_hits"] += 1
            return self._prediction_cache[cache_key], metadata
        
        # Collect predictions from available models
        predictions = {}
        
        # Try CatBoost if requested and available
        if model_type in ['ensemble', 'catboost'] and self.catboost:
            try:
                # Vectorized prediction
                catboost_pred = self.catboost.predict_proba(features)[:, 1]
                predictions['catboost'] = pd.Series(catboost_pred, index=features.index)
            except Exception as e:
                notes.append(f"CatBoost Predict Fail: {e}")
        
        # Try H2O if requested and available
        if model_type in ['ensemble', 'h2o'] and self.h2o_model:
            try:
                # Convert to H2O frame (H2O handles batching internally)
                import h2o
                h2o_df = h2o.H2OFrame(features)
                preds_df = self.h2o_model.predict(h2o_df).as_data_frame()
                
                # Extract predictions (usually 'p1' column for classification)
                if 'p1' in preds_df.columns:
                    predictions['h2o'] = pd.Series(preds_df['p1'].values, index=features.index)
                else:
                    predictions['h2o'] = pd.Series(preds_df.iloc[:, 0].values, index=features.index)
            except Exception as e:
                notes.append(f"H2O Predict Fail: {e}")
        
        # Combine predictions or choose appropriate one
        if model_type == 'ensemble' and len(predictions) > 0:
            # Create DataFrame of all predictions (vectorized)
            pred_df = pd.DataFrame(predictions, index=features.index)
            # Simple averaging ensemble (vectorized)
            final_prediction = pred_df.mean(axis=1)
        elif len(predictions) > 0:
            # Use requested model if available, otherwise first available
            if model_type in predictions:
                final_prediction = predictions[model_type]
            else:
                final_prediction = next(iter(predictions.values()))
        else:
            # No predictions could be made
            notes.append("No valid ML models available/runnable")
            final_prediction = default_output
    
        # Ensure valid output (vectorized)
        output_series = final_prediction.fillna(0.5).clip(0, 1)
        
        # Update metadata
        metadata.processing_time = time.time() - start_time
        metadata.notes = "; ".join(notes) if notes else None
        
        # Cache result
        self._update_cache(self._prediction_cache, cache_key, output_series, self.cache_size)
        
        return output_series, metadata
    
    def get_sanfis_prediction(self, features: pd.DataFrame) -> Tuple[pd.Series, SignalMetadata]:
        """
        Get prediction from pre-trained SANFIS model with caching and vectorization where possible.
        """
        metadata = SignalMetadata(processing_mode_used="fuzzy_sanfis")
        start_time = time.time()
        notes = []
        
        # Default output
        output = pd.Series(0.5, index=features.index)
        
        # Generate cache key
        cache_key = self._safe_cache_key("sanfis_pred", features.shape)
        
        # Check cache
        if cache_key in self._prediction_cache:
            self.metrics["cache_hits"] += 1
            cached_output, cached_notes = self._prediction_cache[cache_key]
            if cached_notes:
                notes.append(cached_notes)
            metadata.notes = "; ".join(notes) if notes else None
            metadata.processing_time = 0.0
            return cached_output, metadata
        
        if not self.sanfis:
            notes.append("SANFIS Model Not Initialized")
            metadata.notes = "; ".join(notes)
            metadata.processing_time = 0.0
            return output, metadata
        
        try:
            # Check if SANFIS has vectorized prediction method
            if hasattr(self.sanfis, "predict_batch") and callable(self.sanfis.predict_batch):
                # Use vectorized batch prediction (much faster)
                feature_array = features.values
                predicted_vals = self.sanfis.predict_batch(feature_array)
                
                # Convert to Series
                if isinstance(predicted_vals, np.ndarray):
                    if predicted_vals.ndim > 1 and predicted_vals.shape[1] > 0:
                        output = pd.Series(predicted_vals[:, 0], index=features.index)
                    else:
                        output = pd.Series(predicted_vals, index=features.index)
                else:
                    output = pd.Series(predicted_vals, index=features.index)
            else:
                # Fallback to row-by-row prediction
                predicted_vals = np.zeros(len(features))
                
                for i in range(len(features)):
                    # Get feature row as numpy array
                    single_feature_row = features.iloc[i].values
                    
                    # Apply SANFIS prediction
                    pred = self.sanfis.predict(single_feature_row)
                    
                    # Handle different return types
                    if isinstance(pred, (list, np.ndarray)) and len(pred) > 0:
                        predicted_vals[i] = float(pred[0])
                    else:
                        predicted_vals[i] = float(pred)
                
                # Create Series from predictions
                output = pd.Series(predicted_vals, index=features.index)
        
        except Exception as e:
            notes.append(f"SANFIS Predict Fail: {e}")
            self.logger.error(f"SANFIS prediction error: {e}")
        
        # Ensure valid output (vectorized)
        output = output.fillna(0.5).clip(0, 1)
        
        # Cache result
        cache_note = "; ".join(notes) if notes else None
        self._update_cache(self._prediction_cache, cache_key, (output, cache_note), self.cache_size)
        
        # Update metadata
        metadata.processing_time = time.time() - start_time
        metadata.notes = cache_note
        
        return output, metadata
    
    def refine_with_qaoa(self, signal_series: pd.Series, **qaoa_params) -> Tuple[pd.Series, SignalMetadata]:
        """
        Uses pre-initialized QAOA Optimizer to refine a signal, with caching.
        """
        metadata = SignalMetadata(processing_mode_used="qaoa_refine")
        start_time = time.time()
        notes = []
        
        # Default to input
        output = signal_series.copy()
        
        # Generate cache key - careful with params!
        cache_key = self._safe_cache_key("qaoa_refine", signal_series.values, 
                                        str(sorted(qaoa_params.items())))
        
        # Check cache
        if cache_key in self._internal_cache:
            self.metrics["cache_hits"] += 1
            cached_output, cached_notes = self._internal_cache[cache_key]
            if cached_notes:
                notes.append(cached_notes)
            metadata.notes = "; ".join(notes) if notes else None
            metadata.processing_time = 0.0
            return cached_output, metadata
        
        if not self.qaoa:
            notes.append("QAOA Instance Unavailable")
            metadata.notes = "; ".join(notes)
            metadata.processing_time = 0.0
            return output, metadata
        
        try:
            # Apply QAOA refinement
            output_series = self.qaoa.refine_signal(signal_series, **qaoa_params)
            
            # Check return type
            if isinstance(output_series, pd.Series):
                output = output_series
            else:
                notes.append("QAOA refine did not return Series")
                
                # Try to convert to Series
                if isinstance(output_series, np.ndarray) and len(output_series) == len(signal_series):
                    output = pd.Series(output_series, index=signal_series.index)
        except Exception as e:
            notes.append(f"QAOA Error: {e}")
            self.logger.error(f"Error in QAOA refinement: {e}")
        
        # Ensure valid output (vectorized)
        output = output.fillna(signal_series).clip(0, 1)
        
        # Cache result
        cache_note = "; ".join(notes) if notes else None
        self._update_cache(self._internal_cache, cache_key, (output, cache_note), self.cache_size)
        
        # Update metadata
        metadata.processing_time = time.time() - start_time
        metadata.notes = cache_note
        
        return output, metadata
    
    def predict_with_quareg(self, features: np.ndarray) -> Tuple[np.ndarray, SignalMetadata]:
        """
        Uses pre-initialized QUAREG model for prediction, with caching.
        """
        metadata = SignalMetadata(processing_mode_used="quareg_predict")
        start_time = time.time()
        notes = []
        
        # Default output
        default_output = np.full(len(features) if features is not None else 1, 0.5)
        
        # Generate cache key
        cache_key = self._safe_cache_key("quareg_pred", features)
        
        # Check cache
        if cache_key in self._internal_cache:
            self.metrics["cache_hits"] += 1
            cached_output, cached_notes = self._internal_cache[cache_key]
            if cached_notes:
                notes.append(cached_notes)
            metadata.notes = "; ".join(notes) if notes else None
            metadata.processing_time = 0.0
            return cached_output, metadata
        
        if not self.quareg:
            notes.append("QUAREG Instance Unavailable")
            metadata.notes = "; ".join(notes)
            metadata.processing_time = 0.0
            return default_output, metadata
        
        try:
            # Apply QUAREG prediction
            predicted_values = self.quareg.predict(features)
            output = predicted_values
        except Exception as e:
            notes.append(f"QUAREG Predict Error: {e}")
            self.logger.error(f"Error in QUAREG prediction: {e}")
            output = default_output
        
        # Cache result
        cache_note = "; ".join(notes) if notes else None
        self._update_cache(self._internal_cache, cache_key, (output, cache_note), self.cache_size)
        
        # Update metadata
        metadata.processing_time = time.time() - start_time
        metadata.notes = cache_note
        
        return output, metadata

    def process_signal(
        self,
        signal_data: np.ndarray,
        mode: Optional[ProcessingMode] = None,
        metadata: Optional[SignalMetadata] = None,
        **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Process signal data using adaptive mode selection between classical and quantum methods.
        
        Args:
            signal_data: Input signal data as a numpy array
            mode: Processing mode (None = use default mode)
            metadata: Additional signal metadata
            **kwargs: Additional processing parameters
        
        Returns:
            Tuple of (processed_signal, processing_metadata)
        """
        start_time = time.time()
        
        # Create metadata if not provided
        if metadata is None:
            metadata = SignalMetadata(
                sample_rate=kwargs.get("sample_rate", 1.0),
                dimension=signal_data.shape[-1] if len(signal_data.shape) > 1 else 1,
                source=kwargs.get("source", "unknown"),
            )
        
        # Determine processing mode
        processing_mode = mode if mode is not None else self.default_processing_mode
        
        # Preprocess signal
        preprocessed_signal = self._preprocess_signal(signal_data, metadata)
        
        # Generate cache key for result caching
        cache_key = self._generate_cache_key(preprocessed_signal, processing_mode, kwargs)
        if cache_key in self._results_cache:
            self.metrics["cache_hits"] += 1
            cached_result, cached_meta = self._results_cache[cache_key]
            cached_meta["cache_hit"] = True
            cached_meta["processing_time"] = time.time() - start_time
            return cached_result, cached_meta
        
        try:
            # Process based on mode and component availability
            if (processing_mode == ProcessingMode.QUANTUM and 
                self.component_status.get("quantum_processing", False)):
                processed_signal, proc_metadata = self._process_quantum(
                    preprocessed_signal, metadata, **kwargs
                )
                actual_mode = "quantum"
            
            elif (processing_mode == ProcessingMode.HYBRID and 
                  self.component_status.get("quantum_processing", False) and
                  self.component_status.get("classical_processing", False)):
                processed_signal, proc_metadata = self._process_hybrid(
                    preprocessed_signal, metadata, **kwargs
                )
                actual_mode = "hybrid"
            
            elif processing_mode == ProcessingMode.AUTO:
                # Auto mode: decide based on signal characteristics and available components
                if (self.component_status.get("quantum_processing", False) and 
                    self._should_use_quantum(preprocessed_signal)):
                    processed_signal, proc_metadata = self._process_quantum(
                        preprocessed_signal, metadata, **kwargs
                    )
                    actual_mode = "quantum"
                else:
                    processed_signal, proc_metadata = self._process_classical(
                        preprocessed_signal, metadata, **kwargs
                    )
                    actual_mode = "classical"
            
            else:
                # Default to classical processing
                processed_signal, proc_metadata = self._process_classical(
                    preprocessed_signal, metadata, **kwargs
                )
                actual_mode = "classical"
            
            # Update metrics
            self.metrics["total_processed_signals"] += 1
            
            # Cache result
            if self.cache_size > 0:
                self._update_cache(cache_key, (processed_signal, proc_metadata))
            
            return processed_signal, proc_metadata
        
        except Exception as e:
            self.logger.error(f"Error processing signal: {str(e)}")
            self.metrics["errors"] += 1
            # Fallback to classical processing
            return self._process_classical(preprocessed_signal, metadata, **kwargs)

    def _preprocess_signal(
        self, signal_data: np.ndarray, metadata: SignalMetadata
    ) -> np.ndarray:
        """
        Preprocess signal data before quantum or classical processing.
        
        Args:
            signal_data: Raw input signal
            metadata: Signal metadata
        
        Returns:
            Preprocessed signal ready for processing
        """
        # Ensure signal is a numpy array
        if not isinstance(signal_data, np.ndarray):
            signal_data = np.array(signal_data)
        
        # Reshape if needed (vectorized operation)
        if signal_data.ndim == 1:
            signal_data = signal_data.reshape(-1, 1)
        
        # Normalize signal to [-1, 1] range for quantum processing (vectorized)
        signal_max = np.max(np.abs(signal_data))
        if signal_max > 0:
            normalized_signal = signal_data / signal_max
        else:
            normalized_signal = signal_data
        
        # Replace NaN values with zeros (vectorized)
        normalized_signal = np.nan_to_num(normalized_signal, nan=0.0)
        
        # Add preprocessing to metadata history
        metadata.processing_history.append("normalization")
        
        return normalized_signal

    def _process_classical(
        self, signal_data: np.ndarray, metadata: SignalMetadata, **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Process signal using classical signal processing techniques.
        
        Args:
            signal_data: Preprocessed input signal
            metadata: Signal metadata
            **kwargs: Additional processing parameters
        
        Returns:
            Tuple of (processed_signal, processing_metadata)
        """
        start_time = time.time()
        
        # Get processing parameters
        filter_type = kwargs.get("filter_type", "lowpass")
        filter_order = kwargs.get("filter_order", 4)
        cutoff_freq = kwargs.get("cutoff_freq", 0.2)
        use_fft = kwargs.get("use_fft", True)
        
        # Generate cache keys for optional caching
        filter_key = None
        fft_key = None
        if filter_type in ["lowpass", "highpass", "bandpass"]:
            filter_key = f"{filter_type}_{filter_order}_{cutoff_freq}_{signal_data.shape}"
        if use_fft:
            fft_key = f"fft_{signal_data.shape}"
        
        # Convert to torch tensor for GPU acceleration if available
        if hasattr(self, "torch_device"):
            signal_tensor = torch.tensor(signal_data, device=self.torch_device)
        else:
            signal_tensor = torch.tensor(signal_data)
        
        # Apply filtering if requested
        if filter_type in ["lowpass", "highpass", "bandpass"]:
            # Check cache first
            if filter_key and filter_key in self._filter_cache:
                signal_tensor = self._filter_cache[filter_key]
            else:
                # Create filter coefficients using SciPy
                if filter_type == "lowpass":
                    b, a = signal.butter(filter_order, cutoff_freq, "lowpass")
                elif filter_type == "highpass":
                    b, a = signal.butter(filter_order, cutoff_freq, "highpass")
                elif filter_type == "bandpass":
                    b, a = signal.butter(filter_order, cutoff_freq, "bandpass")
                
                # Apply filter using PyTorch's conv1d for GPU acceleration
                if signal_tensor.shape[0] > 1:  # Only if we have enough samples
                    # Vectorized operations using PyTorch
                    b_tensor = torch.tensor(b, device=signal_tensor.device).float()
                    padded_signal = torch.nn.functional.pad(
                        signal_tensor.T.float(), (len(b) - 1, 0)
                    )
                    filtered_signal = (
                        torch.nn.functional.conv1d(
                            padded_signal.unsqueeze(0), b_tensor.view(1, 1, -1)
                        )
                        .squeeze(0)
                        .T
                    )
                    signal_tensor = filtered_signal
                    
                    # Cache the result
                    if filter_key and self.cache_size > 0:
                        self._filter_cache[filter_key] = signal_tensor
                        # Trim cache if needed
                        if len(self._filter_cache) > self.cache_size:
                            self._filter_cache.pop(next(iter(self._filter_cache)))
                
                metadata.processing_history.append(
                    f"{filter_type}_filter_order{filter_order}"
                )
        
        # Apply FFT if requested
        if use_fft:
            # Check cache first
            if fft_key and fft_key in self._fft_cache:
                signal_tensor = self._fft_cache[fft_key]
            else:
                # Compute FFT (vectorized via PyTorch)
                signal_fft = torch.fft.rfft(signal_tensor, dim=0)
                
                # Apply simple frequency domain processing (e.g., spectral gating for noise reduction)
                magnitude = torch.abs(signal_fft)
                # Keep components above 10% of mean magnitude
                threshold = torch.mean(magnitude) * 0.1
                mask = magnitude > threshold
                signal_fft_filtered = signal_fft * mask
                
                # Inverse FFT
                signal_tensor = torch.fft.irfft(
                    signal_fft_filtered, dim=0, n=signal_tensor.shape[0]
                )
                
                # Cache the result
                if fft_key and self.cache_size > 0:
                    self._fft_cache[fft_key] = signal_tensor
                    # Trim cache if needed
                    if len(self._fft_cache) > self.cache_size:
                        self._fft_cache.pop(next(iter(self._fft_cache)))
            
            metadata.processing_history.append("fft_spectral_gating")
        
        # Convert back to numpy (vectorized)
        processed_signal = signal_tensor.cpu().numpy()
        
        # Calculate processing time
        processing_time = time.time() - start_time
        self.metrics["classical_processing_time"] += processing_time
        
        # Create metadata
        proc_metadata = {
            "processing_mode": "classical",
            "processing_time": processing_time,
            "filter_applied": filter_type
            if filter_type in ["lowpass", "highpass", "bandpass"]
            else None,
            "fft_applied": use_fft,
            "device": str(self.torch_device)
            if hasattr(self, "torch_device")
            else "cpu",
            "original_shape": signal_data.shape,
            "signal_metadata": metadata.__dict__,
        }
        
        return processed_signal, proc_metadata

    def _process_quantum(
        self, signal_data: np.ndarray, metadata: SignalMetadata, **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Process signal using quantum signal processing techniques.
        
        Args:
            signal_data: Preprocessed input signal
            metadata: Signal metadata
            **kwargs: Additional processing parameters
        
        Returns:
            Tuple of (processed_signal, processing_metadata)
        """
        start_time = time.time()
        
        # Check if quantum device is available
        if not hasattr(self, "quantum_device") or self.quantum_device is None:
            self.logger.warning(
                "Quantum device not available. Falling back to classical processing."
            )
            return self._process_classical(signal_data, metadata, **kwargs)
        
        # Extract processing parameters
        operation = kwargs.get("quantum_operation", "qft")
        batch_size = kwargs.get("batch_size", 32)
        
        # Prepare signal for quantum processing
        # We need to reshape and potentially split into batches that can fit in our quantum circuit
        signal_batches = []
        batch_results = []
        
        # Split into batches (vectorized operations)
        n_samples = signal_data.shape[0]
        batches = [signal_data[i:min(i+batch_size, n_samples)] for i in range(0, n_samples, batch_size)]
        
        # Process each batch with appropriate quantum operation
        for batch in batches:
            if operation == "qft":
                result = self._apply_qft(batch, **kwargs)
                metadata.processing_history.append("quantum_fourier_transform")
            elif operation == "qpe":
                result = self._apply_qpe(batch, **kwargs)
                metadata.processing_history.append("quantum_phase_estimation")
            elif operation == "filter":
                result = self._apply_quantum_filter(batch, **kwargs)
                metadata.processing_history.append("quantum_filtering")
            else:
                raise ValueError(f"Unknown quantum operation: {operation}")
            
            batch_results.append(result)
        
        # Combine batch results (vectorized)
        processed_signal = np.vstack(batch_results) if len(batch_results) > 1 else batch_results[0]
        
        # Calculate processing time
        processing_time = time.time() - start_time
        self.metrics["quantum_processing_time"] += processing_time
        
        # Create metadata
        proc_metadata = {
            "processing_mode": "quantum",
            "processing_time": processing_time,
            "quantum_operation": operation,
            "qubits_used": self.n_qubits,
            "shots": self.shots,
            "device": str(self.quantum_device),
            "batch_count": len(batches),
            "original_shape": signal_data.shape,
            "signal_metadata": metadata.__dict__,
        }
        
        return processed_signal, proc_metadata

    def _apply_qft(self, signal_batch: np.ndarray, **kwargs) -> np.ndarray:
        """
        Apply Quantum Fourier Transform to signal batch using cached QNodes with JIT.
        
        Args:
            signal_batch: Batch of signal data
            **kwargs: Additional parameters
        
        Returns:
            Transformed signal data
        """
        # Check if quantum processing is available
        if not self.component_status.get("quantum_processing", False):
            # Fallback to classical FFT
            self.logger.warning("Quantum processing unavailable, falling back to classical FFT")
            return np.abs(np.fft.rfft(signal_batch, axis=0))**2
        
        # Parameter extraction
        pad_to_power_of_two = kwargs.get("pad_to_power_of_two", True)
        normalize = kwargs.get("normalize", True)
        inverse = kwargs.get("inverse", False)
        
        # Determine how many samples we can process with our qubits
        max_samples = min(2 ** (self.n_qubits - 1), signal_batch.shape[0])
        
        # Prepare data for quantum processing - fully vectorized
        if pad_to_power_of_two:
            # Calculate power of 2 ceiling
            next_power_of_two = int(2 ** np.ceil(np.log2(max_samples)))
            if max_samples < next_power_of_two:
                # Pad to power of 2 using numpy's vectorized padding
                pad_size = next_power_of_two - max_samples
                working_batch = np.pad(
                    signal_batch[:max_samples], ((0, pad_size), (0, 0)), mode="constant"
                )
            else:
                working_batch = signal_batch[:max_samples]
        else:
            working_batch = signal_batch[:max_samples]
        
        # Number of samples to process
        n_samples = working_batch.shape[0]
        
        # Need to determine how many qubits are needed
        n_qubits_needed = int(np.ceil(np.log2(n_samples)))
        
        # Preallocate result array
        result = np.zeros_like(working_batch)
        
        # Process each channel
        for channel in range(working_batch.shape[1]):
            # Extract channel data
            channel_data = working_batch[:, channel]
            
            # Normalize if requested (vectorized)
            if normalize:
                channel_max = np.max(np.abs(channel_data))
                if channel_max > 0:
                    channel_data = channel_data / channel_max
            
            try:
                # Get or create cached QNode with JIT
                qft_circuit = self._get_cached_qnode(
                    f"qft_{n_qubits_needed}_{inverse}", 
                    n_qubits_needed,
                    lambda: self._build_qft_circuit(inverse)
                )
                
                # Execute circuit and get state vector
                state_vector = qft_circuit(channel_data)
                
                # Extract amplitude information (vectorized)
                probabilities = np.abs(state_vector) ** 2
                
                # Truncate to relevant frequencies
                relevant_probs = probabilities[:n_samples]
                
                # Store in result array
                result[:, channel] = relevant_probs
            
            except Exception as e:
                self.logger.error(f"Error in quantum Fourier transform: {e}")
                # Fallback to classical FFT on error (vectorized)
                fft_result = np.abs(np.fft.rfft(channel_data))**2
                # Ensure right size
                if len(fft_result) >= n_samples:
                    result[:, channel] = fft_result[:n_samples]
                else:
                    result[:, channel] = np.pad(fft_result, (0, n_samples - len(fft_result)))
        
        # Return result truncated to original size (vectorized slice)
        return result[: signal_batch.shape[0]]

    def _apply_qpe(self, signal_batch: np.ndarray, **kwargs) -> np.ndarray:
        """
        Apply Quantum Phase Estimation on signal batch using cached QNodes with JIT.
        
        Args:
            signal_batch: Batch of signal data
            **kwargs: Additional parameters
        
        Returns:
            Phase-estimated signal data
        """
        # Check if quantum processing is available
        if not self.component_status.get("quantum_processing", False):
            # Fallback to classical processing
            self.logger.warning("Quantum processing unavailable, falling back to classical phase estimation")
            # Use Hilbert transform as fallback (vectorized)
            analytic_signal = signal.hilbert(signal_batch, axis=0)
            return np.angle(analytic_signal)
        
        # Parameter extraction
        precision = kwargs.get("precision", 3)  # Number of qubits for precision
        
        # Make sure we have enough qubits
        if self.n_qubits < precision + 1:
            self.logger.warning(
                f"Not enough qubits for requested precision. Using {self.n_qubits - 1} qubits."
            )
            precision = self.n_qubits - 1
        
        # Preallocate result array
        result = np.zeros_like(signal_batch)
        
        # Get or create cached QNode with JIT
        qpe_circuit = self._get_cached_qnode(
            f"qpe_{precision}", 
            precision + 1,  # precision qubits + 1 target qubit
            lambda: self._build_qpe_circuit(precision)
        )
        
        # Process each sample - difficult to fully vectorize due to QNode call
        for i in range(signal_batch.shape[0]):
            for channel in range(signal_batch.shape[1]):
                # Get sample value (ensure it's in [-1, 1])
                value = np.clip(signal_batch[i, channel], -1, 1)
                
                try:
                    # Map value to phase angle [0, 2π)
                    phase = (value + 1) / 2  # Map [-1, 1] to [0, 1]
                    
                    # Execute QPE circuit
                    measurements = qpe_circuit(phase)
                    
                    # Convert measurements to phase estimate (vectorized)
                    bit_values = np.array(measurements) < 0
                    bit_weights = 2.0 ** (-np.arange(1, precision + 1))
                    estimated_phase = np.sum(bit_values * bit_weights)
                    
                    # Map back to [-1, 1] range
                    result[i, channel] = estimated_phase * 2 - 1
                
                except Exception as e:
                    self.logger.error(f"Error in quantum phase estimation: {e}")
                    # Fallback to original value on error
                    result[i, channel] = value
        
        return result

    def _apply_quantum_filter(self, signal_batch: np.ndarray, **kwargs) -> np.ndarray:
        """
        Apply quantum filtering to signal batch using cached QNodes with JIT.
        
        Args:
            signal_batch: Batch of signal data
            **kwargs: Additional parameters
        
        Returns:
            Filtered signal data
        """
        # Check if quantum processing is available
        if not self.component_status.get("quantum_processing", False):
            # Fallback to classical filtering
            self.logger.warning("Quantum processing unavailable, falling back to classical filtering")
            # Simple classical filter as fallback (vectorized)
            return signal.medfilt(signal_batch, kernel_size=3)
        
        # Parameter extraction
        filter_type = kwargs.get("filter_type", "lowpass")
        threshold = kwargs.get("threshold", 0.5)
        
        # Create a copy to avoid modifying the input (vectorized)
        result = signal_batch.copy()
        
        # Calculate signal statistics for adaptive filtering (vectorized)
        mean_val = np.mean(signal_batch)
        std_val = np.std(signal_batch)
        
        # Define threshold based on statistics
        if filter_type == "adaptive":
            dynamic_threshold = mean_val + threshold * std_val
        else:
            dynamic_threshold = threshold
        
        try:
            # Get or create cached QNode with JIT
            filter_circuit = self._get_cached_qnode(
                "quantum_filter", 
                1,  # Operates on a single qubit
                lambda: self._build_quantum_filter_circuit()
            )
            
            # Process each sample - difficult to fully vectorize due to QNode call
            for i in range(signal_batch.shape[0]):
                for channel in range(signal_batch.shape[1]):
                    # Normalize sample to [-1, 1]
                    sample_value = np.clip(signal_batch[i, channel], -1, 1)
                    
                    # Apply quantum filter
                    filtered_value = filter_circuit(sample_value, dynamic_threshold)
                    
                    # Store filtered value
                    result[i, channel] = filtered_value
        
        except Exception as e:
            self.logger.error(f"Error in quantum filtering: {e}")
            # Fallback to median filter on error (vectorized)
            result = signal.medfilt(signal_batch, kernel_size=3)
        
        return result

    def _process_hybrid(
        self, signal_data: np.ndarray, metadata: SignalMetadata, **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Process signal using hybrid classical-quantum techniques.
        
        Args:
            signal_data: Preprocessed input signal
            metadata: Signal metadata
            **kwargs: Additional processing parameters
        
        Returns:
            Tuple of (processed_signal, processing_metadata)
        """
        start_time = time.time()
        
        # Extract processing parameters
        hybrid_mode = kwargs.get("hybrid_mode", "sequential")
        quantum_ratio = kwargs.get("quantum_ratio", 0.5)
        
        if hybrid_mode == "sequential":
            # First apply classical processing
            classical_result, classical_meta = self._process_classical(
                signal_data,
                metadata,
                **{k: v for k, v in kwargs.items() if not k.startswith("quantum_")},
            )
            
            # Then apply quantum processing to the classical result
            final_result, quantum_meta = self._process_quantum(
                classical_result,
                metadata,
                **{
                    k: v
                    for k, v in kwargs.items()
                    if k.startswith("quantum_") or k in ["batch_size"]
                },
            )
            
            # Update metadata
            processing_time = time.time() - start_time
            
            proc_metadata = {
                "processing_mode": "hybrid_sequential",
                "processing_time": processing_time,
                "classical_time": classical_meta["processing_time"],
                "quantum_time": quantum_meta["processing_time"],
                "classical_steps": classical_meta["signal_metadata"][
                    "processing_history"
                ],
                "quantum_steps": quantum_meta["signal_metadata"]["processing_history"],
                "device": f"classical:{str(self.torch_device) if hasattr(self, 'torch_device') else 'cpu'}, quantum:{str(self.quantum_device)}",
                "original_shape": signal_data.shape,
                "signal_metadata": metadata.__dict__,
            }
        
        elif hybrid_mode == "parallel":
            # Process parts of the signal with classical and parts with quantum in parallel
            split_idx = int(signal_data.shape[0] * quantum_ratio)
            
            # Split the signal (vectorized)
            classical_part = signal_data[split_idx:]
            quantum_part = signal_data[:split_idx]
            
            # Process in parallel using ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=2) as executor:
                classical_future = executor.submit(
                    self._process_classical,
                    classical_part,
                    metadata,
                    **{k: v for k, v in kwargs.items() if not k.startswith("quantum_")},
                )
                
                quantum_future = executor.submit(
                    self._process_quantum,
                    quantum_part,
                    metadata,
                    **{
                        k: v
                        for k, v in kwargs.items()
                        if k.startswith("quantum_") or k in ["batch_size"]
                    },
                )
                
                # Get results
                classical_result, classical_meta = classical_future.result()
                quantum_result, quantum_meta = quantum_future.result()
            
            # Combine results (vectorized)
            final_result = np.vstack([quantum_result, classical_result])
            
            # Update metadata
            processing_time = time.time() - start_time
            
            proc_metadata = {
                "processing_mode": "hybrid_parallel",
                "processing_time": processing_time,
                "classical_time": classical_meta["processing_time"],
                "quantum_time": quantum_meta["processing_time"],
                "classical_ratio": 1 - quantum_ratio,
                "quantum_ratio": quantum_ratio,
                "classical_steps": classical_meta["signal_metadata"][
                    "processing_history"
                ],
                "quantum_steps": quantum_meta["signal_metadata"]["processing_history"],
                "device": f"classical:{str(self.torch_device) if hasattr(self, 'torch_device') else 'cpu'}, quantum:{str(self.quantum_device)}",
                "original_shape": signal_data.shape,
                "signal_metadata": metadata.__dict__,
            }
        
        elif hybrid_mode == "adaptive":
            # Use frequency domain decomposition
            # Apply FFT to decompose signal (vectorized)
            signal_fft = np.fft.rfft(signal_data, axis=0)
            
            # Low frequencies are more suitable for quantum processing
            freqs = np.fft.rfftfreq(signal_data.shape[0])
            
            # Split based on frequency (vectorized)
            low_freq_mask = freqs < quantum_ratio
            high_freq_mask = ~low_freq_mask
            
            # Process low frequencies with quantum (vectorized)
            low_freq_signal = np.fft.irfft(
                signal_fft * low_freq_mask[:, np.newaxis],
                n=signal_data.shape[0],
                axis=0,
            )
            quantum_result, quantum_meta = self._process_quantum(
                low_freq_signal,
                metadata,
                **{
                    k: v
                    for k, v in kwargs.items()
                    if k.startswith("quantum_") or k in ["batch_size"]
                },
            )
            
            # Process high frequencies with classical (vectorized)
            high_freq_signal = np.fft.irfft(
                signal_fft * high_freq_mask[:, np.newaxis],
                n=signal_data.shape[0],
                axis=0,
            )
            classical_result, classical_meta = self._process_classical(
                high_freq_signal,
                metadata,
                **{k: v for k, v in kwargs.items() if not k.startswith("quantum_")},
            )
            
            # Combine results (vectorized addition)
            final_result = quantum_result + classical_result
            
            # Update metadata
            processing_time = time.time() - start_time
            
            proc_metadata = {
                "processing_mode": "hybrid_adaptive",
                "processing_time": processing_time,
                "classical_time": classical_meta["processing_time"],
                "quantum_time": quantum_meta["processing_time"],
                "frequency_split": quantum_ratio,
                "classical_steps": classical_meta["signal_metadata"][
                    "processing_history"
                ],
                "quantum_steps": quantum_meta["signal_metadata"]["processing_history"],
                "device": f"classical:{str(self.torch_device) if hasattr(self, 'torch_device') else 'cpu'}, quantum:{str(self.quantum_device)}",
                "original_shape": signal_data.shape,
                "signal_metadata": metadata.__dict__,
            }
        
        else:
            raise ValueError(f"Unknown hybrid mode: {hybrid_mode}")
        
        return final_result, proc_metadata

    def _should_use_quantum(self, signal_data: np.ndarray) -> bool:
        """
        Determine if quantum processing should be used based on signal characteristics
        and past performance metrics.
        
        Args:
            signal_data: Preprocessed signal data
        
        Returns:
            Boolean indicating whether to use quantum processing
        """
        # Don't use quantum if not available
        if not self.component_status.get("quantum_processing", False):
            return False
        
        # Don't use quantum for very small signals (overhead not worth it)
        if signal_data.size < 16:
            return False
        
        # Don't use quantum for very large signals (would exceed qubit capacity)
        if signal_data.size > 2**self.n_qubits:
            return False
        
        # Check past performance metrics
        if (self.metrics["quantum_processing_time"] > 0 and 
            self.metrics["classical_processing_time"] > 0):
            avg_quantum_time = self.metrics["quantum_processing_time"] / max(1, self.metrics["total_processed_signals"])
            avg_classical_time = self.metrics["classical_processing_time"] / max(1, self.metrics["total_processed_signals"])
            
            # Use quantum if it's been faster (allowing 20% overhead for better results)
            return avg_quantum_time <= avg_classical_time * 1.2
        
        # Default to using quantum for medium-sized signals
        return 16 <= signal_data.size <= 256

    # ===== MACHINE LEARNING METHODS =====

    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates ML input features based on market data.
        
        Args:
            df: DataFrame with market data
        
        Returns:
            DataFrame with generated features
        """
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            self.logger.warning("Invalid DataFrame for feature generation.")
            return pd.DataFrame()
        
        missing_cols = [col for col in self.required_columns if col not in df.columns]
        if missing_cols:
            self.logger.warning(f"Missing required columns for ML features: {missing_cols}")
            return pd.DataFrame()
        
        try:
            # Generate features using vectorized pandas operations
            features = pd.DataFrame(index=df.index)
            
            # Price-based features (vectorized)
            for col in ["close", "high", "low"]:
                for period in [1, 3, 5, 10, 15]:
                    if len(df) > period:
                        features[f"{col}_change_{period}"] = df[col].pct_change(period)
            
            # Volume features (vectorized)
            features["volume_change"] = df["volume"].pct_change()
            features["volume_ma_ratio"] = (
                df["volume"] / df["volume"].rolling(10, min_periods=1).mean()
            )
            
            # Volatility (vectorized)
            features["volatility_14d"] = (
                df["close"].pct_change().rolling(14, min_periods=3).std()
            )
            
            # RSI (vectorized)
            delta = df["close"].diff()
            gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
            avg_gain = gain.rolling(14).mean()
            avg_loss = loss.rolling(14).mean()
            # Avoid division by zero
            rs = avg_gain / avg_loss.replace(0, 1e-8)
            features["rsi_14"] = 100 - (100 / (1 + rs))
            
            # MACD (vectorized)
            features["macd"] = (
                df["close"].ewm(span=12).mean() - df["close"].ewm(span=26).mean()
            )
            
            # Handle NaN values (vectorized)
            features = features.replace([np.inf, -np.inf], np.nan).fillna(0)
            
            return features
        
        except Exception as e:
            self.logger.error(f"Feature generation error: {str(e)}")
            return pd.DataFrame()

    def predict_ml(self, df: pd.DataFrame, model_type: str = "auto") -> pd.Series:
        """
        Generates ML trading signals using available models with caching.
        
        Args:
            df: DataFrame with market data
            model_type: Model type to use (auto, catboost, h2o, ensemble)
        
        Returns:
            Series with prediction values
        """
        start_time = time.time()
        
        # Generate features
        features = self.generate_features(df)
        if features.empty:
            self.logger.warning("Failed to generate features for ML prediction")
            return pd.Series(0.5, index=df.index)
        
        # Generate cache key based on features data and model type
        cache_key = self._generate_prediction_cache_key(features, model_type)
        if cache_key in self._prediction_cache:
            self.metrics["cache_hits"] += 1
            return self._prediction_cache[cache_key]
        
        try:
            predictions = {}
            
            # CatBoost prediction
            if self.component_status.get("catboost", False) and (
                model_type in ["auto", "catboost", "ensemble"]
            ):
                try:
                    catboost_model = self.ml_models.get("catboost")
                    if catboost_model is not None:
                        # Use model's vectorized predict_proba method
                        catboost_pred = catboost_model.predict_proba(features)[:, 1]
                        predictions["catboost"] = pd.Series(catboost_pred, index=features.index)
                except Exception as e:
                    self.logger.warning(f"CatBoost prediction failed: {e}")
            
            # H2O prediction
            if self.component_status.get("h2o", False) and (
                model_type in ["auto", "h2o", "ensemble"]
            ):
                try:
                    h2o_model = self.ml_models.get("h2o")
                    if h2o_model is not None:
                        h2o_df = h2o.H2OFrame(features)
                        # H2O handles batching internally
                        h2o_pred = h2o_model.predict(h2o_df).as_data_frame().values.flatten()
                        predictions["h2o"] = pd.Series(h2o_pred, index=features.index)
                except Exception as e:
                    self.logger.warning(f"H2O prediction failed: {e}")
            
            # Fuzzy prediction
            if self.component_status.get("sanfis", False) and (
                model_type in ["auto", "fuzzy", "ensemble"]
            ):
                try:
                    fuzzy_signals = self.evaluate_fuzzy(df)
                    predictions["fuzzy"] = fuzzy_signals
                except Exception as e:
                    self.logger.warning(f"Fuzzy prediction failed: {e}")
            
            # Combine predictions
            if not predictions:
                self.logger.warning("No ML models available for prediction")
                return pd.Series(0.5, index=df.index)
            
            if model_type == "ensemble" and len(predictions) > 1:
                # Weighted ensemble based on configured weights or equal weights (vectorized)
                weights = self.config.get("ensemble_weights", {})
                
                # Initialize with zeros
                combined_prediction = pd.Series(0.0, index=features.index)
                total_weight = 0.0
                
                # Add weighted predictions
                for model_name, prediction in predictions.items():
                    weight = weights.get(model_name, 1.0 / len(predictions))
                    combined_prediction += prediction * weight
                    total_weight += weight
                
                # Normalize if weights don't sum to 1
                if total_weight != 1.0 and total_weight > 0:
                    combined_prediction /= total_weight
                
                final_prediction = combined_prediction
            else:
                # Use a single model or the first available one
                if model_type in predictions:
                    final_prediction = predictions[model_type]
                else:
                    final_prediction = next(iter(predictions.values()))
            
            # Ensure values are in [0.01, 0.99] range (vectorized)
            final_prediction = final_prediction.clip(0.01, 0.99)
            
            # Cache the result
            if self.cache_size > 0:
                self._prediction_cache[cache_key] = final_prediction
                # Trim cache if needed
                if len(self._prediction_cache) > self.cache_size:
                    self._prediction_cache.pop(next(iter(self._prediction_cache)))
            
            # Update metrics
            processing_time = time.time() - start_time
            self.metrics["ml_processing_time"] += processing_time
            
            return final_prediction
        
        except Exception as e:
            self.logger.error(f"ML prediction error: {str(e)}")
            # Return neutral signal on error
            return pd.Series(0.5, index=df.index)

    def evaluate_fuzzy(self, df: pd.DataFrame) -> pd.Series:
        """
        Evaluate data using fuzzy logic with SANFIS/ANFIS.
        
        Args:
            df: DataFrame with market data
        
        Returns:
            Series with fuzzy evaluation results
        """
        if not self.component_status.get("sanfis", False) and not self.component_status.get("anfis", False):
            self.logger.warning("No fuzzy models available")
            return pd.Series(0.5, index=df.index)
        
        try:
            # Generate needed features if they don't exist
            if "rsi_14" not in df.columns:
                features = self.generate_features(df)
                df = pd.concat([df, features], axis=1)
            
            # Prepare inputs
            result = pd.Series(index=df.index, dtype=float)
            
            # Use SANFIS if available
            if self.component_status.get("sanfis", False) and hasattr(self, "sanfis"):
                try:
                    # Vectorize SANFIS evaluation if possible
                    if hasattr(self.sanfis, "evaluate_batch"):
                        # Prepare inputs (vectorized)
                        rsi_values = df["rsi_14"].values / 100.0 if "rsi_14" in df.columns else np.full(len(df), 0.5)
                        vol_values = df["volatility_14d"].values if "volatility_14d" in df.columns else np.full(len(df), 0.01)
                        
                        # Call vectorized evaluation
                        fuzzy_outputs = self.sanfis.evaluate_batch(rsi_values, vol_values)
                        result[:] = fuzzy_outputs
                    else:
                        # Fall back to row-by-row processing
                        for idx in df.index:
                            # Normalize RSI to 0-1 range and use volatility directly
                            rsi = df.loc[idx, "rsi_14"] / 100.0 if "rsi_14" in df.columns else 0.5
                            vol = df.loc[idx, "volatility_14d"] if "volatility_14d" in df.columns else 0.01
                            
                            # Call SANFIS evaluate method
                            if hasattr(self.sanfis, "evaluate"):
                                fuzzy_output = self.sanfis.evaluate(rsi, vol)
                                result[idx] = fuzzy_output
                except Exception as e:
                    self.logger.error(f"SANFIS evaluation failed: {e}")
                    # Try ANFIS as fallback
                    if self.component_status.get("anfis", False) and hasattr(self, "anfis"):
                        try:
                            # Use ANFIS filter
                            anfis_filtered = self.anfis.filter_noise(df)
                            result = anfis_filtered["adaptive_signal"]
                        except Exception as e2:
                            self.logger.error(f"ANFIS evaluation failed: {e2}")
                            result.loc[:] = 0.5  # Neutral value on error
                    else:
                        result.loc[:] = 0.5  # Neutral value on error
            
            # If SANFIS not available but ANFIS is
            elif self.component_status.get("anfis", False) and hasattr(self, "anfis"):
                try:
                    # Use ANFIS filter
                    anfis_filtered = self.anfis.filter_noise(df)
                    result = anfis_filtered["adaptive_signal"]
                except Exception as e:
                    self.logger.error(f"ANFIS evaluation failed: {e}")
                    result.loc[:] = 0.5  # Neutral value on error
            
            # Fill any missing values with neutral (vectorized)
            result = result.fillna(0.5)
            
            return result
        
        except Exception as e:
            self.logger.error(f"Fuzzy evaluation error: {str(e)}")
            return pd.Series(0.5, index=df.index)

    # ===== MARKET ANALYSIS METHODS =====

    def detect_market_regime(self, data: pd.DataFrame) -> str:
        """
        Detect current market regime from data.
        
        Args:
            data: DataFrame with market data
        
        Returns:
            String indicating detected regime
        """
        if data is None or data.empty:
            return "unknown"
        
        try:
            # Extract key metrics with memory-efficient operations
            if "close" not in data.columns:
                return "unknown"
            
            # Vectorized operations
            close_values = data["close"].values
            
            if len(close_values) < 20:
                return "unknown"
            
            # Calculate returns using numpy for memory efficiency (vectorized)
            returns = np.diff(close_values) / close_values[:-1]
            
            # Calculate key metrics (vectorized)
            volatility = np.std(returns[-20:]) if len(returns) >= 20 else 0
            short_mom = (
                close_values[-1] / close_values[-5] - 1 if len(close_values) >= 5 else 0
            )
            long_mom = (
                close_values[-1] / close_values[-20] - 1
                if len(close_values) >= 20
                else 0
            )
            
            # Check for quantum regime info (more accurate if available)
            if "regime" in data.columns and not data["regime"].isna().all():
                # Use most recent valid regime value
                for i in range(-1, -min(10, len(data)), -1):
                    regime_value = data["regime"].iloc[i]
                    if not pd.isna(regime_value) and regime_value != "unknown":
                        return regime_value
            
            # Determine regime from metrics
            if volatility > 0.03:
                regime = "volatile"
            elif long_mom > 0.05 and short_mom > 0:
                regime = "bull"
            elif long_mom < -0.05 and short_mom < 0:
                regime = "bear"
            else:
                regime = "normal"
            
            # Update current regime
            self.current_regime = regime
            
            return regime
        
        except Exception as e:
            self.logger.error(f"Error detecting market regime: {e}")
            return "unknown"

    def detect_critical_state(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect market criticality and update critical state detector.
        
        Args:
            data: DataFrame with market data
        
        Returns:
            Dictionary with critical state assessment
        """
        if data is None or data.empty or len(data) < 50:
            return {"warning_level": "normal", "critical_point_estimate": 0.0}
        
        try:
            # Calculate metrics for critical state detection
            if "close" in data.columns:
                # Compute recent returns (vectorized)
                returns = data["close"].pct_change().dropna()
                
                if len(returns) < 30:
                    return {"warning_level": "normal", "critical_point_estimate": 0.0}
                
                # 1. Check for power laws in return distribution (vectorized)
                sorted_returns = np.sort(np.abs(returns.values))[
                    ::-1
                ]  # Descending order
                if len(sorted_returns) > 20:
                    # Fit power law using log-log regression (vectorized)
                    log_x = np.log(np.arange(1, len(sorted_returns) + 1))
                    log_y = np.log(sorted_returns)
                    
                    # Simple regression (vectorized)
                    valid_indices = np.isfinite(log_x) & np.isfinite(log_y)
                    if np.sum(valid_indices) > 10:
                        slope = np.polyfit(
                            log_x[valid_indices], log_y[valid_indices], 1
                        )[0]
                        self.critical_state_detector["power_law_exponent"] = -slope
                
                # 2. Check synchronization between components
                # Check if signals are highly correlated (vectorized)
                signal_columns = [col for col in data.columns if "signal" in col]
                if len(signal_columns) >= 2:
                    # Calculate pairwise correlations (vectorized where possible)
                    correlations = []
                    for i in range(len(signal_columns)):
                        for j in range(i + 1, len(signal_columns)):
                            if (
                                not data[signal_columns[i]].isna().all()
                                and not data[signal_columns[j]].isna().all()
                            ):
                                corr = np.corrcoef(
                                    data[signal_columns[i]].fillna(0),
                                    data[signal_columns[j]].fillna(0),
                                )[0, 1]
                                if not np.isnan(corr):
                                    correlations.append(abs(corr))
                    
                    if correlations:
                        mean_correlation = np.mean(correlations)
                        self.critical_state_detector["critical_indicators"][
                            "synchronization"
                        ] = mean_correlation
                
                # 3. Detect long-range correlations using Hurst exponent
                if len(returns) >= 100:
                    try:
                        # Calculate Hurst exponent
                        hurst = self.calculate_hurst_exponent(returns.values)
                        self.critical_state_detector["critical_indicators"][
                            "long_range_correlations"
                        ] = hurst
                    except Exception as e:
                        self.logger.error(f"Error calculating Hurst exponent: {e}")
                
                # Update warning level based on critical indicators
                self._update_critical_warning_level()
                
                # Return current critical state
                return {
                    "warning_level": self.critical_state_detector["warning_level"],
                    "critical_point_estimate": self.critical_state_detector["critical_point_estimate"],
                    "power_law_exponent": self.critical_state_detector["power_law_exponent"],
                    "synchronization": self.critical_state_detector["critical_indicators"]["synchronization"],
                    "long_range_correlations": self.critical_state_detector["critical_indicators"]["long_range_correlations"]
                }
        
        except Exception as e:
            self.logger.error(f"Error in critical state detection: {e}")
        
        return {"warning_level": "normal", "critical_point_estimate": 0.0}

    @njit
    def calculate_hurst_exponent(self, time_series: np.ndarray, max_lag: int = 20) -> float:
        """
        Calculate Hurst exponent for time series with Numba JIT optimization.
        
        Args:
            time_series: Input time series data
            max_lag: Maximum lag to consider
        
        Returns:
            Hurst exponent (0.5=random, >0.5=trending, <0.5=mean-reverting)
        """
        # Simplified Hurst calculation
        n = len(time_series)
        lags = range(2, min(max_lag, n // 4))
        tau = np.zeros(len(lags))
        
        # Calculate tau for each lag (optimized by numba)
        for i, lag in enumerate(lags):
            # Calculate differences
            diffs = np.zeros(n - lag)
            for j in range(n - lag):
                diffs[j] = time_series[j + lag] - time_series[j]
            
            # Standard deviation
            tau[i] = np.std(diffs)
        
        # Filter valid values
        valid_lags = []
        valid_taus = []
        for i, t in enumerate(tau):
            if np.isfinite(t) and t > 0:
                valid_lags.append(np.log(lags[i]))
                valid_taus.append(np.log(t))
        
        if len(valid_lags) < 4:
            return 0.5  # Default value
        
        # Linear regression
        n_valid = len(valid_lags)
        sum_x = sum(valid_lags)
        sum_y = sum(valid_taus)
        sum_xx = sum(x*x for x in valid_lags)
        sum_xy = sum(x*y for x, y in zip(valid_lags, valid_taus))
        
        # Calculate slope
        slope = (n_valid * sum_xy - sum_x * sum_y) / (n_valid * sum_xx - sum_x * sum_x)
        
        return slope / 2.0  # Hurst = slope/2

    def _update_critical_warning_level(self):
        """Update warning level based on critical state indicators"""
        # Get indicators
        power_law = self.critical_state_detector["power_law_exponent"]
        sync = self.critical_state_detector["critical_indicators"]["synchronization"]
        lrc = self.critical_state_detector["critical_indicators"][
            "long_range_correlations"
        ]
        
        # Calculate criticality score (vectorized where possible)
        criticality_score = 0
        
        # Power law with exponent ~1 indicates criticality
        if 0.9 < power_law < 1.7:
            criticality_score += (1.0 - abs(power_law - 1.3)) * 3
        
        # High synchronization indicates potential system-wide event
        if sync > 0.7:
            criticality_score += (sync - 0.7) * 10
        
        # Hurst exponent ~0.5 is random, ~1.0 is trending, ~0 is mean-reverting
        # Values near 1.0 can indicate critical transitions
        if 0.7 < lrc < 1.0:
            criticality_score += (lrc - 0.7) * 5
        
        # Set warning level based on score
        if criticality_score > 2.0:
            self.critical_state_detector["warning_level"] = "severe"
            self.logger.warning(
                f"CRITICAL STATE DETECTED - Criticality score: {criticality_score:.2f}"
            )
        elif criticality_score > 1.0:
            self.critical_state_detector["warning_level"] = "elevated"
            self.logger.info(
                f"Elevated critical state - Criticality score: {criticality_score:.2f}"
            )
        else:
            self.critical_state_detector["warning_level"] = "normal"
        
        # Store critical point estimate
        self.critical_state_detector["critical_point_estimate"] = criticality_score

    def calculate_system_entropy(self, signals: Dict[str, Union[float, np.ndarray, pd.Series]]) -> float:
        """
        Calculate information entropy of the signal system.
        
        Args:
            signals: Dictionary of signal values
        
        Returns:
            Entropy value (0-1 range)
        """
        try:
            # Get signal values, ensuring they are proper probabilities (vectorized)
            values = []
            for k, v in signals.items():
                if isinstance(v, (pd.Series, np.ndarray)):
                    if len(v) > 0:
                        if isinstance(v, pd.Series):
                            values.append(np.clip(v.iloc[-1], 0.01, 0.99))
                        else:
                            values.append(np.clip(v[-1], 0.01, 0.99))
                else:
                    values.append(np.clip(v, 0.01, 0.99))
            
            if not values:
                return 0.5
            
            # Calculate Shannon entropy (vectorized)
            values = np.array(values)
            entropy = -np.mean(values * np.log2(values) + (1 - values) * np.log2(1 - values))
            
            # Normalize to 0-1 range
            normalized_entropy = entropy
            
            # Update system entropy
            self.system_entropy = normalized_entropy
            
            return normalized_entropy
        except Exception as e:
            self.logger.warning(f"Error calculating system entropy: {e}")
            return 0.5

    def calculate_antifragility(self, data: pd.DataFrame) -> float:
        """
        Calculate antifragility score based on market data.
        
        Args:
            data: DataFrame with market data
        
        Returns:
            Antifragility score (0-1 range)
        """
        if data is None or data.empty:
            return 0.5
        
        try:
            # Check if we already have antifragility in the data
            if "antifragility" in data.columns and not data["antifragility"].isna().all():
                return data["antifragility"].iloc[-1]
            
            # Calculate components of antifragility
            
            # 1. Volatility exposure (higher is better for antifragility)
            if "close" in data.columns and len(data) > 20:
                # Vectorized calculation
                returns = data["close"].pct_change().dropna()
                volatility = returns.std() * np.sqrt(252)  # Annualized
                vol_exposure = min(1.0, volatility * 5)  # Scale for range
            else:
                vol_exposure = 0.5
            
            # 2. Convexity (non-linear positive response to volatility)
            if "close" in data.columns and len(data) > 40:
                # Split in half and compare volatility to returns (vectorized)
                mid_point = len(data) // 2
                first_half_vol = data["close"].iloc[:mid_point].pct_change().std()
                second_half_vol = data["close"].iloc[mid_point:].pct_change().std()
                
                first_half_return = (
                    data["close"].iloc[mid_point - 1] / data["close"].iloc[0] - 1
                )
                second_half_return = (
                    data["close"].iloc[-1] / data["close"].iloc[mid_point] - 1
                )
                
                # Positive convexity: returns increase more than proportionally to volatility
                if second_half_vol > first_half_vol and first_half_vol > 0:
                    vol_ratio = second_half_vol / first_half_vol
                    if first_half_return > 0 and second_half_return > 0:
                        return_ratio = second_half_return / first_half_return
                        convexity = min(1.0, return_ratio / vol_ratio)
                    else:
                        convexity = 0.5
                else:
                    convexity = 0.5
            else:
                convexity = 0.5
            
            # 3. Redundancy (multiple uncorrelated signal sources)
            signal_columns = [col for col in data.columns if "signal" in col]
            if len(signal_columns) >= 2:
                # Vectorized correlation calculation
                signals_df = data[signal_columns].fillna(0)
                corr_matrix = signals_df.corr().abs().values
                
                # Extract upper triangle excluding diagonal
                correlations = []
                for i in range(len(signal_columns)):
                    for j in range(i + 1, len(signal_columns)):
                        correlations.append(corr_matrix[i, j])
                
                if correlations:
                    mean_correlation = np.mean(correlations)
                    redundancy = 1.0 - mean_correlation
                else:
                    redundancy = 0.5
            else:
                redundancy = 0.5
            
            # 4. Adaptation capability (based on system entropy)
            adaptation = self.system_entropy
            
            # 5. Calculate final antifragility score with weights (vectorized)
            weights = {
                "vol_exposure": 0.3,
                "convexity": 0.3,
                "redundancy": 0.2,
                "adaptation": 0.2,
            }
            
            antifragility = (
                weights["vol_exposure"] * vol_exposure +
                weights["convexity"] * convexity +
                weights["redundancy"] * redundancy +
                weights["adaptation"] * adaptation
            )
            
            return np.clip(antifragility, 0.0, 1.0)
        
        except Exception as e:
            self.logger.error(f"Error calculating antifragility: {e}")
            return 0.5

    def via_negativa_filter(self, signals: Dict[str, float]) -> List[str]:
        """
        Apply via negativa filtering to identify risk flags.
        
        Args:
            signals: Dictionary of signal values
        
        Returns:
            List of risk flags
        """
        risk_flags = []
        
        try:
            # 1. Check for signal disagreement (vectorized)
            signal_values = np.array(list(signals.values()))
            if len(signal_values) >= 2:
                bullish_signals = np.sum(signal_values > 0.6)
                bearish_signals = np.sum(signal_values < 0.4)
                
                if bullish_signals > 0 and bearish_signals > 0:
                    # Mixed signals indicate uncertainty
                    if bullish_signals / len(signal_values) > 0.3 and bearish_signals / len(signal_values) > 0.3:
                        risk_flags.append("contradictory_signals")
            
            # 2. Check for extreme values (overconfidence) (vectorized)
            extreme_count = np.sum((signal_values > 0.95) | (signal_values < 0.05))
            if extreme_count / len(signal_values) > 0.5:
                risk_flags.append("signal_overconfidence")
            
            # 3. Check for elevated entropy
            if self.system_entropy > 0.8:
                risk_flags.append("high_entropy")
            
            # 4. Check for critical state warning
            if self.critical_state_detector["warning_level"] != "normal":
                risk_flags.append(f"{self.critical_state_detector['warning_level']}_criticality")
            
            # 5. Check for specific regime warning
            if self.current_regime == "volatile":
                risk_flags.append("volatile_regime")
            
            return risk_flags
        
        except Exception as e:
            self.logger.error(f"Error in via negativa filtering: {e}")
            return risk_flags

    # ===== PARALLEL PROCESSING METHODS =====

    def parallel_process(
        self, data: Union[pd.DataFrame, np.ndarray], 
        method: Union[str, Callable], 
        **params
    ) -> Union[pd.DataFrame, np.ndarray]:
        """
        Process data in parallel using multiple workers.
        
        Args:
            data: Input data to process
            method: Method name or callable to apply
            **params: Parameters to pass to the method
        
        Returns:
            Processed data
        """
        if not self.use_parallel:
            # Call the method directly if parallel processing is disabled
            if isinstance(method, str):
                if hasattr(self, method):
                    return getattr(self, method)(data, **params)
                else:
                    raise ValueError(f"Method {method} not found")
            elif callable(method):
                return method(data, **params)
            else:
                raise ValueError("Method must be a string or callable")
        
        try:
            # Determine chunk size
            if isinstance(data, pd.DataFrame):
                total_size = len(data)
            elif isinstance(data, np.ndarray):
                total_size = data.shape[0]
            else:
                raise ValueError("Data must be a DataFrame or numpy array")
            
            chunk_size = max(1, total_size // self.num_parallel_workers)
            
            # Split data into chunks (vectorized)
            chunks = []
            for i in range(0, total_size, chunk_size):
                if isinstance(data, pd.DataFrame):
                    end = min(i + chunk_size, total_size)
                    chunks.append(data.iloc[i:end])
                else:  # numpy array
                    end = min(i + chunk_size, total_size)
                    chunks.append(data[i:end])
            
            # Process chunks in parallel
            if isinstance(data, pd.DataFrame):
                # Use ProcessPoolExecutor for CPU-bound tasks
                with ProcessPoolExecutor(max_workers=self.num_parallel_workers) as executor:
                    if isinstance(method, str):
                        # Create a wrapper function that calls the method by name
                        def process_chunk(chunk):
                            return getattr(self, method)(chunk, **params)
                        
                        results = list(executor.map(process_chunk, chunks))
                    else:
                        # Pass the callable directly
                        results = list(executor.map(lambda chunk: method(chunk, **params), chunks))
                
                # Combine results (vectorized)
                if isinstance(results[0], pd.DataFrame):
                    return pd.concat(results)
                elif isinstance(results[0], pd.Series):
                    return pd.concat(results)
                else:
                    # If the result is a numpy array, combine them
                    return np.concatenate(results)
            else:
                # For numpy arrays, use ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=self.num_parallel_workers) as executor:
                    if isinstance(method, str):
                        def process_chunk(chunk):
                            return getattr(self, method)(chunk, **params)
                        
                        results = list(executor.map(process_chunk, chunks))
                    else:
                        results = list(executor.map(lambda chunk: method(chunk, **params), chunks))
                
                # Combine results (vectorized)
                return np.concatenate(results)
        
        except Exception as e:
            self.logger.error(f"Error in parallel processing: {e}")
            # Fallback to non-parallel processing
            if isinstance(method, str):
                return getattr(self, method)(data, **params)
            else:
                return method(data, **params)

    def _smooth_chunk_boundaries(self, processed_chunks: List[np.ndarray]) -> np.ndarray:
        """
        Smooth boundaries between processed chunks to prevent artifacts.
        
        Args:
            processed_chunks: List of processed chunks
        
        Returns:
            Smoothed combined array
        """
        if len(processed_chunks) <= 1:
            return processed_chunks[0] if processed_chunks else np.array([])
        
        # Combine chunks into a single array (vectorized)
        combined = np.concatenate(processed_chunks)
        
        # Apply smoothing at the boundary points
        overlap = 10  # Number of points to overlap for smoothing
        smooth_combined = combined.copy()
        
        # Calculate chunk boundaries (vectorized)
        chunk_sizes = np.array([chunk.shape[0] for chunk in processed_chunks])
        boundaries = np.cumsum(chunk_sizes)[:-1]
        
        # Apply smoothing at each boundary
        for boundary in boundaries:
            # Define the overlap region
            start = max(0, boundary - overlap // 2)
            end = min(combined.shape[0], boundary + overlap // 2)
            
            # Apply linear weighting for smooth transition (vectorized)
            weights = np.linspace(0, 1, end - start)
            
            # Create weighted average (vectorized)
            for i in range(start, end):
                weight = weights[i - start]
                smooth_combined[i] = (1 - weight) * combined[i - 1] + weight * combined[i]
        
        return smooth_combined

    # ===== UTILITY METHODS =====

    def batch_process(
        self,
        signals: List[np.ndarray],
        mode: Optional[ProcessingMode] = None,
        metadata: Optional[List[SignalMetadata]] = None,
        **kwargs
    ) -> List[Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Process multiple signals in batch mode.
        
        Args:
            signals: List of signal data arrays
            mode: Processing mode
            metadata: List of metadata for each signal
            **kwargs: Additional processing parameters
        
        Returns:
            List of (processed_signal, processing_metadata) tuples
        """
        results = []
        
        # Create metadata list if not provided
        if metadata is None:
            # Vectorized metadata creation
            metadata = [
                SignalMetadata(
                    sample_rate=kwargs.get("sample_rate", 1.0),
                    dimension=signal.shape[-1] if len(signal.shape) > 1 else 1,
                    source=kwargs.get("source", "unknown"),
                )
                for signal in signals
            ]
        elif len(metadata) != len(signals):
            raise ValueError("Metadata list length must match signals list length")
        
        # Check if we should use parallel processing
        if self.use_parallel and len(signals) > 4:
            # Process signals in parallel using ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=self.num_parallel_workers) as executor:
                futures = []
                for i, signal in enumerate(signals):
                    futures.append(
                        executor.submit(
                            self.process_signal, signal, mode, metadata[i], **kwargs
                        )
                    )
                
                # Collect results
                for future in futures:
                    results.append(future.result())
        else:
            # Process each signal sequentially
            for i, signal in enumerate(signals):
                result = self.process_signal(signal, mode, metadata[i], **kwargs)
                results.append(result)
        
        return results

    def _generate_cache_key(self, signal_data: np.ndarray, mode: ProcessingMode, params: Dict = None) -> str:
        """
        Generate a cache key for the given signal data, mode and parameters
        
        Args:
            signal_data: Signal data array
            mode: Processing mode
            params: Optional processing parameters
        
        Returns:
            Cache key string
        """
        # Create a hashable representation of the signal data
        data_hash = hashlib.md5(signal_data.tobytes()).hexdigest()
        
        # Add mode to the key
        key = f"{data_hash}_{mode.name}"
        
        # Add important parameters to the key if provided
        if params:
            param_str = "_".join(f"{k}:{v}" for k, v in sorted(params.items()) 
                                if k in ["filter_type", "quantum_operation", "batch_size"])
            key += f"_{param_str}"
        
        return key

    def _generate_prediction_cache_key(self, features: pd.DataFrame, model_type: str) -> str:
        """
        Generate a cache key for ML prediction results
        
        Args:
            features: Feature DataFrame
            model_type: Type of model used
        
        Returns:
            Cache key string
        """
        # Create a hashable representation of features
        feature_hash = hashlib.md5(pd.util.hash_pandas_object(features).values.tobytes()).hexdigest()
        
        # Combine with model type
        return f"pred_{feature_hash}_{model_type}"

    def _update_cache(self, key: str, value: Tuple[np.ndarray, Dict[str, Any]]) -> None:
        """Update the results cache with the given key and value"""
        with self._lock:
            # Add new item
            self._results_cache[key] = value
            
            # Trim cache if it exceeds the maximum size
            if len(self._results_cache) > self.cache_size:
                # Remove oldest items (simple FIFO strategy)
                oldest_keys = list(self._results_cache.keys())[
                    : (len(self._results_cache) - self.cache_size)
                ]
                for old_key in oldest_keys:
                    del self._results_cache[old_key]

    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        with self._lock:
            return self.metrics.copy()

    def reset_metrics(self) -> None:
        """Reset performance metrics"""
        with self._lock:
            for key in self.metrics:
                if isinstance(self.metrics[key], (int, float)):
                    self.metrics[key] = type(self.metrics[key])()
    
    def clear_cache(self): # Ensure this method exists and is comprehensive
        """Clears all internal caches."""
        with self._lock:
            caches_to_clear = [
                '_results_cache', '_qnode_cache', '_filter_cache',
                '_fft_cache', '_prediction_cache', '_internal_cache',
                '_tail_risk_cache'
            ]
            cleared_count = 0
            for cache_name in caches_to_clear:
                if hasattr(self, cache_name):
                    cache = getattr(self, cache_name)
                    if hasattr(cache, 'clear') and callable(cache.clear):
                        try:
                             cache.clear()
                             cleared_count += 1
                        except Exception as e:
                             self.logger.error(f"Error clearing USP cache '{cache_name}': {e}")
                    elif isinstance(cache, dict): # Handle plain dict caches
                         cache.clear()
                         cleared_count +=1

            self.logger.info(f"USP internal caches cleared ({cleared_count} caches).")

    def recover(self):
        """Recovers the UniversalSignalProcessor."""
        self.logger.warning("USP recovery triggered!")
        with self._lock:
            try:
                # 1. Clear all internal caches
                self.clear_cache()

                # 2. Re-acquire device handles from HardwareManager
                if self.hw_manager:
                    self.logger.debug("USP Recovery: Re-acquiring device handles from HardwareManager...")
                    # Ensure hw_manager itself is recovered/valid if possible
                    if hasattr(self.hw_manager, 'is_initialized') and not self.hw_manager._is_initialized:
                         self.logger.warning("USP Recovery: HardwareManager is not initialized. Attempting init.")
                         try: self.hw_manager.initialize_hardware()
                         except Exception as e_hw_init: self.logger.error(f"Failed HwM init during USP recovery: {e_hw_init}")

                    # Get potentially updated devices
                    self.torch_device = self.hw_manager.get_torch_device() or (torch.device('cpu') if TORCH_AVAILABLE else None)
                    self.quantum_device = self.hw_manager.get_pennylane_device()
                    self.n_qubits = self.hw_manager.get_optimal_qubits() # Get potentially updated qubit count

                    self.logger.info(f"USP Recovery: Devices re-acquired. Torch: {self.torch_device}, PL: {self.quantum_device.name if self.quantum_device else 'None'}, Qubits: {self.n_qubits}")
                else:
                    self.logger.warning("USP Recovery: HardwareManager not available. Cannot re-acquire devices.")

                # 3. Reset state (e.g., critical state detector, metrics)
                self.current_regime = "unknown"
                self.system_entropy = 0.0
                # Reinitialize detector? Or just reset values? Resetting values is safer.
                if hasattr(self, 'critical_state_detector') and isinstance(self.critical_state_detector, dict):
                    self.critical_state_detector = self._initialize_critical_detector()

                self.reset_metrics() # Reset internal performance metrics

                # 4. Potentially reload models (IF paths were stored or init logic allows)
                # If USP __init__ relies on passed instances, reloading here is difficult.
                # If it loads from paths stored in self.config, call that load method.
                # Example:
                # if hasattr(self, '_load_models') and callable(self._load_models):
                #      self.logger.info("USP Recovery: Reloading models...")
                #      self._load_models()

                self.logger.info("USP recovery attempt finished successfully.")

            except Exception as e_usp_rec:
                self.logger.error(f"Error during USP recovery: {e_usp_rec}", exc_info=True)
    def cleanup(self) -> None:
        """Clean up resources"""
        # Close any open resources
        if hasattr(self, "pool") and self.pool is not None:
            self.pool.close()
            self.pool.join()
        
        # Clear caches
        self.clear_cache()
        
        # Log cleanup
        self.logger.info("Resources cleaned up")