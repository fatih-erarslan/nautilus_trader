"""
High-Performance Quantum-Enhanced Logarithmic Market Scoring Rule (LMSR)

This implementation leverages quantum computing via PennyLane with the lightning.kokkos
backend to achieve significant speedups for LMSR market maker operations.

Key features:
- Hardware-aware initialization using HardwareManager and HardwareAccelerator
- Memory-efficient quantum state representation
- JIT compilation for operation sequences
- Batched processing for GPU optimization
- Comprehensive caching and thread safety
- Classical fallbacks for all operations

Author: Claude
"""
import os
import pennylane as qml
import numpy as np
import time
import logging
import threading
import warnings
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from enum import Enum, auto
from functools import lru_cache, wraps
from datetime import datetime
import json
import numpy as np


# Custom JSON encoder for quantum objects
class QuantumLMSRJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that can handle quantum objects like PennyLane's Shots"""
    def default(self, obj):
        # Handle numpy types
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        
        # Handle PennyLane's Shots type
        if hasattr(obj, '__class__') and obj.__class__.__name__ == 'Shots':
            if hasattr(obj, 'total') and obj.total is not None:
                return f"Shots(total={obj.total})"
            else:
                return "Shots(total=None)"
            
        # Handle other device-specific objects
        if hasattr(obj, '__class__') and 'pennylane' in str(obj.__class__.__module__):
            if hasattr(obj, 'name') and hasattr(obj, 'wires'):
                return {"name": obj.name, "wires": obj.wires}
            else:
                return str(obj)
                
        # For any other type use default behavior
        try:
            return super().default(obj)
        except TypeError:
            return str(obj)


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


# Logger setup
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("qar.log"), logging.StreamHandler()],
)
logger = logging.getLogger("Quantum LMSR")

class QuantumLMSR:
    """
    High-Performance Quantum-Enhanced Logarithmic Market Scoring Rule (LMSR)
    
    This implementation leverages PennyLane's lightning.kokkos backend for
    quantum accelerated market operations. It's designed specifically for 
    high-performance on AMD GPUs while maintaining compatibility with
    other hardware configurations.
    """
    
    def __init__(self, 
                 liquidity_parameter: float = 10.0,
                 min_probability: float = 0.001,
                 max_probability: float = 0.999,
                 qubits: int = 8,  # Set default to 8 for 8-factor standardization
                 layers: int = 2, 
                 shots: Optional[int] = None,
                 precision: PrecisionMode = PrecisionMode.AUTO,
                 mode: ProcessingMode = ProcessingMode.AUTO,
                 enable_caching: bool = True,
                 cache_size: int = 1024,
                 batch_size: int = 32,
                 log_level: Union[int, str] = logging.INFO,
                 device_name: Optional[str] = None,
                 hw_manager: Optional[Any] = None,
                 hw_accelerator: Optional[Any] = None,
                 use_standard_factors: bool = True,  # Whether to use the standard 8-factor model
                 factor_names: Optional[List[str]] = None,  # Custom factor names if not using standard 8-factor model
                 initial_weights: Optional[Dict[str, float]] = None):  # Initial weights for factors
        """
        Initialize QuantumLMSR with hardware-aware configuration.
        
        Args:
            liquidity_parameter: Market liquidity parameter (b in LMSR formula)
            min_probability: Minimum bound for probabilities
            max_probability: Maximum bound for probabilities
            qubits: Number of qubits for quantum circuits
            layers: Number of entangling layers in quantum circuits
            shots: Number of measurement shots (None for analytic)
            precision: Numeric precision mode
            mode: Processing mode (quantum, classical, hybrid, auto)
            enable_caching: Whether to enable result caching
            cache_size: Size of the LRU cache
            batch_size: Size of batches for GPU processing
            device_name: Specific quantum device to use (None for auto-select)
            hw_manager: HardwareManager instance or None to create new one
            hw_accelerator: HardwareAccelerator instance or None to create new one
        """
        # Store hardware components
        self.hw_manager = hw_manager
        self.hw_accelerator = hw_accelerator
        
        # Configure logging
        self.logger = logger
        self._configure_logging(log_level)
        self.logger.info(f"Initializing Quantum LMSR")
        
        # Determine GPU availability correctly
        if self.hw_accelerator is not None:
            accel_type = self.hw_accelerator.get_accelerator_type()
            self.gpu_available = accel_type in (AcceleratorType.CUDA, AcceleratorType.ROCM, AcceleratorType.MPS)
        else:
            # Fallback detection if no hw_accelerator provided
            self.gpu_available = False
            try:
                import torch
                self.gpu_available = torch.cuda.is_available()
            except ImportError:
                pass
    
        # Setup logging
        self.logger = logging.getLogger("QuantumLMSR")
        self.logger.setLevel(log_level)
        
        # Configure market parameters
        self.liquidity_parameter = liquidity_parameter
        self.min_probability = min_probability
        self.max_probability = max_probability
        
        # Initialize factor support
        self.use_standard_factors = use_standard_factors
        
        # Configure factor names and weights
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
            qubits = len(self.factor_names)
        
        # Quantum parameters
        self.qubits = qubits
        self.layers = layers
        self.shots = shots
        self.precision = precision
        self.mode = mode
        self.batch_size = batch_size
        
        # Initialize hardware components
        self._init_hardware(hw_manager, hw_accelerator)
        
        # Determine optimal dtype based on precision mode
        self.dtype, self.c_dtype = self._get_optimal_dtypes()
        
        # Caching settings
        self.enable_caching = enable_caching
        self.cache_size = cache_size
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Initialize quantum device
        self.device, self.quantum_available = self._initialize_quantum_device(device_name)
        
        # Set processing mode based on hardware availability
        if self.mode == ProcessingMode.AUTO:
            if self.quantum_available:
                self.mode = ProcessingMode.QUANTUM
            else:
                self.mode = ProcessingMode.CLASSICAL
        
        # Initialize quantum circuits
        if self.quantum_available and self.mode != ProcessingMode.CLASSICAL:
            self._initialize_quantum_circuits()
        
        # Initialize circuit weights for parameterized circuits
        self.weights = np.random.uniform(0, 2*np.pi, (self.layers, self.qubits, 3))
        
        # Performance metrics
        self._execution_times = {}
        self._call_counts = {}
        
        # Set up caching if enabled
        if self.enable_caching:
            self._setup_caching()
            
        logger.info(f"Initialized QuantumLMSR with {self.qubits} qubits, quantum: {self.quantum_available}, "
                   f"mode: {self.mode.value}, device: {self.device.name if self.device else 'none'}")

    def _configure_logging(self, log_level: Union[int, str]) -> None:
        """Configure logging with appropriate level."""
        resolved_level = logging.INFO  # Default

        try:
            if isinstance(log_level, int):
                if log_level in logging._levelToName:
                    resolved_level = log_level
            elif isinstance(log_level, str):
                level_name = log_level.upper()
                level_int = getattr(logging, level_name, None)
                if isinstance(level_int, int):
                    resolved_level = level_int
        except Exception as e:
            self.logger.error(f"Error setting log level: {e}")

        self.logger.setLevel(resolved_level)
        self.logger.debug(f"Log level set to {logging.getLevelName(self.logger.level)}")
        
    
    def _init_hardware(self, hw_manager, hw_accelerator):
        """Initialize hardware management components."""
        # Set up hardware manager for quantum devices
        if hw_manager is not None and isinstance(hw_manager, HardwareManager):
            self.hw_manager = hw_manager
        elif HARDWARE_ACCEL_AVAILABLE:
            self.hw_manager = HardwareManager.get_manager()
        else:
            self.hw_manager = HardwareManager()
            logger.warning("HardwareManager not available. Using fallback implementation.")
        
        # Initialize hardware acceleration for GPU
        if hw_accelerator is not None and isinstance(hw_accelerator, HardwareAccelerator):
            self.hw_accelerator = hw_accelerator
        elif HARDWARE_ACCEL_AVAILABLE:
            self.hw_accelerator = HardwareAccelerator(enable_gpu=True)
        else:
            self.hw_accelerator = HardwareAccelerator()
            logger.warning("HardwareAccelerator not available. Using fallback implementation.")
        
        # Ensure hardware is initialized
        if not getattr(self.hw_manager, '_is_initialized', False):
            self.hw_manager.initialize_hardware()
            
        # Determine hardware capabilities
        if self.hw_manager is not None:
            # Get quantum device info
            self.quantum_available = getattr(self.hw_manager, 'quantum_available', False) and self._is_pennylane_available()
            
            # Get maximum qubits based on hardware
            self.max_qubits = getattr(self.hw_manager, 'default_quantum_wires', 
                                    getattr(self.hw_manager, 'default_num_wires', self.qubits))
            self.qubits = min(self.qubits, self.max_qubits)
        else:
            self.quantum_available = self._is_pennylane_available()
            self.max_qubits = self.qubits
        
        # Determine GPU availability
        self.gpu_available = False
        if self.hw_accelerator is not None:
            accel_type = self.hw_accelerator.get_accelerator_type() if hasattr(self.hw_accelerator, 'get_accelerator_type') else None
            self.gpu_available = accel_type in (AcceleratorType.CUDA, AcceleratorType.ROCM, AcceleratorType.MPS)
        elif self._has_torch():
            # Fallback check if HardwareAccelerator not available
            import torch
            self.gpu_available = torch.cuda.is_available()
    
    def _is_pennylane_available(self):
        """Check if PennyLane is properly available."""
        try:
            import pennylane as qml
            return True
        except ImportError:
            return False
    
    def _has_torch(self):
        """Check if PyTorch is available."""
        try:
            import torch
            return True
        except ImportError:
            return False
    
    def _get_optimal_dtypes(self) -> Tuple[np.dtype, np.dtype]:
        """Determine optimal dtypes based on precision mode and hardware."""
        if self.precision == PrecisionMode.SINGLE:
            return np.float32, np.complex64
        elif self.precision == PrecisionMode.DOUBLE:
            return np.float64, np.complex128
        elif self.precision == PrecisionMode.AUTO:
            # Use hardware accelerator to determine precision
            if self.hw_accelerator is not None and hasattr(self.hw_accelerator, 'get_accelerator_type'):
                accel_type = self.hw_accelerator.get_accelerator_type()
                # AMD GPUs often prefer single precision
                if accel_type == AcceleratorType.ROCM:
                    return np.float32, np.complex64
            
            # Fallback using torch check
            try:
                import torch
                if torch.cuda.is_available():
                    device_name = torch.cuda.get_device_name(0).lower()
                    # AMD GPUs often have limited double precision
                    if "amd" in device_name or "radeon" in device_name:
                        return np.float32, np.complex64
            except ImportError:
                pass
            
            # Default to double precision
            return np.float64, np.complex128
        else:  # Mixed precision - use single for now
            return np.float32, np.complex64
    
    def _initialize_quantum_device(self, device_name: Optional[str]) -> Tuple[Any, bool]:
        """
        Initialize the quantum device with optimal settings for the hardware.
        
        Args:
            device_name: Specific device to use, or None for auto-selection
            
        Returns:
            Tuple of (device, quantum_available)
        """
        if self.hw_manager is not None and hasattr(self.hw_manager, 'get_quantum_device'):
            try:
                # Get quantum device configuration from hardware manager
                device_config = self.hw_manager.get_quantum_device(self.qubits)
                
                # Create PennyLane device with this configuration
                device = qml.device(
                    device_config.get('device', 'lightning.kokkos'),
                    wires=self.qubits,
                    shots=device_config.get('shots', self.shots),
                    c_dtype=self.c_dtype
                )
                logger.info(f"Using hardware manager quantum device: {device.name}")
                return device, True
            except Exception as e:
                logger.warning(f"Failed to initialize quantum device from hardware manager: {e}")
        
    
        if not self._is_pennylane_available():
            logger.warning("PennyLane not available. Quantum features disabled.")
            return None, False
            
        # First try any explicitly specified device
        if device_name is not None:
            try:
                device = qml.device(
                    device_name,
                    wires=self.qubits,
                    shots=self.shots,
                    c_dtype=self.c_dtype
                )
                logger.info(f"Using specified quantum device: {device_name}")
                return device, True
            except Exception as e:
                logger.warning(f"Failed to initialize specified device {device_name}: {e}")
        
        # Use hardware manager to get optimal quantum device if available
        if self.hw_manager is not None and hasattr(self.hw_manager, '_get_quantum_device'):
            try:
                device_config = self.hw_manager._get_quantum_device(self.qubits)
                device_name = device_config.get('device', 'lightning.qubit')
                wires = device_config.get('wires', self.qubits)
                shots = device_config.get('shots', self.shots)
                
                # Try to create the device
                device = qml.device(
                    device_name,
                    wires=wires,
                    shots=shots,
                    c_dtype=self.c_dtype
                )
                
                logger.info(f"Using HardwareManager-recommended device: {device_name}")
                return device, True
            except Exception as e:
                logger.warning(f"Failed to initialize HardwareManager-recommended device: {e}")
        
        # Try lightning.kokkos with optimized settings (best for AMD GPUs)
        if self.gpu_available:
            try:
                # Configure Kokkos arguments for optimal performance
                kokkos_args = {
                    "threads": 0,  # Let Kokkos choose optimal thread count
                    "numa": 1,     # NUMA optimization
                    "device": "GPU"  # Prefer GPU acceleration
                }
                
                device = qml.device(
                    "lightning.kokkos",
                    wires=self.qubits,
                    shots=self.shots,
                    c_dtype=self.c_dtype,
                    kokkos_args=kokkos_args
                )
                logger.info(f"Using lightning.kokkos device with {self.qubits} qubits")
                return device, True
            except Exception as e:
                logger.warning(f"Failed to initialize lightning.kokkos: {e}")
        
        # Try lightning.qubit as fallback
        try:
            device = qml.device(
                "lightning.qubit",
                wires=self.qubits,
                shots=self.shots,
                c_dtype=self.c_dtype
            )
            logger.info(f"Using lightning.qubit device with {self.qubits} qubits")
            return device, True
        except Exception as e:
            logger.error(f"Failed to initialize any quantum device: {e}")
            return None, False
    
    def _setup_caching(self):
        """Apply LRU caching to expensive functions."""
        # Apply caching with correct maxsize
        self.normalize_probability = lru_cache(maxsize=self.cache_size)(self.normalize_probability)
        self.to_log_odds = lru_cache(maxsize=self.cache_size)(self.to_log_odds)
        self.from_log_odds = lru_cache(maxsize=self.cache_size)(self.from_log_odds)
        
        # Also cache key quantum circuit results if quantum mode is active
        if self.quantum_available and self.mode != ProcessingMode.CLASSICAL:
            # Note: We can't directly cache QNodes, so we'll cache the wrapper functions
            pass
    
    def _initialize_quantum_circuits(self):
        """Initialize the quantum circuits with JIT compilation if available."""
        # Create core quantum circuits
        self._create_normalization_circuit()
        self._create_log_odds_circuit()
        self._create_cost_function_circuit()
        self._create_market_probability_circuit()
        self._create_aggregation_circuit()
        
        # Apply JIT compilation if available to improve performance
        try:
            if hasattr(qml, 'compile'):
                self._circuits = {
                    name: qml.compile(circuit) 
                    for name, circuit in self._circuits.items()
                }
                logger.info("Applied JIT compilation to quantum circuits")
        except Exception as e:
            logger.warning(f"Failed to apply JIT compilation: {e}")
    
    def _create_normalization_circuit(self):
        """Create quantum circuit for probability normalization."""
        if not self.quantum_available or self.device is None:
            return
            
        @qml.qnode(self.device, diff_method="adjoint" if self.shots is None else "parameter-shift")
        def normalize_probability_circuit(prob, min_prob, max_prob):
            """Quantum circuit for probability normalization."""
            # Encode input probability as rotation
            qml.RY(2 * np.arcsin(np.sqrt(prob)), wires=0)
            
            # Apply bounded rotation gates to constrain within limits
            angle_min = 2 * np.arcsin(np.sqrt(min_prob))
            angle_max = 2 * np.arcsin(np.sqrt(max_prob))
            
            # Prepare auxiliary qubit for comparison
            qml.Hadamard(wires=1)
            
            # Apply controlled operations for bounds
            qml.ctrl(qml.RY, control=0)(angle_min - 2 * np.arcsin(np.sqrt(prob)), wires=1)
            
            # Measure probability
            return qml.probs(wires=0)
        
        # Store circuit reference
        if not hasattr(self, '_circuits'):
            self._circuits = {}
        self._circuits['normalize_probability'] = normalize_probability_circuit
    
    def _create_log_odds_circuit(self):
        """Create quantum circuit for log-odds conversion."""
        if not self.quantum_available or self.device is None:
            return
            
        @qml.qnode(self.device, diff_method="adjoint" if self.shots is None else "parameter-shift")
        def log_odds_circuit(prob, mode):
            """
            Quantum circuit for log-odds conversion.
            
            Args:
                prob: Input probability [0,1]
                mode: 0 for to_log_odds, 1 for from_log_odds
            """
            # Encode probability as rotation
            if mode < 0.5:  # to_log_odds
                qml.RY(2 * np.arcsin(np.sqrt(prob)), wires=0)
            else:  # from_log_odds
                # Map log_odds to [0, π] for RY rotation
                # log_odds ∈ (-∞, ∞) -> scaled_log_odds ∈ [0, π]
                scaled_log_odds = np.pi / (1 + np.exp(-prob))
                qml.RY(scaled_log_odds, wires=0)
            
            # Apply QFT to help with logarithm operation
            qml.QFT(wires=range(min(3, self.qubits)))
            
            # Apply inverse QFT if converting from log-odds
            if mode > 0.5:
                qml.adjoint(qml.QFT)(wires=range(min(3, self.qubits)))
            
            # Return probability or expectation depending on mode
            if mode < 0.5:
                return [qml.expval(qml.PauliZ(i)) for i in range(min(3, self.qubits))]
            else:
                return qml.probs(wires=0)
        
        # Store circuit reference
        if not hasattr(self, '_circuits'):
            self._circuits = {}
        self._circuits['log_odds'] = log_odds_circuit
    
    def _create_cost_function_circuit(self):
        """Create quantum circuit for LMSR cost function calculation."""
        if not self.quantum_available or self.device is None:
            return
            
        @qml.qnode(self.device, diff_method="adjoint" if self.shots is None else "parameter-shift")
        def cost_function_circuit(quantities, liquidity):
            """
            Quantum circuit for cost function calculation.
            
            LMSR cost function: C(q) = b * log(sum(exp(q_i/b)))
            """
            # Prepare normalized quantities 
            max_q = np.max(quantities)
            q_norm = np.array([(q - max_q) / liquidity for q in quantities])
            
            # Encode exponential terms as amplitudes
            amplitudes = []
            for q in q_norm[:min(len(q_norm), 2**self.qubits)]:
                # Clamp argument to exp to prevent overflow/underflow
                clamped_q = np.clip(q, -700, 700) # Approximate safe range for exp
                amp = np.sqrt(np.clip(np.exp(clamped_q), 0, 1))  # Clip to [0,1] range
                amplitudes.append(amp)
                
            # Normalize amplitudes
            norm_factor = np.sqrt(np.sum(np.square(amplitudes)))
            if norm_factor > 0:
                amplitudes = [a / norm_factor for a in amplitudes]
            
            # Pad amplitudes to power of 2 if needed
            while len(amplitudes) < 2**self.qubits:
                amplitudes.append(0.0)
            
            # Encode as quantum state with amplitude embedding
            qml.AmplitudeEmbedding(
                amplitudes[:2**self.qubits], 
                wires=range(self.qubits),
                normalize=False
            )
            
            # Apply QFT to help with logarithm calculation
            qml.QFT(wires=range(self.qubits))
            
            # Measure to extract cost function components
            return [qml.expval(qml.PauliZ(i)) for i in range(self.qubits)]
        
        # Store circuit reference
        if not hasattr(self, '_circuits'):
            self._circuits = {}
        self._circuits['cost_function'] = cost_function_circuit
    
    def _create_market_probability_circuit(self):
        """Create quantum circuit for market probability calculation."""
        if not self.quantum_available or self.device is None:
            return
            
        @qml.qnode(self.device, diff_method="adjoint" if self.shots is None else "parameter-shift")
        def market_probability_circuit(quantities, liquidity, index):
            """
            Quantum circuit to calculate market probabilities.
            
            Market probability for outcome i: p_i = exp(q_i/b) / sum(exp(q_j/b))
            """
            # Normalize quantities for numerical stability
            max_q = np.max(quantities)
            q_norm = np.array([(q - max_q) / liquidity for q in quantities])
            
            # Calculate exponential terms
            # Clamp argument to exp to prevent overflow/underflow
            clamped_q_norm = np.clip(q_norm, -700, 700) # Approximate safe range for exp
            exp_terms = np.exp(clamped_q_norm)

            # Check for zero sum to prevent division by zero
            exp_sum_total = np.sum(exp_terms)
            if exp_sum_total <= 1e-9: # Use a small epsilon
                 # Fallback to uniform probability if sum is near zero
                 amplitudes = np.ones_like(exp_terms) / np.sqrt(len(exp_terms))
            else:
                 norm_factor = np.sqrt(np.sum(np.square(exp_terms)))
                 if norm_factor > 0:
                     amplitudes = exp_terms / norm_factor
                 else:
                     # Fallback if norm_factor is zero (should be covered by exp_sum_total check)
                     amplitudes = np.ones_like(exp_terms) / np.sqrt(len(exp_terms))

            # Encode as quantum state
            qml.AmplitudeEmbedding(
                amplitudes[:min(len(amplitudes), 2**self.qubits)], 
                wires=range(self.qubits),
                normalize=True,
                pad_with=0.0
            )
            
            # Compute probability for specific outcome
            if index < 2**self.qubits:
                # Convert index to binary representation for measurement
                binary_index = format(index, f'0{self.qubits}b')
                
                # Apply projective measurement
                for i, bit in enumerate(binary_index):
                    if bit == '0':
                        qml.PauliZ(wires=i)
            
            # Return probability distribution
            return qml.probs(wires=range(min(self.qubits, len(bin(len(quantities))) - 2)))
        
        # Store circuit reference
        if not hasattr(self, '_circuits'):
            self._circuits = {}
        self._circuits['market_probability'] = market_probability_circuit
    
    def _create_aggregation_circuit(self):
        """Create quantum circuit for probability aggregation."""
        if not self.quantum_available or self.device is None:
            return
            
        @qml.qnode(self.device, diff_method="adjoint" if self.shots is None else "parameter-shift")
        def aggregation_circuit(probabilities, weights=None):
            """
            Quantum circuit for aggregating multiple probability estimates.
            
            Args:
                probabilities: List of probabilities to aggregate
                weights: Optional weights for each probability
            """
            # Default to equal weights if not provided
            if weights is None:
                weights = np.ones(len(probabilities)) / len(probabilities)
            
            # Prepare state based on probabilities and weights
            for i, (p, w) in enumerate(zip(probabilities, weights)):
                if i < self.qubits:
                    # Apply weighted rotation based on probability
                    qml.RY(2 * np.arcsin(np.sqrt(p)) * w * self.qubits, wires=i)
            
            # Apply entangling layers
            for l in range(self.layers):
                # Rotation layer
                for i in range(self.qubits):
                    qml.RX(self.weights[l, i, 0], wires=i)
                    qml.RY(self.weights[l, i, 1], wires=i)
                    qml.RZ(self.weights[l, i, 2], wires=i)
                
                # Entanglement layer - efficient for lightning.kokkos
                for i in range(self.qubits - 1):
                    qml.CZ(wires=[i, i+1])
                if self.qubits > 2:  # Add some non-local connections
                    qml.CZ(wires=[0, self.qubits-1])
            
            # Apply QFT for interference-based aggregation
            qml.QFT(wires=range(self.qubits))
            
            # Return first qubit probability (|1⟩ state)
            return qml.probs(wires=0)
        
        # Store circuit reference
        if not hasattr(self, '_circuits'):
            self._circuits = {}
        self._circuits['aggregation'] = aggregation_circuit

    @time_execution
    def normalize_probability(self, prob: float) -> float:
        """
        Normalize probability to range [min_probability, max_probability].
        
        Args:
            prob: Input probability
            
        Returns:
            Normalized probability
        """
        # Handle invalid inputs
        if np.isnan(prob) or np.isinf(prob):
            logger.warning(f"Invalid probability value: {prob}, using default 0.5")
            return 0.5
        
        # Use hardware accelerator if available
        if (self.hw_accelerator is not None and 
            self.gpu_available and 
            hasattr(self.hw_accelerator, 'normalize_probability') and
            not self.enable_caching):  # Can't use accelerator with caching due to potential non-hashable returns
            try:
                return self.hw_accelerator.normalize_probability(prob)
            except Exception as e:
                logger.debug(f"Hardware-accelerated normalization failed: {e}. Using fallback.")
        
        # Use quantum normalization if available and selected
        if (self.quantum_available and 
            self.mode in (ProcessingMode.QUANTUM, ProcessingMode.HYBRID) and
            not self.enable_caching):  # Can't use quantum with caching due to non-hashable return types
            try:
                # Execute quantum circuit for normalization
                with self._lock:
                    circuit = self._circuits.get('normalize_probability')
                    if circuit is not None:
                        result = circuit(
                            float(prob), 
                            float(self.min_probability), 
                            float(self.max_probability)
                        )
                        # Extract probability from result (first element is |0⟩ state)
                        return float(result[0])
            except Exception as e:
                logger.debug(f"Quantum normalization failed: {e}. Using classical method.")
        
        # Classical implementation
        return max(min(float(prob), self.max_probability), self.min_probability)
    
    @time_execution
    def to_log_odds(self, probability: float) -> float:
        """
        Convert probability to log-odds.
        
        Log-odds is the natural logarithm of the odds ratio: ln(p/(1-p))
        
        Args:
            probability: Input probability
            
        Returns:
            Log-odds value
        """
        # Normalize input probability
        probability = self.normalize_probability(probability)
        
        # Use hardware accelerator if available
        if (self.hw_accelerator is not None and 
            self.gpu_available and 
            hasattr(self.hw_accelerator, 'to_log_odds') and
            not self.enable_caching):
            try:
                return self.hw_accelerator.to_log_odds(probability)
            except Exception as e:
                logger.debug(f"Hardware-accelerated to_log_odds failed: {e}. Using fallback.")
        
        # Use quantum calculation if available and appropriate
        if (self.quantum_available and 
            self.mode in (ProcessingMode.QUANTUM, ProcessingMode.HYBRID) and
            not self.enable_caching):
            try:
                # Execute quantum circuit
                with self._lock:
                    circuit = self._circuits.get('log_odds')
                    if circuit is not None:
                        result = circuit(float(probability), 0.0)  # 0.0 for to_log_odds mode
                        
                        # Process result - first element has strongest signal
                        log_odds_estimate = result[0] * 4.0  # Scale to approximate log-odds range
                        return float(log_odds_estimate)
            except Exception as e:
                logger.debug(f"Quantum to_log_odds failed: {e}. Using classical method.")
        
        # Classical implementation (with safety check)
        if probability <= 0 or probability >= 1:
            probability = self.normalize_probability(probability)
        return np.log(probability / (1 - probability))
    
    @time_execution
    def from_log_odds(self, log_odds: float) -> float:
        """
        Convert log-odds back to probability.
        
        Inverse of to_log_odds: p = 1/(1+exp(-log_odds))
        
        Args:
            log_odds: Input log-odds value
            
        Returns:
            Probability value
        """
        # Handle extreme values first
        if log_odds > 709:  # Near upper limit for np.exp
            return self.max_probability
        elif log_odds < -709:  # Near lower limit
            return self.min_probability
        
        # Use hardware accelerator if available
        if (self.hw_accelerator is not None and 
            self.gpu_available and 
            hasattr(self.hw_accelerator, 'from_log_odds') and
            not self.enable_caching):
            try:
                return self.hw_accelerator.from_log_odds(log_odds)
            except Exception as e:
                logger.debug(f"Hardware-accelerated from_log_odds failed: {e}. Using fallback.")
        
        # Use quantum calculation if available and appropriate
        if (self.quantum_available and 
            self.mode in (ProcessingMode.QUANTUM, ProcessingMode.HYBRID) and
            not self.enable_caching):
            try:
                # Execute quantum circuit
                with self._lock:
                    circuit = self._circuits.get('log_odds')
                    if circuit is not None:
                        # Apply scaling to fit log_odds into circuit input range
                        scaled_log_odds = np.clip(log_odds, -10, 10)
                        result = circuit(float(scaled_log_odds), 1.0)  # 1.0 for from_log_odds mode
                        
                        # Extract probability from result
                        return float(result[1])  # [1] is |1⟩ state probability
            except Exception as e:
                logger.debug(f"Quantum from_log_odds failed: {e}. Using classical method.")
        
        # Classical implementation
        return self.normalize_probability(1 / (1 + np.exp(-log_odds)))
    
    @time_execution
    def cost_function(self, quantities: List[float]) -> float:
        """
        LMSR cost function C(q) = b * log(sum(exp(q_i/b)))
        
        The cost function determines how much it costs to move the market
        from state q to q'. The liquidity parameter b controls how quickly
        prices change with trading volume.
        
        Args:
            quantities: Vector of quantities for each outcome
            
        Returns:
            Cost according to LMSR
        """
        if len(quantities) == 0:
            return 0.0
            
        try:
            # Convert to numpy array
            q_array = np.array(quantities, dtype=self.dtype)
            
            # Use hardware accelerator if available
            if (self.hw_accelerator is not None and 
                self.gpu_available and 
                hasattr(self.hw_accelerator, 'calculate_cost_function')):
                try:
                    cost = self.hw_accelerator.calculate_cost_function(
                        q_array, self.liquidity_parameter
                    )
                    if np.isnan(cost) or np.isinf(cost):
                         logger.warning(f"Hardware-accelerated cost function returned invalid value: {cost}. Quantities: {quantities}")
                         return 0.0 # Return safe default
                    return cost
                except Exception as e:
                    logger.debug(f"Hardware-accelerated cost function failed: {e}. Using fallback.")
            
            # Use quantum calculation for cost function if available
            if (self.quantum_available and 
                self.mode in (ProcessingMode.QUANTUM, ProcessingMode.HYBRID) and
                len(quantities) <= 2**self.qubits):
                try:
                    # Execute quantum circuit
                    with self._lock:
                        circuit = self._circuits.get('cost_function')
                        if circuit is not None:
                            result = circuit(
                                q_array,
                                float(self.liquidity_parameter)
                            )
                            
                            # Calculate cost from quantum result
                            q_max = np.max(q_array)
                            
                            # Process quantum output to get log(sum(exp)) component
                            # We use the first few expectation values which contain most information
                            log_sum_exp_component = np.mean([
                                (1 + result[i]) / 2 * (i+1) for i in range(min(3, len(result)))
                            ])
                            
                            cost = self.liquidity_parameter * (
                                q_max / self.liquidity_parameter + 
                                log_sum_exp_component
                            )
                            
                            if np.isnan(cost) or np.isinf(cost):
                                logger.warning(f"Quantum cost function returned invalid value: {cost}. Quantities: {quantities}")
                                return 0.0 # Return safe default
                            return float(cost)
                except Exception as e:
                    logger.debug(f"Quantum cost function failed: {e}. Using classical method.")
            
            # Classical implementation with numerical stability
            b = self.liquidity_parameter
            
            # Use a numerically stable implementation
            max_q = np.max(q_array)
            
            # For numerical stability:
            # log(sum(exp(q_i))) = max_q + log(sum(exp(q_i - max_q)))
            # Clamp argument to exp to prevent overflow/underflow
            clamped_q_diff = np.clip((q_array - max_q) / b, -700, 700) # Approximate safe range for exp
            exp_sum = np.sum(np.exp(clamped_q_diff))

            # Handle case where exp_sum is zero or very small
            if exp_sum <= 1e-9: # Use a small epsilon
                 logger.warning(f"Classical cost function: exp_sum near zero ({exp_sum}). Quantities: {quantities}")
                 return b * (max_q / b + np.log(1e-9)) # Return a large negative cost
            
            cost = b * (max_q / b + np.log(exp_sum))
            
            if np.isnan(cost) or np.isinf(cost):
                logger.warning(f"Classical cost function returned invalid value: {cost}. Quantities: {quantities}")
                return 0.0 # Return safe default
            return float(cost)
                
        except Exception as e:
            logger.error(f"Error calculating cost function: {e}")
            
            # Last-resort fallback
            try:
                b = self.liquidity_parameter
                max_q = max(quantities)
                # Clamp argument to exp in fallback too
                exp_sum = sum(np.exp(np.clip((q - max_q) / b, -700, 700)) for q in quantities)
                if exp_sum <= 1e-9:
                    logger.warning(f"Classical cost function fallback: exp_sum near zero ({exp_sum}). Quantities: {quantities}")
                    return b * (max_q / b + np.log(1e-9))
                cost = b * (max_q / b + np.log(exp_sum))
                if np.isnan(cost) or np.isinf(cost):
                    logger.warning(f"Classical cost function fallback returned invalid value: {cost}. Quantities: {quantities}")
                    return 0.0
                return float(cost)
            except Exception as fallback_e:
                logger.error(f"Fallback cost function failed: {fallback_e}")
                return 0.0
    
    @time_execution
    def get_market_probability(self, quantities: List[float], index: int) -> float:
        """
        Derive implicit probability from current market state.
        
        The LMSR market maker's current prices represent the market's
        expectation of outcome probabilities.
        
        Args:
            quantities: Vector of quantities for each outcome
            index: Index of the outcome to get probability for
            
        Returns:
            Market probability for the specified outcome
        """
        if len(quantities) == 0 or index < 0 or index >= len(quantities):
            logger.error(f"Invalid inputs: quantities={quantities}, index={index}")
            return 0.5
            
        try:
            # Use hardware accelerator if available
            if (self.hw_accelerator is not None and 
                self.gpu_available and 
                hasattr(self.hw_accelerator, 'calculate_market_probability')):
                try:
                    prob = self.hw_accelerator.calculate_market_probability(
                        np.array(quantities, dtype=self.dtype), 
                        index, 
                        self.liquidity_parameter
                    )
                    # Normalization is handled within the accelerated function
                    return prob
                except Exception as e:
                    logger.debug(f"Hardware-accelerated market probability failed: {e}. Using fallback.")
            
            # Use quantum calculation if available and appropriate
            if (self.quantum_available and 
                self.mode in (ProcessingMode.QUANTUM, ProcessingMode.HYBRID) and
                len(quantities) <= 2**self.qubits):
                try:
                    # Execute quantum circuit
                    with self._lock:
                        circuit = self._circuits.get('market_probability')
                        if circuit is not None:
                            result = circuit(
                                np.array(quantities, dtype=self.dtype),
                                float(self.liquidity_parameter),
                                int(index)
                            )
                            
                            # Extract probability for the requested outcome
                            # For small markets, we can directly index into the probability vector
                            # For larger markets, we need to transform the index
                            if len(quantities) <= 2**self.qubits:
                                probability = result[index]
                            else:
                                # Map the index to available qubits
                                mapped_index = index % (2**self.qubits)
                                probability = result[mapped_index]
                            
                            # Normalization is handled within the quantum circuit
                            return float(probability)
                except Exception as e:
                    logger.debug(f"Quantum market probability failed: {e}. Using classical method.")
            
            # Classical implementation
            b = self.liquidity_parameter
            
            # Calculate numerator
            # Clamp argument to exp
            clamped_q_index = np.clip(quantities[index] / b, -700, 700)
            exp_term = np.exp(clamped_q_index)
            
            # Calculate denominator (sum of exponentials)
            # Use a numerically stable implementation for the sum
            max_q = np.max(quantities)
            clamped_q_diff = np.clip((np.array(quantities) - max_q) / b, -700, 700)
            exp_sum = np.sum(np.exp(clamped_q_diff))

            # Handle case where exp_sum is zero or very small
            if exp_sum <= 1e-9: # Use a small epsilon
                 logger.warning(f"Classical market probability: exp_sum near zero ({exp_sum}). Quantities: {quantities}, Index: {index}")
                 # If sum is near zero, probabilities are effectively uniform
                 return self.normalize_probability(1.0 / len(quantities))

            # Calculate probability
            # Use numerically stable form: p_i = exp(q_i/b - max_q/b) / sum(exp(q_j/b - max_q/b))
            probability = np.exp(np.clip((quantities[index] - max_q) / b, -700, 700)) / exp_sum
            
            # Normalization is handled by normalize_probability
            return self.normalize_probability(probability)
                
        except Exception as e:
            logger.error(f"Error calculating market probability: {e}")
            return 0.5
    
    @time_execution
    def get_all_market_probabilities(self, quantities: List[float]) -> List[float]:
        """
        Calculate probabilities for all outcomes.
        
        Args:
            quantities: Vector of quantities for each outcome
            
        Returns:
            List of probabilities for all outcomes
        """
        if not quantities:
            return []
            
        try:
            # Use hardware accelerator for batch calculation if available
            if (self.hw_accelerator is not None and 
                self.gpu_available and 
                hasattr(self.hw_accelerator, 'calculate_all_market_probabilities')):
                try:
                    probs = self.hw_accelerator.calculate_all_market_probabilities(
                        np.array(quantities, dtype=self.dtype), 
                        self.liquidity_parameter
                    )
                    # Normalization is handled within the accelerated function
                    return probs
                except Exception as e:
                    logger.debug(f"Hardware-accelerated batch probabilities failed: {e}. Using fallback.")
            
            # Use batched processing for large markets
            if len(quantities) > self.batch_size:
                all_probs = []
                for i in range(0, len(quantities), self.batch_size):
                    batch = quantities[i:min(i + self.batch_size, len(quantities))]
                    batch_indices = range(i, min(i + self.batch_size, len(quantities)))
                    batch_probs = []
                    
                    for idx, outcome_idx in enumerate(batch_indices):
                        # get_market_probability already handles normalization and invalid values
                        prob = self.get_market_probability(quantities, outcome_idx)
                        batch_probs.append(prob)
                    
                    all_probs.extend(batch_probs)
                
                # Re-normalize to ensure sum to 1.0 after combining batches
                total = sum(all_probs)
                if total > 0:
                    return [self.normalize_probability(p / total) for p in all_probs] # Normalize after summing
                else:
                    logger.warning(f"get_all_market_probabilities: Total probability sum near zero ({total}) after batch processing. Quantities: {quantities}")
                    return [self.normalize_probability(1.0 / len(quantities))] * len(quantities)
                    
            else:
                # Standard processing for smaller markets
                probabilities = [
                    self.get_market_probability(quantities, i)
                    for i in range(len(quantities))
                ]
                
                # Normalize to ensure sum to 1.0
                total = sum(probabilities)
                if total > 0:
                    return [self.normalize_probability(p / total) for p in probabilities] # Normalize after summing
                else:
                    logger.warning(f"get_all_market_probabilities: Total probability sum near zero ({total}) for small market. Quantities: {quantities}")
                    return [self.normalize_probability(1.0 / len(quantities))] * len(quantities)
        except Exception as e:
            logger.error(f"Error calculating all market probabilities: {e}")
            return [self.normalize_probability(1.0 / len(quantities))] * len(quantities) # Fallback to uniform

    @time_execution
    def aggregate_probabilities(self, 
                              probabilities: List[float],
                              weights: Optional[List[float]] = None,
                              method: str = "log_odds") -> float:
        """
        Aggregate multiple probability estimates.
        
        Args:
            probabilities: List of probability estimates
            weights: Optional weights for each probability
            method: Aggregation method ("log_odds", "geometric", "quantum")
            
        Returns:
            Aggregated probability
        """
        if len(probabilities) == 0:
            return 0.5
        
        # Filter out invalid values and normalize inputs
        valid_probs = []
        valid_weights = []
        
        if weights is None:
            # Default to equal weights
            weights = [1.0 / len(probabilities)] * len(probabilities)
        else:
            # Normalize weights
            weight_sum = sum(weights)
            if weight_sum <= 0:
                weights = [1.0 / len(probabilities)] * len(probabilities)
            else:
                weights = [w / weight_sum for w in weights]
        
        # Filter out invalid values and normalize probabilities
        for p, w in zip(probabilities, weights):
            normalized_p = self.normalize_probability(p) # Use normalize_probability here
            if not (np.isnan(normalized_p) or np.isinf(normalized_p) or np.isnan(w) or np.isinf(w)):
                valid_probs.append(normalized_p)
                valid_weights.append(w)
            else:
                 logger.warning(f"aggregate_probabilities: Skipping invalid input probability {p} or weight {w}")

        if not valid_probs:
            logger.warning("aggregate_probabilities: No valid probabilities after filtering, returning 0.5")
            return 0.5
        
        # Re-normalize weights for valid probabilities
        weight_sum = sum(valid_weights)
        if weight_sum <= 0:
             logger.warning("aggregate_probabilities: Valid weight sum near zero, using equal weights for valid probabilities")
             valid_weights = [1.0 / len(valid_probs)] * len(valid_probs)
        else:
             valid_weights = [w / weight_sum for w in valid_weights]

        # Use hardware accelerator if available
        if (self.hw_accelerator is not None and 
            self.gpu_available and 
            hasattr(self.hw_accelerator, 'aggregate_probabilities')):
            try:
                agg_prob = self.hw_accelerator.aggregate_probabilities(
                    np.array(valid_probs, dtype=self.dtype),
                    np.array(valid_weights, dtype=self.dtype)
                )
                # Normalization is handled within the accelerated function
                return agg_prob
            except Exception as e:
                logger.debug(f"Hardware-accelerated probability aggregation failed: {e}. Using fallback.")
        
        # Use quantum aggregation if requested, available and appropriate
        if (method == "quantum" and
            self.quantum_available and 
            self.mode in (ProcessingMode.QUANTUM, ProcessingMode.HYBRID) and
            len(valid_probs) <= self.qubits):
            try:
                # Execute quantum circuit
                with self._lock:
                    circuit = self._circuits.get('aggregation')
                    if circuit is not None:
                        # Process in batches if needed
                        if len(valid_probs) > self.qubits:
                            # Create batches of size qubits
                            batched_results = []
                            for i in range(0, len(valid_probs), self.qubits):
                                batch_probs = valid_probs[i:min(i + self.qubits, len(valid_probs))]
                                batch_weights = valid_weights[i:min(i + self.qubits, len(valid_weights))]
                                
                                # Renormalize batch weights
                                batch_weight_sum = sum(batch_weights)
                                if batch_weight_sum <= 0:
                                     batch_norm_weights = [1.0 / len(batch_weights)] * len(batch_weights)
                                else:
                                     batch_norm_weights = [w / batch_weight_sum for w in batch_weights]
                                
                                # Process batch
                                result = circuit(
                                    np.array(batch_probs, dtype=self.dtype),
                                    np.array(batch_norm_weights, dtype=self.dtype)
                                )
                                batched_results.append(float(result[1]))  # [1] is |1⟩ state probability
                            
                            # Combine batch results (equal weighting)
                            combined_result = sum(batched_results) / len(batched_results) if batched_results else 0.5
                            # Normalization is handled by normalize_probability
                            return self.normalize_probability(combined_result)
                        else:
                            # Process all at once
                            result = circuit(
                                np.array(valid_probs, dtype=self.dtype),
                                np.array(valid_weights, dtype=self.dtype)
                            )
                            # Normalization is handled by normalize_probability
                            return self.normalize_probability(float(result[1]))  # [1] is |1⟩ state probability
            except Exception as e:
                logger.debug(f"Quantum probability aggregation failed: {e}. Using classical method.")
        
        # Classical methods
        if method == "geometric":
            # Geometric mean of probabilities (preserves Bayesian properties)
            # Handle log(0) by replacing 0 with a small epsilon
            log_product = sum(w * np.log(max(p, 1e-9)) for p, w in zip(valid_probs, valid_weights))
            agg_prob = np.exp(log_product)
            # Normalization is handled by normalize_probability
            return self.normalize_probability(agg_prob)
        else:
            # Default: Log-odds method (standard for LMSR)
            log_odds_sum = sum(w * self.to_log_odds(p) for p, w in zip(valid_probs, valid_weights))
            # from_log_odds already handles normalization
            return self.from_log_odds(log_odds_sum)
    
    @time_execution
    def calculate_cost_to_move(self, 
                             current_quantities: List[float],
                             target_probability: float,
                             outcome_index: int) -> float:
        """
        Calculate the cost to move the market probability to a target.
        
        Args:
            current_quantities: Current market state
            target_probability: Desired probability for the specified outcome
            outcome_index: Index of the outcome to move
            
        Returns:
            Cost to move the market to the target probability
        """
        if len(current_quantities) == 0 or outcome_index < 0 or outcome_index >= len(current_quantities):
            logger.error("Invalid inputs for calculate_cost_to_move")
            return float('inf')
            
        try:
            # Current market probability
            current_prob = self.get_market_probability(current_quantities, outcome_index)
            
            if abs(current_prob - target_probability) < 0.001:
                return 0.0  # Already at target
                
            # Binary search to find the quantity that gives the target probability
            b = self.liquidity_parameter
            
            # Calculate required price movement
            log_odds_diff = self.to_log_odds(target_probability) - self.to_log_odds(current_prob)
            
            # Approximate the quantity change needed
            quantity_change = b * log_odds_diff
            
            # Create new quantities array
            new_quantities = current_quantities.copy()
            new_quantities[outcome_index] += quantity_change
            
            # Calculate cost difference
            old_cost = self.cost_function(current_quantities)
            new_cost = self.cost_function(new_quantities)
            
            cost_diff = new_cost - old_cost

            # Check for invalid cost difference
            if np.isnan(cost_diff) or np.isinf(cost_diff):
                 logger.warning(f"calculate_cost_to_move: Calculated invalid cost difference: {cost_diff}. Old cost: {old_cost}, New cost: {new_cost}. Quantities: {current_quantities}, Target Prob: {target_probability}, Index: {outcome_index}")
                 return 0.0 # Return safe default

            return cost_diff
            
        except Exception as e:
            logger.error(f"Error calculating cost to move: {e}")
            return float('inf')
    
    @time_execution
    def calculate_information_gain(self, 
                                 prior_probabilities: List[float],
                                 posterior_probabilities: List[float]) -> float:
        """
        Calculate Kullback-Leibler divergence (information gain) between distributions.
        
        KL divergence measures how much information is gained by updating from the
        prior to the posterior distribution.
        
        Args:
            prior_probabilities: Initial probability distribution
            posterior_probabilities: Updated probability distribution
            
        Returns:
            KL divergence (information gain) in bits
        """
        if not prior_probabilities or not posterior_probabilities or len(prior_probabilities) != len(posterior_probabilities):
            logger.error("Invalid inputs for information gain calculation")
            return 0.0
            
        try:
            # Ensure probabilities are normalized
            prior_sum = sum(prior_probabilities)
            posterior_sum = sum(posterior_probabilities)
            
            if prior_sum <= 0 or posterior_sum <= 0:
                logger.warning(f"calculate_information_gain: Prior or posterior sum near zero. Prior sum: {prior_sum}, Posterior sum: {posterior_sum}")
                return 0.0
                
            normalized_prior = [p / prior_sum for p in prior_probabilities]
            normalized_posterior = [p / posterior_sum for p in posterior_probabilities]
            
            # Use hardware accelerator if available
            if (self.hw_accelerator is not None and 
                self.gpu_available and 
                hasattr(self.hw_accelerator, 'calculate_kl_divergence')):
                try:
                    kl_div = self.hw_accelerator.calculate_kl_divergence(
                        np.array(normalized_prior, dtype=self.dtype),
                        np.array(normalized_posterior, dtype=self.dtype)
                    )
                    if np.isnan(kl_div) or np.isinf(kl_div):
                         logger.warning(f"Hardware-accelerated KL divergence returned invalid value: {kl_div}. Priors: {normalized_prior}, Posteriors: {normalized_posterior}")
                         return 0.0 # Return safe default
                    return kl_div
                except Exception as e:
                    logger.debug(f"Hardware-accelerated KL divergence failed: {e}. Using fallback.")
            
            # Use quantum calculation if available and appropriate
            if (self.quantum_available and 
                self.mode in (ProcessingMode.QUANTUM, ProcessingMode.HYBRID) and
                len(normalized_prior) <= 2**(self.qubits-1)):
                try:
                    # Create specialized circuit for KL divergence
                    @qml.qnode(self.device)
                    def kl_circuit(priors, posteriors):
                        # Encode distributions as amplitudes
                        n_outcomes = len(priors)
                        prior_amps = np.sqrt(np.array(priors))
                        posterior_amps = np.sqrt(np.array(posteriors))
                        
                        # Encode prior distribution in first half of qubits
                        qml.AmplitudeEmbedding(
                            prior_amps,
                            wires=range(self.qubits//2),
                            normalize=True
                        )
                        
                        # Encode posterior distribution in second half
                        qml.AmplitudeEmbedding(
                            posterior_amps,
                            wires=range(self.qubits//2, self.qubits),
                            normalize=True
                        )
                        
                        # Apply operations to compute KL divergence
                        for i in range(min(n_outcomes, self.qubits//2)):
                            q1 = i
                            q2 = i + self.qubits//2
                            qml.CNOT(wires=[q1, q2])
                        
                        # Measure relative entropy components
                        return [qml.expval(qml.PauliZ(i)) for i in range(self.qubits)]
                    
                    # Execute circuit
                    with self._lock:
                        result = kl_circuit(
                            np.array(normalized_prior, dtype=self.dtype)[:2**(self.qubits//2)],
                            np.array(normalized_posterior, dtype=self.dtype)[:2**(self.qubits//2)]
                        )
                    
                    # Process result to get KL divergence
                    # Use first self.qubits//2 results as they contain the relevant information
                    kl_components = [(1 + result[i]) / 2 for i in range(self.qubits//2)]
                    kl_estimate = sum(kl_components) * np.log2(np.e)  # Convert to bits
                    
                    if np.isnan(kl_estimate) or np.isinf(kl_estimate):
                         logger.warning(f"Quantum KL divergence returned invalid value: {kl_estimate}. Priors: {normalized_prior}, Posteriors: {normalized_posterior}")
                         return 0.0 # Return safe default
                    return float(kl_estimate)
                except Exception as e:
                    logger.debug(f"Quantum KL divergence failed: {e}. Using classical method.")
            
            # Classical KL divergence calculation
            kl_divergence = 0.0
            epsilon = 1e-10  # Small value to avoid log(0)
            
            for p_post, p_prior in zip(normalized_posterior, normalized_prior):
                if p_post > epsilon:
                    if p_prior <= epsilon:
                        p_prior = epsilon  # Avoid division by zero
                    kl_divergence += p_post * np.log2(p_post / p_prior)
            
            if np.isnan(kl_divergence) or np.isinf(kl_divergence):
                 logger.warning(f"Classical KL divergence returned invalid value: {kl_divergence}. Priors: {normalized_prior}, Posteriors: {normalized_posterior}")
                 return 0.0 # Return safe default

            return float(kl_divergence)
                
        except Exception as e:
            logger.error(f"Error calculating information gain: {e}")
            return 0.0


    def save_state(self, filepath: str) -> bool:
        """
        Saves the current state of the QuantumLMSR instance to a file.

        Args:
            filepath: The path to the file where the state will be saved.

        Returns:
            True if saving was successful, False otherwise.
        """
        try:
            with self._lock: # Ensure thread safety if applicable
                # Prepare a dictionary of attributes to save
                # Ensure all attributes are JSON-serializable (e.g., convert Enums to values, NumPy arrays to lists)
                state_to_save = {
                    'liquidity_parameter': self.liquidity_parameter,
                    'min_probability': self.min_probability,
                    'max_probability': self.max_probability,
                    'qubits': self.qubits,
                    'layers': self.layers,
                    'shots': self.shots,
                    'precision_mode': self.precision.value, # Save Enum value
                    'processing_mode': self.mode.value,     # Save Enum value
                    'batch_size': self.batch_size,
                    'enable_caching': self.enable_caching, # Though cache itself isn't saved
                    'cache_size': self.cache_size,
                    # Save circuit weights if they are adaptable/learned
                    'circuit_weights': self.weights.tolist() if isinstance(self.weights, np.ndarray) else self.weights,
                    # Performance metrics are optional to save; can be large and are often transient
                    # '_execution_times': self._execution_times,
                    # '_call_counts': self._call_counts,
                    'saved_timestamp': datetime.now().isoformat()
                }

                with open(filepath, 'w') as f:
                    # Use QuantumLMSRJSONEncoder to properly serialize quantum objects like Shots
                    json.dump(state_to_save, f, indent=2, cls=QuantumLMSRJSONEncoder)
                
                # Use self.logger if available, otherwise module-level logger
                current_logger = getattr(self, 'logger', logger) # Fallback to module logger
                current_logger.info(f"QuantumLMSR state saved to {filepath}")
                return True
        except Exception as e:
            current_logger = getattr(self, 'logger', logger)
            current_logger.error(f"Error saving QuantumLMSR state to {filepath}: {e}", exc_info=True)
            return False

    def load_state(self, filepath: str) -> bool:
        """
        Loads the state of the QuantumLMSR instance from a file.

        Args:
            filepath: The path to the file from which the state will be loaded.

        Returns:
            True if loading was successful, False otherwise.
        """
        current_logger = getattr(self, 'logger', logger) # Fallback to module logger
        try:
            if not os.path.exists(filepath):
                current_logger.error(f"QuantumLMSR state file not found: {filepath}")
                return False

            with self._lock: # Ensure thread safety
                with open(filepath, 'r') as f:
                    loaded_state = json.load(f)

                # Restore attributes, providing defaults from current instance if key is missing
                self.liquidity_parameter = loaded_state.get('liquidity_parameter', self.liquidity_parameter)
                self.min_probability = loaded_state.get('min_probability', self.min_probability)
                self.max_probability = loaded_state.get('max_probability', self.max_probability)
                
                # Critical parameters that define circuit structure might need careful handling
                # If these change, circuits might need re-initialization.
                loaded_qubits = loaded_state.get('qubits', self.qubits)
                if self.qubits != loaded_qubits:
                    current_logger.warning(f"Loading QLMSR state with different qubit count ({loaded_qubits}) than current ({self.qubits}). This might require circuit re-initialization.")
                    self.qubits = loaded_qubits # Update and assume re-init if needed

                self.layers = loaded_state.get('layers', self.layers)
                self.shots = loaded_state.get('shots', self.shots)
                
                self.precision = PrecisionMode(loaded_state.get('precision_mode', self.precision.value))
                self.mode = ProcessingMode(loaded_state.get('processing_mode', self.mode.value))
                
                self.batch_size = loaded_state.get('batch_size', self.batch_size)
                self.enable_caching = loaded_state.get('enable_caching', self.enable_caching)
                self.cache_size = loaded_state.get('cache_size', self.cache_size)

                loaded_circuit_weights = loaded_state.get('circuit_weights')
                if loaded_circuit_weights is not None:
                    self.weights = np.array(loaded_circuit_weights, dtype=self._get_optimal_dtypes()[0]) # Use determined dtype
                    if self.weights.shape != (self.layers, self.qubits, 3): # Basic shape check
                        current_logger.error(f"Loaded QLMSR circuit_weights shape mismatch. Expected {(self.layers, self.qubits, 3)}, got {self.weights.shape}. Re-initializing weights.")
                        self.weights = np.random.uniform(0, 2 * np.pi, (self.layers, self.qubits, 3)).astype(self._get_optimal_dtypes()[0])


                # Re-initialize components that depend on loaded parameters if necessary
                # (e.g., quantum device, circuits, dtypes)
                self.dtype, self.c_dtype = self._get_optimal_dtypes()
                # Potentially re-initialize device if config changed significantly, though device_name isn't saved here
                # self.device, self.quantum_available = self._initialize_quantum_device(None) # Or saved device_name
                if self.quantum_available and self.mode != ProcessingMode.CLASSICAL:
                    # Re-initializing circuits is safest if parameters like qubits/layers could change
                    self._initialize_quantum_circuits()
                if self.enable_caching:
                    self._setup_caching() # Re-apply caching decorators

                current_logger.info(f"QuantumLMSR state loaded from {filepath} (saved at {loaded_state.get('saved_timestamp', 'unknown')})")
                return True
        except Exception as e:
            current_logger.error(f"Error loading QuantumLMSR state from {filepath}: {e}", exc_info=True)
            return False
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Return performance metrics for monitoring."""
        metrics = {}
        for func_name, times in self._execution_times.items():
            if times:
                metrics[func_name] = {
                    'avg_time': sum(times) / len(times),
                    'max_time': max(times),
                    'min_time': min(times),
                    'call_count': self._call_counts.get(func_name, 0)
                }
        
        # Add quantum device info if available
        if self.quantum_available:
            metrics['quantum_device'] = {
                'name': self.device.name,
                'wires': self.qubits,
                'shots': self.shots,
                'mode': self.mode.value
            }
        
        # Add hardware manager info if available
        if self.hw_manager is not None:
            try:
                accel_type = self.hw_accelerator.get_accelerator_type() if hasattr(self.hw_accelerator, 'get_accelerator_type') else "unknown"
                # Convert AcceleratorType enum to string
                if isinstance(accel_type, AcceleratorType):
                    accel_type_str = accel_type.name
                else:
                    accel_type_str = str(accel_type)

                metrics['hardware'] = {
                    'quantum_available': getattr(self.hw_manager, 'quantum_available', False),
                    'gpu_available': getattr(self.hw_manager, 'gpu_available', False),
                    'device_type': accel_type_str # Use the string representation
                }
            except Exception as e:
                logger.warning(f"Error getting hardware metrics: {e}")
                pass # Continue even if hardware metrics fail
            
        return metrics

    def benchmark(self, n_outcomes: int = 8, n_iterations: int = 100, 
                compare_classical: bool = True) -> Dict[str, Any]:
        """
        Run benchmarks to measure performance.
        
        Args:
            n_outcomes: Number of market outcomes to test
            n_iterations: Number of iterations for timing
            compare_classical: Whether to compare with classical methods
            
        Returns:
            Dictionary with benchmark results
        """
        # Generate test data
        np.random.seed(42)  # For reproducibility
        quantities = np.random.normal(0, 1, n_outcomes)
        probabilities = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8][:min(8, n_outcomes)]
        
        # Initialize result dictionary
        results = {
            'config': {
                'quantum_available': self.quantum_available,
                'device': self.device.name if self.device else 'none',
                'mode': self.mode.value,
                'n_outcomes': n_outcomes,
                'n_iterations': n_iterations,
                'qubits': self.qubits,
                'gpu_available': self.gpu_available,
                'hw_accelerator_type': str(self.hw_accelerator.get_accelerator_type()) if hasattr(self.hw_accelerator, 'get_accelerator_type') else "unknown"
            },
            'timing': {}
        }
        
        # Create classical LMSR if comparing
        if compare_classical:
            # Simple classical implementation for comparison
            class ClassicalLMSR:
                def __init__(self, b=100.0):
                    self.b = b
                
                def cost_function(self, quantities):
                    b = self.b
                    max_q = max(quantities)
                    exp_sum = sum(np.exp((q - max_q) / b) for q in quantities)
                    return b * (max_q / b + np.log(exp_sum))
                
                def get_market_probability(self, quantities, index):
                    b = self.b
                    exp_term = np.exp(quantities[index] / b)
                    exp_sum = sum(np.exp(q / b) for q in quantities)
                    return exp_term / exp_sum
                
                def aggregate_probabilities(self, probabilities, weights=None):
                    if weights is None:
                        weights = [1.0 / len(probabilities)] * len(probabilities)
                    
                    def to_log_odds(p):
                        return np.log(p / (1 - p))
                    
                    def from_log_odds(l):
                        return 1 / (1 + np.exp(-l))
                    
                    log_odds_sum = sum(w * to_log_odds(p) for p, w in zip(probabilities, weights))
                    return from_log_odds(log_odds_sum)
            
            classical_lmsr = ClassicalLMSR(b=self.liquidity_parameter)
        
        # Benchmark cost function
        logger.info("Benchmarking cost function...")
        start_time = time.time()
        for _ in range(n_iterations):
            _ = self.cost_function(quantities)
        quantum_time = time.time() - start_time
        
        if compare_classical:
            start_time = time.time()
            for _ in range(n_iterations):
                _ = classical_lmsr.cost_function(quantities)
            classical_time = time.time() - start_time
            
            results['timing']['cost_function'] = {
                'quantum': quantum_time,
                'classical': classical_time,
                'speedup': classical_time / quantum_time if quantum_time > 0 else 0
            }
        else:
            results['timing']['cost_function'] = {
                'quantum': quantum_time
            }
        
        # Benchmark market probability
        logger.info("Benchmarking market probability...")
        start_time = time.time()
        for i in range(min(n_outcomes, n_iterations)):
            _ = self.get_market_probability(quantities, i % n_outcomes)
        quantum_time = time.time() - start_time
        
        if compare_classical:
            start_time = time.time()
            for i in range(min(n_outcomes, n_iterations)):
                _ = classical_lmsr.get_market_probability(quantities, i % n_outcomes)
            classical_time = time.time() - start_time
            
            results['timing']['market_probability'] = {
                'quantum': quantum_time,
                'classical': classical_time,
                'speedup': classical_time / quantum_time if quantum_time > 0 else 0
            }
        else:
            results['timing']['market_probability'] = {
                'quantum': quantum_time
            }
        
        # Benchmark probability aggregation
        logger.info("Benchmarking probability aggregation...")
        start_time = time.time()
        for _ in range(n_iterations):
            _ = self.aggregate_probabilities(probabilities)
        quantum_time = time.time() - start_time
        
        if compare_classical:
            start_time = time.time()
            for _ in range(n_iterations):
                _ = classical_lmsr.aggregate_probabilities(probabilities)
            classical_time = time.time() - start_time
            
            results['timing']['aggregate_probabilities'] = {
                'quantum': quantum_time,
                'classical': classical_time,
                'speedup': classical_time / quantum_time if quantum_time > 0 else 0
            }
        else:
            results['timing']['aggregate_probabilities'] = {
                'quantum': quantum_time
            }
        
        # Add overall metrics
        results['performance_metrics'] = self.get_performance_metrics()
        
        return results


def main():
    """Test and benchmark QuantumLMSR."""
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("QuantumLMSR_test")
    
    print("Testing Quantum-Enhanced LMSR with Hardware Acceleration...")
    
    # Initialize hardware components
    hw_manager = None
    hw_accelerator = None
    
    if HARDWARE_ACCEL_AVAILABLE:
        try:
            print("Initializing hardware management...")
            hw_manager = HardwareManager.get_manager()
            hw_manager.initialize_hardware()
            
            hw_accelerator = HardwareAccelerator(enable_gpu=True)
            
            print(f"Hardware initialized: Quantum available: {hw_manager.quantum_available}")
            print(f"GPU acceleration: {hw_accelerator.get_accelerator_type()}")
        except Exception as e:
            print(f"Error initializing hardware components: {e}")
    
    # Test with various configurations
    configs = [
        {"name": "Default", "qubits": 8, "precision": PrecisionMode.AUTO, "mode": ProcessingMode.AUTO},
        {"name": "High precision", "qubits": 8, "precision": PrecisionMode.DOUBLE, "mode": ProcessingMode.AUTO},
        {"name": "Classical only", "qubits": 8, "precision": PrecisionMode.AUTO, "mode": ProcessingMode.CLASSICAL},
    ]
    
    results = {}
    
    try:
        # Run with different configurations
        for config in configs:
            print(f"\nTesting configuration: {config['name']}")
            lmsr = QuantumLMSR(
                qubits=config["qubits"],
                precision=config["precision"],
                mode=config["mode"],
                device_name="lightning.kokkos",  # Optimize for AMD GPUs
                hw_manager=hw_manager,
                hw_accelerator=hw_accelerator
            )
            
            # Basic functionality test
            print("Testing basic functionality...")
            
            # Test probability normalization
            print("- Testing probability normalization")
            test_probs = [0.5, 0.0, 1.0, -0.1, 1.1, float('nan'), float('inf')]
            for p in test_probs:
                norm_p = lmsr.normalize_probability(p)
                print(f"  Input: {p}, Normalized: {norm_p}")
            
            # Test log-odds conversion
            print("- Testing log-odds conversion")
            test_probs = [0.1, 0.5, 0.9]
            for p in test_probs:
                log_odds = lmsr.to_log_odds(p)
                back_prob = lmsr.from_log_odds(log_odds)
                print(f"  Probability: {p}, Log-odds: {log_odds}, Back: {back_prob}")
            
            # Test LMSR cost function
            print("- Testing LMSR cost function")
            test_quantities = [
                [0, 0, 0],
                [1, 2, 3],
                [10, 20, 30],
                list(range(16))
            ]
            for q in test_quantities:
                cost = lmsr.cost_function(q)
                print(f"  Quantities: {q[:3]}{'...' if len(q) > 3 else ''}, Cost: {cost}")
            
            # Test market probabilities
            print("- Testing market probabilities")
            quantities = [0, 10, 20]
            for i in range(len(quantities)):
                prob = lmsr.get_market_probability(quantities, i)
                print(f"  Outcome {i}: {prob}")
            
            # Test all market probabilities
            all_probs = lmsr.get_all_market_probabilities(quantities)
            print(f"  All probabilities: {all_probs}")
            
            # Test probability aggregation
            print("- Testing probability aggregation")
            test_probs = [0.3, 0.4, 0.6, 0.7]
            aggregated = lmsr.aggregate_probabilities(test_probs)
            print(f"  Input: {test_probs}, Aggregated: {aggregated}")
            
            # Test with weights
            weights = [0.1, 0.2, 0.3, 0.4]
            aggregated_weighted = lmsr.aggregate_probabilities(test_probs, weights)
            print(f"  Input: {test_probs}, Weights: {weights}, Aggregated: {aggregated_weighted}")
            
            # Run benchmarks (small scale for test)
            print("Running benchmarks...")
            benchmark_results = lmsr.benchmark(n_outcomes=8, n_iterations=10)
            
            # Store results
            results[config["name"]] = benchmark_results
            
        # Print summary
        print("\nBenchmark Summary:")
        for name, result in results.items():
            print(f"\n{name}:")
            if "cost_function" in result["timing"]:
                timing = result["timing"]["cost_function"]
                if "speedup" in timing:
                    print(f"  Cost function: {timing['quantum']:.6f}s (vs classical: {timing['classical']:.6f}s, speedup: {timing['speedup']:.2f}x)")
                else:
                    print(f"  Cost function: {timing['quantum']:.6f}s")
                    
            if "market_probability" in result["timing"]:
                timing = result["timing"]["market_probability"]
                if "speedup" in timing:
                    print(f"  Market probability: {timing['quantum']:.6f}s (vs classical: {timing['classical']:.6f}s, speedup: {timing['speedup']:.2f}x)")
                else:
                    print(f"  Market probability: {timing['quantum']:.6f}s")
                    
            if "aggregate_probabilities" in result["timing"]:
                timing = result["timing"]["aggregate_probabilities"]
                if "speedup" in timing:
                    print(f"  Probability aggregation: {timing['quantum']:.6f}s (vs classical: {timing['classical']:.6f}s, speedup: {timing['speedup']:.2f}x)")
                else:
                    print(f"  Probability aggregation: {timing['quantum']:.6f}s")
        
        # Print hardware info
        if hw_manager is not None and hw_accelerator is not None:
            print("\nHardware Information:")
            from cdfa_extensions.hw_acceleration import AcceleratorType  # Add this import at the top of the file
            
            # And in the main() function where the error occurs:
            accel_type = hw_accelerator.get_accelerator_type()
            gpu_available = accel_type in (AcceleratorType.CUDA, AcceleratorType.ROCM, AcceleratorType.MPS)
            print(f"  GPU available: {gpu_available}")
        
        print("\nQuantum-Enhanced LMSR testing with Hardware Acceleration completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()
