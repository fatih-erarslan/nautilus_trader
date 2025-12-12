#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 27 14:25:23 2025

@author: ashina
"""
import os
import sys
import gc
import logging
import numpy as np
import pandas as pd
import time
import json
import config
import threading
import math
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
import uuid
import functools
import logging

# Custom JSON encoder for quantum types
class QuantumJSONEncoder(json.JSONEncoder):
    """
    Enhanced JSON encoder for quantum computing objects.
    
    Handles serialization of various quantum types including PennyLane's Shots,
    QNodes, tensors, complex numbers, and quantum device objects.
    
    This encoder ensures that non-serializable quantum objects can be safely
    saved to JSON with appropriate representations that maintain semantic meaning.
    """
    def default(self, obj):
        # Start with basic numerical types
        if isinstance(obj, (int, float, bool, str, type(None))):
            return obj
            
        # Handle numpy numerical types with efficient conversion
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        
        # Handle numpy arrays with shape and dtype preservation
        if isinstance(obj, np.ndarray):
            if obj.size > 1000:  # For very large arrays, consider summary statistics
                return {
                    "__type__": "ndarray_summary",
                    "shape": obj.shape,
                    "dtype": str(obj.dtype),
                    "min": float(np.min(obj)) if obj.size > 0 else None,
                    "max": float(np.max(obj)) if obj.size > 0 else None,
                    "mean": float(np.mean(obj)) if obj.size > 0 else None
                }
            return {
                "__type__": "ndarray",
                "data": obj.tolist(),
                "dtype": str(obj.dtype)
            }
        
        # Handle complex numbers
        if isinstance(obj, complex):
            return {"__type__": "complex", "real": obj.real, "imag": obj.imag}
            
        # Handle datetime objects
        if isinstance(obj, datetime):
            return {"__type__": "datetime", "iso": obj.isoformat()}
        
        # Handle PennyLane Shots object with complete metadata
        if hasattr(obj, '__class__') and obj.__class__.__name__ == 'Shots':
            result = {"__type__": "pennylane.Shots"}
            for attr in ['total', 'counts', 'samples']:
                if hasattr(obj, attr):
                    try:
                        # Handle potential ndarray attributes
                        attr_value = getattr(obj, attr)
                        if isinstance(attr_value, np.ndarray):
                            result[attr] = attr_value.tolist() if attr_value.size < 1000 else f"ndarray(shape={attr_value.shape})"
                        else:
                            result[attr] = attr_value
                    except Exception:
                        result[attr] = None
            return result
        
        # Handle PennyLane QNode objects
        if hasattr(obj, '__class__') and hasattr(obj, 'qtape') and hasattr(obj, 'interface'):
            return {
                "__type__": "pennylane.QNode",
                "interface": obj.interface if hasattr(obj, 'interface') else None,
                "device": str(obj.device) if hasattr(obj, 'device') else None,
                "func_name": obj.func.__name__ if hasattr(obj, 'func') and hasattr(obj.func, '__name__') else None
            }
        
        # Handle other PennyLane objects
        if hasattr(obj, '__class__') and 'pennylane' in str(obj.__class__.__module__):
            # Extract common properties for quantum objects
            result = {
                "__type__": f"pennylane.{obj.__class__.__name__}",
                "module": obj.__class__.__module__
            }
            
            # Add common quantum object attributes if they exist
            for attr in ['name', 'wires', 'num_wires', 'shots', 'interface']:
                if hasattr(obj, attr):
                    try:
                        attr_value = getattr(obj, attr)
                        result[attr] = self.default(attr_value)  # Handle nested objects
                    except Exception:
                        result[attr] = str(attr_value)
            
            # Add params for operations
            if hasattr(obj, 'parameters') and callable(getattr(obj, 'parameters', None)):
                try:
                    params = obj.parameters()
                    result["parameters"] = [self.default(p) for p in params] if params else []
                except Exception:
                    pass
                    
            return result
        
        # Handle enum values
        if isinstance(obj, Enum):
            return {"__type__": "enum", "class": obj.__class__.__name__, "value": obj.value, "name": obj.name}
        
        # Handle sets
        if isinstance(obj, set):
            return {"__type__": "set", "items": list(obj)}
        
        # Handle dataclasses
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return {"__type__": "dataclass", "class": obj.__class__.__name__, **dataclasses.asdict(obj)}
        
        # Handle common collections
        if isinstance(obj, (list, tuple)):
            return [self.default(item) for item in obj]
            
        if isinstance(obj, dict):
            return {str(k): self.default(v) for k, v in obj.items()}
                
        # Failsafe for any other objects
        try:
            # Try the default encoder first
            return super().default(obj)
        except TypeError:
            # If that fails, try to convert to dict if object has __dict__
            if hasattr(obj, '__dict__'):
                try:
                    obj_dict = {"__type__": obj.__class__.__name__}
                    obj_dict.update({k: self.default(v) for k, v in obj.__dict__.items() 
                                   if not k.startswith('_') and not callable(v)})
                    return obj_dict
                except:
                    pass
            
            # Final fallback: convert to string but with class info
            if hasattr(obj, '__class__'):
                return f"{obj.__class__.__name__}({str(obj)})"
            return str(obj)

from numba import njit, prange, vectorize, float64
from concurrent.futures import ThreadPoolExecutor
from enhanced_lmsr import LogarithmicMarketScoringRule, LMSRConfig, ProbabilityConversionMethod, AggregationMethod
from quantum_lmsr import QuantumLMSR, ProcessingMode, PrecisionMode
from quantum_hedge import QuantumHedgeAlgorithm
from cdfa_extensions.hw_acceleration import _apply_shots_to_qnode


# PyTorch for GPU acceleration
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not available. GPU acceleration will be limited.")

# Hardware management
try:
    from hardware_manager import HardwareManager
    HARDWARE_MANAGER_AVAILABLE = True
except ImportError:
    HARDWARE_MANAGER_AVAILABLE = False
    logging.warning("Hardware Manager is not available")

# Hardware acceleration
try:
    from cdfa_extensions.hw_acceleration import HardwareAccelerator, AcceleratorType, MemoryMode
    HARDWARE_ACCEL_AVAILABLE = True
except ImportError:
    HARDWARE_ACCEL_AVAILABLE = False
    logging.warning("Hardware Acceleration is not available.")

# PennyLane for quantum computing
try:
    import pennylane as qml
    from pennylane import numpy as qnp
    QUANTUM_AVAILABLE = True
except ImportError:
    qml = None
    qnp = np  # Fallback to standard numpy
    QUANTUM_AVAILABLE = False
    logging.warning("PennyLane not found. QAR quantum path disabled.")

try:
    from quantum_prospect_theory import QuantumProspectTheory, ProcessingMode
    QUANTUM_PT_AVAILABLE = True
except ImportError:
    QUANTUM_PT_AVAILABLE = False


# Logger setup
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("qar.log"), logging.StreamHandler()],
)
logger = logging.getLogger("Quantum Agentic Reasoning")

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
            cls.TREND.value: 0.60,         # Higher weight for trend
            cls.VOLATILITY.value: 0.50,    # Medium-high weight for volatility
            cls.MOMENTUM.value: 0.55,      # Medium-high weight for momentum
            cls.SENTIMENT.value: 0.45,     # Medium weight for sentiment
            cls.LIQUIDITY.value: 0.35,     # Lower weight for liquidity
            cls.CORRELATION.value: 0.40,   # Medium-low weight for correlation
            cls.CYCLE.value: 0.50,         # Medium-high weight for cycle
            cls.ANOMALY.value: 0.30        # Lower weight for anomaly (rare events)
        }
    
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
            cls.TREND.value: 0.60,         # Higher weight for trend
            cls.VOLATILITY.value: 0.50,    # Medium-high weight for volatility
            cls.MOMENTUM.value: 0.55,      # Medium-high weight for momentum
            cls.SENTIMENT.value: 0.45,     # Medium weight for sentiment
            cls.LIQUIDITY.value: 0.35,     # Lower weight for liquidity
            cls.CORRELATION.value: 0.40,   # Medium-low weight for correlation
            cls.CYCLE.value: 0.50,         # Medium-high weight for cycle
            cls.ANOMALY.value: 0.30        # Lower weight for anomaly (rare events)
        }
    
    @classmethod
    def validate_factor_name(cls, factor_name: str) -> bool:
        """Check if a factor name is in the standard set"""
        return factor_name in cls.get_ordered_list()

class DecisionType(Enum):
    """Trading decision types"""
    BUY = auto()
    SELL = auto()
    HOLD = auto()
    EXIT = auto()
    HEDGE = auto()
    INCREASE = auto()
    DECREASE = auto()

try:
    # Assumes panarchy_analyzer.py is importable from qar.py's location
    from cdfa_extensions.analyzers.panarchy_analyzer import MarketPhase
    logging.debug(f"QAR.PY: Successfully imported MarketPhase: {MarketPhase}")
except ImportError:
    logging.error("CRITICAL: Failed to import MarketPhase from panarchy_analyzer. Using fallback enum (No UNKNOWN phase).")
    class MarketPhase(Enum):
         GROWTH="growth"
         CONSERVATION="conservation"
         RELEASE="release"
         REORGANIZATION="reorganization"
         UNKNOWN="unknown"

         @classmethod
         def from_string(cls, phase_str: str):
             phase_str = str(phase_str).lower()
             for phase in cls:
                 if phase.value == phase_str: return phase
             logging.warning(f"Invalid phase string '{phase_str}' received in fallback enum. Defaulting to UNKNOWN.")
             return cls.UNKNOWN
         
#class MarketPhase(Enum):
    """Market phases based on Panarchy theory"""
#    GROWTH = "growth"
#    CONSERVATION = "conservation"
#    RELEASE = "release"
#    REORGANIZATION = "reorganization"
#    UNKNOWN = "unknown"

#    @classmethod
#    def from_string(cls, phase_str: str):
#        phase_str = str(phase_str).lower()
#        for phase in cls:
#            if phase.value == phase_str:
#                return phase
#        return cls.UNKNOWN

@dataclass
class TradingDecision:
    """Trading decision data structure"""
    decision_type: DecisionType
    confidence: float
    reasoning: str
    timestamp: Optional[datetime] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid.uuid4().hex)

class CircuitCache:
    """Cache for quantum circuits"""
    def __init__(self, max_size: int = 100):
        self.cache = {}
        self.max_size = max_size
        self.hit_count = 0
        self.miss_count = 0
        self._lock = threading.RLock()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key in self.cache:
                self.hit_count += 1
                self.cache[key]["last_access"] = time.time()
                return self.cache[key]["circuit"]
            self.miss_count += 1
            return None

    def put(self, key: str, circuit: Any) -> None:
        with self._lock:
            if len(self.cache) >= self.max_size:
                oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k]["last_access"])
                del self.cache[oldest_key]
            self.cache[key] = {"circuit": circuit, "last_access": time.time()}

    def clear(self) -> None:
        with self._lock:
            self.cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "hit_count": self.hit_count,
                "miss_count": self.miss_count,
                "hit_ratio": self.hit_count / (self.hit_count + self.miss_count)
                if (self.hit_count + self.miss_count) > 0 else 0,
            }

@njit(fastmath=True)
def _calculate_factor_influence_njit(factor_values, weights):
    """Numba-accelerated calculation of factor influence."""
    influences = np.zeros(len(factor_values))
    for i in range(len(factor_values)):
        value = factor_values[i]
        weight = weights[i]
        # Influence = weight * deviation from neutral
        influences[i] = weight * abs(value - 0.5) * 2
    return influences


# Constants for decision making
MAX_ABS_FRAMING_EFFECT = 0.5  # Maximum absolute value for framing effect

def quantum_accelerated(use_hw_accel=True, hw_batch_size=None, device_shots=None):
    """Decorator for hardware-accelerated quantum functions."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            # Check if hardware acceleration is available and requested
            accelerate = getattr(self, 'hw_accelerator', None) is not None and use_hw_accel
            
            # If acceleration is requested but not available, log a warning
            if use_hw_accel and not accelerate:
                self.logger.debug(f"Hardware acceleration requested for {func.__name__} but not available")
            
            # Ensure memory pattern storage is initialized
            if not hasattr(self, 'memory_patterns'):
                self.memory_patterns = []
            
            if not hasattr(self, 'memory_metadata'):
                self.memory_metadata = []
                
            # Apply hardware acceleration settings if available
            if accelerate:
                original_batch_size = None
                
                # Store original settings
                if hasattr(self.device, 'batch_size'):
                    original_batch_size = self.device.batch_size
                    if hw_batch_size is not None:
                        self.device.batch_size = hw_batch_size
                
                # PennyLane no longer allows setting shots directly on device instances
                # Instead, store the shots value as an attribute to be used in QNode calls
                if device_shots is not None:
                    self._saved_device_shots = device_shots
                    self.logger.debug(f"Stored shots configuration: {device_shots}")
                
                self.logger.debug(f"Running {func.__name__} with hardware acceleration")
                
                try:
                    # Run the function with acceleration
                    result = func(self, *args, **kwargs)
                    
                    # Restore original settings
                    if original_batch_size is not None:
                        self.device.batch_size = original_batch_size
                    
                    # Clean up our stored shots configuration
                    if hasattr(self, '_saved_device_shots'):
                        self.logger.debug(f"Cleaning up stored shots configuration")
                        delattr(self, '_saved_device_shots')
                        
                    return result
                except Exception as e:
                    self.logger.error(f"Hardware acceleration failed for {func.__name__}: {e}")
                    # Restore original settings on error
                    if original_batch_size is not None:
                        self.device.batch_size = original_batch_size
                    
                    # Clean up our stored shots configuration on error
                    if hasattr(self, '_saved_device_shots'):
                        self.logger.debug(f"Cleaning up stored shots configuration after error")
                        delattr(self, '_saved_device_shots')
                    
                    # Fall back to standard execution
                    return func(self, *args, **kwargs)
            else:
                # Run without acceleration
                return func(self, *args, **kwargs)
        return wrapper
    return decorator

class QuantumAgenticReasoning:
    """
    Enterprise-grade Quantum Agentic Reasoning system for trading.

    Implements a decision-making framework that combines multiple factors, market states,
    and quantum processing to generate optimal trading actions while adapting to
    evolving market conditions.
    """

    def __init__(
        self,
        hardware_manager = None,
        hw_accelerator = None,
        memory_length: int = 50,
        decision_threshold: float = 0.3,
        num_factors: int = len(StandardFactors.get_ordered_list()),  # Must match standard factors
        quantum_fallback_threshold: int = 3,
        cache_size: int = 100,
        min_probability: float = 0.001,
        max_probability: float = 0.999,
        log_level: Union[int, str] = logging.INFO,
        use_classical: bool = False,
        enable_vectorization: bool = True,
        qha_feature_dim: int = len(StandardFactors.get_ordered_list()),  # Must match standard factors
        qha_num_experts: Optional[int] = None,  # Will be set to len(StandardFactors.get_ordered_list())
        qha_learning_rate: float = 0.01,
        qha_quantum_enhancement: float = 0.1,
        qha_market_adaptive_learning: bool = True,
        qha_weight_decay: float = 0.01,
        qha_min_weight: float = 0.01,
        hw_manager_config: Optional[Dict] = None,
        hw_accelerator_config: Optional[Dict] = None,
        quantum_config: Optional[Dict] = None,
        use_gpu: bool = True,
        memory_mode: str = "auto",
        optimization_level: str = "balanced",
        enable_caching: bool = True,
        enable_parallel: bool = True,
        max_workers: Optional[int] = None,
        log_file: Optional[str] = None,
        state_file: Optional[str] = None,
        enable_state_persistence: bool = True,
        state_save_interval: int = 100,
        enable_resource_monitoring: bool = True,
        resource_monitor_interval: int = 60,
        enable_auto_optimization: bool = True,
        auto_optimization_interval: int = 1000,
        enable_quantum_error_mitigation: bool = True,
        quantum_error_threshold: float = 0.1,
        enable_quantum_circuit_optimization: bool = True,
        quantum_optimization_level: str = "medium",
        enable_quantum_shot_noise_mitigation: bool = True,
        quantum_shots: int = 2048,
        enable_quantum_readout_error_mitigation: bool = True,
        quantum_readout_error_threshold: float = 0.05,
    ) -> None:
        """Initialize the QAR system with standardized 8-factor configuration.
        
        Args:
            hardware_manager: Hardware manager for quantum resource allocation
            hw_accelerator: Hardware accelerator for quantum operations
            memory_length: Length of memory buffer for adaptation
            decision_threshold: Threshold for actionable decisions
            num_factors: Number of factors (standardized to 8)
            quantum_fallback_threshold: Number of quantum failures before fallback
            cache_size: Size of circuit cache
            min_probability: Minimum probability for LMSR calculations
            max_probability: Maximum probability for LMSR calculations
            log_level: Logging level
            use_classical: Force classical implementation
            enable_vectorization: Enable vectorized operations
            qha_feature_dim: QHA feature dimension (standardized to 8)
            qha_num_experts: Number of QHA experts (standardized to 8)
            qha_learning_rate: QHA learning rate
            qha_quantum_enhancement: QHA quantum enhancement factor
            qha_market_adaptive_learning: Enable market-adaptive learning
            qha_weight_decay: Weight decay factor
            qha_min_weight: Minimum weight value
        """
        # Configure logging
        self.logger = logger
        self._configure_logging(log_level)
        self._debug = (self.logger.level == logging.INFO)
        self.logger.debug(f"QAR instance _debug flag set to: {self._debug}")
        self.logger.info(f"Initializing QAR System")

        # Core parameters
        self.memory_length = memory_length
        self.decision_threshold = self._validate_threshold(decision_threshold)
        
        # Enforce 8-factor standardization
        self._validate_factor_configuration(num_factors)
        
        # Core configuration
        self.num_factors = 8  # Always use 8 factors for standardization
        self.quantum_fallback_threshold = quantum_fallback_threshold
        self.cache_size = cache_size
        self.min_probability = min_probability
        self.max_probability = max_probability

        # Initialize factors
        self.standard_factors = StandardFactors.get_ordered_list()
        if len(self.standard_factors) != 8:
            raise ValueError(f"Expected 8 standard factors, found {len(self.standard_factors)}")
        self.factors = self.standard_factors
        self.num_factors = len(self.factors)

        # Initialize QHA parameters
        self.qha_feature_dim = len(self.factors)
        self.qha_learning_rate = qha_learning_rate
        self.qha_quantum_enhancement = qha_quantum_enhancement
        self.qha_market_adaptive_learning = qha_market_adaptive_learning
        self.qha_weight_decay = qha_weight_decay
        self.qha_min_weight = qha_min_weight
        self.qha_num_experts = len(self.factors)  # Must match number of factors
        self.qha_expert_factor_order = self.factors  # Use standard factor order
        self.qha_min_weight = qha_min_weight
        
        # Standard factors that all components should use
        self.standard_factor_names = StandardFactors.get_ordered_list()

        # Thread safety
        self._lock = threading.RLock()

        # Hardware resources
        self.hardware_manager = hardware_manager
        self.use_classical = use_classical or not QUANTUM_AVAILABLE

        # Initialize standardized factors and weights
        self.factor_names = StandardFactors.get_ordered_list()
        self.factor_weights = StandardFactors.get_default_weights()
        
        # Initialize hardware accelerator with standardized configuration
        self.hw_accelerator = hw_accelerator
        if self.hw_accelerator is None and hardware_manager is None:
            try:
                # Create default hardware accelerator with auto device selection
                self.hw_accelerator = HardwareAccelerator(
                    enable_gpu=True,
                    memory_mode=MemoryMode.AUTO,
                    optimization_level="balanced",
                    log_level=log_level
                )
                self.logger.info(f"Initialized hardware accelerator with device: {self.hw_accelerator.get_device()}")
            except Exception as e:
                self.logger.warning(f"Failed to initialize hardware accelerator: {e}")
                self.hw_accelerator = None

        # Quantum configuration
        self.hw_manager: Optional[HardwareManager] = None
        self.hw_accelerator: Optional[HardwareAccelerator] = None
        self.device: Optional[Any] = None # qml.Device
        self.device_name: str = "unknown"
        self.max_qubits = 8  # Force to 8 for standardized factors
        self.quantum_device_shots = None
        self.quantum_available = not self.use_classical and QUANTUM_AVAILABLE
        self.quantum_failure_count = 0

        # Memory and state
        self.factors = StandardFactors.get_ordered_list()  # Initialize with standard factors
        self.factor_importance = {factor: 1.0 for factor in self.factors}  # Equal initial importance
        
        # Initialize all weight dictionaries with standard weights
        standard_weights = StandardFactors.get_default_weights()
        self.baseline_weights = standard_weights.copy()
        self.regime_specific_weights = {
            MarketPhase.GROWTH.value: standard_weights.copy(),
            MarketPhase.CONSERVATION.value: standard_weights.copy(),
            MarketPhase.RELEASE.value: standard_weights.copy(),
            MarketPhase.REORGANIZATION.value: standard_weights.copy()
        }
        
        # Initialize tracking lists
        self.decision_history = []
        self.memory_buffer = []
        self.successful_decisions = []
        self.failed_decisions = []

        # Performance metrics
        self.circuit_cache = CircuitCache(max_size=cache_size)
        self.execution_times = {"quantum": [], "classical": []}
        self.performance_metrics = {}
        self.cumulative_performance = 0.0

        self._initialize_quantum_pt() 
        # Initialize Quantum Hedge Algorithm
        # Initialize QHA expert mapping with standard factors
        self.qha_expert_factor_order = StandardFactors.get_ordered_list()  # Use standard factor order
        self.qha_feature_dim = 8  # Force to 8 for standardization
        self.qha_context_for_feedback: Dict[str, Dict[str, Any]] = {}  # Cache for feedback

        # Initialize quantum components to None
        self.hedge_algorithm = None
        self.quantum_lmsr = None
        self.quantum_pt = None
        
        # Initialize hardware and quantum components
        if self.hardware_manager:
            self._initialize_hardware()
        
        # Initialize quantum components with 8-factor standardization
        try:
            self.initialize_quantum_components()
        except Exception as e:
            self.logger.error(f"Failed to initialize quantum components: {str(e)}")
            self.quantum_available = False

        # Initialize LMSR component
        lmsr_config = LMSRConfig(
            liquidity_parameter=100.0,  # Controls price sensitivity
            min_probability=self.min_probability,
            max_probability=self.max_probability,
            use_numba=True,                     # Enable Numba acceleration
            use_vectorization=enable_vectorization,  # Enable vectorized operations
            batch_size=1024,                    # Efficient batch size for large datasets
            hardware_aware_parallelism=True,    # Optimize for available hardware
            enable_parallel=True,               # Enable parallel processing
            max_workers=4,                      # Adjust based on system capabilities
            log_level=logging.INFO
        )
        self.lmsr = LogarithmicMarketScoringRule(config=lmsr_config)

        # Initialize Quantum LMSR component
        try:
            if self.quantum_available:
                self.quantum_lmsr = QuantumLMSR(
                    liquidity_parameter=100.0,
                    min_probability=self.min_probability,
                    max_probability=self.max_probability,
                    qubits=8,  # Force 8 qubits for standardization
                    shots=self.quantum_device_shots,
                    precision=PrecisionMode.AUTO,
                    use_standard_factors=True,  # Enable standardized 8-factor model
                    factor_names=StandardFactors.get_ordered_list(),
                    initial_weights=StandardFactors.get_default_weights(),
                    mode=ProcessingMode.AUTO,
                    enable_caching=True,
                    cache_size=cache_size,
                    hw_manager=self.hardware_manager,
                    hw_accelerator=self.hw_accelerator  # Use our hardware accelerator
                )
                self.logger.info(f"Quantum LMSR initialized with "
                                f"{getattr(self.quantum_lmsr, 'qubits', 0)} qubits")
            else:
                self.quantum_lmsr = None
                self.logger.info("Quantum LMSR not initialized (quantum not available)")
        except Exception as e:
            self.quantum_lmsr = None
            self.logger.error(f"Failed to initialize Quantum LMSR: {e}")

        # Track factor quantities (market state)
        self.factor_quantities = {}

        # Resource monitoring
        self._resource_monitor_active = False
        self._start_resource_monitoring()

        self.logger.info(f"QAR System initialization complete")

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

    def _validate_threshold(self, threshold: float) -> float:
        """Validate decision threshold is between 0 and 1"""
        if not 0 <= threshold <= 1:
            raise ValueError(f"Decision threshold must be between 0 and 1, got {threshold}")
        return threshold

    def _validate_factor_configuration(self, num_factors: int) -> None:
        """Validate factor configuration and ensure 8-factor standardization"""
        if num_factors != 8:
            self.logger.warning(f"Overriding num_factors={num_factors} to 8 for standardized factor model")
            
        # Verify StandardFactors enum has exactly 8 factors
        standard_factors = StandardFactors.get_ordered_list()
        if len(standard_factors) != 8:
            raise ValueError(f"StandardFactors must define exactly 8 factors, found {len(standard_factors)}")
            
        # Verify default weights exist for all factors
        default_weights = StandardFactors.get_default_weights()
        if set(default_weights.keys()) != set(standard_factors):
            raise ValueError("Default weights must be defined for all standard factors")
            
    def _validate_qha_configuration(self, feature_dim: int, learning_rate: float,
                                   quantum_enhancement: float, market_adaptive: bool,
                                   weight_decay: float, min_weight: float) -> None:
        """Validate Quantum Hedge Algorithm configuration"""
        if feature_dim != 8:
            self.logger.warning(f"Overriding QHA feature_dim={feature_dim} to 8 for standardized model")
            
        # Validate learning parameters
        if not 0 < learning_rate <= 1:
            raise ValueError(f"Learning rate must be between 0 and 1, got {learning_rate}")
            
        if not 0 <= quantum_enhancement <= 1:
            raise ValueError(f"Quantum enhancement must be between 0 and 1, got {quantum_enhancement}")
            
        if not 0 <= weight_decay <= 1:
            raise ValueError(f"Weight decay must be between 0 and 1, got {weight_decay}")
            
        if not 0 <= min_weight <= 1:
            raise ValueError(f"Minimum weight must be between 0 and 1, got {min_weight}")

    def _initialize_hardware(self) -> None:
        """Initialize quantum hardware and circuits with 8-factor standardization"""
        try:
            self.logger.debug("Initializing hardware resources")
            
            # First, determine GPU availability through multiple approaches
            self.gpu_available = False
            
            # 1. Check via PyTorch (most direct method)
            if TORCH_AVAILABLE and torch.cuda.is_available():
                self.gpu_available = True
                gpu_name = torch.cuda.get_device_name(0)
                self.logger.info(f"CUDA GPU detected via PyTorch: {gpu_name}")
            elif TORCH_AVAILABLE and hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and \
                 hasattr(torch.backends.mps, 'is_available') and torch.backends.mps.is_available():
                self.gpu_available = True
                self.logger.info("Apple MPS detected via PyTorch")
            
            # 2. Initialize hardware accelerator if available
            self.acceleration_available = False
            if HARDWARE_ACCEL_AVAILABLE:
                try:
                    # Initialize hardware accelerator with optimal settings for quantum processing
                    self.hw_accelerator = HardwareAccelerator(
                        device="auto",  # Let the system choose the best device
                        memory_mode=MemoryMode.DYNAMIC,  # Allow dynamic memory management
                        #accelerator_type=None,  # Auto-detect the accelerator type
                        #precision="float32",   # Use float32 for quantum calculations
                        enable_profiling=True  # Enable performance profiling
                    )
                    
                    # Get device info for logging
                    device_info = self.hw_accelerator.get_device_info() if hasattr(self.hw_accelerator, 'get_device_info') else {}
                    self.logger.info(f"Hardware accelerator initialized: {device_info.get('name', 'unknown')}")
                    
                    
                    self.acceleration_available = True
                    
                    # Pre-warm the accelerator
                    if hasattr(self.hw_accelerator, 'warmup'):
                        self.logger.info("Pre-warming hardware accelerator...")
                        self.hw_accelerator.warmup()
                except Exception as e:
                    self.logger.error(f"Failed to initialize hardware accelerator: {e}")
                    self.hw_accelerator = None
            else:
                self.logger.warning("Hardware acceleration not available. Performance may be impacted.")
                self.hw_accelerator = None
            
            # 3. Initialize quantum components with device selection fallbacks
            self.quantum_available = not getattr(self, 'use_classical', False) and QUANTUM_AVAILABLE
            
            if self.quantum_available:
                # Try to use device from hardware manager first if available
                if hasattr(self, 'hardware_manager') and self.hardware_manager and \
                   hasattr(self.hardware_manager, "get_pennylane_device"):
                    self.logger.info("Requesting PennyLane device from hardware manager...")
                    self.device = self.hardware_manager.get_pennylane_device()
                    if self.device:
                        # Use device name or class name as fallback
                        self.device_name = getattr(self.device, 'name', 
                                               getattr(self.device, '__class__', 'unknown').__name__)
                else:
                    # Direct device creation with fallbacks 
                    self.logger.info("Creating quantum device with hardware acceleration support...")
                    try:
                        # Try CUDA GPU first
                        self.logger.info("Attempting to create lightning.gpu device...")
                        self.device = qml.device("lightning.gpu", wires=8)  # Standardized to 8 qubits
                        self.device_name = "lightning.gpu"
                        self.logger.info("Successfully created lightning.gpu device")
                    except Exception as e:
                        self.logger.warning(f"Could not create lightning.gpu device: {e}")
                        try:
                            # Try AMD GPU via ROCm/kokkos
                            self.logger.info("Attempting to create lightning.kokkos device...")
                            self.device = qml.device("lightning.kokkos", wires=8)  # Standardized to 8 qubits
                            self.device_name = "lightning.kokkos"
                            self.logger.info("Successfully created lightning.kokkos device")
                        except Exception as e:
                            self.logger.warning(f"Could not create lightning.kokkos device: {e}")
                            try:
                                # Fall back to CPU
                                self.logger.info("Falling back to lightning.qubit device...")
                                self.device = qml.device("lightning.qubit", wires=8)  # Standardized to 8 qubits
                                self.device_name = "lightning.qubit"
                                self.logger.info("Successfully created lightning.qubit device")
                            except Exception as e:
                                self.logger.error(f"All quantum device creation attempts failed: {e}")
                                self.device = None
                                self.device_name = "none"
                                self.quantum_available = False
                
                # Force 8 qubits for standardization
                self.max_qubits = 8
                self.quantum_device_shots = getattr(self.device, 'shots', None) if self.device else None
                
                self.logger.info(f"Hardware initialized: quantum_available={self.quantum_available}, "
                                f"device={self.device_name}, "
                                f"gpu_available={self.gpu_available}, "
                                f"acceleration_available={self.acceleration_available}")
                
                # Initialize quantum circuits if quantum is available
                if self.quantum_available and self.device:
                    try:
                        self.qubits = self.max_qubits  # Already limited to 8
                        self.logger.info("Initializing quantum circuits with hardware acceleration support...")
                        self.circuits = self._initialize_circuits()
                        self.logger.info(f"Quantum circuits initialized with {self.qubits} qubits")
                        self.memory_patterns = []
                        self.memory_metadata = []
                        self.confidence_scaling = 1.2
                        self.quantum_failure_count = 0
                    except Exception as e:
                        self.logger.error(f"Circuit initialization error: {str(e)}")
                        self.quantum_available = False
                        self.quantum_failure_count += 1
            else:
                self.logger.warning("Hardware manager does not support PennyLane devices")
                self.quantum_available = False
        except Exception as e:
            self.logger.error(f"Hardware initialization error: {str(e)}")
            self.quantum_available = False
            self.quantum_failure_count += 1
            
    def _initialize_quantum_lmsr(self) -> None:
        """Initialize the Quantum LMSR component"""
        try:
            self.logger.info(f"Initializing Quantum LMSR")
            
            # Prepare QLMSR configuration
            lmsr_config = {
                "qubits": 8,  # Always use 8 qubits for standardization
                "liquidity_parameter": 100.0
            }
            
            # Add hardware manager if available
            if hasattr(self, 'hardware_manager') and self.hardware_manager:
                lmsr_config['hw_manager'] = self.hardware_manager
                
            # Add hardware accelerator if available
            if hasattr(self, 'hw_accelerator') and self.hw_accelerator:
                lmsr_config['hw_accelerator'] = self.hw_accelerator
                
            # Add device name if available
            if hasattr(self, 'device_name') and self.device_name and self.device_name != 'none':
                lmsr_config['device_name'] = self.device_name
                
            # Initialize the Quantum LMSR with hardware acceleration support
            self.quantum_lmsr = QuantumLMSR(**lmsr_config)
            
            self.logger.info(f"Quantum LMSR initialized with 8 qubits")
        except Exception as e:
            self.logger.error(f"Failed to initialize Quantum LMSR: {e}")
            self.quantum_lmsr = None

    def _initialize_quantum_hedge(self) -> None:
        """Initialize the Quantum Hedge Algorithm with 8-factor standardization.
        
        This ensures we have 9 experts total - 8 standard + 1 ensemble
        """
        try:
            self.logger.debug("Initializing Quantum Hedge Algorithm")
        
            # Set up standard factors
            num_standard_factors = len(self.standard_factors)
            standard_factors = self.standard_factors.copy()  
            standard_weights = StandardFactors.get_default_weights()
            
            # Ensure we're using 8 standard factors
            if num_standard_factors != 8:
                self.logger.warning(f"Expected 8 standard factors, found {num_standard_factors}. Using available factors.")
            
            # Configure QHA with 9 experts (8 factors + 1 ensemble)
            # We use 9 experts total (8 for individual factors + 1 for ensemble)
            qha_num_experts = num_standard_factors + 1
            self.logger.debug(f"Configuring QHA with {qha_num_experts} experts (8 standard + 1 ensemble)")
            
            # Prepare configuration with hardware acceleration if available
            qha_config = {
                "num_experts": qha_num_experts,
                "learning_rate": self.qha_learning_rate,
                "quantum_enhancement": self.qha_quantum_enhancement
            }
            
            # Add hardware manager if available
            if hasattr(self, 'hardware_manager') and self.hardware_manager:
                qha_config['hw_manager'] = self.hardware_manager
                
            # Add hardware accelerator if available
            if hasattr(self, 'hw_accelerator') and self.hw_accelerator:
                qha_config['hw_accelerator'] = self.hw_accelerator
                # Add GPU usage info via processing_mode parameter
                if hasattr(self, 'gpu_available') and self.gpu_available:
                    # Use QUANTUM mode instead of nonexistent GPU mode
                    qha_config['mode'] = ProcessingMode.QUANTUM
            
            # Initialize QHA with proper configuration
            self.hedge_algorithm = QuantumHedgeAlgorithm(**qha_config)
            
            # Log initialization
            self.logger.info(f"QHA initialized with {qha_num_experts} experts (8 standard + 1 ensemble) using standard processing")
            self.logger.info(f"  - Learning rate: {self.qha_learning_rate}")
            self.logger.info(f"  - Quantum enhancement: {self.qha_quantum_enhancement}")
            self.logger.info(f"  - Hardware acceleration: {self.gpu_available if hasattr(self, 'gpu_available') else False}")               
            # Check hardware availability
            if not HARDWARE_MANAGER_AVAILABLE:
                self.logger.warning("Hardware Manager not available. Some quantum features may be limited.")
            if not HARDWARE_ACCEL_AVAILABLE:
                self.logger.warning("Hardware Acceleration not available. Performance may be impacted.")
                    
        except Exception as e:
            self.logger.warning(f"Quantum Hedge Algorithm initialization failed, using classical fallback: {e}")
            self.hedge_algorithm = None
            
    # The simpler _initialize_quantum_pt method has been removed to avoid duplication.
    # The more comprehensive implementation at line ~1298 is now the only one.

    def _validate_quantum_components(self) -> None:
        """Validate that quantum components are properly initialized"""
        try:
            # Map component names to instances
            components = {
                'QHA': self.hedge_algorithm if hasattr(self, 'hedge_algorithm') else None,
                'QLMSR': self.quantum_lmsr if hasattr(self, 'quantum_lmsr') else None,
                'QPT': self.quantum_pt if hasattr(self, 'quantum_pt') else None
            }
            
            # Get list of available components
            available_components = [name for name, comp in components.items() if comp is not None]
            self.logger.info(f"Available quantum components: {available_components}")
            
            # Validate each component
            for name, component in components.items():
                self._validate_component_configuration(name, component)
                
            # Log hardware status
            if hasattr(self, 'hardware_manager'):
                self.logger.info(f"Hardware manager: {'Available' if self.hardware_manager else 'Not available'}")
            if hasattr(self, 'hw_accelerator'):
                self.logger.info(f"Hardware accelerator: {'Available' if self.hw_accelerator else 'Not available'}")
                
            # Check if we have at least one quantum component
            if not available_components:
                self.logger.warning("No quantum components available. System will use classical fallbacks.")
                
        except Exception as e:
            self.logger.error(f"Quantum component validation failed: {e}")
            # Continue execution even if validation fails

    # Log successful initialization of quantum components
    def _log_initialization_summary(self) -> None:
        """Log a summary of the initialized quantum components"""
        try:
            self.logger.info("Quantum components initialized successfully")
            self.logger.info(f"Using {len(self.standard_factors)} standard factors: {self.standard_factors}")
            self.logger.info(f"Initial weights: {StandardFactors.get_default_weights()}")
        except Exception as e:
            self.logger.warning(f"Failed to log initialization summary: {e}")

        except Exception as e:
            self.logger.error(f"Failed to initialize quantum components: {e}")
            raise

    # The duplicate _validate_quantum_components method is removed as it's already defined above
    def initialize_quantum_components(self) -> None:
        """Initialize all quantum components with hardware acceleration"""
        try:
            # Initialize hardware components first
            self._initialize_hardware()
            
            # Initialize quantum components
            self._initialize_quantum_hedge()
            self._initialize_quantum_lmsr()
            self._initialize_quantum_pt()
            
            # Validate the components
            self._validate_quantum_components()
            
            # Log summary
            self._log_initialization_summary()
        except Exception as e:
            self.logger.error(f"Failed to initialize quantum components: {e}")

    def _store_pattern(self, circuit_name, input_values, result, metadata=None):
        """Store quantum circuit execution pattern for later reference and optimization
        
        Args:
            circuit_name: Name of the quantum circuit
            input_values: Input values used for the circuit
            result: Result of the circuit execution
            metadata: Additional metadata to store with the pattern
        """
        if not hasattr(self, 'memory_patterns'):
            self.memory_patterns = []
            
        if not hasattr(self, 'memory_metadata'):
            self.memory_metadata = []
            
        # Create pattern entry
        pattern = {
            'circuit': circuit_name,
            'inputs': input_values,
            'result': result,
            'timestamp': datetime.now().isoformat()
        }
        
        # Create metadata entry
        meta = {
            'circuit': circuit_name,
            'timestamp': datetime.now().isoformat()
        }
        
        # Add any additional metadata
        if metadata:
            meta.update(metadata)
            
        # Store pattern and metadata
        self.memory_patterns.append(pattern)
        self.memory_metadata.append(meta)
        
        # Limit pattern memory to prevent excessive memory usage
        max_patterns = 100
        if len(self.memory_patterns) > max_patterns:
            self.memory_patterns = self.memory_patterns[-max_patterns:]
            self.memory_metadata = self.memory_metadata[-max_patterns:]
            
        return True
            
    def _validate_component_configuration(self, name, component):
        """Validate the configuration of a specific quantum component"""
        if component is None:
            self.logger.warning(f"{name} not available, will use classical fallback")
            return False
            
        # Perform basic validation based on component type
        if name == 'QHA':
            # Special handling for QHA with 9 experts (8 standard + 1 ensemble)
            if hasattr(component, 'num_experts'):
                expected_experts = len(self.standard_factors) + 1
                if component.num_experts != expected_experts:
                    self.logger.warning(f"QHA num_experts ({component.num_experts}) mismatch with expected count ({expected_experts}). QHA state might not load correctly.")
                    
            if hasattr(component, 'feature_dim') and component.feature_dim != 8:
                self.logger.warning(f"QHA feature_dim mismatch: expected 8, got {component.feature_dim}")
        else:
            # For QLMSR and QPT, verify standard 8-factor alignment
            if hasattr(component, 'qubits') and component.qubits != 8:
                self.logger.warning(f"{name} qubits mismatch: expected 8, got {component.qubits}")
        
        self.logger.info(f"{name} validated successfully")
        return True


    def _initialize_circuits(self) -> Dict[str, Callable]:
        """Initialize quantum circuits for QAR."""
        # Initialize circuit cache
        self.circuit_cache = {}
        
        # Quantum Fourier Transform circuit
        @qml.qnode(self.device, interface="autograd")
        def qft_circuit(factors):
            # Encode factors into quantum state
            for i in range(self.qubits):
                qml.RY(np.pi * factors[i], wires=i)

            # Apply QFT
            qml.templates.QFT(wires=range(self.qubits))

            # Measure in computational basis
            return [qml.expval(qml.PauliZ(i)) for i in range(self.qubits)]

        # Dynamic decision optimization circuit - follows the example pattern
        def decision_optimization_factory(num_factor_qubits, num_decision_qubits):
            num_qubits = num_factor_qubits + num_decision_qubits
            
            @qml.qnode(self.device, interface="autograd")
            def decision_circuit_dynamic(ry_angle_params, cry_angle_params, mkt_bias):
                # Input layer using angles derived from raw probabilities
                for i in range(num_factor_qubits):
                    qml.RY(ry_angle_params[i], wires=i)  # Angle based on factor value

                # Entanglement layer using angles derived from weighted probabilities
                for i in range(num_factor_qubits):
                    for j in range(num_decision_qubits):
                        # Use the pre-calculated CRY angle (prob * weight * pi)
                        qml.CRY(cry_angle_params[i], wires=[i, num_factor_qubits + j])

                # Decision layer with market bias
                for i in range(num_decision_qubits):
                    qml.Hadamard(wires=num_factor_qubits + i)
                    qml.RZ(mkt_bias * qnp.pi / 2, wires=num_factor_qubits + i)  # Bias via RZ

                # Entanglement in decision layer
                for i in range(num_decision_qubits - 1):
                    qml.CNOT(wires=[num_factor_qubits + i, num_factor_qubits + i + 1])
                if num_decision_qubits > 0:  # Avoid CNOT on wire 0 if only 1 decision qubit
                    qml.CNOT(wires=[num_factor_qubits + (num_decision_qubits - 1), num_factor_qubits])  # Circular CNOT

                # Measure decision qubits
                return [qml.expval(qml.PauliZ(k)) for k in range(num_factor_qubits, num_qubits)]
            
            return decision_circuit_dynamic
        
        # Pattern recognition circuit with enhanced caching
        def pattern_recognition_factory(num_pattern_qubits, num_memory_qubits):
            @qml.qnode(self.device, interface="autograd")
            def pattern_recognition_dynamic(pattern, memory_patterns, similarity_thresholds):
                # Encode input pattern
                for i in range(num_pattern_qubits):
                    qml.RY(np.pi * pattern[i], wires=i)

                # Apply oracle for pattern matching with similarity thresholds
                for j, (memory_pattern, threshold) in enumerate(zip(memory_patterns, similarity_thresholds)):
                    if j >= num_memory_qubits:  # Respect memory qubit limit
                        break

                    # Apply controlled operations based on similarity and threshold
                    similarity = np.dot(pattern, memory_pattern)
                    if similarity > threshold:
                        # Stronger entanglement for higher similarity
                        qml.CRZ(similarity * np.pi, wires=[j % num_pattern_qubits, (j + 1) % num_pattern_qubits])

                # Apply quantum interference layer
                for i in range(num_pattern_qubits):
                    qml.Hadamard(wires=i)

                # Measure pattern detection results
                return [qml.expval(qml.PauliZ(i)) for i in range(num_pattern_qubits)]
            
            return pattern_recognition_dynamic

        # Create base circuits
        decision_optimization_base = decision_optimization_factory(
            num_factor_qubits=min(5, self.qubits // 2),
            num_decision_qubits=min(3, self.qubits - (self.qubits // 2))
        )
        
        pattern_recognition_base = pattern_recognition_factory(
            num_pattern_qubits=self.qubits,
            num_memory_qubits=min(3, self.qubits)
        )

        # Wrap base circuits with caching mechanism
        def decision_optimization_circuit(factors, weights):
            # Calculate ry_angles from factors
            ry_angles = np.array([np.pi * f for f in factors[:min(5, self.qubits // 2)]])
            
            # Calculate cry_angles from factors and weights
            cry_angles = np.array([
                factors[i] * weights[i] * np.pi 
                for i in range(min(5, self.qubits // 2))
            ])
            
            # Calculate market bias from factors
            market_bias = np.mean(factors) - 0.5  # -0.5 to 0.5 range
            
            # Generate cache key
            cache_key = f"decision_{hash(str(ry_angles))}{hash(str(cry_angles))}{hash(str(market_bias))}"
            
            # Check cache
            if cache_key in self.circuit_cache:
                return self.circuit_cache[cache_key]
            
            # Execute circuit
            try:
                result = decision_optimization_base(ry_angles, cry_angles, market_bias)
                self.circuit_cache[cache_key] = result
                
                # Store pattern for future optimization
                metadata = {
                    'factors': factors.tolist() if isinstance(factors, np.ndarray) else factors,
                    'weights': weights.tolist() if isinstance(weights, np.ndarray) else weights,
                    'market_bias': float(market_bias)
                }
                self._store_pattern('decision_optimization', {'ry_angles': ry_angles, 'cry_angles': cry_angles, 'market_bias': market_bias}, result, metadata)
                
                return result
            except Exception as e:
                self.logger.error(f"Decision circuit execution failed: {e}")
                # Fallback to classical computation
                return np.array([np.dot(factors, weights[:len(factors)]), np.mean(factors), np.std(factors)])
        
        def pattern_recognition_circuit(pattern, memory_patterns):
            # Prepare similarity thresholds based on memory patterns
            similarity_thresholds = np.array([0.6] * len(memory_patterns) if isinstance(memory_patterns, list) else [0.6])
            
            # Extract numerical values if patterns are dictionaries
            pattern_numeric = pattern
            if isinstance(pattern, dict) and 'result' in pattern:
                pattern_numeric = pattern['result'] if isinstance(pattern['result'], (list, np.ndarray, float, int)) else np.array([0.5])
            elif isinstance(pattern, dict):
                pattern_numeric = np.array([0.5])  # Default fallback
            
            # Process memory patterns - ensure they're in numeric format
            memory_patterns_numeric = []
            if isinstance(memory_patterns, list):
                for mem_pattern in memory_patterns:
                    if isinstance(mem_pattern, dict) and 'result' in mem_pattern:
                        if isinstance(mem_pattern['result'], (list, np.ndarray, float, int)):
                            memory_patterns_numeric.append(mem_pattern['result'])
                        else:
                            memory_patterns_numeric.append(np.array([0.5]))  # Default fallback
                    elif isinstance(mem_pattern, (list, np.ndarray, float, int)):
                        memory_patterns_numeric.append(mem_pattern)
                    else:
                        memory_patterns_numeric.append(np.array([0.5]))  # Default fallback
            
            # Generate cache key
            cache_key = f"pattern_{hash(str(pattern_numeric))}{hash(str(memory_patterns_numeric))}"
            
            # Check cache
            if cache_key in self.circuit_cache:
                return self.circuit_cache[cache_key]
            
            # Execute circuit
            try:
                result = pattern_recognition_base(pattern_numeric, memory_patterns_numeric, similarity_thresholds)
                self.circuit_cache[cache_key] = result
                
                # Store pattern for future optimization
                metadata = {
                    'pattern_length': len(pattern_numeric) if hasattr(pattern_numeric, '__len__') else 1,
                    'memory_patterns_count': len(memory_patterns_numeric) if memory_patterns_numeric else 0,
                    'similarity_thresholds': similarity_thresholds.tolist() if isinstance(similarity_thresholds, np.ndarray) else similarity_thresholds
                }
                self._store_pattern('pattern_recognition', {'pattern': pattern, 'memory_patterns': memory_patterns}, result, metadata)
                
                return result
            except Exception as e:
                self.logger.error(f"Pattern recognition circuit execution failed: {e}")
                # Fallback to classical computation
                try:
                    if memory_patterns_numeric and len(memory_patterns_numeric) > 0:
                        # Safe dot product calculation
                        similarities = []
                        for mem_pattern in memory_patterns_numeric:
                            try:
                                if hasattr(pattern_numeric, 'shape') and hasattr(mem_pattern, 'shape'):
                                    # Handle array shape mismatches
                                    min_len = min(pattern_numeric.shape[0] if len(pattern_numeric.shape) > 0 else 1, 
                                                  mem_pattern.shape[0] if len(mem_pattern.shape) > 0 else 1)
                                    p1 = pattern_numeric[:min_len] if hasattr(pattern_numeric, '__getitem__') else pattern_numeric
                                    p2 = mem_pattern[:min_len] if hasattr(mem_pattern, '__getitem__') else mem_pattern
                                    similarities.append(float(np.dot(p1, p2)))
                                else:
                                    # Handle scalar values
                                    similarities.append(float(pattern_numeric * mem_pattern))
                            except Exception:
                                similarities.append(0.5)  # Safe default
                        return np.array(similarities)
                    return np.array([0.0])
                except Exception as e2:
                    self.logger.error(f"Fallback pattern recognition also failed: {e2}")
                    return np.array([0.5])  # Ultimate fallback value

        return {
            'qft': qft_circuit,
            'decision_optimization': decision_optimization_circuit,
            'pattern_recognition': pattern_recognition_circuit
        }


    
    @quantum_accelerated(use_hw_accel=True, hw_batch_size=4, device_shots=1024)
    def _execute_with_fallback(self, circuit_name: str, args: tuple) -> np.ndarray:
        """Execute a quantum circuit with automatic fallback to classical simulation."""
        if not self.quantum_available or not hasattr(self, 'circuits') or circuit_name not in self.circuits:
            # Fall back to classical implementation
            if circuit_name == 'qft':
                # Classical FFT as fallback for QFT
                return np.fft.fft(args[0]).real
            elif circuit_name == 'decision_optimization':
                # Simple weighted sum for decision
                factors, weights = args
                return np.array([np.dot(factors, weights[:len(factors)]), 
                                np.mean(factors), 
                                np.std(factors)])
            elif circuit_name == 'pattern_recognition':
                # Classical similarity calculation
                pattern, memory = args
                if hasattr(memory, 'size') and memory.size > 0:
                    return np.array([np.dot(pattern, mem_pattern) for mem_pattern in memory])
                elif isinstance(memory, list) and len(memory) > 0:
                    return np.array([np.dot(pattern, mem_pattern) for mem_pattern in memory])
                return np.array([0.0])
            else:
                self.logger.warning(f"No fallback for quantum circuit {circuit_name}")
                return np.zeros(self.qubits)
        
        # Generate cache key
        cache_key = f"{circuit_name}_{hash(str(args))}"
        
        # Check cache
        if hasattr(self, 'circuit_cache') and cache_key in self.circuit_cache:
            self.logger.debug(f"Circuit cache hit for {circuit_name}")
            return self.circuit_cache[cache_key]
                
        try:
            # Execute the quantum circuit
            result = self.circuits[circuit_name](*args)
            
            # Cache the result
            if hasattr(self, 'circuit_cache'):
                self.circuit_cache[cache_key] = result
                
            return result
        except Exception as e:
            self.logger.error(f"Quantum circuit {circuit_name} failed: {str(e)}", exc_info=True)
            self.quantum_failure_count += 1
            
            # Fall back to classical implementation (same as above)
            if circuit_name == 'qft':
                return np.fft.fft(args[0]).real
            elif circuit_name == 'decision_optimization':
                factors, weights = args
                return np.array([np.dot(factors, weights[:len(factors)]), 
                                np.mean(factors), 
                                np.std(factors)])
            elif circuit_name == 'pattern_recognition':
                pattern, memory = args
                if hasattr(memory, 'size') and memory.size > 0:
                    return np.array([np.dot(pattern, mem_pattern) for mem_pattern in memory])
                elif isinstance(memory, list) and len(memory) > 0:
                    return np.array([np.dot(pattern, mem_pattern) for mem_pattern in memory])
                return np.array([0.0])
            else:
                return np.zeros(self.qubits)

    def _initialize_quantum_pt(self, config_dict: Dict[str, Any] = None):
        """
        Initialize Quantum Prospect Theory component if available.

        Args:
            config_dict: Configuration dictionary
        """
        global QUANTUM_PT_AVAILABLE # This global is defined at the top of your qar.py

        # <<< VERBOSE DEBUGGING INSERTION >>>
        self.logger.info("--- ENTERING _initialize_quantum_pt ---")
        self.logger.debug(f"  Global QUANTUM_PT_AVAILABLE = {QUANTUM_PT_AVAILABLE}")
        # Check if self.use_classical is set, otherwise assume False for this check's purpose
        use_classical_check = getattr(self, 'use_classical', False)
        self.logger.debug(f"  _VERBOSE_INIT_QPT: self.use_classical = {use_classical_check}")
        # <<< END VERBOSE DEBUGGING INSERTION >>>

        # Your existing conditional return
        if not QUANTUM_PT_AVAILABLE or use_classical_check: # Use the checked value
            self.logger.info(f"_initialize_quantum_pt: Conditions not met for QPT init. QUANTUM_PT_AVAILABLE={QUANTUM_PT_AVAILABLE}, use_classical={use_classical_check}. Setting self.quantum_pt to None.")
            self.quantum_pt = None
            return

        try:
            self.logger.info("_initialize_quantum_pt: Conditions met, attempting QuantumProspectTheory instantiation...")
            config_dict = config_dict or {}
            pt_config = config_dict.get('quantum_pt', {})
            self.logger.debug(f"  _VERBOSE_INIT_QPT: pt_config from config_dict: {pt_config}")

            # Set up proper hardware acceleration
            # --- Determine which hw_manager and hw_accelerator to pass to QPT ---
            # Prefer QAR's existing instances if they are valid
            hw_manager_for_qpt = getattr(self, 'hw_manager', None)
            hw_accelerator_for_qpt = getattr(self, 'hw_accelerator', None)

            if hw_manager_for_qpt is None and HARDWARE_MANAGER_AVAILABLE:
                self.logger.debug("  _VERBOSE_INIT_QPT: QAR's hw_manager is None. QPT will try to get/create its own if needed by its __init__.")
            elif hw_manager_for_qpt:
                 self.logger.debug(f"  _VERBOSE_INIT_QPT: Passing QAR's hw_manager ({type(hw_manager_for_qpt)}) to QPT.")

            if hw_accelerator_for_qpt is None and HARDWARE_ACCEL_AVAILABLE:
                self.logger.debug("  _VERBOSE_INIT_QPT: QAR's hw_accelerator is None. QPT will try to get/create its own if needed by its __init__.")
            elif hw_accelerator_for_qpt:
                 self.logger.debug(f"  _VERBOSE_INIT_QPT: Passing QAR's hw_accelerator ({type(hw_accelerator_for_qpt)}) to QPT.")
            # --- End Hardware Determination ---
            # Determine mode to pass to QPT constructor
            # Ensure ProcessingMode and PrecisionMode are accessible here (e.g. imported)
            qpt_processing_mode = ProcessingMode.AUTO
            if getattr(self, 'use_classical', False):
                qpt_processing_mode = ProcessingMode.CLASSICAL
            
            qpt_precision_mode = PrecisionMode.AUTO  # Default, QPT's __init__ handles this

            self.logger.debug(f" Instantiating QuantumProspectTheory with alpha={pt_config.get('alpha', 0.88)}, mode={qpt_processing_mode}, precision={qpt_precision_mode}")
            
            # Initialize QPT with standardized 8-factor configuration
            self.quantum_pt = QuantumProspectTheory(
                alpha=pt_config.get('alpha', 0.88),
                beta=pt_config.get('beta', 0.88),
                lambda_=pt_config.get('lambda', 2.25),
                gamma=pt_config.get('gamma', 0.61),
                delta=pt_config.get('delta', 0.69),
                qubits=8,  # Force 8 qubits for standardization
                mode=qpt_processing_mode,
                precision=qpt_precision_mode,
                hw_manager=hw_manager_for_qpt,
                hw_accelerator=hw_accelerator_for_qpt,
                debug=getattr(self, '_debug', False),
                use_standard_factors=True,  # Enable standardized 8-factor model
                factor_names=StandardFactors.get_ordered_list(),
                initial_weights=StandardFactors.get_default_weights()
            )
            
            self.logger.info(f"  QuantumProspectTheory INSTANCE CREATED: {type(self.quantum_pt)}")

            # Initialize reference point handling
            if self.quantum_pt:  # Check if instance was created
                self.reference_points = {}
                self.reference_weights = {}
                self.ref_point_history = {}
                self.ref_point_mode = pt_config.get('ref_point_mode', 'adaptive')
                self.logger.debug(f"  Initialized QPT reference point structures. Mode: {self.ref_point_mode}")

            # This log message accesses attributes of self.quantum_pt. If self.quantum_pt is None here,
            # or if those attributes aren't set in QPT's __init__, this line could fail.
            if self.quantum_pt: # Check again before accessing attributes
                qpt_is_using_qar_accelerator = (hw_accelerator_for_qpt is not None and hw_accelerator_for_qpt == self.hw_accelerator)
                qpt_accelerated_status = getattr(hw_accelerator_for_qpt, 'gpu_available', False) if hw_accelerator_for_qpt else False
                qpt_mode_val = self.quantum_pt.processing_mode.value if hasattr(self.quantum_pt, 'processing_mode') and self.quantum_pt.processing_mode else 'N/A'
                
                self.logger.info(
                    f"Quantum Prospect Theory initialized successfully with 8-factor standardization:\n"
                    f"  - Parameters: alpha={self.quantum_pt.alpha:.2f}, beta={self.quantum_pt.beta:.2f}, "
                    f"lambda={self.quantum_pt.lambda_:.2f}, gamma={self.quantum_pt.gamma:.2f}\n"
                    f"  - Mode: {qpt_mode_val}\n"
                    f"  - Hardware: accelerated={qpt_accelerated_status}, using_qar_accelerator={qpt_is_using_qar_accelerator}"
                )
            else:
                self.logger.warning("_initialize_quantum_pt: QPT initialization failed, quantum_pt is None")
                qpt_precision_val = self.quantum_pt.precision.value if hasattr(self.quantum_pt, 'precision') and self.quantum_pt.precision else 'N/A'
                # Check QPT's OWN assessment of GPU availability
                qpt_gpu_available = 'N/A'
                if hasattr(self.quantum_pt, 'hw_accelerator') and self.quantum_pt.hw_accelerator and hasattr(self.quantum_pt.hw_accelerator, 'gpu_available'):
                    qpt_gpu_available = self.quantum_pt.hw_accelerator.gpu_available
                elif hasattr(self.quantum_pt, 'gpu_available'): # Fallback to a direct attribute on QPT
                    qpt_gpu_available = self.quantum_pt.gpu_available

                self.logger.info(
                    f"Quantum Prospect Theory initialized successfully (alpha={self.quantum_pt.alpha:.2f}, "
                    f"mode={qpt_mode_val}, precision={qpt_precision_val}, "
                    f"QPT reports GPU Available={qpt_gpu_available})"
                )

        except Exception as e: # Catch any exception during the try block
            self.logger.error(f"_initialize_quantum_pt: EXCEPTION during QPT initialization: {str(e)}", exc_info=True)
            self.quantum_pt = None # Ensure self.quantum_pt is None on any failure
        
        # <<< VERBOSE DEBUGGING INSERTION >>>
        self.logger.info(f"--- EXITING _initialize_quantum_pt --- self.quantum_pt is now: {type(self.quantum_pt)}")
        if self.quantum_pt is None:
            self.logger.warning("_initialize_quantum_pt: self.quantum_pt is None upon exit. QPT will not be available.")
        # <<< END VERBOSE DEBUGGING INSERTION >>>

    def _start_resource_monitoring(self) -> None:
        """Start resource monitoring in a separate thread."""
        if not self._resource_monitor_active:
            self._resource_monitor_active = True
            self._resource_monitor = ThreadPoolExecutor(max_workers=1)
            self._resource_monitor.submit(self._resource_monitoring_task)
            self.logger.debug("Resource monitoring started")

    def _resource_monitoring_task(self) -> None:
        """Monitor system resources periodically."""
        while self._resource_monitor_active:
            try:
                process = None
                if process:  # Will be replaced with actual process monitoring
                    memory_info = process.memory_info()
                    cpu_percent = process.cpu_percent(interval=0.1)

                    stats = {
                        "timestamp": time.time(),
                        "memory_rss_mb": memory_info.rss / (1024 * 1024),
                        "memory_vms_mb": memory_info.vms / (1024 * 1024),
                        "cpu_percent": cpu_percent,
                        "quantum_failure_count": self.quantum_failure_count,
                        "circuit_cache": self.circuit_cache.get_stats(),
                    }

                    # Log warning if memory usage is high
                    if stats["memory_rss_mb"] > 1000:  # 1GB
                        self.logger.warning(f"High memory usage: {stats['memory_rss_mb']:.2f} MB")

                # Sleep to avoid excessive monitoring
                time.sleep(10)
            except Exception as e:
                self.logger.error(f"Resource monitoring error: {str(e)}")
                time.sleep(30)  # Longer sleep on error

    def stop_resource_monitoring(self) -> None:
        """Stop the resource monitoring thread."""
        if self._resource_monitor_active:
            self._resource_monitor_active = False
            self._resource_monitor.shutdown(wait=False)
            self.logger.debug("Resource monitoring stopped")


    def register_factor(self, factor_name: str, weight: float = 1.0) -> None:
        """
        Register a factor with initial weight.
        If QHA is used, this list of factors (in order of registration)
        will correspond to QHA's experts if QHA is initialized afterwards or num_factors matches.
        """
        with self._lock:
            if not isinstance(factor_name, str) or not factor_name:
                self.logger.warning(f"Invalid factor name: {factor_name}")
                return

            weight = max(0.0, min(1.0, float(weight)))

            if factor_name not in self.factors:
                self.factors.append(factor_name) # Add to ordered list
                self.logger.debug(f"Registered new factor: {factor_name}")
                # If QHA is initialized based on dynamically registered factors,
                # this is where you might update qha_expert_factor_order
                # and potentially re-initialize QHA if its num_experts needs to change.
                # For simplicity, we assume num_factors for QHA is fixed at QAR init.
                # If self.hedge_algorithm and len(self.factors) > self.hedge_algorithm.num_experts:
                #     self.logger.warning("More factors registered than QHA num_experts. QHA might not manage all.")

            self.factor_weights[factor_name] = weight
            self.baseline_weights[factor_name] = weight # QAR's own baseline
            self.logger.debug(f"Updated factor weight for QAR: {factor_name}={weight:.4f}")

            self._normalize_weights() # Normalize QAR's own weights

    def _normalize_weights(self) -> None:
        """Normalize factor weights to sum to 1.0."""
        with self._lock:
            total_weight = sum(self.factor_weights.values())
            if total_weight > 0:
                self.factor_weights = {k: v / total_weight for k, v in self.factor_weights.items()}
                
    def _get_ordered_qha_expert_names(self) -> List[str]:
        """Returns the list of QAR factor names in the order QHA expects."""
        if self.hedge_algorithm:
            # If QAR registers factors and QHA is initialized with len(self.factors)
            # then self.factors IS the ordered list.
            if len(self.factors) == self.hedge_algorithm.num_experts:
                return list(self.factors) # Return a copy
            else:
                self.logger.warning(f"Mismatch between QAR factors ({len(self.factors)}) and QHA experts ({self.hedge_algorithm.num_experts}). Returning first {self.hedge_algorithm.num_experts} QAR factors.")
                return list(self.factors[:self.hedge_algorithm.num_experts])
        return []
    
    def _clip_probability(self, probability: float) -> float:
        """Clip probability to configured min/max bounds."""
        return np.clip(probability, self.min_probability, self.max_probability)

    def _to_log_odds(self, probability: float) -> float:
        """Convert probability to log-odds."""
        clipped_prob = self._clip_probability(probability)
        if clipped_prob <= 0 or clipped_prob >= 1:
            return np.sign(clipped_prob - 0.5) * 10
        return np.log(clipped_prob / (1.0 - clipped_prob))

    def _from_log_odds(self, log_odds: float) -> float:
        """Convert log-odds back to probability."""
        try:
            if log_odds > 15:
                probability = self.max_probability
            elif log_odds < -15:
                probability = self.min_probability
            else:
                probability = 1.0 / (1.0 + np.exp(-log_odds))
            return self._clip_probability(probability)
        except Exception as e:
            self.logger.error(f"Error in _from_log_odds for value {log_odds}: {e}")
            return 0.5  # Neutral default on error


    def apply_prospect_theory_weighting(self, probabilities, ambiguity=None):
        """
        Apply prospect theory probability weighting to prediction probabilities.

        Args:
            probabilities: Dictionary of outcome -> probability
            ambiguity: Optional dictionary of outcome -> ambiguity level

        Returns:
            Dictionary of outcome -> weighted probability
        """
        if not self.quantum_pt or not probabilities:
            return probabilities

        weighted_probs = {}
        for outcome, prob in probabilities.items():
            try:
                # Apply basic PT probability weighting
                weighted_prob = self.quantum_pt.probability_weighting(prob)

                # Apply ambiguity adjustment if provided
                if ambiguity and outcome in ambiguity:
                    amb_level = ambiguity[outcome]
                    if amb_level > 0:
                        weighted_prob = self.quantum_pt.evaluate_ambiguity(
                            prob, amb_level, 0.0  # Use neutral value
                        )

                weighted_probs[outcome] = weighted_prob
            except Exception as e:
                self.logger.warning(f"Error in PT probability weighting for {outcome}: {e}")
                weighted_probs[outcome] = prob  # Fallback to original

        # Normalize probabilities
        total = sum(weighted_probs.values())
        if total > 0:
            weighted_probs = {k: v/total for k, v in weighted_probs.items()}

        return weighted_probs

    def evaluate_prospect(self, current_value, reference_value=None, asset_id=None):
        """
        Evaluate a prospect using PT value function.

        Args:
            current_value: Current value
            reference_value: Reference value (default: 0)
            asset_id: Optional asset identifier for tracking references

        Returns:
            PT value of the prospect
        """
        if not self.quantum_pt:
            return 0.0

        # Use reference_value if provided, otherwise use 0
        ref_value = reference_value if reference_value is not None else 0.0

        try:
            # Calculate relative value
            relative_value = current_value - ref_value

            # Calculate PT value
            return self.quantum_pt.value_function(relative_value)
        except Exception as e:
            self.logger.warning(f"Error in PT value calculation: {e}")
            return 0.0

    def pt_enhanced_factors(self, factors, risk_indicators=None):
        """
        Enhance factor weights with PT principles.

        Args:
            factors: Dictionary of factor_name -> value
            risk_indicators: Optional dict of factor_name -> risk indicator

        Returns:
            Dictionary of PT-adjusted factors
        """
        if not self.quantum_pt or not factors:
            return factors

        # Default risk indicators
        if risk_indicators is None:
            risk_indicators = {k: 0.0 for k in factors}

        factor_names = list(factors.keys())
        factor_values = [factors[name] for name in factor_names]
        base_importance = [self.factor_weights.get(name, 1.0) for name in factor_names]
        risk_values = [risk_indicators.get(name, 0.0) for name in factor_names]

        try:
            # Apply PT-based feature selection
            if hasattr(self.quantum_pt, 'feature_selection'):
                weighted_importance = self.quantum_pt.feature_selection(
                    factor_values, base_importance, risk_values
                )

                # Adjust factors by their PT-weighted importance
                adjusted_factors = {}
                for name, value, importance in zip(factor_names, factor_values, weighted_importance):
                    # Apply importance as a scaling factor
                    adjusted_factors[name] = value * importance

                return adjusted_factors
        except Exception as e:
            self.logger.warning(f"Error in PT feature selection: {e}")

        return factors  # Fallback to original

    @quantum_accelerated(use_hw_accel=True, hw_batch_size=2, device_shots=1024)
    def _analyze_market_regime(self, factors: np.ndarray) -> Dict[str, Any]:
        """Analyze market regime using Quantum Fourier Transform."""
        try:
            # Generate cache key for the entire function
            cache_key = f"regime_analysis_{hash(str(factors))}"
            
            # Check cache
            if hasattr(self, 'circuit_cache') and cache_key in self.circuit_cache:
                self.logger.debug(f"Regime analysis cache hit")
                return self.circuit_cache[cache_key]
                
            # Apply QFT to market factors
            qft_result = self._execute_with_fallback('qft', (factors,))

            # Calculate frequency domain features
            frequencies = np.fft.fftfreq(len(qft_result))
            frequency_strengths = np.abs(qft_result)
            phases = np.angle(qft_result)
            dominant_freq = np.argmax(frequency_strengths[1:]) + 1  # Skip DC component
            
            # Calculate spectral characteristics
            total_power = np.sum(frequency_strengths**2)
            low_freq_power = np.sum(frequency_strengths[:len(frequency_strengths)//4]**2)  # Lower 25% of frequencies
            high_freq_power = np.sum(frequency_strengths[len(frequency_strengths)//2:]**2)  # Upper 50% of frequencies
            
            # Enhanced metrics
            market_volatility = high_freq_power / total_power if total_power > 0 else 0.5
            phase_coherence = np.std(phases) / np.pi  # Normalized to [0,1]
            market_noise = phase_coherence
            
            # Calculate trend strength 
            market_strength = frequency_strengths[dominant_freq] / np.sum(frequency_strengths) if np.sum(frequency_strengths) > 0 else 0.5
            
            # Enhanced regime determination
            if market_strength > 0.6 and market_volatility < 0.3:
                current_regime = "bullish" if frequencies[dominant_freq] > 0 else "bearish"
            elif market_strength > 0.4 and market_volatility > 0.6:
                current_regime = "volatile_bullish" if frequencies[dominant_freq] > 0 else "volatile_bearish"
            elif market_strength < 0.3 and market_volatility < 0.4:
                current_regime = "neutral"
            elif 0.3 < market_strength < 0.5 and 0.3 < market_volatility < 0.6:
                current_regime = "trend_reversal"
            elif market_strength > 0.7 and market_volatility < 0.2:
                current_regime = "breakout"
            elif market_strength < 0.2 and market_volatility < 0.3:
                current_regime = "consolidation"
            else:
                # Fallback to legacy regime mapping if none of the above conditions are met
                regime_mapping = {
                    0: "neutral",
                    1: "bullish",
                    2: "bearish",
                    3: "volatile_bullish",
                    4: "volatile_bearish",
                    5: "trend_reversal",
                    6: "consolidation",
                    7: "breakout"
                }
                current_regime = regime_mapping.get(dominant_freq % len(regime_mapping), "neutral")
            
            # Calculate confidence based on clarity of regime indicators
            confidence = (market_strength + (1 - market_noise)) / 2
            
            result = {
                'regime': current_regime,
                'confidence': float(confidence),
                'strength': float(market_strength),
                'volatility': float(market_volatility),
                'noise': float(market_noise),
                'dominant_frequency': int(dominant_freq),
                'spectral_data': {
                    'magnitudes': frequency_strengths.tolist() if hasattr(frequency_strengths, 'tolist') else [],
                    'phases': phases.tolist() if hasattr(phases, 'tolist') else [],
                    'frequencies': frequencies.tolist() if hasattr(frequencies, 'tolist') else []
                }
            }
            
            # Cache the result
            if hasattr(self, 'circuit_cache'):
                self.circuit_cache[cache_key] = result
                
            return result

        except Exception as e:
            self.logger.error(f"Error in market regime analysis: {e}")
            return {
                'regime': "unknown",
                'confidence': 0.0,
                'strength': 0.0,
                'volatility': 0.0,
                'noise': 0.0,
                'dominant_frequency': 0,
                'error': str(e)
            }
        
    def _get_top_contributing_factors(self, factor_probabilities: Dict[str, float], n: int = 3) -> List[Tuple[str, float]]:
        """Get the top contributing factors based on deviation from neutral."""
        # Calculate contribution as deviation from neutral (0.5)
        contributions = [(name, abs(value - 0.5) * 2) for name, value in factor_probabilities.items()]
        
        # Sort by contribution (descending) and take top n
        return sorted(contributions, key=lambda x: x[1], reverse=True)[:n]

    def _generate_circuit_cache_key(self, factors, market_state, num_qubits):
        """Generate consistent cache key for quantum circuits."""
        try:
            # Convert factors and market_state to hashable form
            factors_hash = hash(tuple(float(f) for f in factors))
            market_hash = hash(str(sorted(market_state.items()))) if isinstance(market_state, dict) else hash(str(market_state))
            # Combine all elements into a single key
            return f"qnode_{factors_hash}_{market_hash}_{num_qubits}"
        except Exception as e:
            self.logger.warning(f"Error generating circuit cache key: {e}")
            # Fallback to basic hash
            return f"qnode_{hash(str(factors))}_{hash(str(market_state))}_{num_qubits}"

# In qar.py, inside the QuantumAgenticReasoning class
# This is the version you confirmed you are using, with the targeted fix.

    @quantum_accelerated(use_hw_accel=True, hw_batch_size=2, device_shots=1024)
    def _recognize_patterns(self, factors: np.ndarray) -> Dict[str, Any]:
        """Recognize patterns using quantum memory, with robust handling of scalar inputs."""
        try:
            # Ensure 'factors' (the input pattern) is a 1D NumPy array
            current_factors_np = np.atleast_1d(np.asarray(factors, dtype=float))
            if np.any(np.isnan(current_factors_np)) or np.any(np.isinf(current_factors_np)):
                self.logger.warning(f"Input 'factors' for pattern recognition contains NaN/Inf: {factors}. Returning default.")
                return {'similarity': 0.0, 'best_match': None, 'note': 'input_factors_invalid'}

            # Generate cache key for the entire function based on the processed factors
            cache_key = f"pattern_rec_{hash(current_factors_np.tobytes())}"
            
            if hasattr(self, 'circuit_cache') and cache_key in self.circuit_cache:
                self.logger.debug("Pattern recognition cache hit")
                return self.circuit_cache[cache_key]
            
            # Skip if memory is empty
            if not hasattr(self, 'memory_patterns') or not self.memory_patterns:
                self.logger.debug("_recognize_patterns: Memory patterns list is empty or not initialized.")
                return {'similarity': 0.0, 'best_match': None, 'note': 'memory_empty'}

            memory_patterns_np_list = []
            valid_memory_indices = [] # Keep track of original indices of valid patterns

            for p_idx, p_item_raw in enumerate(self.memory_patterns):
                # p_item_raw is what _store_pattern saved, e.g.,
                # {'circuit': ..., 
                #  'inputs': ..., 
                #  'result': {'action': ..., 'raw_result': [...], ...},  <-- p_item_raw['result'] is this dict
                #  ...}

                # --- TARGETED FIX START ---
                p_item_data_for_conversion = None
                if isinstance(p_item_raw, dict) and 'result' in p_item_raw:
                    result_content = p_item_raw['result'] # This is the dictionary shown in your error log
                    if isinstance(result_content, dict) and 'raw_result' in result_content:
                        # Extract the actual list of numbers
                        p_item_data_for_conversion = result_content['raw_result'] 
                    elif isinstance(result_content, (list, np.ndarray)): # If result_content itself is the list/array
                        p_item_data_for_conversion = result_content
                    else:
                        self.logger.debug(f"Memory pattern at index {p_idx} has 'result' but 'raw_result' (or list/array) is missing or not in expected format. Result content: {result_content}. Skipping.")
                elif isinstance(p_item_raw, (list, np.ndarray)): # If the stored item is directly the pattern
                    p_item_data_for_conversion = p_item_raw
                else:
                    self.logger.debug(f"Memory pattern at index {p_idx} is not in expected format (dict with result/raw_result or list/array). Content: {p_item_raw}. Skipping.")

                if p_item_data_for_conversion is None:
                    # self.logger.debug(f"Skipping pattern at index {p_idx} due to missing numeric part.") # Already logged above
                    continue
                # --- TARGETED FIX END ---
                
                try:
                    # Now, p_item_data_for_conversion should be the list of numbers
                    processed_item = np.atleast_1d(np.asarray(p_item_data_for_conversion, dtype=float))
                    if np.any(np.isnan(processed_item)) or np.any(np.isinf(processed_item)):
                        self.logger.warning(f"Numeric part of memory pattern at index {p_idx} contains NaN/Inf after conversion. Value: {p_item_data_for_conversion}. Skipping.")
                        continue
                    memory_patterns_np_list.append(processed_item)
                    valid_memory_indices.append(p_idx) # Store original index
                except (ValueError, TypeError) as e_conv:
                    self.logger.warning(f"Could not convert numeric part of memory pattern at index {p_idx} to array: {p_item_data_for_conversion}. Error: {e_conv}. Skipping.")
                    continue
            
            if not memory_patterns_np_list:
                self.logger.debug("_recognize_patterns: No valid numeric memory patterns found after processing.")
                return {'similarity': 0.0, 'best_match': None, 'note': 'no_valid_memory_patterns'}

            similarity_thresholds = np.array([0.6] * len(memory_patterns_np_list))
            
            num_pattern_qubits = min(current_factors_np.size, self.qubits)
            num_memory_qubits = min(len(memory_patterns_np_list), 3) # Max 3 memory patterns for this example circuit

            # Generate circuit cache key using processed inputs
            # For list of arrays, hash their concatenated bytes or tuple of bytes
            memory_hash_content = b"".join([mem_arr.tobytes() for mem_arr in memory_patterns_np_list])
            circuit_instance_cache_key = f"pattern_circuit_instance_{hash(current_factors_np.tobytes())}_{hash(memory_hash_content)}_{self.qubits}"

            if hasattr(self, 'pattern_circuit_cache') and circuit_instance_cache_key in self.pattern_circuit_cache:
                pattern_circuit = self.pattern_circuit_cache[circuit_instance_cache_key]
            else:
                self.logger.debug(f"Creating new pattern recognition circuit instance for key: {circuit_instance_cache_key}")
                @qml.qnode(self.device, interface="autograd")
                def pattern_circuit(pattern_input_arr, memory_list_of_arrs, thresholds_arr):
                    # pattern_input_arr is current_factors_np (1D array)
                    # memory_list_of_arrs is memory_patterns_np_list (list of 1D arrays)

                    for i in range(num_pattern_qubits):
                        # Use .size for NumPy arrays
                        qml.RY(np.pi * pattern_input_arr[i % pattern_input_arr.size], wires=i)

                    for j, (mem_pattern_arr, threshold) in enumerate(zip(memory_list_of_arrs, thresholds_arr)):
                        if j >= num_memory_qubits: break # Corrected from num_pattern_qubits to num_memory_qubits
                        
                        similarity = 0.0
                        try:
                            # Both pattern_input_arr and mem_pattern_arr are 1D arrays
                            len1, len2 = pattern_input_arr.size, mem_pattern_arr.size
                            common_len = min(len1, len2)
                            
                            if common_len > 0:
                                p1_c = pattern_input_arr[:common_len]
                                p2_c = mem_pattern_arr[:common_len]
                                similarity = np.dot(p1_c, p2_c)
                                # Optional normalization for cosine similarity:
                                # norm1, norm2 = np.linalg.norm(p1_c), np.linalg.norm(p2_c)
                                # if norm1 > 1e-9 and norm2 > 1e-9:
                                #     similarity = np.dot(p1_c, p2_c) / (norm1 * norm2)
                                # else: similarity = 0.0
                        except Exception as e_sim_circuit:
                            self.logger.warning(f"Error in circuit similarity calc for mem_pattern {j}: {e_sim_circuit}")
                            similarity = 0.0 # Default similarity on error

                        if similarity > threshold:
                            control = j % num_pattern_qubits
                            target = (j + 1) % num_pattern_qubits
                            if control != target and target < num_pattern_qubits : # Ensure valid and distinct wires
                                qml.CRZ(similarity * np.pi, wires=[control, target])
                    
                    for i in range(num_pattern_qubits): qml.Hadamard(wires=i)
                    return [qml.expval(qml.PauliZ(i)) for i in range(num_pattern_qubits)]

                if not hasattr(self, 'pattern_circuit_cache'): self.pattern_circuit_cache = {}
                self.pattern_circuit_cache[circuit_instance_cache_key] = pattern_circuit
                # pattern_circuit = self.pattern_circuit_cache[circuit_instance_cache_key] # This line was redundant
            
            recognition_result_list = []
            try:
                # Select the subset of memory patterns and thresholds for the QNode call
                memory_subset_for_qnode = memory_patterns_np_list[:num_memory_qubits]
                threshold_subset_for_qnode = similarity_thresholds[:num_memory_qubits]
                
                # Apply shots configuration to the QNode call
                # Use the class-wide shots configuration
                if not hasattr(self, 'shots_for_qnodes'):
                    self.shots_for_qnodes = 1024  # Default value if not set
                pattern_circuit_with_shots = _apply_shots_to_qnode(pattern_circuit, self.shots_for_qnodes)
                self.logger.debug(f"Applying shots={self.shots_for_qnodes} to pattern_circuit call")
                
                recognition_result_list = pattern_circuit_with_shots(
                    current_factors_np, 
                    memory_subset_for_qnode, 
                    threshold_subset_for_qnode
                )
            except Exception as circuit_error:
                self.logger.error(f"Pattern circuit execution failed: {circuit_error}", exc_info=True)
                # Fallback execution
                fallback_result = self._execute_with_fallback('pattern_recognition', (current_factors_np, memory_patterns_np_list))
                recognition_result_list = fallback_result.tolist() if isinstance(fallback_result, np.ndarray) else fallback_result
            
            recognition_result_arr = np.array(recognition_result_list, dtype=float) # Ensure it's an array


            # Calculate classical similarities for verification
            classical_similarities = []
            for i, mem_pattern_np_item in enumerate(memory_patterns_np_list): # Iterate through all valid numeric patterns
                similarity = 0.0
                try:
                    len1, len2 = current_factors_np.size, mem_pattern_np_item.size
                    common_len = min(len1, len2)

                    if common_len > 0:
                        factors_common = current_factors_np[:common_len]
                        mem_pattern_common = mem_pattern_np_item[:common_len]
                        norm_factors = np.linalg.norm(factors_common)
                        norm_pattern = np.linalg.norm(mem_pattern_common)
                        
                        if norm_factors > 1e-9 and norm_pattern > 1e-9:
                            similarity = np.dot(factors_common, mem_pattern_common) / (norm_factors * norm_pattern)
                        elif norm_factors < 1e-9 and norm_pattern < 1e-9: # Both zero vectors
                            similarity = 1.0 
                        # else one is zero, other is not -> similarity is 0.0 (already initialized)
                    # else common_len is 0 -> similarity is 0.0 (already initialized)
                    
                    classical_similarities.append(float(similarity))
                except Exception as e_classical_sim:
                    # Use valid_memory_indices to report the original index in the warning
                    original_item_idx = valid_memory_indices[i] if i < len(valid_memory_indices) else "unknown"
                    self.logger.warning(f"Error calculating classical similarity for memory item (original index {original_item_idx}): {e_classical_sim}")
                    classical_similarities.append(0.0) # Default on error
            
            best_match_idx_in_filtered_list = np.argmax(classical_similarities) if classical_similarities else None
            
            quantum_similarity_val = 0.0
            try:
                if recognition_result_arr.size > 0 and np.issubdtype(recognition_result_arr.dtype, np.number):
                    quantum_similarity_val = float(np.mean(np.abs(recognition_result_arr)))
                else:
                    self.logger.debug(f"Quantum recognition result not suitable for mean: {recognition_result_arr}")
            except Exception as e_mean:
                self.logger.warning(f"Could not calculate mean of quantum recognition_result: {recognition_result_arr}, error: {e_mean}")

            # Determine best_match based on the original index from valid_memory_indices
            best_match_original_idx = None
            best_match_metadata = None
            if best_match_idx_in_filtered_list is not None and best_match_idx_in_filtered_list < len(valid_memory_indices):
                best_match_original_idx = valid_memory_indices[best_match_idx_in_filtered_list]
                if hasattr(self, 'memory_metadata') and best_match_original_idx < len(self.memory_metadata):
                    best_match_metadata = self.memory_metadata[best_match_original_idx]

            result_dict = {
                'quantum_similarity': quantum_similarity_val,
                'classical_similarity': float(max(classical_similarities)) if classical_similarities else 0.0,
                'similarity': float(max(classical_similarities)) if classical_similarities else 0.0, # Using classical as primary
                'best_match_index_original': int(best_match_original_idx) if best_match_original_idx is not None else None, # Report original index
                'best_match': best_match_metadata,
                'recognition_result': recognition_result_arr.tolist()
            }
            
            if hasattr(self, 'circuit_cache'):
                self.circuit_cache[cache_key] = result_dict
            return result_dict
    
        except Exception as e:
            self.logger.error(f"Critical error in _recognize_patterns: {e}", exc_info=True)
            return {'similarity': 0.0, 'best_match': None, 'error': str(e)}
    
    @quantum_accelerated(use_hw_accel=True, hw_batch_size=2, device_shots=1024)
    def _optimize_decision(self, factors: np.ndarray, regime_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize decision using amplitude amplification with dynamic QNode generation."""
        try:
            # Generate cache key for the entire function
            cache_key = f"decision_opt_{hash(str(factors))}_{hash(str(regime_analysis))}"
            
            # Check cache
            if hasattr(self, 'circuit_cache') and cache_key in self.circuit_cache:
                self.logger.debug("Decision optimization cache hit")
                return self.circuit_cache[cache_key]
            
            # Create decision weights based on regime
            weights = np.zeros(self.qubits * 2)

            # First half of weights for factor importance
            if regime_analysis['regime'] == "bullish":
                weights[:self.qubits] = np.array([0.8, 0.9, 0.7, -0.5, 0.6, 0.4, 0.2, 0.3][:self.qubits])
            elif regime_analysis['regime'] == "bearish":
                weights[:self.qubits] = np.array([-0.8, -0.9, -0.7, 0.5, -0.6, -0.4, -0.2, -0.3][:self.qubits])
            elif regime_analysis['regime'] == "volatile_bullish":
                weights[:self.qubits] = np.array([0.5, 0.6, 0.8, 0.7, 0.4, 0.5, 0.3, 0.4][:self.qubits])
            elif regime_analysis['regime'] == "volatile_bearish":
                weights[:self.qubits] = np.array([-0.5, -0.6, -0.8, 0.7, -0.4, -0.5, -0.3, -0.4][:self.qubits])
            elif regime_analysis['regime'] == "trend_reversal":
                weights[:self.qubits] = np.array([0.2, -0.3, 0.1, 0.8, -0.2, 0.7, 0.6, -0.4][:self.qubits])
            else:  # neutral or unknown
                weights[:self.qubits] = np.array([0.1, 0.2, 0.1, 0.3, 0.2, 0.1, 0.2, 0.1][:self.qubits])

            # Second half for decision bias
            weights[self.qubits:] = np.linspace(0.1, 0.8, self.qubits)

            # Ensure weights arrays are the right size
            if len(weights[:self.qubits]) < self.qubits:
                weights[:self.qubits] = np.pad(weights[:self.qubits], (0, self.qubits - len(weights[:self.qubits])))
                
            # Prepare parameters for the dynamic QNode
            num_factor_qubits = min(5, self.qubits // 2)
            num_decision_qubits = min(3, self.qubits - num_factor_qubits)
            
            # Calculate ry_angles from factors
            ry_angles = np.array([np.pi * f for f in factors[:num_factor_qubits]])
            
            # Calculate cry_angles from factors and weights
            cry_angles = np.array([
                factors[i] * weights[i] * np.pi 
                for i in range(min(num_factor_qubits, len(factors), len(weights)))
            ])
            
            # Calculate market bias from regime
            regime_strength = regime_analysis.get('strength', 0.5)
            regime_confidence = regime_analysis.get('confidence', 0.5)
            market_bias = 0.0
            
            if regime_analysis['regime'] in ["bullish", "volatile_bullish", "breakout"]:
                market_bias = regime_strength * regime_confidence
            elif regime_analysis['regime'] in ["bearish", "volatile_bearish"]:
                market_bias = -regime_strength * regime_confidence
            
            # Generate a unique circuit cache key
            circuit_cache_key = self._generate_circuit_cache_key(factors, weights, self.qubits)
            
            # Check if we should use cached circuit or create new one
            if hasattr(self, 'decision_circuit_cache') and circuit_cache_key in self.decision_circuit_cache:
                decision_circuit = self.decision_circuit_cache[circuit_cache_key]
            else:
                self.logger.debug("Creating new decision optimization circuit")
                # Create the dynamic circuit - using the example pattern you provided
                @qml.qnode(self.device, interface="autograd")
                def decision_circuit(ry_angle_params, cry_angle_params, mkt_bias):
                    # Input layer using angles derived from raw probabilities
                    for i in range(num_factor_qubits):
                        qml.RY(ry_angle_params[i], wires=i)  # Angle based on factor value

                    # Entanglement layer using angles derived from weighted probabilities
                    for i in range(num_factor_qubits):
                        for j in range(num_decision_qubits):
                            # Use the pre-calculated CRY angle (prob * weight * pi)
                            qml.CRY(cry_angle_params[i], wires=[i, num_factor_qubits + j])

                    # Decision layer with market bias
                    for i in range(num_decision_qubits):
                        qml.Hadamard(wires=num_factor_qubits + i)
                        qml.RZ(mkt_bias * qnp.pi / 2, wires=num_factor_qubits + i)  # Bias via RZ

                    # Entanglement in decision layer
                    for i in range(num_decision_qubits - 1):
                        qml.CNOT(wires=[num_factor_qubits + i, num_factor_qubits + i + 1])
                    if num_decision_qubits > 0:  # Avoid CNOT on wire 0 if only 1 decision qubit
                        qml.CNOT(wires=[num_factor_qubits + (num_decision_qubits - 1), num_factor_qubits])  # Circular CNOT

                    # Measure decision qubits
                    return [qml.expval(qml.PauliZ(k)) for k in range(num_factor_qubits, num_factor_qubits + num_decision_qubits)]
                
                # Initialize decision circuit cache if needed
                if not hasattr(self, 'decision_circuit_cache'):
                    self.decision_circuit_cache = {}
                
                # Store circuit in cache
                self.decision_circuit_cache[circuit_cache_key] = decision_circuit
                decision_circuit = self.decision_circuit_cache[circuit_cache_key]
            
            # Execute the circuit
            try:
                # Apply shots configuration to the QNode call
                # Use the class-wide shots configuration
                if not hasattr(self, 'shots_for_qnodes'):
                    self.shots_for_qnodes = 1024  # Default value if not set
                decision_circuit_with_shots = _apply_shots_to_qnode(decision_circuit, self.shots_for_qnodes)
                self.logger.debug(f"Applying shots={self.shots_for_qnodes} to decision_circuit call")
                
                measurements = decision_circuit_with_shots(ry_angles, cry_angles, market_bias)
                self.logger.debug(f"Raw quantum decision measurements: {measurements}")
                decision_result = np.array(measurements)
            except Exception as circuit_error:
                self.logger.error(f"Decision circuit execution failed: {circuit_error}")
                # Fallback to standard implementation
                decision_result = self._execute_with_fallback('decision_optimization', (factors, weights))

            # Convert quantum results to actionable decisions
            # Map the decision qubits to buy/sell/hold actions
            if len(decision_result) >= 3:
                buy_signal = float(decision_result[0])
                sell_signal = float(decision_result[1])
                hold_signal = float(decision_result[2])
            else:
                # If we have fewer than 3 qubits, use the first one for buy/sell and derive hold
                if len(decision_result) >= 1:
                    buy_signal = float(decision_result[0])
                    sell_signal = -buy_signal  # Opposite of buy
                    hold_signal = 0.0  # Neutral
                else:
                    # Fallback if no measurements
                    buy_signal = 0.0
                    sell_signal = 0.0
                    hold_signal = 1.0  # Default to hold

            # Normalize signals
            signal_sum = abs(buy_signal) + abs(sell_signal) + abs(hold_signal)
            if signal_sum > 0:
                buy_signal = buy_signal / signal_sum
                sell_signal = sell_signal / signal_sum
                hold_signal = hold_signal / signal_sum

            # Map from [-1,1] to [0,1] range
            buy_signal = (buy_signal + 1) / 2
            sell_signal = (sell_signal + 1) / 2
            hold_signal = (hold_signal + 1) / 2

            # Calculate confidence from decision strength
            decision_signals = [buy_signal, sell_signal, hold_signal]
            action_idx = np.argmax(decision_signals)
            actions = ["buy", "sell", "hold"]
            action = actions[action_idx]
            
            # Calculate action strength
            action_strength = decision_signals[action_idx]
            
            # Calculate consistency
            consistency = max(0.0, 1.0 - np.std(decision_signals))
            
            # Calculate confidence based on action strength, regime confidence, and consistency
            base_confidence = action_strength * regime_analysis.get('confidence', 0.5)
            confidence = base_confidence * (0.7 + 0.3 * consistency) * 1.2  # Scale slightly

            result = {
                'action': action,
                'confidence': float(min(confidence, 1.0)),  # Cap at 1.0
                'buy_signal': float(buy_signal),
                'sell_signal': float(sell_signal),
                'hold_signal': float(hold_signal),
                'consistency': float(consistency),
                'action_strength': float(action_strength),
                'raw_result': decision_result.tolist() if hasattr(decision_result, 'tolist') else [],
                'optimization_weights': weights.tolist() if hasattr(weights, 'tolist') else [],
                'regime_confidence': float(regime_analysis.get('confidence', 0.5)),
                'ry_angles': ry_angles.tolist() if hasattr(ry_angles, 'tolist') else [],
                'cry_angles': cry_angles.tolist() if hasattr(cry_angles, 'tolist') else [],
                'market_bias': float(market_bias)
            }
            
            # Cache the result
            if hasattr(self, 'circuit_cache'):
                self.circuit_cache[cache_key] = result

            return result

        except Exception as e:
            self.logger.error(f"Error in decision optimization: {e}")
            return {
                'action': "hold",  # Default to hold on error
                'confidence': 0.1,
                'buy_signal': 0.0,
                'sell_signal': 0.0,
                'hold_signal': 1.0,
                'consistency': 0.5,
                'error': str(e)
            }
        
    def make_decision(
            self,
            factors: Dict[str, float], # Expects raw factor values, will be converted to probs
            market_data: Dict[str, Any] = None,
            position_state: Dict[str, Any] = None,
            max_factors: Optional[int] = None,
            min_confidence: Optional[float] = None,
            use_quantum: Optional[bool] = None, # This is for QAR's main quantum path (QLMSR)
            decision_params: Dict[str, Any] = None,
            risk_indicators: Optional[Dict[str, float]] = None, # For PT
            ambiguity_levels: Optional[Dict[str, float]] = None # For PT
    ) -> Dict[str, Any]:
        """
            Make a trading decision based on factors and market data.
        """
        start_time_perf = time.perf_counter()

        try:
            with self._lock:
                # --- Start: Your existing initializations and validations ---
                market_data = market_data or {}
                position_state = position_state or {}
                decision_params = decision_params or {}
                risk_indicators = risk_indicators or {}
                ambiguity_levels = ambiguity_levels or {}

                effective_min_confidence = min_confidence if min_confidence is not None else self.decision_threshold
                # This 'should_use_quantum' pertains to QAR's choice between its _quantum_decision (QLMSR) path
                # and its _classical_decision path. QHA can be used in either.
                should_use_qar_main_quantum_path = use_quantum if use_quantum is not None else (self.quantum_available and not self.use_classical)

                hw_is_actually_ready = self.hw_accelerator is not None
                use_accelerated_for_this_run = decision_params.get('use_accelerated', hw_is_actually_ready)
                decision_params['use_accelerated'] = use_accelerated_for_this_run

                regime_str = market_data.get('panarchy_phase', market_data.get('regime', 'unknown'))
                regime_enum = MarketPhase.from_string(regime_str)

                if not factors:
                    self.logger.warning("No factors provided, returning HOLD decision")
                    return {
                        'decision_type': 'HOLD', 'action': 'HOLD', 'confidence': 0.0, 'actionable': False,
                        'explanation': "No factors provided", 'regime': regime_enum.value,
                        'timestamp': datetime.now().timestamp(), 'method': "default_no_factors", # Changed from time.time() for consistency
                        'factors': {}, 'weights': {}, 'metadata': {'error_details': 'No factors input'}
                    }

                validated_factors_as_probs = self._validate_factors(factors)

                if not validated_factors_as_probs:
                    self.logger.warning("No valid factors after validation, returning HOLD decision")
                    return {
                        'decision_type': 'HOLD', 'action': 'HOLD', 'confidence': 0.0, 'actionable': False,
                        'explanation': "No valid factors after validation", 'regime': regime_enum.value,
                        'timestamp': datetime.now().timestamp(), 'method': "validation_failed", # Changed from time.time()
                        'factors': factors, 'weights': {}, 'metadata': {'error_details': 'Factor validation failed'}
                    }

                validated_market_data = self._validate_market_data(market_data)
                
                # position_open_bool, position_direction_int removed as they are extracted inside _classical/_quantum decision paths

                factors_for_decision = validated_factors_as_probs.copy() # Work with a copy
                if hasattr(self, 'quantum_pt') and self.quantum_pt is not None:
                    if risk_indicators:
                        factors_for_decision = self.pt_enhanced_factors(factors_for_decision, risk_indicators)
                        decision_params['pt_factors_enhanced'] = True

                if max_factors and len(factors_for_decision) > max_factors:
                    sorted_factors = sorted(
                        factors_for_decision.items(),
                        key=lambda x: abs(x[1] - 0.5),
                        reverse=True
                    )[:max_factors]
                    factors_for_decision = dict(sorted_factors)
                # --- End: Your existing initializations and validations ---

                # --- QHA INTEGRATION: Prepare market_features_array_for_qha_predict (NumPy array for QHA.predict) ---
                # This block was already present and correct in your provided code.
                # This array is for QHA's internal use if its 'predict' involves quantum processing.
                # Ensure its dimension matches self.qha_feature_dim (QHA's __init__ param).
                _temp_market_features_for_qha_predict = np.array([
                    validated_market_data.get('trend', 0.0),
                    validated_market_data.get('volatility', 0.5),
                    validated_market_data.get('momentum', 0.0),
                    validated_market_data.get('risk_level', 0.5)
                    # Add more or change order to match self.qha_feature_dim if it's different from 4
                ], dtype=np.float64)

                # Ensure the array matches QHA's expected feature_dim
                if len(_temp_market_features_for_qha_predict) > self.qha_feature_dim:
                    market_features_array_for_qha_predict = _temp_market_features_for_qha_predict[:self.qha_feature_dim]
                elif len(_temp_market_features_for_qha_predict) < self.qha_feature_dim:
                    padding = np.zeros(self.qha_feature_dim - len(_temp_market_features_for_qha_predict), dtype=np.float64)
                    market_features_array_for_qha_predict = np.concatenate((_temp_market_features_for_qha_predict, padding))
                else:
                    market_features_array_for_qha_predict = _temp_market_features_for_qha_predict
                # --- END QHA PREDICT CONTEXT PREP ---
                
                # --- Determine Decision Path (QAR's Quantum/QLMSR vs. QAR's Classical) ---
                method_used_for_decision_path_str = "classical_aggregation" # Default path name
                decision_type: DecisionType # Type hint for clarity
                confidence: float
                metadata: Dict[str, Any]

                if should_use_qar_main_quantum_path and self.quantum_failure_count < self.quantum_fallback_threshold:
                    self.logger.debug("Attempting QAR's quantum decision path (_quantum_decision).")
                    decision_type, confidence, metadata = self._quantum_decision(
                        factor_probabilities=factors_for_decision, # Corrected: pass the (potentially PT-enhanced and factor-selected) dict
                        market_data=validated_market_data,
                        position_state=position_state,
                        target_confidence_threshold=effective_min_confidence,
                        decision_params=decision_params,
                        # <<< SENIOR DEV INTEGRATION: Pass QHA predict features to _quantum_decision >>>
                        market_features_for_qha_predict=market_features_array_for_qha_predict
                    )
                    method_used_for_decision_path_str = metadata.get('method', 'quantum_lmsr') # Method string from _quantum_decision
                else:
                    if should_use_qar_main_quantum_path: # Implies quantum failed or threshold met
                        self.logger.info("Falling back to QAR's classical decision path.")
                    self.logger.debug("Using QAR's classical decision path (_classical_decision).")
                    decision_type, confidence, metadata = self._classical_decision(
                        factor_probabilities=factors_for_decision, # Corrected: pass the (potentially PT-enhanced and factor-selected) dict
                        market_data=validated_market_data,
                        position_state=position_state,
                        target_confidence_threshold=effective_min_confidence,
                        decision_params=decision_params,
                        market_features_for_qha_predict=market_features_array_for_qha_predict # Already passed this correctly
                    )
                    method_used_for_decision_path_str = metadata.get('method', 'classical_aggregation') # Method string from _classical_decision

                # --- Modified HOLD override logic with lower fixed threshold (0.3) ---
                # Use a fixed lower threshold of 0.3 to allow more decisions to pass through
                applied_threshold = min(0.3, effective_min_confidence)
                if confidence < applied_threshold and decision_type not in [DecisionType.HOLD, DecisionType.EXIT]:
                    prev_decision_name = decision_type.name # Use decision_type directly
                    decision_type = DecisionType.HOLD
                    metadata["reasoning"] = f"Insufficient confidence ({confidence:.2f} < {applied_threshold:.2f}) for {prev_decision_name}. Original: {metadata.get('reasoning', '')}"
                    metadata["original_decision"] = prev_decision_name
                    metadata["original_confidence"] = confidence
                    confidence = 0.0
                    self.logger.debug(f"Decision overridden to HOLD because confidence {confidence:.2f} < threshold {applied_threshold:.2f} (effective_min_confidence was {effective_min_confidence:.2f})")

                final_reasoning = metadata.get("reasoning", "No reasoning provided.")
                
                # Construct final method string for reporting
                final_method_report_str = method_used_for_decision_path_str # Start with base method from path
                if use_accelerated_for_this_run:
                    final_method_report_str += "_accelerated"
                if decision_params.get('pt_factors_enhanced', False) or 'pt_value' in metadata:
                    final_method_report_str += "_pt"
                # Check metadata if QHA weights were actually used by the chosen path
                if metadata.get('used_qha_weights', False) or metadata.get('used_qha_weights_in_quantum_path', False):
                    final_method_report_str += "_qha"


                decision_obj = TradingDecision(
                    decision_type=decision_type, # Use the final decision_type
                    confidence=confidence,
                    reasoning=final_reasoning,
                    timestamp=datetime.now(),
                    parameters={
                        "input_factors_raw": factors, # Original raw factors before validation
                        "factors_used_in_logic": factors_for_decision, # Factors after validation, PT, selection
                        "market_data_snapshot": validated_market_data,
                        "position_state_snapshot": position_state,
                        "min_confidence_setting": effective_min_confidence,
                        "qar_main_quantum_path_attempted": should_use_qar_main_quantum_path,
                        "qha_context_features_for_predict": market_features_array_for_qha_predict.tolist()
                    },
                    metadata=metadata
                )
                self._update_decision_history(decision_obj) # decision_obj.id is created here

                # --- QHA INTEGRATION: Store context for QHA.update() in provide_feedback ---
                # This block was already present and largely correct in your code.
                # Ensuring it uses the right variables for clarity.
                if self.hedge_algorithm:
                    # This is the DICT of market context for QHA's update method's adaptive LR.
                    market_context_dict_for_qha_update = {
                        'volatility': validated_market_data.get('volatility', 0.5),
                        'trend_strength': validated_market_data.get('trend', 0.0), # QHA's _calculate_adaptive_lr uses 'trend_strength'
                        'trend_direction': validated_market_data.get('trend_direction', 0.0),
                        'volume': validated_market_data.get('volume', 0.0)
                        # Add other keys if QHA's _calculate_adaptive_learning_rate is extended
                    }

                    # QAR's factor values (probabilities) at the time of decision,
                    # ordered as QHA experts. These will be QHA's 'expert_signals'.
                    # 'self.factors' should hold the canonical order of factors QHA was initialized with.
                    ordered_qha_expert_names = self.factors # Use QAR's canonical ordered list of factors

                    # Use 'validated_factors_as_probs' for consistency, as these are the direct inputs
                    # that QAR validated before any further processing like PT enhancement or factor selection.
                    # QHA should learn based on the "rawer" validated signal of each of its experts.
                    qar_factor_values_for_qha_update = np.array([
                        validated_factors_as_probs.get(f_name, 0.5) for f_name in ordered_qha_expert_names
                    ], dtype=np.float64)
                    
                    if len(qar_factor_values_for_qha_update) == self.hedge_algorithm.num_experts:
                        self.qha_context_for_feedback[decision_obj.id] = {
                            'market_features_dict_for_qha_update': market_context_dict_for_qha_update,
                            'qar_factor_values_as_expert_signals': qar_factor_values_for_qha_update
                        }
                    else:
                        self.logger.error(
                            f"QHA Context Store for {decision_obj.id}: Mismatch QAR factors ({len(ordered_qha_expert_names)}) "
                            f"and QHA experts ({self.hedge_algorithm.num_experts}). Cannot store QHA update context."
                        )
                # --- END QHA FEEDBACK CONTEXT ---

                execution_time_ms = (time.perf_counter() - start_time_perf) * 1000
                self.logger.info(
                    f"Decision: {decision_obj.decision_type.name} with confidence {decision_obj.confidence:.2f} "
                    f"in {execution_time_ms:.2f}ms via {final_method_report_str}"
                )
                metadata['execution_time_ms'] = execution_time_ms

                # Determine weights to return in the final dictionary
                # If QHA weights were used by the decision path, that path's metadata should have 'qha_weights_used_map' or similar
                # Otherwise, fallback to QAR's regime or base weights.
                final_weights_for_output_dict = metadata.get('final_weights_applied_map', # Check if _classical or _quantum path provided this
                                    {f_name: self.factor_weights.get(f_name, 0.0) for f_name in factors_for_decision.keys()})


                return {
                    'decision_type': decision_obj.decision_type.name,
                    'action': decision_obj.decision_type.name,
                    'confidence': decision_obj.confidence,
                    'actionable': decision_obj.confidence >= effective_min_confidence and decision_obj.decision_type != DecisionType.HOLD,
                    'explanation': decision_obj.reasoning,
                    'regime': regime_enum.value,
                    'timestamp': decision_obj.timestamp.timestamp(),
                    'method': final_method_report_str, # Use the constructed detailed method string
                    'factors': factors_for_decision,  # Factors after PT enhancement and selection
                    'weights': final_weights_for_output_dict, # Weights actually used for aggregation
                    'metadata': metadata,
                    'decision_id': decision_obj.id
                }

        except Exception as e:
            self.logger.error(f"Critical error in make_decision: {str(e)}", exc_info=True)
            # Ensure the return dict in case of error also has all expected keys for consistency, if possible
            return {
                'decision_type': 'HOLD', 'action': 'HOLD', 'confidence': 0.0, 'actionable': False,
                'explanation': f"Error in decision making: {str(e)}", 'regime': 'unknown',
                'timestamp': datetime.now().timestamp(), 'method': 'error_make_decision',
                'factors': {}, 'weights': {}, 'metadata': {'error_details': str(e)},
                'decision_id': uuid.uuid4().hex # Generate a new UUID for error cases
            }

    def _update_adaptive_reference(self,
                                    asset_id: str,
        current_price: float,
        window_size: int = 20,
        alpha: float = 0.05):
        """
        Update adaptive reference points for Prospect Theory.

        Args:
            asset_id: Identifier for the asset
            current_price: Current market price
            window_size: Window size for moving average calculations
            alpha: Weight for exponential moving average
        """
        try:
            # Initialize storage structures if needed
            if not hasattr(self, 'reference_points'):
                self.reference_points = {}
                self.reference_weights = {}
                self.ref_point_history = {}

            # Initialize history if needed
            if asset_id not in self.ref_point_history:
                self.ref_point_history[asset_id] = []

            # Add current price to history
            self.ref_point_history[asset_id].append(current_price)

            # Trim history to window size
            if len(self.ref_point_history[asset_id]) > window_size:
                self.ref_point_history[asset_id] = self.ref_point_history[asset_id][-window_size:]

            # Calculate different reference points
            history = self.ref_point_history[asset_id]

            # Current/last price
            last_price = history[-1]

            # Simple moving average
            ma_price = sum(history) / len(history)

            # Exponential moving average
            if len(history) > 1:
                # Use existing EMA if available
                ema_key = f"{asset_id}_ema"
                if ema_key in self.ref_point_history:
                    prev_ema = self.ref_point_history[ema_key]
                    ema_price = alpha * current_price + (1 - alpha) * prev_ema
                else:
                    # Initialize with first value
                    ema_price = history[0]

                # Store updated EMA
                self.ref_point_history[ema_key] = ema_price
            else:
                ema_price = current_price
                self.ref_point_history[f"{asset_id}_ema"] = ema_price

            # High watermark (highest price seen)
            high_watermark = max(history)

            # Update reference points with proper weighting
            self._add_reference_point(asset_id, last_price, 0.4, "last")
            self._add_reference_point(asset_id, ma_price, 0.3, "ma")
            self._add_reference_point(asset_id, ema_price, 0.2, "ema")
            self._add_reference_point(asset_id, high_watermark, 0.1, "high")

            # Log update if in debug mode
            if self._debug:
                self.logger.debug(
                    f"Updated reference points for {asset_id}: "
                    f"last={last_price:.4f}, ma={ma_price:.4f}, ema={ema_price:.4f}, high={high_watermark:.4f}"
                )

        except Exception as e:
            self.logger.warning(f"Error updating adaptive reference points: {str(e)}")
            # In case of error, add current price as a single reference point
            if hasattr(self, 'reference_points'):
                self._add_reference_point(asset_id, current_price, 1.0, "fallback")

    def _update_adaptive_reference_accelerated(self,
                                               asset_id: str,
                                               current_price: float,
                                               window_size: int = 20,
                                               alpha: float = 0.05):
        """
        Hardware-accelerated version of adaptive reference point updating.

        Args:
            asset_id: Identifier for the asset
            current_price: Current market price
            window_size: Window size for moving average calculations
            alpha: Weight for exponential moving average
        """
        # Verify hardware acceleration is available
        if hasattr(self, 'hardware_accelerator') and self.hardware_accelerator is not None:
            try:
                # Check if numpy is available through hardware accelerator
                if hasattr(self.hardware_accelerator, 'np'):
                    np = self.hardware_accelerator.np

                    # Initialize storage structures if needed
                    if not hasattr(self, 'reference_points'):
                        self.reference_points = {}
                        self.reference_weights = {}
                        self.ref_point_history = {}

                    # Initialize history if needed - use numpy arrays
                    if asset_id not in self.ref_point_history:
                        self.ref_point_history[asset_id] = np.array([], dtype=np.float32)

                    # Get current history as numpy array
                    history = self.ref_point_history[asset_id]

                    # Append current price
                    history = np.append(history, current_price)

                    # Trim history to window size
                    if len(history) > window_size:
                        history = history[-window_size:]

                    # Store updated history
                    self.ref_point_history[asset_id] = history

                    # Calculate reference points using numpy operations
                    last_price = history[-1]
                    ma_price = np.mean(history)
                    high_watermark = np.max(history)

                    # Calculate EMA efficiently
                    ema_key = f"{asset_id}_ema"
                    if ema_key in self.ref_point_history:
                        prev_ema = self.ref_point_history[ema_key]
                        ema_price = alpha * current_price + (1 - alpha) * prev_ema
                    else:
                        # Initialize with first value if available, otherwise current price
                        ema_price = history[0] if len(history) > 0 else current_price

                    # Store updated EMA
                    self.ref_point_history[ema_key] = ema_price

                    # Update reference points with proper weighting
                    self._add_reference_point(asset_id, float(last_price), 0.4, "last")
                    self._add_reference_point(asset_id, float(ma_price), 0.3, "ma")
                    self._add_reference_point(asset_id, float(ema_price), 0.2, "ema")
                    self._add_reference_point(asset_id, float(high_watermark), 0.1, "high")

                    # Log update if in debug mode
                    if self._debug:
                        self.logger.debug(
                            f"Updated reference points (accelerated) for {asset_id}: "
                            f"last={last_price:.4f}, ma={ma_price:.4f}, ema={ema_price:.4f}, high={high_watermark:.4f}"
                        )

                    return
            except Exception as e:
                self.logger.warning(f"Error in accelerated reference point update: {str(e)}")

        # Fall back to standard implementation if acceleration fails
        return self._update_adaptive_reference(asset_id, current_price, window_size, alpha)

    def _add_reference_point(self,
                             asset_id: str,
                             value: float,
                             weight: float = 1.0,
                             name: str = None):
        """
        Add or update a reference point for an asset.

        Args:
            asset_id: Identifier for the asset
            value: Reference point value
            weight: Weight for this reference point
            name: Optional name identifier for this reference point
        """
        try:
            # Make sure the data structures exist
            if not hasattr(self, 'reference_points'):
                self.reference_points = {}
                self.reference_weights = {}

            if asset_id not in self.reference_points:
                self.reference_points[asset_id] = []
                self.reference_weights[asset_id] = []

            # Add or update reference point
            updated = False
            for i, (ref_name, _) in enumerate(self.reference_points[asset_id]):
                if ref_name == name:
                    # Update existing reference point
                    self.reference_points[asset_id][i] = (name, value)
                    self.reference_weights[asset_id][i] = weight
                    updated = True
                    break

            # Add new if not updated
            if not updated:
                self.reference_points[asset_id].append((name, value))
                self.reference_weights[asset_id].append(weight)

            # Normalize weights to sum to 1.0
            total_weight = sum(self.reference_weights[asset_id])
            if total_weight > 0:
                self.reference_weights[asset_id] = [w / total_weight for w in self.reference_weights[asset_id]]

        except Exception as e:
            self.logger.warning(f"Error adding reference point: {str(e)}")

    def _get_reference_points(self, asset_id: str) -> Tuple[List[float], List[float]]:
        """
        Get reference points and weights for an asset.

        Args:
            asset_id: Identifier for the asset

        Returns:
            Tuple of (reference_points, weights)
        """
        try:
            if not hasattr(self, 'reference_points') or asset_id not in self.reference_points:
                return [], []

            # Extract values and weights
            values = [val for _, val in self.reference_points[asset_id]]
            weights = self.reference_weights[asset_id].copy()

            return values, weights
        except Exception as e:
            self.logger.warning(f"Error getting reference points: {str(e)}")
            return [], []

    def _optimize_position_sizing(self,
                                  decision_type: DecisionType,
                                  confidence: float,
                                  current_value: float,
                                  reference_value: float,
                                  max_position: float = 1.0) -> float:
        """
        Optimize position sizing using Prospect Theory principles.

        Args:
            decision_type: Type of trading decision (enum)
            confidence: Base decision confidence [0,1]
            current_value: Current portfolio/position value
            reference_value: Reference value for PT evaluation
            max_position: Maximum allowed position size [0,1]

        Returns:
            Optimized position size [0,max_position]
        """
        # If no PT available, use the confidence directly
        if not hasattr(self, 'quantum_pt') or self.quantum_pt is None:
            return min(confidence, max_position)

        try:
            # Convert enum to string for backward compatibility
            decision = decision_type.name.lower()

            # Calculate PT value of current state
            relative_value = current_value - reference_value
            pt_value = self.quantum_pt.value_function(relative_value)

            # Base position size on confidence
            base_position = confidence * max_position

            # Adjust based on PT principles and decision type
            if decision in ["buy", "increase"]:
                if relative_value < 0:
                    # In loss domain: loss aversion reduces position size
                    # The deeper in losses, the more we reduce position size
                    loss_adjustment = max(0.5, 1.0 + 0.5 * pt_value)  # pt_value is negative here
                    adjusted_position = base_position * loss_adjustment
                else:
                    # In gain domain: risk aversion
                    risk_adjustment = min(1.2, 1.0 + 0.1 * pt_value)
                    adjusted_position = base_position * risk_adjustment

            elif decision in ["sell", "decrease", "exit"]:
                if relative_value > 0:
                    # In gain domain: risk aversion increases selling
                    risk_adjustment = min(1.5, 1.0 + 0.25 * pt_value)
                    adjusted_position = base_position * risk_adjustment
                else:
                    # In loss domain: loss aversion makes us reluctant to sell (disposition effect)
                    disposition_adjustment = max(0.6, 1.0 + 0.4 * pt_value)  # pt_value is negative here
                    adjusted_position = base_position * disposition_adjustment

            elif decision == "hedge":
                # For hedging, use inverted adjustments - more hedging in gain domain to protect gains
                # Less hedging in loss domain due to loss aversion
                if relative_value > 0:
                    # In gain domain: risk aversion increases hedging to protect gains
                    hedge_adjustment = min(1.4, 1.0 + 0.2 * pt_value)
                    adjusted_position = base_position * hedge_adjustment
                else:
                    # In loss domain: less hedging (tendency to double down instead)
                    hedge_adjustment = max(0.7, 1.0 + 0.3 * pt_value)  # pt_value is negative
                    adjusted_position = base_position * hedge_adjustment

            else:  # hold
                adjusted_position = 0.0

            # Ensure position is in bounds
            position_size = max(0.0, min(max_position, adjusted_position))

            # Log position sizing if in debug mode
            if self._debug:
                self.logger.debug(
                    f"Position sizing for {decision}: confidence={confidence:.2f}, "
                    f"base_size={base_position:.2f}, adjusted_size={position_size:.2f}, "
                    f"pt_value={pt_value:.2f}, relative_value={relative_value:.2f}"
                )

            return position_size

        except Exception as e:
            self.logger.warning(f"Error in position sizing optimization: {str(e)}")
            # Fall back to confidence-based position sizing
            return min(confidence, max_position)


    @quantum_accelerated(use_hw_accel=True, hw_batch_size=1, device_shots=512)
    def _optimize_position_sizing_accelerated(self,
                                              decision: str,
                                              confidence: float,
                                              current_value: float,
                                              reference_value: float,
                                              max_position: float = 1.0) -> float:
        """
        Hardware-accelerated version of position sizing optimization.

        Args:
            decision: Decision type string ('buy', 'sell', 'hold')
            confidence: Base decision confidence [0,1]
            current_value: Current portfolio/position value
            reference_value: Reference value for PT evaluation
            max_position: Maximum allowed position size [0,1]

        Returns:
            Optimized position size [0,max_position]
        """
        # Verify hardware acceleration and quantum PT are available
        if (hasattr(self, 'hardware_accelerator') and self.hardware_accelerator is not None
                and hasattr(self, 'quantum_pt') and self.quantum_pt is not None):
            try:
                # Calculate PT value of current state using accelerated version
                relative_value = current_value - reference_value

                # Use accelerated value function if available
                if hasattr(self.quantum_pt, 'value_function_accelerated'):
                    pt_value = self.quantum_pt.value_function_accelerated(relative_value)
                else:
                    pt_value = self.quantum_pt.value_function(relative_value)

                # Base position size on confidence
                base_position = confidence * max_position

                # Get hardware-accelerated math functions if available
                if hasattr(self.hardware_accelerator, 'np'):
                    np = self.hardware_accelerator.np
                    max_func = np.maximum
                    min_func = np.minimum
                else:
                    max_func = max
                    min_func = min

                # Adjust based on PT principles using vectorized operations if possible
                decision = decision.lower()

                if decision in ["buy", "increase"]:
                    if relative_value < 0:
                        # In loss domain: loss aversion reduces position size
                        loss_adjustment = max_func(0.5, 1.0 + 0.5 * pt_value)
                        adjusted_position = base_position * loss_adjustment
                    else:
                        # In gain domain: risk aversion
                        risk_adjustment = min_func(1.2, 1.0 + 0.1 * pt_value)
                        adjusted_position = base_position * risk_adjustment

                elif decision in ["sell", "decrease", "exit"]:
                    if relative_value > 0:
                        # In gain domain: risk aversion increases selling
                        risk_adjustment = min_func(1.5, 1.0 + 0.25 * pt_value)
                        adjusted_position = base_position * risk_adjustment
                    else:
                        # In loss domain: disposition effect
                        disposition_adjustment = max_func(0.6, 1.0 + 0.4 * pt_value)
                        adjusted_position = base_position * disposition_adjustment
                
                elif decision == "hedge":
                    if relative_value > 0:
                        # In gain domain: protection
                        hedge_adjustment = min_func(1.4, 1.0 + 0.2 * pt_value)
                        adjusted_position = base_position * hedge_adjustment
                    else:
                        # In loss domain: less hedging
                        hedge_adjustment = max_func(0.7, 1.0 + 0.3 * pt_value)
                        adjusted_position = base_position * hedge_adjustment
                else:  # hold
                    adjusted_position = 0.0

                # Ensure position is in bounds
                position_size = max_func(0.0, min_func(max_position, adjusted_position))

                # Convert to standard Python float if necessary
                if hasattr(position_size, 'item'):
                    position_size = position_size.item()

                # Log position sizing if in debug mode
                if self._debug:
                    self.logger.debug(
                        f"Position sizing (accelerated) for {decision}: confidence={confidence:.2f}, "
                        f"base_size={base_position:.2f}, adjusted_size={position_size:.2f}, "
                        f"pt_value={pt_value:.2f}, relative_value={relative_value:.2f}"
                    )

                return float(position_size)

            except Exception as e:
                self.logger.warning(f"Error in accelerated position sizing: {str(e)}")
                
        # Fall back to non-accelerated version
        if isinstance(decision, DecisionType):
            decision_type = decision
        else:
            try:
                decision_type = DecisionType[decision.upper()]
            except (KeyError, AttributeError):
                decision_type = DecisionType.HOLD
                
        return self._optimize_position_sizing(decision_type, confidence, current_value, reference_value, max_position)

    @quantum_accelerated(use_hw_accel=True, hw_batch_size=16, device_shots=1024)
    def _quantum_decision(
        self,
        factor_probabilities: Dict[str, float],
        market_data: Dict[str, Any] = None,
        position_state: Dict[str, Any] = None,
        target_confidence_threshold: float = 0.3,
        decision_params: Dict[str, Any] = None,
        market_features_for_qha_predict: Optional[np.ndarray] = None
    ) -> Tuple[DecisionType, float, Dict[str, Any]]:
        """Make decisions using quantum computing with enhanced circuit integration.
        
        This method integrates QuantumLMSR for market scoring, 
        QuantumHedgeAlgorithm for weight optimization,
        and Quantum Prospect Theory for risk adjustment,
        along with quantum circuits for market regime analysis, pattern recognition,
        and decision optimization.
        
        Args:
            factor_probabilities: Dictionary of factor name -> probability value
            market_data: Dictionary containing market data
            position_state: Dictionary containing position information
            target_confidence_threshold: Target confidence threshold
            decision_params: Additional parameters for decision-making
            market_features_for_qha_predict: Features for QHA predict method
            
        Returns:
            Tuple of (decision_type, confidence, decision_info)
        """
        # Generate cache key for the entire decision process
        if market_data is not None and position_state is not None and market_features_for_qha_predict is not None:
            cache_key = f"quantum_decision_{hash(str(factor_probabilities))}_{hash(str(market_data))}_{hash(str(position_state))}"
            
            # Check cache
            if hasattr(self, 'decision_cache') and cache_key in self.decision_cache:
                self.logger.debug(f"Quantum decision cache hit")
                cached_result = self.decision_cache[cache_key]
                return cached_result[0], cached_result[1], cached_result[2]
        
        position_state = position_state or {}
        decision_params = decision_params or {}
        
        # Existing variable initializations
        current_pos_open = position_state.get('in_position', position_state.get('position_open', False))
        current_pos_direction = position_state.get('position_direction', 0)
        asset_id = market_data.get('asset_id', 'default') if market_data else 'default'
        market_phase_str = market_data.get('panarchy_phase', market_data.get('regime', 'unknown')) if market_data else 'unknown'
        market_phase = MarketPhase.from_string(market_phase_str)
        use_accelerated = decision_params.get('use_accelerated', self.hw_accelerator is not None)
        start_time = time.perf_counter()
        
        # Initialize if not already initialized
        if not hasattr(self, 'qlmsr') or self.qlmsr is None:
            self._initialize_quantum_lmsr()
        if not hasattr(self, 'qha') or self.qha is None:
            self._initialize_quantum_hedge()
        if not hasattr(self, 'qpt') or self.qpt is None:
            self._initialize_quantum_pt()
        if not hasattr(self, 'circuit_cache'):
            self.circuit_cache = {}
        if not hasattr(self, 'decision_cache'):
            self.decision_cache = {}
            
        # Get shots configuration from the accelerator decorator and make it available class-wide
        self.shots_for_qnodes = getattr(self, '_saved_device_shots', 1024)  # Default to 1024 if not set
        self.logger.debug(f"Using shots={self.shots_for_qnodes} for QNode calls in this run")
        
        # Fallback if QLMSR component is not available
        if not hasattr(self, 'quantum_lmsr') or self.quantum_lmsr is None:
            self.logger.warning("QuantumLMSR not available, falling back to classical in _quantum_decision.")
            self.quantum_failure_count += 1 if hasattr(self, 'quantum_failure_count') else 0
            # Pass market_features_for_qha_predict to the classical fallback
            return self._classical_decision(factor_probabilities, market_data, position_state, target_confidence_threshold, decision_params, market_features_for_qha_predict)
        
        # Create metadata storage for circuit analysis results
        metadata = {}
        
        # Check if quantum circuits are available
        quantum_circuits_available = hasattr(self, 'circuits') and self.circuits and self.quantum_available
        if not quantum_circuits_available:
            self.logger.info("Quantum circuits not available, proceeding with standard quantum decision process.")
        else:
            self.logger.info("Enhancing decision process with quantum circuits for market analysis and optimization.")
        
        # Convert factor probabilities to factor array
        # Only use factors that have non-zero values
        factor_names = []
        factor_values = []
        
        for name, value in factor_probabilities.items():
            if abs(value) > 1e-6:  # Filter out negligible factors
                factor_names.append(name)
                factor_values.append(value)
        
        # Normalize factor values to [-1, 1] range if they aren't already
        factor_array = np.array(factor_values)
        if np.max(np.abs(factor_array)) > 1.0:
            factor_array = np.clip(factor_array / np.max(np.abs(factor_array)), -1.0, 1.0)
            
        # Prepare for quantum circuit analysis if available
        if quantum_circuits_available:
            try:
                # Convert factor probabilities to numpy array for quantum processing
                factors_array = np.array(list(factor_probabilities.values()))
                # Ensure array has the right dimensions for quantum circuits
                padded_factors = np.zeros(self.qubits)
                padded_factors[:min(len(factors_array), self.qubits)] = factors_array[:min(len(factors_array), self.qubits)]
                
                # Step 1: Analyze market regime using quantum Fourier transform
                regime_analysis = self._analyze_market_regime(padded_factors)
                metadata['regime_analysis'] = regime_analysis
                
                # Step 2: Recognize patterns using quantum memory
                pattern_recognition = self._recognize_patterns(padded_factors)
                metadata['pattern_recognition'] = pattern_recognition
                
                # Step 3: Optimize decision using quantum amplitude amplification
                decision_result = self._optimize_decision(padded_factors, regime_analysis)
                metadata['decision_optimization'] = decision_result
                
                # Store current pattern in memory for future reference
                current_timestamp = datetime.now().timestamp()
                self._store_pattern(
                    circuit_name='quantum_decision',
                    input_values=padded_factors, 
                    result=decision_result,
                    metadata={
                        'timestamp': current_timestamp,
                        'regime': regime_analysis['regime'],
                        'decision': decision_result['action'],
                        'market_data_snapshot': {k: v for k, v in market_data.items() if k in ['trend', 'volatility', 'momentum', 'volume']} if market_data else {}
                    }
                )
            except Exception as e:
                self.logger.error(f"Error in quantum circuit processing: {e}")
                # Continue with standard quantum processing even if quantum circuit part fails
        
        # QAR's own regime weights (used by QLMSR's quantity update logic)
        regime_weights = self._get_regime_specific_weights(market_phase)

        # --- QHA INTEGRATION: Get dynamic factor weights from QuantumHedgeAlgorithm ---
        hedge_weights_from_qha: Optional[np.ndarray] = None
        used_qha_weights_this_call = False # Flag specific to this decision call
        qha_weights_map_for_metadata: Dict[str, float] = {}

        if hasattr(self, 'hedge_algorithm') and self.hedge_algorithm is not None:
            if market_features_for_qha_predict is not None:
                try:
                    hedge_weights_from_qha = self.hedge_algorithm.predict(market_features_for_qha_predict)
                    self.logger.debug(f"_quantum_decision: QHA contextual predict weights: {hedge_weights_from_qha}")
                    used_qha_weights_this_call = True
                except Exception as e_qha_predict:
                    self.logger.warning(f"Error getting contextual weights from QHA in _quantum_decision: {e_qha_predict}")
                    # Optionally, could try predict without features as a fallback
                    # For now, if contextual predict fails, we won't use QHA weights here.
                    hedge_weights_from_qha = None
                    used_qha_weights_this_call = False
            else: # No contextual features for QHA, get its classical weights
                self.logger.debug("_quantum_decision: No market_features_for_qha_predict; getting QHA classical weights.")
                try:
                    hedge_weights_from_qha = self.hedge_algorithm.predict() # QHA's current base weights
                    used_qha_weights_this_call = True # Still used QHA's output
                except Exception as e_qha_predict_no_feat:
                    self.logger.warning(f"Error getting classical weights from QHA in _quantum_decision: {e_qha_predict_no_feat}")
                    hedge_weights_from_qha = None
                    used_qha_weights_this_call = False
            
            # If QHA weights were obtained, map them for metadata
            if used_qha_weights_this_call and hedge_weights_from_qha is not None:
                # Use self.factors as the canonical ordered list of QAR factors / QHA experts
                ordered_qha_expert_names = self.factors
                if len(ordered_qha_expert_names) == len(hedge_weights_from_qha):
                    qha_weights_map_for_metadata = {
                        name: hedge_weights_from_qha[i] for i, name in enumerate(ordered_qha_expert_names)
                    }
                else:
                    self.logger.warning(
                        f"_quantum_decision: Mismatch QAR factors ({len(ordered_qha_expert_names)}) "
                        f"and obtained QHA weights ({len(hedge_weights_from_qha)}). Cannot map for metadata. QHA weights will not be applied."
                    )
                    used_qha_weights_this_call = False # Invalidate usage if mapping fails
                    hedge_weights_from_qha = None # Don't use potentially misaligned weights
        # --- END QHA INTEGRATION ---

        # Add QLMSR-based market scoring (if available)
        qlmsr_decision = None
        qlmsr_confidence = 0.0
        qlmsr_info = {}
        
        # --- Use QLMSR for market scoring with updated logic ---
        # Use self.factors as the canonical list of factor names for QLMSR processing consistency
        # This ensures alignment with QHA expert ordering if QHA weights are used.
        factor_names_list = self.factors
        
        # Ensure factor_quantities is initialized for all factors QLMSR will process
        for factor_name in factor_names_list:
            if factor_name not in self.factor_quantities:
                self.logger.debug(f"_quantum_decision: Initializing quantity for new/unseen factor '{factor_name}' to 0.0 for QLMSR.")
                self.factor_quantities[factor_name] = 0.0
        
        current_quantities_list = [self.factor_quantities[f] for f in factor_names_list]

        # Fallback if factor_names_list or current_quantities_list is empty
        if not factor_names_list or not current_quantities_list:
            self.logger.warning("_quantum_decision: No factors or quantities available for QLMSR processing. Falling back.")
            self.quantum_failure_count += 1 if hasattr(self, 'quantum_failure_count') else 0
            return self._classical_decision(factor_probabilities, market_data, position_state, target_confidence_threshold, decision_params, market_features_for_qha_predict)

        try:
            prior_probabilities_list = self.quantum_lmsr.get_all_market_probabilities(current_quantities_list)
            # Debug input factors to understand why confidence is consistent
            self.logger.debug(f"INPUT DEBUG: Start of test run with factor_names={factor_names_list}")
            self.logger.debug(f"INPUT DEBUG: Factor probabilities before processing: {factor_probabilities}")
            
            processed_factor_probabilities_list = [factor_probabilities.get(f, 0.5) for f in factor_names_list]
            self.logger.debug(f"INPUT DEBUG: Processed factor probs: {processed_factor_probabilities_list}")
            self.logger.debug(f"INPUT DEBUG: Prior probabilities: {prior_probabilities_list}")
            
            information_gain = self.quantum_lmsr.calculate_information_gain(
                prior_probabilities=prior_probabilities_list,
                posterior_probabilities=processed_factor_probabilities_list
            )
            self.logger.debug(f"INPUT DEBUG: Calculated information_gain={information_gain:.6f}")
            
            # Update quantities based on factor probabilities
            # Process factors in the canonical order to match QLMSR outcomes
            for i, factor_name in enumerate(factor_names_list):
                if factor_name in factor_probabilities:
                    target_prob = processed_factor_probabilities_list[i]
                    base_w = self.factor_weights.get(factor_name, 1.0)
                    regime_w = regime_weights.get(factor_name, 1.0)
                    combined_weight = base_w * regime_w # QAR's internal weighting for QLMSR update
                    try:
                        # Calculate cost to move based on CURRENT quantities
                        # Pass the quantities list which is being updated in place
                        cost_to_move = self.quantum_lmsr.calculate_cost_to_move(
                            current_quantities=current_quantities_list,
                            target_probability=target_prob,
                            outcome_index=i
                        )
                        # Update quantity IN PLACE
                        current_quantities_list[i] += cost_to_move * combined_weight
                    except Exception as e_cost:
                        self.logger.warning(f"Error updating quantity for factor {factor_name} using cost_to_move: {e_cost}")

            # After updating all quantities, update self.factor_quantities and get final market probabilities
            self.factor_quantities = dict(zip(factor_names_list, current_quantities_list))
            market_probabilities_list = self.quantum_lmsr.get_all_market_probabilities(current_quantities_list) # Use updated quantities

            if not isinstance(market_probabilities_list, (list, np.ndarray)) or not market_probabilities_list:
                self.logger.warning(f"_quantum_decision: QLMSR output 'market_probabilities_list' invalid after quantity update. Falling back.")
                self.quantum_failure_count += 1 if hasattr(self, 'quantum_failure_count') else 0
                # Pass market_features_for_qha_predict to the classical fallback
                return self._classical_decision(factor_probabilities, market_data, position_state, target_confidence_threshold, decision_params, market_features_for_qha_predict)

            # Calculate consistency factor
            if processed_factor_probabilities_list: # Check not empty
                consistency = 1.0 - np.std(processed_factor_probabilities_list) if len(processed_factor_probabilities_list) > 1 else 1.0
                consistency = max(0.0, min(1.0, consistency))
            else:
                consistency = 0.0 # Should not happen if factor_names_list is not empty

            # --- Convert QLMSR's market_probabilities_list to a single buy_sell_signal ---
            # Apply QHA weights if available and valid
            buy_sell_signal: float
            if used_qha_weights_this_call and hedge_weights_from_qha is not None and \
               len(hedge_weights_from_qha) == len(market_probabilities_list) and \
               len(market_probabilities_list) > 0: # Added check for non-empty
                
                self.logger.debug(f"_quantum_decision: Applying QHA weights ({hedge_weights_from_qha}) to QLMSR probabilities ({market_probabilities_list}) for buy_sell_signal.")
                # Convert QLMSR factor probabilities [0,1] to directional signals [-1,1]
                # This assumes market_probabilities_list is ordered according to self.factors / QHA experts
                factor_directional_signals = (np.array(market_probabilities_list) - 0.5) * 2.0
                
                weighted_sum_of_directional_signals = np.sum(factor_directional_signals * hedge_weights_from_qha)
                sum_of_qha_applied_weights = np.sum(hedge_weights_from_qha)
                
                if sum_of_qha_applied_weights > 1e-9: # Avoid division by zero
                    buy_sell_signal = weighted_sum_of_directional_signals / sum_of_qha_applied_weights
                else:
                    buy_sell_signal = 0.0
                    self.logger.warning("_quantum_decision: Sum of QHA weights is near zero. Defaulting buy_sell_signal to 0.")
            else: # Fallback: QHA weights not used or invalid, use QAR's own weights for aggregation
                if used_qha_weights_this_call and hedge_weights_from_qha is not None: # Log why QHA weights weren't applied
                    self.logger.warning("_quantum_decision: QHA weights were obtained but not applied to buy_sell_signal due to length mismatch or empty QLMSR output. Using QAR's weights.")

                weighted_signals_sum = 0.0
                total_effective_weights = 0.0
                
                # Use QAR's own weights (potentially regime-adjusted) as fallback weights
                qar_fallback_weights = self._get_regime_specific_weights(market_phase)

                # Iterate using QAR's canonical factor order (self.factors)
                for i, factor_name in enumerate(self.factors):
                    if i < len(market_probabilities_list): # Ensure index is within bounds of QLMSR output
                        factor_prob = market_probabilities_list[i]
                        factor_signal_strength = (factor_prob - 0.5) * 2.0 # Convert probability to directional signal [-1, 1]

                        current_weight_for_this_factor = qar_fallback_weights.get(factor_name, 0.0) # Use QAR's weight

                        weighted_signals_sum += factor_signal_strength * current_weight_for_this_factor
                        total_effective_weights += current_weight_for_this_factor

                if total_effective_weights > 0:
                    buy_sell_signal = weighted_signals_sum / total_effective_weights
                else:
                    buy_sell_signal = 0.0
                    self.logger.warning("_quantum_decision: Fallback aggregation: total_effective_weights is zero. Resulting buy_sell_signal is 0.")
            
            # Calculate action strength for confidence calculation
            action_strength = abs(buy_sell_signal)
            
            # Calculate base confidence from quantum results
            # Log detailed input values to debug the consistent confidence issue
            self.logger.debug(f"CONFIDENCE DEBUG: information_gain={information_gain:.6f}, consistency={consistency:.6f}, action_strength={action_strength:.6f}")
            self.logger.debug(f"CONFIDENCE DEBUG: factor_probs={processed_factor_probabilities_list}")
            self.logger.debug(f"CONFIDENCE DEBUG: market_probs={market_probabilities_list}")
            
            # Calculate confidence with detailed breakdown
            info_gain_component = 0.6 * information_gain
            consistency_component = 0.4 * consistency
            action_component = (0.5 + 0.5 * action_strength)
            base_confidence = (info_gain_component + consistency_component) * action_component
            
            self.logger.debug(f"CONFIDENCE DEBUG: info_component={info_gain_component:.6f}, consist_component={consistency_component:.6f}, action_component={action_component:.6f}")
            self.logger.debug(f"CONFIDENCE DEBUG: raw_confidence={base_confidence:.6f}")
            
            # Ensure confidence is in valid range
            base_confidence = max(0.0, min(1.0, base_confidence))
            
            # Adjust confidence using quantum circuit results if available
            if 'decision_optimization' in metadata:
                decision_opt = metadata['decision_optimization']
                # Use quantum-optimized confidence if it's higher than base confidence
                quantum_confidence = decision_opt.get('confidence', 0.0)
                if quantum_confidence > base_confidence:
                    self.logger.info(f"Enhancing confidence from {base_confidence:.4f} to {quantum_confidence:.4f} using quantum optimization")
                    base_confidence = quantum_confidence
            
            # If strong pattern match found in quantum circuit analysis, boost confidence
            if 'pattern_recognition' in metadata:
                pattern_rec = metadata['pattern_recognition']
                similarity = pattern_rec.get('similarity', 0.0)
                if similarity > 0.7:
                    pattern_boost = min(0.2, similarity * 0.25)  # Cap the boost at 0.2
                    self.logger.info(f"Applying pattern recognition boost of {pattern_boost:.4f} (similarity: {similarity:.4f})")
                    base_confidence = min(1.0, base_confidence * (1.0 + pattern_boost))
                    metadata['confidence_boost'] = f"pattern_match:{similarity:.4f}"
            
            # Determine decision type from buy_sell_signal
            if buy_sell_signal > 0.3:
                decision_type_str = 'BUY'
            elif buy_sell_signal < -0.3:
                decision_type_str = 'SELL'
            else:
                decision_type_str = 'HOLD'
                
            # Store confidence before PT for metadata consistency
            confidence_before_pt_adjustment = base_confidence
        except Exception as e:
            self.logger.error(f"Error in quantum QLMSR processing: {e}")
            self.quantum_failure_count += 1 if hasattr(self, 'quantum_failure_count') else 0
            return self._classical_decision(factor_probabilities, market_data, position_state, target_confidence_threshold, decision_params, market_features_for_qha_predict)
        
        # Apply quantum prospect theory adjustments (if available)
        pt_adjusted_decision = decision_type_str
        pt_adjusted_confidence = base_confidence
        pt_info = {}
        pt_value = 0.0
        framing_effect = 0.0
        
        # Store confidence before PT for metadata
        confidence_before_pt_adjustment = base_confidence
        
        # Apply PT adjustments if we have position data
        if hasattr(self, 'qpt') and self.qpt is not None and position_state is not None:
            try:
                # Current position information
                in_position = position_state.get('in_position', False)
                current_profit = position_state.get('current_profit', 0.0)
                entry_price = position_state.get('entry_price', 0.0)
                
                # Current market information
                current_price = market_data.get('close', 0.0) if market_data else 0.0
                
                # Apply PT adjustment
                pt_result = self.qpt.adjust_decision(
                    decision_type_str,
                    base_confidence,
                    in_position=in_position,
                    current_profit=current_profit,
                    current_price=current_price,
                    entry_price=entry_price,
                    risk_tolerance=self.risk_tolerance
                )
                
                # Extract PT-adjusted decision and confidence
                pt_adjusted_decision = pt_result['adjusted_decision']
                pt_adjusted_confidence = pt_result['adjusted_confidence']
                pt_info = pt_result
                pt_value = pt_result.get('pt_value', 0.0)
                framing_effect = pt_result.get('framing_effect', 0.0)
                
                self.logger.debug(
                    f"PT adjustment: {decision_type_str} ({base_confidence:.4f}) -> "
                    f"{pt_adjusted_decision} ({pt_adjusted_confidence:.4f})"
                )
                
            except Exception as e:
                self.logger.error(f"Quantum PT adjustment failed: {str(e)}")
                # If PT adjustment fails, use the original decision
                pt_adjusted_decision = decision_type_str
                pt_adjusted_confidence = base_confidence
        
        # Handle all valid decision types (BUY, SELL, HOLD, EXIT, HEDGE, INCREASE, DECREASE)
        valid_decision_types = {
            'BUY': DecisionType.BUY,
            'SELL': DecisionType.SELL,
            'HOLD': DecisionType.HOLD,
            'EXIT': DecisionType.EXIT,
            'HEDGE': DecisionType.HEDGE,
            'INCREASE': DecisionType.INCREASE,
            'DECREASE': DecisionType.DECREASE
        }
        
        # Determine final decision type
        # Try to convert string to DecisionType enum
        try:
            if isinstance(pt_adjusted_decision, str):
                final_decision = valid_decision_types.get(pt_adjusted_decision.upper(), DecisionType.HOLD)
            else:
                final_decision = pt_adjusted_decision
                
            # Validate that final_decision is a DecisionType enum
            if not isinstance(final_decision, DecisionType):
                self.logger.warning(f"Invalid decision type: {final_decision}, defaulting to HOLD")
                final_decision = DecisionType.HOLD
                pt_adjusted_confidence = 0.5  # Set moderate confidence for fallback
                
        except (KeyError, TypeError, AttributeError) as e:
            # Fallback to a default decision if conversion fails
            self.logger.warning(f"Invalid decision type: {pt_adjusted_decision}, error: {e}, defaulting to HOLD")
            final_decision = DecisionType.HOLD
            pt_adjusted_confidence = 0.5  # Set moderate confidence for fallback
        
        # Ensure confidence is in valid range [0, 1]
        final_confidence = max(0.0, min(1.0, pt_adjusted_confidence))
        
        # Apply confidence threshold if specified
        if final_confidence < target_confidence_threshold:
            self.logger.debug(
                f"Decision confidence {final_confidence:.4f} below threshold {target_confidence_threshold}, "
                f"defaulting to HOLD"
            )
            final_decision = DecisionType.HOLD
            final_confidence = 0.5  # Set moderate confidence for threshold fallback
        
        # Enhanced decision info incorporating QHA and circuit analysis
        decision_info = {
            # From QHA integration
            'factor_names': factor_names_list if 'factor_names_list' in locals() else [],
            'processed_factor_probs': processed_factor_probabilities_list if 'processed_factor_probabilities_list' in locals() else [],
            'market_probabilities': market_probabilities_list if 'market_probabilities_list' in locals() else [],
            'qha_weights': qha_weights_map_for_metadata if 'qha_weights_map_for_metadata' in locals() else {},
            'used_qha_weights': used_qha_weights_this_call if 'used_qha_weights_this_call' in locals() else False,
            
            # From quantum circuit analysis
            'regime_analysis': metadata.get('regime_analysis', {}),
            'pattern_recognition': metadata.get('pattern_recognition', {}),
            'decision_optimization': metadata.get('decision_optimization', {}),
            
            # Decision data
            'buy_sell_signal': buy_sell_signal if 'buy_sell_signal' in locals() else 0.0,
            'consistency': consistency if 'consistency' in locals() else 0.0,
            'information_gain': information_gain if 'information_gain' in locals() else 0.0,
            'pt_value': pt_value,
            'framing_effect': framing_effect,
            'pt_info': pt_info,
            'initial_decision': decision_type_str,
            'initial_confidence': confidence_before_pt_adjustment,
            'adjusted_decision': pt_adjusted_decision,
            'adjusted_confidence': pt_adjusted_confidence,
            'final_confidence': final_confidence,
            'final_decision': final_decision,
            'target_threshold': target_confidence_threshold,
            'quantum_available': self.quantum_available,
            'execution_time_ms': int((time.perf_counter() - start_time) * 1000),
            'timestamp': time.time()
        }
        
        # Log the final decision
        self.logger.info(
            f"Quantum decision: {final_decision} with confidence {final_confidence:.4f} "
            f"(Market regime: {metadata.get('regime_analysis', {}).get('regime', 'unknown')})"
        )
        
        # Cache the result if caching is enabled
        if hasattr(self, 'decision_cache') and market_data is not None and position_state is not None and 'cache_key' in locals():
            self.decision_cache[cache_key] = (final_decision, final_confidence, decision_info)
            
            # Limit cache size to prevent memory issues
            if len(self.decision_cache) > 100:  # Keep last 100 decisions
                oldest_key = next(iter(self.decision_cache))
                self.decision_cache.pop(oldest_key)
        
        return final_decision, final_confidence, decision_info
        
    def _generate_quantum_pt_reasoning(
            self,
            factor_probabilities: Dict[str, float],
            prior_probabilities: List[float],
            pt_adjusted_probabilities: List[float],
            information_gain: float,
            decision_type: DecisionType,
            adjusted_confidence: float,
            market_data: Dict[str, Any] = None,
            consistency: float = 0.0,
            pt_value: float = 0.0,
            framing_effect: float = 0.0
    ) -> Tuple[Dict[str, float], str]:
        """
        Generate reasoning text and factor contributions for a PT-enhanced quantum decision.

        Args:
            factor_probabilities: Dictionary of factor probabilities
            market_probabilities: Original market-derived probabilities
            pt_adjusted_probabilities: Probabilities after PT weighting
            information_gain: Information gain from prior to posterior
            decision_type: Type of decision made
            adjusted_confidence: Final confidence after adjustments
            market_data: Market context data
            consistency: Consistency measure for factors
            pt_value: Prospect Theory value
            framing_effect: Framing effect magnitude

        Returns:
            Tuple of (factor_contributions, reasoning_text)
        """
        try:
            # Calculate factor contributions
            contributions = {}
            for factor, prob in factor_probabilities.items():
                weight = self.factor_weights.get(factor, 1.0)
                # Calculate influence as weighted deviation from neutral
                influence = weight * abs(prob - 0.5) * 2
                contributions[factor] = influence

            # Sort factors by contribution for reasoning
            top_factors = sorted(contributions.items(), key=lambda x: x[1], reverse=True)

            # Extract market context
            phase = market_data.get('panarchy_phase', market_data.get('regime', 'unknown'))
            trend = market_data.get('trend', 0)
            momentum = market_data.get('momentum', 0)
            volatility = market_data.get('volatility', 0.5)
            risk_level = market_data.get('risk_level', 0.5)

            # Generate reasoning using all components
            reasoning_parts = [
                f"QAR-PT {decision_type.name} signal with {adjusted_confidence:.3f} confidence."
            ]

            # Add top contributing factors
            if top_factors:
                factor_details = []
                for factor, influence in top_factors[:3]:  # Top 3 factors
                    value = factor_probabilities.get(factor, 0.5)
                    direction = "+" if value > 0.55 else ("-" if value < 0.45 else "~")
                    factor_details.append(f"{factor}({direction}{influence:.2f})")

                reasoning_parts.append(f"Key factors: {', '.join(factor_details)}")

            # Add market context
            reasoning_parts.append(
                f"Market context: Phase={phase}, Risk={risk_level:.2f}, "
                f"Trend={trend:.2f}, Momentum={momentum:.2f}, Volatility={volatility:.2f}"
            )

            # Add PT-specific information based on decision type
            if pt_value < 0:
                domain = "loss domain"
                if decision_type == DecisionType.BUY:
                    effect = "Loss aversion active, reducing risk-taking"
                elif decision_type == DecisionType.SELL:
                    effect = "Disposition effect active, reluctant to realize losses"
                elif decision_type == DecisionType.INCREASE:
                    effect = "Loss aversion suggests caution when increasing exposure"
                elif decision_type == DecisionType.DECREASE:
                    effect = "Disposition effect suggests reluctance to reduce positions in loss"
                elif decision_type == DecisionType.EXIT:
                    effect = "Loss aversion may delay closing positions at a loss"
                elif decision_type == DecisionType.HEDGE:
                    effect = "Loss aversion suggests hedging to prevent further losses"
                else:  # HOLD
                    effect = "Neutral effect"
            else:
                domain = "gain domain"
                if decision_type == DecisionType.BUY:
                    effect = "Risk aversion for gains, cautious approach to new positions"
                elif decision_type == DecisionType.SELL:
                    effect = "More willing to realize gains"
                elif decision_type == DecisionType.INCREASE:
                    effect = "Risk aversion for gains suggests caution when increasing exposure"
                elif decision_type == DecisionType.DECREASE:
                    effect = "Risk aversion suggests taking profits by decreasing position"
                elif decision_type == DecisionType.EXIT:
                    effect = "Risk aversion encourages securing gains by exiting"
                elif decision_type == DecisionType.HEDGE:
                    effect = "Risk aversion suggests hedging to protect accumulated gains"
                else:  # HOLD
                    effect = "Neutral effect"

            # Add PT effects to reasoning
            reasoning_parts.append(
                f"PT analysis: In {domain} (value={pt_value:.3f}). {effect}."
            )

            # Add quantum metrics
            prob_delta = sum(abs(p1 - p2) for p1, p2 in zip(prior_probabilities, pt_adjusted_probabilities))
            reasoning_parts.append(
                f"Quantum metrics: Consistency={consistency:.2f}, InfoGain={information_gain:.3f}, ProbDelta={prob_delta:.3f}"
            )

            # Join with spaces
            return contributions, " ".join(reasoning_parts)

        except Exception as e:
            self.logger.warning(f"Error generating PT reasoning: {str(e)}")

            # Simplified fallback reasoning
            return {}, f"QAR-PT {decision_type.name} decision with {adjusted_confidence:.3f} confidence."

    def _determine_decision_type(
        self,
        position_open: bool,
        position_direction: int,
        buy_sell_signal: float,
        confidence: float,
        action_strength: float
    ) -> DecisionType:
        """
        Determine decision type based on position and signals with adaptive thresholds.
        
        Args:
            position_open: Whether a position is currently open
            position_direction: Direction of the current position (1 for long, -1 for short, 0 for none)
            buy_sell_signal: Signal strength (-1.0 to 1.0)
            confidence: Confidence in the signal (0.0 to 1.0)
            action_strength: Strength of the action (0.0 to 1.0)
            
        Returns:
            DecisionType: The determined decision type
        """
        decision_type = DecisionType.HOLD
        decision_reason = []
        
        # Log input parameters for debugging
        self.logger.debug(
            f"Decision inputs - Position: {'OPEN' if position_open else 'CLOSED'}, "
            f"Direction: {'LONG' if position_direction > 0 else 'SHORT' if position_direction < 0 else 'NONE'}, "
            f"Signal: {buy_sell_signal:.4f}, Confidence: {confidence:.4f}, Strength: {action_strength:.4f}"
        )
    
        # Adaptive confidence threshold based on market conditions
        adaptive_threshold = self.decision_threshold
        if hasattr(self, 'market_volatility') and self.market_volatility > 0.7:
            # Be more conservative in high volatility
            adaptive_threshold *= 0.9
            decision_reason.append(f"High volatility: threshold adjusted to {adaptive_threshold:.4f}")
    
        if position_open:  # Existing position
            if position_direction > 0:  # Long
                if buy_sell_signal < -0.25 and confidence > adaptive_threshold * 0.9:
                    decision_type = DecisionType.EXIT
                    decision_reason.append("Strong sell signal with high confidence")
                elif buy_sell_signal > 0.4 and action_strength > 0.6 and confidence > adaptive_threshold * 0.85:
                    decision_type = DecisionType.INCREASE
                    decision_reason.append("Strong buy signal with good confidence and strength")
                elif buy_sell_signal < -0.1 and action_strength > 0.4 and confidence > adaptive_threshold * 0.8:
                    decision_type = DecisionType.DECREASE
                    decision_reason.append("Moderate sell signal with sufficient confidence")
                elif -0.3 <= buy_sell_signal <= -0.1 and 0.4 <= confidence <= 0.7:
                    decision_type = DecisionType.HEDGE
                    decision_reason.append("Moderate signal in HEDGE confidence range")
            else:  # Short (position_direction < 0)
                if buy_sell_signal > 0.25 and confidence > adaptive_threshold * 0.9:
                    decision_type = DecisionType.EXIT
                    decision_reason.append("Strong buy signal with high confidence")
                elif buy_sell_signal < -0.4 and action_strength > 0.6 and confidence > adaptive_threshold * 0.85:
                    decision_type = DecisionType.INCREASE
                    decision_reason.append("Strong sell signal with good confidence and strength")
                elif buy_sell_signal > 0.1 and action_strength > 0.4 and confidence > adaptive_threshold * 0.8:
                    decision_type = DecisionType.DECREASE
                    decision_reason.append("Moderate buy signal with sufficient confidence")
                elif 0.1 <= buy_sell_signal <= 0.3 and 0.4 <= confidence <= 0.7:
                    decision_type = DecisionType.HEDGE
                    decision_reason.append("Moderate signal in HEDGE confidence range")
        else:  # No position
            if buy_sell_signal > 0.25 and confidence > adaptive_threshold * 0.9:
                decision_type = DecisionType.BUY
                decision_reason.append("Strong buy signal with high confidence")
            elif buy_sell_signal < -0.25 and confidence > adaptive_threshold * 0.9:
                decision_type = DecisionType.SELL
                decision_reason.append("Strong sell signal with high confidence")
            elif 0.15 <= abs(buy_sell_signal) <= 0.3 and 0.6 <= confidence <= adaptive_threshold:
                decision_type = DecisionType.HEDGE
                decision_reason.append("Moderate signal in HEDGE range for new position")
    
        # If no specific condition was met, log why we're holding
        if decision_type == DecisionType.HOLD:
            if not position_open:
                if abs(buy_sell_signal) <= 0.25:
                    decision_reason.append(f"Signal strength {abs(buy_sell_signal):.4f} below threshold 0.25")
                if confidence <= adaptive_threshold * 0.9:
                    decision_reason.append(f"Confidence {confidence:.4f} below threshold {adaptive_threshold * 0.9:.4f}")
            else:  # Position is open
                decision_reason.append("No exit or adjustment conditions met")
    
        # Log the final decision with reasoning
        self.logger.info(
            f"Decision: {decision_type.name} - "
            f"Signal: {buy_sell_signal:.4f}, "
            f"Confidence: {confidence:.4f}/{adaptive_threshold:.4f}, "
            f"Strength: {action_strength:.4f}, "
            f"Reason: {'; '.join(decision_reason) or 'No specific reason'}"
        )
    
        return decision_type

    def _update_decision_history(self, decision: TradingDecision) -> None:
        """Update decision history and memory buffer."""
        # Add to decision history
        self.decision_history.append(decision)
        if len(self.decision_history) > self.memory_length:
            self.decision_history = self.decision_history[-self.memory_length:]

        # Add to memory buffer
        self.memory_buffer.append({
            'decision': decision,
            'timestamp': time.time(),
            'outcome': None  # To be filled later with feedback
        })
        if len(self.memory_buffer) > self.memory_length:
            self.memory_buffer = self.memory_buffer[-self.memory_length:]

    def _validate_factors(self, factor_values: Dict[str, float]) -> Dict[str, float]:
        """
        Validate and normalize factor values.

        Args:
            factor_values: Raw factor values

        Returns:
            Normalized factor values for all registered factors
        """
        validated_factors = {}
        neutral_default = 0.5

        for factor_name in self.factors:
            raw_value = factor_values.get(factor_name, None)

            if raw_value is None:
                self.logger.debug(f"Factor '{factor_name}' missing from input. Using default {neutral_default}.")
                validated_factors[factor_name] = neutral_default
                continue

            try:
                float_value = float(raw_value)

                if not np.isfinite(float_value):
                    self.logger.warning(f"Factor '{factor_name}' has non-finite value. Using default {neutral_default}.")
                    validated_factors[factor_name] = neutral_default
                else:
                    clamped_value = max(0.0, min(1.0, float_value))
                    if clamped_value != float_value:
                        self.logger.debug(f"Factor '{factor_name}' clipped from {float_value:.4f} to {clamped_value:.4f}")
                    validated_factors[factor_name] = clamped_value
            except (ValueError, TypeError) as e:
                self.logger.warning(f"Invalid value for factor '{factor_name}': {raw_value}. Error: {e}")
                validated_factors[factor_name] = neutral_default

        return validated_factors

    def _validate_market_data(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate market data.

        Args:
            market_data: Raw market data

        Returns:
            Validated market data with defaults for missing fields
        """
        validated_data = market_data.copy() if market_data else {}

        # Ensure required fields
        required_fields = {
            'close': None,
            'volume': 0,
            'timestamp': time.time(),
            'trend': 0.0,
            'volatility': 0.5,
            'momentum': 0.0,
            'panarchy_phase': 'unknown'
        }

        for field, default in required_fields.items():
            if field not in validated_data:
                validated_data[field] = default

        # Normalize specific fields
        if 'trend' in validated_data:
            validated_data['trend'] = max(-1.0, min(1.0, float(validated_data['trend']))) if isinstance(validated_data['trend'], (int, float)) else 0.0

        if 'momentum' in validated_data:
            validated_data['momentum'] = max(-1.0, min(1.0, float(validated_data['momentum']))) if isinstance(validated_data['momentum'], (int, float)) else 0.0

        if 'volatility' in validated_data:
            validated_data['volatility'] = max(0.0, min(1.0, float(validated_data['volatility']))) if isinstance(validated_data['volatility'], (int, float)) else 0.5

        return validated_data

    def _extract_position_state(self, position_state: Optional[Dict[str, Any]]) -> Tuple[bool, int]:
        """
        Extract position information.

        Args:
            position_state: Position state dictionary

        Returns:
            Tuple of (position_open, position_direction)
        """
        position_open = False
        position_direction = 0

        if position_state:
            try:
                position_open = bool(position_state.get('position_open', False))
                position_direction = position_state.get('position_direction', 0)

                # Normalize direction to -1, 0, 1
                if position_direction > 0:
                    position_direction = 1
                elif position_direction < 0:
                    position_direction = -1
                else:
                    position_direction = 0
            except Exception as e:
                self.logger.warning(f"Error extracting position state: {e}")

        return position_open, position_direction

    def _determine_market_regime(self, market_data: Dict[str, Any]) -> MarketPhase:
        """
        Determine current market regime.

        Args:
            market_data: Market data

        Returns:
            Market phase/regime
        """
        # Try to get panarchy phase from market data
        phase_str = market_data.get('panarchy_phase', 'unknown')
        return MarketPhase.from_string(phase_str)

    def _get_regime_specific_weights(self, regime: MarketPhase) -> Dict[str, float]:
        """
        Get factor weights specific to the current market regime.

        Args:
            regime: Current market regime

        Returns:
            Factor weights for the current regime
        """
        # Start with base weights
        weights = self.factor_weights.copy()

        # Apply regime-specific adjustments if available
        if regime in self.regime_specific_weights:
            regime_weights = self.regime_specific_weights[regime]
            for factor, weight in regime_weights.items():
                if factor in weights:
                    weights[factor] = weight

        # Normalize weights to sum to 1
        weight_sum = sum(weights.values())
        if weight_sum > 0:
            for factor in weights:
                weights[factor] /= weight_sum

        return weights




    def _classical_decision(
            self,
            factor_probabilities: Dict[str, float], # QAR's factor name -> probability map
            market_data: Dict[str, Any],
            position_state: Dict[str, Any] = None,
            target_confidence_threshold: float = 0.6,
            decision_params: Dict[str, Any] = None,
            market_features_for_qha_predict: Optional[np.ndarray] = None # EXPECTING THIS TO BE PASSED BY make_decision
        ) -> Tuple[DecisionType, float, Dict[str, Any]]:
            position_state = position_state or {}
            decision_params = decision_params or {}

            current_pos_open = position_state.get('in_position', position_state.get('position_open', False))
            current_pos_direction = position_state.get('position_direction', 0)
            asset_id = market_data.get('asset_id', 'default')
            market_phase_str = market_data.get('panarchy_phase', market_data.get('regime', 'unknown'))
            market_phase = MarketPhase.from_string(market_phase_str)
            risk_level = market_data.get('risk_level', 0.5)
            use_accelerated = decision_params.get('use_accelerated', self.hw_accelerator is not None)
            start_time = time.perf_counter()

            regime_weights = self._get_regime_specific_weights(market_phase) # QAR's own regime weights

            # --- QHA INTEGRATION: Use QHA to get dynamic factor weights ---
            # This 'hedge_weights' will be the NumPy array from QHA's predict method.
            # It's expected to be ordered according to QAR's self.factors (or self._get_ordered_qha_expert_names())
            hedge_weights_from_qha: Optional[np.ndarray] = None
            used_qha_weights_flag = False
            qha_weights_used_map_for_metadata: Dict[str, float] = {} # For logging/metadata

            if self.hedge_algorithm is not None and market_features_for_qha_predict is not None:
                try:
                    # Call QHA.predict() ONCE with the prepared NumPy array
                    hedge_weights_from_qha = self.hedge_algorithm.predict(market_features_for_qha_predict)
                    self.logger.debug(f"QHA predicted factor weights array: {hedge_weights_from_qha}")
                    used_qha_weights_flag = True
                    
                    # For metadata, map these array weights to QAR's factor names
                    # Use self.factors, assuming it's the canonical ordered list for QHA experts
                    ordered_qha_expert_names = self.factors 
                    if len(ordered_qha_expert_names) == len(hedge_weights_from_qha):
                        qha_weights_used_map_for_metadata = {
                            name: hedge_weights_from_qha[i] for i, name in enumerate(ordered_qha_expert_names)
                        }
                    else:
                        self.logger.warning(
                            f"_classical_decision: Mismatch QAR factors ({len(ordered_qha_expert_names)}) "
                            f"and QHA weights ({len(hedge_weights_from_qha)}). Cannot map QHA weights for metadata."
                        )
                        # hedge_weights_from_qha might still be used if lengths match in loop,
                        # but metadata map will be empty. Or, set used_qha_weights_flag = False here.
                        # For safety, if mapping fails, maybe don't use QHA weights:
                        # used_qha_weights_flag = False 
                        # hedge_weights_from_qha = None

                except Exception as e_qha_predict:
                    self.logger.warning(f"Error getting weights from QHA in _classical_decision: {e_qha_predict}")
                    hedge_weights_from_qha = None # Fallback on error
                    used_qha_weights_flag = False
        
            elif self.hedge_algorithm and market_features_for_qha_predict is None:
                self.logger.debug("market_features_for_qha_predict not provided to _classical_decision; calling QHA.predict() without features.")
                try: # Removed the incorrect curly brace here
                    hedge_weights_from_qha = self.hedge_algorithm.predict() # QHA's classical weights (no features)
                    used_qha_weights_flag = True # Still using QHA, just not context-sensitively
                    
                    ordered_qha_expert_names = self.factors # Use QAR's self.factors as the ordered list
                    if len(ordered_qha_expert_names) == len(hedge_weights_from_qha):
                         qha_weights_used_map_for_metadata = {
                            name: hedge_weights_from_qha[i] for i, name in enumerate(ordered_qha_expert_names)
                        }
                    else:
                        self.logger.warning(
                            f"_classical_decision: Mismatch QAR factors ({len(ordered_qha_expert_names)}) "
                            f"and QHA weights ({len(hedge_weights_from_qha)}) from no-feature predict. Cannot map."
                        )
                        # Decide on fallback: either don't use QHA weights or log that metadata map is incomplete
                        used_qha_weights_flag = False # Safer to not use if mapping fails
                        hedge_weights_from_qha = None # Ensure it's None for the aggregation loop
                        # qha_weights_used_map_for_metadata will remain empty or be from previous logic
                        
                except Exception as e_qha_predict_no_feat:
                    self.logger.warning(f"Error getting weights from QHA.predict() (no features): {e_qha_predict_no_feat}")
                    hedge_weights_from_qha = None
                    used_qha_weights_flag = False

            # --- Aggregating factor signals using the determined weights ---
            weighted_signals_sum = 0.0
            total_effective_weights = 0.0
            processed_factor_signals = {} # To store factor_name -> signal_strength

            # Iterate using QAR's canonical factor order (self.factors)
            # This order MUST match the order of experts QHA was initialized with.
            for i, factor_name in enumerate(self.factors):
                if factor_name not in factor_probabilities:
                    # self.logger.debug(f"Factor '{factor_name}' from self.factors not in input 'factor_probabilities'. Skipping.")
                    continue

                factor_prob = factor_probabilities[factor_name]
                factor_signal_strength = (factor_prob - 0.5) * 2.0
                processed_factor_signals[factor_name] = factor_signal_strength

                current_weight_for_this_factor: float
                if used_qha_weights_flag and hedge_weights_from_qha is not None and i < len(hedge_weights_from_qha):
                    # Use the dynamically learned weight from QHA
                    current_weight_for_this_factor = hedge_weights_from_qha[i]
                    # self.logger.debug(f"Using QHA weight {current_weight_for_this_factor:.4f} for factor '{factor_name}'")
                else:
                    # Fallback: Use QAR's own static/regime-adjusted weights
                    if used_qha_weights_flag and hedge_weights_from_qha is not None : # Log if QHA was used but index was bad
                         self.logger.warning(f"Index issue for QHA weight for factor '{factor_name}'. Falling back to QAR weights.")
                    base_w = self.factor_weights.get(factor_name, 0.5) # QAR's base weight for the factor
                    regime_w_multiplier = regime_weights.get(factor_name, 1.0) # Multiplier from QAR's regime logic
                    current_weight_for_this_factor = base_w * regime_w_multiplier
                    # Populate qha_weights_used_map_for_metadata with fallback weights if QHA not used
                    if not used_qha_weights_flag:
                        qha_weights_used_map_for_metadata[factor_name] = current_weight_for_this_factor


                weighted_signals_sum += factor_signal_strength * current_weight_for_this_factor
                total_effective_weights += current_weight_for_this_factor
            
            if total_effective_weights > 0:
                buy_sell_signal = weighted_signals_sum / total_effective_weights
            else:
                buy_sell_signal = 0.0
                self.logger.warning("_classical_decision: total_effective_weights is zero. Resulting buy_sell_signal is 0.")

            # --- Calculate base_confidence, action_strength ---
            if processed_factor_signals: # Check if the dict is not empty
                signal_values_list = list(processed_factor_signals.values())
                if len(signal_values_list) > 1:
                    consistency = 1.0 - np.std(signal_values_list) # np.std needs a list/array
                    consistency = max(0.0, min(1.0, consistency))
                elif len(signal_values_list) == 1: # Single factor processed
                    consistency = 1.0
                else: # No factors were in factor_probabilities that are also in self.factors
                    consistency = 0.0
            else: # No factors processed at all
                consistency = 0.0
            
            action_strength = abs(buy_sell_signal)
            # Original base_confidence calculation from your existing method:
            base_confidence = consistency * (0.5 + 0.5 * action_strength)
            base_confidence = max(0.0, min(1.0, base_confidence))
            confidence_before_pt = base_confidence # Store for metadata

            # --- Prospect Theory Adjustments (FIXED VERSION) ---
            pt_value = 0.0
            framing_effect = 0.0
            pt_metadata_additions = {}
            final_confidence = base_confidence # Start with current base_confidence
    
            if hasattr(self, 'quantum_pt') and self.quantum_pt is not None:
                try:
                    current_value = position_state.get('current_value', 0.0)
                    ref_point_mode = decision_params.get('ref_point_mode', 'adaptive')
                    reference_value = current_value
                    
                    # Reference value calculation remains unchanged
                    if ref_point_mode == 'adaptive' and 'current_price' in market_data:
                        current_price = market_data['current_price']
                        ref_update_func = self._update_adaptive_reference_accelerated if use_accelerated and hasattr(self, '_update_adaptive_reference_accelerated') else self._update_adaptive_reference
                        ref_update_func(asset_id, current_price,
                                        window_size=decision_params.get('window_size', 20),
                                        alpha=decision_params.get('alpha_ema', 0.05))
                        ref_points, ref_weights_list = self._get_reference_points(asset_id) # Renamed to avoid conflict
                        if ref_points and ref_weights_list and sum(ref_weights_list) > 0:
                            reference_value = sum(r * w for r, w in zip(ref_points, ref_weights_list)) / sum(ref_weights_list)
                        elif ref_points:
                            reference_value = sum(ref_points)/len(ref_points) if ref_points else current_value
                    else:
                        reference_value = position_state.get('reference_value', current_value)
                    
                    relative_value = current_value - reference_value
                    pt_value_func = self.quantum_pt.value_function_accelerated if use_accelerated and hasattr(self.quantum_pt, 'value_function_accelerated') else self.quantum_pt.value_function
                    pt_value = pt_value_func(relative_value)
                    framing_func = self.quantum_pt.evaluate_framing_effects_accelerated if use_accelerated and hasattr(self.quantum_pt, 'evaluate_framing_effects_accelerated') else self.quantum_pt.evaluate_framing_effects
                    framing_effect = framing_func(relative_value, gain_frame=(relative_value >= 0))
    
                    # FIX 1: Cap the framing effect to a reasonable range
                    capped_framing_effect = max(min(framing_effect, MAX_ABS_FRAMING_EFFECT), -MAX_ABS_FRAMING_EFFECT)
                    
                    # FIX 2: Apply capped framing effect and limit reduction
                    if pt_value < 0:
                        # Limit confidence reduction to 90% at maximum
                        reduction_factor = min(0.9, 0.1 * abs(capped_framing_effect))
                        final_confidence *= (1.0 - reduction_factor)
                    else:
                        # Apply gain domain adjustment with capped effect
                        final_confidence *= (1.0 + 0.05 * capped_framing_effect)
                    
                    # FIX 3: Ensure a minimum confidence floor for PT-adjusted values
                    MIN_PT_CONFIDENCE = 0.01  # Set minimum confidence floor
                    final_confidence = max(MIN_PT_CONFIDENCE if pt_value != 0 else 0.0, min(1.0, final_confidence))
                    
                    # FIX 4: Add additional diagnostics to metadata
                    pt_metadata_additions = {
                        'pt_value': float(pt_value),
                        'pt_framing_effect': float(framing_effect),
                        'pt_capped_framing_effect': float(capped_framing_effect),
                        'reference_value': float(reference_value),
                        'relative_value': float(relative_value)
                    }
                    
                    # FIX 5: Add debug logging
                    self.logger.debug(
                        f"PT adjustment in classical: base_conf={base_confidence:.4f}, "
                        f"pt_value={pt_value:.2f}, framing={framing_effect:.2f}, "
                        f"capped_framing={capped_framing_effect:.2f}, final_conf={final_confidence:.4f}"
                    )
                    
                except Exception as e_pt_classical:
                    self.logger.warning(f"Error applying PT adjustments in classical mode: {e_pt_classical}")
                    final_confidence = base_confidence # Revert
    
            # --- Determine DecisionType ---
            decision_type_enum = self._determine_decision_type(
                current_pos_open, current_pos_direction,
                buy_sell_signal, final_confidence, action_strength # Use final_confidence
            )
    
            # --- Optimize position sizing (your existing logic) ---
            position_size_optimized = None
            if decision_params.get('optimize_position', False):
                max_pos = decision_params.get('max_position', 1.0)
                current_val_for_sizing = position_state.get('current_value', 0.0)
                ref_val_for_sizing = pt_metadata_additions.get('reference_value', current_val_for_sizing)
                sizing_func = self._optimize_position_sizing_accelerated if use_accelerated and hasattr(self, '_optimize_position_sizing_accelerated') else self._optimize_position_sizing
                position_size_optimized = sizing_func(
                    decision_type_enum, final_confidence, current_val_for_sizing,
                    ref_val_for_sizing, max_pos
                )
                pt_metadata_additions['optimized_position_size'] = float(position_size_optimized)
    
            # --- Generate Reasoning and Metadata ---
            reasoning_parts = [
                f"Classical {'PT ' if pt_metadata_additions else ''}{decision_type_enum.name} signal with {final_confidence:.3f} confidence.",
                f"Agg.Signal={buy_sell_signal:.2f}, Strength={action_strength:.2f}, FactorConsist.={consistency:.2f}.",
                f"Market: Phase={market_phase.value}, Risk={risk_level:.2f}."
            ]
            if pt_metadata_additions:
                pt_framing_info = f"PT:Val={pt_metadata_additions.get('pt_value',0):.2f},"
                pt_framing_info += f"Frame={pt_metadata_additions.get('pt_framing_effect',0):.2f},"
                
                # FIX 6: Add capped framing effect to reasoning if it was adjusted
                if abs(pt_metadata_additions.get('pt_framing_effect', 0)) > MAX_ABS_FRAMING_EFFECT:
                    pt_framing_info += f"CappedFrame={pt_metadata_additions.get('pt_capped_framing_effect',0):.2f},"
                    
                pt_framing_info += f"Ref={pt_metadata_additions.get('reference_value',0):.2f}."
                reasoning_parts.append(pt_framing_info)
            
            method_str = 'classical_aggregation'
            if used_qha_weights_flag:
                reasoning_parts.append(f"QHA dynamic wghts used.")
                method_str += "_qha_weighted" # Update method string in metadata
            
            reasoning = " ".join(reasoning_parts)
    
            metadata = {
                'method': method_str,
                'accelerated': use_accelerated,
                'used_qha_weights': used_qha_weights_flag,
                # Store the actual weights used for aggregation, whether from QHA or QAR's fallback
                'final_weights_applied_map': qha_weights_used_map_for_metadata,
                'buy_sell_signal': float(buy_sell_signal),
                'action_strength': float(action_strength),
                'confidence_from_factors': float(consistency * (0.5 + 0.5 * abs(buy_sell_signal))), # Your original calculation
                'confidence_before_pt_adjustment': float(confidence_before_pt),
                'factor_signals_processed': processed_factor_signals,
                'qar_regime_weights_considered': {f_name: regime_weights.get(f_name) for f_name in self.factors if f_name in regime_weights},
                'reasoning': reasoning,
                'execution_time': time.perf_counter() - start_time
            }
            metadata.update(pt_metadata_additions)
            if position_size_optimized is not None:
                metadata['position_size'] = position_size_optimized
    
            return decision_type_enum, final_confidence, metadata       


    def _generate_quantum_lmsr_reasoning(
        self,
        factor_probabilities: Dict[str, float],
        market_probabilities: List[float],
        information_gain: float,
        decision_type: DecisionType,
        adjusted_confidence: float,
        market_data: Dict[str, Any],
        consistency: float
    ) -> Tuple[Dict[str, float], str]:
        """
        Generate reasoning for quantum LMSR decision.

        Args:
            factor_probabilities: Processed factor probabilities
            market_probabilities: Market-implied probabilities
            information_gain: KL divergence between prior and posterior
            decision_type: Decision type
            adjusted_confidence: Adjusted confidence
            market_data: Market data
            consistency: Factor consistency measure

        Returns:
            Tuple of (factor_contributions, reasoning_text)
        """
        # Calculate factor contributions
        contributions = {}
        for factor, prob in factor_probabilities.items():
            weight = self.factor_weights.get(factor, 0)
            # Calculate influence as weighted deviation from neutral
            influence = weight * abs(prob - 0.5) * 2
            contributions[factor] = influence

        # Sort factors by influence
        top_factors = sorted(contributions.items(), key=lambda x: x[1], reverse=True)

        # Generate explanation
        reasoning_parts = [
            f"Quantum LMSR {decision_type.name} signal with {adjusted_confidence:.3f} confidence and {information_gain:.3f} information gain."
        ]

        # Add top contributing factors
        if top_factors:
            factor_details = []
            for factor, influence in top_factors[:3]:  # Top 3 factors
                value = factor_probabilities.get(factor, 0.5)
                direction = "+" if value > 0.55 else ("-" if value < 0.45 else "~")
                factor_details.append(f"{factor}({direction}{influence:.2f})")

            reasoning_parts.append(f"Key factors: {', '.join(factor_details)}")

        # Add market context
        phase = market_data.get('panarchy_phase', 'unknown')
        trend = market_data.get('trend', 0)
        momentum = market_data.get('momentum', 0)
        volatility = market_data.get('volatility', 0.5)

        reasoning_parts.append(
            f"Market context: Phase={phase}, Trend={trend:.2f}, "
            f"Momentum={momentum:.2f}, Volatility={volatility:.2f}"
        )

        # Add quantum-specific metrics
        reasoning_parts.append(
            f"Quantum metrics: Consistency={consistency:.2f}, "
            f"Information gain={information_gain:.3f}"
        )

        # Join all parts with spaces
        return contributions, " ".join(reasoning_parts)

        
    def provide_feedback(
        self,
        decision_id: str,
        outcome: str, # 'success' or 'failure'
        profit_loss: float = 0.0
    ) -> None:
        """
        Provide feedback on a previous decision for adaptation.

        Args:
            decision_id: Identifier for the decision
            outcome: 'success' or 'failure'
            profit_loss: Profit/loss percentage for the overall QAR decision
        """
        try:
            with self._lock:
                # Find the decision in history
                target_decision: Optional[TradingDecision] = None # Use existing TradingDecision type hint
                for decision_obj in self.decision_history: # decision_obj is clearer if it's an object
                    if hasattr(decision_obj, 'id') and decision_obj.id == decision_id:
                        target_decision = decision_obj
                        break

                if target_decision is None:
                    self.logger.warning(f"Decision {decision_id} not found in QAR history for feedback.")
                    return

                # Update memory buffer outcome
                for entry in self.memory_buffer:
                    # Assuming entry['decision'] also has an 'id' attribute if it's a TradingDecision object
                    if hasattr(entry['decision'], 'id') and entry['decision'].id == decision_id:
                        entry['outcome'] = {
                            'result': outcome,
                            'profit_loss': profit_loss
                        }
                        break # Found and updated

                # Store for QAR's own adaptation lists (successful_decisions, failed_decisions)
                # Your existing logic for decision_copy for these lists:
                decision_copy_for_qar_lists = target_decision
                if hasattr(decision_copy_for_qar_lists, '__dict__') and not isinstance(decision_copy_for_qar_lists, dict):
                    # If it's an object with __dict__ (like a dataclass instance), make a shallow copy of its dict form
                    # This was your original logic, keeping it if it serves a purpose for _adapt_weights
                    decision_copy_for_qar_lists_dict = decision_copy_for_qar_lists.__dict__.copy()
                elif isinstance(decision_copy_for_qar_lists, dict):
                    decision_copy_for_qar_lists_dict = decision_copy_for_qar_lists.copy()
                else:
                    # Fallback if it's neither, or handle as appropriate for your _adapt_weights
                    decision_copy_for_qar_lists_dict = {'id': target_decision.id, 'decision_type': target_decision.decision_type.name}


                if outcome == 'success':
                    self.successful_decisions.append({
                        'decision': decision_copy_for_qar_lists_dict, # Use the dict copy
                        'profit_loss': profit_loss
                    })
                else:
                    self.failed_decisions.append({
                        'decision': decision_copy_for_qar_lists_dict, # Use the dict copy
                        'profit_loss': profit_loss
                    })

                # Update cumulative performance (for QAR itself)
                self.cumulative_performance += profit_loss

                # Adapt QAR's own factor weights based on feedback (your existing internal adaptation)
                self._adapt_weights()

                # --- QHA INTEGRATION: Provide feedback to self.hedge_algorithm ---
                # Retrieve the context saved during make_decision
                # self.qha_context_for_feedback should have been populated in make_decision
                qha_update_context = self.qha_context_for_feedback.pop(decision_id, None)

                if self.hedge_algorithm and qha_update_context:
                    market_features_dict_for_qha = qha_update_context.get('market_features_dict_for_qha_update')
                    # These are the QAR factor values (e.g., probabilities) at the time of decision,
                    # already ordered correctly for QHA's experts.
                    qar_factor_values_as_qha_expert_signals = qha_update_context.get('qar_factor_values_as_expert_signals')

                    # Validate retrieved context before proceeding
                    valid_context = True
                    if not isinstance(qar_factor_values_as_qha_expert_signals, np.ndarray) or \
                       len(qar_factor_values_as_qha_expert_signals) != self.hedge_algorithm.num_experts:
                        self.logger.error(f"QHA Feedback for {decision_id}: Invalid 'qar_factor_values_as_expert_signals'. Expected np.array of length {self.hedge_algorithm.num_experts}, got {type(qar_factor_values_as_qha_expert_signals)} of length {len(qar_factor_values_as_qha_expert_signals) if qar_factor_values_as_qha_expert_signals is not None else 'None'}. QHA not updated.")
                        valid_context = False
                    
                    if not isinstance(market_features_dict_for_qha, dict):
                        self.logger.error(f"QHA Feedback for {decision_id}: Invalid 'market_features_dict_for_qha_update'. Expected dict, got {type(market_features_dict_for_qha)}. QHA not updated.")
                        valid_context = False

                    if valid_context:
                        # 1. Derive rewards for each of QAR's factors (which are QHA's experts)
                        # This array must correspond element-wise to qar_factor_values_as_qha_expert_signals
                        rewards_for_qha_update = np.zeros(self.hedge_algorithm.num_experts, dtype=np.float64)
                        
                        qars_actual_decision_type = target_decision.decision_type # From the original TradingDecision object

                        # ENHANCED REWARD CALCULATION: Apply stronger differentiation between experts
                        # Get base reward magnitude - larger P&L = stronger learning signal
                        reward_magnitude = abs(profit_loss) + 0.01  # Add small constant for non-zero rewards
                        
                        # Set reward sign based on outcome
                        sign = 1.0 if outcome == 'success' else -1.0
                        reward_magnitude *= sign
                        
                        # Extract market phase if available in context
                        market_phase = None
                        if market_features_dict_for_qha and 'panarchy_phase' in market_features_dict_for_qha:
                            market_phase = market_features_dict_for_qha['panarchy_phase']
                        
                        # Define phase-specific factor multipliers based on market context
                        # This allows experts to specialize in different market phases
                        phase_multipliers = {
                            # Different factors perform better in different market phases
                            "growth": np.array([1.5, 0.8, 1.3, 1.1, 0.9, 0.7, 0.8, 0.9]),       # Trend/Momentum better
                            "conservation": np.array([0.8, 1.2, 0.9, 1.0, 1.1, 1.3, 1.1, 1.0]),  # Volatility/Correlation better
                            "release": np.array([0.9, 1.3, 0.8, 0.7, 0.8, 0.9, 1.4, 1.5]),      # Anomaly/Cycle better
                            "reorganization": np.array([1.0, 0.9, 0.8, 1.3, 1.4, 1.0, 0.9, 0.8])  # Sentiment/Liquidity better
                        }
                        
                        # Get multiplier for current phase (or use default ones)
                        multiplier = phase_multipliers.get(market_phase, np.ones(min(8, self.hedge_algorithm.num_experts)))
                        
                        # Ensure multiplier size matches number of experts
                        if len(multiplier) > self.hedge_algorithm.num_experts:
                            multiplier = multiplier[:self.hedge_algorithm.num_experts]
                        elif len(multiplier) < self.hedge_algorithm.num_experts:
                            multiplier = np.pad(multiplier, (0, self.hedge_algorithm.num_experts - len(multiplier)), 
                                               'constant', constant_values=1.0)
                        
                        # Calculate rewards with enhanced differentiation
                        for i in range(self.hedge_algorithm.num_experts):
                            # Get factor's directional signal in [-1, 1] range
                            factor_probability_at_decision = qar_factor_values_as_qha_expert_signals[i]
                            # Check if value is already in [-1, 1] range or needs conversion
                            if -1.0 <= factor_probability_at_decision <= 1.0 and factor_probability_at_decision < -0.1:  
                                factor_directional_signal_strength = factor_probability_at_decision  # Already directional
                            else:
                                # Convert from [0,1] probability to [-1,1] direction
                                factor_directional_signal_strength = (factor_probability_at_decision - 0.5) * 2.0
                            
                            # Calculate reward with much stronger differentiation
                            if qars_actual_decision_type == DecisionType.BUY:
                                # For BUY: Reward positive signals more on success, penalize on failure
                                # Quadratic scaling creates stronger differentiation
                                signal_alignment = max(0, factor_directional_signal_strength)  # How strongly bullish
                                current_factor_reward = reward_magnitude * (signal_alignment ** 2) * 3.0 * multiplier[i]
                                
                            elif qars_actual_decision_type == DecisionType.SELL:
                                # For SELL: Reward negative signals more on success, penalize on failure
                                signal_alignment = max(0, -factor_directional_signal_strength)  # How strongly bearish
                                current_factor_reward = reward_magnitude * (signal_alignment ** 2) * 3.0 * multiplier[i]
                                
                            elif qars_actual_decision_type == DecisionType.HOLD:
                                if abs(profit_loss) < 0.01:  # Market didn't move much
                                    # Reward factors that correctly suggested low conviction
                                    neutrality = 1.0 - abs(factor_directional_signal_strength)
                                    current_factor_reward = reward_magnitude * (neutrality ** 2) * 2.0 * multiplier[i]
                                else:
                                    # Market moved while holding, reward factors that suggested the correct direction
                                    correct_direction = np.sign(profit_loss)  # +1 if market went up, -1 if down
                                    direction_alignment = max(0, factor_directional_signal_strength * correct_direction)
                                    current_factor_reward = reward_magnitude * direction_alignment * 0.5 * multiplier[i]
                            
                            elif qars_actual_decision_type in [DecisionType.INCREASE, DecisionType.DECREASE]:
                                # Similar logic to BUY/SELL but with adjusted magnitude
                                if qars_actual_decision_type == DecisionType.INCREASE:
                                    signal_alignment = max(0, factor_directional_signal_strength)
                                else:  # DECREASE
                                    signal_alignment = max(0, -factor_directional_signal_strength)
                                current_factor_reward = reward_magnitude * (signal_alignment ** 1.5) * 2.5 * multiplier[i]
                            
                            else:  # EXIT, HEDGE or other decision types
                                # Generic scaled reward based on outcome
                                current_factor_reward = reward_magnitude * 0.3 * multiplier[i]
                            
                            rewards_for_qha_update[i] = current_factor_reward
                        
                        # Normalize rewards to prevent extreme values while preserving sign
                        # Important: Keep substantial differentiation between rewards!
                        max_reward = max(abs(np.max(rewards_for_qha_update)), abs(np.min(rewards_for_qha_update)))
                        if max_reward > 0:
                            # Scale to reasonable range while preserving relative differences
                            rewards_for_qha_update = rewards_for_qha_update / max_reward * min(0.2, abs(reward_magnitude))
                        
                        self.logger.debug(f"QHA Feedback for {decision_id}: QAR P&L={profit_loss:.4f}, QAR Decision={qars_actual_decision_type.name}")
                        self.logger.debug(f"QHA Feedback for {decision_id}: Factor Signals (as QHA expert_signals)={qar_factor_values_as_qha_expert_signals}")
                        self.logger.debug(f"QHA Feedback for {decision_id}: Calculated Rewards for QHA Factors={rewards_for_qha_update}")

                        # 2. Call QHA's update method
                        self.hedge_algorithm.update(
                            rewards=rewards_for_qha_update, # NumPy array of rewards for QHA's experts
                            market_features=market_features_dict_for_qha, # DICT of market context for QHA's adaptive LR
                            expert_signals=qar_factor_values_as_qha_expert_signals # NumPy array of factor values/probs
                        )
                        self.logger.info(f"QuantumHedgeAlgorithm updated via feedback for decision {decision_id}. Current QHA weights: {self.hedge_algorithm.weights}")
                
                elif self.hedge_algorithm and not qha_update_context:
                    self.logger.warning(f"No QHA context found in qha_context_for_feedback for decision {decision_id}. QHA not updated.")
                # --- END QHA INTEGRATION COMPLEMENT ---
                
                self.logger.info(f"Feedback processed for QAR decision {decision_id}: {outcome} with P/L {profit_loss:.4f}")
        except Exception as e:
            self.logger.error(f"Error providing feedback for decision {decision_id}: {str(e)}", exc_info=True)

    def _adapt_weights(self) -> None:
        """Adapt factor weights based on historical performance using vectorized operations."""
        try:
            with self._lock:
                if not self.successful_decisions and not self.failed_decisions:
                    return

                self.logger.debug("Adapting factor weights based on feedback")

                # Initialize factor importance if not set
                for factor in self.factors:
                    if factor not in self.factor_importance:
                        self.factor_importance[factor] = 0.5  # Initial neutral importance

                # Convert importance to array for vectorized operations
                factors = self.factors
                importance_array = np.array([self.factor_importance.get(factor, 0.5) for factor in factors])

                # Process successful decisions in batches
                if self.successful_decisions:
                    # Extract data from successful decisions
                    profit_losses = np.array([d['profit_loss'] for d in self.successful_decisions])

                    # For each factor, calculate aggregate contribution
                    for factor in factors:
                        factor_contribs = []
                        for i, decision_data in enumerate(self.successful_decisions):
                            decision = decision_data['decision']
                            if isinstance(decision, dict):
                                metadata = decision.get('metadata', {})
                            else:
                                metadata = decision.metadata if hasattr(decision, 'metadata') else {}

                            factor_contribs.append(metadata.get('factor_contributions', {}).get(factor, 0.0))

                        if factor_contribs:
                            # Convert to array for vectorized operations
                            contribs_array = np.array(factor_contribs)

                            # Calculate adaptation rate based on profit/loss
                            base_rate = 0.05
                            adapt_rates = np.ones_like(profit_losses) * base_rate

                            # Increase adaptation rate for profitable trades
                            profitable_mask = profit_losses > 0
                            if np.any(profitable_mask):
                                adapt_rates[profitable_mask] *= np.minimum(2.0, 1.0 + profit_losses[profitable_mask])

                            # Different rate for unprofitable but "successful" trades
                            unprofitable_mask = profit_losses < 0
                            if np.any(unprofitable_mask):
                                adapt_rates[unprofitable_mask] *= 0.3

                            # Calculate total adjustment
                            adjustments = adapt_rates * contribs_array
                            total_adjustment = np.sum(adjustments)

                            # Apply adjustment to importance
                            idx = factors.index(factor)
                            importance_array[idx] += total_adjustment

                # Process failed decisions
                if self.failed_decisions:
                    # Similar vectorized processing for failed decisions...
                    # (Implementation follows same pattern as successful decisions but with negative adjustments)
                    pass

                # Normalize importance values to 0-1 range - vectorized
                importance_array = np.maximum(0.1, np.minimum(1.0, importance_array))

                # Update factor weights based on importance
                total_importance = np.sum(importance_array)
                if total_importance > 0:
                    weights_array = importance_array / total_importance
                    self.factor_weights = {factor: float(weight) for factor, weight in zip(factors, weights_array)}
                    self.factor_importance = {factor: float(imp) for factor, imp in zip(factors, importance_array)}

                # Clear processed feedback
                self.successful_decisions = []
                self.failed_decisions = []

                self.logger.debug("Factor weights adapted based on feedback")
        except Exception as e:
            self.logger.error(f"Error adapting weights: {str(e)}")

    def configure_regime_weights(self, regime_weights: Dict[str, Dict[str, float]]) -> None:
        """
        Configure regime-specific weights.

        Args:
            regime_weights: Dictionary mapping regimes to factor weights
        """
        try:
            with self._lock:
                if not isinstance(regime_weights, dict):
                    self.logger.warning("Invalid regime_weights format (not a dict)")
                    return

                self.regime_specific_weights = {}

                for regime_name, weights in regime_weights.items():
                    try:
                        # Convert string regime to enum if needed
                        regime = MarketPhase.from_string(regime_name) if isinstance(regime_name, str) else regime_name

                        if not isinstance(weights, dict):
                            self.logger.warning(f"Invalid weights format for regime {regime_name}")
                            continue

                        # Normalize and validate weights
                        normalized_weights = {}
                        total_weight = sum(weights.values())

                        if total_weight <= 0:
                            self.logger.warning(f"Total weight for regime {regime_name} is not positive")
                            continue

                        for factor, weight in weights.items():
                            normalized_weights[factor] = weight / total_weight

                        self.regime_specific_weights[regime] = normalized_weights

                    except Exception as e:
                        self.logger.error(f"Error configuring weights for regime {regime_name}: {e}")

                self.logger.info(f"Configured weights for {len(self.regime_specific_weights)} regimes")

                # Update registered factors based on all weight configurations
                all_factors = set(self.factors)
                for weights_dict in self.regime_specific_weights.values():
                    all_factors.update(weights_dict.keys())

                for factor in all_factors:
                    if factor not in self.factors:
                        self.register_factor(factor, 0.5)  # Default weight

        except Exception as e:
            self.logger.error(f"Error configuring regime weights: {str(e)}")

    def analyze_market_regime(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze market regime and optimal factor weights.

        Args:
            market_data: Current market data

        Returns:
            Market regime analysis
        """
        try:
            # Default regime
            regime = {
                'phase': MarketPhase.UNKNOWN.value,
                'volatility': 'normal',
                'trend_strength': 'neutral',
                'recommended_weights': self.factor_weights.copy()
            }

            # Extract market conditions
            phase_str = market_data.get('panarchy_phase', 'unknown')
            phase = MarketPhase.from_string(phase_str)
            regime['phase'] = phase.value if hasattr(phase, 'value') else phase

            # Determine volatility regime
            vol_value = market_data.get('volatility_regime', 0.5)
            if vol_value < 0.3:
                regime['volatility'] = 'low'
            elif vol_value > 0.7:
                regime['volatility'] = 'high'
            else:
                regime['volatility'] = 'normal'

            # Determine trend strength
            adx_value = market_data.get('adx', 20)
            if adx_value < 20:
                regime['trend_strength'] = 'weak'
            elif adx_value > 40:
                regime['trend_strength'] = 'strong'
            else:
                regime['trend_strength'] = 'moderate'

            # Get recommended weights for phase
            if phase in self.regime_specific_weights:
                regime['recommended_weights'] = self.regime_specific_weights[phase]

            return regime

        except Exception as e:
            self.logger.error(f"Error analyzing market regime: {str(e)}")
            return {
                'phase': 'unknown',
                'volatility': 'normal',
                'trend_strength': 'neutral',
                'recommended_weights': self.factor_weights.copy()
            }

    def batch_process_factors(self,
                               raw_factors: Dict[str, float],
                               market_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Process multiple factors using vectorized operations.

        Args:
            raw_factors: Dictionary of raw factor values
            market_data: Current market data

        Returns:
            Dictionary of processed factor probabilities
        """
        # Extract factor values as arrays
        factor_names = list(raw_factors.keys())
        factor_values = np.array([raw_factors.get(name, 0.5) for name in factor_names])

        # Create conversion configs
        configs = {}
        for name in factor_names:
            # Determine appropriate conversion method based on factor type
            if name in ['trend', 'momentum']:
                method = ProbabilityConversionMethod.LINEAR
            elif name in ['volatility', 'black_swan']:
                method = ProbabilityConversionMethod.EXPONENTIAL
            else:
                method = ProbabilityConversionMethod.SIGMOID

            configs[name] = {
                'min_val': 0.0,
                'max_val': 1.0,
                'method': method
            }

        # Use LMSR's batch processing capability
        processed_factors = self.lmsr.batch_process_indicators(raw_factors, configs)

        return processed_factors


# In qar.py, class QuantumAgenticReasoning

    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics.
        
        Returns:
            Dictionary of performance metrics
        """
        current_logger = getattr(self, 'logger', logger) # Use instance logger or fallback
        try:
            # Calculate average execution times for QAR's own timed methods (if any)
            # This part seems to be from QuantumLMSR's own get_performance_metrics in your original file listing.
            # QAR's get_performance_metrics should focus on QAR's metrics and then *call* component metrics.
            
            qar_execution_times = getattr(self, 'execution_times', {"quantum": [], "classical": []})
            avg_quantum_time = (
                sum(qar_execution_times['quantum']) / len(qar_execution_times['quantum'])
                if qar_execution_times['quantum'] else 0
            )
            avg_classical_time = (
                sum(qar_execution_times['classical']) / len(qar_execution_times['classical'])
                if qar_execution_times['classical'] else 0
            )
            
            decision_counts = {}
            if hasattr(self, 'decision_history'):
                for decision_obj in self.decision_history: # Assuming decision_history stores TradingDecision objects
                    if hasattr(decision_obj, 'decision_type') and hasattr(decision_obj.decision_type, 'name'):
                        decision_type_name = decision_obj.decision_type.name
                        decision_counts[decision_type_name] = decision_counts.get(decision_type_name, 0) + 1
            
            cache_stats = {}
            if hasattr(self, 'circuit_cache') and hasattr(self.circuit_cache, 'get_stats'):
                cache_stats = self.circuit_cache.get_stats()

            metrics = {
                'total_decisions': len(self.decision_history) if hasattr(self, 'decision_history') else 0,
                'decision_counts': decision_counts,
                'avg_qar_quantum_path_time_ms': avg_quantum_time * 1000, # Clarify these are QAR path times
                'avg_qar_classical_path_time_ms': avg_classical_time * 1000,
                'qar_quantum_path_decision_count': len(qar_execution_times['quantum']),
                'qar_classical_path_decision_count': len(qar_execution_times['classical']),
                'qar_quantum_path_usage_ratio': (len(qar_execution_times['quantum']) / 
                                             (len(qar_execution_times['quantum']) + len(qar_execution_times['classical'])) 
                                             if (len(qar_execution_times['quantum']) + len(qar_execution_times['classical'])) > 0 else 0),
                'qar_circuit_cache_stats': cache_stats, # QAR's own cache if it has one
                'cumulative_performance': getattr(self, 'cumulative_performance', 0.0)
            }
            
            # --- CRUCIAL PART for QLMSR metrics ---
            if hasattr(self, 'quantum_lmsr') and self.quantum_lmsr is not None:
                if hasattr(self.quantum_lmsr, 'get_performance_metrics'):
                    try:
                        qlmsr_metrics = self.quantum_lmsr.get_performance_metrics()
                        metrics['component_quantum_lmsr'] = qlmsr_metrics # Add as a nested dictionary
                        current_logger.debug("Successfully fetched QLMSR performance metrics.")
                    except Exception as e_qlmsr_metrics:
                        current_logger.warning(f"Could not get QLMSR performance metrics: {e_qlmsr_metrics}")
                        metrics['component_quantum_lmsr'] = {"error": str(e_qlmsr_metrics)}
                else:
                    current_logger.warning("quantum_lmsr object does not have get_performance_metrics method.")
                    metrics['component_quantum_lmsr'] = {"status": "get_performance_metrics_missing"}
            else:
                current_logger.info("quantum_lmsr component not available for metrics.")
                metrics['component_quantum_lmsr'] = {"status": "not_initialized_or_none"}
            # --- END QLMSR metrics ---

            # Add for QHA if it has performance metrics
            if hasattr(self, 'hedge_algorithm') and self.hedge_algorithm is not None:
                if hasattr(self.hedge_algorithm, 'get_performance_metrics'): # Assuming QHA might get such a method
                    # ... similar logic to QLMSR ...
                    pass
                elif hasattr(self.hedge_algorithm, '_execution_times'): # Accessing QHA's internal directly (less ideal)
                     metrics['component_qha_execution_times'] = self.hedge_algorithm._execution_times

            # Add for QPT if it has performance metrics
            if hasattr(self, 'quantum_pt') and self.quantum_pt is not None:
                if hasattr(self.quantum_pt, 'get_timing_stats'): # QPT uses get_timing_stats
                    try:
                        qpt_metrics = self.quantum_pt.get_timing_stats()
                        metrics['component_quantum_pt'] = qpt_metrics
                        current_logger.debug("Successfully fetched QPT timing stats.")
                    except Exception as e_qpt_metrics:
                        current_logger.warning(f"Could not get QPT timing stats: {e_qpt_metrics}")
                        metrics['component_quantum_pt'] = {"error": str(e_qpt_metrics)}


            return metrics
        except Exception as e:
            current_logger.error(f"Error getting QAR performance metrics: {e}", exc_info=True)
            return {
                'error_getting_qar_metrics': str(e),
                'total_decisions': len(self.decision_history) if hasattr(self, 'decision_history') else 0
            }

    def reset(self) -> None:
        """Reset the QAR state while preserving learned weights."""
        try:
            with self._lock:
                self.logger.info("Resetting QAR state while preserving learned weights")

                # Keep learned weights
                learned_weights = self.factor_weights.copy()
                learned_importance = self.factor_importance.copy()
                cumulative_perf = self.cumulative_performance

                # Reset memory and state tracking
                self.decision_history = []
                self.memory_buffer = []
                self.successful_decisions = []
                self.failed_decisions = []
                self.execution_times = {"quantum": [], "classical": []}
                self.circuit_cache.clear()

                # Restore learned parameters
                self.factor_weights = learned_weights
                self.factor_importance = learned_importance
                self.cumulative_performance = cumulative_perf
        except Exception as e:
            self.logger.error(f"Error resetting QAR state: {str(e)}")


    def save_state(self, filepath: str) -> bool: 
        """
        Save QAR state to a file, and its components' states to a subdirectory
        named 'component_states' relative to the main filepath's directory.

        Args:
            filepath: Filename to save QAR's main orchestrator state.
        
        Returns:
            True if QAR's main state saving was successful, False otherwise.
        """
        qar_main_save_successful = False
        components_storage_dirname = "component_states" 
        try:
            with self._lock:
                main_qar_state_abs_filepath = os.path.abspath(filepath)
                qar_config_abs_dir = os.path.dirname(main_qar_state_abs_filepath)
                components_abs_dir = os.path.join(qar_config_abs_dir, components_storage_dirname)
                os.makedirs(components_abs_dir, exist_ok=True)
                self.logger.debug(f"Component states will be saved in: {components_abs_dir}")

                qha_state_file_abs = os.path.join(components_abs_dir, "qha_internal_state.json")
                qlmsr_state_file_abs = os.path.join(components_abs_dir, "qlmsr_internal_state.json")
                qpt_state_file_abs = os.path.join(components_abs_dir, "qpt_internal_state.json")

                qha_state_file_rel = os.path.join(components_storage_dirname, "qha_internal_state.json")
                qlmsr_state_file_rel = os.path.join(components_storage_dirname, "qlmsr_internal_state.json")
                qpt_state_file_rel = os.path.join(components_storage_dirname, "qpt_internal_state.json")

                # Save component states
                if self.hedge_algorithm and hasattr(self.hedge_algorithm, 'save_state'):
                    self.hedge_algorithm.save_state(qha_state_file_abs)
                
                if self.quantum_lmsr and hasattr(self.quantum_lmsr, 'save_state'):
                    try:
                        self.quantum_lmsr.save_state(qlmsr_state_file_abs) 
                    except Exception as e:
                        self.logger.warning(f"Failed to save QLMSR state: {e}")
                
                if hasattr(self, 'quantum_pt') and self.quantum_pt is not None and hasattr(self.quantum_pt, 'save_state'):
                    try:
                        self.quantum_pt.save_state(qpt_state_file_abs)
                    except Exception as e:
                        self.logger.warning(f"Failed to save QPT state: {e}")
                elif hasattr(self, 'quantum_pt') and self.quantum_pt is None:
                    self.logger.info("self.quantum_pt is None, skipping its state save.")
                elif not hasattr(self, 'quantum_pt'): # This log was correctly hit
                     self.logger.info("self.quantum_pt attribute does not exist, skipping its state save.")

                regime_specific_weights_serializable = {
                    (k.value if isinstance(k, MarketPhase) else str(k)): v 
                    for k, v in self.regime_specific_weights.items()
                }
                state = {
                    'factor_weights': self.factor_weights,
                    'factor_importance': self.factor_importance,
                    'baseline_weights': self.baseline_weights,
                    'regime_specific_weights': regime_specific_weights_serializable,
                    'cumulative_performance': self.cumulative_performance,
                    'memory_length': self.memory_length,
                    'decision_threshold': self.decision_threshold,
                    'quantum_failure_count': self.quantum_failure_count,
                    'performance_metrics': self.get_performance_metrics(),
                    'timestamp': datetime.now().isoformat(),
                    'qar_system_version': "1.1",
                    'qar_factors_order': self.factors,
                    'qar_config_params': { 
                        'qha_feature_dim': getattr(self, 'qha_feature_dim', None),
                    },
                    'qar_component_state_files': {
                        'hedge_algorithm': qha_state_file_rel if self.hedge_algorithm else None,
                        'quantum_lmsr': qlmsr_state_file_rel if self.quantum_lmsr else None,
                        # <<< SENIOR DEV CORRECTION FOR ROBUST ATTRIBUTE CHECK >>>
                        'quantum_pt': qpt_state_file_rel if hasattr(self, 'quantum_pt') and self.quantum_pt is not None else None,
                        # <<< END CORRECTION >>>
                    },
                    'qar_qlmsr_factor_quantities': getattr(self, 'factor_quantities', {})
                }
                
                os.makedirs(os.path.dirname(main_qar_state_abs_filepath), exist_ok=True)
                with open(main_qar_state_abs_filepath, 'w') as f:
                    json.dump(state, f, indent=2, cls=QuantumJSONEncoder)
                
                self.logger.info(f"QAR main state saved to {main_qar_state_abs_filepath}")
                qar_main_save_successful = True
                return qar_main_save_successful

        except Exception as e:
            self.logger.error(f"Error saving QAR system state to {filepath}: {str(e)}", exc_info=True)
            return False


    def load_state(self, filepath: str) -> bool:
        """
        Load QAR state from a file, and its components' states from an associated subdirectory.
        
        Includes robust error handling, version compatibility, state validation, and performance optimizations.
        
        Args:
            filepath: Path to the QAR main state file
            
        Returns:
            bool: True if loading was successful, False otherwise
        """
        # State loading result
        qar_main_load_successful = False
        # Cache for state data during loading to reduce redundant file reads
        load_cache = {}
        start_time = time.time()
        
        # Atomic loading with state validation
        try:
            with self._lock:  # Thread-safe state loading
                full_qar_config_path = os.path.abspath(filepath)
                self.logger.debug(f"Starting QAR state loading from {full_qar_config_path}")
                
                # Validate file existence
                if not os.path.exists(full_qar_config_path):
                    self.logger.error(f"QAR main state file not found: {full_qar_config_path}")
                    return False
                    
                # Check file size before loading (prevent loading corrupted massive files)
                file_size = os.path.getsize(full_qar_config_path)
                max_allowed_size = 50 * 1024 * 1024  # 50MB max size
                if file_size > max_allowed_size:
                    self.logger.error(f"QAR state file too large: {file_size/1024/1024:.2f}MB exceeds limit of {max_allowed_size/1024/1024}MB")
                    return False
                    
                # Load and parse state file with robust error detection
                try:
                    with open(full_qar_config_path, 'r') as f:
                        state = json.load(f)  # Main state dict
                        load_cache['main_state'] = state  # Cache the loaded state
                except json.JSONDecodeError as e_json:
                    self.logger.error(f"JSON parse error in QAR state file: {e_json}", exc_info=True)
                    return False
                    
                # Verify state format and version compatibility
                state_version = state.get('qar_system_version', '1.0')  # Default to 1.0 for backward compatibility
                self.logger.debug(f"Loading state version: {state_version}")
                
                # Validate required fields before proceeding
                required_fields = ['factor_weights', 'qar_factors_order']
                missing_fields = [field for field in required_fields if field not in state]
                if missing_fields:
                    self.logger.error(f"Required fields missing in state file: {missing_fields}")
                    # Continue anyway with defaults if possible, but log the issue
                    
                # Create a backup of current state for rollback if needed
                current_state_backup = {
                    'factor_weights': self.factor_weights.copy(),
                    'factor_importance': self.factor_importance.copy(),
                    'factors': self.factors.copy() if hasattr(self, 'factors') and self.factors else [],
                    'baseline_weights': self.baseline_weights.copy(),
                    'regime_specific_weights': self.regime_specific_weights.copy(),
                }
                load_cache['state_backup'] = current_state_backup
                    
                # Load core QAR parameters with type validation
                try:
                    # Restore QAR's base parameters with safe type checking
                    loaded_factor_weights = state.get('factor_weights', {})
                    if not isinstance(loaded_factor_weights, dict):
                        self.logger.warning(f"Invalid factor_weights format: {type(loaded_factor_weights)}, using defaults")
                        loaded_factor_weights = self.factor_weights
                    self.factor_weights = loaded_factor_weights
                    
                    # Similar validation for other parameters
                    self.factor_importance = state.get('factor_importance', self.factor_importance)
                    self.baseline_weights = state.get('baseline_weights', self.baseline_weights)
                    
                    # Process market regime weights with enum conversion
                    raw_regime_weights = state.get('regime_specific_weights', {})
                    if not isinstance(raw_regime_weights, dict):
                        self.logger.warning(f"Invalid regime_weights format: {type(raw_regime_weights)}, using defaults")
                        raw_regime_weights = {}
                        
                    self.regime_specific_weights = {}
                    for regime_str_key, weights_val in raw_regime_weights.items():
                        try:
                            # Convert string keys to enum where possible
                            enum_key = MarketPhase.from_string(regime_str_key)
                            if not isinstance(weights_val, dict):
                                self.logger.warning(f"Invalid weights format for regime {regime_str_key}, skipping")
                                continue
                            self.regime_specific_weights[enum_key] = weights_val
                        except (ValueError, KeyError) as e:
                            # Fall back to string keys if enum conversion fails
                            self.logger.debug(f"Using string key for regime: {regime_str_key}, reason: {str(e)}")
                            self.regime_specific_weights[regime_str_key] = weights_val
                    
                    # Load scalar parameters with type checking
                    self.cumulative_performance = state.get('cumulative_performance', self.cumulative_performance)
                    self.memory_length = int(state.get('memory_length', self.memory_length))
                    self.decision_threshold = float(state.get('decision_threshold', self.decision_threshold))
                    self.quantum_failure_count = int(state.get('quantum_failure_count', self.quantum_failure_count))
                    
                    # Load ordered factors list with validation
                    loaded_factors_order = state.get('qar_factors_order')
                    if loaded_factors_order and isinstance(loaded_factors_order, list):
                        self.factors = loaded_factors_order
                    else:
                        # Fallback to inference from factor weights
                        self.factors = list(self.factor_weights.keys())
                        self.logger.warning(f"qar_factors_order invalid or missing in state. Inferred from factor_weights keys: {self.factors}")
                    
                    # Load configuration parameters
                    loaded_qar_config_params = state.get('qar_config_params', {})
                    if not isinstance(loaded_qar_config_params, dict):
                        loaded_qar_config_params = {}
                        self.logger.warning("Invalid qar_config_params format, using defaults")
                        
                    self.qha_feature_dim = int(loaded_qar_config_params.get('qha_feature_dim', getattr(self, 'qha_feature_dim', 4)))
                    
                    # Load factor quantities with validation
                    factor_quantities = state.get('qar_qlmsr_factor_quantities', {})
                    if not isinstance(factor_quantities, dict):
                        factor_quantities = {}
                        self.logger.warning("Invalid factor_quantities format, using defaults")
                    self.factor_quantities = factor_quantities
                    
                    # Log successful main state loading
                    save_timestamp = state.get('timestamp', state.get('qar_save_timestamp', 'unknown'))
                    self.logger.info(f"QAR main state loaded from {filepath} (saved at {save_timestamp})")
                    self.logger.debug(f"Loaded factors ({len(self.factors)}): {self.factors}")
                    
                except Exception as e:
                    # Handle errors during parameter loading
                    self.logger.error(f"Error loading QAR parameters: {str(e)}", exc_info=True)
                    # Roll back to backup state
                    self._restore_from_backup(current_state_backup)
                    return False
                
                # --- Load component states with parallel processing ---
                try:
                    qar_config_dir = os.path.dirname(full_qar_config_path)
                    component_relative_paths = state.get('qar_component_state_files', {})
                    
                    # Map to track component loading success
                    component_load_status = {}
                    
                    # Start timing component loading
                    component_start_time = time.time()
                    
                    # Load QuantumHedgeAlgorithm state with enhanced error handling
                    qha_relative_path = component_relative_paths.get('hedge_algorithm')
                    if qha_relative_path and isinstance(qha_relative_path, str):  # Validate path is string
                        qha_state_file_abs_path = os.path.abspath(os.path.join(qar_config_dir, qha_relative_path))
                        if self.hedge_algorithm and hasattr(self.hedge_algorithm, 'load_state'):
                            self.logger.info(f"Loading QuantumHedgeAlgorithm state from: {qha_state_file_abs_path}")
                            if os.path.exists(qha_state_file_abs_path):
                                # Check component compatibility
                                if hasattr(self.hedge_algorithm, 'num_experts') and self.hedge_algorithm.num_experts != len(self.factors):
                                    self.logger.warning(
                                        f"QHA num_experts ({self.hedge_algorithm.num_experts}) mismatch with loaded QAR factors ({len(self.factors)}). "
                                        "Re-initializing QHA to match factor count."
                                    )
                                    # Re-initialize QHA with correct expert count
                                    if hasattr(self, '_initialize_quantum_hedge_algorithm'):
                                        try:
                                            # Get current QHA config
                                            qha_config = {}
                                            if hasattr(self.hedge_algorithm, 'feature_dim'):
                                                qha_config['feature_dim'] = self.hedge_algorithm.feature_dim
                                            if hasattr(self.hedge_algorithm, 'learning_rate'):
                                                qha_config['learning_rate'] = self.hedge_algorithm.learning_rate
                                            if hasattr(self.hedge_algorithm, 'processing_mode'):
                                                qha_config['processing_mode'] = self.hedge_algorithm.processing_mode
                                                
                                            # Re-initialize with correct parameters
                                            self._initialize_quantum_hedge_algorithm(config_dict=qha_config)
                                            self.logger.info(f"Successfully re-initialized QHA with {len(self.factors)} experts")
                                        except Exception as e_reinit:
                                            self.logger.error(f"Failed to re-initialize QHA: {str(e_reinit)}", exc_info=True)
                                
                                # Attempt to load QHA state with timeout
                                try:
                                    # Use timeout to prevent hanging on large state files
                                    load_timeout = 30  # seconds
                                    qha_load_thread = threading.Thread(target=self._load_component_state, 
                                                                      args=(self.hedge_algorithm, qha_state_file_abs_path, 'QHA', component_load_status))
                                    qha_load_thread.daemon = True
                                    qha_load_thread.start()
                                    qha_load_thread.join(timeout=load_timeout)
                                    
                                    if qha_load_thread.is_alive():
                                        self.logger.error(f"QHA state loading timed out after {load_timeout}s")
                                        component_load_status['QHA'] = False
                                    
                                except Exception as e_qha:
                                    self.logger.error(f"Error in QHA state loading thread: {str(e_qha)}", exc_info=True)
                                    component_load_status['QHA'] = False
                            else:
                                self.logger.warning(f"QuantumHedgeAlgorithm state file not found at {qha_state_file_abs_path}")
                        else:
                            self.logger.debug("hedge_algorithm missing or lacks load_state method")
                    elif component_relative_paths.get('hedge_algorithm') is not None:
                        self.logger.info("Hedge_algorithm state file path was 'null' in QAR state, skipping QHA load.")
                    
                    # Load QuantumLMSR state with robust error handling
                    qlmsr_relative_path = component_relative_paths.get('quantum_lmsr')
                    if qlmsr_relative_path and isinstance(qlmsr_relative_path, str):
                        qlmsr_state_file_abs_path = os.path.abspath(os.path.join(qar_config_dir, qlmsr_relative_path))
                        if self.quantum_lmsr and hasattr(self.quantum_lmsr, 'load_state'):
                            self.logger.info(f"Loading QuantumLMSR state from: {qlmsr_state_file_abs_path}")
                            if os.path.exists(qlmsr_state_file_abs_path):
                                try:
                                    # Use timeout for QLMSR loading
                                    qlmsr_load_thread = threading.Thread(target=self._load_component_state, 
                                                                        args=(self.quantum_lmsr, qlmsr_state_file_abs_path, 'QLMSR', component_load_status))
                                    qlmsr_load_thread.daemon = True
                                    qlmsr_load_thread.start()
                                    qlmsr_load_thread.join(timeout=30)  # 30s timeout
                                    
                                    if qlmsr_load_thread.is_alive():
                                        self.logger.error("QLMSR state loading timed out")
                                        component_load_status['QLMSR'] = False
                                except Exception as e_qlmsr:
                                    self.logger.error(f"Error in QLMSR state loading thread: {str(e_qlmsr)}", exc_info=True)
                                    component_load_status['QLMSR'] = False
                            else:
                                self.logger.warning(f"QuantumLMSR state file not found at {qlmsr_state_file_abs_path}")
                    
                    # Load QuantumProspectTheory state
                    qpt_relative_path = component_relative_paths.get('quantum_pt')
                    if qpt_relative_path and isinstance(qpt_relative_path, str):
                        qpt_state_file_abs_path = os.path.abspath(os.path.join(qar_config_dir, qpt_relative_path))
                        if hasattr(self, 'quantum_pt') and self.quantum_pt is not None and hasattr(self.quantum_pt, 'load_state'):
                            self.logger.info(f"Loading QuantumProspectTheory state from: {qpt_state_file_abs_path}")
                            if os.path.exists(qpt_state_file_abs_path):
                                try:
                                    # Use timeout for QPT loading
                                    qpt_load_thread = threading.Thread(target=self._load_component_state, 
                                                                     args=(self.quantum_pt, qpt_state_file_abs_path, 'QPT', component_load_status))
                                    qpt_load_thread.daemon = True
                                    qpt_load_thread.start()
                                    qpt_load_thread.join(timeout=30)  # 30s timeout
                                    
                                    if qpt_load_thread.is_alive():
                                        self.logger.error("QPT state loading timed out")
                                        component_load_status['QPT'] = False
                                except Exception as e_qpt:
                                    self.logger.error(f"Error in QPT state loading thread: {str(e_qpt)}", exc_info=True)
                                    component_load_status['QPT'] = False
                            else:
                                self.logger.warning(f"QuantumProspectTheory state file not found at {qpt_state_file_abs_path}")
                    
                    # Log component loading results
                    component_load_time = time.time() - component_start_time
                    self.logger.debug(f"Component loading completed in {component_load_time:.2f}s with status: {component_load_status}")
                    
                    # Verify critical components loaded correctly
                    critical_components = ['QHA'] if 'QHA' in component_load_status else []
                    critical_failures = [comp for comp in critical_components if not component_load_status.get(comp, True)]
                    
                    if critical_failures:
                        self.logger.error(f"Critical component(s) failed to load: {critical_failures}")
                        # Consider if this should be a fatal error or continue with degraded functionality
                    
                except Exception as comp_e:
                    self.logger.error(f"Error during component state loading: {str(comp_e)}", exc_info=True)
                    # Continue with main state even if components fail
                
                # State load successful if we reached here without fatal errors
                qar_main_load_successful = True
                
                # Cache invalidation to prevent memory leaks
                load_cache.clear()
                
                # Log total loading time
                total_time = time.time() - start_time
                self.logger.info(f"QAR state loading completed in {total_time:.2f}s")
                
                # Validate state consistency
                self._validate_loaded_state()
                
                # Trigger garbage collection after large state load
                if 'gc' in sys.modules:
                    gc.collect()
                
                return qar_main_load_successful

        except FileNotFoundError:
            self.logger.error(f"QAR state file not found at {filepath}. Cannot load state.")
            return False
        except json.JSONDecodeError as e_json:
            self.logger.error(f"Error decoding JSON from QAR state file {filepath}: {e_json}", exc_info=True)
            return False
        except MemoryError:
            self.logger.critical(f"Memory error while loading QAR state. File may be too large or system low on memory.")
            return False
        except Exception as e:
            self.logger.error(f"General error loading QAR system state from {filepath}: {str(e)}", exc_info=True)
            return False
    def _restore_from_backup(self, backup_state: Dict[str, Any]) -> None:
        """Restore QAR state from a backup dictionary in case of loading failures.
        
        Args:
            backup_state: Dictionary containing backup values for QAR parameters
        """
        try:
            self.logger.warning("Rolling back to previous state due to loading error")
            if 'factor_weights' in backup_state:
                self.factor_weights = backup_state['factor_weights']
            if 'factor_importance' in backup_state:
                self.factor_importance = backup_state['factor_importance']
            if 'factors' in backup_state:
                self.factors = backup_state['factors']
            if 'baseline_weights' in backup_state:
                self.baseline_weights = backup_state['baseline_weights']
            if 'regime_specific_weights' in backup_state:
                self.regime_specific_weights = backup_state['regime_specific_weights']
            self.logger.info("State rollback completed successfully")
        except Exception as e:
            self.logger.error(f"Error during state rollback: {str(e)}", exc_info=True)
            
    def _load_component_state(self, component: Any, state_file_path: str, component_name: str, status_dict: Dict[str, bool]) -> None:
        """Helper method to load a component's state with error handling.
        
        This method is designed to be run in a separate thread with a timeout.
        
        Args:
            component: The component object to load state into
            state_file_path: Path to the component's state file
            component_name: Name of the component for logging
            status_dict: Dictionary to track loading status
        """
        try:
            start_time = time.time()
            component.load_state(state_file_path)
            load_time = time.time() - start_time
            self.logger.info(f"{component_name} state loaded successfully in {load_time:.2f}s")
            status_dict[component_name] = True
        except Exception as e:
            self.logger.error(f"Failed to load {component_name} state: {str(e)}", exc_info=True)
            status_dict[component_name] = False
            
    def _validate_loaded_state(self) -> None:
        """Validate the consistency of the loaded state.
        
        Checks for missing factors, inconsistent weights, and other potential issues.
        Performs auto-correction where possible.
        """
        try:
            # Check factors and weights consistency
            missing_weights = [f for f in self.factors if f not in self.factor_weights]
            if missing_weights:
                self.logger.warning(f"Factors missing from weights: {missing_weights}")
                # Auto-fix by adding default weights
                for f in missing_weights:
                    self.factor_weights[f] = 0.5  # Default weight
                    self.logger.info(f"Added default weight 0.5 for factor: {f}")
            
            # Check for extra weights not in factors list
            extra_weights = [f for f in self.factor_weights if f not in self.factors]
            if extra_weights:
                self.logger.warning(f"Extra weights found for factors not in factor list: {extra_weights}")
            
            # Validate weight ranges
            invalid_weights = [f for f, w in self.factor_weights.items() if not (0 <= w <= 1)]
            if invalid_weights:
                self.logger.warning(f"Invalid weight values found for factors: {invalid_weights}")
                # Auto-correct invalid weights
                for f in invalid_weights:
                    self.factor_weights[f] = max(0, min(1, self.factor_weights[f]))
                    self.logger.info(f"Corrected weight for {f} to {self.factor_weights[f]}")
            
            # Validate regime weights
            for regime, weights in self.regime_specific_weights.items():
                if not isinstance(weights, dict):
                    self.logger.warning(f"Invalid weights format for regime {regime}, correcting")
                    self.regime_specific_weights[regime] = {}
                    continue
                    
                # Check for missing factors in regime weights
                for f in self.factors:
                    if f not in weights:
                        weights[f] = self.factor_weights.get(f, 0.5)
                        self.logger.debug(f"Added missing factor {f} to regime {regime} with weight {weights[f]}")
                        
                # Validate weight ranges for regime weights
                invalid_regime_weights = [f for f, w in weights.items() if not (0 <= w <= 1)]
                if invalid_regime_weights:
                    self.logger.warning(f"Invalid weights in regime {regime} for factors: {invalid_regime_weights}")
                    # Auto-correct invalid weights
                    for f in invalid_regime_weights:
                        weights[f] = max(0, min(1, weights[f]))
            
            # Ensure decision threshold is valid
            if not (0 <= self.decision_threshold <= 1):
                original = self.decision_threshold
                self.decision_threshold = max(0, min(1, self.decision_threshold))
                self.logger.warning(f"Corrected invalid decision threshold from {original} to {self.decision_threshold}")
            
            # Validate memory length
            if self.memory_length < 1:
                self.memory_length = max(1, self.memory_length)
                self.logger.warning(f"Corrected invalid memory length to {self.memory_length}")
            
            self.logger.debug("State validation and auto-correction completed")
        except Exception as e:
            self.logger.error(f"Error during state validation: {str(e)}", exc_info=True)

    def __del__(self):
        """Clean up resources when object is destroyed."""
        try:
            self.stop_resource_monitoring()
            self.logger.debug("QAR resources released")
        except:
            pass
        

def create_quantum_agentic_reasoning(
    hw_manager: HardwareManager,  # Require manager
    num_factors: int = 8,
    decision_threshold: float = 0.6,
) -> QuantumAgenticReasoning:
    """Creates QAR instance using the provided HardwareManager."""
    return QuantumAgenticReasoning(
        hardware_manager=hw_manager,
        num_factors=num_factors,
        decision_threshold=decision_threshold,
    )
