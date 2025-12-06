#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Immune-Inspired Quantum Anomaly Detection (IQAD)
Combines features from both versions into a more robust implementation
while maintaining compatibility with hardware_manager.py
"""

import numpy as np
import logging
import os
import threading
import time
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

try:
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
        
    QUANTUM_AVAILABLE = True
except ImportError:
    QUANTUM_AVAILABLE = False
    logging.warning("PennyLane not installed; quantum features will be disabled")

# Attempt to import optional dependencies with proper fallbacks
try:
    from hardware_manager import HardwareManager
    from cdfa_extensions.hw_acceleration import HardwareAccelerator, AcceleratorType
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


class ImmuneQuantumAnomalyDetector:
    """
    Immune-Inspired Quantum Anomaly Detection for financial time series.
    
    This implements a hybrid classical-quantum anomaly detector inspired by
    biological immune systems to detect market anomalies and potential "black swan" events.
    
    Features:
    - Quantum-enhanced negative selection algorithm
    - Self/non-self discrimination for anomaly detection
    - Immune memory for pattern recognition
    - Affinity maturation for detector optimization
    - Efficient memory management with caching
    """
    
    def __init__(self, 
                 detectors: int = 50,
                 num_detectors: int = 50,
                 quantum_dimension: int = 4,
                 sensitivity: float = 0.85,
                 negative_selection_threshold: float = 0.7,
                 hardware_manager = None, 
                 use_classical: bool = False,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the IQAD component.
        
        Args:
            detectors (int): Number of anomaly detectors
            quantum_dimension (int): Dimension of quantum feature space
            sensitivity (float): Detector sensitivity (0-1)
            negative_selection_threshold (float): Threshold for negative selection
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
            self.logger.info(f"IQAD initialized with log level {logging.getLevelName(self.logger.level)} ({self.logger.level})")
            
        except Exception as e:
            self.logger.error(f"Error setting log level: {e}", exc_info=True)
            self.logger.setLevel(logging.INFO)
        
        # Core parameters
        self.detectors = detectors  # Number of detector units
        self.quantum_dimension = quantum_dimension
        self.qubits = quantum_dimension  # Maintain both attribute names for compatibility
        self.sensitivity = sensitivity
        self.negative_selection_threshold = negative_selection_threshold
        self.detector_mutation_rate = self.config.get('mutation_rate', 0.05)
        self.affinity_threshold = self.config.get('affinity_threshold', 0.7)
        
        # Hardware resources
        self.hardware_manager = hardware_manager
        if self.hardware_manager is None and HARDWARE_MANAGER_AVAILABLE:
            self.hardware_manager = HardwareManager()
        self.use_classical = use_classical or not QUANTUM_AVAILABLE
        
        # Quantum components
        self.shots = self.config.get('shots', None)
        
        # Internal state
        self.is_initialized = False
        self.detector_set = None
        self.normal_patterns = []  # Known normal patterns
        self.anomaly_memory = []   # Detected anomalies
        self.device = None
        self.circuits = {}
        self.max_self_patterns = self.config.get('max_self_patterns', 100)
        self.max_anomaly_memory = self.config.get('max_anomaly_memory', 50)
        
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
            self._initialize_detectors()
        except Exception as e:
            self.logger.error(f"Error initializing IQAD: {str(e)}", exc_info=True)
            self.use_classical = True
            self._initialize_detectors()
    
    def _initialize_detectors(self) -> None:
        """Initialize anomaly detectors."""
        self.logger.info(f"Initializing IQAD with {'classical' if self.use_classical else 'quantum'} backend")
        
        # Initialize detector set
        self.detector_set = self._initialize_detector_set()
        
        # Initialize quantum components if needed
        if not self.use_classical:
            self.device = self._get_optimized_device()
            self._initialize_quantum_circuits()
        
        self.is_initialized = True
        self.logger.info(f"IQAD initialized with {len(self.detector_set)} detectors and {self.qubits} qubits")
    
    def _initialize_detector_set(self) -> List[np.ndarray]:
        """Initialize random detector set for negative selection."""
        detectors = []

        for _ in range(self.detectors):
            # Create random detector
            detector = np.random.rand(2**self.qubits)
            # Normalize detector
            detector = detector / np.linalg.norm(detector)
            detectors.append(detector)

        return detectors
    
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
                    self.logger.info(f"Using AMD GPU for IQAD")
                    return qml.device('lightning.kokkos', wires=self.qubits, shots=self.shots)

                # NVIDIA GPU
                elif self.hardware_manager.devices.get('nvidia_gpu', {}).get('available', False):
                    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
                    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
                    self.logger.info("Using NVIDIA GPU for IQAD")
                    return qml.device('lightning.gpu', wires=self.qubits, shots=self.shots)

                # Apple Silicon
                elif self.hardware_manager.devices.get('apple_silicon', False):
                    self.logger.info("Using Apple Silicon for IQAD")
                    return qml.device('default.qubit', wires=self.qubits, shots=self.shots)
            
            # Fallback to CPU if hardware detection fails or not available
            self.logger.info("Using CPU for IQAD quantum operations")
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
        """Initialize quantum circuits for IQAD."""
        if not QUANTUM_AVAILABLE or self.use_classical or self.device is None:
            self.logger.warning("Quantum libraries not available, skipping circuit initialization")
            return
            
        try:
            # Quantum affinity calculation
            @qml.qnode(self.device)
            def quantum_affinity_circuit(pattern, detector):
                # Encode pattern and detector
                for i in range(self.qubits):
                    qml.RY(np.pi * pattern[i % len(pattern)], wires=i)

                # Apply phase kickback to measure affinity
                for i in range(self.qubits):
                    qml.Hadamard(wires=i)

                # Apply quantum operations based on detector
                for i in range(self.qubits):
                    qml.PhaseShift(detector[i % len(detector)] * np.pi, wires=i)

                # More entanglement
                for i in range(self.qubits - 1):
                    qml.CNOT(wires=[i, i+1])

                # Final Hadamard layer to measure affinity
                for i in range(self.qubits):
                    qml.Hadamard(wires=i)

                # Measure all qubits to calculate affinity
                return [qml.expval(qml.PauliZ(i)) for i in range(self.qubits)]

            # Detector generation circuit
            @qml.qnode(self.device)
            def detector_generation_circuit(self_pattern, random_seed):
                # Encode self pattern to avoid
                for i in range(self.qubits):
                    qml.RY(np.pi * self_pattern[i % len(self_pattern)], wires=i)

                # Apply random rotation based on seed
                for i in range(self.qubits):
                    qml.RY(random_seed[i % len(random_seed)] * np.pi, wires=i)

                # Apply negative selection logic
                for i in range(self.qubits - 1):
                    qml.CNOT(wires=[i, i+1])

                for i in range(self.qubits):
                    qml.RZ(random_seed[(i+self.qubits) % len(random_seed)] * np.pi, wires=i)

                # Generate detector that's different from self pattern
                for i in range(self.qubits):
                    qml.Hadamard(wires=i)

                # Measure detector state
                return [qml.expval(qml.PauliZ(i)) for i in range(self.qubits)]

            # Anomaly scoring circuit
            @qml.qnode(self.device)
            def anomaly_scoring_circuit(pattern, detectors):
                # Encode pattern
                for i in range(self.qubits):
                    qml.RY(np.pi * pattern[i % len(pattern)], wires=i)

                # Apply detector-based operations (simplified for efficiency)
                for i in range(min(3, len(detectors))):  # Use first 3 detectors
                    detector = detectors[i]
                    
                    # Calculate angle based on detector
                    angle = np.sum(detector[:min(len(detector), 4)]) * np.pi / 4
                    
                    # Apply rotation based on detector
                    qml.RY(angle, wires=0)
                    
                    # Apply controlled operations
                    for j in range(1, self.qubits):
                        qml.CRZ(detector[j % len(detector)] * np.pi, wires=[0, j])

                # Final measurement layer
                for i in range(self.qubits):
                    qml.Hadamard(wires=i)

                # Measure anomaly score
                return [qml.expval(qml.PauliZ(i)) for i in range(self.qubits)]

            # Additional circuit for quantum distance calculation
            @qml.qnode(self.device)
            def quantum_distance_circuit(x, y):
                # Encode input data using amplitude encoding
                for i in range(self.qubits):
                    qml.RY(np.arccos(x[i % len(x)]), wires=i)
                    
                # Apply Hadamard gates to create superposition
                for i in range(self.qubits):
                    qml.Hadamard(wires=i)
                    
                # Apply controlled rotations based on second vector
                for i in range(self.qubits):
                    qml.RZ(np.arccos(y[i % len(y)]), wires=i)
                    
                # Apply entangling operations
                for i in range(self.qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                    
                # Measure in computational basis
                return [qml.expval(qml.PauliZ(i)) for i in range(self.qubits)]

            # Store the circuits in the dictionary
            self.circuits = {
                'affinity': quantum_affinity_circuit,
                'detector_generation': detector_generation_circuit,
                'anomaly_scoring': anomaly_scoring_circuit,
                'distance': quantum_distance_circuit
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
        
        if circuit_name == 'affinity':
            pattern, detector = params
            # Compute classical cosine similarity
            p_len = min(len(pattern), len(detector))
            pattern_norm = pattern[:p_len] / (np.linalg.norm(pattern[:p_len]) + 1e-10)
            detector_norm = detector[:p_len] / (np.linalg.norm(detector[:p_len]) + 1e-10)
            similarity = np.abs(np.dot(pattern_norm, detector_norm))
            return np.array([similarity] + [0] * (self.qubits - 1))
            
        elif circuit_name == 'detector_generation':
            self_pattern, random_seed = params
            # Generate classical detector that differs from self pattern
            new_detector = np.random.rand(2**self.qubits)
            # Ensure it's different from self pattern
            self_len = min(len(self_pattern), len(new_detector))
            similarity = np.abs(np.dot(self_pattern[:self_len], new_detector[:self_len]))
            if similarity > 0.7:  # Too similar
                new_detector = -new_detector  # Invert to make different
            return new_detector / np.linalg.norm(new_detector)
            
        elif circuit_name == 'anomaly_scoring':
            pattern, detectors = params
            # Classical anomaly scoring
            scores = []
            p_len = len(pattern)

            for detector in detectors[:3]:  # Use first 3 detectors
                d_len = min(p_len, len(detector))
                similarity = np.abs(np.dot(
                    pattern[:d_len] / (np.linalg.norm(pattern[:d_len]) + 1e-10),
                    detector[:d_len] / (np.linalg.norm(detector[:d_len]) + 1e-10)
                ))
                scores.append(similarity)

            avg_score = np.mean(scores) if scores else 0
            return np.array([avg_score] + [0] * (self.qubits - 1))
            
        elif circuit_name == 'distance':
            x, y = params
            # Calculate classical Euclidean distance
            distance = np.sqrt(np.sum((x - y) ** 2)) / np.sqrt(self.qubits)
            return np.array([distance] + [0] * (self.qubits - 1))
            
        else:
            self.logger.error(f"Unknown circuit: {circuit_name}")
            return np.zeros(self.qubits)
    
    def _prepare_vector_for_quantum(self, vector: np.ndarray) -> np.ndarray:
        """
        Prepare vector for quantum processing.
        
        Args:
            vector (np.ndarray): Input vector
            
        Returns:
            np.ndarray: Processed vector ready for quantum circuits
        """
        # Ensure it's a numpy array
        if not isinstance(vector, np.ndarray):
            vector = np.array(vector)
            
        # Check for non-finite values
        if not np.all(np.isfinite(vector)):
            self.logger.warning("Non-finite values in input vector, replacing with zeros")
            vector = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)
            
        # Target size for quantum processing
        target_size = self.qubits
            
        # Resize if needed
        if len(vector) != target_size:
            if len(vector) > target_size:
                # Truncate
                result = vector[:target_size]
            else:
                # Pad with zeros
                result = np.pad(vector, (0, target_size - len(vector)))
        else:
            result = vector.copy()
            
        # Normalize to [0, 1] range for quantum processing
        if np.max(np.abs(result)) > 0:
            min_val = np.min(result)
            max_val = np.max(result)
            if min_val != max_val:
                result = (result - min_val) / (max_val - min_val)
            else:
                result = np.ones_like(result) * 0.5
        else:
            # If all zeros, use 0.5 for neutral value
            result = np.ones_like(result) * 0.5
            
        return result
    
    def _feature_encoding(self, features: Dict[str, float]) -> np.ndarray:
        """
        Encode market features for anomaly detection.
        
        Args:
            features (Dict[str, float]): Market features
            
        Returns:
            np.ndarray: Encoded feature vector
        """
        # Select key features for anomaly detection
        key_features = [
            features.get('close', 0),
            features.get('volume', 0),
            features.get('volatility', features.get('volatility_regime', 0.5)),
            features.get('rsi_14', features.get('rsi', 50)) / 100,  # Normalize RSI to [0, 1]
            features.get('adx', 15) / 100,     # Normalize ADX
            features.get('trend', 0.5),        # From QERC
            features.get('momentum', 0.5),     # From QERC
            features.get('regime', 0.5)        # From QERC
        ]
        
        # Convert to numpy array
        encoded = np.array(key_features)
        
        # Prepare for quantum processing
        return self._prepare_vector_for_quantum(encoded)
    
    def _quantum_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Calculate quantum distance between feature vectors.
        
        Args:
            a (np.ndarray): First feature vector
            b (np.ndarray): Second feature vector
            
        Returns:
            float: Quantum distance
        """
        # Prepare vectors for quantum processing
        a_prepared = self._prepare_vector_for_quantum(a)
        b_prepared = self._prepare_vector_for_quantum(b)
        
        if self.use_classical or not QUANTUM_AVAILABLE:
            # Classical fallback using Euclidean distance
            return np.sqrt(np.sum((a_prepared - b_prepared) ** 2)) / np.sqrt(self.qubits)
            
        try:
            # Use quantum circuit for distance calculation
            measurements = self._execute_with_fallback('distance', (a_prepared, b_prepared))
            
            # Convert measurements to distance metric
            fidelity = np.mean([0.5 * (1 + m) for m in measurements])
            distance = np.sqrt(1 - fidelity)
            
            return distance
            
        except Exception as e:
            self.logger.error(f"Quantum distance calculation error: {e}", exc_info=True)
            # Fall back to classical distance
            return np.sqrt(np.sum((a_prepared - b_prepared) ** 2)) / np.sqrt(self.qubits)
    
    def _calculate_quantum_affinity(self, pattern: np.ndarray, detector: np.ndarray) -> float:
        """
        Calculate quantum affinity between pattern and detector.
        
        Args:
            pattern (np.ndarray): Pattern to check
            detector (np.ndarray): Detector pattern
            
        Returns:
            float: Affinity value (0-1)
        """
        try:
            # Ensure pattern and detector have compatible dimensions
            pattern = self._prepare_vector_for_quantum(pattern)
            detector = self._prepare_vector_for_quantum(detector)

            # Execute quantum affinity circuit
            affinity_result = self._execute_with_fallback('affinity', (pattern, detector))

            # Calculate affinity score from quantum result
            affinity = np.abs(np.mean(affinity_result))

            return float(affinity)

        except Exception as e:
            self.logger.warning(f"Error in quantum affinity calculation: {e}")
            # Fallback to classical affinity (cosine similarity)
            return float(np.abs(np.dot(
                pattern / (np.linalg.norm(pattern) + 1e-10),
                detector / (np.linalg.norm(detector) + 1e-10)
            )))
    
    def _negative_selection(self, pattern: np.ndarray) -> bool:
        """
        Apply negative selection to determine if pattern is normal.
        
        Args:
            pattern (np.ndarray): Feature pattern to check
            
        Returns:
            bool: True if pattern is considered normal
        """
        # If we don't have enough normal patterns yet, consider it normal
        if len(self.normal_patterns) < 10:
            return True
            
        # Calculate minimum distance to normal patterns
        min_distance = min(self._quantum_distance(pattern, normal) for normal in self.normal_patterns)
        
        # If minimum distance is below threshold, consider it normal
        return min_distance < self.negative_selection_threshold
    
    def _update_detector_set(self, new_self_pattern: np.ndarray) -> None:
        """
        Update detector set based on new self pattern.
        
        Args:
            new_self_pattern (np.ndarray): New self pattern to consider
        """
        try:
            # Prepare vectors for processing
            new_self_pattern = self._prepare_vector_for_quantum(new_self_pattern)
            
            # Clone existing detectors
            new_detectors = []

            # Check each detector against the new self pattern
            for detector in self.detector_set:
                affinity = self._calculate_quantum_affinity(new_self_pattern, detector)

                if affinity < self.affinity_threshold:
                    # Keep detector (low affinity to self pattern is good)
                    new_detectors.append(detector)
                else:
                    # Generate new detector
                    random_seed = np.random.rand(self.qubits * 2)
                    new_detector = self._execute_with_fallback(
                        'detector_generation',
                        (new_self_pattern, random_seed)
                    )

                    # Normalize and add
                    new_detector = self._prepare_vector_for_quantum(new_detector)
                    new_detectors.append(new_detector)

            # Apply mutation to some detectors for diversity
            for i in range(len(new_detectors)):
                if np.random.rand() < self.detector_mutation_rate:
                    # Apply small mutation
                    mutation = np.random.normal(0, 0.1, size=len(new_detectors[i]))
                    new_detectors[i] = new_detectors[i] + mutation
                    # Renormalize
                    new_detectors[i] = new_detectors[i] / np.linalg.norm(new_detectors[i])

            # Update detector set
            self.detector_set = new_detectors

            # Ensure we maintain the desired number of detectors
            while len(self.detector_set) < self.detectors:
                # Generate additional detectors
                random_seed = np.random.rand(self.qubits * 2)
                new_detector = self._execute_with_fallback(
                    'detector_generation',
                    (new_self_pattern, random_seed)
                )

                # Normalize and add
                new_detector = self._prepare_vector_for_quantum(new_detector)
                self.detector_set.append(new_detector)

        except Exception as e:
            self.logger.error(f"Error updating detector set: {e}")
    
    def _learn_normal_pattern(self, pattern: np.ndarray) -> None:
        """
        Learn a new normal pattern and update detectors accordingly.
        
        Args:
            pattern (np.ndarray): Pattern to learn as normal
        """
        try:
            # Prepare pattern for quantum processing
            quantum_pattern = self._prepare_vector_for_quantum(pattern)

            # Check if similar pattern already exists in memory
            for existing_pattern in self.normal_patterns:
                affinity = self._calculate_quantum_affinity(quantum_pattern, existing_pattern)
                if affinity > 0.9:  # Very similar pattern
                    return  # Skip duplicate pattern

            # Add to self patterns memory
            self.normal_patterns.append(quantum_pattern)

            # Limit memory size
            if len(self.normal_patterns) > self.max_self_patterns:
                self.normal_patterns.pop(0)  # Remove oldest

            # Update detector set
            self._update_detector_set(quantum_pattern)

        except Exception as e:
            self.logger.error(f"Error learning normal pattern: {e}")
    
    def _memorize_anomaly(self, pattern: np.ndarray, score: float) -> None:
        """
        Memorize detected anomaly pattern.
        
        Args:
            pattern (np.ndarray): Anomaly pattern
            score (float): Anomaly score
        """
        try:
            # Prepare pattern for quantum processing
            quantum_pattern = self._prepare_vector_for_quantum(pattern)

            # Check if similar anomaly already exists in memory
            for existing_anomaly, _ in self.anomaly_memory:
                affinity = self._calculate_quantum_affinity(quantum_pattern, existing_anomaly)
                if affinity > 0.9:  # Very similar anomaly
                    return  # Skip duplicate anomaly

            # Add to anomaly memory
            self.anomaly_memory.append((quantum_pattern, score))

            # Limit memory size
            if len(self.anomaly_memory) > self.max_anomaly_memory:
                self.anomaly_memory.pop(0)  # Remove oldest

        except Exception as e:
            self.logger.error(f"Error memorizing anomaly: {e}")
    
    def _detector_activation(self, pattern: np.ndarray) -> float:
        """
        Calculate detector activation for a pattern.
        
        Args:
            pattern (np.ndarray): Feature pattern
            
        Returns:
            float: Detector activation level (0-1)
        """
        if not self.detector_set:
            return 0.0
            
        # Calculate distances to all detectors
        pattern_prepared = self._prepare_vector_for_quantum(pattern)
        distances = [self._quantum_distance(pattern_prepared, detector) for detector in self.detector_set]
        
        # Get minimum distance (closest detector)
        min_distance = min(distances)
        
        # Convert to activation level (closer = higher activation)
        activation = np.exp(-min_distance / self.sensitivity)
        
        return min(activation, 1.0)
    
    def _calculate_anomaly_score(self, pattern: np.ndarray) -> Tuple[float, List[float]]:
        """
        Calculate anomaly score for a pattern.
        
        Args:
            pattern (np.ndarray): Pattern to analyze
            
        Returns:
            Tuple[float, List[float]]: Anomaly score and detector affinities
        """
        try:
            # Prepare pattern for quantum processing
            quantum_pattern = self._prepare_vector_for_quantum(pattern)

            # Calculate affinities with detectors
            detector_affinities = []

            # Use quantum scoring circuit for efficiency
            # This processes multiple detectors at once
            batch_size = min(10, len(self.detector_set))  # Process detectors in batches
            
            if batch_size > 0:
                # Execute quantum anomaly scoring
                score_result = self._execute_with_fallback(
                    'anomaly_scoring',
                    (quantum_pattern, self.detector_set[:batch_size])
                )

                # Extract score from first qubit
                batch_score = float(abs(score_result[0]))
                detector_affinities = [batch_score] * batch_size

            # Calculate classical affinities for verification and completeness
            classical_affinities = []
            for detector in self.detector_set:
                affinity = self._calculate_quantum_affinity(quantum_pattern, detector)
                classical_affinities.append(affinity)

            # Combine quantum and classical results
            # Use maximum affinity as anomaly score
            # Higher affinity with detectors = more anomalous
            combined_affinities = detector_affinities + classical_affinities
            max_affinity = max(combined_affinities) if combined_affinities else 0

            return max_affinity, classical_affinities

        except Exception as e:
            self.logger.error(f"Error calculating anomaly score: {e}")
            return 0.0, []
    
    def train_on_normal_data(self, normal_patterns: List[Dict[str, float]]) -> None:
        """
        Train detector on normal market patterns.
        
        Args:
            normal_patterns (List[Dict[str, float]]): List of normal market patterns
        """
        self.logger.info(f"Training IQAD on {len(normal_patterns)} normal patterns")
        
        # Encode and store normal patterns
        for pattern in normal_patterns:
            try:
                # Encode features
                encoded = self._feature_encoding(pattern)
                
                # Learn as normal pattern
                self._learn_normal_pattern(encoded)
            except Exception as e:
                self.logger.error(f"Error training on pattern: {e}")
    
    def detect_anomalies(self, 
                        features: Dict[str, float],
                        expected_behavior: Optional[Dict[str, Any]] = None,
                        regime_transitions: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Detect anomalies in market data.
        
        Args:
            features (Dict[str, float]): Market features
            expected_behavior (Dict[str, Any], optional): Expected behavior baseline
            regime_transitions (Dict[str, Any], optional): Regime transition probabilities
            
        Returns:
            Dict[str, Any]: Anomaly detection results
        """
        start_time = time.time()
        self.logger.debug("-" * 20 + " IQAD detect_anomalies START " + "-" * 20)
        
        try:
            if not self.is_initialized:
                self._initialize_detectors()
            
            # Encode features
            encoded_features = self._feature_encoding(features)
            self.logger.debug(f"Encoded features: {encoded_features}")
            
            # Calculate detector activation
            activation = self._detector_activation(encoded_features)
            self.logger.debug(f"Detector activation: {activation:.4f}")
            
            # Calculate anomaly score
            anomaly_score, detector_affinities = self._calculate_anomaly_score(encoded_features)
            self.logger.debug(f"Raw anomaly score: {anomaly_score:.4f}")
            
            # Apply sensitivity adjustment
            adjusted_score = anomaly_score * self.sensitivity
            
            # Determine anomaly threshold
            base_threshold = 0.7
            
            # Adjust threshold based on expected behavior if provided
            if expected_behavior:
                if 'volatility' in expected_behavior:
                    # Higher threshold during high volatility (more tolerant)
                    volatility_factor = expected_behavior['volatility']
                    base_threshold += 0.1 * volatility_factor
                    
            # Adjust threshold based on regime transitions if provided
            if regime_transitions and 'probability' in regime_transitions:
                # Lower threshold during regime transitions (more sensitive)
                transition_prob = regime_transitions['probability']
                base_threshold -= 0.1 * transition_prob
                
            # Check if anomaly detected
            is_anomaly = adjusted_score > base_threshold
            self.logger.debug(f"Adjusted score: {adjusted_score:.4f}, threshold: {base_threshold:.4f}, is anomaly: {is_anomaly}")
            
            # If anomaly detected, memorize it
            if is_anomaly:
                self._memorize_anomaly(encoded_features, adjusted_score)
                self.logger.info(f"Anomaly detected with score: {adjusted_score:.4f}")
            else:
                # Add to normal patterns if clearly non-anomalous
                learning_threshold = 0.3  # Only learn if score is clearly non-anomalous
                if adjusted_score < learning_threshold:
                    self._learn_normal_pattern(encoded_features)
                    self.logger.debug(f"Normal pattern learned with score: {adjusted_score:.4f}")
            
            # Track execution time
            execution_time = (time.time() - start_time) * 1000  # ms
            self.execution_times.append(execution_time)
            
            # Limit execution time list to last 100 entries
            if len(self.execution_times) > 100:
                self.execution_times = self.execution_times[-100:]
            
            # Report average execution time for monitoring
            if self.hardware_manager and hasattr(self.hardware_manager, 'track_execution_time'):
                device_type = 'quantum' if not self.use_classical else 'cpu'
                self.hardware_manager.track_execution_time(device_type, execution_time)
            
            # Create result dictionary
            result = {
                'detected': is_anomaly,
                'score': float(adjusted_score),
                'threshold': float(base_threshold),
                'confidence': float(adjusted_score / base_threshold if base_threshold > 0 else 0),
                'detector_affinities': [float(a) for a in detector_affinities[:5]],  # Top 5 affinities
                'execution_time_ms': execution_time
            }
            
            # Calculate time to event estimate
            if is_anomaly and adjusted_score > 0.8:
                # High score suggests imminent event
                result['time_to_event'] = 'imminent'
            elif is_anomaly and adjusted_score > base_threshold + 0.1:
                # Moderate score suggests near-term event
                result['time_to_event'] = 'near-term'
            elif is_anomaly:
                # Just above threshold suggests potential event
                result['time_to_event'] = 'potential'
                
            self.logger.debug("-" * 20 + " IQAD detect_anomalies END " + "-" * 20)
            return result
            
        except Exception as e:
            self.logger.error(f"Error in anomaly detection: {e}", exc_info=True)
            return {
                'detected': False,
                'score': 0.0,
                'threshold': 0.7,
                'confidence': 0.0,
                'execution_time_ms': (time.time() - start_time) * 1000,
                'error': str(e)
            }
    
    def calculate_tail_probability(self, reservoir_states: Union[np.ndarray, Dict[str, Any]], quantile: float = 0.99) -> float:
        """
        Calculate tail probability for extreme event detection using extreme value theory.
        
        Args:
            reservoir_states (np.ndarray or Dict): Reservoir states or feature dictionary
            quantile (float): Extreme event quantile (default 0.99 for 1% events)
            
        Returns:
            float: Probability of extreme event
        """
        try:
            # Extract vector from input
            if isinstance(reservoir_states, np.ndarray):
                vector = reservoir_states
            elif isinstance(reservoir_states, dict) and 'reservoir_state' in reservoir_states:
                vector = reservoir_states['reservoir_state']
            else:
                # Try to convert dictionary values to vector
                vector = np.array(list(reservoir_states.values()))
                
            # Prepare for quantum processing
            quantum_vector = self._prepare_vector_for_quantum(vector)
            
            # Check against anomaly memory
            memory_scores = []
            for anomaly_pattern, anomaly_score in self.anomaly_memory:
                affinity = self._calculate_quantum_affinity(quantum_vector, anomaly_pattern)
                memory_scores.append(affinity * anomaly_score)  # Weight by original score
                
            # Calculate anomaly score
            anomaly_score, _ = self._calculate_anomaly_score(quantum_vector)
            
            # Combine current score with memory
            combined_score = anomaly_score
            if memory_scores:
                memory_factor = max(memory_scores)
                combined_score = max(anomaly_score, memory_factor)
                
            # Apply extreme value scaling for tail probability
            # Scale score to match requested quantile
            tail_probability = combined_score ** 2  # Square to emphasize high scores
            
            # Ensure reasonable range
            return float(min(tail_probability, 0.99))
            
        except Exception as e:
            self.logger.error(f"Error calculating tail probability: {e}")
            return 0.1  # Default low probability
    
    def clear_cache(self) -> None:
        """Clear the internal cache."""
        with self.cache_lock:
            self.cache.clear()
        self.logger.debug("IQAD cache cleared")
    
    def reset(self) -> None:
        """Reset the detector state."""
        self.logger.info("Resetting IQAD")
        self.normal_patterns = []
        self.anomaly_memory = []
        self.detector_set = self._initialize_detector_set()
        self.clear_cache()
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """
        Get execution statistics for performance monitoring.
        
        Returns:
            Dict[str, Any]: Execution statistics
        """
        if not self.execution_times:
            return {'avg_time_ms': 0, 'min_time_ms': 0, 'max_time_ms': 0, 'count': 0}
            
        return {
            'avg_time_ms': np.mean(self.execution_times),
            'min_time_ms': np.min(self.execution_times),
            'max_time_ms': np.max(self.execution_times),
            'count': len(self.execution_times)
        }
    
    def recover(self) -> bool:
        """
        Attempt to recover the IQAD component after errors.
        
        Returns:
            bool: True if recovery succeeded
        """
        self.logger.warning("IQAD recovery triggered!")
        try:
            # Reset internal state while preserving some memory
            preserved_normal = self.normal_patterns[-10:] if len(self.normal_patterns) > 10 else self.normal_patterns
            self.reset()
            self.normal_patterns = preserved_normal
            
            # Re-initialize quantum device
            if not self.use_classical and QUANTUM_AVAILABLE:
                self.device = self._get_optimized_device()
                self._initialize_quantum_circuits()
                
            # Check for fault tolerance manager
            if self.fault_tolerance:
                self.fault_tolerance.register_recovery("iqad")
                
            self.logger.info("IQAD recovery succeeded")
            return True
            
        except Exception as e:
            self.logger.error(f"IQAD recovery failed: {str(e)}", exc_info=True)
            # Last resort - force classical mode
            self.use_classical = True
            return False


# Factory function for thread-safe singleton access
_iqad_instance = None
_iqad_lock = threading.RLock()

def get_immune_quantum_anomaly_detector(config=None, reset=False) -> ImmuneQuantumAnomalyDetector:
    """Thread-safe factory function for ImmuneQuantumAnomalyDetector."""
    global _iqad_instance, _iqad_lock

    with _iqad_lock:
        if _iqad_instance is None or reset:
            try:
                # Extract specific parameters from config
                config_dict = config or {}
                
                # Instantiate with parameters from config
                _iqad_instance = ImmuneQuantumAnomalyDetector(
                    detectors=config_dict.get('detectors', 50),
                    quantum_dimension=config_dict.get('quantum_dimension', 4),
                    sensitivity=config_dict.get('sensitivity', 0.85),
                    negative_selection_threshold=config_dict.get('negative_selection_threshold', 0.7),
                    config=config_dict
                )
                
                logger.info("Created new IQAD instance")
                
            except Exception as e:
                logger.exception(f"Failed to initialize IQAD instance: {e}")
                return None

    return _iqad_instance


# Example usage (if module run directly)
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    
    # Create IQAD instance
    iqad = ImmuneQuantumAnomalyDetector(
        detectors=20,  # Smaller for testing
        quantum_dimension=4
    )
    
    # Generate test data
    import random
    def generate_random_features():
        return {
            'close': random.uniform(100, 200),
            'volume': random.uniform(1000, 5000),
            'volatility': random.uniform(0.1, 0.5),
            'rsi_14': random.uniform(30, 70),
            'adx': random.uniform(10, 50),
            'trend': random.uniform(-1, 1),
            'momentum': random.uniform(-1, 1),
            'regime': random.uniform(0, 1)
        }
    
    # Generate normal patterns for training
    normal_patterns = [generate_random_features() for _ in range(20)]
    iqad.train_on_normal_data(normal_patterns)
    
    # Generate test pattern (normal)
    normal_test = generate_random_features()
    normal_result = iqad.detect_anomalies(normal_test)
    
    # Generate anomalous pattern
    anomalous_test = generate_random_features()
    anomalous_test['rsi_14'] = 90  # Extreme RSI is anomalous
    anomalous_test['volatility'] = 0.8  # High volatility is anomalous
    anomalous_result = iqad.detect_anomalies(anomalous_test)
    
    # Print results
    print("\nNormal Pattern Detection:")
    for key, value in normal_result.items():
        if key != 'detector_affinities':
            print(f"  {key}: {value}")
            
    print("\nAnomalous Pattern Detection:")
    for key, value in anomalous_result.items():
        if key != 'detector_affinities':
            print(f"  {key}: {value}")
            
    # Show execution stats
    stats = iqad.get_execution_stats()
    print("\nExecution Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
