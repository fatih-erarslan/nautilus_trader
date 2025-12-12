#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Neuromorphic Quantum Optimizer (NQO)
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
    import pennylane.numpy as qnp
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


class NeuromorphicQuantumOptimizer:
    """
    Neuromorphic Quantum Optimizer (NQO) implementation.

    Features:
    - Neural network preprocessing for quantum parameter optimization
    - Parameter space exploration with quantum circuits
    - Adaptive learning mechanisms with neuromorphic principles
    - Hardware-optimized quantum operations
    - Trading-specific optimization methods

    Resource requirements:
    - 3-5 qubits
    - 1.5GB memory
    - 100-150ms typical execution time
    """

    def __init__(self, 
                 neurons: int = 128,
                 qubits: int = 4,
                 adaptivity: float = 0.7,
                 learning_rate: float = 0.01,
                 hardware_manager = None, 
                 use_classical: bool = False,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the NQO component.
        
        Args:
            neurons (int): Number of neuromorphic neurons
            qubits (int): Number of qubits for quantum kernel
            adaptivity (float): Adaptivity parameter (0-1)
            learning_rate (float): Learning rate for parameter updates
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
            self.logger.info(f"NQO initialized with log level {logging.getLevelName(self.logger.level)} ({self.logger.level})")
            
        except Exception as e:
            self.logger.error(f"Error setting log level: {e}", exc_info=True)
            self.logger.setLevel(logging.INFO)
        
        # Core parameters
        self.neurons = neurons
        self.qubits = qubits
        self.adaptivity = adaptivity
        self.learning_rate = learning_rate
        self.epochs = self.config.get('epochs', 10)
        
        # Hardware resources
        self.hardware_manager = hardware_manager
        if self.hardware_manager is None and HARDWARE_MANAGER_AVAILABLE:
            self.hardware_manager = HardwareManager()
        self.use_classical = use_classical or not QUANTUM_AVAILABLE
        
        # Quantum components
        self.shots = self.config.get('shots', None)
        
        # Internal state
        self.is_initialized = False
        self.weights = None
        self.biases = None
        self.device = None
        self.circuits = {}
        
        # Optimization history
        self.optimization_history = []
        self.max_history = self.config.get('max_history', 50)
        
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
            self._initialize_optimizer()
        except Exception as e:
            self.logger.error(f"Error initializing NQO: {str(e)}", exc_info=True)
            self.use_classical = True
            self._initialize_optimizer()
    
    def _initialize_optimizer(self) -> None:
        """Initialize neuromorphic quantum optimizer."""
        self.logger.info(f"Initializing NQO with {'classical' if self.use_classical else 'quantum'} backend")
        
        # Initialize neural network weights and biases
        self._initialize_neural_network()
        
        # Initialize quantum components if needed
        if not self.use_classical:
            self.device = self._get_optimized_device()
            self._initialize_quantum_circuits()
        
        self.is_initialized = True
        self.logger.info(f"NQO initialized with {self.neurons} neurons and {self.qubits} qubits")
    
    def _initialize_neural_network(self) -> None:
        """Initialize neural network weights and biases."""
        # Simple feedforward structure for surrogate model
        self.weights = {
            'input_hidden': np.random.randn(10, self.neurons) * 0.1,  # Input features to hidden layer
            'hidden_output': np.random.randn(self.neurons, 5) * 0.1,  # Hidden to output
            'biases_hidden': np.zeros(self.neurons),
            'biases_output': np.zeros(5)
        }
        
        # Add recurrent connections for temporal patterns
        self.weights['recurrent'] = np.random.randn(self.neurons, self.neurons) * 0.01
        
        # Initialize hidden state
        self.hidden_state = np.zeros(self.neurons)
    
    def _get_optimized_device(self) -> Any:
        """Get hardware-optimized quantum device based on available hardware."""
        self.logger.debug("NQO: Attempting to get optimized quantum device.")
        if not QUANTUM_AVAILABLE:
            self.logger.warning("NQO: Quantum libraries not available, falling back to classical")
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
                self.logger.info(f"NQO: Using hardware manager's quantum device: {device_name}")
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
                    self.logger.info(f"NQO: Using AMD GPU for quantum operations")
                    return qml.device('lightning.kokkos', wires=self.qubits, shots=self.shots)

                # NVIDIA GPU
                elif self.hardware_manager.devices.get('nvidia_gpu', {}).get('available', False):
                    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
                    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
                    self.logger.info("NQO: Using NVIDIA GPU for quantum operations")
                    return qml.device('lightning.gpu', wires=self.qubits, shots=self.shots)

                # Apple Silicon
                elif self.hardware_manager.devices.get('apple_silicon', False):
                    self.logger.info("NQO: Using Apple Silicon for quantum operations")
                    return qml.device('default.qubit', wires=self.qubits, shots=self.shots)
            
            # Fallback to CPU if hardware detection fails or not available
            self.logger.info("NQO: Using CPU for quantum operations")
            try:
                return qml.device('lightning.qubit', wires=self.qubits, shots=self.shots)
            except Exception as e:
                self.logger.warning(f"NQO: Could not initialize lightning.qubit: {e}, falling back to default.qubit")
                return qml.device('default.qubit', wires=self.qubits, shots=self.shots)

        except Exception as e:
            self.logger.error(f"NQO: Error setting up quantum device: {e}", exc_info=True)
            self.logger.info("NQO: Falling back to default qubit device")
            try:
                return qml.device('default.qubit', wires=self.qubits, shots=self.shots)
            except Exception as e2:
                self.logger.error(f"NQO: Critical error setting up default device: {e2}", exc_info=True)
                self.use_classical = True
                return None
    
    def _initialize_quantum_circuits(self) -> None:
        """Initialize quantum circuits for NQO."""
        self.logger.debug("NQO: Initializing quantum circuits.")
        if not QUANTUM_AVAILABLE or self.use_classical or self.device is None:
            self.logger.warning("NQO: Quantum libraries not available or classical mode enabled, skipping circuit initialization")
            return
            
        try:
            # Parameter space exploration circuit
            @qml.qnode(self.device)
            def parameter_exploration_circuit(parameters):
                # Encode parameters into quantum state
                for i in range(self.qubits):
                    qml.RY(parameters[i % len(parameters)], wires=i)

                # Apply entangling layers
                for i in range(self.qubits - 1):
                    qml.CNOT(wires=[i, i+1])

                # Apply rotation gates
                for i in range(self.qubits):
                    qml.RX(parameters[(i + self.qubits) % len(parameters)], wires=i)
                    qml.RZ(parameters[(i + 2*self.qubits) % len(parameters)], wires=i)

                # Additional entanglement
                for i in range(self.qubits - 1, 0, -1):
                    qml.CNOT(wires=[i, i-1])

                # Measure expectation values
                return [qml.expval(qml.PauliZ(i)) for i in range(self.qubits)]

            # Optimization circuit
            @qml.qnode(self.device)
            def optimization_circuit(parameters, weights):
                # Encode parameters
                for i in range(self.qubits):
                    qml.RY(parameters[i % len(parameters)], wires=i)

                # Apply weighted transformation based on neural network weights
                weight_sum = np.sum(weights)
                for i in range(self.qubits):
                    qml.RZ(weight_sum * 0.1, wires=i)

                # Apply entangling layers
                for i in range(self.qubits - 1):
                    qml.CNOT(wires=[i, i+1])

                # Apply variational layers
                for i in range(self.qubits):
                    qml.RX(parameters[(i + self.qubits) % len(parameters)], wires=i)

                # Final layer
                for i in range(self.qubits):
                    qml.RY(parameters[(i + 2*self.qubits) % len(parameters)], wires=i)

                # Measure probabilities for different parameters
                return [qml.expval(qml.PauliZ(i)) for i in range(self.qubits)]

            # QAOA-inspired optimization circuit
            @qml.qnode(self.device)
            def qaoa_optimization_circuit(params, grads):
                # Prepare initial state
                for i in range(self.qubits):
                    qml.RY(np.pi * params[i % len(params)], wires=i)
                
                # First layer - cost Hamiltonian
                for i in range(self.qubits):
                    qml.RZ(np.pi * grads[i % len(grads)], wires=i)
                
                # Mixing Hamiltonian
                for i in range(self.qubits):
                    qml.RX(np.pi / 2, wires=i)
                
                # Second layer - cost Hamiltonian
                for i in range(self.qubits):
                    qml.RZ(np.pi * grads[i % len(grads)] / 2, wires=i)
                
                # Final mixing for measurement
                for i in range(self.qubits):
                    qml.Hadamard(wires=i)
                
                # Return expectation values
                return [qml.expval(qml.PauliZ(i)) for i in range(self.qubits)]

            # Parameter refinement circuit
            @qml.qnode(self.device)
            def parameter_refinement_circuit(parameters, optimal_direction):
                # Encode parameters
                for i in range(self.qubits):
                    qml.RY(parameters[i % len(parameters)], wires=i)

                # Apply directional bias
                for i in range(self.qubits):
                    qml.RZ(optimal_direction[i % len(optimal_direction)], wires=i)

                # Apply entangling layer
                for i in range(self.qubits - 1):
                    qml.CNOT(wires=[i, i+1])

                # Apply variational layer in optimal direction
                for i in range(self.qubits):
                    qml.RY(optimal_direction[i % len(optimal_direction)] * parameters[i % len(parameters)], wires=i)

                # Measure refined parameters
                return [qml.expval(qml.PauliZ(i)) for i in range(self.qubits)]

            # Store the circuits in the dictionary
            self.circuits = {
                'exploration': parameter_exploration_circuit,
                'optimization': optimization_circuit,
                'qaoa': qaoa_optimization_circuit,
                'refinement': parameter_refinement_circuit
            }
                
            self.logger.info(f"NQO: Quantum circuits initialized with {self.qubits} qubits")
            
        except Exception as e:
            self.logger.error(f"NQO: Error initializing quantum circuits: {str(e)}", exc_info=True)
            self.logger.warning("NQO: Falling back to classical implementation")
            self.use_classical = True
    
    def _execute_with_fallback(self, circuit_name: str, params: Tuple) -> np.ndarray:
        """Execute quantum circuit with fallback to classical computation."""
        self.logger.debug(f"NQO: Attempting to execute circuit '{circuit_name}'. Use classical: {self.use_classical}, Quantum available: {QUANTUM_AVAILABLE}, Circuit exists: {circuit_name in self.circuits}")
        if self.use_classical or not QUANTUM_AVAILABLE or circuit_name not in self.circuits:
            self.logger.debug(f"NQO: Executing classical fallback for circuit: {circuit_name}")
            return self._classical_fallback(circuit_name, params)
            
        try:
            # Generate cache key
            param_str = str([p.tobytes() if hasattr(p, 'tobytes') else str(p) for p in params])
            cache_key = hash(circuit_name + param_str)

            # Check cache
            with self.cache_lock:
                if cache_key in self.cache:
                    self.logger.debug(f"NQO: Cache hit for circuit '{circuit_name}'")
                    return self.cache[cache_key]

            # Execute quantum circuit
            self.logger.debug(f"NQO: Executing quantum circuit '{circuit_name}'")
            result = self.circuits[circuit_name](*params)

            # Cache result
            with self.cache_lock:
                self.cache[cache_key] = result
                self.logger.debug(f"NQO: Cached result for circuit '{circuit_name}'")

            return result

        except Exception as e:
            self.logger.warning(f"NQO: Quantum circuit execution failed for {circuit_name}: {e}. Falling back to classical computation.")
            # Fall back to classical computation
            return self._classical_fallback(circuit_name, params)
    
    def _classical_fallback(self, circuit_name: str, params: Tuple) -> np.ndarray:
        """Provide classical implementations of quantum circuits for fallback."""
        self.logger.debug(f"NQO: Executing classical fallback for circuit: {circuit_name}")
        
        if circuit_name == 'exploration':
            parameters = params[0]
            # Classical approximation of parameter exploration
            result = np.array([np.cos(p) for p in parameters[:self.qubits]])
            return result
            
        elif circuit_name == 'optimization':
            parameters, weights = params
            # Simple weighted sum for optimization
            result = np.array([np.tanh(parameters[i % len(parameters)] *
                           np.sum(weights) * 0.01) for i in range(self.qubits)])
            return result
            
        elif circuit_name == 'qaoa':
            params, grads = params
            # Classical approximation of QAOA circuit
            result = np.array([np.tanh(params[i % len(params)] * grads[i % len(grads)])
                           for i in range(self.qubits)])
            return result
            
        elif circuit_name == 'refinement':
            parameters, optimal_direction = params
            # Simple refinement approximation
            result = np.array([np.tanh(parameters[i % len(parameters)] *
                           optimal_direction[i % len(optimal_direction)])
                          for i in range(self.qubits)])
            return result
            
        else:
            self.logger.error(f"NQO: Unknown circuit: {circuit_name}")
            return np.zeros(self.qubits)
    
    def _neural_forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Forward pass through neural network.
        
        Args:
            inputs (np.ndarray): Input features
            
        Returns:
            np.ndarray: Network output
        """
        self.logger.debug("NQO: Executing neural forward pass.")
        # Ensure input is correctly shaped
        if len(inputs.shape) == 1:
            inputs = inputs.reshape(1, -1)

        # Make sure input has right dimension, pad or truncate as needed
        if inputs.shape[1] < 10:  # Pad
            inputs = np.pad(inputs, ((0, 0), (0, 10 - inputs.shape[1])))
        elif inputs.shape[1] > 10:  # Truncate
            inputs = inputs[:, :10]

        # Update hidden state with recurrent connections
        recurrent_input = np.dot(self.hidden_state, self.weights['recurrent'])

        # Hidden layer with ReLU activation
        hidden = np.dot(inputs, self.weights['input_hidden']) + recurrent_input + self.weights['biases_hidden']
        hidden_activated = np.maximum(0, hidden)  # ReLU
        
        # Store updated hidden state
        self.hidden_state = hidden_activated.flatten()

        # Output layer (no activation - for parameters)
        output = np.dot(hidden_activated, self.weights['hidden_output']) + self.weights['biases_output']

        return output
    
    def _neural_update(self, inputs: np.ndarray, gradients: np.ndarray) -> None:
        """
        Update neural network based on gradients.
        
        Args:
            inputs (np.ndarray): Input features
            gradients (np.ndarray): Gradients for network outputs
        """
        self.logger.debug("NQO: Executing neural update.")
        # Ensure input is correctly shaped
        if len(inputs.shape) == 1:
            inputs = inputs.reshape(1, -1)

        # Make sure input has right dimension
        if inputs.shape[1] < 10:  # Pad
            inputs = np.pad(inputs, ((0, 0), (0, 10 - inputs.shape[1])))
        elif inputs.shape[1] > 10:  # Truncate
            inputs = inputs[:, :10]

        # Forward pass (stored for backward)
        recurrent_input = np.dot(self.hidden_state, self.weights['recurrent'])
        hidden = np.dot(inputs, self.weights['input_hidden']) + recurrent_input + self.weights['biases_hidden']
        hidden_activated = np.maximum(0, hidden)  # ReLU

        # Backward pass (simplified)
        # Gradient for output layer
        d_output = gradients
        d_hidden = np.dot(d_output, self.weights['hidden_output'].T)
        d_hidden_activated = d_hidden * (hidden > 0)  # ReLU derivative

        # Update weights with learning rate and adaptive momentum
        momentum = self.adaptivity  # Use adaptivity parameter as momentum
        lr = self.learning_rate
        
        # Output layer updates
        output_update = lr * np.dot(hidden_activated.T, d_output)
        self.weights['hidden_output'] -= output_update
        self.weights['biases_output'] -= lr * np.sum(d_output, axis=0)
        
        # Hidden layer updates
        hidden_update = lr * np.dot(inputs.T, d_hidden_activated)
        self.weights['input_hidden'] -= hidden_update
        self.weights['biases_hidden'] -= lr * np.sum(d_hidden_activated, axis=0)
        
        # Recurrent layer updates (with momentum)
        recurrent_update = lr * np.dot(self.hidden_state.reshape(-1, 1), d_hidden_activated)
        self.weights['recurrent'] -= momentum * recurrent_update
    
    def _neuromorphic_activation(self, x: np.ndarray) -> np.ndarray:
        """
        Apply neuromorphic activation function with adaptive response.
        
        Args:
            x (np.ndarray): Input values
            
        Returns:
            np.ndarray: Activated values
        """
        self.logger.debug("NQO: Applying neuromorphic activation.")
        # Adaptive sigmoid activation with temperature parameter
        temp = max(0.1, 1.0 - 0.5 * self.adaptivity)  # Lower temperature = sharper response
        return 1 / (1 + np.exp(-x / temp))
    
    def _neuromorphic_processing(self, gradient: np.ndarray) -> np.ndarray:
        """
        Apply neuromorphic processing to gradient.
        
        Args:
            gradient (np.ndarray): Raw gradient
            
        Returns:
            np.ndarray: Processed gradient
        """
        self.logger.debug("NQO: Applying neuromorphic processing to gradient.")
        # Limit gradient elements to process
        num_elements = min(len(gradient), self.neurons)
        
        # Select elements with largest magnitude
        largest_indices = np.argsort(np.abs(gradient))[-num_elements:]
        selected_gradient = np.zeros_like(gradient)
        selected_gradient[largest_indices] = gradient[largest_indices]
        
        # Apply neural network processing
        processed = np.tanh(np.dot(selected_gradient, self.weights['input_hidden'][:num_elements, :num_elements]) + self.weights['biases_hidden'][:num_elements])
        
        # Apply adaptivity
        processed = self.adaptivity * processed + (1 - self.adaptivity) * selected_gradient
        
        return processed
    
    def _quantum_optimization_step(self, parameters: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        """
        Apply quantum optimization step to parameters using QAOA-inspired approach.
        
        Args:
            parameters (np.ndarray): Current parameters
            gradient (np.ndarray): Parameter gradient
            
        Returns:
            np.ndarray: Updated parameters
        """
        self.logger.debug("NQO: Performing quantum optimization step.")
        if self.use_classical or not QUANTUM_AVAILABLE:
            self.logger.debug("NQO: Using classical fallback for quantum optimization step.")
            # Classical fallback
            return parameters - self.learning_rate * gradient
            
        try:
            # Normalize parameters and gradient
            params_norm = parameters / np.max(np.abs(parameters)) if np.max(np.abs(parameters)) > 0 else parameters
            grad_norm = gradient / np.max(np.abs(gradient)) if np.max(np.abs(gradient)) > 0 else gradient
            
            # Select subset of parameters to optimize (limit to number of qubits)
            largest_indices = np.argsort(np.abs(grad_norm))[-self.qubits:]
            selected_params = params_norm[largest_indices]
            selected_grads = grad_norm[largest_indices]
            
            # Execute QAOA-inspired circuit
            expectation_values = self._execute_with_fallback('qaoa', (selected_params, selected_grads))
            
            # Convert expectation values to parameter updates
            updates = np.array([(1 - e) / 2 for e in expectation_values])
            
            # Apply updates to selected parameters
            for i, idx in enumerate(largest_indices):
                parameters[idx] -= self.learning_rate * updates[i] * gradient[idx]
            
            return parameters
            
        except Exception as e:
            self.logger.error(f"NQO: Quantum optimization step error: {str(e)}", exc_info=True)
            # Fall back to classical optimization
            return parameters - self.learning_rate * gradient
    
    def _calculate_gradient(self, objective: Callable, params: np.ndarray, **kwargs) -> np.ndarray:
        """
        Calculate gradient of objective function using numerical approximation.
        
        Args:
            objective (Callable): Objective function
            params (np.ndarray): Current parameters
            **kwargs: Additional arguments for objective function
            
        Returns:
            np.ndarray: Gradient
        """
        self.logger.debug("NQO: Calculating gradient.")
        epsilon = 1e-5
        gradient = np.zeros_like(params)
        
        # Base function value
        base_value = objective(params, **kwargs)
        
        # Calculate gradient for each parameter
        for i in range(len(params)):
            # Perturb parameter
            params_plus = params.copy()
            params_plus[i] += epsilon
            
            # Calculate perturbed value
            value_plus = objective(params_plus, **kwargs)
            
            # Calculate gradient using forward difference
            gradient[i] = (value_plus - base_value) / epsilon
        
        return gradient
    
    def _generate_parameter_range(self, parameter_space: Dict[str, Tuple[float, float]]) -> List[np.ndarray]:
        """
        Generate parameter sets covering the parameter space.
        
        Args:
            parameter_space: Dictionary mapping parameter names to (min, max) ranges
            
        Returns:
            List[np.ndarray]: List of parameter sets
        """
        self.logger.debug("NQO: Generating parameter range.")
        # Extract parameter names and ranges
        param_names = list(parameter_space.keys())
        param_ranges = list(parameter_space.values())

        # Number of test points per dimension
        points_per_dim = min(5, max(2, int(100 / len(param_names))))

        # Generate grid of parameters
        param_sets = []

        # Generate initial set with endpoints and midpoints
        for i in range(len(param_names)):
            low, high = param_ranges[i]
            if i == 0:
                # First parameter - create initial sets
                for val in np.linspace(low, high, points_per_dim):
                    param_sets.append({param_names[i]: val})
            else:
                # Add this parameter to existing sets
                new_param_sets = []
                for param_set in param_sets:
                    for val in np.linspace(low, high, points_per_dim):
                        new_set = param_set.copy()
                        new_set[param_names[i]] = val
                        new_param_sets.append(new_set)
                param_sets = new_param_sets

        # Convert to numpy arrays for quantum processing
        parameter_arrays = []
        for param_set in param_sets:
            # Create array with parameter values in consistent order
            param_array = np.array([param_set[name] for name in param_names])
            parameter_arrays.append(param_array)

        return parameter_arrays
    
    def _quantum_enhance_parameters(self, best_params: np.ndarray, objective_values: List[float],
                                  parameter_sets: List[np.ndarray]) -> np.ndarray:
        """
        Use quantum circuits to enhance parameters.
        
        Args:
            best_params (np.ndarray): Best parameters found so far
            objective_values (List[float]): Objective values for parameter sets
            parameter_sets (List[np.ndarray]): List of parameter sets
            
        Returns:
            np.ndarray: Enhanced parameters
        """
        self.logger.debug("NQO: Attempting quantum parameter enhancement.")
        try:
            # Create quantum-enhanced direction
            # First, execute exploration circuit
            quantum_result = self._execute_with_fallback('exploration', (best_params,))

            # Determine optimal direction from quantum result
            optimal_direction = quantum_result * 2 - 1  # Convert to -1 to 1 range

            # Get top 3 parameter sets
            top_indices = np.argsort(objective_values)[-3:]
            top_params = [parameter_sets[i] for i in top_indices]

            # Combine top parameters into a refinement direction
            refinement_direction = np.mean(top_params, axis=0) - best_params
            refinement_direction = refinement_direction / (np.linalg.norm(refinement_direction) + 1e-10)

            # Combine quantum and classical information
            combined_direction = optimal_direction + self.adaptivity * refinement_direction
            combined_direction = combined_direction / (np.linalg.norm(combined_direction) + 1e-10)

            # Apply refinement circuit
            refined_params = self._execute_with_fallback(
                'refinement', (best_params, combined_direction)
            )

            # Map back to parameter space (using best params as reference)
            param_scale = np.abs(best_params) + 0.1
            enhanced_params = best_params + refined_params * param_scale * 0.2

            return enhanced_params

        except Exception as e:
            self.logger.error(f"NQO: Error in quantum parameter enhancement: {e}")
            return best_params
    
    def _store_optimization_result(self, result: Dict[str, Any]) -> None:
        """
        Store optimization result in history.
        
        Args:
            result (Dict[str, Any]): Optimization result
        """
        try:
            # Add timestamp
            result_record = result.copy()
            result_record['timestamp'] = time.time()

            # Add to history
            self.optimization_history.append(result_record)

            # Trim history if too long
            if len(self.optimization_history) > self.max_history:
                self.optimization_history = self.optimization_history[-self.max_history:]

        except Exception as e:
            self.logger.error(f"NQO: Error storing optimization result: {e}")
    
    def optimize_parameters(self,
                          objective: Callable,
                          initial_params: np.ndarray,
                          iterations: int = 10,
                          **kwargs) -> Dict[str, Any]:
        """
        Optimize parameters for an objective function.
        
        Args:
            objective (Callable): Objective function to minimize
            initial_params (np.ndarray): Initial parameters
            iterations (int): Number of optimization iterations
            **kwargs: Additional arguments for objective function
            
        Returns:
            Dict[str, Any]: Optimization results
        """
        start_time = time.time()
        self.logger.info(f"NQO: Starting parameter optimization with {iterations} iterations.")
        
        try:
            if not self.is_initialized:
                self._initialize_optimizer()
            
            # Initialize parameters
            params = initial_params.copy()
            best_params = params.copy()
            best_value = objective(params, **kwargs)
            initial_value = best_value  # Store for improvement calculation
            history = [best_value]
            
            # Iterative optimization
            for i in range(iterations):
                # Evaluate objective function
                value = objective(params, **kwargs)
                
                # Update best parameters if needed
                if value < best_value:
                    best_value = value
                    best_params = params.copy()
                
                # Calculate gradient (numerical approximation)
                gradient = self._calculate_gradient(objective, params, **kwargs)
                
                # Apply neuromorphic processing
                processed_gradient = self._neuromorphic_processing(gradient)
                
                # Apply quantum optimization step
                params = self._quantum_optimization_step(params, processed_gradient)
                
                # Feed information to neural network for learning
                if i > 0:
                    input_features = np.concatenate([
                        params,
                        [value],
                        [np.mean(history)],
                        [np.std(history)]
                    ])
                    
                    # Get neural network prediction
                    nn_output = self._neural_forward(input_features)
                    
                    # Use actual outcome to update network
                    target_output = nn_output.copy()
                    
                    # Check if we have at least 2 history entries before comparing
                    if len(history) >= 2 and history[-1] < history[-2]:
                        # Safely update target_output considering dimension constraints
                        param_diff = params - best_params
                        update_length = min(len(target_output), len(param_diff))
                        
                        for j in range(update_length):
                            target_output[j] = param_diff[j]
                    
                    gradient = target_output - nn_output
                    self._neural_update(input_features, gradient.reshape(1, -1))
            
            # Try quantum enhancement as final step
            if iterations > 0 and len(history) > 1:
                parameter_sets = [params, best_params]  # Use current and best params
                if len(parameter_sets) >= 2:
                    enhanced_params = self._quantum_enhance_parameters(
                        best_params, history, parameter_sets
                    )
                    
                    # Evaluate enhanced parameters
                    enhanced_value = objective(enhanced_params, **kwargs)
                    
                    # Take the best between enhanced and original
                    if enhanced_value > best_value:
                        best_params = enhanced_params
                        best_value = enhanced_value
                        history.append(enhanced_value)
            
            # Calculate confidence
            improvement = (initial_value - best_value) / (abs(initial_value) + 1e-10)
            confidence = min(0.95, max(0.2, improvement))
            
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
            
            # Prepare result dictionary
            result = {
                'params': best_params,
                'value': float(best_value),
                'initial_value': float(initial_value),
                'history': history,
                'iterations': iterations,
                'confidence': float(confidence),
                'execution_time_ms': execution_time
            }
            
            # Store in history
            self._store_optimization_result(result)
            
            self.logger.info(f"NQO: Optimization completed in {execution_time:.2f}ms. Final value: {best_value:.6f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"NQO: Error in parameter optimization: {e}", exc_info=True)
            execution_time = (time.time() - start_time) * 1000
            
            # Return fallback result
            return {
                'params': initial_params,
                'value': objective(initial_params, **kwargs) if callable(objective) else None,
                'error': str(e),
                'confidence': 0.1,
                'execution_time_ms': execution_time
            }
    
    def optimize_trading_parameters(self, potential_matches: Dict[str, float], 
                                   order_book_data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Optimize trading parameters based on pattern matches and order book.
        
        Args:
            potential_matches (Dict[str, float]): Pattern match confidences
            order_book_data (Dict, optional): Order book data
            
        Returns:
            Dict[str, Any]: Optimized trading parameters
        """
        self.logger.info("NQO: Optimizing trading parameters.")
        # Define objective function for trading parameter optimization
        def trading_objective(params, **kwargs):
            # Extract parameters
            entry_threshold = params[0]
            stop_loss = params[1]
            take_profit = params[2]
            
            # Calculate expected value based on probability and risk/reward
            probability = min(max(sum(kwargs.get('matches', {}).values()) / len(kwargs.get('matches', {})), 0), 1)
            risk = stop_loss
            reward = take_profit
            
            # Expected value
            expected_value = probability * reward - (1 - probability) * risk
            
            # Penalize extreme values
            penalty = 0
            if entry_threshold < 0.3 or entry_threshold > 0.9:
                penalty += 10
            if stop_loss < 0.01 or stop_loss > 0.1:
                penalty += 10
            if take_profit < stop_loss * 1.5:
                penalty += 10
            
            # Return negative expected value (minimize)
            return -expected_value + penalty
        
        # Initial parameters: [entry_threshold, stop_loss, take_profit]
        initial_params = np.array([0.6, 0.03, 0.06])
        
        # Optimize
        result = self.optimize_parameters(
            objective=trading_objective,
            initial_params=initial_params,
            iterations=5,
            matches=potential_matches,
            order_book=order_book_data
        )
        
        # Extract optimized parameters
        optimized_params = result.get('params', initial_params)
        
        return {
            'entry_threshold': float(optimized_params[0]),
            'stop_loss': float(optimized_params[1]),
            'take_profit': float(optimized_params[2]),
            'confidence': float(1 - result.get('value', 0) if result.get('value', 0) <= 0 else 0.5)
        }
    
    def optimize_allocation(self, pair: str, edge: float, win_rate: float, 
                           market_data: Dict) -> Dict[str, Any]:
        """
        Optimize portfolio allocation based on edge and market conditions.
        
        Args:
            pair (str): Trading pair
            edge (float): Edge (expected return)
            win_rate (float): Win rate
            market_data (Dict): Market data
            
        Returns:
            Dict[str, Any]: Optimized allocation
        """
        self.logger.info(f"NQO: Optimizing allocation for {pair}.")
        # Define objective function for allocation optimization
        def allocation_objective(params, **kwargs):
            # Extract parameters
            allocation = params[0]  # Position size as percentage
            
            # Extract kwargs
            win_rate = kwargs.get('win_rate', 0.5)
            edge = kwargs.get('edge', 0)
            
            # Calculate Kelly criterion
            kelly = win_rate - (1 - win_rate) / (edge / 0.03)  # Normalize edge
            optimal_allocation = max(0, kelly / 2)  # Half-Kelly for safety
            
            # Penalty for deviating from optimal
            deviation_penalty = (allocation - optimal_allocation) ** 2
            
            # Risk penalty
            volatility = kwargs.get('market_data', {}).get('volatility', kwargs.get('market_data', {}).get('volatility_regime', 0.5))
            risk_penalty = allocation * volatility
            
            # Return combined penalty (minimize)
            return deviation_penalty + risk_penalty
        
        # Initial parameters: [allocation]
        initial_params = np.array([0.05])  # 5% allocation
        
        # Optimize
        result = self.optimize_parameters(
            objective=allocation_objective,
            initial_params=initial_params,
            iterations=5,
            win_rate=win_rate,
            edge=edge,
            market_data=market_data
        )
        
        # Extract optimized parameters
        optimized_params = result.get('params', initial_params)
        
        return {
            'allocation': float(optimized_params[0]),
            'confidence': float(1 - result.get('value', 0) if result.get('value', 0) <= 0 else 0.5)
        }
    
    def optimize_parameters_with_constraints(self,
                                           parameter_space: Dict[str, Tuple[float, float]],
                                           objective_function: Callable,
                                           constraint_functions: List[Callable] = None) -> Dict[str, Any]:
        """
        Optimize parameters using neuromorphic quantum approach with constraints.

        Args:
            parameter_space: Dictionary mapping parameter names to (min, max) ranges
            objective_function: Function to maximize, takes parameter dictionary as input
            constraint_functions: List of constraint functions, each returns True if constraint is satisfied

        Returns:
            Dict containing optimized parameters and metadata
        """
        start_time = time.time()
        self.logger.info("NQO: Starting constrained parameter optimization.")

        try:
            # Extract parameter names
            param_names = list(parameter_space.keys())

            # Generate initial parameter sets
            parameter_sets = self._generate_parameter_range(parameter_space)

            # Apply constraints if provided
            if constraint_functions:
                filtered_parameter_sets = []
                for params in parameter_sets:
                    param_dict = {name: params[i] for i, name in enumerate(param_names)}
                    if all(constraint(param_dict) for constraint in constraint_functions):
                        filtered_parameter_sets.append(params)
                
                # If all parameter sets violate constraints, relax constraints
                if not filtered_parameter_sets:
                    self.logger.warning("NQO: All parameter sets violate constraints, using original sets")
                    filtered_parameter_sets = parameter_sets
                
                parameter_sets = filtered_parameter_sets

            # Evaluate objective function for each parameter set
            objective_values = []
            for params in parameter_sets:
                param_dict = {name: params[i] for i, name in enumerate(param_names)}
                try:
                    value = objective_function(param_dict)
                    objective_values.append(value)
                except Exception as e:
                    self.logger.warning(f"NQO: Error evaluating objective function: {e}")
                    objective_values.append(float('-inf'))  # Worst possible value

            # Find best parameter set
            if objective_values:
                best_idx = np.argmax(objective_values)
                best_params = parameter_sets[best_idx]
                best_value = objective_values[best_idx]
                
                # Convert to dictionary
                best_param_dict = {name: best_params[i] for i, name in enumerate(param_names)}
                
                # Apply quantum enhancement
                enhanced_params = self._quantum_enhance_parameters(
                    best_params, objective_values, parameter_sets
                )
                
                # Convert to dictionary
                enhanced_param_dict = {name: enhanced_params[i] for i, name in enumerate(param_names)}
                
                # Evaluate enhanced parameters
                try:
                    enhanced_value = objective_function(enhanced_param_dict)
                except Exception:
                    enhanced_value = float('-inf')
                
                # Take the best between enhanced and original
                if enhanced_value > best_value:
                    final_params = enhanced_param_dict
                    final_value = enhanced_value
                else:
                    final_params = best_param_dict
                    final_value = best_value
                
                # Calculate confidence
                mean_value = np.mean(objective_values) if objective_values else 0
                std_value = np.std(objective_values) if objective_values else 1
                confidence = min(0.95, max(0.1, (final_value - mean_value) / (std_value + 1e-10)))
                
                result = {
                    'optimized_parameters': final_params,
                    'objective_value': float(final_value),
                    'confidence': float(confidence),
                    'execution_time_ms': (time.time() - start_time) * 1000
                }
                
                # Store in history
                self._store_optimization_result(result)
                
                self.logger.info(f"NQO: Constrained optimization completed in {result['execution_time_ms']:.2f}ms. Final value: {final_value:.6f}")
                
                return result
            else:
                self.logger.warning("NQO: No valid parameter sets found after constraint filtering.")
                raise ValueError("No valid parameter sets found")

        except Exception as e:
            self.logger.error(f"NQO: Error in constrained parameter optimization: {e}", exc_info=True)
            execution_time = (time.time() - start_time) * 1000
            
            # Default midpoint of parameter ranges
            default_params = {name: (min_val + max_val) / 2 for name, (min_val, max_val) in parameter_space.items()}
            
            return {
                'optimized_parameters': default_params,
                'objective_value': 0.0,
                'confidence': 0.1,
                'execution_time_ms': execution_time,
                'error': str(e)
            }
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Calculate performance metrics for past optimizations.
        
        Returns:
            Dict[str, float]: Performance metrics
        """
        try:
            if not self.optimization_history:
                return {'mean_improvement': 0.0, 'success_rate': 0.0, 'sample_size': 0}

            # Calculate improvement from initial to final value
            improvements = []
            successes = 0

            for result in self.optimization_history:
                if 'initial_value' in result and 'value' in result:
                    improvement = (result['initial_value'] - result['value']) / (abs(result['initial_value']) + 1e-10)
                    improvements.append(improvement)

                    if result['value'] < result['initial_value']:
                        successes += 1

            # Calculate metrics
            mean_improvement = np.mean(improvements) if improvements else 0.0
            success_rate = successes / len(self.optimization_history) if self.optimization_history else 0.0

            return {
                'mean_improvement': float(mean_improvement),
                'success_rate': float(success_rate),
                'sample_size': len(self.optimization_history)
            }

        except Exception as e:
            self.logger.error(f"NQO: Error calculating performance metrics: {e}")
            return {'mean_improvement': 0.0, 'success_rate': 0.0, 'sample_size': 0, 'error': str(e)}
    
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
    
    def clear_cache(self) -> None:
        """Clear the internal cache."""
        with self.cache_lock:
            self.cache.clear()
        self.logger.debug("NQO cache cleared")
    
    def reset(self) -> None:
        """Reset the optimizer state."""
        self.logger.info("NQO: Resetting NQO")
        # Reset neural network
        self._initialize_neural_network()
        # Clear history
        self.optimization_history = []
        # Clear cache
        self.clear_cache()
    
    def recover(self) -> bool:
        """
        Attempt to recover the NQO component after errors.
        
        Returns:
            bool: True if recovery succeeded
        """
        self.logger.warning("NQO: Recovery triggered!")
        try:
            # Reset state
            self.reset()
            
            # Re-initialize quantum device
            if not self.use_classical and QUANTUM_AVAILABLE:
                self.device = self._get_optimized_device()
                self._initialize_quantum_circuits()
                
            # Check for fault tolerance manager
            if self.fault_tolerance:
                self.fault_tolerance.register_recovery("nqo")
                
            self.logger.info("NQO: Recovery succeeded")
            return True
            
        except Exception as e:
            self.logger.error(f"NQO: Recovery failed: {str(e)}", exc_info=True)
            # Last resort - force classical mode
            self.use_classical = True
            return False


# Factory function for thread-safe singleton access
_nqo_instance = None
_nqo_lock = threading.RLock()

def get_neuromorphic_quantum_optimizer(config=None, reset=False) -> NeuromorphicQuantumOptimizer:
    """Thread-safe factory function for NeuromorphicQuantumOptimizer."""
    global _nqo_instance, _nqo_lock

    with _nqo_lock:
        if _nqo_instance is None or reset:
            try:
                # Extract specific parameters from config
                config_dict = config or {}
                
                # Instantiate with parameters from config
                _nqo_instance = NeuromorphicQuantumOptimizer(
                    neurons=config_dict.get('neurons', 128),
                    qubits=config_dict.get('qubits', 4),
                    adaptivity=config_dict.get('adaptivity', 0.7),
                    learning_rate=config_dict.get('learning_rate', 0.01),
                    config=config_dict
                )
                
                logger.info("NQO: Created new NQO instance")
                
            except Exception as e:
                logger.exception(f"NQO: Failed to initialize NQO instance: {e}")
                return None

    return _nqo_instance


# Example usage (if module run directly)
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    
    # Create NQO instance
    nqo = NeuromorphicQuantumOptimizer(
        neurons=64,  # Smaller for testing
        qubits=4,
        adaptivity=0.7,
        learning_rate=0.01
    )
    
    # Test simple optimization - find minimum of a parabola
    def parabola(params, **kwargs):
        x = params[0]
        y = params[1]
        return (x - 3)**2 + (y - 4)**2
    
    # Optimize
    initial_params = np.array([0.0, 0.0])
    result = nqo.optimize_parameters(parabola, initial_params, iterations=10)
    
    # Print results
    print("\nOptimization Results:")
    print(f"  Initial parameters: {initial_params}")
    print(f"  Optimized parameters: {result['params']}")
    print(f"  Final value: {result['value']}")
    print(f"  Expected minimum: x=3, y=4, value=0")
    print(f"  Confidence: {result['confidence']}")
    print(f"  Execution time: {result['execution_time_ms']:.2f} ms")
    
    # Test trading parameter optimization
    potential_matches = {'pattern1': 0.8, 'pattern2': 0.6, 'pattern3': 0.7}
    trading_result = nqo.optimize_trading_parameters(potential_matches)
    
    print("\nTrading Parameter Optimization:")
    for key, value in trading_result.items():
        print(f"  {key}: {value}")
    
    # Show execution stats
    stats = nqo.get_execution_stats()
    print("\nExecution Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
