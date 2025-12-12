"""
GPU-accelerated quantum simulation backend management.
"""

import numpy as np
import pennylane as qml
from typing import Dict, Any, Optional, List, Tuple, Union
import logging
import os
from dataclasses import dataclass
import threading

logger = logging.getLogger(__name__)

@dataclass
class SimulatorConfig:
    """Configuration for quantum simulator."""
    backend: str  # 'lightning.gpu', 'lightning.kokkos', 'lightning.qubit'
    num_wires: int
    shots: Optional[int]
    batch_size: int = 1
    gpu_index: int = 0
    precision: str = 'float32'  # 'float32' or 'float64'

class SimulatorBackend:
    """
    Unified interface for GPU-accelerated quantum simulation.
    """

    def __init__(self, hw_optimizer: Any):
        """
        Initialize simulator backend.

        Args:
            hw_optimizer: Hardware optimizer instance
        """
        self.hw_optimizer = hw_optimizer
        self.active_devices = {}
        self.device_lock = threading.Lock()
        self.performance_stats = {}

        # Detect available backends
        self.available_backends = self._detect_backends()
        self.default_backend = self._select_default_backend()

        logger.info(f"Available backends: {self.available_backends}")
        logger.info(f"Default backend: {self.default_backend}")

    def _detect_backends(self) -> List[str]:
        """Detect available PennyLane backends."""
        backends = ['default.qubit', 'lightning.qubit']  # Always available

        # Check for GPU backends
        try:
            # Test lightning.gpu (NVIDIA)
            test_dev = qml.device('lightning.gpu', wires=2)
            backends.append('lightning.gpu')
            logger.info("lightning.gpu backend available")
        except Exception as e:
            logger.debug(f"lightning.gpu not available: {e}")

        try:
            # Test lightning.kokkos (AMD/Multi-backend)
            test_dev = qml.device('lightning.kokkos', wires=2)
            backends.append('lightning.kokkos')
            logger.info("lightning.kokkos backend available")
        except Exception as e:
            logger.debug(f"lightning.kokkos not available: {e}")

        return backends

    def _select_default_backend(self) -> str:
        """Select optimal backend based on hardware."""
        # Priority order based on performance
        if 'lightning.gpu' in self.available_backends and self.hw_optimizer._device_type == 'cuda':
            return 'lightning.gpu'
        elif 'lightning.kokkos' in self.available_backends:
            return 'lightning.kokkos'
        else:
            return 'lightning.qubit'

    def create_device(self, config: SimulatorConfig) -> Any:
        """
        Create quantum device with specified configuration.

        Args:
            config: Simulator configuration

        Returns:
            PennyLane device
        """
        device_key = f"{config.backend}_{config.num_wires}_{config.shots}_{config.gpu_index}"

        with self.device_lock:
            # Return cached device if exists
            if device_key in self.active_devices:
                logger.debug(f"Returning cached device: {device_key}")
                return self.active_devices[device_key]

            # Create new device
            device_kwargs = {
                'wires': config.num_wires
            }

            # Add backend-specific configurations
            if config.backend == 'lightning.gpu':
                # Set CUDA device
                os.environ['CUDA_VISIBLE_DEVICES'] = str(config.gpu_index)

                # Precision setting
                if config.precision == 'float64':
                    device_kwargs['c_dtype'] = np.complex128
                else:
                    device_kwargs['c_dtype'] = np.complex64

                # Batch execution
                if config.batch_size > 1:
                    device_kwargs['batch_obs'] = True

            elif config.backend == 'lightning.kokkos':
                # Kokkos backend configuration
                if 'gpu' in self.hw_optimizer._device:
                    device_kwargs['kokkos_args'] = {
                        '--device-id': config.gpu_index,
                        '--execution-space': 'cuda' if 'cuda' in self.hw_optimizer._device else 'hip'
                    }

            # Create device
            device = qml.device(config.backend, **device_kwargs)

            # Cache device
            self.active_devices[device_key] = device

            logger.info(f"Created device: {config.backend} with {config.num_wires} wires")
            return device

    def optimize_circuit(self, circuit: qml.QNode, config: SimulatorConfig) -> qml.QNode:
        """
        Optimize quantum circuit for execution.

        Args:
            circuit: Quantum circuit (QNode)
            config: Simulator configuration

        Returns:
            Optimized circuit
        """
        # Apply backend-specific optimizations
        if config.backend == 'lightning.gpu':
            # GPU-specific optimizations
            circuit = qml.transforms.merge_rotations(circuit)
            circuit = qml.transforms.cancel_inverses(circuit)
            circuit = qml.transforms.commute_controlled(circuit)

        elif config.backend == 'lightning.kokkos':
            # Kokkos optimizations
            circuit = qml.transforms.merge_rotations(circuit)
            circuit = qml.transforms.single_qubit_fusion(circuit)

        # Common optimizations
        circuit = qml.transforms.remove_barrier(circuit)

        return circuit

    def execute_batch(self, circuits: List[qml.QNode],
                     config: SimulatorConfig,
                     parameters: List[np.ndarray]) -> List[Any]:
        """
        Execute batch of quantum circuits.

        Args:
            circuits: List of quantum circuits
            config: Simulator configuration
            parameters: Parameters for each circuit

        Returns:
            Execution results
        """
        device = self.create_device(config)
        results = []

        # Batch execution for GPU backends
        if config.backend in ['lightning.gpu', 'lightning.kokkos'] and len(circuits) > 1:
            # Create batched circuit
            batch_size = min(len(circuits), config.batch_size)

            for i in range(0, len(circuits), batch_size):
                batch_circuits = circuits[i:i+batch_size]
                batch_params = parameters[i:i+batch_size]

                # Execute batch
                batch_results = self._execute_batch_gpu(
                    batch_circuits, batch_params, device
                )
                results.extend(batch_results)
        else:
            # Sequential execution
            for circuit, params in zip(circuits, parameters):
                result = circuit(*params)
                results.append(result)

        return results

    def _execute_batch_gpu(self, circuits: List[qml.QNode],
                          parameters: List[np.ndarray],
                          device: Any) -> List[Any]:
        """Execute batch on GPU backend."""
        # Prepare batch data
        batch_results = []

        try:
            # Use device's batch execution if available
            if hasattr(device, 'batch_execute'):
                # Prepare tapes for batch execution
                tapes = []
                for circuit, params in zip(circuits, parameters):
                    tape = qml.tape.QuantumTape()
                    with tape:
                        circuit(*params)
                    tapes.append(tape)

                # Batch execute
                results = device.batch_execute(tapes)
                batch_results.extend(results)
            else:
                # Fallback to sequential
                for circuit, params in zip(circuits, parameters):
                    result = circuit(*params)
                    batch_results.append(result)

        except Exception as e:
            logger.error(f"Batch execution failed: {e}")
            # Fallback to sequential
            for circuit, params in zip(circuits, parameters):
                result = circuit(*params)
                batch_results.append(result)

        return batch_results

    def estimate_memory_usage(self, config: SimulatorConfig) -> int:
        """
        Estimate GPU memory usage for simulation.

        Args:
            config: Simulator configuration

        Returns:
            Estimated memory in MB
        """
        # State vector size
        state_size = 2 ** config.num_wires

        # Complex number size
        if config.precision == 'float64':
            bytes_per_complex = 16  # 8 bytes real + 8 bytes imag
        else:
            bytes_per_complex = 8   # 4 bytes real + 4 bytes imag

        # Base memory for state vector
        base_memory = state_size * bytes_per_complex

        # Additional memory for operations (roughly 2x for workspace)
        total_memory = base_memory * 2

        # Add overhead for batch execution
        if config.batch_size > 1:
            total_memory *= config.batch_size

        # Convert to MB
        return int(total_memory / (1024 * 1024))

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return {
            'active_devices': len(self.active_devices),
            'default_backend': self.default_backend,
            'available_backends': self.available_backends,
            'device_stats': self.performance_stats
        }

    def cleanup(self):
        """Clean up resources."""
        with self.device_lock:
            self.active_devices.clear()
        logger.info("Simulator backend cleaned up")
