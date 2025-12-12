"""Main QKS Plugin class - Entry point for all cognitive operations.

This module provides the unified interface to the Quantum Knowledge System.
"""

from typing import Optional, Dict, Any
import numpy as np
from dataclasses import dataclass

from qks._ffi import _lib, QksHandle, QksConfig, check_error
from qks.consciousness import Consciousness
from qks.metacognition import Metacognition
from qks.integration import Integration


@dataclass
class QKSConfig:
    """Configuration for QKS plugin.

    Attributes:
        num_qubits: Number of qubits for quantum simulation (default: 10)
        use_gpu: Enable GPU acceleration if available (default: True)
        enable_consciousness: Enable consciousness APIs (default: True)
        enable_metacognition: Enable metacognition APIs (default: True)
    """
    num_qubits: int = 10
    use_gpu: bool = True
    enable_consciousness: bool = True
    enable_metacognition: bool = True


class QKSPlugin:
    """Quantum Knowledge System - Drop-in Cognitive Super Pill.

    The main interface to QKS cognitive capabilities across all layers:
    - Layer 6: Consciousness (IIT, Global Workspace)
    - Layer 7: Metacognition (Introspection, Self-monitoring)
    - Layer 8: Integration (Full cognitive cycles)

    Example:
        >>> plugin = QKSPlugin(QKSConfig(num_qubits=20, use_gpu=True))
        >>> phi = plugin.consciousness.compute_phi(network_state)
        >>> if phi > 1.0:
        ...     print("System exhibits consciousness!")
        >>>
        >>> report = plugin.metacognition.introspect()
        >>> print(f"Confidence: {report.confidence:.2%}")
        >>>
        >>> output = plugin.integration.cognitive_cycle(sensory_input)
        >>> print(f"Action: {output.action}")

    Architecture:
        The plugin wraps a Rust core library that provides:
        - GPU-accelerated quantum simulation (Metal on macOS)
        - IIT 3.0 Φ computation (Tononi et al.)
        - Global Workspace Theory implementation
        - Real-time metacognitive monitoring
        - Full perception-cognition-action loops
    """

    def __init__(self, config: Optional[QKSConfig] = None):
        """Initialize QKS plugin.

        Args:
            config: Plugin configuration. Uses defaults if None.

        Raises:
            RuntimeError: If plugin initialization fails.
        """
        self._config = config or QKSConfig()

        # Convert to C struct
        c_config = QksConfig()
        c_config.num_qubits = self._config.num_qubits
        c_config.use_gpu = self._config.use_gpu
        c_config.enable_consciousness = self._config.enable_consciousness
        c_config.enable_metacognition = self._config.enable_metacognition

        # Create native handle
        self._handle = _lib.qks_create(ctypes.byref(c_config))
        if not self._handle:
            raise RuntimeError("Failed to create QKS plugin instance")

        # Initialize layer APIs
        self._consciousness = Consciousness(self._handle)
        self._metacognition = Metacognition(self._handle)
        self._integration = Integration(self._handle)

    def __del__(self):
        """Cleanup native resources."""
        if hasattr(self, '_handle') and self._handle:
            _lib.qks_destroy(self._handle)
            self._handle = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.__del__()
        return False

    @property
    def consciousness(self) -> Consciousness:
        """Access Layer 6: Consciousness APIs.

        Returns:
            Consciousness API instance for IIT and Global Workspace operations.
        """
        return self._consciousness

    @property
    def metacognition(self) -> Metacognition:
        """Access Layer 7: Metacognition APIs.

        Returns:
            Metacognition API instance for introspection and self-monitoring.
        """
        return self._metacognition

    @property
    def integration(self) -> Integration:
        """Access Layer 8: Integration APIs.

        Returns:
            Integration API instance for full cognitive cycles.
        """
        return self._integration

    @property
    def config(self) -> QKSConfig:
        """Get plugin configuration.

        Returns:
            Current plugin configuration.
        """
        return self._config

    @staticmethod
    def version() -> str:
        """Get QKS plugin version.

        Returns:
            Version string (e.g., "0.1.0")
        """
        return _lib.qks_version().decode('utf-8')

    def info(self) -> Dict[str, Any]:
        """Get plugin information and status.

        Returns:
            Dictionary with plugin metadata:
            - version: Plugin version string
            - config: Current configuration
            - capabilities: Available features
            - status: Runtime status
        """
        return {
            "version": self.version(),
            "config": {
                "num_qubits": self._config.num_qubits,
                "use_gpu": self._config.use_gpu,
                "enable_consciousness": self._config.enable_consciousness,
                "enable_metacognition": self._config.enable_metacognition,
            },
            "capabilities": {
                "consciousness": self._config.enable_consciousness,
                "metacognition": self._config.enable_metacognition,
                "quantum_simulation": True,
                "gpu_acceleration": self._config.use_gpu,
            },
            "status": "initialized",
        }


# Import ctypes for use in __init__
import ctypes
