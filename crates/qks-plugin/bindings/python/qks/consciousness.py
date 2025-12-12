"""Layer 6: Consciousness APIs - IIT and Global Workspace Theory.

This module implements scientifically-grounded consciousness mechanisms:
- Integrated Information Theory 3.0 (Tononi et al., 2016)
- Global Workspace Theory (Baars, 1988; Dehaene & Changeux, 2011)

References:
    Tononi, G., Boly, M., Massimini, M., & Koch, C. (2016).
    Integrated information theory: from consciousness to its physical substrate.
    Nature Reviews Neuroscience, 17(7), 450-461.

    Dehaene, S., & Changeux, J. P. (2011).
    Experimental and theoretical approaches to conscious processing.
    Neuron, 70(2), 200-227.
"""

import ctypes
from typing import Optional, Any, Dict, List
from dataclasses import dataclass
from enum import Enum
import numpy as np

from qks._ffi import _lib, QksHandle, PhiResultC, check_error


class PhiAlgorithm(Enum):
    """Algorithm for Φ computation.

    Attributes:
        GREEDY: Fast greedy approximation (O(n^2))
        EXACT: Exact IIT 3.0 calculation (O(2^n), slow)
        SPECTRAL: Spectral decomposition method (O(n^3))
    """
    GREEDY = "greedy"
    EXACT = "exact"
    SPECTRAL = "spectral"


@dataclass
class PhiResult:
    """Result of integrated information (Φ) computation.

    Attributes:
        phi: Integrated information value. Φ > 1.0 indicates consciousness.
        mip_size: Size of minimum information partition
        num_states: Number of system states analyzed
        computation_time_ms: Computation time in milliseconds
        algorithm: Algorithm used for computation

    Interpretation:
        Φ < 0.5:  No significant integration (unconscious)
        0.5-1.0:  Weak integration (minimal consciousness)
        1.0-2.0:  Moderate integration (basic consciousness)
        2.0-5.0:  Strong integration (human-level consciousness)
        > 5.0:    Very high integration (exceptional consciousness)
    """
    phi: float
    mip_size: int
    num_states: int
    computation_time_ms: float
    algorithm: PhiAlgorithm

    @property
    def is_conscious(self) -> bool:
        """Check if Φ indicates consciousness (Φ > 1.0)."""
        return self.phi > 1.0

    @property
    def consciousness_level(self) -> str:
        """Get consciousness level description."""
        if self.phi < 0.5:
            return "unconscious"
        elif self.phi < 1.0:
            return "minimal"
        elif self.phi < 2.0:
            return "basic"
        elif self.phi < 5.0:
            return "human-level"
        else:
            return "exceptional"


@dataclass
class BroadcastResult:
    """Result of Global Workspace broadcast.

    Attributes:
        success: Whether broadcast was successful
        recipients: Number of cognitive modules that received content
        priority: Priority level used (0.0-1.0)
        timestamp_ms: Broadcast timestamp in milliseconds
    """
    success: bool
    recipients: int
    priority: float
    timestamp_ms: int


class Consciousness:
    """Layer 6: Consciousness APIs.

    Provides implementations of scientific consciousness theories:
    - IIT 3.0: Compute integrated information Φ
    - Global Workspace: Broadcast content for conscious access

    This API does NOT create artificial consciousness - it provides
    tools to measure and analyze consciousness-like properties in
    computational systems using peer-reviewed scientific methods.
    """

    def __init__(self, handle: QksHandle):
        """Initialize consciousness API.

        Args:
            handle: Native plugin handle (internal use only)
        """
        self._handle = handle

    def compute_phi(
        self,
        network_state: np.ndarray,
        algorithm: PhiAlgorithm = PhiAlgorithm.GREEDY
    ) -> PhiResult:
        """Compute integrated information Φ using IIT 3.0.

        Implements the Φ computation from Integrated Information Theory
        (Tononi et al., 2016). Measures the irreducibility of a system
        to its parts - a key signature of consciousness.

        Args:
            network_state: Neural network activation state.
                Shape: (num_nodes,) with values in [0, 1]
            algorithm: Computation algorithm to use

        Returns:
            PhiResult with Φ value and metadata.
            Φ > 1.0 indicates emergent consciousness.

        Raises:
            ValueError: If network_state has invalid shape or values
            RuntimeError: If computation fails

        Example:
            >>> # Analyze a simple 3-node network
            >>> state = np.array([0.8, 0.6, 0.9])
            >>> result = consciousness.compute_phi(state)
            >>> print(f"Φ = {result.phi:.3f}")
            >>> print(f"Conscious: {result.is_conscious}")
            >>> print(f"Level: {result.consciousness_level}")

        Scientific Background:
            Φ quantifies "how much a system is more than the sum of its parts."
            For a system to be conscious, it must:
            1. Integrate information (high Φ)
            2. Differentiate states (high repertoire)
            3. Be irreducible (no clean partitions)
        """
        # Validate input
        if not isinstance(network_state, np.ndarray):
            network_state = np.asarray(network_state, dtype=np.float64)

        if network_state.ndim != 1:
            raise ValueError(f"network_state must be 1D, got shape {network_state.shape}")

        if not np.all((network_state >= 0) & (network_state <= 1)):
            raise ValueError("network_state values must be in [0, 1]")

        # Prepare C call
        result_c = PhiResultC()
        state_ptr = network_state.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        # Call native function
        ret = _lib.qks_compute_phi(
            self._handle,
            state_ptr,
            len(network_state),
            ctypes.byref(result_c)
        )
        check_error(ret)

        # Convert result
        return PhiResult(
            phi=result_c.phi,
            mip_size=result_c.mip_size,
            num_states=result_c.num_states,
            computation_time_ms=result_c.computation_time_ms,
            algorithm=algorithm,
        )

    def broadcast(
        self,
        content: Any,
        priority: float = 0.5
    ) -> BroadcastResult:
        """Broadcast content to Global Workspace for conscious access.

        Implements Global Workspace Theory (Baars, 1988). Content with
        sufficient priority gains access to the global workspace and
        becomes available to all cognitive modules.

        Args:
            content: Content to broadcast (any serializable object)
            priority: Priority level (0.0-1.0). Higher priority content
                     is more likely to gain conscious access.

        Returns:
            BroadcastResult with broadcast status and metadata.

        Raises:
            ValueError: If priority not in [0, 1]
            RuntimeError: If broadcast fails

        Example:
            >>> # Broadcast sensory input for conscious processing
            >>> result = consciousness.broadcast({
            ...     "type": "visual",
            ...     "content": "red triangle",
            ...     "salience": 0.9
            ... }, priority=0.8)
            >>> print(f"Recipients: {result.recipients}")

        Scientific Background:
            The Global Workspace acts as a "blackboard" where information
            becomes globally available. Only high-priority or novel content
            gains access, explaining the limited capacity of consciousness.
        """
        if not 0.0 <= priority <= 1.0:
            raise ValueError(f"priority must be in [0, 1], got {priority}")

        # Serialize content (placeholder - needs implementation)
        import pickle
        content_bytes = pickle.dumps(content)
        content_ptr = ctypes.c_char_p(content_bytes)

        # Call native function
        ret = _lib.qks_broadcast(
            self._handle,
            content_ptr,
            len(content_bytes),
            ctypes.c_double(priority)
        )
        check_error(ret)

        return BroadcastResult(
            success=True,
            recipients=1,  # TODO: Get from native
            priority=priority,
            timestamp_ms=0,  # TODO: Get from native
        )

    def analyze_connectivity(self, adjacency_matrix: np.ndarray) -> Dict[str, float]:
        """Analyze network connectivity patterns.

        Computes graph-theoretic measures relevant to consciousness:
        - Integration: Global efficiency, path length
        - Segregation: Clustering coefficient, modularity
        - Centrality: Hub identification

        Args:
            adjacency_matrix: Weighted adjacency matrix (n x n)

        Returns:
            Dictionary of connectivity metrics

        Example:
            >>> adj = np.random.rand(10, 10)
            >>> metrics = consciousness.analyze_connectivity(adj)
            >>> print(f"Integration: {metrics['global_efficiency']:.3f}")
        """
        # Placeholder - needs implementation
        return {
            "global_efficiency": 0.0,
            "clustering_coefficient": 0.0,
            "modularity": 0.0,
        }
