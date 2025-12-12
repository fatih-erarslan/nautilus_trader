"""Quantum Knowledge System - Python Bindings

Drop-in super pill of wisdom for cognitive computing applications.

This package provides Python bindings to the QKS Rust core, enabling:
- Integrated Information Theory (IIT 3.0) - Φ computation
- Global Workspace Theory - Broadcast mechanisms
- Metacognitive monitoring - Real-time introspection
- Full cognitive cycles - Perception-cognition-action

Example:
    >>> from qks import QKSPlugin
    >>> plugin = QKSPlugin()
    >>> phi = plugin.consciousness.compute_phi(network_state)
    >>> print(f"Consciousness level: {phi:.3f}")

Architecture:
    Layer 6: Consciousness (IIT + Global Workspace)
    Layer 7: Metacognition (Introspection + Self-monitoring)
    Layer 8: Integration (Full cognitive cycles)
"""

__version__ = "0.1.0"
__author__ = "QKS Development Team"
__license__ = "MIT"

from qks.plugin import QKSPlugin
from qks.consciousness import Consciousness, PhiResult, BroadcastResult
from qks.metacognition import Metacognition, IntrospectionReport, BeliefState
from qks.integration import Integration, CognitiveOutput, SensoryInput

__all__ = [
    "QKSPlugin",
    "Consciousness",
    "PhiResult",
    "BroadcastResult",
    "Metacognition",
    "IntrospectionReport",
    "BeliefState",
    "Integration",
    "CognitiveOutput",
    "SensoryInput",
]
