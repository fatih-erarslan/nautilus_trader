"""
Quantum-Biological Market Intuition Agent (QBMIA)

A sophisticated market analysis system combining quantum computing simulations,
biological learning patterns, and game-theoretic strategies for advanced
decision-making in financial markets.
"""

__version__ = "1.0.0"
__author__ = "QBMIA Development Team"

import logging
import warnings
from typing import Optional

# Configure logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

# Version tracking for state serialization
SERIALIZATION_VERSION = "1.0.0"

# Lazy imports to reduce initial load time
_modules = {
    'agent': None,
    'state_manager': None,
    'resource_manager': None,
    'hardware_optimizer': None
}

def _lazy_import(module_name: str):
    """Lazy import mechanism for heavy modules."""
    if _modules[module_name] is None:
        if module_name == 'agent':
            from .core.agent import QBMIAAgent
            _modules['agent'] = QBMIAAgent
        elif module_name == 'state_manager':
            from .core.state_manager import StateManager
            _modules['state_manager'] = StateManager
        elif module_name == 'resource_manager':
            from .orchestration.resource_manager import ResourceManager
            _modules['resource_manager'] = ResourceManager
        elif module_name == 'hardware_optimizer':
            from .core.hardware_optimizer import QBMIAHardwareOptimizer
            _modules['hardware_optimizer'] = QBMIAHardwareOptimizer
    return _modules[module_name]

# Export main classes through properties
@property
def QBMIAAgent():
    return _lazy_import('agent')

@property
def StateManager():
    return _lazy_import('state_manager')

@property
def ResourceManager():
    return _lazy_import('resource_manager')

# Hardware capability detection
def detect_capabilities() -> dict:
    """Detect available hardware acceleration capabilities."""
    try:
        from .core.hardware_optimizer import detect_hardware_capabilities
        return detect_hardware_capabilities()
    except ImportError as e:
        warnings.warn(f"Hardware detection failed: {e}")
        return {'cpu': True, 'cuda': False, 'rocm': False}

# Version compatibility check
def check_serialization_compatibility(version: str) -> bool:
    """Check if a serialized state version is compatible."""
    major_current = int(SERIALIZATION_VERSION.split('.')[0])
    major_check = int(version.split('.')[0])
    return major_current == major_check

__all__ = [
    'QBMIAAgent',
    'StateManager',
    'ResourceManager',
    'SERIALIZATION_VERSION',
    'detect_capabilities',
    'check_serialization_compatibility'
]
