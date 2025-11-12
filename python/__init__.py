"""
HyperPhysics Python Bridge

GPU-accelerated financial computing with hyperbolic geometry.

Modules:
    hyperphysics_torch: PyTorch GPU kernels for order book and risk
    rocm_setup: AMD GPU configuration and optimization
    integration_example: Freqtrade integration examples

Example:
    >>> from hyperphysics_torch import HyperbolicOrderBook, GPURiskEngine
    >>> ob = HyperbolicOrderBook(device="cuda:0")
    >>> risk = GPURiskEngine(device="cuda:0")
"""

__version__ = "0.1.0"
__author__ = "HyperPhysics Team"

from .hyperphysics_torch import (
    HyperbolicOrderBook,
    GPURiskEngine,
    get_device_info,
)

from .rocm_setup import (
    ROCmConfig,
    setup_rocm_for_freqtrade,
)

from .integration_example import (
    HyperPhysicsFinancialEngine,
)

__all__ = [
    "HyperbolicOrderBook",
    "GPURiskEngine",
    "get_device_info",
    "ROCmConfig",
    "setup_rocm_for_freqtrade",
    "HyperPhysicsFinancialEngine",
]
