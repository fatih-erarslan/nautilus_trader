"""Low-level FFI bindings to QKS Rust library using ctypes.

This module provides the ctypes interface to the Rust shared library.
DO NOT use these functions directly - use the high-level APIs instead.
"""

import ctypes
import platform
import os
from pathlib import Path
from typing import Optional

# Determine library name based on platform
_SYSTEM = platform.system()
if _SYSTEM == "Darwin":
    _LIB_NAME = "libqks_plugin.dylib"
elif _SYSTEM == "Linux":
    _LIB_NAME = "libqks_plugin.so"
elif _SYSTEM == "Windows":
    _LIB_NAME = "qks_plugin.dll"
else:
    raise RuntimeError(f"Unsupported platform: {_SYSTEM}")

# Search paths for the shared library
_SEARCH_PATHS = [
    # Development path (relative to bindings)
    Path(__file__).parent.parent.parent.parent / "target" / "release",
    Path(__file__).parent.parent.parent.parent / "target" / "debug",
    # Installed path (site-packages)
    Path(__file__).parent / "lib",
    # System paths
    Path("/usr/local/lib"),
    Path("/usr/lib"),
]


def _find_library() -> Optional[Path]:
    """Find the QKS shared library."""
    for search_path in _SEARCH_PATHS:
        lib_path = search_path / _LIB_NAME
        if lib_path.exists():
            return lib_path

    # Try system library search
    lib = ctypes.util.find_library("qks_plugin")
    if lib:
        return Path(lib)

    return None


# Load the library
_lib_path = _find_library()
if _lib_path is None:
    raise RuntimeError(
        f"QKS library not found. Searched: {[str(p) for p in _SEARCH_PATHS]}\n"
        "Please build the Rust library with: cargo build --release"
    )

_lib = ctypes.CDLL(str(_lib_path))


# ============================================================================
# Type Definitions
# ============================================================================

class QksHandle(ctypes.c_void_p):
    """Opaque handle to QKS instance."""
    pass


class QksConfig(ctypes.Structure):
    """QKS configuration."""
    _fields_ = [
        ("num_qubits", ctypes.c_size_t),
        ("use_gpu", ctypes.c_bool),
        ("enable_consciousness", ctypes.c_bool),
        ("enable_metacognition", ctypes.c_bool),
    ]


class PhiResultC(ctypes.Structure):
    """IIT Φ computation result."""
    _fields_ = [
        ("phi", ctypes.c_double),
        ("mip_size", ctypes.c_size_t),
        ("num_states", ctypes.c_size_t),
        ("computation_time_ms", ctypes.c_double),
    ]


class IntrospectionReportC(ctypes.Structure):
    """Metacognitive introspection report."""
    _fields_ = [
        ("num_beliefs", ctypes.c_size_t),
        ("num_goals", ctypes.c_size_t),
        ("num_capabilities", ctypes.c_size_t),
        ("confidence", ctypes.c_double),
        ("timestamp_ms", ctypes.c_uint64),
    ]


# ============================================================================
# Core Functions
# ============================================================================

# Plugin lifecycle
_lib.qks_create.argtypes = [ctypes.POINTER(QksConfig)]
_lib.qks_create.restype = QksHandle

_lib.qks_destroy.argtypes = [QksHandle]
_lib.qks_destroy.restype = None

_lib.qks_version.argtypes = []
_lib.qks_version.restype = ctypes.c_char_p


# ============================================================================
# Layer 6: Consciousness
# ============================================================================

# IIT Φ computation
_lib.qks_compute_phi.argtypes = [
    QksHandle,
    ctypes.POINTER(ctypes.c_double),  # network_state
    ctypes.c_size_t,                  # state_size
    ctypes.POINTER(PhiResultC),       # result
]
_lib.qks_compute_phi.restype = ctypes.c_int

# Global Workspace broadcast
_lib.qks_broadcast.argtypes = [
    QksHandle,
    ctypes.c_void_p,      # content (opaque)
    ctypes.c_size_t,      # content_size
    ctypes.c_double,      # priority
]
_lib.qks_broadcast.restype = ctypes.c_int


# ============================================================================
# Layer 7: Metacognition
# ============================================================================

# Introspection
_lib.qks_introspect.argtypes = [
    QksHandle,
    ctypes.POINTER(IntrospectionReportC),
]
_lib.qks_introspect.restype = ctypes.c_int

# Self-monitoring
_lib.qks_monitor_performance.argtypes = [
    QksHandle,
    ctypes.POINTER(ctypes.c_double),  # metrics
    ctypes.c_size_t,                  # num_metrics
]
_lib.qks_monitor_performance.restype = ctypes.c_int


# ============================================================================
# Layer 8: Integration
# ============================================================================

# Cognitive cycle
_lib.qks_cognitive_cycle.argtypes = [
    QksHandle,
    ctypes.c_void_p,      # sensory_input
    ctypes.c_size_t,      # input_size
    ctypes.c_void_p,      # output_buffer
    ctypes.c_size_t,      # output_size
]
_lib.qks_cognitive_cycle.restype = ctypes.c_int


# ============================================================================
# Error Handling
# ============================================================================

_lib.qks_get_last_error.argtypes = []
_lib.qks_get_last_error.restype = ctypes.c_char_p


def check_error(result: int) -> None:
    """Check FFI call result and raise exception on error."""
    if result != 0:
        error_msg = _lib.qks_get_last_error()
        if error_msg:
            msg = error_msg.decode('utf-8')
        else:
            msg = f"QKS error code: {result}"
        raise RuntimeError(msg)


# Export library and handle type
__all__ = [
    "_lib",
    "QksHandle",
    "QksConfig",
    "PhiResultC",
    "IntrospectionReportC",
    "check_error",
]
