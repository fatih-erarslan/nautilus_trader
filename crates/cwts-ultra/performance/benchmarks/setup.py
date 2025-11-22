#!/usr/bin/env python3
"""
Setup script for compiling Cython extensions with optimal performance flags
Implements scientific compilation for zero computational waste
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os
import sys
import platform

def get_optimization_flags():
    """Get platform-specific optimization flags"""
    base_flags = [
        "-O3",                    # Maximum optimization
        "-march=native",          # Native CPU architecture
        "-mtune=native",          # Tune for native CPU
        "-ffast-math",           # Fast math optimizations
        "-funroll-loops",        # Loop unrolling
        "-fomit-frame-pointer",  # Omit frame pointer for better performance
        "-msse4.2",              # SSE 4.2 support
        "-mavx",                 # AVX support
        "-mavx2",                # AVX2 support
        "-mfma",                 # FMA support
    ]
    
    # Platform-specific flags
    if platform.system() == "Linux":
        base_flags.extend([
            "-flto",              # Link-time optimization
            "-fuse-linker-plugin" # Use linker plugin for LTO
        ])
    elif platform.system() == "Darwin":  # macOS
        base_flags.extend([
            "-flto",
            "-Wl,-dead_strip"     # Remove dead code
        ])
    
    return base_flags

def get_link_flags():
    """Get linking optimization flags"""
    link_flags = []
    
    if platform.system() == "Linux":
        link_flags.extend([
            "-flto",
            "-Wl,--gc-sections",  # Garbage collect sections
            "-Wl,-O1"            # Linker optimization
        ])
    elif platform.system() == "Darwin":
        link_flags.extend([
            "-flto",
            "-Wl,-dead_strip"
        ])
        
    return link_flags

# Compiler and linker flags
optimization_flags = get_optimization_flags()
link_flags = get_link_flags()

# Extensions to compile
extensions = [
    Extension(
        "optimized_market_data",
        sources=["../cython/optimized_market_data.pyx"],
        include_dirs=[
            np.get_include(),
            "/usr/include",
            "/usr/local/include"
        ],
        extra_compile_args=optimization_flags + [
            "-fopenmp",           # OpenMP support
            "-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION",
            "-DCYTHON_WITHOUT_ASSERTIONS"  # Remove Cython assertions
        ],
        extra_link_args=link_flags + ["-fopenmp"],
        language="c++",
        define_macros=[
            ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"),
            ("CYTHON_WITHOUT_ASSERTIONS", None)
        ]
    ),
    Extension(
        "vectorized_computations",
        sources=["../simd/vectorized_computations.pyx"],
        include_dirs=[
            np.get_include(),
            "/usr/include",
            "/usr/local/include"
        ],
        extra_compile_args=optimization_flags + [
            "-fopenmp",
            "-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION",
            "-DCYTHON_WITHOUT_ASSERTIONS"
        ],
        extra_link_args=link_flags + ["-fopenmp"],
        language="c++",
        define_macros=[
            ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"),
            ("CYTHON_WITHOUT_ASSERTIONS", None)
        ]
    )
]

# Cython compilation directives for maximum performance
compiler_directives = {
    "language_level": 3,
    "boundscheck": False,
    "wraparound": False,
    "nonecheck": False,
    "cdivision": True,
    "profile": False,
    "linetrace": False,
    "embedsignature": True,
    "optimize.use_switch": True,
    "optimize.unpack_method_calls": True,
    "warn.undeclared": True,
    "warn.unreachable": True,
    "warn.maybe_uninitialized": True
}

if __name__ == "__main__":
    setup(
        name="cwts_performance_extensions",
        version="1.0.0",
        description="High-performance Cython extensions for CWTS trading system",
        author="CWTS Development Team",
        ext_modules=cythonize(
            extensions,
            compiler_directives=compiler_directives,
            annotate=True,  # Generate HTML annotation files
            nthreads=os.cpu_count()  # Parallel compilation
        ),
        zip_safe=False,
        python_requires=">=3.8",
        install_requires=[
            "numpy>=1.21.0",
            "cython>=0.29.0"
        ]
    )