"""
Setup script for CWTS Ultra Python/Cython bindings
Builds high-performance extensions for FreqTrade integration
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os

# Compilation flags for maximum performance
extra_compile_args = [
    '-O3',                # Maximum optimization
    '-march=native',      # CPU-specific optimizations
    '-mtune=native',      
    '-ffast-math',        # Fast math operations
    '-funroll-loops',     # Loop unrolling
    '-ftree-vectorize',   # Auto-vectorization
    '-fopenmp',           # OpenMP support
    '-std=c11',           # C11 standard
    '-Wall',              # All warnings
    '-Wno-unused-function',
]

extra_link_args = [
    '-fopenmp',           # Link OpenMP
    '-lrt',               # Real-time library (for shared memory)
    '-lpthread',          # POSIX threads
]

# Define the Cython extension
extensions = [
    Extension(
        "cwts_client",
        ["cwts_client.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language="c",
    )
]

# Package configuration
setup(
    name="cwts-ultra-freqtrade",
    version="2.0.0",
    description="Ultra-low latency CWTS Ultra bindings for FreqTrade",
    author="CWTS Ultra Team",
    python_requires=">=3.8",
    
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            'language_level': 3,
            'boundscheck': False,
            'wraparound': False,
            'nonecheck': False,
            'cdivision': True,
            'profile': False,
            'embedsignature': True,
        },
        annotate=True,  # Generate HTML annotation files
    ),
    
    install_requires=[
        "numpy>=1.20.0",
        "cython>=0.29.30",
        "websockets>=10.0",
        "msgpack>=1.0.0",
        "freqtrade>=2023.0",
    ],
    
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-asyncio>=0.18.0',
            'pytest-benchmark>=3.4.0',
            'black>=22.0.0',
            'mypy>=0.950',
        ],
    },
    
    package_data={
        'cwts_ultra_freqtrade': ['*.pyx', '*.pxd'],
    },
    
    zip_safe=False,
)