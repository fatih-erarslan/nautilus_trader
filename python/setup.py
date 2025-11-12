"""
Setup script for HyperPhysics Python package.

Builds the PyO3 Rust extension and installs Python dependencies.
Optimized for AMD 6800XT with ROCm support.

Installation:
    # From HyperPhysics directory
    pip install -e python/

    # Or with maturin (recommended for development)
    maturin develop --release --features python

Requirements:
    - Python 3.9+
    - PyTorch 2.2.2 with ROCm 5.7+
    - Rust 1.70+
    - maturin 1.0+
"""

from setuptools import setup
from setuptools_rust import Binding, RustExtension

# Read README
with open("../README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="hyperphysics-finance",
    version="0.1.0",
    author="HyperPhysics Team",
    author_email="team@hyperphysics.io",
    description="GPU-accelerated financial computing with hyperbolic geometry",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hyperphysics/hyperphysics",
    project_urls={
        "Bug Tracker": "https://github.com/hyperphysics/hyperphysics/issues",
        "Documentation": "https://hyperphysics.readthedocs.io",
        "Source Code": "https://github.com/hyperphysics/hyperphysics",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Rust",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Office/Business :: Financial",
    ],
    rust_extensions=[
        RustExtension(
            "hyperphysics_finance.hyperphysics_finance",
            path="../Cargo.toml",
            binding=Binding.PyO3,
            features=["python"],
            debug=False,
        )
    ],
    packages=["hyperphysics_finance"],
    package_dir={"hyperphysics_finance": "."},
    py_modules=[
        "hyperphysics_torch",
        "rocm_setup",
        "integration_example",
    ],
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.2.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.10.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-benchmark>=4.0.0",
            "maturin>=1.0.0",
            "black>=23.0.0",
            "ruff>=0.1.0",
        ],
        "freqtrade": [
            "freqtrade>=2023.0.0",
            "ccxt>=4.0.0",
        ],
        "viz": [
            "matplotlib>=3.7.0",
            "plotly>=5.0.0",
            "dash>=2.0.0",
        ],
    },
    zip_safe=False,
)
