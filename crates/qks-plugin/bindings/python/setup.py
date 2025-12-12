"""Setup script for QKS Python bindings."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = ""
if readme_file.exists():
    long_description = readme_file.read_text(encoding="utf-8")
else:
    long_description = "Quantum Knowledge System - Python bindings for cognitive computing"

setup(
    name="qks-plugin",
    version="0.1.0",
    author="QKS Development Team",
    author_email="qks@example.com",
    description="Quantum Knowledge System - Drop-in cognitive super pill for Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/qks/qks-plugin",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Rust",
    ],
    keywords="quantum consciousness metacognition cognitive-computing IIT neuroscience",
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "mypy>=1.0.0",
            "ruff>=0.0.250",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
    package_data={
        "qks": [
            "lib/*.so",
            "lib/*.dylib",
            "lib/*.dll",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    project_urls={
        "Documentation": "https://qks.readthedocs.io",
        "Source": "https://github.com/qks/qks-plugin",
        "Bug Reports": "https://github.com/qks/qks-plugin/issues",
    },
)
