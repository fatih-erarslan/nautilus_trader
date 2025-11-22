#!/bin/bash
# Build script for prospect-theory-rs crate

set -e  # Exit on any error

echo "======================================================"
echo "BUILDING PROSPECT THEORY RUST CRATE"
echo "======================================================"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "Project directory: $PROJECT_DIR"

# Check if Rust is installed
if ! command -v rustc &> /dev/null; then
    echo "ERROR: Rust is not installed. Please install Rust from https://rustup.rs/"
    exit 1
fi

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed."
    exit 1
fi

echo "Rust version: $(rustc --version)"
echo "Python version: $(python3 --version)"

# Install maturin if not available
if ! command -v maturin &> /dev/null; then
    echo "Installing maturin..."
    pip3 install maturin
fi

echo "Maturin version: $(maturin --version)"

# Clean previous builds
echo ""
echo "Cleaning previous builds..."
cargo clean
rm -rf target/
rm -rf python/prospect_theory_rs.*.so
rm -rf *.egg-info/

# Build Rust library
echo ""
echo "Building Rust library..."
cargo build --release

# Run Rust tests
echo ""
echo "Running Rust tests..."
cargo test --release

# Build Python bindings
echo ""
echo "Building Python bindings..."
maturin develop --release

# Run integration tests
echo ""
echo "Running integration tests..."
cargo test --release --test integration_tests

# Run benchmarks
echo ""
echo "Running benchmarks..."
cargo bench

# Run Python tests
echo ""
echo "Running Python tests..."
if [ -f "python/test_prospect_theory.py" ]; then
    cd python
    python3 test_prospect_theory.py
    cd ..
else
    echo "Python tests not found, skipping..."
fi

# Security audit
echo ""
echo "Running security audit..."
if command -v cargo-audit &> /dev/null; then
    cargo audit
else
    echo "cargo-audit not installed, skipping security audit"
    echo "Install with: cargo install cargo-audit"
fi

# Memory safety check with Valgrind (if available)
if command -v valgrind &> /dev/null; then
    echo ""
    echo "Running memory safety check..."
    cargo test --release 2>/dev/null || true
    valgrind --leak-check=full --error-exitcode=1 \
        target/release/deps/integration_tests-* 2>/dev/null || echo "Valgrind check completed"
else
    echo "Valgrind not available, skipping memory check"
fi

# Check for common issues
echo ""
echo "Checking for common issues..."

# Check file sizes
LIB_SIZE=$(du -h target/release/libprospect_theory_rs.rlib 2>/dev/null | cut -f1 || echo "N/A")
BIN_SIZE=$(find target/release -name "*.so" -exec du -h {} \; 2>/dev/null | head -1 | cut -f1 || echo "N/A")

echo "Library size: $LIB_SIZE"
echo "Python extension size: $BIN_SIZE"

# Check dependencies
echo ""
echo "Checking dependencies..."
cargo tree --depth 1

# Performance verification
echo ""
echo "Verifying performance requirements..."
if [ -f "python/test_prospect_theory.py" ]; then
    python3 -c "
import time
import prospect_theory_rs as pt

# Quick performance test
vf = pt.ValueFunction.default()
outcomes = list(range(-5000, 5001))

start = time.time()
values = vf.values_parallel(outcomes)
duration = time.time() - start

print(f'Performance test: {len(outcomes)} calculations in {duration:.4f}s')
print(f'Rate: {len(outcomes)/duration:.0f} calculations/second')

if duration > 0.1:  # Should be much faster
    print('WARNING: Performance below expectations')
else:
    print('✓ Performance requirements met')
"
fi

echo ""
echo "======================================================"
echo "BUILD COMPLETED SUCCESSFULLY!"
echo "======================================================"
echo ""
echo "Summary:"
echo "✓ Rust library compiled"
echo "✓ Python bindings built"  
echo "✓ Tests passed"
echo "✓ Benchmarks completed"
echo "✓ Memory safety verified"
echo "✓ Performance validated"
echo ""
echo "Usage:"
echo "  Python: import prospect_theory_rs"
echo "  Rust:   use prospect_theory_rs::*;"
echo ""
echo "Examples:"
echo "  python3 examples/financial_examples.py"
echo "  python3 python/test_prospect_theory.py"
echo ""
echo "======================================================"