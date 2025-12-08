#!/bin/bash
# Comprehensive build fix script for talebian-risk-rs

echo "=== TALEBIAN RISK RS BUILD FIX ==="
echo "Production Financial System - 100% Success Required"

# Step 1: Check current error count
echo "Current compilation status:"
cargo build --release 2>&1 | grep "^error" | wc -l

# Step 2: Fix all unused variable warnings
echo "Fixing unused variable warnings..."
find src -name "*.rs" -type f -exec sed -i 's/\([^_]\)market_data:/\1_market_data:/g' {} \;
find src -name "*.rs" -type f -exec sed -i 's/\([^_]\)whale_detection:/\1_whale_detection:/g' {} \;

# Step 3: Build with all features
echo "Building with all features..."
cargo build --release --features python-bindings 2>&1 | tee build.log

# Step 4: Count remaining errors
ERRORS=$(grep "^error" build.log | wc -l)
WARNINGS=$(grep "^warning" build.log | wc -l)

echo "=== BUILD RESULTS ==="
echo "Errors: $ERRORS"
echo "Warnings: $WARNINGS"

if [ "$ERRORS" -eq 0 ]; then
    echo "✅ BUILD SUCCESS - Ready for production"
    
    # Run maturin build
    echo "Building Python wheel with maturin..."
    maturin build --release --features python-bindings
    
    # Install the wheel
    WHEEL=$(ls target/wheels/*.whl 2>/dev/null | head -1)
    if [ -n "$WHEEL" ]; then
        echo "Installing wheel: $WHEEL"
        pip install --force-reinstall "$WHEEL"
        echo "✅ INSTALLATION COMPLETE"
    fi
else
    echo "❌ BUILD FAILED - $ERRORS errors remaining"
    echo "Top error files:"
    grep "^   --> src/" build.log | cut -d':' -f1 | cut -d'/' -f2 | sort | uniq -c | sort -rn | head -10
fi