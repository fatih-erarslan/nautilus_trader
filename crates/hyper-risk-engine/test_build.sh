#!/bin/bash
# Test build script for hyper-risk-engine

cd /Volumes/Kingston/Developer/Ashina/HyperPhysics

echo "=========================================="
echo "Building hyper-risk-engine..."
echo "=========================================="

cargo build -p hyper-risk-engine --lib --no-default-features 2>&1

BUILD_STATUS=$?

echo ""
echo "=========================================="
if [ $BUILD_STATUS -eq 0 ]; then
    echo "✅ BUILD SUCCESSFUL"
else
    echo "❌ BUILD FAILED (Exit code: $BUILD_STATUS)"
fi
echo "=========================================="

exit $BUILD_STATUS
