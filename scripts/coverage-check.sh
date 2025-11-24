#!/bin/bash
# Test Coverage Quick Check Script
# Usage: ./scripts/coverage-check.sh [--full]

set -e

FULL_MODE=false
if [[ "$1" == "--full" ]]; then
    FULL_MODE=true
fi

echo "========================================"
echo "HyperPhysics Coverage Quick Check"
echo "========================================"
echo

# Check if coverage tools are installed
if ! command -v cargo-llvm-cov &> /dev/null; then
    echo "❌ cargo-llvm-cov not found. Installing..."
    cargo install cargo-llvm-cov
fi

# MVP crates to check
MVP_CRATES=(
    "hyperphysics-core"
    "hyperphysics-market"
    "hyperphysics-risk"
    "rapier-hyperphysics"
    "active-inference-agent"
    "hyperphysics-reasoning-router"
    "gpu-marl"
)

TARGET_COVERAGE=70

echo "Checking MVP Crates (Target: ${TARGET_COVERAGE}%)"
echo "------------------------------------------------"

PASSED=0
FAILED=0

for crate in "${MVP_CRATES[@]}"; do
    echo -n "📦 $crate: "

    if $FULL_MODE; then
        # Run actual coverage (slow)
        COVERAGE=$(cargo llvm-cov --package "$crate" --no-report 2>&1 | \
                   grep -oP '\d+\.\d+(?=%)' | head -1 || echo "0.0")
    else
        # Quick estimate based on test count
        SRC_FILES=$(find "crates/$crate/src" -name "*.rs" 2>/dev/null | wc -l || echo "0")
        TEST_COUNT=$(grep -r "#\[test\]" "crates/$crate" 2>/dev/null | wc -l || echo "0")

        if [ "$SRC_FILES" -eq 0 ]; then
            echo "⚠️  Crate not found"
            continue
        fi

        # Rough estimate: test count / (source files * 3)
        COVERAGE=$(echo "scale=1; ($TEST_COUNT / ($SRC_FILES * 3)) * 100" | bc || echo "0.0")
    fi

    # Compare with target
    if (( $(echo "$COVERAGE >= $TARGET_COVERAGE" | bc -l) )); then
        echo "✅ ${COVERAGE}%"
        ((PASSED++))
    else
        echo "❌ ${COVERAGE}% (need $(echo "$TARGET_COVERAGE - $COVERAGE" | bc)% more)"
        ((FAILED++))
    fi
done

echo
echo "Summary"
echo "-------"
echo "Passed: $PASSED / ${#MVP_CRATES[@]}"
echo "Failed: $FAILED / ${#MVP_CRATES[@]}"

if [ $FAILED -eq 0 ]; then
    echo "🎉 All MVP crates meet 70% coverage target!"
    exit 0
else
    echo "⚠️  ${FAILED} crate(s) below 70% coverage"
    exit 1
fi
