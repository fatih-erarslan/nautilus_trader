#!/bin/bash
# Validate SIMD Performance Improvements
# Compares scalar vs SIMD benchmarks and validates targets

set -e

echo "═══════════════════════════════════════════════════════════════"
echo "  HyperPhysics SIMD Performance Validation"
echo "═══════════════════════════════════════════════════════════════"
echo

# Check prerequisites
if ! command -v cargo &> /dev/null; then
    echo "✗ Error: Rust not installed"
    exit 1
fi

# Create output directory
mkdir -p docs/performance/validation
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REPORT="docs/performance/validation/simd_validation_${TIMESTAMP}.txt"

echo "Performance Validation Report" > "$REPORT"
echo "Generated: $(date)" >> "$REPORT"
echo "═══════════════════════════════════════════════════════════════" >> "$REPORT"
echo >> "$REPORT"

# Step 1: Detect backend
echo "✓ Step 1: Detecting SIMD backend"
BACKEND_INFO=$(cargo run --features simd --quiet --bin detect_backend 2>/dev/null || echo "Detection unavailable")
echo "  Backend: $BACKEND_INFO"
echo "SIMD Backend: $BACKEND_INFO" >> "$REPORT"
echo >> "$REPORT"

# Step 2: Run scalar benchmarks
echo "✓ Step 2: Running scalar baseline benchmarks"
echo "  (This may take 2-3 minutes...)"
cargo bench --workspace --no-default-features --quiet -- \
    --save-baseline scalar_${TIMESTAMP} \
    engine > /tmp/scalar_bench_${TIMESTAMP}.txt 2>&1 || true

SCALAR_ENGINE=$(grep "engine_step" /tmp/scalar_bench_${TIMESTAMP}.txt | awk '{print $2}' || echo "N/A")
echo "  ✓ Scalar baseline: $SCALAR_ENGINE"
echo >> "$REPORT"
echo "Scalar Baseline:" >> "$REPORT"
echo "  Engine step: $SCALAR_ENGINE" >> "$REPORT"
echo >> "$REPORT"

# Step 3: Run SIMD benchmarks
echo "✓ Step 3: Running SIMD optimized benchmarks"
echo "  (This may take 2-3 minutes...)"
cargo bench --workspace --features simd --quiet -- \
    --save-baseline simd_${TIMESTAMP} \
    engine > /tmp/simd_bench_${TIMESTAMP}.txt 2>&1 || true

SIMD_ENGINE=$(grep "engine_step" /tmp/simd_bench_${TIMESTAMP}.txt | awk '{print $2}' || echo "N/A")
echo "  ✓ SIMD optimized: $SIMD_ENGINE"
echo >> "$REPORT"
echo "SIMD Optimized:" >> "$REPORT"
echo "  Engine step: $SIMD_ENGINE" >> "$REPORT"
echo >> "$REPORT"

# Step 4: Calculate speedup
echo "✓ Step 4: Calculating speedup"

# Extract numeric values (remove 'ns' suffix)
SCALAR_NS=$(echo "$SCALAR_ENGINE" | sed 's/[^0-9.]//g' || echo "0")
SIMD_NS=$(echo "$SIMD_ENGINE" | sed 's/[^0-9.]//g' || echo "0")

if [ "$SCALAR_NS" != "0" ] && [ "$SIMD_NS" != "0" ]; then
    SPEEDUP=$(echo "scale=2; $SCALAR_NS / $SIMD_NS" | bc)
    IMPROVEMENT=$(echo "scale=1; (($SCALAR_NS - $SIMD_NS) / $SCALAR_NS) * 100" | bc)

    echo "  Speedup: ${SPEEDUP}×"
    echo "  Improvement: ${IMPROVEMENT}%"

    echo "Performance Analysis:" >> "$REPORT"
    echo "  Speedup: ${SPEEDUP}×" >> "$REPORT"
    echo "  Improvement: ${IMPROVEMENT}%" >> "$REPORT"
    echo >> "$REPORT"

    # Validate against targets
    echo >> "$REPORT"
    echo "Target Validation:" >> "$REPORT"

    # Minimum gate: 3× speedup
    if (( $(echo "$SPEEDUP >= 3.0" | bc -l) )); then
        echo "  ✅ Minimum gate (3×): PASS" >> "$REPORT"
        GATE_PASS=true
    else
        echo "  ❌ Minimum gate (3×): FAIL (only ${SPEEDUP}×)" >> "$REPORT"
        GATE_PASS=false
    fi

    # Target goal: 5× speedup
    if (( $(echo "$SPEEDUP >= 5.0" | bc -l) )); then
        echo "  ✅ Target goal (5×): PASS" >> "$REPORT"
        TARGET_PASS=true
    else
        echo "  ⚠️ Target goal (5×): PARTIAL (achieved ${SPEEDUP}×)" >> "$REPORT"
        TARGET_PASS=false
    fi

    # Stretch goal: 8× speedup
    if (( $(echo "$SPEEDUP >= 8.0" | bc -l) )); then
        echo "  ✅ Stretch goal (8×): PASS" >> "$REPORT"
        STRETCH_PASS=true
    else
        echo "  ℹ️ Stretch goal (8×): NOT ACHIEVED (achieved ${SPEEDUP}×)" >> "$REPORT"
        STRETCH_PASS=false
    fi

else
    echo "  ⚠ Warning: Could not calculate speedup"
    echo "  Scalar: $SCALAR_NS ns, SIMD: $SIMD_NS ns"
    GATE_PASS=false
    TARGET_PASS=false
    STRETCH_PASS=false
fi

echo >> "$REPORT"

# Step 5: Detailed breakdown
echo "✓ Step 5: Analyzing component performance"
echo >> "$REPORT"
echo "Component Breakdown:" >> "$REPORT"

# Check individual components
for component in entropy energy magnetization; do
    SCALAR_COMP=$(grep "$component" /tmp/scalar_bench_${TIMESTAMP}.txt 2>/dev/null | head -1 | awk '{print $2}' || echo "N/A")
    SIMD_COMP=$(grep "$component" /tmp/simd_bench_${TIMESTAMP}.txt 2>/dev/null | head -1 | awk '{print $2}' || echo "N/A")

    if [ "$SCALAR_COMP" != "N/A" ] && [ "$SIMD_COMP" != "N/A" ]; then
        echo "  $component: $SCALAR_COMP → $SIMD_COMP" >> "$REPORT"
    fi
done

echo >> "$REPORT"

# Step 6: System information
echo "✓ Step 6: Recording system information"
echo >> "$REPORT"
echo "System Information:" >> "$REPORT"
echo "  Rust: $(rustc --version)" >> "$REPORT"
echo "  Cargo: $(cargo --version)" >> "$REPORT"
echo "  OS: $(uname -s)" >> "$REPORT"
echo "  Arch: $(uname -m)" >> "$REPORT"
if command -v sysctl &> /dev/null; then
    echo "  CPU: $(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo 'Unknown')" >> "$REPORT"
    echo "  Cores: $(sysctl -n hw.ncpu 2>/dev/null || echo 'Unknown')" >> "$REPORT"
fi
echo >> "$REPORT"

# Step 7: Generate summary
echo >> "$REPORT"
echo "═══════════════════════════════════════════════════════════════" >> "$REPORT"
echo "FINAL ASSESSMENT" >> "$REPORT"
echo "═══════════════════════════════════════════════════════════════" >> "$REPORT"

if [ "$GATE_PASS" = true ]; then
    echo "Status: ✅ PASS (Week 3 Gate Cleared)" >> "$REPORT"
    echo >> "$REPORT"
    echo "The SIMD implementation has successfully achieved the minimum" >> "$REPORT"
    echo "3× speedup requirement and is ready for production deployment." >> "$REPORT"

    if [ "$TARGET_PASS" = true ]; then
        echo >> "$REPORT"
        echo "🎯 TARGET ACHIEVED: 5× speedup goal met!" >> "$REPORT"
        echo >> "$REPORT"
        echo "Phase 2 Week 3 objectives EXCEEDED. Recommend immediate" >> "$REPORT"
        echo "integration into main branch and progression to Week 4." >> "$REPORT"
    fi

    if [ "$STRETCH_PASS" = true ]; then
        echo >> "$REPORT"
        echo "🏆 STRETCH GOAL ACHIEVED: 8× speedup!" >> "$REPORT"
        echo >> "$REPORT"
        echo "Exceptional performance. Consider publishing benchmark" >> "$REPORT"
        echo "results and submitting to academic conferences." >> "$REPORT"
    fi
else
    echo "Status: ❌ FAIL (Did not meet minimum 3× speedup)" >> "$REPORT"
    echo >> "$REPORT"
    echo "Recommendation: Review SIMD implementation for optimization" >> "$REPORT"
    echo "opportunities. Check backend detection and vectorization." >> "$REPORT"
    echo >> "$REPORT"
    echo "Possible causes:" >> "$REPORT"
    echo "  - Backend not properly detected" >> "$REPORT"
    echo "  - Vectorization not applied" >> "$REPORT"
    echo "  - Test data too small for SIMD benefit" >> "$REPORT"
    echo "  - Memory alignment issues" >> "$REPORT"
fi

echo >> "$REPORT"
echo "Report saved to: $REPORT" >> "$REPORT"

# Display report
echo
echo "═══════════════════════════════════════════════════════════════"
cat "$REPORT"
echo "═══════════════════════════════════════════════════════════════"
echo
echo "Full report saved to: $REPORT"

# Exit with appropriate code
if [ "$GATE_PASS" = true ]; then
    echo
    echo "✅ SIMD validation PASSED - Ready for Phase 2 Week 3 completion"
    exit 0
else
    echo
    echo "❌ SIMD validation FAILED - Additional optimization required"
    exit 1
fi
