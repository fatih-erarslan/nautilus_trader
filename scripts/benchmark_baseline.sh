#!/bin/bash
#
# Benchmark Baseline Script
# Establishes performance baselines for Phase 2 SIMD comparison
#

set -e

echo "⚡ HyperPhysics Performance Baseline"
echo "===================================="
echo ""

cd "$(dirname "$0")/.."

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASELINE_DIR="docs/performance/baselines"
mkdir -p "$BASELINE_DIR"

echo -e "${YELLOW}⏳${NC} Running benchmarks (this may take 5-10 minutes)..."
echo ""

# Run benchmarks and save output
BASELINE_FILE="$BASELINE_DIR/baseline_$TIMESTAMP.txt"
cargo bench --workspace --all-features -- --save-baseline scalar_baseline 2>&1 | tee "$BASELINE_FILE"

echo ""
echo -e "${GREEN}✓${NC} Baseline saved to: $BASELINE_FILE"

# Extract key metrics
echo ""
echo "📊 Key Metrics:"
echo "==============="

# Extract benchmark results
grep "time:" "$BASELINE_FILE" | head -10

# Generate summary
SUMMARY_FILE="$BASELINE_DIR/SUMMARY_$TIMESTAMP.md"
cat > "$SUMMARY_FILE" << EOF
# Performance Baseline - $TIMESTAMP

## Test Environment
- Date: $(date)
- Rust: $(rustc --version)
- CPU: $(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "Unknown")
- Cores: $(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo "Unknown")

## Benchmark Results

See full output: baseline_$TIMESTAMP.txt

### Critical Paths
$(grep "time:" "$BASELINE_FILE" | head -10)

## SIMD Optimization Targets

Based on these baselines, target 3-5× improvement in:
1. Engine step time
2. Entropy calculation
3. Φ/CI metrics
4. Energy calculations

## Next Steps

1. Implement SIMD optimizations
2. Re-run with: cargo bench --features simd
3. Compare with: cargo benchcmp scalar_baseline simd_optimized
EOF

echo ""
echo -e "${GREEN}✓${NC} Summary saved to: $SUMMARY_FILE"
echo ""
echo "Next steps:"
echo "  1. Review baseline metrics"
echo "  2. Implement SIMD optimizations"
echo "  3. Re-benchmark with: cargo bench --features simd"
echo "  4. Compare: cargo benchcmp scalar_baseline simd_optimized"
echo ""
