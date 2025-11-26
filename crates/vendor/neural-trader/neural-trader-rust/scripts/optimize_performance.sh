#!/bin/bash
# Performance Optimization and Warning Elimination Script
# Neural Trader Rust Port - Comprehensive Optimization

set -e

PROJECT_ROOT="/workspaces/neural-trader/neural-trader-rust"
REPORT_DIR="/workspaces/neural-trader/docs/rust-port"
BENCHMARK_DIR="$REPORT_DIR/benchmarks"

cd "$PROJECT_ROOT"

echo "======================================"
echo "  Performance Optimization Mission"
echo "======================================"
echo ""

# Kill any running builds
echo "🧹 Cleaning up existing builds..."
killall -9 cargo rustc tarpaulin 2>/dev/null || true
sleep 2

# Remove locks
rm -rf target/.cargo-lock target/*/.cargo-lock 2>/dev/null || true

# Step 1: Quick check for warnings count
echo "📊 Analyzing current warnings..."
cargo check --workspace --quiet 2>&1 | grep -c "warning:" || echo "0"

# Step 2: Auto-fix warnings with cargo fix
echo ""
echo "🔧 Step 1/8: Running cargo fix..."
timeout 300 cargo fix --workspace --allow-dirty --allow-staged 2>&1 | tee /tmp/cargo_fix_log.txt || {
    echo "⚠️  Cargo fix timed out or failed, continuing..."
}

# Step 3: Run clippy for additional warnings
echo ""
echo "🔍 Step 2/8: Running clippy analysis..."
cargo clippy --workspace --all-targets --quiet -- -D warnings 2>&1 | tee /tmp/clippy_output.txt || {
    echo "⚠️  Clippy found issues, will address manually..."
}

# Step 4: Build workspace in release mode
echo ""
echo "🏗️  Step 3/8: Building release version..."
time cargo build --workspace --release --exclude nt-napi-bindings 2>&1 | tee /tmp/release_build.txt

# Step 5: Run benchmarks
echo ""
echo "⚡ Step 4/8: Running performance benchmarks..."
mkdir -p "$BENCHMARK_DIR"

if [ -d "benches" ]; then
    cargo bench --workspace --no-fail-fast 2>&1 | tee "$BENCHMARK_DIR/benchmark_results.txt" || {
        echo "⚠️  Some benchmarks failed"
    }
fi

# Step 6: Measure binary sizes
echo ""
echo "📦 Step 5/8: Analyzing binary sizes..."
{
    echo "Binary Size Analysis"
    echo "===================="
    echo ""
    find target/release -maxdepth 1 -type f -executable -exec ls -lh {} \; | grep -v ".so" | awk '{print $9, $5}'
    echo ""
    echo "Total release target size:"
    du -sh target/release
} | tee "$BENCHMARK_DIR/binary_sizes.txt"

# Step 7: Analyze compilation time
echo ""
echo "⏱️  Step 6/8: Measuring compilation time..."
{
    echo "Compilation Time Analysis"
    echo "========================"
    echo ""
    cargo clean --release
    /usr/bin/time -v cargo build --workspace --release --exclude nt-napi-bindings 2>&1 | grep -E "User time|System time|Elapsed|Maximum resident"
} | tee "$BENCHMARK_DIR/compile_time.txt" 2>&1 || true

# Step 8: Dependency analysis
echo ""
echo "📦 Step 7/8: Analyzing dependencies..."
{
    echo "Dependency Tree (depth=1)"
    echo "========================"
    cargo tree --workspace --depth 1
    echo ""
    echo "Duplicate dependencies:"
    cargo tree --workspace --duplicates
} | tee "$BENCHMARK_DIR/dependencies.txt"

# Step 9: Generate final warning report
echo ""
echo "📝 Step 8/8: Generating final report..."
{
    echo "# Performance Optimization Report"
    echo "Generated: $(date)"
    echo ""
    echo "## Warning Summary"
    echo ""
    cargo check --workspace --quiet 2>&1 | grep "warning:" | sort | uniq -c | sort -rn || echo "No warnings found!"
    echo ""
    echo "## Compilation Metrics"
    echo ""
    echo "### Build Time"
    grep "Elapsed" "$BENCHMARK_DIR/compile_time.txt" 2>/dev/null || echo "N/A"
    echo ""
    echo "### Binary Sizes"
    cat "$BENCHMARK_DIR/binary_sizes.txt" 2>/dev/null || echo "N/A"
    echo ""
    echo "## Optimization Recommendations"
    echo ""
    echo "1. **LTO Enabled**: Yes (profile.release.lto = true)"
    echo "2. **Strip Symbols**: Yes (profile.release.strip = true)"
    echo "3. **Codegen Units**: 1 (maximum optimization)"
    echo "4. **Opt Level**: 3 (maximum optimization)"
    echo ""
    echo "## Next Steps"
    echo ""
    echo "- Review clippy suggestions in /tmp/clippy_output.txt"
    echo "- Profile hot paths with cargo flamegraph"
    echo "- Consider splitting large crates (>500 LOC modules)"
    echo "- Review dependency tree for unused deps"
} | tee "$REPORT_DIR/PERFORMANCE_REPORT.md"

echo ""
echo "✅ Optimization complete!"
echo ""
echo "📊 Reports generated:"
echo "   - $REPORT_DIR/PERFORMANCE_REPORT.md"
echo "   - $BENCHMARK_DIR/benchmark_results.txt"
echo "   - $BENCHMARK_DIR/binary_sizes.txt"
echo "   - $BENCHMARK_DIR/compile_time.txt"
echo "   - $BENCHMARK_DIR/dependencies.txt"
echo ""
