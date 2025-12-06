#!/bin/bash

# IQAD Performance Benchmarks Runner
# 
# This script runs comprehensive performance benchmarks for the IQAD crate
# and generates detailed HTML reports for analysis.

set -e

echo "🚀 Starting IQAD Performance Benchmarks..."
echo "============================================"

# Check if criterion is available
if ! command -v cargo &> /dev/null; then
    echo "❌ Cargo not found. Please install Rust."
    exit 1
fi

# Create benchmark results directory
mkdir -p benchmark_results

# Set environment variables for optimal performance
export RUSTFLAGS="-C target-cpu=native -C opt-level=3"
export CARGO_PROFILE_BENCH_LTO=true
export CARGO_PROFILE_BENCH_CODEGEN_UNITS=1

echo "📊 Running benchmarks with optimizations..."

# Run benchmarks with HTML output
cargo bench --bench performance_benchmarks -- --output-format html

echo ""
echo "✅ Benchmarks completed!"
echo ""
echo "📋 Benchmark Categories Tested:"
echo "   • Detection Speed - Core algorithm performance"
echo "   • Batch Processing - Throughput measurements"
echo "   • Memory Usage - Resource consumption analysis"
echo "   • Concurrent Performance - Multi-threading efficiency"
echo "   • SIMD Acceleration - Hardware optimization benefits"
echo "   • Cache Performance - Caching system effectiveness"
echo "   • Quantum Complexity - Circuit complexity scaling"
echo "   • Production Scenarios - Real-world use cases"
echo "   • Baseline Comparisons - Classical vs quantum-immune"
echo ""
echo "📈 Results saved to:"
echo "   • HTML Report: target/criterion/performance_benchmarks/report/index.html"
echo "   • Raw Data: target/criterion/"
echo ""
echo "🔍 To view results:"
echo "   • Open target/criterion/performance_benchmarks/report/index.html in browser"
echo "   • Or run: python -m http.server 8000 --directory target/criterion/performance_benchmarks/report/"
echo ""

# Generate summary report
echo "📝 Generating summary report..."
echo "IQAD Benchmark Summary - $(date)" > benchmark_results/summary.txt
echo "================================" >> benchmark_results/summary.txt
echo "" >> benchmark_results/summary.txt
echo "Benchmark Categories:" >> benchmark_results/summary.txt
echo "- Detection Speed: Single-threaded anomaly detection performance" >> benchmark_results/summary.txt
echo "- Batch Processing: Throughput with different batch sizes" >> benchmark_results/summary.txt
echo "- Memory Usage: Peak memory consumption scaling" >> benchmark_results/summary.txt
echo "- Concurrent Performance: Multi-threading scalability" >> benchmark_results/summary.txt
echo "- SIMD Acceleration: AVX-512 performance benefits" >> benchmark_results/summary.txt
echo "- Cache Performance: Caching effectiveness and hit ratios" >> benchmark_results/summary.txt
echo "- Quantum Complexity: Quantum circuit scaling behavior" >> benchmark_results/summary.txt
echo "- Production Scenarios: Real-world use case performance" >> benchmark_results/summary.txt
echo "- Baseline Comparisons: Quantum-immune vs classical methods" >> benchmark_results/summary.txt
echo "" >> benchmark_results/summary.txt
echo "For detailed results, see: target/criterion/performance_benchmarks/report/index.html" >> benchmark_results/summary.txt

echo "✨ Summary saved to: benchmark_results/summary.txt"
echo ""
echo "🎯 Next Steps:"
echo "   1. Review HTML report for performance insights"
echo "   2. Identify optimization opportunities"
echo "   3. Compare against baseline metrics"
echo "   4. Monitor performance regression over time"