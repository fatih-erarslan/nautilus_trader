#!/bin/bash

# NQO Performance Benchmarks Runner
# 
# This script runs comprehensive performance benchmarks for the NQO crate
# and generates detailed HTML reports for analysis.

set -e

echo "🚀 Starting NQO Performance Benchmarks..."
echo "=========================================="

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
echo "   • Optimization Speed - Core algorithm performance"
echo "   • Batch Processing - Neural network batch efficiency"
echo "   • Memory Usage - Resource consumption analysis"
echo "   • Concurrent Performance - Multi-threading scalability"
echo "   • SIMD Acceleration - Vector optimization benefits"
echo "   • Cache Performance - Neural gradient caching"
echo "   • Quantum Complexity - Circuit complexity scaling"
echo "   • Multi-objective - Pareto optimization performance"
echo "   • Production Scenarios - Real-world optimization tasks"
echo "   • Baseline Comparisons - Quantum-neural vs classical"
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
echo "NQO Benchmark Summary - $(date)" > benchmark_results/summary.txt
echo "===============================" >> benchmark_results/summary.txt
echo "" >> benchmark_results/summary.txt
echo "Benchmark Categories:" >> benchmark_results/summary.txt
echo "- Optimization Speed: Single and multi-objective optimization performance" >> benchmark_results/summary.txt
echo "- Batch Processing: Neural network training batch efficiency" >> benchmark_results/summary.txt
echo "- Memory Usage: Peak memory consumption with problem scaling" >> benchmark_results/summary.txt
echo "- Concurrent Performance: Parallel optimization scalability" >> benchmark_results/summary.txt
echo "- SIMD Acceleration: Vector operation performance benefits" >> benchmark_results/summary.txt
echo "- Cache Performance: Neural gradient and optimization caching" >> benchmark_results/summary.txt
echo "- Quantum Complexity: Quantum circuit parameter scaling" >> benchmark_results/summary.txt
echo "- Multi-objective: Pareto-optimal solution finding performance" >> benchmark_results/summary.txt
echo "- Production Scenarios: Real-world optimization use cases" >> benchmark_results/summary.txt
echo "- Baseline Comparisons: Hybrid quantum-neural vs classical methods" >> benchmark_results/summary.txt
echo "" >> benchmark_results/summary.txt
echo "For detailed results, see: target/criterion/performance_benchmarks/report/index.html" >> benchmark_results/summary.txt

echo "✨ Summary saved to: benchmark_results/summary.txt"
echo ""
echo "🎯 Next Steps:"
echo "   1. Review HTML report for optimization insights"
echo "   2. Analyze neural network vs quantum contributions"
echo "   3. Compare multi-objective vs single-objective performance"
echo "   4. Identify scaling bottlenecks"
echo "   5. Monitor performance trends over development cycles"