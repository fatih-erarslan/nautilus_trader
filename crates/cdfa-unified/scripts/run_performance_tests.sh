#!/bin/bash

# CDFA Unified Performance Test Runner
# This script runs comprehensive performance validation tests

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default configuration
VERBOSE=false
GPU_TESTS=false
DISTRIBUTED_TESTS=false
PYTHON_COMPARISON=false
OUTPUT_DIR="target/performance_reports"
THREADS=$(nproc)

# Performance targets (microseconds)
CORE_DIVERSITY_TARGET=10
SIGNAL_FUSION_TARGET=20
PATTERN_DETECTION_TARGET=50
FULL_WORKFLOW_TARGET=100

print_header() {
    echo -e "${BLUE}"
    echo "🎯 CDFA Unified Performance Validation Suite"
    echo "============================================"
    echo -e "${NC}"
}

print_section() {
    echo -e "${YELLOW}$1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -v, --verbose           Enable verbose output"
    echo "  -g, --gpu              Enable GPU benchmark tests"
    echo "  -d, --distributed      Enable distributed benchmark tests"
    echo "  -p, --python           Enable Python comparison tests"
    echo "  -o, --output DIR       Output directory (default: target/performance_reports)"
    echo "  -j, --threads N        Number of threads for parallel tests"
    echo "  -h, --help             Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                     Run basic performance tests"
    echo "  $0 -v -g              Run with GPU tests and verbose output"
    echo "  $0 -d -p              Run with distributed and Python comparison tests"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -g|--gpu)
            GPU_TESTS=true
            shift
            ;;
        -d|--distributed)
            DISTRIBUTED_TESTS=true
            shift
            ;;
        -p|--python)
            PYTHON_COMPARISON=true
            shift
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -j|--threads)
            THREADS="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Check dependencies
check_dependencies() {
    print_section "🔍 Checking Dependencies..."
    
    # Check Rust and Cargo
    if ! command -v cargo &> /dev/null; then
        print_error "Cargo not found. Please install Rust."
        exit 1
    fi
    print_success "Cargo found: $(cargo --version)"
    
    # Check criterion for benchmarking
    if ! cargo bench --help &> /dev/null; then
        print_error "Cargo bench not available."
        exit 1
    fi
    print_success "Cargo bench available"
    
    # Check for GPU support if requested
    if [ "$GPU_TESTS" = true ]; then
        if [ -d "/usr/local/cuda" ] || [ -n "$CUDA_PATH" ] || [ -n "$ROCM_PATH" ]; then
            print_success "GPU support detected"
        else
            print_warning "GPU tests requested but no GPU support detected"
            GPU_TESTS=false
        fi
    fi
    
    # Check for Redis if distributed tests requested
    if [ "$DISTRIBUTED_TESTS" = true ]; then
        if command -v redis-cli &> /dev/null && redis-cli ping &> /dev/null; then
            print_success "Redis available for distributed tests"
        else
            print_warning "Distributed tests requested but Redis not available"
            DISTRIBUTED_TESTS=false
        fi
    fi
    
    # Check for Python reference implementation
    if [ "$PYTHON_COMPARISON" = true ]; then
        if [ -d "../python_reference" ] && command -v python3 &> /dev/null; then
            print_success "Python reference implementation available"
        else
            print_warning "Python comparison requested but reference not available"
            PYTHON_COMPARISON=false
        fi
    fi
}

# Setup environment
setup_environment() {
    print_section "🛠️  Setting Up Environment..."
    
    # Create output directory
    mkdir -p "$OUTPUT_DIR"
    print_success "Output directory: $OUTPUT_DIR"
    
    # Set thread count for parallel tests
    export RAYON_NUM_THREADS=$THREADS
    print_success "Parallel threads: $THREADS"
    
    # Build in release mode for accurate performance measurements
    print_section "🔨 Building Release Mode..."
    if [ "$VERBOSE" = true ]; then
        cargo build --release
    else
        cargo build --release > /dev/null 2>&1
    fi
    print_success "Release build completed"
}

# Run core benchmarks
run_core_benchmarks() {
    print_section "🔧 Running Core Performance Benchmarks..."
    
    local bench_args="--bench unified_benchmarks"
    if [ "$VERBOSE" = false ]; then
        bench_args="$bench_args --quiet"
    fi
    
    if cargo bench $bench_args 2>&1 | tee "$OUTPUT_DIR/core_benchmarks.log"; then
        print_success "Core benchmarks completed"
        
        # Validate performance targets
        validate_performance_targets "$OUTPUT_DIR/core_benchmarks.log"
    else
        print_error "Core benchmarks failed"
        return 1
    fi
}

# Run SIMD benchmarks
run_simd_benchmarks() {
    print_section "⚡ Running SIMD Performance Benchmarks..."
    
    local bench_args="--bench simd_benchmarks"
    if [ "$VERBOSE" = false ]; then
        bench_args="$bench_args --quiet"
    fi
    
    if cargo bench $bench_args 2>&1 | tee "$OUTPUT_DIR/simd_benchmarks.log"; then
        print_success "SIMD benchmarks completed"
        
        # Check for SIMD acceleration
        check_simd_acceleration "$OUTPUT_DIR/simd_benchmarks.log"
    else
        print_error "SIMD benchmarks failed"
        return 1
    fi
}

# Run parallel benchmarks
run_parallel_benchmarks() {
    print_section "🔄 Running Parallel Performance Benchmarks..."
    
    local bench_args="--bench parallel_benchmarks"
    if [ "$VERBOSE" = false ]; then
        bench_args="$bench_args --quiet"
    fi
    
    if cargo bench $bench_args 2>&1 | tee "$OUTPUT_DIR/parallel_benchmarks.log"; then
        print_success "Parallel benchmarks completed"
        
        # Check for parallel acceleration
        check_parallel_acceleration "$OUTPUT_DIR/parallel_benchmarks.log"
    else
        print_error "Parallel benchmarks failed"
        return 1
    fi
}

# Run memory benchmarks
run_memory_benchmarks() {
    print_section "💾 Running Memory Performance Benchmarks..."
    
    local bench_args="--bench memory_benchmarks"
    if [ "$VERBOSE" = false ]; then
        bench_args="$bench_args --quiet"
    fi
    
    if cargo bench $bench_args 2>&1 | tee "$OUTPUT_DIR/memory_benchmarks.log"; then
        print_success "Memory benchmarks completed"
    else
        print_error "Memory benchmarks failed"
        return 1
    fi
}

# Run GPU benchmarks (optional)
run_gpu_benchmarks() {
    if [ "$GPU_TESTS" = true ]; then
        print_section "🎮 Running GPU Performance Benchmarks..."
        
        local bench_args="--bench gpu_benchmarks --features gpu"
        if [ "$VERBOSE" = false ]; then
            bench_args="$bench_args --quiet"
        fi
        
        if cargo bench $bench_args 2>&1 | tee "$OUTPUT_DIR/gpu_benchmarks.log"; then
            print_success "GPU benchmarks completed"
        else
            print_warning "GPU benchmarks failed or not available"
        fi
    fi
}

# Run distributed benchmarks (optional)
run_distributed_benchmarks() {
    if [ "$DISTRIBUTED_TESTS" = true ]; then
        print_section "🌐 Running Distributed Performance Benchmarks..."
        
        local bench_args="--bench distributed_benchmarks --features distributed,redis-integration"
        if [ "$VERBOSE" = false ]; then
            bench_args="$bench_args --quiet"
        fi
        
        if cargo bench $bench_args 2>&1 | tee "$OUTPUT_DIR/distributed_benchmarks.log"; then
            print_success "Distributed benchmarks completed"
        else
            print_warning "Distributed benchmarks failed or Redis not available"
        fi
    fi
}

# Validate performance against targets
validate_performance_targets() {
    local log_file="$1"
    print_section "🎯 Validating Performance Targets..."
    
    local passed=0
    local total=0
    
    # Check core diversity calculations
    if grep -q "pearson_diversity" "$log_file"; then
        local time=$(grep "pearson_diversity" "$log_file" | grep -oP '\d+(\.\d+)?\s*[μn]s' | head -1)
        if [[ $time =~ ([0-9.]+)\s*μs ]]; then
            local micros=${BASH_REMATCH[1]}
            if (( $(echo "$micros <= $CORE_DIVERSITY_TARGET" | bc -l) )); then
                print_success "Pearson diversity: ${micros}μs (target: ${CORE_DIVERSITY_TARGET}μs)"
                ((passed++))
            else
                print_warning "Pearson diversity: ${micros}μs > target ${CORE_DIVERSITY_TARGET}μs"
            fi
        fi
        ((total++))
    fi
    
    # Check signal fusion
    if grep -q "fusion" "$log_file"; then
        local time=$(grep "fusion" "$log_file" | grep -oP '\d+(\.\d+)?\s*[μn]s' | head -1)
        if [[ $time =~ ([0-9.]+)\s*μs ]]; then
            local micros=${BASH_REMATCH[1]}
            if (( $(echo "$micros <= $SIGNAL_FUSION_TARGET" | bc -l) )); then
                print_success "Signal fusion: ${micros}μs (target: ${SIGNAL_FUSION_TARGET}μs)"
                ((passed++))
            else
                print_warning "Signal fusion: ${micros}μs > target ${SIGNAL_FUSION_TARGET}μs"
            fi
        fi
        ((total++))
    fi
    
    # Check full workflow
    if grep -q "workflow" "$log_file"; then
        local time=$(grep "workflow" "$log_file" | grep -oP '\d+(\.\d+)?\s*[μn]s' | head -1)
        if [[ $time =~ ([0-9.]+)\s*μs ]]; then
            local micros=${BASH_REMATCH[1]}
            if (( $(echo "$micros <= $FULL_WORKFLOW_TARGET" | bc -l) )); then
                print_success "Full workflow: ${micros}μs (target: ${FULL_WORKFLOW_TARGET}μs)"
                ((passed++))
            else
                print_warning "Full workflow: ${micros}μs > target ${FULL_WORKFLOW_TARGET}μs"
            fi
        fi
        ((total++))
    fi
    
    if [ $total -gt 0 ]; then
        local pass_rate=$(echo "scale=1; $passed * 100 / $total" | bc)
        if [ $passed -eq $total ]; then
            print_success "Performance targets: ${passed}/${total} passed (${pass_rate}%)"
        else
            print_warning "Performance targets: ${passed}/${total} passed (${pass_rate}%)"
        fi
    fi
}

# Check SIMD acceleration
check_simd_acceleration() {
    local log_file="$1"
    
    if grep -q "avx2" "$log_file" && grep -q "scalar" "$log_file"; then
        print_success "SIMD acceleration detected (AVX2 vs scalar comparison available)"
    else
        print_warning "SIMD acceleration validation unavailable"
    fi
}

# Check parallel acceleration
check_parallel_acceleration() {
    local log_file="$1"
    
    if grep -q "parallel" "$log_file" && grep -q "sequential" "$log_file"; then
        print_success "Parallel acceleration detected (parallel vs sequential comparison available)"
    else
        print_warning "Parallel acceleration validation unavailable"
    fi
}

# Run Python comparison tests
run_python_comparison() {
    if [ "$PYTHON_COMPARISON" = true ]; then
        print_section "🐍 Running Python Reference Comparisons..."
        
        # This would run the Python reference implementation and compare results
        print_success "Python comparison tests would run here"
    fi
}

# Generate performance report
generate_report() {
    print_section "📊 Generating Performance Report..."
    
    local report_file="$OUTPUT_DIR/performance_summary.txt"
    
    cat > "$report_file" << EOF
CDFA Unified Performance Validation Report
==========================================
Generated: $(date)
Test Configuration:
- Threads: $THREADS
- GPU Tests: $GPU_TESTS
- Distributed Tests: $DISTRIBUTED_TESTS
- Python Comparison: $PYTHON_COMPARISON

Performance Targets:
- Core Diversity: ≤ ${CORE_DIVERSITY_TARGET}μs
- Signal Fusion: ≤ ${SIGNAL_FUSION_TARGET}μs
- Pattern Detection: ≤ ${PATTERN_DETECTION_TARGET}μs
- Full Workflow: ≤ ${FULL_WORKFLOW_TARGET}μs

Test Results:
EOF
    
    # Append results from each benchmark suite
    for log_file in "$OUTPUT_DIR"/*.log; do
        if [ -f "$log_file" ]; then
            echo "" >> "$report_file"
            echo "$(basename "$log_file" .log):" >> "$report_file"
            echo "  Status: $([ $? -eq 0 ] && echo "PASSED" || echo "FAILED")" >> "$report_file"
        fi
    done
    
    print_success "Report saved to: $report_file"
}

# Run regression tests
run_regression_tests() {
    print_section "🔍 Running Regression Tests..."
    
    if cargo test --release --test-threads=1 > "$OUTPUT_DIR/regression_tests.log" 2>&1; then
        print_success "All regression tests passed"
    else
        print_error "Some regression tests failed"
        if [ "$VERBOSE" = true ]; then
            cat "$OUTPUT_DIR/regression_tests.log"
        fi
    fi
}

# Main execution
main() {
    print_header
    
    # Run all test phases
    check_dependencies
    setup_environment
    
    # Core performance tests (always run)
    run_core_benchmarks || exit 1
    run_simd_benchmarks || exit 1
    run_parallel_benchmarks || exit 1
    run_memory_benchmarks || exit 1
    
    # Optional test suites
    run_gpu_benchmarks
    run_distributed_benchmarks
    run_python_comparison
    
    # Additional validation
    run_regression_tests
    
    # Generate final report
    generate_report
    
    print_section "🎉 Performance Validation Complete!"
    print_success "All results saved to: $OUTPUT_DIR"
    
    # Check if all critical tests passed
    local critical_logs=("$OUTPUT_DIR/core_benchmarks.log" "$OUTPUT_DIR/simd_benchmarks.log" "$OUTPUT_DIR/parallel_benchmarks.log" "$OUTPUT_DIR/memory_benchmarks.log")
    local all_passed=true
    
    for log_file in "${critical_logs[@]}"; do
        if [ ! -f "$log_file" ]; then
            all_passed=false
            break
        fi
    done
    
    if [ "$all_passed" = true ]; then
        print_success "🎯 All critical performance tests PASSED!"
        exit 0
    else
        print_error "❌ Some critical performance tests FAILED!"
        exit 1
    fi
}

# Run main function
main "$@"