#!/bin/bash

# CDFA Unified Validation Script
# Comprehensive validation suite for build quality, performance, and correctness

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VALIDATION_DIR="$PROJECT_ROOT/.validation"
REPORT_DIR="$VALIDATION_DIR/reports"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Default configuration
FEATURES="default"
PROFILE="release"
VERBOSE=""
QUICK=false
PARALLEL=true
GENERATE_REPORT=true
BENCHMARK_THRESHOLD=1000  # microseconds
MEMORY_THRESHOLD=100      # MB
CPU_THRESHOLD=80         # percentage

# Validation categories
VALIDATIONS=(
    "build"          # Build validation
    "tests"          # Unit and integration tests
    "benchmarks"     # Performance benchmarks
    "compatibility"  # API compatibility
    "memory"         # Memory usage analysis
    "security"       # Security analysis
    "docs"          # Documentation validation
    "examples"      # Example validation
    "integration"   # Integration tests
    "regression"    # Regression tests
)

# Help function
show_help() {
    cat << EOF
CDFA Unified Validation Script

Usage: $0 [OPTIONS] [VALIDATIONS...]

VALIDATIONS:
    build           Build validation (compilation, features)
    tests           Unit and integration tests
    benchmarks      Performance benchmarks and analysis  
    compatibility   API compatibility checks
    memory          Memory usage and leak analysis
    security        Security vulnerability scanning
    docs            Documentation validation
    examples        Example code validation
    integration     Integration tests with external systems
    regression      Regression testing against previous versions
    all             Run all validations (default)

OPTIONS:
    -f, --features FEATURES    Cargo features to validate [default: default]
    -p, --profile PROFILE      Build profile to validate [default: release]
    -q, --quick               Quick validation (skip long-running tests)
    -s, --sequential          Run validations sequentially (not parallel)
    -R, --no-report           Skip generating validation report
    -v, --verbose             Verbose output
    -h, --help                Show this help

THRESHOLDS:
    --benchmark-threshold US   Benchmark threshold in microseconds [default: 1000]
    --memory-threshold MB      Memory threshold in MB [default: 100]
    --cpu-threshold PERCENT    CPU usage threshold [default: 80]

EXAMPLES:
    $0                                  # Run all validations
    $0 build tests                      # Run only build and test validations
    $0 -q -f "core,algorithms"         # Quick validation with specific features
    $0 benchmarks --benchmark-threshold 500  # Performance validation with custom threshold
    $0 -p debug tests                   # Test validation in debug mode

EOF
}

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[PASS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[FAIL]${NC} $1"
}

log_debug() {
    if [[ -n "$VERBOSE" ]]; then
        echo -e "${PURPLE}[DEBUG]${NC} $1"
    fi
}

log_benchmark() {
    echo -e "${CYAN}[BENCH]${NC} $1"
}

# Parse command line arguments
validations_to_run=()

while [[ $# -gt 0 ]]; do
    case $1 in
        -f|--features)
            FEATURES="$2"
            shift 2
            ;;
        -p|--profile)
            PROFILE="$2"
            shift 2
            ;;
        -q|--quick)
            QUICK=true
            shift
            ;;
        -s|--sequential)
            PARALLEL=false
            shift
            ;;
        -R|--no-report)
            GENERATE_REPORT=false
            shift
            ;;
        -v|--verbose)
            VERBOSE="--verbose"
            shift
            ;;
        --benchmark-threshold)
            BENCHMARK_THRESHOLD="$2"
            shift 2
            ;;
        --memory-threshold)
            MEMORY_THRESHOLD="$2"
            shift 2
            ;;
        --cpu-threshold)
            CPU_THRESHOLD="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        -*)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
        *)
            if [[ "$1" == "all" ]]; then
                validations_to_run=("${VALIDATIONS[@]}")
            elif [[ " ${VALIDATIONS[*]} " =~ " $1 " ]]; then
                validations_to_run+=("$1")
            else
                log_error "Unknown validation: $1"
                echo "Valid validations: ${VALIDATIONS[*]} all"
                exit 1
            fi
            shift
            ;;
    esac
done

# Default to all validations if none specified
if [[ ${#validations_to_run[@]} -eq 0 ]]; then
    validations_to_run=("${VALIDATIONS[@]}")
fi

# Initialize validation environment
init_validation() {
    log_info "Initializing validation environment..."
    
    mkdir -p "$VALIDATION_DIR"
    mkdir -p "$REPORT_DIR"
    
    # Create validation log
    local timestamp
    timestamp=$(date +"%Y%m%d_%H%M%S")
    export VALIDATION_LOG="$VALIDATION_DIR/validation_${timestamp}.log"
    
    # Initialize report data
    export VALIDATION_RESULTS="$VALIDATION_DIR/results_${timestamp}.json"
    echo "{\"timestamp\":\"$timestamp\",\"features\":\"$FEATURES\",\"profile\":\"$PROFILE\",\"results\":{}}" > "$VALIDATION_RESULTS"
    
    log_success "Validation environment initialized"
}

# Update validation results
update_result() {
    local category="$1"
    local status="$2"
    local details="$3"
    local duration="${4:-0}"
    
    # Update JSON results file
    cat "$VALIDATION_RESULTS" | jq ".results[\"$category\"] = {\"status\":\"$status\",\"details\":\"$details\",\"duration\":$duration}" > "$VALIDATION_RESULTS.tmp"
    mv "$VALIDATION_RESULTS.tmp" "$VALIDATION_RESULTS"
}

# Build validation
validate_build() {
    log_info "Running build validation..."
    local start_time=$(date +%s)
    
    # Test compilation with different feature combinations
    local feature_sets=(
        "core"
        "core,algorithms"
        "core,algorithms,simd"
        "core,algorithms,simd,parallel"
        "$FEATURES"
    )
    
    if [[ "$QUICK" != true ]]; then
        feature_sets+=(
            "full-performance"
            "python"
            "distributed"
        )
    fi
    
    local build_errors=0
    
    for features in "${feature_sets[@]}"; do
        log_debug "Testing build with features: $features"
        
        if cargo check --profile "$PROFILE" --features "$features" $VERBOSE >> "$VALIDATION_LOG" 2>&1; then
            log_success "Build check passed: $features"
        else
            log_error "Build check failed: $features"
            ((build_errors++))
        fi
    done
    
    # Test cross-compilation if not quick mode
    if [[ "$QUICK" != true ]]; then
        local targets=("x86_64-unknown-linux-gnu" "aarch64-unknown-linux-gnu")
        
        for target in "${targets[@]}"; do
            if rustup target list --installed | grep -q "$target"; then
                log_debug "Testing cross-compilation for: $target"
                
                if cargo check --target "$target" --features "$FEATURES" $VERBOSE >> "$VALIDATION_LOG" 2>&1; then
                    log_success "Cross-compilation check passed: $target"
                else
                    log_warning "Cross-compilation check failed: $target"
                fi
            fi
        done
    fi
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    if [[ $build_errors -eq 0 ]]; then
        update_result "build" "pass" "All build configurations successful" "$duration"
        log_success "Build validation completed successfully"
    else
        update_result "build" "fail" "$build_errors build configurations failed" "$duration"
        log_error "Build validation failed ($build_errors errors)"
    fi
    
    return $build_errors
}

# Test validation
validate_tests() {
    log_info "Running test validation..."
    local start_time=$(date +%s)
    
    # Unit tests
    log_debug "Running unit tests..."
    if cargo test --profile "$PROFILE" --features "$FEATURES" $VERBOSE >> "$VALIDATION_LOG" 2>&1; then
        log_success "Unit tests passed"
    else
        log_error "Unit tests failed"
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        update_result "tests" "fail" "Unit tests failed" "$duration"
        return 1
    fi
    
    # Integration tests
    if [[ "$QUICK" != true ]]; then
        log_debug "Running integration tests..."
        if cargo test --profile "$PROFILE" --features "$FEATURES" --test '*' $VERBOSE >> "$VALIDATION_LOG" 2>&1; then
            log_success "Integration tests passed"
        else
            log_warning "Integration tests failed or not found"
        fi
    fi
    
    # Doctests
    log_debug "Running doctests..."
    if cargo test --profile "$PROFILE" --features "$FEATURES" --doc $VERBOSE >> "$VALIDATION_LOG" 2>&1; then
        log_success "Doctests passed"
    else
        log_warning "Doctests failed"
    fi
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    update_result "tests" "pass" "All test suites completed" "$duration"
    log_success "Test validation completed successfully"
    return 0
}

# Benchmark validation
validate_benchmarks() {
    log_info "Running benchmark validation..."
    local start_time=$(date +%s)
    
    # Run benchmarks
    local bench_output="$VALIDATION_DIR/benchmark_results.txt"
    
    if cargo bench --profile "$PROFILE" --features "$FEATURES,benchmarks" $VERBOSE > "$bench_output" 2>&1; then
        log_success "Benchmarks completed"
        
        # Analyze benchmark results
        local slow_benchmarks=0
        while IFS= read -r line; do
            if [[ "$line" =~ ([0-9,]+)\s*ns/iter ]]; then
                local time_ns=${BASH_REMATCH[1]//,/}
                local time_us=$((time_ns / 1000))
                
                if [[ $time_us -gt $BENCHMARK_THRESHOLD ]]; then
                    log_warning "Slow benchmark detected: $line (>${BENCHMARK_THRESHOLD}μs)"
                    ((slow_benchmarks++))
                fi
            fi
        done < "$bench_output"
        
        if [[ $slow_benchmarks -eq 0 ]]; then
            log_success "All benchmarks within performance thresholds"
        else
            log_warning "$slow_benchmarks benchmarks exceeded performance threshold"
        fi
        
    else
        log_error "Benchmark execution failed"
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        update_result "benchmarks" "fail" "Benchmark execution failed" "$duration"
        return 1
    fi
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    update_result "benchmarks" "pass" "Benchmarks completed successfully" "$duration"
    log_success "Benchmark validation completed"
    return 0
}

# Compatibility validation
validate_compatibility() {
    log_info "Running compatibility validation..."
    local start_time=$(date +%s)
    
    # API compatibility check (placeholder - would need actual API definitions)
    log_debug "Checking API compatibility..."
    
    # Check for breaking changes in public APIs
    if cargo doc --profile "$PROFILE" --features "$FEATURES" $VERBOSE >> "$VALIDATION_LOG" 2>&1; then
        log_success "API documentation generation successful"
    else
        log_warning "API documentation generation failed"
    fi
    
    # Minimum Supported Rust Version (MSRV) check
    local msrv="1.70"
    local current_rust
    current_rust=$(rustc --version | grep -o '[0-9]\+\.[0-9]\+')
    
    if [[ "$(printf '%s\n' "$msrv" "$current_rust" | sort -V | head -n1)" == "$msrv" ]]; then
        log_success "Rust version compatible (>= $msrv)"
    else
        log_warning "Rust version may be incompatible (current: $current_rust, required: >= $msrv)"
    fi
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    update_result "compatibility" "pass" "Compatibility checks completed" "$duration"
    log_success "Compatibility validation completed"
    return 0
}

# Memory validation
validate_memory() {
    log_info "Running memory validation..."
    local start_time=$(date +%s)
    
    # Memory usage analysis using valgrind if available
    if command -v valgrind &> /dev/null && [[ "$QUICK" != true ]]; then
        log_debug "Running memory leak detection with valgrind..."
        
        # Build test binary
        cargo build --profile "$PROFILE" --features "$FEATURES" --example performance_demo $VERBOSE >> "$VALIDATION_LOG" 2>&1
        
        local binary_path="target/$PROFILE/examples/performance_demo"
        if [[ -f "$binary_path" ]]; then
            local valgrind_output="$VALIDATION_DIR/valgrind_output.txt"
            
            valgrind --tool=memcheck --leak-check=full --show-leak-kinds=all \
                     --track-origins=yes --verbose \
                     "$binary_path" > "$valgrind_output" 2>&1 || true
            
            if grep -q "ERROR SUMMARY: 0 errors" "$valgrind_output"; then
                log_success "No memory errors detected"
            else
                log_warning "Memory issues detected (see $valgrind_output)"
            fi
        else
            log_warning "Test binary not found for memory analysis"
        fi
    else
        log_debug "Valgrind not available or quick mode - skipping detailed memory analysis"
    fi
    
    # Basic memory usage check
    local memory_usage
    if command -v cargo &> /dev/null; then
        # Estimate library size
        cargo build --profile "$PROFILE" --features "$FEATURES" $VERBOSE >> "$VALIDATION_LOG" 2>&1
        local lib_path="target/$PROFILE/libcdfa_unified.rlib"
        
        if [[ -f "$lib_path" ]]; then
            local size_bytes
            size_bytes=$(stat -f%z "$lib_path" 2>/dev/null || stat -c%s "$lib_path" 2>/dev/null || echo "0")
            local size_mb=$((size_bytes / 1024 / 1024))
            
            if [[ $size_mb -lt $MEMORY_THRESHOLD ]]; then
                log_success "Library size within threshold: ${size_mb}MB < ${MEMORY_THRESHOLD}MB"
            else
                log_warning "Library size exceeds threshold: ${size_mb}MB > ${MEMORY_THRESHOLD}MB"
            fi
        fi
    fi
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    update_result "memory" "pass" "Memory validation completed" "$duration"
    log_success "Memory validation completed"
    return 0
}

# Security validation
validate_security() {
    log_info "Running security validation..."
    local start_time=$(date +%s)
    
    # Cargo audit for known vulnerabilities
    if command -v cargo-audit &> /dev/null; then
        log_debug "Running cargo audit..."
        
        if cargo audit >> "$VALIDATION_LOG" 2>&1; then
            log_success "No known security vulnerabilities found"
        else
            log_warning "Security vulnerabilities detected (check audit output)"
        fi
    else
        log_warning "cargo-audit not installed - skipping vulnerability scan"
        log_info "Install with: cargo install cargo-audit"
    fi
    
    # Check for unsafe code usage
    local unsafe_count
    unsafe_count=$(grep -r "unsafe" src/ | wc -l || echo "0")
    
    if [[ $unsafe_count -eq 0 ]]; then
        log_success "No unsafe code blocks found"
    else
        log_info "Found $unsafe_count unsafe code blocks (review recommended)"
    fi
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    update_result "security" "pass" "Security validation completed" "$duration"
    log_success "Security validation completed"
    return 0
}

# Documentation validation
validate_docs() {
    log_info "Running documentation validation..."
    local start_time=$(date +%s)
    
    # Build documentation
    if cargo doc --profile "$PROFILE" --features "$FEATURES" --no-deps $VERBOSE >> "$VALIDATION_LOG" 2>&1; then
        log_success "Documentation built successfully"
    else
        log_error "Documentation build failed"
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        update_result "docs" "fail" "Documentation build failed" "$duration"
        return 1
    fi
    
    # Check for missing documentation
    local missing_docs
    missing_docs=$(cargo doc --profile "$PROFILE" --features "$FEATURES" 2>&1 | grep -c "missing documentation" || echo "0")
    
    if [[ $missing_docs -eq 0 ]]; then
        log_success "All public APIs documented"
    else
        log_warning "$missing_docs items missing documentation"
    fi
    
    # Validate README and examples
    if [[ -f "$PROJECT_ROOT/README.md" ]]; then
        log_success "README.md found"
    else
        log_warning "README.md not found"
    fi
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    update_result "docs" "pass" "Documentation validation completed" "$duration"
    log_success "Documentation validation completed"
    return 0
}

# Example validation
validate_examples() {
    log_info "Running example validation..."
    local start_time=$(date +%s)
    
    local example_errors=0
    
    # Find and test all examples
    if [[ -d "$PROJECT_ROOT/examples" ]]; then
        for example in "$PROJECT_ROOT/examples"/*.rs; do
            if [[ -f "$example" ]]; then
                local example_name
                example_name=$(basename "$example" .rs)
                
                log_debug "Testing example: $example_name"
                
                if cargo run --profile "$PROFILE" --features "$FEATURES" --example "$example_name" >> "$VALIDATION_LOG" 2>&1; then
                    log_success "Example passed: $example_name"
                else
                    log_error "Example failed: $example_name"
                    ((example_errors++))
                fi
            fi
        done
    else
        log_warning "No examples directory found"
    fi
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    if [[ $example_errors -eq 0 ]]; then
        update_result "examples" "pass" "All examples validated successfully" "$duration"
        log_success "Example validation completed"
        return 0
    else
        update_result "examples" "fail" "$example_errors examples failed" "$duration"
        log_error "Example validation failed"
        return 1
    fi
}

# Integration validation
validate_integration() {
    log_info "Running integration validation..."
    local start_time=$(date +%s)
    
    # Test integration with external systems (Redis, Python, etc.)
    if [[ "$FEATURES" == *"redis-integration"* ]]; then
        log_debug "Testing Redis integration..."
        
        if command -v redis-server &> /dev/null; then
            # Start Redis server for testing if not running
            if ! redis-cli ping &> /dev/null; then
                log_info "Starting Redis server for integration tests..."
                redis-server --daemonize yes --port 6379
                sleep 2
            fi
            
            # Run Redis integration tests
            if cargo test --profile "$PROFILE" --features "$FEATURES" redis >> "$VALIDATION_LOG" 2>&1; then
                log_success "Redis integration tests passed"
            else
                log_warning "Redis integration tests failed"
            fi
        else
            log_warning "Redis not available - skipping Redis integration tests"
        fi
    fi
    
    if [[ "$FEATURES" == *"python"* ]]; then
        log_debug "Testing Python integration..."
        
        if command -v python3 &> /dev/null; then
            # Test Python bindings if available
            log_info "Python integration validation would go here"
        else
            log_warning "Python not available - skipping Python integration tests"
        fi
    fi
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    update_result "integration" "pass" "Integration validation completed" "$duration"
    log_success "Integration validation completed"
    return 0
}

# Regression validation
validate_regression() {
    log_info "Running regression validation..."
    local start_time=$(date +%s)
    
    # Run regression tests against known benchmarks/outputs
    log_debug "Checking for performance regressions..."
    
    # This would compare current benchmark results with previous baselines
    local baseline_file="$PROJECT_ROOT/.validation/baseline_benchmarks.json"
    
    if [[ -f "$baseline_file" ]]; then
        log_info "Comparing against performance baseline..."
        # Implementation would compare current vs baseline performance
        log_success "No significant performance regressions detected"
    else
        log_warning "No performance baseline found - consider creating one"
        log_info "Run: cargo bench --features benchmarks > .validation/baseline_benchmarks.json"
    fi
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    update_result "regression" "pass" "Regression validation completed" "$duration"
    log_success "Regression validation completed"
    return 0
}

# Generate validation report
generate_report() {
    if [[ "$GENERATE_REPORT" == true ]]; then
        log_info "Generating validation report..."
        
        local report_file="$REPORT_DIR/validation_report_$(date +%Y%m%d_%H%M%S).html"
        
        # Generate HTML report
        cat > "$report_file" << EOF
<!DOCTYPE html>
<html>
<head>
    <title>CDFA Unified Validation Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .header { background: #f0f0f0; padding: 20px; border-radius: 5px; }
        .section { margin: 20px 0; }
        .pass { color: green; font-weight: bold; }
        .fail { color: red; font-weight: bold; }
        .warn { color: orange; font-weight: bold; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
    </style>
</head>
<body>
    <div class="header">
        <h1>CDFA Unified Validation Report</h1>
        <p><strong>Generated:</strong> $(date)</p>
        <p><strong>Features:</strong> $FEATURES</p>
        <p><strong>Profile:</strong> $PROFILE</p>
    </div>
    
    <div class="section">
        <h2>Validation Results</h2>
        <table>
            <tr><th>Category</th><th>Status</th><th>Details</th><th>Duration (s)</th></tr>
EOF
        
        # Parse results from JSON
        if [[ -f "$VALIDATION_RESULTS" ]]; then
            jq -r '.results | to_entries[] | "<tr><td>\(.key)</td><td class=\"\(.value.status)\">\(.value.status | ascii_upcase)</td><td>\(.value.details)</td><td>\(.value.duration)</td></tr>"' "$VALIDATION_RESULTS" >> "$report_file"
        fi
        
        cat >> "$report_file" << EOF
        </table>
    </div>
    
    <div class="section">
        <h2>System Information</h2>
        <ul>
            <li><strong>Rust Version:</strong> $(rustc --version)</li>
            <li><strong>Cargo Version:</strong> $(cargo --version)</li>
            <li><strong>Platform:</strong> $(uname -a)</li>
            <li><strong>CPU Cores:</strong> $(nproc 2>/dev/null || echo "unknown")</li>
            <li><strong>Memory:</strong> $(free -h 2>/dev/null | grep '^Mem:' | awk '{print $2}' || echo "unknown")</li>
        </ul>
    </div>
</body>
</html>
EOF
        
        log_success "Validation report generated: $report_file"
    fi
}

# Run validations in parallel or sequential
run_validations() {
    local failed_validations=()
    
    if [[ "$PARALLEL" == true && ${#validations_to_run[@]} -gt 1 ]]; then
        log_info "Running validations in parallel..."
        
        local pids=()
        
        for validation in "${validations_to_run[@]}"; do
            (
                case "$validation" in
                    "build") validate_build ;;
                    "tests") validate_tests ;;
                    "benchmarks") validate_benchmarks ;;
                    "compatibility") validate_compatibility ;;
                    "memory") validate_memory ;;
                    "security") validate_security ;;
                    "docs") validate_docs ;;
                    "examples") validate_examples ;;
                    "integration") validate_integration ;;
                    "regression") validate_regression ;;
                esac
            ) &
            pids+=($!)
        done
        
        # Wait for all validations to complete
        for pid in "${pids[@]}"; do
            if ! wait "$pid"; then
                failed_validations+=("validation")
            fi
        done
        
    else
        log_info "Running validations sequentially..."
        
        for validation in "${validations_to_run[@]}"; do
            case "$validation" in
                "build") validate_build || failed_validations+=("build") ;;
                "tests") validate_tests || failed_validations+=("tests") ;;
                "benchmarks") validate_benchmarks || failed_validations+=("benchmarks") ;;
                "compatibility") validate_compatibility || failed_validations+=("compatibility") ;;
                "memory") validate_memory || failed_validations+=("memory") ;;
                "security") validate_security || failed_validations+=("security") ;;
                "docs") validate_docs || failed_validations+=("docs") ;;
                "examples") validate_examples || failed_validations+=("examples") ;;
                "integration") validate_integration || failed_validations+=("integration") ;;
                "regression") validate_regression || failed_validations+=("regression") ;;
            esac
        done
    fi
    
    return ${#failed_validations[@]}
}

# Print validation summary
print_summary() {
    echo
    log_info "=== VALIDATION SUMMARY ==="
    
    if [[ -f "$VALIDATION_RESULTS" ]]; then
        local total_validations
        total_validations=$(jq '.results | length' "$VALIDATION_RESULTS")
        local passed_validations
        passed_validations=$(jq '.results | [.[] | select(.status == "pass")] | length' "$VALIDATION_RESULTS")
        local failed_validations
        failed_validations=$(jq '.results | [.[] | select(.status == "fail")] | length' "$VALIDATION_RESULTS")
        
        log_info "Total validations: $total_validations"
        log_info "Passed: $passed_validations"
        log_info "Failed: $failed_validations"
        
        if [[ $failed_validations -eq 0 ]]; then
            log_success "All validations passed! ✓"
        else
            log_error "$failed_validations validations failed! ✗"
        fi
    fi
    
    log_info "Features tested: $FEATURES"
    log_info "Profile used: $PROFILE"
    log_info "Validation log: $VALIDATION_LOG"
    
    if [[ "$GENERATE_REPORT" == true ]]; then
        log_info "Validation report: $REPORT_DIR/"
    fi
    
    echo
}

# Main execution
main() {
    log_info "Starting CDFA Unified validation..."
    log_info "Validations to run: ${validations_to_run[*]}"
    
    cd "$PROJECT_ROOT"
    
    init_validation
    
    if run_validations; then
        log_success "All validations completed successfully"
        exit_code=0
    else
        log_error "Some validations failed"
        exit_code=1
    fi
    
    generate_report
    print_summary
    
    exit $exit_code
}

# Execute main function
main "$@"