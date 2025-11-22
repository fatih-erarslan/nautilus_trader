#!/bin/bash

# Comprehensive test runner script for the autopoiesis framework
# Provides automated testing with coverage reporting and performance analysis

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DEFAULT_TIMEOUT=300
DEFAULT_PROPTEST_CASES=1000
TEST_RESULTS_DIR="target/test-results"
COVERAGE_DIR="target/coverage"

# Function to print colored output
print_status() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

print_header() {
    echo
    print_status $BLUE "=================================================================================="
    print_status $BLUE "$1"
    print_status $BLUE "=================================================================================="
    echo
}

print_success() {
    print_status $GREEN "✅ $1"
}

print_error() {
    print_status $RED "❌ $1"
}

print_warning() {
    print_status $YELLOW "⚠️  $1"
}

print_info() {
    print_status $BLUE "ℹ️  $1"
}

# Function to show usage
show_usage() {
    cat << EOF
Autopoiesis Test Runner

USAGE:
    ./run_tests.sh [OPTIONS] [TEST_CATEGORY]

TEST CATEGORIES:
    unit                Run unit tests only
    integration         Run integration tests only
    property           Run property-based tests only
    performance        Run performance tests only
    chaos              Run chaos engineering tests only
    benchmarks         Run criterion benchmarks only
    all                Run all tests (default)

OPTIONS:
    -c, --coverage     Generate coverage report
    -v, --verbose      Enable verbose output
    -t, --timeout SEC  Set test timeout in seconds (default: $DEFAULT_TIMEOUT)
    -p, --proptest N   Set property test cases (default: $DEFAULT_PROPTEST_CASES)
    -j, --parallel     Run tests in parallel (default)
    -s, --sequential   Run tests sequentially
    --no-capture       Don't capture test output
    --html             Generate HTML reports
    --junit            Generate JUnit XML for CI
    --clean            Clean before running tests
    --install-deps     Install test dependencies
    -h, --help         Show this help message

EXAMPLES:
    ./run_tests.sh                          # Run all tests
    ./run_tests.sh unit --coverage          # Run unit tests with coverage
    ./run_tests.sh integration --verbose    # Run integration tests verbosely
    ./run_tests.sh benchmarks               # Run performance benchmarks
    ./run_tests.sh --clean --coverage       # Clean build and run with coverage

ENVIRONMENT VARIABLES:
    RUST_LOG           Set logging level (debug, info, warn, error)
    RUST_BACKTRACE     Enable backtraces (0, 1, full)
    PROPTEST_CASES     Override property test case count
    TEST_TIMEOUT       Override default timeout
EOF
}

# Function to check prerequisites
check_prerequisites() {
    print_info "Checking prerequisites..."
    
    # Check Rust installation
    if ! command -v cargo &> /dev/null; then
        print_error "Cargo not found. Please install Rust: https://rustup.rs/"
        exit 1
    fi
    
    # Check if we're in the right directory
    if [[ ! -f "Cargo.toml" ]]; then
        print_error "Cargo.toml not found. Please run from the project root."
        exit 1
    fi
    
    # Create directories
    mkdir -p "$TEST_RESULTS_DIR" "$COVERAGE_DIR"
    
    print_success "Prerequisites checked"
}

# Function to install test dependencies
install_test_dependencies() {
    print_info "Installing test dependencies..."
    
    # Install cargo-tarpaulin for coverage (if not installed)
    if ! command -v cargo-tarpaulin &> /dev/null; then
        print_info "Installing cargo-tarpaulin for coverage reporting..."
        cargo install cargo-tarpaulin || {
            print_warning "Failed to install cargo-tarpaulin. Coverage may not work."
        }
    fi
    
    # Install cargo-criterion for benchmarking (if not installed)
    if ! command -v cargo-criterion &> /dev/null; then
        print_info "Installing cargo-criterion for benchmarking..."
        cargo install cargo-criterion || {
            print_warning "Failed to install cargo-criterion. Benchmarks may not work."
        }
    fi
    
    # Install additional tools
    local tools=("cargo-audit" "cargo-outdated" "cargo-bloat")
    for tool in "${tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            print_info "Installing $tool..."
            cargo install "$tool" || {
                print_warning "Failed to install $tool"
            }
        fi
    done
    
    print_success "Test dependencies checked"
}

# Function to clean build artifacts
clean_build() {
    print_info "Cleaning build artifacts..."
    cargo clean
    rm -rf "$TEST_RESULTS_DIR" "$COVERAGE_DIR"
    mkdir -p "$TEST_RESULTS_DIR" "$COVERAGE_DIR"
    print_success "Build artifacts cleaned"
}

# Function to run unit tests
run_unit_tests() {
    print_header "Running Unit Tests"
    
    local args=("test" "--lib")
    
    if [[ "$VERBOSE" == "true" ]]; then
        args+=("--verbose")
    fi
    
    if [[ "$PARALLEL" == "false" ]]; then
        args+=("--" "--test-threads=1")
    fi
    
    if [[ "$NO_CAPTURE" == "true" ]]; then
        args+=("--" "--nocapture")
    fi
    
    # Set environment variables
    export RUST_TEST_TIME_UNIT="60000"  # 60 second timeout per test
    
    if cargo "${args[@]}"; then
        print_success "Unit tests passed"
        return 0
    else
        print_error "Unit tests failed"
        return 1
    fi
}

# Function to run integration tests
run_integration_tests() {
    print_header "Running Integration Tests"
    
    local args=("test" "--test" "*")
    
    if [[ "$VERBOSE" == "true" ]]; then
        args+=("--verbose")
    fi
    
    if [[ "$PARALLEL" == "false" ]]; then
        args+=("--" "--test-threads=1")
    fi
    
    # Set longer timeout for integration tests
    export RUST_TEST_TIME_UNIT="180000"  # 3 minute timeout per test
    
    if cargo "${args[@]}"; then
        print_success "Integration tests passed"
        return 0
    else
        print_error "Integration tests failed"
        return 1
    fi
}

# Function to run property-based tests
run_property_tests() {
    print_header "Running Property-Based Tests"
    
    local args=("test" "property")
    
    if [[ "$VERBOSE" == "true" ]]; then
        args+=("--verbose")
    fi
    
    # Set property test configuration
    export PROPTEST_CASES="$PROPTEST_CASES"
    export PROPTEST_MAX_SHRINK_ITERS="10000"
    
    if cargo "${args[@]}"; then
        print_success "Property-based tests passed"
        return 0
    else
        print_error "Property-based tests failed"
        return 1
    fi
}

# Function to run performance tests
run_performance_tests() {
    print_header "Running Performance Tests"
    
    local args=("test" "benchmark" "--release")
    
    if [[ "$VERBOSE" == "true" ]]; then
        args+=("--verbose")
    fi
    
    if cargo "${args[@]}"; then
        print_success "Performance tests passed"
        return 0
    else
        print_error "Performance tests failed"
        return 1
    fi
}

# Function to run chaos engineering tests
run_chaos_tests() {
    print_header "Running Chaos Engineering Tests"
    
    local args=("test" "chaos" "--release")
    
    if [[ "$VERBOSE" == "true" ]]; then
        args+=("--verbose")
    fi
    
    # Chaos tests need more time
    export RUST_TEST_TIME_UNIT="300000"  # 5 minute timeout per test
    
    if cargo "${args[@]}"; then
        print_success "Chaos engineering tests passed"
        return 0
    else
        print_error "Chaos engineering tests failed"
        return 1
    fi
}

# Function to run benchmarks
run_benchmarks() {
    print_header "Running Criterion Benchmarks"
    
    local args=("bench")
    
    if [[ "$VERBOSE" == "true" ]]; then
        args+=("--verbose")
    fi
    
    # Generate benchmark report
    if cargo "${args[@]}" -- --output-format pretty; then
        print_success "Benchmarks completed"
        
        # Copy benchmark results to test results directory
        if [[ -d "target/criterion" ]]; then
            cp -r target/criterion "$TEST_RESULTS_DIR/"
            print_info "Benchmark results saved to $TEST_RESULTS_DIR/criterion/"
        fi
        
        return 0
    else
        print_error "Benchmarks failed"
        return 1
    fi
}

# Function to generate coverage report
generate_coverage() {
    print_header "Generating Coverage Report"
    
    if command -v cargo-tarpaulin &> /dev/null; then
        local args=(
            "tarpaulin"
            "--out" "Html" "--out" "Xml" "--out" "Json"
            "--output-dir" "$COVERAGE_DIR"
            "--timeout" "$TIMEOUT"
        )
        
        if [[ "$VERBOSE" == "true" ]]; then
            args+=("--verbose")
        fi
        
        # Include all features
        args+=("--all-features")
        
        # Exclude test files from coverage
        args+=("--exclude-files" "tests/*" "benches/*")
        
        if cargo "${args[@]}"; then
            print_success "Coverage report generated"
            print_info "HTML report: $COVERAGE_DIR/tarpaulin-report.html"
            print_info "XML report: $COVERAGE_DIR/cobertura.xml"
            print_info "JSON report: $COVERAGE_DIR/tarpaulin-report.json"
            return 0
        else
            print_error "Coverage generation failed"
            return 1
        fi
    else
        print_warning "cargo-tarpaulin not found. Skipping coverage report."
        print_info "Install with: cargo install cargo-tarpaulin"
        return 1
    fi
}

# Function to generate reports
generate_reports() {
    print_header "Generating Test Reports"
    
    # Generate JUnit XML if requested
    if [[ "$JUNIT" == "true" ]]; then
        print_info "Generating JUnit XML report..."
        # This would require parsing cargo test output
        # For now, create a basic structure
        cat > "$TEST_RESULTS_DIR/junit.xml" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<testsuites name="autopoiesis" tests="0" failures="0" errors="0" time="0">
    <testsuite name="placeholder" tests="0" failures="0" errors="0" time="0">
    </testsuite>
</testsuites>
EOF
        print_info "JUnit XML saved to $TEST_RESULTS_DIR/junit.xml"
    fi
    
    # Generate HTML report if requested
    if [[ "$HTML" == "true" ]]; then
        print_info "Generating HTML test report..."
        # Copy the template and create a basic report
        if [[ -f "tests/assets/test_report_template.html" ]]; then
            cp "tests/assets/test_report_template.html" "$TEST_RESULTS_DIR/report.html"
            print_info "HTML report saved to $TEST_RESULTS_DIR/report.html"
        else
            print_warning "HTML template not found"
        fi
    fi
    
    print_success "Reports generated"
}

# Function to run all tests
run_all_tests() {
    print_header "Running All Tests"
    
    local failed_categories=()
    
    # Run each category and track failures
    if ! run_unit_tests; then
        failed_categories+=("unit")
    fi
    
    if ! run_integration_tests; then
        failed_categories+=("integration")
    fi
    
    if ! run_property_tests; then
        failed_categories+=("property")
    fi
    
    if ! run_performance_tests; then
        failed_categories+=("performance")
    fi
    
    # Report results
    if [[ ${#failed_categories[@]} -eq 0 ]]; then
        print_success "All test categories passed!"
        return 0
    else
        print_error "Failed test categories: ${failed_categories[*]}"
        return 1
    fi
}

# Function to run security audit
run_security_audit() {
    print_header "Running Security Audit"
    
    if command -v cargo-audit &> /dev/null; then
        if cargo audit; then
            print_success "Security audit passed"
        else
            print_warning "Security audit found issues"
        fi
    else
        print_warning "cargo-audit not found. Skipping security audit."
    fi
}

# Function to check for outdated dependencies
check_outdated_deps() {
    print_header "Checking for Outdated Dependencies"
    
    if command -v cargo-outdated &> /dev/null; then
        cargo outdated || true  # Don't fail on outdated deps
    else
        print_warning "cargo-outdated not found. Skipping dependency check."
    fi
}

# Function to analyze binary size
analyze_binary_size() {
    print_header "Analyzing Binary Size"
    
    if command -v cargo-bloat &> /dev/null; then
        print_info "Building release binary for analysis..."
        cargo build --release --quiet || true
        
        print_info "Binary size analysis:"
        cargo bloat --release --crates || true
    else
        print_warning "cargo-bloat not found. Skipping binary size analysis."
    fi
}

# Function to display final summary
display_summary() {
    local start_time=$1
    local end_time=$2
    local exit_code=$3
    
    local duration=$((end_time - start_time))
    local minutes=$((duration / 60))
    local seconds=$((duration % 60))
    
    print_header "Test Summary"
    
    print_info "Total execution time: ${minutes}m ${seconds}s"
    
    if [[ $exit_code -eq 0 ]]; then
        print_success "All tests completed successfully! 🎉"
        print_info "Test results: $TEST_RESULTS_DIR/"
        
        if [[ "$COVERAGE" == "true" ]]; then
            print_info "Coverage report: $COVERAGE_DIR/"
        fi
    else
        print_error "Some tests failed. Check the output above for details."
        print_info "Exit code: $exit_code"
    fi
    
    echo
    print_info "For detailed results, check:"
    print_info "  • Test results: $TEST_RESULTS_DIR/"
    print_info "  • Coverage: $COVERAGE_DIR/"
    print_info "  • Benchmarks: target/criterion/"
    echo
}

# Main execution function
main() {
    local start_time=$(date +%s)
    
    # Default values
    COVERAGE="false"
    VERBOSE="false"
    TIMEOUT="$DEFAULT_TIMEOUT"
    PROPTEST_CASES="$DEFAULT_PROPTEST_CASES"
    PARALLEL="true"
    NO_CAPTURE="false"
    HTML="false"
    JUNIT="false"
    CLEAN="false"
    INSTALL_DEPS="false"
    TEST_CATEGORY="all"
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -c|--coverage)
                COVERAGE="true"
                shift
                ;;
            -v|--verbose)
                VERBOSE="true"
                shift
                ;;
            -t|--timeout)
                TIMEOUT="$2"
                shift 2
                ;;
            -p|--proptest)
                PROPTEST_CASES="$2"
                shift 2
                ;;
            -j|--parallel)
                PARALLEL="true"
                shift
                ;;
            -s|--sequential)
                PARALLEL="false"
                shift
                ;;
            --no-capture)
                NO_CAPTURE="true"
                shift
                ;;
            --html)
                HTML="true"
                shift
                ;;
            --junit)
                JUNIT="true"
                shift
                ;;
            --clean)
                CLEAN="true"
                shift
                ;;
            --install-deps)
                INSTALL_DEPS="true"
                shift
                ;;
            -h|--help)
                show_usage
                exit 0
                ;;
            unit|integration|property|performance|chaos|benchmarks|all)
                TEST_CATEGORY="$1"
                shift
                ;;
            *)
                print_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    # Set environment variables
    export RUST_BACKTRACE="${RUST_BACKTRACE:-1}"
    export PROPTEST_CASES="$PROPTEST_CASES"
    
    if [[ "$VERBOSE" == "true" ]]; then
        export RUST_LOG="${RUST_LOG:-debug}"
    fi
    
    # Welcome message
    print_header "Autopoiesis Test Suite"
    print_info "Test category: $TEST_CATEGORY"
    print_info "Coverage: $COVERAGE"
    print_info "Timeout: ${TIMEOUT}s"
    print_info "Property test cases: $PROPTEST_CASES"
    print_info "Parallel execution: $PARALLEL"
    
    # Check prerequisites
    check_prerequisites
    
    # Install dependencies if requested
    if [[ "$INSTALL_DEPS" == "true" ]]; then
        install_test_dependencies
    fi
    
    # Clean if requested
    if [[ "$CLEAN" == "true" ]]; then
        clean_build
    fi
    
    # Set exit code tracking
    local overall_exit_code=0
    
    # Run tests based on category
    case "$TEST_CATEGORY" in
        unit)
            run_unit_tests || overall_exit_code=$?
            ;;
        integration)
            run_integration_tests || overall_exit_code=$?
            ;;
        property)
            run_property_tests || overall_exit_code=$?
            ;;
        performance)
            run_performance_tests || overall_exit_code=$?
            ;;
        chaos)
            run_chaos_tests || overall_exit_code=$?
            ;;
        benchmarks)
            run_benchmarks || overall_exit_code=$?
            ;;
        all)
            run_all_tests || overall_exit_code=$?
            ;;
        *)
            print_error "Unknown test category: $TEST_CATEGORY"
            exit 1
            ;;
    esac
    
    # Generate coverage if requested and tests passed
    if [[ "$COVERAGE" == "true" ]] && [[ $overall_exit_code -eq 0 ]]; then
        generate_coverage || true  # Don't fail overall if coverage fails
    fi
    
    # Run additional checks
    if [[ $overall_exit_code -eq 0 ]]; then
        run_security_audit || true
        check_outdated_deps || true
        
        if [[ "$TEST_CATEGORY" == "all" ]]; then
            analyze_binary_size || true
        fi
    fi
    
    # Generate reports
    generate_reports || true
    
    # Display summary
    local end_time=$(date +%s)
    display_summary "$start_time" "$end_time" "$overall_exit_code"
    
    exit $overall_exit_code
}

# Run main function with all arguments
main "$@"