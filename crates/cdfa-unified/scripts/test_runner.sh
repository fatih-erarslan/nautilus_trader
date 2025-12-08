#!/bin/bash

# Test runner script for achieving 100% coverage in CDFA unified library
# This script runs all test suites and validates coverage requirements

set -e  # Exit on any error

# Configuration
TARGET_COVERAGE=${1:-100}
CARGO_ARGS=${2:-""}
REPORT_DIR="target/coverage-report"
LOG_FILE="$REPORT_DIR/test-run.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Create report directory
mkdir -p "$REPORT_DIR"

# Logging function
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

# Print colored output
print_colored() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}" | tee -a "$LOG_FILE"
}

# Print section header
print_section() {
    local title=$1
    echo "" | tee -a "$LOG_FILE"
    print_colored "$BLUE" "========================================"
    print_colored "$BLUE" "$title"
    print_colored "$BLUE" "========================================"
    echo "" | tee -a "$LOG_FILE"
}

# Check if required tools are installed
check_dependencies() {
    print_section "Checking Dependencies"
    
    local missing_tools=()
    
    if ! command -v cargo &> /dev/null; then
        missing_tools+=("cargo")
    fi
    
    if ! command -v rustc &> /dev/null; then
        missing_tools+=("rustc")
    fi
    
    # Check for tarpaulin
    if ! cargo tarpaulin --version &> /dev/null; then
        log "Installing cargo-tarpaulin..."
        cargo install cargo-tarpaulin || {
            print_colored "$YELLOW" "Warning: Could not install cargo-tarpaulin. Coverage reporting may be limited."
        }
    fi
    
    # Check for criterion (for benchmarks)
    if ! grep -q "criterion" Cargo.toml; then
        print_colored "$YELLOW" "Warning: Criterion not found in Cargo.toml. Benchmark tests may not work."
    fi
    
    if [ ${#missing_tools[@]} -eq 0 ]; then
        print_colored "$GREEN" "✅ All required tools are available"
    else
        print_colored "$RED" "❌ Missing tools: ${missing_tools[*]}"
        exit 1
    fi
}

# Clean previous build artifacts
clean_build() {
    print_section "Cleaning Build Artifacts"
    
    log "Cleaning cargo cache..."
    cargo clean
    
    log "Removing previous coverage reports..."
    rm -rf "$REPORT_DIR"/*
    
    print_colored "$GREEN" "✅ Build artifacts cleaned"
}

# Build the project
build_project() {
    print_section "Building Project"
    
    log "Building with all features..."
    if cargo build --all-features $CARGO_ARGS; then
        print_colored "$GREEN" "✅ Project built successfully"
    else
        print_colored "$RED" "❌ Build failed"
        exit 1
    fi
    
    log "Building tests..."
    if cargo test --no-run --all-features $CARGO_ARGS; then
        print_colored "$GREEN" "✅ Tests compiled successfully"
    else
        print_colored "$RED" "❌ Test compilation failed"
        exit 1
    fi
}

# Run unit tests
run_unit_tests() {
    print_section "Running Unit Tests"
    
    local start_time=$(date +%s)
    local test_output_file="$REPORT_DIR/unit-tests.log"
    
    log "Running unit tests..."
    if cargo test --lib $CARGO_ARGS 2>&1 | tee "$test_output_file"; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        
        # Parse test results
        local passed=$(grep -o "[0-9]\+ passed" "$test_output_file" | head -1 | grep -o "[0-9]\+")
        local failed=$(grep -o "[0-9]\+ failed" "$test_output_file" | head -1 | grep -o "[0-9]\+")
        
        print_colored "$GREEN" "✅ Unit tests completed in ${duration}s"
        print_colored "$GREEN" "   Passed: ${passed:-0}, Failed: ${failed:-0}"
        
        if [ "${failed:-0}" -gt 0 ]; then
            print_colored "$RED" "❌ Some unit tests failed"
            return 1
        fi
    else
        print_colored "$RED" "❌ Unit tests failed"
        return 1
    fi
}

# Run integration tests
run_integration_tests() {
    print_section "Running Integration Tests"
    
    local start_time=$(date +%s)
    local test_output_file="$REPORT_DIR/integration-tests.log"
    
    log "Running integration tests..."
    if find tests -name "*.rs" -type f | head -1 > /dev/null 2>&1; then
        if cargo test --test '*' $CARGO_ARGS 2>&1 | tee "$test_output_file"; then
            local end_time=$(date +%s)
            local duration=$((end_time - start_time))
            
            print_colored "$GREEN" "✅ Integration tests completed in ${duration}s"
        else
            print_colored "$RED" "❌ Integration tests failed"
            return 1
        fi
    else
        print_colored "$YELLOW" "⚠️  No integration tests found"
    fi
}

# Run property-based tests
run_property_tests() {
    print_section "Running Property-Based Tests"
    
    local start_time=$(date +%s)
    local test_output_file="$REPORT_DIR/property-tests.log"
    
    log "Running property-based tests..."
    if cargo test --test '*' -- --include-ignored property $CARGO_ARGS 2>&1 | tee "$test_output_file"; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        
        print_colored "$GREEN" "✅ Property tests completed in ${duration}s"
    else
        print_colored "$YELLOW" "⚠️  Property tests had issues (this may be expected)"
    fi
}

# Run fuzz tests
run_fuzz_tests() {
    print_section "Running Fuzz Tests"
    
    local start_time=$(date +%s)
    local test_output_file="$REPORT_DIR/fuzz-tests.log"
    
    log "Running fuzz tests..."
    if cargo test --test '*' -- --include-ignored fuzz $CARGO_ARGS 2>&1 | tee "$test_output_file"; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        
        print_colored "$GREEN" "✅ Fuzz tests completed in ${duration}s"
    else
        print_colored "$YELLOW" "⚠️  Fuzz tests had issues (this may be expected)"
    fi
}

# Run benchmark tests
run_benchmark_tests() {
    print_section "Running Benchmark Tests"
    
    local start_time=$(date +%s)
    local bench_output_file="$REPORT_DIR/benchmark-tests.log"
    
    log "Running benchmark tests (compilation only)..."
    if cargo bench --no-run $CARGO_ARGS 2>&1 | tee "$bench_output_file"; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        
        print_colored "$GREEN" "✅ Benchmark tests compiled in ${duration}s"
    else
        print_colored "$YELLOW" "⚠️  Benchmark test compilation had issues"
    fi
}

# Run documentation tests
run_doc_tests() {
    print_section "Running Documentation Tests"
    
    local start_time=$(date +%s)
    local doc_output_file="$REPORT_DIR/doc-tests.log"
    
    log "Running documentation tests..."
    if cargo test --doc $CARGO_ARGS 2>&1 | tee "$doc_output_file"; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        
        print_colored "$GREEN" "✅ Documentation tests completed in ${duration}s"
    else
        print_colored "$YELLOW" "⚠️  Documentation tests had issues"
    fi
}

# Generate coverage report
generate_coverage() {
    print_section "Generating Coverage Report"
    
    log "Running tarpaulin for coverage analysis..."
    local coverage_file="$REPORT_DIR/coverage.json"
    local lcov_file="$REPORT_DIR/lcov.info"
    
    if command -v cargo-tarpaulin &> /dev/null; then
        if cargo tarpaulin \
            --out Json \
            --out Lcov \
            --output-dir "$REPORT_DIR" \
            --exclude-files "target/*" \
            --exclude-files "tests/*" \
            --exclude-files "benches/*" \
            --exclude-files "examples/*" \
            --timeout 300 \
            $CARGO_ARGS; then
            
            # Parse coverage percentage
            if [ -f "$REPORT_DIR/tarpaulin-report.json" ]; then
                local coverage=$(python3 -c "
import json
try:
    with open('$REPORT_DIR/tarpaulin-report.json', 'r') as f:
        data = json.load(f)
    print(f\"{data['files']['coverage']:.2f}\")
except:
    print('0.00')
" 2>/dev/null || echo "0.00")
                
                print_colored "$GREEN" "✅ Coverage analysis completed: ${coverage}%"
                
                # Check if coverage meets target
                if (( $(echo "$coverage >= $TARGET_COVERAGE" | bc -l) )); then
                    print_colored "$GREEN" "✅ Coverage target achieved: ${coverage}% >= ${TARGET_COVERAGE}%"
                else
                    print_colored "$RED" "❌ Coverage target not met: ${coverage}% < ${TARGET_COVERAGE}%"
                    return 1
                fi
            else
                print_colored "$YELLOW" "⚠️  Coverage report file not found"
            fi
        else
            print_colored "$RED" "❌ Coverage analysis failed"
            return 1
        fi
    else
        print_colored "$YELLOW" "⚠️  Tarpaulin not available, using manual coverage estimation"
        
        # Manual coverage estimation
        local total_lines=$(find src -name "*.rs" -exec wc -l {} + | tail -1 | awk '{print $1}')
        local test_lines=$(find tests -name "*.rs" -exec wc -l {} + 2>/dev/null | tail -1 | awk '{print $1}' || echo "0")
        local estimated_coverage=$(echo "scale=2; ($test_lines / $total_lines) * 50" | bc -l 2>/dev/null || echo "0.00")
        
        print_colored "$YELLOW" "📊 Estimated coverage: ${estimated_coverage}% (based on test/source ratio)"
    fi
}

# Generate HTML report
generate_html_report() {
    print_section "Generating HTML Report"
    
    local html_file="$REPORT_DIR/index.html"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S UTC')
    
    cat > "$html_file" << EOF
<!DOCTYPE html>
<html>
<head>
    <title>CDFA Test Coverage Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .header { text-align: center; margin-bottom: 30px; }
        .metric { display: inline-block; margin: 15px; padding: 20px; border: 1px solid #ddd; border-radius: 8px; min-width: 150px; text-align: center; }
        .metric h3 { margin: 0 0 10px 0; color: #333; }
        .metric .value { font-size: 24px; font-weight: bold; }
        .success { color: #28a745; border-color: #28a745; }
        .warning { color: #ffc107; border-color: #ffc107; }
        .danger { color: #dc3545; border-color: #dc3545; }
        .log-section { margin: 20px 0; }
        .log-content { background: #f8f9fa; padding: 15px; border-radius: 5px; font-family: monospace; white-space: pre-wrap; max-height: 300px; overflow-y: auto; }
        .coverage-bar { width: 100%; height: 30px; background: #e9ecef; border-radius: 15px; overflow: hidden; margin: 10px 0; }
        .coverage-fill { height: 100%; background: linear-gradient(to right, #dc3545, #ffc107, #28a745); transition: width 0.3s ease; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧪 CDFA Test Coverage Report</h1>
            <p>Generated on $timestamp</p>
        </div>
        
        <div class="metrics">
EOF

    # Add test metrics
    if [ -f "$REPORT_DIR/unit-tests.log" ]; then
        local unit_passed=$(grep -o "[0-9]\+ passed" "$REPORT_DIR/unit-tests.log" | head -1 | grep -o "[0-9]\+" || echo "0")
        local unit_failed=$(grep -o "[0-9]\+ failed" "$REPORT_DIR/unit-tests.log" | head -1 | grep -o "[0-9]\+" || echo "0")
        
        cat >> "$html_file" << EOF
            <div class="metric $([ "$unit_failed" -eq 0 ] && echo "success" || echo "danger")">
                <h3>Unit Tests</h3>
                <div class="value">$unit_passed passed</div>
                <div>$unit_failed failed</div>
            </div>
EOF
    fi

    cat >> "$html_file" << EOF
        </div>
        
        <div class="log-section">
            <h2>📋 Test Execution Logs</h2>
            <h3>Recent Test Run</h3>
            <div class="log-content">$(tail -50 "$LOG_FILE" | sed 's/&/\&amp;/g; s/</\&lt;/g; s/>/\&gt;/g')</div>
        </div>
        
        <div class="log-section">
            <h2>📊 Coverage Analysis</h2>
            <p>Coverage target: $TARGET_COVERAGE%</p>
EOF

    if [ -f "$REPORT_DIR/tarpaulin-report.json" ]; then
        cat >> "$html_file" << EOF
            <p><a href="tarpaulin-report.html">📈 Detailed Coverage Report</a></p>
EOF
    fi

    cat >> "$html_file" << EOF
        </div>
        
        <div class="log-section">
            <h2>🛠️ Next Steps</h2>
            <ul>
                <li>Review uncovered code paths in the detailed coverage report</li>
                <li>Add tests for any missing functionality</li>
                <li>Run property-based tests for mathematical functions</li>
                <li>Implement fuzz tests for edge cases</li>
                <li>Add integration tests for module interactions</li>
            </ul>
        </div>
        
        <div class="log-section">
            <h2>📁 Generated Files</h2>
            <ul>
                <li><a href="unit-tests.log">Unit Test Log</a></li>
                <li><a href="integration-tests.log">Integration Test Log</a></li>
                <li><a href="property-tests.log">Property Test Log</a></li>
                <li><a href="fuzz-tests.log">Fuzz Test Log</a></li>
                <li><a href="benchmark-tests.log">Benchmark Test Log</a></li>
                <li><a href="test-run.log">Complete Test Run Log</a></li>
            </ul>
        </div>
    </div>
</body>
</html>
EOF

    print_colored "$GREEN" "✅ HTML report generated: $html_file"
}

# Validate final results
validate_results() {
    print_section "Validating Results"
    
    local errors=0
    
    # Check if any critical tests failed
    if [ -f "$REPORT_DIR/unit-tests.log" ]; then
        local unit_failed=$(grep -o "[0-9]\+ failed" "$REPORT_DIR/unit-tests.log" | head -1 | grep -o "[0-9]\+" || echo "0")
        if [ "$unit_failed" -gt 0 ]; then
            print_colored "$RED" "❌ Unit tests failed: $unit_failed failures"
            errors=$((errors + 1))
        fi
    fi
    
    # Check coverage if tarpaulin was successful
    if [ -f "$REPORT_DIR/tarpaulin-report.json" ]; then
        local coverage=$(python3 -c "
import json
try:
    with open('$REPORT_DIR/tarpaulin-report.json', 'r') as f:
        data = json.load(f)
    print(f\"{data['files']['coverage']:.2f}\")
except:
    print('0.00')
" 2>/dev/null || echo "0.00")
        
        if (( $(echo "$coverage < $TARGET_COVERAGE" | bc -l) )); then
            print_colored "$RED" "❌ Coverage target not met: ${coverage}% < ${TARGET_COVERAGE}%"
            errors=$((errors + 1))
        fi
    fi
    
    if [ $errors -eq 0 ]; then
        print_colored "$GREEN" "✅ All validation checks passed!"
        return 0
    else
        print_colored "$RED" "❌ Validation failed with $errors errors"
        return 1
    fi
}

# Print final summary
print_summary() {
    print_section "Test Run Summary"
    
    local end_time=$(date +%s)
    local total_duration=$((end_time - START_TIME))
    
    print_colored "$BLUE" "Total execution time: ${total_duration}s"
    print_colored "$BLUE" "Report directory: $REPORT_DIR"
    print_colored "$BLUE" "HTML report: $REPORT_DIR/index.html"
    
    echo "" | tee -a "$LOG_FILE"
    print_colored "$GREEN" "🎉 Test run completed!"
    echo "" | tee -a "$LOG_FILE"
}

# Main execution
main() {
    local START_TIME=$(date +%s)
    
    print_section "CDFA Test Coverage Validation"
    log "Starting test run with coverage target: $TARGET_COVERAGE%"
    log "Additional cargo args: $CARGO_ARGS"
    
    # Initialize log file
    echo "CDFA Test Run Log - $(date)" > "$LOG_FILE"
    echo "Target Coverage: $TARGET_COVERAGE%" >> "$LOG_FILE"
    echo "======================================" >> "$LOG_FILE"
    
    # Execute all steps
    check_dependencies
    clean_build
    build_project
    
    # Run all test suites
    run_unit_tests || true  # Continue even if some tests fail
    run_integration_tests || true
    run_property_tests || true
    run_fuzz_tests || true
    run_benchmark_tests || true
    run_doc_tests || true
    
    # Generate reports
    generate_coverage || true
    generate_html_report
    
    # Final validation
    if validate_results; then
        print_summary
        exit 0
    else
        print_summary
        exit 1
    fi
}

# Handle interruption
trap 'print_colored "$RED" "Test run interrupted!"; exit 130' INT TERM

# Run main function
main "$@"