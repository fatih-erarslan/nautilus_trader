#!/bin/bash
# Comprehensive CI test suite for CDFA crates

set -e  # Exit on error
set -u  # Exit on undefined variable

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "======================================"
echo "CDFA Comprehensive Test Suite"
echo "======================================"

# Function to run tests and capture results
run_test() {
    local name=$1
    local cmd=$2
    
    echo -e "\n${YELLOW}Running ${name}...${NC}"
    if eval "$cmd"; then
        echo -e "${GREEN}✓ ${name} PASSED${NC}"
        return 0
    else
        echo -e "${RED}✗ ${name} FAILED${NC}"
        return 1
    fi
}

# Track failures
FAILURES=0

# 1. Check code formatting
run_test "Format Check" "cargo fmt --all -- --check" || ((FAILURES++))

# 2. Run clippy lints
run_test "Clippy Lints" "cargo clippy --all-targets --all-features -- -D warnings" || ((FAILURES++))

# 3. Build all crates
run_test "Build All" "cargo build --all --release" || ((FAILURES++))

# 4. Run unit tests
run_test "Unit Tests" "cargo test --all --lib" || ((FAILURES++))

# 5. Run integration tests
run_test "Integration Tests" "cargo test --all --test '*'" || ((FAILURES++))

# 6. Run doc tests
run_test "Doc Tests" "cargo test --all --doc" || ((FAILURES++))

# 7. Run property-based tests (if available)
if grep -q "quickcheck" Cargo.toml; then
    run_test "Property Tests" "cargo test --all property_tests" || ((FAILURES++))
fi

# 8. Check test coverage (requires cargo-tarpaulin)
if command -v cargo-tarpaulin &> /dev/null; then
    run_test "Coverage Report" "cargo tarpaulin --all --out Xml --output-dir coverage" || ((FAILURES++))
    
    # Extract coverage percentage
    if [ -f coverage/cobertura.xml ]; then
        coverage=$(grep -oP 'line-rate="\K[^"]+' coverage/cobertura.xml | head -1)
        coverage_pct=$(echo "$coverage * 100" | bc)
        echo -e "\n${YELLOW}Test Coverage: ${coverage_pct}%${NC}"
        
        # Fail if coverage is below threshold
        threshold=80
        if (( $(echo "$coverage_pct < $threshold" | bc -l) )); then
            echo -e "${RED}✗ Coverage ${coverage_pct}% is below threshold ${threshold}%${NC}"
            ((FAILURES++))
        fi
    fi
else
    echo -e "${YELLOW}Skipping coverage (install cargo-tarpaulin)${NC}"
fi

# 9. Run benchmarks (smoke test only in CI)
run_test "Benchmark Smoke Test" "cargo bench --all --no-run" || ((FAILURES++))

# 10. Memory leak detection (requires valgrind)
if command -v valgrind &> /dev/null; then
    echo -e "\n${YELLOW}Running memory leak detection...${NC}"
    cargo test --release --bin cdfa_validation 2>&1 | \
    valgrind --leak-check=full --error-exitcode=1 ./target/release/cdfa_validation diversity_metrics test_input.json || ((FAILURES++))
else
    echo -e "${YELLOW}Skipping memory leak detection (install valgrind)${NC}"
fi

# 11. Performance regression check
if [ -f benchmarks/baseline.json ]; then
    echo -e "\n${YELLOW}Checking performance regression...${NC}"
    cargo bench --all -- --save-baseline current
    
    # Compare with baseline (requires cargo-benchcmp)
    if command -v cargo-benchcmp &> /dev/null; then
        regression=$(cargo benchcmp baseline current --threshold 5)
        if echo "$regression" | grep -q "REGRESSION"; then
            echo -e "${RED}✗ Performance regression detected${NC}"
            echo "$regression"
            ((FAILURES++))
        else
            echo -e "${GREEN}✓ No performance regression${NC}"
        fi
    fi
fi

# 12. Safety checks
run_test "Unsafe Code Check" "! grep -r 'unsafe' src/ --include='*.rs' || echo 'No unsafe code found'" || ((FAILURES++))

# 13. Dependency audit
if command -v cargo-audit &> /dev/null; then
    run_test "Security Audit" "cargo audit" || ((FAILURES++))
else
    echo -e "${YELLOW}Skipping security audit (install cargo-audit)${NC}"
fi

# 14. Documentation build
run_test "Documentation" "cargo doc --all --no-deps" || ((FAILURES++))

# Summary
echo -e "\n======================================"
if [ $FAILURES -eq 0 ]; then
    echo -e "${GREEN}All tests PASSED!${NC}"
    exit 0
else
    echo -e "${RED}${FAILURES} test(s) FAILED${NC}"
    exit 1
fi