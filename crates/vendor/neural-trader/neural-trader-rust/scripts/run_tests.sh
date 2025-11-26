#!/bin/bash
# Comprehensive test execution script for neural-trader Rust port

set -e

echo "🧪 Neural Trader Rust - Test Suite Runner"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to run a test category
run_test_category() {
    local category=$1
    local command=$2

    echo -e "${YELLOW}Running ${category}...${NC}"
    if eval "$command"; then
        echo -e "${GREEN}✓ ${category} passed${NC}"
        echo ""
        return 0
    else
        echo -e "${RED}✗ ${category} failed${NC}"
        echo ""
        return 1
    fi
}

# Track failures
FAILED_TESTS=()

# 1. Unit Tests
if ! run_test_category "Unit Tests" "cargo test --lib --all-features"; then
    FAILED_TESTS+=("Unit Tests")
fi

# 2. Integration Tests
if ! run_test_category "Integration Tests" "cargo test --test '*' --all-features"; then
    FAILED_TESTS+=("Integration Tests")
fi

# 3. Property Tests
if ! run_test_category "Property Tests" "cargo test --test test_invariants --all-features"; then
    FAILED_TESTS+=("Property Tests")
fi

# 4. E2E Tests
if ! run_test_category "End-to-End Tests" "cargo test --test test_full_trading_loop --test test_backtesting --all-features"; then
    FAILED_TESTS+=("E2E Tests")
fi

# 5. Load Tests (in release mode for performance)
if ! run_test_category "Load Tests" "cargo test --test stress_tests --release"; then
    FAILED_TESTS+=("Load Tests")
fi

# 6. Fault Tolerance Tests
if ! run_test_category "Fault Tolerance Tests" "cargo test --test error_injection --all-features"; then
    FAILED_TESTS+=("Fault Tolerance Tests")
fi

# Summary
echo "=========================================="
echo "Test Summary"
echo "=========================================="
echo ""

if [ ${#FAILED_TESTS[@]} -eq 0 ]; then
    echo -e "${GREEN}✓ All test categories passed!${NC}"
    echo ""
    exit 0
else
    echo -e "${RED}✗ Some tests failed:${NC}"
    for test in "${FAILED_TESTS[@]}"; do
        echo "  - $test"
    done
    echo ""
    exit 1
fi
