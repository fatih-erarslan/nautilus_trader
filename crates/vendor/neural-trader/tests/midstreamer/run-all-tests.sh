#!/bin/bash

echo "🧪 Running Midstreamer Integration Test Suite"
echo "=============================================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Test results
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

run_test_suite() {
    local suite_name=$1
    local test_path=$2
    
    echo -e "${YELLOW}Running ${suite_name}...${NC}"
    
    if npm test -- "$test_path" --silent 2>&1 | grep -q "PASS"; then
        echo -e "${GREEN}✓ ${suite_name} passed${NC}"
        ((PASSED_TESTS++))
    else
        echo -e "${RED}✗ ${suite_name} failed${NC}"
        ((FAILED_TESTS++))
    fi
    ((TOTAL_TESTS++))
    echo ""
}

# Run all test suites
echo "📊 Test Suites:"
echo ""

run_test_suite "DTW Pattern Matching" "tests/midstreamer/dtw"
run_test_suite "LCS Strategy Correlation" "tests/midstreamer/lcs"
run_test_suite "ReasoningBank Learning" "tests/midstreamer/reasoningbank"
run_test_suite "QUIC Coordination" "tests/midstreamer/quic"
run_test_suite "End-to-End Integration" "tests/midstreamer/integration"
run_test_suite "Performance Benchmarks" "tests/midstreamer/benchmarks"

# Summary
echo "=============================================="
echo -e "📈 Test Summary:"
echo -e "   Total Suites: ${TOTAL_TESTS}"
echo -e "   ${GREEN}Passed: ${PASSED_TESTS}${NC}"
echo -e "   ${RED}Failed: ${FAILED_TESTS}${NC}"
echo ""

if [ $FAILED_TESTS -eq 0 ]; then
    echo -e "${GREEN}✓ All test suites passed!${NC}"
    exit 0
else
    echo -e "${RED}✗ Some test suites failed${NC}"
    exit 1
fi
