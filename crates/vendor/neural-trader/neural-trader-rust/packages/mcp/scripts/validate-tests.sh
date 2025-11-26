#!/bin/bash
# Level 2: Unit Tests
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "🧪 Level 2: Unit Tests"
echo "=============================="

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

ERRORS=0
TOTAL_TESTS=0
PASSED_TESTS=0

# 1. Run Rust unit tests
echo -e "\n${YELLOW}2.1 Running Rust unit tests...${NC}"
cd "${PROJECT_ROOT}/../../.."

if cargo test --lib --manifest-path neural-trader-rust/crates/mcp-server/Cargo.toml 2>&1 | tee /tmp/rust-tests.log; then
    RUST_PASSED=$(grep -o "test result: ok\. [0-9]* passed" /tmp/rust-tests.log | grep -o "[0-9]*" || echo "0")
    RUST_FAILED=$(grep -o "[0-9]* failed" /tmp/rust-tests.log | grep -o "[0-9]*" || echo "0")

    echo -e "${GREEN}✓ Rust tests: ${RUST_PASSED} passed${NC}"
    TOTAL_TESTS=$((TOTAL_TESTS + RUST_PASSED))
    PASSED_TESTS=$((PASSED_TESTS + RUST_PASSED))

    if [ "$RUST_FAILED" -gt 0 ]; then
        echo -e "${RED}✗ Rust tests: ${RUST_FAILED} failed${NC}"
        ERRORS=$((ERRORS + RUST_FAILED))
    fi
else
    echo -e "${RED}✗ Rust tests failed to run${NC}"
    ERRORS=$((ERRORS + 1))
fi

# 2. Run JavaScript unit tests
echo -e "\n${YELLOW}2.2 Running JavaScript unit tests...${NC}"
cd "${PROJECT_ROOT}"

if [ -d "tests" ]; then
    if npm test 2>&1 | tee /tmp/js-tests.log; then
        # Mocha uses "X passing" not "X passed"
        JS_PASSED=$(grep -o "[0-9]* passing" /tmp/js-tests.log | head -1 | grep -o "[0-9]*" || echo "0")
        JS_FAILED=$(grep -o "[0-9]* failing" /tmp/js-tests.log | head -1 | grep -o "[0-9]*" || echo "0")

        echo -e "${GREEN}✓ JavaScript tests: ${JS_PASSED} passed${NC}"
        TOTAL_TESTS=$((TOTAL_TESTS + JS_PASSED))
        PASSED_TESTS=$((PASSED_TESTS + JS_PASSED))

        if [ "$JS_FAILED" -gt 0 ]; then
            echo -e "${RED}✗ JavaScript tests: ${JS_FAILED} failed${NC}"
            ERRORS=$((ERRORS + JS_FAILED))
        fi
    else
        echo -e "${RED}✗ JavaScript tests failed${NC}"
        ERRORS=$((ERRORS + 1))
    fi
else
    echo -e "${YELLOW}⚠ No tests directory found${NC}"
fi

# 3. Check test coverage
echo -e "\n${YELLOW}2.3 Checking test coverage...${NC}"
if [ -f "coverage/coverage-summary.json" ]; then
    COVERAGE=$(node -e "console.log(require('./coverage/coverage-summary.json').total.lines.pct)" || echo "0")
    echo "Code coverage: ${COVERAGE}%"

    if (( $(echo "$COVERAGE < 80" | bc -l) )); then
        echo -e "${YELLOW}⚠ Coverage below 80% threshold${NC}"
    else
        echo -e "${GREEN}✓ Coverage meets 80% threshold${NC}"
    fi
fi

# 4. Run integration tests
echo -e "\n${YELLOW}2.4 Running integration tests...${NC}"
if [ -f "tests/integration.test.js" ]; then
    if npm run test:integration 2>&1 | tee /tmp/integration-tests.log; then
        echo -e "${GREEN}✓ Integration tests passed${NC}"
    else
        echo -e "${RED}✗ Integration tests failed${NC}"
        ERRORS=$((ERRORS + 1))
    fi
else
    echo -e "${YELLOW}⚠ No integration tests found${NC}"
fi

# Calculate pass rate
if [ $TOTAL_TESTS -gt 0 ]; then
    PASS_RATE=$(echo "scale=2; $PASSED_TESTS * 100 / $TOTAL_TESTS" | bc)
else
    PASS_RATE=0
fi

# Summary
echo -e "\n=============================="
echo "Level 2 Summary:"
echo "  Total Tests: $TOTAL_TESTS"
echo "  Passed: $PASSED_TESTS"
echo "  Failed: $ERRORS"
echo "  Pass Rate: ${PASS_RATE}%"

if [ $ERRORS -eq 0 ] && [ $TOTAL_TESTS -gt 0 ]; then
    echo -e "${GREEN}✅ Level 2: PASSED${NC}"
    exit 0
else
    echo -e "${RED}❌ Level 2: FAILED${NC}"
    exit 1
fi
