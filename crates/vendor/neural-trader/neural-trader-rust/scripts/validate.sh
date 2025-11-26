#!/bin/bash

# Neural Trader Rust Validation Script
# Runs comprehensive validation of all components

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║         Neural Trader Rust - Validation Suite                   ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

PASS_COUNT=0
FAIL_COUNT=0

# Function to run test
run_test() {
    local crate=$1
    local name=$2
    
    echo -n "Testing $name... "
    if cargo test --package $crate --lib 2>&1 | grep -q "test result: ok"; then
        echo "✅ PASS"
        ((PASS_COUNT++))
    else
        echo "❌ FAIL"
        ((FAIL_COUNT++))
    fi
}

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Running Unit Tests"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

run_test "nt-core" "Core Types & Config"
run_test "nt-market-data" "Market Data & Streaming"
run_test "nt-features" "Feature Engineering"
run_test "nt-risk" "Risk Management (may have 2 failures)"
run_test "nt-backtesting" "Backtesting Engine"
run_test "nt-portfolio" "Portfolio Management"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Results"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "✅ Passed: $PASS_COUNT"
echo "❌ Failed: $FAIL_COUNT"
echo ""

if [ $FAIL_COUNT -eq 0 ]; then
    echo "🎉 All tests passed!"
else
    echo "⚠️  Some tests failed. See details above."
fi

echo ""
echo "For detailed validation report, see:"
echo "  - docs/FINAL_VALIDATION_REPORT.md"
echo "  - docs/VALIDATION_EXECUTIVE_SUMMARY.md"
echo "  - VALIDATION_RESULTS.txt"
echo ""
