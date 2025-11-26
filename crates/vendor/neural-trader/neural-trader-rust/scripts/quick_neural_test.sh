#!/bin/bash

# Quick Neural Network Test Runner - runs fast subset of tests

set -e

RESULTS_DIR="/workspaces/neural-trader/neural-trader-rust/packages/docs/tests"
mkdir -p "$RESULTS_DIR"

cd /workspaces/neural-trader/neural-trader-rust/crates/neural

echo "Running quick neural network tests..."
echo ""

# Run non-ignored tests (fast)
cargo test --features candle --test comprehensive_neural_test -- --nocapture

echo ""
echo "Quick tests completed!"
echo "For full test suite, run: /workspaces/neural-trader/neural-trader-rust/scripts/run_neural_tests.sh"
