#!/bin/bash
# Mutation Testing Runner for HyperPhysics
# Usage: ./scripts/run_mutation_tests.sh [--crate CRATE_NAME] [--timeout SECONDS]

set -euo pipefail

# Configuration
WORKSPACE_ROOT="/Users/ashina/Desktop/Kurultay/HyperPhysics"
DEFAULT_TIMEOUT=300
OUTPUT_DIR="$WORKSPACE_ROOT/target/mutants"

# Parse arguments
CRATE=""
TIMEOUT=$DEFAULT_TIMEOUT

while [[ $# -gt 0 ]]; do
    case $1 in
        --crate)
            CRATE="$2"
            shift 2
            ;;
        --timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--crate CRATE_NAME] [--timeout SECONDS]"
            exit 1
            ;;
    esac
done

# Check if cargo-mutants is installed
if ! command -v cargo-mutants &> /dev/null; then
    echo "cargo-mutants not found. Installing..."
    cargo install cargo-mutants
fi

# Change to workspace root
cd "$WORKSPACE_ROOT"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "===========================================" echo "Running Mutation Tests"
echo "==========================================="
echo "Workspace: $WORKSPACE_ROOT"
echo "Timeout: ${TIMEOUT}s"
echo "Output: $OUTPUT_DIR"

if [ -n "$CRATE" ]; then
    echo "Crate: $CRATE"
    cargo mutants --package "$CRATE" --timeout "$TIMEOUT" --output "$OUTPUT_DIR"
else
    echo "Crate: All workspace crates"
    cargo mutants --workspace --timeout "$TIMEOUT" --output "$OUTPUT_DIR"
fi

echo ""
echo "==========================================="
echo "Mutation Testing Complete"
echo "==========================================="
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "To view detailed results:"
echo "  cat $OUTPUT_DIR/mutants.out"
echo ""
echo "To test a specific mutant:"
echo "  cargo mutants --mutant 'path/to/file.rs:LINE'"
