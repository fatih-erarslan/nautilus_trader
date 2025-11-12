#!/bin/bash
# Fuzzing Test Runner for HyperPhysics
# Usage: ./scripts/run_fuzz_tests.sh [--target TARGET] [--time SECONDS] [--jobs N]

set -euo pipefail

# Configuration
WORKSPACE_ROOT="/Users/ashina/Desktop/Kurultay/HyperPhysics"
DEFAULT_TIME=3600  # 1 hour
DEFAULT_JOBS=1

# Parse arguments
TARGET=""
TIME=$DEFAULT_TIME
JOBS=$DEFAULT_JOBS

while [[ $# -gt 0 ]]; do
    case $1 in
        --target)
            TARGET="$2"
            shift 2
            ;;
        --time)
            TIME="$2"
            shift 2
            ;;
        --jobs)
            JOBS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--target TARGET] [--time SECONDS] [--jobs N]"
            exit 1
            ;;
    esac
done

# Check if cargo-fuzz is installed
if ! command -v cargo-fuzz &> /dev/null; then
    echo "cargo-fuzz not found. Installing..."
    cargo install cargo-fuzz
fi

# Change to workspace root
cd "$WORKSPACE_ROOT"

echo "==========================================="
echo "Running Fuzz Tests"
echo "==========================================="
echo "Workspace: $WORKSPACE_ROOT"
echo "Max Time: ${TIME}s"
echo "Jobs: $JOBS"

if [ -n "$TARGET" ]; then
    echo "Target: $TARGET"
    echo ""
    cargo fuzz run "$TARGET" -- -max_total_time="$TIME" -jobs="$JOBS"
else
    echo "Running all fuzz targets sequentially..."
    echo ""

    for target in fuzz_gillespie fuzz_metropolis fuzz_lattice; do
        echo "Running $target..."
        cargo fuzz run "$target" -- -max_total_time="$TIME" -jobs="$JOBS" || true
        echo ""
    done
fi

echo "==========================================="
echo "Fuzzing Complete"
echo "==========================================="
echo ""
echo "To minimize a corpus:"
echo "  cargo fuzz cmin TARGET"
echo ""
echo "To debug a crash:"
echo "  cargo fuzz run TARGET crash-file.txt"
