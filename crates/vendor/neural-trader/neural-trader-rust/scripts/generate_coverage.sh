#!/bin/bash
# Generate test coverage reports

set -e

echo "📊 Generating Test Coverage Report"
echo "===================================="
echo ""

# Check if tarpaulin is installed
if ! command -v cargo-tarpaulin &> /dev/null; then
    echo "Installing cargo-tarpaulin..."
    cargo install cargo-tarpaulin
fi

# Generate coverage in multiple formats
echo "Running tarpaulin..."
cargo tarpaulin \
    --workspace \
    --all-features \
    --timeout 600 \
    --out Html \
    --out Xml \
    --out Lcov \
    --output-dir coverage/

echo ""
echo "✓ Coverage reports generated:"
echo "  - HTML: coverage/index.html"
echo "  - XML: coverage/cobertura.xml"
echo "  - LCOV: coverage/lcov.info"
echo ""

# Display summary
if [ -f "coverage/index.html" ]; then
    echo "To view the HTML report, open:"
    echo "  file://$(pwd)/coverage/index.html"
    echo ""
fi

# Extract coverage percentage if available
if command -v grep &> /dev/null && [ -f "coverage/cobertura.xml" ]; then
    COVERAGE=$(grep -oP 'line-rate="\K[^"]+' coverage/cobertura.xml | head -1)
    if [ ! -z "$COVERAGE" ]; then
        COVERAGE_PCT=$(echo "$COVERAGE * 100" | bc)
        echo "Overall Line Coverage: ${COVERAGE_PCT}%"

        if (( $(echo "$COVERAGE_PCT >= 90.0" | bc -l) )); then
            echo "✓ Coverage target (90%) met!"
        else
            echo "⚠ Coverage below 90% target"
        fi
    fi
fi

echo ""
echo "===================================="
