#!/bin/bash
# Verify all NAPI exports in the .node file

echo "=== NAPI Export Verification Script ==="
echo ""

NODE_FILE=$(find /workspaces/neural-trader/neural-trader-rust -name "*.node" -type f | head -1)

if [ -z "$NODE_FILE" ]; then
    echo "❌ No .node file found. Build first with: cargo build --release"
    exit 1
fi

echo "✓ Found .node file: $NODE_FILE"
echo ""

echo "Analyzing exports..."
TOTAL_EXPORTS=$(nm "$NODE_FILE" | grep " T " | wc -l)
echo "Total exported symbols: $TOTAL_EXPORTS"

echo ""
echo "NAPI function exports:"
nm "$NODE_FILE" | grep " T " | grep -v "__" | head -20

echo ""
echo "Searching for key functions..."
FUNCTIONS="ping list_strategies neural_forecast analyze_news calculate_kelly execute_trade"

for func in $FUNCTIONS; do
    if nm "$NODE_FILE" | grep -q "$func"; then
        echo "  ✓ $func"
    else
        echo "  ❌ $func (NOT FOUND)"
    fi
done

echo ""
echo "=== Summary ==="
echo "Total exports: $TOTAL_EXPORTS"
echo "Expected: 91+ NAPI functions"
