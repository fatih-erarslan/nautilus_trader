#!/bin/bash

echo "=== SYSTEMATIC COMPILATION FIX ==="
echo "Financial System - Fixing struct field and method signature mismatches"

# Step 1: Build and analyze errors
echo "Analyzing current errors..."
cargo build --release --features python-bindings 2>&1 | tee current_errors.log

# Count error categories
echo ""
echo "Error Analysis:"
echo "- Field errors: $(grep "no field" current_errors.log | wc -l)"
echo "- Method errors: $(grep "no method" current_errors.log | wc -l)"
echo "- Import errors: $(grep "unresolved import" current_errors.log | wc -l)"
echo "- Argument errors: $(grep "this method takes" current_errors.log | wc -l)"
echo "- Type errors: $(grep "the trait" current_errors.log | wc -l)"

# Step 2: Extract unique field errors
echo ""
echo "Field mismatches to fix:"
grep "no field" current_errors.log | sed 's/.*no field `\([^`]*\)`.*/\1/' | sort -u | head -20

# Step 3: Extract unique method errors
echo ""
echo "Method mismatches to fix:"
grep "no method" current_errors.log | sed 's/.*no method named `\([^`]*\)`.*/\1/' | sort -u | head -20

# Step 4: Show top error files
echo ""
echo "Files with most errors:"
grep "^   --> src/" current_errors.log | cut -d':' -f1 | cut -d'/' -f2- | sort | uniq -c | sort -rn | head -10

TOTAL_ERRORS=$(grep "^error" current_errors.log | wc -l)
echo ""
echo "Total errors to fix: $TOTAL_ERRORS"

# Provide recommendations
echo ""
echo "=== RECOMMENDATIONS ==="
echo "1. Fix ReturnData struct - add missing fields or update usage"
echo "2. Fix PerformanceTracker struct - add missing fields"
echo "3. Fix MarketObservation struct - add missing fields"
echo "4. Fix BlackSwanDetector methods - add missing methods"
echo "5. Align method signatures with their calls"