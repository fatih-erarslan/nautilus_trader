#!/bin/bash
# CQGS Post-Edit Hook - Comprehensive Analysis
# Runs after file edit for full synthetic data analysis

set -euo pipefail

FILE="$1"

echo "🔍 CQGS Post-Edit Analysis: $FILE"

# Skip analysis for documentation and reports
case "$FILE" in
    *.md|**/target/**|**/.git/**)
        echo "ℹ️  Skipping analysis for documentation/build file"
        exit 0
        ;;
esac

# Comprehensive synthetic data scan
SYNTHETIC_SCORE=0

# Check for fastrand usage
if grep -q "fastrand::" "$FILE" 2>/dev/null; then
    echo "❌ CRITICAL: fastrand usage found - synthetic data contamination"
    SYNTHETIC_SCORE=$((SYNTHETIC_SCORE + 50))
fi

# Check for synthetic patterns
SYNTHETIC_PATTERNS=(
    "random.*\*.*2\.0.*-.*1\.0"
    "fake.*market"
    "synthetic.*data"
    "dummy.*data"
    "placeholder.*data"
    "artificial.*data"
    "toy.*data"
)

for pattern in "${SYNTHETIC_PATTERNS[@]}"; do
    if grep -E "$pattern" "$FILE" >/dev/null 2>&1; then
        echo "⚠️  Synthetic pattern detected: $pattern"
        SYNTHETIC_SCORE=$((SYNTHETIC_SCORE + 10))
    fi
done

# Check for proper API integration in financial files
if [[ "$FILE" == *"financial"* ]] || [[ "$FILE" == *"market_data"* ]]; then
    if ! grep -E "(API_KEY|authenticate|env::var)" "$FILE" >/dev/null 2>&1; then
        echo "⚠️  Missing API authentication in financial module"
        SYNTHETIC_SCORE=$((SYNTHETIC_SCORE + 5))
    fi
    
    if ! grep -E "(validate|verify|check.*integrity)" "$FILE" >/dev/null 2>&1; then
        echo "⚠️  Missing data validation in financial module"
        SYNTHETIC_SCORE=$((SYNTHETIC_SCORE + 5))
    fi
fi

# Calculate final score
if [ $SYNTHETIC_SCORE -ge 50 ]; then
    echo "❌ CRITICAL: High synthetic data risk (score: $SYNTHETIC_SCORE)"
    echo "   File contains critical synthetic data patterns"
    echo "   Action: Review and remove all synthetic data generation"
    exit 1
elif [ $SYNTHETIC_SCORE -ge 20 ]; then
    echo "⚠️  WARNING: Moderate synthetic data risk (score: $SYNTHETIC_SCORE)"
    echo "   File contains suspicious patterns"
    echo "   Recommendation: Review for potential synthetic data"
elif [ $SYNTHETIC_SCORE -gt 0 ]; then
    echo "ℹ️  INFO: Minor concerns detected (score: $SYNTHETIC_SCORE)"
else
    echo "✅ Post-edit analysis passed - clean file (score: 0)"
fi

# Log analysis result
echo "$(date -Iseconds) | $FILE | SCORE:$SYNTHETIC_SCORE" >> .cqgs/cqgs-alerts.log