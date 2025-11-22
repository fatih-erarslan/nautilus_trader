#!/bin/bash
# CQGS Pre-Edit Hook - Synthetic Data Prevention
# Runs before any file edit to prevent synthetic data contamination

set -euo pipefail

FILE="$1"
TEMP_CONTENT="$2"

echo "🛡️  CQGS Pre-Edit Analysis: $FILE"

# Quick synthetic data detection
if grep -q "fastrand::" "$TEMP_CONTENT" 2>/dev/null; then
    echo "❌ CRITICAL: fastrand usage detected in edit - BLOCKED"
    echo "   File: $FILE"
    echo "   Rule: Synthetic Data Prohibition (SDP-001)"
    echo "   Action: Edit blocked to prevent synthetic data contamination"
    exit 1
fi

# Check for synthetic patterns
if grep -E "(fake|synthetic|mock|dummy).*(market|data|price)" "$TEMP_CONTENT" 2>/dev/null; then
    echo "⚠️  WARNING: Synthetic data patterns detected"
    echo "   File: $FILE"
    echo "   Recommendation: Use real market data APIs instead"
fi

# Check for random multipliers (common synthetic pattern)
if grep -E "\*\s*\(.*random.*\)" "$TEMP_CONTENT" 2>/dev/null; then
    echo "❌ CRITICAL: Random multiplier pattern detected - BLOCKED"
    echo "   File: $FILE"
    echo "   Pattern: Multiplication with random values"
    echo "   Rule: Zero Synthetic Data Policy"
    exit 1
fi

# Check for hardcoded credentials
if grep -E "(password|api_key|secret|token)\s*=\s*\"[^\"]+\"" "$TEMP_CONTENT" 2>/dev/null; then
    echo "❌ CRITICAL: Hardcoded credentials detected - BLOCKED"
    echo "   File: $FILE"
    echo "   Rule: Security Compliance (SEC-001)"
    echo "   Action: Use environment variables instead"
    exit 1
fi

echo "✅ Pre-edit analysis passed - no synthetic data patterns detected"