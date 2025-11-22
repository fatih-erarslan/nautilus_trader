#!/bin/bash
# CQGS Pre-Commit Hook - Final Synthetic Data Check
# Comprehensive analysis before commit to prevent synthetic data from entering repo

set -euo pipefail

echo "🚨 CQGS Pre-Commit Analysis - Final Synthetic Data Check"

# Initialize counters
TOTAL_FILES=0
CLEAN_FILES=0
WARNING_FILES=0
CRITICAL_FILES=0

# Create reports directory
mkdir -p .cqgs/reports

# Scan all staged files
for file in $(git diff --cached --name-only --diff-filter=ACM | grep -E '\.(rs|toml)$' || true); do
    if [[ -f "$file" ]]; then
        TOTAL_FILES=$((TOTAL_FILES + 1))
        echo "Analyzing: $file"
        
        # Run post-edit analysis
        if .cqgs/hooks/post-edit.sh "$file" > /tmp/cqgs-analysis 2>&1; then
            if grep -q "score: 0" /tmp/cqgs-analysis; then
                CLEAN_FILES=$((CLEAN_FILES + 1))
            else
                WARNING_FILES=$((WARNING_FILES + 1))
                echo "⚠️  Warnings in $file:"
                cat /tmp/cqgs-analysis | grep -E "⚠️|ℹ️"
            fi
        else
            CRITICAL_FILES=$((CRITICAL_FILES + 1))
            echo "❌ CRITICAL issues in $file:"
            cat /tmp/cqgs-analysis | grep -E "❌|CRITICAL"
        fi
    fi
done

# Generate commit report
REPORT_FILE=".cqgs/reports/commit-$(date +%Y%m%d-%H%M%S).md"
cat > "$REPORT_FILE" << EOF
# CQGS Pre-Commit Analysis Report

**Timestamp:** $(date -Iseconds)  
**Commit:** $(git rev-parse --short HEAD 2>/dev/null || echo "pending")

## Summary

- **Total Files Analyzed:** $TOTAL_FILES
- **Clean Files:** $CLEAN_FILES ✅
- **Files with Warnings:** $WARNING_FILES ⚠️
- **Files with Critical Issues:** $CRITICAL_FILES ❌

## Analysis Results

$(if [ $CRITICAL_FILES -gt 0 ]; then
    echo "### ❌ CRITICAL ISSUES DETECTED"
    echo "The following files contain critical synthetic data patterns:"
    echo ""
    for file in $(git diff --cached --name-only --diff-filter=ACM | grep -E '\.(rs|toml)$' || true); do
        if [[ -f "$file" ]] && .cqgs/hooks/post-edit.sh "$file" > /tmp/check 2>&1; then
            if ! grep -q "score: 0" /tmp/check; then
                echo "- $file"
            fi
        fi
    done
    echo ""
    echo "**ACTION REQUIRED:** Fix critical issues before committing"
else
    echo "### ✅ NO CRITICAL ISSUES"
    echo "All files passed synthetic data analysis"
fi)

## Compliance Status

- **Synthetic Data Policy:** $(if [ $CRITICAL_FILES -eq 0 ]; then echo "✅ COMPLIANT"; else echo "❌ VIOLATION"; fi)
- **Security Policy:** $(if grep -r "password.*=" . --include="*.rs" 2>/dev/null | grep -v ".cqgs" >/dev/null; then echo "❌ VIOLATION"; else echo "✅ COMPLIANT"; fi)
- **Real Data Requirement:** $(if [ $WARNING_FILES -eq 0 ]; then echo "✅ COMPLIANT"; else echo "⚠️ REVIEW NEEDED"; fi)

EOF

echo ""
echo "📊 CQGS Analysis Summary:"
echo "   Total Files: $TOTAL_FILES"
echo "   Clean: $CLEAN_FILES"
echo "   Warnings: $WARNING_FILES"
echo "   Critical: $CRITICAL_FILES"
echo ""

# Block commit if critical issues found
if [ $CRITICAL_FILES -gt 0 ]; then
    echo "❌ COMMIT BLOCKED - Critical synthetic data issues detected"
    echo "   Files with critical issues: $CRITICAL_FILES"
    echo "   Action: Fix all critical issues before committing"
    echo "   Report: $REPORT_FILE"
    exit 1
fi

# Warn about warnings but allow commit
if [ $WARNING_FILES -gt 0 ]; then
    echo "⚠️  COMMIT ALLOWED - Warnings detected"
    echo "   Files with warnings: $WARNING_FILES"
    echo "   Recommendation: Review warnings when possible"
    echo "   Report: $REPORT_FILE"
fi

echo "✅ CQGS Analysis Passed - Commit approved"
echo "   Report: $REPORT_FILE"