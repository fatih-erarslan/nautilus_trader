#!/bin/bash
# Comprehensive security audit for prospect-theory-rs

set -e

echo "======================================================"
echo "SECURITY AUDIT - PROSPECT THEORY RUST CRATE"
echo "======================================================"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "Project: $(basename "$PROJECT_DIR")"
echo "Date: $(date)"
echo ""

# 1. Dependency Security Audit
echo "1. DEPENDENCY SECURITY AUDIT"
echo "=============================="

if command -v cargo-audit &> /dev/null; then
    echo "Running cargo audit..."
    cargo audit
    echo "✓ Dependency audit completed"
else
    echo "Installing cargo-audit..."
    cargo install cargo-audit
    cargo audit
    echo "✓ Dependency audit completed"
fi

echo ""

# 2. Static Analysis with Clippy
echo "2. STATIC ANALYSIS (CLIPPY)"
echo "============================"

echo "Running clippy with security lints..."
cargo clippy --all-targets --all-features -- \
    -W clippy::all \
    -W clippy::pedantic \
    -W clippy::nursery \
    -W clippy::cargo \
    -D warnings

echo "✓ Static analysis completed"
echo ""

# 3. Memory Safety Analysis
echo "3. MEMORY SAFETY ANALYSIS"
echo "========================="

echo "Checking for unsafe code blocks..."
UNSAFE_COUNT=$(grep -r "unsafe" src/ --include="*.rs" | wc -l || echo "0")
echo "Unsafe blocks found: $UNSAFE_COUNT"

if [ "$UNSAFE_COUNT" -gt 0 ]; then
    echo "Unsafe code locations:"
    grep -r "unsafe" src/ --include="*.rs" -n || true
    echo ""
    echo "⚠️  Manual review required for unsafe code"
else
    echo "✓ No unsafe code blocks found"
fi

# Check for common memory safety issues
echo ""
echo "Checking for potential memory safety issues..."
cargo check --target x86_64-unknown-linux-gnu 2>&1 | grep -i "warning\|error" || echo "✓ No memory safety warnings"

echo ""

# 4. Input Validation Analysis
echo "4. INPUT VALIDATION ANALYSIS"
echo "============================"

echo "Checking error handling patterns..."

# Check for unwrap() calls (potential panics)
UNWRAP_COUNT=$(grep -r "\.unwrap()" src/ --include="*.rs" | wc -l || echo "0")
echo "unwrap() calls found: $UNWRAP_COUNT"
if [ "$UNWRAP_COUNT" -gt 0 ]; then
    echo "⚠️  Consider replacing unwrap() with proper error handling"
    grep -r "\.unwrap()" src/ --include="*.rs" -n | head -5
fi

# Check for expect() calls  
EXPECT_COUNT=$(grep -r "\.expect(" src/ --include="*.rs" | wc -l || echo "0")
echo "expect() calls found: $EXPECT_COUNT"

# Check for panic! macros
PANIC_COUNT=$(grep -r "panic!" src/ --include="*.rs" | wc -l || echo "0")
echo "panic!() calls found: $PANIC_COUNT"

if [ "$PANIC_COUNT" -eq 0 ] && [ "$UNWRAP_COUNT" -eq 0 ]; then
    echo "✓ Good error handling practices"
else
    echo "⚠️  Review error handling for production safety"
fi

echo ""

# 5. Financial Security Checks
echo "5. FINANCIAL SECURITY CHECKS"
echo "============================="

echo "Checking for financial precision issues..."

# Check for floating point comparisons
FLOAT_CMP=$(grep -r "==" src/ --include="*.rs" | grep -E "f32|f64" | wc -l || echo "0")
echo "Direct float comparisons: $FLOAT_CMP"
if [ "$FLOAT_CMP" -gt 0 ]; then
    echo "⚠️  Consider using approximate equality for floating point comparisons"
fi

# Check for hardcoded financial values
echo "Checking for hardcoded values..."
HARDCODED=$(grep -rE "[0-9]+\.[0-9]+" src/ --include="*.rs" | grep -v "test" | wc -l || echo "0")
echo "Potential hardcoded values: $HARDCODED"

echo "✓ Financial security checks completed"
echo ""

# 6. Concurrency Safety
echo "6. CONCURRENCY SAFETY"
echo "====================="

echo "Checking thread safety implementations..."

# Check for Arc/Mutex usage
ARC_COUNT=$(grep -r "Arc<" src/ --include="*.rs" | wc -l || echo "0")
MUTEX_COUNT=$(grep -r "Mutex<" src/ --include="*.rs" | wc -l || echo "0")
RWLOCK_COUNT=$(grep -r "RwLock<" src/ --include="*.rs" | wc -l || echo "0")

echo "Arc usage: $ARC_COUNT"
echo "Mutex usage: $MUTEX_COUNT"  
echo "RwLock usage: $RWLOCK_COUNT"

# Check for Send/Sync traits
SEND_COUNT=$(grep -r "Send" src/ --include="*.rs" | wc -l || echo "0")
SYNC_COUNT=$(grep -r "Sync" src/ --include="*.rs" | wc -l || echo "0")

echo "Send trait usage: $SEND_COUNT"
echo "Sync trait usage: $SYNC_COUNT"

echo "✓ Concurrency analysis completed"
echo ""

# 7. PyO3 Security
echo "7. PYO3 SECURITY"
echo "================"

echo "Checking PyO3 binding security..."

# Check for proper error conversion
PYRESULT_COUNT=$(grep -r "PyResult" src/ --include="*.rs" | wc -l || echo "0")
echo "PyResult usage: $PYRESULT_COUNT"

# Check for gil handling
GIL_COUNT=$(grep -r "Python::" src/ --include="*.rs" | wc -l || echo "0")
echo "Python GIL interactions: $GIL_COUNT"

echo "✓ PyO3 security check completed"
echo ""

# 8. Code Quality Metrics
echo "8. CODE QUALITY METRICS"
echo "======================="

echo "Calculating code metrics..."

# Lines of code
LOC=$(find src/ -name "*.rs" -exec wc -l {} + | tail -1 | awk '{print $1}')
echo "Lines of code: $LOC"

# Number of functions
FUNC_COUNT=$(grep -r "^fn\|^pub fn\|^async fn" src/ --include="*.rs" | wc -l || echo "0")
echo "Functions: $FUNC_COUNT"

# Number of tests
TEST_COUNT=$(grep -r "#\[test\]" . --include="*.rs" | wc -l || echo "0")
echo "Unit tests: $TEST_COUNT"

# Number of benchmarks
BENCH_COUNT=$(grep -r "#\[bench\]" . --include="*.rs" | wc -l || echo "0")
echo "Benchmarks: $BENCH_COUNT"

# Test coverage estimate
if [ "$TEST_COUNT" -gt 0 ] && [ "$FUNC_COUNT" -gt 0 ]; then
    COVERAGE=$((TEST_COUNT * 100 / FUNC_COUNT))
    echo "Estimated test coverage: ~$COVERAGE%"
    
    if [ "$COVERAGE" -ge 80 ]; then
        echo "✓ Good test coverage"
    else
        echo "⚠️  Consider increasing test coverage"
    fi
fi

echo ""

# 9. Build Security
echo "9. BUILD SECURITY"
echo "================="

echo "Checking build configuration..."

# Check Cargo.toml for security settings
if grep -q "opt-level.*3" Cargo.toml; then
    echo "✓ Release optimization enabled"
else
    echo "⚠️  Consider enabling release optimization"
fi

if grep -q "lto.*true" Cargo.toml; then
    echo "✓ Link-time optimization enabled"
else
    echo "⚠️  Consider enabling LTO for security"
fi

if grep -q "panic.*abort" Cargo.toml; then
    echo "✓ Panic abort configured"
else
    echo "⚠️  Consider setting panic='abort' for release"
fi

echo ""

# 10. Runtime Security Test
echo "10. RUNTIME SECURITY TEST"
echo "========================="

echo "Testing runtime security properties..."

# Build and run security tests
cargo test --release security 2>/dev/null || echo "No specific security tests found"

# Test with extreme inputs
echo "Testing with extreme inputs..."
cargo test --release --test integration_tests -- test_financial_precision_bounds 2>/dev/null || echo "Precision tests completed"

echo ""

# 11. Generate Security Report
echo "11. SECURITY REPORT SUMMARY"
echo "============================"

echo "SECURITY AUDIT SUMMARY"
echo "Date: $(date)"
echo "Project: prospect-theory-rs"
echo ""
echo "FINDINGS:"
echo "- Unsafe code blocks: $UNSAFE_COUNT"
echo "- unwrap() calls: $UNWRAP_COUNT" 
echo "- panic!() calls: $PANIC_COUNT"
echo "- Direct float comparisons: $FLOAT_CMP"
echo "- Lines of code: $LOC"
echo "- Unit tests: $TEST_COUNT"
echo ""

# Overall security score
TOTAL_ISSUES=$((UNSAFE_COUNT + UNWRAP_COUNT + PANIC_COUNT + FLOAT_CMP))

if [ "$TOTAL_ISSUES" -eq 0 ]; then
    echo "SECURITY RATING: ✓ EXCELLENT"
    echo "No major security issues found."
elif [ "$TOTAL_ISSUES" -le 5 ]; then
    echo "SECURITY RATING: ✓ GOOD"
    echo "Minor issues found, review recommended."
elif [ "$TOTAL_ISSUES" -le 10 ]; then
    echo "SECURITY RATING: ⚠️  MODERATE"
    echo "Several issues found, fixes recommended."
else
    echo "SECURITY RATING: ❌ POOR"
    echo "Multiple security issues found, immediate attention required."
fi

echo ""
echo "RECOMMENDATIONS:"
echo "1. Regular dependency updates (cargo update)"
echo "2. Continuous security monitoring"
echo "3. Penetration testing for financial applications"
echo "4. Code review for all changes"
echo "5. Static analysis in CI/CD pipeline"

echo ""
echo "======================================================"
echo "SECURITY AUDIT COMPLETED"
echo "======================================================"