#!/bin/bash
#
# System Validation Script
# Comprehensive validation of HyperPhysics system
#

set -e

echo "🔍 HyperPhysics System Validation"
echo "=================================="
echo ""

cd "$(dirname "$0")/.."

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

PASSED=0
FAILED=0

check_result() {
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ PASS${NC}: $1"
        ((PASSED++))
    else
        echo -e "${RED}✗ FAIL${NC}: $1"
        ((FAILED++))
    fi
}

# 1. Build Check
echo "1️⃣  Build Test..."
cargo build --workspace --all-features --quiet 2>&1 > /dev/null
check_result "Workspace builds successfully"

# 2. Test Suite
echo "2️⃣  Test Suite..."
TEST_OUTPUT=$(cargo test --workspace --quiet 2>&1)
echo "$TEST_OUTPUT" | grep -q "test result: ok"
check_result "All tests pass"

# Extract test count
TEST_COUNT=$(echo "$TEST_OUTPUT" | grep -oP '\d+(?= passed)' | tail -1)
echo "   Tests passed: $TEST_COUNT"

# 3. No Warnings
echo "3️⃣  Compiler Warnings..."
WARNINGS=$(cargo build --workspace --all-features 2>&1 | grep -c "warning:" || true)
if [ "$WARNINGS" -eq 0 ]; then
    check_result "Zero compiler warnings"
else
    echo -e "${YELLOW}⚠ ${NC} $WARNINGS warnings found"
    ((FAILED++))
fi

# 4. Forbidden Patterns
echo "4️⃣  Forbidden Patterns..."
FORBIDDEN_COUNT=0
for pattern in "np.random" "random.random" "mock." "TODO" "FIXME" "dummy" "placeholder"; do
    COUNT=$(grep -r "$pattern" crates/ --include="*.rs" | wc -l)
    FORBIDDEN_COUNT=$((FORBIDDEN_COUNT + COUNT))
done

if [ "$FORBIDDEN_COUNT" -eq 0 ]; then
    check_result "No forbidden patterns (mock data, TODOs)"
else
    echo -e "${RED}✗ FAIL${NC}: Found $FORBIDDEN_COUNT forbidden patterns"
    ((FAILED++))
fi

# 5. Documentation Coverage
echo "5️⃣  Documentation..."
cargo doc --workspace --no-deps --quiet 2>&1 > /dev/null
check_result "Documentation builds"

# 6. Clippy Lints
echo "6️⃣  Clippy Lints..."
cargo clippy --workspace --all-features --quiet 2>&1 > /dev/null
check_result "Clippy passes"

# 7. Format Check
echo "7️⃣  Code Formatting..."
cargo fmt --all -- --check 2>&1 > /dev/null
check_result "Code is formatted"

# 8. Dependency Audit
echo "8️⃣  Security Audit..."
if command -v cargo-audit &> /dev/null; then
    cargo audit --quiet 2>&1 > /dev/null
    check_result "No security vulnerabilities"
else
    echo -e "${YELLOW}⚠ ${NC} cargo-audit not installed (skipped)"
fi

# 9. Benchmark Compilation
echo "9️⃣  Benchmarks..."
cargo bench --workspace --no-run --quiet 2>&1 > /dev/null
check_result "Benchmarks compile"

# 10. Feature Combinations
echo "🔟 Feature Flags..."
cargo build --workspace --no-default-features --quiet 2>&1 > /dev/null
check_result "Builds without default features"

cargo build --workspace --all-features --quiet 2>&1 > /dev/null
check_result "Builds with all features"

# Summary
echo ""
echo "=================================="
echo "📊 Validation Summary"
echo "=================================="
echo -e "${GREEN}Passed: $PASSED${NC}"
echo -e "${RED}Failed: $FAILED${NC}"
echo ""

if [ "$FAILED" -eq 0 ]; then
    echo -e "${GREEN}✅ All validation checks passed!${NC}"
    echo ""
    echo "System Status: PRODUCTION READY"
    exit 0
else
    echo -e "${RED}❌ Some validation checks failed${NC}"
    echo ""
    echo "System Status: NEEDS ATTENTION"
    exit 1
fi
