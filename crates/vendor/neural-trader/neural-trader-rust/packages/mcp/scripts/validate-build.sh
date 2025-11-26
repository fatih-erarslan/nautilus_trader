#!/bin/bash
# Level 1: Build Validation
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "🔨 Level 1: Build Validation"
echo "=============================="

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

ERRORS=0
WARNINGS=0

# 1. Check Rust crates compile
echo -e "\n${YELLOW}1.1 Checking Rust crates...${NC}"
cd "${PROJECT_ROOT}/../../.."

if cargo build --release --manifest-path neural-trader-rust/crates/mcp-server/Cargo.toml 2>&1 | tee /tmp/rust-build.log; then
    echo -e "${GREEN}✓ Rust crates compile successfully${NC}"
else
    echo -e "${RED}✗ Rust compilation failed${NC}"
    ERRORS=$((ERRORS + 1))
    cat /tmp/rust-build.log
fi

# Check for warnings
if grep -q "warning:" /tmp/rust-build.log; then
    WARN_COUNT=$(grep -c "warning:" /tmp/rust-build.log || echo "0")
    echo -e "${YELLOW}⚠ Found ${WARN_COUNT} compiler warnings${NC}"
    WARNINGS=$((WARNINGS + WARN_COUNT))
fi

# 2. Check NPM packages build
echo -e "\n${YELLOW}1.2 Checking NPM package build...${NC}"
cd "${PROJECT_ROOT}"

if npm run build 2>&1 | tee /tmp/npm-build.log; then
    echo -e "${GREEN}✓ NPM package builds successfully${NC}"
else
    echo -e "${RED}✗ NPM build failed${NC}"
    ERRORS=$((ERRORS + 1))
    cat /tmp/npm-build.log
fi

# 3. Check NAPI binaries created
echo -e "\n${YELLOW}1.3 Checking NAPI binaries...${NC}"
NAPI_DIR="${PROJECT_ROOT}/../../crates/napi-bindings"

if [ -d "${NAPI_DIR}/target/release" ]; then
    BINARY_COUNT=$(find "${NAPI_DIR}/target/release" -name "*.node" 2>/dev/null | wc -l || echo "0")
    if [ "$BINARY_COUNT" -gt 0 ]; then
        echo -e "${GREEN}✓ Found ${BINARY_COUNT} NAPI binary files${NC}"
    else
        echo -e "${YELLOW}⚠ No NAPI binaries found (may need to build)${NC}"
        WARNINGS=$((WARNINGS + 1))
    fi
else
    echo -e "${YELLOW}⚠ NAPI bindings directory not found${NC}"
    WARNINGS=$((WARNINGS + 1))
fi

# 4. Check dependencies resolve
echo -e "\n${YELLOW}1.4 Checking dependency resolution...${NC}"
cd "${PROJECT_ROOT}"

if npm ls --depth=0 2>&1 | tee /tmp/npm-deps.log; then
    echo -e "${GREEN}✓ Dependencies resolve correctly${NC}"
else
    echo -e "${RED}✗ Dependency resolution issues found${NC}"
    ERRORS=$((ERRORS + 1))
    cat /tmp/npm-deps.log
fi

# 5. Check TypeScript compilation
echo -e "\n${YELLOW}1.5 Checking TypeScript compilation...${NC}"
if [ -f "${PROJECT_ROOT}/tsconfig.json" ]; then
    if npx tsc --noEmit 2>&1 | tee /tmp/tsc-check.log; then
        echo -e "${GREEN}✓ TypeScript compiles without errors${NC}"
    else
        echo -e "${RED}✗ TypeScript compilation errors${NC}"
        ERRORS=$((ERRORS + 1))
        cat /tmp/tsc-check.log
    fi
else
    echo -e "${YELLOW}⚠ No tsconfig.json found${NC}"
fi

# Summary
echo -e "\n=============================="
echo "Level 1 Summary:"
echo "  Errors: $ERRORS"
echo "  Warnings: $WARNINGS"

if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}✅ Level 1: PASSED${NC}"
    exit 0
else
    echo -e "${RED}❌ Level 1: FAILED${NC}"
    exit 1
fi
