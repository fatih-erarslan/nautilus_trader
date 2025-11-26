#!/bin/bash
# Iterative fix and validation loop
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "🔧 Iterative Fix and Validation Loop"
echo "====================================="

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

MAX_ITERATIONS=5
ITERATION=0

cd "${PROJECT_ROOT}"

while [ $ITERATION -lt $MAX_ITERATIONS ]; do
    ITERATION=$((ITERATION + 1))

    echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}Iteration ${ITERATION}/${MAX_ITERATIONS}${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    # Run validation
    echo -e "\n${YELLOW}Running validation suite...${NC}"

    if bash "${SCRIPT_DIR}/validate-all.sh" 2>&1 | tee "/tmp/validation-iter-${ITERATION}.log"; then
        echo -e "\n${GREEN}✅ All validations passed!${NC}"
        echo -e "${GREEN}Certification complete after ${ITERATION} iteration(s)${NC}"
        exit 0
    fi

    # Capture errors
    echo -e "\n${YELLOW}Analyzing failures...${NC}"

    ERRORS_FOUND=false

    # Check for build errors
    if grep -q "✗.*build\|✗.*compile" "/tmp/validation-iter-${ITERATION}.log"; then
        echo -e "${RED}Build errors detected${NC}"
        ERRORS_FOUND=true

        # Attempt automatic fixes
        echo -e "${YELLOW}Attempting to fix build issues...${NC}"

        # Clean and rebuild
        npm run clean 2>/dev/null || true
        rm -rf node_modules package-lock.json
        npm install
        npm run build || true

        # Rebuild Rust
        cd "${PROJECT_ROOT}/../../.."
        cargo clean --manifest-path neural-trader-rust/crates/mcp-server/Cargo.toml
        cargo build --release --manifest-path neural-trader-rust/crates/mcp-server/Cargo.toml || true
        cd "${PROJECT_ROOT}"
    fi

    # Check for test failures
    if grep -q "✗.*test" "/tmp/validation-iter-${ITERATION}.log"; then
        echo -e "${RED}Test failures detected${NC}"
        ERRORS_FOUND=true

        echo -e "${YELLOW}Test failures require manual review${NC}"
        echo "Please check the test output above and fix failing tests"
    fi

    # Check for MCP protocol issues
    if grep -q "✗.*MCP\|✗.*JSON-RPC\|✗.*tool" "/tmp/validation-iter-${ITERATION}.log"; then
        echo -e "${RED}MCP protocol issues detected${NC}"
        ERRORS_FOUND=true

        echo -e "${YELLOW}Checking tool registry...${NC}"
        if [ ! -f "tools/toolRegistry.json" ]; then
            echo "Creating tool registry..."
            mkdir -p tools
            echo '{"tools":[]}' > tools/toolRegistry.json
        fi
    fi

    # Check for Docker issues
    if grep -q "✗.*Docker\|✗.*container" "/tmp/validation-iter-${ITERATION}.log"; then
        echo -e "${RED}Docker issues detected${NC}"
        ERRORS_FOUND=true

        echo -e "${YELLOW}Docker issues may require manual intervention${NC}"
        echo "Check Docker installation and permissions"
    fi

    # Check for performance issues
    if grep -q "✗.*latency\|✗.*throughput\|✗.*memory" "/tmp/validation-iter-${ITERATION}.log"; then
        echo -e "${RED}Performance issues detected${NC}"
        ERRORS_FOUND=true

        echo -e "${YELLOW}Performance optimization needed${NC}"
        echo "Consider:"
        echo "  - Code optimization"
        echo "  - Caching improvements"
        echo "  - Resource limits"
    fi

    if [ "$ERRORS_FOUND" = false ]; then
        echo -e "${YELLOW}No specific errors detected, but validation failed${NC}"
        echo "Manual investigation required"
        break
    fi

    echo -e "\n${YELLOW}Fixes applied. Running next iteration...${NC}"
    sleep 2
done

# Generate final report
echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Fix and Validation Summary${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "Iterations completed: ${ITERATION}/${MAX_ITERATIONS}"
echo ""

if [ $ITERATION -eq $MAX_ITERATIONS ]; then
    echo -e "${RED}❌ Maximum iterations reached${NC}"
    echo -e "${RED}Some issues require manual intervention${NC}"
    echo ""
    echo "Manual fixes needed:"
    echo "1. Review validation logs in /tmp/validation-iter-*.log"
    echo "2. Fix failing tests"
    echo "3. Address MCP protocol compliance issues"
    echo "4. Optimize performance bottlenecks"
    echo ""
    echo "After manual fixes, run: bash scripts/validate-all.sh"
    exit 1
else
    echo -e "${YELLOW}Validation incomplete after ${ITERATION} iteration(s)${NC}"
    exit 1
fi
