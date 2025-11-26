#!/bin/bash
# Main validation script - runs all levels
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "🔍 Neural Trader MCP - Full Validation Suite"
echo "============================================="
echo ""

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

TOTAL_LEVELS=6
PASSED_LEVELS=0
FAILED_LEVELS=0

# Array to track results
declare -a LEVEL_RESULTS

run_level() {
    local level=$1
    local script=$2
    local name=$3

    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}Running Level ${level}: ${name}${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""

    if bash "${SCRIPT_DIR}/${script}"; then
        LEVEL_RESULTS[$level]="PASSED"
        PASSED_LEVELS=$((PASSED_LEVELS + 1))
        echo -e "\n${GREEN}✅ Level ${level} PASSED${NC}\n"
        return 0
    else
        LEVEL_RESULTS[$level]="FAILED"
        FAILED_LEVELS=$((FAILED_LEVELS + 1))
        echo -e "\n${RED}❌ Level ${level} FAILED${NC}\n"
        return 1
    fi
}

# Level 1: Build Validation
run_level 1 "validate-build.sh" "Build Validation" || true

# Level 2: Unit Tests
run_level 2 "validate-tests.sh" "Unit Tests" || true

# Level 3: MCP Protocol Compliance
run_level 3 "validate-mcp.sh" "MCP Protocol Compliance" || true

# Level 4: End-to-End Testing
run_level 4 "validate-e2e.sh" "End-to-End Testing" || true

# Level 5: Docker Validation
run_level 5 "validate-docker.sh" "Docker Validation" || true

# Level 6: Performance Validation
run_level 6 "validate-performance.sh" "Performance Validation" || true

# Final Summary
echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}FINAL VALIDATION SUMMARY${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

for i in {1..6}; do
    result=${LEVEL_RESULTS[$i]}
    if [ "$result" = "PASSED" ]; then
        echo -e "  Level $i: ${GREEN}✅ PASSED${NC}"
    else
        echo -e "  Level $i: ${RED}❌ FAILED${NC}"
    fi
done

echo ""
echo "Results:"
echo "  Total Levels: $TOTAL_LEVELS"
echo "  Passed: $PASSED_LEVELS"
echo "  Failed: $FAILED_LEVELS"

PASS_RATE=$(( PASSED_LEVELS * 100 / TOTAL_LEVELS ))
echo "  Pass Rate: ${PASS_RATE}%"

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

if [ $FAILED_LEVELS -eq 0 ]; then
    echo -e "${GREEN}🎉 CERTIFICATION: PASSED${NC}"
    echo -e "${GREEN}All validation levels completed successfully!${NC}"
    exit 0
else
    echo -e "${RED}❌ CERTIFICATION: FAILED${NC}"
    echo -e "${RED}${FAILED_LEVELS} level(s) failed validation${NC}"
    echo ""
    echo "Please review the output above and run the fix script:"
    echo "  bash scripts/fix-and-validate.sh"
    exit 1
fi
