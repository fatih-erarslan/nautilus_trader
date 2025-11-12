#!/bin/bash
# ============================================
# HyperPhysics Scientific Validation Script
# Version: 1.0
# Date: 2025-11-12
# Authority: Queen Seraphina
# ============================================

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  HyperPhysics Scientific Validation       ║${NC}"
echo -e "${BLUE}║  PRINCIPLE 0 ACTIVATION: Scientific Rigor  ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════╝${NC}"
echo ""

# Configuration
PROJECT_ROOT="/Users/ashina/Desktop/Kurultay/HyperPhysics"
SRC_DIR="$PROJECT_ROOT/src"
TEST_DIR="$PROJECT_ROOT/tests"
DOCS_DIR="$PROJECT_ROOT/docs/scientific"

# Check if we're in the right directory
if [ ! -d "$SRC_DIR" ]; then
    echo -e "${RED}❌ ERROR: Source directory not found at $SRC_DIR${NC}"
    exit 1
fi

# Initialize counters
FAIL_COUNT=0
CITATION_COUNT=0
DATA_SOURCE_COUNT=0
TOTAL_SCORE=0

echo -e "${YELLOW}Phase 1: Forbidden Pattern Detection${NC}"
echo "======================================"

# Forbidden patterns that cause instant failure
declare -a FORBIDDEN_PATTERNS=(
    "np.random.normal"
    "np.random.uniform"
    "np.random.rand"
    "random.random"
    "random.uniform"
    "mock."
    "MockData"
    "TODO"
    "FIXME"
    "placeholder"
    "hardcoded"
    "dummy"
    "test_data"
    "synthetic_"
    "generate_fake"
    "lorem_ipsum"
)

# Check for forbidden patterns
for pattern in "${FORBIDDEN_PATTERNS[@]}"; do
    # Search in Python files only
    if [ -d "$SRC_DIR" ]; then
        COUNT=$(grep -r "$pattern" "$SRC_DIR" --include="*.py" 2>/dev/null | wc -l | tr -d ' ')
        if [ "$COUNT" -gt 0 ]; then
            echo -e "${RED}❌ CRITICAL: Found $COUNT instances of '$pattern'${NC}"
            grep -rn "$pattern" "$SRC_DIR" --include="*.py" 2>/dev/null | head -5
            FAIL_COUNT=$((FAIL_COUNT + COUNT))
        fi
    fi
done

if [ $FAIL_COUNT -gt 0 ]; then
    echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${RED}🚫 GATE 0 FAILURE: $FAIL_COUNT forbidden patterns detected${NC}"
    echo -e "${RED}   Score: 0/100 (Automatic Failure)${NC}"
    echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo "Required Actions:"
    echo "1. Remove ALL synthetic/mock data generators"
    echo "2. Replace with real data sources (APIs/historical files)"
    echo "3. Complete TODO/FIXME items"
    echo "4. Re-run validation"
    exit 1
else
    echo -e "${GREEN}✅ GATE 0 PASSED: Zero forbidden patterns detected${NC}"
fi

echo ""
echo -e "${YELLOW}Phase 2: Scientific Citation Verification${NC}"
echo "======================================"

# Required peer-reviewed citations
declare -a REQUIRED_CITATIONS=(
    "Krioukov"      # Hyperbolic geometry
    "Landauer"      # Thermodynamics
    "Gillespie"     # Stochastic algorithms
    "Tononi"        # Integrated information
    "Sagawa"        # Information thermodynamics
    "Mantegna"      # Econophysics
    "Bennett"       # Reversible computation
    "Metropolis"    # MCMC
    "Farmer"        # Agent-based models
    "Boguñá"        # Network geometry
)

for citation in "${REQUIRED_CITATIONS[@]}"; do
    if [ -d "$SRC_DIR" ]; then
        COUNT=$(grep -r "$citation" "$SRC_DIR" --include="*.py" 2>/dev/null | wc -l | tr -d ' ')
        if [ "$COUNT" -gt 0 ]; then
            echo -e "${GREEN}✅ Found citations to $citation ($COUNT instances)${NC}"
            CITATION_COUNT=$((CITATION_COUNT + 1))
        else
            echo -e "${YELLOW}⚠️  WARNING: No citations to $citation found${NC}"
        fi
    fi
done

CITATION_SCORE=$((CITATION_COUNT * 10))
echo -e "${BLUE}📚 Citation Score: $CITATION_COUNT/10 required authors (${CITATION_SCORE}/100)${NC}"

if [ $CITATION_COUNT -lt 5 ]; then
    echo -e "${YELLOW}⚠️  Insufficient scientific citations. Minimum 5 required.${NC}"
fi

echo ""
echo -e "${YELLOW}Phase 3: Real Data Source Detection${NC}"
echo "======================================"

# Real data source APIs and libraries
declare -a DATA_SOURCES=(
    "yfinance"
    "alpha_vantage"
    "bloomberg"
    "reuters"
    "iex"
    "fred"
    "pandas_datareader"
    "quandl"
)

for source in "${DATA_SOURCES[@]}"; do
    if [ -d "$SRC_DIR" ]; then
        COUNT=$(grep -r "$source" "$SRC_DIR" --include="*.py" 2>/dev/null | wc -l | tr -d ' ')
        if [ "$COUNT" -gt 0 ]; then
            echo -e "${GREEN}✅ Found $source integration ($COUNT instances)${NC}"
            DATA_SOURCE_COUNT=$((DATA_SOURCE_COUNT + 1))
        fi
    fi
done

if [ $DATA_SOURCE_COUNT -eq 0 ]; then
    echo -e "${RED}❌ CRITICAL: No real data source APIs detected${NC}"
    echo -e "${RED}   All financial data MUST come from real sources${NC}"
else
    echo -e "${GREEN}✅ Found $DATA_SOURCE_COUNT real data source(s)${NC}"
fi

echo ""
echo -e "${YELLOW}Phase 4: Algorithm Implementation Verification${NC}"
echo "======================================"

# Check for specific algorithm implementations
declare -A ALGORITHMS=(
    ["mercator"]="Hyperbolic embedding (Boguñá et al. 2019)"
    ["poincare"]="Poincaré disc model (Krioukov et al. 2010)"
    ["landauer"]="Landauer bound (Landauer 1961)"
    ["jarzynski"]="Jarzynski equality (Sagawa & Ueda 2010)"
    ["gillespie"]="Gillespie SSA (Gillespie 1977)"
    ["metropolis"]="Metropolis-Hastings MCMC (1953/1970)"
    ["pyphi"]="Integrated information (Tononi et al. 2014)"
)

ALGORITHM_COUNT=0
for algo in "${!ALGORITHMS[@]}"; do
    if [ -d "$SRC_DIR" ]; then
        COUNT=$(grep -ri "$algo" "$SRC_DIR" --include="*.py" 2>/dev/null | wc -l | tr -d ' ')
        if [ "$COUNT" -gt 0 ]; then
            echo -e "${GREEN}✅ ${ALGORITHMS[$algo]} detected${NC}"
            ALGORITHM_COUNT=$((ALGORITHM_COUNT + 1))
        else
            echo -e "${YELLOW}⚠️  ${ALGORITHMS[$algo]} not found${NC}"
        fi
    fi
done

echo -e "${BLUE}🧮 Algorithm Score: $ALGORITHM_COUNT/${#ALGORITHMS[@]} implemented${NC}"

echo ""
echo -e "${YELLOW}Phase 5: Test Coverage Analysis${NC}"
echo "======================================"

# Check if pytest is available
if command -v pytest &> /dev/null; then
    if [ -d "$TEST_DIR" ]; then
        echo "Running pytest coverage analysis..."
        cd "$PROJECT_ROOT"

        # Run pytest with coverage (suppress verbose output)
        COVERAGE_OUTPUT=$(pytest --cov="$SRC_DIR" --cov-report=term-missing --cov-fail-under=0 -q 2>&1 || true)

        # Extract coverage percentage
        COVERAGE_PCT=$(echo "$COVERAGE_OUTPUT" | grep -oP "TOTAL.*\K[0-9]+" | tail -1 || echo "0")

        if [ -z "$COVERAGE_PCT" ]; then
            COVERAGE_PCT=0
        fi

        echo -e "${BLUE}📊 Test Coverage: ${COVERAGE_PCT}%${NC}"

        if [ "$COVERAGE_PCT" -ge 90 ]; then
            echo -e "${GREEN}✅ Coverage exceeds 90% threshold${NC}"
        elif [ "$COVERAGE_PCT" -ge 70 ]; then
            echo -e "${YELLOW}⚠️  Coverage below 90% target (current: ${COVERAGE_PCT}%)${NC}"
        else
            echo -e "${RED}❌ Coverage critically low (< 70%)${NC}"
        fi
    else
        echo -e "${YELLOW}⚠️  Test directory not found${NC}"
        COVERAGE_PCT=0
    fi
else
    echo -e "${YELLOW}⚠️  pytest not installed, skipping coverage analysis${NC}"
    COVERAGE_PCT=0
fi

echo ""
echo -e "${YELLOW}Phase 6: Documentation Check${NC}"
echo "======================================"

# Check for required documentation files
declare -a REQUIRED_DOCS=(
    "$DOCS_DIR/LITERATURE_REVIEW.md"
    "$DOCS_DIR/REFERENCES.bib"
    "$DOCS_DIR/VALIDATION_CHECKLIST.md"
)

DOC_COUNT=0
for doc in "${REQUIRED_DOCS[@]}"; do
    if [ -f "$doc" ]; then
        echo -e "${GREEN}✅ Found $(basename $doc)${NC}"
        DOC_COUNT=$((DOC_COUNT + 1))
    else
        echo -e "${RED}❌ Missing $(basename $doc)${NC}"
    fi
done

echo ""
echo -e "${BLUE}╔════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║          VALIDATION SUMMARY                ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════╝${NC}"
echo ""

# Calculate overall score
SCIENTIFIC_RIGOR=$((CITATION_COUNT * 8 + ALGORITHM_COUNT * 6))  # Max 100
ARCHITECTURE=$((DATA_SOURCE_COUNT * 20))  # Max 100 (5 sources)
QUALITY=$COVERAGE_PCT  # Max 100
DOCUMENTATION=$((DOC_COUNT * 33))  # Max 100 (3 docs)

# Ensure scores don't exceed 100
[ $SCIENTIFIC_RIGOR -gt 100 ] && SCIENTIFIC_RIGOR=100
[ $ARCHITECTURE -gt 100 ] && ARCHITECTURE=100
[ $DOCUMENTATION -gt 100 ] && DOCUMENTATION=100

# Weighted total (using simplified weights)
TOTAL_SCORE=$(( (SCIENTIFIC_RIGOR * 25 + ARCHITECTURE * 20 + QUALITY * 20 + DOCUMENTATION * 10) / 75 ))

echo -e "${BLUE}📊 Dimension Scores:${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
printf "%-25s %3d/100 (Weight: 25%%)\n" "Scientific Rigor:" $SCIENTIFIC_RIGOR
printf "%-25s %3d/100 (Weight: 20%%)\n" "Architecture:" $ARCHITECTURE
printf "%-25s %3d/100 (Weight: 20%%)\n" "Quality:" $QUALITY
printf "%-25s %3d/100 (Weight: 10%%)\n" "Documentation:" $DOCUMENTATION
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
printf "%-25s %3d/100\n" "TOTAL SCORE:" $TOTAL_SCORE
echo ""

# Gate evaluation
echo -e "${BLUE}🚪 Gate Evaluation:${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ $FAIL_COUNT -gt 0 ]; then
    echo -e "${RED}🚫 GATE 0: FAILED - Forbidden patterns detected${NC}"
    echo -e "${RED}   Action: Remove all synthetic/mock data${NC}"
    EXIT_CODE=1
elif [ $TOTAL_SCORE -lt 60 ]; then
    echo -e "${RED}🚫 GATE 1: FAILED - Score < 60 (requires redesign)${NC}"
    echo -e "${RED}   Action: Complete redesign required${NC}"
    EXIT_CODE=1
elif [ $TOTAL_SCORE -lt 80 ]; then
    echo -e "${YELLOW}⚠️  GATE 2: PASSED - Score ≥ 60 (integration allowed)${NC}"
    echo -e "${YELLOW}   Action: Implement improvements before production${NC}"
    EXIT_CODE=0
elif [ $TOTAL_SCORE -lt 95 ]; then
    echo -e "${GREEN}✅ GATE 3: PASSED - Score ≥ 80 (testing phase)${NC}"
    echo -e "${GREEN}   Action: Proceed to comprehensive testing${NC}"
    EXIT_CODE=0
else
    echo -e "${GREEN}🎯 GATE 4: PASSED - Score ≥ 95 (production candidate)${NC}"
    echo -e "${GREEN}   Action: Formal verification and deployment${NC}"
    EXIT_CODE=0
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Recommendations
echo -e "${BLUE}📋 Recommendations:${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ $CITATION_COUNT -lt 5 ]; then
    echo -e "${YELLOW}1. Add scientific citations (current: $CITATION_COUNT, target: 5+)${NC}"
fi

if [ $DATA_SOURCE_COUNT -eq 0 ]; then
    echo -e "${RED}2. CRITICAL: Integrate real data sources (APIs/datasets)${NC}"
fi

if [ $COVERAGE_PCT -lt 90 ]; then
    echo -e "${YELLOW}3. Increase test coverage to 90%+ (current: ${COVERAGE_PCT}%)${NC}"
fi

if [ $ALGORITHM_COUNT -lt 5 ]; then
    echo -e "${YELLOW}4. Implement missing peer-reviewed algorithms${NC}"
fi

if [ $DOC_COUNT -lt 3 ]; then
    echo -e "${YELLOW}5. Complete scientific documentation${NC}"
fi

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Validation Report: /tmp/hyperphysics_validation_$(date +%Y%m%d_%H%M%S).log${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# Save detailed report
REPORT_FILE="/tmp/hyperphysics_validation_$(date +%Y%m%d_%H%M%S).log"
{
    echo "HyperPhysics Scientific Validation Report"
    echo "Date: $(date)"
    echo "Score: $TOTAL_SCORE/100"
    echo ""
    echo "Forbidden Patterns: $FAIL_COUNT"
    echo "Citations: $CITATION_COUNT/10"
    echo "Data Sources: $DATA_SOURCE_COUNT"
    echo "Algorithms: $ALGORITHM_COUNT/${#ALGORITHMS[@]}"
    echo "Test Coverage: ${COVERAGE_PCT}%"
    echo "Documentation: $DOC_COUNT/3"
} > "$REPORT_FILE"

echo "Report saved to: $REPORT_FILE"

exit $EXIT_CODE
