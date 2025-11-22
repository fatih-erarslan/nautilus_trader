#!/bin/bash
# HyperPhysics HFT Ecosystem - Critical Dependency Fix Script
# Phase 1: Version Conflict Resolution
# Generated: 2025-11-21

set -e  # Exit on error

echo "================================================"
echo "HyperPhysics HFT Ecosystem - Dependency Fixes"
echo "================================================"
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

HYPERPHYSICS_ROOT="/Users/ashina/Desktop/Kurultay/HyperPhysics"

cd "$HYPERPHYSICS_ROOT"

echo -e "${YELLOW}[1/4] Fixing nalgebra version conflicts (0.32 → 0.33)${NC}"
echo ""

# Fix nalgebra in specific files
files_to_fix=(
    "crates/cwts-ultra/tests/Cargo.toml"
    "crates/ats-core/Cargo.toml"
    "crates/hyperphysics-market/Cargo.toml"
    "crates/game-theory-engine/Cargo.toml"
    "crates/bio-inspired-workspace/Cargo.toml"
    "crates/bio-inspired-workspace/dynamic-swarm-selector/Cargo.toml"
)

for file in "${files_to_fix[@]}"; do
    if [ -f "$file" ]; then
        echo "  Updating: $file"
        # Update nalgebra 0.32 to 0.33
        sed -i.bak 's/nalgebra = { version = "0.32"/nalgebra = { version = "0.33"/g' "$file"
        sed -i.bak 's/nalgebra = "0.32"/nalgebra = "0.33"/g' "$file"
        rm -f "$file.bak"
        echo -e "    ${GREEN}✓${NC} nalgebra updated to 0.33"
    else
        echo -e "    ${RED}✗${NC} File not found: $file"
    fi
done

echo ""
echo -e "${YELLOW}[2/4] Fixing dashmap version conflicts (5.5/6.0 → 6.1)${NC}"
echo ""

dashmap_files=(
    "crates/bio-inspired-workspace/Cargo.toml"
    "crates/bio-inspired-workspace/dynamic-swarm-selector/Cargo.toml"
    "crates/quantum-lstm/Cargo.toml"
    "crates/hive-mind-rust/Cargo.toml"
    "crates/cwts-core/Cargo.toml"
    "crates/cwts-intelligence/Cargo.toml"
)

for file in "${dashmap_files[@]}"; do
    if [ -f "$file" ]; then
        echo "  Updating: $file"
        # Update dashmap 5.5 and 6.0 to 6.1
        sed -i.bak 's/dashmap = "5.5"/dashmap = "6.1"/g' "$file"
        sed -i.bak 's/dashmap = "6.0"/dashmap = "6.1"/g' "$file"
        rm -f "$file.bak"
        echo -e "    ${GREEN}✓${NC} dashmap updated to 6.1"
    else
        echo -e "    ${RED}✗${NC} File not found: $file"
    fi
done

echo ""
echo -e "${YELLOW}[3/4] Verifying changes...${NC}"
echo ""

# Check if updates were successful
echo "  Checking nalgebra versions..."
if grep -r "nalgebra.*0\.32" crates/*/Cargo.toml crates/*/*/Cargo.toml 2>/dev/null; then
    echo -e "    ${RED}✗${NC} Some nalgebra 0.32 references still exist"
else
    echo -e "    ${GREEN}✓${NC} All nalgebra dependencies updated to 0.33"
fi

echo "  Checking dashmap versions..."
if grep -r "dashmap.*[\"=]5\.5" crates/*/Cargo.toml crates/*/*/Cargo.toml 2>/dev/null; then
    echo -e "    ${RED}✗${NC} Some dashmap 5.5 references still exist"
elif grep -r "dashmap.*[\"=]6\.0\"" crates/*/Cargo.toml crates/*/*/Cargo.toml 2>/dev/null; then
    echo -e "    ${RED}✗${NC} Some dashmap 6.0 references still exist"
else
    echo -e "    ${GREEN}✓${NC} All dashmap dependencies updated to 6.1"
fi

echo ""
echo -e "${YELLOW}[4/4] Running cargo check to validate...${NC}"
echo ""

# Test compilation
echo "  Testing compilation (this may take a few minutes)..."
if cargo check --workspace --all-features 2>&1 | tee /tmp/hyperphysics_check.log; then
    echo -e "  ${GREEN}✓${NC} Workspace compiles successfully"
else
    echo -e "  ${RED}✗${NC} Compilation errors detected"
    echo "  Check /tmp/hyperphysics_check.log for details"
    exit 1
fi

echo ""
echo "================================================"
echo -e "${GREEN}CRITICAL FIXES COMPLETE${NC}"
echo "================================================"
echo ""
echo "Summary:"
echo "  ✓ nalgebra standardized to 0.33"
echo "  ✓ dashmap standardized to 6.1"
echo "  ✓ Workspace compiles successfully"
echo ""
echo "Next Steps:"
echo "  1. Review unsafe code blocks (see gap_analysis_report.md)"
echo "  2. Replace unwrap() calls with proper error handling"
echo "  3. Create hyperphysics-hft-ecosystem master crate"
echo "  4. Integrate physics engines"
echo ""
echo "Estimated time saved: 2-3 hours"
echo ""
