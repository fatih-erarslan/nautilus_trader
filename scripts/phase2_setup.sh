#!/bin/bash
#
# Phase 2 Setup Script
# Installs Rust, Lean 4, and all necessary development tools
#

set -e

echo "🚀 HyperPhysics Phase 2 Setup"
echo "=============================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Rust is installed
if command -v rustc &> /dev/null; then
    echo -e "${GREEN}✓${NC} Rust already installed: $(rustc --version)"
else
    echo -e "${YELLOW}⏳${NC} Installing Rust..."
    curl --proto='=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
    echo -e "${GREEN}✓${NC} Rust installed: $(rustc --version)"
fi

# Check if Lean 4 is installed
if command -v lean &> /dev/null; then
    echo -e "${GREEN}✓${NC} Lean 4 already installed: $(lean --version)"
else
    echo -e "${YELLOW}⏳${NC} Installing Lean 4..."
    curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh -s -- -y
    source "$HOME/.elan/env"
    echo -e "${GREEN}✓${NC} Lean 4 installed: $(lean --version)"
fi

# Install development tools
echo ""
echo "📦 Installing development tools..."

if cargo install --list | grep -q cargo-mutants; then
    echo -e "${GREEN}✓${NC} cargo-mutants already installed"
else
    echo -e "${YELLOW}⏳${NC} Installing cargo-mutants..."
    cargo install cargo-mutants
fi

if cargo install --list | grep -q cargo-fuzz; then
    echo -e "${GREEN}✓${NC} cargo-fuzz already installed"
else
    echo -e "${YELLOW}⏳${NC} Installing cargo-fuzz..."
    cargo install cargo-fuzz
fi

if cargo install --list | grep -q flamegraph; then
    echo -e "${GREEN}✓${NC} flamegraph already installed"
else
    echo -e "${YELLOW}⏳${NC} Installing flamegraph..."
    cargo install flamegraph
fi

if cargo install --list | grep -q cargo-benchcmp; then
    echo -e "${GREEN}✓${NC} cargo-benchcmp already installed"
else
    echo -e "${YELLOW}⏳${NC} Installing cargo-benchcmp..."
    cargo install cargo-benchcmp
fi

echo ""
echo "🏗️  Building HyperPhysics..."
cd "$(dirname "$0")/.."
cargo build --workspace --all-features

echo ""
echo "🧪 Running tests..."
cargo test --workspace --quiet

echo ""
echo -e "${GREEN}✅ Phase 2 setup complete!${NC}"
echo ""
echo "Next steps:"
echo "  1. cargo bench --workspace              # Run performance baselines"
echo "  2. ./scripts/run_mutation_tests.sh      # Mutation testing"
echo "  3. ./scripts/run_fuzz_tests.sh --time 3600  # Fuzzing"
echo ""
