#!/bin/bash

# Build and Publish Script for Neural Trader Examples
# Exit on error, but continue to next package if one fails

EXAMPLES_DIR="/home/user/neural-trader/packages/examples"
SUCCESS_LOG="$EXAMPLES_DIR/publish-success.log"
ERROR_LOG="$EXAMPLES_DIR/publish-errors.log"

# Clear previous logs
> "$SUCCESS_LOG"
> "$ERROR_LOG"

echo "=== Neural Trader Package Build & Publish ==="
echo "Starting at: $(date)"
echo ""

# Define packages to build and publish
declare -a PACKAGES=(
    # Shared packages (publish first - others may depend on them)
    "shared/openrouter-integration"
    "shared/benchmark-swarm-framework"
    "shared/self-learning-framework"

    # Example packages
    "portfolio-optimization"
    "multi-strategy-backtest"
    "market-microstructure"
    "energy-forecasting"
    "supply-chain-prediction"
    "anomaly-detection"
    "logistics-optimization"
    "dynamic-pricing"
    "energy-grid-optimization"
    "healthcare-optimization"
    "quantum-optimization"
    "adaptive-systems"
    "neuromorphic-computing"
    "evolutionary-game-theory"
)

# Function to build and publish a package
build_and_publish() {
    local package_path="$1"
    local package_dir="$EXAMPLES_DIR/$package_path"

    if [ ! -d "$package_dir" ]; then
        echo "❌ Directory not found: $package_dir" | tee -a "$ERROR_LOG"
        return 1
    fi

    cd "$package_dir" || return 1

    # Get package name from package.json
    local package_name=$(node -p "require('./package.json').name" 2>/dev/null)

    if [ -z "$package_name" ]; then
        echo "❌ Could not read package name from $package_path/package.json" | tee -a "$ERROR_LOG"
        return 1
    fi

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📦 Processing: $package_name"
    echo "📁 Path: $package_path"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # Install dependencies if needed
    if [ ! -d "node_modules" ]; then
        echo "📥 Installing dependencies..."
        npm install --legacy-peer-deps 2>&1 | tail -5
    fi

    # Build
    echo "🔨 Building package..."
    if npm run build 2>&1 | tee /tmp/build-output.log | tail -10; then
        echo "✅ Build successful"
    else
        echo "❌ Build failed for $package_name" | tee -a "$ERROR_LOG"
        cat /tmp/build-output.log | tail -20 | tee -a "$ERROR_LOG"
        return 1
    fi

    # Publish
    echo "📤 Publishing to npm..."
    if npm publish --access public 2>&1 | tee /tmp/publish-output.log | tail -10; then
        echo "✅ Published successfully: $package_name" | tee -a "$SUCCESS_LOG"
        echo "🔗 https://www.npmjs.com/package/$package_name" | tee -a "$SUCCESS_LOG"
        return 0
    else
        # Check if already published
        if grep -q "You cannot publish over the previously published versions" /tmp/publish-output.log; then
            echo "⚠️  Already published: $package_name" | tee -a "$SUCCESS_LOG"
            echo "🔗 https://www.npmjs.com/package/$package_name" | tee -a "$SUCCESS_LOG"
            return 0
        else
            echo "❌ Publish failed for $package_name" | tee -a "$ERROR_LOG"
            cat /tmp/publish-output.log | tail -20 | tee -a "$ERROR_LOG"
            return 1
        fi
    fi
}

# Build and publish all packages
TOTAL=${#PACKAGES[@]}
SUCCESS_COUNT=0
FAIL_COUNT=0

for package in "${PACKAGES[@]}"; do
    if build_and_publish "$package"; then
        ((SUCCESS_COUNT++))
    else
        ((FAIL_COUNT++))
    fi
done

# Summary
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 PUBLICATION SUMMARY"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Total packages: $TOTAL"
echo "✅ Successful: $SUCCESS_COUNT"
echo "❌ Failed: $FAIL_COUNT"
echo ""
echo "Completed at: $(date)"
echo ""

if [ $SUCCESS_COUNT -gt 0 ]; then
    echo "✅ Successfully published packages:"
    cat "$SUCCESS_LOG"
fi

if [ $FAIL_COUNT -gt 0 ]; then
    echo ""
    echo "❌ Failed packages (see details in $ERROR_LOG):"
    grep "❌" "$ERROR_LOG"
fi

echo ""
echo "Full logs:"
echo "  Success: $SUCCESS_LOG"
echo "  Errors: $ERROR_LOG"
