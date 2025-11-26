#!/bin/bash
# publish-all-packages.sh
# Publishes all 18 Neural Trader packages in dependency order

set -e

PACKAGES_DIR="/workspaces/neural-trader/neural-trader-rust/packages"
LOG_FILE="$PACKAGES_DIR/docs/npm-publish-log.txt"

echo "🚀 Neural Trader NPM Publishing" | tee "$LOG_FILE"
echo "================================" | tee -a "$LOG_FILE"
echo "Started: $(date)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Check npm authentication
echo "Checking npm authentication..." | tee -a "$LOG_FILE"
if ! npm whoami &>/dev/null; then
  echo "❌ Not logged in to npm" | tee -a "$LOG_FILE"
  echo "Please run: npm login" | tee -a "$LOG_FILE"
  exit 1
fi

NPM_USER=$(npm whoami)
echo "✅ Logged in as: $NPM_USER" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Publishing order (dependency-aware)
PACKAGES=(
  "core"
  "mcp-protocol"
  "risk"
  "features"
  "market-data"
  "backtesting"
  "neural"
  "strategies"
  "portfolio"
  "execution"
  "brokers"
  "sports-betting"
  "prediction-markets"
  "news-trading"
  "syndicate"
  "benchoptimizer"
  "mcp"
  "neural-trader"
)

SUCCESS_COUNT=0
FAILED_COUNT=0
FAILED_PACKAGES=()

for package in "${PACKAGES[@]}"; do
  echo "📦 Publishing: @neural-trader/$package" | tee -a "$LOG_FILE"

  cd "$PACKAGES_DIR/$package"

  # Verify package.json exists
  if [ ! -f "package.json" ]; then
    echo "  ❌ package.json not found" | tee -a "$LOG_FILE"
    FAILED_COUNT=$((FAILED_COUNT + 1))
    FAILED_PACKAGES+=("$package")
    continue
  fi

  # Get version
  VERSION=$(node -p "require('./package.json').version")
  echo "  Version: $VERSION" | tee -a "$LOG_FILE"

  # Check if already published
  if npm view "@neural-trader/$package@$VERSION" &>/dev/null; then
    echo "  ⚠️  Already published: @neural-trader/$package@$VERSION" | tee -a "$LOG_FILE"
    echo "  Skipping..." | tee -a "$LOG_FILE"
    SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    echo "" | tee -a "$LOG_FILE"
    continue
  fi

  # Dry run first
  echo "  Running dry-run..." | tee -a "$LOG_FILE"
  if npm publish --dry-run --access public &>> "$LOG_FILE"; then
    echo "  ✅ Dry-run successful" | tee -a "$LOG_FILE"

    # Actual publish
    echo "  Publishing to npm..." | tee -a "$LOG_FILE"
    if npm publish --access public &>> "$LOG_FILE"; then
      echo "  ✅ Published: @neural-trader/$package@$VERSION" | tee -a "$LOG_FILE"
      echo "  URL: https://www.npmjs.com/package/@neural-trader/$package" | tee -a "$LOG_FILE"
      SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
      echo "  ❌ Publish failed: @neural-trader/$package" | tee -a "$LOG_FILE"
      FAILED_COUNT=$((FAILED_COUNT + 1))
      FAILED_PACKAGES+=("$package")
    fi
  else
    echo "  ❌ Dry-run failed: @neural-trader/$package" | tee -a "$LOG_FILE"
    FAILED_COUNT=$((FAILED_COUNT + 1))
    FAILED_PACKAGES+=("$package")
  fi

  echo "" | tee -a "$LOG_FILE"

  # Rate limiting: wait 2 seconds between publishes
  sleep 2
done

# Summary
echo "================================" | tee -a "$LOG_FILE"
echo "📊 Publishing Summary" | tee -a "$LOG_FILE"
echo "================================" | tee -a "$LOG_FILE"
echo "Total packages: ${#PACKAGES[@]}" | tee -a "$LOG_FILE"
echo "✅ Successful: $SUCCESS_COUNT" | tee -a "$LOG_FILE"
echo "❌ Failed: $FAILED_COUNT" | tee -a "$LOG_FILE"

if [ $FAILED_COUNT -gt 0 ]; then
  echo "" | tee -a "$LOG_FILE"
  echo "Failed packages:" | tee -a "$LOG_FILE"
  for pkg in "${FAILED_PACKAGES[@]}"; do
    echo "  - $pkg" | tee -a "$LOG_FILE"
  done
fi

echo "" | tee -a "$LOG_FILE"
echo "Completed: $(date)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "📋 Next Steps:" | tee -a "$LOG_FILE"
echo "1. Verify publications: npm view @neural-trader/syndicate" | tee -a "$LOG_FILE"
echo "2. Test installations: npm install neural-trader" | tee -a "$LOG_FILE"
echo "3. Create GitHub release: gh release create v1.0.0" | tee -a "$LOG_FILE"
echo "4. Update documentation with npm badges" | tee -a "$LOG_FILE"

# Exit with error if any failed
if [ $FAILED_COUNT -gt 0 ]; then
  exit 1
fi

exit 0
