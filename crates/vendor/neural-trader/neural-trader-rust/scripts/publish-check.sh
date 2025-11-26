#!/bin/bash
# Pre-publish validation script
# Ensures everything is ready before publishing to npm

set -e

echo "🔍 Running pre-publish checks for Neural Trader..."
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Track errors
errors=0

# Check 1: Verify package.json exists and is valid
echo -n "Checking package.json... "
if [ -f "package.json" ] && node -e "JSON.parse(require('fs').readFileSync('package.json'))"; then
  echo -e "${GREEN}✓${NC}"
else
  echo -e "${RED}✗ Invalid package.json${NC}"
  errors=$((errors + 1))
fi

# Check 2: Verify version in package.json matches Cargo.toml
echo -n "Checking version consistency... "
PKG_VERSION=$(node -p "require('./package.json').version")
CARGO_VERSION=$(grep "^version" Cargo.toml | head -1 | sed 's/.*"\(.*\)".*/\1/')
if [ "$PKG_VERSION" = "$CARGO_VERSION" ]; then
  echo -e "${GREEN}✓ v$PKG_VERSION${NC}"
else
  echo -e "${RED}✗ Version mismatch: package.json=$PKG_VERSION, Cargo.toml=$CARGO_VERSION${NC}"
  errors=$((errors + 1))
fi

# Check 3: Verify all platform packages exist
echo "Checking platform packages..."
platforms=("darwin-arm64" "darwin-x64" "linux-x64-gnu" "linux-x64-musl" "win32-x64-msvc")
for platform in "${platforms[@]}"; do
  echo -n "  $platform... "
  if [ -f "npm/$platform/package.json" ]; then
    echo -e "${GREEN}✓${NC}"
  else
    echo -e "${RED}✗ Missing${NC}"
    errors=$((errors + 1))
  fi
done

# Check 4: Verify entry point files exist
echo "Checking entry points..."
files=("index.js" "index.d.ts" "bin/cli.js" "scripts/postinstall.js")
for file in "${files[@]}"; do
  echo -n "  $file... "
  if [ -f "$file" ]; then
    echo -e "${GREEN}✓${NC}"
  else
    echo -e "${RED}✗ Missing${NC}"
    errors=$((errors + 1))
  fi
done

# Check 5: Verify README exists
echo -n "Checking README.md... "
if [ -f "README.md" ]; then
  echo -e "${GREEN}✓${NC}"
else
  echo -e "${YELLOW}⚠ Missing (recommended)${NC}"
fi

# Check 6: Verify LICENSE exists
echo -n "Checking LICENSE... "
if [ -f "LICENSE" ] || [ -f "LICENSE.md" ] || [ -f "LICENSE.txt" ]; then
  echo -e "${GREEN}✓${NC}"
else
  echo -e "${YELLOW}⚠ Missing (recommended)${NC}"
fi

# Check 7: Verify .npmignore exists
echo -n "Checking .npmignore... "
if [ -f ".npmignore" ]; then
  echo -e "${GREEN}✓${NC}"
else
  echo -e "${YELLOW}⚠ Missing (will use .gitignore)${NC}"
fi

# Check 8: Check for common security issues
echo -n "Checking for secrets in code... "
if git grep -q -E "(api_key|secret_key|password|token).*=.*['\"][^'\"]{20,}['\"]" 2>/dev/null; then
  echo -e "${RED}✗ Possible secrets found${NC}"
  echo "  Please review and remove any hardcoded secrets"
  errors=$((errors + 1))
else
  echo -e "${GREEN}✓${NC}"
fi

# Check 9: Run tests
echo -n "Running tests... "
if npm test &>/dev/null; then
  echo -e "${GREEN}✓${NC}"
else
  echo -e "${YELLOW}⚠ Tests failed or not configured${NC}"
fi

# Check 10: Verify git status
echo -n "Checking git status... "
if git diff-index --quiet HEAD -- 2>/dev/null; then
  echo -e "${GREEN}✓ Clean${NC}"
else
  echo -e "${YELLOW}⚠ Uncommitted changes${NC}"
fi

# Summary
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ $errors -eq 0 ]; then
  echo -e "${GREEN}✓ All checks passed!${NC}"
  echo ""
  echo "Ready to publish. Run:"
  echo "  npm publish"
  echo ""
  echo "For a dry run, use:"
  echo "  npm publish --dry-run"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  exit 0
else
  echo -e "${RED}✗ $errors error(s) found${NC}"
  echo ""
  echo "Please fix the errors above before publishing."
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  exit 1
fi
