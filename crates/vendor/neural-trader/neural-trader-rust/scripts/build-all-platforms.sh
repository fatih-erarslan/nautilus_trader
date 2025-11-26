#!/bin/bash
# Build script for cross-platform compilation
# Builds native modules for all supported platforms

set -e

echo "🔨 Building Neural Trader for all platforms..."
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running in CI
if [ -n "$CI" ]; then
  echo "Running in CI environment"
  IS_CI=true
else
  IS_CI=false
fi

# Function to build for a specific target
build_target() {
  local target=$1
  local display_name=$2

  echo -e "${YELLOW}Building for ${display_name} (${target})...${NC}"

  if cargo build --release --target "$target" --manifest-path crates/napi-bindings/Cargo.toml; then
    echo -e "${GREEN}✓ ${display_name} build successful${NC}"
    return 0
  else
    echo -e "${RED}✗ ${display_name} build failed${NC}"
    return 1
  fi
}

# Track successes and failures
declare -a successes
declare -a failures

# Linux x86_64 GNU
if build_target "x86_64-unknown-linux-gnu" "Linux x86_64 (GNU)"; then
  successes+=("linux-x64-gnu")
else
  failures+=("linux-x64-gnu")
fi

# Linux x86_64 MUSL (static linking)
if build_target "x86_64-unknown-linux-musl" "Linux x86_64 (MUSL)"; then
  successes+=("linux-x64-musl")
else
  failures+=("linux-x64-musl")
fi

# macOS x86_64
if [ "$(uname)" = "Darwin" ] || [ "$IS_CI" = true ]; then
  if build_target "x86_64-apple-darwin" "macOS x86_64"; then
    successes+=("darwin-x64")
  else
    failures+=("darwin-x64")
  fi

  # macOS ARM64 (Apple Silicon)
  if build_target "aarch64-apple-darwin" "macOS ARM64"; then
    successes+=("darwin-arm64")
  else
    failures+=("darwin-arm64")
  fi
else
  echo -e "${YELLOW}Skipping macOS builds (not on macOS)${NC}"
fi

# Windows x86_64 MSVC
if [ "$(uname)" = "MINGW"* ] || [ "$IS_CI" = true ]; then
  if build_target "x86_64-pc-windows-msvc" "Windows x86_64 (MSVC)"; then
    successes+=("win32-x64-msvc")
  else
    failures+=("win32-x64-msvc")
  fi
else
  echo -e "${YELLOW}Skipping Windows build (not on Windows)${NC}"
fi

# Summary
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Build Summary"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ ${#successes[@]} -gt 0 ]; then
  echo -e "${GREEN}Successful builds (${#successes[@]}):${NC}"
  for target in "${successes[@]}"; do
    echo "  ✓ $target"
  done
fi

if [ ${#failures[@]} -gt 0 ]; then
  echo -e "${RED}Failed builds (${#failures[@]}):${NC}"
  for target in "${failures[@]}"; do
    echo "  ✗ $target"
  done
  echo ""
  echo "Some builds failed. This is normal if you don't have all toolchains installed."
  echo "For CI/CD, ensure all cross-compilation toolchains are available."
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Exit with error if any required build failed in CI
if [ "$IS_CI" = true ] && [ ${#failures[@]} -gt 0 ]; then
  exit 1
fi
