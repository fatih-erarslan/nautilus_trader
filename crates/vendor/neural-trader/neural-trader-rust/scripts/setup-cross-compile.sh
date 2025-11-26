#!/bin/bash
# Setup cross-compilation toolchains for Neural Trader
# Run this once to install all necessary build tools

set -e

echo "🔧 Setting up cross-compilation environment for Neural Trader..."
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check OS
OS="$(uname -s)"

# Install Rust targets
echo -e "${YELLOW}Installing Rust targets...${NC}"
rustup target add x86_64-unknown-linux-gnu
rustup target add x86_64-unknown-linux-musl
rustup target add x86_64-apple-darwin
rustup target add aarch64-apple-darwin
rustup target add x86_64-pc-windows-msvc

echo -e "${GREEN}✓ Rust targets installed${NC}"
echo ""

# Install cross-compilation tools based on OS
case "$OS" in
  Linux*)
    echo -e "${YELLOW}Installing Linux cross-compilation tools...${NC}"

    # Check if apt is available
    if command -v apt-get &> /dev/null; then
      sudo apt-get update
      sudo apt-get install -y \
        gcc-multilib \
        g++-multilib \
        musl-tools \
        mingw-w64

      echo -e "${GREEN}✓ Linux tools installed${NC}"
    else
      echo "Please install gcc-multilib, g++-multilib, musl-tools, and mingw-w64 manually"
    fi
    ;;

  Darwin*)
    echo -e "${YELLOW}Installing macOS cross-compilation tools...${NC}"

    # Install cross-compilation SDK if needed
    echo "macOS native compilation should work out of the box"
    echo "For cross-compiling to Linux/Windows, consider using Docker"
    echo -e "${GREEN}✓ macOS ready${NC}"
    ;;

  MINGW*|MSYS*|CYGWIN*)
    echo -e "${YELLOW}Installing Windows cross-compilation tools...${NC}"
    echo "Windows native compilation should work with MSVC"
    echo "Ensure Visual Studio Build Tools are installed"
    echo -e "${GREEN}✓ Windows ready${NC}"
    ;;

  *)
    echo "Unknown OS: $OS"
    echo "Please install cross-compilation tools manually"
    exit 1
    ;;
esac

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${GREEN}Setup complete!${NC}"
echo ""
echo "You can now build for all platforms with:"
echo "  npm run build:all"
echo ""
echo "Or build for specific platforms:"
echo "  npm run build:linux"
echo "  npm run build:darwin"
echo "  npm run build:windows"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
