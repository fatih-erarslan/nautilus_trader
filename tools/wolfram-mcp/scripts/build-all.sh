#!/bin/bash
# Build script for Wolfram MCP with native bindings
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "🔨 Building Wolfram MCP Server v2.0"
echo "=================================="

cd "$PROJECT_DIR"

# 1. Build Rust native module
echo ""
echo "📦 Building Rust native module (NAPI-RS)..."
cd native
if [ ! -d "node_modules" ]; then
    npm install
fi
npm run build
cd ..

# 2. Build Swift module (macOS only)
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo ""
    echo "🍎 Building Swift module..."
    cd native/swift
    swift build -c release
    # Copy CLI to accessible location
    cp .build/release/wolfram-swift-cli ../
    cd ../..
fi

# 3. Install Bun dependencies
echo ""
echo "🥟 Installing Bun dependencies..."
if ! command -v bun &> /dev/null; then
    echo "Installing Bun..."
    curl -fsSL https://bun.sh/install | bash
fi
bun install

# 4. Build TypeScript with Bun
echo ""
echo "📜 Building TypeScript..."
bun run build

echo ""
echo "✅ Build complete!"
echo ""
echo "To start the server:"
echo "  bun run start"
echo ""
echo "To run in development mode:"
echo "  bun run dev"
