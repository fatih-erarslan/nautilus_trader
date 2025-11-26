#!/bin/bash
# E2B Trading Swarm Tests - Automated Run Script

set -e  # Exit on error

echo "🚀 E2B Trading Swarm Integration Tests"
echo "========================================"
echo ""

# Check prerequisites
echo "📋 Checking prerequisites..."

# Check Node.js version
NODE_VERSION=$(node --version)
echo "  ✅ Node.js: $NODE_VERSION"

# Check E2B API key
if [ -z "$E2B_API_KEY" ] && [ -z "$E2B_ACCESS_TOKEN" ]; then
    echo "  ❌ E2B API key not found in environment"
    echo "     Please set E2B_API_KEY or E2B_ACCESS_TOKEN"
    exit 1
else
    echo "  ✅ E2B API key configured"
fi

# Check if dependencies are installed
if [ ! -d "node_modules" ]; then
    echo ""
    echo "📦 Installing dependencies..."
    npm install
    echo "  ✅ Dependencies installed"
fi

echo ""
echo "🧪 Running test suite..."
echo ""

# Run tests with hooks
npx claude-flow@alpha hooks pre-task --description "Running E2B swarm integration tests" || true

npm test

npx claude-flow@alpha hooks post-task --task-id "e2b-tests-$(date +%s)" || true

echo ""
echo "✅ All tests completed successfully!"
echo ""
echo "📊 To view coverage report:"
echo "   npm run test:coverage"
echo "   open coverage/lcov-report/index.html"
