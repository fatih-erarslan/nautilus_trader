#!/bin/bash

# E2B Deployment Patterns Test Runner
# Runs comprehensive tests for all 8 deployment patterns

set -e

echo "=========================================="
echo "  E2B DEPLOYMENT PATTERNS TEST SUITE"
echo "=========================================="
echo ""

# Check for E2B API key
if [ -z "$E2B_API_KEY" ]; then
    echo "⚠️  No E2B_API_KEY found - running in MOCK mode"
    echo "   Set E2B_API_KEY for live testing"
    echo ""
else
    echo "✅ E2B_API_KEY detected - running in LIVE mode"
    echo ""
fi

# Check dependencies
if [ ! -d "node_modules" ]; then
    echo "📦 Installing dependencies..."
    npm install
    echo ""
fi

# Run tests
echo "🧪 Running deployment pattern tests..."
echo ""

npm test

echo ""
echo "=========================================="
echo "  TESTS COMPLETE"
echo "=========================================="
echo ""
echo "📄 See docs/e2b/DEPLOYMENT_PATTERNS_RESULTS.md for detailed analysis"
