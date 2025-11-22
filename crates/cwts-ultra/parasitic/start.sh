#!/bin/bash

echo "🐝 Starting Parasitic Trading System..."

# Set environment variables
export NODE_ENV=production
export CONFIG_PATH="$(pwd)/config/production"
export RUST_LOG=info
export QUANTUM_MODE=enhanced
export MCP_PORT=8081

# Start MCP server
echo "🚀 Starting MCP server on port 8081..."
node mcp/server.js &
MCP_PID=$!

echo "MCP Server PID: $MCP_PID"
echo "WebSocket endpoint: ws://localhost:8081"
echo "System ready for trading operations"

# Save PID for stop script
echo $MCP_PID > .mcp.pid

wait $MCP_PID