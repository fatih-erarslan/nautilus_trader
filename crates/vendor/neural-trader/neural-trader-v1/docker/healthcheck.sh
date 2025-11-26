#!/bin/bash
# Health check script for Docker containers

set -e

# Check if MCP server is responding
if command -v curl &> /dev/null; then
    curl -f http://localhost:3000/health || exit 1
elif command -v nc &> /dev/null; then
    nc -z localhost 3000 || exit 1
else
    echo "No health check tool available"
    exit 1
fi

# Check if NAPI bindings are loaded
if [ -n "$CHECK_NAPI" ]; then
    node -e "require('./index.js')" || exit 1
fi

echo "Health check passed"
exit 0
