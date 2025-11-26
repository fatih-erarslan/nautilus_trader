#!/bin/bash
# Level 3: MCP Protocol Compliance
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "📡 Level 3: MCP Protocol Compliance"
echo "=============================="

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

ERRORS=0

# 1. JSON-RPC 2.0 compliance
echo -e "\n${YELLOW}3.1 Checking JSON-RPC 2.0 compliance...${NC}"
cd "${PROJECT_ROOT}"

# Check if server exports JSON-RPC handler
if grep -q "jsonrpc.*2.0" src/server.ts 2>/dev/null || grep -q "jsonrpc.*2.0" index.js 2>/dev/null; then
    echo -e "${GREEN}✓ JSON-RPC 2.0 support detected${NC}"
else
    echo -e "${YELLOW}⚠ JSON-RPC 2.0 support not clearly detected${NC}"
fi

# 2. STDIO transport validation
echo -e "\n${YELLOW}3.2 Checking STDIO transport...${NC}"
if grep -q "stdio" src/server.ts 2>/dev/null || grep -q "process.stdin\|process.stdout" index.js 2>/dev/null; then
    echo -e "${GREEN}✓ STDIO transport implemented${NC}"
else
    echo -e "${RED}✗ STDIO transport not found${NC}"
    ERRORS=$((ERRORS + 1))
fi

# 3. Tool discovery verification
echo -e "\n${YELLOW}3.3 Verifying tool discovery...${NC}"
if [ -f "tools/toolRegistry.json" ] || [ -f "src/tools/registry.ts" ]; then
    TOOL_COUNT=$(grep -c "\"name\":" tools/toolRegistry.json 2>/dev/null || echo "0")
    if [ "$TOOL_COUNT" -gt 0 ]; then
        echo -e "${GREEN}✓ Found ${TOOL_COUNT} registered tools${NC}"

        # Verify 107 tools expected
        if [ "$TOOL_COUNT" -eq 107 ]; then
            echo -e "${GREEN}✓ All 107 expected tools registered${NC}"
        else
            echo -e "${YELLOW}⚠ Expected 107 tools, found ${TOOL_COUNT}${NC}"
        fi
    else
        echo -e "${RED}✗ No tools found in registry${NC}"
        ERRORS=$((ERRORS + 1))
    fi
else
    echo -e "${YELLOW}⚠ Tool registry file not found${NC}"
fi

# 4. Schema validation (87 tools)
echo -e "\n${YELLOW}3.4 Validating tool schemas...${NC}"
if [ -d "src/tools" ]; then
    SCHEMA_FILES=$(find src/tools -name "*.schema.json" 2>/dev/null | wc -l || echo "0")
    echo "Schema files found: ${SCHEMA_FILES}"

    if [ "$SCHEMA_FILES" -ge 87 ]; then
        echo -e "${GREEN}✓ Sufficient schema files (${SCHEMA_FILES} ≥ 87)${NC}"
    else
        echo -e "${YELLOW}⚠ Expected at least 87 schemas, found ${SCHEMA_FILES}${NC}"
    fi

    # Validate schema format
    for schema in src/tools/**/*.schema.json; do
        if [ -f "$schema" ]; then
            if ! node -e "JSON.parse(require('fs').readFileSync('$schema', 'utf8'))" 2>/dev/null; then
                echo -e "${RED}✗ Invalid JSON schema: $schema${NC}"
                ERRORS=$((ERRORS + 1))
            fi
        fi
    done
else
    echo -e "${YELLOW}⚠ No src/tools directory found${NC}"
fi

# 5. Audit logging check
echo -e "\n${YELLOW}3.5 Checking audit logging...${NC}"
if [ -d "logs" ]; then
    echo -e "${GREEN}✓ Logs directory exists${NC}"

    # Check for audit log implementation
    if grep -rq "audit\|logger" src/ 2>/dev/null; then
        echo -e "${GREEN}✓ Audit logging implementation found${NC}"
    else
        echo -e "${YELLOW}⚠ Audit logging not clearly implemented${NC}"
    fi
else
    echo -e "${YELLOW}⚠ No logs directory found${NC}"
fi

# 6. MCP protocol version check
echo -e "\n${YELLOW}3.6 Checking MCP protocol version...${NC}"
if grep -q "\"mcp\":" package.json 2>/dev/null; then
    MCP_VERSION=$(grep "\"mcp\":" package.json | grep -o "[0-9]*\.[0-9]*\.[0-9]*" || echo "unknown")
    echo "MCP version: ${MCP_VERSION}"
    echo -e "${GREEN}✓ MCP dependency found${NC}"
else
    echo -e "${YELLOW}⚠ MCP dependency not found in package.json${NC}"
fi

# 7. Check for proper error handling
echo -e "\n${YELLOW}3.7 Checking error handling...${NC}"
if grep -rq "try.*catch\|Error" src/ 2>/dev/null; then
    echo -e "${GREEN}✓ Error handling detected${NC}"
else
    echo -e "${YELLOW}⚠ Limited error handling found${NC}"
fi

# Summary
echo -e "\n=============================="
echo "Level 3 Summary:"
echo "  Errors: $ERRORS"

if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}✅ Level 3: PASSED${NC}"
    exit 0
else
    echo -e "${RED}❌ Level 3: FAILED${NC}"
    exit 1
fi
