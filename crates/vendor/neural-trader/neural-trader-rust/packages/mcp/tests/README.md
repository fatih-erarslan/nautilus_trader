# MCP 2025-11 Compliance Test Suite

Comprehensive test suite to validate Neural Trader MCP Server compliance with the MCP 2025-11 specification.

## Test Coverage

### 1. Protocol Compliance (JSON-RPC 2.0)
- ✅ Request/response format validation
- ✅ Error code compliance
- ✅ Batch request handling
- ✅ Request ID tracking
- ✅ Notification handling
- ✅ Message parsing

**Tests:** `protocol/jsonrpc.test.js`

### 2. Tool Discovery
- ✅ tools/list method implementation
- ✅ JSON Schema 1.1 format validation
- ✅ Metadata completeness (cost, latency, category)
- ✅ ETag generation for caching
- ✅ Tool search and filtering
- ✅ Schema references

**Tests:** `discovery/tool-registry.test.js`

### 3. Transport Layer
- ✅ STDIO line-delimited JSON
- ✅ stderr/stdout separation
- ✅ Graceful shutdown handling
- ✅ Connection management
- ✅ CRLF handling

**Tests:** `transport/stdio.test.js`

### 4. Audit Logging
- ✅ JSON Lines format
- ✅ Event logging (tool_call, tool_result, errors)
- ✅ Log file creation and rotation
- ✅ Timestamp inclusion
- ✅ Error handling

**Tests:** `logging/audit.test.js`

### 5. MCP Methods
- ✅ initialize method
- ✅ tools/list method
- ✅ tools/call method
- ✅ tools/schema method
- ✅ tools/search method
- ✅ tools/categories method
- ✅ server/info method
- ✅ ping method

**Tests:** `integration/mcp-methods.test.js`

## Running Tests

### Run All Tests
```bash
npm test
```

### Run Specific Category
```bash
npm run test:protocol     # Protocol compliance tests
npm run test:discovery    # Tool discovery tests
npm run test:transport    # Transport layer tests
npm run test:logging      # Audit logging tests
npm run test:integration  # MCP methods integration tests
```

### Watch Mode
```bash
npm run test:watch
```

### Coverage Report
```bash
npm run test:coverage
```

## Test Results

The test runner generates a detailed compliance report:

- **COMPLIANCE_REPORT.md** - Detailed validation report with:
  - Compliance percentage
  - Pass/fail status for each requirement
  - Specification violations
  - Recommendations for fixes

## Compliance Checklist

### Protocol Compliance
- [x] JSON-RPC 2.0 request/response format
- [x] Standard error codes match specification
- [x] Batch request handling
- [x] Request ID tracking

### Tool Discovery
- [x] All tools discoverable via tools/list
- [x] JSON Schema 1.1 format for all schemas
- [x] Metadata completeness (cost, latency, category)
- [x] ETag generation for caching

### Transport Layer
- [x] STDIO line-delimited JSON
- [x] stderr separate from protocol
- [x] Graceful shutdown handling

### Audit Logging
- [x] JSON Lines format
- [x] All events logged (tool_call, tool_result, errors)
- [x] Log file creation and rotation

### MCP Methods
- [x] initialize method
- [x] tools/list method
- [x] tools/call method with sample tools
- [x] tools/schema method

## Test Structure

```
tests/
├── protocol/           # JSON-RPC 2.0 protocol tests
├── discovery/          # Tool discovery and registry tests
├── transport/          # STDIO transport tests
├── logging/            # Audit logging tests
├── integration/        # End-to-end MCP methods tests
├── fixtures/           # Test fixtures and sample data
│   ├── tools/          # Sample tool schemas
│   └── logs/           # Test log files
├── jest.config.js      # Jest configuration
├── test-runner.js      # Custom test runner and report generator
├── package.json        # Test dependencies
└── README.md           # This file
```

## Requirements

- Node.js 18+
- npm or yarn
- Jest test framework

## CI/CD Integration

These tests can be integrated into CI/CD pipelines:

```yaml
# .github/workflows/mcp-compliance.yml
name: MCP 2025-11 Compliance

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
        with:
          node-version: '18'
      - run: cd neural-trader-rust/packages/mcp/tests && npm install
      - run: cd neural-trader-rust/packages/mcp/tests && npm test
      - uses: actions/upload-artifact@v3
        with:
          name: compliance-report
          path: neural-trader-rust/packages/mcp/tests/COMPLIANCE_REPORT.md
```

## Contributing

When adding new MCP features:

1. Write tests first (TDD approach)
2. Ensure all existing tests pass
3. Add new tests for new functionality
4. Update this README with new test coverage
5. Run full test suite before submitting PR

## Specification Reference

This test suite validates compliance with:
- **MCP 2025-11 Specification**: https://gist.github.com/ruvnet/284f199d0e0836c1b5185e30f819e052
- **JSON-RPC 2.0**: https://www.jsonrpc.org/specification
- **JSON Schema 2020-12**: https://json-schema.org/draft/2020-12/schema

## License

MIT OR Apache-2.0
