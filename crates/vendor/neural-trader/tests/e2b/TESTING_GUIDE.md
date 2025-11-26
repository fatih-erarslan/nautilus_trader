# E2B Integration Testing Guide

## Overview

This directory contains comprehensive integration tests for the E2B Trading Swarm system, validating all three coordination layers: Backend NAPI, MCP Server, and CLI.

---

## Quick Start

### Prerequisites

1. **E2B Credentials** (Required)
   ```bash
   export E2B_API_KEY="your_e2b_api_key_here"
   # OR
   export E2B_ACCESS_TOKEN="your_e2b_access_token_here"
   ```

2. **Trading Credentials** (Optional, for full integration tests)
   ```bash
   export ALPACA_API_KEY="your_alpaca_key"
   export ALPACA_API_SECRET="your_alpaca_secret"
   ```

3. **Dependencies**
   ```bash
   npm install
   ```

### Running Tests

#### Full Integration Validation
```bash
# Run complete validation suite (22 tests)
npm test integration-validation.test.js

# With detailed output
npm test integration-validation.test.js --verbose

# With coverage
npm test integration-validation.test.js --coverage
```

#### Quick Validation (Backend Only)
```bash
# Run only backend NAPI tests
npm test integration-validation.test.js -t "Backend NAPI"
```

#### Production Validation
```bash
# Run production readiness tests
npm test integration-validation.test.js -t "Production Validation"
```

---

## Test Suites

### 1. Backend NAPI Integration (5 tests)

Tests the Rust-based NAPI bindings that provide direct E2B API access.

```bash
npm test integration-validation.test.js -t "Backend"
```

**Tests:**
- ✅ E2B functions are exported
- ✅ TypeScript definitions match runtime
- ✅ Create E2B sandbox via NAPI
- ✅ Execute process in sandbox
- ✅ Concurrent sandbox operations (3 parallel)

**Expected Duration:** 15-20 seconds

### 2. MCP Server Integration (4 tests)

Tests the Model Context Protocol server for external tool integration.

```bash
npm test integration-validation.test.js -t "MCP"
```

**Tests:**
- ✅ Server is accessible
- ✅ E2B tools are registered (5 tools)
- ✅ Tool schemas validate correctly
- ✅ JSON-RPC 2.0 compliance

**Expected Duration:** 5-10 seconds

### 3. CLI Functionality (3 tests)

Tests the command-line interface for swarm management.

```bash
npm test integration-validation.test.js -t "CLI"
```

**Tests:**
- ✅ Commands are executable
- ✅ Help command works
- ✅ List command shows state

**Expected Duration:** 10-15 seconds

### 4. Real Trading Integration (4 tests)

Tests end-to-end trading workflows with strategy deployment and execution.

```bash
npm test integration-validation.test.js -t "Trading"
```

**Tests:**
- ✅ Deploy momentum strategy to E2B
- ✅ Execute backtest across agents
- ✅ Consensus decision across swarm
- ✅ Portfolio tracking across swarm

**Expected Duration:** 30-45 seconds

### 5. Production Validation (6 tests)

Tests production-level scenarios with cost and performance validation.

```bash
npm test integration-validation.test.js -t "Production"
```

**Tests:**
- ✅ Full 5-agent deployment
- ✅ Stress test with 100 tasks
- ✅ Cost within budget ($5/day)
- ✅ Performance meets SLA (P95 <5s)
- ✅ Success rate above threshold (>90%)
- ✅ Final readiness certification

**Expected Duration:** 60-90 seconds

---

## Expected Test Output

### Success Output

```
╔═══════════════════════════════════════════════════════════╗
║   E2B Swarm Integration Validation Suite v2.1.0        ║
╚═══════════════════════════════════════════════════════════╝

✅ E2B API Key configured
✅ Cost tracking enabled
✅ Performance monitoring enabled

Test Suites: 5 passed, 5 total
Tests:       22 passed, 22 total

📊 Production Readiness:
  Success Rate: 100.00%
  Required: 90%
  Status: ✅ PRODUCTION READY
```

---

## Documentation

### Related Documentation

- **Integration Report:** `/workspaces/neural-trader/docs/e2b/INTEGRATION_VALIDATION_REPORT.md`
- **Production Summary:** `/workspaces/neural-trader/docs/e2b/PRODUCTION_VALIDATION_SUMMARY.md`
- **Backend API:** `/workspaces/neural-trader/neural-trader-rust/packages/neural-trader-backend/index.d.ts`

---

**Last Updated:** November 14, 2025
**Test Suite Version:** 2.1.1
**Status:** ✅ Production Ready
