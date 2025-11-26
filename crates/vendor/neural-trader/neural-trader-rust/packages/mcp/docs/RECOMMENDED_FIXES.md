# Recommended Fixes for Neural Trader MCP

This document provides detailed instructions for fixing all validation failures.

---

## Priority 1: Critical Server Issues

### Fix 1: Create Server Entry Point

**Issue:** `bin/neural-trader.js` missing - server cannot start

**Fix:**
```bash
mkdir -p bin
```

Create `bin/neural-trader.js`:
```javascript
#!/usr/bin/env node
const { MCPServer } = require('../index.js');

async function main() {
  const server = new MCPServer({
    transport: 'stdio',
    toolRegistry: require('../tools/toolRegistry.json')
  });

  await server.start();

  process.on('SIGINT', async () => {
    await server.stop();
    process.exit(0);
  });
}

main().catch(console.error);
```

Make executable:
```bash
chmod +x bin/neural-trader.js
```

**Validation:** `node bin/neural-trader.js` should start server

---

### Fix 2: Implement STDIO Transport

**Issue:** MCP protocol requires STDIO, not detected

**Fix:**

Create `src/transport/stdio.ts`:
```typescript
import { Transport } from './types';
import * as readline from 'readline';

export class StdioTransport implements Transport {
  private rl: readline.Interface;

  constructor() {
    this.rl = readline.createInterface({
      input: process.stdin,
      output: process.stdout,
      terminal: false
    });
  }

  async start(handler: (message: any) => void): Promise<void> {
    this.rl.on('line', (line) => {
      try {
        const message = JSON.parse(line);
        handler(message);
      } catch (error) {
        console.error('Invalid JSON:', error);
      }
    });
  }

  async send(message: any): Promise<void> {
    process.stdout.write(JSON.stringify(message) + '\n');
  }

  async stop(): Promise<void> {
    this.rl.close();
  }
}
```

**Validation:** Test with JSON-RPC messages via stdin/stdout

---

## Priority 2: MCP Protocol Compliance

### Fix 3: Create Tool Registry

**Issue:** No tool registry for 107 tools

**Fix:**

Create `tools/toolRegistry.json`:
```json
{
  "version": "1.0.0",
  "tools": [
    {
      "name": "ping",
      "description": "Simple ping tool to verify server connectivity",
      "inputSchema": {
        "type": "object",
        "properties": {},
        "required": []
      }
    },
    {
      "name": "list_strategies",
      "description": "List all available trading strategies with GPU capabilities",
      "inputSchema": {
        "type": "object",
        "properties": {},
        "required": []
      }
    },
    {
      "name": "get_strategy_info",
      "description": "Get detailed information about a trading strategy",
      "inputSchema": {
        "type": "object",
        "properties": {
          "strategy": {
            "type": "string",
            "description": "Strategy name"
          }
        },
        "required": ["strategy"]
      }
    }
    // ... add all 107 tools
  ]
}
```

**Script to generate from Rust code:**
```bash
cd ../../crates/mcp-server
cargo run --bin generate-registry > ../../packages/mcp/tools/toolRegistry.json
```

**Validation:** Verify 107 tools present with valid schemas

---

### Fix 4: Add Tool Schemas

**Issue:** No schema validation files

**Fix:**

Create schema generator:
```bash
npm install --save-dev @apidevtools/json-schema-ref-parser
```

Create `scripts/generate-schemas.js`:
```javascript
const fs = require('fs');
const path = require('path');

const registry = require('../tools/toolRegistry.json');
const outputDir = path.join(__dirname, '../src/tools/schemas');

// Create schemas directory
if (!fs.existsSync(outputDir)) {
  fs.mkdirSync(outputDir, { recursive: true });
}

// Generate individual schema files
registry.tools.forEach(tool => {
  const schema = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "title": tool.name,
    "description": tool.description,
    "properties": tool.inputSchema.properties,
    "required": tool.inputSchema.required
  };

  const filename = path.join(outputDir, `${tool.name}.schema.json`);
  fs.writeFileSync(filename, JSON.stringify(schema, null, 2));
});

console.log(`Generated ${registry.tools.length} schema files`);
```

Run:
```bash
node scripts/generate-schemas.js
```

**Validation:** Should have 107 schema files in `src/tools/schemas/`

---

## Priority 3: Testing Infrastructure

### Fix 5: Fix Rust Module Path

**Issue:** Example fails with `use of unresolved module 'mcp_server'`

**Fix:**

Edit `examples/mcp_tools_demo.rs`:
```rust
// Change this:
use mcp_server::tools::{account, neural_extended, risk, config};

// To this:
use neural_trader_mcp::tools::{account, neural_extended, risk, config};
```

Or update `Cargo.toml`:
```toml
[[example]]
name = "mcp_tools_demo"
crate-type = ["bin"]
```

**Validation:** `cargo test --examples` should pass

---

### Fix 6: Create Unit Test Suite

**Issue:** 0 tests executed

**Fix:**

Create `tests/unit/tools.test.ts`:
```typescript
import { describe, it, expect } from '@jest/globals';
import { MCPServer } from '../../src/server';

describe('MCP Tools', () => {
  let server: MCPServer;

  beforeAll(() => {
    server = new MCPServer();
  });

  describe('ping', () => {
    it('should respond with success', async () => {
      const result = await server.callTool('ping', {});
      expect(result).toHaveProperty('success', true);
    });
  });

  describe('list_strategies', () => {
    it('should return array of strategies', async () => {
      const result = await server.callTool('list_strategies', {});
      expect(result).toHaveProperty('strategies');
      expect(Array.isArray(result.strategies)).toBe(true);
    });
  });

  // Add tests for all 107 tools
});
```

Install test dependencies:
```bash
npm install --save-dev jest @jest/globals @types/jest ts-jest
```

Create `jest.config.js`:
```javascript
module.exports = {
  preset: 'ts-jest',
  testEnvironment: 'node',
  roots: ['<rootDir>/tests'],
  testMatch: ['**/*.test.ts'],
  collectCoverage: true,
  coverageDirectory: 'coverage',
  coverageReporters: ['text', 'lcov', 'html'],
  collectCoverageFrom: [
    'src/**/*.ts',
    '!src/**/*.d.ts'
  ],
  coverageThresholds: {
    global: {
      branches: 80,
      functions: 80,
      lines: 80,
      statements: 80
    }
  }
};
```

**Validation:** `npm test` should run all tests

---

## Priority 4: Build & Docker

### Fix 7: Add Build Scripts

**Issue:** No build script in package.json

**Fix:**

Edit `package.json`:
```json
{
  "scripts": {
    "build": "tsc && npm run build:rust",
    "build:rust": "cd ../../crates/mcp-server && cargo build --release",
    "clean": "rm -rf dist coverage && cargo clean",
    "test": "jest",
    "test:integration": "jest --config jest.integration.config.js",
    "lint": "eslint src/**/*.ts",
    "format": "prettier --write 'src/**/*.ts'"
  }
}
```

**Validation:** `npm run build` should succeed

---

### Fix 8: Fix Docker Build

**Issue:** Docker build fails on `npm ci`

**Fix:**

Generate lock file:
```bash
npm install
# This creates package-lock.json
```

Improve `Dockerfile`:
```dockerfile
FROM node:18-alpine AS builder

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy source
COPY . .

# Build
RUN npm run build

# Production image
FROM node:18-alpine

WORKDIR /app

# Copy built files
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/package.json ./

# Run
CMD ["node", "bin/neural-trader.js"]
```

**Validation:** `docker build -t neural-trader-mcp .` should succeed

---

## Priority 5: Configuration & Optimization

### Fix 9: Add TypeScript Configuration

**Issue:** No tsconfig.json

**Fix:**

Create `tsconfig.json`:
```json
{
  "compilerOptions": {
    "target": "ES2020",
    "module": "commonjs",
    "lib": ["ES2020"],
    "outDir": "./dist",
    "rootDir": "./src",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true,
    "resolveJsonModule": true,
    "declaration": true,
    "declarationMap": true,
    "sourceMap": true,
    "moduleResolution": "node"
  },
  "include": ["src/**/*"],
  "exclude": ["node_modules", "dist", "tests"]
}
```

**Validation:** `npx tsc --noEmit` should pass

---

### Fix 10: Optimize Throughput

**Issue:** 50 req/s (target: 100+)

**Recommendations:**

1. **Enable Connection Pooling:**
```typescript
const server = new MCPServer({
  maxConnections: 100,
  connectionTimeout: 5000
});
```

2. **Add Caching:**
```typescript
import { LRUCache } from 'lru-cache';

const cache = new LRUCache({
  max: 1000,
  ttl: 60000 // 1 minute
});
```

3. **Optimize Tool Execution:**
```rust
// Use tokio runtime for async operations
#[tokio::main]
async fn main() {
    // Parallel tool execution
}
```

4. **Load Balancing:**
```typescript
// Use cluster module for multi-core
const cluster = require('cluster');
const numCPUs = require('os').cpus().length;

if (cluster.isMaster) {
  for (let i = 0; i < numCPUs; i++) {
    cluster.fork();
  }
} else {
  startServer();
}
```

**Validation:** Re-run performance tests

---

## Automated Fix Script

Run all automated fixes:
```bash
bash scripts/fix-and-validate.sh
```

This will:
1. Generate package-lock.json
2. Create tool registry
3. Generate schemas
4. Fix Rust examples
5. Set up tests
6. Rebuild everything
7. Re-run validation

---

## Manual Steps Checklist

- [ ] Create `bin/neural-trader.js`
- [ ] Implement STDIO transport
- [ ] Generate tool registry (107 tools)
- [ ] Create schema files (87+ files)
- [ ] Fix Rust example imports
- [ ] Write unit tests for all tools
- [ ] Add integration tests
- [ ] Create TypeScript config
- [ ] Generate package-lock.json
- [ ] Update Dockerfile
- [ ] Add build scripts
- [ ] Optimize performance
- [ ] Run full validation
- [ ] Verify 100% pass rate

---

## Verification Commands

After applying fixes:

```bash
# Level 1: Build
npm run build
cargo build --release

# Level 2: Tests
npm test
cargo test

# Level 3: Protocol
node scripts/verify-mcp-compliance.js

# Level 4: E2E
node bin/neural-trader.js &
npm run test:e2e

# Level 5: Docker
docker build -t neural-trader-mcp .
docker run neural-trader-mcp

# Level 6: Performance
npm run benchmark

# Full validation
bash scripts/validate-all.sh
```

---

## Expected Results After Fixes

- ✅ Level 1: Build (0 errors, 0 warnings)
- ✅ Level 2: Tests (107+ tests, 100% pass, >80% coverage)
- ✅ Level 3: Protocol (Full MCP compliance)
- ✅ Level 4: E2E (All tools functional)
- ✅ Level 5: Docker (Clean build, <500MB image)
- ✅ Level 6: Performance (>100 req/s, <100ms latency)

**Certification Status:** ✅ **PASSED**

---

## Support

For questions or issues:
- GitHub Issues: https://github.com/your-repo/issues
- Documentation: See README.md
- Validation Logs: Check `/tmp/validation-*.log`
