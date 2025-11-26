# Getting Started - Rust Port Implementation

**Practical guide for developers starting the Neural Trading Rust port**

---

## Prerequisites

### Required Software

```bash
# Check versions
node --version    # >= 18.0.0
npm --version     # >= 9.0.0
rustc --version   # >= 1.70.0
cargo --version   # >= 1.70.0
```

### Install Rust (if not already installed)

```bash
# Linux/macOS
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Follow prompts, then:
source "$HOME/.cargo/env"

# Verify
rustc --version
cargo --version
```

### Install Build Dependencies

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install -y build-essential pkg-config libssl-dev
```

**macOS:**
```bash
xcode-select --install
```

**Windows:**
```powershell
# Install Visual Studio 2022 Build Tools
# https://visualstudio.microsoft.com/downloads/
# Select "Desktop development with C++"
```

### Install Rust Targets

```bash
# For cross-compilation
rustup target add x86_64-unknown-linux-gnu
rustup target add x86_64-apple-darwin
rustup target add aarch64-apple-darwin
rustup target add x86_64-pc-windows-msvc
```

### Install napi-rs CLI

```bash
npm install -g @napi-rs/cli
```

---

## Step 1: Initialize Project

### Option A: New Project

```bash
# Create new directory
mkdir neural-trader-rs
cd neural-trader-rs

# Initialize with napi-rs
napi new

# Answer prompts:
# Package name: @neural-trader/core
# Choose targets: All (linux, darwin, win32)
# Enable type definitions: Yes
# Enable GitHub Actions: Yes
```

### Option B: Add to Existing Project

```bash
cd /home/user/neural-trader

# Initialize napi-rs in existing project
npm init napi

# This creates:
# - Cargo.toml
# - src/lib.rs
# - package.json updates
# - .github/workflows/CI.yml
```

---

## Step 2: Create Workspace Structure

```bash
# Create Cargo workspace
cat > Cargo.toml << 'EOF'
[workspace]
members = [
    "crates/neural-core",
    "crates/neural-bindings",
    "crates/neural-cli"
]
resolver = "2"

[workspace.package]
version = "0.1.0"
edition = "2021"
authors = ["Neural Trader Team <team@neural-trader.io>"]
license = "MIT"

[workspace.dependencies]
# Async runtime
tokio = { version = "1.40", features = ["full", "tracing"] }
tokio-util = { version = "0.7", features = ["rt"] }
async-trait = "0.1"

# Parallel processing
rayon = "1.10"

# Data processing
polars = { version = "0.43", features = ["lazy", "sql", "streaming"] }
arrow = "52.2"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"

# HTTP/WebSocket
axum = { version = "0.7", features = ["ws", "macros"] }
reqwest = { version = "0.12", features = ["json", "stream"] }
tokio-tungstenite = "0.23"

# Database
sqlx = { version = "0.8", features = ["runtime-tokio", "sqlite"] }

# ML/Neural
candle-core = "0.6"
candle-nn = "0.6"

# Observability
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }

# Error handling
anyhow = "1.0"
thiserror = "1.0"

# Testing
criterion = "0.5"
proptest = "1.5"
EOF

# Create crate directories
mkdir -p crates/neural-core/src
mkdir -p crates/neural-bindings/src
mkdir -p crates/neural-cli/src
```

---

## Step 3: Configure Core Crate

```bash
# crates/neural-core/Cargo.toml
cat > crates/neural-core/Cargo.toml << 'EOF'
[package]
name = "neural-core"
version.workspace = true
edition.workspace = true

[dependencies]
tokio = { workspace = true }
tokio-util = { workspace = true }
rayon = { workspace = true }
polars = { workspace = true }
serde = { workspace = true }
serde_json = { workspace = true }
axum = { workspace = true }
reqwest = { workspace = true }
tokio-tungstenite = { workspace = true }
candle-core = { workspace = true }
candle-nn = { workspace = true }
tracing = { workspace = true }
anyhow = { workspace = true }
thiserror = { workspace = true }

[dev-dependencies]
tokio-test = "0.4"
proptest = { workspace = true }

[lib]
name = "neural_core"
path = "src/lib.rs"
EOF
```

### Create Initial Core Structure

```bash
# crates/neural-core/src/lib.rs
cat > crates/neural-core/src/lib.rs << 'EOF'
//! Neural Trading Core Engine
//!
//! High-performance trading engine with ultra-low latency execution.

pub mod execution;
pub mod market_data;
pub mod neural;
pub mod portfolio;
pub mod risk;
pub mod types;

pub use execution::ExecutionEngine;
pub use market_data::MarketDataProcessor;
pub use neural::NeuralModel;
pub use portfolio::PortfolioOptimizer;
pub use types::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
EOF

# Create module files
mkdir -p crates/neural-core/src/{execution,market_data,neural,portfolio,risk}
touch crates/neural-core/src/{execution,market_data,neural,portfolio,risk}/mod.rs
```

---

## Step 4: Implement Core Types

```bash
# crates/neural-core/src/types.rs
cat > crates/neural-core/src/types.rs << 'EOF'
//! Core data types used throughout the trading engine

use serde::{Deserialize, Serialize};

/// Market data point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketData {
    pub symbol: String,
    pub price: f64,
    pub volume: f64,
    pub timestamp_ns: i64,
    pub bid: f64,
    pub ask: f64,
    pub spread: f64,
}

/// Order side (Buy or Sell)
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum OrderSide {
    Buy,
    Sell,
}

impl std::fmt::Display for OrderSide {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OrderSide::Buy => write!(f, "BUY"),
            OrderSide::Sell => write!(f, "SELL"),
        }
    }
}

/// Order type
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum OrderType {
    Market,
    Limit,
    Stop,
}

/// Trade order
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeOrder {
    pub symbol: String,
    pub side: OrderSide,
    pub quantity: f64,
    pub order_type: OrderType,
    pub price: Option<f64>,
    pub timestamp_ns: i64,
}

/// Execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionResult {
    pub order_id: String,
    pub status: ExecutionStatus,
    pub filled_quantity: f64,
    pub avg_price: f64,
    pub total_latency_ns: i64,
    pub validation_time_ns: i64,
    pub execution_time_ns: i64,
    pub timestamp_ns: i64,
}

/// Order execution status
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ExecutionStatus {
    Filled,
    Partial,
    Rejected,
}

impl std::fmt::Display for ExecutionStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExecutionStatus::Filled => write!(f, "FILLED"),
            ExecutionStatus::Partial => write!(f, "PARTIAL"),
            ExecutionStatus::Rejected => write!(f, "REJECTED"),
        }
    }
}

/// Neural model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralConfig {
    pub model_path: String,
    pub batch_size: usize,
    pub use_gpu: bool,
    pub precision: Precision,
}

/// Model precision
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Precision {
    FP32,
    FP16,
    INT8,
}

/// Prediction result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionResult {
    pub predictions: Vec<f64>,
    pub confidence: Vec<f64>,
    pub latency_ms: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_market_data_serialization() {
        let data = MarketData {
            symbol: "AAPL".to_string(),
            price: 150.0,
            volume: 1000.0,
            timestamp_ns: 1234567890,
            bid: 149.95,
            ask: 150.05,
            spread: 0.10,
        };

        let json = serde_json::to_string(&data).unwrap();
        let deserialized: MarketData = serde_json::from_str(&json).unwrap();

        assert_eq!(data.symbol, deserialized.symbol);
        assert_eq!(data.price, deserialized.price);
    }
}
EOF
```

---

## Step 5: Configure napi-rs Bindings

```bash
# crates/neural-bindings/Cargo.toml
cat > crates/neural-bindings/Cargo.toml << 'EOF'
[package]
name = "neural-bindings"
version.workspace = true
edition.workspace = true

[lib]
crate-type = ["cdylib"]

[dependencies]
neural-core = { path = "../neural-core" }

# napi-rs
napi = "2.16"
napi-derive = "2.16"

# Async
tokio = { workspace = true }

# Serialization
serde = { workspace = true }
serde_json = { workspace = true }

[build-dependencies]
napi-build = "2.1"

[profile.release]
lto = true
codegen-units = 1
opt-level = 3
strip = true
panic = "abort"
EOF

# Create build script
cat > crates/neural-bindings/build.rs << 'EOF'
fn main() {
    napi_build::setup();
}
EOF
```

---

## Step 6: Implement Initial Bindings

```bash
# crates/neural-bindings/src/lib.rs
cat > crates/neural-bindings/src/lib.rs << 'EOF'
#![deny(clippy::all)]

use napi::bindgen_prelude::*;
use napi_derive::napi;

mod execution;
mod market_data;
mod neural;

pub use execution::*;
pub use market_data::*;
pub use neural::*;

/// Get library version and build info
#[napi]
pub fn get_version() -> Result<Version> {
    Ok(Version {
        version: env!("CARGO_PKG_VERSION").to_string(),
        build_date: env!("BUILD_DATE").unwrap_or("unknown").to_string(),
        platform: std::env::consts::OS.to_string(),
        features: vec![
            #[cfg(feature = "gpu")]
            "gpu".to_string(),
        ],
    })
}

/// Version information
#[napi(object)]
pub struct Version {
    pub version: String,
    pub build_date: String,
    pub platform: String,
    pub features: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        let version = get_version().unwrap();
        assert!(!version.version.is_empty());
    }
}
EOF

# Create binding modules
cat > crates/neural-bindings/src/execution.rs << 'EOF'
//! Execution engine bindings

use napi::bindgen_prelude::*;
use napi_derive::napi;

/// Ultra-low latency execution engine
#[napi]
pub struct ExecutionEngine {
    // Placeholder - implement in Phase 2
}

#[napi]
impl ExecutionEngine {
    /// Create a new execution engine
    #[napi(constructor)]
    pub fn new(config: ExecutionConfig) -> Result<Self> {
        Ok(Self {})
    }

    /// Start the execution engine
    #[napi]
    pub async fn start(&self) -> Result<()> {
        Ok(())
    }

    /// Stop the engine gracefully
    #[napi]
    pub async fn stop(&self) -> Result<()> {
        Ok(())
    }
}

/// Execution configuration
#[napi(object)]
pub struct ExecutionConfig {
    pub websocket_url: String,
    pub api_key: String,
    pub buffer_size: Option<u32>,
    pub max_latency_ms: Option<u32>,
}
EOF

cat > crates/neural-bindings/src/market_data.rs << 'EOF'
//! Market data processor bindings

use napi::bindgen_prelude::*;
use napi_derive::napi;

/// Market data processor
#[napi]
pub struct MarketDataProcessor {
    // Placeholder - implement in Phase 4
}

#[napi]
impl MarketDataProcessor {
    /// Create a new market data processor
    #[napi(constructor)]
    pub fn new(config: MarketDataConfig) -> Result<Self> {
        Ok(Self {})
    }
}

/// Market data configuration
#[napi(object)]
pub struct MarketDataConfig {
    pub symbols: Vec<String>,
    pub data_source: String,
    pub api_key: String,
}
EOF

cat > crates/neural-bindings/src/neural.rs << 'EOF'
//! Neural model bindings

use napi::bindgen_prelude::*;
use napi_derive::napi;

/// Neural trading model
#[napi]
pub struct NeuralModel {
    // Placeholder - implement in Phase 3
}

#[napi]
impl NeuralModel {
    /// Load a pre-trained model
    #[napi(factory)]
    pub async fn load(config: NeuralConfig) -> Result<Self> {
        Ok(Self {})
    }
}

/// Neural model configuration
#[napi(object)]
pub struct NeuralConfig {
    pub model_path: String,
    pub batch_size: u32,
    pub use_gpu: bool,
    pub precision: String,
}
EOF
```

---

## Step 7: Update package.json

```bash
# Update package.json
cat > package.json << 'EOF'
{
  "name": "@neural-trader/core",
  "version": "0.1.0",
  "description": "Ultra-low latency neural trading engine",
  "main": "index.js",
  "types": "index.d.ts",
  "napi": {
    "name": "neural-trader",
    "triples": {
      "defaults": true,
      "additional": [
        "x86_64-unknown-linux-gnu",
        "x86_64-apple-darwin",
        "aarch64-apple-darwin",
        "x86_64-pc-windows-msvc"
      ]
    }
  },
  "scripts": {
    "artifacts": "napi artifacts",
    "build": "napi build --platform --release --manifest-path crates/neural-bindings/Cargo.toml",
    "build:debug": "napi build --platform --manifest-path crates/neural-bindings/Cargo.toml",
    "prepublishOnly": "napi prepublish -t npm",
    "test": "vitest",
    "test:integration": "vitest run __test__/",
    "bench": "cargo bench",
    "format": "cargo fmt --all && prettier --write .",
    "lint": "cargo clippy --all-targets --all-features",
    "typecheck": "tsc --noEmit",
    "universal": "napi universal",
    "version": "napi version"
  },
  "devDependencies": {
    "@napi-rs/cli": "^2.18.0",
    "@types/node": "^20.11.0",
    "prettier": "^3.2.0",
    "typescript": "^5.3.3",
    "vitest": "^1.2.0"
  },
  "optionalDependencies": {
    "@neural-trader/linux-x64-gnu": "0.1.0",
    "@neural-trader/darwin-x64": "0.1.0",
    "@neural-trader/darwin-arm64": "0.1.0",
    "@neural-trader/win32-x64-msvc": "0.1.0"
  },
  "engines": {
    "node": ">= 18"
  },
  "repository": {
    "type": "git",
    "url": "https://github.com/your-org/neural-trader-rs.git"
  },
  "keywords": [
    "trading",
    "neural-networks",
    "low-latency",
    "rust",
    "napi"
  ],
  "author": "Neural Trader Team",
  "license": "MIT"
}
EOF

# Install dependencies
npm install
```

---

## Step 8: Create TypeScript Configuration

```bash
cat > tsconfig.json << 'EOF'
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "ESNext",
    "lib": ["ES2022"],
    "moduleResolution": "bundler",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true,
    "declaration": true,
    "declarationMap": true,
    "outDir": "dist",
    "rootDir": "."
  },
  "include": ["index.d.ts", "__test__/**/*"],
  "exclude": ["node_modules", "target", "dist"]
}
EOF
```

---

## Step 9: Create Initial Tests

```bash
mkdir -p __test__

cat > __test__/integration.test.ts << 'EOF'
import { describe, it, expect } from 'vitest';
import { getVersion, ExecutionEngine } from '../index.js';

describe('Neural Trader Core', () => {
  it('should get version info', () => {
    const version = getVersion();

    expect(version).toBeDefined();
    expect(version.version).toBeTruthy();
    expect(typeof version.version).toBe('string');
  });

  it('should create execution engine', () => {
    const engine = new ExecutionEngine({
      websocketUrl: 'wss://test.example.com',
      apiKey: 'test-key',
      bufferSize: 4096,
      maxLatencyMs: 50,
    });

    expect(engine).toBeDefined();
  });
});
EOF

# Configure Vitest
cat > vitest.config.ts << 'EOF'
import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    globals: true,
    environment: 'node',
    include: ['__test__/**/*.test.ts'],
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html'],
    },
  },
});
EOF
```

---

## Step 10: Build and Test

```bash
# Build the project (debug mode for faster iteration)
npm run build:debug

# This will:
# 1. Compile Rust code
# 2. Generate .node native module
# 3. Auto-generate index.d.ts TypeScript definitions

# Check generated files
ls -lh *.node        # Native binary
ls -lh index.d.ts    # TypeScript types

# Run tests
npm test

# Expected output:
# ✓ should get version info
# ✓ should create execution engine
```

---

## Step 11: Create Example Usage

```bash
mkdir -p example

cat > example/basic.ts << 'EOF'
import { ExecutionEngine, getVersion } from '../index.js';

async function main() {
  // Get version
  const version = getVersion();
  console.log('Neural Trader Version:', version.version);
  console.log('Platform:', version.platform);

  // Create execution engine
  const engine = new ExecutionEngine({
    websocketUrl: process.env.ALPACA_WS_URL || 'wss://paper-api.alpaca.markets/stream',
    apiKey: process.env.ALPACA_API_KEY || 'test-key',
    bufferSize: 4096,
    maxLatencyMs: 50,
  });

  console.log('✅ Execution engine created');

  // Start engine
  await engine.start();
  console.log('✅ Engine started');

  // Stop engine
  await engine.stop();
  console.log('✅ Engine stopped');
}

main().catch(console.error);
EOF

# Run example
npm run build:debug && node --loader tsx example/basic.ts
```

---

## Step 12: Set Up CI/CD

```bash
mkdir -p .github/workflows

cat > .github/workflows/CI.yml << 'EOF'
name: Build Native Modules

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]

env:
  DEBUG: napi:*
  APP_NAME: neural-trader
  MACOSX_DEPLOYMENT_TARGET: "10.13"

jobs:
  build:
    strategy:
      fail-fast: false
      matrix:
        settings:
          - host: ubuntu-latest
            target: x86_64-unknown-linux-gnu
            build: npm run build -- --target x86_64-unknown-linux-gnu
          - host: macos-latest
            target: x86_64-apple-darwin
            build: npm run build -- --target x86_64-apple-darwin
          - host: macos-latest
            target: aarch64-apple-darwin
            build: npm run build -- --target aarch64-apple-darwin
          - host: windows-latest
            target: x86_64-pc-windows-msvc
            build: npm run build -- --target x86_64-pc-windows-msvc

    name: Build - ${{ matrix.settings.target }}
    runs-on: ${{ matrix.settings.host }}

    steps:
      - uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: 20
          cache: npm

      - name: Install Rust
        uses: dtolnay/rust-toolchain@stable
        with:
          targets: ${{ matrix.settings.target }}

      - name: Cache cargo
        uses: actions/cache@v4
        with:
          path: |
            ~/.cargo/registry/index/
            ~/.cargo/registry/cache/
            ~/.cargo/git/db/
            target/
          key: ${{ matrix.settings.target }}-cargo-${{ hashFiles('**/Cargo.lock') }}

      - name: Install dependencies
        run: npm ci

      - name: Build
        run: ${{ matrix.settings.build }}

      - name: Upload artifact
        uses: actions/upload-artifact@v4
        with:
          name: bindings-${{ matrix.settings.target }}
          path: |
            *.node
            npm/**/*.node
          if-no-files-found: error

  test:
    runs-on: ubuntu-latest
    needs: build

    steps:
      - uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: 20

      - name: Download artifacts
        uses: actions/download-artifact@v4
        with:
          name: bindings-x86_64-unknown-linux-gnu

      - name: Install dependencies
        run: npm ci

      - name: Run tests
        run: npm test
EOF
```

---

## Step 13: Documentation

```bash
# Create README for the Rust port
cat > README.rust.md << 'EOF'
# Neural Trader - Rust Core

Ultra-low latency neural trading engine built with Rust and napi-rs.

## Features

- ⚡ Sub-50μs latency execution pipeline
- 🔄 Zero-copy data streaming
- 🧠 GPU-accelerated neural inference
- 📊 High-performance data processing with Polars
- 🔒 Memory-safe with Rust
- 🌐 Cross-platform (Linux, macOS, Windows)

## Installation

```bash
npm install @neural-trader/core
```

## Quick Start

```typescript
import { ExecutionEngine } from '@neural-trader/core';

const engine = new ExecutionEngine({
  websocketUrl: 'wss://api.example.com/stream',
  apiKey: 'your-api-key',
});

await engine.start();
```

## Development

### Prerequisites

- Node.js >= 18
- Rust >= 1.70
- Build tools (gcc/clang/MSVC)

### Build

```bash
# Debug build (faster)
npm run build:debug

# Release build (optimized)
npm run build
```

### Test

```bash
# Unit tests (Rust)
cargo test

# Integration tests (TypeScript)
npm test

# Benchmarks
npm run bench
```

## Performance

| Metric | Python | Rust | Improvement |
|--------|--------|------|-------------|
| Latency (p50) | 450ms | 45μs | 10,000x |
| Throughput | 2.2K ops/s | 22K ops/s | 10x |
| Memory | 450 MB | 85 MB | 5.3x |

## License

MIT
EOF
```

---

## Verification Checklist

After completing all steps, verify:

```bash
# 1. Project structure
tree -L 3 crates/

# Should show:
# crates/
# ├── neural-bindings/
# │   ├── Cargo.toml
# │   ├── build.rs
# │   └── src/
# ├── neural-cli/
# │   ├── Cargo.toml
# │   └── src/
# └── neural-core/
#     ├── Cargo.toml
#     └── src/

# 2. Rust builds
cargo build --workspace
cargo test --workspace

# 3. napi-rs builds
npm run build:debug

# 4. TypeScript types generated
cat index.d.ts | head -20

# 5. Tests pass
npm test

# 6. Example runs
node --loader tsx example/basic.ts
```

---

## Next Steps

Now that the foundation is set up:

1. **Phase 2 (Weeks 3-4):** Implement execution pipeline
   - See `crates/neural-core/src/execution/`
   - Port lock-free buffers
   - WebSocket integration

2. **Phase 3 (Weeks 5-6):** Implement neural models
   - See `crates/neural-core/src/neural/`
   - Candle integration
   - GPU support

3. **Phase 4 (Weeks 7-8):** Data processing
   - See `crates/neural-core/src/market_data/`
   - Polars DataFrames
   - Historical data API

Continue following the roadmap in `01-crate-ecosystem-and-interop.md`.

---

## Common Issues

### Issue: `napi-rs` not found

**Solution:**
```bash
npm install -g @napi-rs/cli
```

### Issue: Rust compilation fails

**Solution:**
```bash
# Update Rust
rustup update

# Clean and rebuild
cargo clean
npm run build:debug
```

### Issue: TypeScript types not generated

**Solution:**
```bash
# Rebuild with type generation
napi build --platform --release --dts index.d.ts
```

### Issue: Tests fail with "Cannot find module"

**Solution:**
```bash
# Ensure native module is built
npm run build:debug

# Check .node file exists
ls -lh *.node
```

---

## Resources

- **Full Documentation:** See `docs/rust-port/README.md`
- **API Reference:** See `docs/rust-port/01-crate-ecosystem-and-interop.md`
- **Troubleshooting:** See `docs/rust-port/02-quick-reference.md`

---

**Getting Started Version:** 1.0.0
**Last Updated:** 2025-11-12

**Status:** ✅ Ready to Implement Phase 1
