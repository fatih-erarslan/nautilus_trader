# NAPI-RS Development Guide

Complete guide for developing and extending the Neural Trader NAPI-RS implementation.

## Table of Contents

- [Overview](#overview)
- [Build System](#build-system)
- [Adding New Tools](#adding-new-tools)
- [Type System](#type-system)
- [Testing](#testing)
- [Performance Optimization](#performance-optimization)
- [Debugging](#debugging)
- [Publishing](#publishing)

---

## Overview

Neural Trader uses NAPI-RS to create high-performance Node.js bindings for Rust code, providing:

- **Native Performance** - 10-100x faster than pure JavaScript
- **Memory Safety** - Rust's ownership system prevents memory leaks
- **Type Safety** - Full TypeScript definitions generated automatically
- **Cross-Platform** - Builds for Linux, macOS, Windows
- **GPU Support** - Optional CUDA/Metal acceleration

### Architecture

```
┌─────────────────────────────────────────┐
│         Node.js Application             │
│    (@neural-trader/mcp package)         │
└─────────────┬───────────────────────────┘
              │ NAPI-RS bindings
┌─────────────▼───────────────────────────┐
│         Rust Implementation             │
│  • crates/mcp-server                    │
│  • crates/mcp-protocol                  │
│  • crates/neural                        │
│  • crates/strategies                    │
│  • crates/risk                          │
└─────────────────────────────────────────┘
```

---

## Build System

### Prerequisites

**Required:**
- Rust 1.70+ (`rustup install stable`)
- Node.js 18+ (`nvm install 18`)
- NAPI-RS CLI (`npm install -g @napi-rs/cli`)

**Optional (for GPU):**
- CUDA Toolkit 11.8+ (NVIDIA)
- Metal SDK (Apple Silicon)

### Initial Setup

```bash
# Clone repository
git clone https://github.com/ruvnet/neural-trader.git
cd neural-trader/neural-trader-rust

# Install dependencies
npm install

# Build Rust code and generate bindings
npm run build

# Run tests
npm test
```

### Build Commands

```bash
# Development build (debug mode)
npm run build

# Production build (optimized)
npm run build:release

# Build for specific platform
npm run build -- --target x86_64-unknown-linux-gnu

# Watch mode (rebuild on changes)
npm run build:watch

# Clean build artifacts
npm run clean
```

### Build Configuration

**package.json:**

```json
{
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
  }
}
```

**Cargo.toml:**

```toml
[lib]
crate-type = ["cdylib"]

[dependencies]
napi = { version = "2.16", features = ["async", "tokio_rt"] }
napi-derive = "2.16"

[build-dependencies]
napi-build = "2.1"
```

---

## Adding New Tools

### Step 1: Define Tool Schema

Create schema in `crates/mcp-protocol/src/schema.rs`:

```rust
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct CustomIndicatorParams {
    pub symbol: String,
    pub period: u32,
    pub indicator_type: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CustomIndicatorResult {
    pub symbol: String,
    pub value: f64,
    pub signal: String,
    pub timestamp: String,
}
```

### Step 2: Implement Tool Logic

Add implementation in appropriate crate (e.g., `crates/strategies/src/indicators.rs`):

```rust
use crate::schema::{CustomIndicatorParams, CustomIndicatorResult};
use anyhow::Result;

pub async fn calculate_custom_indicator(
    params: CustomIndicatorParams
) -> Result<CustomIndicatorResult> {
    // Fetch data
    let prices = fetch_prices(&params.symbol, params.period).await?;

    // Calculate indicator
    let value = match params.indicator_type.as_str() {
        "rsi" => calculate_rsi(&prices, params.period),
        "macd" => calculate_macd(&prices),
        _ => return Err(anyhow::anyhow!("Unknown indicator type")),
    }?;

    // Generate signal
    let signal = if value > 70.0 {
        "OVERBOUGHT".to_string()
    } else if value < 30.0 {
        "OVERSOLD".to_string()
    } else {
        "NEUTRAL".to_string()
    };

    Ok(CustomIndicatorResult {
        symbol: params.symbol,
        value,
        signal,
        timestamp: chrono::Utc::now().to_rfc3339(),
    })
}
```

### Step 3: Register MCP Tool

Add to `crates/mcp-server/src/tools/trading.rs`:

```rust
use crate::schema::*;
use napi::bindgen_prelude::*;

#[napi]
pub async fn custom_indicator(
    params: String
) -> Result<String> {
    // Parse parameters
    let params: CustomIndicatorParams = serde_json::from_str(&params)
        .map_err(|e| Error::from_reason(format!("Invalid params: {}", e)))?;

    // Execute tool
    let result = calculate_custom_indicator(params)
        .await
        .map_err(|e| Error::from_reason(e.to_string()))?;

    // Serialize response
    serde_json::to_string(&result)
        .map_err(|e| Error::from_reason(e.to_string()))
}
```

### Step 4: Export Tool

Add to `crates/mcp-server/src/tools/mod.rs`:

```rust
pub mod trading;

pub use trading::custom_indicator;
```

And in `crates/mcp-server/src/lib.rs`:

```rust
pub use tools::custom_indicator;
```

### Step 5: Add Tool Metadata

Update `crates/mcp-server/src/handlers/tools.rs`:

```rust
pub fn get_tool_schema(name: &str) -> Option<ToolSchema> {
    match name {
        "custom_indicator" => Some(ToolSchema {
            name: "custom_indicator".to_string(),
            description: "Calculate custom technical indicator".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Stock symbol"
                    },
                    "period": {
                        "type": "integer",
                        "description": "Indicator period"
                    },
                    "indicator_type": {
                        "type": "string",
                        "enum": ["rsi", "macd"],
                        "description": "Indicator type"
                    }
                },
                "required": ["symbol", "period", "indicator_type"]
            }),
        }),
        // ... other tools
        _ => None,
    }
}
```

### Step 6: Build and Test

```bash
# Build
npm run build

# Test manually
node -e "
const nt = require('./index.js');
const result = await nt.customIndicator(JSON.stringify({
  symbol: 'AAPL',
  period: 14,
  indicator_type: 'rsi'
}));
console.log(JSON.parse(result));
"
```

### Step 7: Add Unit Tests

Create `crates/strategies/tests/indicators_test.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_custom_indicator_rsi() {
        let params = CustomIndicatorParams {
            symbol: "AAPL".to_string(),
            period: 14,
            indicator_type: "rsi".to_string(),
        };

        let result = calculate_custom_indicator(params).await;
        assert!(result.is_ok());

        let result = result.unwrap();
        assert_eq!(result.symbol, "AAPL");
        assert!(result.value >= 0.0 && result.value <= 100.0);
        assert!(["OVERBOUGHT", "OVERSOLD", "NEUTRAL"].contains(&result.signal.as_str()));
    }

    #[tokio::test]
    async fn test_custom_indicator_invalid_type() {
        let params = CustomIndicatorParams {
            symbol: "AAPL".to_string(),
            period: 14,
            indicator_type: "invalid".to_string(),
        };

        let result = calculate_custom_indicator(params).await;
        assert!(result.is_err());
    }
}
```

Run tests:

```bash
cargo test --package strategies
```

---

## Type System

### NAPI-RS Type Mapping

| Rust Type | TypeScript Type | JavaScript |
|-----------|----------------|------------|
| `String` | `string` | string |
| `i32`, `u32` | `number` | number |
| `f64` | `number` | number |
| `bool` | `boolean` | boolean |
| `Vec<T>` | `Array<T>` | Array |
| `HashMap<K, V>` | `Record<K, V>` | Object |
| `Option<T>` | `T \| undefined` | value or undefined |
| `Result<T, E>` | `Promise<T>` | Promise |

### Custom Types

Define shared types in `crates/mcp-protocol/src/schema.rs`:

```rust
use napi_derive::napi;
use serde::{Deserialize, Serialize};

#[napi(object)]
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Position {
    pub symbol: String,
    pub quantity: i32,
    pub avg_price: f64,
    pub current_price: f64,
    pub market_value: f64,
    pub unrealized_pnl: f64,
}

#[napi(object)]
#[derive(Debug, Serialize, Deserialize)]
pub struct Portfolio {
    pub total_value: f64,
    pub cash: f64,
    pub positions: Vec<Position>,
    pub daily_pnl: f64,
}
```

This generates TypeScript definitions:

```typescript
export interface Position {
  symbol: string;
  quantity: number;
  avgPrice: number;
  currentPrice: number;
  marketValue: number;
  unrealizedPnl: number;
}

export interface Portfolio {
  totalValue: number;
  cash: number;
  positions: Array<Position>;
  dailyPnl: number;
}
```

### Async Functions

All async functions return `Promise`:

```rust
#[napi]
pub async fn fetch_data(symbol: String) -> Result<String> {
    // Async implementation
    tokio::time::sleep(Duration::from_millis(100)).await;
    Ok(format!("Data for {}", symbol))
}
```

TypeScript:

```typescript
export function fetchData(symbol: string): Promise<string>;
```

### Error Handling

Use `napi::Result` for all public functions:

```rust
use napi::bindgen_prelude::*;

#[napi]
pub fn validate_symbol(symbol: String) -> Result<bool> {
    if symbol.is_empty() {
        return Err(Error::from_reason("Symbol cannot be empty"));
    }

    if symbol.len() > 5 {
        return Err(Error::from_reason("Symbol too long"));
    }

    Ok(true)
}
```

JavaScript error handling:

```javascript
try {
  await validateSymbol('AAPL');
} catch (error) {
  console.error('Validation error:', error.message);
}
```

---

## Testing

### Unit Tests

**Rust unit tests:**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validation() {
        assert!(validate_symbol("AAPL".to_string()).is_ok());
        assert!(validate_symbol("".to_string()).is_err());
        assert!(validate_symbol("TOOLONG".to_string()).is_err());
    }

    #[tokio::test]
    async fn test_async_operation() {
        let result = fetch_data("AAPL".to_string()).await;
        assert!(result.is_ok());
    }
}
```

Run Rust tests:

```bash
# All tests
cargo test

# Specific package
cargo test --package mcp-server

# With output
cargo test -- --nocapture

# Test coverage
cargo tarpaulin --out Html
```

### Integration Tests

**JavaScript/TypeScript tests:**

```javascript
const test = require('ava');
const nt = require('./index.js');

test('custom_indicator returns valid result', async (t) => {
  const result = await nt.customIndicator(JSON.stringify({
    symbol: 'AAPL',
    period: 14,
    indicator_type: 'rsi'
  }));

  const parsed = JSON.parse(result);

  t.is(parsed.symbol, 'AAPL');
  t.true(parsed.value >= 0 && parsed.value <= 100);
  t.true(['OVERBOUGHT', 'OVERSOLD', 'NEUTRAL'].includes(parsed.signal));
});

test('custom_indicator handles invalid input', async (t) => {
  await t.throwsAsync(
    async () => {
      await nt.customIndicator(JSON.stringify({
        symbol: 'AAPL',
        period: 14,
        indicator_type: 'invalid'
      }));
    },
    { message: /Unknown indicator type/ }
  );
});
```

Run integration tests:

```bash
npm test
```

### Benchmarking

**Criterion benchmarks:**

Create `benches/indicators.rs`:

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use neural_trader_strategies::calculate_custom_indicator;

fn benchmark_rsi(c: &mut Criterion) {
    let params = CustomIndicatorParams {
        symbol: "AAPL".to_string(),
        period: 14,
        indicator_type: "rsi".to_string(),
    };

    c.bench_function("calculate_rsi", |b| {
        b.iter(|| {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(calculate_custom_indicator(black_box(params.clone())))
        })
    });
}

criterion_group!(benches, benchmark_rsi);
criterion_main!(benches);
```

Run benchmarks:

```bash
cargo bench
```

---

## Performance Optimization

### SIMD Acceleration

Use `packed_simd` for vectorized operations:

```rust
use packed_simd::f64x4;

pub fn calculate_sma_simd(prices: &[f64], period: usize) -> Vec<f64> {
    let mut result = Vec::with_capacity(prices.len());

    for i in 0..prices.len() - period + 1 {
        let mut sum = f64x4::splat(0.0);
        let mut j = i;

        // Process 4 elements at a time
        while j + 4 <= i + period {
            let chunk = f64x4::from_slice_unaligned(&prices[j..j + 4]);
            sum += chunk;
            j += 4;
        }

        // Handle remaining elements
        let mut scalar_sum = sum.sum();
        for k in j..i + period {
            scalar_sum += prices[k];
        }

        result.push(scalar_sum / period as f64);
    }

    result
}
```

### Parallel Processing

Use `rayon` for parallel iteration:

```rust
use rayon::prelude::*;

pub fn analyze_portfolio_parallel(symbols: Vec<String>) -> Vec<AnalysisResult> {
    symbols
        .par_iter()
        .map(|symbol| {
            analyze_symbol(symbol.clone())
        })
        .collect()
}
```

### Memory Pool

Reuse allocations for hot paths:

```rust
use std::sync::Arc;
use parking_lot::Mutex;

pub struct BufferPool {
    buffers: Arc<Mutex<Vec<Vec<f64>>>>,
    capacity: usize,
}

impl BufferPool {
    pub fn new(capacity: usize) -> Self {
        Self {
            buffers: Arc::new(Mutex::new(Vec::new())),
            capacity,
        }
    }

    pub fn get(&self) -> Vec<f64> {
        self.buffers.lock().pop().unwrap_or_else(|| Vec::with_capacity(self.capacity))
    }

    pub fn return_buffer(&self, mut buffer: Vec<f64>) {
        buffer.clear();
        self.buffers.lock().push(buffer);
    }
}
```

### GPU Acceleration

Use `cudarc` for CUDA operations:

```rust
#[cfg(feature = "gpu")]
use cudarc::driver::*;

#[cfg(feature = "gpu")]
pub fn calculate_var_gpu(returns: &[f64], confidence: f64) -> Result<f64> {
    let dev = CudaDevice::new(0)?;

    // Allocate GPU memory
    let returns_gpu = dev.htod_copy(returns)?;

    // Launch kernel
    let result = dev.launch_kernel(
        var_kernel,
        LaunchConfig::for_num_elems(returns.len() as u32),
        (&returns_gpu, confidence)
    )?;

    Ok(result)
}
```

### Profile-Guided Optimization

```bash
# Generate profile data
RUSTFLAGS="-C profile-generate=/tmp/pgo-data" cargo build --release
./target/release/neural-trader-bench

# Use profile data for optimization
RUSTFLAGS="-C profile-use=/tmp/pgo-data/merged.profdata" cargo build --release
```

---

## Debugging

### Rust Debugging

**Enable debug symbols:**

```toml
[profile.release]
debug = true
```

**Use `dbg!` macro:**

```rust
pub fn calculate_indicator(prices: &[f64]) -> f64 {
    let sma = calculate_sma(prices);
    dbg!(sma);  // Prints: [src/indicators.rs:42] sma = 185.25

    let result = sma * 1.5;
    dbg!(result);  // Prints: [src/indicators.rs:45] result = 277.875

    result
}
```

**Use GDB/LLDB:**

```bash
# Build with debug symbols
cargo build

# Debug with GDB
gdb ./target/debug/neural-trader-mcp

# Debug with LLDB (macOS)
lldb ./target/debug/neural-trader-mcp
```

### JavaScript Debugging

**Node.js inspector:**

```bash
node --inspect-brk -e "
const nt = require('./index.js');
debugger;
const result = await nt.customIndicator(JSON.stringify({...}));
"
```

Open Chrome DevTools at `chrome://inspect`.

**Logging:**

```rust
use tracing::{info, debug, error};

#[napi]
pub async fn custom_indicator(params: String) -> Result<String> {
    info!("custom_indicator called with params: {}", params);

    let parsed = serde_json::from_str(&params)
        .map_err(|e| {
            error!("Failed to parse params: {}", e);
            Error::from_reason(format!("Invalid params: {}", e))
        })?;

    debug!("Parsed params: {:?}", parsed);

    // ... implementation

    Ok(result)
}
```

### Memory Profiling

**Valgrind:**

```bash
valgrind --leak-check=full \
  --show-leak-kinds=all \
  ./target/debug/neural-trader-mcp
```

**Heaptrack:**

```bash
heaptrack ./target/release/neural-trader-mcp
heaptrack_gui heaptrack.neural-trader-mcp.*.gz
```

### Performance Profiling

**perf (Linux):**

```bash
# Record
perf record --call-graph dwarf ./target/release/neural-trader-mcp

# Analyze
perf report
```

**Instruments (macOS):**

```bash
instruments -t "Time Profiler" ./target/release/neural-trader-mcp
```

**flamegraph:**

```bash
cargo install flamegraph
cargo flamegraph --bench indicators
```

---

## Publishing

### Pre-Release Checklist

- [ ] All tests pass (`cargo test && npm test`)
- [ ] Benchmarks show no regressions
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Version bumped in Cargo.toml and package.json

### Build for All Platforms

```bash
# Install cross-compilation tools
npm install -g @napi-rs/cli

# Build for all platforms
npm run build -- --release --target x86_64-unknown-linux-gnu
npm run build -- --release --target x86_64-apple-darwin
npm run build -- --release --target aarch64-apple-darwin
npm run build -- --release --target x86_64-pc-windows-msvc
```

### Create Universal Binary (macOS)

```bash
npm run build:universal
```

### Publish to npm

```bash
# Login to npm
npm login

# Publish
npm publish --access public
```

### Publish to crates.io

```bash
# Login to crates.io
cargo login

# Publish core crates first
cd crates/mcp-protocol
cargo publish

cd ../mcp-server
cargo publish
```

### GitHub Release

```bash
# Create tag
git tag -a v1.0.0 -m "Release v1.0.0"
git push origin v1.0.0

# Create release with binaries
gh release create v1.0.0 \
  --title "v1.0.0" \
  --notes "Release notes here" \
  target/release/neural-trader.*.node
```

---

## Next Steps

- [API Reference](/workspaces/neural-trader/neural-trader-rust/docs/api/NEURAL_TRADER_MCP_API.md)
- [Migration Guide](/workspaces/neural-trader/neural-trader-rust/docs/guides/PYTHON_TO_NAPI_MIGRATION.md)
- [Examples](/workspaces/neural-trader/neural-trader-rust/docs/examples/)

---

**Last Updated**: 2025-01-14
**Maintained By**: Neural Trader Team
