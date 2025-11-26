# Rust Port Quick Reference

**Last Updated:** 2025-11-12

---

## Decision Matrix

### Which Interop Strategy Should I Use?

```
┌─────────────────────────────────────────────────────────────────┐
│                     DECISION FLOWCHART                           │
└─────────────────────────────────────────────────────────────────┘

START
  │
  ├─ Need BEST performance? ──▶ napi-rs
  │                              - Sub-50μs latency
  │                              - Zero-copy buffers
  │                              - Auto TypeScript types
  │
  ├─ Windows build issues? ──▶ Neon
  │                             - More Windows-friendly
  │                             - 15% slower than napi-rs
  │                             - Manual TypeScript types
  │
  ├─ Need max portability? ──▶ WASI + Wasmtime
  │                             - Runs anywhere
  │                             - 30-40% slower
  │                             - No GPU support
  │
  └─ Last resort? ──▶ CLI + STDIO
                      - Always works
                      - 2-3x slower
                      - High latency
```

---

## Command Cheat Sheet

### napi-rs

```bash
# Initial setup
npm init napi
cargo new --lib crates/neural-core
cargo new --lib crates/neural-bindings

# Development
npm run build              # Build for current platform
npm run build:debug        # Debug build (faster compile)

# Cross-compilation
npm run build -- --target x86_64-unknown-linux-gnu
npm run build -- --target x86_64-apple-darwin
npm run build -- --target aarch64-apple-darwin
npm run build -- --target x86_64-pc-windows-msvc

# Publishing
npm run artifacts          # Collect binaries
npm run universal          # Create universal macOS binary
npm run prepublishOnly     # Prepare for npm publish
npm publish                # Publish to npm
```

### Cargo

```bash
# Build
cargo build --release                    # Release build
cargo build --release --features gpu     # With GPU support

# Testing
cargo test                               # Run tests
cargo test --release                     # Release mode tests
cargo bench                              # Benchmarks (Criterion)

# Profiling
cargo flamegraph --bin neural-cli        # CPU flamegraph
cargo instruments -t time                # macOS Instruments

# Linting
cargo clippy                             # Linter
cargo fmt                                # Format code
```

### Debugging

```bash
# Node.js debugging
node --inspect-brk example/trading.js

# Rust debugging (VSCode)
# Add to .vscode/launch.json:
{
  "type": "lldb",
  "request": "launch",
  "name": "Debug napi-rs",
  "program": "node",
  "args": ["${workspaceFolder}/example/trading.js"],
  "env": {
    "RUST_BACKTRACE": "1",
    "RUST_LOG": "debug"
  }
}
```

---

## Common Patterns

### 1. Export Async Function

```rust
#[napi]
pub async fn fetch_market_data(symbol: String) -> Result<MarketData> {
    let data = api::fetch(&symbol).await
        .map_err(|e| Error::from_reason(e.to_string()))?;

    Ok(MarketData {
        symbol: data.symbol,
        price: data.price,
        // ...
    })
}
```

**JavaScript:**
```typescript
const data = await fetchMarketData('AAPL');
```

---

### 2. Export Class with Methods

```rust
#[napi]
pub struct TradingEngine {
    inner: Arc<RwLock<Engine>>,
}

#[napi]
impl TradingEngine {
    #[napi(constructor)]
    pub fn new(api_key: String) -> Result<Self> {
        let engine = Engine::new(api_key)?;
        Ok(Self {
            inner: Arc::new(RwLock::new(engine)),
        })
    }

    #[napi]
    pub async fn submit_order(&self, order: Order) -> Result<String> {
        let engine = self.inner.read().await;
        engine.submit(order).await
    }
}
```

**JavaScript:**
```typescript
const engine = new TradingEngine('api-key');
const orderId = await engine.submitOrder({ symbol: 'AAPL', ... });
```

---

### 3. Streaming with Callbacks

```rust
#[napi]
pub fn stream_prices(
    symbols: Vec<String>,
    callback: JsFunction,
) -> Result<()> {
    let tsfn: ThreadsafeFunction<Price> = callback
        .create_threadsafe_function(0, |ctx| Ok(vec![ctx.value]))?;

    tokio::spawn(async move {
        let mut stream = api::subscribe(symbols).await;

        while let Some(price) = stream.next().await {
            tsfn.call(price, ThreadsafeFunctionCallMode::NonBlocking);
        }
    });

    Ok(())
}
```

**JavaScript:**
```typescript
streamPrices(['AAPL', 'TSLA'], (price) => {
    console.log('Price update:', price);
});
```

---

### 4. Zero-Copy Buffers

```rust
#[napi]
pub struct DataBuffer {
    data: Arc<Vec<f64>>,
}

#[napi]
impl DataBuffer {
    #[napi]
    pub fn as_buffer(&self, env: Env) -> Result<Buffer> {
        unsafe {
            env.create_buffer_with_borrowed_data(
                self.data.as_ptr() as *const u8,
                self.data.len() * 8,
                self.data.clone(),
                |_data, _hint| {},
            )
        }
    }
}
```

**JavaScript:**
```typescript
const buffer = dataBuffer.asBuffer();
const array = new Float64Array(buffer.buffer, buffer.byteOffset, buffer.byteLength / 8);
```

---

### 5. Error Handling

```rust
use thiserror::Error;

#[derive(Error, Debug)]
pub enum TradingError {
    #[error("Invalid symbol: {0}")]
    InvalidSymbol(String),

    #[error("Order rejected: {0}")]
    OrderRejected(String),

    #[error("Network error: {0}")]
    Network(#[from] reqwest::Error),
}

#[napi]
pub async fn place_order(symbol: String) -> Result<String> {
    if symbol.is_empty() {
        return Err(Error::from_reason("Symbol cannot be empty"));
    }

    // Rust error auto-converts to JS Error
    let result = api::place_order(&symbol).await
        .map_err(|e| Error::from_reason(e.to_string()))?;

    Ok(result.order_id)
}
```

**JavaScript:**
```typescript
try {
    await placeOrder('AAPL');
} catch (error) {
    console.error('Order failed:', error.message);
}
```

---

## Performance Tips

### 1. Use `Arc` for Shared Data

```rust
// ❌ BAD: Clone entire vector
pub struct Engine {
    data: Vec<f64>,
}

impl Engine {
    pub fn get_data(&self) -> Vec<f64> {
        self.data.clone() // Expensive!
    }
}

// ✅ GOOD: Share reference with Arc
pub struct Engine {
    data: Arc<Vec<f64>>,
}

impl Engine {
    pub fn get_data(&self) -> Arc<Vec<f64>> {
        Arc::clone(&self.data) // Cheap!
    }
}
```

---

### 2. Batch Operations

```rust
// ❌ BAD: One order at a time
#[napi]
pub async fn submit_order(&self, order: Order) -> Result<String> {
    // ...
}

// ✅ GOOD: Batch multiple orders
#[napi]
pub async fn submit_orders(&self, orders: Vec<Order>) -> Result<Vec<String>> {
    let results = futures::future::join_all(
        orders.into_iter().map(|o| self.submit_one(o))
    ).await;

    results.into_iter().collect()
}
```

---

### 3. Minimize JS ↔ Rust Crossings

```rust
// ❌ BAD: Call JS callback for every item
#[napi]
pub fn process_data(&self, callback: JsFunction) {
    for item in &self.data {
        callback.call(vec![item]); // Slow!
    }
}

// ✅ GOOD: Process in Rust, return batch
#[napi]
pub fn process_data(&self) -> Vec<ProcessedItem> {
    self.data.iter()
        .map(|item| process(item))
        .collect() // Single return
}
```

---

### 4. Use Rayon for Parallelism

```rust
use rayon::prelude::*;

// ❌ BAD: Sequential processing
#[napi]
pub fn calculate_indicators(&self, data: Vec<f64>) -> Vec<f64> {
    data.iter()
        .map(|&x| expensive_calculation(x))
        .collect()
}

// ✅ GOOD: Parallel processing
#[napi]
pub fn calculate_indicators(&self, data: Vec<f64>) -> Vec<f64> {
    data.par_iter()
        .map(|&x| expensive_calculation(x))
        .collect()
}
```

---

## Testing Patterns

### Unit Tests (Rust)

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_market_data_parsing() {
        let json = r#"{"symbol": "AAPL", "price": 150.0}"#;
        let data: MarketData = serde_json::from_str(json).unwrap();

        assert_eq!(data.symbol, "AAPL");
        assert_eq!(data.price, 150.0);
    }

    #[tokio::test]
    async fn test_async_order_submission() {
        let engine = Engine::new("test-key").await.unwrap();
        let order = Order { symbol: "AAPL".into(), quantity: 10 };

        let result = engine.submit_order(order).await;
        assert!(result.is_ok());
    }
}
```

---

### Integration Tests (TypeScript)

```typescript
import { describe, it, expect } from 'vitest';
import { TradingEngine } from '../index.js';

describe('TradingEngine', () => {
  it('should submit orders successfully', async () => {
    const engine = new TradingEngine('test-key');

    const orderId = await engine.submitOrder({
      symbol: 'AAPL',
      side: 'BUY',
      quantity: 10,
      orderType: 'MARKET',
    });

    expect(orderId).toBeTruthy();
    expect(typeof orderId).toBe('string');
  });

  it('should handle errors gracefully', async () => {
    const engine = new TradingEngine('invalid-key');

    await expect(
      engine.submitOrder({ symbol: '', side: 'BUY', quantity: 10, orderType: 'MARKET' })
    ).rejects.toThrow('Symbol cannot be empty');
  });
});
```

---

### Benchmark (Criterion)

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn benchmark_order_processing(c: &mut Criterion) {
    let engine = Engine::new("test").unwrap();

    c.bench_function("process_1000_orders", |b| {
        b.iter(|| {
            let orders = (0..1000).map(|i| Order {
                symbol: format!("SYM{}", i),
                quantity: 10.0,
            }).collect::<Vec<_>>();

            engine.process_orders(black_box(orders))
        });
    });
}

criterion_group!(benches, benchmark_order_processing);
criterion_main!(benches);
```

---

## Troubleshooting

### Build Errors

#### Error: `napi-rs` not found

```bash
# Solution: Install napi-rs CLI
npm install -D @napi-rs/cli
```

#### Error: Rust toolchain missing

```bash
# Solution: Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Add target for cross-compilation
rustup target add x86_64-unknown-linux-gnu
rustup target add x86_64-apple-darwin
rustup target add aarch64-apple-darwin
rustup target add x86_64-pc-windows-msvc
```

#### Error: Linking failed (macOS)

```bash
# Solution: Install Xcode Command Line Tools
xcode-select --install
```

---

### Runtime Errors

#### Error: `Cannot find module 'neural-trader.node'`

```bash
# Solution: Rebuild native module
npm run build

# Or install from npm (if published)
npm install @neural-trader/core
```

#### Error: Segmentation fault

```rust
// Likely cause: Unsafe memory access
// Solution: Review all `unsafe` blocks

// Check for:
// 1. Buffer outliving its source
// 2. Incorrect pointer casts
// 3. Data races in multi-threaded code
```

#### Error: Promise never resolves

```rust
// Likely cause: Tokio runtime not running
// Solution: Use `#[napi]` with async functions

// ❌ BAD
#[napi]
pub fn async_fn() -> Result<JsPromise> {
    // Manual promise handling
}

// ✅ GOOD
#[napi]
pub async fn async_fn() -> Result<String> {
    // napi-rs handles Promise automatically
}
```

---

## Resources

### Documentation

- **napi-rs:** https://napi.rs
- **Tokio:** https://tokio.rs
- **Polars:** https://pola-rs.github.io/polars-book/
- **Candle:** https://github.com/huggingface/candle

### Examples

- **napi-rs examples:** https://github.com/napi-rs/napi-rs/tree/main/examples
- **Native Node modules:** https://nodejs.org/api/n-api.html

### Community

- **Discord:** https://discord.gg/napi-rs
- **GitHub Discussions:** https://github.com/napi-rs/napi-rs/discussions

---

## Appendix: File Structure

```
neural-trader-rs/
├── .github/
│   └── workflows/
│       └── build.yml           # CI/CD for all platforms
├── crates/
│   ├── neural-core/            # Pure Rust logic (no Node.js)
│   │   ├── Cargo.toml
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── execution/
│   │   │   │   ├── mod.rs
│   │   │   │   ├── engine.rs
│   │   │   │   └── buffer.rs
│   │   │   ├── neural/
│   │   │   │   ├── mod.rs
│   │   │   │   ├── model.rs
│   │   │   │   └── inference.rs
│   │   │   └── market_data/
│   │   │       ├── mod.rs
│   │   │       ├── feed.rs
│   │   │       └── processor.rs
│   │   └── tests/
│   │       └── integration.rs
│   ├── neural-bindings/        # napi-rs bindings
│   │   ├── Cargo.toml
│   │   ├── build.rs
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── execution.rs    # ExecutionEngine bindings
│   │       ├── neural.rs       # NeuralModel bindings
│   │       └── market_data.rs  # MarketDataProcessor bindings
│   └── neural-cli/             # Standalone CLI (fallback)
│       ├── Cargo.toml
│       └── src/
│           └── main.rs
├── npm/                        # Pre-built binaries
│   ├── linux-x64-gnu/
│   │   ├── package.json
│   │   └── neural-trader.node
│   ├── darwin-x64/
│   │   ├── package.json
│   │   └── neural-trader.node
│   ├── darwin-arm64/
│   │   ├── package.json
│   │   └── neural-trader.node
│   └── win32-x64-msvc/
│       ├── package.json
│       └── neural-trader.node
├── __test__/                   # JavaScript tests
│   ├── execution.test.ts
│   ├── neural.test.ts
│   └── integration.test.ts
├── example/                    # Usage examples
│   ├── basic-trading.ts
│   ├── streaming.ts
│   └── full-bot.ts
├── Cargo.toml                  # Workspace root
├── package.json                # npm package
├── index.js                    # JS entry point
├── index.d.ts                  # TypeScript definitions (auto-generated)
└── README.md
```

---

**Quick Reference Version:** 1.0.0
**See also:** `01-crate-ecosystem-and-interop.md` for full details
