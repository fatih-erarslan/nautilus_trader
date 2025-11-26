# Neural Trading Rust Port - Crate Ecosystem and Node.js Interop

**Version:** 1.0.0
**Date:** 2025-11-12
**Status:** Design Complete
**Cross-References:** [Architecture](03_Architecture.md) | [Parity](02_Parity_Requirements.md) | [Node.js Interop Docs](../../docs/rust-architecture/05-nodejs-interop-strategy.md)

---

## Table of Contents

1. [Primary Path: napi-rs Strategy](#primary-path-napi-rs-strategy)
2. [TypeScript Type Definitions](#typescript-type-definitions)
3. [Async Bridge (Promise ↔ Future)](#async-bridge-promise--future)
4. [Memory Management and Lifecycle](#memory-management-and-lifecycle)
5. [Fallback Strategies](#fallback-strategies)
6. [Build Targets](#build-targets)
7. [Complete Crate Dependency Matrix](#complete-crate-dependency-matrix)
8. [Feature Gates and Build Flags](#feature-gates-and-build-flags)
9. [Code Examples for Each Approach](#code-examples-for-each-approach)

---

## Primary Path: napi-rs Strategy

### Why napi-rs?

**Technical Advantages:**
- ✅ **Zero-copy**: Direct memory access between Rust and Node.js
- ✅ **Type-safe**: Auto-generated TypeScript definitions
- ✅ **Async**: Native Promise support with Tokio integration
- ✅ **Performance**: <0.1ms overhead for FFI calls
- ✅ **Cross-platform**: Official support for all major platforms
- ✅ **Production-proven**: Used by Prisma, SWC, Parcel, Next.js

**Performance Comparison:**

| Approach | FFI Overhead | Throughput | Use Case |
|----------|-------------|------------|----------|
| napi-rs | <0.1ms | 1M ops/sec | ✅ Primary (production) |
| Neon | <0.15ms | 800K ops/sec | Fallback 1 (Windows) |
| WASI | <1ms | 100K ops/sec | Fallback 2 (portable) |
| CLI+IPC | 5-10ms | 1K ops/sec | Fallback 3 (simple) |

---

### Project Structure

```
neural-trader-rs/
├── crates/
│   ├── nt-core/              # Shared Rust logic (no FFI)
│   ├── nt-strategies/        # Strategy implementations
│   ├── nt-execution/         # Order execution
│   └── nt-napi/              # ⭐ Node.js bindings crate
│       ├── Cargo.toml
│       ├── build.rs          # Pre-build script
│       ├── package.json      # NPM package config
│       ├── src/
│       │   ├── lib.rs        # Main FFI entry point
│       │   ├── market_data.rs
│       │   ├── strategies.rs
│       │   ├── execution.rs
│       │   ├── portfolio.rs
│       │   └── risk.rs
│       └── __test__/
│           └── index.spec.ts # JS integration tests
└── npm/
    └── neural-trader/        # 📦 Published NPM package
        ├── package.json
        ├── index.js          # JS entry point
        ├── index.d.ts        # TypeScript definitions (auto-generated)
        └── lib/
            ├── neural-trader.darwin-arm64.node
            ├── neural-trader.darwin-x64.node
            ├── neural-trader.linux-x64-gnu.node
            ├── neural-trader.linux-x64-musl.node
            └── neural-trader.win32-x64-msvc.node
```

---

### Cargo.toml Configuration

```toml
# crates/nt-napi/Cargo.toml

[package]
name = "nt-napi"
version = "1.0.0"
edition = "2021"
authors = ["Neural Trader Team"]
description = "Node.js bindings for neural-trader"

[lib]
crate-type = ["cdylib"]  # Dynamic library for Node.js

[dependencies]
# napi-rs core
napi = { version = "2.16", features = ["async", "tokio_rt", "napi8"] }
napi-derive = "2.16"

# Async runtime
tokio = { version = "1.35", features = ["rt-multi-thread", "macros"] }

# Internal crates
nt-core = { path = "../nt-core" }
nt-market-data = { path = "../nt-market-data" }
nt-strategies = { path = "../nt-strategies" }
nt-execution = { path = "../nt-execution" }
nt-portfolio = { path = "../nt-portfolio" }
nt-risk = { path = "../nt-risk" }

# Serialization
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"

[build-dependencies]
napi-build = "2.1"

[profile.release]
lto = true                   # Link-time optimization
codegen-units = 1           # Better optimization (slower compile)
strip = true                # Strip debug symbols
opt-level = 3               # Maximum optimization
panic = "abort"             # Smaller binary size

[profile.dev]
opt-level = 0
debug = true
```

---

### API Surface Design

#### Market Data Bindings

```rust
// crates/nt-napi/src/market_data.rs

use napi::bindgen_prelude::*;
use napi_derive::napi;
use nt_core::{Symbol, Quote, Bar};
use nt_market_data::AlpacaMarketData;

/// JavaScript Quote object
#[napi(object)]
pub struct JsQuote {
    pub symbol: String,
    pub timestamp: i64,        // Unix milliseconds
    pub bid: f64,
    pub ask: f64,
    pub bid_size: f64,
    pub ask_size: f64,
}

impl From<Quote> for JsQuote {
    fn from(quote: Quote) -> Self {
        Self {
            symbol: quote.symbol.to_string(),
            timestamp: quote.timestamp.timestamp_millis(),
            bid: quote.bid.to_f64().unwrap(),
            ask: quote.ask.to_f64().unwrap(),
            bid_size: quote.bid_size.to_f64().unwrap(),
            ask_size: quote.ask_size.to_f64().unwrap(),
        }
    }
}

/// Market Data Stream
#[napi]
pub struct MarketDataStream {
    inner: Arc<Mutex<AlpacaMarketData>>,
}

#[napi]
impl MarketDataStream {
    /// Create new market data stream
    #[napi(constructor)]
    pub fn new(api_key: String, secret_key: String, paper_trading: bool) -> Result<Self> {
        let provider = AlpacaMarketData::new(api_key, secret_key, paper_trading)
            .map_err(|e| Error::from_reason(e.to_string()))?;

        Ok(Self {
            inner: Arc::new(Mutex::new(provider)),
        })
    }

    /// Subscribe to real-time quotes
    #[napi]
    pub async fn subscribe_quotes(
        &self,
        symbols: Vec<String>,
        callback: JsFunction,
    ) -> Result<()> {
        let symbols: Vec<Symbol> = symbols
            .into_iter()
            .map(|s| Symbol::from(s))
            .collect();

        let provider = self.inner.lock().await;
        let mut quote_rx = provider.subscribe_quotes(&symbols)
            .await
            .map_err(|e| Error::from_reason(e.to_string()))?;

        // Create threadsafe function for callbacks
        let tsfn: ThreadsafeFunction<JsQuote, ErrorStrategy::Fatal> =
            callback.create_threadsafe_function(0, |ctx| {
                Ok(vec![ctx.value])
            })?;

        // Spawn background task to forward quotes to JS
        tokio::spawn(async move {
            while let Some(quote) = quote_rx.recv().await {
                let js_quote = JsQuote::from(quote);
                tsfn.call(js_quote, ThreadsafeFunctionCallMode::NonBlocking);
            }
        });

        Ok(())
    }

    /// Get current quote (async)
    #[napi]
    pub async fn get_quote(&self, symbol: String) -> Result<JsQuote> {
        let symbol = Symbol::from(symbol);
        let provider = self.inner.lock().await;

        let quote = provider.get_quote(&symbol)
            .await
            .map_err(|e| Error::from_reason(e.to_string()))?;

        Ok(JsQuote::from(quote))
    }

    /// Get historical bars
    #[napi]
    pub async fn get_historical_bars(
        &self,
        symbol: String,
        start: i64,          // Unix milliseconds
        end: i64,
        timeframe: String,   // "1Min", "5Min", "1Hour", "1Day"
    ) -> Result<Vec<JsBar>> {
        let symbol = Symbol::from(symbol);
        let start = DateTime::from_timestamp_millis(start)
            .ok_or_else(|| Error::from_reason("Invalid start timestamp"))?;
        let end = DateTime::from_timestamp_millis(end)
            .ok_or_else(|| Error::from_reason("Invalid end timestamp"))?;

        let provider = self.inner.lock().await;
        let bars = provider.get_bars(&symbol, start, end, &timeframe)
            .await
            .map_err(|e| Error::from_reason(e.to_string()))?;

        Ok(bars.into_iter().map(JsBar::from).collect())
    }
}

/// Historical bar data
#[napi(object)]
pub struct JsBar {
    pub timestamp: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

impl From<Bar> for JsBar {
    fn from(bar: Bar) -> Self {
        Self {
            timestamp: bar.timestamp.timestamp_millis(),
            open: bar.open.to_f64().unwrap(),
            high: bar.high.to_f64().unwrap(),
            low: bar.low.to_f64().unwrap(),
            close: bar.close.to_f64().unwrap(),
            volume: bar.volume as f64,
        }
    }
}
```

#### Strategy Bindings

```rust
// crates/nt-napi/src/strategies.rs

use napi::bindgen_prelude::*;
use napi_derive::napi;
use nt_strategies::{MomentumStrategy, MirrorStrategy, StrategyConfig};

/// JavaScript Signal object
#[napi(object)]
pub struct JsSignal {
    pub id: String,
    pub strategy_id: String,
    pub symbol: String,
    pub direction: String,      // "long", "short", "close"
    pub confidence: f64,        // 0.0-1.0
    pub entry_price: Option<f64>,
    pub stop_loss: Option<f64>,
    pub take_profit: Option<f64>,
    pub reasoning: String,
    pub timestamp: i64,
}

/// Strategy runner
#[napi]
pub struct StrategyRunner {
    strategies: Arc<Mutex<Vec<Box<dyn Strategy>>>>,
    signal_tx: mpsc::Sender<Signal>,
}

#[napi]
impl StrategyRunner {
    /// Create new strategy runner
    #[napi(constructor)]
    pub fn new() -> Self {
        let (signal_tx, _) = mpsc::channel(100);
        Self {
            strategies: Arc::new(Mutex::new(Vec::new())),
            signal_tx,
        }
    }

    /// Add momentum strategy
    #[napi]
    pub async fn add_momentum_strategy(
        &mut self,
        config: JsObject,
    ) -> Result<String> {
        let config: StrategyConfig = serde_json::from_value(
            serde_json::to_value(&config)?
        )?;

        let strategy = MomentumStrategy::new(config)
            .map_err(|e| Error::from_reason(e.to_string()))?;

        let id = strategy.id().to_string();
        self.strategies.lock().await.push(Box::new(strategy));

        Ok(id)
    }

    /// Generate signals
    #[napi]
    pub async fn generate_signals(&self) -> Result<Vec<JsSignal>> {
        let strategies = self.strategies.lock().await;
        let mut all_signals = Vec::new();

        for strategy in strategies.iter() {
            let signals = strategy.generate_signals()
                .await
                .map_err(|e| Error::from_reason(e.to_string()))?;

            for signal in signals {
                all_signals.push(JsSignal::from(signal));
            }
        }

        Ok(all_signals)
    }

    /// Subscribe to signals (stream)
    #[napi]
    pub fn subscribe_signals(&self, callback: JsFunction) -> Result<()> {
        let mut signal_rx = self.signal_tx.subscribe();

        let tsfn: ThreadsafeFunction<JsSignal, ErrorStrategy::Fatal> =
            callback.create_threadsafe_function(0, |ctx| {
                Ok(vec![ctx.value])
            })?;

        tokio::spawn(async move {
            while let Ok(signal) = signal_rx.recv().await {
                let js_signal = JsSignal::from(signal);
                tsfn.call(js_signal, ThreadsafeFunctionCallMode::NonBlocking);
            }
        });

        Ok(())
    }
}
```

#### Order Execution Bindings

```rust
// crates/nt-napi/src/execution.rs

use napi::bindgen_prelude::*;
use napi_derive::napi;
use nt_execution::{OrderManager, Order, OrderRequest};

/// JavaScript Order object
#[napi(object)]
pub struct JsOrder {
    pub id: String,
    pub symbol: String,
    pub side: String,           // "buy", "sell"
    pub order_type: String,     // "market", "limit", "stop_loss"
    pub quantity: u32,
    pub limit_price: Option<f64>,
    pub status: String,         // "pending", "filled", "cancelled"
    pub filled_qty: u32,
    pub filled_avg_price: Option<f64>,
}

/// Order executor
#[napi]
pub struct OrderExecutor {
    manager: Arc<Mutex<OrderManager>>,
}

#[napi]
impl OrderExecutor {
    /// Create new order executor
    #[napi(constructor)]
    pub fn new(api_key: String, secret_key: String, paper_trading: bool) -> Result<Self> {
        let manager = OrderManager::new(api_key, secret_key, paper_trading)
            .map_err(|e| Error::from_reason(e.to_string()))?;

        Ok(Self {
            manager: Arc::new(Mutex::new(manager)),
        })
    }

    /// Place market order
    #[napi]
    pub async fn place_market_order(
        &self,
        symbol: String,
        side: String,
        quantity: u32,
    ) -> Result<JsOrder> {
        let order_request = OrderRequest {
            symbol: Symbol::from(symbol),
            side: match side.as_str() {
                "buy" => OrderSide::Buy,
                "sell" => OrderSide::Sell,
                _ => return Err(Error::from_reason("Invalid side")),
            },
            order_type: OrderType::Market,
            quantity,
            limit_price: None,
            stop_price: None,
            time_in_force: TimeInForce::Day,
        };

        let manager = self.manager.lock().await;
        let order = manager.place_order(order_request)
            .await
            .map_err(|e| Error::from_reason(e.to_string()))?;

        Ok(JsOrder::from(order))
    }

    /// Get order status
    #[napi]
    pub async fn get_order_status(&self, order_id: String) -> Result<JsOrder> {
        let manager = self.manager.lock().await;
        let order = manager.get_order(&order_id)
            .await
            .map_err(|e| Error::from_reason(e.to_string()))?;

        Ok(JsOrder::from(order))
    }

    /// Cancel order
    #[napi]
    pub async fn cancel_order(&self, order_id: String) -> Result<()> {
        let manager = self.manager.lock().await;
        manager.cancel_order(&order_id)
            .await
            .map_err(|e| Error::from_reason(e.to_string()))
    }
}
```

---

## TypeScript Type Definitions

### Auto-Generated Definitions

napi-rs automatically generates TypeScript definitions from Rust code:

```typescript
// index.d.ts (auto-generated)

/** Market quote data */
export interface JsQuote {
  symbol: string;
  timestamp: number;
  bid: number;
  ask: number;
  bidSize: number;
  askSize: number;
}

/** Historical bar data */
export interface JsBar {
  timestamp: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

/** Trading signal */
export interface JsSignal {
  id: string;
  strategyId: string;
  symbol: string;
  direction: 'long' | 'short' | 'close';
  confidence: number;
  entryPrice?: number;
  stopLoss?: number;
  takeProfit?: number;
  reasoning: string;
  timestamp: number;
}

/** Market data stream */
export class MarketDataStream {
  constructor(apiKey: string, secretKey: string, paperTrading: boolean);

  /** Subscribe to real-time quotes */
  subscribeQuotes(symbols: string[], callback: (quote: JsQuote) => void): Promise<void>;

  /** Get current quote */
  getQuote(symbol: string): Promise<JsQuote>;

  /** Get historical bars */
  getHistoricalBars(
    symbol: string,
    start: number,
    end: number,
    timeframe: string
  ): Promise<JsBar[]>;
}

/** Strategy runner */
export class StrategyRunner {
  constructor();

  /** Add momentum strategy */
  addMomentumStrategy(config: Record<string, any>): Promise<string>;

  /** Generate signals */
  generateSignals(): Promise<JsSignal[]>;

  /** Subscribe to signal stream */
  subscribeSignals(callback: (signal: JsSignal) => void): void;
}

/** Order executor */
export class OrderExecutor {
  constructor(apiKey: string, secretKey: string, paperTrading: boolean);

  /** Place market order */
  placeMarketOrder(symbol: string, side: 'buy' | 'sell', quantity: number): Promise<JsOrder>;

  /** Get order status */
  getOrderStatus(orderId: string): Promise<JsOrder>;

  /** Cancel order */
  cancelOrder(orderId: string): Promise<void>;
}
```

---

## Async Bridge (Promise ↔ Future)

### Promise → Future (JS calls Rust)

```rust
#[napi]
pub async fn async_operation(input: String) -> Result<String> {
    // This function returns a Future
    // napi-rs automatically converts it to a Promise
    tokio::time::sleep(Duration::from_millis(100)).await;
    Ok(format!("Processed: {}", input))
}
```

```typescript
// JavaScript usage
const result = await asyncOperation("test");
console.log(result); // "Processed: test"
```

### Future → Promise (Rust calls JS)

```rust
#[napi]
pub fn call_js_async(callback: JsFunction) -> Result<()> {
    // Create threadsafe function
    let tsfn: ThreadsafeFunction<String, ErrorStrategy::Fatal> =
        callback.create_threadsafe_function(0, |ctx| {
            Ok(vec![ctx.value])
        })?;

    // Spawn Rust async task
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_secs(1)).await;
        tsfn.call("Hello from Rust!".to_string(), ThreadsafeFunctionCallMode::Blocking);
    });

    Ok(())
}
```

```typescript
// JavaScript usage
callJsAsync((message: string) => {
  console.log(message); // "Hello from Rust!" (after 1 second)
});
```

### Streaming Pattern (Rust → JS)

```rust
#[napi]
pub fn subscribe_to_stream(callback: JsFunction) -> Result<()> {
    let tsfn: ThreadsafeFunction<i32, ErrorStrategy::Fatal> =
        callback.create_threadsafe_function(0, |ctx| {
            Ok(vec![ctx.value])
        })?;

    tokio::spawn(async move {
        for i in 0..10 {
            tsfn.call(i, ThreadsafeFunctionCallMode::NonBlocking);
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    });

    Ok(())
}
```

```typescript
// JavaScript usage
subscribeToStream((value: number) => {
  console.log(`Received: ${value}`);
});
// Output: Received: 0, 1, 2, ..., 9 (every 100ms)
```

---

## Memory Management and Lifecycle

### Zero-Copy Buffer Transfer

```rust
use napi::bindgen_prelude::*;

#[napi]
pub fn process_buffer(buffer: Buffer) -> Result<Buffer> {
    // Zero-copy read from Node.js buffer
    let input_data: &[u8] = buffer.as_ref();

    // Process data
    let mut output_data = Vec::with_capacity(input_data.len());
    for &byte in input_data {
        output_data.push(byte ^ 0xFF); // Invert bits
    }

    // Return as Node.js Buffer (ownership transferred)
    Ok(Buffer::from(output_data))
}
```

```typescript
const input = Buffer.from([0x01, 0x02, 0x03]);
const output = processBuffer(input);
console.log(output); // <Buffer fe fd fc>
```

### External References (Opaque Handles)

```rust
use napi::bindgen_prelude::*;

#[napi]
pub fn create_large_dataframe() -> Result<External<DataFrame>> {
    // Create large data structure
    let df = create_dataframe_with_million_rows()?;

    // Return as opaque handle (no serialization)
    Ok(External::new(df))
}

#[napi]
pub fn process_dataframe(df_handle: External<DataFrame>) -> Result<Vec<f64>> {
    // Access data without copying
    let df = &*df_handle;

    // Extract column
    let values = df.column("close")?.f64()?.to_vec();
    Ok(values)
}
```

```typescript
// JavaScript usage
const df = createLargeDataframe(); // Fast (no copy)
const values = processDataframe(df); // Fast (no copy)
```

### Reference Counting with Arc

```rust
#[napi]
pub struct SharedState {
    inner: Arc<Mutex<HashMap<String, String>>>,
}

#[napi]
impl SharedState {
    #[napi(constructor)]
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    #[napi]
    pub async fn set(&self, key: String, value: String) {
        self.inner.lock().await.insert(key, value);
    }

    #[napi]
    pub async fn get(&self, key: String) -> Option<String> {
        self.inner.lock().await.get(&key).cloned()
    }
}
```

```typescript
const state = new SharedState();
await state.set("key", "value");
const value = await state.get("key"); // "value"
```

---

## Fallback Strategies

### Fallback 1: Neon (Windows-Friendly)

**When to use:** napi-rs build issues on Windows

```rust
// Using Neon instead of napi-rs
use neon::prelude::*;

fn get_quote(mut cx: FunctionContext) -> JsResult<JsObject> {
    let symbol = cx.argument::<JsString>(0)?.value(&mut cx);

    let quote = get_quote_sync(&symbol)
        .or_else(|e| cx.throw_error(e.to_string()))?;

    let obj = cx.empty_object();
    let symbol_val = cx.string(quote.symbol);
    obj.set(&mut cx, "symbol", symbol_val)?;

    let bid_val = cx.number(quote.bid);
    obj.set(&mut cx, "bid", bid_val)?;

    Ok(obj)
}

#[neon::main]
fn main(mut cx: ModuleContext) -> NeonResult<()> {
    cx.export_function("getQuote", get_quote)?;
    Ok(())
}
```

**Tradeoffs:**
- ✅ Better Windows support
- ✅ Mature ecosystem
- ❌ No auto-generated TypeScript definitions
- ❌ 15% slower than napi-rs
- ❌ More verbose API

### Fallback 2: WASI + Wasmtime

**When to use:** Maximum portability, no native modules

```rust
// Compile to WASM
#[no_mangle]
pub extern "C" fn get_quote(symbol_ptr: *const u8, symbol_len: usize) -> *mut u8 {
    let symbol = unsafe {
        std::str::from_utf8_unchecked(std::slice::from_raw_parts(symbol_ptr, symbol_len))
    };

    let quote = get_quote_sync(symbol).unwrap();
    let json = serde_json::to_string(&quote).unwrap();

    // Allocate and return
    let bytes = json.into_bytes();
    let ptr = bytes.as_ptr() as *mut u8;
    std::mem::forget(bytes);
    ptr
}
```

```bash
# Build WASM module
cargo build --target wasm32-wasi --release
wasm-opt -O3 target/wasm32-wasi/release/neural_trader.wasm -o neural_trader.wasm
```

```typescript
// JavaScript usage with Wasmtime
import { WASI } from 'wasi';
import { readFile } from 'fs/promises';

const wasm = await readFile('./neural_trader.wasm');
const wasi = new WASI();
const { instance } = await WebAssembly.instantiate(wasm, wasi.getImportObject());

wasi.start(instance);

const getQuote = instance.exports.get_quote;
const result = getQuote(symbolPtr, symbolLen);
```

**Tradeoffs:**
- ✅ Maximum portability (runs anywhere)
- ✅ No build toolchain required
- ✅ Sandboxed execution
- ❌ 30% slower than native
- ❌ More complex memory management
- ❌ Limited async support

### Fallback 3: CLI + STDIO (Last Resort)

**When to use:** Simple integration, no FFI complexity

```rust
// CLI binary
fn main() -> anyhow::Result<()> {
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();

    for line in stdin.lock().lines() {
        let request: Request = serde_json::from_str(&line?)?;

        let response = match request {
            Request::GetQuote { symbol } => {
                let quote = get_quote_sync(&symbol)?;
                Response::Quote(quote)
            }
            Request::PlaceOrder { order } => {
                let result = place_order_sync(order)?;
                Response::OrderResult(result)
            }
        };

        serde_json::to_writer(&stdout, &response)?;
        writeln!(&stdout)?;
    }

    Ok(())
}
```

```typescript
// JavaScript wrapper
import { spawn } from 'child_process';
import { createInterface } from 'readline';

class CliClient {
  private process: ChildProcess;
  private rl: ReadLine;

  constructor() {
    this.process = spawn('neural-trader-cli');
    this.rl = createInterface({ input: this.process.stdout });
  }

  async getQuote(symbol: string): Promise<Quote> {
    return new Promise((resolve, reject) => {
      const request = { GetQuote: { symbol } };
      this.process.stdin.write(JSON.stringify(request) + '\n');

      this.rl.once('line', (line) => {
        const response = JSON.parse(line);
        resolve(response.Quote);
      });
    });
  }
}
```

**Tradeoffs:**
- ✅ Zero FFI complexity
- ✅ Easy debugging
- ✅ Works everywhere
- ❌ 2-3x slower (IPC overhead)
- ❌ No streaming support
- ❌ Process management complexity

---

## Build Targets

### Supported Platforms

| Platform | Target Triple | Status | Priority |
|----------|--------------|--------|----------|
| macOS Intel | `x86_64-apple-darwin` | ✅ Tier 1 | P0 |
| macOS ARM | `aarch64-apple-darwin` | ✅ Tier 1 | P0 |
| Linux GNU | `x86_64-unknown-linux-gnu` | ✅ Tier 1 | P0 |
| Linux musl | `x86_64-unknown-linux-musl` | ✅ Tier 1 | P1 |
| Windows MSVC | `x86_64-pc-windows-msvc` | ✅ Tier 1 | P0 |
| Linux ARM | `aarch64-unknown-linux-gnu` | ⚠️ Tier 2 | P1 |
| Linux ARM musl | `aarch64-unknown-linux-musl` | ⚠️ Tier 2 | P2 |

### Cross-Compilation Setup

```bash
# Install cross-compilation tool
cargo install cross

# Build for all platforms
cross build --release --target x86_64-apple-darwin
cross build --release --target aarch64-apple-darwin
cross build --release --target x86_64-unknown-linux-gnu
cross build --release --target x86_64-unknown-linux-musl
cross build --release --target x86_64-pc-windows-msvc
```

### NPM Package Configuration

```json
{
  "name": "neural-trader",
  "version": "1.0.0",
  "main": "index.js",
  "types": "index.d.ts",
  "napi": {
    "name": "neural-trader",
    "triples": {
      "defaults": true,
      "additional": [
        "x86_64-unknown-linux-musl",
        "aarch64-unknown-linux-gnu",
        "aarch64-apple-darwin"
      ]
    }
  },
  "scripts": {
    "build": "napi build --platform --release",
    "build:debug": "napi build --platform",
    "prepublishOnly": "napi prepublish -t npm",
    "artifacts": "napi artifacts",
    "test": "node --test"
  },
  "devDependencies": {
    "@napi-rs/cli": "^2.18.0",
    "typescript": "^5.3.0"
  },
  "optionalDependencies": {
    "neural-trader-darwin-x64": "1.0.0",
    "neural-trader-darwin-arm64": "1.0.0",
    "neural-trader-linux-x64-gnu": "1.0.0",
    "neural-trader-linux-x64-musl": "1.0.0",
    "neural-trader-win32-x64-msvc": "1.0.0"
  }
}
```

### GitHub Actions CI/CD

```yaml
# .github/workflows/build.yml
name: Build Native Modules

on:
  push:
    branches: [main]
  pull_request:

jobs:
  build:
    strategy:
      matrix:
        settings:
          - host: macos-latest
            target: x86_64-apple-darwin
          - host: macos-latest
            target: aarch64-apple-darwin
          - host: ubuntu-latest
            target: x86_64-unknown-linux-gnu
          - host: ubuntu-latest
            target: x86_64-unknown-linux-musl
          - host: windows-latest
            target: x86_64-pc-windows-msvc

    runs-on: ${{ matrix.settings.host }}

    steps:
      - uses: actions/checkout@v4

      - name: Setup Rust
        uses: dtolnay/rust-toolchain@stable
        with:
          targets: ${{ matrix.settings.target }}

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: 20

      - name: Install dependencies
        run: npm install

      - name: Build
        run: npm run build -- --target ${{ matrix.settings.target }}

      - name: Upload artifacts
        uses: actions/upload-artifact@v4
        with:
          name: bindings-${{ matrix.settings.target }}
          path: npm/neural-trader/lib/*.node
```

---

## Complete Crate Dependency Matrix

### Core Dependencies

```toml
[workspace]
members = [
    "crates/nt-core",
    "crates/nt-config",
    "crates/nt-market-data",
    "crates/nt-features",
    "crates/nt-strategies",
    "crates/nt-signals",
    "crates/nt-portfolio",
    "crates/nt-risk",
    "crates/nt-execution",
    "crates/nt-backtesting",
    "crates/nt-neural",
    "crates/nt-news",
    "crates/nt-agentdb",
    "crates/nt-api",
    "crates/nt-napi",
    "crates/nt-cli",
]

[workspace.dependencies]
# Async runtime
tokio = { version = "1.35", features = ["full"] }
async-trait = "0.1"

# Data processing
polars = { version = "0.36", features = ["lazy", "temporal", "dtype-full"] }
ndarray = "0.15"

# Math & stats
nalgebra = "0.32"
statrs = "0.17"

# Decimal precision
rust_decimal = { version = "1.33", features = ["serde-with-float"] }

# Time & date
chrono = { version = "0.4", features = ["serde"] }

# Serialization
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
bincode = "1.3"

# HTTP client
reqwest = { version = "0.11", features = ["json", "rustls-tls"] }

# WebSocket
tokio-tungstenite = { version = "0.21", features = ["rustls-tls-webpki-roots"] }

# Database
sqlx = { version = "0.7", features = ["runtime-tokio-rustls", "postgres"] }

# Logging & tracing
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter", "json"] }

# Error handling
thiserror = "1.0"
anyhow = "1.0"

# Testing
criterion = "0.5"
proptest = "1.4"
```

### Dependency Graph

```
nt-napi (Node.js bindings)
  ├─ nt-core
  ├─ nt-config
  │   └─ nt-core
  ├─ nt-market-data
  │   └─ nt-core
  ├─ nt-features
  │   └─ nt-core
  ├─ nt-strategies
  │   ├─ nt-core
  │   ├─ nt-features
  │   └─ nt-neural (optional)
  ├─ nt-signals
  │   ├─ nt-core
  │   └─ nt-strategies
  ├─ nt-portfolio
  │   └─ nt-core
  ├─ nt-risk
  │   ├─ nt-core
  │   └─ nt-portfolio
  ├─ nt-execution
  │   ├─ nt-core
  │   └─ nt-risk
  └─ nt-agentdb
      └─ nt-core
```

---

## Feature Gates and Build Flags

### Cargo Features

```toml
[features]
default = ["alpaca", "rest-api"]

# Trading providers
alpaca = ["nt-market-data/alpaca"]
polygon = ["nt-market-data/polygon"]
binance = ["nt-market-data/binance"]

# APIs
rest-api = ["nt-api/rest"]
grpc-api = ["nt-api/grpc"]
graphql-api = ["nt-api/graphql"]

# Neural models
neural-nhits = ["nt-neural/nhits"]
neural-lstm = ["nt-neural/lstm"]
neural-transformer = ["nt-neural/transformer"]

# GPU acceleration
cuda = ["nt-neural/cuda", "nt-features/cuda"]
opencl = ["nt-neural/opencl"]

# Databases
postgres = ["sqlx/postgres"]
sqlite = ["sqlx/sqlite"]
mysql = ["sqlx/mysql"]

# Observability
metrics = ["prometheus"]
tracing = ["opentelemetry"]
profiling = ["pprof"]

# Build profiles
production = ["metrics", "tracing"]
development = ["profiling"]
minimal = []
```

### Conditional Compilation

```rust
#[cfg(feature = "cuda")]
use cudarc::driver::CudaDevice;

#[cfg(feature = "cuda")]
pub fn calculate_features_gpu(data: &[f64]) -> Result<Vec<f64>> {
    let device = CudaDevice::new(0)?;
    // GPU implementation
}

#[cfg(not(feature = "cuda"))]
pub fn calculate_features_gpu(data: &[f64]) -> Result<Vec<f64>> {
    // Fallback to CPU
    calculate_features_cpu(data)
}
```

---

## Code Examples for Each Approach

### Complete Example: napi-rs

```typescript
// example.ts
import {
  MarketDataStream,
  StrategyRunner,
  OrderExecutor,
} from 'neural-trader';

async function main() {
  // Initialize market data
  const marketData = new MarketDataStream(
    process.env.ALPACA_API_KEY!,
    process.env.ALPACA_SECRET_KEY!,
    true // paper trading
  );

  // Subscribe to quotes
  marketData.subscribeQuotes(['AAPL', 'GOOGL'], (quote) => {
    console.log(`${quote.symbol}: ${quote.bid} x ${quote.ask}`);
  });

  // Initialize strategy runner
  const strategies = new StrategyRunner();
  await strategies.addMomentumStrategy({
    period: 20,
    threshold: 0.02,
  });

  // Subscribe to signals
  strategies.subscribeSignals((signal) => {
    console.log(`Signal: ${signal.direction} ${signal.symbol} @ ${signal.confidence}`);
  });

  // Initialize order executor
  const executor = new OrderExecutor(
    process.env.ALPACA_API_KEY!,
    process.env.ALPACA_SECRET_KEY!,
    true
  );

  // Place test order
  const order = await executor.placeMarketOrder('AAPL', 'buy', 10);
  console.log(`Order placed: ${order.id}`);
}

main().catch(console.error);
```

---

## Cross-References

- **Architecture:** [03_Architecture.md](03_Architecture.md)
- **Memory & AgentDB:** [05_Memory_and_AgentDB.md](05_Memory_and_AgentDB.md)
- **Strategy Implementation:** [06_Strategy_and_Sublinear_Solvers.md](06_Strategy_and_Sublinear_Solvers.md)
- **Streaming:** [07_Streaming_and_Midstreamer.md](07_Streaming_and_Midstreamer.md)

---

**Document Status:** ✅ Complete
**Last Updated:** 2025-11-12
**Next Review:** Phase 2 (Week 5)
**Owner:** Backend Developer + DevOps
