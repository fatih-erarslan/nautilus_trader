# Node.js Interoperability Strategy

## Multi-Tier Fallback Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   TypeScript/JavaScript                      │
│                       Application                            │
└───────────────────────┬─────────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┬─────────────┐
        │               │               │             │
        ▼               ▼               ▼             ▼
┌──────────────┐ ┌──────────────┐ ┌─────────┐ ┌──────────┐
│ Tier 1:      │ │ Tier 2:      │ │ Tier 3: │ │ Tier 4:  │
│ napi-rs      │ │ WASI/WASM    │ │ CLI+IPC │ │ gRPC     │
│ (Primary)    │ │ (Fallback 1) │ │ (FB 2)  │ │ (FB 3)   │
└──────┬───────┘ └──────┬───────┘ └────┬────┘ └────┬─────┘
       │                │               │           │
       └────────────────┴───────────────┴───────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │      Rust Core Library        │
        │   (Shared Implementation)     │
        └───────────────────────────────┘
```

## Tier 1: napi-rs (Primary)

**Best for:** Performance-critical paths, native module support

### Project Structure

```
neural-trader-rs/
├── crates/
│   ├── nt-core/              # Shared Rust logic
│   └── nt-napi/              # Node.js bindings
│       ├── Cargo.toml
│       ├── build.rs
│       ├── src/
│       │   ├── lib.rs
│       │   ├── market_data.rs
│       │   ├── strategies.rs
│       │   ├── signals.rs
│       │   └── execution.rs
│       └── package.json
└── npm/
    └── neural-trader/
        ├── package.json
        ├── index.js          # JS entry point
        ├── index.d.ts        # TypeScript definitions
        └── lib/
            └── bindings.node # Built native module
```

### Implementation

#### Cargo.toml Configuration

```toml
[package]
name = "nt-napi"
version = "1.0.0"
edition = "2021"

[lib]
crate-type = ["cdylib"]  # Dynamic library for Node.js

[dependencies]
napi = { version = "2.16", features = ["async", "tokio_rt"] }
napi-derive = "2.16"
tokio = { version = "1.35", features = ["rt-multi-thread"] }
nt-core = { path = "../nt-core" }

[build-dependencies]
napi-build = "2.1"

[profile.release]
lto = true                    # Link-time optimization
codegen-units = 1            # Better optimization
strip = true                 # Strip symbols
opt-level = 3                # Maximum optimization
```

#### Market Data Bindings

```rust
// src/market_data.rs
use napi::bindgen_prelude::*;
use napi_derive::napi;
use nt_core::{Symbol, Quote, MarketDataProvider};

#[napi(object)]
pub struct JsQuote {
    pub symbol: String,
    pub timestamp: i64,
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

#[napi]
pub struct MarketDataStream {
    inner: Arc<Mutex<Box<dyn MarketDataProvider>>>,
}

#[napi]
impl MarketDataStream {
    #[napi(constructor)]
    pub fn new(api_key: String, secret_key: String) -> Result<Self> {
        let provider = create_provider(api_key, secret_key)?;
        Ok(Self {
            inner: Arc::new(Mutex::new(provider)),
        })
    }

    #[napi]
    pub async fn subscribe_quotes(
        &self,
        symbols: Vec<String>,
        callback: JsFunction,
    ) -> Result<()> {
        let symbols: Vec<Symbol> = symbols.into_iter()
            .map(Symbol::from)
            .collect();

        let provider = self.inner.lock().await;
        let mut quote_rx = provider.subscribe_quotes(&symbols).await
            .map_err(|e| Error::from_reason(e.to_string()))?;

        // Spawn background task to forward quotes to JS
        let tsfn: ThreadsafeFunction<JsQuote, ErrorStrategy::Fatal> =
            callback.create_threadsafe_function(0, |ctx| {
                Ok(vec![ctx.value])
            })?;

        tokio::spawn(async move {
            while let Some(quote) = quote_rx.recv().await {
                let js_quote = JsQuote::from(quote);
                tsfn.call(js_quote, ThreadsafeFunctionCallMode::NonBlocking);
            }
        });

        Ok(())
    }

    #[napi]
    pub async fn get_historical_bars(
        &self,
        symbol: String,
        start: i64,
        end: i64,
        timeframe: String,
    ) -> Result<Vec<JsBar>> {
        // Implementation...
        todo!()
    }
}
```

#### Zero-Copy Buffer Transfer

```rust
// src/lib.rs
use napi::bindgen_prelude::*;

#[napi]
pub fn process_market_data_batch(buffer: Buffer) -> Result<Buffer> {
    // Zero-copy read from Node.js buffer
    let input_data: &[u8] = buffer.as_ref();

    // Parse with zero-copy (using Cap'n Proto or FlatBuffers)
    let events = parse_events(input_data)
        .map_err(|e| Error::from_reason(e.to_string()))?;

    // Process events
    let results = process_events(events)?;

    // Serialize back to buffer
    let output_data = serialize_results(&results)?;

    // Return as Node.js Buffer (ownership transferred)
    Ok(Buffer::from(output_data))
}

#[napi]
pub fn create_dataframe_from_buffer(buffer: Buffer) -> Result<External<DataFrame>> {
    // Parse Arrow IPC format
    let cursor = std::io::Cursor::new(buffer.as_ref());
    let df = polars::io::ipc::IpcReader::new(cursor)
        .finish()
        .map_err(|e| Error::from_reason(e.to_string()))?;

    // Return as opaque handle to avoid serialization
    Ok(External::new(df))
}

#[napi]
pub fn calculate_features(df_handle: External<DataFrame>) -> Result<External<DataFrame>> {
    let df = &*df_handle;

    // Feature calculation (all in Rust, no JS boundary crossing)
    let features_df = calculate_technical_features(df)
        .map_err(|e| Error::from_reason(e.to_string()))?;

    Ok(External::new(features_df))
}
```

#### TypeScript Usage

```typescript
// Generated automatically by napi-rs
import { MarketDataStream, JsQuote } from 'neural-trader';

const stream = new MarketDataStream(apiKey, secretKey);

// Async/await support
const bars = await stream.getHistoricalBars(
    'AAPL',
    Date.now() - 86400000,
    Date.now(),
    '5m'
);

// Callback support
stream.subscribeQuotes(['AAPL', 'GOOGL'], (quote: JsQuote) => {
    console.log(`${quote.symbol}: ${quote.bid} x ${quote.ask}`);
});

// Zero-copy buffer operations
const inputBuffer = Buffer.from(marketDataBytes);
const outputBuffer = processMarketDataBatch(inputBuffer);
```

### Build Process

```json
// package.json
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
    "test": "node --test",
    "artifacts": "napi artifacts"
  },
  "devDependencies": {
    "@napi-rs/cli": "^2.18.0"
  }
}
```

Build commands:
```bash
# Development build
npm run build:debug

# Release build for current platform
npm run build

# Cross-compile for all targets
npm run build -- --target x86_64-apple-darwin
npm run build -- --target aarch64-apple-darwin
npm run build -- --target x86_64-pc-windows-msvc
npm run build -- --target x86_64-unknown-linux-gnu
```

---

## Tier 2: WASI/WebAssembly (Fallback 1)

**Best for:** Environments without native module support, browser compatibility

### Configuration

```toml
# Cargo.toml
[lib]
crate-type = ["cdylib"]

[dependencies]
wasm-bindgen = "0.2"
wasm-bindgen-futures = "0.4"
js-sys = "0.3"
serde-wasm-bindgen = "0.6"

[target.'cfg(target_arch = "wasm32")'.dependencies]
getrandom = { version = "0.2", features = ["js"] }

[profile.release]
opt-level = "z"     # Optimize for size
lto = true
```

### Implementation

```rust
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct WasmMarketData {
    provider: Box<dyn MarketDataProvider>,
}

#[wasm_bindgen]
impl WasmMarketData {
    #[wasm_bindgen(constructor)]
    pub fn new(api_key: String, secret_key: String) -> Result<WasmMarketData, JsValue> {
        let provider = create_provider(api_key, secret_key)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(Self { provider })
    }

    #[wasm_bindgen]
    pub async fn get_quote(&self, symbol: String) -> Result<JsValue, JsValue> {
        let quote = self.provider.get_quote(&Symbol::from(symbol))
            .await
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        // Serialize to JS object
        serde_wasm_bindgen::to_value(&quote)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}
```

### TypeScript Usage

```typescript
import init, { WasmMarketData } from './neural_trader_wasm';

// Initialize WASM module
await init();

const marketData = new WasmMarketData(apiKey, secretKey);
const quote = await marketData.getQuote('AAPL');
console.log(quote);
```

### Build Process

```bash
# Install wasm-pack
cargo install wasm-pack

# Build for Node.js
wasm-pack build --target nodejs --out-dir npm/wasm-nodejs

# Build for bundlers (Webpack, etc.)
wasm-pack build --target bundler --out-dir npm/wasm-bundler

# Build for browsers
wasm-pack build --target web --out-dir npm/wasm-web
```

---

## Tier 3: CLI + IPC (Fallback 2)

**Best for:** Simple integration, no build requirements, cross-platform

### CLI Binary

```rust
// src/bin/nt-cli.rs
use clap::{Parser, Subcommand};
use serde_json::json;

#[derive(Parser)]
#[command(name = "nt-cli")]
#[command(about = "Neural Trader CLI")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Get market quote
    GetQuote {
        #[arg(long)]
        symbol: String,
    },
    /// Subscribe to market data stream
    Subscribe {
        #[arg(long)]
        symbols: Vec<String>,
    },
    /// Calculate features
    CalculateFeatures {
        #[arg(long)]
        input: String,
        #[arg(long)]
        output: String,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::GetQuote { symbol } => {
            let quote = get_quote(&symbol).await?;
            println!("{}", serde_json::to_string(&quote)?);
        }
        Commands::Subscribe { symbols } => {
            subscribe_quotes(symbols).await?;
        }
        Commands::CalculateFeatures { input, output } => {
            calculate_features(&input, &output).await?;
        }
    }

    Ok(())
}

async fn subscribe_quotes(symbols: Vec<String>) -> anyhow::Result<()> {
    let provider = create_provider_from_env()?;
    let mut rx = provider.subscribe_quotes(&parse_symbols(&symbols)).await?;

    while let Some(quote) = rx.recv().await {
        // Output JSON to stdout (one per line)
        println!("{}", serde_json::to_string(&quote)?);
    }

    Ok(())
}
```

### Node.js Wrapper

```typescript
// lib/cli-client.ts
import { spawn, ChildProcess } from 'child_process';
import { EventEmitter } from 'events';
import { createInterface } from 'readline';

export class CliClient extends EventEmitter {
    private process: ChildProcess | null = null;

    async getQuote(symbol: string): Promise<Quote> {
        const result = await this.execCommand([
            'get-quote',
            '--symbol', symbol
        ]);

        return JSON.parse(result);
    }

    subscribeQuotes(symbols: string[], callback: (quote: Quote) => void): void {
        this.process = spawn('nt-cli', [
            'subscribe',
            '--symbols', ...symbols
        ]);

        const rl = createInterface({
            input: this.process.stdout!,
            crlfDelay: Infinity
        });

        rl.on('line', (line) => {
            try {
                const quote = JSON.parse(line);
                callback(quote);
            } catch (err) {
                this.emit('error', err);
            }
        });

        this.process.on('error', (err) => {
            this.emit('error', err);
        });
    }

    private async execCommand(args: string[]): Promise<string> {
        return new Promise((resolve, reject) => {
            const process = spawn('nt-cli', args);
            let stdout = '';
            let stderr = '';

            process.stdout.on('data', (data) => {
                stdout += data.toString();
            });

            process.stderr.on('data', (data) => {
                stderr += data.toString();
            });

            process.on('close', (code) => {
                if (code === 0) {
                    resolve(stdout);
                } else {
                    reject(new Error(`Command failed: ${stderr}`));
                }
            });
        });
    }

    close(): void {
        if (this.process) {
            this.process.kill();
            this.process = null;
        }
    }
}
```

---

## Tier 4: gRPC Service (Fallback 3)

**Best for:** Language-agnostic, microservices architecture

### Protocol Definition

```protobuf
// proto/neural_trader.proto
syntax = "proto3";

package neural_trader;

service MarketDataService {
    rpc GetQuote(GetQuoteRequest) returns (Quote);
    rpc SubscribeQuotes(SubscribeRequest) returns (stream Quote);
    rpc GetHistoricalBars(HistoricalRequest) returns (BarResponse);
}

message GetQuoteRequest {
    string symbol = 1;
}

message Quote {
    string symbol = 1;
    int64 timestamp = 2;
    double bid = 3;
    double ask = 4;
    double bid_size = 5;
    double ask_size = 6;
}

message SubscribeRequest {
    repeated string symbols = 1;
}

message Bar {
    string symbol = 1;
    int64 timestamp = 2;
    double open = 3;
    double high = 4;
    double low = 5;
    double close = 6;
    double volume = 7;
}

message HistoricalRequest {
    string symbol = 1;
    int64 start_time = 2;
    int64 end_time = 3;
    string timeframe = 4;
}

message BarResponse {
    repeated Bar bars = 1;
}
```

### Rust Implementation

```rust
// src/grpc/server.rs
use tonic::{transport::Server, Request, Response, Status};

pub struct MarketDataServer {
    provider: Arc<Box<dyn MarketDataProvider>>,
}

#[tonic::async_trait]
impl market_data_service_server::MarketDataService for MarketDataServer {
    async fn get_quote(
        &self,
        request: Request<GetQuoteRequest>,
    ) -> Result<Response<Quote>, Status> {
        let symbol = Symbol::from(request.into_inner().symbol);

        let quote = self.provider.get_quote(&symbol)
            .await
            .map_err(|e| Status::internal(e.to_string()))?;

        Ok(Response::new(quote.into()))
    }

    type SubscribeQuotesStream = ReceiverStream<Result<Quote, Status>>;

    async fn subscribe_quotes(
        &self,
        request: Request<SubscribeRequest>,
    ) -> Result<Response<Self::SubscribeQuotesStream>, Status> {
        let symbols: Vec<Symbol> = request.into_inner().symbols
            .into_iter()
            .map(Symbol::from)
            .collect();

        let mut quote_rx = self.provider.subscribe_quotes(&symbols)
            .await
            .map_err(|e| Status::internal(e.to_string()))?;

        let (tx, rx) = mpsc::channel(100);

        tokio::spawn(async move {
            while let Some(quote) = quote_rx.recv().await {
                if tx.send(Ok(quote.into())).await.is_err() {
                    break;
                }
            }
        });

        Ok(Response::new(ReceiverStream::new(rx)))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let addr = "[::1]:50051".parse()?;
    let server = MarketDataServer {
        provider: Arc::new(create_provider_from_env()?),
    };

    println!("gRPC server listening on {}", addr);

    Server::builder()
        .add_service(market_data_service_server::MarketDataServiceServer::new(server))
        .serve(addr)
        .await?;

    Ok(())
}
```

### Node.js Client

```typescript
// lib/grpc-client.ts
import * as grpc from '@grpc/grpc-js';
import * as protoLoader from '@grpc/proto-loader';
import { EventEmitter } from 'events';

const packageDefinition = protoLoader.loadSync('proto/neural_trader.proto');
const proto = grpc.loadPackageDefinition(packageDefinition) as any;

export class GrpcClient extends EventEmitter {
    private client: any;

    constructor(address: string = 'localhost:50051') {
        super();
        this.client = new proto.neural_trader.MarketDataService(
            address,
            grpc.credentials.createInsecure()
        );
    }

    async getQuote(symbol: string): Promise<Quote> {
        return new Promise((resolve, reject) => {
            this.client.GetQuote({ symbol }, (err: any, response: any) => {
                if (err) {
                    reject(err);
                } else {
                    resolve(response);
                }
            });
        });
    }

    subscribeQuotes(symbols: string[], callback: (quote: Quote) => void): void {
        const call = this.client.SubscribeQuotes({ symbols });

        call.on('data', (quote: Quote) => {
            callback(quote);
        });

        call.on('error', (err: Error) => {
            this.emit('error', err);
        });

        call.on('end', () => {
            this.emit('end');
        });
    }

    close(): void {
        this.client.close();
    }
}
```

---

## Unified API Layer

Provide consistent API across all tiers:

```typescript
// lib/index.ts
export interface NeuralTraderClient {
    getQuote(symbol: string): Promise<Quote>;
    subscribeQuotes(symbols: string[], callback: (quote: Quote) => void): void;
    close(): void;
}

export async function createClient(options?: ClientOptions): Promise<NeuralTraderClient> {
    const preferredTier = options?.tier || detectBestTier();

    switch (preferredTier) {
        case 'napi':
            return new NapiClient(options);
        case 'wasm':
            return new WasmClient(options);
        case 'cli':
            return new CliClient(options);
        case 'grpc':
            return new GrpcClient(options);
        default:
            throw new Error(`Unknown tier: ${preferredTier}`);
    }
}

function detectBestTier(): string {
    // Try to load napi module
    try {
        require('./lib/bindings.node');
        return 'napi';
    } catch {}

    // Check for WASM support
    if (typeof WebAssembly !== 'undefined') {
        return 'wasm';
    }

    // Check for CLI binary
    if (commandExists('nt-cli')) {
        return 'cli';
    }

    // Default to gRPC (requires separate service)
    return 'grpc';
}
```

---

## Performance Comparison

| Tier | Latency | Throughput | Build Time | Distribution |
|------|---------|------------|------------|--------------|
| napi-rs | <0.1ms | 1M ops/sec | 2-5 min | Binary per platform |
| WASM | <1ms | 100K ops/sec | 1-2 min | Universal (~2MB) |
| CLI+IPC | 5-10ms | 1K ops/sec | 2-5 min | Binary per platform |
| gRPC | 10-50ms | 10K ops/sec | 2-5 min | Separate service |

---

**Next:** [06-performance-concurrency.md](./06-performance-concurrency.md)
