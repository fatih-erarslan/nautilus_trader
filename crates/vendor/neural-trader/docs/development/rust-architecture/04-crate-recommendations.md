# Rust Crate Recommendations

## Core Dependencies

### Async Runtime

#### tokio (v1.35+)
**Purpose:** Async runtime and I/O

**Features:**
```toml
[dependencies]
tokio = { version = "1.35", features = [
    "rt-multi-thread",  # Multi-threaded work-stealing scheduler
    "macros",           # #[tokio::main] and #[tokio::test]
    "sync",             # Channels, mutexes, semaphores
    "time",             # Sleep, interval, timeout
    "net",              # TCP/UDP networking
    "io-util",          # AsyncRead/AsyncWrite utilities
    "signal",           # Unix signal handling
] }
```

**Justification:**
- Industry standard for async Rust
- Excellent performance (work-stealing scheduler)
- Rich ecosystem of compatible libraries
- Built-in tracing integration
- Mature and well-maintained

**Alternatives Considered:**
- `async-std`: Similar API, smaller ecosystem
- `smol`: Lightweight, but less feature-complete
- Custom runtime: Unnecessary complexity

---

### Data Processing

#### polars (v0.36+)
**Purpose:** DataFrame operations and analytics

**Features:**
```toml
[dependencies]
polars = { version = "0.36", features = [
    "lazy",              # Lazy evaluation and query optimization
    "temporal",          # DateTime operations
    "rolling_window",    # Time-series rolling operations
    "rank",              # Ranking functions
    "round_series",      # Rounding operations
    "is_in",             # Membership testing
    "parquet",           # Parquet I/O
    "json",              # JSON I/O
    "performant",        # Performance optimizations
    "dtype-date",        # Date types
    "dtype-datetime",    # DateTime types
    "dtype-time",        # Time types
    "dtype-duration",    # Duration types
] }
```

**Justification:**
- Native Rust implementation (no Python overhead)
- Apache Arrow memory format (zero-copy interop)
- Lazy evaluation with query optimization
- SIMD acceleration built-in
- 5-10x faster than Pandas in benchmarks
- Memory-efficient chunked arrays

**Alternatives:**
- `ndarray`: Lower-level, no DataFrame abstractions
- `datafusion`: SQL-focused, heavier weight
- `arrow`: Lower-level Arrow primitives

**Usage Example:**
```rust
use polars::prelude::*;

fn calculate_features(df: &DataFrame) -> PolarsResult<DataFrame> {
    df.lazy()
        .group_by([col("symbol")])
        .agg([
            col("close").mean().alias("price_mean"),
            col("close").std(1).alias("price_std"),
            col("volume").sum().alias("volume_sum"),
        ])
        .collect()
}
```

---

### Node.js Interoperability

#### napi-rs (v2.16+)
**Purpose:** Node.js native module bindings

**Features:**
```toml
[dependencies]
napi = { version = "2.16", features = ["async", "tokio_rt"] }
napi-derive = "2.16"
```

**Build Dependencies:**
```toml
[build-dependencies]
napi-build = "2.1"
```

**Justification:**
- Type-safe Rust ↔ JavaScript conversions
- Automatic TypeScript definition generation
- N-API stable ABI (no recompilation across Node versions)
- Excellent performance (near-native)
- Zero-copy buffer transfers
- Async support with Tokio integration

**Example:**
```rust
use napi::bindgen_prelude::*;
use napi_derive::napi;

#[napi(object)]
pub struct MarketData {
    pub symbol: String,
    pub price: f64,
    pub volume: f64,
}

#[napi]
pub async fn fetch_market_data(symbol: String) -> Result<MarketData> {
    // Rust async implementation
    Ok(MarketData {
        symbol,
        price: 150.0,
        volume: 1000000.0,
    })
}
```

**Generated TypeScript:**
```typescript
export interface MarketData {
    symbol: string;
    price: number;
    volume: number;
}

export function fetchMarketData(symbol: string): Promise<MarketData>;
```

---

### Serialization

#### serde (v1.0+)
**Purpose:** Serialization/deserialization framework

```toml
[dependencies]
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"      # JSON
rmp-serde = "1.1"       # MessagePack (compact binary)
bincode = "1.3"         # Binary format
toml = "0.8"            # Configuration files
```

**Justification:**
- Universal serialization framework
- Zero-copy deserialization where possible
- Code generation via derive macros
- Extensive format support

---

### HTTP and WebSockets

#### reqwest (v0.11+)
**Purpose:** HTTP client

```toml
[dependencies]
reqwest = { version = "0.11", features = [
    "json",
    "stream",
    "rustls-tls",  # Use rustls instead of OpenSSL
] }
```

**Justification:**
- Async-first design
- Connection pooling
- Automatic retries
- TLS support

#### tokio-tungstenite (v0.21+)
**Purpose:** WebSocket client/server

```toml
[dependencies]
tokio-tungstenite = { version = "0.21", features = ["rustls-tls-native-roots"] }
```

**Justification:**
- Tokio-compatible
- Low overhead
- RFC 6455 compliant

---

### Decimal Precision

#### rust_decimal (v1.33+)
**Purpose:** Fixed-precision decimal arithmetic

```toml
[dependencies]
rust_decimal = { version = "1.33", features = ["serde-float"] }
rust_decimal_macros = "1.33"
```

**Justification:**
- No floating-point errors
- Essential for financial calculations
- Exact decimal representation
- Serde integration

**Example:**
```rust
use rust_decimal::Decimal;
use rust_decimal_macros::dec;

let price = dec!(123.45);
let quantity = dec!(100);
let total = price * quantity;  // Exact: 12345.00
```

---

### Date and Time

#### chrono (v0.4+)
**Purpose:** Date and time handling

```toml
[dependencies]
chrono = { version = "0.4", features = ["serde"] }
```

**Justification:**
- Comprehensive timezone support
- Parsing and formatting
- Duration arithmetic
- Integration with Polars

**Alternative:** `time` crate (simpler, no timezone support)

---

### Error Handling

#### thiserror (v1.0+)
**Purpose:** Derive Error trait

```toml
[dependencies]
thiserror = "1.0"
```

**Example:**
```rust
use thiserror::Error;

#[derive(Error, Debug)]
pub enum TradingError {
    #[error("Market data error: {0}")]
    MarketData(String),

    #[error("Execution failed: {0}")]
    Execution(String),

    #[error("Risk limit exceeded: {reason}")]
    RiskLimit { reason: String },

    #[error(transparent)]
    Io(#[from] std::io::Error),
}
```

#### anyhow (v1.0+)
**Purpose:** Application-level error handling

```toml
[dependencies]
anyhow = "1.0"
```

**Usage:** Use `anyhow::Result<T>` for application code, `thiserror` for library errors.

---

### Logging and Tracing

#### tracing (v0.1+)
**Purpose:** Structured logging and distributed tracing

```toml
[dependencies]
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = [
    "env-filter",
    "json",
    "fmt",
] }
tracing-appender = "0.2"
```

**Justification:**
- Async-aware (no blocking)
- Structured fields
- Span-based tracing
- Tokio integration
- OpenTelemetry export

**Example:**
```rust
use tracing::{info, instrument, span, Level};

#[instrument(skip(data))]
async fn process_signal(signal_id: &str, data: &MarketData) -> Result<()> {
    let span = span!(Level::INFO, "signal_processing");
    let _enter = span.enter();

    info!(
        signal_id = %signal_id,
        symbol = %data.symbol,
        price = %data.price,
        "Processing trading signal"
    );

    // Processing logic...
    Ok(())
}
```

---

### Configuration Management

#### config (v0.14+)
**Purpose:** Hierarchical configuration

```toml
[dependencies]
config = { version = "0.14", features = ["toml", "json", "yaml"] }
```

**Justification:**
- Multiple format support
- Environment variable override
- Hierarchical merging
- Type-safe deserialization

**Example:**
```rust
use config::{Config, File, Environment};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct Settings {
    api_key: String,
    max_position_size: Decimal,
    risk_level: String,
}

fn load_config() -> Result<Settings> {
    let settings = Config::builder()
        .add_source(File::with_name("config/default"))
        .add_source(File::with_name("config/production").required(false))
        .add_source(Environment::with_prefix("NT"))
        .build()?;

    settings.try_deserialize()
}
```

---

### Database and Storage

#### sqlx (v0.7+)
**Purpose:** Async SQL database access

```toml
[dependencies]
sqlx = { version = "0.7", features = [
    "runtime-tokio-rustls",
    "postgres",
    "sqlite",
    "macros",
    "migrate",
    "chrono",
    "decimal",
] }
```

**Justification:**
- Compile-time checked queries
- Connection pooling
- Async/await support
- Migration management

#### rocksdb (v0.21+)
**Purpose:** Embedded key-value store

```toml
[dependencies]
rocksdb = { version = "0.21", features = ["snappy"] }
```

**Justification:**
- High-performance local storage
- LSM tree design
- Prefix iterators
- Compression support

---

### Concurrency Primitives

#### crossbeam (v0.8+)
**Purpose:** Lock-free data structures

```toml
[dependencies]
crossbeam = "0.8"
```

**Justification:**
- Lock-free channels
- Scoped threads
- Epoch-based memory reclamation
- Excellent performance

#### dashmap (v5.5+)
**Purpose:** Concurrent HashMap

```toml
[dependencies]
dashmap = "5.5"
```

**Justification:**
- Sharded locking for low contention
- No single lock bottleneck
- Drop-in replacement for `RwLock<HashMap>`

**Example:**
```rust
use dashmap::DashMap;

let positions: DashMap<Symbol, Position> = DashMap::new();

// Concurrent writes from multiple threads
positions.insert(symbol, position);

// Concurrent reads
if let Some(pos) = positions.get(&symbol) {
    println!("Position: {:?}", pos);
}
```

---

### Neural Network Integration

#### ort (ONNX Runtime) (v2.0+)
**Purpose:** Neural network inference

```toml
[dependencies]
ort = { version = "2.0", features = ["cuda", "tensorrt"] }
```

**Justification:**
- Run ONNX models from Python training
- CPU and GPU support
- Optimized kernels
- Cross-platform

**Example:**
```rust
use ort::{Environment, SessionBuilder, Value};

let env = Environment::builder().build()?;
let session = SessionBuilder::new(&env)?
    .with_model_from_file("model.onnx")?;

let input_tensor = Array::from_shape_vec((1, 10), input_data)?;
let outputs = session.run(vec![Value::from_array(session.allocator(), &input_tensor)?])?;
```

---

### Metrics and Observability

#### prometheus (v0.13+)
**Purpose:** Metrics collection

```toml
[dependencies]
prometheus = "0.13"
```

**Example:**
```rust
use prometheus::{Counter, Histogram, Registry};

lazy_static! {
    static ref ORDERS_TOTAL: Counter = Counter::new(
        "orders_total",
        "Total number of orders submitted"
    ).unwrap();

    static ref ORDER_LATENCY: Histogram = Histogram::with_opts(
        HistogramOpts::new("order_latency_seconds", "Order submission latency")
            .buckets(vec![0.001, 0.01, 0.1, 0.5, 1.0, 5.0])
    ).unwrap();
}

async fn submit_order(order: Order) -> Result<()> {
    let timer = ORDER_LATENCY.start_timer();

    // Submit order...

    ORDERS_TOTAL.inc();
    timer.observe_duration();
    Ok(())
}
```

---

### Testing

#### mockall (v0.12+)
**Purpose:** Mocking framework

```toml
[dev-dependencies]
mockall = "0.12"
```

**Example:**
```rust
use mockall::mock;

mock! {
    MarketDataProvider {}

    #[async_trait]
    impl MarketDataProvider for MarketDataProvider {
        async fn get_quote(&self, symbol: &Symbol) -> Result<Quote>;
    }
}

#[tokio::test]
async fn test_strategy() {
    let mut mock_provider = MockMarketDataProvider::new();
    mock_provider
        .expect_get_quote()
        .returning(|_| Ok(Quote { /* ... */ }));

    // Test with mock...
}
```

#### criterion (v0.5+)
**Purpose:** Benchmarking

```toml
[dev-dependencies]
criterion = { version = "0.5", features = ["async_tokio"] }
```

**Example:**
```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn benchmark_feature_extraction(c: &mut Criterion) {
    let df = /* setup dataframe */;

    c.bench_function("extract_features", |b| {
        b.iter(|| extract_features(black_box(&df)))
    });
}

criterion_group!(benches, benchmark_feature_extraction);
criterion_main!(benches);
```

---

## Integration Crates

### AgentDB Client

**Custom Implementation:**
```toml
[dependencies]
tonic = "0.11"           # gRPC client
prost = "0.12"           # Protobuf
```

**Rationale:** AgentDB likely has gRPC API. Implement client using Tonic.

### E2B Sandbox Client

**Custom Implementation:**
```toml
[dependencies]
reqwest = "0.11"         # HTTP client
serde = "1.0"            # JSON
```

**Rationale:** E2B has REST API. Use reqwest for HTTP calls.

### Agentic Flow Federation

**Custom Implementation:**
```toml
[dependencies]
tonic = "0.11"           # gRPC
nats = "0.25"            # NATS messaging
```

**Rationale:** Inter-agent communication via gRPC and pub/sub via NATS.

---

## Dependency Management Strategy

### Version Pinning

Use `=` for exact versions in production:
```toml
tokio = "=1.35.1"  # Lock to specific version
```

Use `^` for compatible updates in development:
```toml
tokio = "^1.35"    # Allow 1.35.x updates
```

### Feature Flags

Enable features conditionally:
```toml
[features]
default = ["postgres", "rest-api"]
postgres = ["sqlx/postgres"]
sqlite = ["sqlx/sqlite"]
gpu = ["ort/cuda"]
rest-api = ["axum"]
grpc-api = ["tonic"]
```

Build with features:
```bash
cargo build --release --features "postgres,rest-api,gpu"
```

### Workspace Dependencies

Centralize version management:
```toml
# Workspace root Cargo.toml
[workspace.dependencies]
tokio = { version = "1.35", features = ["full"] }
polars = { version = "0.36", features = ["lazy"] }

# Member crate
[dependencies]
tokio.workspace = true
polars.workspace = true
```

---

## Security Considerations

### Supply Chain Security

1. **Audit dependencies:**
   ```bash
   cargo audit
   ```

2. **Check for unmaintained crates:**
   ```bash
   cargo outdated
   ```

3. **Review security advisories:**
   - RustSec Advisory Database
   - GitHub Security Advisories

### Cryptography

Use audited cryptography crates:
```toml
[dependencies]
ring = "0.17"            # Cryptographic primitives
rustls = "0.22"          # TLS library (audited)
ed25519-dalek = "2.1"    # Digital signatures
```

**Avoid:** Self-rolled cryptography

---

**Next:** [05-nodejs-interop-strategy.md](./05-nodejs-interop-strategy.md)
