# Data Flow Diagrams

## Complete Trading Pipeline Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Market Data Sources                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐          │
│  │ Alpaca   │  │ Polygon  │  │ Coinbase │  │  News    │          │
│  │ WebSocket│  │   API    │  │ WebSocket│  │   API    │          │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘          │
└───────┼─────────────┼─────────────┼─────────────┼─────────────────┘
        │             │             │             │
        └─────────────┴─────────────┴─────────────┘
                      ▼
        ┌─────────────────────────────┐
        │   Market Data Multiplexer   │◀───── Raw events
        │   (Tokio async streams)     │
        └──────────────┬──────────────┘
                       │ Normalized events
                       ▼
        ┌─────────────────────────────┐
        │    Data Ingestion Layer     │
        │  ┌────────────────────────┐ │
        │  │  Rate Limiting         │ │
        │  │  Deduplication         │ │
        │  │  Validation            │ │
        │  └────────────┬───────────┘ │
        └───────────────┼─────────────┘
                        │ Validated events
                        ▼
        ┌─────────────────────────────┐
        │  Polars DataFrame Builder   │
        │  (Batch accumulation)       │
        └──────────────┬──────────────┘
                       │ DataFrames (1000 rows)
                       ▼
        ┌─────────────────────────────┐
        │    Event Bus (Midstreamer)  │◀───── Pub/Sub
        │  Topic: market.data.*       │
        └──────┬─────────────┬────────┘
               │             │
     ┌─────────┴────┐   ┌────┴──────────┐
     ▼              ▼   ▼               ▼
┌─────────┐   ┌─────────────┐   ┌─────────────┐
│ AgentDB │   │  Feature    │   │  Strategy   │
│ Storage │   │  Extraction │   │  Engines    │
└─────────┘   └──────┬──────┘   └──────┬──────┘
                     │ Features        │ Raw data
                     ▼                 │
              ┌─────────────┐          │
              │  Feature    │          │
              │  Store      │◀─────────┘
              │  (Cache)    │
              └──────┬──────┘
                     │ Enriched features
                     ▼
              ┌─────────────┐
              │   Signal    │
              │  Generation │
              │  (Sublinear)│
              └──────┬──────┘
                     │ Trading signals
                     ▼
              ┌─────────────┐
              │    Risk     │
              │  Management │
              └──────┬──────┘
                     │ Approved orders
                     ▼
              ┌─────────────┐
              │   Order     │
              │  Execution  │
              └──────┬──────┘
                     │ Filled orders
                     ▼
        ┌─────────────────────────────┐
        │       Settlement            │
        │  ┌────────────────────────┐ │
        │  │  Position Updates      │ │
        │  │  PnL Calculation       │ │
        │  │  Cost Attribution      │ │
        │  └────────────────────────┘ │
        └─────────────────────────────┘
                     │
                     ▼
        ┌─────────────────────────────┐
        │   Agentic Payments Tracker  │
        └─────────────────────────────┘
```

## Detailed Component Data Flows

### 1. Market Data Ingestion Flow

```
WebSocket Connection
       │
       ▼
┌─────────────────┐
│  Raw Message    │  Format: JSON/Protobuf
│  {"type": ...}  │
└────────┬────────┘
         │ deserialize
         ▼
┌─────────────────┐
│  Typed Event    │  Rust struct
│  Quote { ... }  │
└────────┬────────┘
         │ validate
         ▼
┌─────────────────┐     Yes    ┌─────────────┐
│  Validation     │───────────▶│  Accept     │
│  Checks         │            └──────┬──────┘
└────────┬────────┘                   │
         │ No                         │
         ▼                            │
┌─────────────────┐                  │
│  Error Log &    │                  │
│  Metrics        │                  │
└─────────────────┘                  │
                                     ▼
                          ┌─────────────────┐
                          │  Batch Buffer   │
                          │  [Quote; 1000]  │
                          └────────┬────────┘
                                   │ full?
                                   ▼
                          ┌─────────────────┐
                          │  Convert to     │
                          │  DataFrame      │
                          └────────┬────────┘
                                   │
                                   ▼
                          ┌─────────────────┐
                          │  Publish to     │
                          │  Event Bus      │
                          └─────────────────┘
```

### 2. Feature Extraction Pipeline

```
DataFrame Input
       │
       ▼
┌──────────────────┐
│  Group by Symbol │  Polars: group_by("symbol")
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Sort by Time    │  Polars: sort("timestamp")
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Rolling Window  │  window_size = 20
│  Operations      │
└────────┬─────────┘
         │
    ┌────┴────────┬────────┬─────────┐
    ▼             ▼        ▼         ▼
┌────────┐  ┌────────┐ ┌────────┐ ┌────────┐
│  SMA   │  │  EMA   │ │  RSI   │ │  MACD  │
└───┬────┘  └───┬────┘ └───┬────┘ └───┬────┘
    │           │          │          │
    └───────────┴──────────┴──────────┘
                │ Horizontal concatenation
                ▼
    ┌─────────────────────┐
    │  Feature DataFrame  │
    │  [symbol, time,     │
    │   price, sma_20,    │
    │   ema_12, rsi_14,   │
    │   macd, ...]        │
    └──────────┬──────────┘
               │
               ▼
    ┌─────────────────────┐
    │  Feature Store      │
    │  (TTL: 1 hour)      │
    └──────────┬──────────┘
               │
               ▼
    ┌─────────────────────┐
    │  Signal Generators  │
    └─────────────────────┘
```

### 3. Signal Generation with Sublinear Optimization

```
Feature Set
       │
       ▼
┌──────────────────────┐
│  Feature Vector      │  Shape: [batch_size, num_features]
│  f32 array           │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Sketching Algorithm │  HyperLogLog / Count-Min Sketch
│  (Sublinear)         │  O(log n) space complexity
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Pattern Matching    │  Locality-Sensitive Hashing
│  (AgentDB query)     │  Find similar historical patterns
└──────────┬───────────┘
           │ Top-K matches
           ▼
┌──────────────────────┐
│  Neural Model        │  ONNX Runtime inference
│  Inference           │  Input: features + context
└──────────┬───────────┘
           │ Raw predictions
           ▼
┌──────────────────────┐
│  Online Convex       │  Hedge algorithm
│  Optimization        │  Combine multiple models
└──────────┬───────────┘
           │ Weighted signal
           ▼
┌──────────────────────┐
│  Signal Struct       │
│  {                   │
│    symbol,           │
│    side,             │
│    strength,         │
│    confidence,       │
│    timestamp         │
│  }                   │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Risk Manager        │
└──────────────────────┘
```

### 4. Order Execution Flow

```
Signal
   │
   ▼
┌─────────────────┐
│  Risk Check     │
│  ┌───────────┐  │
│  │ Position  │  │
│  │ Limits    │  │
│  └───────────┘  │
│  ┌───────────┐  │
│  │ Portfolio │  │
│  │ VaR       │  │
│  └───────────┘  │
│  ┌───────────┐  │
│  │ Circuit   │  │
│  │ Breaker   │  │
│  └───────────┘  │
└────────┬────────┘
         │ Approved
         ▼
┌─────────────────┐
│  Order Builder  │
│  ┌───────────┐  │
│  │ Size      │  │  Kelly Criterion
│  │ Calc      │  │
│  └───────────┘  │
│  ┌───────────┐  │
│  │ Price     │  │  Limit price = mid + spread/2
│  │ Calc      │  │
│  └───────────┘  │
│  ┌───────────┐  │
│  │ Time In   │  │  GTC / Day / IOC
│  │ Force     │  │
│  └───────────┘  │
└────────┬────────┘
         │ Order struct
         ▼
┌─────────────────┐
│  Exchange       │
│  Router         │
│  ┌───────────┐  │
│  │ Alpaca    │  │
│  ├───────────┤  │
│  │ Coinbase  │  │
│  ├───────────┤  │
│  │ Binance   │  │
│  └───────────┘  │
└────────┬────────┘
         │ REST API / WebSocket
         ▼
┌─────────────────┐
│  Exchange       │
│  (External)     │
└────────┬────────┘
         │ Order confirmation
         ▼
┌─────────────────┐
│  Order Tracker  │
│  HashMap<       │
│    OrderId,     │
│    OrderStatus  │
│  >              │
└────────┬────────┘
         │ Updates
         ▼
┌─────────────────┐
│  Event Bus      │  Topic: orders.filled
└────────┬────────┘
         │
    ┌────┴─────┐
    ▼          ▼
┌────────┐  ┌────────┐
│Strategy│  │ Settle-│
│Notif.  │  │ ment   │
└────────┘  └────────┘
```

### 5. AgentDB Memory Integration

```
Strategy Event
       │
       ▼
┌──────────────────┐
│  Feature Vector  │  f32[512]
│  Embedding       │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  AgentDB Client  │
│  (Async)         │
└────────┬─────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐  ┌────────┐
│ Store  │  │ Query  │
└───┬────┘  └───┬────┘
    │           │
    ▼           ▼
┌──────────────────┐
│  gRPC Connection │
│  (Connection     │
│   Pool)          │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  AgentDB Server  │
│  ┌────────────┐  │
│  │ HNSW Index │  │  k-NN search
│  └────────────┘  │
│  ┌────────────┐  │
│  │ RocksDB    │  │  Key-value store
│  └────────────┘  │
└────────┬─────────┘
         │ Results
         ▼
┌──────────────────┐
│  Strategy        │
│  Context         │
│  ┌────────────┐  │
│  │ Historical │  │
│  │ Patterns   │  │
│  └────────────┘  │
│  ┌────────────┐  │
│  │ Similar    │  │
│  │ Situations │  │
│  └────────────┘  │
└──────────────────┘
```

## Message Formats

### Market Data Event

```rust
// Binary format using Cap'n Proto or FlatBuffers
struct MarketDataEvent {
    event_type: EventType,      // 1 byte
    symbol_id: u32,              // 4 bytes (intern strings)
    timestamp_ns: i64,           // 8 bytes
    price: f64,                  // 8 bytes
    size: f64,                   // 8 bytes
    side: Side,                  // 1 byte
    exchange_id: u16,            // 2 bytes
}
// Total: 32 bytes per event

// Zero-copy deserialization
impl MarketDataEvent {
    fn from_bytes(bytes: &[u8]) -> &Self {
        unsafe { &*(bytes.as_ptr() as *const Self) }
    }
}
```

### Signal Message

```rust
#[derive(Serialize, Deserialize)]
struct SignalMessage {
    signal_id: Uuid,
    strategy_id: String,
    symbol: Symbol,
    side: Side,
    strength: f64,           // 0.0 to 1.0
    confidence: f64,         // 0.0 to 1.0
    features: Vec<f32>,      // Feature vector
    metadata: HashMap<String, String>,
    timestamp: DateTime<Utc>,
}

// JSON format for event bus
// MessagePack for storage
```

### Order Message

```rust
#[derive(Serialize, Deserialize)]
struct OrderMessage {
    order_id: OrderId,
    client_order_id: String,
    symbol: Symbol,
    side: Side,
    order_type: OrderType,
    quantity: Decimal,
    price: Option<Decimal>,
    time_in_force: TimeInForce,
    strategy_id: String,
    signal_id: Option<Uuid>,
    timestamp: DateTime<Utc>,
}
```

## Data Storage Patterns

### Time-Series Data (Market Data)

```
Storage: Polars Parquet files
Schema:
  - symbol: String (dictionary encoded)
  - timestamp: i64 (nanoseconds)
  - price: f64
  - volume: f64
  - Features: f64 (columns)

Partitioning: By date and symbol
  /data/market/2025-11-12/AAPL.parquet
  /data/market/2025-11-12/GOOGL.parquet

Compression: Snappy (fast) or Zstd (better ratio)
Retention: 30 days hot, 1 year cold storage
```

### Vector Embeddings (AgentDB)

```
Storage: AgentDB vector database
Format: f32 arrays
Dimension: 512 (configurable)
Index: HNSW (M=16, ef_construction=200)
Distance: Cosine similarity

Key format: "strategy:{id}:pattern:{timestamp}"
TTL: 90 days
```

### Orders and Fills

```
Storage: PostgreSQL / SQLite
Schema:
  orders:
    - id (UUID primary key)
    - strategy_id (indexed)
    - symbol (indexed)
    - status (indexed)
    - created_at (indexed)
    - ... other fields

  fills:
    - id (UUID primary key)
    - order_id (foreign key)
    - price
    - quantity
    - timestamp
```

## Performance Characteristics

### Data Flow Latency Budget

```
Market Event → Ingestion:        <1ms  (p99)
Ingestion → DataFrame:           <5ms  (batching delay)
DataFrame → Features:            <10ms (p95)
Features → Signal:               <50ms (p95)
Signal → Risk Check:             <10ms (p99)
Risk Check → Order Submit:       <30ms (p95)
Order Submit → Exchange ACK:     <100ms (network)
─────────────────────────────────────────────
Total (data to order):           <200ms (p95)
```

### Throughput Targets

```
Market events ingested:   100,000/sec
DataFrames processed:     1,000/sec
Features calculated:      10,000/sec
Signals generated:        1,000/sec
Orders submitted:         100/sec
```

### Memory Usage

```
Market data buffer:       100MB  (1M events)
Feature cache:            500MB  (hot features)
AgentDB connection pool:  50MB   (10 connections)
Strategy instances:       100MB  (20 strategies @ 5MB each)
Order book cache:         200MB  (100 symbols)
─────────────────────────────────────────
Total baseline:           ~1GB
Peak (under load):        ~3GB
```

## Data Quality and Consistency

### Validation Rules

1. **Timestamp Monotonicity:** Events must have increasing timestamps
2. **Price Sanity:** Price changes >10% in 1 second flagged
3. **Volume Outliers:** Volume >3 std dev from mean flagged
4. **Missing Data:** Gaps >5 seconds trigger alerts
5. **Duplicate Detection:** Hash-based deduplication window

### Consistency Guarantees

- **At-least-once delivery:** Event bus guarantees
- **Idempotent processing:** Duplicate signals ignored
- **Eventual consistency:** AgentDB replication lag <1s
- **Strong consistency:** Order state in PostgreSQL

---

**Next:** [04-crate-recommendations.md](./04-crate-recommendations.md)
