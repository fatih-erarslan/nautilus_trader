# Neural Trader v2.1.0 - System Architecture

**High-Performance Trading Platform with Rust NAPI Backend**

---

## Table of Contents

1. [Overview](#overview)
2. [System Components](#system-components)
3. [Architecture Layers](#architecture-layers)
4. [Data Flow](#data-flow)
5. [Technology Stack](#technology-stack)
6. [Design Patterns](#design-patterns)
7. [Performance Characteristics](#performance-characteristics)
8. [Security Architecture](#security-architecture)
9. [Deployment Architecture](#deployment-architecture)
10. [Scalability Design](#scalability-design)

---

## Overview

Neural Trader is a **hybrid TypeScript/Rust** trading platform that combines the flexibility of Node.js with the raw performance of Rust. The architecture follows a **layered design** with clear separation of concerns and optimized data paths for high-frequency operations.

### Key Architectural Goals

1. **Performance:** Sub-millisecond latency for critical operations
2. **Reliability:** 99.99% uptime with graceful degradation
3. **Scalability:** Horizontal scaling to 10K+ concurrent operations
4. **Security:** Defense-in-depth with multiple validation layers
5. **Maintainability:** Modular design with clear interfaces

---

## System Components

```
┌─────────────────────────────────────────────────────────────┐
│                     User Applications                        │
│  (Claude Desktop, CLI, Custom Integrations, Web Clients)    │
└──────────────────────┬──────────────────────────────────────┘
                       │ JSON-RPC 2.0
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              MCP Server (TypeScript)                         │
│  • Protocol Handling    • Request Validation                │
│  • Authentication       • Rate Limiting                      │
│  • Response Formatting  • Error Handling                     │
└──────────────────────┬──────────────────────────────────────┘
                       │ NAPI Bridge
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              NAPI-RS Interface Layer                         │
│  • Type Conversion      • Memory Management                  │
│  • Error Marshalling    • Thread Safety                      │
└──────────────────────┬──────────────────────────────────────┘
                       │ Function Calls
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                 Rust Backend (Core Logic)                    │
│  ┌─────────────┬──────────────┬─────────────┬──────────┐   │
│  │   Trading   │    Neural    │    Risk     │  Sports  │   │
│  │   Engine    │   Networks   │  Analysis   │  Betting │   │
│  └─────────────┴──────────────┴─────────────┴──────────┘   │
│  ┌─────────────┬──────────────┬─────────────┬──────────┐   │
│  │   Market    │     News     │   E2B       │ Syndicates│  │
│  │    Data     │  Sentiment   │   Cloud     │  Manager │   │
│  └─────────────┴──────────────┴─────────────┴──────────┘   │
└──────────────────────┬──────────────────────────────────────┘
                       │ API Calls
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              External Services Integration                   │
│  • Alpaca Markets    • NewsAPI        • The Odds API        │
│  • Finnhub           • E2B Cloud      • Database            │
└─────────────────────────────────────────────────────────────┘
```

---

## Architecture Layers

### Layer 1: Protocol & Interface (TypeScript)

**Responsibilities:**
- MCP protocol implementation (JSON-RPC 2.0)
- Request routing and validation
- Authentication and authorization
- Response formatting and error handling

**Key Files:**
- `/packages/mcp/src/index.ts` - MCP server entry point
- `/packages/mcp/src/tools/*.ts` - Tool definitions
- `/packages/mcp/src/bridge/rust.js` - NAPI bridge

**Design Decisions:**
- TypeScript for type safety and developer experience
- Async/await for all I/O operations
- Promise-based API for consistency

---

### Layer 2: NAPI Bridge (Rust/Node.js)

**Responsibilities:**
- Type conversion between JavaScript and Rust
- Memory management across FFI boundary
- Thread safety guarantees
- Error marshalling

**Key Files:**
- `/neural-trader-rust/crates/napi-bindings/src/lib.rs`
- `/neural-trader-rust/crates/napi-bindings/src/*/mod.rs`

**Technical Constraints:**
- All data must be serializable
- No direct pointer sharing
- Explicit lifetime management
- Thread-safe operations only

**Performance Characteristics:**
- Type conversion overhead: ~10-50 microseconds
- Memory copying for large datasets
- Zero-copy optimization where possible

---

### Layer 3: Core Business Logic (Rust)

**Responsibilities:**
- Trading strategy execution
- Neural network training/inference
- Risk calculations
- Market data processing

**Module Structure:**
```
neural-trader-rust/
├── backend-rs/           # Core backend library
│   ├── trading/          # Trading strategies
│   ├── neural/           # Neural networks
│   ├── risk/             # Risk management
│   └── sports/           # Sports betting
├── napi-bindings/        # NAPI exports
└── data-structures/      # Shared types
```

**Design Patterns:**
- **Builder Pattern:** Complex configuration objects
- **Strategy Pattern:** Pluggable trading strategies
- **Factory Pattern:** Neural model creation
- **Repository Pattern:** Data access abstraction

---

### Layer 4: External Services

**Service Integration:**

```rust
// Market Data Flow
External API → Rate Limiter → Cache → Parser → Type Conversion → Business Logic

// Error Handling Flow
API Error → Categorize → Retry Logic → Fallback Provider → User Error
```

**Rate Limiting:**
- Per-provider token buckets
- Exponential backoff on errors
- Circuit breaker pattern

**Caching:**
- Redis for distributed cache
- In-memory LRU for hot data
- Configurable TTL per data type

---

## Data Flow

### Read Path (Market Data Query)

```
User Request
    ↓
MCP Protocol Handler
    ↓
Request Validator
    ↓
NAPI Bridge (TS → Rust)
    ↓
Check Cache (Redis)
    ↓ (miss)
External API Client
    ↓
Rate Limiter
    ↓
HTTP Request
    ↓
Response Parser
    ↓
Type Conversion
    ↓
Cache Update
    ↓
NAPI Bridge (Rust → TS)
    ↓
Response Formatter
    ↓
User Response
```

**Latency Breakdown:**
- Protocol handling: 1-2ms
- NAPI bridge: 0.05ms
- Cache lookup: 0.1-0.5ms
- External API: 50-200ms
- Total: ~50-205ms

---

### Write Path (Trade Execution)

```
Trade Request
    ↓
Authentication
    ↓
Risk Validation
    ↓
Strategy Evaluation
    ↓
Pre-trade Checks
    ↓
Order Submission
    ↓
Confirmation Wait
    ↓
Portfolio Update
    ↓
Audit Log
    ↓
Response
```

**Critical Sections:**
- Risk validation: Atomic operations
- Order submission: Idempotency keys
- Portfolio update: Transactional

---

### Neural Training Path (GPU Accelerated)

```
Training Request
    ↓
GPU Availability Check
    ↓
Data Loading (Parallel)
    ↓
Data Preprocessing (GPU)
    ↓
Model Initialization
    ↓
Training Loop (GPU)
    ├─→ Forward Pass
    ├─→ Loss Calculation
    ├─→ Backpropagation
    └─→ Optimizer Step
    ↓
Model Serialization
    ↓
Metrics Collection
    ↓
Response
```

**Performance:**
- CPU Training: 45 minutes (typical)
- GPU Training: 4.5 minutes (10x speedup)

---

## Technology Stack

### Frontend Layer
- **Language:** TypeScript 5.0+
- **Runtime:** Node.js 18+
- **Framework:** MCP Protocol (JSON-RPC 2.0)
- **Validation:** Zod schemas

### Backend Layer
- **Language:** Rust 1.75+
- **FFI:** NAPI-RS 2.x
- **Async Runtime:** Tokio
- **Serialization:** Serde

### Neural Networks
- **Framework:** Burn (Rust-native)
- **GPU Acceleration:** CUDA/ROCm
- **Model Format:** Safetensors

### Data Storage
- **Cache:** Redis 7.x
- **Database:** PostgreSQL 15+
- **Time Series:** TimescaleDB
- **Blob Storage:** S3-compatible

### External APIs
- **Trading:** Alpaca Markets API v2
- **News:** NewsAPI, Finnhub, Alpha Vantage
- **Sports:** The Odds API
- **Cloud:** E2B Sandboxes

---

## Design Patterns

### 1. Adapter Pattern (External APIs)

```rust
trait MarketDataProvider {
    async fn get_quote(&self, symbol: &str) -> Result<Quote>;
    async fn get_bars(&self, symbol: &str, timeframe: Timeframe) -> Result<Vec<Bar>>;
}

struct AlpacaAdapter { /* ... */ }
struct PolygonAdapter { /* ... */ }
```

**Benefits:**
- Uniform interface across providers
- Easy provider switching
- Testability with mocks

---

### 2. Strategy Pattern (Trading)

```rust
trait TradingStrategy {
    async fn analyze(&self, data: &MarketData) -> Signal;
    async fn generate_orders(&self, signal: Signal) -> Vec<Order>;
}

struct MomentumStrategy { /* ... */ }
struct MeanReversionStrategy { /* ... */ }
```

**Benefits:**
- Runtime strategy selection
- Strategy composition
- A/B testing support

---

### 3. Builder Pattern (Configuration)

```rust
let config = NeuralConfigBuilder::new()
    .architecture(Architecture::LSTM)
    .layers(vec![128, 64, 32])
    .epochs(100)
    .batch_size(32)
    .learning_rate(0.001)
    .use_gpu(true)
    .build()?;
```

**Benefits:**
- Fluent API
- Compile-time validation
- Default values

---

### 4. Repository Pattern (Data Access)

```rust
trait TradeRepository {
    async fn save(&self, trade: &Trade) -> Result<()>;
    async fn find_by_id(&self, id: &str) -> Result<Option<Trade>>;
    async fn find_by_strategy(&self, strategy: &str) -> Result<Vec<Trade>>;
}
```

**Benefits:**
- Abstraction over storage
- Easy testing with mocks
- Migration flexibility

---

### 5. Circuit Breaker (Resilience)

```rust
struct CircuitBreaker {
    state: Arc<RwLock<State>>,
    failure_threshold: u32,
    timeout: Duration,
}

impl CircuitBreaker {
    async fn call<F, T>(&self, f: F) -> Result<T>
    where F: Future<Output = Result<T>> { /* ... */ }
}
```

**States:**
- **Closed:** Normal operation
- **Open:** Failing fast
- **Half-Open:** Testing recovery

---

## Performance Characteristics

### Latency (P50/P95/P99)

| Operation | P50 | P95 | P99 |
|-----------|-----|-----|-----|
| Market Data Query | 52ms | 120ms | 250ms |
| Trade Execution | 150ms | 300ms | 500ms |
| Neural Prediction | 15ms | 35ms | 80ms |
| Risk Analysis (CPU) | 180ms | 400ms | 1200ms |
| Risk Analysis (GPU) | 18ms | 40ms | 120ms |

### Throughput

| Operation | Requests/sec |
|-----------|-------------|
| Market Data | 1,000 |
| Predictions | 500 |
| Trade Orders | 100 |
| Backtest | 10 |

### Resource Usage

**CPU:**
- Idle: 5-10%
- Trading: 20-40%
- Training: 90-100%

**Memory:**
- Base: 200 MB
- With Cache: 500 MB
- Training (GPU): 2-4 GB

**GPU:**
- Training: 80-90% utilization
- Inference: 30-50% utilization

---

## Security Architecture

### Defense in Depth

```
Layer 1: Network (TLS 1.3, Rate Limiting)
    ↓
Layer 2: Authentication (JWT, API Keys)
    ↓
Layer 3: Authorization (RBAC, Permissions)
    ↓
Layer 4: Input Validation (Schema Validation)
    ↓
Layer 5: Business Logic (Risk Checks)
    ↓
Layer 6: Data Access (Prepared Statements)
    ↓
Layer 7: Audit Logging (Immutable Logs)
```

### Authentication Flow

```
User Credentials → Bcrypt Verification → JWT Generation → Response

JWT on Request → Signature Verification → Claims Validation → Authorize
```

### API Key Security

```rust
struct ApiKey {
    hash: String,        // Argon2 hash
    scopes: Vec<String>, // Permissions
    rate_limit: u32,     // Requests/minute
    expires_at: DateTime<Utc>,
}
```

---

## Deployment Architecture

### Single-Node Deployment

```
┌─────────────────────────────────────┐
│           Load Balancer             │
│         (nginx/Caddy)               │
└────────────┬────────────────────────┘
             │
┌────────────▼────────────────────────┐
│        MCP Server Instance          │
│  • Neural Trader                    │
│  • Redis Cache (local)              │
│  • PostgreSQL (local)               │
└─────────────────────────────────────┘
```

**Use Cases:**
- Development
- Small-scale trading
- Personal use

---

### Multi-Node Deployment (Kubernetes)

```
┌─────────────────────────────────────┐
│         Ingress Controller          │
│     (nginx-ingress/Traefik)         │
└────────────┬────────────────────────┘
             │
     ┌───────┴───────┐
     │               │
┌────▼────┐    ┌────▼────┐
│ MCP Pod │    │ MCP Pod │ ... (auto-scaling)
│  (3 GB) │    │  (3 GB) │
└────┬────┘    └────┬────┘
     │               │
     └───────┬───────┘
             │
┌────────────▼────────────────────────┐
│        Shared Services              │
│  • Redis Cluster                    │
│  • PostgreSQL Primary/Replicas      │
│  • S3 Object Storage                │
└─────────────────────────────────────┘
```

**Scaling Parameters:**
- CPU: 60% threshold
- Memory: 70% threshold
- Min replicas: 2
- Max replicas: 20

---

### E2B Cloud Architecture

```
┌─────────────────────────────────────┐
│        Neural Trader MCP            │
└────────────┬────────────────────────┘
             │ E2B SDK
             ▼
┌─────────────────────────────────────┐
│         E2B Platform                │
│  ┌──────────┐  ┌──────────┐        │
│  │ Sandbox 1│  │ Sandbox 2│  ...   │
│  │ Trading  │  │ Training │        │
│  │ Agent    │  │ Neural   │        │
│  └──────────┘  └──────────┘        │
└─────────────────────────────────────┘
```

**Isolation:**
- Separate VMs per sandbox
- Network isolation
- Resource limits

---

## Scalability Design

### Horizontal Scaling

**Stateless Design:**
- No local state in MCP servers
- Session data in Redis
- Database for persistence

**Load Balancing:**
- Round-robin for general requests
- Consistent hashing for cache affinity
- Sticky sessions for WebSocket

---

### Vertical Scaling

**Resource Limits:**
```yaml
resources:
  requests:
    cpu: 1000m
    memory: 2Gi
  limits:
    cpu: 4000m
    memory: 8Gi
```

**GPU Nodes:**
```yaml
nvidia.com/gpu: 1  # Request 1 GPU
```

---

### Database Scaling

**Read Replicas:**
- 1 Primary (writes)
- 3+ Replicas (reads)
- Automatic failover

**Partitioning:**
- Time-based partitioning for trades
- Hash partitioning for users
- Range partitioning for market data

**Caching Strategy:**
- Cache-aside for reads
- Write-through for critical data
- TTL-based expiration

---

## Monitoring & Observability

### Metrics (Prometheus)

```
# Request metrics
http_requests_total{method, status}
http_request_duration_seconds{method, quantile}

# Trading metrics
trades_executed_total{strategy, symbol}
trade_pnl{strategy, symbol}

# System metrics
rust_memory_usage_bytes
gpu_utilization_percent
```

### Logging (Structured JSON)

```json
{
  "timestamp": "2025-11-14T12:00:00Z",
  "level": "INFO",
  "module": "trading::engine",
  "message": "Trade executed",
  "context": {
    "trade_id": "12345",
    "symbol": "SPY",
    "quantity": 100,
    "price": 450.25
  }
}
```

### Tracing (OpenTelemetry)

```
Span: execute_trade
├─ Span: validate_order
├─ Span: check_risk_limits
├─ Span: submit_to_broker
└─ Span: update_portfolio
```

---

## Disaster Recovery

### Backup Strategy

**Database:**
- Continuous WAL archiving
- Daily full backups
- Point-in-time recovery

**Models:**
- Version control in S3
- Immutable snapshots
- Metadata registry

**Configuration:**
- Git repository
- Encrypted secrets
- Automated sync

### Recovery Procedures

**RTO (Recovery Time Objective):** 15 minutes
**RPO (Recovery Point Objective):** 5 minutes

**Runbooks:**
1. Database failure → Promote replica
2. Service crash → Kubernetes auto-restart
3. Data corruption → Restore from backup
4. Regional outage → Failover to DR region

---

## Architecture Decision Records (ADRs)

### ADR-001: Rust Backend

**Decision:** Use Rust for performance-critical backend

**Rationale:**
- 10x performance vs Python
- Memory safety guarantees
- Zero-cost abstractions
- Excellent GPU support

**Trade-offs:**
- Steeper learning curve
- Longer compilation times
- Smaller ecosystem vs Python

---

### ADR-002: NAPI-RS Bridge

**Decision:** Use NAPI-RS for Node.js integration

**Rationale:**
- Type-safe bindings
- Automatic memory management
- Async/await support
- Active maintenance

**Alternatives Considered:**
- Pure Rust CLI: Loses Node.js ecosystem
- WebAssembly: Performance overhead
- C bindings: Manual memory management

---

### ADR-003: MCP Protocol

**Decision:** Implement MCP for AI agent integration

**Rationale:**
- Standard protocol for AI tools
- Claude Desktop compatibility
- JSON-RPC 2.0 based
- Growing ecosystem

**Benefits:**
- No custom protocol needed
- Built-in validation
- Tool discovery

---

## Future Architecture Evolution

### Planned Improvements

**Q1 2026:**
- WebSocket streaming for real-time data
- GraphQL API for flexible queries
- Multi-region deployment

**Q2 2026:**
- Microservices decomposition
- Event-driven architecture (Kafka)
- Service mesh (Istio)

**Q3 2026:**
- Machine learning pipeline (MLflow)
- A/B testing framework
- Advanced observability

---

**Document Version:** 2.1.0
**Last Updated:** November 14, 2025
**Authors:** Neural Trader Architecture Team
