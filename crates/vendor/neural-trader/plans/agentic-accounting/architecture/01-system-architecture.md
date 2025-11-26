# Agentic Accounting System - System Architecture

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Interface Layer                      │
│         (CLI, REST API, GraphQL, MCP Tools Interface)           │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Orchestration Layer                          │
│  ┌──────────────┐  ┌─────────────┐  ┌──────────────────────┐  │
│  │ Agentic Flow │→ │ Coordinator │→ │ ReasoningBank Memory │  │
│  │ (MCP Server) │  │   Agent     │  │   (AgentDB)          │  │
│  └──────────────┘  └─────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      Agent Swarm Layer                          │
│  ┌────────────┐ ┌──────────┐ ┌─────────┐ ┌──────────────┐    │
│  │ Ingestion  │ │   Tax    │ │ Harvest │ │  Compliance  │    │
│  │   Agent    │ │ Compute  │ │  Agent  │ │    Agent     │    │
│  └────────────┘ └──────────┘ └─────────┘ └──────────────┘    │
│  ┌────────────┐ ┌──────────┐ ┌─────────┐ ┌──────────────┐    │
│  │  Forensic  │ │ Reporting│ │ Learning│ │  Verification│    │
│  │   Agent    │ │  Agent   │ │  Agent  │ │    Agent     │    │
│  └────────────┘ └──────────┘ └─────────┘ └──────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Computation Layer                            │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Rust Core (via napi-rs)                    │   │
│  │  • Tax calculations     • Portfolio optimization        │   │
│  │  • Cost basis tracking  • VaR/Risk calculations         │   │
│  │  • Report generation    • Cryptographic operations      │   │
│  │  • Technical indicators • SIMD optimizations            │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                       Data Layer                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐     │
│  │   AgentDB    │  │  PostgreSQL  │  │  Lean4 Verifier  │     │
│  │  (Vectors)   │  │  (Relational)│  │  (Proofs)        │     │
│  │  • HNSW      │  │  • pgvector  │  │  • Theorems      │     │
│  │  • Embeddings│  │  • Audit logs│  │  • Invariants    │     │
│  │  • O(log n)  │  │  • ACID      │  │  • Certificates  │     │
│  └──────────────┘  └──────────────┘  └──────────────────┘     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Integration Layer                            │
│  External APIs: Exchanges, ERPs, Blockchain, Market Data       │
└─────────────────────────────────────────────────────────────────┘
```

## Component Breakdown

### 1. User Interface Layer

**Purpose**: Expose system functionality to users and external systems

**Components**:
- **MCP Tools Interface**: Primary interface for Claude Code integration
- **REST API**: HTTP endpoints for programmatic access
- **GraphQL API**: Flexible query interface for complex data retrieval
- **CLI**: Command-line tools for administrative tasks

**Technology Stack**:
- Node.js/Express for REST API
- Apollo Server for GraphQL
- Commander.js for CLI
- MCP Protocol for tool exposure

**Key Endpoints**:
```typescript
// MCP Tools
accounting_add_transaction(transaction: Transaction): Result
accounting_calculate_tax(disposalId: string, method: AccountingMethod): TaxResult
accounting_harvest_losses(portfolioId: string): HarvestOpportunity[]
compliance_check_trade(transaction: Transaction): ComplianceResult
forensic_find_similar(transactionId: string, threshold: number): Match[]
accounting_generate_report(year: number, format: ReportFormat): Report
learning_feedback(agentId: string, outcome: Outcome, feedback: string): void

// REST API
POST   /api/v1/transactions
GET    /api/v1/transactions/:id
GET    /api/v1/positions
POST   /api/v1/disposals/calculate
GET    /api/v1/tax-summary/:year
GET    /api/v1/harvest-opportunities
POST   /api/v1/compliance/check
GET    /api/v1/audit-trail
POST   /api/v1/reports/generate
```

---

### 2. Orchestration Layer

**Purpose**: Coordinate multi-agent workflows and manage system state

**Components**:

#### A. Agentic Flow MCP Server
- Provides 213+ orchestration tools
- Manages agent lifecycle (spawn, monitor, terminate)
- Implements ReasoningBank memory integration
- Handles inter-agent communication
- Supports hierarchical, mesh, and adaptive topologies

#### B. Coordinator Agent
- Central orchestration hub
- Task allocation and routing
- Conflict resolution
- Load balancing
- Failure recovery
- Performance monitoring

#### C. ReasoningBank Memory
- Stores decision rationales and outcomes
- Enables experience replay
- Supports similarity-based retrieval
- Facilitates continuous learning
- Cross-agent knowledge sharing

**Data Flow**:
```
User Request → MCP Tool → Coordinator Agent → Task Queue
                                   ↓
                         Assign to Specialized Agent
                                   ↓
                         Agent executes with Rust core
                                   ↓
                         Store outcome in ReasoningBank
                                   ↓
                         Return result to user
```

---

### 3. Agent Swarm Layer

**Purpose**: Specialized agents for domain-specific tasks

**Agent Topology**: Adaptive hybrid (hierarchical + mesh)

**Agent Communication Protocols**:
- **Synchronous**: Direct MCP tool calls for immediate responses
- **Asynchronous**: Message queue for batch processing
- **Broadcast**: Coordinator publishes to all agents for state updates
- **Peer-to-Peer**: Forensic ↔ Compliance direct coordination

**Agent State Management**:
- Each agent maintains local state in memory
- Shared state persisted in PostgreSQL
- ReasoningBank for decision history
- AgentDB for semantic queries

**Scaling Strategy**:
- **Static agents**: 5 core agents always running
- **Dynamic agents**: Spawned on-demand for workload spikes
- **Horizontal scaling**: Multiple instances per agent type
- **Load balancing**: Round-robin with health checks

---

### 4. Computation Layer (Rust Core)

**Purpose**: High-performance calculations with sub-millisecond latency

**Architecture**:
```
┌─────────────────────────────────────────────────────┐
│              Node.js Process                        │
│  ┌──────────────────────────────────────────────┐  │
│  │      TypeScript Business Logic              │  │
│  └──────────────┬──────────────────────────────┘  │
│                 │ N-API Calls                      │
│  ┌──────────────▼──────────────────────────────┐  │
│  │      Rust Addon (via napi-rs)              │  │
│  │  ┌────────────────────────────────────┐    │  │
│  │  │  Tax Calculation Module            │    │  │
│  │  │  • FIFO/LIFO/HIFO algorithms       │    │  │
│  │  │  • Wash sale detection             │    │  │
│  │  │  • Cost basis tracking             │    │  │
│  │  └────────────────────────────────────┘    │  │
│  │  ┌────────────────────────────────────┐    │  │
│  │  │  Forensic Analysis Module          │    │  │
│  │  │  • Similarity hashing              │    │  │
│  │  │  • Merkle tree construction        │    │  │
│  │  │  • Ed25519 signatures              │    │  │
│  │  └────────────────────────────────────┘    │  │
│  │  ┌────────────────────────────────────┐    │  │
│  │  │  Report Generation Module          │    │  │
│  │  │  • PDF rendering                   │    │  │
│  │  │  • Tax form templates              │    │  │
│  │  │  • Data aggregation                │    │  │
│  │  └────────────────────────────────────┘    │  │
│  │  ┌────────────────────────────────────┐    │  │
│  │  │  Performance Module                │    │  │
│  │  │  • Technical indicators (SIMD)     │    │  │
│  │  │  • Portfolio optimization          │    │  │
│  │  │  • VaR/CVaR calculations           │    │  │
│  │  └────────────────────────────────────┘    │  │
│  └─────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

**Rust Module Structure**:
```
agentic-accounting-core/
├── src/
│   ├── lib.rs                 # napi-rs entry point
│   ├── tax/
│   │   ├── mod.rs
│   │   ├── fifo.rs
│   │   ├── lifo.rs
│   │   ├── hifo.rs
│   │   ├── specific_id.rs
│   │   ├── average_cost.rs
│   │   └── wash_sale.rs
│   ├── forensic/
│   │   ├── mod.rs
│   │   ├── merkle.rs
│   │   ├── signatures.rs
│   │   └── hashing.rs
│   ├── reports/
│   │   ├── mod.rs
│   │   ├── pdf.rs
│   │   └── forms.rs
│   ├── performance/
│   │   ├── mod.rs
│   │   ├── indicators.rs
│   │   └── optimization.rs
│   └── types.rs              # Shared types
├── Cargo.toml
└── build.rs
```

**Performance Optimizations**:
- **SIMD**: Vectorized operations for batch calculations
- **Rayon**: Parallel processing across CPU cores
- **Unsafe blocks**: Zero-copy data transfer where safe
- **Memory pools**: Reuse allocations for frequent operations
- **Cache-friendly data structures**: Minimize cache misses

---

### 5. Data Layer

**Purpose**: Persistent and in-memory storage with performance guarantees

#### A. AgentDB (Vector Database)

**Use Cases**:
- Transaction similarity search
- Fraud pattern matching
- Communication-transaction linking
- ReasoningBank decision retrieval

**Configuration**:
```typescript
const agentdb = new AgentDB({
  dimensions: 768,              // Embedding size
  distanceMetric: 'cosine',
  indexType: 'hnsw',
  hnswParams: {
    m: 16,                      // Number of connections
    efConstruction: 200,        // Build-time accuracy
    efSearch: 100,              // Query-time accuracy
  },
  quantization: 'int8',         // 4x memory reduction
  persistence: {
    enabled: true,
    path: './data/agentdb',
    syncInterval: 60000,        // Sync every 60s
  },
});
```

**Collections**:
1. `transactions`: All transaction embeddings
2. `fraud_signatures`: Known fraud patterns
3. `communications`: Email/message embeddings
4. `reasoning_bank`: Agent decision history

**Performance Targets**:
- Insert: <1ms per vector
- Search: <100µs for top-10 results
- Throughput: 10,000+ queries/sec

#### B. PostgreSQL (Relational Database)

**Schema**:
- 10 core tables (see Data Models document)
- pgvector extension for hybrid queries
- Partitioning by tax year for scalability
- Read replicas for reporting workloads

**Indexes**:
```sql
-- B-tree indexes for common queries
CREATE INDEX idx_transactions_timestamp ON transactions(timestamp);
CREATE INDEX idx_transactions_asset ON transactions(asset);
CREATE INDEX idx_tax_lots_status ON tax_lots(status) WHERE status != 'CLOSED';

-- pgvector indexes for hybrid search
CREATE INDEX idx_embeddings_vector ON embeddings
  USING hnsw (vector vector_cosine_ops);

-- Composite indexes for complex queries
CREATE INDEX idx_disposals_year_term ON disposals(tax_year, term);
```

**Backup & Recovery**:
- Continuous archiving with WAL
- Point-in-time recovery (PITR)
- Daily full backups
- Hourly incremental backups
- 7-year retention for compliance

#### C. Lean4 Verifier (Formal Verification)

**Purpose**: Mathematically prove accounting invariants

**Theorem Examples**:
```lean
-- Balance sheet equation
theorem balance_consistency (ledger : Ledger) :
  ledger.assets = ledger.liabilities + ledger.equity

-- No negative holdings
theorem non_negative_positions (position : Position) :
  position.quantity ≥ 0

-- Cost basis accuracy
theorem cost_basis_sum (lots : List TaxLot) (disposal : Disposal) :
  disposal.costBasis = (lots.filter (λ lot => lot.id ∈ disposal.lotIds))
    .map (λ lot => lot.unitCostBasis * disposal.quantityFromLot lot.id)
    .sum
```

**Integration**:
- Proofs generated on critical operations
- Verification runs in background (async)
- Failed proofs trigger alerts
- Proof certificates stored in audit trail

---

### 6. Integration Layer

**Purpose**: Connect to external data sources and services

**Integrations**:

#### A. Exchange APIs
- **Coinbase Pro**, **Binance**, **Kraken**: Real-time trade data
- **Webhooks**: Instant transaction notifications
- **Rate limiting**: Respect API quotas

#### B. Blockchain APIs
- **Etherscan**, **Blockchair**: Transaction history
- **Web3 providers**: Direct blockchain queries
- **IPFS**: Decentralized storage for proofs

#### C. Market Data
- **CoinGecko**, **CoinMarketCap**: Price feeds
- **Alpha Vantage**: Historical data
- **WebSocket streams**: Real-time prices

#### D. ERP Systems
- **QuickBooks**, **Xero**, **SAP**: Accounting integration
- **CSV/Excel imports**: Manual data entry
- **SFTP**: Automated file transfers

**Error Handling**:
- Exponential backoff for rate limits
- Circuit breakers for failed services
- Fallback data sources
- Queue for retry processing

---

## Deployment Architecture

### Development Environment
```
Local Machine
├── Node.js app (TypeScript)
├── Rust addon (compiled)
├── PostgreSQL (Docker)
├── AgentDB (in-process)
└── Lean4 (optional)
```

### Production Environment
```
Kubernetes Cluster
├── Agent Pods (5-50 instances)
│   ├── Coordinator Agent
│   ├── Specialized Agents
│   └── Sidecar: AgentDB cache
├── API Gateway (NGINX)
├── PostgreSQL (Cloud SQL / RDS)
│   ├── Primary instance
│   └── Read replicas (3x)
├── Redis (caching & queues)
├── Monitoring (Prometheus + Grafana)
└── Logging (ELK stack)
```

### Scaling Strategy
- **Horizontal**: Add more agent pods
- **Vertical**: Increase pod resources for Rust computations
- **Database**: Read replicas + connection pooling
- **Caching**: Redis for frequently accessed data
- **CDN**: Static assets and reports

---

## Security Architecture

### Authentication & Authorization
- **JWT tokens**: Stateless auth
- **RBAC**: Role-based permissions
- **API keys**: Service-to-service auth
- **OAuth 2.0**: Third-party integrations

### Encryption
- **At rest**: AES-256 for database
- **In transit**: TLS 1.3 for all connections
- **Secrets**: Vault for key management
- **Backups**: Encrypted before storage

### Audit & Compliance
- **Immutable logs**: Append-only audit trail
- **Ed25519 signatures**: Verify agent actions
- **Merkle proofs**: Tamper detection
- **Access logs**: Track all data access

---

## Monitoring & Observability

### Metrics
- Agent performance (task duration, success rate)
- API latency (p50, p95, p99)
- Database query performance
- Vector search latency
- Rust addon execution time

### Logging
- Structured JSON logs
- Distributed tracing (OpenTelemetry)
- Error tracking (Sentry)
- Agent decision logs

### Alerts
- High anomaly scores (fraud detection)
- System errors (>0.1% rate)
- Performance degradation (latency spikes)
- Compliance violations
- Resource exhaustion

---

## Technology Stack Summary

| Layer | Technology |
|-------|------------|
| Runtime | Node.js 18+, TypeScript |
| High-Performance | Rust (napi-rs) |
| Orchestration | Agentic Flow, MCP Protocol |
| Vector DB | AgentDB (HNSW) |
| Relational DB | PostgreSQL 15+ (pgvector) |
| Verification | Lean4 |
| Caching | Redis |
| Message Queue | BullMQ |
| API | Express, Apollo GraphQL |
| Monitoring | Prometheus, Grafana |
| Logging | Winston, ELK stack |
| Deployment | Docker, Kubernetes |
| CI/CD | GitHub Actions |
