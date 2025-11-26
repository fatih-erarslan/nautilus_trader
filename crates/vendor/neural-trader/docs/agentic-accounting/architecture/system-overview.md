# System Architecture Overview - Agentic Accounting

## Executive Summary

The Agentic Accounting System is a **multi-agent autonomous accounting platform** that combines:
- **66 specialized agents** via Agentic Flow
- **High-performance Rust** for sub-millisecond computations
- **Vector-based semantic search** via AgentDB (150×-12,500× faster)
- **Formal verification** via Lean4 theorem proving
- **Self-learning** through ReasoningBank persistent memory

**Performance Targets**:
- Vector search: <100µs
- Tax calculation: <10ms per transaction
- Compliance check: <1 second
- API latency: <200ms p95

---

## System Layers

### 1. User Interface Layer
**Purpose**: Expose system functionality to users and external systems

**Components**:
- **CLI** (Commander.js): Interactive and batch command execution
- **REST API** (Express): HTTP endpoints for programmatic access
- **GraphQL API** (Apollo): Flexible query interface
- **MCP Server**: Claude Code integration with 10+ accounting tools

**Key Endpoints**:
```typescript
// MCP Tools
accounting_add_transaction(transaction)
accounting_calculate_tax(disposalId, method)
accounting_harvest_losses(portfolioId)
compliance_check_trade(transaction)
forensic_find_similar(transactionId, threshold)
accounting_generate_report(year, format)

// REST API
POST   /api/v1/transactions
GET    /api/v1/positions
POST   /api/v1/disposals/calculate
GET    /api/v1/tax-summary/:year
```

---

### 2. Orchestration Layer
**Purpose**: Coordinate multi-agent workflows and manage system state

**Components**:

#### A. Agentic Flow Coordinator
- **Task Routing**: Assigns tasks to appropriate agents
- **Lifecycle Management**: Spawn, monitor, terminate agents
- **Load Balancing**: Distributes work across agent pool
- **Failure Recovery**: Automatic retry and fallback strategies

#### B. ReasoningBank Memory
- **Decision Storage**: Records agent decisions with rationales
- **Experience Replay**: Retrieves similar past decisions
- **Continuous Learning**: Updates strategies based on outcomes
- **Cross-Agent Knowledge**: Shares learning between agents

**Data Flow**:
```
User Request → MCP Tool → Coordinator
                ↓
         Task Assignment → Specialized Agent
                ↓
         Execute (Rust Core) → Store Result
                ↓
         Log Decision (ReasoningBank) → Return to User
```

---

### 3. Agent Swarm Layer
**Purpose**: Specialized agents for domain-specific tasks

**Agent Types**:

1. **Transaction Ingestion Agent**
   - Import from exchanges, blockchains, ERPs
   - Normalize data formats
   - Validate and enrich transactions

2. **Tax Computation Agent**
   - Calculate gains/losses (FIFO/LIFO/HIFO)
   - Track cost basis per lot
   - Generate tax-year summaries

3. **Tax-Loss Harvesting Agent**
   - Scan for harvestable losses
   - Check wash-sale compliance
   - Recommend optimal timing

4. **Compliance Agent**
   - Validate against rules
   - Trigger alerts for violations
   - Monitor thresholds

5. **Forensic Analysis Agent**
   - Detect fraud patterns via vector search
   - Link communications with transactions
   - Generate Merkle proofs

6. **Reporting Agent**
   - Generate P&L statements
   - Create tax forms (Schedule D, 8949)
   - Export audit documentation

7. **Learning & Optimization Agent**
   - Train on overnight batches
   - Update ReasoningBank
   - Tune agent parameters

8. **Coordinator Agent (Queen)**
   - Orchestrate workflows
   - Resolve conflicts
   - Monitor system health

**Communication**:
- **Synchronous**: Direct MCP calls for immediate responses
- **Asynchronous**: Message queue (BullMQ/Redis) for batch jobs
- **Broadcast**: Coordinator publishes state updates
- **Peer-to-Peer**: Direct agent-to-agent for specific workflows

---

### 4. Computation Layer (Rust Core)
**Purpose**: High-performance calculations with sub-millisecond latency

**Modules**:

#### A. Tax Calculation
```rust
pub mod tax {
  pub fn calculate_fifo(...) -> TaxResult;
  pub fn calculate_lifo(...) -> TaxResult;
  pub fn calculate_hifo(...) -> TaxResult;
  pub fn detect_wash_sale(...) -> WashSaleResult;
}
```

#### B. Forensic Operations
```rust
pub mod forensic {
  pub fn build_merkle_tree(...) -> MerkleRoot;
  pub fn sign_audit_entry(...) -> Signature;
  pub fn hash_transaction(...) -> Hash;
}
```

#### C. Report Generation
```rust
pub mod reports {
  pub fn generate_pdf(...) -> Vec<u8>;
  pub fn render_schedule_d(...) -> String;
}
```

#### D. Performance Optimization
```rust
pub mod performance {
  pub fn calculate_indicators_simd(...) -> Indicators;
  pub fn optimize_portfolio(...) -> Portfolio;
}
```

**Performance Features**:
- **SIMD Vectorization**: Process 4-8 values per instruction
- **Rayon Parallelism**: Utilize all CPU cores
- **Zero-Copy**: Minimize memory allocations
- **Cache-Friendly**: Optimize data structures for L1/L2 cache

---

### 5. Data Layer
**Purpose**: Persistent and in-memory storage with performance guarantees

#### A. AgentDB (Vector Database)
**Configuration**:
```typescript
const agentdb = new AgentDB({
  dimensions: 768,
  distanceMetric: 'cosine',
  indexType: 'hnsw',
  quantization: 'int8',
  persistence: true,
});
```

**Collections**:
1. `transactions`: Transaction embeddings for fraud detection
2. `fraud_signatures`: Known fraud patterns
3. `communications`: Email/message embeddings
4. `reasoning_bank`: Agent decision history

**Performance**: <100µs for top-10 similarity search

#### B. PostgreSQL (Relational Database)
**Tables**:
1. `transactions`: All financial transactions
2. `tax_lots`: Individual lot tracking
3. `disposals`: Sale records with gains/losses
4. `positions`: Current holdings (materialized view)
5. `tax_summaries`: Annual tax data
6. `compliance_rules`: Rule definitions
7. `audit_trail`: Immutable log with cryptographic hashing
8. `embeddings`: pgvector for hybrid queries
9. `reasoning_bank`: Agent learning memory
10. `verification_proofs`: Lean4 proof certificates

**Extensions**:
- `uuid-ossp`: UUID generation
- `pgvector`: Vector similarity search

**Indexes**:
- B-tree: Primary keys, foreign keys, timestamps
- HNSW: Vector similarity (pgvector)
- GiST: Multi-column searches

#### C. Lean4 Verifier (Formal Verification)
**Theorems**:
```lean
-- Balance consistency
theorem balance_equation (ledger : Ledger) :
  ledger.assets = ledger.liabilities + ledger.equity

-- Non-negative holdings
theorem positions_non_negative (position : Position) :
  position.quantity ≥ 0

-- Cost basis accuracy
theorem disposal_cost_basis_correct (lots, disposal) :
  disposal.costBasis = lots.selectedSum
```

**Integration**: Async verification with proof certificates stored in DB

---

### 6. Integration Layer
**Purpose**: Connect to external data sources

**Integrations**:
- **Exchanges**: Coinbase, Binance, Kraken (REST + WebSocket)
- **Blockchains**: Etherscan, Blockchair (transaction history)
- **Market Data**: CoinGecko, Alpha Vantage (price feeds)
- **ERPs**: QuickBooks, Xero (accounting integration)

**Error Handling**:
- Exponential backoff for rate limits
- Circuit breakers for failed services
- Fallback data sources
- Retry queue

---

## Package Structure

```
neural-trader/
└── packages/
    ├── agentic-accounting-core/       # Core TypeScript library
    ├── agentic-accounting-rust-core/  # Rust addon via napi-rs
    ├── agentic-accounting-agents/     # Agent implementations
    ├── agentic-accounting-types/      # Shared TypeScript types
    ├── agentic-accounting-mcp/        # MCP server
    ├── agentic-accounting-api/        # REST/GraphQL APIs
    └── agentic-accounting-cli/        # Command-line interface
```

**Dependencies**:
- `core` depends on: `types`, `rust-core`
- `agents` depends on: `core`, `types`
- `mcp` depends on: `core`, `agents`, `types`
- `api` depends on: `core`, `agents`, `types`
- `cli` depends on: `core`, `types`

---

## Deployment Architecture

### Development
```
Local Machine
├── Node.js (TypeScript)
├── Rust addon (compiled)
├── PostgreSQL (Docker)
├── AgentDB (in-process)
└── Redis (Docker)
```

### Production (Kubernetes)
```
┌─────────────────────────────────────────┐
│           Ingress (NGINX)               │
└──────────────┬──────────────────────────┘
               │
       ┌───────┴────────┐
       ▼                ▼
 ┌──────────┐     ┌──────────┐
 │   API    │     │   MCP    │
 │(5 pods)  │     │(3 pods)  │
 └────┬─────┘     └────┬─────┘
      │                │
      └────────┬───────┘
               ▼
      ┌─────────────────┐
      │  Coordinator    │
      │   (2 pods)      │
      └────────┬────────┘
               ▼
      ┌─────────────────┐
      │  Agent Workers  │
      │  (20-50 pods)   │
      └────────┬────────┘
               ▼
      ┌─────────────────┐
      │      Redis      │
      │   (3 nodes)     │
      └─────────────────┘
               │
       ┌───────┴────────┐
       ▼                ▼
 ┌──────────┐     ┌──────────┐
 │PostgreSQL│     │ AgentDB  │
 │(Primary +│     │ (Cluster)│
 │3 Replicas│     │          │
 └──────────┘     └──────────┘
```

**Scaling**:
- Horizontal Pod Autoscaling (HPA) based on CPU/memory
- Agent pods scale 5-50 based on queue depth
- PostgreSQL read replicas for reporting queries
- Redis for distributed task queues and caching

---

## Security Architecture

### Authentication & Authorization
- **JWT tokens**: Stateless authentication
- **RBAC**: Role-based access control
- **API keys**: Service-to-service authentication

### Encryption
- **At rest**: AES-256 for database
- **In transit**: TLS 1.3 for all connections
- **Secrets**: HashiCorp Vault

### Audit & Compliance
- **Immutable logs**: Append-only audit trail
- **Ed25519 signatures**: Cryptographic verification
- **Merkle proofs**: Tamper detection
- **Access logging**: Track all data access

---

## Monitoring & Observability

### Metrics (Prometheus)
- Agent performance (task duration, success rate)
- API latency (p50, p95, p99)
- Database query performance
- Vector search latency
- Rust addon execution time

### Logging (Winston + ELK)
- Structured JSON logs
- Distributed tracing (OpenTelemetry)
- Error tracking (Sentry)
- Agent decision logs

### Alerts
- High anomaly scores (fraud detection)
- System errors (>0.1% rate)
- Performance degradation
- Compliance violations
- Resource exhaustion

---

## Technology Stack Summary

| Layer | Technology |
|-------|------------|
| **Runtime** | Node.js 18+, TypeScript 5.3+ |
| **High-Performance** | Rust 1.75+ (napi-rs) |
| **Orchestration** | Agentic Flow, MCP Protocol |
| **Vector DB** | AgentDB (HNSW) |
| **Relational DB** | PostgreSQL 15+ (pgvector) |
| **Verification** | Lean4 |
| **Caching** | Redis 7+ |
| **Message Queue** | BullMQ |
| **API** | Express, Apollo GraphQL |
| **Monitoring** | Prometheus, Grafana |
| **Logging** | Winston, ELK Stack |
| **Deployment** | Docker, Kubernetes |
| **CI/CD** | GitHub Actions |

---

## Next Steps

1. **Phase 1: Foundation** (Weeks 1-2)
   - Complete monorepo setup
   - Implement database schema
   - Build Rust addon foundation

2. **Phase 2: Tax Engine** (Weeks 3-4)
   - Implement FIFO/LIFO/HIFO in Rust
   - Build TaxComputeAgent
   - Achieve <10ms performance

3. **Phase 3: Agents** (Weeks 5-10)
   - Deploy all 8 core agents
   - Integrate ReasoningBank
   - Build compliance and forensic systems

4. **Phase 4: Production** (Weeks 11-20)
   - APIs and CLI
   - Kubernetes deployment
   - Full testing and validation
