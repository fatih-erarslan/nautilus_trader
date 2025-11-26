# Architecture Documentation

System architecture, design patterns, and technical specifications for Neural Trader.

## 📐 Architecture Overview

Neural Trader is built on a modular, high-performance architecture:

- **Core:** Rust for performance-critical operations (NAPI bindings)
- **Orchestration:** Node.js/TypeScript for coordination
- **Integration:** MCP protocol for AI assistants
- **Deployment:** E2B sandboxes for distributed execution

## 📚 Documentation

### Core Architecture
- [Architecture Diagrams](./architecture-diagrams.md) - Visual system diagrams
- [Technical Specifications](./technical-specifications.md) - Detailed specs
- [Workspace Architecture](./WORKSPACE_ARCHITECTURE.md) - Monorepo structure

### FFI & Interop
- [FFI Design](./FFI_DESIGN.md) - Foreign Function Interface design
- [NAPI Bindings](../development/NAPI_RS_IMPLEMENTATION_PLAN.md) - Rust ↔ Node.js bridge

### System Components

#### Trading System
```
┌─────────────────────────────────────────┐
│           User Interface                │
│  (CLI, MCP Server, API)                 │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│      Orchestration Layer (Node.js)      │
│  - Task Management                      │
│  - Agent Coordination                   │
│  - Workflow Engine                      │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│        Core Engine (Rust NAPI)          │
│  - Strategy Execution                   │
│  - Neural Networks                      │
│  - Risk Management                      │
│  - Portfolio Optimization               │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│         External Services               │
│  - Brokers (Alpaca, IB, Binance)        │
│  - Data Providers                       │
│  - E2B Sandboxes                        │
└─────────────────────────────────────────┘
```

## 🏗️ Design Patterns

### 1. Modular Package Architecture
Each feature is an independent npm package:

```
@neural-trader/
├── core            # Type definitions
├── strategies      # Trading strategies
├── neural          # Neural networks
├── portfolio       # Portfolio management
├── risk            # Risk management
├── backtesting     # Backtesting engine
├── execution       # Order execution
├── mcp             # MCP server
└── ...            # 17 total packages
```

### 2. Event-Driven Architecture
- Pub/sub for real-time data
- Message queues for task distribution
- WebSocket streams for live updates

### 3. Agent-Based Coordination
- Swarm topologies (hierarchical, mesh, ring, star)
- Autonomous decision-making
- Distributed execution via E2B

## 🔧 Performance Architecture

### Native Performance (Rust)
- **NAPI-RS:** Zero-copy FFI bindings
- **SIMD:** Vectorized operations
- **Async Runtime:** Tokio for concurrency
- **Memory:** Zero-allocation hot paths

### Distributed Execution
- **E2B Sandboxes:** Isolated execution environments
- **Horizontal Scaling:** Add sandboxes on demand
- **Load Balancing:** Adaptive task distribution

### Caching Strategy
- **L1:** In-memory (Node.js)
- **L2:** Redis (shared state)
- **L3:** Database (Supabase)

## 📊 Data Flow

### Trading Workflow
```
Market Data → Feature Engineering → Strategy Engine → Risk Check → Execution → Portfolio Update
     ↓              ↓                     ↓              ↓             ↓            ↓
  Cache      Neural Network          Backtest       VaR/CVaR      Broker API   Database
```

### Neural Training Workflow
```
Historical Data → Preprocessing → Model Training → Validation → Deployment → Inference
      ↓               ↓                ↓              ↓            ↓           ↓
  Supabase        WASM/Rust        GPU/Cloud    Test Metrics   Model Store  Real-time
```

## 🔐 Security Architecture

### Authentication & Authorization
- JWT tokens for API access
- API key rotation
- Role-based access control (RBAC)

### Data Security
- Encrypted at rest (Supabase)
- TLS for data in transit
- Secret management (environment variables)

### Execution Isolation
- E2B sandboxes for untrusted code
- Resource limits per sandbox
- Network isolation

## 🌐 Deployment Architecture

### Local Development
```
Developer Machine
├── Node.js runtime
├── Rust toolchain
├── Local database (optional)
└── MCP server (stdio)
```

### Production (Fly.io)
```
Fly.io Infrastructure
├── App instances (Node.js + Rust NAPI)
├── PostgreSQL (Supabase)
├── Redis cache
├── E2B sandboxes (on-demand)
└── Load balancer
```

### Distributed Trading (E2B)
```
E2B Cloud
├── Sandbox 1 → Strategy A
├── Sandbox 2 → Strategy B
├── Sandbox 3 → Neural Training
├── Sandbox 4 → Risk Analysis
└── Coordinator → Results aggregation
```

## 📖 Related Documentation

- [Rust Port Documentation](../development/rust-port/)
- [Distributed Systems](../advanced/distributed-systems-architecture.md)
- [Integration Architecture](../advanced/integration-architecture.md)
- [Development Guide](../development/)

## 🔗 External Resources

- [NAPI-RS Documentation](https://napi.rs/)
- [E2B Platform](https://e2b.dev/)
- [Supabase](https://supabase.com/)
- [Fly.io](https://fly.io/)

---

[← Back to Main Docs](../README.md) | [Development →](../development/)
