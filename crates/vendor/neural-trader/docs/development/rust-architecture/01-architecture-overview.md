# Neural Trading System - Rust Architecture Overview

**Version:** 1.0.0
**Date:** 2025-11-12
**Status:** Design Phase

## Executive Summary

This document outlines the comprehensive architecture for porting the Neural Trading system from Python/TypeScript to Rust. The design prioritizes performance, type safety, concurrency, and seamless Node.js interoperability while maintaining the system's advanced AI trading capabilities.

## Core Design Principles

### 1. Zero-Cost Abstractions
- Compile-time polymorphism using traits
- Generic programming without runtime overhead
- Stack allocation by default with explicit heap usage
- SIMD acceleration where applicable

### 2. Fearless Concurrency
- Message-passing via Tokio channels
- Lock-free data structures where possible
- Actor-based patterns for strategy isolation
- Work-stealing runtime for optimal CPU utilization

### 3. Type-Driven Development
- Strong type system prevents entire classes of bugs
- Builder patterns for complex configurations
- Type states for lifecycle management
- Compile-time guarantees for trading rules

### 4. Modular Architecture
- Clear separation of concerns
- Plugin-based strategy system
- Dependency injection via traits
- Feature flags for optional components

## System Architecture Layers

```
┌─────────────────────────────────────────────────────────────────┐
│                     External Interfaces                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Node.js    │  │   CLI/MCP    │  │   gRPC API   │          │
│  │  (napi-rs)   │  │  Interface   │  │              │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Application Layer                           │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │    Trading Orchestrator (Agentic Flow Federation)        │  │
│  │  - Portfolio Manager    - Risk Manager   - Order Router  │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Strategy Layer                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │  Strategy   │  │  Signal     │  │  Execution  │            │
│  │  Plugins    │  │  Generator  │  │  Engine     │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
└─────────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Processing Layer                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │  Feature    │  │  Sublinear  │  │  Neural     │            │
│  │  Extraction │  │  Solvers    │  │  Inference  │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
└─────────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Data Layer                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │  Market     │  │  AgentDB    │  │  Event      │            │
│  │  Data       │  │  Memory     │  │  Streaming  │            │
│  │  (Polars)   │  │  (Vectors)  │  │  (Midstream)│            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
└─────────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Infrastructure Layer                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │  E2B        │  │  Observ-    │  │  AIDefence  │            │
│  │  Sandboxes  │  │  ability    │  │  Governance │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
└─────────────────────────────────────────────────────────────────┘
```

## High-Level Component Interaction

```
Market Data Feed
      │
      ▼
┌─────────────────┐
│  Data Ingestion │───────┐
│   (Tokio async) │       │
└─────────────────┘       │
                          ▼
                  ┌───────────────┐
                  │  AgentDB      │
                  │  Memory Store │
                  └───────────────┘
                          │
                          ▼
┌─────────────────┐   ┌──────────────┐   ┌─────────────────┐
│  Feature        │──▶│  Signal      │──▶│  Order          │
│  Extraction     │   │  Generation  │   │  Execution      │
│  (Polars)       │   │  (Sublinear) │   │  (Exchange API) │
└─────────────────┘   └──────────────┘   └─────────────────┘
      │                     │                     │
      ▼                     ▼                     ▼
┌─────────────────┐   ┌──────────────┐   ┌─────────────────┐
│  Midstreamer    │   │  Risk        │   │  Settlement     │
│  Event Bus      │   │  Checks      │   │  & Reporting    │
└─────────────────┘   └──────────────┘   └─────────────────┘
      │                     │                     │
      └─────────────────────┴─────────────────────┘
                          ▼
                  ┌───────────────┐
                  │  Agentic      │
                  │  Payments     │
                  └───────────────┘
```

## Key Architectural Decisions

### ADR-001: Tokio Async Runtime
**Decision:** Use Tokio as the primary async runtime

**Rationale:**
- Most mature async ecosystem in Rust
- Excellent performance characteristics (work-stealing scheduler)
- Rich ecosystem of compatible libraries
- Built-in tracing integration
- Multi-threaded by default with configurable worker threads

**Alternatives Considered:**
- async-std: Less ecosystem support, similar performance
- smol: Simpler but less feature-complete
- Custom runtime: Too much engineering overhead

**Implications:**
- All async code must use Tokio-compatible futures
- I/O operations leverage Tokio's non-blocking primitives
- Enables efficient handling of thousands of concurrent market data streams

### ADR-002: Polars for DataFrame Operations
**Decision:** Use Polars instead of ndarray or custom structures

**Rationale:**
- Native Rust implementation with zero Python overhead
- Apache Arrow interoperability for efficient data exchange
- Lazy evaluation for query optimization
- SIMD acceleration built-in
- Memory-efficient chunked arrays
- Better performance than Pandas in benchmarks (5-10x faster)

**Alternatives Considered:**
- ndarray: Lower-level, no DataFrame abstractions
- DataFusion: More focused on SQL, heavier weight
- Custom implementation: Reinventing the wheel

**Implications:**
- Market data represented as Polars DataFrames
- Time-series operations benefit from vectorization
- Easy interop with Python via Arrow IPC

### ADR-003: napi-rs for Node.js Interop
**Decision:** napi-rs as primary Node.js binding layer

**Rationale:**
- Type-safe bindings generated from Rust code
- Automatic TypeScript definition generation
- N-API stable ABI (no recompilation across Node versions)
- Excellent performance (near-native speed)
- Active maintenance and community
- Zero-copy buffer transfers

**Fallback Strategy:**
1. **Primary:** napi-rs (best performance, native modules)
2. **Fallback 1:** WASI + WebAssembly (portable, sandbox)
3. **Fallback 2:** CLI + IPC (JSON over stdin/stdout)
4. **Fallback 3:** gRPC service (language-agnostic)

**Implications:**
- Separate workspace member for napi bindings
- Build scripts generate .node binaries
- TypeScript definitions auto-generated
- Careful memory management at FFI boundary

### ADR-004: AgentDB Integration
**Decision:** Integrate AgentDB via FFI or gRPC

**Rationale:**
- Vector storage essential for neural features
- Memory persistence across agent restarts
- Fast similarity search for pattern matching
- Built-in HNSW indexing

**Implementation:**
- Rust client library wrapping AgentDB API
- Connection pooling for concurrent access
- Local cache with write-through semantics
- Async/await interface using Tokio

### ADR-005: Sublinear Time Solvers
**Decision:** Implement custom sublinear algorithms in Rust

**Rationale:**
- Portfolio optimization benefits from approximation algorithms
- Sketching algorithms for streaming data
- Locality-sensitive hashing for pattern matching
- Count-min sketch for frequency estimation

**Key Algorithms:**
- **Reservoir sampling:** Fixed-memory stream sampling
- **HyperLogLog:** Cardinality estimation
- **Bloom filters:** Membership testing
- **Online convex optimization:** Parameter updates

### ADR-006: Event Streaming with Midstreamer
**Decision:** Use Midstreamer for event-driven architecture

**Rationale:**
- Decouples system components
- Enables audit trail and replay
- Facilitates backtesting with recorded events
- Natural fit for trading event flows

**Event Types:**
- Market data updates
- Signal generation
- Order placement/execution
- Risk violations
- System health metrics

### ADR-007: E2B Sandboxes for Execution
**Decision:** Deploy strategies in E2B isolated environments

**Rationale:**
- Security isolation between strategies
- Resource limits prevent runaway processes
- Clean environment for reproducibility
- Easy scaling and orchestration

**Implementation:**
- Strategies compiled as standalone binaries
- Docker-compatible deployment
- Health checks and auto-restart
- Graceful shutdown on SIGTERM

### ADR-008: Agentic Flow Federation
**Decision:** Scale-out using federated agent architecture

**Rationale:**
- Horizontal scaling across machines
- Geographic distribution for latency
- Fault isolation
- Independent strategy lifecycles

**Communication:**
- gRPC for inter-agent RPC
- NATS for pub/sub messaging
- Consistent hashing for routing
- Gossip protocol for membership

### ADR-009: AIDefence & Lean Agentic Governance
**Decision:** Embed governance checks at compile-time and runtime

**Rationale:**
- Prevent unauthorized trades
- Enforce risk limits
- Audit trail compliance
- Type-level enforcement where possible

**Mechanisms:**
- Policy engine with rule evaluation
- Circuit breakers for anomalies
- Rate limiting on API calls
- Cryptographic signatures on orders

### ADR-010: Agentic Payments Integration
**Decision:** Track costs at task/operation granularity

**Rationale:**
- Attribute infrastructure costs to strategies
- Fair billing for multi-tenant scenarios
- Budget controls per strategy
- ROI tracking

**Tracking Points:**
- API calls (market data, news)
- Compute time (strategy execution)
- Data storage (AgentDB, logs)
- Network egress

## Performance Targets

### Latency Requirements

| Operation | Target Latency | Max Latency |
|-----------|---------------|-------------|
| Market data ingestion | <1ms p99 | <5ms p99.9 |
| Feature extraction | <10ms p95 | <50ms p99 |
| Signal generation | <50ms p95 | <200ms p99 |
| Order placement | <100ms p95 | <500ms p99 |
| End-to-end (data→order) | <200ms p95 | <1s p99 |

### Throughput Requirements

| Metric | Target | Peak |
|--------|--------|------|
| Market data events/sec | 100k | 500k |
| Feature calculations/sec | 10k | 50k |
| Signal evaluations/sec | 1k | 5k |
| Order submissions/sec | 100 | 500 |

### Resource Utilization

| Resource | Baseline | Under Load |
|----------|----------|------------|
| Memory per strategy | <100MB | <500MB |
| CPU per strategy | <5% | <50% |
| Network bandwidth | <10Mbps | <100Mbps |
| Disk I/O | <10MB/s | <100MB/s |

### Efficiency Metrics

- **Memory overhead:** <10% vs Python implementation
- **CPU efficiency:** >90% for compute-bound tasks
- **Zero-copy operations:** >95% of data transfers
- **GC pauses:** N/A (no garbage collection)

## Trade-offs and Constraints

### Development Velocity
**Trade-off:** Rust's strict type system slows initial development
**Mitigation:** Rich type system prevents entire bug classes, reducing debugging time

### Ecosystem Maturity
**Trade-off:** Some Python libraries lack Rust equivalents
**Mitigation:** FFI bridges where necessary, prioritize pure-Rust alternatives

### Binary Size
**Trade-off:** Rust binaries larger than interpreted Python
**Mitigation:** Strip symbols, use dynamic linking for common deps, feature flags

### Learning Curve
**Trade-off:** Rust has steeper learning curve than Python/TypeScript
**Mitigation:** Comprehensive documentation, training sessions, gradual migration

## Migration Strategy

### Phase 1: Core Data Pipeline (4 weeks)
- Market data ingestion
- Polars integration
- Basic feature extraction
- AgentDB connection

### Phase 2: Strategy Framework (6 weeks)
- Strategy trait definitions
- Plugin architecture
- Signal generation framework
- Backtesting engine

### Phase 3: Execution Layer (4 weeks)
- Order management system
- Exchange API integrations
- Settlement and reconciliation
- Error handling and retries

### Phase 4: Node.js Interop (3 weeks)
- napi-rs bindings
- TypeScript definitions
- Integration tests
- Fallback mechanisms

### Phase 5: Advanced Features (8 weeks)
- Sublinear solvers
- Midstreamer integration
- E2B sandbox deployment
- Agentic Flow federation

### Phase 6: Production Hardening (4 weeks)
- Observability setup
- Security audits
- Performance tuning
- Documentation

**Total Estimated Time:** 29 weeks

## Success Criteria

1. **Performance:** 10x latency improvement over Python
2. **Reliability:** 99.9% uptime in production
3. **Maintainability:** <100 lines per module average
4. **Testing:** >90% code coverage
5. **Documentation:** Complete API docs and architecture guides
6. **Interop:** Seamless Node.js integration with <1% overhead

## Next Steps

1. Review and approve architecture
2. Set up Rust workspace structure
3. Define module interfaces (traits)
4. Begin Phase 1 implementation
5. Establish CI/CD pipeline
6. Create integration test framework

---

**Document Revision History:**
- 1.0.0 (2025-11-12): Initial architecture design
