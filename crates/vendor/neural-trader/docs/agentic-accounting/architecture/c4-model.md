# C4 Model - Agentic Accounting System

## Level 1: System Context Diagram

```
┌───────────────────────────────────────────────────────────────────┐
│                        External Systems                           │
│                                                                   │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────────┐  │
│  │  Exchanges  │  │  Blockchain  │  │   Market Data APIs    │  │
│  │ (Coinbase,  │  │   (Etherscan,│  │  (CoinGecko, Alpha    │  │
│  │  Binance)   │  │  Blockchair) │  │      Vantage)         │  │
│  └──────┬──────┘  └──────┬───────┘  └───────────┬────────────┘  │
│         │                │                       │                │
└─────────┼────────────────┼───────────────────────┼────────────────┘
          │                │                       │
          │                │                       │
          ▼                ▼                       ▼
┌───────────────────────────────────────────────────────────────────┐
│                                                                   │
│              Agentic Accounting System                           │
│        Multi-Agent Autonomous Accounting Platform                │
│                                                                   │
│   • Tax calculation (FIFO/LIFO/HIFO)                            │
│   • Tax-loss harvesting & wash-sale detection                   │
│   • Forensic fraud detection (vector search)                    │
│   • Compliance rule engine                                       │
│   • Automated reporting & tax forms                              │
│   • Self-learning via ReasoningBank                             │
│                                                                   │
└───────────────────────────┬───────────────────────────────────────┘
                            │
                            │
                            ▼
┌───────────────────────────────────────────────────────────────────┐
│                       Users & Clients                             │
│                                                                   │
│  ┌──────────────┐  ┌────────────┐  ┌───────────────────────┐   │
│  │   Traders    │  │ Accountants│  │    Claude Code AI     │   │
│  │   (Crypto,   │  │ (Tax Prep, │  │  (MCP Integration)    │   │
│  │   Equities)  │  │  Auditing) │  │                       │   │
│  └──────────────┘  └────────────┘  └───────────────────────┘   │
└───────────────────────────────────────────────────────────────────┘
```

**Key Interactions**:
- **Users** interact via CLI, REST API, GraphQL, or MCP tools
- **System** ingests data from exchanges, blockchains, and market APIs
- **System** exposes results through multiple interfaces

---

## Level 2: Container Diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│                     Agentic Accounting System                        │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                  User Interface Layer                        │   │
│  │  ┌──────────┐  ┌─────────┐  ┌─────────┐  ┌──────────────┐ │   │
│  │  │   CLI    │  │REST API │  │GraphQL  │  │  MCP Server  │ │   │
│  │  │(Commander│  │(Express)│  │(Apollo) │  │  (Tools)     │ │   │
│  │  └────┬─────┘  └────┬────┘  └────┬────┘  └──────┬───────┘ │   │
│  └───────┼─────────────┼────────────┼───────────────┼─────────┘   │
│          │             │            │               │             │
│          └─────────────┴────────────┴───────────────┘             │
│                              ▼                                     │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │              Orchestration Layer                            │  │
│  │  ┌─────────────────────────────────────────────────────┐   │  │
│  │  │          Agentic Flow Coordinator                   │   │  │
│  │  │  • Task routing & agent lifecycle                   │   │  │
│  │  │  • ReasoningBank memory integration                 │   │  │
│  │  │  • Inter-agent communication                        │   │  │
│  │  └─────────────────────────────────────────────────────┘   │  │
│  └──────────────────────────┬──────────────────────────────────┘  │
│                             ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │              Agent Swarm (Multi-Agent Workers)              │  │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐      │  │
│  │  │Ingestion │ │   Tax    │ │ Harvest  │ │Compliance│      │  │
│  │  │  Agent   │ │  Compute │ │  Agent   │ │  Agent   │      │  │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘      │  │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐      │  │
│  │  │ Forensic │ │ Reporting│ │ Learning │ │Verification     │  │
│  │  │  Agent   │ │  Agent   │ │  Agent   │ │   Agent  │      │  │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘      │  │
│  └──────────────────────────┬──────────────────────────────────┘  │
│                             ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │            Rust Core (High-Performance Addon)               │  │
│  │  • Tax calculations (FIFO/LIFO/HIFO)                       │  │
│  │  • Cryptographic operations (Ed25519, SHA-256)             │  │
│  │  • PDF generation & report formatting                      │  │
│  │  • SIMD-optimized technical indicators                     │  │
│  └──────────────────────────┬──────────────────────────────────┘  │
│                             ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │                    Data Storage Layer                       │  │
│  │  ┌──────────────┐  ┌───────────────┐  ┌──────────────────┐│  │
│  │  │   AgentDB    │  │  PostgreSQL   │  │  Lean4 Verifier  ││  │
│  │  │  (Vectors)   │  │  (Relational) │  │    (Proofs)      ││  │
│  │  │• HNSW index  │  │• Transactions │  │• Theorems        ││  │
│  │  │• <100µs query│  │• Audit logs   │  │• Invariants      ││  │
│  │  └──────────────┘  └───────────────┘  └──────────────────┘│  │
│  └─────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Level 3: Component Diagram - Core Package

```
┌────────────────────────────────────────────────────────────────────┐
│        @neural-trader/agentic-accounting-core                      │
│                                                                    │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │                 Transaction Management                        │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌──────────────────┐    │ │
│  │  │  Ingestion  │→ │ Validation  │→ │  Normalization   │    │ │
│  │  └─────────────┘  └─────────────┘  └──────────────────┘    │ │
│  └──────────────────────────┬───────────────────────────────────┘ │
│                             ▼                                      │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │                  Tax Calculation Engine                       │ │
│  │  ┌────────────┐  ┌──────────────┐  ┌────────────────────┐  │ │
│  │  │ Calculator │→ │ Method Router│→ │  Rust Core Calls   │  │ │
│  │  │ Coordinator│  │ (FIFO/LIFO)  │  │  (via napi-rs)     │  │ │
│  │  └────────────┘  └──────────────┘  └────────────────────┘  │ │
│  │  ┌────────────┐  ┌──────────────┐                          │ │
│  │  │ Wash Sale  │  │ Tax-Loss     │                          │ │
│  │  │ Detection  │  │  Harvesting  │                          │ │
│  │  └────────────┘  └──────────────┘                          │ │
│  └──────────────────────────┬───────────────────────────────────┘ │
│                             ▼                                      │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │                Position & Lot Management                      │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌────────────────────┐  │ │
│  │  │   Position  │  │ Lot Tracker │  │ Disposal Handler   │  │ │
│  │  │   Manager   │  │   (FIFO)    │  │                    │  │ │
│  │  └─────────────┘  └─────────────┘  └────────────────────┘  │ │
│  └──────────────────────────┬───────────────────────────────────┘ │
│                             ▼                                      │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │                 Compliance Engine                             │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌────────────────────┐  │ │
│  │  │    Rules    │  │  Validator  │  │  Alert Manager     │  │ │
│  │  │   Engine    │  │             │  │                    │  │ │
│  │  └─────────────┘  └─────────────┘  └────────────────────┘  │ │
│  └──────────────────────────┬───────────────────────────────────┘ │
│                             ▼                                      │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │                Forensic Analysis                              │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌────────────────────┐  │ │
│  │  │    Fraud    │  │  Similarity │  │ Merkle Proof Gen   │  │ │
│  │  │  Detection  │  │   Search    │  │  (Rust Core)       │  │ │
│  │  └─────────────┘  └─────────────┘  └────────────────────┘  │ │
│  └──────────────────────────┬───────────────────────────────────┘ │
│                             ▼                                      │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │              Reporting & Forms Generation                     │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌────────────────────┐  │ │
│  │  │   Report    │  │  Tax Forms  │  │   Formatters       │  │ │
│  │  │  Generator  │  │ (Schedule D)│  │   (JSON/PDF)       │  │ │
│  │  └─────────────┘  └─────────────┘  └────────────────────┘  │ │
│  └──────────────────────────┬───────────────────────────────────┘ │
│                             ▼                                      │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │            Learning & Optimization                            │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌────────────────────┐  │ │
│  │  │ Reasoning   │  │  Feedback   │  │   Strategy         │  │ │
│  │  │    Bank     │  │  Processor  │  │  Optimization      │  │ │
│  │  └─────────────┘  └─────────────┘  └────────────────────┘  │ │
│  └──────────────────────────┬───────────────────────────────────┘ │
│                             ▼                                      │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │                  Database Layer                               │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌────────────────────┐  │ │
│  │  │ PostgreSQL  │  │   AgentDB   │  │    Migrations      │  │ │
│  │  │   Client    │  │   Client    │  │                    │  │ │
│  │  └─────────────┘  └─────────────┘  └────────────────────┘  │ │
│  └──────────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────┘
```

---

## Level 4: Code Structure - Tax Calculation Module

```typescript
// @neural-trader/agentic-accounting-core/src/tax/calculator.ts

import { TaxLot, Disposal, AccountingMethod } from '@neural-trader/agentic-accounting-types';
import * as rustCore from '@neural-trader/agentic-accounting-rust-core';

export class TaxCalculator {
  /**
   * Calculate tax for a disposal using the specified method
   */
  async calculateTax(
    saleTransactionId: string,
    method: AccountingMethod,
    lotIds?: string[]
  ): Promise<TaxCalculationResult> {
    // 1. Fetch sale transaction
    const saleTransaction = await this.db.getTransaction(saleTransactionId);

    // 2. Fetch eligible lots
    const eligibleLots = await this.getLots(saleTransaction.asset, method);

    // 3. Call Rust core for high-performance calculation
    const result = rustCore.calculateTax({
      saleTransaction,
      eligibleLots,
      method,
      specificLotIds: lotIds,
    });

    // 4. Store disposals in database
    await this.db.insertDisposals(result.disposals);

    // 5. Update lot statuses
    await this.updateLotStatuses(result.disposals);

    // 6. Log decision to ReasoningBank
    await this.reasoningBank.logDecision({
      agentType: 'TaxComputeAgent',
      scenario: `Tax calculation for ${saleTransaction.asset} sale`,
      decision: `Used ${method} method`,
      rationale: `Selected ${method} to ${this.getMethodRationale(method)}`,
      outcome: 'SUCCESS',
      metrics: { gain: result.totalGain },
    });

    return result;
  }

  private async getLots(
    asset: string,
    method: AccountingMethod
  ): Promise<TaxLot[]> {
    const lots = await this.db.getOpenLots(asset);

    // Sort based on method
    switch (method) {
      case AccountingMethod.FIFO:
        return lots.sort((a, b) =>
          a.acquiredDate.getTime() - b.acquiredDate.getTime()
        );
      case AccountingMethod.LIFO:
        return lots.sort((a, b) =>
          b.acquiredDate.getTime() - a.acquiredDate.getTime()
        );
      case AccountingMethod.HIFO:
        return lots.sort((a, b) =>
          b.unitCostBasis - a.unitCostBasis
        );
      default:
        return lots;
    }
  }
}
```

---

## Deployment Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Kubernetes Cluster                          │
│                                                                 │
│  ┌────────────────────────────────────────────────────────┐   │
│  │                  Ingress / API Gateway                  │   │
│  │                       (NGINX)                           │   │
│  └──────────────────────┬─────────────────────────────────┘   │
│                         │                                       │
│         ┌───────────────┼───────────────┐                      │
│         ▼               ▼               ▼                      │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐                │
│  │   MCP    │    │   REST   │    │ GraphQL  │                │
│  │  Server  │    │   API    │    │   API    │                │
│  │ (3 pods) │    │ (5 pods) │    │ (3 pods) │                │
│  └────┬─────┘    └────┬─────┘    └────┬─────┘                │
│       │               │               │                        │
│       └───────────────┼───────────────┘                        │
│                       ▼                                         │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │            Coordinator Agent (Queen)                    │  │
│  │                   (2 pods, HA)                          │  │
│  └──────────────────────┬──────────────────────────────────┘  │
│                         ▼                                      │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │              Agent Worker Pods                          │  │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │  │
│  │  │Ingestion │ │   Tax    │ │ Harvest  │ │Compliance│  │  │
│  │  │(3 pods)  │ │ Compute  │ │(2 pods)  │ │(2 pods)  │  │  │
│  │  │          │ │(5 pods)  │ │          │ │          │  │  │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘  │  │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐              │  │
│  │  │ Forensic │ │ Reporting│ │ Learning │              │  │
│  │  │(2 pods)  │ │(3 pods)  │ │(2 pods)  │              │  │
│  │  └──────────┘ └──────────┘ └──────────┘              │  │
│  └──────────────────────┬──────────────────────────────────┘  │
│                         ▼                                      │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │                    Redis Cluster                        │  │
│  │           (Message Queue & Caching)                     │  │
│  │                  (3 nodes)                              │  │
│  └─────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Managed Database Services                      │
│                                                                 │
│  ┌──────────────────┐          ┌─────────────────────────┐    │
│  │   PostgreSQL     │          │      AgentDB            │    │
│  │   (Cloud SQL)    │          │   (Self-hosted)         │    │
│  │   - Primary      │          │   - Vector Storage      │    │
│  │   - 3 Replicas   │          │   - HNSW Indices        │    │
│  └──────────────────┘          └─────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

**Scaling Strategy**:
- Horizontal Pod Autoscaling (HPA) based on CPU/memory
- Redis for distributed task queues
- PostgreSQL read replicas for reporting
- AgentDB sharding for large vector datasets

---

## Technology Mapping

| Layer | Technologies |
|-------|-------------|
| **UI** | TypeScript, Commander.js, Express, Apollo Server |
| **Orchestration** | Agentic Flow, MCP Protocol, BullMQ |
| **Agents** | TypeScript, Node.js 18+ |
| **Computation** | Rust (napi-rs), SIMD, Rayon |
| **Vector DB** | AgentDB (HNSW) |
| **Relational DB** | PostgreSQL 15+, pgvector |
| **Verification** | Lean4 |
| **Caching** | Redis 7+ |
| **Monitoring** | Prometheus, Grafana |
| **Logging** | Winston, ELK Stack |
| **Deployment** | Docker, Kubernetes |
| **CI/CD** | GitHub Actions |
