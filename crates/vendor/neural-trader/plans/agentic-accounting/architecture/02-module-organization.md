# Agentic Accounting System - Module Organization

## Package Structure

```
@neural-trader/agentic-accounting/
├── packages/
│   ├── core/                          # Core TypeScript library
│   ├── rust-core/                     # Rust high-performance addon
│   ├── agents/                        # Agent implementations
│   ├── mcp-server/                    # MCP tool server
│   ├── api/                           # REST/GraphQL APIs
│   ├── cli/                           # Command-line interface
│   └── types/                         # Shared TypeScript types
├── apps/
│   ├── web/                           # Web dashboard (optional)
│   └── desktop/                       # Electron app (optional)
├── tests/
│   ├── unit/                          # Unit tests
│   ├── integration/                   # Integration tests
│   └── e2e/                           # End-to-end tests
├── docs/                              # Documentation
└── examples/                          # Example usage
```

---

## Core Package (@neural-trader/agentic-accounting-core)

```
packages/core/
├── src/
│   ├── index.ts                       # Main entry point
│   ├── transactions/
│   │   ├── index.ts
│   │   ├── ingestion.ts               # Transaction ingestion
│   │   ├── validation.ts              # Data validation
│   │   └── normalization.ts           # Data normalization
│   ├── tax/
│   │   ├── index.ts
│   │   ├── calculator.ts              # Tax calculation coordinator
│   │   ├── methods.ts                 # Accounting method wrappers
│   │   ├── wash-sale.ts               # Wash sale detection
│   │   └── harvesting.ts              # Tax-loss harvesting
│   ├── positions/
│   │   ├── index.ts
│   │   ├── manager.ts                 # Position management
│   │   ├── lots.ts                    # Lot tracking
│   │   └── disposals.ts               # Disposal handling
│   ├── compliance/
│   │   ├── index.ts
│   │   ├── rules.ts                   # Compliance rules engine
│   │   ├── validator.ts               # Transaction validation
│   │   └── alerts.ts                  # Alert management
│   ├── forensic/
│   │   ├── index.ts
│   │   ├── fraud-detection.ts         # Fraud detection
│   │   ├── similarity.ts              # Vector similarity
│   │   └── merkle.ts                  # Merkle proof utilities
│   ├── reporting/
│   │   ├── index.ts
│   │   ├── generator.ts               # Report generation
│   │   ├── templates/                 # Report templates
│   │   │   ├── schedule-d.ts
│   │   │   ├── form-8949.ts
│   │   │   └── custom.ts
│   │   └── formatters.ts              # Output formatters
│   ├── learning/
│   │   ├── index.ts
│   │   ├── reasoning-bank.ts          # ReasoningBank interface
│   │   ├── feedback.ts                # Feedback processing
│   │   └── optimization.ts            # Strategy optimization
│   ├── database/
│   │   ├── index.ts
│   │   ├── postgresql.ts              # PostgreSQL client
│   │   ├── agentdb.ts                 # AgentDB client
│   │   ├── migrations/                # Database migrations
│   │   └── seeds/                     # Test data
│   ├── integrations/
│   │   ├── index.ts
│   │   ├── exchanges/
│   │   │   ├── coinbase.ts
│   │   │   ├── binance.ts
│   │   │   └── kraken.ts
│   │   ├── blockchain/
│   │   │   ├── etherscan.ts
│   │   │   └── blockchair.ts
│   │   └── market-data/
│   │       ├── coingecko.ts
│   │       └── alpha-vantage.ts
│   ├── utils/
│   │   ├── decimal.ts                 # Precise decimal math
│   │   ├── date.ts                    # Date utilities
│   │   ├── crypto.ts                  # Cryptographic utilities
│   │   └── logger.ts                  # Logging
│   └── config/
│       ├── index.ts
│       ├── database.ts
│       ├── jurisdictions.ts           # Jurisdiction configs
│       └── constants.ts
├── package.json
├── tsconfig.json
└── README.md
```

**Key Exports**:
```typescript
export {
  // Transaction management
  TransactionManager,
  TransactionValidator,

  // Tax calculations
  TaxCalculator,
  AccountingMethod,
  WashSaleDetector,
  TaxLossHarvester,

  // Position management
  PositionManager,
  LotTracker,
  DisposalHandler,

  // Compliance
  ComplianceEngine,
  RuleValidator,
  AlertManager,

  // Forensics
  FraudDetector,
  SimilaritySearch,
  MerkleProofGenerator,

  // Reporting
  ReportGenerator,
  TaxFormGenerator,

  // Learning
  ReasoningBank,
  FeedbackProcessor,

  // Types
  Transaction,
  TaxLot,
  Disposal,
  Position,
  ComplianceRule,
  AuditEntry,
};
```

---

## Rust Core Package (agentic-accounting-rust-core)

```
packages/rust-core/
├── src/
│   ├── lib.rs                         # napi-rs entry point
│   ├── tax/
│   │   ├── mod.rs
│   │   ├── fifo.rs                    # FIFO implementation
│   │   ├── lifo.rs                    # LIFO implementation
│   │   ├── hifo.rs                    # HIFO implementation
│   │   ├── specific_id.rs             # Specific ID implementation
│   │   ├── average_cost.rs            # Average cost implementation
│   │   ├── wash_sale.rs               # Wash sale detection
│   │   └── calculator.rs              # Main tax calculator
│   ├── forensic/
│   │   ├── mod.rs
│   │   ├── merkle.rs                  # Merkle tree operations
│   │   ├── signatures.rs              # Ed25519 signing
│   │   ├── hashing.rs                 # SHA-256 hashing
│   │   └── similarity.rs              # Fast similarity hashing
│   ├── reports/
│   │   ├── mod.rs
│   │   ├── pdf.rs                     # PDF generation
│   │   ├── forms.rs                   # Tax form rendering
│   │   └── aggregation.rs             # Data aggregation
│   ├── performance/
│   │   ├── mod.rs
│   │   ├── indicators.rs              # Technical indicators (SIMD)
│   │   ├── optimization.rs            # Portfolio optimization
│   │   └── risk.rs                    # VaR/CVaR calculations
│   ├── types.rs                       # Shared Rust types
│   ├── utils.rs                       # Utility functions
│   └── error.rs                       # Error types
├── benches/                           # Benchmarks
│   ├── tax_calculations.rs
│   └── vector_operations.rs
├── tests/                             # Rust tests
│   ├── integration.rs
│   └── unit/
├── Cargo.toml
├── build.rs
└── README.md
```

**Cargo.toml Dependencies**:
```toml
[dependencies]
napi = "2"
napi-derive = "2"
rust_decimal = "1"                     # Precise decimal math
chrono = "0.4"                         # Date/time
ed25519-dalek = "2"                    # Signatures
sha2 = "0.10"                          # Hashing
rayon = "1.7"                          # Parallelism
serde = { version = "1", features = ["derive"] }
serde_json = "1"

[profile.release]
lto = true                             # Link-time optimization
codegen-units = 1                      # Better optimization
opt-level = 3                          # Maximum optimization
```

---

## Agents Package (@neural-trader/agentic-accounting-agents)

```
packages/agents/
├── src/
│   ├── index.ts
│   ├── base/
│   │   ├── agent.ts                   # Base agent class
│   │   ├── coordinator.ts             # Coordinator agent
│   │   └── lifecycle.ts               # Agent lifecycle management
│   ├── ingestion/
│   │   ├── index.ts
│   │   └── ingestion-agent.ts         # Transaction ingestion agent
│   ├── tax-compute/
│   │   ├── index.ts
│   │   └── tax-compute-agent.ts       # Tax computation agent
│   ├── harvesting/
│   │   ├── index.ts
│   │   └── harvest-agent.ts           # Tax-loss harvesting agent
│   ├── compliance/
│   │   ├── index.ts
│   │   └── compliance-agent.ts        # Compliance agent
│   ├── forensic/
│   │   ├── index.ts
│   │   └── forensic-agent.ts          # Forensic analysis agent
│   ├── reporting/
│   │   ├── index.ts
│   │   └── reporting-agent.ts         # Reporting agent
│   ├── learning/
│   │   ├── index.ts
│   │   └── learning-agent.ts          # Learning & optimization agent
│   ├── verification/
│   │   ├── index.ts
│   │   └── verification-agent.ts      # Formal verification agent
│   └── communication/
│       ├── index.ts
│       ├── message-queue.ts           # Message queue client
│       └── protocols.ts               # Communication protocols
├── package.json
└── README.md
```

**Base Agent Interface**:
```typescript
export abstract class BaseAgent {
  abstract agentId: string;
  abstract agentType: string;

  abstract async initialize(): Promise<void>;
  abstract async execute(task: Task): Promise<Result>;
  abstract async shutdown(): Promise<void>;

  protected async logDecision(
    scenario: string,
    decision: string,
    rationale: string,
    outcome: Outcome
  ): Promise<void>;

  protected async retrieveSimilarDecisions(
    scenario: string,
    topK: number
  ): Promise<ReasoningEntry[]>;

  protected async sendMessage(
    targetAgent: string,
    message: Message
  ): Promise<void>;

  protected async subscribeToEvents(
    eventType: string,
    handler: EventHandler
  ): Promise<void>;
}
```

---

## MCP Server Package (@neural-trader/agentic-accounting-mcp)

```
packages/mcp-server/
├── src/
│   ├── index.ts                       # MCP server entry point
│   ├── server.ts                      # Server implementation
│   ├── tools/
│   │   ├── index.ts
│   │   ├── transaction-tools.ts       # accounting_add_transaction, etc.
│   │   ├── tax-tools.ts               # accounting_calculate_tax, etc.
│   │   ├── compliance-tools.ts        # compliance_check_trade, etc.
│   │   ├── forensic-tools.ts          # forensic_find_similar, etc.
│   │   ├── reporting-tools.ts         # accounting_generate_report, etc.
│   │   └── learning-tools.ts          # learning_feedback, etc.
│   ├── prompts/
│   │   ├── index.ts
│   │   └── accounting-prompts.ts      # Prompt templates
│   └── config.ts
├── package.json
└── README.md
```

**Tool Definitions**:
```typescript
export const tools: MCPTool[] = [
  {
    name: "accounting_add_transaction",
    description: "Add a new transaction to the accounting system",
    inputSchema: {
      type: "object",
      properties: {
        type: { type: "string", enum: ["BUY", "SELL", "TRADE", "INCOME"] },
        asset: { type: "string" },
        quantity: { type: "number" },
        price: { type: "number" },
        timestamp: { type: "string", format: "date-time" },
        source: { type: "string" },
        fees: { type: "number" },
      },
      required: ["type", "asset", "quantity", "price", "timestamp"],
    },
  },
  {
    name: "accounting_calculate_tax",
    description: "Calculate tax for a disposal using specified method",
    inputSchema: {
      type: "object",
      properties: {
        transactionId: { type: "string" },
        method: { type: "string", enum: ["FIFO", "LIFO", "HIFO", "SPECIFIC_ID"] },
        lotIds: { type: "array", items: { type: "string" } },
      },
      required: ["transactionId", "method"],
    },
  },
  // ... more tools
];
```

---

## API Package (@neural-trader/agentic-accounting-api)

```
packages/api/
├── src/
│   ├── index.ts
│   ├── rest/
│   │   ├── index.ts
│   │   ├── server.ts                  # Express server
│   │   ├── routes/
│   │   │   ├── transactions.ts
│   │   │   ├── positions.ts
│   │   │   ├── tax.ts
│   │   │   ├── compliance.ts
│   │   │   ├── forensic.ts
│   │   │   └── reports.ts
│   │   ├── middleware/
│   │   │   ├── auth.ts
│   │   │   ├── validation.ts
│   │   │   ├── error-handler.ts
│   │   │   └── rate-limit.ts
│   │   └── controllers/
│   │       ├── transaction-controller.ts
│   │       └── ...
│   ├── graphql/
│   │   ├── index.ts
│   │   ├── schema.ts                  # GraphQL schema
│   │   ├── resolvers/
│   │   │   ├── transaction-resolvers.ts
│   │   │   ├── position-resolvers.ts
│   │   │   └── ...
│   │   └── dataloaders/
│   │       └── batch-loaders.ts
│   └── config/
│       ├── index.ts
│       └── cors.ts
├── package.json
└── README.md
```

---

## CLI Package (@neural-trader/agentic-accounting-cli)

```
packages/cli/
├── src/
│   ├── index.ts
│   ├── commands/
│   │   ├── import.ts                  # Import transactions
│   │   ├── calculate.ts               # Calculate taxes
│   │   ├── harvest.ts                 # Find harvest opportunities
│   │   ├── report.ts                  # Generate reports
│   │   ├── audit.ts                   # View audit trail
│   │   └── agent.ts                   # Manage agents
│   ├── ui/
│   │   ├── prompts.ts                 # Interactive prompts
│   │   └── formatting.ts              # Output formatting
│   └── config.ts
├── bin/
│   └── accounting-cli.js              # CLI entry point
├── package.json
└── README.md
```

---

## Types Package (@neural-trader/agentic-accounting-types)

```
packages/types/
├── src/
│   ├── index.ts
│   ├── transaction.ts
│   ├── tax-lot.ts
│   ├── disposal.ts
│   ├── position.ts
│   ├── compliance.ts
│   ├── audit.ts
│   ├── agent.ts
│   └── api.ts
├── package.json
└── README.md
```

---

## Testing Structure

```
tests/
├── unit/
│   ├── tax-calculations.test.ts
│   ├── wash-sale.test.ts
│   ├── fraud-detection.test.ts
│   └── ...
├── integration/
│   ├── agent-coordination.test.ts
│   ├── database-integration.test.ts
│   ├── api-integration.test.ts
│   └── ...
└── e2e/
    ├── full-workflow.test.ts
    ├── tax-year-end.test.ts
    └── compliance-audit.test.ts
```

---

## Build & Distribution

**Monorepo Management**: Nx or Turborepo

**Package Publishing**:
- All packages published to npm under `@neural-trader` scope
- Semantic versioning
- Automated releases via CI/CD

**Rust Addon Distribution**:
- Precompiled binaries for common platforms (via `@napi-rs/cli`)
- Fallback to source compilation if needed
- Separate npm packages per platform (optional)

**Docker Images**:
- `neural-trader/agentic-accounting:latest` - Full system
- `neural-trader/agentic-accounting-agents:latest` - Agent workers only
- `neural-trader/agentic-accounting-api:latest` - API server only
