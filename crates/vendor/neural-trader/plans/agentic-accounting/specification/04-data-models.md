# Agentic Accounting System - Data Models

## Core Data Structures

### 1. Transaction Model

```typescript
interface Transaction {
  id: string;                    // UUID
  timestamp: Date;               // ISO 8601
  type: TransactionType;         // BUY, SELL, TRADE, INCOME, EXPENSE
  asset: string;                 // Symbol/ticker
  quantity: Decimal;             // Precise decimal (not float)
  price: Decimal;                // Price per unit
  fees: Decimal;                 // Transaction fees
  currency: string;              // Base currency (USD, EUR, etc.)
  source: string;                // Exchange/wallet/platform
  sourceId: string;              // External transaction ID
  taxable: boolean;              // Is this a taxable event?
  metadata: Record<string, any>; // Flexible additional data
  embedding?: Vector;            // For semantic search
}

enum TransactionType {
  BUY = 'BUY',           // Purchase/acquisition
  SELL = 'SELL',         // Sale/disposal
  TRADE = 'TRADE',       // Exchange one asset for another
  INCOME = 'INCOME',     // Interest, dividends, rewards
  EXPENSE = 'EXPENSE',   // Fees, payments
  TRANSFER = 'TRANSFER', // Non-taxable movement
}
```

---

### 2. Lot Model (Tax Lot Tracking)

```typescript
interface TaxLot {
  id: string;                  // UUID
  transactionId: string;       // Source transaction
  asset: string;               // Symbol/ticker
  acquiredDate: Date;          // Acquisition timestamp
  quantity: Decimal;           // Remaining quantity
  originalQuantity: Decimal;   // Initial quantity
  costBasis: Decimal;          // Total cost basis
  unitCostBasis: Decimal;      // Cost per unit
  currency: string;            // Currency
  source: string;              // Origin exchange/wallet
  method: AccountingMethod;    // FIFO/LIFO/HIFO/SPECIFIC_ID
  disposals: Disposal[];       // History of sales from this lot
  status: LotStatus;           // OPEN, CLOSED, PARTIAL
}

enum AccountingMethod {
  FIFO = 'FIFO',                 // First-In, First-Out
  LIFO = 'LIFO',                 // Last-In, First-Out
  HIFO = 'HIFO',                 // Highest-In, First-Out
  SPECIFIC_ID = 'SPECIFIC_ID',   // Manual selection
  AVERAGE_COST = 'AVERAGE_COST', // Average cost basis
}

enum LotStatus {
  OPEN = 'OPEN',       // Fully available
  PARTIAL = 'PARTIAL', // Partially sold
  CLOSED = 'CLOSED',   // Fully disposed
}
```

---

### 3. Disposal Model (Sale/Trade Records)

```typescript
interface Disposal {
  id: string;                  // UUID
  lotId: string;               // Source lot
  transactionId: string;       // Sale transaction
  disposalDate: Date;          // Sale timestamp
  quantity: Decimal;           // Quantity sold
  proceeds: Decimal;           // Sale proceeds
  costBasis: Decimal;          // Cost basis of sold quantity
  gain: Decimal;               // Realized gain/loss
  term: CapitalGainTerm;       // SHORT or LONG
  taxYear: number;             // Tax year
  method: AccountingMethod;    // Method used
}

enum CapitalGainTerm {
  SHORT = 'SHORT',  // < 1 year holding period
  LONG = 'LONG',    // >= 1 year holding period
}
```

---

### 4. Position Model (Current Holdings)

```typescript
interface Position {
  asset: string;                   // Symbol/ticker
  totalQuantity: Decimal;          // Total holdings
  totalCostBasis: Decimal;         // Sum of all lot cost bases
  averageCostBasis: Decimal;       // Average cost per unit
  currentPrice: Decimal;           // Market price
  marketValue: Decimal;            // Current market value
  unrealizedGain: Decimal;         // Unrealized gain/loss
  unrealizedGainPercent: number;   // % gain/loss
  lots: TaxLot[];                  // Individual tax lots
  lastUpdated: Date;               // Price update timestamp
}
```

---

### 5. Tax Summary Model

```typescript
interface TaxSummary {
  taxYear: number;                // Tax year
  shortTermGains: Decimal;        // Short-term capital gains
  longTermGains: Decimal;         // Long-term capital gains
  totalGains: Decimal;            // Total realized gains
  totalLosses: Decimal;           // Total realized losses
  netGains: Decimal;              // Net gains/losses
  harvestedLosses: Decimal;       // Losses from tax-loss harvesting
  washSaleAdjustments: Decimal;   // Wash-sale disallowed losses
  disposals: Disposal[];          // All disposals for year
  income: Income[];               // Interest, dividends, etc.
  forms: TaxForm[];               // Generated tax forms
}

interface Income {
  id: string;
  type: IncomeType;
  amount: Decimal;
  asset: string;
  date: Date;
  source: string;
}

enum IncomeType {
  INTEREST = 'INTEREST',
  DIVIDEND = 'DIVIDEND',
  STAKING = 'STAKING',
  MINING = 'MINING',
  AIRDROP = 'AIRDROP',
}
```

---

### 6. Compliance Rule Model

```typescript
interface ComplianceRule {
  id: string;                      // UUID
  name: string;                    // Rule name
  type: RuleType;                  // Category
  condition: RuleCondition;        // Evaluation logic
  action: RuleAction;              // What to do if triggered
  severity: Severity;              // Alert level
  enabled: boolean;                // Active status
  jurisdiction: string[];          // Applicable jurisdictions
  metadata: Record<string, any>;   // Additional config
}

enum RuleType {
  WASH_SALE = 'WASH_SALE',
  TRADING_LIMIT = 'TRADING_LIMIT',
  SEGREGATION_DUTY = 'SEGREGATION_DUTY',
  SUSPICIOUS_ACTIVITY = 'SUSPICIOUS_ACTIVITY',
  POLICY_VIOLATION = 'POLICY_VIOLATION',
}

interface RuleCondition {
  field: string;      // Field to check
  operator: string;   // Comparison operator
  value: any;         // Threshold value
  logic?: Logic;      // AND/OR for compound rules
}

enum RuleAction {
  ALERT = 'ALERT',             // Notify but allow
  BLOCK = 'BLOCK',             // Prevent transaction
  FLAG = 'FLAG',               // Mark for review
  REQUIRE_APPROVAL = 'REQUIRE_APPROVAL', // Human review
}

enum Severity {
  INFO = 'INFO',
  WARNING = 'WARNING',
  ERROR = 'ERROR',
  CRITICAL = 'CRITICAL',
}
```

---

### 7. Audit Trail Model

```typescript
interface AuditEntry {
  id: string;                      // UUID
  timestamp: Date;                 // Event time
  agentId: string;                 // Acting agent
  agentType: string;               // Agent role
  action: string;                  // Action taken
  entityType: string;              // Affected entity type
  entityId: string;                // Affected entity ID
  changes: Change[];               // Before/after values
  reason: string;                  // Decision rationale
  hash: string;                    // SHA-256 hash
  signature: string;               // Ed25519 signature
  previousHash?: string;           // Previous entry hash (chain)
  merkleRoot?: string;             // Merkle tree root
  metadata: Record<string, any>;   // Additional context
}

interface Change {
  field: string;
  before: any;
  after: any;
}
```

---

### 8. Vector Embedding Model

```typescript
interface Embedding {
  id: string;                      // UUID
  entityType: string;              // Transaction, Communication, etc.
  entityId: string;                // Foreign key
  vector: Float32Array;            // Dense embedding (384/768/1536 dims)
  model: string;                   // Embedding model used
  metadata: Record<string, any>;   // Searchable metadata
  timestamp: Date;                 // Creation time
}
```

---

### 9. ReasoningBank Entry Model

```typescript
interface ReasoningEntry {
  id: string;                      // UUID
  agentType: string;               // Agent role
  scenario: string;                // Problem description
  decision: string;                // Action taken
  rationale: string;               // Why this decision
  outcome: Outcome;                // Success/failure
  metrics: Record<string, number>; // Performance metrics
  embedding: Float32Array;         // For similarity search
  timestamp: Date;                 // When occurred
  feedbackScore?: number;          // Human/auto feedback
  references: string[];            // Related entries
}

enum Outcome {
  SUCCESS = 'SUCCESS',
  FAILURE = 'FAILURE',
  PARTIAL = 'PARTIAL',
  PENDING = 'PENDING',
}
```

---

### 10. Formal Verification Model

```typescript
interface VerificationProof {
  id: string;                      // UUID
  theorem: string;                 // Lean4 theorem statement
  proof: string;                   // Lean4 proof code
  invariant: Invariant;            // What's being proven
  verified: boolean;               // Proof validated?
  timestamp: Date;                 // Verification time
  context: Record<string, any>;    // Transaction context
  error?: string;                  // Verification error if failed
}

enum Invariant {
  BALANCE_CONSISTENCY = 'BALANCE_CONSISTENCY',     // Assets = Liabilities + Equity
  NON_NEGATIVE_HOLDINGS = 'NON_NEGATIVE_HOLDINGS', // No negative positions
  SEGREGATION_DUTIES = 'SEGREGATION_DUTIES',       // Role separation
  COST_BASIS_ACCURACY = 'COST_BASIS_ACCURACY',     // Basis calculations correct
  WASH_SALE_COMPLIANCE = 'WASH_SALE_COMPLIANCE',   // No violations
}
```

---

## Database Schema (PostgreSQL)

### Tables

1. **transactions** - All financial transactions
2. **tax_lots** - Individual lot records
3. **disposals** - Sale/trade records
4. **positions** - Current holdings (materialized view)
5. **tax_summaries** - Annual tax data
6. **compliance_rules** - Rule definitions
7. **audit_trail** - Immutable log entries
8. **embeddings** - Vector data (with pgvector)
9. **reasoning_bank** - Agent learning memory
10. **verification_proofs** - Formal proofs

### Indexes

- **B-tree**: Primary keys, foreign keys, timestamps
- **HNSW**: Vector similarity searches (pgvector)
- **GiST**: Multi-column searches
- **Hash**: Equality lookups (asset symbols)

### Constraints

- **Primary Keys**: UUID v4 for all tables
- **Foreign Keys**: Referential integrity
- **Check Constraints**: Non-negative quantities, valid dates
- **Unique Constraints**: External IDs per source

---

## AgentDB Schema

### Collections

1. **transactions** - Transaction embeddings
2. **communications** - Email/message embeddings
3. **patterns** - Fraud pattern signatures
4. **strategies** - Successful decision embeddings

### Vector Dimensions

- **Small**: 384 dims (all-MiniLM-L6-v2)
- **Medium**: 768 dims (BERT-base)
- **Large**: 1536 dims (OpenAI ada-002)

### Distance Metrics

- **Cosine**: Default for semantic similarity
- **Euclidean**: For magnitude-sensitive comparisons
- **Dot Product**: For pre-normalized vectors

---

## Data Flow

1. **Ingestion**: External → Transaction → AgentDB embedding
2. **Processing**: Transaction → Tax Lot → Disposal
3. **Compliance**: Transaction → Rule Check → Audit Trail
4. **Learning**: Audit Trail → ReasoningBank → AgentDB
5. **Reporting**: Tax Lots + Disposals → Tax Summary → Forms
6. **Forensics**: AgentDB Query → Similar Transactions → Alert

---

## Serialization & APIs

- **Internal**: MessagePack for inter-agent communication
- **External**: JSON for REST APIs
- **Storage**: Protocol Buffers for efficient disk storage
- **Streaming**: NDJSON for real-time feeds

## Data Retention

- **Transactions**: Indefinite (immutable)
- **Audit Trail**: Indefinite (compliance requirement)
- **Embeddings**: 7 years (regulatory requirement)
- **ReasoningBank**: Continuous pruning of low-value entries
- **Temp Data**: 30 days (debugging, replays)
