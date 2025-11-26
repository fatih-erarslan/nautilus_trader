# Database Schema Reference

## Table Definitions

### 1. transactions

Financial transaction records (buys, sells, trades, income).

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | UUID | PRIMARY KEY | Unique transaction ID |
| timestamp | TIMESTAMPTZ | NOT NULL | Transaction time |
| type | transaction_type | NOT NULL | BUY, SELL, TRADE, INCOME, EXPENSE, TRANSFER |
| asset | VARCHAR(50) | NOT NULL | Asset symbol (BTC, ETH, etc.) |
| quantity | DECIMAL(30,18) | NOT NULL, CHECK > 0 | Quantity traded |
| price | DECIMAL(30,18) | NOT NULL, CHECK >= 0 | Price per unit |
| fees | DECIMAL(30,18) | DEFAULT 0, CHECK >= 0 | Transaction fees |
| currency | VARCHAR(10) | DEFAULT 'USD' | Base currency |
| source | VARCHAR(100) | NOT NULL | Exchange/wallet name |
| source_id | VARCHAR(255) | | External transaction ID |
| taxable | BOOLEAN | DEFAULT true | Is taxable event? |
| metadata | JSONB | DEFAULT '{}' | Additional data |
| created_at | TIMESTAMPTZ | DEFAULT NOW() | Record creation time |
| updated_at | TIMESTAMPTZ | DEFAULT NOW() | Last update time |

**Indexes:**
- `idx_transactions_timestamp` - B-tree on timestamp
- `idx_transactions_asset` - B-tree on asset
- `idx_transactions_type` - B-tree on type
- `idx_transactions_asset_timestamp` - Composite (asset, timestamp)

---

### 2. tax_lots

Individual tax lot tracking for cost basis calculations.

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | UUID | PRIMARY KEY | Unique lot ID |
| transaction_id | UUID | FK → transactions | Source transaction |
| asset | VARCHAR(50) | NOT NULL | Asset symbol |
| acquired_date | TIMESTAMPTZ | NOT NULL | Acquisition date |
| quantity | DECIMAL(30,18) | CHECK >= 0 | Remaining quantity |
| original_quantity | DECIMAL(30,18) | CHECK > 0 | Initial quantity |
| cost_basis | DECIMAL(30,18) | CHECK >= 0 | Total cost basis |
| unit_cost_basis | DECIMAL(30,18) | CHECK >= 0 | Cost per unit |
| currency | VARCHAR(10) | DEFAULT 'USD' | Currency |
| source | VARCHAR(100) | NOT NULL | Origin exchange |
| method | accounting_method | DEFAULT 'FIFO' | FIFO, LIFO, HIFO, SPECIFIC_ID, AVERAGE_COST |
| status | lot_status | DEFAULT 'OPEN' | OPEN, PARTIAL, CLOSED |
| created_at | TIMESTAMPTZ | DEFAULT NOW() | Record creation |
| updated_at | TIMESTAMPTZ | DEFAULT NOW() | Last update |

**Indexes:**
- `idx_tax_lots_transaction_id` - FK lookup
- `idx_tax_lots_asset_acquired` - FIFO queries (asset, acquired_date ASC)
- `idx_tax_lots_asset_cost` - HIFO queries (asset, unit_cost_basis DESC)

---

### 3. disposals

Sale/trade records with realized gains and losses.

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | UUID | PRIMARY KEY | Unique disposal ID |
| lot_id | UUID | FK → tax_lots | Source lot |
| transaction_id | UUID | FK → transactions | Sale transaction |
| disposal_date | TIMESTAMPTZ | NOT NULL | Sale timestamp |
| quantity | DECIMAL(30,18) | CHECK > 0 | Quantity sold |
| proceeds | DECIMAL(30,18) | CHECK >= 0 | Sale proceeds |
| cost_basis | DECIMAL(30,18) | CHECK >= 0 | Cost basis |
| gain | DECIMAL(30,18) | | Realized gain/loss |
| term | capital_gain_term | | SHORT or LONG |
| tax_year | INTEGER | CHECK 2000-2100 | Tax year |
| method | accounting_method | | Accounting method |
| wash_sale_disallowed | DECIMAL(30,18) | DEFAULT 0 | Disallowed loss |
| wash_sale_deferred_to | UUID | FK → tax_lots | Deferred lot |
| created_at | TIMESTAMPTZ | DEFAULT NOW() | Record creation |
| updated_at | TIMESTAMPTZ | DEFAULT NOW() | Last update |

**Indexes:**
- `idx_disposals_year_term` - Tax reporting (tax_year, term)
- `idx_disposals_disposal_date` - Temporal queries

---

### 4. positions (Materialized View)

Current holdings aggregated from tax_lots.

| Column | Type | Description |
|--------|------|-------------|
| asset | VARCHAR(50) | Asset symbol (PK) |
| total_quantity | DECIMAL(30,18) | Total holdings |
| total_cost_basis | DECIMAL(30,18) | Sum of cost bases |
| average_cost_basis | DECIMAL(30,18) | Average cost per unit |
| current_price | DECIMAL(30,18) | Market price |
| market_value | DECIMAL(30,18) | Current value |
| unrealized_gain | DECIMAL(30,18) | Unrealized P&L |
| unrealized_gain_percent | DECIMAL(30,2) | P&L percentage |
| last_updated | TIMESTAMPTZ | Last refresh |

**Refresh:** `REFRESH MATERIALIZED VIEW CONCURRENTLY positions`

---

### 5. tax_summaries

Annual tax summaries with aggregated gains/losses.

| Column | Type | Description |
|--------|------|-------------|
| id | UUID | PRIMARY KEY |
| tax_year | INTEGER | Tax year (UNIQUE) |
| short_term_gains | DECIMAL(30,18) | Short-term gains |
| long_term_gains | DECIMAL(30,18) | Long-term gains |
| total_gains | DECIMAL(30,18) | Total gains |
| total_losses | DECIMAL(30,18) | Total losses |
| net_gains | DECIMAL(30,18) | Net gains/losses |
| harvested_losses | DECIMAL(30,18) | Harvested losses |
| wash_sale_adjustments | DECIMAL(30,18) | Wash sale adjustments |
| total_income | DECIMAL(30,18) | Total income |
| created_at | TIMESTAMPTZ | Record creation |
| updated_at | TIMESTAMPTZ | Last update |

**Function:** `compute_tax_summary(year INTEGER)` - Auto-compute summary

---

### 6. compliance_rules

Configurable compliance rules for validation.

| Column | Type | Description |
|--------|------|-------------|
| id | UUID | PRIMARY KEY |
| name | VARCHAR(255) | Rule name (UNIQUE) |
| description | TEXT | Rule description |
| type | rule_type | WASH_SALE, TRADING_LIMIT, etc. |
| condition | JSONB | Evaluation logic |
| action | rule_action | ALERT, BLOCK, FLAG, REQUIRE_APPROVAL |
| severity | severity | INFO, WARNING, ERROR, CRITICAL |
| enabled | BOOLEAN | Active status |
| jurisdictions | TEXT[] | Applicable jurisdictions |
| metadata | JSONB | Additional config |
| created_at | TIMESTAMPTZ | Record creation |
| updated_at | TIMESTAMPTZ | Last update |

---

### 7. audit_trail

Immutable audit log with cryptographic verification.

| Column | Type | Description |
|--------|------|-------------|
| id | UUID | PRIMARY KEY |
| timestamp | TIMESTAMPTZ | Event time |
| agent_id | VARCHAR(255) | Acting agent |
| agent_type | VARCHAR(100) | Agent role |
| action | VARCHAR(255) | Action taken |
| entity_type | VARCHAR(100) | Affected entity |
| entity_id | UUID | Entity ID |
| changes | JSONB | Before/after values |
| reason | TEXT | Decision rationale |
| hash | VARCHAR(64) | SHA-256 hash |
| signature | VARCHAR(128) | Ed25519 signature |
| previous_hash | VARCHAR(64) | Previous entry hash |
| merkle_root | VARCHAR(64) | Merkle tree root |
| metadata | JSONB | Additional context |
| created_at | TIMESTAMPTZ | Record creation |

**Immutable:** Updates and deletes are blocked via rules

**Function:** `insert_audit_entry()` - Add with automatic hash chaining

---

### 8. embeddings

Vector embeddings for semantic search (pgvector).

| Column | Type | Description |
|--------|------|-------------|
| id | UUID | PRIMARY KEY |
| entity_type | VARCHAR(100) | Entity type |
| entity_id | UUID | Entity ID |
| vector | vector(768) | 768-dim embedding |
| model | VARCHAR(100) | Model name |
| metadata | JSONB | Searchable metadata |
| timestamp | TIMESTAMPTZ | Creation time |

**Index:** HNSW on vector for fast cosine similarity search

**Function:** `find_similar_transactions()` - Find similar transactions

---

### 9. reasoning_bank

Agent learning memory with decision tracking.

| Column | Type | Description |
|--------|------|-------------|
| id | UUID | PRIMARY KEY |
| agent_type | VARCHAR(100) | Agent role |
| scenario | TEXT | Problem description |
| decision | TEXT | Action taken |
| rationale | TEXT | Why this decision |
| outcome | outcome | SUCCESS, FAILURE, PARTIAL, PENDING |
| metrics | JSONB | Performance metrics |
| embedding | vector(768) | Decision embedding |
| timestamp | TIMESTAMPTZ | When occurred |
| feedback_score | DECIMAL(3,2) | Human/auto feedback (0-1) |
| references | UUID[] | Related entries |
| metadata | JSONB | Additional context |
| created_at | TIMESTAMPTZ | Record creation |

**Function:** `find_similar_decisions()` - Retrieve similar past decisions

---

### 10. verification_proofs

Formal verification proofs using Lean4.

| Column | Type | Description |
|--------|------|-------------|
| id | UUID | PRIMARY KEY |
| theorem | TEXT | Lean4 theorem statement |
| proof | TEXT | Lean4 proof code |
| invariant | invariant | What's being proven |
| verified | BOOLEAN | Proof validated? |
| timestamp | TIMESTAMPTZ | Verification time |
| context | JSONB | Transaction context |
| error | TEXT | Error if failed |
| verification_time_ms | INTEGER | Time taken (ms) |
| lean_version | VARCHAR(50) | Lean version |
| created_at | TIMESTAMPTZ | Record creation |

**Function:** `record_verification()` - Record verification attempt

---

## Enums

### transaction_type
- `BUY` - Purchase/acquisition
- `SELL` - Sale/disposal
- `TRADE` - Exchange one asset for another
- `INCOME` - Interest, dividends, rewards
- `EXPENSE` - Fees, payments
- `TRANSFER` - Non-taxable movement

### accounting_method
- `FIFO` - First-In, First-Out
- `LIFO` - Last-In, First-Out
- `HIFO` - Highest-In, First-Out
- `SPECIFIC_ID` - Manual selection
- `AVERAGE_COST` - Average cost basis

### lot_status
- `OPEN` - Fully available
- `PARTIAL` - Partially sold
- `CLOSED` - Fully disposed

### capital_gain_term
- `SHORT` - < 1 year holding period
- `LONG` - >= 1 year holding period

### outcome
- `SUCCESS` - Successful outcome
- `FAILURE` - Failed outcome
- `PARTIAL` - Partially successful
- `PENDING` - Still in progress

### invariant
- `BALANCE_CONSISTENCY` - Assets = Liabilities + Equity
- `NON_NEGATIVE_HOLDINGS` - No negative positions
- `SEGREGATION_DUTIES` - Role separation
- `COST_BASIS_ACCURACY` - Basis calculations correct
- `WASH_SALE_COMPLIANCE` - No violations

---

## Key Functions

### Tax Calculations
- `calculate_holding_period(acquired_date, disposal_date)` → capital_gain_term
- `compute_tax_summary(year)` → UUID

### Compliance
- `evaluate_compliance_rules(transaction_data)` → TABLE

### Audit
- `insert_audit_entry(...)` → UUID
- `verify_audit_chain()` → TABLE
- `compute_audit_hash(...)` → VARCHAR

### Vector Search
- `find_similar_entities(query_vector, ...)` → TABLE
- `find_similar_transactions(transaction_id, ...)` → TABLE
- `find_similar_decisions(query_scenario, ...)` → TABLE

### Views
- `refresh_positions_with_prices(price_updates)` → void
- `refresh_all_materialized_views()` → void

---

## Performance Tips

1. **Use covering indexes** for frequently accessed columns
2. **Refresh materialized views** periodically (hourly/daily)
3. **Vacuum and analyze** regularly for query performance
4. **Partition large tables** by tax year in production
5. **Use connection pooling** (max 20 connections)
6. **Monitor slow queries** with pg_stat_statements
7. **Use EXPLAIN ANALYZE** to optimize queries
