# Database Architecture

## Overview

The agentic accounting system uses a hybrid database architecture combining PostgreSQL for relational data and AgentDB for vector similarity search.

## PostgreSQL Schema

### Core Tables

1. **transactions** - All financial transactions
2. **tax_lots** - Individual lot records for cost basis tracking
3. **disposals** - Sale/trade records with realized gains/losses
4. **positions** - Current holdings (materialized view)
5. **tax_summaries** - Annual tax summaries
6. **compliance_rules** - Configurable compliance rules
7. **audit_trail** - Immutable audit log with cryptographic verification
8. **embeddings** - Vector embeddings for semantic search (pgvector)
9. **reasoning_bank** - Agent learning memory
10. **verification_proofs** - Formal verification proofs (Lean4)

### Entity Relationship Diagram

```
┌─────────────────┐
│  transactions   │
│─────────────────│
│ id (PK)         │◄──┐
│ timestamp       │   │
│ type            │   │
│ asset           │   │
│ quantity        │   │
│ price           │   │
│ fees            │   │
│ source          │   │
└─────────────────┘   │
                      │
                      │ FK: transaction_id
                      │
┌─────────────────┐   │
│   tax_lots      │◄──┤
│─────────────────│   │
│ id (PK)         │◄──┼──┐
│ transaction_id  │───┘  │
│ asset           │      │
│ acquired_date   │      │
│ quantity        │      │
│ cost_basis      │      │
│ status          │      │
└─────────────────┘      │
                         │ FK: lot_id
                         │
┌─────────────────┐      │
│   disposals     │      │
│─────────────────│      │
│ id (PK)         │      │
│ lot_id          │──────┘
│ transaction_id  │──────┐
│ disposal_date   │      │
│ proceeds        │      │
│ gain            │      │
│ term            │      │
│ tax_year        │      │
└─────────────────┘      │
                         │
                         │
┌─────────────────┐      │
│   positions     │      │
│─────────────────│      │
│ asset (PK)      │      │
│ total_quantity  │      │
│ total_cost_basis│      │
│ market_value    │      │
│ unrealized_gain │      │
└─────────────────┘      │
                         │
                         │
┌─────────────────┐      │
│ tax_summaries   │      │
│─────────────────│      │
│ id (PK)         │      │
│ tax_year        │      │
│ short_term_gains│      │
│ long_term_gains │      │
│ net_gains       │      │
└─────────────────┘      │
                         │
                         │
┌─────────────────┐      │
│ audit_trail     │      │
│─────────────────│      │
│ id (PK)         │      │
│ timestamp       │      │
│ agent_id        │      │
│ entity_id       │──────┘
│ action          │
│ hash            │
│ signature       │
└─────────────────┘

┌─────────────────┐      ┌─────────────────┐
│  embeddings     │      │ reasoning_bank  │
│─────────────────│      │─────────────────│
│ id (PK)         │      │ id (PK)         │
│ entity_type     │      │ agent_type      │
│ entity_id       │      │ scenario        │
│ vector (768d)   │      │ decision        │
│ metadata        │      │ outcome         │
└─────────────────┘      │ embedding (768d)│
                         └─────────────────┘
```

## Indexes

### B-tree Indexes
- Primary keys (all tables)
- Foreign keys (relationships)
- Timestamp columns (temporal queries)
- Asset symbols (filtering)
- Status fields (active record filtering)

### HNSW Indexes (pgvector)
- `embeddings.vector` - Fast cosine similarity search
- `reasoning_bank.embedding` - Decision similarity search

### Composite Indexes
- `(asset, timestamp)` - Transaction history queries
- `(asset, acquired_date)` - FIFO lot selection
- `(asset, unit_cost_basis)` - HIFO lot selection
- `(tax_year, term)` - Tax reporting queries
- `(entity_type, entity_id, timestamp)` - Audit trail queries

## Data Types

### Numeric Precision
- **Quantities**: `DECIMAL(30, 18)` - Up to 18 decimal places for crypto precision
- **Prices**: `DECIMAL(30, 18)` - Same precision for accurate calculations
- **Gains/Losses**: `DECIMAL(30, 18)` - Precise gain calculations

### Vectors
- **Embedding dimensions**: 768 (BERT-base, all-MiniLM-L6-v2)
- **Distance metric**: Cosine similarity
- **Index type**: HNSW (Hierarchical Navigable Small World)

## Constraints

### Check Constraints
- Non-negative quantities and prices
- Valid date ranges
- Lot quantity ≤ original quantity
- Feedback scores between 0 and 1
- Tax years between 2000 and 2100

### Unique Constraints
- External transaction IDs per source
- One embedding per entity
- One tax summary per year
- Unique rule names

### Foreign Keys
- Cascading deletes for dependent records
- Referential integrity for all relationships

## Functions

### Tax Calculations
- `calculate_holding_period()` - Determine short/long term
- `compute_tax_summary()` - Aggregate annual tax data

### Compliance
- `evaluate_compliance_rules()` - Check transaction against rules

### Audit
- `insert_audit_entry()` - Add entry with automatic hashing
- `verify_audit_chain()` - Verify blockchain-style chain integrity
- `compute_audit_hash()` - SHA-256 hash computation

### Vector Search
- `find_similar_entities()` - Generic similarity search
- `find_similar_transactions()` - Transaction-specific similarity
- `find_similar_decisions()` - Reasoning pattern matching

### Views
- `refresh_positions_with_prices()` - Update positions with market data
- `refresh_all_materialized_views()` - Batch refresh

## Materialized Views

### positions
- Aggregates current holdings from tax_lots
- Updated via `REFRESH MATERIALIZED VIEW CONCURRENTLY`
- Includes unrealized gains/losses

### hot_assets
- High-volume trading assets (last 30 days)
- Useful for performance optimization

### tax_year_summary
- Annual tax summary aggregated from disposals
- Quick access to tax reporting data

## Performance Optimization

### Query Performance
- **HNSW indexes**: O(log n) vector search
- **Partial indexes**: Filter only relevant records
- **Covering indexes**: Include frequently accessed columns
- **Connection pooling**: Max 20 connections

### Data Retention
- **Transactions**: Indefinite (immutable)
- **Audit trail**: Indefinite (compliance)
- **Embeddings**: 7 years (regulatory)
- **Reasoning bank**: Pruned by performance

### Caching Strategy
- Materialized views for aggregated data
- AgentDB in-memory cache for vectors
- Connection pooling for PostgreSQL

## Security

### Authentication
- PostgreSQL password authentication
- Connection pooling with secure credentials
- Environment variable configuration

### Encryption
- TLS for connections (configurable)
- SHA-256 hashing for audit trail
- Ed25519 signatures for verification

### Immutability
- Audit trail: No updates or deletes allowed
- Blockchain-style hash chaining
- Cryptographic verification

## Backup & Recovery

### Continuous Archiving
- PostgreSQL WAL (Write-Ahead Logging)
- Point-in-time recovery (PITR)

### Backup Schedule
- Daily full backups
- Hourly incremental backups
- 7-year retention for compliance

## Environment Variables

```bash
# PostgreSQL
DB_HOST=localhost
DB_PORT=5432
DB_NAME=agentic_accounting
DB_USER=postgres
DB_PASSWORD=your_password
DB_POOL_SIZE=20
DB_SSL=false

# AgentDB
AGENTDB_DIMENSIONS=768
AGENTDB_METRIC=cosine
AGENTDB_QUANTIZATION=int8
AGENTDB_PERSISTENCE=true
AGENTDB_PATH=./data/agentdb
```

## Migration Commands

```bash
# Run all pending migrations
npm run db:migrate

# Run seed data
npm run db:seed

# Rollback last migration
npm run db:rollback

# Reset database (rollback + migrate)
npm run db:reset
```

## Development Setup

1. Install PostgreSQL 15+
2. Install pgvector extension
3. Create database: `createdb agentic_accounting`
4. Set environment variables
5. Run migrations: `npm run db:migrate`
6. (Optional) Run seeds: `npm run db:seed`

## Production Considerations

- Use read replicas for reporting workloads
- Partition large tables by tax year
- Regular VACUUM and ANALYZE
- Monitor query performance
- Set up automated backups
- Configure connection pooling
- Enable SSL/TLS connections
- Implement row-level security if multi-tenant

## References

- PostgreSQL Documentation: https://www.postgresql.org/docs/
- pgvector: https://github.com/pgvector/pgvector
- HNSW Algorithm: https://arxiv.org/abs/1603.09320
- AgentDB: https://github.com/ruvnet/agentdb
