# Agentic Accounting Core

Core TypeScript library for the agentic accounting system with PostgreSQL, pgvector, and AgentDB integration.

## Features

- **PostgreSQL Database** with pgvector extension for vector similarity search
- **AgentDB Client** for semantic pattern matching
- **10 Core Tables** for comprehensive accounting tracking
- **Migration System** for database schema management
- **Type-Safe API** with TypeScript
- **High Performance** with optimized indexes (B-tree, HNSW)
- **Compliance Rules** engine for transaction validation
- **Audit Trail** with cryptographic verification
- **ReasoningBank** for agent learning memory

## Installation

```bash
npm install @neural-trader/agentic-accounting-core
```

## Database Setup

### Prerequisites

1. PostgreSQL 15+ installed
2. pgvector extension available

```bash
# Ubuntu/Debian
sudo apt install postgresql-15 postgresql-15-pgvector

# macOS (Homebrew)
brew install postgresql@15 pgvector
```

### Initialize Database

```bash
# Create database
createdb agentic_accounting

# Configure environment
cp .env.example .env
# Edit .env with your database credentials

# Run migrations
npm run db:migrate

# (Optional) Seed test data
npm run db:seed
```

## Usage

### Initialize Databases

```typescript
import { initializeAllDatabases, closeAllDatabases } from '@neural-trader/agentic-accounting-core';

// Initialize PostgreSQL and AgentDB
await initializeAllDatabases();

// ... use the databases

// Close connections
await closeAllDatabases();
```

### Query PostgreSQL

```typescript
import { query, transaction } from '@neural-trader/agentic-accounting-core';

// Simple query
const result = await query(
  'SELECT * FROM transactions WHERE asset = $1',
  ['BTC']
);

// Transaction
await transaction(async (client) => {
  await client.query('INSERT INTO transactions (...) VALUES (...)');
  await client.query('INSERT INTO tax_lots (...) VALUES (...)');
});
```

### Use AgentDB

```typescript
import { getAgentDB } from '@neural-trader/agentic-accounting-core';

const agentDB = getAgentDB();
await agentDB.initialize();

// Insert vector
await agentDB.insert('transactions', 'tx-123', embedding, {
  asset: 'BTC',
  amount: 50000,
});

// Search similar
const results = await agentDB.search('transactions', queryVector, 10);
```

## Database Schema

### Core Tables

1. **transactions** - All financial transactions
2. **tax_lots** - Individual lot tracking for cost basis
3. **disposals** - Sale records with realized gains/losses
4. **positions** - Current holdings (materialized view)
5. **tax_summaries** - Annual tax summaries
6. **compliance_rules** - Configurable compliance rules
7. **audit_trail** - Immutable audit log
8. **embeddings** - Vector embeddings (pgvector)
9. **reasoning_bank** - Agent learning memory
10. **verification_proofs** - Formal verification proofs

See [Database Documentation](./docs/database/README.md) for full schema details.

## Migration Commands

```bash
# Run pending migrations
npm run db:migrate

# Seed test data
npm run db:seed

# Rollback last migration
npm run db:rollback

# Reset database (rollback + migrate)
npm run db:reset

# Full setup (migrate + seed)
npm run db:setup
```

## Environment Variables

```bash
# PostgreSQL
DB_HOST=localhost
DB_PORT=5432
DB_NAME=agentic_accounting
DB_USER=postgres
DB_PASSWORD=your_password
DB_POOL_SIZE=20

# AgentDB
AGENTDB_DIMENSIONS=768
AGENTDB_METRIC=cosine
AGENTDB_PERSISTENCE=true
AGENTDB_PATH=./data/agentdb
```

## Development

```bash
# Install dependencies
npm install

# Build
npm run build

# Watch mode
npm run dev

# Run tests
npm test

# Lint
npm run lint

# Format
npm run format
```

## Architecture

### PostgreSQL Features
- **pgvector extension** for vector similarity search
- **HNSW indexes** for O(log n) vector search
- **Materialized views** for aggregated data
- **Audit trail** with SHA-256 hash chaining
- **ACID transactions** for data integrity

### AgentDB Features
- **In-memory vector database** with disk persistence
- **HNSW indexing** for fast similarity search
- **Quantization support** (int8) for memory efficiency
- **Multiple distance metrics** (cosine, euclidean, dot)

## Performance

- **Vector search**: <100µs for top-10 results
- **Connection pooling**: Max 20 concurrent connections
- **Optimized indexes**: B-tree + HNSW for hybrid queries
- **Materialized views**: Pre-aggregated data for fast reporting

## Security

- **Immutable audit trail**: No updates or deletes
- **Cryptographic verification**: SHA-256 + Ed25519 signatures
- **TLS encryption**: Secure database connections
- **Input validation**: Parameterized queries prevent SQL injection

## Documentation

- [Database README](./docs/database/README.md) - Architecture and design
- [Schema Reference](./docs/database/SCHEMA.md) - Detailed table definitions
- [Migration Guide](./src/database/migrations/) - SQL migrations

## License

MIT

## Contributing

Contributions welcome! Please read the contributing guidelines before submitting PRs.

## Support

For issues and questions:
- GitHub Issues: https://github.com/neural-trader/neural-trader/issues
- Documentation: https://docs.neural-trader.io
