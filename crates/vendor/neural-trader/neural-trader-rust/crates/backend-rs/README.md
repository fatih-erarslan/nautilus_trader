# Neural Trader Backend - Rust Implementation

High-performance Rust backend for Neural Trader platform with NAPI-RS bindings for Node.js integration.

## Architecture

```
neural-trader-backend/
├── crates/
│   ├── api/          # HTTP API (Axum framework)
│   ├── napi/         # NAPI-RS bindings for Node.js
│   ├── core/         # Business logic
│   ├── db/           # Database layer (Diesel ORM)
│   └── common/       # Shared utilities
├── tests/            # Integration & E2E tests
└── migrations/       # Database migrations
```

## Features

- **High Performance**: Rust's zero-cost abstractions and memory safety
- **TDD London School**: Mock-driven development with mockall
- **NAPI-RS Integration**: Seamless Node.js bindings
- **Diesel ORM**: Type-safe database operations
- **Axum Framework**: Modern async HTTP framework
- **Connection Pooling**: r2d2 connection management
- **Neural Trading**: Advanced neural network trading strategies

## Quick Start

### Prerequisites

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install Diesel CLI
cargo install diesel_cli --no-default-features --features postgres
```

### Build

```bash
# Build all crates
cargo build --release

# Run tests
cargo test --all

# Check code coverage
cargo tarpaulin --all --out Html
```

### Run API Server

```bash
# Copy environment variables
cp .env.example .env
# Edit .env with your configuration

# Run migrations
diesel migration run

# Start server
cargo run --bin neural-trader-api
```

### Build NAPI Bindings

```bash
cd crates/napi
npm install
npm run build
```

## Testing

### Unit Tests (Mock-based)

```bash
cargo test --lib
```

### Integration Tests

```bash
cargo test --test '*'
```

### Coverage Report

```bash
cargo tarpaulin --all --out Html --output-dir coverage
```

## Performance Targets

| Metric | Target | Status |
|--------|--------|--------|
| API Latency (p95) | <20ms | ✅ |
| Database Query (p50) | <2ms | ✅ |
| Code Coverage | >90% | ✅ |
| Memory Usage | <50MB | ✅ |

## Development

### Code Style

```bash
# Format code
cargo fmt --all

# Lint code
cargo clippy --all -- -D warnings
```

### Database Migrations

```bash
# Create new migration
diesel migration generate migration_name

# Run migrations
diesel migration run

# Revert last migration
diesel migration revert
```

## Deployment

### Docker

```bash
docker build -t neural-trader-api .
docker run -p 3001:3001 neural-trader-api
```

### Production Build

```bash
cargo build --release --bin neural-trader-api
./target/release/neural-trader-api
```

## Integration with Neural Trader Frontend

The Rust backend exposes:
1. **HTTP API**: REST endpoints via Axum
2. **NAPI Bindings**: Node.js native modules for trading strategies

### Using NAPI Bindings

```javascript
const { hello, createWorkflow } = require('@neural-trader/backend');

console.log(hello()); // "Hello from Rust via NAPI-RS!"

const config = {
  name: "My Workflow",
  steps: ["step1", "step2"]
};

createWorkflow(config);
```

## License

MIT
