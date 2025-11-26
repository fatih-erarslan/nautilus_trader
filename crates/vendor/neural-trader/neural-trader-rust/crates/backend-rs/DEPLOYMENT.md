# Swarm Rust Backend - Deployment Guide

## 📦 Package Contents

This package contains the complete BeClever Rust backend API server with:

- **Multi-crate workspace** architecture
- **E2B integration** for agent deployment in cloud sandboxes
- **OpenRouter integration** for LLM capabilities
- **SQLite database** with migrations
- **Authentication middleware**
- **API Scanner** capabilities
- **Workflow management**
- **Analytics & monitoring**

## 🚀 Quick Start

### Prerequisites

1. **Rust toolchain** (1.70.0 or later)
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```

2. **System dependencies**
   ```bash
   # Ubuntu/Debian
   sudo apt-get update
   sudo apt-get install -y build-essential pkg-config libssl-dev sqlite3

   # macOS
   brew install pkg-config openssl sqlite3

   # Fedora/RHEL
   sudo dnf install -y gcc pkg-config openssl-devel sqlite-devel
   ```

3. **API Keys** (optional but recommended)
   - E2B API Key: https://e2b.dev
   - OpenRouter API Key: https://openrouter.ai

### Installation Steps

#### 1. Extract the package

```bash
tar -xzf beclever-backend-rs.tar.gz
cd backend-rs
```

#### 2. Configure environment variables

```bash
# Copy example environment file
cat > .env << 'EOF'
# API Keys
E2B_API_KEY=e2b_your_api_key_here
OPENROUTER_API_KEY=sk-or-v1-your_openrouter_key_here

# Database
DATABASE_URL=sqlite://data/beclever.db

# Server
HOST=0.0.0.0
PORT=8001

# Authentication (optional)
JWT_SECRET=your-secret-key-change-this-in-production
EOF
```

#### 3. Build the project

```bash
# Development build
cargo build

# Production build (optimized)
cargo build --release
```

#### 4. Run the server

```bash
# Development mode
cargo run

# Production mode (recommended)
E2B_API_KEY="your_key" OPENROUTER_API_KEY="your_key" ./target/release/beclever-api
```

The server will start on `http://0.0.0.0:8001`

## 📁 Project Structure

```
backend-rs/
├── Cargo.toml              # Workspace configuration
├── Cargo.lock              # Dependency lock file
├── crates/
│   ├── api/                # Main API server
│   │   ├── src/
│   │   │   ├── main.rs           # Server entry point
│   │   │   ├── agents.rs         # Agent deployment endpoints
│   │   │   ├── db.rs             # Database operations
│   │   │   ├── auth.rs           # Authentication middleware
│   │   │   ├── e2b_client.rs     # E2B sandbox client
│   │   │   ├── openrouter_client.rs # LLM client
│   │   │   ├── analytics.rs      # Analytics endpoints
│   │   │   └── scanner.rs        # API scanning
│   │   └── Cargo.toml
│   ├── common/             # Shared utilities
│   ├── core/               # Business logic
│   ├── db/                 # Database models
│   └── napi/               # Node.js bindings
├── migrations/             # SQL migrations
│   ├── 001_*.sql
│   ├── 002_*.sql
│   ├── 003_*.sql
│   └── 004_agent_deployment_sqlite.sql
├── data/                   # Database files (created at runtime)
└── examples/               # Example code
```

## 🔧 Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `E2B_API_KEY` | No | - | E2B cloud sandbox API key |
| `OPENROUTER_API_KEY` | No | - | OpenRouter LLM API key |
| `DATABASE_URL` | No | `sqlite://data/beclever.db` | SQLite database path |
| `HOST` | No | `0.0.0.0` | Server bind address |
| `PORT` | No | `8001` | Server port |
| `JWT_SECRET` | No | Auto-generated | JWT signing secret |

### Database Migrations

Migrations run automatically on server startup. Manual migration:

```bash
# Check migration status
sqlite3 data/beclever.db ".schema"

# Run migrations manually if needed
sqlite3 data/beclever.db < migrations/001_initial_schema.sql
```

## 🌐 API Endpoints

### Agent Deployment

- `GET /api/agents` - List all deployed agents
- `POST /api/agents/deploy` - Deploy new agent in E2B sandbox
- `GET /api/agents/:id` - Get agent status
- `DELETE /api/agents/:id` - Terminate/destroy agent
- `GET /api/agents/:id/logs` - Get agent execution logs
- `POST /api/agents/:id/execute` - Execute code in agent sandbox

### Workflows

- `GET /api/workflows` - List workflows
- `POST /api/workflows` - Create workflow
- `POST /api/workflows/:id/execute` - Execute workflow

### Analytics

- `GET /api/analytics/dashboard` - Dashboard statistics
- `GET /api/analytics/usage` - Usage analytics
- `GET /api/analytics/activity` - Activity feed
- `POST /api/analytics/activity` - Log activity
- `GET /api/analytics/performance` - Performance metrics

### API Scanner

- `GET /api/scanner/scans` - List API scans
- `POST /api/scanner/scans` - Create new scan
- `GET /api/scanner/scans/:id` - Get scan details
- `PATCH /api/scanner/scans/:id` - Update scan
- `DELETE /api/scanner/scans/:id` - Delete scan
- `GET /api/scanner/stats` - Scanner statistics

### System

- `GET /api/stats` - System statistics
- `GET /health` - Health check endpoint

## 🔐 Authentication

The server includes authentication middleware. To disable for development:

Edit `crates/api/src/main.rs` and remove the auth layer:

```rust
// Comment out or remove this line
// .layer(middleware::from_fn(auth::auth_middleware))
```

## 🧪 Testing

```bash
# Run unit tests
cargo test

# Run with output
cargo test -- --nocapture

# Run specific test
cargo test test_name

# Test agent deployment (requires E2B key)
curl -X POST http://localhost:8001/api/agents/deploy \
  -H "Content-Type: application/json" \
  -d '{
    "agent_name": "test-agent",
    "agent_type": "researcher",
    "task_description": "Test task",
    "template": "base",
    "environment": {},
    "capabilities": [],
    "config": {"timeout": 3600}
  }'

# List agents
curl http://localhost:8001/api/agents
```

## 📊 Monitoring

### Logs

The server uses `tracing` for structured logging:

```bash
# Set log level
RUST_LOG=debug cargo run
RUST_LOG=info cargo run        # Default
RUST_LOG=error cargo run
```

### Metrics

- Server startup logs show configuration
- Database connection status
- Migration completion
- Request/response logging

## 🐛 Troubleshooting

### Port Already in Use

```bash
# Find process on port 8001
lsof -ti:8001

# Kill process
lsof -ti:8001 | xargs kill -9
```

### Database Locked

```bash
# Stop all running instances
pkill -f beclever-api

# Remove lock files
rm data/*.db-shm data/*.db-wal
```

### Build Failures

```bash
# Clean build cache
cargo clean

# Update dependencies
cargo update

# Rebuild
cargo build --release
```

### Missing Dependencies

```bash
# Ubuntu/Debian
sudo apt-get install build-essential pkg-config libssl-dev

# macOS
xcode-select --install
brew install pkg-config openssl
```

## 🚢 Production Deployment

### Using systemd

Create `/etc/systemd/system/beclever-api.service`:

```ini
[Unit]
Description=BeClever Rust API Server
After=network.target

[Service]
Type=simple
User=beclever
WorkingDirectory=/opt/beclever/backend-rs
Environment="E2B_API_KEY=your_key"
Environment="OPENROUTER_API_KEY=your_key"
ExecStart=/opt/beclever/backend-rs/target/release/beclever-api
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable beclever-api
sudo systemctl start beclever-api
sudo systemctl status beclever-api
```

### Using Docker

```dockerfile
FROM rust:1.75 as builder
WORKDIR /app
COPY . .
RUN cargo build --release

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y libssl3 sqlite3 && rm -rf /var/lib/apt/lists/*
COPY --from=builder /app/target/release/beclever-api /usr/local/bin/
COPY --from=builder /app/migrations /app/migrations
WORKDIR /app
EXPOSE 8001
CMD ["beclever-api"]
```

```bash
docker build -t beclever-api .
docker run -p 8001:8001 \
  -e E2B_API_KEY=your_key \
  -e OPENROUTER_API_KEY=your_key \
  -v $(pwd)/data:/app/data \
  beclever-api
```

### Reverse Proxy (Nginx)

```nginx
server {
    listen 80;
    server_name api.beclever.ai;

    location / {
        proxy_pass http://127.0.0.1:8001;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## 📈 Performance Tuning

### Database

```bash
# Enable WAL mode for better concurrency
sqlite3 data/beclever.db "PRAGMA journal_mode=WAL;"

# Optimize database
sqlite3 data/beclever.db "VACUUM; ANALYZE;"
```

### Build Optimizations

Add to `Cargo.toml`:

```toml
[profile.release]
opt-level = 3
lto = true
codegen-units = 1
strip = true
```

## 🔒 Security

### Production Checklist

- [ ] Change default `JWT_SECRET`
- [ ] Enable HTTPS/TLS
- [ ] Configure firewall rules
- [ ] Set up rate limiting
- [ ] Enable authentication middleware
- [ ] Rotate API keys regularly
- [ ] Review CORS configuration
- [ ] Set up monitoring/alerting
- [ ] Configure backup strategy
- [ ] Review database permissions

## 📚 Additional Resources

- **E2B Documentation**: https://e2b.dev/docs
- **OpenRouter API**: https://openrouter.ai/docs
- **Rust Documentation**: https://doc.rust-lang.org/
- **Axum Web Framework**: https://docs.rs/axum/latest/axum/

## 🆘 Support

For issues or questions:

1. Check the troubleshooting section above
2. Review server logs: `journalctl -u beclever-api -f`
3. GitHub Issues: https://github.com/FoxRev/beclever/issues
4. API Documentation: http://localhost:8001/docs (when running)

## 📄 License

See LICENSE file in the project root.

---

**Version**: 1.0.0
**Last Updated**: 2025-11-14
**Status**: Production Ready ✅
