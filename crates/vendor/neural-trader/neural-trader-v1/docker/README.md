# Neural Trader Docker Validation Environment

Complete Docker-based validation environment for the Neural Trader NAPI-RS implementation with MCP 2025-11 compliance testing.

## 🚀 Quick Start

```bash
# Build and test everything
./scripts/docker-test.sh

# Run with fresh build
./scripts/docker-test.sh --fresh

# Include performance benchmarks
./scripts/docker-test.sh --benchmark

# Skip validation checks
./scripts/docker-test.sh --skip-validation
```

## 📦 Docker Images

### Multi-Stage Build Architecture

1. **rust-builder**: Compiles all Rust crates and NAPI bindings
2. **node-builder**: Builds Node.js artifacts with NAPI-RS
3. **testing**: Full test environment with Rust + Node.js
4. **mcp-server**: Production-ready MCP server
5. **validation**: MCP 2025-11 compliance testing

## 🔧 Services

### MCP Server
Production MCP server with health checks and monitoring.

```bash
# Start server
docker-compose -f docker-compose.validation.yml up -d mcp-server

# Check logs
docker-compose -f docker-compose.validation.yml logs -f mcp-server

# Check health
curl http://localhost:3000/health
```

### Testing Service
Comprehensive test suite execution.

```bash
# Run all tests
docker-compose -f docker-compose.validation.yml run --rm testing

# Run specific test
docker-compose -f docker-compose.validation.yml run --rm testing npm test -- --grep "specific test"
```

### Validation Service
MCP 2025-11 protocol compliance validation.

```bash
# Run validation
docker-compose -f docker-compose.validation.yml run --rm validation

# View results
cat reports/validation-*.json
```

### Benchmark Service
Performance benchmarking.

```bash
# Run benchmarks
docker-compose -f docker-compose.validation.yml run --rm benchmark

# View results
cat benchmark-results/*
```

## 🏗️ Building

### Build All Images
```bash
docker-compose -f docker-compose.validation.yml build
```

### Build Specific Stage
```bash
docker build -t neural-trader:test --target testing -f Dockerfile.validation .
```

### Multi-Platform Build
```bash
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  --target mcp-server \
  -t neural-trader:latest \
  -f Dockerfile.validation .
```

## 🧪 Testing

### Full Test Suite
```bash
./scripts/docker-test.sh
```

### Individual Test Categories
```bash
# Unit tests only
docker-compose -f docker-compose.validation.yml run --rm testing npm test -- unit

# Integration tests
docker-compose -f docker-compose.validation.yml run --rm testing npm test -- integration

# E2E tests
docker-compose -f docker-compose.validation.yml run --rm testing npm test -- e2e
```

## ✅ Validation

### MCP Protocol Validation
```bash
./scripts/validate-docker.sh
```

### Manual Validation
```bash
# Start server
docker-compose -f docker-compose.validation.yml up -d mcp-server

# Wait for healthy status
docker-compose -f docker-compose.validation.yml ps

# Test connectivity
curl http://localhost:3000/health

# List tools (should be ≥107)
curl http://localhost:3000/tools | jq '. | length'

# Test specific tool
curl http://localhost:3000/tools/ping
```

## 📊 Performance Benchmarks

```bash
# Run all benchmarks
docker-compose -f docker-compose.validation.yml run --rm benchmark

# Run specific benchmark
docker-compose -f docker-compose.validation.yml run --rm benchmark \
  cargo bench --bench portfolio_optimization
```

## 🔍 Debugging

### View Logs
```bash
# All services
docker-compose -f docker-compose.validation.yml logs

# Specific service
docker-compose -f docker-compose.validation.yml logs mcp-server

# Follow logs
docker-compose -f docker-compose.validation.yml logs -f --tail=100
```

### Interactive Shell
```bash
# Access testing environment
docker-compose -f docker-compose.validation.yml run --rm testing /bin/bash

# Access MCP server
docker exec -it neural-trader-mcp /bin/bash
```

### Debug Build Failures
```bash
# Build with verbose output
docker-compose -f docker-compose.validation.yml build --progress=plain

# No cache build
docker-compose -f docker-compose.validation.yml build --no-cache
```

## 🚨 CI/CD Integration

GitHub Actions workflow at `.github/workflows/docker-validation.yml` runs:

1. Multi-platform Docker builds
2. Full test suite
3. MCP 2025-11 validation
4. Performance benchmarks
5. Security scanning with Trivy

### Triggered On
- Push to main, develop, rust-port branches
- Pull requests to main, develop
- Manual workflow dispatch

### Artifacts
- Test results
- Validation reports
- Benchmark results
- Coverage reports

## 📈 Performance Metrics

Expected performance characteristics:

- **Build Time**: ~3-5 minutes (cached)
- **Test Execution**: ~2-3 minutes
- **MCP Response Latency**: <100ms
- **Tool Count**: ≥107 tools
- **Protocol Version**: 2025-11

## 🔐 Security

### Image Scanning
```bash
# Scan with Trivy
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
  aquasec/trivy image neural-trader:latest
```

### Security Best Practices
- No root user in production images
- Minimal base images (slim variants)
- No secrets in images
- Security options enabled
- Regular dependency updates

## 🐛 Troubleshooting

### Port Already in Use
```bash
# Find process using port 3000
lsof -i :3000

# Kill process
kill -9 <PID>
```

### Out of Disk Space
```bash
# Clean up Docker
docker system prune -a --volumes

# Remove old images
docker image prune -a
```

### Container Won't Start
```bash
# Check logs
docker-compose -f docker-compose.validation.yml logs mcp-server

# Inspect container
docker inspect neural-trader-mcp

# Check health
docker exec neural-trader-mcp /healthcheck.sh
```

### NAPI Bindings Not Found
```bash
# Verify NAPI build
docker-compose -f docker-compose.validation.yml run --rm testing \
  ls -la neural-trader-rust/crates/napi-bindings/*.node

# Rebuild bindings
docker-compose -f docker-compose.validation.yml run --rm testing \
  npm run build:release
```

## 📝 Environment Variables

### Build Time
- `RUST_VERSION`: Rust toolchain version (default: 1.75)
- `NODE_VERSION`: Node.js version (default: 18)

### Runtime
- `NODE_ENV`: Environment mode (production/test)
- `MCP_PORT`: Server port (default: 3000)
- `RUST_LOG`: Logging level (info/debug/trace)
- `RUST_BACKTRACE`: Enable backtraces (0/1/full)
- `MCP_VALIDATION`: Enable validation mode (true/false)
- `MCP_PROTOCOL_VERSION`: Protocol version (2025-11)

## 📚 Additional Resources

- [Neural Trader Documentation](../docs/)
- [NAPI-RS Guide](https://napi.rs/)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [MCP Protocol Spec](https://spec.modelcontextprotocol.io/)

## 🤝 Contributing

When adding new Docker features:

1. Update `Dockerfile.validation`
2. Add corresponding service to `docker-compose.validation.yml`
3. Update validation scripts if needed
4. Document changes in this README
5. Test with `./scripts/docker-test.sh --fresh`

## 📄 License

MIT OR Apache-2.0
