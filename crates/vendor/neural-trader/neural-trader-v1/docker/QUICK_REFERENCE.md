# Docker Validation - Quick Reference Card

## 🚀 One-Line Commands

```bash
# Run everything (recommended)
./scripts/docker-test.sh --fresh --benchmark

# Test only
./scripts/docker-test.sh

# Validate MCP compliance
./scripts/validate-docker.sh

# Start MCP server
docker-compose -f docker-compose.validation.yml up -d mcp-server

# Stop everything
docker-compose -f docker-compose.validation.yml down
```

## 📋 Service Commands

```bash
# Build all images
docker-compose -f docker-compose.validation.yml build

# Run tests
docker-compose -f docker-compose.validation.yml run --rm testing

# Run validation
docker-compose -f docker-compose.validation.yml run --rm validation

# Run benchmarks
docker-compose -f docker-compose.validation.yml run --rm benchmark

# View logs
docker-compose -f docker-compose.validation.yml logs -f mcp-server
```

## 🔍 Debug Commands

```bash
# Interactive shell
docker-compose -f docker-compose.validation.yml run --rm testing /bin/bash

# Check health
curl http://localhost:3000/health

# List tools
curl http://localhost:3000/tools | jq '. | length'

# Test specific tool
curl http://localhost:3000/tools/ping

# View validation report
cat reports/validation-*.json | jq .
```

## 📊 Validation Checklist

- [ ] `./scripts/docker-test.sh --fresh` runs successfully
- [ ] MCP server starts and passes health check
- [ ] All 107+ tools are accessible
- [ ] Validation report shows 100% success
- [ ] Benchmarks complete without errors
- [ ] No security vulnerabilities found

## 🎯 Expected Results

```
✅ Build Time: <5 minutes (fresh), <1 minute (cached)
✅ Test Execution: <3 minutes
✅ MCP Response: <100ms
✅ Tool Count: ≥107
✅ Protocol: 2025-11
✅ Success Rate: 100%
```

## 📁 Key Files

```
Dockerfile.validation              # Multi-stage build
docker-compose.validation.yml      # Services configuration
scripts/docker-test.sh             # Test automation
scripts/validate-docker.sh         # MCP validation
docker/README.md                   # Full documentation
```

## 🐛 Quick Troubleshooting

| Problem | Solution |
|---------|----------|
| Port 3000 in use | `lsof -i :3000` and kill process |
| Build fails | `docker-compose build --no-cache` |
| Tests fail | Check logs: `docker-compose logs testing` |
| NAPI not found | Rebuild: `npm run build:release` |
| Health check fails | `docker exec neural-trader-mcp /healthcheck.sh` |

## 📞 Quick Links

- Full Docs: `docker/README.md`
- Setup Guide: `docs/DOCKER_VALIDATION_SETUP.md`
- Summary: `DOCKER_VALIDATION_SUMMARY.md`
- CI/CD: `.github/workflows/docker-validation.yml`
