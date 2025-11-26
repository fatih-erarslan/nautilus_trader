# Integration Layer Architecture

## Quick Reference

This crate provides the **central integration layer** for neural-trader.

### Key Components

1. **NeuralTrader** - Main facade (single entry point)
2. **Services** - Business logic (Trading, Analytics, Risk, Neural)
3. **Coordination** - Resource management (Brokers, Strategies, Models, Memory)
4. **APIs** - External interfaces (REST, WebSocket, CLI)

### Documentation

- **Full Architecture**: `/workspaces/neural-trader/docs/integration-architecture.md`
- **Quick Start**: `/workspaces/neural-trader/docs/integration-quickstart.md`
- **README**: `README.md`

### Quick Start

```bash
# Copy config
cp example.config.toml config.toml

# Edit credentials
vim config.toml

# Run
cargo run --release
```

### Integration Status

✅ Core infrastructure complete
✅ Services layer complete
✅ Coordination layer complete
✅ API layer complete
✅ Documentation complete
🔄 Connecting to other crates (as they are implemented)

See `/workspaces/neural-trader/docs/INTEGRATION_COMPLETE.md` for full status.
