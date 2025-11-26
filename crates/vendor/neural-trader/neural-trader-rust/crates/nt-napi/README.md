# nt-napi - Neural Trader NAPI Native Addon

High-performance Rust implementation of Neural Trader exposed to Node.js via NAPI-RS.

## Overview

This crate provides Node.js bindings for the entire neural-trader Rust implementation, exposing 99 trading tools across 9 categories through native N-API addons.

## Features

- **Trading Strategies**: 15+ strategy implementations with GPU acceleration
- **Neural Networks**: LSTM/GAN forecasting with 27+ models
- **Sports Betting**: Real-time odds, arbitrage detection, Kelly Criterion
- **Syndicates**: Collaborative betting with profit distribution
- **Prediction Markets**: Market analysis and sentiment tracking
- **E2B Deployment**: Isolated sandbox execution
- **Portfolio Management**: Risk analysis, optimization, rebalancing
- **News Analysis**: Sentiment tracking and trend analysis
- **Fantasy Sports**: (Coming soon)

## Architecture

```
nt-napi/
├── src/
│   ├── lib.rs           # Main module, initialization
│   ├── error.rs         # Error handling utilities
│   ├── trading.rs       # Trading strategies (15 tools)
│   ├── neural.rs        # Neural networks (12 tools)
│   ├── sports.rs        # Sports betting (18 tools)
│   ├── syndicate.rs     # Syndicates (17 tools)
│   ├── prediction.rs    # Prediction markets (7 tools)
│   ├── e2b.rs          # E2B deployment (10 tools)
│   ├── portfolio.rs     # Portfolio management (13 tools)
│   ├── news.rs         # News analysis (4 tools)
│   └── fantasy.rs      # Fantasy sports (3 tools)
├── Cargo.toml          # Dependencies and build config
└── build.rs            # NAPI build setup
```

## Building

### Prerequisites

- Rust 1.70+
- Node.js 16+
- NAPI-RS CLI: `npm install -g @napi-rs/cli`

### Build Commands

```bash
# Development build
cargo build

# Release build (optimized)
cargo build --release

# Build Node.js addon
napi build --release

# Build for specific platform
napi build --release --platform linux --arch x64
```

## Integration

This crate is designed to be used by the `neural-trader` package:

```typescript
// In neural-trader package
import { initNeuralTrader, quickAnalysis } from './index.js';

await initNeuralTrader();
const analysis = await quickAnalysis('AAPL', true);
```

## Dependencies

### Core NAPI

- `napi = "3"` - Core N-API bindings with async support
- `napi-derive = "3"` - Procedural macros for NAPI
- `napi-build = "3"` - Build script support

### Neural Trader Crates

- `nt-core` - Core types and utilities
- `nt-strategies` - Trading strategy implementations
- `nt-neural` - Neural network models
- `nt-sports` - Sports betting engine
- `nt-prediction` - Prediction market tools
- `nt-syndicate` - Syndicate management
- `nt-e2b` - E2B deployment
- `nt-portfolio` - Portfolio optimization

### Runtime

- `tokio` - Async runtime for NAPI async functions
- `serde/serde_json` - Serialization for complex types
- `chrono` - Date/time handling

## Performance Optimizations

The release profile includes aggressive optimizations:

```toml
[profile.release]
lto = true              # Link-time optimization
strip = true            # Strip debug symbols
opt-level = 3           # Maximum optimization
codegen-units = 1       # Single codegen unit for better optimization
```

## Error Handling

All NAPI functions return `napi::Result<T>` with custom error types:

```rust
use crate::error::*;

#[napi]
pub async fn example() -> Result<String> {
    // Rust errors are automatically converted to NAPI errors
    Ok("Success".to_string())
}
```

## Testing

```bash
# Run Rust unit tests
cargo test

# Run integration tests with Node.js
npm test
```

## Platform Support

- **Linux**: x64, arm64
- **macOS**: x64, arm64 (Apple Silicon)
- **Windows**: x64

## Contributing

This crate is part of the neural-trader monorepo. See the main repository for contribution guidelines.

## License

MIT
