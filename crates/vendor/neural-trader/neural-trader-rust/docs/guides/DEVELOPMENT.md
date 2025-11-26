# Development Guide - Neural Trader Rust Port

Complete development guide for contributing to Neural Trader.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Building](#building)
- [Testing](#testing)
- [Code Style](#code-style)
- [Contributing](#contributing)

## Getting Started

### Prerequisites

```bash
# Rust 1.70+
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Node.js 18+
nvm install 18
nvm use 18

# Development tools
cargo install cargo-watch cargo-tarpaulin cargo-audit
npm install -g @napi-rs/cli
```

### Initial Setup

```bash
# Clone repository
git clone https://github.com/ruvnet/neural-trader.git
cd neural-trader/neural-trader-rust

# Install dependencies
npm install

# Build native module
npm run build:debug

# Run tests
npm test
cargo test
```

## Development Workflow

### Hot Reloading

Use `cargo-watch` for automatic rebuilds:

```bash
# Watch Rust code
cargo watch -x 'build --manifest-path crates/napi-bindings/Cargo.toml'

# Watch and test
cargo watch -x test

# Watch specific crate
cargo watch -x 'test -p nt-core'
```

### Project Structure

```
neural-trader-rust/
├── crates/
│   ├── core/              # Core trading types and traits
│   ├── market-data/       # Real-time market data feeds
│   ├── strategies/        # Trading strategies
│   ├── execution/         # Order execution engine
│   ├── portfolio/         # Portfolio management
│   ├── risk/              # Risk management
│   ├── backtesting/       # Backtesting engine
│   ├── neural/            # Neural network integration
│   ├── napi-bindings/     # Node.js FFI bindings
│   └── cli/               # Command-line interface
├── npm/                   # Platform-specific packages
├── scripts/               # Build and utility scripts
├── tests/                 # Integration tests
└── benches/               # Performance benchmarks
```

## Building

### Debug Build (Fast)

```bash
npm run build:debug
```

- Faster compilation
- Includes debug symbols
- No optimizations
- Good for development

### Release Build (Optimized)

```bash
npm run build
```

- Full optimizations (opt-level=3)
- LTO (Link-Time Optimization) enabled
- Stripped debug symbols
- Production-ready

### Platform-Specific Builds

```bash
# Linux
npm run build:linux       # x86_64-unknown-linux-gnu + musl

# macOS
npm run build:darwin      # x86_64 + aarch64 (Apple Silicon)

# Windows
npm run build:windows     # x86_64-pc-windows-msvc

# All platforms
npm run build:all
```

### Cross-Compilation Setup

```bash
./scripts/setup-cross-compile.sh
```

This installs:
- Rust target toolchains
- Cross-compilation linkers
- MUSL tools (static linking)
- MinGW (Windows cross-compilation on Linux)

## Testing

### Rust Tests

```bash
# All tests
cargo test --workspace

# Specific crate
cargo test -p nt-core

# With output
cargo test -- --nocapture

# Specific test
cargo test test_market_order

# Integration tests only
cargo test --test '*'

# Doc tests
cargo test --doc
```

### Node.js Tests

```bash
# All tests
npm test

# Watch mode
npm run test:watch

# Single test file
npm test -- index.test.js
```

### Integration Tests

```bash
# Run full integration suite
cargo test --test integration_test

# Specific integration test
cargo test --test backtesting_integration
```

### Benchmarks

```bash
# Run all benchmarks
cargo bench

# Specific benchmark
cargo bench execution_latency

# Compare with baseline
cargo bench -- --save-baseline main
git checkout feature-branch
cargo bench -- --baseline main
```

## Code Style

### Rust Style

We follow the [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/):

```bash
# Format code
cargo fmt

# Check formatting
cargo fmt -- --check

# Lint with Clippy
cargo clippy --all-targets --all-features

# Fix Clippy warnings
cargo clippy --fix
```

**Key conventions:**
- Snake_case for functions, variables, modules
- CamelCase for types, traits
- SCREAMING_SNAKE_CASE for constants
- Prefer `?` over `unwrap()`
- Document all public APIs
- Write unit tests for all logic

### TypeScript Style

```bash
# Type check
npx tsc --noEmit

# Lint
npm run lint

# Format (if using Prettier)
npm run format
```

## Performance Profiling

### CPU Profiling

```bash
# Linux (perf)
cargo build --release
perf record --call-graph=dwarf ./target/release/neural-trader
perf report

# macOS (Instruments)
cargo instruments --release --bench execution_engine

# Flamegraph
cargo install flamegraph
cargo flamegraph --bench execution_engine
```

### Memory Profiling

```bash
# Valgrind (Linux)
cargo build
valgrind --tool=massif ./target/debug/neural-trader
ms_print massif.out.*

# Heaptrack (Linux)
heaptrack ./target/debug/neural-trader
```

### Benchmarking

```bash
# Criterion.rs benchmarks
cargo bench

# Custom benchmark
cargo bench --bench execution_latency -- --verbose

# Compare baselines
cargo bench -- --baseline before
# Make changes
cargo bench -- --baseline after
```

## Debugging

### Rust Debugging

```bash
# Build with debug symbols
cargo build

# GDB (Linux)
gdb ./target/debug/neural-trader
# (gdb) break main
# (gdb) run

# LLDB (macOS)
lldb ./target/debug/neural-trader
# (lldb) breakpoint set -n main
# (lldb) run

# VSCode
# Use launch.json configuration
```

### Node.js Debugging

```bash
# Node inspector
node --inspect-brk node_modules/.bin/vitest

# Chrome DevTools
# Navigate to: chrome://inspect
```

### Logging

```bash
# Enable Rust tracing
RUST_LOG=debug npm test

# Specific module
RUST_LOG=neural_trader::execution=trace npm test

# Save logs
RUST_LOG=info npm test > test.log 2>&1
```

## Contributing

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add momentum strategy implementation
fix: correct order execution timing bug
docs: update API documentation
test: add backtesting integration tests
perf: optimize market data parsing
refactor: simplify portfolio calculation logic
```

### Pull Request Process

1. **Fork & Branch**: Create a feature branch
   ```bash
   git checkout -b feat/my-feature
   ```

2. **Implement**: Write code + tests
   ```bash
   cargo test
   npm test
   ```

3. **Format & Lint**:
   ```bash
   cargo fmt
   cargo clippy --fix
   ```

4. **Commit**: Use conventional commits
   ```bash
   git commit -m "feat: add new feature"
   ```

5. **Push & PR**:
   ```bash
   git push origin feat/my-feature
   # Create PR on GitHub
   ```

6. **CI Checks**: Ensure all checks pass
   - Rust tests
   - Node.js tests
   - Linting (rustfmt, clippy)
   - Security audit

### Code Review Guidelines

- **Test Coverage**: Aim for 80%+ coverage
- **Documentation**: Document all public APIs
- **Performance**: Benchmark performance-critical code
- **Safety**: Avoid unsafe code unless necessary (with justification)
- **Error Handling**: Use Result/Error types, avoid panics

## Advanced Topics

### SIMD Optimization

```rust
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// SIMD-accelerated calculations
fn process_batch_simd(data: &[f64]) -> Vec<f64> {
    // Implementation using SIMD intrinsics
}
```

### Async Runtime

```rust
#[tokio::main]
async fn main() {
    // Async execution
}

// Testing async code
#[tokio::test]
async fn test_async_function() {
    // Test async code
}
```

### FFI Safety

```rust
// Safe FFI wrapper
#[napi]
pub fn safe_function(input: String) -> Result<String> {
    // Validate input
    // Process
    // Return Result
}
```

## Resources

- **Rust Book**: https://doc.rust-lang.org/book/
- **napi-rs Docs**: https://napi.rs
- **Tokio Docs**: https://tokio.rs
- **Criterion.rs**: https://bheisler.github.io/criterion.rs/book/

## Getting Help

- **GitHub Issues**: https://github.com/ruvnet/neural-trader/issues
- **Discussions**: https://github.com/ruvnet/neural-trader/discussions
- **Discord**: https://discord.gg/neural-trader

---

Happy coding! 🦀🚀
