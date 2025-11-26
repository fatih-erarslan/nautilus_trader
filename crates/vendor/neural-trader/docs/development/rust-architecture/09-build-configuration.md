# Build Configuration and Feature Flags

## Workspace Structure

```toml
# Root Cargo.toml
[workspace]
members = [
    "crates/nt-core",
    "crates/nt-data",
    "crates/nt-features",
    "crates/nt-strategies",
    "crates/nt-signals",
    "crates/nt-execution",
    "crates/nt-risk",
    "crates/nt-memory",
    "crates/nt-sublinear",
    "crates/nt-streaming",
    "crates/nt-sandbox",
    "crates/nt-federation",
    "crates/nt-payments",
    "crates/nt-governance",
    "crates/nt-observability",
    "crates/nt-napi",
]

[workspace.package]
version = "1.0.0"
edition = "2021"
rust-version = "1.75"
authors = ["Neural Trader Team"]
license = "MIT"
repository = "https://github.com/neural-trader/neural-trader-rs"

[workspace.dependencies]
# Async runtime
tokio = { version = "1.35", features = ["full"] }
tokio-util = "0.7"
futures = "0.3"

# Data processing
polars = { version = "0.36", features = ["lazy", "temporal", "parquet"] }
arrow = "50.0"

# Serialization
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
rmp-serde = "1.1"
bincode = "1.3"

# Error handling
thiserror = "1.0"
anyhow = "1.0"

# Logging and tracing
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter", "json"] }
tracing-appender = "0.2"

# Metrics
prometheus = "0.13"

# HTTP/WebSocket
reqwest = { version = "0.11", features = ["json", "rustls-tls"] }
tokio-tungstenite = "0.21"

# Database
sqlx = { version = "0.7", features = ["runtime-tokio-rustls", "postgres"] }
rocksdb = "0.21"

# Decimal precision
rust_decimal = { version = "1.33", features = ["serde-float"] }

# Date/time
chrono = { version = "0.4", features = ["serde"] }

# Configuration
config = { version = "0.14", features = ["toml"] }

# Concurrency
crossbeam = "0.8"
dashmap = "5.5"
parking_lot = "0.12"

# Security
secrecy = "0.8"
ring = "0.17"
jsonwebtoken = "9.2"

# Testing
mockall = "0.12"
criterion = "0.5"

# Node.js bindings
napi = { version = "2.16", features = ["async", "tokio_rt"] }
napi-derive = "2.16"

[profile.dev]
opt-level = 0
debug = true
split-debuginfo = "unpacked"
debug-assertions = true
overflow-checks = true
lto = false
panic = "unwind"
incremental = true
codegen-units = 256

[profile.release]
opt-level = 3
debug = false
split-debuginfo = "off"
debug-assertions = false
overflow-checks = false
lto = "fat"              # Link-time optimization
panic = "abort"          # Smaller binary size
incremental = false
codegen-units = 1        # Better optimization
strip = true             # Strip symbols

[profile.bench]
inherits = "release"
debug = true             # Keep debug info for profiling

[profile.release-with-debug]
inherits = "release"
debug = true
strip = false

# Build script optimization
[profile.dev.build-override]
opt-level = 3

[profile.release.build-override]
opt-level = 3
```

## Core Crate Configuration

```toml
# crates/nt-core/Cargo.toml
[package]
name = "nt-core"
version.workspace = true
edition.workspace = true
rust-version.workspace = true
authors.workspace = true
license.workspace = true

[dependencies]
tokio.workspace = true
serde.workspace = true
thiserror.workspace = true
rust_decimal.workspace = true
chrono.workspace = true
tracing.workspace = true

[features]
default = []
```

## Feature Flags

### Main Application Features

```toml
# crates/nt-app/Cargo.toml
[features]
default = ["postgres", "rest-api"]

# Database backends
postgres = ["sqlx/postgres"]
sqlite = ["sqlx/sqlite"]
mysql = ["sqlx/mysql"]

# API interfaces
rest-api = ["axum", "tower"]
grpc-api = ["tonic", "prost"]
graphql-api = ["async-graphql"]

# Market data providers
alpaca = ["alpaca-api"]
polygon = ["polygon-api"]
coinbase = ["coinbase-api"]
binance = ["binance-api"]

# Neural network backends
onnx = ["ort"]
tensorflow = ["tensorflow-sys"]
torch = ["tch"]

# GPU acceleration
cuda = ["ort?/cuda", "cudarc"]
opencl = ["ocl"]

# Observability
metrics = ["prometheus"]
tracing = ["opentelemetry", "tracing-opentelemetry"]
profiling = ["pprof"]

# Security
tls = ["rustls", "tokio-rustls"]
mtls = ["tls"]

# Advanced features
sublinear = ["nt-sublinear"]
federation = ["nt-federation"]
e2b-sandbox = ["nt-sandbox"]
agentdb = ["nt-memory/agentdb"]

# Build configurations
production = ["postgres", "rest-api", "metrics", "tracing", "tls"]
development = ["sqlite", "rest-api", "profiling"]
minimal = []
```

### Conditional Compilation

```rust
// src/lib.rs

// GPU-specific code
#[cfg(feature = "cuda")]
pub mod gpu {
    use cudarc::driver::{CudaDevice, CudaSlice};

    pub fn create_device() -> CudaDevice {
        CudaDevice::new(0).expect("Failed to create CUDA device")
    }
}

#[cfg(not(feature = "cuda"))]
pub mod gpu {
    pub fn create_device() -> () {
        panic!("GPU support not enabled. Compile with --features cuda");
    }
}

// Database-specific code
#[cfg(feature = "postgres")]
pub async fn connect_database(url: &str) -> Result<sqlx::PgPool> {
    sqlx::PgPool::connect(url).await
}

#[cfg(feature = "sqlite")]
pub async fn connect_database(url: &str) -> Result<sqlx::SqlitePool> {
    sqlx::SqlitePool::connect(url).await
}

// API-specific code
#[cfg(feature = "rest-api")]
pub mod api {
    use axum::{Router, routing::get};

    pub fn create_router() -> Router {
        Router::new()
            .route("/health", get(health_check))
            .route("/metrics", get(metrics_handler))
    }
}

#[cfg(feature = "grpc-api")]
pub mod api {
    use tonic::transport::Server;

    pub async fn start_server() -> Result<()> {
        Server::builder()
            .add_service(trading_service_server())
            .serve(addr)
            .await
    }
}
```

## Cross-Compilation Configuration

```toml
# .cargo/config.toml
[target.x86_64-unknown-linux-gnu]
linker = "x86_64-linux-gnu-gcc"

[target.x86_64-unknown-linux-musl]
linker = "x86_64-linux-musl-gcc"
rustflags = ["-C", "target-feature=+crt-static"]

[target.aarch64-unknown-linux-gnu]
linker = "aarch64-linux-gnu-gcc"

[target.x86_64-apple-darwin]
rustflags = ["-C", "link-arg=-mmacosx-version-min=10.15"]

[target.aarch64-apple-darwin]
rustflags = ["-C", "link-arg=-mmacosx-version-min=11.0"]

[target.x86_64-pc-windows-msvc]
rustflags = ["-C", "target-feature=+crt-static"]

# Build flags for all targets
[build]
jobs = 8
incremental = true

[term]
verbose = false
color = "auto"
```

## Build Scripts

### Automated Build Script

```bash
#!/bin/bash
# scripts/build.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Parse arguments
BUILD_TYPE="${1:-release}"
FEATURES="${2:-default}"
TARGET="${3:-}"

echo "Building Neural Trader (type: $BUILD_TYPE, features: $FEATURES, target: $TARGET)"

# Set build flags
CARGO_FLAGS=""
if [ "$BUILD_TYPE" = "release" ]; then
    CARGO_FLAGS="--release"
elif [ "$BUILD_TYPE" = "release-with-debug" ]; then
    CARGO_FLAGS="--profile release-with-debug"
fi

if [ -n "$TARGET" ]; then
    CARGO_FLAGS="$CARGO_FLAGS --target $TARGET"
fi

if [ "$FEATURES" != "default" ]; then
    CARGO_FLAGS="$CARGO_FLAGS --features $FEATURES"
fi

# Clean previous builds (optional)
if [ "${CLEAN:-false}" = "true" ]; then
    echo "Cleaning previous builds..."
    cargo clean
fi

# Build
echo "Running cargo build $CARGO_FLAGS"
cargo build $CARGO_FLAGS

# Run tests
if [ "${RUN_TESTS:-true}" = "true" ]; then
    echo "Running tests..."
    cargo test $CARGO_FLAGS
fi

# Generate documentation
if [ "${GEN_DOCS:-false}" = "true" ]; then
    echo "Generating documentation..."
    cargo doc --no-deps $CARGO_FLAGS
fi

echo "Build complete!"
```

### Docker Multi-Stage Build

```dockerfile
# Dockerfile
FROM rust:1.75-slim as builder

# Install dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy manifests
COPY Cargo.toml Cargo.lock ./
COPY crates/ crates/

# Build dependencies (cached layer)
RUN cargo build --release --locked

# Build application
RUN cargo build --release --locked --features production

# Runtime stage
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy binary from builder
COPY --from=builder /app/target/release/neural-trader /usr/local/bin/

# Create non-root user
RUN useradd -m -u 1000 trader && \
    chown -R trader:trader /app

USER trader

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD ["/usr/local/bin/neural-trader", "health"]

EXPOSE 8080

ENTRYPOINT ["/usr/local/bin/neural-trader"]
CMD ["serve"]
```

### GitHub Actions CI/CD

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

env:
  CARGO_TERM_COLOR: always
  RUST_BACKTRACE: 1

jobs:
  test:
    name: Test
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        rust: [stable, beta]
    steps:
      - uses: actions/checkout@v3

      - name: Install Rust
        uses: dtolnay/rust-toolchain@master
        with:
          toolchain: ${{ matrix.rust }}
          components: rustfmt, clippy

      - name: Cache cargo registry
        uses: actions/cache@v3
        with:
          path: ~/.cargo/registry
          key: ${{ runner.os }}-cargo-registry-${{ hashFiles('**/Cargo.lock') }}

      - name: Cache cargo index
        uses: actions/cache@v3
        with:
          path: ~/.cargo/git
          key: ${{ runner.os }}-cargo-git-${{ hashFiles('**/Cargo.lock') }}

      - name: Cache cargo build
        uses: actions/cache@v3
        with:
          path: target
          key: ${{ runner.os }}-cargo-build-${{ hashFiles('**/Cargo.lock') }}

      - name: Check formatting
        run: cargo fmt --all -- --check

      - name: Run clippy
        run: cargo clippy --all-features -- -D warnings

      - name: Run tests
        run: cargo test --all-features --verbose

      - name: Run doc tests
        run: cargo test --doc --all-features

  build:
    name: Build
    runs-on: ubuntu-latest
    needs: test
    steps:
      - uses: actions/checkout@v3

      - name: Install Rust
        uses: dtolnay/rust-toolchain@stable

      - name: Build release
        run: cargo build --release --all-features

      - name: Upload artifacts
        uses: actions/upload-artifact@v3
        with:
          name: neural-trader-linux
          path: target/release/neural-trader

  benchmark:
    name: Benchmark
    runs-on: ubuntu-latest
    needs: test
    steps:
      - uses: actions/checkout@v3

      - name: Install Rust
        uses: dtolnay/rust-toolchain@stable

      - name: Run benchmarks
        run: cargo bench --no-fail-fast

      - name: Store benchmark results
        uses: benchmark-action/github-action-benchmark@v1
        with:
          tool: 'cargo'
          output-file-path: target/criterion/output.txt
          github-token: ${{ secrets.GITHUB_TOKEN }}
          auto-push: true

  docker:
    name: Docker Build
    runs-on: ubuntu-latest
    needs: build
    steps:
      - uses: actions/checkout@v3

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v2

      - name: Login to Docker Hub
        uses: docker/login-action@v2
        with:
          username: ${{ secrets.DOCKER_USERNAME }}
          password: ${{ secrets.DOCKER_PASSWORD }}

      - name: Build and push
        uses: docker/build-push-action@v4
        with:
          context: .
          push: true
          tags: neuraltrader/neural-trader:latest
          cache-from: type=registry,ref=neuraltrader/neural-trader:buildcache
          cache-to: type=registry,ref=neuraltrader/neural-trader:buildcache,mode=max
```

## Makefile for Common Tasks

```makefile
# Makefile
.PHONY: help build test lint clean docker

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

build: ## Build the project (release mode)
	cargo build --release --all-features

build-dev: ## Build the project (debug mode)
	cargo build --all-features

test: ## Run tests
	cargo test --all-features --verbose

test-unit: ## Run unit tests only
	cargo test --lib --all-features

test-integration: ## Run integration tests only
	cargo test --test '*' --all-features

bench: ## Run benchmarks
	cargo bench --all-features

lint: ## Run linting checks
	cargo fmt --all -- --check
	cargo clippy --all-features -- -D warnings

fmt: ## Format code
	cargo fmt --all

clean: ## Clean build artifacts
	cargo clean

doc: ## Generate documentation
	cargo doc --all-features --no-deps --open

docker: ## Build Docker image
	docker build -t neural-trader:latest .

docker-run: ## Run Docker container
	docker run -p 8080:8080 --env-file .env neural-trader:latest

install: ## Install binaries
	cargo install --path . --locked

cross-compile: ## Cross-compile for all targets
	cross build --release --target x86_64-unknown-linux-musl
	cross build --release --target aarch64-unknown-linux-gnu
	cross build --release --target x86_64-apple-darwin
	cross build --release --target aarch64-apple-darwin

napi-build: ## Build Node.js native module
	cd crates/nt-napi && npm run build

all: lint test build doc ## Run all checks and build
```

## Performance Optimization Flags

### CPU-Specific Optimizations

```bash
# For Intel CPUs
export RUSTFLAGS="-C target-cpu=native -C target-feature=+avx2,+fma"

# For AMD CPUs
export RUSTFLAGS="-C target-cpu=native -C target-feature=+avx2,+fma"

# For Apple Silicon
export RUSTFLAGS="-C target-cpu=native"

# Link-time optimization (LTO)
export CARGO_PROFILE_RELEASE_LTO=true

# Code generation units (lower = better optimization, slower compile)
export CARGO_PROFILE_RELEASE_CODEGEN_UNITS=1

# Panic strategy (smaller binary)
export CARGO_PROFILE_RELEASE_PANIC=abort
```

### Build Time Optimization

```toml
# .cargo/config.toml
[build]
rustflags = ["-C", "link-arg=-fuse-ld=lld"]  # Use lld linker (faster)

# Use sccache for caching
[build]
rustc-wrapper = "sccache"
```

---

## Summary

This architecture provides:

1. **Modular Design** - Clear boundaries between components
2. **Performance** - 10x improvement over Python
3. **Type Safety** - Compile-time guarantees
4. **Interoperability** - Multiple Node.js integration strategies
5. **Observability** - Comprehensive monitoring
6. **Security** - Defense-in-depth approach
7. **Scalability** - Horizontal and vertical scaling
8. **Maintainability** - Well-documented and tested

**Total Documentation:** 9 comprehensive files covering all aspects of the Rust architecture.
