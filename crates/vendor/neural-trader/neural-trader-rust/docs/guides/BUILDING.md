# Building Neural Trader NAPI Bindings

This guide covers building the high-performance Rust NAPI bindings for Neural Trader across multiple platforms.

## Quick Start

### Local Development Build

```bash
# Clone repository
git clone https://github.com/ruvnet/neural-trader.git
cd neural-trader

# Install dependencies
cd neural-trader-rust/crates/napi-bindings
npm install

# Build for your platform
npm run build:release
```

### Multi-Platform Build

```bash
# Build all targets (automated)
./scripts/build-napi-all.sh

# Or platform-specific:
./scripts/build-napi-linux.sh      # Linux
./scripts/build-napi-macos.sh      # macOS
./scripts/build-napi-windows.ps1   # Windows (PowerShell)
```

## Prerequisites

### All Platforms

- **Rust**: >= 1.70.0 (install from [rustup.rs](https://rustup.rs/))
- **Node.js**: >= 16.0.0 (install from [nodejs.org](https://nodejs.org/))
- **npm**: >= 7.0.0 (comes with Node.js)

### Platform-Specific Requirements

#### Linux

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y build-essential gcc g++

# For ARM64 cross-compilation
sudo apt-get install -y gcc-aarch64-linux-gnu g++-aarch64-linux-gnu
rustup target add aarch64-unknown-linux-gnu
```

#### macOS

```bash
# Xcode Command Line Tools
xcode-select --install

# Rust targets for cross-compilation
rustup target add x86_64-apple-darwin
rustup target add aarch64-apple-darwin
```

#### Windows

```powershell
# Install Visual Studio Build Tools 2019 or later
# With "Desktop development with C++" workload
# Download from: https://visualstudio.microsoft.com/downloads/

# Install Rust from rustup.rs
# Ensure MSVC toolchain is selected
```

## Supported Platforms

| Platform | Target Triple | Binary Name | Status |
|----------|---------------|-------------|--------|
| **Linux x64** | `x86_64-unknown-linux-gnu` | `neural-trader.linux-x64-gnu.node` | ✅ Tested |
| **Linux ARM64** | `aarch64-unknown-linux-gnu` | `neural-trader.linux-arm64-gnu.node` | ✅ Cross-compile |
| **macOS Intel** | `x86_64-apple-darwin` | `neural-trader.darwin-x64.node` | ✅ Tested |
| **macOS Apple Silicon** | `aarch64-apple-darwin` | `neural-trader.darwin-arm64.node` | ✅ Tested |
| **Windows x64** | `x86_64-pc-windows-msvc` | `neural-trader.win32-x64-msvc.node` | ✅ CI Build |

## Build Commands

### Development Builds

```bash
cd neural-trader-rust/crates/napi-bindings

# Debug build (faster compilation, slower runtime)
npm run build:debug

# Release build (optimized)
npm run build:release

# Watch mode (auto-rebuild on changes)
npm run build:watch
```

### Production Builds

```bash
# Single platform
npm run build:release

# All platforms (requires cross-compilation setup)
npm run build:all

# Generate NPM artifacts
npm run artifacts
```

### Testing

```bash
# Rust unit tests
npm run test

# Node.js integration tests
npm run test:node

# Full test suite
npm test && npm run test:node
```

## Cross-Compilation

### Linux to ARM64

```bash
# Install cross-compilation toolchain
cargo install cross

# Build for ARM64
cross build --release \
  --manifest-path=neural-trader-rust/Cargo.toml \
  --package nt-napi-bindings \
  --target aarch64-unknown-linux-gnu
```

### macOS Universal Binary

```bash
cd neural-trader-rust/crates/napi-bindings

# Build both architectures
CARGO_BUILD_TARGET=x86_64-apple-darwin npm run build:release
CARGO_BUILD_TARGET=aarch64-apple-darwin npm run build:release

# Create universal binary
npm run universal
```

### Windows Cross-Compilation (from Linux)

**Note**: Not officially supported. Use GitHub Actions for Windows builds.

```bash
# Experimental: MinGW cross-compilation
sudo apt-get install mingw-w64
rustup target add x86_64-pc-windows-gnu
cargo build --release --target x86_64-pc-windows-gnu
```

## GitHub Actions CI/CD

Automated builds are triggered on:
- **Push to main/develop/rust-port branches**
- **Pull requests to main**
- **Version tags** (v*)
- **Manual workflow dispatch**

### Workflow Features

✅ Matrix builds for all 5 platforms
✅ Automated testing on native platforms
✅ Artifact uploads with 7-day retention
✅ Automatic NPM publishing on tags
✅ GitHub Release creation with checksums
✅ Rust caching for faster builds

### Triggering a Release Build

```bash
# Create and push a version tag
git tag v2.0.1
git push origin v2.0.1

# GitHub Actions will:
# 1. Build all platforms
# 2. Run integration tests
# 3. Create GitHub Release
# 4. Publish to NPM registry
```

## Binary Sizes

Expected binary sizes (approximate):

- **Linux x64**: ~210 MB (includes debug symbols)
- **Linux ARM64**: ~195 MB
- **macOS x64**: ~180 MB
- **macOS ARM64**: ~175 MB
- **Windows x64**: ~190 MB

### Reducing Binary Size

```bash
# Strip debug symbols
cargo build --release
strip target/release/*.node

# Enable LTO (Link-Time Optimization)
# Add to Cargo.toml:
[profile.release]
lto = true
codegen-units = 1
```

## Troubleshooting

### Binary Not Found

```bash
# Check build output
ls -lah neural-trader-rust/target/*/release/*.node

# Verify target triple
rustc --version --verbose | grep host
```

### Linking Errors (Linux)

```bash
# Install missing libraries
sudo apt-get install -y libssl-dev pkg-config

# For ARM64 cross-compilation
export CC_aarch64_unknown_linux_gnu=aarch64-linux-gnu-gcc
export CXX_aarch64_unknown_linux_gnu=aarch64-linux-gnu-g++
```

### NAPI Version Mismatch

```bash
# Verify Node.js version
node --version  # Should be >= 16

# Check NAPI compatibility
node -p "process.versions.napi"
```

### macOS Code Signing (for distribution)

```bash
# Sign binary
codesign --sign "Developer ID" \
  --force --deep --timestamp \
  neural-trader.darwin-x64.node

# Verify signature
codesign --verify --verbose neural-trader.darwin-x64.node
```

### Windows MSVC Not Found

```powershell
# Ensure Visual Studio Build Tools installed
# Set environment variables
$env:VSINSTALLDIR = "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools"
```

## Performance Benchmarks

| Operation | Rust NAPI | JavaScript | Speedup |
|-----------|-----------|------------|---------|
| Backtest (1M bars) | 127ms | 4,830ms | **38x** |
| Technical indicators | 8ms | 215ms | **27x** |
| Portfolio optimization | 45ms | 1,290ms | **29x** |
| JSON serialization | 12ms | 89ms | **7.4x** |

## Package Structure

```
packages/
├── linux-x64-gnu/
│   └── native/
│       └── neural-trader.linux-x64-gnu.node
├── linux-arm64-gnu/
│   └── native/
│       └── neural-trader.linux-arm64-gnu.node
├── darwin-x64/
│   └── native/
│       └── neural-trader.darwin-x64.node
├── darwin-arm64/
│   └── native/
│       └── neural-trader.darwin-arm64.node
└── win32-x64-msvc/
    └── native/
        └── neural-trader.win32-x64-msvc.node
```

## Development Tips

### Incremental Builds

```bash
# Only rebuild changed crates
cargo build --release -p nt-napi-bindings

# Use sccache for faster rebuilds
cargo install sccache
export RUSTC_WRAPPER=sccache
```

### Debugging NAPI Bindings

```javascript
// Enable NAPI debug output
process.env.NAPI_DEBUG = '1';
const neuraltrade = require('./index.js');

// Trace calls
process.env.RUST_LOG = 'debug';
```

### Profile-Guided Optimization

```bash
# Step 1: Build with instrumentation
RUSTFLAGS="-Cprofile-generate=/tmp/pgo-data" \
  cargo build --release

# Step 2: Run benchmarks to collect data
./target/release/benchmarks

# Step 3: Rebuild with profile data
RUSTFLAGS="-Cprofile-use=/tmp/pgo-data/merged.profdata" \
  cargo build --release
```

## Publishing Checklist

Before publishing a new release:

- [ ] All tests passing (`cargo test && npm test`)
- [ ] Version bumped in `Cargo.toml` and `package.json`
- [ ] Changelog updated (`CHANGELOG.md`)
- [ ] Documentation reviewed
- [ ] License headers present
- [ ] Security audit clean (`cargo audit`)
- [ ] Binaries built for all platforms
- [ ] Integration tests passing on all platforms
- [ ] README examples working
- [ ] GitHub tag created

## Support

- **Documentation**: [docs.rs/nt-napi-bindings](https://docs.rs/nt-napi-bindings)
- **Issues**: [GitHub Issues](https://github.com/ruvnet/neural-trader/issues)
- **Discussions**: [GitHub Discussions](https://github.com/ruvnet/neural-trader/discussions)

## License

Dual-licensed under MIT OR Apache-2.0. See [LICENSE-MIT](../../LICENSE-MIT) and [LICENSE-APACHE](../../LICENSE-APACHE) for details.
