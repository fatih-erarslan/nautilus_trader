# Neuro-Divergent Multi-Platform Build Guide

## Overview

This guide covers building the Neuro-Divergent NAPI binaries for all supported platforms. The Neuro-Divergent package provides 27+ neural forecasting models for time series prediction with Node.js bindings.

## Supported Platforms

| Platform | Target Triple | Binary Name | Notes |
|----------|--------------|-------------|-------|
| **Linux x64 (glibc)** | `x86_64-unknown-linux-gnu` | `neuro-divergent.linux-x64-gnu.node` | Standard Linux |
| **Linux x64 (musl)** | `x86_64-unknown-linux-musl` | `neuro-divergent.linux-x64-musl.node` | Alpine Linux |
| **Linux ARM64** | `aarch64-unknown-linux-gnu` | `neuro-divergent.linux-arm64-gnu.node` | ARM servers |
| **macOS Intel** | `x86_64-apple-darwin` | `neuro-divergent.darwin-x64.node` | Intel Macs |
| **macOS Apple Silicon** | `aarch64-apple-darwin` | `neuro-divergent.darwin-arm64.node` | M1/M2/M3 |
| **Windows x64** | `x86_64-pc-windows-msvc` | `neuro-divergent.win32-x64-msvc.node` | Windows |

## Prerequisites

### All Platforms
- **Rust** (1.70+): https://rustup.rs/
- **Node.js** (16+): https://nodejs.org/
- **npm** or **yarn**

### Linux-Specific
For ARM64 cross-compilation:
```bash
sudo apt-get update
sudo apt-get install -y gcc-aarch64-linux-gnu g++-aarch64-linux-gnu
```

For musl builds:
```bash
# Docker required
docker --version
```

### macOS-Specific
- **Xcode Command Line Tools**:
  ```bash
  xcode-select --install
  ```

### Windows-Specific
- **Visual Studio 2019+** with C++ tools
- **Windows SDK**

## Quick Start

### Build All Platforms (Automated)

```bash
cd neural-trader-rust/crates/neuro-divergent
npm install
npm run build:all
```

This script will:
1. Install all Rust targets
2. Build binaries for all 6 platforms
3. Copy binaries to `artifacts/` directory
4. Verify binary sizes
5. Test binaries (on compatible platforms)

### Platform-Specific Builds

#### Linux Only
```bash
npm run build:linux
```

Builds:
- x86_64-unknown-linux-gnu
- x86_64-unknown-linux-musl (requires Docker)
- aarch64-unknown-linux-gnu (cross-compile)

#### macOS Only
```bash
npm run build:macos
```

Builds:
- x86_64-apple-darwin
- aarch64-apple-darwin
- darwin-universal (fat binary)

#### Windows Only
```powershell
npm run build:windows
```

Builds:
- x86_64-pc-windows-msvc

## Manual Build Process

### 1. Install Rust Target

```bash
rustup target add <target-triple>
```

Example:
```bash
rustup target add x86_64-unknown-linux-gnu
```

### 2. Install Dependencies

```bash
cd neural-trader-rust/crates/neuro-divergent
npm install
```

### 3. Build Binary

```bash
npm run build -- --target <target-triple> --release --strip
```

Example:
```bash
npm run build -- --target x86_64-apple-darwin --release --strip
```

### 4. Locate Binary

Binary will be at:
```
neural-trader-rust/crates/neuro-divergent/target/<target>/release/neuro-divergent.node
```

## Build Scripts Reference

### Shell Scripts (Linux/macOS)

Located in `scripts/`:

- **`build-neuro-divergent-all.sh`**: Build all platforms
- **`build-neuro-divergent-linux.sh`**: Linux platforms only
- **`build-neuro-divergent-macos.sh`**: macOS platforms only

Usage:
```bash
bash scripts/build-neuro-divergent-all.sh
```

### PowerShell Script (Windows)

Located in `scripts/`:

- **`build-neuro-divergent-windows.ps1`**: Windows x64 build

Usage:
```powershell
powershell -ExecutionPolicy Bypass -File scripts/build-neuro-divergent-windows.ps1
```

## GitHub Actions CI/CD

Workflow file: `.github/workflows/build-neuro-divergent.yml`

### Triggers
- Push to `main`, `develop`, `rust-port` branches
- Pull requests to `main`
- Tags matching `neuro-divergent-v*`
- Manual workflow dispatch

### Jobs

1. **Build**: Parallel builds for all 6 platforms
2. **Test**: Run integration tests on native platforms
3. **Release**: Create GitHub release (on tag)
4. **Publish**: Publish to npm (on tag)

### Artifacts

Build artifacts are uploaded with 7-day retention:
- `neuro-divergent-linux-x64-gnu`
- `neuro-divergent-linux-x64-musl`
- `neuro-divergent-linux-arm64-gnu`
- `neuro-divergent-darwin-x64`
- `neuro-divergent-darwin-arm64`
- `neuro-divergent-win32-x64-msvc`

## Cross-Compilation

### Linux ARM64 (from x64)

1. Install cross-compilation tools:
   ```bash
   sudo apt-get install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu
   ```

2. Set environment variables:
   ```bash
   export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc
   export CC_aarch64_unknown_linux_gnu=aarch64-linux-gnu-gcc
   export CXX_aarch64_unknown_linux_gnu=aarch64-linux-gnu-g++
   ```

3. Build:
   ```bash
   npm run build -- --target aarch64-unknown-linux-gnu --release
   ```

### macOS ARM64 (from Intel)

```bash
rustup target add aarch64-apple-darwin
npm run build -- --target aarch64-apple-darwin --release
```

### musl (using Docker)

```bash
docker run --rm \
  -v $(pwd):/work \
  -w /work/neural-trader-rust/crates/neuro-divergent \
  rust:alpine \
  sh -c "
    apk add --no-cache nodejs npm musl-dev &&
    rustup target add x86_64-unknown-linux-musl &&
    npm install &&
    npm run build -- --target x86_64-unknown-linux-musl --release
  "
```

## Binary Verification

### Check Binary Size

```bash
ls -lh artifacts/*/native/*.node
```

Target: < 10MB per binary (20MB acceptable)

### Test Binary Loading

```bash
cd neural-trader-rust/crates/neuro-divergent
node -e "
  try {
    const binding = require('./index.js');
    console.log('✓ Binary loaded');
    console.log('Platform:', binding.platformInfo());
    console.log('Models:', binding.listModels());
  } catch(e) {
    console.error('✗ Failed:', e.message);
    process.exit(1);
  }
"
```

### Run Integration Tests

```bash
npm run test:node
```

## Troubleshooting

### "Binary not found" Error

**Cause**: Build failed or binary in wrong location

**Solution**:
1. Check build output for errors
2. Verify target is installed: `rustup target list --installed`
3. Re-run build with verbose output: `npm run build -- --target <target> --release -vv`

### "Cannot load binary" Error

**Cause**: Binary architecture mismatch or missing dependencies

**Solution**:
1. Verify architecture: `file path/to/binary.node`
2. Check dependencies: `ldd path/to/binary.node` (Linux) or `otool -L path/to/binary.node` (macOS)
3. Ensure platform matches: Don't load ARM binary on x64

### musl Build Fails

**Cause**: Docker not available or musl-dev not installed

**Solution**:
1. Ensure Docker is running: `docker ps`
2. Use Alpine Rust image: `rust:alpine`
3. Install musl-dev inside container

### Large Binary Size

**Cause**: Debug symbols not stripped

**Solution**:
1. Use `--strip` flag: `npm run build -- --target <target> --release --strip`
2. Manual strip: `strip path/to/binary.node`

### ARM64 Cross-Compile Fails

**Cause**: Missing cross-compilation toolchain

**Solution**:
1. Install ARM64 GCC: `sudo apt-get install gcc-aarch64-linux-gnu`
2. Set linker environment variables (see Cross-Compilation section)

## Binary Size Optimization

### 1. Enable LTO (Link-Time Optimization)

Add to `Cargo.toml`:
```toml
[profile.release]
lto = true
codegen-units = 1
```

### 2. Strip Symbols

```bash
strip -s path/to/binary.node
```

### 3. Use opt-level

```toml
[profile.release]
opt-level = "z"  # Optimize for size
```

### 4. Exclude Unused Features

Build with minimal features:
```bash
npm run build -- --target <target> --release --no-default-features --features cpu
```

## Continuous Integration

### Local Testing

Before pushing:
```bash
# Run all builds
npm run build:all

# Run tests
npm run test
npm run test:node

# Verify artifacts
ls -lh artifacts/*/native/*.node
```

### GitHub Actions

Workflow automatically:
1. Builds on push/PR
2. Runs tests on native platforms
3. Creates release on tag
4. Publishes to npm on tag

## Publishing to npm

### Prerequisites
- npm account with publish permissions
- `NPM_TOKEN` secret configured in GitHub

### Manual Publish

1. Build all platforms:
   ```bash
   npm run build:all
   ```

2. Prepare package:
   ```bash
   npm run prepublishOnly
   ```

3. Publish:
   ```bash
   npm publish --access public
   ```

### Automated Publish (GitHub Actions)

1. Tag release:
   ```bash
   git tag neuro-divergent-v0.1.0
   git push origin neuro-divergent-v0.1.0
   ```

2. GitHub Actions will:
   - Build all platforms
   - Run tests
   - Create GitHub release
   - Publish to npm

## Binary Distribution

### npm Package Structure

```
@neural-trader/neuro-divergent/
├── index.js                          # Platform detection
├── index.d.ts                        # TypeScript definitions
└── native/
    ├── neuro-divergent.linux-x64-gnu.node
    ├── neuro-divergent.linux-x64-musl.node
    ├── neuro-divergent.linux-arm64-gnu.node
    ├── neuro-divergent.darwin-x64.node
    ├── neuro-divergent.darwin-arm64.node
    └── neuro-divergent.win32-x64-msvc.node
```

### Platform Detection

`index.js` automatically loads the correct binary for the current platform:

```javascript
const binding = require('@neural-trader/neuro-divergent');
// Automatically loads platform-specific .node file
```

## Available Neural Models

The package includes 27+ state-of-the-art neural forecasting models:

### Basic Models
- MLP, DLinear, NLinear, MLPMultivariate

### Recurrent Models
- RNN, LSTM, GRU

### Advanced Models
- NBEATS, NBEATSx, NHITS, TiDE

### Transformer Models
- TFT, Informer, AutoFormer, FedFormer, PatchTST, ITransformer

### Specialized Models
- DeepAR, DeepNPTS, TCN, BiTCN, TimesNet, StemGNN, TSMixer, TimeLLM

## Performance Benchmarks

Target performance for each platform:

- **Build Time**: < 10 minutes per platform
- **Binary Size**: < 10MB (20MB acceptable)
- **Load Time**: < 100ms
- **Inference**: < 10ms per prediction

## Support

For build issues:
1. Check this guide
2. Review GitHub Actions logs
3. Open issue: https://github.com/ruvnet/neural-trader/issues

## License

MIT OR Apache-2.0
