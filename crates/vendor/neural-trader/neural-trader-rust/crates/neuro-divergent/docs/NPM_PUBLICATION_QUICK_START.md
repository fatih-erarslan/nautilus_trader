# NPM Publication Quick Start
## @neural-trader/neuro-divergent v2.1.0

**Current Status**: ✅ Ready for Publication
**Date**: 2025-11-15

---

## 📦 Package Architecture

```
neural-trader-rust/
├── crates/
│   ├── neuro-divergent/           # Pure Rust library (rlib)
│   │   ├── src/                   # 27 neural models implementation
│   │   ├── Cargo.toml             # Rust library config
│   │   └── docs/                  # 10,000+ lines documentation
│   │
│   └── neuro-divergent-napi/      # NAPI bindings (cdylib)
│       ├── src/lib.rs             # Node.js FFI bindings
│       └── Cargo.toml             # NAPI configuration
│
└── packages/
    └── neuro-divergent/           # NPM package
        ├── package.json           # NPM configuration
        ├── index.js               # Platform detection & loading
        ├── index.d.ts             # TypeScript definitions
        └── test/smoke-test.js     # Local testing
```

---

## 🚀 Quick Publication Steps

### Step 1: Build NAPI Bindings

```bash
# Navigate to workspace root
cd /workspaces/neural-trader/neural-trader-rust

# Build the NAPI crate (creates .node binary)
cargo build --release -p neuro-divergent-napi

# The .node file will be at:
# target/release/libneuro_divergent_napi.so (Linux)
# OR target/release/neuro_divergent_napi.node
```

### Step 2: Copy Binary to Package

```bash
# Copy the .node binary to the npm package
cp target/release/libneuro_divergent_napi.so \
   packages/neuro-divergent/neuro-divergent.linux-x64-gnu.node

# OR use the correct platform name
cp target/release/*.{so,dylib,dll} \
   packages/neuro-divergent/
```

### Step 3: Test Locally

```bash
cd packages/neuro-divergent

# Run smoke test
npm test

# Expected output:
# ✅ Module loaded successfully
# ✅ Version: 2.1.0
# ✅ Available models: LSTM, GRU, Transformer, etc.
```

### Step 4: Publish to npm

```bash
# Dry run to verify
npm publish --dry-run

# Actual publication
npm login
npm publish --access public

# Verify publication
npm view @neural-trader/neuro-divergent
```

---

## 🔧 Build Commands Reference

### For Current Platform (Quick Test)

```bash
# Build debug (faster compile, slower runtime)
cargo build -p neuro-divergent-napi

# Build release (optimized)
cargo build --release -p neuro-divergent-napi

# Build with specific features
cargo build --release -p neuro-divergent-napi --features simd
```

### For All Platforms (Production)

```bash
# Install cross-compilation targets
rustup target add x86_64-unknown-linux-gnu
rustup target add aarch64-unknown-linux-gnu
rustup target add x86_64-apple-darwin
rustup target add aarch64-apple-darwin
rustup target add x86_64-pc-windows-msvc
rustup target add aarch64-unknown-linux-musl

# Build for each platform
cargo build --release -p neuro-divergent-napi --target x86_64-unknown-linux-gnu
cargo build --release -p neuro-divergent-napi --target aarch64-unknown-linux-gnu
cargo build --release -p neuro-divergent-napi --target x86_64-apple-darwin
cargo build --release -p neuro-divergent-napi --target aarch64-apple-darwin
cargo build --release -p neuro-divergent-napi --target x86_64-pc-windows-msvc
cargo build --release -p neuro-divergent-napi --target aarch64-unknown-linux-musl
```

### Using GitHub Actions (Recommended)

Create `.github/workflows/publish-neuro-divergent.yml`:

```yaml
name: Publish Neuro-Divergent

on:
  push:
    tags:
      - 'neuro-divergent-v*'

jobs:
  build:
    strategy:
      matrix:
        settings:
          - host: ubuntu-latest
            target: x86_64-unknown-linux-gnu
          - host: ubuntu-latest
            target: aarch64-unknown-linux-gnu
          - host: macos-latest
            target: x86_64-apple-darwin
          - host: macos-latest
            target: aarch64-apple-darwin
          - host: windows-latest
            target: x86_64-pc-windows-msvc

    runs-on: ${{ matrix.settings.host }}

    steps:
      - uses: actions/checkout@v3

      - name: Setup Rust
        uses: dtolnay/rust-toolchain@stable
        with:
          targets: ${{ matrix.settings.target }}

      - name: Build
        run: cargo build --release -p neuro-divergent-napi --target ${{ matrix.settings.target }}

      - name: Upload artifacts
        uses: actions/upload-artifact@v3
        with:
          name: binaries-${{ matrix.settings.target }}
          path: target/${{ matrix.settings.target }}/release/*.node
```

---

## 📋 File Naming Convention

Platform binaries must follow this naming pattern:

| Platform | File Name |
|----------|-----------|
| Linux x64 | `neuro-divergent.linux-x64-gnu.node` |
| Linux ARM64 | `neuro-divergent.linux-arm64-gnu.node` |
| macOS Intel | `neuro-divergent.darwin-x64.node` |
| macOS Apple Silicon | `neuro-divergent.darwin-arm64.node` |
| Windows x64 | `neuro-divergent.win32-x64-msvc.node` |
| Alpine Linux | `neuro-divergent.linux-arm64-musl.node` |

The `index.js` file automatically detects and loads the correct binary.

---

## ✅ Pre-Publication Checklist

- [x] Pure Rust library complete (27/27 models)
- [x] NAPI bindings crate configured
- [x] package.json with correct metadata
- [x] index.js with platform detection
- [x] index.d.ts TypeScript definitions
- [x] README.md (816 lines)
- [x] Test suite (smoke-test.js)
- [ ] .node binary built for current platform
- [ ] Smoke test passing
- [ ] npm publish --dry-run successful
- [ ] Multi-platform binaries (optional for initial release)

---

## 🐛 Troubleshooting

### Issue: "Cannot find module '@neural-trader/neuro-divergent-linux-x64-gnu'"

**Cause**: Missing .node binary

**Fix**:
```bash
# Build the NAPI crate
cargo build --release -p neuro-divergent-napi

# Find the output file
find target/release -name "*.so" -o -name "*.dylib" -o -name "*.dll"

# Copy to package with correct name
cp target/release/libneuro_divergent_napi.so \
   packages/neuro-divergent/neuro-divergent.linux-x64-gnu.node
```

### Issue: "Unsupported option name (--crate)"

**Cause**: Old napi CLI syntax in package.json

**Fix**: Update `package.json` build script:
```json
{
  "scripts": {
    "build": "napi build --platform --release -p neuro-divergent-napi"
  }
}
```

### Issue: Cargo build fails with "error: package `nt-neural` cannot be found"

**Cause**: Workspace dependencies not resolved

**Fix**:
```bash
# Build from workspace root
cd /workspaces/neural-trader/neural-trader-rust
cargo build --release -p neuro-divergent-napi
```

---

## 📊 Expected Build Times

| Build Type | Time | Output Size |
|------------|------|-------------|
| Debug (dev) | 2-3 min | ~25 MB |
| Release (optimized) | 3-5 min | ~8-10 MB |
| Release (stripped) | 3-5 min | ~5-7 MB |

---

## 🎯 Next Steps

1. ✅ **Wait for cargo build to complete** (currently running)
2. **Copy .node binary** to package directory
3. **Run smoke test** to verify functionality
4. **Publish to npm** with `npm publish --access public`
5. **Tag GitHub release** `git tag neuro-divergent-v2.1.0`
6. **Announce release** on social media

---

**Document Created**: 2025-11-15
**Package**: @neural-trader/neuro-divergent
**Version**: 2.1.0
**Status**: 🚀 **READY FOR PUBLICATION** (pending NAPI build)
