# NPM Publishing Guide for Neural Trader

Complete guide for building, testing, and publishing the Neural Trader Rust port to NPM.

## Overview

Neural Trader uses **napi-rs** to create native Node.js bindings from Rust. This allows JavaScript/TypeScript applications to leverage the ultra-low-latency Rust trading engine.

## Architecture

### Package Structure

```
@neural-trader/core                    # Main package
├── @neural-trader/darwin-arm64       # macOS ARM64 (Apple Silicon)
├── @neural-trader/darwin-x64         # macOS Intel
├── @neural-trader/linux-x64-gnu      # Linux x86_64 (glibc)
├── @neural-trader/linux-x64-musl     # Linux x86_64 (musl/Alpine)
└── @neural-trader/win32-x64-msvc     # Windows x86_64
```

### How It Works

1. **Main Package** (`@neural-trader/core`): Contains JavaScript entry point and TypeScript definitions
2. **Platform Packages**: Contain pre-built native `.node` bindings for each platform
3. **Runtime Loading**: `index.js` detects the current platform and loads the appropriate native addon
4. **Optional Dependencies**: Platform packages are listed as `optionalDependencies`, so only the relevant one is installed

## Prerequisites

### Development Environment

```bash
# Rust toolchain (1.70+)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Node.js (18+)
nvm install 18
nvm use 18

# napi-rs CLI
npm install -g @napi-rs/cli
```

### Cross-Compilation Setup

Run the setup script to install all cross-compilation toolchains:

```bash
chmod +x scripts/setup-cross-compile.sh
./scripts/setup-cross-compile.sh
```

This installs:
- Rust targets for all platforms
- GCC/Clang cross-compilers
- MUSL tools for static linking
- MinGW for Windows cross-compilation (on Linux)

## Building

### Local Development Build

Build for your current platform:

```bash
npm run build:debug    # Debug build (faster compile, slower runtime)
npm run build          # Release build (optimized)
```

### Platform-Specific Builds

```bash
npm run build:linux    # Linux x86_64 (GNU + MUSL)
npm run build:darwin   # macOS (x64 + ARM64)
npm run build:windows  # Windows x86_64 (MSVC)
```

### Build All Platforms

```bash
npm run build:all      # Build for all supported platforms
```

Or use the comprehensive build script:

```bash
chmod +x scripts/build-all-platforms.sh
./scripts/build-all-platforms.sh
```

## Testing

### Run Tests

```bash
npm test              # Run all tests
npm run test:watch    # Watch mode
```

### Test Native Bindings

```bash
# Test in Node.js
node -e "const nt = require('.'); console.log(nt.getVersion())"

# Test CLI
npx neural-trader --version
```

### Test Cross-Platform (Docker)

```bash
# Test Linux builds in Docker
docker run --rm -v $(pwd):/app -w /app node:18-alpine \
  sh -c "npm install && npm test"

# Test Linux MUSL
docker run --rm -v $(pwd):/app -w /app node:18-alpine \
  sh -c "npm install && node -e \"require('.').getVersion()\""
```

## Pre-Publish Validation

Run comprehensive pre-publish checks:

```bash
chmod +x scripts/publish-check.sh
./scripts/publish-check.sh
```

This validates:
- ✅ package.json validity
- ✅ Version consistency across files
- ✅ All platform packages exist
- ✅ Entry points (index.js, index.d.ts, CLI)
- ✅ Documentation (README, LICENSE)
- ✅ No hardcoded secrets
- ✅ Tests pass
- ✅ Git status is clean

## Publishing

### 1. Prepare for Publishing

```bash
# Update version (updates both package.json and Cargo.toml)
npm version patch   # 0.1.0 -> 0.1.1
npm version minor   # 0.1.0 -> 0.2.0
npm version major   # 0.1.0 -> 1.0.0
```

### 2. Build Artifacts

```bash
# Build all platforms
npm run build:all

# Prepare npm packages
npm run artifacts
```

This creates platform-specific packages in `npm/`:
- `npm/darwin-arm64/package.json` + `.node` binary
- `npm/darwin-x64/package.json` + `.node` binary
- `npm/linux-x64-gnu/package.json` + `.node` binary
- `npm/linux-x64-musl/package.json` + `.node` binary
- `npm/win32-x64-msvc/package.json` + `.node` binary

### 3. Test Package Locally

```bash
# Dry-run publishing to see what would be included
npm publish --dry-run

# Install locally to test
npm pack
npm install -g ./neural-trader-core-0.1.0.tgz

# Test installation
npx neural-trader --version
```

### 4. Publish to NPM

```bash
# Login to npm (first time only)
npm login

# Publish main package
npm publish --access public

# Publish platform packages
cd npm/darwin-arm64 && npm publish --access public && cd ../..
cd npm/darwin-x64 && npm publish --access public && cd ../..
cd npm/linux-x64-gnu && npm publish --access public && cd ../..
cd npm/linux-x64-musl && npm publish --access public && cd ../..
cd npm/win32-x64-msvc && npm publish --access public && cd ../..
```

Or use napi-rs automated publishing:

```bash
npm run prepublishOnly  # Runs napi prepublish
npm publish            # Publishes all packages
```

### 5. Verify Published Package

```bash
# Install from npm
npm install -g @neural-trader/core

# Test it works
neural-trader --version

# Check package contents
npm view @neural-trader/core
npm view @neural-trader/darwin-arm64
```

## CI/CD Setup (GitHub Actions)

Create `.github/workflows/release.yml`:

```yaml
name: Release

on:
  push:
    tags:
      - 'v*'

jobs:
  build:
    strategy:
      matrix:
        settings:
          - host: ubuntu-latest
            target: x86_64-unknown-linux-gnu
          - host: ubuntu-latest
            target: x86_64-unknown-linux-musl
          - host: macos-latest
            target: x86_64-apple-darwin
          - host: macos-latest
            target: aarch64-apple-darwin
          - host: windows-latest
            target: x86_64-pc-windows-msvc

    runs-on: ${{ matrix.settings.host }}

    steps:
      - uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: 18

      - name: Setup Rust
        uses: dtolnay/rust-toolchain@stable
        with:
          targets: ${{ matrix.settings.target }}

      - name: Install dependencies
        run: npm install

      - name: Build
        run: npm run build -- --target ${{ matrix.settings.target }}

      - name: Upload artifacts
        uses: actions/upload-artifact@v3
        with:
          name: bindings-${{ matrix.settings.target }}
          path: '*.node'

  publish:
    needs: build
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: 18
          registry-url: 'https://registry.npmjs.org'

      - name: Download artifacts
        uses: actions/download-artifact@v3

      - name: Publish to npm
        run: |
          npm run artifacts
          npm publish --access public
        env:
          NODE_AUTH_TOKEN: ${{ secrets.NPM_TOKEN }}
```

## Troubleshooting

### Native Module Not Found

```bash
Error: Cannot find module '@neural-trader/darwin-arm64'
```

**Solution**: Install optional dependencies:
```bash
npm install --include=optional
```

### Platform Not Supported

```bash
Error: Unsupported platform: linux-arm64
```

**Solution**: The platform is not yet supported. Either:
1. Build from source: `npm run build`
2. Request support: https://github.com/ruvnet/neural-trader/issues

### Cross-Compilation Fails

```bash
error: linker `x86_64-linux-gnu-gcc` not found
```

**Solution**: Install cross-compilation tools:
```bash
./scripts/setup-cross-compile.sh
```

### Version Mismatch

```bash
Error: Version mismatch between package.json and Cargo.toml
```

**Solution**: Use npm version command to sync:
```bash
npm version patch  # Updates both files
```

## Best Practices

### 1. Semantic Versioning

- **MAJOR**: Breaking API changes
- **MINOR**: New features, backward compatible
- **PATCH**: Bug fixes

### 2. Testing Before Publishing

Always test on multiple platforms:
- Linux (Ubuntu, Alpine)
- macOS (Intel, Apple Silicon)
- Windows (x64)

### 3. Security

- Never commit API keys or secrets
- Run `npm audit` before publishing
- Use `.npmignore` to exclude sensitive files

### 4. Documentation

- Keep README.md updated
- Document breaking changes in CHANGELOG.md
- Update TypeScript definitions (`index.d.ts`)

### 5. Performance

- Always use release builds for production (`npm run build`)
- Enable LTO (Link-Time Optimization) in Cargo.toml
- Profile critical paths with Rust benchmarks

## Advanced Topics

### Custom Native Module Path

Override the module loading logic:

```javascript
process.env.NEURAL_TRADER_NATIVE_PATH = '/custom/path/neural-trader.node';
const neuralTrader = require('@neural-trader/core');
```

### Fallback for Unsupported Platforms

Implement a JavaScript fallback:

```javascript
let nativeAddon;
try {
  nativeAddon = require('@neural-trader/core');
} catch (err) {
  // Use WebAssembly fallback or pure JS implementation
  nativeAddon = require('./fallback.js');
}
```

### Debugging Native Code

```bash
# Build with debug symbols
npm run build:debug

# Run with RUST_BACKTRACE
RUST_BACKTRACE=1 node your-script.js

# Use lldb (macOS) or gdb (Linux)
lldb -- node your-script.js
```

## Resources

- **napi-rs Documentation**: https://napi.rs
- **Rust FFI Guide**: https://doc.rust-lang.org/nomicon/ffi.html
- **Node.js Native Addons**: https://nodejs.org/api/addons.html
- **Neural Trader Issues**: https://github.com/ruvnet/neural-trader/issues

## Support

For questions or issues:
- GitHub Issues: https://github.com/ruvnet/neural-trader/issues
- Documentation: https://neural-trader.io
- Discord: https://discord.gg/neural-trader

---

**Happy Trading! 🚀📈**
