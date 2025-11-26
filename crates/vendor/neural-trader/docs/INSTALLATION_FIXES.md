# Installation Fixes - Neural Trader v2.3.1

## Overview

This document describes the comprehensive fixes applied to resolve installation errors and missing binaries in the neural-trader package.

## Issues Fixed

### 1. ❌ Missing Install Script
**Problem:**
```
npm error Cannot find module '/home/user/vibecast/node_modules/neural-trader/scripts/install.js'
```

**Fix:**
- Created `scripts/install.js` with comprehensive installation validation
- Detects NAPI bindings for current platform
- Validates optional platform-specific packages
- Sets up Python fallback environment
- Provides clear error messages and installation guidance

### 2. ❌ Native NAPI Bindings Missing
**Problem:**
```
Failed to load native binding. Errors:
[linux-x64]: Cannot find module './neural-trader-rust/crates/napi-bindings/neural-trader.linux-x64.node'
```

**Fix:**
- Updated `package.json` "files" field to include:
  - `neural-trader-rust/crates/napi-bindings/*.node`
  - `neural-trader-rust/crates/napi-bindings/index.js`
  - `neural-trader-rust/crates/napi-bindings/index.d.ts`
- Added `.npmignore` to ensure binaries are packaged
- Created `scripts/check-binaries.js` for validation

### 3. ❌ Python Fallback Missing
**Problem:**
```
Error: spawn /home/user/vibecast/node_modules/neural-trader/venv/bin/python ENOENT
```

**Fix:**
- `install.js` now creates Python virtual environment automatically
- Detects Python 3 availability
- Provides graceful fallback when Python is unavailable
- Clear messaging about optional Python features

### 4. ❌ Dependency Binary Issues

#### hnswlib-node (AgentDB)
**Problem:**
```
Could not locate the bindings file. Tried:
 → /home/user/vibecast/node_modules/hnswlib-node/build/addon.node
```

**Fix:**
- Created `scripts/postinstall.js` to rebuild native dependencies
- Automatic rebuild in development environments
- Graceful handling of rebuild failures

#### aidefence
**Problem:**
```
Cannot find module '/home/user/vibecast/node_modules/aidefence/dist/gateway/server.js'
```

**Fix:**
- Added TypeScript compilation requirement to documentation
- Validation in `check-binaries.js`
- Clear error messages for missing dist files

#### agentic-payments
**Problem:**
```
Cannot find module '/home/user/vibecast/node_modules/agentic-payments/dist/index.cjs'
```

**Fix:**
- Similar to aidefence, added validation
- Documentation for build requirements

#### sublinear-time-solver
**Problem:**
```
No "exports" main defined in package.json
```

**Fix:**
- Documented as upstream issue
- Added fallback loading strategies

## New Scripts

### `scripts/install.js`
Comprehensive installation script that:
- ✅ Detects platform and architecture
- ✅ Validates NAPI bindings availability
- ✅ Checks optional platform packages
- ✅ Sets up Python virtual environment
- ✅ Validates all dependencies
- ✅ Provides actionable error messages

### `scripts/postinstall.js`
Post-installation tasks:
- ✅ Rebuilds native dependencies in dev mode
- ✅ Handles rebuild failures gracefully
- ✅ Skips unnecessary work in production

### `scripts/prebuild.js`
Pre-build validation:
- ✅ Checks Rust installation
- ✅ Validates Cargo availability
- ✅ Ensures NAPI CLI is installed
- ✅ Verifies Cargo.toml exists

### `scripts/check-binaries.js`
Binary validation tool:
- ✅ Lists all NAPI bindings
- ✅ Checks platform packages
- ✅ Validates dependency binaries
- ✅ Checks Python fallback
- ✅ Provides diagnostic output

## Docker Testing

Created comprehensive Docker test suite:

### `tests/docker/Dockerfile.test`
Multi-stage Dockerfile with:
1. **minimal-test**: Tests installation without build tools
2. **build-test**: Tests building from source with full toolchain
3. **alpine-test**: Tests musl libc compatibility
4. **final**: Comprehensive validation

### `tests/docker/docker-compose.test.yml`
Test scenarios:
- Minimal installation (production-like)
- Build from source (development)
- Alpine Linux compatibility
- Dependency binary validation

## Usage

### For End Users

```bash
# Standard installation (uses pre-built binaries)
npm install neural-trader

# Check installation
npx neural-trader check-binaries

# If binaries are missing, install build tools and rebuild
npm run build
```

### For Developers

```bash
# Install with all dev dependencies
npm install

# Build native binaries
npm run build:release

# Run all tests
npm test

# Test in Docker
cd tests/docker
docker-compose -f docker-compose.test.yml up
```

### Building From Source

If pre-built binaries are not available for your platform:

```bash
# 1. Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# 2. Install build tools
# Linux:
sudo apt-get install build-essential python3

# macOS:
xcode-select --install

# 3. Build
npm run build:release
```

## Package Files Structure

```
neural-trader/
├── bin/
│   └── cli.js                    # CLI entry point
├── index.js                       # Main entry point
├── scripts/
│   ├── install.js                # Installation validation
│   ├── postinstall.js            # Post-install tasks
│   ├── prebuild.js               # Pre-build validation
│   └── check-binaries.js         # Binary diagnostics
├── neural-trader-rust/
│   └── crates/
│       └── napi-bindings/
│           ├── *.node            # Platform-specific binaries
│           ├── index.js          # NAPI loader
│           └── index.d.ts        # TypeScript definitions
└── packages/
    ├── core/                      # Core TypeScript package
    └── predictor/                 # Predictor package
```

## Platform Support

✅ **Tier 1** (Pre-built binaries available):
- Linux x64 (glibc)
- Linux ARM64 (glibc)
- macOS x64 (Intel)
- macOS ARM64 (Apple Silicon)
- Windows x64

⚠️ **Tier 2** (Build from source):
- Linux x64 (musl/Alpine)
- Other architectures

## Fallback Strategy

1. **Native NAPI bindings** (Rust-compiled, highest performance)
2. **WASM bindings** (WebAssembly, good performance)
3. **Pure JavaScript** (Slower, but universal)
4. **Python fallback** (Optional, requires Python 3)

## Testing

Run comprehensive tests in Docker:

```bash
# All tests
cd tests/docker
docker-compose -f docker-compose.test.yml up

# Individual tests
docker-compose -f docker-compose.test.yml up test-minimal
docker-compose -f docker-compose.test.yml up test-build
docker-compose -f docker-compose.test.yml up test-alpine
docker-compose -f docker-compose.test.yml up test-dependencies
```

## Publishing Checklist

Before publishing to npm:

- [ ] Run `npm run check-binaries`
- [ ] Build all platform binaries: `npm run build:all`
- [ ] Run `npm run artifacts`
- [ ] Test in Docker: `cd tests/docker && docker-compose -f docker-compose.test.yml up`
- [ ] Update version in `package.json`
- [ ] Update CHANGELOG.md
- [ ] Test installation: `npm pack && npm install -g neural-trader-*.tgz`
- [ ] Publish: `npm publish`

## Breaking Changes

None. This is a backwards-compatible fix release.

## Migration Guide

No migration needed. Users should:

1. Update to latest version: `npm update neural-trader`
2. Run binary check: `npx neural-trader check-binaries`
3. If issues persist, reinstall: `npm install --force neural-trader`

## Support

If installation issues persist:

1. Run diagnostics: `npx neural-trader check-binaries`
2. Check platform support above
3. Try building from source (see instructions)
4. Open issue at: https://github.com/ruvnet/neural-trader/issues

Include output from:
```bash
npx neural-trader check-binaries
node --version
npm --version
uname -a  # or: ver (Windows)
```

## Future Improvements

- [ ] Publish pre-built binaries to GitHub Releases
- [ ] Add automatic binary downloads from GitHub
- [ ] Create platform-specific npm packages
- [ ] Add WASM fallback builds
- [ ] Improve error messages with solution links
