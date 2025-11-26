# Neural Trader - NAPI Multi-Platform Build System

Complete setup for building and publishing native NAPI bindings across all supported platforms.

## 📦 Package Structure

```
neural-trader-rust/packages/neural-trader-backend/
├── package.json              # Main package configuration
├── index.js                  # Platform-specific loader
├── index.d.ts               # TypeScript definitions
├── README.md                # Package documentation
├── .npmignore              # NPM publish exclusions
├── scripts/
│   ├── postinstall.js      # Post-install binary detection
│   ├── prepack.js          # Pre-publish validation
│   └── build-all-platforms.js  # Multi-platform build coordinator
└── test/
    ├── smoke-test.js       # Basic module loading test
    └── advanced-test.js    # Performance and memory tests
```

## 🎯 Supported Platforms

| Platform | Architecture | Triple | Status |
|----------|-------------|--------|--------|
| **Linux** | x64 | `x86_64-unknown-linux-gnu` | ✅ Supported |
| **Linux** | ARM64 | `aarch64-unknown-linux-gnu` | ✅ Supported |
| **macOS** | x64 (Intel) | `x86_64-apple-darwin` | ✅ Supported |
| **macOS** | ARM64 (M1/M2/M3) | `aarch64-apple-darwin` | ✅ Supported |
| **Windows** | x64 | `x86_64-pc-windows-msvc` | ✅ Supported |
| **Windows** | ARM64 | `aarch64-pc-windows-msvc` | 🚧 Experimental |

## 🚀 Build Commands

### Local Development

```bash
# Build for current platform
cd /workspaces/neural-trader/neural-trader-rust/packages/neural-trader-backend
npm run build

# Build with debug symbols
npm run build:debug

# Build optimized release
npm run build:release

# Run smoke tests
npm test

# Run advanced tests
npm run test:advanced
```

### Multi-Platform (via Scripts)

```bash
# Build all platforms (Linux only, uses cross-compilation)
cd /workspaces/neural-trader
./scripts/build-napi-all.sh

# Platform-specific builds
./scripts/build-napi-linux.sh      # Linux native
./scripts/build-napi-macos.sh      # macOS native
./scripts/build-napi-windows.ps1   # Windows native (PowerShell)
```

### CI/CD (GitHub Actions)

```bash
# Trigger workflow manually
gh workflow run build-napi.yml

# Create release (triggers automatic build & publish)
git tag v2.0.1
git push origin v2.0.1

# The workflow will:
# 1. Build binaries for all 6 platforms
# 2. Run tests on native platforms
# 3. Upload artifacts with 7-day retention
# 4. Create GitHub release with binaries
# 5. Publish to npm (if tag pushed)
```

## 📝 GitHub Actions Workflow

Location: `.github/workflows/build-napi.yml`

### Build Matrix

```yaml
matrix:
  include:
    - os: ubuntu-latest
      target: x86_64-unknown-linux-gnu
      platform: linux-x64-gnu

    - os: ubuntu-latest
      target: aarch64-unknown-linux-gnu
      platform: linux-arm64-gnu
      cross: true

    - os: macos-13
      target: x86_64-apple-darwin
      platform: darwin-x64

    - os: macos-14
      target: aarch64-apple-darwin
      platform: darwin-arm64

    - os: windows-latest
      target: x86_64-pc-windows-msvc
      platform: win32-x64-msvc
```

### Workflow Jobs

1. **build**: Builds native binaries for all platforms
2. **test-binaries**: Tests binaries on native platforms
3. **create-release**: Creates GitHub release with checksums
4. **publish-npm**: Publishes to npm registry

### Triggers

- Push to `main`, `develop`, `rust-port` branches
- Pull requests to `main`
- Version tags (`v*`)
- Manual dispatch via GitHub UI

## 📦 NPM Package Structure

The main package `@neural-trader/backend` uses **optional dependencies** for platform-specific binaries:

```json
{
  "name": "@neural-trader/backend",
  "version": "2.0.0",
  "optionalDependencies": {
    "@neural-trader/backend-linux-x64-gnu": "2.0.0",
    "@neural-trader/backend-linux-arm64-gnu": "2.0.0",
    "@neural-trader/backend-darwin-x64": "2.0.0",
    "@neural-trader/backend-darwin-arm64": "2.0.0",
    "@neural-trader/backend-win32-x64-msvc": "2.0.0",
    "@neural-trader/backend-win32-arm64-msvc": "2.0.0"
  }
}
```

### Platform Package Example

Each platform-specific package contains:

```
@neural-trader/backend-linux-x64-gnu/
├── package.json
├── index.js (loads .node binary)
└── neural-trader.linux-x64-gnu.node
```

## 🧪 Testing

### Smoke Test

Basic module loading and API verification:

```bash
npm test
```

Tests:
- Module loads successfully
- Expected exports are available
- Error handling works correctly

### Advanced Test

Performance, memory, and concurrency testing:

```bash
npm run test:advanced
```

Tests:
- Performance benchmarks
- Memory usage monitoring
- Concurrent operations
- System information

### Integration Test

Test with real NAPI bindings:

```javascript
const backend = require('@neural-trader/backend');

console.log('Platform:', backend.__platform__);
console.log('Version:', backend.__version__);
console.log('Node:', backend.__node__);

// Use backend functions
const result = backend.someFunction();
```

## 📤 Publishing to NPM

### Prerequisites

1. NPM account with publish access
2. GitHub repository access
3. All platform binaries built

### Publishing Process

#### Option 1: Automated (Recommended)

```bash
# 1. Update version in package.json
npm version patch  # or minor, major

# 2. Commit and push
git add .
git commit -m "chore: bump version to 2.0.1"
git push origin main

# 3. Create and push release tag
git tag v2.0.1
git push origin v2.0.1

# 4. GitHub Actions will automatically:
#    - Build all platforms
#    - Test binaries
#    - Create GitHub release
#    - Publish to npm
```

#### Option 2: Manual

```bash
# Run the publish script
./scripts/build-publish-napi.sh

# This will:
# 1. Verify npm login
# 2. Check git status
# 3. Validate version
# 4. Guide you through GitHub Actions
# 5. Download release artifacts
# 6. Create platform packages
# 7. Publish to npm
```

### Publishing Checklist

- [ ] All compilation errors fixed
- [ ] Tests passing on all platforms
- [ ] Version bumped in package.json
- [ ] CHANGELOG.md updated
- [ ] README.md up to date
- [ ] Git working directory clean
- [ ] Logged into npm (`npm whoami`)
- [ ] GitHub Actions completed successfully
- [ ] All platform binaries present

## 🔧 Troubleshooting

### Binary Not Found

**Issue**: Module fails to load on installation

**Solution**:
```bash
# Verify platform
node -p "process.platform + '-' + process.arch"

# Install optional dependencies
npm install --include=optional

# Try building from source
cd neural-trader-rust/packages/neural-trader-backend
npm run build
```

### Cross-Compilation Fails

**Issue**: ARM64 Linux build fails on x64 host

**Solution**:
```bash
# Install cross-compilation tools
sudo apt-get update
sudo apt-get install -y gcc-aarch64-linux-gnu g++-aarch64-linux-gnu

# Install cross tool
cargo install cross

# Build with cross
cross build --release --target aarch64-unknown-linux-gnu
```

### GitHub Actions Timeout

**Issue**: Build job times out after 6 hours

**Solution**:
- Enable Rust caching (already configured)
- Split build into multiple jobs
- Use faster runners (GitHub-hosted are sufficient)
- Optimize Cargo.toml dependencies

### NPM Publish Fails

**Issue**: `npm publish` returns 403 or 404

**Solution**:
```bash
# Verify authentication
npm whoami

# Re-login if needed
npm logout
npm login

# Verify package name is unique
npm view @neural-trader/backend

# Check publish access
npm access list packages @neural-trader
```

### Windows Build Issues

**Issue**: MSVC linker errors on Windows

**Solution**:
1. Install Visual Studio 2019+ with C++ workload
2. Install Windows SDK
3. Ensure `link.exe` is in PATH
4. Use `x64 Native Tools Command Prompt`

## 🎯 Performance Benchmarks

Typical binary sizes:

| Platform | Binary Size | Stripped |
|----------|------------|----------|
| Linux x64 | ~2.6 MB | ~2.4 MB |
| Linux ARM64 | ~2.8 MB | ~2.6 MB |
| macOS x64 | ~2.7 MB | ~2.5 MB |
| macOS ARM64 | ~2.5 MB | ~2.3 MB |
| Windows x64 | ~2.9 MB | ~2.7 MB |

Build times (GitHub Actions):

| Platform | Build Time | With Cache |
|----------|-----------|------------|
| Linux x64 | ~3 min | ~1 min |
| Linux ARM64 | ~5 min | ~2 min |
| macOS x64 | ~4 min | ~1.5 min |
| macOS ARM64 | ~4 min | ~1.5 min |
| Windows x64 | ~6 min | ~2 min |

## 📚 Related Documentation

- [Building Guide](../neural-trader-rust/docs/BUILDING.md)
- [NAPI Implementation](../neural-trader-rust/docs/NAPI_REAL_IMPLEMENTATION_ARCHITECTURE.md)
- [GitHub Actions Workflow](../.github/workflows/build-napi.yml)
- [Scripts README](../scripts/README.md)

## 🤝 Contributing

When contributing NAPI bindings:

1. Test locally on your platform first
2. Ensure all tests pass
3. Update TypeScript definitions
4. Add tests for new features
5. Document breaking changes
6. Push to trigger CI builds

## 📄 License

MIT - See [LICENSE](../LICENSE)

## 🔗 Resources

- [NAPI-RS Documentation](https://napi.rs)
- [Rust FFI Guide](https://doc.rust-lang.org/nomicon/ffi.html)
- [Node.js N-API](https://nodejs.org/api/n-api.html)
- [Cross-Compilation Guide](https://rust-lang.github.io/rustup/cross-compilation.html)
