# Package Development Documentation

Development guides for contributing to and publishing Neural Trader packages.

## 📁 Documentation Structure

This directory contains comprehensive documentation organized into the following categories:

### 📦 [Build](./build/)
Build system, compilation, and platform support documentation.

**Contents:**
- Multi-platform build guides (Linux, macOS, Windows)
- Cross-compilation setup
- NAPI-RS bindings configuration
- Platform support matrix
- Build optimization and benchmarks

### 📤 [Publishing](./publishing/)
NPM publishing workflow, checklists, and success reports.

**Contents:**
- Publishing guides and checklists
- NPM publishing logs and reports
- Package release workflows
- Publishing automation scripts

### 🧪 [Testing](./testing/)
Comprehensive test reports, metrics, and quality assurance.

**Contents:**
- Package test results and summaries
- Test metrics and visualizations
- Coverage reports
- Test suite documentation

### ✅ [Verification](./verification/)
Package verification and validation reports.

**Contents:**
- Verification procedures
- Validation summaries
- Quality assurance reports

### ⚡ [Features](./features/)
Feature-specific implementation documentation.

**Contents:**
- Syndicate package documentation
- MCP package additions
- Feature parity reports
- Implementation guides

### 📖 [Guides](./guides/)
Development guides, migration docs, and improvements.

**Contents:**
- Migration guides
- README templates and improvements
- Fix documentation
- GitHub issues summary

### 🔧 [Scripts](./scripts/)
Automation scripts for development workflow.

**Contents:**
- Publishing automation
- Validation scripts
- Build automation
- Testing utilities

## Quick Start

### Building for Current Platform
```bash
cd packages/<package-name>
npm run build
```

### Building for All Platforms
```bash
cd packages/<package-name>
npm run build:all
```

### Testing Bindings
```bash
node -e "const pkg = require('@neural-trader/neural'); console.log('Success!')"
```

## Packages with NAPI Bindings

All 9 packages support multi-platform builds:

1. **@neural-trader/backtesting** - High-performance backtesting engine
2. **@neural-trader/neural** - Neural network models (LSTM, GRU, TCN)
3. **@neural-trader/risk** - Risk management (VaR, CVaR, Kelly)
4. **@neural-trader/strategies** - Trading strategies
5. **@neural-trader/portfolio** - Portfolio optimization
6. **@neural-trader/execution** - Order execution (TWAP, VWAP)
7. **@neural-trader/brokers** - Broker integrations
8. **@neural-trader/market-data** - Market data providers
9. **@neural-trader/features** - Technical indicators (150+)

## Supported Platforms

- **Linux x64 GNU** (Ubuntu, Debian, CentOS)
- **Linux x64 MUSL** (Alpine Linux)
- **Linux ARM64** (ARM servers, Raspberry Pi 4+)
- **macOS Intel** (x86_64)
- **macOS ARM** (M1/M2/M3 Apple Silicon)
- **Windows x64** (MSVC)

## CI/CD

Automated builds are configured in:
- `.github/workflows/build-bindings.yml`

The workflow automatically:
- Builds for all platforms on push/PR
- Tests on multiple Node.js versions (16, 18, 20)
- Caches Cargo artifacts for faster builds
- Uploads platform-specific artifacts
- Publishes to NPM (on main branch)

## Architecture

```
packages/
├── backtesting/          # Backtesting engine
│   ├── package.json      # Multi-platform config
│   └── *.node           # Native bindings
├── neural/              # Neural networks
├── risk/                # Risk management
├── strategies/          # Trading strategies
├── portfolio/           # Portfolio optimization
├── execution/           # Order execution
├── brokers/             # Broker integrations
├── market-data/         # Market data
├── features/            # Technical indicators
└── docs/                # This documentation
    ├── README.md        # This file
    ├── MULTI_PLATFORM_BUILD.md
    ├── QUICK_BUILD_REFERENCE.md
    └── PLATFORM_MATRIX.md
```

## Configuration

Each package contains:

```json
{
  "napi": {
    "name": "neural-trader",
    "triples": {
      "defaults": true,
      "additional": [
        "x86_64-unknown-linux-gnu",
        "x86_64-unknown-linux-musl",
        "aarch64-unknown-linux-gnu",
        "x86_64-apple-darwin",
        "aarch64-apple-darwin",
        "x86_64-pc-windows-msvc"
      ]
    }
  },
  "optionalDependencies": {
    "@neural-trader/<pkg>-linux-x64-gnu": "1.0.0",
    "@neural-trader/<pkg>-linux-x64-musl": "1.0.0",
    "@neural-trader/<pkg>-linux-arm64-gnu": "1.0.0",
    "@neural-trader/<pkg>-darwin-x64": "1.0.0",
    "@neural-trader/<pkg>-darwin-arm64": "1.0.0",
    "@neural-trader/<pkg>-win32-x64-msvc": "1.0.0"
  }
}
```

## Prerequisites

### Required
- Node.js 16.x, 18.x, or 20.x
- Rust 1.70+ (`rustup`)
- Cargo (comes with Rust)
- @napi-rs/cli (`npm install -g @napi-rs/cli`)

### Platform-Specific
- **macOS**: Xcode Command Line Tools
- **Windows**: Visual Studio 2019+ or Build Tools
- **Linux**: gcc, g++, make

## Performance

Native bindings provide significant performance improvements:

| Operation | JS Baseline | Rust NAPI |
|-----------|-------------|-----------|
| Neural Network Training | 1x | 12-18x faster |
| Risk Calculations | 1x | 20-30x faster |
| Backtesting | 1x | 30-50x faster |
| Technical Indicators | 1x | 50-150x faster |

## Troubleshooting

Common issues and solutions:

### Missing Binary
```bash
# Rebuild for your platform
npm run build
```

### Permission Errors
```bash
# Clean and rebuild
npm run clean
npm run build
```

### Cross-Compilation Issues
See [MULTI_PLATFORM_BUILD.md](./MULTI_PLATFORM_BUILD.md#cross-compilation-setup)

## Resources

- [NAPI-RS Documentation](https://napi.rs/)
- [Rust Platform Support](https://doc.rust-lang.org/nightly/rustc/platform-support.html)
- [Node.js Native Addons](https://nodejs.org/api/addons.html)
- [GitHub Actions Matrix Builds](https://docs.github.com/en/actions/using-jobs/using-a-matrix-for-your-jobs)

## Contributing

When adding new NAPI bindings:

1. Add platform configuration to `package.json`
2. Add `build:all` script
3. Add optional dependencies for platform packages
4. Test on multiple platforms
5. Update documentation

## Support

For issues:
1. Check the troubleshooting guides
2. Search existing GitHub issues
3. Open a new issue with:
   - Platform details (`node -p "process.platform + '-' + process.arch"`)
   - Node.js version (`node --version`)
   - Error messages
   - Build logs

## License

MIT OR Apache-2.0 (see project root for details)
