# Platform Selection - Neural Trader

## Overview

Neural Trader automatically detects your platform and loads the correct native binary (.node file) for your system. No manual configuration required!

## How It Works

### Automatic Platform Detection

Each Neural Trader package includes a `load-binary.js` module that:

1. **Detects your OS**: Linux, macOS, or Windows
2. **Detects your CPU architecture**: x64, ARM64
3. **Detects Linux libc variant**: glibc vs musl (for Alpine Linux)
4. **Loads the correct binary** from the `native/` directory

### Binary Naming Convention

Neural Trader uses standardized binary names:

| Platform | Architecture | libc | Binary Name |
|----------|-------------|------|-------------|
| Linux | x86_64 | glibc | `neural-trader.linux-x64-gnu.node` |
| Linux | x86_64 | musl | `neural-trader.linux-x64-musl.node` |
| Linux | ARM64 | glibc | `neural-trader.linux-arm64-gnu.node` |
| macOS | x86_64 (Intel) | - | `neural-trader.darwin-x64.node` |
| macOS | ARM64 (M1/M2/M3) | - | `neural-trader.darwin-arm64.node` |
| Windows | x86_64 | - | `neural-trader.win32-x64-msvc.node` |

## Usage

### Standard Installation

Just install any Neural Trader package normally - platform detection is automatic:

```bash
npm install @neural-trader/backtesting
npm install @neural-trader/strategies
npm install @neural-trader/risk
```

### Using in Code

Platform detection happens automatically when you require a package:

```javascript
const { BacktestEngine } = require('@neural-trader/backtesting');
// ✅ Automatically loads the correct binary for your platform

const backtest = new BacktestEngine({
  initialCapital: 100000,
  // ...config
});
```

## Platform Detection Logic

### 1. Linux Systems

On Linux, the system determines if you're using glibc or musl:

```javascript
// Detect libc type
function detectLibc() {
  try {
    // Try using detect-libc package
    const detectLibc = require('detect-libc');
    const family = detectLibc.familySync();
    return family === 'musl' ? 'musl' : 'gnu';
  } catch (err) {
    // Fallback: check for Alpine-specific files
    if (fs.existsSync('/etc/alpine-release')) {
      return 'musl';
    }
    return 'gnu'; // Default to glibc
  }
}
```

**Result:**
- Ubuntu/Debian/CentOS/RHEL → `neural-trader.linux-x64-gnu.node`
- Alpine Linux → `neural-trader.linux-x64-musl.node` (v2.1.1+)

### 2. macOS Systems

macOS detection distinguishes between Intel and Apple Silicon:

```javascript
const platform = os.platform(); // 'darwin'
const arch = os.arch(); // 'x64' or 'arm64'

if (arch === 'x64') {
  binaryName = 'neural-trader.darwin-x64.node';
} else if (arch === 'arm64') {
  binaryName = 'neural-trader.darwin-arm64.node';
}
```

### 3. Windows Systems

Windows detection (v2.2.0+):

```javascript
const platform = os.platform(); // 'win32'
const arch = os.arch(); // 'x64'

binaryName = 'neural-trader.win32-x64-msvc.node';
```

## Fallback Behavior

The loader tries multiple locations in order:

1. **`native/` directory** (v2.1.1+): `./native/neural-trader.linux-x64-gnu.node`
2. **Package root** (v2.1.0 compatibility): `./neural-trader.linux-x64-gnu.node`
3. **Legacy path** (pre-v2.1.0): `../../neural-trader.linux-x64-gnu.node`

This ensures backward compatibility while supporting the new multi-platform structure.

## Error Handling

If no compatible binary is found, you'll get a helpful error message:

```
Neural Trader native binary not found for linux-x64

Expected: neural-trader.linux-x64-gnu.node

Searched in:
  - /node_modules/@neural-trader/backtesting/native/neural-trader.linux-x64-gnu.node
  - /node_modules/@neural-trader/backtesting/neural-trader.linux-x64-gnu.node
  - /neural-trader-rust/neural-trader.linux-x64-gnu.node

Platform Support:
  ✅ Linux x64 (glibc) - v2.1.0+
  ✅ Linux x64 (musl) - v2.1.1+
  ✅ macOS Intel - v2.2.0+
  ✅ macOS ARM64 - v2.2.0+
  ✅ Windows x64 - v2.2.0+
  ✅ Linux ARM64 - v2.3.0+

Please ensure you're using a compatible Neural Trader version.
See: https://github.com/ruvnet/neural-trader/blob/main/neural-trader-rust/docs/PLATFORM_COMPATIBILITY.md
```

## Docker Containers

### Debian-based (Recommended)

```dockerfile
FROM node:20-slim
WORKDIR /app

RUN npm install @neural-trader/backtesting@2.1.0
# ✅ Automatically uses neural-trader.linux-x64-gnu.node

CMD ["node", "app.js"]
```

### Alpine Linux (v2.1.1+)

```dockerfile
FROM node:20-alpine
WORKDIR /app

RUN npm install @neural-trader/backtesting@2.1.1
# ✅ Automatically uses neural-trader.linux-x64-musl.node

CMD ["node", "app.js"]
```

### Multi-Stage Build

```dockerfile
# Build stage
FROM node:20-slim AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci --production

# Runtime stage (Alpine)
FROM node:20-alpine
WORKDIR /app
COPY --from=builder /app/node_modules ./node_modules
COPY . .

# ✅ Works with v2.1.1+ musl binaries
CMD ["node", "app.js"]
```

## Cloud Platform Deployments

### AWS Lambda

```javascript
// Lambda function with automatic platform detection
const { BacktestEngine } = require('@neural-trader/backtesting');

exports.handler = async (event) => {
  // ✅ Automatically uses linux-x64-gnu binary
  const backtest = new BacktestEngine({
    initialCapital: 100000,
    // ...
  });

  // Run backtest...
};
```

**Lambda Container Image:**
```dockerfile
FROM public.ecr.aws/lambda/nodejs:20
COPY package*.json ./
RUN npm install
COPY . .
CMD ["index.handler"]
```

### Google Cloud Run

```dockerfile
FROM node:20-slim
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
CMD ["node", "server.js"]
```

### Azure Functions

Uses Node.js runtime with automatic platform detection.

## Verification

### Check Platform Detection

```javascript
const { getPlatformBinary, detectLibc } = require('@neural-trader/backtesting/load-binary');

console.log('Platform binary:', getPlatformBinary());
// Linux (glibc): "neural-trader.linux-x64-gnu.node"
// Linux (musl): "neural-trader.linux-x64-musl.node"
// macOS Intel: "neural-trader.darwin-x64.node"
// macOS ARM: "neural-trader.darwin-arm64.node"
// Windows: "neural-trader.win32-x64-msvc.node"

if (process.platform === 'linux') {
  console.log('libc:', detectLibc());
  // "gnu" or "musl"
}
```

### Test Binary Loading

```javascript
const { loadNativeBinary } = require('@neural-trader/backtesting/load-binary');

try {
  const nativeBindings = loadNativeBinary();
  console.log('✅ Binary loaded successfully');
  console.log('Exported functions:', Object.keys(nativeBindings).length);
} catch (err) {
  console.error('❌ Failed to load binary:', err.message);
}
```

## Dependencies

### detect-libc

Neural Trader packages include `detect-libc` for Linux libc detection:

```json
{
  "dependencies": {
    "detect-libc": "^2.0.2"
  }
}
```

This library:
- Detects glibc vs musl on Linux systems
- Returns `null` on non-Linux platforms
- Provides synchronous and asynchronous detection methods

## Troubleshooting

### Problem: "Cannot find module"

```bash
# Reinstall the package
npm install --force @neural-trader/backtesting
```

### Problem: "Unsupported platform"

Check your Node.js environment:

```bash
node -e "console.log(process.platform, process.arch)"
# Expected: linux x64, darwin arm64, etc.
```

### Problem: Alpine Linux fails with v2.1.0

```bash
# Upgrade to v2.1.1+ for musl support
npm install @neural-trader/backtesting@latest
```

Or use Debian-based image:

```dockerfile
# ❌ Won't work with v2.1.0
FROM node:20-alpine

# ✅ Works with all versions
FROM node:20-slim
```

## Version Support Matrix

| Version | Linux glibc | Linux musl | macOS Intel | macOS ARM | Windows | Linux ARM |
|---------|------------|------------|-------------|-----------|---------|-----------|
| v2.1.0 | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| v2.1.1 | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| v2.2.0 | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| v2.3.0 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

## CI/CD Integration

### GitHub Actions

```yaml
name: Test Neural Trader

on: [push]

jobs:
  test:
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        node: [18, 20]

    runs-on: ${{ matrix.os }}

    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: ${{ matrix.node }}

      - run: npm install
      # ✅ Automatically uses correct binary for each OS

      - run: npm test
```

### GitLab CI

```yaml
test:
  parallel:
    matrix:
      - IMAGE: ['node:20-slim', 'node:20-alpine']
  image: $IMAGE
  script:
    - npm install
    - npm test
```

## Advanced: Custom Binary Locations

If you need custom binary paths:

```javascript
const path = require('path');
const customBinaryPath = path.join(__dirname, 'custom-binaries', 'my-binary.node');

const nativeBindings = require(customBinaryPath);
```

## Contributing

Want to add support for a new platform? See:
- [CONTRIBUTING.md](../CONTRIBUTING.md)
- [BUILDING.md](./BUILDING.md)
- [Cross-compilation guide](https://rust-lang.github.io/rustup/cross-compilation.html)

## Support

- 💬 Discussions: https://github.com/ruvnet/neural-trader/discussions
- 🐛 Issues: https://github.com/ruvnet/neural-trader/issues
- 📧 Email: support@neural-trader.ruv.io
