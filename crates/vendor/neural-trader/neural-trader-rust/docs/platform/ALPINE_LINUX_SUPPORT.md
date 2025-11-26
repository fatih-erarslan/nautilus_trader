# Alpine Linux (musl) Support for Neural Trader

## Current Status: ⚠️ Limited Support (v2.1.0)

Neural Trader v2.1.0 packages currently include **glibc** binaries only, which work on:
- ✅ Ubuntu, Debian, CentOS, RHEL, Fedora
- ❌ Alpine Linux (requires musl binaries)

## Why Alpine Linux Needs Special Builds

Alpine Linux uses **musl libc** instead of **glibc**. The .node binaries compiled for glibc will not work on Alpine. You'll see this error:

```
Error loading shared library ld-linux-x86-64.so.2: No such file or directory
```

## Workaround: Use Debian-Based Images

For Docker deployments, use Debian-based Node.js images instead of Alpine:

```dockerfile
# ❌ Won't work
FROM node:20-alpine

# ✅ Works perfectly
FROM node:20-slim
```

## Adding musl Support (Future v2.1.1+)

To add Alpine Linux support, we need to:

### 1. Build musl Binaries

Requires musl cross-compilation toolchain:

```bash
# Install musl tools (on Debian/Ubuntu)
sudo apt-get install musl-tools musl-dev

# Add Rust musl target
rustup target add x86_64-unknown-linux-musl

# Build for musl
cd crates/napi-bindings
cargo build --release --target x86_64-unknown-linux-musl
```

### 2. Configure NAPI-RS for musl

The challenge: **Rust's musl target doesn't support cdylib** (dynamic libraries) by default.

Solution options:
1. **Cross-compile with musl-gcc** (requires musl-tools)
2. **Use Docker for musl builds** (recommended for CI)
3. **GitHub Actions with cross-rs** (production approach)

### 3. GitHub Actions Workflow (Recommended)

```yaml
name: Build Multi-Platform Binaries

on:
  push:
    tags: ['v*']

jobs:
  build-musl:
    runs-on: ubuntu-latest
    container: ghcr.io/napi-rs/napi-rs/nodejs-rust:lts-alpine
    steps:
      - uses: actions/checkout@v4
      - name: Build musl binary
        run: |
          cd crates/napi-bindings
          cargo build --release --target x86_64-unknown-linux-musl

      - name: Copy musl binaries to packages
        run: |
          for pkg in packages/{backtesting,strategies,risk,portfolio,brokers,execution,features,market-data,neural,news-trading,prediction-markets,sports-betting}/native; do
            cp target/x86_64-unknown-linux-musl/release/libnt_napi_bindings.so \
               $pkg/neural-trader.linux-x64-musl.node
          done

      - name: Publish packages
        run: |
          npm config set //registry.npmjs.org/:_authToken ${{ secrets.NPM_TOKEN }}
          cd packages/backtesting && npm publish --access public
          # ... repeat for other packages
```

## Docker Multi-Stage Build for Alpine

If you must use Alpine, use multi-stage builds:

```dockerfile
# Build stage (Debian with glibc)
FROM node:20-slim AS build
WORKDIR /app
COPY package*.json ./
RUN npm install --production

# Runtime stage (Alpine - requires musl binaries!)
FROM node:20-alpine
WORKDIR /app
COPY --from=build /app/node_modules ./node_modules
COPY . .

# This only works if packages include musl binaries
CMD ["node", "index.js"]
```

## Platform Support Matrix

| Platform | Architecture | libc | Status | Package Version |
|----------|-------------|------|--------|----------------|
| Linux | x86_64 | glibc | ✅ Supported | v2.1.0+ |
| Linux | x86_64 | musl | ⚠️ Planned | v2.1.1+ |
| Linux | ARM64 | glibc | 📝 Future | TBD |
| Linux | ARM64 | musl | 📝 Future | TBD |
| macOS | x86_64 | - | 📝 Future | TBD |
| macOS | ARM64 (M1/M2) | - | 📝 Future | TBD |
| Windows | x86_64 | - | 📝 Future | TBD |

## Current Best Practice (v2.1.0)

**For production Docker deployments:**

1. Use `node:20-slim` (Debian-based) instead of `node:20-alpine`
2. All Neural Trader packages work perfectly
3. Slightly larger image (~100MB vs ~50MB)
4. No compatibility issues

**Example Dockerfile:**

```dockerfile
FROM node:20-slim

# Install Neural Trader
RUN npm install -g neural-trader@2.1.0

# Install strategy packages
RUN npm install \
  @neural-trader/backtesting@2.1.0 \
  @neural-trader/strategies@2.1.0 \
  @neural-trader/risk@2.1.0

# Test that binaries work
RUN node -e "console.log(require('@neural-trader/backtesting/native/neural-trader.linux-x64-gnu.node').getVersionInfo())"

CMD ["neural-trader", "--help"]
```

## Tracking musl Support

- 📋 Issue: https://github.com/ruvnet/neural-trader/issues/XXX
- 🎯 Milestone: v2.1.1
- 📅 Target: Q1 2025

## Testing Alpine Compatibility

Once musl binaries are added, test with:

```bash
docker run --rm node:20-alpine sh -c "
  npm install @neural-trader/backtesting@2.1.1 &&
  node -e \"console.log(require('@neural-trader/backtesting').getVersionInfo())\"
"
```

Expected output (with musl support):
```json
{
  "rustCore": "2.0.0",
  "napiBindings": "2.0.0",
  "rustCompiler": "1.91.1"
}
```

## Contributing musl Builds

Want to help add Alpine Linux support? See:
- [`CONTRIBUTING.md`](../CONTRIBUTING.md)
- [`docs/BUILDING.md`](./BUILDING.md)
- Cross-compilation guide: https://rust-lang.github.io/rustup/cross-compilation.html

## Questions?

- 💬 Discussions: https://github.com/ruvnet/neural-trader/discussions
- 🐛 Issues: https://github.com/ruvnet/neural-trader/issues
- 📧 Email: support@neural-trader.ruv.io
