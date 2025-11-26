# Platform Compatibility - Neural Trader v2.1.0

## ✅ Currently Supported (v2.1.0)

### Linux x86_64 (glibc)
**Status:** ✅ **Fully Supported** - Production Ready

**Works on:**
- Ubuntu (18.04+, 20.04, 22.04, 24.04)
- Debian (10+, 11, 12)  
- CentOS / RHEL (7+, 8, 9)
- Fedora (35+)
- Amazon Linux 2
- Google Cloud Platform (Debian/Ubuntu images)
- AWS (Amazon Linux, Ubuntu)
- Azure (Ubuntu)

**Docker Images:**
```dockerfile
# ✅ These work perfectly
FROM node:20-slim       # Debian slim
FROM node:20            # Debian full
FROM ubuntu:22.04
FROM debian:12-slim
```

**Installation:**
```bash
npm install -g neural-trader@2.1.0
npx @neural-trader/mcp@2.1.0 --version
```

**Binary:** `neural-trader.linux-x64-gnu.node` (2.49 MB)
**Functions:** 129 exported Rust functions per package
**Tested:** ✅ Verified in Docker containers

---

## ⚠️ Limited Support (Workaround Available)

### Alpine Linux x86_64 (musl)
**Status:** ⚠️ **Use Debian Images Instead** - musl binaries planned for v2.1.1

**Current Limitation:**
- v2.1.0 packages only include glibc binaries
- Alpine Linux uses musl libc (incompatible)

**Workaround - Use Debian Instead:**
```dockerfile
# ❌ Won't work with v2.1.0
FROM node:20-alpine

# ✅ Use this instead
FROM node:20-slim  # Only 40MB larger, fully compatible
```

**Future Support:**
- 📅 Target: v2.1.1 (Q1 2025)
- 📋 Tracking: [ALPINE_LINUX_SUPPORT.md](./ALPINE_LINUX_SUPPORT.md)
- 🔧 Requires: musl cross-compilation in GitHub Actions

---

## 📝 Planned Support (Future Versions)

### Windows x86_64
**Status:** 📝 Planned for v2.2.0

- Target: Windows 10/11 x64
- Binary: `neural-trader.win32-x64-msvc.node`
- Requires: Visual Studio Build Tools

### macOS x86_64 (Intel)
**Status:** 📝 Planned for v2.2.0

- Target: macOS 10.15+ (Intel chips)
- Binary: `neural-trader.darwin-x64.node`
- Requires: Xcode Command Line Tools

### macOS ARM64 (M1/M2/M3)
**Status:** 📝 Planned for v2.2.0

- Target: macOS 11+ (Apple Silicon)
- Binary: `neural-trader.darwin-arm64.node`
- Native ARM performance

### Linux ARM64 (aarch64)
**Status:** 📝 Planned for v2.3.0

- Target: ARM64 servers, Raspberry Pi 4/5
- Binary: `neural-trader.linux-arm64-gnu.node`
- Use cases: Edge computing, cost-effective cloud

---

## Platform Detection

Neural Trader **automatically detects your platform** and loads the correct native binary:

```javascript
const { getVersionInfo } = require('@neural-trader/backtesting');

console.log(getVersionInfo());
// {
//   "rustCore": "2.0.0",
//   "napiBindings": "2.0.0",
//   "rustCompiler": "1.91.1"
// }
```

### Automatic Platform Selection (v2.1.1+)

Starting with v2.1.1, all packages include intelligent platform detection:

- **Linux glibc**: `neural-trader.linux-x64-gnu.node`
- **Linux musl**: `neural-trader.linux-x64-musl.node` (Alpine)
- **macOS Intel**: `neural-trader.darwin-x64.node` (v2.2.0+)
- **macOS ARM**: `neural-trader.darwin-arm64.node` (v2.2.0+)
- **Windows**: `neural-trader.win32-x64-msvc.node` (v2.2.0+)
- **Linux ARM64**: `neural-trader.linux-arm64-gnu.node` (v2.3.0+)

The correct binary is loaded automatically based on:
1. Operating system (`linux`, `darwin`, `win32`)
2. CPU architecture (`x64`, `arm64`)
3. Linux libc variant (`glibc` vs `musl`)

**For complete details**, see: [PLATFORM_SELECTION.md](./PLATFORM_SELECTION.md)

### Manual Platform Check

```javascript
const { getPlatformBinary } = require('@neural-trader/backtesting/load-binary');
console.log('Using binary:', getPlatformBinary());
// "neural-trader.linux-x64-gnu.node"
```

If the binary doesn't match your platform, you'll see a detailed error with:
- Expected binary name
- Search paths attempted
- Supported platforms and versions
- Link to compatibility documentation

---

## Multi-Platform Publishing Roadmap

### v2.1.0 (Current) ✅
- ✅ Linux x86_64 glibc
- ✅ 14 packages published
- ✅ 129 functions per package

### v2.1.1 (Q1 2025) 🎯
- 📋 Alpine Linux x86_64 musl
- 📋 GitHub Actions for musl builds
- 📋 Multi-platform package tests

### v2.2.0 (Q2 2025)
- 📝 Windows x86_64 MSVC
- 📝 macOS x86_64 (Intel)
- 📝 macOS ARM64 (Apple Silicon)
- 📝 Automated multi-platform CI/CD

### v2.3.0 (Q3 2025)
- 📝 Linux ARM64 glibc
- 📝 Linux ARM64 musl
- 📝 Raspberry Pi support

---

## Quick Compatibility Check

```bash
# Check your platform
uname -a

# Check libc version
ldd --version | head -1

# Test Neural Trader installation
npx neural-trader@2.1.0 --version

# Test package loading
node -e "console.log(require('@neural-trader/backtesting').getVersionInfo())"
```

---

## Platform-Specific Notes

### Docker Users
**Best Practice:** Use Debian-based images for guaranteed compatibility

```dockerfile
FROM node:20-slim
WORKDIR /app
RUN npm install -g neural-trader@2.1.0
CMD ["neural-trader", "--help"]
```

### Cloud Platforms
- **AWS Lambda:** Use `provided.al2` or custom container with Debian base
- **Google Cloud Run:** Use Node.js runtime or Debian containers
- **Azure Functions:** Use Node.js runtime or containers

### CI/CD
All platforms work in standard CI/CD:
- ✅ GitHub Actions (ubuntu-latest)
- ✅ GitLab CI (debian/ubuntu images)
- ✅ CircleCI (linux docker executors)
- ✅ Jenkins (linux agents)

---

## Troubleshooting

### "No such file or directory" Error
```bash
# Wrong platform/libc - check what you have:
file /lib/ld-linux-*.so.2

# If musl: Use v2.1.1+ or switch to Debian image
# If glibc: Should work - check npm cache:
npm cache clean --force
npm install -g neural-trader@2.1.0
```

### "Cannot find module" Error
```bash
# Reinstall with native binary
npm rebuild neural-trader
```

---

## Contributing Platform Support

Want to help add support for your platform? See:
- [CONTRIBUTING.md](../CONTRIBUTING.md)
- [docs/BUILDING.md](./BUILDING.md)
- [Cross-compilation guide](https://rust-lang.github.io/rustup/cross-compilation.html)

---

## Questions & Support

- 💬 Discussions: https://github.com/ruvnet/neural-trader/discussions
- 🐛 Platform Issues: https://github.com/ruvnet/neural-trader/issues
- 📧 Email: support@neural-trader.ruv.io
