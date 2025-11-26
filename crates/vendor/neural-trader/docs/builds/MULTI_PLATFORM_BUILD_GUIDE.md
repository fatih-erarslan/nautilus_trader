# Multi-Platform NAPI Binary Build Guide

**Purpose:** Build NAPI binaries for all platforms (Linux, macOS, Windows)
**Status:** Linux x64 complete, others documented for future builds

---

## 📋 Quick Reference

| Platform | Architecture | Binary Name | Status |
|----------|--------------|-------------|--------|
| Linux | x64 GNU | `neural-trader.linux-x64-gnu.node` | ✅ Built |
| macOS | x64 Intel | `neural-trader.darwin-x64.node` | 📋 Documented |
| macOS | ARM64 (M1+) | `neural-trader.darwin-arm64.node` | 📋 Documented |
| Windows | x64 | `neural-trader.win32-x64.node` | 📋 Documented |
| Windows | ARM64 | `neural-trader.win32-arm64.node` | 📋 Documented |

---

## 🔨 Building for Each Platform

### 1. Linux x64 (Current) ✅

**Environment:** Linux x86_64 (Ubuntu, Debian, etc.)

```bash
cd /workspaces/neural-trader/neural-trader-rust/crates/napi-bindings
cargo build --release
```

**Output:** `target/release/neural-trader.linux-x64-gnu.node` (2.5 MB)

---

### 2. macOS Intel (darwin-x64)

**Option A: Build on macOS x64**
```bash
# On macOS Intel machine
cd neural-trader-rust/crates/napi-bindings
cargo build --release
```

**Option B: Cross-compile from Linux**
```bash
# Install macOS toolchain
rustup target add x86_64-apple-darwin

# Install osxcross (requires macOS SDK)
git clone https://github.com/tpoechtrager/osxcross
cd osxcross
./build.sh

# Build
export PATH="$PWD/target/bin:$PATH"
cargo build --release --target x86_64-apple-darwin
```

**Output:** `target/x86_64-apple-darwin/release/neural-trader.darwin-x64.node`

---

### 3. macOS ARM (darwin-arm64)

**Option A: Build on macOS ARM (M1/M2/M3)**
```bash
# On macOS ARM machine
cd neural-trader-rust/crates/napi-bindings
cargo build --release
```

**Option B: Cross-compile from Linux**
```bash
# Install macOS ARM toolchain
rustup target add aarch64-apple-darwin

# Use osxcross with ARM SDK
export PATH="/path/to/osxcross/target/bin:$PATH"
cargo build --release --target aarch64-apple-darwin
```

**Output:** `target/aarch64-apple-darwin/release/neural-trader.darwin-arm64.node`

---

### 4. Windows x64 (win32-x64)

**Option A: Build on Windows**
```powershell
# On Windows with Visual Studio
cd neural-trader-rust\crates\napi-bindings
cargo build --release
```

**Option B: Cross-compile from Linux**
```bash
# Install Windows toolchain
rustup target add x86_64-pc-windows-msvc

# Install mingw-w64
sudo apt-get install mingw-w64

# Build
cargo build --release --target x86_64-pc-windows-msvc
```

**Output:** `target/x86_64-pc-windows-msvc/release/neural-trader.win32-x64.node`

---

### 5. Windows ARM (win32-arm64)

**Build on Windows ARM or Cross-compile**
```bash
# Install Windows ARM toolchain
rustup target add aarch64-pc-windows-msvc

# Build
cargo build --release --target aarch64-pc-windows-msvc
```

**Output:** `target/aarch64-pc-windows-msvc/release/neural-trader.win32-arm64.node`

---

## 🤖 GitHub Actions Automated Builds

Create `.github/workflows/build-napi.yml`:

```yaml
name: Build Multi-Platform NAPI Binaries

on:
  push:
    tags:
      - 'v*'
  workflow_dispatch:

jobs:
  build:
    name: Build ${{ matrix.os }} ${{ matrix.arch }}
    strategy:
      fail-fast: false
      matrix:
        include:
          # Linux
          - os: ubuntu-latest
            arch: x64
            target: x86_64-unknown-linux-gnu
            binary: neural-trader.linux-x64-gnu.node
          
          # macOS Intel
          - os: macos-latest
            arch: x64
            target: x86_64-apple-darwin
            binary: neural-trader.darwin-x64.node
          
          # macOS ARM
          - os: macos-latest
            arch: arm64
            target: aarch64-apple-darwin
            binary: neural-trader.darwin-arm64.node
          
          # Windows x64
          - os: windows-latest
            arch: x64
            target: x86_64-pc-windows-msvc
            binary: neural-trader.win32-x64.node
          
          # Windows ARM
          - os: windows-latest
            arch: arm64
            target: aarch64-pc-windows-msvc
            binary: neural-trader.win32-arm64.node
    
    runs-on: ${{ matrix.os }}
    
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4
      
      - name: Install Rust
        uses: actions-rs/toolchain@v1
        with:
          toolchain: stable
          target: ${{ matrix.target }}
          override: true
      
      - name: Install Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '18'
      
      - name: Install dependencies
        run: |
          cd neural-trader-rust/crates/napi-bindings
          npm install
      
      - name: Build NAPI binary
        run: |
          cd neural-trader-rust/crates/napi-bindings
          cargo build --release --target ${{ matrix.target }}
      
      - name: Rename binary
        shell: bash
        run: |
          cd neural-trader-rust/crates/napi-bindings
          if [ "${{ matrix.os }}" = "windows-latest" ]; then
            mv target/${{ matrix.target }}/release/*.node ${{ matrix.binary }} || true
          else
            mv target/${{ matrix.target }}/release/*.node ${{ matrix.binary }} || true
          fi
      
      - name: Upload artifact
        uses: actions/upload-artifact@v4
        with:
          name: ${{ matrix.binary }}
          path: neural-trader-rust/crates/napi-bindings/${{ matrix.binary }}
      
      - name: Upload to release
        if: startsWith(github.ref, 'refs/tags/v')
        uses: softprops/action-gh-release@v1
        with:
          files: neural-trader-rust/crates/napi-bindings/${{ matrix.binary }}
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}

  distribute:
    name: Distribute binaries to packages
    needs: build
    runs-on: ubuntu-latest
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4
      
      - name: Download all artifacts
        uses: actions/download-artifact@v4
        with:
          path: binaries
      
      - name: Distribute to packages
        run: |
          cd neural-trader-rust
          
          for binary in binaries/*/*.node; do
            filename=$(basename "$binary")
            echo "Distributing $filename..."
            
            for pkg in packages/*/native; do
              if [ -d "$pkg" ]; then
                cp "$binary" "$pkg/"
                echo "  ✅ Copied to $pkg"
              fi
            done
          done
      
      - name: Commit updated binaries
        run: |
          git config user.name "GitHub Actions"
          git config user.email "actions@github.com"
          git add neural-trader-rust/packages/*/native/*.node
          git commit -m "chore: update NAPI binaries for all platforms" || true
          git push || true
```

---

## 🚀 Triggering Builds

### Manual Trigger
```bash
# Push a tag
git tag v2.0.3
git push origin v2.0.3
```

### Automatic on Release
- GitHub Actions will build all platforms
- Binaries uploaded to release assets
- Packages updated automatically

---

## 📦 Platform-Specific Packages (Optional)

Create separate packages for each platform to reduce download size:

```json
{
  "name": "@neural-trader/linux-x64",
  "version": "2.0.0",
  "os": ["linux"],
  "cpu": ["x64"],
  "files": ["native/neural-trader.linux-x64-gnu.node"]
}
```

Then use `@napi-rs/cli` to automatically select the right package:

```bash
npm install @neural-trader/linux-x64  # Only on Linux x64
npm install @neural-trader/darwin-arm64  # Only on macOS ARM
```

---

## 🧪 Testing Multi-Platform Binaries

### Create Test Script
```javascript
// test-binary.js
const os = require('os');
const path = require('path');

const platform = os.platform();
const arch = os.arch();

let binaryName;
if (platform === 'linux' && arch === 'x64') {
  binaryName = 'neural-trader.linux-x64-gnu.node';
} else if (platform === 'darwin' && arch === 'x64') {
  binaryName = 'neural-trader.darwin-x64.node';
} else if (platform === 'darwin' && arch === 'arm64') {
  binaryName = 'neural-trader.darwin-arm64.node';
} else if (platform === 'win32' && arch === 'x64') {
  binaryName = 'neural-trader.win32-x64.node';
} else if (platform === 'win32' && arch === 'arm64') {
  binaryName = 'neural-trader.win32-arm64.node';
} else {
  console.error(`Unsupported platform: ${platform}-${arch}`);
  process.exit(1);
}

try {
  const napi = require(path.join(__dirname, 'native', binaryName));
  console.log('✅ NAPI module loaded successfully');
  console.log(`Platform: ${platform}-${arch}`);
  console.log(`Binary: ${binaryName}`);
  console.log(`Exported functions: ${Object.keys(napi).length}`);
  
  // Test ping function
  const result = napi.ping();
  console.log('✅ ping() test:', result);
} catch (error) {
  console.error('❌ Failed to load NAPI module:', error.message);
  process.exit(1);
}
```

### Run Tests
```bash
node test-binary.js
```

---

## 📊 Binary Size Expectations

| Platform | Expected Size | Compression |
|----------|--------------|-------------|
| Linux x64 | 2.5 MB | ~800 KB gzip |
| macOS x64 | 2.8 MB | ~900 KB gzip |
| macOS ARM | 2.6 MB | ~850 KB gzip |
| Windows x64 | 3.0 MB | ~1.0 MB gzip |
| Windows ARM | 2.8 MB | ~900 KB gzip |

---

## 🔧 Troubleshooting

### Issue: Cross-compilation fails
**Solution:** Use GitHub Actions runners for native builds

### Issue: Binary doesn't load on target platform
**Solution:** Verify target triple matches platform exactly

### Issue: Missing symbols in binary
**Solution:** Ensure all dependencies are included in Cargo.toml

### Issue: Large binary size
**Solution:** Enable LTO and strip debug symbols:
```toml
[profile.release]
lto = true
strip = true
opt-level = 3
```

---

**Next Steps:**
1. Set up GitHub Actions workflow
2. Test on each platform
3. Create platform-specific packages (optional)
4. Publish to npm with all binaries

---

*This guide provides everything needed for multi-platform NAPI builds.*
