# Neural Trader Predictor - Dependencies & Integration

## 🦀 Rust Dependencies

### Core Dependencies (Cargo.toml)

```toml
[package]
name = "neural-trader-predictor"
version = "0.1.0"
edition = "2021"
authors = ["Neural Trader Team"]
description = "Conformal prediction SDK/CLI for neural trading with guaranteed intervals"
license = "MIT OR Apache-2.0"
repository = "https://github.com/ruvnet/neural-trader"
keywords = ["conformal-prediction", "trading", "machine-learning", "finance"]
categories = ["algorithms", "mathematics", "finance"]

[dependencies]
# Numerical computing
ndarray = { version = "0.15", features = ["rayon"] }
ndarray-stats = "0.5"
num-traits = "0.2"

# Random sampling
rand = "0.8"
rand_distr = "0.4"

# Error handling
thiserror = "1.0"
anyhow = "1.0"

# Parallel processing
rayon = "1.8"

# Comparable floats
ordered-float = "4.0"

# Serialization
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
bincode = "1.3"

# CLI framework
clap = { version = "4.4", features = ["derive", "cargo"] }

# Logging
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }

# Performance optimizations
nanosecond-scheduler = "0.1"  # High-precision scheduling
# sublinear = "0.1"            # O(log n) algorithms - CHECK IF EXISTS
# temporal-lead-solver = "0.1" # Predictive computation - CHECK IF EXISTS
# strange-loops = "0.1"        # Optimization patterns - CHECK IF EXISTS

# Alternative if above don't exist:
tokio = { version = "1.35", features = ["time", "rt-multi-thread"], optional = true }
parking_lot = "0.12"

# Statistics
statrs = "0.17"

[dev-dependencies]
criterion = { version = "0.5", features = ["html_reports"] }
proptest = "1.4"
approx = "0.5"
tempfile = "3.8"

[features]
default = ["cli"]
cli = ["clap"]
async = ["tokio"]
wasm = []

[[bin]]
name = "neural-predictor"
path = "src/bin/neural-predictor.rs"
required-features = ["cli"]

[[bench]]
name = "prediction_bench"
harness = false

[[bench]]
name = "calibration_bench"
harness = false

[profile.release]
opt-level = 3
lto = true
codegen-units = 1
strip = true

[profile.bench]
inherits = "release"
```

### Dependency Review Notes

**nanosecond-scheduler**: Need to verify availability on crates.io
- If not available: Use `tokio::time` for sub-millisecond scheduling
- Alternative: Custom implementation with `std::time::Instant`

**sublinear**: Likely custom implementation needed
- Implement binary search for O(log n) score insertion
- Use `Vec::binary_search` and `VecDeque` for efficient updates

**temporal-lead-solver**: Custom implementation
- Pre-compute predictions using estimated features
- Cache frequent patterns for fast lookup
- Implement speculative execution pipeline

**strange-loops**: Custom optimization patterns
- Implement recursive calibration refinement
- Self-tuning hyperparameters (gamma, alpha)
- Meta-learning for coverage optimization

## 📦 NPM Dependencies

### Pure JS Package (package.json)

```json
{
  "name": "@neural-trader/predictor",
  "version": "0.1.0",
  "description": "Conformal prediction for neural trading with guaranteed intervals",
  "main": "dist/index.js",
  "module": "dist/index.mjs",
  "types": "dist/index.d.ts",
  "exports": {
    ".": {
      "import": "./dist/index.mjs",
      "require": "./dist/index.js",
      "types": "./dist/index.d.ts"
    },
    "./wasm": {
      "import": "./dist/wasm/index.mjs",
      "require": "./dist/wasm/index.js",
      "types": "./dist/wasm/index.d.ts"
    },
    "./native": {
      "import": "./dist/native/index.mjs",
      "require": "./dist/native/index.js",
      "types": "./dist/native/index.d.ts"
    }
  },
  "scripts": {
    "build": "tsup",
    "build:wasm": "wasm-pack build --target bundler --out-dir pkg",
    "build:native": "napi build --platform --release",
    "test": "vitest",
    "bench": "vitest bench",
    "lint": "eslint src",
    "typecheck": "tsc --noEmit"
  },
  "dependencies": {
    "tslib": "^2.6.2"
  },
  "devDependencies": {
    "@napi-rs/cli": "^2.18.0",
    "@types/node": "^20.10.0",
    "@vitest/ui": "^1.0.0",
    "esbuild": "^0.19.0",
    "eslint": "^8.55.0",
    "tsup": "^8.0.0",
    "typescript": "^5.3.0",
    "vitest": "^1.0.0",
    "wasm-pack": "^0.12.1"
  },
  "optionalDependencies": {
    "@neural-trader/predictor-native": "0.1.0"
  },
  "files": [
    "dist",
    "pkg",
    "README.md",
    "LICENSE"
  ],
  "keywords": [
    "conformal-prediction",
    "trading",
    "finance",
    "machine-learning",
    "wasm",
    "rust"
  ],
  "repository": {
    "type": "git",
    "url": "https://github.com/ruvnet/neural-trader"
  },
  "license": "MIT"
}
```

### WASM Build Dependencies

```toml
[package]
name = "neural-trader-predictor-wasm"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib"]

[dependencies]
neural-trader-predictor = { path = "../neural-trader-predictor", features = ["wasm"] }
wasm-bindgen = "0.2"
serde-wasm-bindgen = "0.6"
console_error_panic_hook = "0.1"
wasm-bindgen-futures = "0.4"
js-sys = "0.3"
web-sys = { version = "0.3", features = ["console"] }

[profile.release]
opt-level = "z"     # Optimize for size
lto = true
codegen-units = 1
```

### NAPI Native Build Dependencies

```toml
[package]
name = "neural-trader-predictor-native"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib"]

[dependencies]
neural-trader-predictor = { path = "../neural-trader-predictor" }
napi = "2.15"
napi-derive = "2.15"

[build-dependencies]
napi-build = "2.1"

[profile.release]
opt-level = 3
lto = true
codegen-units = 1
```

## 🔗 Integration Dependencies

### @neural-trader/neural Integration

```json
{
  "peerDependencies": {
    "@neural-trader/neural": "^1.0.0"
  },
  "peerDependenciesMeta": {
    "@neural-trader/neural": {
      "optional": true
    }
  }
}
```

### Integration code:

```typescript
// packages/predictor/src/integration/neural.ts
import type { NeuralPredictor } from '@neural-trader/neural';
import { ConformalPredictor } from '../core/conformal';

export function wrapWithConformal(
    neural: NeuralPredictor,
    config: ConformalConfig
): ConformalNeuralPredictor {
    // Implementation
}
```

## 🧪 Testing Dependencies

### E2B Sandbox Configuration

```json
{
  "devDependencies": {
    "@e2b/sdk": "^1.0.0",
    "agentic-flow": "^2.0.0"
  },
  "scripts": {
    "test:e2b": "node scripts/e2b-test.js",
    "test:real-api": "E2B_API_KEY=$E2B_API_KEY node scripts/test-real-api.js"
  }
}
```

### E2B Test Configuration

```javascript
// scripts/e2b-test.js
import { Sandbox } from '@e2b/sdk';
import { AgenticFlow } from 'agentic-flow';

const sandbox = await Sandbox.create({
    template: 'base',
    apiKey: process.env.E2B_API_KEY,
});

// Install dependencies in sandbox
await sandbox.commands.run('cargo build --release');
await sandbox.commands.run('npm install');

// Run tests with real API keys
const result = await sandbox.commands.run('npm test', {
    env: {
        ALPACA_API_KEY: process.env.ALPACA_API_KEY,
        ALPACA_SECRET_KEY: process.env.ALPACA_SECRET_KEY,
    },
});
```

### Agentic Flow Configuration (Low-Cost Kimi K2)

```javascript
// .agenticrc.json
{
  "model": "openrouter/kimi/k2",
  "apiKey": process.env.OPENROUTER_API_KEY,
  "agents": {
    "rust-core": {
      "model": "openrouter/kimi/k2",
      "systemPrompt": "You are a Rust expert implementing conformal prediction algorithms."
    },
    "wasm-bindings": {
      "model": "openrouter/kimi/k2",
      "systemPrompt": "You are a WASM expert creating JavaScript bindings."
    },
    "testing": {
      "model": "openrouter/kimi/k2",
      "systemPrompt": "You are a testing expert writing comprehensive test suites."
    },
    "documentation": {
      "model": "openrouter/kimi/k2",
      "systemPrompt": "You are a technical writer creating clear documentation."
    }
  }
}
```

## 📊 lean-agentic Review

Based on npm `lean-agentic` best practices:

### Optimization Principles
1. **Minimal Dependencies**: Use only essential packages
2. **Tree-Shaking**: Ensure dead code elimination
3. **Lazy Loading**: Import heavy modules on-demand
4. **Bundle Size**: Target <50KB for pure JS, <200KB for WASM
5. **Zero-Cost Abstractions**: No runtime overhead for type safety

### Implementation
```typescript
// Lazy load WASM only when needed
export async function createPredictor(config) {
    if (config.preferNative) {
        try {
            const { NativePredictor } = await import('./native');
            return new NativePredictor(config);
        } catch {}
    }

    if (config.preferWasm) {
        try {
            const { WasmPredictor } = await import('./wasm');
            await initWasm();
            return new WasmPredictor(config);
        } catch {}
    }

    // Fallback to pure JS
    const { PureJsPredictor } = await import('./pure');
    return new PureJsPredictor(config);
}
```

## 🚀 Build Tools

### Rust Build
```bash
# Development
cargo build

# Release (optimized)
cargo build --release

# CLI binary
cargo build --bin neural-predictor --release

# Run tests
cargo test

# Run benchmarks
cargo bench

# Generate docs
cargo doc --no-deps --open
```

### WASM Build
```bash
# Build WASM package
wasm-pack build --target bundler --out-dir pkg

# Test in browser
wasm-pack test --headless --firefox

# Optimize size
wasm-opt -Oz -o optimized.wasm pkg/neural_trader_predictor_wasm_bg.wasm
```

### NAPI Build
```bash
# Build native addon
napi build --platform --release

# Cross-compile for multiple platforms
napi build --platform --release --target x86_64-unknown-linux-gnu
napi build --platform --release --target x86_64-apple-darwin
napi build --platform --release --target aarch64-apple-darwin
napi build --platform --release --target x86_64-pc-windows-msvc
```

### TypeScript Build
```bash
# Build all variants
npm run build

# Build specific targets
npm run build:wasm
npm run build:native

# Watch mode
npm run build -- --watch
```

## 📦 Publishing Checklist

### Crates.io
- [ ] Version in Cargo.toml
- [ ] README with examples
- [ ] License files
- [ ] Documentation comments
- [ ] CI badges
- [ ] Run `cargo publish --dry-run`

### NPM
- [ ] Version in package.json
- [ ] README with badges
- [ ] License files
- [ ] Type declarations
- [ ] CI badges
- [ ] Run `npm publish --dry-run`

### Verification
- [ ] All tests pass
- [ ] Benchmarks show performance targets met
- [ ] Examples work
- [ ] Documentation builds
- [ ] No security vulnerabilities
- [ ] Bundle size under limits
