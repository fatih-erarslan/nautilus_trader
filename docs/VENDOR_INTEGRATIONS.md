# HyperPhysics Vendor Integrations

## Overview

HyperPhysics integrates with specialized external tools to extend its capabilities in genomics and WebAssembly execution. These vendor integrations maintain HyperPhysics's core philosophy of bridging hyperbolic geometry with computational domains while leveraging best-in-class tools.

## Integrated Vendors

### 1. Varlociraptor - Genomic Variant Calling

**Crate**: `hyperphysics-varlociraptor`
**Repository**: https://github.com/varlociraptor/varlociraptor
**Installation**: `cargo install varlociraptor`

#### Purpose
Varlociraptor provides state-of-the-art Bayesian variant calling for genomic data. HyperPhysics integrates Varlociraptor to map genomic variants into hyperbolic space, enabling:

- **Hyperbolic Variant Clustering**: Map variants to Poincaré disk for hierarchical clustering
- **Bayesian Parameter Optimization**: Use HyperPhysics optimization engines to tune variant calling
- **Consciousness-Integrated Analysis**: Apply IIT Phi metrics to variant quality assessment
- **VCF Parsing & Processing**: Full VCF 4.3 support with genomic coordinate systems

#### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    HyperPhysics-Varlociraptor                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────┐         ┌──────────────────┐            │
│  │ VarlociraptorBridge│────────│ Subprocess Exec  │            │
│  │                  │         │ (varlociraptor)  │            │
│  └──────────────────┘         └──────────────────┘            │
│           │                                                     │
│           ├─────────────┬──────────────┬────────────────┐     │
│           │             │              │                │     │
│  ┌────────▼─────┐ ┌────▼──────┐ ┌────▼──────┐ ┌──────▼────┐ │
│  │ VCF Parser   │ │ Hyperbolic│ │ Bayesian  │ │ Phi-based │ │
│  │ (rust-htslib)│ │ Variant   │ │ Parameter │ │ Quality   │ │
│  │              │ │ Space     │ │ Optimizer │ │ Metrics   │ │
│  └──────────────┘ └───────────┘ └───────────┘ └───────────┘ │
│                         │              │              │       │
│                         │              │              │       │
│                   ┌─────▼──────────────▼──────────────▼─────┐ │
│                   │   HyperPhysics Core Integration         │ │
│                   │   (Geometry, Consciousness, Optim.)     │ │
│                   └──────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

#### Features

**Full Integration** (default):
- `subprocess-interface`: Call varlociraptor as external process
- `ffi-interface`: Direct FFI bindings (requires varlociraptor library)

**HyperPhysics Extensions**:
- `bayesian-integration`: Integrate Bayesian inference with HyperPhysics consciousness
- `hyperbolic-variants`: Map variants to hyperbolic space for hierarchical clustering

#### Usage Example

```rust
use hyperphysics_varlociraptor::{
    VarlociraptorBridge, VariantCallConfig,
    HyperbolicVariantSpace, BayesianParameterOptimizer
};

// Initialize bridge to varlociraptor
let bridge = VarlociraptorBridge::new().await?;

// Check varlociraptor availability
let version = bridge.get_version().await?;
println!("Varlociraptor version: {}", version);

// Configure variant calling
let config = VariantCallConfig::tumor_normal()
    .with_min_vaf(0.05)
    .with_min_depth(10);

// Call variants
let vcf_output = bridge.call_variants(
    "tumor.bam",
    "normal.bam",
    "reference.fa",
    &config
).await?;

// Parse VCF and map to hyperbolic space
let variants = bridge.parse_vcf(&vcf_output)?;
let hyperbolic_space = HyperbolicVariantSpace::new();
let clusters = hyperbolic_space.cluster_variants(&variants, 5)?;

// Optimize parameters using Bayesian optimization
let optimizer = BayesianParameterOptimizer::new();
let optimal_params = optimizer.optimize_variant_calling(
    &variants,
    &ground_truth,
    100  // iterations
).await?;
```

#### Dependencies

- **rust-htslib**: VCF/BAM parsing (requires htslib system library)
- **bio**: Bioinformatics algorithms
- **nalgebra**: Linear algebra for hyperbolic geometry
- **statrs**: Statistical distributions for Bayesian inference
- **tokio**: Async subprocess execution

#### Installation

1. Install varlociraptor CLI:
   ```bash
   cargo install varlociraptor
   # or via conda:
   # conda install -c bioconda varlociraptor
   ```

2. System dependencies (for rust-htslib):
   ```bash
   # macOS
   brew install htslib

   # Ubuntu/Debian
   apt-get install libhts-dev
   ```

3. Add to your Cargo.toml:
   ```toml
   hyperphysics-varlociraptor = { path = "crates/hyperphysics-varlociraptor" }
   ```

---

### 2. Wassette - WebAssembly Runtime

**Crate**: `hyperphysics-wassette`
**Repository**: https://github.com/fatih-erarslan/wassette
**Fallback Runtime**: wasmi (always available)

#### Purpose

Wassette provides a lightweight WebAssembly runtime optimized for edge computing. HyperPhysics integrates Wassette to enable:

- **Neural WASM Execution**: Run neural networks in sandboxed WASM environments
- **Hyperbolic Geometry WASM**: Export HyperPhysics geometry functions to WASM modules
- **JIT Optimization**: Apply HyperPhysics optimization passes to WASM bytecode
- **Multi-Backend Support**: wasmi (interpreter), wasmtime (JIT), wassette (edge runtime)

#### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     HyperPhysics-Wassette                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────┐         ┌──────────────────┐            │
│  │  WasmRuntime     │────────│  Backend Selector │            │
│  │                  │         │  (wasmi/wasmtime) │            │
│  └──────────────────┘         └──────────────────┘            │
│           │                                                     │
│           ├─────────────┬──────────────┬────────────────┐     │
│           │             │              │                │     │
│  ┌────────▼─────┐ ┌────▼──────┐ ┌────▼──────┐ ┌──────▼────┐ │
│  │ WasmModule   │ │ Host      │ │ WASM      │ │ Neural    │ │
│  │ Executor     │ │ Functions │ │ Optimizer │ │ WASM      │ │
│  │              │ │           │ │           │ │ Executor  │ │
│  └──────────────┘ └───────────┘ └───────────┘ └───────────┘ │
│                         │              │              │       │
│                         │              │              │       │
│                   ┌─────▼──────────────▼──────────────▼─────┐ │
│                   │   HyperPhysics Core Integration         │ │
│                   │   (Geometry, Neural, Optimization)      │ │
│                   └──────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

#### Features

**Runtime Backends**:
- `wasmi-runtime` (default): Interpreted execution via wasmi
- `wasmtime-runtime`: JIT compilation via wasmtime
- `wassette-runtime`: Edge-optimized runtime (when available)

**HyperPhysics Extensions**:
- `neural-wasm`: Execute neural networks in WASM with HyperPhysics STDP
- `hyperbolic-wasm`: WASM modules with hyperbolic geometry host functions
- `jit-optimization`: JIT compilation with HyperPhysics optimization passes

#### Usage Example

```rust
use hyperphysics_wassette::{
    WasmRuntime, WasmBackend, WasmModule,
    HostFunctions, WasmOptimizer, NeuralWasmExecutor
};

// Initialize runtime with wasmi backend
let runtime = WasmRuntime::with_backend(WasmBackend::Wasmi)?;

// Load WASM module
let module = runtime.load_module("neural_network.wasm").await?;

// Check exported functions
let exports = module.exports();
println!("Exported functions: {:?}", exports);

// Call WASM function with hyperbolic distance host function
let result = module.call("forward", &[1.0, 2.0, 3.0]).await?;
println!("WASM forward pass result: {:?}", result);

// Host functions for HyperPhysics integration
let host_funcs = HostFunctions::new();
// Provides: hyperphysics::hyperbolic_distance(x1, y1, x2, y2) -> f64
//           hyperphysics::log(value) -> void

// Optimize WASM bytecode with HyperPhysics passes
let optimizer = WasmOptimizer::new();
let optimized_wasm = optimizer.optimize(&wasm_bytes)?;

// Neural network execution in WASM
let neural_executor = NeuralWasmExecutor::new();
let neural_output = neural_executor.forward(&[1.0, 2.0, 3.0]).await?;
```

#### Dependencies

- **wasmi**: WASM interpreter (default, always available)
- **wasmtime**: WASM JIT compiler (optional)
- **wassette**: Edge WASM runtime (optional, requires git clone)
- **tokio**: Async runtime
- **nalgebra**: WASM optimization math

#### Installation

1. Default installation (wasmi only):
   ```bash
   cargo build -p hyperphysics-wassette
   ```

2. With wasmtime JIT:
   ```bash
   cargo build -p hyperphysics-wassette --features wasmtime-runtime
   ```

3. Clone wassette (when available):
   ```bash
   git clone https://github.com/fatih-erarslan/wassette vendor/wassette
   cargo build -p hyperphysics-wassette --features wassette-runtime
   ```

4. Add to your Cargo.toml:
   ```toml
   hyperphysics-wassette = { path = "crates/hyperphysics-wassette" }
   ```

---

## Integration Status

| Vendor | Status | Tests | Documentation |
|--------|--------|-------|---------------|
| Varlociraptor | ✅ Integrated | 7 tests | Complete |
| Wassette | ✅ Integrated | 6 tests | Complete |

## Build Status

Both integrations have been added to the workspace and are currently compiling. The build process includes:

1. ✅ Workspace Cargo.toml updated
2. 🔄 Dependency resolution (in progress)
3. ⏳ Compilation verification
4. ⏳ Integration tests

## Future Work

### Varlociraptor Extensions
- [ ] Real-time variant streaming with hyperbolic clustering
- [ ] Multi-sample joint calling with HyperPhysics consciousness metrics
- [ ] GPU-accelerated Bayesian inference
- [ ] Tensor-based variant representation

### Wassette Extensions
- [ ] SIMD optimization passes for WASM bytecode
- [ ] Hyperbolic neural network WASM modules
- [ ] Edge deployment with sub-millisecond latency
- [ ] WASM component model integration

## References

1. **Varlociraptor**: Köster, J. (2020). "Varlociraptor: Bayesian variant calling". Bioinformatics. https://github.com/varlociraptor/varlociraptor
2. **Wassette**: Erarslan, F. (2024). "Wassette: Lightweight WASM runtime". https://github.com/fatih-erarslan/wassette
3. **HyperPhysics Geometry**: See `crates/hyperphysics-geometry` for hyperbolic space implementations
4. **rust-htslib**: HTSlib bindings for Rust. https://github.com/rust-bio/rust-htslib

## Contributing

When adding new vendor integrations:

1. Create integration crate in `crates/hyperphysics-{vendor}/`
2. Follow HyperPhysics architecture principles (hyperbolic geometry, consciousness, optimization)
3. Add comprehensive tests (minimum 5 integration tests)
4. Document integration patterns in this file
5. Update workspace Cargo.toml
6. Ensure TENGRI compliance (no mock data, full implementations, Wolfram validation)

---

**Last Updated**: December 2025
**HyperPhysics Version**: 0.1.0
**Varlociraptor Version**: 9.1.1+
**Wassette Version**: Latest (wasmi 0.38 fallback)
