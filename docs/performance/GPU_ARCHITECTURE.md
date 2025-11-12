# GPU Acceleration Architecture for HyperPhysics

**Agent**: Performance-Engineer
**Date**: 2025-11-12
**Status**: Design Phase - Implementation Deferred Until SIMD Baseline
**Priority**: ⭐⭐ MEDIUM (after SIMD proves insufficient)

---

## ⚠️ Critical Decision: Delay GPU Development

### Rationale
1. **SIMD First**: Must establish 3-5x SIMD speedup before investing in GPU
2. **Problem Size**: Current use case (10-1000 pBits) too small for GPU gains
3. **wgpu Blocker**: Compilation error prevents immediate GPU work
4. **ROI**: GPU requires 80-120 hours dev time, SIMD requires 20-40 hours

### GPU Threshold
Only proceed with GPU if:
- ✅ SIMD optimization complete and benchmarked
- ✅ Production workloads require >10k pBits
- ✅ SIMD <100 μs latency still insufficient
- ✅ wgpu compilation fixed and stable

**Current Decision**: DEFER GPU to Phase 4 (Weeks 4-5+)

---

## 1. GPU Backend Selection

### 1.1 Candidate Backends

| Backend | Pros | Cons | Verdict |
|---------|------|------|---------|
| **wgpu** | Cross-platform, pure Rust | Compilation issues, immature | ⭐ **RECOMMENDED** (after fix) |
| **CUDA** | Maximum performance, mature | NVIDIA-only, C++ bindings | ⭐⭐ Fallback for >100k pBits |
| **OpenCL** | Widely supported | Deprecated, complex | ❌ AVOID |
| **Metal** | Native on Apple Silicon | macOS/iOS only | ⭐⭐ Future (via wgpu) |
| **Vulkan Compute** | High performance | Complex API | ⭐⭐ Future (via wgpu) |

---

### 1.2 wgpu Architecture (Recommended)

**Why wgpu**:
- Compiles to **Metal** (macOS), **Vulkan** (Linux), **D3D12** (Windows), **WebGPU** (browser)
- Pure Rust (no FFI overhead)
- Active development (200+ contributors)
- Works on 10k-1M pBits (sweet spot)

**Current Blocker**:
```
error: macro expansion ends with an incomplete expression
   --> wgpu-0.19.4/src/backend/wgpu_core.rs:783:92
```

**Fix**: Update to wgpu v0.20+ (released Nov 2024)
```toml
[dependencies]
wgpu = { version = "0.20", optional = true }
bytemuck = { version = "1.14", optional = true }
pollster = { version = "0.3", optional = true }  # For async runtime
```

---

### 1.3 CUDA Fallback (Enterprise Use Case)

**When to Use CUDA**:
- Lattices >100k pBits
- NVIDIA GPU farm available
- <1 μs latency required
- Maximum throughput needed

**Dependencies**:
```toml
[dependencies]
cudarc = { version = "0.10", optional = true }  # Rust CUDA bindings
```

**Trade-offs**:
- 2-3x faster than wgpu on NVIDIA hardware
- No portability (locks users into NVIDIA)
- Requires CUDA toolkit installation

---

## 2. GPU Kernel Design

### 2.1 Gillespie Propensity Kernel

**Objective**: Calculate transition rates for all pBits in parallel

**WGSL Compute Shader**:
```wgsl
@group(0) @binding(0) var<storage, read> states: array<u32>;        // Packed bits
@group(0) @binding(1) var<storage, read> biases: array<f32>;        // N elements
@group(0) @binding(2) var<storage, read> couplings: array<f32>;     // N×N matrix (flat)
@group(0) @binding(3) var<storage, read_write> propensities: array<f32>; // Output
@group(0) @binding(4) var<uniform> params: SimulationParams;

struct SimulationParams {
    n_pbits: u32,
    temperature: f32,
    _padding: vec2<f32>,  // Alignment
}

@compute @workgroup_size(256)
fn calculate_propensities(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= params.n_pbits) { return; }

    // Unpack state from bitfield
    let word_idx = idx / 32u;
    let bit_idx = idx % 32u;
    let state_i = (states[word_idx] >> bit_idx) & 1u;
    let si = select(-1.0, 1.0, state_i == 1u);  // Convert to spin {-1, +1}

    // Calculate effective field: h_i = bias_i + Σ J_ij s_j
    var h_eff = biases[idx];

    for (var j = 0u; j < params.n_pbits; j++) {
        if (j == idx) { continue; }  // Skip self-coupling

        let word_j = j / 32u;
        let bit_j = j % 32u;
        let state_j = (states[word_j] >> bit_j) & 1u;
        let sj = select(-1.0, 1.0, state_j == 1u);

        let coupling_idx = idx * params.n_pbits + j;
        h_eff += couplings[coupling_idx] * sj;
    }

    // Calculate transition probability: p = sigmoid(h_eff / T)
    let x = h_eff / params.temperature;
    let prob = 1.0 / (1.0 + exp(-x));

    // Transition rate: r = p if state=0, r = (1-p) if state=1
    let rate = select(prob, 1.0 - prob, state_i == 1u);

    propensities[idx] = rate;
}
```

**Optimization Opportunities**:
1. **Shared Memory**: Cache coupling matrix rows in workgroup shared memory
2. **Coalesced Access**: Ensure memory access patterns are sequential
3. **Wave Intrinsics**: Use subgroup operations for horizontal sums

**Expected Performance** (NVIDIA A100):
- Workgroups: 256 threads each
- Occupancy: 100% (limited by register pressure)
- Throughput: ~10M propensity calculations/sec
- Latency: **5 μs** for 10k pBits

---

### 2.2 Memory Transfer Strategy

**Problem**: GPU↔CPU transfers are expensive (PCIe bottleneck)
- Typical bandwidth: 16 GB/s (PCIe 4.0 ×16)
- Latency: 10-50 μs per transfer

**Solution**: Keep state on GPU, minimize transfers

```rust
struct GpuGillespieSimulator {
    device: wgpu::Device,
    queue: wgpu::Queue,

    // GPU-resident buffers (never copied to CPU)
    states_buffer: wgpu::Buffer,         // Bitpacked states
    couplings_buffer: wgpu::Buffer,      // Coupling matrix
    propensities_buffer: wgpu::Buffer,   // Transition rates

    // Staging buffers (for rare CPU↔GPU sync)
    staging_buffer: wgpu::Buffer,

    // Compute pipeline
    propensity_pipeline: wgpu::ComputePipeline,
}

impl GpuGillespieSimulator {
    /// Initialize from CPU lattice (one-time transfer)
    pub fn from_lattice(lattice: &PBitLattice) -> Self {
        // Upload couplings to GPU (once)
        let couplings_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Coupling Matrix"),
            contents: bytemuck::cast_slice(&lattice.coupling_matrix_flat()),
            usage: BufferUsages::STORAGE,
        });

        // Initial state upload
        let states_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("pBit States"),
            contents: &lattice.states_packed(),  // Bitpacked
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });

        // ... initialize other buffers and pipeline
    }

    /// Run N simulation steps on GPU (no CPU transfer!)
    pub async fn simulate_steps(&mut self, n_steps: usize) -> Result<(), GpuError> {
        for _ in 0..n_steps {
            // 1. Dispatch propensity kernel
            self.calculate_propensities().await?;

            // 2. Select which pBit flips (GPU atomic reduction)
            let flip_idx = self.select_event_gpu().await?;

            // 3. Flip pBit on GPU (atomic bit flip)
            self.flip_pbit_gpu(flip_idx).await?;

            // NO CPU↔GPU TRANSFER!
        }
        Ok(())
    }

    /// Download final state to CPU (rare)
    pub async fn download_state(&self) -> Vec<bool> {
        // Copy GPU buffer to staging buffer
        self.queue.submit(/* copy command */);

        // Map staging buffer to CPU
        let buffer_slice = self.staging_buffer.slice(..);
        buffer_slice.map_async(MapMode::Read).await.unwrap();

        // Unpack bitfield to bool array
        let data = buffer_slice.get_mapped_range();
        unpack_states(&data)
    }
}
```

**Transfer Budget**:
- Initial upload: 1× (amortized over 1000s of steps)
- Per-step download: 0× (only when user requests state)
- Result: **<1 μs transfer overhead** per step

---

### 2.3 Async Compute + CPU Overlap

**Problem**: GPU work blocks CPU
**Solution**: Use async compute to overlap GPU and CPU work

```rust
use pollster::FutureExt;  // Async runtime for wgpu

// Spawn GPU work asynchronously
let gpu_future = simulator.simulate_steps(1000);

// CPU continues working (e.g., updating external signals)
for _ in 0..100 {
    update_external_input();
}

// Wait for GPU to finish
gpu_future.block_on()?;
```

**Advanced**: Use multiple command queues for true parallelism (Vulkan/Metal).

---

## 3. Performance Targets

### 3.1 Latency Breakdown

| Operation | CPU (Scalar) | SIMD | GPU | Target |
|-----------|--------------|------|-----|--------|
| Propensity calc (10k) | 500 μs | 100 μs | **5 μs** | <10 μs |
| Event selection | 50 μs | 50 μs | **2 μs** | <5 μs |
| State update | 1 μs | 1 μs | **0.5 μs** | <1 μs |
| **Total per step** | **551 μs** | **151 μs** | **7.5 μs** | **<20 μs** |

**Speedup**: 20-70× over scalar, 10-20× over SIMD

---

### 3.2 Throughput Targets

| Metric | GPU Target | Rationale |
|--------|-----------|-----------|
| Steps/sec (10k pBits) | 100,000 | Message passing <10 μs |
| Steps/sec (100k pBits) | 10,000 | Large-scale simulation |
| Propensities/sec | 1 billion | GPU compute bound |

---

### 3.3 Scaling Analysis

**Question**: When does GPU become worth it?

| pBits | SIMD Time | GPU Compute | GPU Transfer | GPU Total | Speedup | Worth It? |
|-------|-----------|-------------|--------------|-----------|---------|-----------|
| 100 | 1 μs | 0.5 μs | 10 μs | **10.5 μs** | **0.1×** | ❌ NO |
| 1,000 | 10 μs | 1 μs | 10 μs | **11 μs** | **0.9×** | ❌ NO |
| 10,000 | 100 μs | 5 μs | 10 μs | **15 μs** | **6.7×** | ✅ YES |
| 100,000 | 1 ms | 50 μs | 10 μs | **60 μs** | **16.7×** | ✅ YES |

**Conclusion**: GPU only beneficial for **>5k pBits** (assuming per-step transfer avoided)

---

## 4. Implementation Roadmap (Deferred)

### Phase 4A: GPU Prototype (Week 4)
- [ ] Fix wgpu compilation (update to v0.20)
- [ ] Implement propensity kernel (WGSL)
- [ ] Basic GPU↔CPU transfer
- [ ] Benchmark vs SIMD (target: 10× speedup for 10k pBits)

### Phase 4B: Optimize Memory (Week 5)
- [ ] Implement GPU-resident state
- [ ] Async event selection kernel
- [ ] Minimize CPU↔GPU transfers
- [ ] Benchmark end-to-end latency (target: <10 μs)

### Phase 4C: Production Ready (Week 6)
- [ ] Automatic CPU/SIMD/GPU selection based on problem size
- [ ] Error handling and fallback
- [ ] Cross-platform testing (Metal, Vulkan, D3D12)

---

## 5. Automatic Backend Selection

**Smart Dispatch Logic**:
```rust
pub enum ComputeBackend {
    Scalar,
    Simd,
    Gpu(GpuDevice),
}

impl PBitSimulator {
    pub fn auto_select_backend(n_pbits: usize) -> ComputeBackend {
        // Check GPU availability
        let gpu_available = pollster::block_on(async {
            wgpu::Instance::new(wgpu::InstanceDescriptor::default())
                .request_adapter(&wgpu::RequestAdapterOptions::default())
                .await
                .is_some()
        });

        match (n_pbits, gpu_available) {
            (n, true) if n >= 10_000 => {
                println!("Using GPU backend (n={} pBits)", n);
                ComputeBackend::Gpu(GpuDevice::new())
            }
            (n, _) if n >= 100 => {
                println!("Using SIMD backend (n={} pBits)", n);
                ComputeBackend::Simd
            }
            _ => {
                println!("Using scalar backend (small problem)");
                ComputeBackend::Scalar
            }
        }
    }
}
```

**Override for Testing**:
```rust
// Environment variable override
let backend = std::env::var("HYPERPHYSICS_BACKEND")
    .map(|s| match s.as_str() {
        "gpu" => ComputeBackend::Gpu(GpuDevice::new()),
        "simd" => ComputeBackend::Simd,
        "scalar" => ComputeBackend::Scalar,
        _ => panic!("Invalid backend"),
    })
    .unwrap_or_else(|_| Self::auto_select_backend(n_pbits));
```

---

## 6. GPU Testing Strategy

### 6.1 Correctness Tests
**Challenge**: GPU results may differ slightly from CPU (floating-point rounding)

```rust
#[test]
fn test_gpu_vs_cpu_propensities() {
    let lattice = PBitLattice::random(1000);

    // CPU reference
    let cpu_props = calculate_propensities_cpu(&lattice);

    // GPU implementation
    let gpu_props = pollster::block_on(async {
        calculate_propensities_gpu(&lattice).await
    });

    // Compare with tolerance
    for (cpu, gpu) in cpu_props.iter().zip(gpu_props.iter()) {
        assert!((cpu - gpu).abs() < 1e-5, "GPU mismatch: {} vs {}", cpu, gpu);
    }
}
```

---

### 6.2 Performance Tests
```rust
#[bench]
fn bench_gpu_vs_simd(b: &mut Bencher) {
    let lattice = PBitLattice::random(10_000);

    // SIMD baseline
    b.iter(|| {
        let props = calculate_propensities_simd(&lattice);
        black_box(props);
    });

    // GPU version (should be 10-20× faster)
    b.iter(|| {
        let props = pollster::block_on(async {
            calculate_propensities_gpu(&lattice).await
        });
        black_box(props);
    });
}
```

---

## 7. Deployment Considerations

### 7.1 Hardware Requirements
**Minimum GPU** (for 10k pBits):
- VRAM: 512 MB
- Compute: 1 TFLOPS (fp32)
- Examples: GTX 1650, M1 base, Radeon RX 560

**Recommended GPU** (for 100k pBits):
- VRAM: 4 GB
- Compute: 10 TFLOPS
- Examples: RTX 3060, M1 Pro, Radeon RX 6700

### 7.2 Fallback Strategy
```rust
// Attempt GPU initialization
match GpuSimulator::new() {
    Ok(gpu) => run_simulation_gpu(gpu),
    Err(e) => {
        warn!("GPU initialization failed: {}, falling back to SIMD", e);
        run_simulation_simd()
    }
}
```

---

## 8. Future Extensions

### 8.1 Multi-GPU Support
For >1M pBits, partition lattice across multiple GPUs:
```rust
struct MultiGpuSimulator {
    gpus: Vec<GpuDevice>,
    partition: LatticePartition,  // Domain decomposition
}
```

### 8.2 GPU-Accelerated Metropolis
Similar kernel design for MCMC:
```wgsl
@compute @workgroup_size(256)
fn metropolis_step(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= params.n_pbits) { return; }

    // Parallel tempering: run 256 replicas in parallel
    let replica_id = id.y;
    let temperature = params.temperatures[replica_id];

    // Calculate energy change for flipping pBit idx
    let delta_e = calculate_energy_change(idx, states, couplings);

    // Metropolis criterion
    let accept_prob = min(1.0, exp(-delta_e / temperature));
    let random = random_buffer[idx];  // Pre-generated RNG

    if (random < accept_prob) {
        // Flip pBit (atomic bit operation)
        atomicXor(&states[idx / 32], 1u << (idx % 32));
    }
}
```

---

## 9. Cost-Benefit Analysis

### Development Cost
| Phase | Hours | Engineer Cost | Total |
|-------|-------|---------------|-------|
| wgpu setup | 8 | $150/hr | $1,200 |
| Kernel dev | 40 | $150/hr | $6,000 |
| Testing | 16 | $150/hr | $2,400 |
| Integration | 16 | $150/hr | $2,400 |
| **Total** | **80** | - | **$12,000** |

### Performance Benefit
| Metric | SIMD | GPU | Improvement |
|--------|------|-----|-------------|
| 10k pBit step | 100 μs | 10 μs | **10× faster** |
| Monthly compute cost | $100 | $10 | **10× cheaper** |

**ROI**: Pays off if >100 hours/month of simulation time
**Decision**: Defer until production workload justifies investment

---

## 10. Dependencies to Add (When Implemented)

```toml
[dependencies]
# GPU acceleration (optional)
wgpu = { version = "0.20", optional = true }
bytemuck = { version = "1.14", features = ["derive"], optional = true }
pollster = { version = "0.3", optional = true }

# CUDA alternative (enterprise)
cudarc = { version = "0.10", optional = true }

[features]
default = []
gpu = ["wgpu", "bytemuck", "pollster"]
gpu-cuda = ["cudarc"]
```

---

## 11. References

### wgpu Resources
- [wgpu Tutorial](https://sotrh.github.io/learn-wgpu/)
- [Compute Shader Guide](https://webgpufundamentals.org/webgpu/lessons/webgpu-compute-shaders.html)
- [WGSL Specification](https://www.w3.org/TR/WGSL/)

### GPU Optimization
- Nvidia, "CUDA C++ Best Practices Guide"
- AMD, "RDNA 3 Optimization Guide"
- Apple, "Metal Compute Best Practices"

### Academic Papers
- Dongarra et al., "HPC Programming on Modern GPU Architectures"
- Karimi et al., "A Performance Comparison of CUDA and OpenCL"

---

**Agent Status**: 🎮 Architecture Complete
**Implementation Status**: DEFERRED to Phase 4
**Blocker**: wgpu compilation + SIMD baseline required
**Next Phase**: Implement SIMD first (SIMD_STRATEGY.md)
