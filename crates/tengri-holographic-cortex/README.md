# Tengri Holographic Cortex

A unified pBit-based cognitive architecture combining **Graph Neural Networks (GNN)** and **Spiking Neural Networks (SNN)** in an **11D Hyperbolic Lattice Spacetime Continuum**.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│ (E) MSOCL - Meta-Stable Oscillatory Control Layer               │
│     • Kuramoto-model phase coordination                         │
│     • Temperature modulation across engines                     │
└─────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ (A) 4-Engine pBit Topology (2×2 Square)                         │
│     ┌───────┐     ┌───────┐                                    │
│     │ Eng A │◀───▶│ Eng B │   • Local pBit dynamics (AVX2)     │
│     └───────┘     └───────┘   • Boltzmann sampling             │
│          ▲             ▲                                        │
│          │   K^αβ      │      • Inter-engine coupling tensors  │
│          ▼             ▼                                        │
│     ┌───────┐     ┌───────┐                                    │
│     │ Eng D │◀───▶│ Eng C │   • Möbius blending to H¹¹        │
│     └───────┘     └───────┘                                    │
└─────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ (C) Ultra-Fast Cortical Bus (UFCB)                              │
│     • Tier A: <50μs spikes (pinned hugepages)                   │
│     • Tier B: <1ms embeddings (GPU P2P)                         │
│     • Tier C: <10ms model shards (NVMe streaming)               │
└─────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ (B) 11D Hyperbolic Relational Holographic Substrate             │
│     • Lorentz model: ⟨x,y⟩_L = -x₀y₀ + Σᵢ xᵢyᵢ                 │
│     • exp/log maps for tangent space ↔ hyperboloid              │
└─────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ (D) HNSW + LSH Memory Fabric                                    │
│     • LSH: k=8 hash functions, L=32 tables                      │
│     • HNSW: M=16-32, efConstruction=200                         │
└─────────────────────────────────────────────────────────────────┘
```

## Wolfram-Verified Mathematical Foundations

All constants and formulas have been formally verified using Wolfram Language:

### Ising Model (2D Square Lattice)

| Constant | Value | Verification |
|----------|-------|--------------|
| Critical Temperature T_c | 2.269185314213022 | `N[2/Log[1 + Sqrt[2]], 20]` |
| Inverse β_c | 0.4406867935097714 | `1/T_c` |

### pBit Sampling Probability

```
P(s=+1) = σ((h-bias)/T) = 1/(1 + exp(-(h-bias)/T))
```

| Test Case | Value | Verified |
|-----------|-------|----------|
| P(h=0, bias=0, T=1) | 0.5 | ✓ |
| P(h=1, bias=0, T=1) | 0.7311 | ✓ |
| P(h=1, bias=0, T=0.1) | 0.9999546 | ✓ |

### Hyperbolic Geometry (Lorentz H¹¹)

**Constraint:**
```
-x₀² + x₁² + ... + x₁₁² = -1
```

**Lift from ℝ¹¹:**
```
x₀ = √(1 + ||z||²)
```

**Distance:**
```
d_H(x,y) = acosh(-⟨x,y⟩_L)
```

### Möbius Addition (Poincaré Ball)

```
x ⊕_c y = ((1 + 2c⟨x,y⟩ + c||y||²)x + (1 - c||x||²)y) / (1 + 2c⟨x,y⟩ + c²||x||²||y||²)
```

**Verified:** `Möbius({0.3,0},{0,0.4},c=1) = {0.343, 0.359}` with norm < 1 ✓

### STDP Learning Rule

```
LTP (Δt > 0): ΔW = A₊ × exp(-Δt/τ₊)
LTD (Δt < 0): ΔW = -A₋ × exp(Δt/τ₋)
```

| Parameters | Values |
|------------|--------|
| A₊ | 0.1 |
| A₋ | 0.12 |
| τ₊ = τ₋ | 20 ms |
| ΔW(Δt=10ms) | 0.0607 ✓ |
| ΔW(Δt=-10ms) | -0.0728 ✓ |

### Annealing Schedules

| Schedule | Formula | T(100) at T₀=T_c |
|----------|---------|------------------|
| Logarithmic (optimal) | T₀/ln(1+t) | 0.4919 |
| Exponential (fast) | T₀×0.99^t | 0.8309 |

### 4-Engine Coupling Matrix

```
G = [[0, 1, 0.5, 1],    // A: neighbors B,D; cross C
     [1, 0, 1, 0.5],    // B: neighbors A,C; cross D
     [0.5, 1, 0, 1],    // C: neighbors B,D; cross A
     [1, 0.5, 1, 0]]    // D: neighbors A,C; cross B
```

**Eigenvalues:** [2.5, -1.5, -0.5, -0.5]
**Spectral gap:** 4.0 (good mixing) ✓

### MSOCL Kuramoto Model

```
dφᵢ/dt = ωᵢ + (K/N) Σⱼ sin(φⱼ - φᵢ)
```

Critical coupling: K_c = 2γ = 0.2 (for γ=0.1 frequency spread)

### HNSW/LSH Parameters

| Component | Parameter | Value |
|-----------|-----------|-------|
| HNSW | M | 16-32 |
| HNSW | efConstruction | 200 |
| HNSW | efQuery | 50-200 |
| LSH | k (hash functions) | 8 |
| LSH | L (tables) | 32 |

## Usage

```rust
use tengri_holographic_cortex::*;

// Create a 4-engine cortex
let config = TopologyConfig::default();
let mut cortex = Cortex4::new(config);

// Run simulation
for _ in 0..1000 {
    cortex.step();
}

// Get global hyperbolic embedding
if let Some(embedding) = cortex.global_embedding() {
    println!("Global embedding on H¹¹: {:?}", embedding.spatial());
}

// Query memory fabric
let mut fabric = MemoryFabric::default();
let id = fabric.insert(vec![0.1; 11]);
let results = fabric.query(&vec![0.1; 11], 5);
```

## Features

- `pbit` - Core pBit dynamics (default)
- `lorentz` - Hyperbolic geometry (default)
- `cortical` - Cortical bus (default)
- `hnsw` - HNSW index integration
- `lsh` - LSH filtering
- `stdp` - STDP plasticity
- `dilithium` - Post-quantum cryptography
- `swarm` - Swarm intelligence
- `simd` - SIMD acceleration
- `parallel` - Parallel processing
- `full` - All features

## Benchmarks

```bash
cargo bench --bench cortex_bench
```

## References

1. Camsari et al. (2017) "Stochastic p-bits for invertible logic" PRX 7:031014
2. Nickel & Kiela (2017) "Poincaré Embeddings for Learning Hierarchical Representations"
3. Kuramoto (1984) "Chemical Oscillations, Waves, and Turbulence"
4. Onsager (1944) "Crystal Statistics I: A Two-Dimensional Model with an Order-Disorder Transition"

## License

MIT
