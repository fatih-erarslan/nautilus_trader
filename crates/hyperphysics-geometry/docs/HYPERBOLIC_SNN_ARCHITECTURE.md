# Hyperbolic Spiking Neural Network Architecture

## Comprehensive Technical Specification

**Version**: 1.0.0
**Date**: 2025-12-05
**Authors**: HyperPhysics Team

---

## Table of Contents

1. [Theoretical Foundation](#theoretical-foundation)
2. [Multi-Timescale Framework](#multi-timescale-framework)
3. [Mathematical Specifications](#mathematical-specifications)
4. [Module Architecture](#module-architecture)
5. [Implementation Details](#implementation-details)
6. [References](#references)

---

## 1. Theoretical Foundation

### 1.1 Core Synthesis

This architecture synthesizes four foundational frameworks:

1. **Hyperbolic Geometry** (Kollár et al., 2019)
   - Exponential boundary growth for natural dissipation
   - Geodesic distance encodes axonal delay
   - {p,q} tilings provide precise branching control

2. **Self-Organized Criticality** (Bak et al., 1987)
   - Systems at criticality (σ = 1) maximize information transmission
   - Power-law avalanche distributions: P(s) ~ s^{-3/2}
   - Scale-free dynamics for optimal computation

3. **Active Inference** (Friston, 2010)
   - Free energy minimization as universal principle
   - Bayesian belief propagation along geodesics
   - Prediction error drives learning

4. **Multi-Timescale Language Creation** (Christiansen & Chater, 2016)
   - Utterance timescale (ms): Processing → Spike dynamics
   - Individual timescale (years): Acquisition → STDP learning
   - Historical timescale (generations): Evolution → Topology adaptation

### 1.2 Why Hyperbolic Geometry for SNNs?

#### 1.2.1 Branching Ratio Control

SOC requires branching ratio σ ≈ 1 (each spike triggers ~1 subsequent spike on average).
Hyperbolic tilings give precise geometric control:

```
{p,q} tiling → valence q neighbors → natural branching factor (q-1)

{7,3}: 3 neighbors → σ_base ≈ 2 (supercritical)
{5,4}: 4 neighbors → σ_base ≈ 3 (supercritical)

With inhibition/threshold tuning → drive σ → 1 (critical)
```

#### 1.2.2 Exponential Boundary = Natural Dissipation

SOC requires open boundaries for energy dissipation. Hyperbolic space provides this intrinsically:

```
Perimeter at depth n ∝ (q-1)^n    (exponential)
Interior at depth n  ∝ Σ(q-1)^k   (polynomial sum)

Perimeter/Interior → ∞ as n → ∞
```

This massive boundary surface acts as the "sand pile edge" where avalanches dissipate.

#### 1.2.3 Geodesic Distance = Axonal Delay

STDP (Spike-Timing-Dependent Plasticity) is critically dependent on timing:

```
ΔW ∝ exp(-|Δt|/τ) × sign(t_post - t_pre)
```

Hyperbolic distance provides geometry-grounded delay:

```
delay(i→j) = d_H(p_i, p_j) / c_propagation
```

This creates natural timing relationships that respect the geometric structure.

#### 1.2.4 Lorentz Boost = Spike Wavefront

The Lorentz boost maps perfectly to spike propagation:

```rust
// Rapidity η encodes spike "velocity" / intensity
spike_wavefront = neuron_position.boost(rapidity, direction);

// Information travels along geodesics at finite "speed"
// Creating natural light-cone causality for spikes
```

---

## 2. Multi-Timescale Framework

### 2.1 Mapping from "Creating Language"

| Language Creation | Timescale | SNN Implementation |
|-------------------|-----------|-------------------|
| **Utterance** (Processing) | ~1-100ms | Spike dynamics, membrane potential (LIF) |
| **Acquisition** | ~1-10 years | STDP weight learning, pattern formation |
| **Evolution** | generations | Structural plasticity, topology optimization |

### 2.2 Chunk-and-Pass Hierarchy

From the "Now-or-Never" bottleneck (Christiansen & Chater):

> Because memory is fleeting, new material will rapidly obliterate previous material.
> To successfully deal with the continual deluge of linguistic information, the brain
> must compress and recode its input into "chunks" as rapidly as possible.

**Implementation in Hyperbolic Space:**

```
Layer 0 (origin):    τ = τ_base           (~1ms timescale)
Layer 1:             τ = τ_base × r       (r = growth_rate from tiling)
Layer 2:             τ = τ_base × r²
Layer n:             τ = τ_base × r^n     (exponentially slower)

Hyperbolic depth naturally encodes temporal abstraction level:
  - Boundary: Fast, sensory features
  - Core: Slow, abstract patterns
  - Origin: Global context/attractor
```

### 2.3 Timescale Hierarchy

```
┌─────────────────────────────────────────────────────────────┐
│  ULTRA-FAST (~μs):    SIMD Lorentz operations               │
│  FAST (~1ms):         Membrane potential dynamics           │
│  MEDIUM (~10-100ms):  STDP plasticity window                │
│  SLOW (~1s):          Chunk formation, working memory       │
│  VERY SLOW (~hours):  Weight consolidation                  │
│  STRUCTURAL (~days):  Topology adaptation via SOC           │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Mathematical Specifications

### 3.1 Spiking Neuron Model (LIF on H²)

**Leaky Integrate-and-Fire on Hyperbolic Lattice:**

```
dV_i/dt = -V_i/τ_m + I_syn + I_ext

I_syn = Σ_j W_ij × K(d_H(i,j)) × S_j(t - d_H(i,j)/c)

Where:
  V_i     = membrane potential at node i (on hyperboloid)
  τ_m     = membrane time constant (configurable, typically 10-20ms)
  W_ij    = synaptic weight (learned via STDP)
  K(d)    = hyperbolic coupling kernel
  S_j(t)  = spike train from neuron j: Σ_k δ(t - t_k^j)
  d_H     = hyperbolic distance (geodesic on H²)
  c       = propagation speed (encoded as rapidity)
  I_ext   = external input current
```

**Spike Generation:**
```
if V_i > V_threshold:
    emit spike
    V_i = V_reset
    enter refractory period (τ_ref)
```

**Hyperbolic Coupling Kernel:**
```
K(d) = A × exp(-d/λ) / sinh(d)

Where:
  A = amplitude scaling
  λ = length constant (characteristic distance)
  sinh(d) accounts for hyperbolic volume element
```

### 3.2 Hyperbolic STDP Learning Rule

**Standard STDP:**
```
STDP(Δt) = A+ × exp(-Δt/τ+)  if Δt > 0 (LTP: post after pre)
         = A- × exp(+Δt/τ-)  if Δt < 0 (LTD: post before pre)

Δt = t_post - t_pre
```

**Hyperbolic Modulation:**
```
ΔW_ij = η × STDP(Δt) × Locality(d_H(i,j)) × SOC_factor(σ)

Locality(d) = exp(-d/λ_STDP) × (1 + |K| × curvature_boost)

  K = -1 (hyperbolic curvature)
  curvature_boost > 0 → strengthens local connections

SOC_factor(σ) = 1 + α × (σ_target - σ_measured)

  σ_target = 1.0 (critical point)
  σ_measured = running average of branching ratio
  α = adaptation rate
```

**Properties:**
- Nearby neurons (small d_H) have stronger plasticity
- Distant neurons have weaker but non-zero plasticity
- SOC factor drives system toward criticality

### 3.3 Markovian Heat Kernel on H²

**Heat Equation on Hyperbolic Space:**
```
∂u/∂t = Δ_H u

Where Δ_H is the Laplace-Beltrami operator on H²
```

**Heat Kernel (Fundamental Solution):**
```
K_t(x,y) = (4πt)^{-1} × (d/sinh(d)) × exp(-d²/4t - t/4)

Where d = d_H(x,y) is the hyperbolic distance
```

**Markov Transition Operator:**
```
P(x→y|t) = K_t(x,y) / Z_t(x)

Z_t(x) = ∫_H² K_t(x,z) dμ(z)  (normalization)
```

**Chapman-Kolmogorov Equation:**
```
K_{s+t}(x,y) = ∫_H² K_s(x,z) × K_t(z,y) dμ(z)
```

**Discretization on Lattice:**
```
For lattice nodes {v_i}:

P_ij(t) = K_t(v_i, v_j) / Σ_k K_t(v_i, v_k)

Transition matrix: P(t) with entries P_ij(t)
```

### 3.4 Self-Organized Criticality

**At Criticality (σ = 1):**
```
P(avalanche_size = s) ∝ s^{-τ}        where τ ≈ 3/2
P(avalanche_duration = T) ∝ T^{-α}    where α ≈ 2
⟨s⟩(T) ∝ T^{γ}                        where γ ≈ 2
```

**Scaling Relation:**
```
γ = (α - 1)/(τ - 1)

For τ = 3/2, α = 2: γ = 1/0.5 = 2 ✓
```

**Branching Ratio:**
```
σ = (1/N_spikes) × Σ_i (spikes triggered by spike i)

σ < 1: Subcritical (activity dies out)
σ = 1: Critical (optimal computation)
σ > 1: Supercritical (explosive activity)
```

**Hyperbolic Enhancement of SOC:**
```
Natural boundary dissipation from exponential perimeter
→ No need for artificial boundary conditions
→ Avalanches naturally terminate at boundary
→ System self-organizes to criticality
```

### 3.5 Active Inference

**Free Energy:**
```
F = E_q[log q(z|x) - log p(x,z)]
  = KL[q(z|x)||p(z)] - E_q[log p(x|z)]
  = Complexity - Accuracy
```

**Belief Update (Variational):**
```
q*(z) = argmin_q F[q]

For Gaussian approximate posterior:
  μ_new = μ + η × ∇_μ F
  Σ_new = Σ + η × ∇_Σ F
```

**Integration with Hyperbolic Geometry:**
```
Prior p(z) encodes hyperbolic metric structure:
  - Distance from origin → prior threat probability
  - Perimeter nodes have higher prior uncertainty

Likelihood p(x|z) from observation model:
  - Anomaly score as noisy observation
  - Precision modulated by layer depth
```

### 3.6 Enactive Cognition

**Sensorimotor Loop:**
```
Action: a_t = π(belief_t, affordances)
Observation: o_t = env(a_t)
Belief update: belief_{t+1} = update(belief_t, o_t)
```

**Affordance Encoding in Hyperbolic Space:**
```
Affordance vectors embedded in H² tangent space
Distance between affordances = semantic similarity
Actions = geodesics toward desired affordance
```

**Embodied Interface at Boundary:**
```
Boundary neurons: Sensor/actuator interface
Interior neurons: Processing and representation
Origin neurons: Global state/attractor
```

---

## 4. Module Architecture

### 4.1 File Structure

```
hyperphysics-geometry/src/
├── lib.rs                    # Module exports
├── adversarial_lattice.rs    # Base hyperbolic topology (existing)
├── sentry_integration.rs     # SIMD + active inference (existing)
├── hyperbolic_snn.rs         # NEW: Core SNN module
├── markov_kernels.rs         # NEW: Markovian dynamics
├── chunk_processor.rs        # NEW: Chunk-and-Pass
└── enactive_layer.rs         # NEW: Enactive cognition
```

### 4.2 Module Dependencies

```
┌─────────────────────────────────────────────────────────────┐
│                      lib.rs (exports)                       │
└─────────────────────────────────────────────────────────────┘
                              │
      ┌───────────────────────┼───────────────────────┐
      │                       │                       │
      ▼                       ▼                       ▼
┌─────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ hyperbolic_ │     │ markov_kernels  │     │ chunk_processor │
│    snn      │     │                 │     │                 │
└─────────────┘     └─────────────────┘     └─────────────────┘
      │                       │                       │
      └───────────────────────┼───────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │ enactive_layer  │
                    └─────────────────┘
                              │
      ┌───────────────────────┼───────────────────────┐
      │                       │                       │
      ▼                       ▼                       ▼
┌─────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ adversarial │     │    sentry_      │     │   poincare      │
│  _lattice   │     │  integration    │     │                 │
└─────────────┘     └─────────────────┘     └─────────────────┘
```

### 4.3 Key Types

#### hyperbolic_snn.rs
```rust
pub struct SpikingNeuron {
    pub id: usize,
    pub position: LorentzVec,
    pub membrane_potential: f64,
    pub threshold: f64,
    pub reset_potential: f64,
    pub refractory_remaining: f64,
    pub spike_times: Vec<f64>,
    pub layer: usize,
}

pub struct Synapse {
    pub pre_id: usize,
    pub post_id: usize,
    pub weight: f64,
    pub delay: f64,  // From hyperbolic distance
}

pub struct HyperbolicSTDP {
    pub a_plus: f64,
    pub a_minus: f64,
    pub tau_plus: f64,
    pub tau_minus: f64,
    pub lambda_stdp: f64,
    pub curvature_boost: f64,
}

pub struct HyperbolicSNN {
    pub neurons: Vec<SpikingNeuron>,
    pub synapses: Vec<Synapse>,
    pub stdp: HyperbolicSTDP,
    pub soc_monitor: SOCMonitor,
    pub time: f64,
}
```

#### markov_kernels.rs
```rust
pub struct HyperbolicHeatKernel {
    pub diffusion_time: f64,
}

pub struct TransitionMatrix {
    pub matrix: Vec<Vec<f64>>,
    pub node_positions: Vec<LorentzVec>,
}

pub struct ChapmanKolmogorov {
    pub kernel: HyperbolicHeatKernel,
}
```

#### chunk_processor.rs
```rust
pub struct TemporalChunk {
    pub content: Vec<f64>,
    pub timestamp: f64,
    pub layer: usize,
    pub compression_ratio: f64,
}

pub struct ChunkProcessor {
    pub layers: Vec<ChunkLayer>,
    pub timescale_ratio: f64,
}

pub struct WorkingMemory {
    pub capacity: usize,
    pub chunks: VecDeque<TemporalChunk>,
}
```

#### enactive_layer.rs
```rust
pub struct SensorimotorLoop {
    pub sensors: Vec<usize>,      // Boundary neuron IDs
    pub actuators: Vec<usize>,    // Boundary neuron IDs
    pub belief_state: Vec<f64>,
}

pub struct Affordance {
    pub position: LorentzVec,
    pub action_type: ActionType,
    pub salience: f64,
}

pub struct EnactiveAgent {
    pub snn: HyperbolicSNN,
    pub sensorimotor: SensorimotorLoop,
    pub affordances: Vec<Affordance>,
}
```

---

## 5. Implementation Details

### 5.1 Numerical Stability

**Hyperbolic Distance:**
```rust
// Avoid numerical issues near boundary
pub fn safe_acosh(x: f64) -> f64 {
    if x <= 1.0 {
        0.0
    } else if x > 1e10 {
        x.ln() + std::f64::consts::LN_2
    } else {
        x.acosh()
    }
}
```

**Heat Kernel Normalization:**
```rust
// Ensure proper probability distribution
pub fn normalize_transitions(row: &mut [f64]) {
    let sum: f64 = row.iter().sum();
    if sum > 1e-10 {
        for p in row.iter_mut() {
            *p /= sum;
        }
    }
}
```

### 5.2 Performance Optimization

**SIMD Operations:**
- Use `LorentzVec` for all position computations
- Batch distance calculations
- Vectorized spike propagation

**Memory Layout:**
- Struct of Arrays (SoA) for neuron data
- Cache-friendly iteration patterns
- Preallocated spike buffers

### 5.3 Testing Strategy

**Unit Tests:**
- Neuron dynamics (LIF equation)
- STDP weight updates
- Heat kernel normalization
- Chunk formation

**Integration Tests:**
- Full network simulation
- SOC emergence verification
- Power-law validation

**Property-Based Tests:**
- Chapman-Kolmogorov consistency
- Distance metric properties
- Energy conservation

---

## 6. References

### Primary Sources

1. Kollár, A. J., et al. (2019). "Hyperbolic lattices in circuit quantum electrodynamics." *Nature*, 571:45-50.

2. Bak, P., Tang, C., & Wiesenfeld, K. (1987). "Self-organized criticality: An explanation of 1/f noise." *Physical Review Letters*, 59(4):381-384.

3. Friston, K. (2010). "The free-energy principle: a unified brain theory?" *Nature Reviews Neuroscience*, 11:127-138.

4. Christiansen, M. H., & Chater, N. (2016). *Creating Language: Integrating Evolution, Acquisition, and Processing*. MIT Press.

### Secondary Sources

5. Kinouchi, O., & Copelli, M. (2006). "Optimal dynamical range of excitable networks at criticality." *Nature Physics*, 2:348-351.

6. Bertschinger, N., & Natschläger, T. (2004). "Real-time computation at the edge of chaos in recurrent neural networks." *Neural Computation*, 16:1413-1436.

7. Nickel, M., & Kiela, D. (2017). "Poincaré embeddings for learning hierarchical representations." *NeurIPS*.

8. Parisi, G. (1988). *Statistical Field Theory*. Perseus Books.

### Mathematical Background

9. Cannon, J. W., et al. (1997). "Hyperbolic Geometry." In *Flavors of Geometry*, MSRI Publications 31.

10. Grigor'yan, A. (2009). *Heat Kernel and Analysis on Manifolds*. AMS/IP Studies in Advanced Mathematics.

---

## Appendix A: Quick Reference

### Constants

| Symbol | Value | Description |
|--------|-------|-------------|
| K | -1 | Hyperbolic curvature |
| τ_m | 10-20ms | Membrane time constant |
| τ_+ | 20ms | LTP time constant |
| τ_- | 20ms | LTD time constant |
| σ_target | 1.0 | Critical branching ratio |
| τ | 3/2 | Avalanche size exponent |
| α | 2.0 | Avalanche duration exponent |

### Key Equations Summary

```
LIF:     dV/dt = -V/τ + I_syn + I_ext
STDP:    ΔW = η × STDP(Δt) × Locality(d) × SOC(σ)
Heat:    K_t(x,y) ∝ (d/sinh d) × exp(-d²/4t - t/4)
SOC:     P(s) ~ s^{-3/2}, P(T) ~ T^{-2}
FE:      F = KL[q||p] - E_q[log p(x|z)]
```

---

*Document generated for HyperPhysics project. For questions, see the source code documentation.*
