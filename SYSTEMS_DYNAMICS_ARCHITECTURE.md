# Systems Dynamics Architecture Overview

## Module Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        SYSTEMS DYNAMICS TRACKER                              │
│                     (systems_dynamics.rs, ~1,100 lines)                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                ┌─────────────────────┼─────────────────────┐
                │                     │                     │
                ▼                     ▼                     ▼
        ┌───────────────┐    ┌────────────────┐   ┌──────────────┐
        │   Recording   │    │   Analysis     │   │    Export    │
        │   Pipeline    │    │   Pipeline     │   │   Pipeline   │
        └───────────────┘    └────────────────┘   └──────────────┘
                │                     │                     │
        ┌───────▼────────┐   ┌────────▼─────────┐  ┌────────▼──────┐
        │ StateSnapshot  │   │ TemporalStats    │  │ CSV Format    │
        │ + Circular     │   │ + Criticality    │  │ + JSON Format │
        │   Buffer       │   │   Metrics        │  │ + Series API  │
        └────────────────┘   │ + Spectral       │  └───────────────┘
                             │   Analysis       │
                             │ + Emergence      │
                             │   Indicators     │
                             └──────────────────┘
```

## Data Flow

```
Agent Step Loop:
    │
    ▼
┌─────────────────────────────────────┐
│  Agent Computes:                    │
│  - Φ (consciousness)                │
│  - F (free energy)                  │
│  - u (control)                      │
│  - S (survival)                     │
│  - A (model accuracy)               │
│  - Prediction error                 │
│  - Belief magnitude                 │
│  - Avg precision                    │
└──────────────┬──────────────────────┘
               │
               ▼
    ┌──────────────────────┐
    │  record_state()      │
    │  8 parameters        │
    └──────────┬───────────┘
               │
               ▼
    ┌──────────────────────────────┐
    │  StateSnapshot created       │
    │  with all metrics            │
    └──────────┬───────────────────┘
               │
               ▼
    ┌──────────────────────────────┐
    │  Circular Buffer             │
    │  (max 10,000 snapshots)      │
    │  - FIFO eviction             │
    │  - Cache invalidation        │
    └──────────────────────────────┘
```

## Analysis Pipelines

### Pipeline 1: Criticality Detection

```
History Buffer (N snapshots)
    │
    ├─► Extract Control Series ──┐
    │                             │
    ├─► Count Active States       ├─► Branching Ratio (σ)
    │   (control > 0.1)           │   σ ≈ 1 = Critical
    │                             │
    └─────────────────────────────┘
           │
           ├─► Φ Series ──────────┐
           │                      │
           ├─► Rescaled Range ────┼─► Hurst Exponent (H)
           │                      │   H > 0.5 = Trending
           │                      │
           └──────────────────────┘
           │
           ├─► Control Divergence ┐
           │                      │
           └─────────────────────┬┼─► Lyapunov (λ)
                                 │   λ > 0 = Chaotic
                              FE Series
                                 │
                         ┌────────▼──────┐
                         │ Quantize to   │
                         │ 10 bins       │
                         │ Shannon H     │
                         └───────────────┘
                              │
                              ▼
                        Entropy Rate
```

### Pipeline 2: Spectral Analysis

```
Φ Time Series
    │
    ├─► Test 8 Frequencies ┐
    │   (0.1 to 0.8 norm)  │
    │                      ├─► Periodogram
    ├─► Compute cos/sin    │   P(f) = |sum|²
    │   components for     │
    │   each frequency     └─► Peak Frequency
    │                         Peak Power
    ├─► Normalize to Total
    │   Power
    │
    ├─► Shannon Entropy ────────► Spectral Entropy
    │   on normalized power
    │
    └─► Detect Harmonics ──────► Harmonic List
        at multiples of peak
```

### Pipeline 3: Statistical Analysis

```
For each metric (Φ, F, u, S, A):

Time Series Values
    │
    ├─► Sort ──────────────► Min, Max
    │
    ├─► Sum/Count ─────────► Mean
    │
    ├─► (X - μ)² ──────────► Std Dev
    │
    ├─► Lag-1 Covariance ──► Autocorrelation
    │   Cov(X_t, X_{t+1})
    │
    ├─► First Differences ─► Volatility
    │   std(ΔX)
    │
    ├─► Normalized ────────► Skewness
    │   Moment 3
    │
    └─► Normalized ────────► Kurtosis
        Moment 4 - 3
```

## State Dimensions

```
┌─────────────────────────────────────────────────────────────┐
│              AGENT STATE (9 dimensions)                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Dimension 1: CONSCIOUSNESS                                │
│  ├─ Φ (Integrated Information) ∈ [0, 10]                  │
│  └─ Measures: System integration, coherence               │
│                                                             │
│  Dimension 2: ENERGY                                       │
│  ├─ F (Free Energy) ∈ [0, ∞)                              │
│  └─ Measures: Surprise, prediction error                  │
│                                                             │
│  Dimension 3: AGENCY                                       │
│  ├─ u (Control Authority) ∈ [0, 1]                        │
│  └─ Measures: Ability to influence environment            │
│                                                             │
│  Dimension 4: SURVIVAL                                     │
│  ├─ S (Survival Drive) ∈ [0, 1]                           │
│  └─ Measures: Response to energy threats                  │
│                                                             │
│  Dimension 5: ACCURACY                                     │
│  ├─ A (Model Accuracy) ∈ [0, 1]                           │
│  └─ Measures: Prediction correctness                      │
│                                                             │
│  Dimension 6: STATE MAGNITUDE                              │
│  ├─ ||b|| (Belief Magnitude) ∈ [0, ∞)                     │
│  └─ Measures: Strength of internal state                  │
│                                                             │
│  Dimension 7: UNCERTAINTY                                  │
│  ├─ p̄ (Avg Precision) ∈ [0, ∞)                            │
│  └─ Measures: 1/uncertainty inverse                       │
│                                                             │
│  Dimension 8: ERROR                                        │
│  ├─ ε (Prediction Error) ∈ [0, ∞)                         │
│  └─ Measures: Observation surprise                        │
│                                                             │
│  Dimension 9: TIME                                         │
│  ├─ t ∈ [0, ∞)                                            │
│  └─ Measures: Temporal ordering                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Metrics Computation Graph

```
StateSnapshot (9 values)
    │
    ├──────────────────────────────────────┐
    │                                      │
    ▼                                      ▼
Phi Stats                          Control Stats
├─ mean_phi                        ├─ mean_control
├─ std_phi                         ├─ std_control
├─ autocorr_phi                    ├─ autocorr_control
├─ volatility_phi                  ├─ volatility_control
├─ skew_phi                        └─ skew_control
└─ kurtosis_phi
    │                                  │
    │                                  │
    └────────────────────┬─────────────┘
                         │
                         ▼
              DynamicsStats Aggregation
              ├─ phi: TemporalStats
              ├─ free_energy: TemporalStats
              ├─ control: TemporalStats
              ├─ survival: TemporalStats
              └─ samples: usize
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
   Emergence       Robustness         Criticality
   Indicator       Score              Metrics
   │               │                  │
   0.30×Φ/10      0.60×              Branching
   0.20×FE_stab    stability          Ratio (σ)
   0.30×Control    0.40×              Hurst (H)
   0.20×ρ_surv     surv_skew          Lyapunov (λ)
   │               │                  Entropy (H)
   Range: [0,1]   Range: [0,1]       Range: varies
```

## Emergence Evolution

```
Time →

Initial State (t=0):
Φ ═════ Control ═════ Emergence ═════ Robustness
1.0     0.2          0.15            0.30
█ low   █ low        █ random        █ fragile


Developing (t=250):
Φ ═════════════ Control ═════════ Emergence ═════════ Robustness
3.0     [growing]   0.5  [growing]  0.45         [growing] 0.50
████ moderate       ████ moderate    ███ moderate            ████


Mature (t=500):
Φ ════════════════════ Control ══════════════ Emergence ════════════ Robustness
5.0  [strong]         0.75  [strong]        0.70 [strong]        0.75
████████ strong       ████████ strong       ███████ strong       ███████


Critical (t=750):
Φ ══════════════════════════ Control ════════════════════ Emergence ════════════
6.5 [very strong]           0.85 [very strong]          0.82 [very strong]
████████████ critical       ██████████ critical        ████████ critical

σ (Branching Ratio) → 0.98 ≈ 1.0 (CRITICALITY!)
```

## Export Flow

```
History Buffer
    │
    ├─────────────────────────────────────┐
    │                                     │
    ▼                                     ▼
CSV Export                          JSON Export
├─ Header row                       └─ Array of
│  time,phi,free_energy,...            StateSnapshot
├─ Data rows                           objects
│  0,2.5,1.5,...                       with all
│  1,2.6,1.4,...                       fields
│  ...
└─ Plain text                      JSON format
   (Excel/Pandas)                  (Web/Analytics)
    │                                  │
    └──────────────┬───────────────────┘
                   │
         File Export or API
```

## Integration with CyberneticAgent

```
┌────────────────────────────────────────┐
│       CyberneticAgent::step()          │
├────────────────────────────────────────┤
│                                        │
│ 1. Perception Phase                    │
│    └─ update_beliefs()                 │
│                                        │
│ 2. Consciousness Phase                 │
│    └─ compute_phi() → Φ                │
│                                        │
│ 3. Free Energy Phase                   │
│    └─ free_energy.compute() → F        │
│                                        │
│ 4. Survival Phase                      │
│    └─ survival.compute_drive() → S     │
│                                        │
│ 5. Control Phase                       │
│    └─ update_control() → u             │
│                                        │
│ 6. Action Phase                        │
│    └─ policy_selector.select()         │
│                                        │
│ 7. Adaptation Phase                    │
│    └─ adapt() [impermanence]           │
│                                        │
│ 8. Dynamics Tracking                   │
│    └─ dynamics.record_state(           │
│         Φ, F, u, S, A, ε, ||b||, p̄)   │
│                                        │
└────────────────────────────────────────┘
         │
         └─────► AgencyDynamics History
                 (10,000 snapshots max)
```

## Memory Layout

```
AgencyDynamics
├─ history: VecDeque<StateSnapshot>
│  └─ [0]: StateSnapshot { φ: 1.0, fe: 2.5, ... }
│  └─ [1]: StateSnapshot { φ: 1.1, fe: 2.4, ... }
│  └─ ...
│  └─ [9999]: StateSnapshot { φ: 5.2, fe: 1.1, ... }
│
├─ max_history: usize = 10000
│
├─ criticality: CriticalityMetrics
│  ├─ branching_ratio: Option<f64>
│  ├─ lyapunov_exponent: Option<f64>
│  ├─ hurst_exponent: Option<f64>
│  └─ entropy_rate: Option<f64>
│
├─ spectral: SpectralAnalysis
│  ├─ peak_frequency: Option<f64>
│  ├─ peak_power: Option<f64>
│  ├─ spectral_entropy: Option<f64>
│  └─ harmonics: Vec<f64>
│
├─ stats_cache: Option<DynamicsStats>
│  ├─ phi: TemporalStats
│  ├─ free_energy: TemporalStats
│  ├─ control: TemporalStats
│  └─ survival: TemporalStats
│
└─ cache_valid: bool
```

## Computation Complexity

```
Operation                  Time Complexity    Space Complexity
──────────────────────────────────────────────────────────────
record_state()             O(1)               O(1)
branching_ratio()          O(n)               O(1)
compute_criticality()      O(n)               O(1)
  - estimate_lyapunov()    O(n)               O(1)
  - estimate_hurst()       O(n)               O(1)
  - compute_entropy_rate() O(n)               O(1)
analyze_spectral()         O(n)               O(8)
get_stats()                O(n) [on first]    O(n) [cached]
                           O(1) [on cache]    O(1)
export_csv()               O(n)               O(n)
export_json()              O(n)               O(n)
──────────────────────────────────────────────────────────────

where n = number of snapshots in history
```

## Test Coverage Matrix

```
┌──────────────────────────┬──────┬────────┬─────────┬──────────┐
│ Test Name                │ Φ    │ F      │ Control │ Emerging │
├──────────────────────────┼──────┼────────┼─────────┼──────────┤
│ Emergence (500 steps)    │ ✓    │ ✓      │ ✓       │ ✓        │
│ Criticality Detection    │ ✓    │ -      │ ✓       │ ✓        │
│ Spectral Analysis        │ ✓    │ -      │ -       │ -        │
│ FE Minimization Traje... │ -    │ ✓      │ -       │ ✓        │
│ Temporal Statistics      │ ✓    │ -      │ ✓       │ -        │
│ Robustness Indicators    │ ✓    │ ✓      │ ✓       │ ✓        │
│ CSV/JSON Export          │ ✓    │ ✓      │ ✓       │ ✓        │
│ Full Criticality + Spec. │ ✓    │ ✓      │ ✓       │ ✓        │
│ Ensemble (10 agents)     │ ✓    │ -      │ ✓       │ ✓        │
│ Long Horizon (2000 steps)│ ✓    │ ✓      │ ✓       │ ✓        │
└──────────────────────────┴──────┴────────┴─────────┴──────────┘

Legend: ✓ = tested, - = not applicable
```

This architecture provides a complete, mathematically rigorous framework for tracking and analyzing the emergence of agency in cybernetic systems.
