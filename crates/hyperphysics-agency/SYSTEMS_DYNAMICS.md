# Systems Dynamics Tracker for Cybernetic Agency

## Overview

The Systems Dynamics module provides comprehensive time series tracking and analysis of agent dynamics across multiple dimensions:

- **Consciousness Metrics**: Integrated Information (Φ)
- **Energy Dynamics**: Free Energy (F), Survival Drive
- **Control Theory**: Control Authority, Model Accuracy
- **Criticality Analysis**: Branching Ratio, Lyapunov Exponents
- **Spectral Properties**: Power Spectral Density, Harmonics
- **Emergence Detection**: Agency indicators, Robustness scores

## Mathematical Foundation

### Core Metrics

#### 1. **Integrated Information (Φ)**
Measures consciousness using Integrated Information Theory:
```
Φ = min(p_partition) I(X; Y | partition)
```
Where the minimum is taken over all possible bipartitions of the system.

#### 2. **Variational Free Energy (F)**
Free Energy Principle minimization metric:
```
F = D[q(h)||p(h|o)] - ln p(o)
  = KL(q||p) - ln p(o)
```
Bounds surprise (prediction error) from above.

#### 3. **Control Authority (u)**
Emerges from consciousness and model accuracy:
```
u = Φ × A × (1 + α × S)
```
Where:
- Φ = Integrated Information
- A = Model Accuracy
- S = Survival Drive
- α = Modulation factor

#### 4. **Branching Ratio (σ)** - Self-Organized Criticality
Criticality metric from avalanche dynamics:
```
σ = N_{t+1} / N_t
```
Where N_t is the number of "active" agents at time t.
- σ < 1.0: Sub-critical (activity dies out)
- σ ≈ 1.0: Critical (self-similar cascades at all scales)
- σ > 1.0: Super-critical (runaway cascades)

### Advanced Analysis

#### Autocorrelation Function (ACF)
Temporal dependency measure:
```
ρ(k) = Cov(X_t, X_{t+k}) / Var(X)
```
High ACF indicates persistent dynamics.

#### Hurst Exponent
Long-range dependence indicator:
```
H ∈ [0, 2]
H = 0.5: Random walk
H > 0.5: Persistent (trending)
H < 0.5: Mean-reverting
```

#### Lyapunov Exponent
Divergence rate in phase space:
```
λ = lim_{t→∞} (1/t) ln|Δx(t)|/|Δx(0)|
```
λ > 0: Chaotic
λ = 0: Critical
λ < 0: Stable

#### Entropy Rate
Information production rate:
```
H = -Σ p(X_t) log p(X_t)
```

## API Reference

### Core Types

#### `StateSnapshot`
Single time-step snapshot of agent state:

```rust
pub struct StateSnapshot {
    pub phi: f64,                  // Integrated information
    pub free_energy: f64,          // Surprise minimization
    pub control: f64,              // Control authority [0,1]
    pub survival: f64,             // Survival drive [0,1]
    pub model_accuracy: f64,       // Prediction accuracy [0,1]
    pub time: u64,                 // Time step
    pub prediction_error: f64,     // Magnitude of error
    pub belief_magnitude: f64,     // L2 norm of beliefs
    pub avg_precision: f64,        // Inverse uncertainty
}
```

#### `AgencyDynamics`
Main tracking struct with history buffer:

```rust
pub struct AgencyDynamics {
    // Recording
    pub fn record_state(&mut self, phi, free_energy, control,
                       survival, model_accuracy, ...) { }

    // Analysis
    pub fn branching_ratio(&mut self) -> Option<f64> { }
    pub fn compute_criticality(&mut self) -> CriticalityMetrics { }
    pub fn analyze_spectral(&mut self) -> SpectralAnalysis { }
    pub fn get_stats(&mut self) -> Option<DynamicsStats> { }

    // Export
    pub fn export_csv(&self) -> String { }
    pub fn export_json(&self) -> Result<String, Error> { }

    // Time Series
    pub fn get_series(&self, metric: &str) -> Vec<f64> { }
}
```

#### `TemporalStats`
Statistical summary of time series:

```rust
pub struct TemporalStats {
    pub mean: f64,              // Mean value
    pub std: f64,               // Standard deviation
    pub min: f64,               // Minimum value
    pub max: f64,               // Maximum value
    pub autocorr_lag1: f64,     // First-order autocorrelation
    pub volatility: f64,        // First-difference std
    pub skewness: f64,          // Asymmetry (-3 to +3)
    pub kurtosis: f64,          // Tail heaviness (excess)
    pub count: usize,           // Number of samples
}
```

#### `DynamicsStats`
Aggregated dynamics statistics:

```rust
pub struct DynamicsStats {
    pub phi: TemporalStats,             // Consciousness statistics
    pub free_energy: TemporalStats,     // Energy statistics
    pub control: TemporalStats,         // Control statistics
    pub survival: TemporalStats,        // Survival statistics
    pub samples: usize,                 // Total snapshots
}

// Methods
impl DynamicsStats {
    // Emergence indicator [0, 1]
    pub fn emergence_indicator(&self) -> f64 { }

    // Robustness score [0, 1]
    pub fn robustness_score(&self) -> f64 { }
}
```

#### `CriticalityMetrics`
Self-organized criticality analysis:

```rust
pub struct CriticalityMetrics {
    pub branching_ratio: Option<f64>,      // σ (criticality)
    pub avalanche_mean: Option<f64>,       // Mean cascade size
    pub avalanche_exponent: Option<f64>,   // Power law exponent
    pub lyapunov_exponent: Option<f64>,    // Divergence rate
    pub hurst_exponent: Option<f64>,       // Long-range dependence
    pub entropy_rate: Option<f64>,         // Information production
}
```

#### `SpectralAnalysis`
Frequency domain analysis:

```rust
pub struct SpectralAnalysis {
    pub peak_frequency: Option<f64>,    // Dominant frequency
    pub peak_power: Option<f64>,        // Spectral magnitude
    pub pink_noise_exponent: Option<f64>, // 1/f slope
    pub spectral_entropy: Option<f64>,  // Frequency complexity
    pub harmonics: Vec<f64>,            // Harmonic frequencies
}
```

## Usage Examples

### Basic Recording

```rust
use hyperphysics_agency::systems_dynamics::AgencyDynamics;

let mut dynamics = AgencyDynamics::new();

for step in 0..1000 {
    // Get agent metrics (from your agent implementation)
    let (phi, fe, control, survival, model_acc) = agent.get_metrics();

    // Record state
    dynamics.record_state(
        phi,           // Integrated information
        fe,            // Free energy
        control,       // Control authority
        survival,      // Survival drive
        model_acc,     // Model accuracy
        pred_error,    // Prediction error
        belief_mag,    // Belief magnitude
        avg_prec,      // Average precision
    );
}
```

### Criticality Detection

```rust
// Compute criticality metrics
let criticality = dynamics.compute_criticality();

match criticality.branching_ratio {
    Some(sigma) if (sigma - 1.0).abs() < 0.1 => {
        println!("System at criticality!");
    }
    Some(sigma) if sigma < 1.0 => {
        println!("Sub-critical: cascades die out");
    }
    Some(sigma) => {
        println!("Super-critical: runaway cascades");
    }
    None => println!("Insufficient data"),
}

if let Some(hurst) = criticality.hurst_exponent {
    if hurst > 0.5 {
        println!("System shows trending behavior");
    } else {
        println!("System is mean-reverting");
    }
}
```

### Spectral Analysis

```rust
let spectral = dynamics.analyze_spectral();

println!("Dominant oscillation: {:?} Hz", spectral.peak_frequency);
println!("Spectral complexity: {:?}", spectral.spectral_entropy);
println!("Harmonics: {:?}", spectral.harmonics);
```

### Emergence Metrics

```rust
if let Some(stats) = dynamics.get_stats() {
    let emergence = stats.emergence_indicator();
    let robustness = stats.robustness_score();

    println!("Agency emergence: {:.1}%", emergence * 100.0);
    println!("System robustness: {:.1}%", robustness * 100.0);
}
```

### Data Export

```rust
// CSV for Excel/Pandas
let csv = dynamics.export_csv();
std::fs::write("dynamics.csv", csv)?;

// JSON for web/analytics
let json = dynamics.export_json()?;
std::fs::write("dynamics.json", json)?;

// Get specific time series
let phi_series = dynamics.get_series("phi");
let control_series = dynamics.get_series("control");
```

## Emergence Indicator Calculation

The emergence indicator combines multiple dimensions:

```
Emergence = 0.3 × (Φ/10)
          + 0.2 × (1 - σ_FE/(σ̄_FE + 0.1))
          + 0.3 × Control
          + 0.2 × |ρ_survival|

Range: [0, 1]
```

**Interpretation:**
- **0.0-0.3**: Minimal agency (random fluctuations)
- **0.3-0.6**: Moderate agency (goal-directed behavior emerging)
- **0.6-0.9**: Strong agency (stable control)
- **0.9-1.0**: Perfect agency (full consciousness + control)

## Robustness Score Calculation

Measures stability and persistence:

```
Robustness = 0.6 × (1 - σ_control/(σ̄_control + 0.1))
           + 0.4 × (1 - |skew_survival|/10)

Range: [0, 1]
```

## Time Series Metrics

### Autocorrelation (Lag-1)
Measures temporal dependencies:
```
ρ(1) = Cov(X_t, X_{t+1}) / Var(X)
```

**Interpretation:**
- **ρ > 0.7**: Strong persistence (trending)
- **ρ ≈ 0.0**: No correlation (random)
- **ρ < -0.5**: Mean-reverting

### Volatility
First-order differences volatility:
```
σ = √(E[ΔX²])
```

Lower volatility indicates stability.

### Skewness & Kurtosis
Higher-order moments:
- **Skewness**: Asymmetry in distribution
- **Kurtosis**: Tail heaviness (excess >0 = heavy tails)

## Spectral Analysis Details

### Simple Periodogram
Simplified FFT at key frequencies:

```
P(f) = (1/N) |Σ_t x_t × e^(-2πift)|²
```

### Peak Detection
Finds dominant oscillation frequency and power.

### Harmonic Identification
Detects multiples of peak frequency.

### Spectral Entropy
Frequency domain complexity:
```
H_s = -Σ_f p(f) log₂ p(f)
```

## Performance Characteristics

- **Recording**: O(1) per state
- **History buffer**: Circular buffer, max 10,000 states (configurable)
- **Branching ratio**: O(n) where n = history length
- **Spectral analysis**: O(n log n) via simplified FFT
- **Statistics**: O(n) with caching

## Data Format

### CSV Export
Headers:
```
time,phi,free_energy,control,survival,model_accuracy,belief_magnitude,avg_precision
```

Example:
```
0,2.5000,1.5000,0.6000,0.7000,0.8000,8.0000,1.0000
1,2.6542,1.4821,0.6234,0.7100,0.8050,8.1234,1.0050
...
```

### JSON Export
```json
[
  {
    "phi": 2.5,
    "free_energy": 1.5,
    "control": 0.6,
    "survival": 0.7,
    "model_accuracy": 0.8,
    "time": 0,
    "prediction_error": 0.1,
    "belief_magnitude": 8.0,
    "avg_precision": 1.0
  },
  ...
]
```

## Testing

Comprehensive test suite covers:

- ✓ State snapshot creation and validation
- ✓ Temporal statistics computation
- ✓ Branching ratio detection
- ✓ Spectral analysis
- ✓ Hurst exponent estimation
- ✓ Emergence indicator calculation
- ✓ Data export (CSV/JSON)
- ✓ Long-horizon trajectory analysis
- ✓ Ensemble statistics
- ✓ Criticality detection

## Future Extensions

1. **Multi-scale Analysis**: Wavelets for time-frequency decomposition
2. **Stability Analysis**: Lyapunov spectra
3. **Bifurcation Detection**: Parameter-space analysis
4. **Causal Analysis**: Granger causality, transfer entropy
5. **Geometric Analysis**: Manifold dimension estimation
6. **Quantum Extensions**: Wigner functions, quantum entanglement

## References

1. Friston, K. (2010). "The free-energy principle: A unified brain theory?" Nature Reviews Neuroscience.
2. Tononi, G. (2004). "An information integration theory of consciousness." BMC Neuroscience.
3. Bak, P., Tang, C., Wiesenfeld, K. (1987). "Self-organized criticality." Physical Review Letters.
4. Peters, E. E. (1994). "Fractal Market Analysis: Applying Chaos Theory to Investment and Economics."
5. Hurst, H. E. (1951). "Long-term storage capacity of reservoirs." Transactions of the American Society of Civil Engineers.

## See Also

- `/Volumes/Tengritek/Ashina/HyperPhysics/crates/hyperphysics-agency/src/systems_dynamics.rs` - Implementation
- `/Volumes/Tengritek/Ashina/HyperPhysics/crates/hyperphysics-agency/examples/systems_dynamics_demo.rs` - Demo
- `/Volumes/Tengritek/Ashina/HyperPhysics/crates/hyperphysics-agency/tests/dynamics_integration_test.rs` - Tests
