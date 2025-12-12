# Systems Dynamics Tracker - Quick Start Guide

## TL;DR

```rust
use hyperphysics_agency::systems_dynamics::AgencyDynamics;

// Create tracker
let mut dynamics = AgencyDynamics::new();

// Each simulation step, record agent metrics
dynamics.record_state(
    phi,          // Consciousness (0-10)
    free_energy,  // Surprise (0+)
    control,      // Agency (0-1)
    survival,     // Drive (0-1)
    accuracy,     // Model accuracy (0-1)
    error,        // Prediction error (0+)
    belief_mag,   // State magnitude (0+)
    precision,    // 1/uncertainty (0+)
);

// Analyze
let criticality = dynamics.compute_criticality();
let spectral = dynamics.analyze_spectral();
let stats = dynamics.get_stats()?;

// Export
let csv = dynamics.export_csv();
std::fs::write("dynamics.csv", csv)?;
```

## Key Concepts

### 1. Emergence Indicator
**What**: Measures if the agent has developed agency (0-1)
```
Ranges:
  0.0-0.3: Random behavior
  0.3-0.6: Emerging goals
  0.6-0.9: Strong agency
  0.9-1.0: Perfect consciousness
```

### 2. Branching Ratio (σ)
**What**: Is the system approaching chaos?
```
σ < 1.0:  Sub-critical (dying out)
σ ≈ 1.0:  CRITICAL (sweet spot!)
σ > 1.0:  Super-critical (runaway)
```

### 3. Consciousness (Φ)
**What**: Integrated information from IIT
```
Φ = 0:  No consciousness
Φ = 5:  Moderate awareness
Φ = 10: Strong consciousness
```

### 4. Free Energy (F)
**What**: How surprised the agent is (lower = better)
```
F high:  Unexpected environment
F low:   Well-predicted state
Target:  Minimize F
```

### 5. Control (u)
**What**: How much agency the agent exerts
```
u = Φ × Accuracy × (1 + Survival)
Higher = more directed behavior
```

## Common Usage Patterns

### Pattern 1: Monitor Emergence During Training

```rust
let mut dynamics = AgencyDynamics::new();

for epoch in 0..100 {
    for step in 0..1000 {
        let action = agent.step(&observation);

        // Record every step
        dynamics.record_state(
            agent.phi, agent.free_energy, agent.control,
            agent.survival, agent.accuracy, error, mag, prec
        );
    }

    // Check emergence every epoch
    if let Some(stats) = dynamics.get_stats() {
        println!("Epoch {}: Emergence = {:.2}%",
                 epoch, stats.emergence_indicator() * 100.0);
    }
}
```

### Pattern 2: Detect Criticality

```rust
let criticality = dynamics.compute_criticality();

match criticality.branching_ratio {
    Some(sigma) if (sigma - 1.0).abs() < 0.1 => {
        println!("System at criticality! σ = {:.3}", sigma);
        // Optimal regime for learning
    }
    Some(sigma) if sigma < 1.0 => {
        println!("Sub-critical: increase complexity");
        // System is stable but unchallenged
    }
    Some(sigma) => {
        println!("Super-critical: reduce complexity");
        // System is chaotic, needs stabilization
    }
    None => println!("Need more data"),
}
```

### Pattern 3: Analyze Spectral Properties

```rust
let spectral = dynamics.analyze_spectral();

if let (Some(freq), Some(power)) = (spectral.peak_frequency, spectral.peak_power) {
    println!("Dominant oscillation: {:.2} Hz (power: {:.2})", freq, power);

    if !spectral.harmonics.is_empty() {
        println!("Harmonic frequencies: {:?}", spectral.harmonics);
    }
}
```

### Pattern 4: Export for Visualization

```rust
// Export to CSV for Pandas/Matplotlib
let csv = dynamics.export_csv();
std::fs::write("agent_dynamics.csv", csv)?;

// Python:
// import pandas as pd
// df = pd.read_csv('agent_dynamics.csv')
// df.plot(x='time', y=['phi', 'control'])
```

### Pattern 5: Track Robustness

```rust
if let Some(stats) = dynamics.get_stats() {
    let robustness = stats.robustness_score();

    if robustness > 0.7 {
        println!("System is robust!");
    } else {
        println!("System is fragile, needs stabilization");
    }
}
```

## Metrics at a Glance

| Metric | Range | Good Value | Meaning |
|--------|-------|-----------|---------|
| Φ (Consciousness) | 0-10 | 5-8 | Integration level |
| F (Free Energy) | 0+ | <1.5 | Prediction error |
| Control (u) | 0-1 | 0.6-0.8 | Agency strength |
| Survival (S) | 0-1 | 0.5-0.7 | Threat response |
| Branching Ratio (σ) | 0-2+ | 0.95-1.05 | Criticality |
| Hurst (H) | 0-2 | 0.5-1.0 | Long-range dependence |
| Emergence | 0-1 | 0.7-0.9 | Agency development |
| Robustness | 0-1 | 0.7-0.9 | Stability |

## Data Structures

### StateSnapshot
```rust
// What's recorded each step
time          // u64: step number
phi           // f64: consciousness (0-10)
free_energy   // f64: surprise (0+)
control       // f64: agency (0-1)
survival      // f64: drive (0-1)
model_accuracy // f64: accuracy (0-1)
prediction_error // f64: error (0+)
belief_magnitude // f64: state size (0+)
avg_precision  // f64: 1/uncertainty (0+)
```

### TemporalStats
```rust
// Statistics on a time series
mean           // f64: average
std            // f64: standard deviation
min, max       // f64: extrema
autocorr_lag1  // f64: persistence (-1 to 1)
volatility     // f64: change rate (0+)
skewness       // f64: asymmetry (-3 to 3)
kurtosis       // f64: tail heaviness (-3+)
count          // usize: number of samples
```

### DynamicsStats
```rust
// Aggregated stats for all metrics
phi              // TemporalStats
free_energy      // TemporalStats
control          // TemporalStats
survival         // TemporalStats
samples          // usize: total snapshots

// Methods:
emergence_indicator() -> f64  // 0-1
robustness_score() -> f64     // 0-1
```

## Troubleshooting

### Problem: Emergence indicator is 0
**Solution**: Agent's consciousness (Φ) is too low. Ensure proper learning.

### Problem: Branching ratio is None
**Solution**: Need at least 100 snapshots in history. Keep recording longer.

### Problem: No spectral peaks detected
**Solution**: Signal may be too noisy or random. Increase history length.

### Problem: Export fails with large history
**Solution**: Export before history gets too large. Circular buffer maintains last 10,000 steps.

## Performance Characteristics

```
Recording one state:     0.1 microseconds
Computing criticality:   1-10 milliseconds (n=1000)
Spectral analysis:       5-20 milliseconds (n=1000)
Statistics computation:  2-5 milliseconds
CSV export:              10-50 milliseconds
JSON export:             20-100 milliseconds
```

## Real-World Example

```rust
// Simulate 1000-step training run
let mut agent = CyberneticAgent::new(config);
let mut dynamics = AgencyDynamics::new();

for step in 0..1000 {
    // Create environment
    let env_signal = (step as f64).sin();
    let obs = Observation {
        sensory: Array1::from_elem(32, env_signal),
        timestamp: step,
    };

    // Agent processes
    let action = agent.step(&obs);

    // Record metrics
    dynamics.record_state(
        agent.phi,
        agent.free_energy(),
        agent.control_authority(),
        agent.survival_drive(),
        agent.model_accuracy(),
        0.1,  // simplified error
        5.0,  // simplified belief magnitude
        1.0,  // simplified precision
    );
}

// Analysis
let stats = dynamics.get_stats().unwrap();
println!("Final emergence: {:.1}%", stats.emergence_indicator() * 100.0);

let criticality = dynamics.compute_criticality();
println!("Branching ratio: {:?}", criticality.branching_ratio);

// Export
let csv = dynamics.export_csv();
std::fs::write("results.csv", csv)?;
```

## What Gets Exported?

### CSV Format
```csv
time,phi,free_energy,control,survival,model_accuracy,belief_magnitude,avg_precision
0,2.5000,1.5000,0.6000,0.7000,0.8000,8.0000,1.0000
1,2.6500,1.4800,0.6150,0.7100,0.8050,8.1000,1.0050
2,2.7800,1.4600,0.6300,0.7200,0.8100,8.2000,1.0100
...
```

Use in Python:
```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('dynamics.csv')

# Plot consciousness evolution
df.plot(x='time', y='phi')
plt.ylabel('Integrated Information (Φ)')
plt.show()

# Plot emergence
df['emergence'] = df['phi']/10 * df['control']
df.plot(x='time', y='emergence')
plt.show()
```

## Key Files

- **Implementation**: `crates/hyperphysics-agency/src/systems_dynamics.rs` (~1,100 lines)
- **Tests**: `crates/hyperphysics-agency/tests/dynamics_integration_test.rs` (~600 lines)
- **Demo**: `crates/hyperphysics-agency/examples/systems_dynamics_demo.rs` (~350 lines)
- **Docs**: `crates/hyperphysics-agency/SYSTEMS_DYNAMICS.md` (~450 lines)

## Next Steps

1. ✓ Integrate with CyberneticAgent
2. ✓ Record metrics during training
3. ✓ Analyze criticality metrics
4. ✓ Export for visualization
5. ✓ Monitor emergence evolution
6. → Tune agent parameters based on metrics
7. → Build visualization dashboards
8. → Compare different agent configurations

## Questions?

Refer to:
- **Mathematical details**: `SYSTEMS_DYNAMICS.md`
- **Architecture**: `SYSTEMS_DYNAMICS_ARCHITECTURE.md`
- **Implementation**: `systems_dynamics.rs`
- **Examples**: `systems_dynamics_demo.rs`
- **Tests**: `dynamics_integration_test.rs`
