# CDFA SOC Analyzer

High-performance Self-Organized Criticality (SOC) analyzer with sub-microsecond performance, ported from Python to Rust with SIMD optimization.

## Features

- **Sub-microsecond performance** for core calculations
- **SIMD optimization** using the `wide` crate for 8x parallel processing
- **Sample Entropy** calculation with Numba-style performance
- **Entropy Rate** analysis for predictability measurement
- **Avalanche detection** and size distribution analysis
- **Power law fitting** with maximum likelihood estimation
- **SOC regime classification** (Critical, Near-Critical, Unstable, Stable, Normal)
- **Momentum and divergence** calculations
- **Critical transition detection** with early warning signals
- **Parallel processing** support using Rayon
- **Python bindings** available (optional feature)

## Performance

Benchmark results on modern hardware (example):
- SOC Index (1k points): ~800ns
- Sample Entropy (100 points): ~500ns
- Full Analysis (100 points): ~950ns
- Momentum Calculation (1k points): ~600ns
- Avalanche Detection (1k points): ~1.2µs

## Usage

```rust
use cdfa_soc_analyzer::{SocAnalyzer, SocParameters};

// Create analyzer with default parameters
let analyzer = SocAnalyzer::new();

// Or with custom parameters
let params = SocParameters {
    sample_entropy_m: 2,
    sample_entropy_r: 0.2,
    critical_threshold_complexity: 0.7,
    ..Default::default()
};
let analyzer = SocAnalyzer::with_params(params);

// Analyze price and volume data
let prices = vec![100.0, 101.0, 99.5, 102.0, ...];
let volumes = vec![1000.0, 1100.0, 950.0, 1200.0, ...];

let metrics = analyzer.analyze(&prices, &volumes)?;

// Access results
println!("SOC Index: {:?}", metrics.soc_index);
println!("Current Regime: {:?}", metrics.regime.last());
println!("Fragility: {}", metrics.fragility.last().unwrap());
println!("Momentum: {}", metrics.momentum.last().unwrap());
```

## Features

### Default Features
- `simd`: SIMD optimization (enabled by default)
- `parallel`: Parallel processing using Rayon (enabled by default)

### Optional Features
- `python`: Python bindings via PyO3
- `serialize`: Serde support for serialization
- `benchmark`: Enable benchmarking features

Enable features in `Cargo.toml`:
```toml
[dependencies]
cdfa-soc-analyzer = { version = "0.1", features = ["python", "serialize"] }
```

## Mathematical Background

The SOC analyzer implements several key concepts from Self-Organized Criticality theory:

1. **Sample Entropy**: Measures complexity and predictability of time series
2. **Entropy Rate**: Quantifies information production rate
3. **Avalanche Detection**: Identifies cascading events in the system
4. **Power Law Fitting**: Tests for scale-free behavior characteristic of SOC
5. **Regime Classification**: Determines current system state based on multiple metrics

## Architecture

The crate is organized into modules:
- `entropy`: SIMD-optimized entropy calculations
- `avalanche`: Avalanche detection and statistics
- `power_law`: Power law fitting and distribution comparison
- `regime`: SOC regime classification and transition detection
- `simd_utils`: Common SIMD utility functions

## Building

```bash
# Standard build
cargo build --release

# With Python bindings
cargo build --release --features python

# Run benchmarks
cargo bench
```

## Testing

```bash
cargo test
```

## License

MIT License