# CDFA Panarchy Analyzer

High-performance Rust implementation of Panarchy adaptive cycle analysis for CDFA (Composable Distributed Financial Analytics), achieving sub-microsecond performance targets.

## Overview

The Panarchy Analyzer implements the four-phase adaptive cycle model:

1. **Growth (r)** - Exploitation of opportunities
2. **Conservation (K)** - Stability and efficiency  
3. **Release (Ω)** - Creative destruction
4. **Reorganization (α)** - Innovation and renewal

## Features

- **Sub-microsecond performance** for critical operations
- **SIMD optimization** using the `wide` crate for vectorized computations
- **PCR Analysis** - Potential, Connectedness, and Resilience components
- **Phase identification** with hysteresis to prevent oscillation
- **ADX calculation** for trend strength analysis
- **Batch processing** capabilities for multiple time series
- **Compatible** with CDFA server integration

## Performance Targets

- PCR Calculation: < 300ns
- Phase Classification: < 200ns  
- Regime Score: < 150ns
- Full Analysis: < 800ns

## Usage

```rust
use cdfa_panarchy_analyzer::{PanarchyAnalyzer, PanarchyParameters};

// Create analyzer with default parameters
let mut analyzer = PanarchyAnalyzer::new();

// Prepare market data
let prices = vec![100.0, 101.0, 99.5, 102.0, 103.0];
let volumes = vec![1000.0, 1100.0, 950.0, 1200.0, 1150.0];

// Perform analysis
let result = analyzer.analyze(&prices, &volumes)?;

println!("Current phase: {}", result.phase);
println!("Signal: {:.2}", result.signal);
println!("Confidence: {:.2}%", result.confidence * 100.0);
```

## PCR Components

- **Potential (P)**: Capacity for growth/change, normalized price position
- **Connectedness (C)**: Internal connections/rigidity via autocorrelation
- **Resilience (R)**: Ability to withstand disturbance, inverse volatility

## Custom Parameters

```rust
use cdfa_panarchy_analyzer::PanarchyParameters;

let mut params = PanarchyParameters::default();
params.adx_period = 20;
params.autocorr_lag = 2;
params.hysteresis_min_score_threshold = 0.4;

let analyzer = PanarchyAnalyzer::with_params(params);
```

## Batch Processing

```rust
use cdfa_panarchy_analyzer::BatchPanarchyAnalyzer;

let mut batch = BatchPanarchyAnalyzer::new(10);
let results = batch.analyze_batch(&price_series, &volume_series);
```

## Building and Testing

```bash
# Build with optimizations
cargo build --release

# Run tests
cargo test

# Run benchmarks
cargo bench

# Run example
cargo run --release --example panarchy_analysis
```

## Features

- `default`: Standard features with SIMD and parallel processing
- `gpu`: GPU acceleration support (optional)
- `jemalloc`: Use jemalloc for better memory performance

## Integration with CDFA

The analyzer implements the standard CDFA analyzer interface:

```rust
pub fn analyze(&mut self, prices: &[f64], volumes: &[f64]) -> Result<PanarchyResult>
```

This makes it compatible with the CDFA server infrastructure for real-time analysis.

## License

MIT OR Apache-2.0