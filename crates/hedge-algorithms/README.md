# Hedge Algorithms

A comprehensive Rust crate for advanced hedge algorithms with quantum enhancement and expert weighting systems.

## Features

- **🔬 Quantum Hedge Algorithm**: Adaptive expert weighting with quantum state evolution
- **⚖️ Multiplicative Weights**: Advanced update rules for expert performance tracking
- **📉 Regret Minimization**: Both external and internal regret tracking and bounds
- **📊 8-Factor Model**: Standardized factor integration for risk attribution
- **🎯 Options Hedging**: Greeks-based hedging strategies with Black-Scholes pricing
- **📈 Pair Trading**: Statistical arbitrage with cointegration analysis
- **📊 Volatility Hedging**: Adaptive grid-based hedging with GARCH modeling
- **🐋 Whale Detection**: Momentum-based large order detection and tracking
- **⚡ High Performance**: SIMD acceleration and parallel processing
- **🐍 Python Bindings**: Easy integration with Python via PyO3

## Quick Start

### Rust Usage

```rust
use hedge_algorithms::{HedgeAlgorithms, HedgeConfig, MarketData};

// Create configuration
let config = HedgeConfig::default();
let mut hedge = HedgeAlgorithms::new(config)?;

// Update with market data
let market_data = MarketData::new(
    "BTCUSD".to_string(),
    chrono::Utc::now(),
    [100.0, 105.0, 95.0, 102.0, 1000.0] // [open, high, low, close, volume]
);

hedge.update_market_data(&market_data)?;

// Get hedge recommendation
let recommendation = hedge.get_hedge_recommendation()?;
println!("Position size: {}", recommendation.position_size);
println!("Hedge ratio: {}", recommendation.hedge_ratio);
println!("Confidence: {:.2}%", recommendation.confidence * 100.0);
```

### Python Usage

```python
import hedge_algorithms as ha

# Create configuration
config = ha.HedgeConfig(
    learning_rate=0.01,
    weight_decay=0.999,
    confidence_threshold=0.7
)

# Create hedge algorithm
hedge = ha.HedgeAlgorithms(config)

# Create market data
market_data = ha.MarketData(
    symbol="BTCUSD",
    timestamp=1640995200.0,  # Unix timestamp
    open=100.0,
    high=105.0,
    low=95.0,
    close=102.0,
    volume=1000.0
)

# Update and get recommendation
hedge.update_market_data(market_data)
recommendation = hedge.get_hedge_recommendation()

print(f"Position size: {recommendation.position_size}")
print(f"Hedge ratio: {recommendation.hedge_ratio}")
print(f"Confidence: {recommendation.confidence:.2%}")
```

### CLI Usage

```bash
# Install the CLI
cargo install --path .

# Run hedge algorithm on market data
hedge_cli run --data market_data.csv --output recommendations.json

# Calculate Black-Scholes option price
hedge_cli black-scholes --spot 100 --strike 100 --time 1 --volatility 0.2

# Test pairs trading
hedge_cli pairs-test --asset-a prices_a.txt --asset-b prices_b.txt

# Detect whale activity
hedge_cli whale-detect --data market_data.csv --output whale_activity.json

# Run performance benchmark
hedge_cli benchmark --iterations 10000 --experts 20

# Generate sample configuration
hedge_cli generate-config --output config.json

# Validate configuration
hedge_cli validate-config --config config.json
```

## Architecture

### Core Components

1. **QuantumHedgeAlgorithm**: Main algorithm with quantum state management
2. **ExpertSystem**: Expert registration, evaluation, and ensemble methods
3. **StandardFactorModel**: 8-factor risk model with attribution
4. **RegretMinimizer**: External and internal regret tracking
5. **OptionsHedger**: Black-Scholes pricing and Greeks calculation
6. **PairsTrader**: Statistical arbitrage with cointegration
7. **VolatilityHedger**: Grid-based hedging with volatility forecasting
8. **WhaleDetector**: Large order detection and momentum analysis

### Quantum Enhancement

The quantum hedge algorithm uses quantum state evolution to enhance expert weighting:

- **Quantum Superposition**: Experts exist in superposition states
- **Quantum Entanglement**: Correlated expert relationships
- **Quantum Measurement**: Probabilistic expert selection
- **Decoherence**: Natural decay of quantum coherence over time

### Expert System

The expert system supports multiple expert types:

- **Trend Following**: Momentum-based experts
- **Mean Reversion**: Contrarian experts
- **Volatility Trading**: Volatility-based experts
- **Custom Experts**: User-defined expert implementations

### 8-Factor Model

The standardized factor model includes:

1. **Volatility Factor**: Market volatility exposure
2. **Momentum Factor**: Price momentum exposure
3. **Mean Reversion Factor**: Contrarian exposure
4. **Liquidity Factor**: Liquidity risk exposure
5. **Quality Factor**: Earnings quality exposure
6. **Growth Factor**: Growth characteristics
7. **Size Factor**: Market capitalization effects
8. **Profitability Factor**: Profitability measures

## Configuration

### Basic Configuration

```rust
let config = HedgeConfig {
    learning_rate: 0.01,
    weight_decay: 0.999,
    min_weight: 0.001,
    max_weight: 0.5,
    max_history: 10000,
    confidence_threshold: 0.7,
    risk_tolerance: 0.05,
    // ... other fields
};
```

### Quantum Configuration

```rust
let quantum_config = QuantumConfig {
    enabled: true,
    decoherence_rate: 0.01,
    entanglement_factor: 0.1,
    quantum_iterations: 100,
    noise_level: 0.001,
};
```

### Expert Configuration

```rust
let expert_config = ExpertConfig {
    max_experts: 50,
    evaluation_window: 1000,
    pruning_threshold: 0.1,
    diversity_requirement: 0.3,
    consistency_weight: 0.2,
};
```

## Performance

### Benchmarks

Run benchmarks with:

```bash
cargo bench
```

Typical performance on modern hardware:

- **Hedge Algorithm Update**: ~10 μs per update
- **Quantum State Evolution**: ~50 μs per update
- **Options Greeks Calculation**: ~5 μs per calculation
- **Factor Model Update**: ~20 μs per update
- **Regret Minimization**: ~2 μs per update

### Optimization Features

- **SIMD Acceleration**: Vectorized mathematical operations
- **Parallel Processing**: Multi-threaded expert evaluation
- **Memory Optimization**: Efficient data structures and caching
- **Hardware Acceleration**: GPU support for neural networks (optional)

## Error Handling

The crate provides comprehensive error handling:

```rust
use hedge_algorithms::{HedgeError, HedgeResult};

fn example() -> HedgeResult<()> {
    let config = HedgeConfig::default();
    let hedge = HedgeAlgorithms::new(config)?;
    
    // Errors are automatically propagated
    let recommendation = hedge.get_hedge_recommendation()?;
    
    Ok(())
}
```

Error categories:
- **Configuration**: Invalid parameters
- **Mathematical**: Computation errors
- **Data**: Invalid or insufficient data
- **Expert**: Expert system errors
- **Quantum**: Quantum algorithm errors

## Testing

### Unit Tests

```bash
cargo test
```

### Integration Tests

```bash
cargo test --test integration_tests
```

### Property-Based Tests

```bash
cargo test --features proptest
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

### Code Style

- Use `rustfmt` for formatting
- Follow Rust naming conventions
- Add documentation for public APIs
- Include examples in documentation

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Quantum algorithms based on research in quantum machine learning
- Options pricing models from Black-Scholes-Merton framework
- Regret minimization algorithms from online learning theory
- Factor models from quantitative finance literature

## Support

For support, please:

1. Check the documentation
2. Search existing issues
3. Create a new issue with detailed description
4. Provide minimal reproduction case

## Roadmap

### Version 0.2.0
- [ ] GPU acceleration for neural networks
- [ ] Distributed processing support
- [ ] Real-time streaming data integration
- [ ] Advanced volatility models (Heston, SABR)

### Version 0.3.0
- [ ] Machine learning integration
- [ ] Alternative data sources
- [ ] Risk management enhancements
- [ ] Performance dashboard

### Version 1.0.0
- [ ] Production-ready stability
- [ ] Comprehensive documentation
- [ ] Enterprise features
- [ ] Regulatory compliance tools

## Examples

See the `examples/` directory for complete usage examples:

- `basic_hedge.rs`: Basic hedge algorithm usage
- `quantum_enhancement.rs`: Quantum-enhanced expert weighting
- `options_hedging.rs`: Options hedging with Greeks
- `pairs_trading.rs`: Statistical arbitrage
- `whale_detection.rs`: Large order detection
- `performance_analysis.rs`: Performance metrics and attribution

## API Reference

Full API documentation is available at: https://docs.rs/hedge_algorithms

## Changelog

### Version 0.1.0
- Initial release
- Core hedge algorithm implementation
- Quantum enhancement features
- Expert system framework
- 8-factor model integration
- Options hedging capabilities
- Pair trading functionality
- Whale detection system
- Python bindings
- CLI interface
- Comprehensive test suite