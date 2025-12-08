# Neural Testing Framework

A comprehensive zero-mock neural network testing framework for Nautilus Trader that implements real neural network testing with actual data, hardware acceleration, and live trading simulation.

## 🚀 Features

### Core Testing Capabilities
- **NHITS Neural Network Tests**: Comprehensive testing of Neural Hierarchical Interpolation for Time Series
- **CDFA Algorithm Tests**: Consensus Diversity Fusion Algorithm validation with real market scenarios
- **GPU/CUDA Acceleration**: Hardware acceleration testing with real GPU kernels
- **Real-Time Trading Simulation**: Live trading simulation with neural predictions
- **Quantum Neural Components**: Quantum-enhanced neural network testing
- **Zero-Mock Testing**: All tests use real implementations, no mocking

### Performance Requirements
- **Sub-100μs Inference**: Ultra-low latency neural inference testing
- **Real Market Data**: Authentic market data generation with multiple regimes
- **Memory Efficiency**: Comprehensive memory usage and efficiency testing
- **Hardware Utilization**: CPU, GPU, and memory bandwidth optimization

## 📦 Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
neural-testing-framework = { path = "../neural-testing-framework" }

# Optional features
[features]
gpu = ["neural-testing-framework/gpu"]
cuda = ["neural-testing-framework/cuda"]
quantum = ["neural-testing-framework/quantum"]
full = ["neural-testing-framework/full"]
```

## 🏃‍♂️ Quick Start

### Basic Usage

```rust
use neural_testing_framework::{NeuralTestRunner, NeuralTestConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create default configuration
    let config = NeuralTestConfig::default();
    
    // Initialize test runner
    let mut runner = NeuralTestRunner::new(config);
    
    // Run all tests
    runner.run_all_tests().await?;
    
    Ok(())
}
```

### Command Line Interface

```bash
# Run all tests
cargo run --bin neural-test-runner all

# Run specific test suites
cargo run --bin neural-test-runner nhits
cargo run --bin neural-test-runner cdfa --sources 10
cargo run --bin neural-test-runner gpu --memory-stress
cargo run --bin neural-test-runner simulation --duration 600

# Run with GPU support
cargo run --features gpu --bin neural-test-runner gpu

# Quick tests only
cargo run --bin neural-test-runner nhits --quick

# Verbose output
cargo run --bin neural-test-runner all --verbose
```

## 🧠 NHITS Neural Network Tests

The NHITS test suite provides comprehensive validation of Neural Hierarchical Interpolation for Time Series models:

```rust
use neural_testing_framework::nhits_tests::{NHITSTestSuite, NHITSConfig};

let config = NHITSConfig {
    input_size: 24,
    output_size: 12,
    num_stacks: 4,
    stack_types: vec![
        StackType::Trend,
        StackType::Seasonality(24),
        StackType::Seasonality(168),
        StackType::Residual,
    ],
    // ... other configuration
};

let mut test_suite = NHITSTestSuite::new(config);
let results = test_suite.run_comprehensive_tests().await?;
```

### NHITS Test Coverage

1. **Basic Functionality**: Model creation, training, and inference
2. **Multi-Scale Decomposition**: Hierarchical pattern decomposition validation
3. **Real-Time Inference**: Sub-100μs latency requirements
4. **Memory Efficiency**: Memory usage optimization under different batch sizes
5. **Training Stability**: Gradient flow and convergence monitoring
6. **Interpretability**: Component analysis and decomposition quality
7. **Edge Cases**: Robustness testing with extreme inputs
8. **Scalability**: Performance across different input sizes
9. **Multi-Asset**: Cross-asset forecasting capabilities
10. **Regime Adaptation**: Performance across different market conditions

## 🔬 CDFA Algorithm Tests

The CDFA (Consensus Diversity Fusion Algorithm) test suite validates fusion methods:

```rust
use neural_testing_framework::cdfa_tests::{CDFATestSuite, CDFAConfig};

let config = CDFAConfig {
    num_sources: 5,
    diversity_threshold: 0.5,
    adaptive_fusion_enabled: true,
    real_time_mode: true,
    // ... other configuration
};

let mut test_suite = CDFATestSuite::new(config);
let results = test_suite.run_comprehensive_tests().await?;
```

### CDFA Test Coverage

1. **Fusion Methods Comparison**: Average, weighted, Borda count, adaptive
2. **Diversity Metrics Validation**: Kendall's tau, Spearman, Jensen-Shannon
3. **Adaptive Fusion Performance**: Dynamic method selection
4. **Real-Time Latency**: Sub-microsecond fusion requirements
5. **Market Regime Adaptation**: Performance across different market conditions
6. **Scalability**: Performance with varying source counts
7. **Noise Robustness**: Handling of noisy and corrupted sources
8. **Multi-Asset Fusion**: Cross-asset consensus building
9. **Streaming Data**: Continuous fusion with live data
10. **Consensus Stability**: Stability of consensus under volatility

## 🚀 GPU/CUDA Tests

Comprehensive GPU acceleration testing with real hardware:

```rust
use neural_testing_framework::gpu_tests::{GPUTestSuite, GPUTestConfig};

let config = GPUTestConfig {
    test_cuda: true,
    test_memory_transfers: true,
    test_concurrent_streams: true,
    memory_stress_levels: vec![512, 1024, 2048, 4096],
    // ... other configuration
};

let mut test_suite = GPUTestSuite::new(config)?;
let results = test_suite.run_comprehensive_tests().await?;
```

### GPU Test Coverage

1. **Device Detection**: CUDA runtime and device initialization
2. **Memory Operations**: Host-device transfers and bandwidth testing
3. **CUDA Kernels**: Real kernel execution and validation
4. **Neural Forward Pass**: GPU vs CPU performance comparison
5. **Batch Processing**: Optimization for different batch sizes
6. **Memory Stress**: Large dataset handling and memory management
7. **Concurrent Streams**: Multi-stream execution and synchronization
8. **Mixed Precision**: FP16/FP32 performance and accuracy
9. **Multi-GPU**: Coordination across multiple devices

## 📈 Real-Time Trading Simulation

Live trading simulation with neural predictions:

```rust
use neural_testing_framework::real_time_simulation::{
    RealTimeSimulationSuite, SimulationConfig
};

let config = SimulationConfig {
    simulation_duration_s: 300,
    update_frequency_ms: 100,
    num_strategies: 5,
    // ... risk configuration
};

let mut test_suite = RealTimeSimulationSuite::new(config);
let results = test_suite.run_comprehensive_tests().await?;
```

### Simulation Test Coverage

1. **Prediction Latency**: Real-time neural prediction performance
2. **Market Regime Adaptation**: Strategy adaptation across regimes
3. **High-Frequency Trading**: Ultra-low latency execution
4. **Risk Management**: Real-time risk monitoring and controls
5. **Multi-Strategy Coordination**: Portfolio-level strategy management
6. **System Stability**: Continuous operation under load
7. **Emergency Stops**: Crisis response mechanisms
8. **Memory Usage**: Long-running operation memory efficiency
9. **Prediction Accuracy**: Accuracy degradation over time
10. **Network Latency**: Impact of network delays on performance

## 🔧 Configuration

### Test Data Configuration

```rust
use neural_testing_framework::{TestDataConfig, MarketRegime};

let data_config = TestDataConfig {
    num_assets: 10,
    sequence_length: 24,
    num_features: 5,
    forecast_horizon: 12,
    market_regimes: vec![
        MarketRegime::Bull,
        MarketRegime::Bear,
        MarketRegime::HighVolatility,
        MarketRegime::Crisis,
    ],
    noise_levels: vec![0.01, 0.02, 0.05],
};
```

### Performance Thresholds

```rust
use neural_testing_framework::PerformanceThresholds;

let thresholds = PerformanceThresholds {
    max_inference_time_us: 100.0,    // 100 microseconds
    max_memory_usage_mb: 1024.0,     // 1 GB
    min_accuracy: 0.8,               // 80% accuracy
    max_training_time_s: 300.0,      // 5 minutes
    min_gpu_utilization: 0.7,        // 70% GPU utilization
};
```

### Hardware Test Configuration

```rust
use neural_testing_framework::HardwareTestConfig;

let hardware_config = HardwareTestConfig {
    test_cpu: true,
    test_gpu: true,
    test_quantum: false,
    test_distributed: false,
    memory_stress_levels: vec![512, 1024, 2048, 4096], // MB
};
```

## 📊 Test Results and Reporting

### Test Result Structure

```rust
pub struct NeuralTestResults {
    pub test_name: String,
    pub success: bool,
    pub metrics: PerformanceMetrics,
    pub errors: Vec<String>,
    pub execution_time: Duration,
    pub memory_stats: MemoryStats,
    pub hardware_utilization: HardwareUtilization,
}
```

### Performance Metrics

```rust
pub struct PerformanceMetrics {
    pub inference_latency_us: f64,
    pub training_time_s: f64,
    pub accuracy_metrics: AccuracyMetrics,
    pub throughput_pps: f64,
    pub memory_efficiency: f64,
}
```

### Accuracy Metrics

```rust
pub struct AccuracyMetrics {
    pub mae: f64,                    // Mean Absolute Error
    pub rmse: f64,                   // Root Mean Square Error
    pub mape: f64,                   // Mean Absolute Percentage Error
    pub r2: f64,                     // R-squared coefficient
    pub sharpe_ratio: Option<f64>,   // Sharpe ratio (trading)
    pub max_drawdown: Option<f64>,   // Maximum drawdown
    pub hit_rate: Option<f64>,       // Direction prediction accuracy
}
```

## 🎯 Test Execution Examples

### Running NHITS Tests

```bash
# Basic NHITS tests
cargo run --bin neural-test-runner nhits

# Quick NHITS tests (reduced complexity)
cargo run --bin neural-test-runner nhits --quick

# NHITS with verbose output
cargo run --bin neural-test-runner nhits --verbose
```

### Running CDFA Tests

```bash
# CDFA with 5 sources
cargo run --bin neural-test-runner cdfa

# CDFA with 10 sources
cargo run --bin neural-test-runner cdfa --sources 10
```

### Running GPU Tests

```bash
# Basic GPU tests
cargo run --features gpu --bin neural-test-runner gpu

# GPU with memory stress testing
cargo run --features gpu --bin neural-test-runner gpu --memory-stress

# CUDA-specific tests
cargo run --features cuda --bin neural-test-runner gpu
```

### Running Trading Simulation

```bash
# 5-minute simulation
cargo run --bin neural-test-runner simulation

# 10-minute simulation
cargo run --bin neural-test-runner simulation --duration 600

# High-frequency simulation
cargo run --bin neural-test-runner simulation --duration 60
```

## 📈 Performance Benchmarks

### Benchmark Execution

```bash
# Run performance benchmarks
cargo run --bin neural-test-runner benchmark

# Extended benchmarks with more iterations
cargo run --bin neural-test-runner benchmark --iterations 1000

# Quick benchmarks
cargo run --bin neural-test-runner benchmark --iterations 10
```

### Expected Performance

| Component | Target Latency | Expected Accuracy | Memory Usage |
|-----------|----------------|-------------------|--------------|
| NHITS Inference | < 100μs | R² > 0.8 | < 512MB |
| CDFA Fusion | < 1μs | Accuracy > 75% | < 256MB |
| GPU Transfer | < 10ms | 100% integrity | Variable |
| Trading Prediction | < 50μs | Hit rate > 65% | < 128MB |

## 🔗 Integration with Nautilus Trader

### Using with Nautilus Strategies

```rust
// Example integration in a Nautilus strategy
use nautilus_trader::model::prelude::*;
use neural_testing_framework::{NeuralTestRunner, NeuralTestConfig};

pub struct NeuralStrategy {
    test_runner: NeuralTestRunner,
}

impl NeuralStrategy {
    pub fn new() -> Self {
        let config = NeuralTestConfig::default();
        let test_runner = NeuralTestRunner::new(config);
        
        Self { test_runner }
    }
    
    pub async fn validate_neural_components(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        // Run neural tests before strategy execution
        self.test_runner.run_all_tests().await?;
        Ok(())
    }
}
```

## 🧪 Custom Test Development

### Creating Custom Tests

```rust
use neural_testing_framework::{NeuralTestResults, PerformanceMetrics};

pub struct CustomTestSuite;

impl CustomTestSuite {
    pub async fn run_custom_test(&mut self) -> Result<NeuralTestResults, Box<dyn std::error::Error>> {
        let start_time = std::time::Instant::now();
        
        // Implement your test logic here
        let test_passed = self.execute_custom_logic().await?;
        
        Ok(NeuralTestResults {
            test_name: "custom_neural_test".to_string(),
            success: test_passed,
            metrics: PerformanceMetrics::default(),
            errors: Vec::new(),
            execution_time: start_time.elapsed(),
            memory_stats: Default::default(),
            hardware_utilization: Default::default(),
        })
    }
    
    async fn execute_custom_logic(&self) -> Result<bool, Box<dyn std::error::Error>> {
        // Your custom test implementation
        Ok(true)
    }
}
```

## 📝 Logging and Debugging

### Enable Detailed Logging

```bash
# Debug level logging
RUST_LOG=debug cargo run --bin neural-test-runner all

# Trace level logging (very verbose)
RUST_LOG=trace cargo run --bin neural-test-runner all

# Component-specific logging
RUST_LOG=neural_testing_framework::nhits_tests=debug cargo run --bin neural-test-runner nhits
```

### Test Configuration File

```json
{
  "data_config": {
    "num_assets": 10,
    "sequence_length": 24,
    "num_features": 5,
    "forecast_horizon": 12,
    "market_regimes": ["Bull", "Bear", "HighVolatility"],
    "noise_levels": [0.01, 0.02, 0.05]
  },
  "performance_thresholds": {
    "max_inference_time_us": 100.0,
    "max_memory_usage_mb": 1024.0,
    "min_accuracy": 0.8,
    "max_training_time_s": 300.0,
    "min_gpu_utilization": 0.7
  },
  "hardware_config": {
    "test_cpu": true,
    "test_gpu": true,
    "test_quantum": false,
    "test_distributed": false,
    "memory_stress_levels": [512, 1024, 2048]
  }
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add comprehensive tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Neural Hierarchical Interpolation for Time Series (NHITS) research
- CUDA and GPU acceleration libraries
- Nautilus Trader core team
- Contributors to the neural forecasting community