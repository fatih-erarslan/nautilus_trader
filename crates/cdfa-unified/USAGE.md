# CDFA Unified Usage Guide

Comprehensive guide for using CDFA Unified across different scenarios and applications.

## 📋 Table of Contents

- [Getting Started](#getting-started)
- [Core Functionality](#core-functionality)
- [Advanced Features](#advanced-features)
- [Performance Optimization](#performance-optimization)
- [Integration Examples](#integration-examples)
- [Best Practices](#best-practices)
- [API Reference](#api-reference)

## 🚀 Getting Started

### Basic Setup

```rust
use cdfa_unified::prelude::*;
use ndarray::array;

fn main() -> Result<()> {
    // Initialize logging
    env_logger::init();
    
    // Create sample data
    let data = array![
        [0.8, 0.6, 0.9, 0.3, 0.7],
        [0.7, 0.8, 0.6, 0.4, 0.9],
        [0.9, 0.5, 0.8, 0.5, 0.6]
    ];
    
    // Calculate basic metrics
    let correlation = pearson_correlation(&data.row(0), &data.row(1))?;
    println!("Pearson correlation: {:.4}", correlation);
    
    Ok(())
}
```

### Configuration

```rust
use cdfa_unified::config::*;

// Load configuration from file
let config = CdfaConfig::load_from_file("config.toml")?;

// Or create programmatically
let config = CdfaConfig::builder()
    .threads(8)
    .simd_level(SIMDLevel::AVX2)
    .enable_gpu(true)
    .cache_size(1024 * 1024 * 1024) // 1GB
    .build()?;

// Apply configuration globally
config.apply_global()?;
```

## 🔧 Core Functionality

### Diversity Metrics

```rust
use cdfa_unified::core::diversity::*;
use ndarray::Array1;

fn diversity_analysis() -> Result<()> {
    let series1: Array1<f64> = array![1.0, 2.0, 3.0, 4.0, 5.0];
    let series2: Array1<f64> = array![1.1, 2.1, 2.9, 4.1, 4.9];
    
    // Correlation metrics
    let pearson = pearson_correlation(&series1.view(), &series2.view())?;
    let spearman = spearman_correlation(&series1.view(), &series2.view())?;
    let kendall = kendall_tau(&series1.view(), &series2.view())?;
    
    println!("Pearson: {:.4}", pearson);
    println!("Spearman: {:.4}", spearman);
    println!("Kendall: {:.4}", kendall);
    
    // Distance metrics
    let js_divergence = jensen_shannon_divergence(&series1.view(), &series2.view())?;
    let dtw_distance = dynamic_time_warping(&series1.view(), &series2.view())?;
    
    println!("JS Divergence: {:.4}", js_divergence);
    println!("DTW Distance: {:.4}", dtw_distance);
    
    Ok(())
}
```

### Fusion Methods

```rust
use cdfa_unified::core::fusion::*;
use ndarray::Array2;

fn fusion_example() -> Result<()> {
    let data: Array2<f64> = array![
        [0.8, 0.6, 0.9, 0.3],
        [0.7, 0.8, 0.6, 0.4],
        [0.9, 0.5, 0.8, 0.5]
    ];
    
    // Score fusion
    let score_fusion = ScoreFusion::new();
    let fused_scores = score_fusion.fuse(&data.view(), FusionMethod::Average)?;
    
    // Rank fusion
    let rank_fusion = RankFusion::new();
    let fused_ranks = rank_fusion.fuse(&data.view(), FusionMethod::BordaCount)?;
    
    // Adaptive fusion
    let adaptive_fusion = AdaptiveScoreFusion::new()?;
    let adaptive_result = adaptive_fusion.fuse(&data.view())?;
    
    println!("Score fusion: {:?}", fused_scores);
    println!("Rank fusion: {:?}", fused_ranks);
    println!("Adaptive fusion: {:?}", adaptive_result);
    
    Ok(())
}
```

### Signal Processing

```rust
use cdfa_unified::algorithms::*;
use ndarray::Array1;

fn signal_processing() -> Result<()> {
    let signal: Array1<f64> = Array1::linspace(0.0, 10.0, 1000)
        .mapv(|x| (x * 2.0 * std::f64::consts::PI).sin() + 0.1 * (x * 20.0 * std::f64::consts::PI).sin());
    
    // Wavelet transform
    let (approx, detail) = WaveletTransform::dwt_haar(&signal.view())?;
    println!("Wavelet decomposition: {} approx, {} detail coefficients", approx.len(), detail.len());
    
    // Entropy measures
    let sample_entropy = SampleEntropy::calculate(&signal.view(), 2, 0.2)?;
    let permutation_entropy = PermutationEntropy::calculate(&signal.view(), 3)?;
    
    println!("Sample entropy: {:.4}", sample_entropy);
    println!("Permutation entropy: {:.4}", permutation_entropy);
    
    // Volatility analysis
    let vol_clustering = VolatilityClustering::analyze(&signal.view())?;
    println!("Volatility clusters: {}", vol_clustering.clusters.len());
    
    Ok(())
}
```

## 🔍 Advanced Features

### Pattern Detection

```rust
use cdfa_unified::detectors::*;

fn pattern_detection() -> Result<()> {
    // Generate sample price data
    let prices: Array1<f64> = Array1::linspace(100.0, 150.0, 1000)
        .mapv(|x| x + 5.0 * (x / 10.0).sin());
    
    // Fibonacci pattern detection
    let fib_detector = FibonacciPatternDetector::builder()
        .tolerance(0.02)
        .min_pattern_length(50)
        .build()?;
    
    let fib_patterns = fib_detector.detect_patterns(&prices.view())?;
    println!("Found {} Fibonacci patterns", fib_patterns.len());
    
    // Black swan detection
    let swan_detector = BlackSwanDetector::builder()
        .threshold(3.0)
        .window_size(100)
        .build()?;
    
    let extreme_events = swan_detector.analyze(&prices.view())?;
    println!("Detected {} potential black swan events", extreme_events.len());
    
    // Antifragility analysis
    let antifragility_analyzer = AntifragilityAnalyzer::new();
    let metrics = antifragility_analyzer.analyze(&prices.view())?;
    
    println!("Antifragility score: {:.4}", metrics.antifragility_score);
    println!("Stress benefit ratio: {:.4}", metrics.stress_benefit_ratio);
    
    Ok(())
}
```

### Machine Learning Integration

```rust
use cdfa_unified::ml::*;

fn ml_integration() -> Result<()> {
    // Prepare training data
    let features: Array2<f64> = Array2::random((1000, 10), rand::distributions::Uniform::new(-1.0, 1.0));
    let targets: Array1<f64> = features.column(0).mapv(|x| if x > 0.0 { 1.0 } else { 0.0 });
    
    // Neural pattern detector
    let mut neural_detector = NeuralPatternDetector::builder()
        .hidden_layers(vec![20, 10])
        .learning_rate(0.001)
        .epochs(100)
        .build()?;
    
    neural_detector.train(&features.view(), &targets.view())?;
    let predictions = neural_detector.predict(&features.view())?;
    
    let accuracy = predictions.iter()
        .zip(targets.iter())
        .filter(|(pred, target)| (pred.round() - target).abs() < 0.5)
        .count() as f64 / targets.len() as f64;
    
    println!("Neural detector accuracy: {:.4}", accuracy);
    
    // Classical ML ensemble
    let mut ensemble = EnsembleDetector::new()
        .add_detector(Box::new(SvmDetector::new()))
        .add_detector(Box::new(RandomForestDetector::new()))
        .add_detector(Box::new(GradientBoostingDetector::new()));
    
    ensemble.train(&features.view(), &targets.view())?;
    let ensemble_predictions = ensemble.predict(&features.view())?;
    
    println!("Ensemble predictions shape: {:?}", ensemble_predictions.shape());
    
    Ok(())
}
```

### GPU Acceleration

```rust
use cdfa_unified::gpu::*;

#[cfg(feature = "gpu")]
fn gpu_acceleration() -> Result<()> {
    // Initialize GPU context
    let gpu_context = GpuContext::new()?;
    println!("GPU: {}", gpu_context.device_name());
    
    // GPU-accelerated correlation matrix
    let data: Array2<f32> = Array2::random((1000, 100), rand::distributions::Uniform::new(-1.0f32, 1.0f32));
    
    let gpu_correlations = gpu_context.correlation_matrix(&data.view())?;
    println!("GPU correlation matrix: {:?}", gpu_correlations.shape());
    
    // GPU Fibonacci detector
    let gpu_fib_detector = GpuFibonacciDetector::new(&gpu_context)?;
    let prices: Array1<f32> = Array1::linspace(100.0, 150.0, 10000);
    
    let gpu_patterns = gpu_fib_detector.detect_realtime(&prices.view())?;
    println!("GPU detected {} patterns", gpu_patterns.len());
    
    Ok(())
}

#[cfg(not(feature = "gpu"))]
fn gpu_acceleration() -> Result<()> {
    println!("GPU features not enabled. Rebuild with --features gpu");
    Ok(())
}
```

## ⚡ Performance Optimization

### SIMD Optimization

```rust
use cdfa_unified::simd::*;

fn simd_optimization() -> Result<()> {
    let data1: Array1<f32> = Array1::linspace(0.0, 1.0, 10000);
    let data2: Array1<f32> = Array1::linspace(0.1, 1.1, 10000);
    
    // Check SIMD availability
    println!("SIMD features available: {:?}", SIMDFeatures::detect());
    
    // SIMD-accelerated correlation
    let simd_correlation = simd_pearson_correlation(&data1.view(), &data2.view())?;
    println!("SIMD correlation: {:.6}", simd_correlation);
    
    // SIMD dot product
    let simd_dot_product = simd_dot_product(&data1.view(), &data2.view())?;
    println!("SIMD dot product: {:.6}", simd_dot_product);
    
    // Vector operations
    let mut result = Array1::<f32>::zeros(data1.len());
    simd_vector_add(&data1.view(), &data2.view(), &mut result.view_mut())?;
    
    println!("SIMD vector addition completed for {} elements", result.len());
    
    Ok(())
}
```

### Parallel Processing

```rust
use cdfa_unified::parallel::*;
use rayon::prelude::*;

fn parallel_processing() -> Result<()> {
    // Configure thread pool
    let thread_pool = ThreadPoolBuilder::new()
        .num_threads(8)
        .build()?;
    
    // Parallel correlation matrix
    let data: Array2<f64> = Array2::random((500, 1000), rand::distributions::Standard);
    
    let parallel_analyzer = ParallelCdfaAnalyzer::new(8)?;
    let correlation_matrix = parallel_analyzer.correlation_matrix(&data.view())?;
    
    println!("Parallel correlation matrix: {:?}", correlation_matrix.shape());
    
    // Parallel batch processing
    let datasets: Vec<Array2<f64>> = (0..10)
        .map(|_| Array2::random((100, 50), rand::distributions::Standard))
        .collect();
    
    let batch_results = parallel_analyzer.analyze_batch(&datasets)?;
    println!("Processed {} datasets in parallel", batch_results.len());
    
    // Custom parallel operation
    let results: Vec<f64> = (0..1000)
        .into_par_iter()
        .map(|i| {
            let x = i as f64 / 1000.0;
            x.sin() * x.cos()
        })
        .collect();
    
    println!("Parallel computation completed: {} results", results.len());
    
    Ok(())
}
```

### Memory Optimization

```rust
use cdfa_unified::config::*;

fn memory_optimization() -> Result<()> {
    // Configure memory-efficient settings
    let config = CdfaConfig::builder()
        .memory_pool_size(512 * 1024 * 1024) // 512MB
        .enable_memory_mapping(true)
        .compression_level(6)
        .cache_policy(CachePolicy::LRU)
        .build()?;
    
    config.apply_global()?;
    
    // Use memory-mapped arrays for large datasets
    let large_data = MemoryMappedArray::<f64>::create("large_dataset.dat", (10000, 1000))?;
    
    // Process in chunks to avoid memory pressure
    let chunk_size = 1000;
    for i in (0..large_data.nrows()).step_by(chunk_size) {
        let end = std::cmp::min(i + chunk_size, large_data.nrows());
        let chunk = large_data.slice(s![i..end, ..]);
        
        // Process chunk
        let chunk_mean = chunk.mean().unwrap();
        println!("Chunk {} mean: {:.4}", i / chunk_size, chunk_mean);
    }
    
    Ok(())
}
```

## 🔗 Integration Examples

### Python Integration

```python
import cdfa_unified as cdfa
import numpy as np
import pandas as pd

def python_integration():
    # Load data
    data = np.random.rand(1000, 5)
    df = pd.DataFrame(data, columns=['A', 'B', 'C', 'D', 'E'])
    
    # Calculate correlations
    corr_matrix = cdfa.correlation_matrix(data, method='pearson')
    print(f"Correlation matrix shape: {corr_matrix.shape}")
    
    # Fibonacci pattern detection
    prices = np.cumsum(np.random.randn(1000)) + 100
    detector = cdfa.FibonacciDetector(tolerance=0.02)
    patterns = detector.detect(prices)
    print(f"Found {len(patterns)} Fibonacci patterns")
    
    # Fusion analysis
    scores = np.random.rand(50, 10)
    fused = cdfa.fuse_scores(scores, method='adaptive')
    print(f"Fused scores: {fused[:5]}")
    
    # Performance comparison
    import time
    
    start = time.time()
    rust_corr = cdfa.pearson_correlation(data[:, 0], data[:, 1])
    rust_time = time.time() - start
    
    start = time.time()
    numpy_corr = np.corrcoef(data[:, 0], data[:, 1])[0, 1]
    numpy_time = time.time() - start
    
    print(f"Rust: {rust_corr:.6f} ({rust_time*1000:.2f}ms)")
    print(f"NumPy: {numpy_corr:.6f} ({numpy_time*1000:.2f}ms)")
    print(f"Speedup: {numpy_time/rust_time:.1f}x")

if __name__ == "__main__":
    python_integration()
```

### C/C++ Integration

```c
#include "cdfa_unified.h"
#include <stdio.h>
#include <stdlib.h>

int main() {
    // Initialize CDFA
    CDFAContext* ctx = cdfa_context_new();
    if (!ctx) {
        fprintf(stderr, "Failed to initialize CDFA context\n");
        return 1;
    }
    
    // Create sample data
    size_t n = 1000;
    double* data1 = malloc(n * sizeof(double));
    double* data2 = malloc(n * sizeof(double));
    
    for (size_t i = 0; i < n; i++) {
        data1[i] = (double)i / n;
        data2[i] = data1[i] + 0.1 * ((double)rand() / RAND_MAX - 0.5);
    }
    
    // Calculate correlation
    double correlation;
    CDFAResult result = cdfa_pearson_correlation(ctx, data1, data2, n, &correlation);
    
    if (result == CDFA_SUCCESS) {
        printf("Pearson correlation: %.6f\n", correlation);
    } else {
        fprintf(stderr, "Correlation calculation failed\n");
    }
    
    // Fibonacci pattern detection
    CDFAFibonacciDetector* detector = cdfa_fibonacci_detector_new(ctx, 0.02);
    CDFAPatterns* patterns = cdfa_fibonacci_detect(detector, data1, n);
    
    if (patterns) {
        printf("Found %zu Fibonacci patterns\n", cdfa_patterns_count(patterns));
        cdfa_patterns_free(patterns);
    }
    
    // Cleanup
    cdfa_fibonacci_detector_free(detector);
    cdfa_context_free(ctx);
    free(data1);
    free(data2);
    
    return 0;
}
```

### WebAssembly Integration

```javascript
import init, { CdfaWasm } from './pkg/cdfa_unified.js';

async function wasmIntegration() {
    // Initialize WebAssembly module
    await init();
    
    // Create CDFA instance
    const cdfa = new CdfaWasm();
    
    // Generate sample data
    const data1 = new Float64Array(1000);
    const data2 = new Float64Array(1000);
    
    for (let i = 0; i < 1000; i++) {
        data1[i] = Math.sin(i * 0.01);
        data2[i] = Math.cos(i * 0.01);
    }
    
    // Calculate correlation
    const correlation = cdfa.pearsonCorrelation(data1, data2);
    console.log(`Correlation: ${correlation.toFixed(6)}`);
    
    // Fusion analysis
    const scores = new Float64Array([
        0.8, 0.6, 0.9, 0.7,
        0.7, 0.8, 0.6, 0.9,
        0.9, 0.5, 0.8, 0.6
    ]);
    
    const fused = cdfa.fuseScores(scores, 3, 4, 'average');
    console.log(`Fused scores: ${Array.from(fused)}`);
    
    // Performance measurement
    const start = performance.now();
    for (let i = 0; i < 1000; i++) {
        cdfa.pearsonCorrelation(data1, data2);
    }
    const end = performance.now();
    
    console.log(`1000 correlations in ${end - start:.2f}ms`);
    console.log(`Average: ${(end - start)/1000:.3f}ms per correlation`);
}

wasmIntegration().catch(console.error);
```

## 📚 Best Practices

### Data Preprocessing

```rust
use cdfa_unified::prelude::*;

fn data_preprocessing() -> Result<()> {
    let mut data: Array2<f64> = Array2::random((1000, 10), rand::distributions::Standard);
    
    // Handle missing values
    data.mapv_inplace(|x| if x.is_nan() { 0.0 } else { x });
    
    // Normalize data
    for mut column in data.axis_iter_mut(Axis(1)) {
        let mean = column.mean().unwrap();
        let std = column.std(0.0);
        column.mapv_inplace(|x| (x - mean) / std);
    }
    
    // Remove outliers (3-sigma rule)
    let threshold = 3.0;
    data.mapv_inplace(|x| if x.abs() > threshold { threshold * x.signum() } else { x });
    
    // Verify data quality
    let has_nan = data.iter().any(|&x| x.is_nan());
    let has_inf = data.iter().any(|&x| x.is_infinite());
    
    println!("Data quality: NaN={}, Inf={}", has_nan, has_inf);
    
    Ok(())
}
```

### Error Handling

```rust
use cdfa_unified::prelude::*;

fn robust_analysis() -> Result<()> {
    let data: Array2<f64> = Array2::random((100, 5), rand::distributions::Standard);
    
    // Use try-catch pattern for operations
    match pearson_correlation(&data.column(0), &data.column(1)) {
        Ok(correlation) => println!("Correlation: {:.4}", correlation),
        Err(CdfaError::InvalidInput(msg)) => {
            eprintln!("Input validation failed: {}", msg);
        },
        Err(CdfaError::ComputationError(msg)) => {
            eprintln!("Computation failed: {}", msg);
        },
        Err(e) => eprintln!("Unexpected error: {}", e),
    }
    
    // Validate inputs before processing
    if data.nrows() < 2 {
        return Err(CdfaError::InvalidInput("Insufficient data points".into()));
    }
    
    if data.iter().any(|&x| !x.is_finite()) {
        return Err(CdfaError::InvalidInput("Data contains non-finite values".into()));
    }
    
    // Use fallback methods for edge cases
    let correlation = match pearson_correlation(&data.column(0), &data.column(1)) {
        Ok(corr) => corr,
        Err(_) => {
            eprintln!("Pearson correlation failed, falling back to Spearman");
            spearman_correlation(&data.column(0), &data.column(1))?
        }
    };
    
    println!("Robust correlation: {:.4}", correlation);
    
    Ok(())
}
```

### Performance Monitoring

```rust
use cdfa_unified::prelude::*;
use std::time::Instant;

fn performance_monitoring() -> Result<()> {
    let data: Array2<f64> = Array2::random((10000, 100), rand::distributions::Standard);
    
    // Measure operation performance
    let start = Instant::now();
    let correlation_matrix = data.correlation_matrix()?;
    let duration = start.elapsed();
    
    println!("Correlation matrix ({} x {}) computed in {:?}", 
             correlation_matrix.nrows(), correlation_matrix.ncols(), duration);
    
    // Memory usage tracking
    let memory_before = std::alloc::System.alloc_size();
    
    let large_result = expensive_computation(&data)?;
    
    let memory_after = std::alloc::System.alloc_size();
    let memory_used = memory_after - memory_before;
    
    println!("Memory used: {} MB", memory_used / 1024 / 1024);
    
    // Performance profiling
    let mut profiler = PerformanceProfiler::new();
    
    profiler.start("fibonacci_detection");
    let fib_detector = FibonacciPatternDetector::new()?;
    let patterns = fib_detector.detect_patterns(&data.column(0))?;
    profiler.end("fibonacci_detection");
    
    profiler.start("fusion_analysis");
    let fused = CdfaFusion::fuse(&data.view(), FusionMethod::Average, None)?;
    profiler.end("fusion_analysis");
    
    profiler.print_report();
    
    Ok(())
}

fn expensive_computation(data: &Array2<f64>) -> Result<Array1<f64>> {
    // Simulate expensive computation
    Ok(data.sum_axis(Axis(1)))
}
```

## 📖 API Reference

### Core Types

```rust
// Main result type
type Result<T> = std::result::Result<T, CdfaError>;

// Array types
type Array1<T> = ndarray::Array1<T>;
type Array2<T> = ndarray::Array2<T>;
type ArrayView1<'a, T> = ndarray::ArrayView1<'a, T>;
type ArrayView2<'a, T> = ndarray::ArrayView2<'a, T>;

// Configuration
struct CdfaConfig {
    pub threads: usize,
    pub simd_level: SIMDLevel,
    pub enable_gpu: bool,
    pub memory_pool_size: usize,
}

// Fusion methods
enum FusionMethod {
    Average,
    Median,
    Weighted(Vec<f64>),
    RankBased,
    BordaCount,
    Adaptive,
}
```

### Common Functions

```rust
// Correlation functions
fn pearson_correlation(x: &ArrayView1<f64>, y: &ArrayView1<f64>) -> Result<f64>;
fn spearman_correlation(x: &ArrayView1<f64>, y: &ArrayView1<f64>) -> Result<f64>;
fn kendall_tau(x: &ArrayView1<f64>, y: &ArrayView1<f64>) -> Result<f64>;

// Fusion functions
fn fuse_scores(data: &ArrayView2<f64>, method: FusionMethod) -> Result<Array1<f64>>;
fn fuse_ranks(data: &ArrayView2<f64>, method: FusionMethod) -> Result<Array1<usize>>;

// Pattern detection
struct FibonacciPatternDetector;
impl FibonacciPatternDetector {
    fn new() -> Result<Self>;
    fn detect_patterns(&self, data: &ArrayView1<f64>) -> Result<Vec<FibonacciPattern>>;
}

struct BlackSwanDetector;
impl BlackSwanDetector {
    fn new() -> Result<Self>;
    fn analyze(&self, data: &ArrayView1<f64>) -> Result<Vec<ExtremeEvent>>;
}
```

For complete API documentation, visit [docs.rs/cdfa-unified](https://docs.rs/cdfa-unified).

---

**Ready to dive deeper?** Check out our [examples directory](examples/) for complete working examples and use cases.