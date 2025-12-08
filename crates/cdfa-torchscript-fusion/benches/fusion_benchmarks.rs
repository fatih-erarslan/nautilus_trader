//! Performance benchmarks for TorchScript Fusion operations
//!
//! These benchmarks measure the performance of all fusion types across different
//! signal sizes and device configurations.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use cdfa_torchscript_fusion::{TorchScriptFusion, FusionType, FusionParams};
use ndarray::Array2;
use tokio::runtime::Runtime;

/// Benchmark configuration
struct BenchConfig {
    num_signals: usize,
    sequence_length: usize,
    fusion_type: FusionType,
}

impl BenchConfig {
    fn new(num_signals: usize, sequence_length: usize, fusion_type: FusionType) -> Self {
        Self {
            num_signals,
            sequence_length,
            fusion_type,
        }
    }

    fn id(&self) -> String {
        format!(
            "{}sig_{}len_{}",
            self.num_signals,
            self.sequence_length,
            self.fusion_type.as_str()
        )
    }
}

/// Create test data for benchmarks
fn create_test_data(num_signals: usize, sequence_length: usize) -> (Array2<f32>, Array2<f32>) {
    let signals = Array2::from_shape_fn((num_signals, sequence_length), |(i, j)| {
        // Create diverse signals with different characteristics
        let base = (i as f32 + 1.0) * 0.1;
        let time = j as f32 * 0.01;
        let noise = (i as f32 * 1.37 + j as f32 * 2.71).sin() * 0.05;
        
        match i % 4 {
            0 => base + time.sin() + noise,           // Sine wave
            1 => base + time.cos() + noise,           // Cosine wave
            2 => base + (time * 2.0).sin() * 0.5 + noise, // Higher frequency
            _ => base + time.powi(2) * 0.001 + noise, // Quadratic trend
        }
    });

    let confidences = Array2::from_shape_fn((num_signals, sequence_length), |(i, j)| {
        // Varying confidence patterns
        let base_conf = 0.7 + (i as f32 * 0.05);
        let time_factor = (j as f32 * 0.1).cos() * 0.1;
        (base_conf + time_factor).clamp(0.1, 1.0)
    });

    (signals, confidences)
}

/// Benchmark all fusion types with different signal configurations
fn bench_fusion_types(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let configs = vec![
        // Small signals - high frequency trading
        BenchConfig::new(3, 100, FusionType::Score),
        BenchConfig::new(3, 100, FusionType::Rank),
        BenchConfig::new(3, 100, FusionType::Hybrid),
        BenchConfig::new(3, 100, FusionType::Weighted),
        BenchConfig::new(3, 100, FusionType::Layered),
        BenchConfig::new(3, 100, FusionType::Adaptive),
        
        // Medium signals - standard analysis
        BenchConfig::new(5, 500, FusionType::Score),
        BenchConfig::new(5, 500, FusionType::Rank),
        BenchConfig::new(5, 500, FusionType::Hybrid),
        BenchConfig::new(5, 500, FusionType::Weighted),
        BenchConfig::new(5, 500, FusionType::Layered),
        BenchConfig::new(5, 500, FusionType::Adaptive),
        
        // Large signals - comprehensive analysis
        BenchConfig::new(10, 1000, FusionType::Score),
        BenchConfig::new(10, 1000, FusionType::Rank),
        BenchConfig::new(10, 1000, FusionType::Hybrid),
        BenchConfig::new(10, 1000, FusionType::Weighted),
        BenchConfig::new(10, 1000, FusionType::Layered),
        BenchConfig::new(10, 1000, FusionType::Adaptive),
    ];

    let mut group = c.benchmark_group("fusion_operations");
    
    for config in configs {
        let (signals, confidences) = create_test_data(config.num_signals, config.sequence_length);
        let params = FusionParams::default();
        
        group.throughput(Throughput::Elements(
            (config.num_signals * config.sequence_length) as u64
        ));
        
        group.bench_with_input(
            BenchmarkId::new("fusion", config.id()),
            &config,
            |b, config| {
                b.to_async(&rt).iter(|| async {
                    let mut fusion = TorchScriptFusion::new().await.unwrap();
                    
                    let result = fusion.fuse_signals(
                        black_box(&signals),
                        black_box(&confidences),
                        black_box(config.fusion_type),
                        black_box(&params),
                    ).await.unwrap();
                    
                    black_box(result)
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark sub-microsecond performance targets
fn bench_sub_microsecond_targets(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    // Ultra-small signals for sub-microsecond performance
    let test_cases = vec![
        (2, 10),   // Minimal case
        (2, 50),   // Small case
        (3, 10),   // Few signals
        (5, 10),   // More signals
    ];

    let mut group = c.benchmark_group("sub_microsecond");
    group.measurement_time(std::time::Duration::from_secs(10));
    group.sample_size(1000);
    
    for (num_signals, seq_len) in test_cases {
        let (signals, confidences) = create_test_data(num_signals, seq_len);
        let params = FusionParams::default().with_chunk_size(seq_len);
        
        group.throughput(Throughput::Elements((num_signals * seq_len) as u64));
        
        // Test score fusion (fastest)
        group.bench_with_input(
            BenchmarkId::new("score", format!("{}x{}", num_signals, seq_len)),
            &(num_signals, seq_len),
            |b, _| {
                b.to_async(&rt).iter(|| async {
                    let mut fusion = TorchScriptFusion::new().await.unwrap();
                    
                    let result = fusion.fuse_signals(
                        black_box(&signals),
                        black_box(&confidences),
                        black_box(FusionType::Score),
                        black_box(&params),
                    ).await.unwrap();
                    
                    black_box(result)
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark memory efficiency and allocation patterns
fn bench_memory_efficiency(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("memory_efficiency");
    
    // Test different signal sizes to measure memory scaling
    let sizes = vec![
        (10, 100),   // 1K elements
        (20, 500),   // 10K elements  
        (50, 1000),  // 50K elements
        (100, 1000), // 100K elements
    ];
    
    for (num_signals, seq_len) in sizes {
        let (signals, confidences) = create_test_data(num_signals, seq_len);
        let params = FusionParams::default();
        
        group.throughput(Throughput::Bytes(
            (num_signals * seq_len * std::mem::size_of::<f32>() * 2) as u64
        ));
        
        group.bench_with_input(
            BenchmarkId::new("weighted_fusion", format!("{}x{}", num_signals, seq_len)),
            &(num_signals, seq_len),
            |b, _| {
                b.to_async(&rt).iter(|| async {
                    let mut fusion = TorchScriptFusion::new().await.unwrap();
                    
                    let result = fusion.fuse_signals(
                        black_box(&signals),
                        black_box(&confidences),
                        black_box(FusionType::Weighted), // Most memory intensive
                        black_box(&params),
                    ).await.unwrap();
                    
                    black_box(result)
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark parameter sensitivity
fn bench_parameter_sensitivity(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let (signals, confidences) = create_test_data(5, 500);
    
    let mut group = c.benchmark_group("parameter_sensitivity");
    
    let param_configs = vec![
        ("default", FusionParams::default()),
        ("min_weight_005", FusionParams::default().with_min_weight(0.05)),
        ("min_weight_001", FusionParams::default().with_min_weight(0.001)),
        ("score_alpha_08", FusionParams::default().with_score_alpha(0.8)),
        ("score_alpha_02", FusionParams::default().with_score_alpha(0.2)),
        ("nonlinear_disabled", FusionParams::default().with_nonlinear_weighting(false)),
        ("chunk_100", FusionParams::default().with_chunk_size(100)),
        ("chunk_1000", FusionParams::default().with_chunk_size(1000)),
    ];
    
    for (name, params) in param_configs {
        group.bench_with_input(
            BenchmarkId::new("hybrid_fusion", name),
            &params,
            |b, params| {
                b.to_async(&rt).iter(|| async {
                    let mut fusion = TorchScriptFusion::new().await.unwrap();
                    
                    let result = fusion.fuse_signals(
                        black_box(&signals),
                        black_box(&confidences),
                        black_box(FusionType::Hybrid),
                        black_box(params),
                    ).await.unwrap();
                    
                    black_box(result)
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark device performance comparison
fn bench_device_performance(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let (signals, confidences) = create_test_data(8, 1000);
    let params = FusionParams::default();
    
    let mut group = c.benchmark_group("device_performance");
    
    // Test CPU performance
    group.bench_function("cpu_fusion", |b| {
        b.to_async(&rt).iter(|| async {
            use candle_core::Device;
            let mut fusion = TorchScriptFusion::with_device(Device::Cpu).await.unwrap();
            
            let result = fusion.fuse_signals(
                black_box(&signals),
                black_box(&confidences),
                black_box(FusionType::Score),
                black_box(&params),
            ).await.unwrap();
            
            black_box(result)
        });
    });
    
    // Test GPU performance if available
    if cdfa_torchscript_fusion::gpu_available() {
        group.bench_function("gpu_fusion", |b| {
            b.to_async(&rt).iter(|| async {
                let mut fusion = TorchScriptFusion::new().await.unwrap();
                
                let result = fusion.fuse_signals(
                    black_box(&signals),
                    black_box(&confidences),
                    black_box(FusionType::Score),
                    black_box(&params),
                ).await.unwrap();
                
                black_box(result)
            });
        });
    }
    
    group.finish();
}

/// Benchmark compilation overhead
fn bench_compilation_overhead(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let (signals, confidences) = create_test_data(5, 200);
    let params = FusionParams::default();
    
    let mut group = c.benchmark_group("compilation_overhead");
    
    // Measure first-time compilation
    group.bench_function("first_compile", |b| {
        b.to_async(&rt).iter(|| async {
            let mut fusion = TorchScriptFusion::new().await.unwrap();
            
            let result = fusion.fuse_signals(
                black_box(&signals),
                black_box(&confidences),
                black_box(FusionType::Score),
                black_box(&params),
            ).await.unwrap();
            
            black_box(result)
        });
    });
    
    // Measure cached execution
    group.bench_function("cached_execution", |b| {
        b.to_async(&rt).iter_batched(
            || {
                rt.block_on(async {
                    let mut fusion = TorchScriptFusion::new().await.unwrap();
                    // Pre-compile by running once
                    let _ = fusion.fuse_signals(&signals, &confidences, FusionType::Score, &params).await.unwrap();
                    fusion
                })
            },
            |mut fusion| async move {
                let result = fusion.fuse_signals(
                    black_box(&signals),
                    black_box(&confidences),
                    black_box(FusionType::Score),
                    black_box(&params),
                ).await.unwrap();
                
                black_box(result)
            },
            criterion::BatchSize::SmallInput,
        );
    });
    
    group.finish();
}

criterion_group!(
    benches,
    bench_fusion_types,
    bench_sub_microsecond_targets,
    bench_memory_efficiency,
    bench_parameter_sensitivity,
    bench_device_performance,
    bench_compilation_overhead
);

criterion_main!(benches);