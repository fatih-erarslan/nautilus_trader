//! # IQAD Performance Benchmarks
//! 
//! Comprehensive performance benchmarks for the Immune-inspired Quantum Anomaly Detection system.
//! Tests cover all critical performance aspects under realistic production scenarios.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use iqad::{
    ImmuneQuantumAnomalyDetector, 
    DetectorConfig, 
    AnomalyScore, 
    DataPoint,
    IqadResult,
    types::{QuantumCircuitConfig, ImmuneSystemConfig, HardwareConfig}
};
use ndarray::{Array1, Array2};
use rayon::prelude::*;
use std::time::Duration;
use tokio::runtime::Runtime;

// Test data generators
fn generate_normal_data(size: usize, dimensions: usize) -> Vec<DataPoint> {
    use rand::prelude::*;
    use rand_distr::Normal;
    
    let mut rng = thread_rng();
    let normal = Normal::new(0.0, 1.0).unwrap();
    
    (0..size)
        .map(|_| {
            let features: Vec<f64> = (0..dimensions)
                .map(|_| normal.sample(&mut rng))
                .collect();
            DataPoint::new(features, None)
        })
        .collect()
}

fn generate_anomaly_data(size: usize, dimensions: usize) -> Vec<DataPoint> {
    use rand::prelude::*;
    use rand_distr::Normal;
    
    let mut rng = thread_rng();
    let anomaly_dist = Normal::new(5.0, 2.0).unwrap(); // Shifted distribution for anomalies
    
    (0..size)
        .map(|_| {
            let features: Vec<f64> = (0..dimensions)
                .map(|_| anomaly_dist.sample(&mut rng))
                .collect();
            DataPoint::new(features, Some(true)) // Mark as anomaly
        })
        .collect()
}

fn generate_mixed_dataset(normal_size: usize, anomaly_size: usize, dimensions: usize) -> Vec<DataPoint> {
    let mut dataset = generate_normal_data(normal_size, dimensions);
    dataset.extend(generate_anomaly_data(anomaly_size, dimensions));
    
    // Shuffle to simulate realistic mixed data
    use rand::seq::SliceRandom;
    let mut rng = rand::thread_rng();
    dataset.shuffle(&mut rng);
    
    dataset
}

// Configuration generators
fn create_performance_config() -> DetectorConfig {
    DetectorConfig {
        quantum: QuantumCircuitConfig {
            num_qubits: 8,
            circuit_depth: 4,
            measurement_shots: 1000,
            optimization_level: 2,
            noise_model: None,
        },
        immune: ImmuneSystemConfig {
            num_detectors: 100,
            negative_selection_threshold: 0.8,
            clonal_selection_rate: 0.1,
            mutation_rate: 0.05,
            memory_cell_lifetime: 1000,
        },
        hardware: HardwareConfig {
            use_simd: true,
            use_gpu: false, // Disabled for benchmarking consistency
            max_threads: num_cpus::get(),
            cache_size: 10_000,
        },
        detection_threshold: 0.7,
        batch_size: 100,
    }
}

fn create_simd_config() -> DetectorConfig {
    let mut config = create_performance_config();
    config.hardware.use_simd = true;
    config
}

fn create_gpu_config() -> DetectorConfig {
    let mut config = create_performance_config();
    config.hardware.use_gpu = true;
    config
}

// 1. Detection Speed Benchmarks
fn bench_detection_speed(c: &mut Criterion) {
    let mut group = c.benchmark_group("detection_speed");
    
    let rt = Runtime::new().unwrap();
    let config = create_performance_config();
    
    // Test different data sizes
    for size in [100, 500, 1000, 5000, 10000].iter() {
        let data = generate_mixed_dataset(*size * 9 / 10, *size / 10, 10); // 90% normal, 10% anomalies
        
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(
            BenchmarkId::new("single_threaded", size),
            size,
            |b, _| {
                b.to_async(&rt).iter(|| async {
                    let detector = ImmuneQuantumAnomalyDetector::new(config.clone()).await.unwrap();
                    let results: Vec<AnomalyScore> = black_box(
                        detector.detect(&data).await.unwrap()
                    );
                    black_box(results)
                });
            },
        );
    }
    
    group.finish();
}

// 2. Batch Processing Benchmarks
fn bench_batch_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_processing");
    
    let rt = Runtime::new().unwrap();
    let config = create_performance_config();
    
    // Different batch sizes
    for batch_size in [10, 50, 100, 500, 1000].iter() {
        let data = generate_mixed_dataset(5000, 500, 10);
        let mut config = config.clone();
        config.batch_size = *batch_size;
        
        group.throughput(Throughput::Elements(data.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("batch_size", batch_size),
            batch_size,
            |b, _| {
                b.to_async(&rt).iter(|| async {
                    let detector = ImmuneQuantumAnomalyDetector::new(config.clone()).await.unwrap();
                    let results: Vec<AnomalyScore> = black_box(
                        detector.detect_batch(&data).await.unwrap()
                    );
                    black_box(results)
                });
            },
        );
    }
    
    group.finish();
}

// 3. Memory Usage Benchmarks
fn bench_memory_usage(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_usage");
    group.sample_size(10); // Fewer samples for memory tests
    
    let rt = Runtime::new().unwrap();
    
    // Test memory usage with different dataset sizes
    for size in [1000, 5000, 10000, 50000].iter() {
        let data = generate_mixed_dataset(*size * 9 / 10, *size / 10, 20);
        
        group.bench_with_input(
            BenchmarkId::new("memory_footprint", size),
            size,
            |b, _| {
                b.to_async(&rt).iter(|| async {
                    let config = create_performance_config();
                    let detector = ImmuneQuantumAnomalyDetector::new(config).await.unwrap();
                    
                    // Measure peak memory during detection
                    let results = black_box(
                        detector.detect(&data).await.unwrap()
                    );
                    
                    // Force memory cleanup
                    drop(detector);
                    black_box(results)
                });
            },
        );
    }
    
    group.finish();
}

// 4. Concurrent Performance Benchmarks
fn bench_concurrent_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent_performance");
    group.measurement_time(Duration::from_secs(30));
    
    let rt = Runtime::new().unwrap();
    let config = create_performance_config();
    
    // Test different thread counts
    for thread_count in [1, 2, 4, 8, 16].iter() {
        let data = generate_mixed_dataset(2000, 200, 10);
        let mut config = config.clone();
        config.hardware.max_threads = *thread_count;
        
        group.throughput(Throughput::Elements(data.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("threads", thread_count),
            thread_count,
            |b, _| {
                b.to_async(&rt).iter(|| async {
                    let detector = ImmuneQuantumAnomalyDetector::new(config.clone()).await.unwrap();
                    
                    // Simulate concurrent detection requests
                    let tasks: Vec<_> = (0..4)
                        .map(|_| {
                            let detector = detector.clone();
                            let data = data.clone();
                            tokio::spawn(async move {
                                detector.detect(&data).await.unwrap()
                            })
                        })
                        .collect();
                    
                    let results: Vec<Vec<AnomalyScore>> = futures::future::join_all(tasks)
                        .await
                        .into_iter()
                        .map(|r| r.unwrap())
                        .collect();
                    
                    black_box(results)
                });
            },
        );
    }
    
    group.finish();
}

// 5. SIMD Acceleration Benchmarks
fn bench_simd_acceleration(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_acceleration");
    
    let rt = Runtime::new().unwrap();
    let data = generate_mixed_dataset(5000, 500, 16); // Use 16D for better SIMD utilization
    
    // Compare SIMD enabled vs disabled
    for simd_enabled in [false, true].iter() {
        let mut config = create_performance_config();
        config.hardware.use_simd = *simd_enabled;
        
        let label = if *simd_enabled { "simd_enabled" } else { "simd_disabled" };
        
        group.throughput(Throughput::Elements(data.len() as u64));
        group.bench_function(label, |b| {
            b.to_async(&rt).iter(|| async {
                let detector = ImmuneQuantumAnomalyDetector::new(config.clone()).await.unwrap();
                let results: Vec<AnomalyScore> = black_box(
                    detector.detect(&data).await.unwrap()
                );
                black_box(results)
            });
        });
    }
    
    // SIMD-specific operations benchmark
    group.bench_function("simd_vector_operations", |b| {
        let vectors: Vec<Array1<f64>> = (0..1000)
            .map(|_| Array1::from_vec(generate_normal_data(1, 16)[0].features().to_vec()))
            .collect();
        
        b.iter(|| {
            let results: Vec<f64> = vectors
                .par_iter()
                .map(|v| {
                    // Simulate SIMD-accelerated vector operations
                    let norm = v.dot(v).sqrt();
                    black_box(norm)
                })
                .collect();
            black_box(results)
        });
    });
    
    group.finish();
}

// 6. Cache Performance Benchmarks
fn bench_cache_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_performance");
    
    let rt = Runtime::new().unwrap();
    
    // Test different cache sizes
    for cache_size in [100, 1000, 10000, 100000].iter() {
        let data = generate_mixed_dataset(2000, 200, 10);
        let mut config = create_performance_config();
        config.hardware.cache_size = *cache_size;
        
        group.bench_with_input(
            BenchmarkId::new("cache_size", cache_size),
            cache_size,
            |b, _| {
                b.to_async(&rt).iter(|| async {
                    let detector = ImmuneQuantumAnomalyDetector::new(config.clone()).await.unwrap();
                    
                    // First pass - populate cache
                    let _first_results = detector.detect(&data).await.unwrap();
                    
                    // Second pass - should hit cache
                    let results: Vec<AnomalyScore> = black_box(
                        detector.detect(&data).await.unwrap()
                    );
                    black_box(results)
                });
            },
        );
    }
    
    // Cache hit vs miss comparison
    group.bench_function("cache_hit_ratio", |b| {
        b.to_async(&rt).iter(|| async {
            let config = create_performance_config();
            let detector = ImmuneQuantumAnomalyDetector::new(config).await.unwrap();
            
            // Test with repeated vs unique data
            let repeated_data = generate_mixed_dataset(500, 50, 10);
            let unique_data = generate_mixed_dataset(500, 50, 10);
            
            // Warm up cache with repeated data
            let _warm_up = detector.detect(&repeated_data).await.unwrap();
            
            // Measure cache hits (repeated data)
            let cache_hits: Vec<AnomalyScore> = black_box(
                detector.detect(&repeated_data).await.unwrap()
            );
            
            // Measure cache misses (unique data)  
            let cache_misses: Vec<AnomalyScore> = black_box(
                detector.detect(&unique_data).await.unwrap()
            );
            
            black_box((cache_hits, cache_misses))
        });
    });
    
    group.finish();
}

// 7. Quantum Circuit Complexity Benchmarks
fn bench_quantum_complexity(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantum_complexity");
    group.sample_size(10); // Quantum operations are expensive
    
    let rt = Runtime::new().unwrap();
    let data = generate_mixed_dataset(1000, 100, 8);
    
    // Test different quantum circuit configurations
    for num_qubits in [4, 6, 8, 10, 12].iter() {
        let mut config = create_performance_config();
        config.quantum.num_qubits = *num_qubits;
        
        group.bench_with_input(
            BenchmarkId::new("qubits", num_qubits),
            num_qubits,
            |b, _| {
                b.to_async(&rt).iter(|| async {
                    let detector = ImmuneQuantumAnomalyDetector::new(config.clone()).await.unwrap();
                    let results: Vec<AnomalyScore> = black_box(
                        detector.detect(&data).await.unwrap()
                    );
                    black_box(results)
                });
            },
        );
    }
    
    for circuit_depth in [2, 4, 6, 8, 10].iter() {
        let mut config = create_performance_config();
        config.quantum.circuit_depth = *circuit_depth;
        
        group.bench_with_input(
            BenchmarkId::new("depth", circuit_depth),
            circuit_depth,
            |b, _| {
                b.to_async(&rt).iter(|| async {
                    let detector = ImmuneQuantumAnomalyDetector::new(config.clone()).await.unwrap();
                    let results: Vec<AnomalyScore> = black_box(
                        detector.detect(&data).await.unwrap()
                    );
                    black_box(results)
                });
            },
        );
    }
    
    group.finish();
}

// 8. Real-world Scenario Benchmarks
fn bench_production_scenarios(c: &mut Criterion) {
    let mut group = c.benchmark_group("production_scenarios");
    group.measurement_time(Duration::from_secs(60));
    group.sample_size(10);
    
    let rt = Runtime::new().unwrap();
    
    // Scenario 1: High-frequency trading anomaly detection
    group.bench_function("hft_trading_scenario", |b| {
        // Simulate financial time series data
        let trading_data = generate_mixed_dataset(10000, 100, 20); // 10k trades, 100 anomalies, 20 features
        let config = create_performance_config();
        
        b.to_async(&rt).iter(|| async {
            let detector = ImmuneQuantumAnomalyDetector::new(config.clone()).await.unwrap();
            let results: Vec<AnomalyScore> = black_box(
                detector.detect(&trading_data).await.unwrap()
            );
            black_box(results)
        });
    });
    
    // Scenario 2: Network intrusion detection
    group.bench_function("network_intrusion_scenario", |b| {
        // Simulate network traffic data
        let network_data = generate_mixed_dataset(50000, 500, 15); // 50k packets, 500 intrusions, 15 features
        let mut config = create_performance_config();
        config.batch_size = 1000; // Larger batches for network data
        
        b.to_async(&rt).iter(|| async {
            let detector = ImmuneQuantumAnomalyDetector::new(config.clone()).await.unwrap();
            let results: Vec<AnomalyScore> = black_box(
                detector.detect_batch(&network_data).await.unwrap()
            );
            black_box(results)
        });
    });
    
    // Scenario 3: IoT sensor anomaly detection
    group.bench_function("iot_sensor_scenario", |b| {
        // Simulate IoT sensor data
        let sensor_data = generate_mixed_dataset(100000, 1000, 8); // 100k readings, 1k anomalies, 8 sensors
        let mut config = create_performance_config();
        config.hardware.max_threads = num_cpus::get(); // Use all cores for IoT scale
        
        b.to_async(&rt).iter(|| async {
            let detector = ImmuneQuantumAnomalyDetector::new(config.clone()).await.unwrap();
            let results: Vec<AnomalyScore> = black_box(
                detector.detect(&sensor_data).await.unwrap()
            );
            black_box(results)
        });
    });
    
    group.finish();
}

// 9. Baseline Comparison Benchmarks
fn bench_baseline_comparisons(c: &mut Criterion) {
    let mut group = c.benchmark_group("baseline_comparisons");
    
    let rt = Runtime::new().unwrap();
    let data = generate_mixed_dataset(5000, 500, 10);
    
    // Compare against classical detection methods
    group.bench_function("quantum_immune_system", |b| {
        let config = create_performance_config();
        
        b.to_async(&rt).iter(|| async {
            let detector = ImmuneQuantumAnomalyDetector::new(config.clone()).await.unwrap();
            let results: Vec<AnomalyScore> = black_box(
                detector.detect(&data).await.unwrap()
            );
            black_box(results)
        });
    });
    
    group.bench_function("classical_statistical", |b| {
        b.iter(|| {
            // Simple statistical anomaly detection baseline
            let results: Vec<f64> = data
                .iter()
                .map(|point| {
                    let features = point.features();
                    let mean = features.iter().sum::<f64>() / features.len() as f64;
                    let variance = features
                        .iter()
                        .map(|x| (x - mean).powi(2))
                        .sum::<f64>() / features.len() as f64;
                    
                    // Z-score based anomaly detection
                    let z_score = (mean - 0.0) / variance.sqrt();
                    black_box(z_score.abs())
                })
                .collect();
            black_box(results)
        });
    });
    
    group.finish();
}

// Criterion configuration
criterion_group!(
    name = benches;
    config = Criterion::default()
        .measurement_time(Duration::from_secs(20))
        .warm_up_time(Duration::from_secs(5))
        .sample_size(50);
    targets = 
        bench_detection_speed,
        bench_batch_processing,
        bench_memory_usage,
        bench_concurrent_performance,
        bench_simd_acceleration,
        bench_cache_performance,
        bench_quantum_complexity,
        bench_production_scenarios,
        bench_baseline_comparisons
);

criterion_main!(benches);