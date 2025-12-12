//! # Quantum Uncertainty Benchmarks
//!
//! Performance benchmarks and quantum advantage validation for the 
//! quantum uncertainty quantification system.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use ndarray::{Array1, Array2};
use quantum_uncertainty::*;
use std::time::Duration;
use tokio::runtime::Runtime;

fn bench_quantum_uncertainty_pipeline(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("quantum_uncertainty_pipeline");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(30));
    
    // Test different configurations
    let configs = vec![
        ("lightweight", QuantumConfig::lightweight()),
        ("default", QuantumConfig::default()),
        ("high_performance", QuantumConfig::high_performance()),
    ];
    
    // Test different data sizes
    let data_sizes = vec![5, 10, 20, 50];
    
    for (config_name, config) in configs {
        for &size in &data_sizes {
            let engine = rt.block_on(async {
                QuantumUncertaintyEngine::new(config.clone()).await.unwrap()
            });
            
            let data = Array2::from_shape_fn((size, 4), |(i, j)| (i + j) as f64 / 10.0);
            let target = Array1::from_shape_fn(size, |i| i as f64 / 5.0);
            
            group.bench_with_input(
                BenchmarkId::new(config_name, size),
                &(data, target),
                |b, (data, target)| {
                    b.to_async(&rt).iter(|| async {
                        engine.quantify_uncertainty(
                            black_box(data),
                            black_box(target),
                        ).await.unwrap()
                    });
                },
            );
        }
    }
    
    group.finish();
}

fn bench_quantum_circuit_simulation(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantum_circuit_simulation");
    group.sample_size(20);
    
    // Test different qubit counts
    let qubit_counts = vec![2, 4, 6, 8, 10];
    
    for &n_qubits in &qubit_counts {
        let mut simulator = QuantumCircuitSimulator::new(n_qubits).unwrap();
        
        // Create a test circuit
        let circuit = QuantumCircuit::hardware_efficient_ansatz(
            n_qubits,
            3,
            format!("benchmark_circuit_{}", n_qubits),
        );
        
        group.bench_with_input(
            BenchmarkId::new("circuit_execution", n_qubits),
            &circuit,
            |b, circuit| {
                b.iter(|| {
                    simulator.execute_circuit(black_box(circuit)).unwrap();
                });
            },
        );
    }
    
    group.finish();
}

fn bench_quantum_feature_extraction(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("quantum_feature_extraction");
    group.sample_size(15);
    
    let config = QuantumConfig::lightweight();
    let extractor = QuantumFeatureExtractor::new(config).unwrap();
    
    // Test different data sizes and feature counts
    let test_cases = vec![
        (10, 3),   // Small dataset
        (50, 5),   // Medium dataset
        (100, 8),  // Large dataset
    ];
    
    for &(n_samples, n_features) in &test_cases {
        let data = Array2::from_shape_fn((n_samples, n_features), |(i, j)| {
            (i as f64 * j as f64).sin() / 10.0
        });
        
        group.bench_with_input(
            BenchmarkId::new("feature_extraction", format!("{}x{}", n_samples, n_features)),
            &data,
            |b, data| {
                b.to_async(&rt).iter(|| async {
                    extractor.extract_features(black_box(data)).await.unwrap()
                });
            },
        );
    }
    
    group.finish();
}

fn bench_quantum_correlation_analysis(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("quantum_correlation_analysis");
    group.sample_size(10);
    
    let config = QuantumConfig::lightweight();
    let analyzer = QuantumCorrelationAnalyzer::new(config).unwrap();
    
    // Test different feature counts
    let feature_counts = vec![3, 5, 8, 12];
    
    for &n_features in &feature_counts {
        let features = QuantumFeatures::new(
            (0..n_features).map(|i| i as f64 / n_features as f64).collect()
        );
        
        group.bench_with_input(
            BenchmarkId::new("correlation_analysis", n_features),
            &features,
            |b, features| {
                b.to_async(&rt).iter(|| async {
                    analyzer.analyze_correlations(black_box(features)).await.unwrap()
                });
            },
        );
    }
    
    group.finish();
}

fn bench_conformal_prediction(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("conformal_prediction");
    group.sample_size(15);
    
    let config = QuantumConfig::lightweight();
    let predictor = QuantumConformalPredictor::new(config).unwrap();
    
    // Test different ensemble sizes
    let ensemble_sizes = vec![3, 5, 10, 20];
    
    for &ensemble_size in &ensemble_sizes {
        let features = QuantumFeatures::new(vec![0.1, 0.3, 0.5, 0.7, 0.9]);
        let target = Array1::from_vec(vec![0.2, 0.4, 0.6, 0.8, 1.0]);
        let estimates: Vec<UncertaintyEstimate> = (0..ensemble_size)
            .map(|i| {
                UncertaintyEstimate::new(
                    0.1 + i as f64 * 0.05,
                    0.01 + i as f64 * 0.002,
                    (0.05 + i as f64 * 0.05, 0.15 + i as f64 * 0.05),
                    format!("vqc_{}", i),
                    0.95 - i as f64 * 0.01,
                )
            })
            .collect();
        
        group.bench_with_input(
            BenchmarkId::new("conformal_intervals", ensemble_size),
            &(features, target, estimates),
            |b, (features, target, estimates)| {
                b.to_async(&rt).iter(|| async {
                    predictor.create_prediction_intervals(
                        black_box(features),
                        black_box(target),
                        black_box(estimates),
                    ).await.unwrap()
                });
            },
        );
    }
    
    group.finish();
}

fn bench_measurement_optimization(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("measurement_optimization");
    group.sample_size(10);
    
    let config = QuantumConfig::lightweight();
    let optimizer = QuantumMeasurementOptimizer::new(config).unwrap();
    
    // Test different feature complexities
    let complexity_levels = vec![
        ("simple", vec![0.1, 0.2, 0.3]),
        ("medium", vec![0.1, 0.3, 0.5, 0.7, 0.9]),
        ("complex", vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]),
    ];
    
    for (complexity_name, feature_values) in complexity_levels {
        let features = QuantumFeatures::new(feature_values.clone());
        let estimates: Vec<UncertaintyEstimate> = feature_values
            .iter()
            .enumerate()
            .map(|(i, &val)| {
                UncertaintyEstimate::new(
                    val * 0.1,
                    val * 0.01,
                    (val * 0.05, val * 0.15),
                    format!("test_{}", i),
                    0.9 + val * 0.05,
                )
            })
            .collect();
        
        group.bench_with_input(
            BenchmarkId::new("optimization", complexity_name),
            &(features, estimates),
            |b, (features, estimates)| {
                b.to_async(&rt).iter(|| async {
                    optimizer.optimize_measurements(
                        black_box(features),
                        black_box(estimates),
                    ).await.unwrap()
                });
            },
        );
    }
    
    group.finish();
}

fn bench_quantum_advantage_validation(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("quantum_advantage_validation");
    group.sample_size(5);
    group.measurement_time(Duration::from_secs(60));
    
    // Compare quantum vs classical approaches
    let approaches = vec![
        ("quantum_lightweight", QuantumConfig::lightweight()),
        ("quantum_default", QuantumConfig::default()),
    ];
    
    let test_scenarios = vec![
        ("low_noise", 0.01),
        ("medium_noise", 0.05),
        ("high_noise", 0.1),
    ];
    
    for (approach_name, config) in approaches {
        for (scenario_name, noise_level) in &test_scenarios {
            let engine = rt.block_on(async {
                QuantumUncertaintyEngine::new(config.clone()).await.unwrap()
            });
            
            // Generate test data with controlled noise
            let data = Array2::from_shape_fn((20, 5), |(i, j)| {
                let base_value = (i as f64 * j as f64).sin() / 5.0;
                let noise = (rand::random::<f64>() - 0.5) * noise_level;
                base_value + noise
            });
            let target = Array1::from_shape_fn(20, |i| {
                let base_value = (i as f64 / 4.0).cos();
                let noise = (rand::random::<f64>() - 0.5) * noise_level;
                base_value + noise
            });
            
            group.bench_with_input(
                BenchmarkId::new(approach_name, scenario_name),
                &(data, target),
                |b, (data, target)| {
                    b.to_async(&rt).iter(|| async {
                        let result = engine.quantify_uncertainty(
                            black_box(data),
                            black_box(target),
                        ).await.unwrap();
                        black_box(result.quantum_advantage)
                    });
                },
            );
        }
    }
    
    group.finish();
}

fn bench_classical_quantum_integration(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("classical_quantum_integration");
    group.sample_size(8);
    
    let config = QuantumConfig::lightweight();
    let interface = ClassicalQuantumInterface::new(config).unwrap();
    
    // Test different integration scenarios
    let scenarios = vec![
        ("balanced", 0.5, 0.5),    // Equal quantum-classical weight
        ("quantum_heavy", 0.8, 0.2), // Quantum-heavy
        ("classical_heavy", 0.2, 0.8), // Classical-heavy
    ];
    
    for (scenario_name, _q_weight, _c_weight) in scenarios {
        let classical_data = Array2::from_shape_fn((15, 4), |(i, j)| {
            (i as f64 + j as f64) / 10.0
        });
        let quantum_features = QuantumFeatures::new(vec![0.2, 0.4, 0.6, 0.8]);
        let target = Array1::from_shape_fn(15, |i| i as f64 / 10.0);
        
        group.bench_with_input(
            BenchmarkId::new("hybrid_uncertainty", scenario_name),
            &(classical_data, quantum_features, target),
            |b, (classical_data, quantum_features, target)| {
                b.to_async(&rt).iter(|| async {
                    interface.hybrid_uncertainty_quantification(
                        black_box(classical_data),
                        black_box(quantum_features),
                        black_box(target),
                    ).await.unwrap()
                });
            },
        );
    }
    
    group.finish();
}

fn bench_pennylane_integration(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("pennylane_integration");
    group.sample_size(5);
    
    let config = QuantumConfig::lightweight();
    
    // Test different model types and sizes
    let model_configs = vec![
        ("vqc_small", 3, 2),
        ("vqc_medium", 4, 3),
        ("vqc_large", 6, 4),
    ];
    
    for (model_name, n_qubits, n_layers) in model_configs {
        group.bench_function(
            &format!("model_creation_{}", model_name),
            |b| {
                b.to_async(&rt).iter(|| async {
                    let mut interface = PennyLaneInterface::new(config.clone()).unwrap();
                    interface.initialize().await.unwrap();
                    
                    let model_id = interface.create_vqc(
                        black_box(format!("bench_{}", model_name)),
                        black_box(n_qubits),
                        black_box(n_layers),
                    ).await.unwrap();
                    
                    black_box(model_id)
                });
            },
        );
    }
    
    // Benchmark training performance
    let features = QuantumFeatures::new(vec![0.1, 0.2, 0.3, 0.4]);
    let targets = vec![0.5, 0.6, 0.7, 0.8];
    
    group.bench_function("model_training", |b| {
        b.to_async(&rt).iter(|| async {
            let mut interface = PennyLaneInterface::new(config.clone()).unwrap();
            interface.initialize().await.unwrap();
            
            let model_id = interface.create_vqc("training_bench".to_string(), 4, 2).await.unwrap();
            
            let training_config = TrainingConfig {
                max_epochs: 5,
                learning_rate: 0.1,
                ..Default::default()
            };
            
            let result = interface.train_model(
                &model_id,
                black_box(&features),
                black_box(&targets),
                black_box(training_config),
            ).await.unwrap();
            
            black_box(result.final_loss)
        });
    });
    
    group.finish();
}

fn bench_memory_and_scaling(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("memory_and_scaling");
    group.sample_size(5);
    
    // Test memory usage with different qubit counts
    let qubit_counts = vec![4, 6, 8, 10];
    
    for &n_qubits in &qubit_counts {
        let config = QuantumConfig {
            n_qubits,
            n_layers: 2,
            ensemble_size: 3,
            ..QuantumConfig::lightweight()
        };
        
        let estimated_memory = config.estimated_memory_mb();
        let estimated_complexity = config.estimated_complexity();
        
        group.bench_with_input(
            BenchmarkId::new("memory_scaling", n_qubits),
            &config,
            |b, config| {
                b.to_async(&rt).iter(|| async {
                    let engine = QuantumUncertaintyEngine::new(config.clone()).await.unwrap();
                    
                    let data_size = 2_usize.pow(n_qubits.min(4) as u32);
                    let data = Array2::from_shape_fn((data_size, 3), |(i, j)| {
                        (i + j) as f64 / 10.0
                    });
                    let target = Array1::from_shape_fn(data_size, |i| i as f64 / 5.0);
                    
                    let result = engine.quantify_uncertainty(
                        black_box(&data),
                        black_box(&target),
                    ).await.unwrap();
                    
                    black_box((result.quantum_advantage, estimated_memory, estimated_complexity))
                });
            },
        );
    }
    
    group.finish();
}

fn bench_real_time_performance(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("real_time_performance");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(10));
    
    let config = QuantumConfig::lightweight();
    let engine = rt.block_on(async {
        QuantumUncertaintyEngine::new(config).await.unwrap()
    });
    
    // Simulate real-time trading data updates
    let batch_sizes = vec![1, 5, 10, 20];
    
    for &batch_size in &batch_sizes {
        let data = Array2::from_shape_fn((batch_size, 5), |(i, j)| {
            // Simulate realistic trading features (price, volume, etc.)
            let base_value = 100.0 + i as f64;
            let feature_type = j as f64;
            base_value + feature_type * 0.1
        });
        let target = Array1::from_shape_fn(batch_size, |i| 100.0 + i as f64 * 0.1);
        
        group.bench_with_input(
            BenchmarkId::new("real_time_batch", batch_size),
            &(data, target),
            |b, (data, target)| {
                b.to_async(&rt).iter(|| async {
                    let start_time = std::time::Instant::now();
                    let result = engine.quantify_uncertainty(
                        black_box(data),
                        black_box(target),
                    ).await.unwrap();
                    let duration = start_time.elapsed();
                    
                    black_box((result.quantum_advantage, duration))
                });
            },
        );
    }
    
    group.finish();
}

criterion_group!(
    benches,
    bench_quantum_uncertainty_pipeline,
    bench_quantum_circuit_simulation,
    bench_quantum_feature_extraction,
    bench_quantum_correlation_analysis,
    bench_conformal_prediction,
    bench_measurement_optimization,
    bench_quantum_advantage_validation,
    bench_classical_quantum_integration,
    bench_pennylane_integration,
    bench_memory_and_scaling,
    bench_real_time_performance,
);

criterion_main!(benches);