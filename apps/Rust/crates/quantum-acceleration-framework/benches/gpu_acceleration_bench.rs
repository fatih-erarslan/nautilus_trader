//! Comprehensive benchmarks for GPU acceleration performance validation

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use tokio::runtime::Runtime;
use qbmia_acceleration::*;
use std::time::Duration;

/// Benchmark quantum state evolution performance
fn bench_quantum_evolution(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    // Initialize accelerator
    let accelerator = rt.block_on(async {
        QBMIAAccelerator::new().await.unwrap()
    });
    
    let mut group = c.benchmark_group("quantum_evolution");
    group.measurement_time(Duration::from_secs(30));
    group.sample_size(1000);
    
    // Test different qubit counts
    for n_qubits in [4, 8, 12, 16].iter() {
        group.bench_with_input(
            BenchmarkId::new("hadamard_gates", n_qubits),
            n_qubits,
            |b, &n_qubits| {
                let state = QuantumState::new(n_qubits).unwrap();
                let gates = vec![UnitaryGate::hadamard(); n_qubits];
                let indices = (0..n_qubits).map(|i| vec![i]).collect::<Vec<_>>();
                
                b.to_async(&rt).iter(|| async {
                    let result = accelerator.evolve_quantum_state(
                        black_box(&state),
                        black_box(&gates),
                        black_box(&indices),
                    ).await;
                    black_box(result.unwrap());
                });
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("cnot_gates", n_qubits),
            n_qubits,
            |b, &n_qubits| {
                let state = QuantumState::new(n_qubits).unwrap();
                let gates = vec![UnitaryGate::cnot(); n_qubits - 1];
                let indices = (0..n_qubits-1).map(|i| vec![i, i+1]).collect::<Vec<_>>();
                
                b.to_async(&rt).iter(|| async {
                    let result = accelerator.evolve_quantum_state(
                        black_box(&state),
                        black_box(&gates),
                        black_box(&indices),
                    ).await;
                    black_box(result.unwrap());
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark Nash equilibrium solving performance
fn bench_nash_solving(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let accelerator = rt.block_on(async {
        QBMIAAccelerator::new().await.unwrap()
    });
    
    let mut group = c.benchmark_group("nash_solving");
    group.measurement_time(Duration::from_secs(30));
    group.sample_size(1000);
    
    // Test different matrix sizes
    for size in [2, 4, 8, 16, 32].iter() {
        group.bench_with_input(
            BenchmarkId::new("random_matrix", size),
            size,
            |b, &size| {
                let matrix = PayoffMatrix::random(size, size).unwrap();
                let strategies = StrategyVector::uniform(size).unwrap();
                let params = NashSolverParams::default();
                
                b.to_async(&rt).iter(|| async {
                    let result = accelerator.solve_nash_equilibrium(
                        black_box(&matrix),
                        black_box(&strategies),
                        black_box(&params),
                    ).await;
                    black_box(result.unwrap());
                });
            },
        );
    }
    
    // Special benchmark for 2x2 analytical solution (should be < 50ns)
    group.bench_function("2x2_analytical", |b| {
        let matrix = PayoffMatrix::new(2, 2, vec![3.0, 0.0, 0.0, 1.0]).unwrap();
        let strategies = StrategyVector::uniform(2).unwrap();
        let params = NashSolverParams::default();
        
        b.to_async(&rt).iter(|| async {
            let result = accelerator.solve_nash_equilibrium(
                black_box(&matrix),
                black_box(&strategies),
                black_box(&params),
            ).await;
            black_box(result.unwrap());
        });
    });
    
    group.finish();
}

/// Benchmark pattern matching performance
fn bench_pattern_matching(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let accelerator = rt.block_on(async {
        QBMIAAccelerator::new().await.unwrap()
    });
    
    let mut group = c.benchmark_group("pattern_matching");
    group.measurement_time(Duration::from_secs(30));
    group.sample_size(500);
    
    // Test different pattern counts and dimensions
    for &(n_patterns, dim) in &[(100, 64), (1000, 64), (10000, 64), (1000, 128), (1000, 256)] {
        group.bench_with_input(
            BenchmarkId::new("cosine_similarity", format!("{}x{}", n_patterns, dim)),
            &(n_patterns, dim),
            |b, &(n_patterns, dim)| {
                let patterns = (0..n_patterns)
                    .map(|_| Pattern::random(dim).unwrap())
                    .collect::<Vec<_>>();
                let query = Pattern::random(dim).unwrap();
                let threshold = 0.8;
                
                b.to_async(&rt).iter(|| async {
                    let result = accelerator.pattern_match(
                        black_box(&patterns),
                        black_box(&query),
                        black_box(threshold),
                    ).await;
                    black_box(result.unwrap());
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark SIMD operations performance
fn bench_simd_operations(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let accelerator = rt.block_on(async {
        QBMIAAccelerator::new().await.unwrap()
    });
    
    let mut group = c.benchmark_group("simd_operations");
    group.measurement_time(Duration::from_secs(20));
    group.sample_size(10000);
    
    // Test different vector sizes
    for size in [16, 64, 256, 1024, 4096].iter() {
        group.bench_with_input(
            BenchmarkId::new("dot_product", size),
            size,
            |b, &size| {
                let a = vec![1.0f32; size];
                let b = vec![2.0f32; size];
                
                b.iter(|| {
                    let result = accelerator.simd_processor.dot_product(
                        black_box(&a),
                        black_box(&b),
                    );
                    black_box(result.unwrap());
                });
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("matrix_vector_multiply", size),
            size,
            |b, &size| {
                let matrix = vec![1.0f32; size * size];
                let vector = vec![2.0f32; size];
                
                b.iter(|| {
                    let result = accelerator.simd_processor.matrix_vector_multiply(
                        black_box(&matrix),
                        black_box(&vector),
                        black_box(size),
                        black_box(size),
                    );
                    black_box(result.unwrap());
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark memory operations performance
fn bench_memory_operations(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let accelerator = rt.block_on(async {
        QBMIAAccelerator::new().await.unwrap()
    });
    
    let mut group = c.benchmark_group("memory_operations");
    group.measurement_time(Duration::from_secs(20));
    group.sample_size(100);
    
    // Test different buffer sizes
    for size_kb in [1, 10, 100, 1000].iter() {
        let size_bytes = size_kb * 1024;
        
        group.bench_with_input(
            BenchmarkId::new("buffer_create_read", format!("{}KB", size_kb)),
            &size_bytes,
            |b, &size_bytes| {
                let data = vec![42u8; size_bytes];
                
                b.to_async(&rt).iter(|| async {
                    let buffer = accelerator.memory_manager.create_buffer(
                        black_box(&data)
                    ).await.unwrap();
                    
                    let read_data = accelerator.memory_manager.read_buffer(
                        black_box(&buffer)
                    ).await.unwrap();
                    
                    black_box(read_data);
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark end-to-end pipeline performance
fn bench_pipeline_performance(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let accelerator = rt.block_on(async {
        QBMIAAccelerator::new().await.unwrap()
    });
    
    let mut group = c.benchmark_group("pipeline_performance");
    group.measurement_time(Duration::from_secs(30));
    group.sample_size(100);
    
    // Test parallel execution of mixed operations
    group.bench_function("mixed_operations_parallel", |b| {
        b.to_async(&rt).iter(|| async {
            // Prepare operations
            let quantum_state = QuantumState::new(8).unwrap();
            let gates = vec![UnitaryGate::hadamard(); 4];
            let indices = (0..4).map(|i| vec![i]).collect::<Vec<_>>();
            
            let payoff_matrix = PayoffMatrix::random(4, 4).unwrap();
            let strategies = StrategyVector::uniform(4).unwrap();
            let nash_params = NashSolverParams::default();
            
            let patterns = (0..1000)
                .map(|_| Pattern::random(64).unwrap())
                .collect::<Vec<_>>();
            let query = Pattern::random(64).unwrap();
            
            // Execute operations in parallel
            let (quantum_result, nash_result, pattern_result) = tokio::join!(
                accelerator.evolve_quantum_state(&quantum_state, &gates, &indices),
                accelerator.solve_nash_equilibrium(&payoff_matrix, &strategies, &nash_params),
                accelerator.pattern_match(&patterns, &query, 0.8)
            );
            
            black_box((quantum_result.unwrap(), nash_result.unwrap(), pattern_result.unwrap()));
        });
    });
    
    // Test warmup performance
    group.bench_function("warmup", |b| {
        b.to_async(&rt).iter(|| async {
            accelerator.warmup().await.unwrap();
        });
    });
    
    group.finish();
}

/// Validate performance targets
fn validate_performance_targets(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let accelerator = rt.block_on(async {
        QBMIAAccelerator::new().await.unwrap()
    });
    
    let mut group = c.benchmark_group("performance_targets");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(10000);
    
    // Target: Single Hadamard gate < 100ns
    group.bench_function("single_hadamard_target_100ns", |b| {
        let state = QuantumState::new(4).unwrap();
        let gates = vec![UnitaryGate::hadamard()];
        let indices = vec![vec![0]];
        
        b.to_async(&rt).iter(|| async {
            let start = std::time::Instant::now();
            let result = accelerator.evolve_quantum_state(
                black_box(&state),
                black_box(&gates),
                black_box(&indices),
            ).await.unwrap();
            let elapsed = start.elapsed();
            
            // Validate < 100ns target
            if elapsed.as_nanos() > 100 {
                eprintln!("WARNING: Single Hadamard took {}ns, exceeding 100ns target", elapsed.as_nanos());
            }
            
            black_box(result);
        });
    });
    
    // Target: 2x2 Nash equilibrium < 500ns
    group.bench_function("nash_2x2_target_500ns", |b| {
        let matrix = PayoffMatrix::new(2, 2, vec![3.0, 0.0, 0.0, 1.0]).unwrap();
        let strategies = StrategyVector::uniform(2).unwrap();
        let params = NashSolverParams::default();
        
        b.to_async(&rt).iter(|| async {
            let start = std::time::Instant::now();
            let result = accelerator.solve_nash_equilibrium(
                black_box(&matrix),
                black_box(&strategies),
                black_box(&params),
            ).await.unwrap();
            let elapsed = start.elapsed();
            
            // Validate < 500ns target
            if elapsed.as_nanos() > 500 {
                eprintln!("WARNING: 2x2 Nash took {}ns, exceeding 500ns target", elapsed.as_nanos());
            }
            
            black_box(result);
        });
    });
    
    // Target: Kernel launch overhead < 50ns
    group.bench_function("kernel_launch_target_50ns", |b| {
        let data = vec![1.0f32; 16];
        
        b.iter(|| {
            let start = std::time::Instant::now();
            let result = accelerator.simd_processor.dot_product(
                black_box(&data),
                black_box(&data),
            ).unwrap();
            let elapsed = start.elapsed();
            
            // Validate < 50ns target for small operations
            if elapsed.as_nanos() > 50 {
                eprintln!("WARNING: Small dot product took {}ns, exceeding 50ns target", elapsed.as_nanos());
            }
            
            black_box(result);
        });
    });
    
    group.finish();
}

/// Stress test with high frequency operations
fn stress_test_performance(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let accelerator = rt.block_on(async {
        QBMIAAccelerator::new().await.unwrap()
    });
    
    let mut group = c.benchmark_group("stress_test");
    group.measurement_time(Duration::from_secs(60));
    group.sample_size(10);
    
    // Stress test: 10,000 operations per second
    group.bench_function("high_frequency_operations", |b| {
        b.to_async(&rt).iter(|| async {
            let operations_per_batch = 100;
            let batches = 100; // Total: 10,000 operations
            
            for _ in 0..batches {
                let mut futures = Vec::with_capacity(operations_per_batch);
                
                for _ in 0..operations_per_batch {
                    let state = QuantumState::new(4).unwrap();
                    let gates = vec![UnitaryGate::hadamard()];
                    let indices = vec![vec![0]];
                    
                    futures.push(accelerator.evolve_quantum_state(&state, &gates, &indices));
                }
                
                let results = futures::future::join_all(futures).await;
                
                // Verify all operations succeeded
                for result in results {
                    black_box(result.unwrap());
                }
            }
        });
    });
    
    group.finish();
}

criterion_group!(
    benches,
    bench_quantum_evolution,
    bench_nash_solving,
    bench_pattern_matching,
    bench_simd_operations,
    bench_memory_operations,
    bench_pipeline_performance,
    validate_performance_targets,
    stress_test_performance
);

criterion_main!(benches);