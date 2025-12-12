//! Performance benchmarks for Quantum Nash Equilibrium solver

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use qbmia_core::{
    quantum::{QuantumNashEquilibrium, GameMatrix},
    config::QuantumConfig,
};
use ndarray::Array4;
use tokio::runtime::Runtime;

fn bench_quantum_nash_equilibrium(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("quantum_nash_equilibrium");
    
    // Test different problem sizes
    for num_qubits in [4, 8, 12, 16].iter() {
        group.bench_with_input(
            BenchmarkId::new("find_equilibrium", num_qubits),
            num_qubits,
            |b, &num_qubits| {
                let config = QuantumConfig {
                    num_qubits,
                    max_iterations: 50, // Shorter for benchmarks
                    ..QuantumConfig::default()
                };
                
                // Create test game matrix
                let matrix = Array4::from_elem((2, 2, 2, 2), 1.0);
                let game = GameMatrix::new(matrix).unwrap();
                
                b.to_async(&rt).iter(|| async {
                    let mut solver = QuantumNashEquilibrium::new(config.clone()).await.unwrap();
                    let result = solver.find_equilibrium(&game, None).await.unwrap();
                    black_box(result)
                });
            },
        );
    }
    
    group.finish();
}

fn bench_quantum_circuit_execution(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("quantum_circuit_execution");
    
    for num_qubits in [4, 8, 12, 16].iter() {
        group.bench_with_input(
            BenchmarkId::new("circuit_execution", num_qubits),
            num_qubits,
            |b, &num_qubits| {
                let config = QuantumConfig {
                    num_qubits,
                    num_layers: 1,
                    max_iterations: 1,
                    ..QuantumConfig::default()
                };
                
                let matrix = Array4::from_elem((2, 2, 2, 2), 1.0);
                let game = GameMatrix::new(matrix).unwrap();
                
                b.to_async(&rt).iter(|| async {
                    let mut solver = QuantumNashEquilibrium::new(config.clone()).await.unwrap();
                    let result = solver.find_equilibrium(&game, None).await.unwrap();
                    black_box(result)
                });
            },
        );
    }
    
    group.finish();
}

fn bench_entropy_calculation(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    c.bench_function("entropy_calculation", |b| {
        let config = QuantumConfig {
            num_qubits: 8,
            ..QuantumConfig::default()
        };
        
        b.to_async(&rt).iter(|| async {
            let solver = QuantumNashEquilibrium::new(config.clone()).await.unwrap();
            let probabilities = vec![0.25, 0.25, 0.25, 0.25];
            let entropy = solver.calculate_entropy(&probabilities);
            black_box(entropy)
        });
    });
}

criterion_group!(
    benches,
    bench_quantum_nash_equilibrium,
    bench_quantum_circuit_execution,
    bench_entropy_calculation
);
criterion_main!(benches);