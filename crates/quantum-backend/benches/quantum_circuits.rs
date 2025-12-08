//! Quantum circuit execution benchmarks

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use quantum_backend::*;
use quantum_core::*;
use tokio::runtime::Runtime;

fn bench_circuit_execution(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    c.bench_function("4-qubit quantum circuit", |b| {
        b.iter(|| {
            rt.block_on(async {
                let backend = QuantumBackend::new().await.unwrap();
                
                let mut circuit = QuantumCircuit::new(4);
                circuit.add_gate(QuantumGate::hadamard(0));
                circuit.add_gate(QuantumGate::cnot(0, 1));
                circuit.add_gate(QuantumGate::cnot(1, 2));
                circuit.add_gate(QuantumGate::cnot(2, 3));
                
                let result = backend.execute_circuit(&circuit).await.unwrap();
                black_box(result);
            })
        })
    });
    
    c.bench_function("8-qubit quantum circuit", |b| {
        b.iter(|| {
            rt.block_on(async {
                let backend = QuantumBackend::new().await.unwrap();
                
                let mut circuit = QuantumCircuit::new(8);
                for i in 0..8 {
                    circuit.add_gate(QuantumGate::hadamard(i));
                }
                for i in 0..7 {
                    circuit.add_gate(QuantumGate::cnot(i, i + 1));
                }
                
                let result = backend.execute_circuit(&circuit).await.unwrap();
                black_box(result);
            })
        })
    });
    
    c.bench_function("12-qubit quantum circuit", |b| {
        b.iter(|| {
            rt.block_on(async {
                let backend = QuantumBackend::new().await.unwrap();
                
                let mut circuit = QuantumCircuit::new(12);
                // Bell state preparation
                for i in 0..12 {
                    circuit.add_gate(QuantumGate::hadamard(i));
                }
                // Entangling layer
                for i in 0..11 {
                    circuit.add_gate(QuantumGate::cnot(i, i + 1));
                }
                // Rotation layer
                for i in 0..12 {
                    circuit.add_gate(QuantumGate::rz(i, 0.5));
                }
                
                let result = backend.execute_circuit(&circuit).await.unwrap();
                black_box(result);
            })
        })
    });
}

fn bench_nash_solver(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    c.bench_function("2x2 Nash equilibrium", |b| {
        b.iter(|| {
            rt.block_on(async {
                let backend = QuantumBackend::new().await.unwrap();
                
                let payoff = ndarray::Array2::from_shape_vec(
                    (2, 2),
                    vec![3.0, 0.0, 5.0, 1.0]
                ).unwrap();
                
                let solution = backend.solve_nash_equilibrium(&payoff, 4).await.unwrap();
                black_box(solution);
            })
        })
    });
    
    c.bench_function("3x3 Nash equilibrium", |b| {
        b.iter(|| {
            rt.block_on(async {
                let backend = QuantumBackend::new().await.unwrap();
                
                let payoff = ndarray::Array2::from_shape_vec(
                    (3, 3),
                    vec![1.0, 2.0, 0.0, 0.0, 1.0, 2.0, 2.0, 0.0, 1.0]
                ).unwrap();
                
                let solution = backend.solve_nash_equilibrium(&payoff, 6).await.unwrap();
                black_box(solution);
            })
        })
    });
}

criterion_group!(benches, bench_circuit_execution, bench_nash_solver);
criterion_main!(benches);