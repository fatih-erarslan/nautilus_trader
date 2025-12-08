// Quantum Execution Benchmark for Quantum Bridge
// Copyright (c) 2025 TENGRI Trading Swarm

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use quantum_bridge::{
    QuantumBridge, QuantumCircuit, QuantumGate, QuantumState, QuantumMeasurement,
    QuantumExecutor, QuantumBackend, QuantumSimulator, QuantumDevice,
    CircuitBuilder, StatePreparation, QuantumAlgorithm
};
use std::time::Instant;

fn create_sample_circuit(qubits: usize, depth: usize) -> QuantumCircuit {
    let mut builder = CircuitBuilder::new(qubits);
    
    for layer in 0..depth {
        for qubit in 0..qubits {
            match layer % 4 {
                0 => builder.add_gate(QuantumGate::Hadamard, qubit),
                1 => builder.add_gate(QuantumGate::PauliX, qubit),
                2 => builder.add_gate(QuantumGate::PauliY, qubit),
                3 => builder.add_gate(QuantumGate::PauliZ, qubit),
                _ => unreachable!(),
            }
        }
        
        // Add entangling gates
        for qubit in 0..(qubits - 1) {
            builder.add_gate(QuantumGate::CNOT, qubit);
        }
    }
    
    builder.build()
}

fn benchmark_circuit_execution(c: &mut Criterion) {
    let mut group = c.benchmark_group("circuit_execution");
    
    let bridge = QuantumBridge::new();
    
    for qubits in [4, 6, 8, 10, 12].iter() {
        for depth in [5, 10, 15, 20].iter() {
            let circuit = create_sample_circuit(*qubits, *depth);
            
            group.bench_with_input(
                BenchmarkId::new("execute_circuit", format!("{}q_{}d", qubits, depth)),
                &(qubits, depth),
                |b, _| {
                    b.iter(|| {
                        bridge.execute_circuit(&circuit)
                    })
                }
            );
        }
    }
    group.finish();
}

fn benchmark_quantum_state_preparation(c: &mut Criterion) {
    let mut group = c.benchmark_group("state_preparation");
    
    let bridge = QuantumBridge::new();
    
    for qubits in [4, 6, 8, 10].iter() {
        group.bench_with_input(
            BenchmarkId::new("prepare_bell_state", qubits),
            qubits,
            |b, &q| {
                b.iter(|| {
                    bridge.prepare_bell_state(q)
                })
            }
        );
        
        group.bench_with_input(
            BenchmarkId::new("prepare_ghz_state", qubits),
            qubits,
            |b, &q| {
                b.iter(|| {
                    bridge.prepare_ghz_state(q)
                })
            }
        );
        
        group.bench_with_input(
            BenchmarkId::new("prepare_random_state", qubits),
            qubits,
            |b, &q| {
                b.iter(|| {
                    bridge.prepare_random_state(q)
                })
            }
        );
    }
    group.finish();
}

fn benchmark_quantum_measurements(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantum_measurements");
    
    let bridge = QuantumBridge::new();
    
    for qubits in [4, 6, 8, 10].iter() {
        let circuit = create_sample_circuit(*qubits, 10);
        let state = bridge.execute_circuit(&circuit);
        
        group.bench_with_input(
            BenchmarkId::new("measure_all", qubits),
            qubits,
            |b, _| {
                b.iter(|| {
                    bridge.measure_all(&state)
                })
            }
        );
        
        group.bench_with_input(
            BenchmarkId::new("measure_single", qubits),
            qubits,
            |b, &q| {
                b.iter(|| {
                    bridge.measure_qubit(&state, q / 2)
                })
            }
        );
        
        group.bench_with_input(
            BenchmarkId::new("measure_expectation", qubits),
            qubits,
            |b, _| {
                b.iter(|| {
                    bridge.measure_expectation_value(&state, &QuantumGate::PauliZ)
                })
            }
        );
    }
    group.finish();
}

fn benchmark_quantum_algorithms(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantum_algorithms");
    
    let bridge = QuantumBridge::new();
    
    // Quantum Fourier Transform
    for qubits in [4, 6, 8].iter() {
        group.bench_with_input(
            BenchmarkId::new("qft", qubits),
            qubits,
            |b, &q| {
                b.iter(|| {
                    bridge.quantum_fourier_transform(q)
                })
            }
        );
    }
    
    // Grover's Algorithm
    for qubits in [4, 6, 8].iter() {
        group.bench_with_input(
            BenchmarkId::new("grover", qubits),
            qubits,
            |b, &q| {
                b.iter(|| {
                    bridge.grovers_algorithm(q, vec![0]) // Search for |0...0⟩
                })
            }
        );
    }
    
    // Variational Quantum Eigensolver (VQE)
    for qubits in [2, 4, 6].iter() {
        group.bench_with_input(
            BenchmarkId::new("vqe", qubits),
            qubits,
            |b, &q| {
                b.iter(|| {
                    bridge.variational_quantum_eigensolver(q)
                })
            }
        );
    }
    
    group.finish();
}

fn benchmark_quantum_backends(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantum_backends");
    
    let circuit = create_sample_circuit(8, 10);
    
    // Simulator backend
    group.bench_function("simulator_backend", |b| {
        b.iter(|| {
            let backend = QuantumSimulator::new();
            backend.execute(&circuit)
        })
    });
    
    // Different simulator types
    let simulators = vec![
        ("statevector", QuantumBackend::StateVector),
        ("density_matrix", QuantumBackend::DensityMatrix),
        ("stabilizer", QuantumBackend::Stabilizer),
    ];
    
    for (name, backend_type) in simulators.iter() {
        group.bench_with_input(
            BenchmarkId::new("backend_type", name),
            backend_type,
            |b, backend| {
                b.iter(|| {
                    let executor = QuantumExecutor::new(backend.clone());
                    executor.execute(&circuit)
                })
            }
        );
    }
    
    group.finish();
}

fn benchmark_quantum_noise_simulation(c: &mut Criterion) {
    let mut group = c.benchmark_group("noise_simulation");
    
    let bridge = QuantumBridge::new();
    let circuit = create_sample_circuit(6, 10);
    
    // Different noise models
    let noise_levels = vec![0.0, 0.001, 0.01, 0.1];
    
    for noise_level in noise_levels.iter() {
        group.bench_with_input(
            BenchmarkId::new("depolarizing_noise", format!("{:.3}", noise_level)),
            noise_level,
            |b, &noise| {
                b.iter(|| {
                    bridge.execute_with_noise(&circuit, noise)
                })
            }
        );
    }
    
    group.finish();
}

fn benchmark_quantum_optimization(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantum_optimization");
    
    let bridge = QuantumBridge::new();
    
    // Quantum Approximate Optimization Algorithm (QAOA)
    for layers in [1, 2, 3, 4].iter() {
        group.bench_with_input(
            BenchmarkId::new("qaoa", format!("{}layers", layers)),
            layers,
            |b, &p| {
                b.iter(|| {
                    bridge.qaoa_optimization(6, p) // 6 qubits, p layers
                })
            }
        );
    }
    
    // Quantum Adiabatic Algorithm
    for qubits in [4, 6, 8].iter() {
        group.bench_with_input(
            BenchmarkId::new("adiabatic", qubits),
            qubits,
            |b, &q| {
                b.iter(|| {
                    bridge.adiabatic_optimization(q)
                })
            }
        );
    }
    
    group.finish();
}

fn benchmark_parallel_quantum_execution(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_execution");
    
    let bridge = QuantumBridge::new();
    let circuits: Vec<_> = (0..10).map(|i| create_sample_circuit(6, 5 + i)).collect();
    
    for thread_count in [1, 2, 4, 8].iter() {
        group.bench_with_input(
            BenchmarkId::new("parallel_circuits", thread_count),
            thread_count,
            |b, &threads| {
                b.iter(|| {
                    bridge.execute_parallel(&circuits, threads)
                })
            }
        );
    }
    
    group.finish();
}

criterion_group!(
    benches,
    benchmark_circuit_execution,
    benchmark_quantum_state_preparation,
    benchmark_quantum_measurements,
    benchmark_quantum_algorithms,
    benchmark_quantum_backends,
    benchmark_quantum_noise_simulation,
    benchmark_quantum_optimization,
    benchmark_parallel_quantum_execution
);
criterion_main!(benches);