//! Benchmarks for quantum-circuit crate

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use quantum_circuit::{
    Circuit, CircuitBuilder, VariationalCircuit, EntanglementPattern,
    Simulator, BatchSimulator,
    QAOAOptimizer, OptimizerConfig, Optimizer,
    AmplitudeEmbedding, AngleEmbedding, ParametricEmbedding, NormalizationMethod, QuantumEmbedding,
    gates::*,
    constants,
};
use std::f64::consts::PI;

fn bench_gate_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("gate_operations");
    
    // Benchmark single-qubit gates
    group.bench_function("hadamard_gate", |b| {
        let mut state = constants::zero_state();
        let h = Hadamard::new(0);
        b.iter(|| {
            h.apply(&mut state).unwrap();
        });
    });
    
    group.bench_function("rx_gate", |b| {
        let mut state = constants::zero_state();
        let rx = RX::new(0, PI / 4.0);
        b.iter(|| {
            rx.apply(&mut state).unwrap();
        });
    });
    
    // Benchmark two-qubit gates with larger state
    group.bench_function("cnot_gate", |b| {
        let mut state = ndarray::Array1::zeros(4);
        state[0] = quantum_circuit::Complex::new(1.0, 0.0);
        let cnot = CNOT::new(0, 1);
        b.iter(|| {
            cnot.apply(&mut state).unwrap();
        });
    });
    
    group.finish();
}

fn bench_circuit_execution(c: &mut Criterion) {
    let mut group = c.benchmark_group("circuit_execution");
    
    for n_qubits in [2, 4, 6, 8, 10].iter() {
        group.throughput(Throughput::Elements(*n_qubits as u64));
        
        // Random circuit benchmark
        let circuit = create_random_circuit(*n_qubits, 20);
        
        group.bench_with_input(
            BenchmarkId::new("random_circuit", n_qubits),
            n_qubits,
            |b, _| {
                b.iter(|| {
                    black_box(circuit.execute().unwrap());
                });
            },
        );
        
        // Parameterized circuit benchmark
        let param_circuit = create_parameterized_circuit(*n_qubits, 10);
        let params: Vec<f64> = (0..param_circuit.parameter_count())
            .map(|i| (i as f64) * 0.1)
            .collect();
        
        group.bench_with_input(
            BenchmarkId::new("parameterized_circuit", n_qubits),
            n_qubits,
            |b, _| {
                b.iter(|| {
                    black_box(param_circuit.execute_with_parameters(&params).unwrap());
                });
            },
        );
    }
    
    group.finish();
}

fn bench_variational_circuits(c: &mut Criterion) {
    let mut group = c.benchmark_group("variational_circuits");
    
    for (n_qubits, n_layers) in [(3, 2), (4, 3), (5, 2)].iter() {
        let mut vqc = VariationalCircuit::new(*n_qubits, *n_layers, EntanglementPattern::Linear);
        vqc.build_random().unwrap();
        
        group.bench_with_input(
            BenchmarkId::new("vqc_execution", format!("{}q_{}l", n_qubits, n_layers)),
            &(*n_qubits, *n_layers),
            |b, _| {
                b.iter(|| {
                    black_box(vqc.circuit().execute().unwrap());
                });
            },
        );
    }
    
    group.finish();
}

fn bench_simulation_methods(c: &mut Criterion) {
    let mut group = c.benchmark_group("simulation_methods");
    
    let circuit = CircuitBuilder::new(4)
        .h(0)
        .cnot(0, 1)
        .cnot(1, 2)
        .cnot(2, 3)
        .rx(0, PI / 3.0)
        .ry(1, PI / 4.0)
        .build();
    
    // Single simulation
    group.bench_function("single_simulation", |b| {
        b.iter(|| {
            let mut sim = Simulator::new(4);
            sim.execute_circuit(black_box(&circuit)).unwrap();
        });
    });
    
    // Batch simulation
    let circuits = vec![&circuit; 10];
    group.bench_function("batch_simulation", |b| {
        b.iter(|| {
            let mut batch_sim = BatchSimulator::new(4, 10);
            batch_sim.execute_batch(black_box(circuits.clone())).unwrap();
        });
    });
    
    // Measurement sampling
    group.bench_function("measurement_sampling", |b| {
        let mut sim = Simulator::new(4);
        sim.execute_circuit(&circuit).unwrap();
        
        b.iter(|| {
            sim.sample_measurements(black_box(100)).unwrap();
        });
    });
    
    group.finish();
}

fn bench_quantum_embeddings(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantum_embeddings");
    
    let data_sizes = [4, 8, 16];
    let test_data: Vec<f64> = (0..16).map(|i| (i as f64) / 16.0).collect();
    
    for &size in data_sizes.iter() {
        let data = &test_data[..size];
        
        // Amplitude embedding
        let amp_embedding = AmplitudeEmbedding::new(size, NormalizationMethod::L2);
        group.bench_with_input(
            BenchmarkId::new("amplitude_embedding", size),
            &size,
            |b, _| {
                b.iter(|| {
                    black_box(amp_embedding.embed(black_box(data)).unwrap());
                });
            },
        );
        
        // Angle embedding
        let angle_embedding = AngleEmbedding::new(size);
        group.bench_with_input(
            BenchmarkId::new("angle_embedding", size),
            &size,
            |b, _| {
                b.iter(|| {
                    black_box(angle_embedding.embed(black_box(data)).unwrap());
                });
            },
        );
        
        // Parametric embedding
        let n_qubits = (size as f64).log2().ceil() as usize;
        let param_embedding = ParametricEmbedding::new(size, n_qubits, 2);
        group.bench_with_input(
            BenchmarkId::new("parametric_embedding", size),
            &size,
            |b, _| {
                b.iter(|| {
                    black_box(param_embedding.embed(black_box(data)).unwrap());
                });
            },
        );
    }
    
    group.finish();
}

fn bench_optimization_algorithms(c: &mut Criterion) {
    let mut group = c.benchmark_group("optimization_algorithms");
    group.sample_size(10); // Reduce sample size for slower benchmarks
    
    // QAOA optimization
    let config = OptimizerConfig {
        max_iterations: 20,
        tolerance: 1e-6,
        learning_rate: 0.1,
        verbose: false,
        ..Default::default()
    };
    
    let mut qaoa = QAOAOptimizer::new(config, 2);
    let objective = |params: &[f64]| -> f64 {
        params.iter().map(|x| x * x).sum::<f64>()
    };
    
    group.bench_function("qaoa_optimization", |b| {
        b.iter(|| {
            let initial_params = vec![0.5, 0.3, 0.2, 0.1];
            black_box(qaoa.optimize(black_box(objective), black_box(&initial_params)).unwrap());
        });
    });
    
    group.finish();
}

fn bench_gradient_computation(c: &mut Criterion) {
    let mut group = c.benchmark_group("gradient_computation");

    for n_params in [1, 3, 5, 10].iter() {
        let mut circuit = Circuit::new(2);

        // Add parameterized gates
        for i in 0..*n_params {
            let qubit = i % 2;
            circuit.add_gate(Box::new(RY::new(qubit, PI / 4.0))).unwrap();
        }

        // Create a 4x4 observable for the 2-qubit system (tensor product of Z with I)
        // Z ⊗ I = diag(1, 1, -1, -1)
        let mut observable = ndarray::Array2::zeros((4, 4));
        observable[[0, 0]] = quantum_circuit::Complex::new(1.0, 0.0);
        observable[[1, 1]] = quantum_circuit::Complex::new(1.0, 0.0);
        observable[[2, 2]] = quantum_circuit::Complex::new(-1.0, 0.0);
        observable[[3, 3]] = quantum_circuit::Complex::new(-1.0, 0.0);

        group.bench_with_input(
            BenchmarkId::new("parameter_shift_gradients", n_params),
            n_params,
            |b, _| {
                b.iter(|| {
                    black_box(circuit.parameter_gradients(black_box(&observable)).unwrap());
                });
            },
        );
    }

    group.finish();
}

fn bench_quantum_utils(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantum_utils");

    // Benchmark state operations
    let state1 = quantum_circuit::utils::random_state(3);
    let state2 = quantum_circuit::utils::random_state(3);

    // Create an 8x8 diagonal operator for 3-qubit system (Z ⊗ I ⊗ I)
    let mut operator = ndarray::Array2::zeros((8, 8));
    for i in 0..4 {
        operator[[i, i]] = quantum_circuit::Complex::new(1.0, 0.0);
    }
    for i in 4..8 {
        operator[[i, i]] = quantum_circuit::Complex::new(-1.0, 0.0);
    }

    group.bench_function("fidelity_computation", |b| {
        b.iter(|| {
            black_box(quantum_circuit::utils::fidelity(
                black_box(&state1),
                black_box(&state2)
            ).unwrap());
        });
    });

    group.bench_function("expectation_value", |b| {
        b.iter(|| {
            black_box(quantum_circuit::utils::expectation_value(
                black_box(&state1),
                black_box(&operator)
            ).unwrap());
        });
    });
    
    group.bench_function("random_state_generation", |b| {
        b.iter(|| {
            black_box(quantum_circuit::utils::random_state(black_box(4)));
        });
    });
    
    // Amplitude encoding benchmark
    let classical_data = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
    group.bench_function("amplitude_encoding", |b| {
        b.iter(|| {
            black_box(quantum_circuit::utils::amplitude_encode(black_box(&classical_data)).unwrap());
        });
    });
    
    group.finish();
}

fn bench_memory_usage(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_usage");
    
    // Test memory scaling with qubit count
    for n_qubits in [8, 10, 12].iter() {
        let dim = 1 << n_qubits;
        
        group.bench_with_input(
            BenchmarkId::new("state_allocation", n_qubits),
            n_qubits,
            |b, _| {
                b.iter(|| {
                    let state = ndarray::Array1::<quantum_circuit::Complex>::zeros(black_box(dim));
                    black_box(state);
                });
            },
        );
        
        // Circuit with many gates
        let circuit = create_deep_circuit(*n_qubits, 50);
        group.bench_with_input(
            BenchmarkId::new("deep_circuit_execution", n_qubits),
            n_qubits,
            |b, _| {
                b.iter(|| {
                    black_box(circuit.execute().unwrap());
                });
            },
        );
    }
    
    group.finish();
}

// Helper functions
fn create_random_circuit(n_qubits: usize, n_gates: usize) -> Circuit {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let mut circuit = Circuit::new(n_qubits);
    
    for _ in 0..n_gates {
        match rng.gen_range(0..5) {
            0 => circuit.add_gate(Box::new(Hadamard::new(rng.gen_range(0..n_qubits)))).unwrap(),
            1 => circuit.add_gate(Box::new(PauliX::new(rng.gen_range(0..n_qubits)))).unwrap(),
            2 => circuit.add_gate(Box::new(PauliY::new(rng.gen_range(0..n_qubits)))).unwrap(),
            3 => circuit.add_gate(Box::new(PauliZ::new(rng.gen_range(0..n_qubits)))).unwrap(),
            4 => {
                if n_qubits > 1 {
                    let control = rng.gen_range(0..n_qubits);
                    let target = (control + 1) % n_qubits;
                    circuit.add_gate(Box::new(CNOT::new(control, target))).unwrap();
                }
            },
            _ => unreachable!(),
        }
    }
    
    circuit
}

fn create_parameterized_circuit(n_qubits: usize, n_layers: usize) -> Circuit {
    let mut circuit = Circuit::new(n_qubits);
    
    for layer in 0..n_layers {
        // Parameterized rotations
        for qubit in 0..n_qubits {
            let angle = (layer as f64 + qubit as f64) * PI / 10.0;
            circuit.add_gate(Box::new(RY::new(qubit, angle))).unwrap();
        }
        
        // Entangling gates
        for qubit in 0..n_qubits-1 {
            circuit.add_gate(Box::new(CNOT::new(qubit, qubit + 1))).unwrap();
        }
    }
    
    circuit
}

fn create_deep_circuit(n_qubits: usize, depth: usize) -> Circuit {
    let mut circuit = Circuit::new(n_qubits);
    
    for layer in 0..depth {
        for qubit in 0..n_qubits {
            let angle = (layer as f64) * PI / (depth as f64);
            match layer % 3 {
                0 => circuit.add_gate(Box::new(RX::new(qubit, angle))).unwrap(),
                1 => circuit.add_gate(Box::new(RY::new(qubit, angle))).unwrap(),
                2 => circuit.add_gate(Box::new(RZ::new(qubit, angle))).unwrap(),
                _ => unreachable!(),
            }
        }
        
        // Add entangling layer every few steps
        if layer % 3 == 2 {
            for qubit in 0..n_qubits-1 {
                circuit.add_gate(Box::new(CNOT::new(qubit, qubit + 1))).unwrap();
            }
        }
    }
    
    circuit
}

criterion_group!(
    benches,
    bench_gate_operations,
    bench_circuit_execution,
    bench_variational_circuits,
    bench_simulation_methods,
    bench_quantum_embeddings,
    bench_optimization_algorithms,
    bench_gradient_computation,
    bench_quantum_utils,
    bench_memory_usage,
);

criterion_main!(benches);