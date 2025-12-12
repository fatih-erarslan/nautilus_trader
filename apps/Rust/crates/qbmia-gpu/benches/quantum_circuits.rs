//! Quantum Circuit GPU Acceleration Benchmarks

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use qbmia_gpu::{
    quantum::{GpuQuantumCircuit, gates},
    initialize,
};

fn benchmark_quantum_gates(c: &mut Criterion) {
    // Initialize GPU backend (will fall back to CPU if no GPU)
    let _ = initialize();
    
    let mut group = c.benchmark_group("quantum_gates");
    
    for num_qubits in [5, 10, 15, 20].iter() {
        group.bench_with_input(
            BenchmarkId::new("hadamard_gate", num_qubits),
            num_qubits,
            |b, &num_qubits| {
                b.iter(|| {
                    let mut circuit = GpuQuantumCircuit::new(black_box(num_qubits), 0);
                    circuit.add_gate(gates::h(), 0);
                    
                    // Execute circuit (may fall back to CPU)
                    let result = circuit.execute();
                    black_box(result)
                });
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("cnot_gate", num_qubits),
            num_qubits,
            |b, &num_qubits| {
                b.iter(|| {
                    let mut circuit = GpuQuantumCircuit::new(black_box(num_qubits), 0);
                    if num_qubits >= 2 {
                        circuit.add_two_gate(gates::cnot(), 0, 1);
                    }
                    
                    let result = circuit.execute();
                    black_box(result)
                });
            },
        );
    }
    
    group.finish();
}

fn benchmark_quantum_circuit_depth(c: &mut Criterion) {
    let _ = initialize();
    
    let mut group = c.benchmark_group("quantum_circuit_depth");
    
    for depth in [10, 50, 100, 200].iter() {
        group.bench_with_input(
            BenchmarkId::new("circuit_depth", depth),
            depth,
            |b, &depth| {
                b.iter(|| {
                    let mut circuit = GpuQuantumCircuit::new(10, 0);
                    
                    // Add gates in sequence
                    for i in 0..black_box(depth) {
                        let qubit = i % 10;
                        match i % 4 {
                            0 => circuit.add_gate(gates::h(), qubit),
                            1 => circuit.add_gate(gates::x(), qubit),
                            2 => circuit.add_gate(gates::y(), qubit),
                            3 => circuit.add_gate(gates::z(), qubit),
                            _ => unreachable!(),
                        }
                    }
                    
                    let result = circuit.execute();
                    black_box(result)
                });
            },
        );
    }
    
    group.finish();
}

fn benchmark_quantum_entanglement(c: &mut Criterion) {
    let _ = initialize();
    
    c.bench_function("bell_state_preparation", |b| {
        b.iter(|| {
            let mut circuit = GpuQuantumCircuit::new(black_box(2), 0);
            
            // Create Bell state |00⟩ + |11⟩
            circuit.add_gate(gates::h(), 0);
            circuit.add_two_gate(gates::cnot(), 0, 1);
            
            let result = circuit.execute();
            black_box(result)
        });
    });
    
    c.bench_function("ghz_state_preparation", |b| {
        b.iter(|| {
            let mut circuit = GpuQuantumCircuit::new(black_box(5), 0);
            
            // Create GHZ state
            circuit.add_gate(gates::h(), 0);
            for i in 1..5 {
                circuit.add_two_gate(gates::cnot(), 0, i);
            }
            
            let result = circuit.execute();
            black_box(result)
        });
    });
}

fn benchmark_quantum_algorithms(c: &mut Criterion) {
    let _ = initialize();
    
    c.bench_function("quantum_fourier_transform", |b| {
        b.iter(|| {
            let num_qubits = black_box(8);
            let mut circuit = GpuQuantumCircuit::new(num_qubits, 0);
            
            // Simplified QFT implementation
            for i in 0..num_qubits {
                circuit.add_gate(gates::h(), i);
                for j in (i+1)..num_qubits {
                    // Phase gates would go here in full implementation
                    circuit.add_two_gate(gates::cnot(), i, j);
                }
            }
            
            let result = circuit.execute();
            black_box(result)
        });
    });
}

criterion_group!(
    benches,
    benchmark_quantum_gates,
    benchmark_quantum_circuit_depth,
    benchmark_quantum_entanglement,
    benchmark_quantum_algorithms
);
criterion_main!(benches);