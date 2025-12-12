//! Quantum State Processing Benchmarks
//! 
//! Benchmarks for quantum state manipulation and processing with GPU acceleration

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use qbmia_acceleration::quantum::QuantumProcessor;
use qbmia_acceleration::types::QuantumState;
use std::time::Duration;

fn bench_quantum_state_creation(c: &mut Criterion) {
    let processor = QuantumProcessor::new();
    
    c.bench_function("quantum_state_4_qubits", |b| {
        b.iter(|| {
            let _state = processor.create_quantum_state(black_box(4));
        })
    });
    
    c.bench_function("quantum_state_8_qubits", |b| {
        b.iter(|| {
            let _state = processor.create_quantum_state(black_box(8));
        })
    });
}

fn bench_quantum_state_operations(c: &mut Criterion) {
    let processor = QuantumProcessor::new();
    let state = processor.create_quantum_state(4);
    
    c.bench_function("quantum_hadamard_gate", |b| {
        b.iter(|| {
            let _result = processor.apply_hadamard_gate(black_box(&state), black_box(0));
        })
    });
    
    c.bench_function("quantum_cnot_gate", |b| {
        b.iter(|| {
            let _result = processor.apply_cnot_gate(black_box(&state), black_box(0), black_box(1));
        })
    });
}

fn bench_quantum_measurement(c: &mut Criterion) {
    let processor = QuantumProcessor::new();
    let state = processor.create_quantum_state(4);
    
    c.bench_function("quantum_measurement", |b| {
        b.iter(|| {
            let _result = processor.measure_quantum_state(black_box(&state));
        })
    });
}

fn bench_large_quantum_states(c: &mut Criterion) {
    let processor = QuantumProcessor::new();
    
    let mut group = c.benchmark_group("large_quantum_states");
    group.measurement_time(Duration::from_secs(20));
    
    group.bench_function("quantum_state_16_qubits", |b| {
        b.iter(|| {
            let _state = processor.create_quantum_state(black_box(16));
        })
    });
    
    group.bench_function("quantum_state_32_qubits", |b| {
        b.iter(|| {
            let _state = processor.create_quantum_state(black_box(32));
        })
    });
    
    group.finish();
}

criterion_group!(benches, 
    bench_quantum_state_creation, 
    bench_quantum_state_operations, 
    bench_quantum_measurement,
    bench_large_quantum_states
);
criterion_main!(benches);