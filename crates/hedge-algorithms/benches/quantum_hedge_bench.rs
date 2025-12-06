use criterion::{black_box, criterion_group, criterion_main, Criterion};
use hedge_algorithms::*;
use std::collections::HashMap;

fn benchmark_quantum_operations(c: &mut Criterion) {
    let config = HedgeConfig::default();
    let expert_names = vec!["expert1".to_string(), "expert2".to_string(), "expert3".to_string(), "expert4".to_string()];
    let mut quantum_hedge = QuantumHedgeAlgorithm::new(expert_names, config).unwrap();
    
    let market_data = MarketData::new(
        "BTCUSD".to_string(),
        chrono::Utc::now(),
        [100.0, 105.0, 95.0, 102.0, 1000.0]
    );
    
    let mut predictions = HashMap::new();
    predictions.insert("expert1".to_string(), 0.05);
    predictions.insert("expert2".to_string(), -0.02);
    predictions.insert("expert3".to_string(), 0.03);
    predictions.insert("expert4".to_string(), -0.01);
    
    c.bench_function("quantum_state_update", |b| {
        b.iter(|| {
            quantum_hedge.update(black_box(&market_data), black_box(&predictions)).unwrap();
        })
    });
    
    c.bench_function("quantum_measurement", |b| {
        b.iter(|| {
            quantum_hedge.measure().unwrap();
        })
    });
    
    c.bench_function("quantum_entropy", |b| {
        b.iter(|| {
            quantum_hedge.get_entropy();
        })
    });
    
    c.bench_function("quantum_purity", |b| {
        b.iter(|| {
            quantum_hedge.get_purity();
        })
    });
}

fn benchmark_quantum_gates(c: &mut Criterion) {
    let labels = vec!["q1".to_string(), "q2".to_string(), "q3".to_string(), "q4".to_string()];
    let mut quantum_state = QuantumState::new(labels, 0.01);
    
    let hadamard = QuantumGate::hadamard(1);
    let rotation = QuantumGate::rotation(std::f64::consts::PI / 4.0, 1);
    let phase = QuantumGate::phase(std::f64::consts::PI / 8.0, 1);
    
    c.bench_function("quantum_gate_hadamard", |b| {
        b.iter(|| {
            quantum_state.apply_gate(black_box(&hadamard)).unwrap();
        })
    });
    
    c.bench_function("quantum_gate_rotation", |b| {
        b.iter(|| {
            quantum_state.apply_gate(black_box(&rotation)).unwrap();
        })
    });
    
    c.bench_function("quantum_gate_phase", |b| {
        b.iter(|| {
            quantum_state.apply_gate(black_box(&phase)).unwrap();
        })
    });
    
    c.bench_function("quantum_normalize", |b| {
        b.iter(|| {
            quantum_state.normalize().unwrap();
        })
    });
    
    c.bench_function("quantum_decoherence", |b| {
        b.iter(|| {
            quantum_state.apply_decoherence(black_box(0.01)).unwrap();
        })
    });
}

fn benchmark_quantum_annealing(c: &mut Criterion) {
    let mut annealer = QuantumAnnealer::new(2);
    let labels = vec!["q1".to_string(), "q2".to_string()];
    let mut quantum_state = QuantumState::new(labels, 0.01);
    
    c.bench_function("quantum_annealing", |b| {
        b.iter(|| {
            annealer.anneal(black_box(&mut quantum_state)).unwrap();
        })
    });
}

fn benchmark_quantum_error_correction(c: &mut Criterion) {
    let mut error_correction = QuantumErrorCorrection::new();
    let labels = vec!["q1".to_string(), "q2".to_string()];
    let mut quantum_state = QuantumState::new(labels, 0.01);
    
    c.bench_function("quantum_error_detection", |b| {
        b.iter(|| {
            error_correction.detect_errors(black_box(&quantum_state)).unwrap();
        })
    });
    
    c.bench_function("quantum_error_correction", |b| {
        b.iter(|| {
            error_correction.correct_errors(black_box(&mut quantum_state)).unwrap();
        })
    });
}

fn benchmark_quantum_scaling(c: &mut Criterion) {
    let config = HedgeConfig::default();
    
    // Test with different numbers of experts
    let expert_counts = vec![2, 4, 8, 16];
    
    for &count in &expert_counts {
        let expert_names: Vec<String> = (0..count).map(|i| format!("expert{}", i)).collect();
        let mut quantum_hedge = QuantumHedgeAlgorithm::new(expert_names.clone(), config.clone()).unwrap();
        
        let market_data = MarketData::new(
            "BTCUSD".to_string(),
            chrono::Utc::now(),
            [100.0, 105.0, 95.0, 102.0, 1000.0]
        );
        
        let mut predictions = HashMap::new();
        for (i, name) in expert_names.iter().enumerate() {
            predictions.insert(name.clone(), (i as f64 - count as f64 / 2.0) * 0.01);
        }
        
        c.bench_function(&format!("quantum_hedge_experts_{}", count), |b| {
            b.iter(|| {
                quantum_hedge.update(black_box(&market_data), black_box(&predictions)).unwrap();
            })
        });
    }
}

criterion_group!(
    quantum_benches,
    benchmark_quantum_operations,
    benchmark_quantum_gates,
    benchmark_quantum_annealing,
    benchmark_quantum_error_correction,
    benchmark_quantum_scaling
);
criterion_main!(quantum_benches);