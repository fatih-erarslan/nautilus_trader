// Prospect Theory Benchmarks for Quantum-Enhanced Framework
// Copyright (c) 2025 TENGRI Trading Swarm

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use quantum_prospect_theory::{
    ProspectTheoryEngine, QuantumProspectEngine, DecisionMaker, UtilityFunction,
    ReferencePoint, CumulativeWeights, ProspectValue, LossAversion, RiskProfile
};

fn create_sample_outcomes(size: usize) -> Vec<f64> {
    (0..size).map(|i| (i as f64 - size as f64 / 2.0) * 100.0).collect()
}

fn create_sample_probabilities(size: usize) -> Vec<f64> {
    (0..size).map(|i| 1.0 / size as f64).collect()
}

fn benchmark_utility_calculation(c: &mut Criterion) {
    let mut group = c.benchmark_group("utility_calculation");
    
    for size in [10, 50, 100, 500, 1000].iter() {
        let outcomes = create_sample_outcomes(*size);
        let engine = ProspectTheoryEngine::new();
        
        group.bench_with_input(BenchmarkId::new("calculate_utility", size), size, |b, _| {
            b.iter(|| {
                engine.calculate_utility(&outcomes)
            })
        });
    }
    group.finish();
}

fn benchmark_quantum_prospect_calculation(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantum_prospect_calculation");
    
    for size in [10, 50, 100, 500].iter() {
        let outcomes = create_sample_outcomes(*size);
        let probabilities = create_sample_probabilities(*size);
        let engine = QuantumProspectEngine::new();
        
        group.bench_with_input(BenchmarkId::new("quantum_prospect_value", size), size, |b, _| {
            b.iter(|| {
                engine.calculate_quantum_prospect_value(&outcomes, &probabilities)
            })
        });
    }
    group.finish();
}

fn benchmark_reference_point_update(c: &mut Criterion) {
    let mut group = c.benchmark_group("reference_point_update");
    
    let outcomes = create_sample_outcomes(1000);
    let engine = ProspectTheoryEngine::new();
    
    group.bench_function("update_reference_point", |b| {
        b.iter(|| {
            engine.update_reference_point(&outcomes)
        })
    });
    
    group.finish();
}

fn benchmark_loss_aversion_calculation(c: &mut Criterion) {
    let mut group = c.benchmark_group("loss_aversion_calculation");
    
    for lambda in [1.5, 2.0, 2.5, 3.0].iter() {
        let outcomes = create_sample_outcomes(1000);
        let engine = ProspectTheoryEngine::with_loss_aversion(*lambda);
        
        group.bench_with_input(BenchmarkId::new("loss_aversion", lambda), lambda, |b, _| {
            b.iter(|| {
                engine.calculate_loss_aversion(&outcomes)
            })
        });
    }
    group.finish();
}

fn benchmark_cumulative_weights(c: &mut Criterion) {
    let mut group = c.benchmark_group("cumulative_weights");
    
    for size in [10, 50, 100, 500].iter() {
        let probabilities = create_sample_probabilities(*size);
        let engine = ProspectTheoryEngine::new();
        
        group.bench_with_input(BenchmarkId::new("calculate_weights", size), size, |b, _| {
            b.iter(|| {
                engine.calculate_cumulative_weights(&probabilities)
            })
        });
    }
    group.finish();
}

fn benchmark_decision_making(c: &mut Criterion) {
    let mut group = c.benchmark_group("decision_making");
    
    let decision_maker = DecisionMaker::new();
    
    for choice_count in [2, 5, 10, 20].iter() {
        let outcomes = (0..*choice_count).map(|_| create_sample_outcomes(100)).collect::<Vec<_>>();
        let probabilities = (0..*choice_count).map(|_| create_sample_probabilities(100)).collect::<Vec<_>>();
        
        group.bench_with_input(BenchmarkId::new("make_decision", choice_count), choice_count, |b, _| {
            b.iter(|| {
                decision_maker.make_decision(&outcomes, &probabilities)
            })
        });
    }
    group.finish();
}

fn benchmark_risk_profile_analysis(c: &mut Criterion) {
    let mut group = c.benchmark_group("risk_profile_analysis");
    
    let outcomes = create_sample_outcomes(1000);
    let probabilities = create_sample_probabilities(1000);
    let engine = ProspectTheoryEngine::new();
    
    group.bench_function("analyze_risk_profile", |b| {
        b.iter(|| {
            engine.analyze_risk_profile(&outcomes, &probabilities)
        })
    });
    
    group.bench_function("calculate_certainty_equivalent", |b| {
        b.iter(|| {
            engine.calculate_certainty_equivalent(&outcomes, &probabilities)
        })
    });
    
    group.finish();
}

fn benchmark_quantum_enhancement(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantum_enhancement");
    
    let outcomes = create_sample_outcomes(500);
    let probabilities = create_sample_probabilities(500);
    let quantum_engine = QuantumProspectEngine::new();
    
    group.bench_function("quantum_superposition", |b| {
        b.iter(|| {
            quantum_engine.create_superposition(&outcomes, &probabilities)
        })
    });
    
    group.bench_function("quantum_entanglement", |b| {
        b.iter(|| {
            quantum_engine.apply_entanglement(&outcomes)
        })
    });
    
    group.bench_function("quantum_measurement", |b| {
        b.iter(|| {
            quantum_engine.measure_quantum_state(&outcomes, &probabilities)
        })
    });
    
    group.finish();
}

criterion_group!(
    benches,
    benchmark_utility_calculation,
    benchmark_quantum_prospect_calculation,
    benchmark_reference_point_update,
    benchmark_loss_aversion_calculation,
    benchmark_cumulative_weights,
    benchmark_decision_making,
    benchmark_risk_profile_analysis,
    benchmark_quantum_enhancement
);
criterion_main!(benches);