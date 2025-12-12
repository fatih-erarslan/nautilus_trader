//! Cognitive API Benchmarks
//!
//! Benchmarks for all 8 cognitive layers

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use qks_plugin::api::prelude::*;

fn bench_thermodynamic(c: &mut Criterion) {
    c.bench_function("thermodynamic_boltzmann_weight", |b| {
        b.iter(|| {
            black_box(boltzmann_weight(1.0, ISING_CRITICAL_TEMP))
        });
    });

    c.bench_function("thermodynamic_partition_function", |b| {
        let energies = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        b.iter(|| {
            black_box(partition_function(&energies, ISING_CRITICAL_TEMP))
        });
    });
}

fn bench_cognitive(c: &mut Criterion) {
    c.bench_function("cognitive_focus_attention", |b| {
        let saliences = vec![0.5, 0.3, 0.9, 0.2, 0.7];
        b.iter(|| {
            black_box(focus_attention(&saliences).unwrap())
        });
    });

    c.bench_function("cognitive_cosine_similarity", |b| {
        let a = vec![0.5, 0.3, 0.8, 0.2];
        let b = vec![0.4, 0.3, 0.9, 0.1];
        b.iter(|| {
            black_box(cosine_similarity(&a, &b))
        });
    });
}

fn bench_decision(c: &mut Criterion) {
    c.bench_function("decision_belief_entropy", |b| {
        let states = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let belief = BeliefState::uniform(states);
        b.iter(|| {
            black_box(belief.entropy())
        });
    });
}

fn bench_learning(c: &mut Criterion) {
    c.bench_function("learning_stdp", |b| {
        b.iter(|| {
            black_box(apply_stdp(5.0, 1.0, 0.01).unwrap())
        });
    });

    c.bench_function("learning_eligibility_trace", |b| {
        b.iter(|| {
            black_box(eligibility_trace(10.0, FIBONACCI_TAU))
        });
    });
}

fn bench_collective(c: &mut Criterion) {
    c.bench_function("collective_swarm_cohesion", |b| {
        let states = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.1],
            vec![0.2, 0.2],
            vec![0.15, 0.15],
        ];
        b.iter(|| {
            black_box(swarm_cohesion(&states))
        });
    });
}

fn bench_consciousness(c: &mut Criterion) {
    c.bench_function("consciousness_is_conscious", |b| {
        b.iter(|| {
            black_box(is_conscious(1.5))
        });
    });

    c.bench_function("consciousness_level", |b| {
        b.iter(|| {
            black_box(consciousness_level(2.5))
        });
    });
}

fn bench_metacognition(c: &mut Criterion) {
    c.bench_function("metacognition_assess_capability", |b| {
        let model = SelfModel::new();
        b.iter(|| {
            black_box(assess_capability("test_task", &model))
        });
    });
}

fn bench_integration(c: &mut Criterion) {
    c.bench_function("integration_homeostasis_update", |b| {
        let mut state = HomeostasisState::new();
        b.iter(|| {
            state.update(0.1);
            black_box(&state)
        });
    });
}

criterion_group!(
    benches,
    bench_thermodynamic,
    bench_cognitive,
    bench_decision,
    bench_learning,
    bench_collective,
    bench_consciousness,
    bench_metacognition,
    bench_integration
);

criterion_main!(benches);
