//! Benchmark for Thermodynamic Scheduler

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use hyperphysics_thermodynamic_scheduler::{ThermodynamicScheduler, SchedulerConfig};

fn bench_scheduler_step(c: &mut Criterion) {
    let mut scheduler = ThermodynamicScheduler::hyperphysics_default();

    c.bench_function("scheduler_step", |b| {
        b.iter(|| {
            black_box(scheduler.step(0.1, 0.5))
        })
    });
}

fn bench_phase_classification(c: &mut Criterion) {
    use hyperphysics_thermodynamic_scheduler::ThermodynamicState;

    c.bench_function("phase_classification", |b| {
        b.iter(|| {
            black_box(ThermodynamicState::classify_phase(2.2))
        })
    });
}

fn bench_pbit_activation(c: &mut Criterion) {
    let scheduler = ThermodynamicScheduler::new(SchedulerConfig {
        t0: 0.15,
        ..Default::default()
    });

    c.bench_function("pbit_activation", |b| {
        b.iter(|| {
            black_box(scheduler.pbit_activation_probability(0.5, 0.0))
        })
    });
}

criterion_group!(benches, bench_scheduler_step, bench_phase_classification, bench_pbit_activation);
criterion_main!(benches);
