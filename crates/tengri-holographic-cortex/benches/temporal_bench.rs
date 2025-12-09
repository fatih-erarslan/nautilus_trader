//! Benchmarks for Phase 7: Temporal Consciousness Fabric
//!
//! Wolfram-verified performance targets:
//! - Time embedding: <100ns
//! - Gamma oscillator step: <50ns
//! - Temporal binding step: <1μs
//! - Memory consolidation update: <10μs for 1000 memories
//! - Temporal attention: <5μs for 100 timestamps
//! - Full fabric step: <50μs

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use rand::SeedableRng;
use rand::rngs::SmallRng;
use tengri_holographic_cortex::{
    HyperbolicTimeEmbedding, GammaOscillator, TemporalBinder, TemporalEvent,
    MemoryConsolidator, SubjectiveTimeClock, TemporalAttention,
    TemporalConsciousnessFabric, TemporalConfig,
    TEMPORAL_TAU, GAMMA_FREQUENCY,
};

fn bench_hyperbolic_time_embedding(c: &mut Criterion) {
    let mut group = c.benchmark_group("hyperbolic_time_embedding");

    group.bench_function("embed", |b| {
        let embedding = HyperbolicTimeEmbedding::new(TEMPORAL_TAU);
        b.iter(|| {
            black_box(embedding.embed(black_box(1000.0)))
        })
    });

    group.bench_function("inverse_embed", |b| {
        let embedding = HyperbolicTimeEmbedding::new(TEMPORAL_TAU);
        b.iter(|| {
            black_box(embedding.inverse_embed(black_box(2.398)))
        })
    });

    group.bench_function("temporal_distance", |b| {
        let embedding = HyperbolicTimeEmbedding::new(TEMPORAL_TAU);
        b.iter(|| {
            black_box(embedding.temporal_distance(black_box(100.0), black_box(1000.0)))
        })
    });

    group.bench_function("multi_scale_embed", |b| {
        let embedding = HyperbolicTimeEmbedding::new(TEMPORAL_TAU);
        b.iter(|| {
            black_box(embedding.multi_scale_embed(black_box(60000.0)))
        })
    });

    group.bench_function("recency_weight", |b| {
        let embedding = HyperbolicTimeEmbedding::new(TEMPORAL_TAU);
        b.iter(|| {
            black_box(embedding.recency_weight(black_box(500.0), black_box(1000.0)))
        })
    });

    group.finish();
}

fn bench_gamma_oscillator(c: &mut Criterion) {
    let mut group = c.benchmark_group("gamma_oscillator");

    group.bench_function("step_1ms", |b| {
        let mut osc = GammaOscillator::new(GAMMA_FREQUENCY);
        b.iter(|| {
            osc.step(black_box(1.0))
        })
    });

    group.bench_function("value", |b| {
        let osc = GammaOscillator::new(GAMMA_FREQUENCY);
        b.iter(|| {
            black_box(osc.value())
        })
    });

    group.bench_function("binding_strength", |b| {
        let osc = GammaOscillator::new(GAMMA_FREQUENCY);
        b.iter(|| {
            black_box(osc.binding_strength(black_box(1.5)))
        })
    });

    group.bench_function("couple", |b| {
        let mut osc = GammaOscillator::new(GAMMA_FREQUENCY);
        b.iter(|| {
            osc.couple(black_box(0.5), black_box(1.0))
        })
    });

    group.finish();
}

fn bench_temporal_binder(c: &mut Criterion) {
    let mut group = c.benchmark_group("temporal_binder");

    group.bench_function("step", |b| {
        let mut binder = TemporalBinder::new(50.0);
        let mut t = 0.0;
        b.iter(|| {
            t += 1.0;
            binder.step(black_box(1.0), black_box(t))
        })
    });

    group.bench_function("add_event", |b| {
        let mut binder = TemporalBinder::new(50.0);
        let mut id = 0u64;
        b.iter(|| {
            id += 1;
            binder.add_event(TemporalEvent {
                timestamp: black_box(id as f64),
                id: black_box(id),
                strength: black_box(0.8),
            })
        })
    });

    group.bench_function("step_with_events", |b| {
        let mut binder = TemporalBinder::new(50.0);
        let mut t = 0.0;
        b.iter(|| {
            t += 1.0;
            binder.add_event(TemporalEvent {
                timestamp: t,
                id: t as u64,
                strength: 0.8,
            });
            binder.step(black_box(1.0), black_box(t))
        })
    });

    group.finish();
}

fn bench_memory_consolidator(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_consolidator");

    group.bench_function("encode", |b| {
        let mut consolidator = MemoryConsolidator::default();
        let mut id = 0u64;
        b.iter(|| {
            id += 1;
            consolidator.encode(black_box(id), black_box(0.8), black_box(id as f64))
        })
    });

    group.bench_function("access", |b| {
        let mut consolidator = MemoryConsolidator::default();
        for i in 0..100 {
            consolidator.encode(i, 0.8, i as f64);
        }
        let mut t = 100.0;
        b.iter(|| {
            t += 1.0;
            black_box(consolidator.access(black_box(50), black_box(t)))
        })
    });

    for num_memories in [100, 500, 1000].iter() {
        group.bench_with_input(BenchmarkId::new("update", num_memories), num_memories, |b, &n| {
            let mut consolidator = MemoryConsolidator::default();
            for i in 0..n {
                consolidator.encode(i as u64, 0.8, i as f64);
                if i % 3 == 0 {
                    consolidator.access(i as u64, (i + 10) as f64);
                }
            }
            let mut t = (n + 100) as f64;
            b.iter(|| {
                t += 10.0;
                consolidator.update(black_box(t))
            })
        });
    }

    group.bench_function("replay_10", |b| {
        let mut consolidator = MemoryConsolidator::default();
        for i in 0..100 {
            consolidator.encode(i, 0.8, i as f64);
        }
        let mut t = 200.0;
        b.iter(|| {
            t += 1.0;
            consolidator.replay(black_box(t), black_box(10))
        })
    });

    group.finish();
}

fn bench_subjective_time(c: &mut Criterion) {
    let mut group = c.benchmark_group("subjective_time");

    group.bench_function("step", |b| {
        let mut clock = SubjectiveTimeClock::default();
        b.iter(|| {
            clock.step(black_box(1.0))
        })
    });

    group.bench_function("dilation", |b| {
        let mut clock = SubjectiveTimeClock::default();
        clock.step(100.0);
        b.iter(|| {
            black_box(clock.dilation())
        })
    });

    group.bench_function("in_flow_state", |b| {
        let mut clock = SubjectiveTimeClock::new(0.5, 1.0);
        clock.step(100.0);
        b.iter(|| {
            black_box(clock.in_flow_state())
        })
    });

    group.finish();
}

fn bench_temporal_attention(c: &mut Criterion) {
    let mut group = c.benchmark_group("temporal_attention");

    for num_times in [10, 50, 100].iter() {
        group.bench_with_input(BenchmarkId::new("attention_weights", num_times), num_times, |b, &n| {
            let attention = TemporalAttention::default();
            let key_times: Vec<f64> = (0..n).map(|i| (i * 10) as f64).collect();
            b.iter(|| {
                black_box(attention.attention_weights(black_box(1000.0), black_box(&key_times)))
            })
        });
    }

    group.bench_function("attend", |b| {
        let mut attention = TemporalAttention::default();
        let mut t = 0.0;
        b.iter(|| {
            t += 1.0;
            attention.attend(black_box(t), black_box(0.8))
        })
    });

    group.bench_function("weighted_average_100", |b| {
        let mut attention = TemporalAttention::default();
        for i in 0..100 {
            attention.attend(i as f64 * 10.0, 0.8);
        }
        b.iter(|| {
            black_box(attention.weighted_average(black_box(1000.0)))
        })
    });

    group.finish();
}

fn bench_temporal_fabric(c: &mut Criterion) {
    let mut group = c.benchmark_group("temporal_fabric");

    group.bench_function("step", |b| {
        let config = TemporalConfig::default();
        let mut fabric = TemporalConsciousnessFabric::new(config);
        b.iter(|| {
            fabric.step(black_box(1.0))
        })
    });

    group.bench_function("process_event", |b| {
        let config = TemporalConfig::default();
        let mut fabric = TemporalConsciousnessFabric::new(config);
        let mut id = 0u64;
        b.iter(|| {
            id += 1;
            fabric.step(1.0);
            fabric.process_event(black_box(id), black_box(0.8))
        })
    });

    group.bench_function("embedded_time", |b| {
        let config = TemporalConfig::default();
        let mut fabric = TemporalConsciousnessFabric::new(config);
        fabric.step(1000.0);
        b.iter(|| {
            black_box(fabric.embedded_time())
        })
    });

    group.bench_function("multi_scale_time", |b| {
        let config = TemporalConfig::default();
        let mut fabric = TemporalConsciousnessFabric::new(config);
        fabric.step(60000.0);
        b.iter(|| {
            black_box(fabric.multi_scale_time())
        })
    });

    group.bench_function("replay_10", |b| {
        let config = TemporalConfig::default();
        let mut fabric = TemporalConsciousnessFabric::new(config);
        for i in 0..100 {
            fabric.step(1.0);
            if i % 5 == 0 {
                fabric.process_event(i as u64, 0.8);
            }
        }
        b.iter(|| {
            fabric.replay(black_box(10))
        })
    });

    group.bench_function("stats", |b| {
        let config = TemporalConfig::default();
        let mut fabric = TemporalConsciousnessFabric::new(config);
        for i in 0..100 {
            fabric.step(1.0);
            if i % 10 == 0 {
                fabric.process_event(i as u64, 0.8);
            }
        }
        b.iter(|| {
            black_box(fabric.stats())
        })
    });

    group.finish();
}

fn bench_full_simulation(c: &mut Criterion) {
    let mut group = c.benchmark_group("full_temporal_simulation");

    group.bench_function("run_1000_steps", |b| {
        b.iter(|| {
            let config = TemporalConfig::default();
            let mut fabric = TemporalConsciousnessFabric::new(config);
            let mut rng = SmallRng::seed_from_u64(42);

            for i in 0..1000 {
                fabric.step(1.0);
                // Random events
                if i % 25 == 0 {
                    fabric.process_event(i as u64, 0.8);
                }
                // Periodic replay
                if i % 100 == 0 {
                    fabric.replay(5);
                }
            }
            black_box(fabric.stats())
        })
    });

    group.bench_function("run_1000_steps_high_event_rate", |b| {
        b.iter(|| {
            let config = TemporalConfig::default();
            let mut fabric = TemporalConsciousnessFabric::new(config);

            for i in 0..1000 {
                fabric.step(1.0);
                // High event rate (every 5 steps)
                if i % 5 == 0 {
                    fabric.process_event(i as u64, 0.8);
                }
            }
            black_box(fabric.stats())
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_hyperbolic_time_embedding,
    bench_gamma_oscillator,
    bench_temporal_binder,
    bench_memory_consolidator,
    bench_subjective_time,
    bench_temporal_attention,
    bench_temporal_fabric,
    bench_full_simulation,
);

criterion_main!(benches);
