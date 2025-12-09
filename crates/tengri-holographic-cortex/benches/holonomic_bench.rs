//! Benchmarks for Phase 9: Holonomic Memory Integration
//!
//! Wolfram-verified performance targets:
//! - Weber-Fechner encoding: <10ns
//! - Hyperbolic distance H^11: <100ns
//! - Working memory push: <50ns
//! - STM encode: <100ns
//! - LTM query (k-NN): <10μs for 1000 traces
//! - Hopfield energy: <1μs for 100 patterns
//! - Pattern completion: <100μs for 50 patterns
//! - Path integral action: <1μs
//! - Full recall: <50μs

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use std::collections::HashMap;
use tengri_holographic_cortex::{
    HolonomicConfig, HolonomicMemory, HyperbolicHopfield, HyperbolicMemoryOps,
    LongTermMemory, HolonomicTrace as MemoryTrace, PathIntegralRecall, RecallGating,
    ShortTermMemory, WorkingMemory, EMBEDDING_DIM, HOPFIELD_KAPPA, HOPFIELD_TEMPERATURE,
    PATTERN_COMPLETION_ETA, WM_CAPACITY, WM_DECAY_TAU,
};
use tengri_holographic_cortex::holonomic_memory::{STM_DECAY_TAU, LTM_DECAY_TAU};

fn bench_weber_fechner(c: &mut Criterion) {
    let mut group = c.benchmark_group("weber_fechner");

    group.bench_function("encode", |b| {
        b.iter(|| black_box(MemoryTrace::weber_fechner_encode(black_box(0.5))))
    });

    group.bench_function("decode", |b| {
        b.iter(|| black_box(MemoryTrace::weber_fechner_decode(black_box(1.7))))
    });

    group.finish();
}

fn bench_hyperbolic_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("hyperbolic_ops");

    let mut x = [0.0; EMBEDDING_DIM];
    x[0] = 1.0;

    let mut y = [0.0; EMBEDDING_DIM];
    y[0] = 1.1;
    y[1] = 0.458;

    group.bench_function("lorentz_inner", |b| {
        b.iter(|| black_box(HyperbolicMemoryOps::lorentz_inner(black_box(&x), black_box(&y))))
    });

    group.bench_function("hyperbolic_distance", |b| {
        b.iter(|| {
            black_box(HyperbolicMemoryOps::hyperbolic_distance(
                black_box(&x),
                black_box(&y),
            ))
        })
    });

    group.bench_function("lift_to_hyperboloid", |b| {
        let euclidean = [0.5, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        b.iter(|| black_box(HyperbolicMemoryOps::lift_to_hyperboloid(black_box(&euclidean))))
    });

    group.bench_function("exp_map", |b| {
        let mut v = [0.0; EMBEDDING_DIM];
        v[1] = 0.1;
        b.iter(|| black_box(HyperbolicMemoryOps::exp_map(black_box(&x), black_box(&v))))
    });

    group.bench_function("log_map", |b| {
        b.iter(|| black_box(HyperbolicMemoryOps::log_map(black_box(&x), black_box(&y))))
    });

    group.bench_function("hyperbolic_similarity", |b| {
        b.iter(|| {
            black_box(HyperbolicMemoryOps::hyperbolic_similarity(
                black_box(&x),
                black_box(&y),
                black_box(1.0),
            ))
        })
    });

    group.bench_function("mobius_add", |b| {
        let x_vec = vec![0.1, 0.2, 0.1];
        let y_vec = vec![0.1, -0.1, 0.05];
        b.iter(|| {
            black_box(HyperbolicMemoryOps::mobius_add(
                black_box(&x_vec),
                black_box(&y_vec),
                black_box(-1.0),
            ))
        })
    });

    group.finish();
}

fn bench_working_memory(c: &mut Criterion) {
    let mut group = c.benchmark_group("working_memory");

    group.bench_function("push", |b| {
        let mut wm = WorkingMemory::new(WM_CAPACITY, WM_DECAY_TAU);
        let mut id = 0u64;
        b.iter(|| {
            id += 1;
            let trace = MemoryTrace::new(id, vec![1.0, 2.0, 3.0], id as f64);
            black_box(wm.push(trace))
        })
    });

    group.bench_function("get", |b| {
        let mut wm = WorkingMemory::new(WM_CAPACITY, WM_DECAY_TAU);
        for i in 0..WM_CAPACITY {
            let trace = MemoryTrace::new(i as u64, vec![i as f64], 0.0);
            wm.push(trace);
        }
        b.iter(|| black_box(wm.get(black_box(3))))
    });

    group.bench_function("step", |b| {
        let mut wm = WorkingMemory::new(WM_CAPACITY, WM_DECAY_TAU);
        for i in 0..WM_CAPACITY {
            let trace = MemoryTrace::new(i as u64, vec![i as f64], 0.0);
            wm.push(trace);
        }
        b.iter(|| wm.step(black_box(1.0)))
    });

    group.bench_function("stats", |b| {
        let mut wm = WorkingMemory::new(WM_CAPACITY, WM_DECAY_TAU);
        for i in 0..WM_CAPACITY {
            let trace = MemoryTrace::new(i as u64, vec![i as f64], 0.0);
            wm.push(trace);
        }
        b.iter(|| black_box(wm.stats()))
    });

    group.finish();
}

fn bench_short_term_memory(c: &mut Criterion) {
    let mut group = c.benchmark_group("short_term_memory");

    group.bench_function("encode", |b| {
        let mut stm = ShortTermMemory::new(100, STM_DECAY_TAU);
        let mut id = 0u64;
        b.iter(|| {
            id += 1;
            let trace = MemoryTrace::new(id, vec![1.0, 2.0], id as f64);
            stm.encode(trace)
        })
    });

    group.bench_function("access", |b| {
        let mut stm = ShortTermMemory::new(100, STM_DECAY_TAU);
        for i in 0..50 {
            let trace = MemoryTrace::new(i, vec![i as f64], 0.0);
            stm.encode(trace);
        }
        b.iter(|| black_box(stm.access(black_box(25), black_box(0.1)).is_some()))
    });

    group.bench_function("query", |b| {
        let mut stm = ShortTermMemory::new(100, STM_DECAY_TAU);
        for i in 0..50 {
            let trace = MemoryTrace::new(i, vec![i as f64, (i * 2) as f64], 0.0);
            stm.encode(trace);
        }
        let query = vec![25.0, 50.0];
        b.iter(|| black_box(stm.query(black_box(&query), black_box(0.7))))
    });

    group.bench_function("step", |b| {
        let mut stm = ShortTermMemory::new(100, STM_DECAY_TAU);
        for i in 0..50 {
            let trace = MemoryTrace::new(i, vec![i as f64], 0.0);
            stm.encode(trace);
        }
        b.iter(|| black_box(stm.step(black_box(1.0))))
    });

    group.finish();
}

fn bench_long_term_memory(c: &mut Criterion) {
    let mut group = c.benchmark_group("long_term_memory");

    for num_traces in [100, 500, 1000].iter() {
        group.bench_with_input(
            BenchmarkId::new("query_hyperbolic", num_traces),
            num_traces,
            |b, &n| {
                let mut ltm = LongTermMemory::default();
                let mut rng = SmallRng::seed_from_u64(42);

                for i in 0..n {
                    let pattern: Vec<f64> = (0..10).map(|_| rng.gen_range(-1.0..1.0)).collect();
                    let embedding =
                        HyperbolicMemoryOps::lift_to_hyperboloid(&pattern[..EMBEDDING_DIM - 1]);
                    let trace = MemoryTrace::with_embedding(i as u64, pattern, embedding, 0.0);
                    ltm.consolidate(trace, 1.0, 2.0);
                }

                let query_embedding = HyperbolicMemoryOps::lift_to_hyperboloid(&[0.5; 11]);

                b.iter(|| {
                    black_box(ltm.query_hyperbolic(
                        black_box(&query_embedding),
                        black_box(10),
                        black_box(0.01),
                    ))
                })
            },
        );
    }

    group.bench_function("consolidate", |b| {
        let mut ltm = LongTermMemory::default();
        let mut id = 0u64;
        b.iter(|| {
            id += 1;
            let trace = MemoryTrace::new(id, vec![1.0, 2.0], id as f64);
            ltm.consolidate(trace, black_box(1.0), black_box(2.0))
        })
    });

    group.bench_function("step_1000", |b| {
        let mut ltm = LongTermMemory::default();
        for i in 0..1000 {
            let trace = MemoryTrace::new(i, vec![i as f64], 0.0);
            ltm.consolidate(trace, 1.0, 2.0);
        }
        b.iter(|| ltm.step(black_box(10.0)))
    });

    group.bench_function("replay_10", |b| {
        let mut ltm = LongTermMemory::default();
        for i in 0..100 {
            let trace = MemoryTrace::new(i, vec![i as f64], 0.0);
            ltm.consolidate(trace, 1.0, 2.0);
        }
        let ids: Vec<u64> = (0..10).collect();
        b.iter(|| ltm.replay(black_box(&ids), black_box(0.1)))
    });

    group.finish();
}

fn bench_hopfield(c: &mut Criterion) {
    let mut group = c.benchmark_group("hopfield");

    group.bench_function("store_pattern", |b| {
        let mut hopfield = HyperbolicHopfield::new(HOPFIELD_TEMPERATURE, HOPFIELD_KAPPA, PATTERN_COMPLETION_ETA);
        let mut id = 0u64;
        b.iter(|| {
            id += 1;
            let mut embedding = [0.0; EMBEDDING_DIM];
            embedding[0] = 1.0;
            embedding[1] = (id as f64 * 0.1).sin();
            HyperbolicMemoryOps::project_to_hyperboloid(&mut embedding);
            hopfield.store_pattern(black_box(id), black_box(embedding))
        })
    });

    for num_patterns in [10, 50, 100].iter() {
        group.bench_with_input(
            BenchmarkId::new("energy", num_patterns),
            num_patterns,
            |b, &n| {
                let mut hopfield = HyperbolicHopfield::new(HOPFIELD_TEMPERATURE, HOPFIELD_KAPPA, PATTERN_COMPLETION_ETA);
                let mut rng = SmallRng::seed_from_u64(42);

                for i in 0..n {
                    let mut embedding = [0.0; EMBEDDING_DIM];
                    embedding[0] = 1.0;
                    for j in 1..EMBEDDING_DIM {
                        embedding[j] = rng.gen_range(-0.3..0.3);
                    }
                    HyperbolicMemoryOps::project_to_hyperboloid(&mut embedding);
                    hopfield.store_pattern(i as u64, embedding);
                }

                let mut state: HashMap<u64, [f64; EMBEDDING_DIM]> = HashMap::new();
                for i in 0..n {
                    if let Some(p) = hopfield.get_pattern(i as u64) {
                        state.insert(i as u64, *p);
                    }
                }

                b.iter(|| black_box(hopfield.energy(black_box(&state))))
            },
        );

        group.bench_with_input(
            BenchmarkId::new("complete_pattern", num_patterns),
            num_patterns,
            |b, &n| {
                let mut hopfield = HyperbolicHopfield::new(HOPFIELD_TEMPERATURE, HOPFIELD_KAPPA, PATTERN_COMPLETION_ETA);
                let mut rng = SmallRng::seed_from_u64(42);

                for i in 0..n {
                    let mut embedding = [0.0; EMBEDDING_DIM];
                    embedding[0] = 1.0;
                    for j in 1..EMBEDDING_DIM {
                        embedding[j] = rng.gen_range(-0.3..0.3);
                    }
                    HyperbolicMemoryOps::project_to_hyperboloid(&mut embedding);
                    hopfield.store_pattern(i as u64, embedding);
                }

                // Partial state (clamp half the patterns)
                let mut partial: HashMap<u64, [f64; EMBEDDING_DIM]> = HashMap::new();
                let clamped: Vec<u64> = (0..n / 2).map(|i| i as u64).collect();
                for &id in &clamped {
                    if let Some(p) = hopfield.get_pattern(id) {
                        partial.insert(id, *p);
                    }
                }

                b.iter(|| {
                    black_box(hopfield.complete_pattern(black_box(&partial), black_box(&clamped)))
                })
            },
        );
    }

    group.bench_function("nearest_pattern_100", |b| {
        let mut hopfield = HyperbolicHopfield::new(HOPFIELD_TEMPERATURE, HOPFIELD_KAPPA, PATTERN_COMPLETION_ETA);
        let mut rng = SmallRng::seed_from_u64(42);

        for i in 0..100 {
            let mut embedding = [0.0; EMBEDDING_DIM];
            embedding[0] = 1.0;
            for j in 1..EMBEDDING_DIM {
                embedding[j] = rng.gen_range(-0.3..0.3);
            }
            HyperbolicMemoryOps::project_to_hyperboloid(&mut embedding);
            hopfield.store_pattern(i, embedding);
        }

        let mut query = [0.0; EMBEDDING_DIM];
        query[0] = 1.0;
        query[1] = 0.2;
        HyperbolicMemoryOps::project_to_hyperboloid(&mut query);

        b.iter(|| black_box(hopfield.nearest_pattern(black_box(&query))))
    });

    group.finish();
}

fn bench_path_integral(c: &mut Criterion) {
    let mut group = c.benchmark_group("path_integral");

    let mut start = [0.0; EMBEDDING_DIM];
    start[0] = 1.0;

    let mut target = [0.0; EMBEDDING_DIM];
    target[0] = 1.0;
    target[1] = 0.5;
    HyperbolicMemoryOps::project_to_hyperboloid(&mut target);

    group.bench_function("generate_path", |b| {
        let recall = PathIntegralRecall::new(1.0, 0.1, 10);
        b.iter(|| black_box(recall.generate_path(black_box(&start), black_box(&target))))
    });

    group.bench_function("compute_action", |b| {
        let recall = PathIntegralRecall::new(1.0, 0.1, 10);
        let path = recall.generate_path(&start, &target);
        b.iter(|| black_box(recall.compute_action(black_box(&path), black_box(&target))))
    });

    group.bench_function("recall_probability", |b| {
        let recall = PathIntegralRecall::new(1.0, 0.1, 10);
        b.iter(|| black_box(recall.recall_probability(black_box(0.5))))
    });

    group.finish();
}

fn bench_recall_gating(c: &mut Criterion) {
    let mut group = c.benchmark_group("recall_gating");

    group.bench_function("update", |b| {
        let mut gating = RecallGating::new();
        let mut curvature: f64 = 0.0;
        b.iter(|| {
            curvature += 0.01;
            gating.update(black_box(curvature.sin()))
        })
    });

    group.bench_function("gating_factor", |b| {
        let mut gating = RecallGating::new();
        for i in 0..50 {
            gating.update(i as f64 * 0.02 - 0.5);
        }
        b.iter(|| black_box(gating.gating_factor(black_box(0.5))))
    });

    group.bench_function("should_gate", |b| {
        let mut gating = RecallGating::new();
        for i in 0..50 {
            gating.update(i as f64 * 0.02 - 0.5);
        }
        b.iter(|| black_box(gating.should_gate(black_box(0.3))))
    });

    group.finish();
}

fn bench_holonomic_memory(c: &mut Criterion) {
    let mut group = c.benchmark_group("holonomic_memory");

    group.bench_function("encode", |b| {
        let mut memory = HolonomicMemory::default();
        b.iter(|| black_box(memory.encode(black_box(vec![1.0, 2.0, 3.0]), black_box(0.8))))
    });

    for k in [1, 5, 10].iter() {
        group.bench_with_input(BenchmarkId::new("recall", k), k, |b, &k| {
            let mut memory = HolonomicMemory::default();
            let mut rng = SmallRng::seed_from_u64(42);

            for _ in 0..100 {
                let pattern: Vec<f64> = (0..10).map(|_| rng.gen_range(-1.0..1.0)).collect();
                memory.encode(pattern, rng.gen_range(0.3..1.0));
            }
            memory.step(10.0);

            let query: Vec<f64> = (0..10).map(|_| rng.gen_range(-1.0..1.0)).collect();

            b.iter(|| black_box(memory.recall(black_box(&query), black_box(k))))
        });
    }

    group.bench_function("step", |b| {
        let mut memory = HolonomicMemory::default();
        let mut rng = SmallRng::seed_from_u64(42);

        for _ in 0..50 {
            let pattern: Vec<f64> = (0..5).map(|_| rng.gen_range(-1.0..1.0)).collect();
            memory.encode(pattern, rng.gen_range(0.3..1.0));
        }

        b.iter(|| memory.step(black_box(1.0)))
    });

    group.bench_function("replay_10", |b| {
        let mut memory = HolonomicMemory::default();
        let mut rng = SmallRng::seed_from_u64(42);

        for _ in 0..100 {
            let pattern: Vec<f64> = (0..5).map(|_| rng.gen_range(-1.0..1.0)).collect();
            memory.encode(pattern, rng.gen_range(0.3..1.0));
            memory.step(1.0);
        }

        b.iter(|| memory.replay(black_box(10)))
    });

    group.bench_function("update_regime", |b| {
        let mut memory = HolonomicMemory::default();
        let mut curvature: f64 = 0.0;
        b.iter(|| {
            curvature += 0.01;
            memory.update_regime(black_box(curvature.sin()))
        })
    });

    group.bench_function("stats", |b| {
        let mut memory = HolonomicMemory::default();
        let mut rng = SmallRng::seed_from_u64(42);

        for _ in 0..50 {
            let pattern: Vec<f64> = (0..5).map(|_| rng.gen_range(-1.0..1.0)).collect();
            memory.encode(pattern, rng.gen_range(0.3..1.0));
        }

        b.iter(|| black_box(memory.stats()))
    });

    group.finish();
}

fn bench_full_memory_simulation(c: &mut Criterion) {
    let mut group = c.benchmark_group("full_memory_simulation");

    group.bench_function("encode_recall_cycle_100", |b| {
        b.iter(|| {
            let mut memory = HolonomicMemory::default();
            let mut rng = SmallRng::seed_from_u64(42);

            for i in 0..100 {
                // Encode
                let pattern: Vec<f64> = (0..8).map(|_| rng.gen_range(-1.0..1.0)).collect();
                memory.encode(pattern.clone(), rng.gen_range(0.5..1.0));

                // Step
                memory.step(1.0);

                // Periodic recall
                if i % 10 == 0 {
                    let query: Vec<f64> = (0..8).map(|_| rng.gen_range(-1.0..1.0)).collect();
                    memory.recall(&query, 5);
                }

                // Periodic replay
                if i % 25 == 0 {
                    memory.replay(5);
                }

                // Update regime
                memory.update_regime(rng.gen_range(-1.0..1.0));
            }

            black_box(memory.stats())
        })
    });

    group.bench_function("high_throughput_encoding_1000", |b| {
        b.iter(|| {
            let mut memory = HolonomicMemory::default();
            let mut rng = SmallRng::seed_from_u64(42);

            for _ in 0..1000 {
                let pattern: Vec<f64> = (0..5).map(|_| rng.gen_range(-1.0..1.0)).collect();
                memory.encode(pattern, rng.gen_range(0.3..1.0));
            }

            black_box(memory.stats())
        })
    });

    group.bench_function("consolidation_stress_test", |b| {
        b.iter(|| {
            let config = HolonomicConfig {
                wm_capacity: 3,
                stm_capacity: 10,
                ..Default::default()
            };
            let mut memory = HolonomicMemory::new(config);
            let mut rng = SmallRng::seed_from_u64(42);

            // Force rapid consolidation
            for i in 0..50 {
                let pattern: Vec<f64> = (0..5).map(|_| rng.gen_range(-1.0..1.0)).collect();
                memory.encode(pattern, 0.8);
                memory.step(10.0);

                // Access to boost consolidation
                if i % 5 == 0 {
                    for j in 0..3 {
                        memory.recall(&vec![j as f64], 1);
                    }
                }
            }

            black_box(memory.stats())
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_weber_fechner,
    bench_hyperbolic_ops,
    bench_working_memory,
    bench_short_term_memory,
    bench_long_term_memory,
    bench_hopfield,
    bench_path_integral,
    bench_recall_gating,
    bench_holonomic_memory,
    bench_full_memory_simulation,
);

criterion_main!(benches);
