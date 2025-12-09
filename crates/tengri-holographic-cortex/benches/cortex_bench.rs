//! Benchmarks for Tengri Holographic Cortex
//!
//! Run with: cargo bench --bench cortex_bench
//!
//! Hardware target:
//! - Intel i9-13900K (24 cores, AVX2/AVX-512)
//! - AMD Radeon 6800XT (16GB VRAM)
//! - 96GB RAM

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use tengri_holographic_cortex::*;

// ============================================================================
// pBit Engine Benchmarks
// ============================================================================

fn bench_pbit_engine(c: &mut Criterion) {
    let mut group = c.benchmark_group("pbit_engine");
    
    for num_pbits in [256, 1024, 4096, 16384] {
        let config = EngineConfig {
            num_pbits,
            seed: Some(42),
            ..Default::default()
        };
        let mut engine = PBitEngine::new(0, config);
        
        group.throughput(Throughput::Elements(num_pbits as u64));
        group.bench_with_input(
            BenchmarkId::new("step", num_pbits),
            &num_pbits,
            |b, _| {
                b.iter(|| {
                    engine.step();
                    black_box(engine.spike_rate())
                })
            },
        );
    }
    
    group.finish();
}

fn bench_pbit_engine_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("pbit_engine_batch");
    
    let config = EngineConfig {
        num_pbits: 1024,
        seed: Some(42),
        ..Default::default()
    };
    let mut engine = PBitEngine::new(0, config);
    
    for batch_size in [1, 8, 16, 32] {
        group.throughput(Throughput::Elements(batch_size as u64 * 1024));
        group.bench_with_input(
            BenchmarkId::new("step_n", batch_size),
            &batch_size,
            |b, &n| {
                b.iter(|| {
                    engine.step_n(n);
                    black_box(engine.spike_rate())
                })
            },
        );
    }
    
    group.finish();
}

// ============================================================================
// Cortex4 Topology Benchmarks
// ============================================================================

fn bench_cortex4_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("cortex4");
    
    for num_pbits in [256, 1024, 4096] {
        let config = TopologyConfig {
            engine_config: EngineConfig {
                num_pbits,
                seed: Some(42),
                ..Default::default()
            },
            ..Default::default()
        };
        let mut cortex = Cortex4::new(config);
        
        group.throughput(Throughput::Elements(4 * num_pbits as u64));
        group.bench_with_input(
            BenchmarkId::new("step", num_pbits),
            &num_pbits,
            |b, _| {
                b.iter(|| {
                    cortex.step();
                    black_box(cortex.spike_rates())
                })
            },
        );
    }
    
    group.finish();
}

// ============================================================================
// Hyperbolic Geometry Benchmarks
// ============================================================================

fn bench_hyperbolic(c: &mut Criterion) {
    let mut group = c.benchmark_group("hyperbolic");
    
    let p1 = LorentzPoint11::from_euclidean(&vec![0.1; 11]);
    let p2 = LorentzPoint11::from_euclidean(&vec![0.2; 11]);
    
    group.bench_function("distance", |b| {
        b.iter(|| black_box(p1.distance(&p2)))
    });
    
    group.bench_function("lorentz_constraint", |b| {
        b.iter(|| black_box(p1.lorentz_constraint()))
    });
    
    group.bench_function("to_poincare", |b| {
        b.iter(|| black_box(p1.to_poincare()))
    });
    
    group.bench_function("from_euclidean", |b| {
        let z = vec![0.1; 11];
        b.iter(|| black_box(LorentzPoint11::from_euclidean(&z)))
    });
    
    // Möbius operations
    let blender = MobiusBlend::new(-1.0);
    let points: Vec<Vec<f64>> = (0..4).map(|i| vec![0.1 * i as f64; 11]).collect();
    let weights = vec![0.25; 4];
    
    group.bench_function("mobius_blend_4", |b| {
        b.iter(|| black_box(blender.blend(&points, &weights)))
    });
    
    group.finish();
}

// ============================================================================
// Memory Fabric Benchmarks
// ============================================================================

fn bench_memory_fabric(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_fabric");
    
    // Different sizes
    for size in [100, 1000, 10000] {
        let mut fabric = MemoryFabric::default();
        
        for i in 0..size {
            let v: Vec<f64> = (0..11).map(|j| (i * 11 + j) as f64 / (size * 11) as f64).collect();
            fabric.insert(v);
        }
        
        let query = vec![0.5; 11];
        
        group.throughput(Throughput::Elements(size as u64));
        
        group.bench_with_input(
            BenchmarkId::new("query_k10", size),
            &size,
            |b, _| b.iter(|| black_box(fabric.query(&query, 10))),
        );
    }
    
    // Insertion benchmark
    group.bench_function("insert_1000", |b| {
        b.iter(|| {
            let mut fabric = MemoryFabric::default();
            for i in 0..1000 {
                let v: Vec<f64> = (0..11).map(|j| (i * 11 + j) as f64 / 10000.0).collect();
                fabric.insert(v);
            }
            black_box(fabric.len())
        })
    });
    
    group.finish();
}

// ============================================================================
// SIMD Benchmarks
// ============================================================================

fn bench_simd(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd");
    
    // Fast exp
    group.bench_function("fast_exp_1000", |b| {
        let inputs: Vec<f32> = (0..1000).map(|i| -i as f32 * 0.01).collect();
        b.iter(|| {
            for &x in &inputs {
                black_box(simd::fast_exp_f32(x));
            }
        })
    });
    
    // Fast sigmoid
    group.bench_function("fast_sigmoid_1000", |b| {
        let inputs: Vec<f32> = (0..1000).map(|i| (i as f32 - 500.0) * 0.01).collect();
        b.iter(|| {
            for &x in &inputs {
                black_box(simd::fast_sigmoid_f32(x));
            }
        })
    });
    
    // Batch probabilities
    for size in [1000, 10000, 100000] {
        let fields: Vec<f32> = (0..size).map(|i| (i as f32 - size as f32 / 2.0) * 0.001).collect();
        let biases = vec![0.0f32; size];

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::new("pbit_probs_batch", size),
            &size,
            |b, _| {
                let mut probs = vec![0.0f32; size];
                b.iter(|| {
                    simd::pbit_probabilities_batch(&fields, &biases, 1.0, &mut probs);
                    black_box(probs[0])
                })
            },
        );
    }
    
    // Lorentz inner product (f32)
    let x_f32 = [1.0f32, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1];
    let y_f32 = [1.2f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2];

    group.bench_function("lorentz_inner_f32", |b| {
        b.iter(|| black_box(simd::lorentz_inner_simd(&x_f32, &y_f32)))
    });

    // Lorentz inner product (f64) - scalar vs SIMD comparison
    let x_f64 = [1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1];
    let y_f64 = [1.2, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2];

    group.bench_function("lorentz_inner_scalar_f64", |b| {
        b.iter(|| black_box(simd::lorentz_inner_scalar(&x_f64, &y_f64)))
    });

    group.bench_function("lorentz_inner_simd_f64", |b| {
        b.iter(|| black_box(simd::lorentz_inner_f64(&x_f64, &y_f64)))
    });

    // AVX2 direct benchmark (if available)
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            group.bench_function("lorentz_inner_avx2_f64", |b| {
                b.iter(|| unsafe { black_box(simd::lorentz_inner_avx2(&x_f64, &y_f64)) })
            });
        }
    }

    // Hyperbolic distance (single pair)
    group.bench_function("hyperbolic_distance_simd", |b| {
        b.iter(|| black_box(simd::hyperbolic_distance_simd(&x_f64, &y_f64)))
    });

    // Stable acosh benchmarks
    group.bench_function("stable_acosh_near_1", |b| {
        let x = 1.001;
        b.iter(|| black_box(simd::stable_acosh_f64(x)))
    });

    group.bench_function("stable_acosh_far_from_1", |b| {
        let x = 2.5;
        b.iter(|| black_box(simd::stable_acosh_f64(x)))
    });

    // Batch distance computation benchmarks
    for corpus_size in [10, 100, 1000, 10000] {
        let query = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        // Generate random corpus points on hyperboloid
        let mut corpus = Vec::with_capacity(corpus_size);
        for i in 0..corpus_size {
            let mut point = [0.0; 12];
            for j in 1..12 {
                point[j] = ((i * 11 + j) as f64 * 0.001).sin() * 0.1;
            }
            // Lift to hyperboloid
            let spatial_norm_sq: f64 = point[1..].iter().map(|x| x * x).sum();
            point[0] = (1.0 + spatial_norm_sq).sqrt();
            corpus.push(point);
        }

        group.throughput(Throughput::Elements(corpus_size as u64));

        group.bench_with_input(
            BenchmarkId::new("batch_distances", corpus_size),
            &corpus_size,
            |b, _| {
                b.iter(|| black_box(simd::batch_hyperbolic_distances(&query, &corpus)))
            },
        );

        group.bench_with_input(
            BenchmarkId::new("batch_distances_into", corpus_size),
            &corpus_size,
            |b, _| {
                let mut output = vec![0.0; corpus_size];
                b.iter(|| {
                    simd::batch_hyperbolic_distances_into(&query, &corpus, &mut output);
                    black_box(output[0])
                })
            },
        );

        // Parallel version (if feature enabled)
        #[cfg(feature = "rayon")]
        {
            if corpus_size >= 1000 {
                group.bench_with_input(
                    BenchmarkId::new("batch_distances_parallel", corpus_size),
                    &corpus_size,
                    |b, _| {
                        b.iter(|| black_box(simd::batch_hyperbolic_distances_parallel(&query, &corpus)))
                    },
                );
            }
        }
    }

    group.finish();
}

// ============================================================================
// Boltzmann Statistics Benchmarks
// ============================================================================

fn bench_boltzmann(c: &mut Criterion) {
    let mut group = c.benchmark_group("boltzmann");
    
    for size in [10, 100, 1000] {
        let energies: Vec<f64> = (0..size).map(|i| i as f64 * 0.1).collect();
        
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::new("probabilities", size),
            &size,
            |b, _| b.iter(|| black_box(boltzmann_probabilities(&energies, 1.0))),
        );
    }
    
    // Annealing schedules
    group.bench_function("annealing_log_1000", |b| {
        b.iter(|| {
            for t in 0..1000 {
                black_box(annealing_logarithmic(ISING_CRITICAL_TEMP, t));
            }
        })
    });
    
    group.bench_function("annealing_exp_1000", |b| {
        b.iter(|| {
            for t in 0..1000 {
                black_box(annealing_exponential(ISING_CRITICAL_TEMP, 0.99, t));
            }
        })
    });
    
    group.finish();
}

// ============================================================================
// MSOCL Benchmarks
// ============================================================================

fn bench_msocl(c: &mut Criterion) {
    let mut group = c.benchmark_group("msocl");
    
    let mut msocl = Msocl::new();
    
    group.bench_function("tick", |b| {
        b.iter(|| {
            msocl.tick();
            black_box(msocl.current_phase())
        })
    });
    
    group.bench_function("engine_temperatures", |b| {
        b.iter(|| black_box(msocl.engine_temperatures(ISING_CRITICAL_TEMP)))
    });
    
    group.finish();
}

// ============================================================================
// Criterion Groups
// ============================================================================

criterion_group!(
    benches,
    bench_pbit_engine,
    bench_pbit_engine_batch,
    bench_cortex4_step,
    bench_hyperbolic,
    bench_memory_fabric,
    bench_simd,
    bench_boltzmann,
    bench_msocl,
);

criterion_main!(benches);
