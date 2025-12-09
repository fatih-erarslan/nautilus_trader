//! Benchmarks for Phase 8: Morphogenetic Field Dynamics
//!
//! Wolfram-verified performance targets:
//! - Morphogen gradient: <100μs for 100x100 grid
//! - Reaction-diffusion step: <1ms for 100x100 grid
//! - Pattern wavelength: <100ns
//! - Attractor potential: <1μs
//! - Full system step: <5ms for 50x50

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use rand::SeedableRng;
use rand::rngs::SmallRng;
use tengri_holographic_cortex::{
    MorphogenField, ReactionDiffusion, AttractorLandscape, AttractorType,
    FieldPBitCoupler, MorphogeneticSystem, MorphogeneticConfig,
    MORPHOGEN_DECAY_LENGTH,
};

fn bench_morphogen_field(c: &mut Criterion) {
    let mut group = c.benchmark_group("morphogen_field");

    for size in [20, 50, 100].iter() {
        group.bench_with_input(BenchmarkId::new("compute_gradient", size), size, |b, &n| {
            let mut field = MorphogenField::new(n, n, MORPHOGEN_DECAY_LENGTH);
            field.set_source(n / 2, n / 2, 1.0);
            b.iter(|| {
                field.compute_gradient(black_box(n / 2), black_box(n / 2))
            })
        });

        group.bench_with_input(BenchmarkId::new("step", size), size, |b, &n| {
            let mut field = MorphogenField::new(n, n, MORPHOGEN_DECAY_LENGTH);
            field.set_source(n / 2, n / 2, 1.0);
            field.compute_gradient(n / 2, n / 2);
            b.iter(|| {
                field.step(black_box(0.1))
            })
        });
    }

    group.bench_function("french_flag_encoding", |b| {
        let mut field = MorphogenField::new(100, 100, MORPHOGEN_DECAY_LENGTH);
        field.set_source(50, 50, 1.0);
        field.compute_gradient(50, 50);
        b.iter(|| {
            black_box(field.french_flag_encoding(black_box(25), black_box(25), black_box(0.5)))
        })
    });

    group.bench_function("stats_100", |b| {
        let mut field = MorphogenField::new(100, 100, MORPHOGEN_DECAY_LENGTH);
        field.set_source(50, 50, 1.0);
        field.compute_gradient(50, 50);
        b.iter(|| {
            black_box(field.stats())
        })
    });

    group.finish();
}

fn bench_reaction_diffusion(c: &mut Criterion) {
    let mut group = c.benchmark_group("reaction_diffusion");

    for size in [20, 50, 100].iter() {
        group.bench_with_input(BenchmarkId::new("step", size), size, |b, &n| {
            let mut rd = ReactionDiffusion::new(n, n);
            let mut rng = SmallRng::seed_from_u64(42);
            rd.perturb(0.1, &mut rng);
            b.iter(|| {
                rd.step()
            })
        });

        group.bench_with_input(BenchmarkId::new("evolve_10", size), size, |b, &n| {
            let mut rd = ReactionDiffusion::new(n, n);
            let mut rng = SmallRng::seed_from_u64(42);
            rd.perturb(0.1, &mut rng);
            b.iter(|| {
                rd.evolve(black_box(10))
            })
        });
    }

    group.bench_function("pattern_wavelength", |b| {
        let rd = ReactionDiffusion::new(50, 50);
        b.iter(|| {
            black_box(rd.pattern_wavelength())
        })
    });

    group.bench_function("pattern_stats_50", |b| {
        let mut rd = ReactionDiffusion::new(50, 50);
        let mut rng = SmallRng::seed_from_u64(42);
        rd.perturb(0.1, &mut rng);
        rd.evolve(100);
        b.iter(|| {
            black_box(rd.pattern_stats())
        })
    });

    group.bench_function("is_turing_unstable", |b| {
        let rd = ReactionDiffusion::new(50, 50);
        b.iter(|| {
            black_box(rd.is_turing_unstable())
        })
    });

    group.finish();
}

fn bench_attractor_landscape(c: &mut Criterion) {
    let mut group = c.benchmark_group("attractor_landscape");

    group.bench_function("potential_3_attractors", |b| {
        let mut landscape = AttractorLandscape::new(2);
        landscape.add_attractor(vec![0.0, 0.0], 1.0, 1.0, AttractorType::PointAttractor);
        landscape.add_attractor(vec![5.0, 0.0], 1.0, 0.8, AttractorType::PointAttractor);
        landscape.add_attractor(vec![2.5, 4.0], 1.0, 0.6, AttractorType::PointAttractor);
        landscape.set_state(vec![1.0, 1.0]);
        b.iter(|| {
            black_box(landscape.potential())
        })
    });

    group.bench_function("gradient_3_attractors", |b| {
        let mut landscape = AttractorLandscape::new(2);
        landscape.add_attractor(vec![0.0, 0.0], 1.0, 1.0, AttractorType::PointAttractor);
        landscape.add_attractor(vec![5.0, 0.0], 1.0, 0.8, AttractorType::PointAttractor);
        landscape.add_attractor(vec![2.5, 4.0], 1.0, 0.6, AttractorType::PointAttractor);
        landscape.set_state(vec![1.0, 1.0]);
        b.iter(|| {
            black_box(landscape.gradient())
        })
    });

    group.bench_function("step", |b| {
        let mut landscape = AttractorLandscape::new(2);
        landscape.add_attractor(vec![0.0, 0.0], 1.0, 1.0, AttractorType::PointAttractor);
        landscape.set_state(vec![1.0, 1.0]);
        let mut rng = SmallRng::seed_from_u64(42);
        b.iter(|| {
            landscape.step(black_box(0.01), &mut rng)
        })
    });

    group.bench_function("nearest_attractor_10", |b| {
        let mut landscape = AttractorLandscape::new(2);
        for i in 0..10 {
            landscape.add_attractor(
                vec![i as f64, (i * 2) as f64],
                1.0,
                1.0,
                AttractorType::PointAttractor,
            );
        }
        landscape.set_state(vec![5.0, 5.0]);
        b.iter(|| {
            black_box(landscape.nearest_attractor())
        })
    });

    group.finish();
}

fn bench_field_pbit_coupler(c: &mut Criterion) {
    let mut group = c.benchmark_group("field_pbit_coupler");

    group.bench_function("morphogen_to_bias", |b| {
        let coupler = FieldPBitCoupler::default();
        b.iter(|| {
            black_box(coupler.morphogen_to_bias(black_box(0.8), black_box(0.5)))
        })
    });

    group.bench_function("pbit_probability", |b| {
        let coupler = FieldPBitCoupler::default();
        b.iter(|| {
            black_box(coupler.pbit_probability(black_box(0.8), black_box(0.5), black_box(1.0)))
        })
    });

    group.bench_function("effective_temperature", |b| {
        let coupler = FieldPBitCoupler::default();
        b.iter(|| {
            black_box(coupler.effective_temperature(black_box(1.0), black_box(0.5)))
        })
    });

    group.finish();
}

fn bench_morphogenetic_system(c: &mut Criterion) {
    let mut group = c.benchmark_group("morphogenetic_system");

    for size in [20, 50].iter() {
        let config = MorphogeneticConfig {
            width: *size,
            height: *size,
            enable_reaction_diffusion: true,
            enable_gradients: true,
            enable_attractors: false,
            pbit_coupling: 1.0,
        };

        group.bench_with_input(BenchmarkId::new("step", size), size, |b, _| {
            let mut system = MorphogeneticSystem::new(config.clone());
            let mut rng = SmallRng::seed_from_u64(42);
            system.initialize(&mut rng);
            b.iter(|| {
                system.step(&mut rng)
            })
        });

        group.bench_with_input(BenchmarkId::new("stats", size), size, |b, _| {
            let mut system = MorphogeneticSystem::new(config.clone());
            let mut rng = SmallRng::seed_from_u64(42);
            system.initialize(&mut rng);
            for _ in 0..10 {
                system.step(&mut rng);
            }
            b.iter(|| {
                black_box(system.stats())
            })
        });
    }

    group.bench_function("get_pbit_bias", |b| {
        let config = MorphogeneticConfig::default();
        let mut system = MorphogeneticSystem::new(config);
        let mut rng = SmallRng::seed_from_u64(42);
        system.initialize(&mut rng);
        b.iter(|| {
            black_box(system.get_pbit_bias(black_box(25), black_box(25)))
        })
    });

    group.finish();
}

fn bench_full_pattern_formation(c: &mut Criterion) {
    let mut group = c.benchmark_group("full_pattern_formation");

    group.bench_function("evolve_100_steps_50x50", |b| {
        b.iter(|| {
            let mut rd = ReactionDiffusion::new(50, 50);
            let mut rng = SmallRng::seed_from_u64(42);
            rd.perturb(0.1, &mut rng);
            rd.evolve(100);
            black_box(rd.pattern_stats())
        })
    });

    group.bench_function("system_100_steps_30x30", |b| {
        b.iter(|| {
            let config = MorphogeneticConfig {
                width: 30,
                height: 30,
                enable_reaction_diffusion: true,
                enable_gradients: true,
                enable_attractors: false,
                pbit_coupling: 1.0,
            };
            let mut system = MorphogeneticSystem::new(config);
            let mut rng = SmallRng::seed_from_u64(42);
            system.initialize(&mut rng);
            for _ in 0..100 {
                system.step(&mut rng);
            }
            black_box(system.stats())
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_morphogen_field,
    bench_reaction_diffusion,
    bench_attractor_landscape,
    bench_field_pbit_coupler,
    bench_morphogenetic_system,
    bench_full_pattern_formation,
);

criterion_main!(benches);
