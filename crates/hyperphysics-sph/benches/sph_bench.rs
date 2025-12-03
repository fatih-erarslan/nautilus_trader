//! SPH Benchmarks
//!
//! Performance benchmarks for SPH simulation components.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use hyperphysics_sph::{
    SphWorld, SphConfig, ParticleType, PhysicsConstants,
    IntegrationMethod,
};

fn bench_density_computation(c: &mut Criterion) {
    let mut group = c.benchmark_group("density_computation");

    for num_particles in [100, 1000, 5000] {
        group.bench_with_input(
            BenchmarkId::from_parameter(num_particles),
            &num_particles,
            |b, &n| {
                let mut world = SphWorld::fluid();
                let spacing = 0.5;
                let size = ((n as f32).cbrt() * spacing) as f32;

                let _ = world.add_particle_block(
                    [0.0, 0.0, 0.0],
                    [size, size, size],
                    spacing,
                    ParticleType::Liquid,
                );

                b.iter(|| {
                    world.step();
                    black_box(&world);
                });
            },
        );
    }
    group.finish();
}

fn bench_integration_methods(c: &mut Criterion) {
    let mut group = c.benchmark_group("integration_methods");

    for method in [
        IntegrationMethod::SemiImplicitEuler,
        IntegrationMethod::Leapfrog,
        IntegrationMethod::VelocityVerlet,
    ] {
        let name = match method {
            IntegrationMethod::SemiImplicitEuler => "euler",
            IntegrationMethod::Leapfrog => "leapfrog",
            IntegrationMethod::VelocityVerlet => "verlet",
        };

        group.bench_function(name, |b| {
            let mut config = SphConfig::fluid();
            config.solver.integration_method = method;
            let mut world = SphWorld::new(config);

            let _ = world.add_particle_block(
                [0.0, 5.0, 0.0],
                [3.0, 8.0, 3.0],
                0.5,
                ParticleType::Liquid,
            );

            b.iter(|| {
                world.step();
                black_box(&world);
            });
        });
    }
    group.finish();
}

fn bench_elastic_network(c: &mut Criterion) {
    let mut group = c.benchmark_group("elastic_network");

    for num_connections in [100, 500, 1000] {
        group.bench_with_input(
            BenchmarkId::from_parameter(num_connections),
            &num_connections,
            |b, &n| {
                let mut world = SphWorld::worm();

                // Create a chain of particles
                let num_particles = (n as f32).sqrt() as usize + 1;
                let mut indices = Vec::with_capacity(num_particles);

                for i in 0..num_particles {
                    let idx = world.add_particle(
                        [i as f32 * 0.5, 5.0, 0.0],
                        [0.0, 0.0, 0.0],
                        ParticleType::Elastic,
                    ).unwrap();
                    indices.push(idx);
                }

                // Connect adjacent particles
                for i in 0..indices.len() - 1 {
                    world.connect(indices[i], indices[i + 1], 100.0);
                }

                b.iter(|| {
                    world.step();
                    black_box(&world);
                });
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_density_computation,
    bench_integration_methods,
    bench_elastic_network,
);
criterion_main!(benches);
