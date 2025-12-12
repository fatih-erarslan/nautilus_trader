use criterion::{black_box, criterion_group, criterion_main, Criterion};
use swarm_intelligence::pso::ParticleSwarmOptimizer;
use swarm_intelligence::aco::AntColonyOptimizer;
use swarm_intelligence::abc::ArtificialBeeColony;
use swarm_intelligence::config::SwarmConfig;

fn pso_optimization_benchmark(c: &mut Criterion) {
    let config = SwarmConfig::default();
    let mut optimizer = ParticleSwarmOptimizer::new(config);
    let problem_size = 10;
    
    c.bench_function("pso_optimization", |b| {
        b.iter(|| {
            optimizer.optimize(black_box(problem_size))
        })
    });
}

fn aco_optimization_benchmark(c: &mut Criterion) {
    let config = SwarmConfig::default();
    let mut optimizer = AntColonyOptimizer::new(config);
    let graph_size = 20;
    
    c.bench_function("aco_optimization", |b| {
        b.iter(|| {
            optimizer.find_path(black_box(graph_size))
        })
    });
}

fn abc_optimization_benchmark(c: &mut Criterion) {
    let config = SwarmConfig::default();
    let mut colony = ArtificialBeeColony::new(config);
    let food_sources = 50;
    
    c.bench_function("abc_optimization", |b| {
        b.iter(|| {
            colony.forage(black_box(food_sources))
        })
    });
}

criterion_group!(benches, pso_optimization_benchmark, aco_optimization_benchmark, abc_optimization_benchmark);
criterion_main!(benches);