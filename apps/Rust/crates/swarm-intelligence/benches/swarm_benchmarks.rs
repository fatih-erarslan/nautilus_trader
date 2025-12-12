// Swarm Intelligence Benchmarks
// Copyright (c) 2025 TENGRI Trading Swarm

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use swarm_intelligence::{
    SwarmOptimizer, SwarmAlgorithm, OptimizationProblem, Population,
    ParticleSwarmOptimizer, AntColonyOptimizer, BeeColonyOptimizer,
    FireflyOptimizer, CuckooSearchOptimizer, GreyWolfOptimizer
};
use std::time::Instant;

fn create_optimization_problem(dimensions: usize) -> OptimizationProblem {
    OptimizationProblem::new(
        dimensions,
        vec![-100.0; dimensions],  // lower bounds
        vec![100.0; dimensions],   // upper bounds
        "rosenbrock"               // function type
    )
}

fn benchmark_particle_swarm(c: &mut Criterion) {
    let mut group = c.benchmark_group("particle_swarm");
    
    for dimensions in [10, 30, 50, 100].iter() {
        let problem = create_optimization_problem(*dimensions);
        let optimizer = ParticleSwarmOptimizer::new(50, 1000); // 50 particles, 1000 iterations
        
        group.bench_with_input(BenchmarkId::new("optimize", dimensions), dimensions, |b, _| {
            b.iter(|| {
                optimizer.optimize(&problem)
            })
        });
    }
    group.finish();
}

fn benchmark_ant_colony(c: &mut Criterion) {
    let mut group = c.benchmark_group("ant_colony");
    
    for dimensions in [10, 30, 50].iter() {
        let problem = create_optimization_problem(*dimensions);
        let optimizer = AntColonyOptimizer::new(30, 500); // 30 ants, 500 iterations
        
        group.bench_with_input(BenchmarkId::new("optimize", dimensions), dimensions, |b, _| {
            b.iter(|| {
                optimizer.optimize(&problem)
            })
        });
    }
    group.finish();
}

fn benchmark_bee_colony(c: &mut Criterion) {
    let mut group = c.benchmark_group("bee_colony");
    
    for dimensions in [10, 30, 50].iter() {
        let problem = create_optimization_problem(*dimensions);
        let optimizer = BeeColonyOptimizer::new(40, 500); // 40 bees, 500 iterations
        
        group.bench_with_input(BenchmarkId::new("optimize", dimensions), dimensions, |b, _| {
            b.iter(|| {
                optimizer.optimize(&problem)
            })
        });
    }
    group.finish();
}

fn benchmark_firefly(c: &mut Criterion) {
    let mut group = c.benchmark_group("firefly");
    
    for dimensions in [10, 30, 50].iter() {
        let problem = create_optimization_problem(*dimensions);
        let optimizer = FireflyOptimizer::new(25, 500); // 25 fireflies, 500 iterations
        
        group.bench_with_input(BenchmarkId::new("optimize", dimensions), dimensions, |b, _| {
            b.iter(|| {
                optimizer.optimize(&problem)
            })
        });
    }
    group.finish();
}

fn benchmark_cuckoo_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("cuckoo_search");
    
    for dimensions in [10, 30, 50].iter() {
        let problem = create_optimization_problem(*dimensions);
        let optimizer = CuckooSearchOptimizer::new(25, 500); // 25 nests, 500 iterations
        
        group.bench_with_input(BenchmarkId::new("optimize", dimensions), dimensions, |b, _| {
            b.iter(|| {
                optimizer.optimize(&problem)
            })
        });
    }
    group.finish();
}

fn benchmark_grey_wolf(c: &mut Criterion) {
    let mut group = c.benchmark_group("grey_wolf");
    
    for dimensions in [10, 30, 50].iter() {
        let problem = create_optimization_problem(*dimensions);
        let optimizer = GreyWolfOptimizer::new(30, 500); // 30 wolves, 500 iterations
        
        group.bench_with_input(BenchmarkId::new("optimize", dimensions), dimensions, |b, _| {
            b.iter(|| {
                optimizer.optimize(&problem)
            })
        });
    }
    group.finish();
}

fn benchmark_population_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("population_operations");
    
    for population_size in [25, 50, 100, 200].iter() {
        let problem = create_optimization_problem(30);
        let mut population = Population::new(*population_size, &problem);
        
        group.bench_with_input(BenchmarkId::new("evaluate_population", population_size), population_size, |b, _| {
            b.iter(|| {
                population.evaluate_fitness(&problem)
            })
        });
        
        group.bench_with_input(BenchmarkId::new("update_population", population_size), population_size, |b, _| {
            b.iter(|| {
                population.update_positions(&problem)
            })
        });
    }
    group.finish();
}

fn benchmark_convergence_analysis(c: &mut Criterion) {
    let mut group = c.benchmark_group("convergence_analysis");
    
    let problem = create_optimization_problem(30);
    let optimizer = ParticleSwarmOptimizer::new(50, 1000);
    
    group.bench_function("convergence_tracking", |b| {
        b.iter(|| {
            optimizer.track_convergence(&problem)
        })
    });
    
    group.bench_function("diversity_measurement", |b| {
        b.iter(|| {
            optimizer.measure_diversity(&problem)
        })
    });
    
    group.finish();
}

fn benchmark_parallel_execution(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_execution");
    
    let problem = create_optimization_problem(50);
    
    for thread_count in [1, 2, 4, 8].iter() {
        let optimizer = ParticleSwarmOptimizer::with_threads(50, 500, *thread_count);
        
        group.bench_with_input(BenchmarkId::new("parallel_optimize", thread_count), thread_count, |b, _| {
            b.iter(|| {
                optimizer.optimize_parallel(&problem)
            })
        });
    }
    group.finish();
}

fn benchmark_algorithm_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("algorithm_comparison");
    
    let problem = create_optimization_problem(30);
    let algorithms = vec![
        ("PSO", SwarmAlgorithm::ParticleSwarm),
        ("ACO", SwarmAlgorithm::AntColony),
        ("ABC", SwarmAlgorithm::BeeColony),
        ("FA", SwarmAlgorithm::Firefly),
        ("CS", SwarmAlgorithm::CuckooSearch),
        ("GWO", SwarmAlgorithm::GreyWolf),
    ];
    
    for (name, algorithm) in algorithms.iter() {
        group.bench_with_input(BenchmarkId::new("solve_problem", name), algorithm, |b, algo| {
            b.iter(|| {
                let optimizer = SwarmOptimizer::new(algo.clone(), 50, 500);
                optimizer.solve(&problem)
            })
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    benchmark_particle_swarm,
    benchmark_ant_colony,
    benchmark_bee_colony,
    benchmark_firefly,
    benchmark_cuckoo_search,
    benchmark_grey_wolf,
    benchmark_population_operations,
    benchmark_convergence_analysis,
    benchmark_parallel_execution,
    benchmark_algorithm_comparison
);
criterion_main!(benches);