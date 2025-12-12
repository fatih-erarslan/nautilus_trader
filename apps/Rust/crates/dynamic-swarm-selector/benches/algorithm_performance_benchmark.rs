// Algorithm Performance Benchmark for Dynamic Swarm Selector
// Copyright (c) 2025 TENGRI Trading Swarm

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use dynamic_swarm_selector::{
    SwarmSelector, AlgorithmType, OptimizationProblem, BenchmarkSuite,
    AlgorithmPerformanceAnalyzer, PerformanceProfiler
};
use std::time::Instant;

fn create_optimization_problems() -> Vec<OptimizationProblem> {
    vec![
        OptimizationProblem::new("portfolio_optimization", 50, 1000),
        OptimizationProblem::new("risk_minimization", 30, 500),
        OptimizationProblem::new("return_maximization", 40, 750),
        OptimizationProblem::new("sharpe_optimization", 35, 600),
    ]
}

fn benchmark_algorithm_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("algorithm_performance");
    
    let problems = create_optimization_problems();
    let algorithms = vec![
        AlgorithmType::ParticleSwarm,
        AlgorithmType::AntColony,
        AlgorithmType::GeneticAlgorithm,
        AlgorithmType::DifferentialEvolution,
        AlgorithmType::GreyWolf,
        AlgorithmType::WhaleOptimization,
        AlgorithmType::BatAlgorithm,
        AlgorithmType::FireflyAlgorithm,
        AlgorithmType::CuckooSearch,
        AlgorithmType::ArtificialBeeColony,
    ];
    
    for algorithm in algorithms.iter() {
        for problem in problems.iter() {
            group.bench_with_input(
                BenchmarkId::new("solve_problem", format!("{:?}_{}", algorithm, problem.name())),
                &(algorithm, problem),
                |b, (algo, prob)| {
                    b.iter(|| {
                        let selector = SwarmSelector::new();
                        selector.solve_problem(algo.clone(), prob.clone())
                    })
                }
            );
        }
    }
    group.finish();
}

fn benchmark_convergence_analysis(c: &mut Criterion) {
    let mut group = c.benchmark_group("convergence_analysis");
    
    let problems = create_optimization_problems();
    let analyzer = AlgorithmPerformanceAnalyzer::new();
    
    for problem in problems.iter() {
        group.bench_with_input(
            BenchmarkId::new("analyze_convergence", problem.name()),
            problem,
            |b, prob| {
                b.iter(|| {
                    analyzer.analyze_convergence(prob.clone())
                })
            }
        );
    }
    group.finish();
}

fn benchmark_solution_quality(c: &mut Criterion) {
    let mut group = c.benchmark_group("solution_quality");
    
    let problems = create_optimization_problems();
    let analyzer = AlgorithmPerformanceAnalyzer::new();
    
    for problem in problems.iter() {
        group.bench_with_input(
            BenchmarkId::new("evaluate_quality", problem.name()),
            problem,
            |b, prob| {
                b.iter(|| {
                    analyzer.evaluate_solution_quality(prob.clone())
                })
            }
        );
    }
    group.finish();
}

fn benchmark_computational_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("computational_efficiency");
    
    let profiler = PerformanceProfiler::new();
    let algorithms = vec![
        AlgorithmType::ParticleSwarm,
        AlgorithmType::AntColony,
        AlgorithmType::GeneticAlgorithm,
    ];
    
    for algorithm in algorithms.iter() {
        group.bench_with_input(
            BenchmarkId::new("profile_efficiency", format!("{:?}", algorithm)),
            algorithm,
            |b, algo| {
                b.iter(|| {
                    profiler.profile_algorithm(algo.clone())
                })
            }
        );
    }
    group.finish();
}

fn benchmark_parallel_execution(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_execution");
    
    let problems = create_optimization_problems();
    let selector = SwarmSelector::new();
    
    for thread_count in [1, 2, 4, 8, 16].iter() {
        group.bench_with_input(
            BenchmarkId::new("parallel_solve", thread_count),
            thread_count,
            |b, &threads| {
                b.iter(|| {
                    selector.solve_parallel(&problems, threads)
                })
            }
        );
    }
    group.finish();
}

fn benchmark_memory_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_efficiency");
    
    let selector = SwarmSelector::new();
    
    for population_size in [50, 100, 200, 500, 1000].iter() {
        group.bench_with_input(
            BenchmarkId::new("memory_usage", population_size),
            population_size,
            |b, &size| {
                b.iter(|| {
                    selector.test_memory_efficiency(size)
                })
            }
        );
    }
    group.finish();
}

fn benchmark_benchmark_suite(c: &mut Criterion) {
    let mut group = c.benchmark_group("benchmark_suite");
    
    let suite = BenchmarkSuite::new();
    
    group.bench_function("run_full_suite", |b| {
        b.iter(|| {
            suite.run_full_benchmark()
        })
    });
    
    group.bench_function("run_quick_benchmark", |b| {
        b.iter(|| {
            suite.run_quick_benchmark()
        })
    });
    
    group.finish();
}

criterion_group!(
    benches,
    benchmark_algorithm_performance,
    benchmark_convergence_analysis,
    benchmark_solution_quality,
    benchmark_computational_efficiency,
    benchmark_parallel_execution,
    benchmark_memory_efficiency,
    benchmark_benchmark_suite
);
criterion_main!(benches);