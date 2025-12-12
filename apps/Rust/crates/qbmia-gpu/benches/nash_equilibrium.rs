//! Nash Equilibrium GPU Solver Benchmarks

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use qbmia_gpu::{
    nash::{GpuNashSolver, PayoffMatrix, SolverConfig, NashAlgorithm},
    initialize,
};
use ndarray::Array;

fn create_test_game(num_players: usize, strategies_per_player: usize) -> PayoffMatrix {
    let total_outcomes = strategies_per_player.pow(num_players as u32);
    
    let payoffs = (0..num_players).map(|player| {
        // Create random payoff matrix for each player
        let mut shape = vec![strategies_per_player; num_players];
        let data: Vec<f64> = (0..total_outcomes)
            .map(|i| (i as f64 + player as f64) % 10.0) // Deterministic "random" values
            .collect();
        
        Array::from_shape_vec(ndarray::IxDyn(&shape), data).unwrap()
    }).collect();
    
    PayoffMatrix {
        num_players,
        strategies: vec![strategies_per_player; num_players],
        payoffs,
    }
}

fn benchmark_nash_solver_creation(c: &mut Criterion) {
    let _ = initialize();
    
    let mut group = c.benchmark_group("nash_solver_creation");
    
    for &num_players in [2, 3, 4, 5].iter() {
        for &strategies in [2, 3, 4, 5].iter() {
            group.bench_with_input(
                BenchmarkId::new("create_solver", format!("{}p_{}s", num_players, strategies)),
                &(num_players, strategies),
                |b, &(num_players, strategies)| {
                    b.iter(|| {
                        let payoff_matrix = create_test_game(black_box(num_players), black_box(strategies));
                        let config = SolverConfig::default();
                        
                        // Note: This will likely fail without GPU, but tests the interface
                        let result = GpuNashSolver::new(0, payoff_matrix, config);
                        black_box(result)
                    });
                },
            );
        }
    }
    
    group.finish();
}

fn benchmark_nash_algorithms(c: &mut Criterion) {
    let _ = initialize();
    
    let mut group = c.benchmark_group("nash_algorithms");
    
    let algorithms = [
        ("projected_gradient", NashAlgorithm::ProjectedGradient),
        ("fictitious_play", NashAlgorithm::FictitiousPlay),
        ("regret_minimization", NashAlgorithm::RegretMinimization),
    ];
    
    for (name, algorithm) in algorithms.iter() {
        group.bench_with_input(
            BenchmarkId::new("algorithm", name),
            algorithm,
            |b, &algorithm| {
                b.iter(|| {
                    let payoff_matrix = create_test_game(2, 3);
                    let mut config = SolverConfig::default();
                    config.algorithm = black_box(algorithm);
                    config.max_iterations = 100; // Limit iterations for benchmarking
                    
                    // Create solver and attempt to solve
                    match GpuNashSolver::new(0, payoff_matrix, config) {
                        Ok(mut solver) => {
                            let result = solver.solve();
                            black_box(result)
                        }
                        Err(e) => black_box(Err(e)), // Expected without GPU
                    }
                });
            },
        );
    }
    
    group.finish();
}

fn benchmark_quantum_nash_enhancement(c: &mut Criterion) {
    let _ = initialize();
    
    c.bench_function("quantum_enhanced_nash", |b| {
        b.iter(|| {
            let payoff_matrix = create_test_game(black_box(2), black_box(3));
            let mut config = SolverConfig::default();
            config.quantum_enhanced = true;
            config.quantum_qubits = 6;
            config.max_iterations = 50;
            
            match GpuNashSolver::new(0, payoff_matrix, config) {
                Ok(mut solver) => {
                    let result = solver.solve();
                    black_box(result)
                }
                Err(e) => black_box(Err(e)),
            }
        });
    });
}

fn benchmark_large_games(c: &mut Criterion) {
    let _ = initialize();
    
    let mut group = c.benchmark_group("large_games");
    
    // Test scalability with larger games
    for &(players, strategies) in [(3, 5), (4, 4), (5, 3)].iter() {
        group.bench_with_input(
            BenchmarkId::new("large_game", format!("{}p_{}s", players, strategies)),
            &(players, strategies),
            |b, &(players, strategies)| {
                b.iter(|| {
                    let payoff_matrix = create_test_game(black_box(players), black_box(strategies));
                    let mut config = SolverConfig::default();
                    config.max_iterations = 50; // Limit for benchmarking
                    config.batch_size = 512;
                    
                    match GpuNashSolver::new(0, payoff_matrix, config) {
                        Ok(mut solver) => {
                            let result = solver.solve();
                            black_box(result)
                        }
                        Err(e) => black_box(Err(e)),
                    }
                });
            },
        );
    }
    
    group.finish();
}

fn benchmark_convergence_rates(c: &mut Criterion) {
    let _ = initialize();
    
    let mut group = c.benchmark_group("convergence_rates");
    
    for &tolerance in [1e-3, 1e-6, 1e-9].iter() {
        group.bench_with_input(
            BenchmarkId::new("tolerance", format!("{:.0e}", tolerance)),
            &tolerance,
            |b, &tolerance| {
                b.iter(|| {
                    let payoff_matrix = create_test_game(2, 4);
                    let mut config = SolverConfig::default();
                    config.tolerance = black_box(tolerance);
                    config.max_iterations = 1000;
                    
                    match GpuNashSolver::new(0, payoff_matrix, config) {
                        Ok(mut solver) => {
                            let result = solver.solve();
                            black_box(result)
                        }
                        Err(e) => black_box(Err(e)),
                    }
                });
            },
        );
    }
    
    group.finish();
}

fn benchmark_parallel_solving(c: &mut Criterion) {
    let _ = initialize();
    
    c.bench_function("parallel_nash_solving", |b| {
        b.iter(|| {
            let payoff_matrix = create_test_game(black_box(3), black_box(4));
            let mut config = SolverConfig::default();
            config.batch_size = 2048; // Large batch for parallel processing
            config.max_iterations = 100;
            
            match GpuNashSolver::new(0, payoff_matrix, config) {
                Ok(mut solver) => {
                    let result = solver.solve();
                    black_box(result)
                }
                Err(e) => black_box(Err(e)),
            }
        });
    });
}

criterion_group!(
    benches,
    benchmark_nash_solver_creation,
    benchmark_nash_algorithms,
    benchmark_quantum_nash_enhancement,
    benchmark_large_games,
    benchmark_convergence_rates,
    benchmark_parallel_solving
);
criterion_main!(benches);