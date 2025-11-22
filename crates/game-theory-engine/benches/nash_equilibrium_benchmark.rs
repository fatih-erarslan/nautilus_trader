// Nash Equilibrium Benchmark for Game Theory Engine
// Copyright (c) 2025 TENGRI Trading Swarm

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use game_theory_engine::{
    NashEquilibrium, GameTheory, StrategyProfile, PayoffMatrix, Player,
    PureStrategy, MixedStrategy, EvolutionaryStableStrategy, Replicator,
    IteratedElimination, BestResponse, CoreSolution, ShapleyValue,
    CoalitionalGame, CooperativeGame, NonCooperativeGame, GameSolver
};

fn create_players(count: usize) -> Vec<Player> {
    (0..count).map(|i| Player {
        id: i,
        strategy_space: vec![0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        payoff_function: Box::new(move |strategies: &[f64]| {
            strategies.iter().sum::<f64>() * (1.0 + i as f64 * 0.1)
        }),
        rationality_level: 0.8 + (i as f64 * 0.02),
        risk_preference: 0.1 + (i as f64 * 0.05),
    }).collect()
}

fn create_payoff_matrix(size: usize) -> PayoffMatrix {
    PayoffMatrix::new(
        (0..size).map(|i| {
            (0..size).map(|j| {
                if i == j { 3.0 } else { 1.0 + (i + j) as f64 * 0.1 }
            }).collect()
        }).collect()
    )
}

fn benchmark_nash_equilibrium_computation(c: &mut Criterion) {
    let mut group = c.benchmark_group("nash_equilibrium_computation");
    
    for player_count in [2, 3, 4, 5, 8].iter() {
        let players = create_players(*player_count);
        let payoff_matrix = create_payoff_matrix(*player_count);
        let nash_solver = NashEquilibrium::new();
        
        group.bench_with_input(BenchmarkId::new("compute_pure_nash", player_count), player_count, |b, _| {
            b.iter(|| {
                nash_solver.compute_pure_nash_equilibrium(&players, &payoff_matrix)
            })
        });
        
        group.bench_with_input(BenchmarkId::new("compute_mixed_nash", player_count), player_count, |b, _| {
            b.iter(|| {
                nash_solver.compute_mixed_nash_equilibrium(&players, &payoff_matrix)
            })
        });
    }
    group.finish();
}

fn benchmark_strategy_profiles(c: &mut Criterion) {
    let mut group = c.benchmark_group("strategy_profiles");
    
    let players = create_players(4);
    let payoff_matrix = create_payoff_matrix(4);
    let profile_analyzer = StrategyProfile::new();
    
    group.bench_function("pure_strategy_profile", |b| {
        b.iter(|| {
            profile_analyzer.analyze_pure_strategy_profile(&players, &payoff_matrix)
        })
    });
    
    group.bench_function("mixed_strategy_profile", |b| {
        b.iter(|| {
            profile_analyzer.analyze_mixed_strategy_profile(&players, &payoff_matrix)
        })
    });
    
    group.bench_function("symmetric_strategy_profile", |b| {
        b.iter(|| {
            profile_analyzer.analyze_symmetric_strategy_profile(&players, &payoff_matrix)
        })
    });
    
    group.finish();
}

fn benchmark_evolutionary_stable_strategies(c: &mut Criterion) {
    let mut group = c.benchmark_group("evolutionary_stable_strategies");
    
    let players = create_players(6);
    let payoff_matrix = create_payoff_matrix(6);
    let ess_analyzer = EvolutionaryStableStrategy::new();
    
    group.bench_function("find_ess", |b| {
        b.iter(|| {
            ess_analyzer.find_evolutionary_stable_strategies(&players, &payoff_matrix)
        })
    });
    
    group.bench_function("stability_analysis", |b| {
        b.iter(|| {
            ess_analyzer.analyze_strategy_stability(&players, &payoff_matrix)
        })
    });
    
    group.bench_function("invasion_analysis", |b| {
        b.iter(|| {
            ess_analyzer.analyze_invasion_resistance(&players, &payoff_matrix)
        })
    });
    
    group.finish();
}

fn benchmark_replicator_dynamics(c: &mut Criterion) {
    let mut group = c.benchmark_group("replicator_dynamics");
    
    let players = create_players(5);
    let payoff_matrix = create_payoff_matrix(5);
    let replicator = Replicator::new();
    
    group.bench_function("simulate_dynamics", |b| {
        b.iter(|| {
            replicator.simulate_replicator_dynamics(&players, &payoff_matrix, 1000)
        })
    });
    
    group.bench_function("find_fixed_points", |b| {
        b.iter(|| {
            replicator.find_fixed_points(&players, &payoff_matrix)
        })
    });
    
    group.bench_function("stability_analysis", |b| {
        b.iter(|| {
            replicator.analyze_dynamic_stability(&players, &payoff_matrix)
        })
    });
    
    group.finish();
}

fn benchmark_iterated_elimination(c: &mut Criterion) {
    let mut group = c.benchmark_group("iterated_elimination");
    
    let players = create_players(4);
    let payoff_matrix = create_payoff_matrix(4);
    let eliminator = IteratedElimination::new();
    
    group.bench_function("eliminate_dominated_strategies", |b| {
        b.iter(|| {
            eliminator.eliminate_dominated_strategies(&players, &payoff_matrix)
        })
    });
    
    group.bench_function("eliminate_weakly_dominated", |b| {
        b.iter(|| {
            eliminator.eliminate_weakly_dominated_strategies(&players, &payoff_matrix)
        })
    });
    
    group.bench_function("iterative_elimination", |b| {
        b.iter(|| {
            eliminator.iterative_elimination_process(&players, &payoff_matrix)
        })
    });
    
    group.finish();
}

fn benchmark_best_response_analysis(c: &mut Criterion) {
    let mut group = c.benchmark_group("best_response_analysis");
    
    let players = create_players(4);
    let payoff_matrix = create_payoff_matrix(4);
    let best_response = BestResponse::new();
    
    group.bench_function("compute_best_response", |b| {
        b.iter(|| {
            best_response.compute_best_response(&players, &payoff_matrix)
        })
    });
    
    group.bench_function("best_response_dynamics", |b| {
        b.iter(|| {
            best_response.simulate_best_response_dynamics(&players, &payoff_matrix, 500)
        })
    });
    
    group.bench_function("rationalizability_analysis", |b| {
        b.iter(|| {
            best_response.analyze_rationalizability(&players, &payoff_matrix)
        })
    });
    
    group.finish();
}

fn benchmark_cooperative_games(c: &mut Criterion) {
    let mut group = c.benchmark_group("cooperative_games");
    
    let players = create_players(5);
    let cooperative_game = CooperativeGame::new();
    
    group.bench_function("compute_core", |b| {
        b.iter(|| {
            cooperative_game.compute_core(&players)
        })
    });
    
    group.bench_function("compute_shapley_value", |b| {
        b.iter(|| {
            cooperative_game.compute_shapley_value(&players)
        })
    });
    
    group.bench_function("compute_nucleolus", |b| {
        b.iter(|| {
            cooperative_game.compute_nucleolus(&players)
        })
    });
    
    group.finish();
}

fn benchmark_coalitional_games(c: &mut Criterion) {
    let mut group = c.benchmark_group("coalitional_games");
    
    let players = create_players(6);
    let coalitional_game = CoalitionalGame::new();
    
    group.bench_function("analyze_coalitions", |b| {
        b.iter(|| {
            coalitional_game.analyze_coalition_formation(&players)
        })
    });
    
    group.bench_function("stability_analysis", |b| {
        b.iter(|| {
            coalitional_game.analyze_coalition_stability(&players)
        })
    });
    
    group.bench_function("bargaining_analysis", |b| {
        b.iter(|| {
            coalitional_game.analyze_bargaining_power(&players)
        })
    });
    
    group.finish();
}

fn benchmark_game_solver_algorithms(c: &mut Criterion) {
    let mut group = c.benchmark_group("game_solver_algorithms");
    
    let players = create_players(4);
    let payoff_matrix = create_payoff_matrix(4);
    let solver = GameSolver::new();
    
    group.bench_function("lemke_howson_algorithm", |b| {
        b.iter(|| {
            solver.lemke_howson_algorithm(&players, &payoff_matrix)
        })
    });
    
    group.bench_function("support_enumeration", |b| {
        b.iter(|| {
            solver.support_enumeration_algorithm(&players, &payoff_matrix)
        })
    });
    
    group.bench_function("fictitious_play", |b| {
        b.iter(|| {
            solver.fictitious_play_algorithm(&players, &payoff_matrix, 1000)
        })
    });
    
    group.finish();
}

fn benchmark_mechanism_design_games(c: &mut Criterion) {
    let mut group = c.benchmark_group("mechanism_design_games");
    
    let players = create_players(5);
    let game_theory = GameTheory::new();
    
    group.bench_function("auction_mechanism", |b| {
        b.iter(|| {
            game_theory.analyze_auction_mechanism(&players)
        })
    });
    
    group.bench_function("voting_mechanism", |b| {
        b.iter(|| {
            game_theory.analyze_voting_mechanism(&players)
        })
    });
    
    group.bench_function("contract_mechanism", |b| {
        b.iter(|| {
            game_theory.analyze_contract_mechanism(&players)
        })
    });
    
    group.finish();
}

fn benchmark_real_time_game_analysis(c: &mut Criterion) {
    let mut group = c.benchmark_group("real_time_game_analysis");
    
    let nash_solver = NashEquilibrium::new();
    
    for batch_size in [2, 4, 8, 16].iter() {
        let players = create_players(*batch_size);
        let payoff_matrix = create_payoff_matrix(*batch_size);
        
        group.bench_with_input(BenchmarkId::new("streaming_analysis", batch_size), batch_size, |b, _| {
            b.iter(|| {
                nash_solver.streaming_game_analysis(&players, &payoff_matrix)
            })
        });
    }
    
    group.bench_function("low_latency_equilibrium", |b| {
        b.iter(|| {
            let players = create_players(2);
            let payoff_matrix = create_payoff_matrix(2);
            nash_solver.low_latency_equilibrium_computation(&players, &payoff_matrix)
        })
    });
    
    group.finish();
}

criterion_group!(
    benches,
    benchmark_nash_equilibrium_computation,
    benchmark_strategy_profiles,
    benchmark_evolutionary_stable_strategies,
    benchmark_replicator_dynamics,
    benchmark_iterated_elimination,
    benchmark_best_response_analysis,
    benchmark_cooperative_games,
    benchmark_coalitional_games,
    benchmark_game_solver_algorithms,
    benchmark_mechanism_design_games,
    benchmark_real_time_game_analysis
);
criterion_main!(benches);