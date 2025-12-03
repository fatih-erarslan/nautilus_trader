//! Nash Equilibrium Benchmark for Game Theory Engine
//!
//! Benchmarks for Nash equilibrium solving algorithms based on:
//! - Nash, J. (1951). "Non-Cooperative Games". Annals of Mathematics. 54 (2): 286–295
//! - Lemke, C.E. & Howson, J.T. (1964). "Equilibrium Points of Bimatrix Games". SIAM Journal

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use game_theory_engine::{
    NashSolver, GameState, GameType, PayoffMatrix, MarketRegime, MarketContext,
    RegulatoryEnvironment, TransparencyLevel, EvolutionaryAnalyzer, Population,
};
use std::collections::HashMap;

/// Create a 2-player symmetric payoff matrix of given size
fn create_symmetric_payoff_matrix(size: usize) -> PayoffMatrix {
    let mut payoffs = HashMap::new();
    for i in 0..size {
        for j in 0..size {
            // Symmetric game: diagonal has high payoffs, off-diagonal varies
            let p1_payoff = if i == j { 3.0 } else { 1.0 + (i + j) as f64 * 0.1 };
            let p2_payoff = if i == j { 3.0 } else { 1.0 + (i + j) as f64 * 0.1 };
            payoffs.insert(format!("P1_payoff_{}_{}", i, j), p1_payoff);
            payoffs.insert(format!("P2_payoff_{}_{}", i, j), p2_payoff);
        }
    }

    let mut strategies = HashMap::new();
    let strats: Vec<String> = (0..size).map(|i| format!("S{}", i)).collect();
    strategies.insert("P1".to_string(), strats.clone());
    strategies.insert("P2".to_string(), strats);

    PayoffMatrix {
        players: vec!["P1".to_string(), "P2".to_string()],
        strategies,
        payoffs,
        dimension: vec![size, size],
    }
}

/// Create a zero-sum game payoff matrix
fn create_zero_sum_matrix(size: usize) -> PayoffMatrix {
    let mut payoffs = HashMap::new();
    for i in 0..size {
        for j in 0..size {
            // Zero-sum: P1 wins what P2 loses
            let p1_payoff = if i == j { 0.0 }
                else if (i + 1) % size == j { -1.0 }
                else { 1.0 };
            payoffs.insert(format!("P1_payoff_{}_{}", i, j), p1_payoff);
            payoffs.insert(format!("P2_payoff_{}_{}", i, j), -p1_payoff);
        }
    }

    let mut strategies = HashMap::new();
    let strats: Vec<String> = (0..size).map(|i| format!("S{}", i)).collect();
    strategies.insert("P1".to_string(), strats.clone());
    strategies.insert("P2".to_string(), strats);

    PayoffMatrix {
        players: vec!["P1".to_string(), "P2".to_string()],
        strategies,
        payoffs,
        dimension: vec![size, size],
    }
}

/// Create a game state with the given payoff matrix
fn create_game_state(game_type: GameType, payoff_matrix: PayoffMatrix) -> GameState {
    GameState {
        game_type,
        players: vec![],
        market_context: MarketContext {
            regime: MarketRegime::LowVolatility,
            volatility: 0.1,
            liquidity: 1_000_000.0,
            volume: 500_000.0,
            spread: 0.01,
            market_impact: 0.001,
            information_asymmetry: 0.1,
            regulatory_environment: RegulatoryEnvironment {
                short_selling_allowed: true,
                position_limits: None,
                circuit_breakers: true,
                market_making_obligations: false,
                transparency_requirements: TransparencyLevel::Full,
            },
        },
        information_sets: HashMap::new(),
        action_history: vec![],
        current_round: 0,
        payoff_matrix: Some(payoff_matrix),
        nash_equilibria: vec![],
        nash_equilibrium_found: false,
        dominant_strategies: HashMap::new(),
        cooperation_level: 0.5,
        competition_intensity: 0.7,
    }
}

fn benchmark_nash_solver_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("nash_solver_creation");

    let configs = [
        (1e-6, 100),
        (1e-9, 1000),
        (1e-12, 10000),
    ];

    for (tolerance, max_iter) in configs {
        group.bench_with_input(
            BenchmarkId::new("new", format!("tol{:.0e}_iter{}", tolerance, max_iter)),
            &(tolerance, max_iter),
            |b, &(tol, max)| {
                b.iter(|| black_box(NashSolver::new(black_box(tol), black_box(max))))
            },
        );
    }

    group.finish();
}

fn benchmark_pure_nash_equilibrium(c: &mut Criterion) {
    let mut group = c.benchmark_group("pure_nash_equilibrium");

    let solver = NashSolver::new(1e-9, 1000);

    for size in [2, 3, 4, 5] {
        let game_state = create_game_state(
            GameType::PrisonersDilemma,
            create_symmetric_payoff_matrix(size),
        );

        group.throughput(Throughput::Elements(size as u64 * size as u64));
        group.bench_with_input(
            BenchmarkId::new("symmetric", size),
            &size,
            |b, _| {
                b.iter(|| solver.find_pure_nash(black_box(&game_state)))
            },
        );
    }

    // Zero-sum games (usually no pure equilibrium)
    for size in [2, 3, 4] {
        let game_state = create_game_state(
            GameType::MatchingPennies,
            create_zero_sum_matrix(size),
        );

        group.bench_with_input(
            BenchmarkId::new("zero_sum", size),
            &size,
            |b, _| {
                b.iter(|| solver.find_pure_nash(black_box(&game_state)))
            },
        );
    }

    group.finish();
}

fn benchmark_mixed_nash_equilibrium(c: &mut Criterion) {
    let mut group = c.benchmark_group("mixed_nash_equilibrium");

    let solver = NashSolver::new(1e-9, 1000);

    // 2x2 games - classical case
    let game_2x2 = create_game_state(GameType::MatchingPennies, create_zero_sum_matrix(2));
    group.bench_function("2x2_zero_sum", |b| {
        b.iter(|| solver.find_mixed_nash(black_box(&game_2x2)))
    });

    // 2x2 symmetric coordination game
    let game_coord = create_game_state(GameType::StagHunt, create_symmetric_payoff_matrix(2));
    group.bench_function("2x2_coordination", |b| {
        b.iter(|| solver.find_mixed_nash(black_box(&game_coord)))
    });

    // Larger games (3x3, 4x4) - support enumeration scales combinatorially
    for size in [3, 4] {
        let game_state = create_game_state(
            GameType::RockPaperScissors,
            create_zero_sum_matrix(size),
        );

        group.bench_with_input(
            BenchmarkId::new("nxn_zero_sum", size),
            &size,
            |b, _| {
                b.iter(|| solver.find_mixed_nash(black_box(&game_state)))
            },
        );
    }

    group.finish();
}

fn benchmark_full_nash_solve(c: &mut Criterion) {
    let mut group = c.benchmark_group("full_nash_solve");

    let solver = NashSolver::new(1e-9, 1000);

    // Standard game types
    let games = [
        ("prisoners_dilemma_2x2", GameType::PrisonersDilemma, create_symmetric_payoff_matrix(2)),
        ("matching_pennies_2x2", GameType::MatchingPennies, create_zero_sum_matrix(2)),
        ("coordination_3x3", GameType::StagHunt, create_symmetric_payoff_matrix(3)),
        ("rps_3x3", GameType::RockPaperScissors, create_zero_sum_matrix(3)),
    ];

    for (name, game_type, payoff_matrix) in games {
        let game_state = create_game_state(game_type, payoff_matrix);

        group.bench_function(name, |b| {
            b.iter(|| solver.solve(black_box(&game_state)))
        });
    }

    group.finish();
}

fn benchmark_solver_tolerance_impact(c: &mut Criterion) {
    let mut group = c.benchmark_group("tolerance_impact");

    let game_state = create_game_state(GameType::MatchingPennies, create_zero_sum_matrix(2));

    for tolerance in [1e-3, 1e-6, 1e-9, 1e-12] {
        let solver = NashSolver::new(tolerance, 1000);

        group.bench_with_input(
            BenchmarkId::new("solve", format!("{:.0e}", tolerance)),
            &tolerance,
            |b, _| {
                b.iter(|| solver.solve(black_box(&game_state)))
            },
        );
    }

    group.finish();
}

fn benchmark_evolutionary_analyzer(c: &mut Criterion) {
    let mut group = c.benchmark_group("evolutionary_analyzer");

    let configs = [
        (100, 0.01, 1.0),
        (500, 0.01, 1.0),
        (1000, 0.01, 1.0),
    ];

    for (pop_size, mutation_rate, selection_pressure) in configs {
        group.bench_with_input(
            BenchmarkId::new("create", pop_size),
            &(pop_size, mutation_rate, selection_pressure),
            |b, &(ps, mr, sp)| {
                b.iter(|| black_box(EvolutionaryAnalyzer::new(black_box(ps), black_box(mr), black_box(sp))))
            },
        );
    }

    // Find ESS
    let analyzer = EvolutionaryAnalyzer::new(100, 0.01, 1.0);
    let game_state = create_game_state(GameType::HawkDove, create_symmetric_payoff_matrix(2));

    group.bench_function("find_ess", |b| {
        b.iter(|| analyzer.find_ess(black_box(&game_state)))
    });

    // Simulate evolution
    let initial_population = Population {
        strategies: [("Hawk".to_string(), 0.5), ("Dove".to_string(), 0.5)]
            .into_iter()
            .collect(),
        fitness: [("Hawk".to_string(), 1.0), ("Dove".to_string(), 1.0)]
            .into_iter()
            .collect(),
    };

    for generations in [10, 100, 500] {
        group.bench_with_input(
            BenchmarkId::new("simulate_evolution", generations),
            &generations,
            |b, &gen| {
                b.iter(|| analyzer.simulate_evolution(black_box(&initial_population), black_box(gen)))
            },
        );
    }

    group.finish();
}

fn benchmark_payoff_matrix_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("payoff_matrix_creation");

    for size in [2, 3, 4, 5, 8, 10] {
        group.throughput(Throughput::Elements((size * size) as u64));

        group.bench_with_input(
            BenchmarkId::new("symmetric", size),
            &size,
            |b, &s| {
                b.iter(|| black_box(create_symmetric_payoff_matrix(black_box(s))))
            },
        );

        group.bench_with_input(
            BenchmarkId::new("zero_sum", size),
            &size,
            |b, &s| {
                b.iter(|| black_box(create_zero_sum_matrix(black_box(s))))
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    benchmark_nash_solver_creation,
    benchmark_pure_nash_equilibrium,
    benchmark_mixed_nash_equilibrium,
    benchmark_full_nash_solve,
    benchmark_solver_tolerance_impact,
    benchmark_evolutionary_analyzer,
    benchmark_payoff_matrix_creation,
);

criterion_main!(benches);
