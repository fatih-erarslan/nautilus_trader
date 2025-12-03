//! Performance benchmarks for Game Theory Engine
//!
//! Benchmarks the core components:
//! - GameTree operations (construction, traversal, MCTS)
//! - NashSolver equilibrium finding (pure and mixed)
//!
//! Based on:
//! - Nash, J. (1951). "Non-Cooperative Games". Annals of Mathematics. 54 (2): 286–295
//! - Lemke, C.E. & Howson, J.T. (1964). "Equilibrium Points of Bimatrix Games". SIAM Journal

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use game_theory_engine::{
    GameTree, GameNode, NashSolver,
    GameState, GameType, PayoffMatrix, MarketRegime, MarketContext,
    RegulatoryEnvironment, TransparencyLevel,
};
use std::collections::HashMap;

/// Create a Prisoner's Dilemma payoff matrix for benchmarking
fn create_prisoners_dilemma() -> PayoffMatrix {
    // Classic Prisoner's Dilemma:
    //           Cooperate  Defect
    // Cooperate  (-1,-1)   (-3, 0)
    // Defect     ( 0,-3)   (-2,-2)
    let mut payoffs = HashMap::new();
    payoffs.insert("P1_payoff_0_0".to_string(), -1.0);
    payoffs.insert("P1_payoff_0_1".to_string(), -3.0);
    payoffs.insert("P1_payoff_1_0".to_string(), 0.0);
    payoffs.insert("P1_payoff_1_1".to_string(), -2.0);
    payoffs.insert("P2_payoff_0_0".to_string(), -1.0);
    payoffs.insert("P2_payoff_0_1".to_string(), 0.0);
    payoffs.insert("P2_payoff_1_0".to_string(), -3.0);
    payoffs.insert("P2_payoff_1_1".to_string(), -2.0);

    let mut strategies = HashMap::new();
    strategies.insert("P1".to_string(), vec!["Cooperate".to_string(), "Defect".to_string()]);
    strategies.insert("P2".to_string(), vec!["Cooperate".to_string(), "Defect".to_string()]);

    PayoffMatrix {
        players: vec!["P1".to_string(), "P2".to_string()],
        strategies,
        payoffs,
        dimension: vec![2, 2],
    }
}

/// Create a Matching Pennies payoff matrix (zero-sum game with only mixed equilibrium)
fn create_matching_pennies() -> PayoffMatrix {
    // Matching Pennies (zero-sum):
    //           Heads   Tails
    // Heads     (1,-1)  (-1,1)
    // Tails     (-1,1)  (1,-1)
    let mut payoffs = HashMap::new();
    payoffs.insert("P1_payoff_0_0".to_string(), 1.0);
    payoffs.insert("P1_payoff_0_1".to_string(), -1.0);
    payoffs.insert("P1_payoff_1_0".to_string(), -1.0);
    payoffs.insert("P1_payoff_1_1".to_string(), 1.0);
    payoffs.insert("P2_payoff_0_0".to_string(), -1.0);
    payoffs.insert("P2_payoff_0_1".to_string(), 1.0);
    payoffs.insert("P2_payoff_1_0".to_string(), 1.0);
    payoffs.insert("P2_payoff_1_1".to_string(), -1.0);

    let mut strategies = HashMap::new();
    strategies.insert("P1".to_string(), vec!["Heads".to_string(), "Tails".to_string()]);
    strategies.insert("P2".to_string(), vec!["Heads".to_string(), "Tails".to_string()]);

    PayoffMatrix {
        players: vec!["P1".to_string(), "P2".to_string()],
        strategies,
        payoffs,
        dimension: vec![2, 2],
    }
}

/// Create a 3x3 game for larger benchmarks
fn create_3x3_game() -> PayoffMatrix {
    // Rock-Paper-Scissors variant with asymmetric payoffs
    let mut payoffs = HashMap::new();
    for i in 0..3 {
        for j in 0..3 {
            // Player 1 payoff
            let p1_payoff = if i == j { 0.0 }
                else if (i + 1) % 3 == j { -1.0 }
                else { 1.0 };
            // Player 2 payoff (zero-sum)
            let p2_payoff = -p1_payoff;
            payoffs.insert(format!("P1_payoff_{}_{}", i, j), p1_payoff);
            payoffs.insert(format!("P2_payoff_{}_{}", i, j), p2_payoff);
        }
    }

    let mut strategies = HashMap::new();
    strategies.insert("P1".to_string(), vec!["Rock".to_string(), "Paper".to_string(), "Scissors".to_string()]);
    strategies.insert("P2".to_string(), vec!["Rock".to_string(), "Paper".to_string(), "Scissors".to_string()]);

    PayoffMatrix {
        players: vec!["P1".to_string(), "P2".to_string()],
        strategies,
        payoffs,
        dimension: vec![3, 3],
    }
}

/// Create a default game state with given payoff matrix
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

fn bench_game_tree_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("game_tree_creation");

    group.bench_function("new_game_tree", |b| {
        b.iter(|| {
            black_box(GameTree::new())
        })
    });

    group.bench_function("game_node_root", |b| {
        b.iter(|| {
            black_box(GameNode::new_root())
        })
    });

    group.finish();
}

fn bench_game_tree_build(c: &mut Criterion) {
    let mut group = c.benchmark_group("game_tree_build");

    let game_state = create_game_state(GameType::PrisonersDilemma, create_prisoners_dilemma());

    for depth in [1, 3, 5, 7] {
        group.bench_with_input(
            BenchmarkId::new("build_from_state", depth),
            &depth,
            |b, &depth| {
                let mut tree = GameTree::new();
                b.iter(|| {
                    tree.build_from_state(black_box(&game_state), black_box(depth))
                })
            },
        );
    }

    group.finish();
}

fn bench_game_tree_mcts(c: &mut Criterion) {
    let mut group = c.benchmark_group("game_tree_mcts");

    let game_state = create_game_state(GameType::PrisonersDilemma, create_prisoners_dilemma());
    let mut tree = GameTree::new();
    let _ = tree.build_from_state(&game_state, 3);

    for iterations in [10, 100, 500] {
        group.throughput(Throughput::Elements(iterations as u64));
        group.bench_with_input(
            BenchmarkId::new("monte_carlo_tree_search", iterations),
            &iterations,
            |b, &iterations| {
                b.iter(|| {
                    tree.monte_carlo_tree_search(black_box(iterations))
                })
            },
        );
    }

    group.finish();
}

fn bench_nash_solver_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("nash_solver_creation");

    let tolerances = [1e-6, 1e-9, 1e-12];
    let max_iterations = [100, 1000, 10000];

    for &tolerance in &tolerances {
        for &max_iter in &max_iterations {
            group.bench_with_input(
                BenchmarkId::new(format!("tol_{:.0e}", tolerance), max_iter),
                &(tolerance, max_iter),
                |b, &(tolerance, max_iter)| {
                    b.iter(|| {
                        black_box(NashSolver::new(black_box(tolerance), black_box(max_iter)))
                    })
                },
            );
        }
    }

    group.finish();
}

fn bench_nash_solver_pure(c: &mut Criterion) {
    let mut group = c.benchmark_group("nash_solver_pure_equilibrium");

    let solver = NashSolver::new(1e-9, 1000);

    // Prisoner's Dilemma - has one pure Nash equilibrium
    let pd_state = create_game_state(GameType::PrisonersDilemma, create_prisoners_dilemma());
    group.bench_function("prisoners_dilemma", |b| {
        b.iter(|| {
            solver.find_pure_nash(black_box(&pd_state))
        })
    });

    // Matching Pennies - has no pure Nash equilibrium
    let mp_state = create_game_state(GameType::MatchingPennies, create_matching_pennies());
    group.bench_function("matching_pennies", |b| {
        b.iter(|| {
            solver.find_pure_nash(black_box(&mp_state))
        })
    });

    // 3x3 game (Rock-Paper-Scissors)
    let rps_state = create_game_state(GameType::RockPaperScissors, create_3x3_game());
    group.bench_function("rock_paper_scissors_3x3", |b| {
        b.iter(|| {
            solver.find_pure_nash(black_box(&rps_state))
        })
    });

    group.finish();
}

fn bench_nash_solver_mixed(c: &mut Criterion) {
    let mut group = c.benchmark_group("nash_solver_mixed_equilibrium");

    let solver = NashSolver::new(1e-9, 1000);

    // Matching Pennies - has unique mixed Nash equilibrium (0.5, 0.5)
    let mp_state = create_game_state(GameType::MatchingPennies, create_matching_pennies());
    group.bench_function("matching_pennies_mixed", |b| {
        b.iter(|| {
            solver.find_mixed_nash(black_box(&mp_state))
        })
    });

    // 3x3 game (Rock-Paper-Scissors) - mixed equilibrium (1/3, 1/3, 1/3)
    let rps_state = create_game_state(GameType::RockPaperScissors, create_3x3_game());
    group.bench_function("rock_paper_scissors_mixed", |b| {
        b.iter(|| {
            solver.find_mixed_nash(black_box(&rps_state))
        })
    });

    group.finish();
}

fn bench_nash_solver_full(c: &mut Criterion) {
    let mut group = c.benchmark_group("nash_solver_full_solve");

    let solver = NashSolver::new(1e-9, 1000);

    // Full solve includes both pure and mixed equilibria
    let pd_state = create_game_state(GameType::PrisonersDilemma, create_prisoners_dilemma());
    group.bench_function("prisoners_dilemma_full", |b| {
        b.iter(|| {
            solver.solve(black_box(&pd_state))
        })
    });

    let mp_state = create_game_state(GameType::MatchingPennies, create_matching_pennies());
    group.bench_function("matching_pennies_full", |b| {
        b.iter(|| {
            solver.solve(black_box(&mp_state))
        })
    });

    let rps_state = create_game_state(GameType::RockPaperScissors, create_3x3_game());
    group.bench_function("rock_paper_scissors_full", |b| {
        b.iter(|| {
            solver.solve(black_box(&rps_state))
        })
    });

    group.finish();
}

fn bench_game_node_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("game_node_operations");

    // Create a node with children
    let mut root = GameNode::new_root();
    for i in 0..10 {
        let mut child = GameNode::new_root();
        child.value = i as f64 * 0.1;
        root.children.push(child);
    }

    group.bench_function("is_terminal_leaf", |b| {
        let leaf = GameNode::new_root();
        b.iter(|| {
            black_box(leaf.is_terminal())
        })
    });

    group.bench_function("is_terminal_internal", |b| {
        b.iter(|| {
            black_box(root.is_terminal())
        })
    });

    group.bench_function("best_child_10_children", |b| {
        b.iter(|| {
            black_box(root.best_child())
        })
    });

    // Create node with more children
    let mut large_root = GameNode::new_root();
    for i in 0..100 {
        let mut child = GameNode::new_root();
        child.value = (i as f64) * 0.01;
        large_root.children.push(child);
    }

    group.bench_function("best_child_100_children", |b| {
        b.iter(|| {
            black_box(large_root.best_child())
        })
    });

    group.finish();
}

fn bench_tolerance_impact(c: &mut Criterion) {
    let mut group = c.benchmark_group("tolerance_impact_on_solver");

    let mp_state = create_game_state(GameType::MatchingPennies, create_matching_pennies());

    for tolerance in [1e-3, 1e-6, 1e-9, 1e-12] {
        let solver = NashSolver::new(tolerance, 1000);
        group.bench_with_input(
            BenchmarkId::new("solve", format!("{:.0e}", tolerance)),
            &tolerance,
            |b, _| {
                b.iter(|| {
                    solver.solve(black_box(&mp_state))
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_game_tree_creation,
    bench_game_tree_build,
    bench_game_tree_mcts,
    bench_nash_solver_creation,
    bench_nash_solver_pure,
    bench_nash_solver_mixed,
    bench_nash_solver_full,
    bench_game_node_operations,
    bench_tolerance_impact,
);

criterion_main!(benches);
