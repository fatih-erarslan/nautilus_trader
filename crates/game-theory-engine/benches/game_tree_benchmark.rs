use criterion::{black_box, criterion_group, criterion_main, Criterion};
use game_theory_engine::tree::GameTree;
use game_theory_engine::solver::GameSolver;
use game_theory_engine::config::GameTheoryConfig;

fn game_tree_benchmark(c: &mut Criterion) {
    let config = GameTheoryConfig::default();
    let tree = GameTree::new(config);
    let game_state = vec![1, 2, 3, 4, 5];
    
    c.bench_function("game_tree", |b| {
        b.iter(|| {
            tree.build_tree(black_box(&game_state))
        })
    });
}

fn game_solver_benchmark(c: &mut Criterion) {
    let solver = GameSolver::new();
    let payoff_matrix = vec![
        vec![3.0, 1.0, 4.0],
        vec![2.0, 5.0, 1.0],
        vec![4.0, 2.0, 3.0],
    ];
    
    c.bench_function("game_solver", |b| {
        b.iter(|| {
            solver.solve_game(black_box(&payoff_matrix))
        })
    });
}

criterion_group!(benches, game_tree_benchmark, game_solver_benchmark);
criterion_main!(benches);