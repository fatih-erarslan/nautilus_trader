use criterion::{black_box, criterion_group, criterion_main, Criterion};
use quantum_backend::nash::NashSolver;
use quantum_backend::game::GameTheory;
use quantum_backend::config::QuantumBackendConfig;

fn nash_solver_benchmark(c: &mut Criterion) {
    let config = QuantumBackendConfig::default();
    let solver = NashSolver::new(config);
    let payoff_matrix = vec![
        vec![3.0, 1.0],
        vec![0.0, 2.0],
    ];
    
    c.bench_function("nash_solver", |b| {
        b.iter(|| {
            solver.find_nash_equilibrium(black_box(&payoff_matrix))
        })
    });
}

fn game_theory_benchmark(c: &mut Criterion) {
    let game = GameTheory::new();
    let strategies = vec![0.5, 0.3, 0.2];
    let opponent_strategies = vec![0.4, 0.6];
    
    c.bench_function("game_theory", |b| {
        b.iter(|| {
            game.calculate_expected_payoff(
                black_box(&strategies),
                black_box(&opponent_strategies)
            )
        })
    });
}

criterion_group!(benches, nash_solver_benchmark, game_theory_benchmark);
criterion_main!(benches);