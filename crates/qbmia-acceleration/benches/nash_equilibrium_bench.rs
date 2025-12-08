//! Nash Equilibrium Computation Benchmarks
//! 
//! Benchmarks for quantum-enhanced Nash equilibrium solving with GPU acceleration

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use qbmia_acceleration::nash::QuantumNashSolver;
use qbmia_acceleration::types::GameMatrix;
use std::time::Duration;

fn bench_small_game(c: &mut Criterion) {
    let solver = QuantumNashSolver::new();
    
    // 2x2 game matrix
    let game = GameMatrix {
        player1_payoffs: vec![vec![3.0, 0.0], vec![5.0, 1.0]],
        player2_payoffs: vec![vec![3.0, 5.0], vec![0.0, 1.0]],
    };
    
    c.bench_function("nash_2x2_game", |b| {
        b.iter(|| {
            let _result = solver.solve_quantum_nash(black_box(&game));
        })
    });
}

fn bench_medium_game(c: &mut Criterion) {
    let solver = QuantumNashSolver::new();
    
    // 5x5 game matrix
    let mut player1_payoffs = vec![];
    let mut player2_payoffs = vec![];
    
    for i in 0..5 {
        let mut row1 = vec![];
        let mut row2 = vec![];
        for j in 0..5 {
            row1.push((i + j) as f64);
            row2.push((i * j + 1) as f64);
        }
        player1_payoffs.push(row1);
        player2_payoffs.push(row2);
    }
    
    let game = GameMatrix {
        player1_payoffs,
        player2_payoffs,
    };
    
    c.bench_function("nash_5x5_game", |b| {
        b.iter(|| {
            let _result = solver.solve_quantum_nash(black_box(&game));
        })
    });
}

fn bench_large_game(c: &mut Criterion) {
    let solver = QuantumNashSolver::new();
    
    // 10x10 game matrix
    let mut player1_payoffs = vec![];
    let mut player2_payoffs = vec![];
    
    for i in 0..10 {
        let mut row1 = vec![];
        let mut row2 = vec![];
        for j in 0..10 {
            row1.push((i + j + i * j) as f64 / 10.0);
            row2.push((i * j + i + j + 1) as f64 / 10.0);
        }
        player1_payoffs.push(row1);
        player2_payoffs.push(row2);
    }
    
    let game = GameMatrix {
        player1_payoffs,
        player2_payoffs,
    };
    
    let mut group = c.benchmark_group("nash_large_games");
    group.measurement_time(Duration::from_secs(10));
    
    group.bench_function("nash_10x10_game", |b| {
        b.iter(|| {
            let _result = solver.solve_quantum_nash(black_box(&game));
        })
    });
    
    group.finish();
}

criterion_group!(benches, bench_small_game, bench_medium_game, bench_large_game);
criterion_main!(benches);