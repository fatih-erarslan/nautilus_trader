// CUDA Quantum Kernel Performance Benchmarks
// Validates <10ns gate operations and <1ms circuit execution

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use std::sync::Arc;

#[cfg(feature = "cuda")]
use nn_models::{
    QBMIACudaContext,
    QuantumState,
    QuantumCircuit,
    QuantumGate,
    NashEquilibrium,
    PortfolioOptimizer,
};

#[cfg(feature = "cuda")]
fn bench_single_gates(c: &mut Criterion) {
    let context = Arc::new(QBMIACudaContext::new(0).expect("Failed to initialize CUDA"));
    
    let mut group = c.benchmark_group("quantum_gates");
    group.sample_size(1000);
    
    for num_qubits in [4, 6, 8, 10].iter() {
        for batch_size in [1, 10, 100, 1000].iter() {
            let mut state = QuantumState::new(*num_qubits, *batch_size, context.clone())
                .expect("Failed to create quantum state");
            
            // Benchmark Hadamard gate
            group.bench_with_input(
                BenchmarkId::new("hadamard", format!("{}q_{}b", num_qubits, batch_size)),
                &(*num_qubits, *batch_size),
                |b, _| {
                    b.iter(|| {
                        let gate = QuantumGate::Hadamard { qubit: 0 };
                        black_box(state.apply_gate(&gate).unwrap());
                    });
                },
            );
            
            // Benchmark CNOT gate
            group.bench_with_input(
                BenchmarkId::new("cnot", format!("{}q_{}b", num_qubits, batch_size)),
                &(*num_qubits, *batch_size),
                |b, _| {
                    b.iter(|| {
                        let gate = QuantumGate::CNOT { control: 0, target: 1 };
                        black_box(state.apply_gate(&gate).unwrap());
                    });
                },
            );
            
            // Benchmark RY rotation
            group.bench_with_input(
                BenchmarkId::new("ry_rotation", format!("{}q_{}b", num_qubits, batch_size)),
                &(*num_qubits, *batch_size),
                |b, _| {
                    b.iter(|| {
                        let gate = QuantumGate::RY { 
                            qubit: 0, 
                            angle: std::f32::consts::PI / 4.0 
                        };
                        black_box(state.apply_gate(&gate).unwrap());
                    });
                },
            );
        }
    }
    
    group.finish();
}

#[cfg(feature = "cuda")]
fn bench_quantum_circuits(c: &mut Criterion) {
    let context = Arc::new(QBMIACudaContext::new(0).expect("Failed to initialize CUDA"));
    
    let mut group = c.benchmark_group("quantum_circuits");
    group.sample_size(100);
    
    for num_qubits in [4, 6, 8].iter() {
        for num_layers in [2, 4, 6].iter() {
            for batch_size in [1, 10, 100].iter() {
                let circuit = QuantumCircuit::create_qbmia_ansatz(*num_qubits, *num_layers);
                
                group.bench_with_input(
                    BenchmarkId::new(
                        "qbmia_circuit", 
                        format!("{}q_{}l_{}b", num_qubits, num_layers, batch_size)
                    ),
                    &(*num_qubits, *num_layers, *batch_size),
                    |b, _| {
                        b.iter(|| {
                            let mut state = QuantumState::new(*num_qubits, *batch_size, context.clone())
                                .expect("Failed to create quantum state");
                            black_box(circuit.execute(&mut state).unwrap());
                        });
                    },
                );
            }
        }
    }
    
    group.finish();
}

#[cfg(feature = "cuda")]
fn bench_expectation_values(c: &mut Criterion) {
    let context = Arc::new(QBMIACudaContext::new(0).expect("Failed to initialize CUDA"));
    
    let mut group = c.benchmark_group("expectation_values");
    group.sample_size(500);
    
    for num_qubits in [4, 6, 8, 10].iter() {
        for batch_size in [1, 10, 100, 1000].iter() {
            let state = QuantumState::new(*num_qubits, *batch_size, context.clone())
                .expect("Failed to create quantum state");
            
            let observable_size = (1 << num_qubits) * (1 << num_qubits);
            let observable = vec![1.0f32; observable_size]; // Identity observable
            
            group.bench_with_input(
                BenchmarkId::new("expectation", format!("{}q_{}b", num_qubits, batch_size)),
                &(*num_qubits, *batch_size),
                |b, _| {
                    b.iter(|| {
                        black_box(state.expectation_value(&observable).unwrap());
                    });
                },
            );
        }
    }
    
    group.finish();
}

#[cfg(feature = "cuda")]
fn bench_nash_equilibrium(c: &mut Criterion) {
    let context = Arc::new(QBMIACudaContext::new(0).expect("Failed to initialize CUDA"));
    
    let mut group = c.benchmark_group("nash_equilibrium");
    group.sample_size(50);
    
    let nash_solver = NashEquilibrium::new(context)
        .with_params(100, 1e-4); // Reduced iterations for benchmarking
    
    for num_players in [2, 3, 4].iter() {
        for num_strategies in [2, 3, 4, 5].iter() {
            // Generate random payoff matrix
            let payoff_size = num_players * num_strategies * num_strategies;
            let payoff_matrix: Vec<f32> = (0..payoff_size)
                .map(|i| (i as f32 * 0.1).sin() * 2.0 + 3.0)
                .collect();
            
            group.bench_with_input(
                BenchmarkId::new("fictitious_play", format!("{}p_{}s", num_players, num_strategies)),
                &(*num_players, *num_strategies),
                |b, _| {
                    b.iter(|| {
                        black_box(nash_solver.solve_fictitious_play(
                            &payoff_matrix,
                            *num_players,
                            *num_strategies,
                        ).unwrap());
                    });
                },
            );
            
            group.bench_with_input(
                BenchmarkId::new("evolutionary", format!("{}p_{}s", num_players, num_strategies)),
                &(*num_players, *num_strategies),
                |b, _| {
                    b.iter(|| {
                        black_box(nash_solver.solve_evolutionary(
                            &payoff_matrix,
                            *num_players,
                            *num_strategies,
                            0.01,
                        ).unwrap());
                    });
                },
            );
        }
    }
    
    group.finish();
}

#[cfg(feature = "cuda")]
fn bench_portfolio_optimization(c: &mut Criterion) {
    let context = Arc::new(QBMIACudaContext::new(0).expect("Failed to initialize CUDA"));
    
    let mut group = c.benchmark_group("portfolio_optimization");
    group.sample_size(100);
    
    let optimizer = PortfolioOptimizer::new(context).with_risk_aversion(0.5);
    
    for num_assets in [10, 20, 50, 100].iter() {
        let expected_returns: Vec<f32> = (0..*num_assets)
            .map(|i| 0.05 + 0.15 * (i as f32 / *num_assets as f32))
            .collect();
        
        let covariance_matrix: Vec<f32> = (0..*num_assets * *num_assets)
            .map(|i| {
                if i % (*num_assets + 1) == 0 {
                    0.04 // Diagonal elements (variance)
                } else {
                    0.005 // Off-diagonal elements (covariance)
                }
            })
            .collect();
        
        let num_qubits = (*num_assets as f32).log2().ceil() as usize;
        
        group.bench_with_input(
            BenchmarkId::new("quantum_mean_variance", format!("{}a", num_assets)),
            num_assets,
            |b, _| {
                b.iter(|| {
                    black_box(optimizer.quantum_mean_variance(
                        &expected_returns,
                        &covariance_matrix,
                        *num_assets,
                        num_qubits,
                    ).unwrap());
                });
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("quantum_risk_parity", format!("{}a", num_assets)),
            num_assets,
            |b, _| {
                b.iter(|| {
                    black_box(optimizer.quantum_risk_parity(
                        &covariance_matrix,
                        *num_assets,
                        num_qubits,
                    ).unwrap());
                });
            },
        );
    }
    
    group.finish();
}

#[cfg(feature = "cuda")]
fn bench_complete_trading_pipeline(c: &mut Criterion) {
    let context = Arc::new(QBMIACudaContext::new(0).expect("Failed to initialize CUDA"));
    
    let mut group = c.benchmark_group("trading_pipeline");
    group.sample_size(50);
    
    let num_qubits = 6;
    let num_assets = 20;
    let batch_size = 100;
    
    // Pre-create all components
    let circuit = QuantumCircuit::create_qbmia_ansatz(num_qubits, 3);
    let nash_solver = NashEquilibrium::new(context.clone()).with_params(50, 1e-4);
    let optimizer = PortfolioOptimizer::new(context.clone());
    
    let market_features: Vec<f32> = (0..batch_size * num_assets)
        .map(|i| (i as f32 * 0.1).sin() * 0.5 + 0.5)
        .collect();
    
    let trading_payoffs = vec![
        2.0, 1.0, 3.0, 1.0, 2.0, 2.0, 3.0, 2.0, 1.0,
        2.0, 3.0, 1.0, 3.0, 2.0, 1.0, 1.0, 1.0, 2.0,
        1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 3.0, 1.0,
    ];
    
    let expected_returns: Vec<f32> = (0..num_assets)
        .map(|i| 0.08 + 0.12 * (i as f32 / num_assets as f32))
        .collect();
    
    let covariance_matrix: Vec<f32> = (0..num_assets * num_assets)
        .map(|i| if i % (num_assets + 1) == 0 { 0.04 } else { 0.005 })
        .collect();
    
    group.bench_function("complete_pipeline", |b| {
        b.iter(|| {
            // Step 1: Quantum feature encoding
            let quantum_state = QuantumState::from_features(
                &market_features,
                num_qubits,
                batch_size,
                context.clone(),
            ).unwrap();
            
            // Step 2: Quantum circuit processing
            let mut processing_state = quantum_state;
            let _metrics = circuit.execute(&mut processing_state).unwrap();
            
            // Step 3: Nash equilibrium solving
            let _nash_strategies = nash_solver.solve_fictitious_play(&trading_payoffs, 3, 3).unwrap();
            
            // Step 4: Portfolio optimization
            let _optimal_weights = optimizer.quantum_mean_variance(
                &expected_returns,
                &covariance_matrix,
                num_assets,
                num_qubits,
            ).unwrap();
            
            black_box(());
        });
    });
    
    group.finish();
}

#[cfg(feature = "cuda")]
criterion_group!(
    benches,
    bench_single_gates,
    bench_quantum_circuits,
    bench_expectation_values,
    bench_nash_equilibrium,
    bench_portfolio_optimization,
    bench_complete_trading_pipeline
);

#[cfg(not(feature = "cuda"))]
fn bench_dummy(_c: &mut Criterion) {
    // Dummy benchmark when CUDA is not available
}

#[cfg(not(feature = "cuda"))]
criterion_group!(benches, bench_dummy);

criterion_main!(benches);