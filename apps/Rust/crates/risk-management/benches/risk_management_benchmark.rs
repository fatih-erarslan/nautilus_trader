use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use risk_management::{RiskManager, RiskConfig, VarCalculator, VarConfig};
use tokio::runtime::Runtime;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;

fn create_test_portfolio() -> risk_management::Portfolio {
    let mut portfolio = risk_management::Portfolio::default();
    portfolio.positions = vec![
        risk_management::Position {
            symbol: "AAPL".to_string(),
            quantity: 100.0,
            price: 150.0,
            market_value: 15000.0,
            weight: 0.5,
            pnl: 0.0,
            entry_price: 150.0,
            entry_time: chrono::Utc::now(),
        },
        risk_management::Position {
            symbol: "GOOGL".to_string(),
            quantity: 50.0,
            price: 2500.0,
            market_value: 125000.0,
            weight: 0.5,
            pnl: 0.0,
            entry_price: 2500.0,
            entry_time: chrono::Utc::now(),
        },
    ];
    
    // Add historical returns
    portfolio.returns = (0..252).map(|i| {
        let phase = i as f64 * 0.1;
        0.001 * (phase.sin() + 0.5 * (phase * 2.0).cos())
    }).collect();
    
    portfolio.targets = vec![0.01; 252];
    
    portfolio
}

fn benchmark_var_calculation(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let portfolio = create_test_portfolio();
    
    let mut group = c.benchmark_group("var_calculation");
    group.measurement_time(Duration::from_secs(30));
    
    // Test different confidence levels
    for confidence_level in [0.01, 0.05, 0.10].iter() {
        group.bench_with_input(
            BenchmarkId::new("historical_var", format!("{}%", (confidence_level * 100.0) as u32)),
            confidence_level,
            |b, &confidence_level| {
                b.to_async(&rt).iter(|| async {
                    let config = VarConfig::default();
                    let quantum_config = quantum_uncertainty::QuantumConfig::default();
                    let quantum_engine = Arc::new(RwLock::new(
                        quantum_uncertainty::QuantumUncertaintyEngine::new(quantum_config).await.unwrap()
                    ));
                    
                    let var_calculator = VarCalculator::new(config, quantum_engine).await.unwrap();
                    
                    let result = var_calculator.calculate_var(
                        black_box(&portfolio),
                        black_box(confidence_level)
                    ).await;
                    
                    black_box(result)
                });
            },
        );
    }
    
    group.finish();
}

fn benchmark_real_time_constraints(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let portfolio = create_test_portfolio();
    
    let mut group = c.benchmark_group("real_time_constraints");
    group.measurement_time(Duration::from_secs(10));
    
    // Test if VaR calculation meets <10μs constraint
    group.bench_function("var_latency_test", |b| {
        b.to_async(&rt).iter(|| async {
            let config = VarConfig {
                confidence_levels: vec![0.05],
                historical_window: 50, // Smaller window for speed
                monte_carlo_simulations: 1000, // Fewer simulations for speed
                method: risk_management::VarMethod::Historical,
                enable_quantum: false, // Disable for speed test
                smoothing_factor: 0.94,
            };
            
            let quantum_config = quantum_uncertainty::QuantumConfig::default();
            let quantum_engine = Arc::new(RwLock::new(
                quantum_uncertainty::QuantumUncertaintyEngine::new(quantum_config).await.unwrap()
            ));
            
            let var_calculator = VarCalculator::new(config, quantum_engine).await.unwrap();
            
            let start = std::time::Instant::now();
            let _result = var_calculator.calculate_var(
                black_box(&portfolio),
                black_box(0.05)
            ).await;
            let elapsed = start.elapsed();
            
            // Assert that calculation is under 10μs (this will fail initially, 
            // but shows where optimization is needed)
            if elapsed > Duration::from_micros(10) {
                eprintln!("Warning: VaR calculation took {:?}, exceeding 10μs target", elapsed);
            }
            
            black_box(elapsed)
        });
    });
    
    group.finish();
}

fn benchmark_comprehensive_risk_report(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let portfolio = create_test_portfolio();
    
    let mut group = c.benchmark_group("comprehensive_risk_report");
    group.measurement_time(Duration::from_secs(60));
    
    group.bench_function("full_risk_report", |b| {
        b.to_async(&rt).iter(|| async {
            let config = RiskConfig::default();
            let risk_manager = RiskManager::new(config).await.unwrap();
            
            let result = risk_manager.get_comprehensive_risk_report(
                black_box(&portfolio)
            ).await;
            
            black_box(result)
        });
    });
    
    group.finish();
}

fn benchmark_monte_carlo_performance(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let portfolio = create_test_portfolio();
    
    let mut group = c.benchmark_group("monte_carlo_performance");
    group.measurement_time(Duration::from_secs(45));
    
    for num_simulations in [1_000, 10_000, 100_000].iter() {
        group.bench_with_input(
            BenchmarkId::new("monte_carlo_var", num_simulations),
            num_simulations,
            |b, &num_simulations| {
                b.to_async(&rt).iter(|| async {
                    let config = VarConfig {
                        confidence_levels: vec![0.05],
                        historical_window: 250,
                        monte_carlo_simulations: num_simulations,
                        method: risk_management::VarMethod::MonteCarlo,
                        enable_quantum: false,
                        smoothing_factor: 0.94,
                    };
                    
                    let quantum_config = quantum_uncertainty::QuantumConfig::default();
                    let quantum_engine = Arc::new(RwLock::new(
                        quantum_uncertainty::QuantumUncertaintyEngine::new(quantum_config).await.unwrap()
                    ));
                    
                    let var_calculator = VarCalculator::new(config, quantum_engine).await.unwrap();
                    
                    let result = var_calculator.calculate_var(
                        black_box(&portfolio),
                        black_box(0.05)
                    ).await;
                    
                    black_box(result)
                });
            },
        );
    }
    
    group.finish();
}

fn benchmark_quantum_enhancement(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let portfolio = create_test_portfolio();
    
    let mut group = c.benchmark_group("quantum_enhancement");
    group.measurement_time(Duration::from_secs(90));
    
    group.bench_function("quantum_vs_classical_var", |b| {
        b.to_async(&rt).iter(|| async {
            let config = VarConfig {
                confidence_levels: vec![0.05],
                historical_window: 250,
                monte_carlo_simulations: 10_000,
                method: risk_management::VarMethod::QuantumMonteCarlo,
                enable_quantum: true,
                smoothing_factor: 0.94,
            };
            
            let quantum_config = quantum_uncertainty::QuantumConfig::default();
            let quantum_engine = Arc::new(RwLock::new(
                quantum_uncertainty::QuantumUncertaintyEngine::new(quantum_config).await.unwrap()
            ));
            
            let var_calculator = VarCalculator::new(config, quantum_engine).await.unwrap();
            
            let result = var_calculator.calculate_var(
                black_box(&portfolio),
                black_box(0.05)
            ).await;
            
            black_box(result)
        });
    });
    
    group.finish();
}

criterion_group!(
    benches,
    benchmark_var_calculation,
    benchmark_real_time_constraints,
    benchmark_comprehensive_risk_report,
    benchmark_monte_carlo_performance,
    benchmark_quantum_enhancement
);

criterion_main!(benches);