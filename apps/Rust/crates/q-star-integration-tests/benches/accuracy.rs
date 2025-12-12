use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use q_star_core::*;
use q_star_neural::*;
use q_star_quantum::*;
use q_star_trading::*;
use q_star_orchestrator::*;
use std::time::Duration;

fn benchmark_prediction_accuracy(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("Prediction Accuracy");
    group.measurement_time(Duration::from_secs(30));
    
    // Initialize Q-Star components
    let core_config = QStarCoreConfig::default();
    let neural_config = QStarNeuralConfig::default();
    let quantum_config = QStarQuantumConfig::default();
    let trading_config = QStarTradingConfig::default();
    
    let core = rt.block_on(async {
        QStarCore::new(core_config).await.unwrap()
    });
    
    let neural = rt.block_on(async {
        QStarNeural::new(neural_config).await.unwrap()
    });
    
    let quantum = rt.block_on(async {
        QStarQuantum::new(quantum_config).await.unwrap()
    });
    
    let trading = rt.block_on(async {
        QStarTrading::new(trading_config).await.unwrap()
    });
    
    // Create orchestrator
    let orchestrator_config = QStarOrchestratorConfig::default();
    let orchestrator = rt.block_on(async {
        QStarOrchestrator::new(orchestrator_config, core, neural, quantum, trading).await.unwrap()
    });
    
    // Test market data
    let market_data = create_test_market_data(1000);
    
    for window_size in [50, 100, 200, 500].iter() {
        group.bench_with_input(
            BenchmarkId::new("price_prediction_accuracy", window_size),
            window_size,
            |b, &window_size| {
                b.to_async(&rt).iter(|| async {
                    let windowed_data = &market_data[..window_size];
                    orchestrator.predict_price_with_accuracy(black_box(windowed_data)).await.unwrap()
                })
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("direction_prediction_accuracy", window_size),
            window_size,
            |b, &window_size| {
                b.to_async(&rt).iter(|| async {
                    let windowed_data = &market_data[..window_size];
                    orchestrator.predict_direction_with_accuracy(black_box(windowed_data)).await.unwrap()
                })
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("volatility_prediction_accuracy", window_size),
            window_size,
            |b, &window_size| {
                b.to_async(&rt).iter(|| async {
                    let windowed_data = &market_data[..window_size];
                    orchestrator.predict_volatility_with_accuracy(black_box(windowed_data)).await.unwrap()
                })
            },
        );
    }
    
    group.finish();
}

fn benchmark_ensemble_accuracy(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("Ensemble Accuracy");
    group.measurement_time(Duration::from_secs(25));
    
    // Initialize multiple Q-Star instances for ensemble
    let ensemble_size = 5;
    let orchestrators = rt.block_on(async {
        let mut orchestrators = Vec::new();
        for i in 0..ensemble_size {
            let core_config = QStarCoreConfig::with_id(i);
            let neural_config = QStarNeuralConfig::with_id(i);
            let quantum_config = QStarQuantumConfig::with_id(i);
            let trading_config = QStarTradingConfig::with_id(i);
            
            let core = QStarCore::new(core_config).await.unwrap();
            let neural = QStarNeural::new(neural_config).await.unwrap();
            let quantum = QStarQuantum::new(quantum_config).await.unwrap();
            let trading = QStarTrading::new(trading_config).await.unwrap();
            
            let orchestrator_config = QStarOrchestratorConfig::with_id(i);
            let orchestrator = QStarOrchestrator::new(orchestrator_config, core, neural, quantum, trading).await.unwrap();
            orchestrators.push(orchestrator);
        }
        orchestrators
    });
    
    let ensemble_orchestrator = rt.block_on(async {
        QStarEnsemble::new(orchestrators).await.unwrap()
    });
    
    let market_data = create_test_market_data(500);
    
    group.bench_function("ensemble_prediction_accuracy", |b| {
        b.to_async(&rt).iter(|| async {
            ensemble_orchestrator.predict_ensemble_with_accuracy(black_box(&market_data)).await.unwrap()
        })
    });
    
    group.bench_function("weighted_ensemble_accuracy", |b| {
        b.to_async(&rt).iter(|| async {
            ensemble_orchestrator.predict_weighted_ensemble_with_accuracy(black_box(&market_data)).await.unwrap()
        })
    });
    
    group.bench_function("adaptive_ensemble_accuracy", |b| {
        b.to_async(&rt).iter(|| async {
            ensemble_orchestrator.predict_adaptive_ensemble_with_accuracy(black_box(&market_data)).await.unwrap()
        })
    });
    
    group.finish();
}

fn benchmark_real_time_accuracy(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("Real-time Accuracy");
    group.measurement_time(Duration::from_secs(20));
    
    // Initialize Q-Star with real-time configuration
    let core_config = QStarCoreConfig::real_time();
    let neural_config = QStarNeuralConfig::real_time();
    let quantum_config = QStarQuantumConfig::real_time();
    let trading_config = QStarTradingConfig::real_time();
    
    let core = rt.block_on(async {
        QStarCore::new(core_config).await.unwrap()
    });
    
    let neural = rt.block_on(async {
        QStarNeural::new(neural_config).await.unwrap()
    });
    
    let quantum = rt.block_on(async {
        QStarQuantum::new(quantum_config).await.unwrap()
    });
    
    let trading = rt.block_on(async {
        QStarTrading::new(trading_config).await.unwrap()
    });
    
    let orchestrator_config = QStarOrchestratorConfig::real_time();
    let orchestrator = rt.block_on(async {
        QStarOrchestrator::new(orchestrator_config, core, neural, quantum, trading).await.unwrap()
    });
    
    // Streaming market data simulation
    let streaming_data = create_streaming_test_data(100);
    
    group.bench_function("streaming_prediction_accuracy", |b| {
        b.to_async(&rt).iter(|| async {
            orchestrator.predict_streaming_with_accuracy(black_box(&streaming_data)).await.unwrap()
        })
    });
    
    group.bench_function("low_latency_accuracy", |b| {
        b.to_async(&rt).iter(|| async {
            orchestrator.predict_low_latency_with_accuracy(black_box(&streaming_data)).await.unwrap()
        })
    });
    
    group.bench_function("adaptive_accuracy", |b| {
        b.to_async(&rt).iter(|| async {
            orchestrator.predict_adaptive_with_accuracy(black_box(&streaming_data)).await.unwrap()
        })
    });
    
    group.finish();
}

fn benchmark_cross_validation_accuracy(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("Cross-validation Accuracy");
    group.measurement_time(Duration::from_secs(40));
    
    // Initialize Q-Star for cross-validation
    let core_config = QStarCoreConfig::default();
    let neural_config = QStarNeuralConfig::default();
    let quantum_config = QStarQuantumConfig::default();
    let trading_config = QStarTradingConfig::default();
    
    let core = rt.block_on(async {
        QStarCore::new(core_config).await.unwrap()
    });
    
    let neural = rt.block_on(async {
        QStarNeural::new(neural_config).await.unwrap()
    });
    
    let quantum = rt.block_on(async {
        QStarQuantum::new(quantum_config).await.unwrap()
    });
    
    let trading = rt.block_on(async {
        QStarTrading::new(trading_config).await.unwrap()
    });
    
    let orchestrator_config = QStarOrchestratorConfig::default();
    let orchestrator = rt.block_on(async {
        QStarOrchestrator::new(orchestrator_config, core, neural, quantum, trading).await.unwrap()
    });
    
    let market_data = create_test_market_data(1000);
    
    for k_fold in [3, 5, 10].iter() {
        group.bench_with_input(
            BenchmarkId::new("k_fold_cross_validation", k_fold),
            k_fold,
            |b, &k_fold| {
                b.to_async(&rt).iter(|| async {
                    orchestrator.k_fold_cross_validation_accuracy(black_box(&market_data), k_fold).await.unwrap()
                })
            },
        );
    }
    
    group.bench_function("time_series_cross_validation", |b| {
        b.to_async(&rt).iter(|| async {
            orchestrator.time_series_cross_validation_accuracy(black_box(&market_data)).await.unwrap()
        })
    });
    
    group.bench_function("walk_forward_validation", |b| {
        b.to_async(&rt).iter(|| async {
            orchestrator.walk_forward_validation_accuracy(black_box(&market_data)).await.unwrap()
        })
    });
    
    group.finish();
}

// Helper functions for test data generation
fn create_test_market_data(size: usize) -> Vec<MarketDataPoint> {
    (0..size)
        .map(|i| MarketDataPoint {
            timestamp: i as u64,
            price: 100.0 + (i as f64 * 0.01).sin(),
            volume: 1000.0 + (i as f64 * 0.1).cos() * 100.0,
            volatility: 0.1 + (i as f64 * 0.001).sin().abs() * 0.05,
        })
        .collect()
}

fn create_streaming_test_data(size: usize) -> Vec<StreamingDataPoint> {
    (0..size)
        .map(|i| StreamingDataPoint {
            timestamp: i as u64,
            price: 100.0 + (i as f64 * 0.01).sin(),
            volume: 1000.0 + (i as f64 * 0.1).cos() * 100.0,
            bid: 99.9 + (i as f64 * 0.01).sin(),
            ask: 100.1 + (i as f64 * 0.01).sin(),
        })
        .collect()
}

criterion_group!(
    benches,
    benchmark_prediction_accuracy,
    benchmark_ensemble_accuracy,
    benchmark_real_time_accuracy,
    benchmark_cross_validation_accuracy
);
criterion_main!(benches);