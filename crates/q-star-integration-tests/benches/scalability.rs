use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use q_star_core::*;
use q_star_neural::*;
use q_star_quantum::*;
use q_star_trading::*;
use q_star_orchestrator::*;
use std::time::Duration;

fn benchmark_data_scaling(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("Data Scaling");
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
    
    let orchestrator_config = QStarOrchestratorConfig::default();
    let orchestrator = rt.block_on(async {
        QStarOrchestrator::new(orchestrator_config, core, neural, quantum, trading).await.unwrap()
    });
    
    // Test with different data sizes
    for data_size in [100, 500, 1000, 5000, 10000, 50000].iter() {
        let market_data = create_test_market_data(*data_size);
        
        group.bench_with_input(
            BenchmarkId::new("neural_processing", data_size),
            data_size,
            |b, _| {
                b.to_async(&rt).iter(|| async {
                    orchestrator.process_neural_data(black_box(&market_data)).await.unwrap()
                })
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("quantum_processing", data_size),
            data_size,
            |b, _| {
                b.to_async(&rt).iter(|| async {
                    orchestrator.process_quantum_data(black_box(&market_data)).await.unwrap()
                })
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("trading_processing", data_size),
            data_size,
            |b, _| {
                b.to_async(&rt).iter(|| async {
                    orchestrator.process_trading_data(black_box(&market_data)).await.unwrap()
                })
            },
        );
    }
    
    group.finish();
}

fn benchmark_concurrent_processing(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("Concurrent Processing");
    group.measurement_time(Duration::from_secs(25));
    
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
    
    let orchestrator_config = QStarOrchestratorConfig::default();
    let orchestrator = rt.block_on(async {
        QStarOrchestrator::new(orchestrator_config, core, neural, quantum, trading).await.unwrap()
    });
    
    let market_data = create_test_market_data(1000);
    
    // Test with different concurrency levels
    for concurrency in [1, 2, 4, 8, 16, 32].iter() {
        group.bench_with_input(
            BenchmarkId::new("concurrent_predictions", concurrency),
            concurrency,
            |b, &concurrency| {
                b.to_async(&rt).iter(|| async {
                    orchestrator.concurrent_predictions(black_box(&market_data), concurrency).await.unwrap()
                })
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("parallel_processing", concurrency),
            concurrency,
            |b, &concurrency| {
                b.to_async(&rt).iter(|| async {
                    orchestrator.parallel_processing(black_box(&market_data), concurrency).await.unwrap()
                })
            },
        );
    }
    
    group.finish();
}

fn benchmark_memory_scaling(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("Memory Scaling");
    group.measurement_time(Duration::from_secs(20));
    
    // Test with different memory configurations
    for memory_limit in [64, 128, 256, 512, 1024].iter() {
        let core_config = QStarCoreConfig::with_memory_limit(*memory_limit);
        let neural_config = QStarNeuralConfig::with_memory_limit(*memory_limit);
        let quantum_config = QStarQuantumConfig::with_memory_limit(*memory_limit);
        let trading_config = QStarTradingConfig::with_memory_limit(*memory_limit);
        
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
        
        let orchestrator_config = QStarOrchestratorConfig::with_memory_limit(*memory_limit);
        let orchestrator = rt.block_on(async {
            QStarOrchestrator::new(orchestrator_config, core, neural, quantum, trading).await.unwrap()
        });
        
        let market_data = create_test_market_data(1000);
        
        group.bench_with_input(
            BenchmarkId::new("memory_efficient_processing", memory_limit),
            memory_limit,
            |b, _| {
                b.to_async(&rt).iter(|| async {
                    orchestrator.memory_efficient_processing(black_box(&market_data)).await.unwrap()
                })
            },
        );
    }
    
    group.finish();
}

fn benchmark_distributed_scaling(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("Distributed Scaling");
    group.measurement_time(Duration::from_secs(35));
    
    // Test with different node counts
    for node_count in [1, 2, 4, 8].iter() {
        let cluster_config = QStarClusterConfig::with_nodes(*node_count);
        let cluster = rt.block_on(async {
            QStarCluster::new(cluster_config).await.unwrap()
        });
        
        let market_data = create_test_market_data(2000);
        
        group.bench_with_input(
            BenchmarkId::new("distributed_processing", node_count),
            node_count,
            |b, _| {
                b.to_async(&rt).iter(|| async {
                    cluster.distributed_processing(black_box(&market_data)).await.unwrap()
                })
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("cluster_coordination", node_count),
            node_count,
            |b, _| {
                b.to_async(&rt).iter(|| async {
                    cluster.cluster_coordination(black_box(&market_data)).await.unwrap()
                })
            },
        );
    }
    
    group.finish();
}

fn benchmark_throughput_scaling(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("Throughput Scaling");
    group.measurement_time(Duration::from_secs(30));
    
    // Initialize high-throughput Q-Star configuration
    let core_config = QStarCoreConfig::high_throughput();
    let neural_config = QStarNeuralConfig::high_throughput();
    let quantum_config = QStarQuantumConfig::high_throughput();
    let trading_config = QStarTradingConfig::high_throughput();
    
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
    
    let orchestrator_config = QStarOrchestratorConfig::high_throughput();
    let orchestrator = rt.block_on(async {
        QStarOrchestrator::new(orchestrator_config, core, neural, quantum, trading).await.unwrap()
    });
    
    // Test with different request rates
    for requests_per_second in [10, 100, 1000, 5000, 10000].iter() {
        let market_data = create_test_market_data(100);
        
        group.bench_with_input(
            BenchmarkId::new("high_throughput_processing", requests_per_second),
            requests_per_second,
            |b, &requests_per_second| {
                b.to_async(&rt).iter(|| async {
                    orchestrator.high_throughput_processing(black_box(&market_data), requests_per_second).await.unwrap()
                })
            },
        );
    }
    
    group.finish();
}

fn benchmark_resource_utilization(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("Resource Utilization");
    group.measurement_time(Duration::from_secs(25));
    
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
    
    let orchestrator_config = QStarOrchestratorConfig::default();
    let orchestrator = rt.block_on(async {
        QStarOrchestrator::new(orchestrator_config, core, neural, quantum, trading).await.unwrap()
    });
    
    let market_data = create_test_market_data(1000);
    
    group.bench_function("cpu_utilization", |b| {
        b.to_async(&rt).iter(|| async {
            orchestrator.cpu_intensive_processing(black_box(&market_data)).await.unwrap()
        })
    });
    
    group.bench_function("memory_utilization", |b| {
        b.to_async(&rt).iter(|| async {
            orchestrator.memory_intensive_processing(black_box(&market_data)).await.unwrap()
        })
    });
    
    group.bench_function("io_utilization", |b| {
        b.to_async(&rt).iter(|| async {
            orchestrator.io_intensive_processing(black_box(&market_data)).await.unwrap()
        })
    });
    
    group.bench_function("network_utilization", |b| {
        b.to_async(&rt).iter(|| async {
            orchestrator.network_intensive_processing(black_box(&market_data)).await.unwrap()
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
            market_cap: 1_000_000.0 + (i as f64 * 0.1).sin() * 100_000.0,
            open_interest: 500.0 + (i as f64 * 0.05).cos() * 50.0,
        })
        .collect()
}

criterion_group!(
    benches,
    benchmark_data_scaling,
    benchmark_concurrent_processing,
    benchmark_memory_scaling,
    benchmark_distributed_scaling,
    benchmark_throughput_scaling,
    benchmark_resource_utilization
);
criterion_main!(benches);