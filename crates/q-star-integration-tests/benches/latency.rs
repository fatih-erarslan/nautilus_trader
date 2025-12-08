//! Q* Latency Benchmarks
//! 
//! Measures decision latency across all components

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use q_star_core::*;
use q_star_neural::*;
use q_star_quantum::*;
use q_star_trading::*;
use q_star_orchestrator::*;
use std::time::Duration;

fn bench_core_decision_latency(c: &mut Criterion) {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let engine = runtime.block_on(async {
        QStarEngine::new(QStarConfig::default()).await.unwrap()
    });
    
    let mut group = c.benchmark_group("core_decision_latency");
    group.measurement_time(Duration::from_secs(10));
    
    for complexity in ["simple", "moderate", "complex"].iter() {
        let state = match *complexity {
            "simple" => create_simple_state(),
            "moderate" => create_moderate_state(),
            "complex" => create_complex_state(),
            _ => unreachable!(),
        };
        
        group.bench_with_input(
            BenchmarkId::from_parameter(complexity),
            &state,
            |b, state| {
                b.to_async(&runtime).iter(|| async {
                    black_box(engine.decide(state).await.unwrap())
                });
            },
        );
    }
    
    group.finish();
}

fn bench_neural_inference_latency(c: &mut Criterion) {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("neural_inference_latency");
    group.measurement_time(Duration::from_secs(5));
    
    for (name, config) in [
        ("small", QStarNeuralConfig {
            hidden_sizes: vec![64, 32],
            activation: Activation::ReLU,
            dropout: 0.0,
            use_batch_norm: false,
        }),
        ("medium", QStarNeuralConfig {
            hidden_sizes: vec![128, 64, 32],
            activation: Activation::GELU,
            dropout: 0.1,
            use_batch_norm: true,
        }),
        ("large", QStarNeuralConfig {
            hidden_sizes: vec![256, 128, 64, 32],
            activation: Activation::Swish,
            dropout: 0.2,
            use_batch_norm: true,
        }),
    ] {
        let network = QStarPolicyNetwork::new(config);
        let state = create_moderate_state();
        
        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &state,
            |b, state| {
                b.to_async(&runtime).iter(|| async {
                    black_box(network.forward(state).await.unwrap())
                });
            },
        );
    }
    
    group.finish();
}

fn bench_quantum_computation_latency(c: &mut Criterion) {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("quantum_computation_latency");
    group.measurement_time(Duration::from_secs(5));
    
    for num_qubits in [5, 10, 15, 20] {
        let config = QuantumConfig {
            num_qubits,
            circuit_depth: 5,
            error_correction: true,
            measurement_shots: 1000,
        };
        
        let quantum_agent = QuantumQStarAgent::new(config);
        let state = create_moderate_state();
        
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}qubits", num_qubits)),
            &state,
            |b, state| {
                b.to_async(&runtime).iter(|| async {
                    black_box(quantum_agent.decide(state).await.unwrap())
                });
            },
        );
    }
    
    group.finish();
}

fn bench_reward_calculation_latency(c: &mut Criterion) {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let calculator = TradingRewardCalculator::new(RewardConfig::default());
    
    let mut group = c.benchmark_group("reward_calculation_latency");
    
    for (scenario, action, impact) in [
        ("simple_buy", create_buy_action(), create_simple_impact()),
        ("complex_sell", create_sell_action(), create_complex_impact()),
        ("high_freq", create_hft_action(), create_hft_impact()),
    ] {
        group.bench_with_input(
            BenchmarkId::from_parameter(scenario),
            &(action.clone(), impact.clone()),
            |b, (action, impact)| {
                b.to_async(&runtime).iter(|| async {
                    black_box(calculator.calculate(action, impact).await.unwrap())
                });
            },
        );
    }
    
    group.finish();
}

fn bench_coordination_overhead(c: &mut Criterion) {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    
    let orchestrator = runtime.block_on(async {
        let config = OrchestratorConfig {
            topology: SwarmTopology::Mesh,
            max_agents: 20,
            min_agents: 5,
            spawn_strategy: SpawnStrategy::Adaptive,
            coordination_strategy: CoordinationStrategy::Parallel,
            consensus_mechanism: ConsensusMechanism::Optimistic,
            health_check_interval: Duration::from_secs(10),
            auto_scale: false,
            fault_tolerance: false,
            performance_targets: PerformanceTargets::default(),
        };
        
        let orch = QStarOrchestrator::new(config).await.unwrap();
        
        // Pre-spawn agents
        for _ in 0..10 {
            orch.spawn_agent(AgentType::Explorer).await.unwrap();
        }
        
        orch
    });
    
    let mut group = c.benchmark_group("coordination_overhead");
    group.measurement_time(Duration::from_secs(10));
    
    for num_agents in [5, 10, 15, 20] {
        let task = create_test_task();
        
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}agents", num_agents)),
            &task,
            |b, task| {
                b.to_async(&runtime).iter(|| async {
                    let id = orchestrator.submit_task(task.clone()).await.unwrap();
                    black_box(orchestrator.await_result(&id).await.unwrap())
                });
            },
        );
    }
    
    group.finish();
}

// Helper functions
fn create_simple_state() -> MarketState {
    MarketState {
        timestamp: chrono::Utc::now(),
        prices: vec![100.0; 5],
        volumes: vec![1000.0; 5],
        technical_indicators: vec![0.5; 5],
        market_regime: MarketRegime::Stable,
        volatility: 0.01,
        liquidity: 0.9,
    }
}

fn create_moderate_state() -> MarketState {
    MarketState {
        timestamp: chrono::Utc::now(),
        prices: vec![100.0, 101.0, 99.5, 100.5, 101.5],
        volumes: vec![1000.0, 1100.0, 900.0, 1050.0, 1200.0],
        technical_indicators: vec![0.5, 0.6, 0.55, 0.58, 0.62, 0.48, 0.52, 0.59, 0.61, 0.63],
        market_regime: MarketRegime::Trending,
        volatility: 0.02,
        liquidity: 0.85,
    }
}

fn create_complex_state() -> MarketState {
    MarketState {
        timestamp: chrono::Utc::now(),
        prices: (0..100).map(|i| 100.0 + (i as f64).sin() * 5.0).collect(),
        volumes: (0..100).map(|i| 1000.0 * (1.0 + (i as f64 * 0.1).cos())).collect(),
        technical_indicators: (0..50).map(|i| (i as f64 / 50.0).sin().abs()).collect(),
        market_regime: MarketRegime::Volatile,
        volatility: 0.05,
        liquidity: 0.7,
    }
}

fn create_buy_action() -> QStarAction {
    QStarAction {
        action_type: ActionType::Buy,
        size: 1.0,
        price: Some(100.0),
        confidence: 0.9,
        risk_level: RiskLevel::Low,
        priority: ActionPriority::High,
        metadata: Default::default(),
    }
}

fn create_sell_action() -> QStarAction {
    QStarAction {
        action_type: ActionType::Sell,
        size: 2.0,
        price: Some(101.0),
        confidence: 0.85,
        risk_level: RiskLevel::Medium,
        priority: ActionPriority::Medium,
        metadata: Default::default(),
    }
}

fn create_hft_action() -> QStarAction {
    QStarAction {
        action_type: ActionType::MarketMake,
        size: 0.1,
        price: Some(100.05),
        confidence: 0.95,
        risk_level: RiskLevel::Low,
        priority: ActionPriority::Critical,
        metadata: Default::default(),
    }
}

fn create_simple_impact() -> MarketImpact {
    MarketImpact {
        price: 100.1,
        volume: 1000.0,
        slippage: 0.001,
        execution_time_ms: 1,
    }
}

fn create_complex_impact() -> MarketImpact {
    MarketImpact {
        price: 101.5,
        volume: 5000.0,
        slippage: 0.01,
        execution_time_ms: 50,
    }
}

fn create_hft_impact() -> MarketImpact {
    MarketImpact {
        price: 100.051,
        volume: 100.0,
        slippage: 0.0001,
        execution_time_ms: 0,
    }
}

fn create_test_task() -> QStarTask {
    QStarTask {
        id: "bench_task".to_string(),
        state: create_moderate_state(),
        constraints: TaskConstraints::default(),
        priority: TaskPriority::Medium,
    }
}

criterion_group!(
    benches,
    bench_core_decision_latency,
    bench_neural_inference_latency,
    bench_quantum_computation_latency,
    bench_reward_calculation_latency,
    bench_coordination_overhead
);

criterion_main!(benches);