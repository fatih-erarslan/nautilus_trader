//! Q* System Integration Tests
//! 
//! Validates end-to-end functionality of the complete Q* system

use async_trait::async_trait;
use chrono::Utc;
use q_star_core::*;
use q_star_neural::*;
use q_star_orchestrator::*;
use q_star_quantum::*;
use q_star_trading::*;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::info;

#[tokio::test]
async fn test_full_system_integration() {
    tracing_test::traced_test("test_full_system_integration");
    
    // 1. Create orchestrator configuration
    let config = OrchestratorConfig {
        topology: SwarmTopology::Hierarchical,
        max_agents: 10,
        min_agents: 3,
        spawn_strategy: SpawnStrategy::Adaptive,
        coordination_strategy: CoordinationStrategy::Weighted,
        consensus_mechanism: ConsensusMechanism::Byzantine,
        health_check_interval: Duration::from_secs(1),
        auto_scale: true,
        fault_tolerance: true,
        performance_targets: PerformanceTargets {
            max_latency_us: 10,
            min_throughput: 100_000,
            max_memory_mb: 100,
            target_accuracy: 0.95,
        },
    };
    
    // 2. Initialize orchestrator
    let orchestrator = QStarOrchestrator::new(config).await.unwrap();
    
    // 3. Spawn initial agents
    let explorer_id = orchestrator.spawn_agent(AgentType::Explorer).await.unwrap();
    let exploiter_id = orchestrator.spawn_agent(AgentType::Exploiter).await.unwrap();
    let coordinator_id = orchestrator.spawn_agent(AgentType::Coordinator).await.unwrap();
    let quantum_id = orchestrator.spawn_agent(AgentType::Quantum).await.unwrap();
    
    // 4. Create market state
    let state = MarketState {
        timestamp: Utc::now(),
        prices: vec![100.0, 100.5, 101.0, 100.8, 101.2],
        volumes: vec![1000.0, 1500.0, 1200.0, 1100.0, 1300.0],
        technical_indicators: vec![0.6, 0.65, 0.7, 0.68, 0.72],
        market_regime: MarketRegime::Trending,
        volatility: 0.02,
        liquidity: 0.85,
    };
    
    // 5. Submit task for decision
    let task = QStarTask {
        id: "integration_test_001".to_string(),
        state: state.clone(),
        constraints: TaskConstraints {
            max_latency_us: 50,
            required_confidence: 0.8,
            risk_limit: 0.02,
        },
        priority: TaskPriority::High,
    };
    
    let task_id = orchestrator.submit_task(task).await.unwrap();
    
    // 6. Wait for result
    let result = orchestrator.await_result(&task_id).await.unwrap();
    
    // 7. Validate result
    assert!(result.decision_time_us < 50);
    assert!(result.confidence >= 0.8);
    assert!(result.consensus_achieved);
    assert!(result.agents_participated.len() >= 3);
    
    // 8. Check system health
    let health = orchestrator.health_check().await.unwrap();
    assert!(health.all_agents_healthy);
    assert!(health.performance_targets_met);
    
    info!("Integration test completed successfully");
}

#[tokio::test]
async fn test_multi_agent_coordination() {
    tracing_test::traced_test("test_multi_agent_coordination");
    
    let config = OrchestratorConfig::default();
    let orchestrator = QStarOrchestrator::new(config).await.unwrap();
    
    // Spawn multiple agents of each type
    let mut agent_ids = vec![];
    for _ in 0..2 {
        agent_ids.push(orchestrator.spawn_agent(AgentType::Explorer).await.unwrap());
        agent_ids.push(orchestrator.spawn_agent(AgentType::Exploiter).await.unwrap());
    }
    agent_ids.push(orchestrator.spawn_agent(AgentType::Coordinator).await.unwrap());
    agent_ids.push(orchestrator.spawn_agent(AgentType::Quantum).await.unwrap());
    
    // Create complex market scenario
    let states = vec![
        create_trending_market(),
        create_ranging_market(),
        create_volatile_market(),
        create_crash_scenario(),
    ];
    
    // Submit multiple concurrent tasks
    let mut task_ids = vec![];
    for (i, state) in states.iter().enumerate() {
        let task = QStarTask {
            id: format!("coordination_test_{}", i),
            state: state.clone(),
            constraints: TaskConstraints::default(),
            priority: TaskPriority::Medium,
        };
        task_ids.push(orchestrator.submit_task(task).await.unwrap());
    }
    
    // Await all results
    let results = futures::future::join_all(
        task_ids.iter().map(|id| orchestrator.await_result(id))
    ).await;
    
    // Validate coordination
    for result in results {
        let result = result.unwrap();
        assert!(result.consensus_achieved);
        assert!(result.agents_participated.len() >= 3);
        assert!(result.coordination_overhead_us < 10);
    }
}

#[tokio::test]
async fn test_neural_network_inference() {
    let config = QStarNeuralConfig {
        hidden_sizes: vec![128, 64, 32],
        activation: Activation::GELU,
        dropout: 0.1,
        use_batch_norm: true,
    };
    
    let network = QStarPolicyNetwork::new(config);
    let state = create_test_market_state();
    
    // Measure inference time
    let start = Instant::now();
    let probs = network.forward(&state).await.unwrap();
    let inference_time = start.elapsed();
    
    // Validate performance
    assert!(inference_time.as_micros() < 1);
    assert_eq!(probs.len(), 10); // 10 action types
    assert!((probs.iter().sum::<f64>() - 1.0).abs() < 1e-6);
}

#[tokio::test]
async fn test_quantum_enhancement() {
    let quantum_agent = QuantumQStarAgent::new(QuantumConfig {
        num_qubits: 10,
        circuit_depth: 5,
        error_correction: true,
        measurement_shots: 1000,
    });
    
    let state = create_test_market_state();
    
    // Test quantum advantage
    let start = Instant::now();
    let decision = quantum_agent.decide(&state).await.unwrap();
    let quantum_time = start.elapsed();
    
    // Quantum should provide 2^n advantage
    assert!(quantum_time.as_micros() < 100);
    assert!(decision.quantum_confidence > 0.9);
    assert!(decision.entanglement_utilized);
}

#[tokio::test]
async fn test_reward_calculation() {
    let calculator = TradingRewardCalculator::new(RewardConfig {
        profit_weight: 0.4,
        risk_weight: 0.3,
        efficiency_weight: 0.2,
        timing_weight: 0.1,
    });
    
    let action = QStarAction {
        action_type: ActionType::Buy,
        size: 1.0,
        price: Some(100.0),
        confidence: 0.9,
        risk_level: RiskLevel::Medium,
        priority: ActionPriority::High,
        metadata: Default::default(),
    };
    
    let market_impact = MarketImpact {
        price: 100.5,
        volume: 1000.0,
        slippage: 0.005,
        execution_time_ms: 5,
    };
    
    let start = Instant::now();
    let reward = calculator.calculate(&action, &market_impact).await.unwrap();
    let calc_time = start.elapsed();
    
    assert!(calc_time.as_micros() < 1);
    assert!(reward.total_reward > 0.0);
}

// Helper functions
fn create_test_market_state() -> MarketState {
    MarketState {
        timestamp: Utc::now(),
        prices: vec![100.0, 101.0, 102.0, 101.5, 102.5],
        volumes: vec![1000.0, 1200.0, 1100.0, 1300.0, 1250.0],
        technical_indicators: vec![0.6, 0.65, 0.7, 0.68, 0.72],
        market_regime: MarketRegime::Trending,
        volatility: 0.02,
        liquidity: 0.85,
    }
}

fn create_trending_market() -> MarketState {
    MarketState {
        timestamp: Utc::now(),
        prices: vec![100.0, 101.0, 102.0, 103.0, 104.0],
        volumes: vec![1000.0, 1100.0, 1200.0, 1300.0, 1400.0],
        technical_indicators: vec![0.7, 0.75, 0.8, 0.85, 0.9],
        market_regime: MarketRegime::Trending,
        volatility: 0.015,
        liquidity: 0.9,
    }
}

fn create_ranging_market() -> MarketState {
    MarketState {
        timestamp: Utc::now(),
        prices: vec![100.0, 100.5, 99.8, 100.2, 100.0],
        volumes: vec![1000.0, 900.0, 950.0, 1000.0, 980.0],
        technical_indicators: vec![0.5, 0.48, 0.52, 0.49, 0.51],
        market_regime: MarketRegime::Ranging,
        volatility: 0.01,
        liquidity: 0.7,
    }
}

fn create_volatile_market() -> MarketState {
    MarketState {
        timestamp: Utc::now(),
        prices: vec![100.0, 105.0, 98.0, 103.0, 95.0],
        volumes: vec![2000.0, 2500.0, 3000.0, 2200.0, 2800.0],
        technical_indicators: vec![0.3, 0.8, 0.2, 0.7, 0.4],
        market_regime: MarketRegime::Volatile,
        volatility: 0.05,
        liquidity: 0.6,
    }
}

fn create_crash_scenario() -> MarketState {
    MarketState {
        timestamp: Utc::now(),
        prices: vec![100.0, 95.0, 90.0, 85.0, 80.0],
        volumes: vec![5000.0, 6000.0, 7000.0, 8000.0, 9000.0],
        technical_indicators: vec![0.2, 0.15, 0.1, 0.05, 0.0],
        market_regime: MarketRegime::Crisis,
        volatility: 0.1,
        liquidity: 0.3,
    }
}