//! Q* Performance Tests
//! 
//! Validates that all components meet their performance targets

use criterion::{black_box, Criterion};
use q_star_core::*;
use q_star_neural::*;
use q_star_orchestrator::*;
use q_star_quantum::*;
use q_star_trading::*;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::runtime::Runtime;

#[tokio::test]
async fn test_decision_latency() {
    // Target: <10μs decision latency
    let engine = create_test_engine().await;
    let state = create_test_state();
    
    // Warm up
    for _ in 0..100 {
        let _ = engine.decide(&state).await;
    }
    
    // Measure
    let mut latencies = Vec::new();
    for _ in 0..1000 {
        let start = Instant::now();
        let _ = engine.decide(&state).await.unwrap();
        latencies.push(start.elapsed().as_micros());
    }
    
    let avg_latency = latencies.iter().sum::<u128>() / latencies.len() as u128;
    let p99_latency = percentile(&mut latencies, 99.0);
    
    println!("Average latency: {}μs", avg_latency);
    println!("P99 latency: {}μs", p99_latency);
    
    assert!(avg_latency < 10);
    assert!(p99_latency < 20);
}

#[tokio::test]
async fn test_neural_inference_speed() {
    // Target: <1μs inference
    let network = create_test_network();
    let state = create_test_state();
    
    // Warm up
    for _ in 0..100 {
        let _ = network.forward(&state).await;
    }
    
    // Measure
    let mut times = Vec::new();
    for _ in 0..10000 {
        let start = Instant::now();
        let _ = network.forward(&state).await.unwrap();
        times.push(start.elapsed().as_nanos());
    }
    
    let avg_time = times.iter().sum::<u128>() / times.len() as u128;
    let p99_time = percentile(&mut times, 99.0);
    
    println!("Neural inference - Average: {}ns, P99: {}ns", avg_time, p99_time);
    
    assert!(avg_time < 1000); // <1μs
    assert!(p99_time < 2000); // <2μs at P99
}

#[tokio::test]
async fn test_quantum_computation_advantage() {
    // Target: 2^n quantum advantage
    let quantum_agent = create_quantum_agent();
    let classical_agent = create_classical_agent();
    let complex_state = create_complex_state();
    
    // Measure classical time
    let start = Instant::now();
    let _ = classical_agent.decide(&complex_state).await.unwrap();
    let classical_time = start.elapsed();
    
    // Measure quantum time
    let start = Instant::now();
    let _ = quantum_agent.decide(&complex_state).await.unwrap();
    let quantum_time = start.elapsed();
    
    let speedup = classical_time.as_micros() as f64 / quantum_time.as_micros() as f64;
    println!("Quantum speedup: {}x", speedup);
    
    // With 10 qubits, expect at least 10x speedup
    assert!(speedup > 10.0);
}

#[tokio::test]
async fn test_reward_calculation_speed() {
    // Target: <1μs reward calculation
    let calculator = create_reward_calculator();
    let action = create_test_action();
    let impact = create_test_impact();
    
    // Warm up
    for _ in 0..100 {
        let _ = calculator.calculate(&action, &impact).await;
    }
    
    // Measure
    let mut times = Vec::new();
    for _ in 0..10000 {
        let start = Instant::now();
        let _ = calculator.calculate(&action, &impact).await.unwrap();
        times.push(start.elapsed().as_nanos());
    }
    
    let avg_time = times.iter().sum::<u128>() / times.len() as u128;
    println!("Reward calculation average: {}ns", avg_time);
    
    assert!(avg_time < 1000); // <1μs
}

#[tokio::test]
async fn test_coordination_overhead() {
    // Target: <10μs coordination overhead
    let orchestrator = create_test_orchestrator().await;
    
    // Spawn agents
    for _ in 0..5 {
        orchestrator.spawn_agent(AgentType::Explorer).await.unwrap();
    }
    
    let task = create_test_task();
    
    // Warm up
    for _ in 0..10 {
        let id = orchestrator.submit_task(task.clone()).await.unwrap();
        let _ = orchestrator.await_result(&id).await;
    }
    
    // Measure coordination overhead
    let mut overheads = Vec::new();
    for _ in 0..100 {
        let id = orchestrator.submit_task(task.clone()).await.unwrap();
        let result = orchestrator.await_result(&id).await.unwrap();
        overheads.push(result.coordination_overhead_us);
    }
    
    let avg_overhead = overheads.iter().sum::<u64>() / overheads.len() as u64;
    println!("Average coordination overhead: {}μs", avg_overhead);
    
    assert!(avg_overhead < 10);
}

#[tokio::test]
async fn test_throughput() {
    // Target: >1M decisions/second
    let engine = create_test_engine().await;
    let states: Vec<MarketState> = (0..10000).map(|_| create_test_state()).collect();
    
    let start = Instant::now();
    let mut count = 0;
    let deadline = start + Duration::from_secs(1);
    
    while Instant::now() < deadline {
        for state in &states {
            let _ = engine.decide(state).await;
            count += 1;
        }
    }
    
    let elapsed = start.elapsed();
    let throughput = count as f64 / elapsed.as_secs_f64();
    
    println!("Throughput: {:.0} decisions/second", throughput);
    assert!(throughput > 1_000_000.0);
}

#[tokio::test]
async fn test_memory_efficiency() {
    // Target: <1MB per agent
    let orchestrator = create_test_orchestrator().await;
    
    // Get baseline memory
    let baseline = get_memory_usage();
    
    // Spawn 100 agents
    for _ in 0..100 {
        orchestrator.spawn_agent(AgentType::Explorer).await.unwrap();
    }
    
    // Measure memory after spawning
    let after_spawn = get_memory_usage();
    let memory_per_agent = (after_spawn - baseline) / 100;
    
    println!("Memory per agent: {} KB", memory_per_agent / 1024);
    assert!(memory_per_agent < 1_048_576); // <1MB
}

#[tokio::test]
async fn test_scalability() {
    // Test linear scaling up to 1000 agents
    let orchestrator = create_test_orchestrator().await;
    let task = create_test_task();
    
    let mut results = Vec::new();
    
    for num_agents in [10, 50, 100, 500, 1000] {
        // Clear previous agents
        orchestrator.clear_agents().await.unwrap();
        
        // Spawn agents
        for _ in 0..num_agents {
            orchestrator.spawn_agent(AgentType::Explorer).await.unwrap();
        }
        
        // Measure performance
        let start = Instant::now();
        let mut completed = 0;
        
        for _ in 0..100 {
            let id = orchestrator.submit_task(task.clone()).await.unwrap();
            let _ = orchestrator.await_result(&id).await;
            completed += 1;
        }
        
        let elapsed = start.elapsed();
        let throughput = completed as f64 / elapsed.as_secs_f64();
        
        results.push((num_agents, throughput));
        println!("{} agents: {:.2} tasks/second", num_agents, throughput);
    }
    
    // Verify near-linear scaling
    let scaling_efficiency = results[4].1 / (results[0].1 * 100.0);
    println!("Scaling efficiency: {:.2}%", scaling_efficiency * 100.0);
    assert!(scaling_efficiency > 0.8); // >80% scaling efficiency
}

// Helper functions
async fn create_test_engine() -> Arc<QStarEngine> {
    let config = QStarConfig::default();
    Arc::new(QStarEngine::new(config).await.unwrap())
}

fn create_test_network() -> QStarPolicyNetwork {
    QStarPolicyNetwork::new(QStarNeuralConfig::default())
}

fn create_quantum_agent() -> QuantumQStarAgent {
    QuantumQStarAgent::new(QuantumConfig::default())
}

fn create_classical_agent() -> ExplorerAgent {
    ExplorerAgent::default()
}

fn create_reward_calculator() -> TradingRewardCalculator {
    TradingRewardCalculator::new(RewardConfig::default())
}

async fn create_test_orchestrator() -> Arc<QStarOrchestrator> {
    let config = OrchestratorConfig::default();
    Arc::new(QStarOrchestrator::new(config).await.unwrap())
}

fn create_test_state() -> MarketState {
    MarketState {
        timestamp: chrono::Utc::now(),
        prices: vec![100.0; 5],
        volumes: vec![1000.0; 5],
        technical_indicators: vec![0.5; 10],
        market_regime: MarketRegime::Trending,
        volatility: 0.02,
        liquidity: 0.85,
    }
}

fn create_complex_state() -> MarketState {
    MarketState {
        timestamp: chrono::Utc::now(),
        prices: (0..100).map(|i| 100.0 + (i as f64).sin()).collect(),
        volumes: (0..100).map(|i| 1000.0 * (1.0 + (i as f64 * 0.1).cos())).collect(),
        technical_indicators: (0..50).map(|i| (i as f64 / 50.0)).collect(),
        market_regime: MarketRegime::Volatile,
        volatility: 0.05,
        liquidity: 0.7,
    }
}

fn create_test_action() -> QStarAction {
    QStarAction {
        action_type: ActionType::Buy,
        size: 1.0,
        price: Some(100.0),
        confidence: 0.9,
        risk_level: RiskLevel::Medium,
        priority: ActionPriority::High,
        metadata: Default::default(),
    }
}

fn create_test_impact() -> MarketImpact {
    MarketImpact {
        price: 100.5,
        volume: 1000.0,
        slippage: 0.005,
        execution_time_ms: 5,
    }
}

fn create_test_task() -> QStarTask {
    QStarTask {
        id: "test_task".to_string(),
        state: create_test_state(),
        constraints: TaskConstraints::default(),
        priority: TaskPriority::Medium,
    }
}

fn percentile(data: &mut Vec<u128>, p: f64) -> u128 {
    data.sort_unstable();
    let idx = ((data.len() as f64 - 1.0) * p / 100.0) as usize;
    data[idx]
}

fn get_memory_usage() -> usize {
    // Platform-specific memory measurement
    #[cfg(target_os = "linux")]
    {
        use std::fs;
        let status = fs::read_to_string("/proc/self/status").unwrap();
        for line in status.lines() {
            if line.starts_with("VmRSS:") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                return parts[1].parse::<usize>().unwrap() * 1024; // Convert KB to bytes
            }
        }
    }
    0
}