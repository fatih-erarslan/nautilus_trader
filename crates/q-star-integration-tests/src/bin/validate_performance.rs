//! Q* Performance Validation Tool
//! 
//! Validates that all Q* components meet their performance targets

use std::time::{Duration, Instant};
use tokio::runtime::Runtime;
use q_star_core::*;
use q_star_neural::*;
use q_star_quantum::*;
use q_star_trading::*;
use q_star_orchestrator::*;
use q_star_integration_tests::*;

#[derive(Default)]
struct ValidationResults {
    core_latency: (f64, bool),
    neural_inference: (f64, bool),
    quantum_advantage: (f64, bool),
    reward_calculation: (f64, bool),
    coordination_overhead: (f64, bool),
    throughput: (f64, bool),
    memory_per_agent: (f64, bool),
    scaling_efficiency: (f64, bool),
}

fn main() {
    println!("🔍 Q* Performance Validation");
    println!("============================\n");
    
    let rt = Runtime::new().unwrap();
    let mut results = ValidationResults::default();
    
    // Test 1: Core Decision Latency (<10μs)
    print!("Testing Core Decision Latency... ");
    results.core_latency = rt.block_on(test_core_latency());
    println!("{:.2}μs {}", results.core_latency.0, 
        if results.core_latency.1 { "✅" } else { "❌" });
    
    // Test 2: Neural Inference (<1μs)
    print!("Testing Neural Inference Speed... ");
    results.neural_inference = rt.block_on(test_neural_inference());
    println!("{:.2}μs {}", results.neural_inference.0,
        if results.neural_inference.1 { "✅" } else { "❌" });
    
    // Test 3: Quantum Advantage (>10x)
    print!("Testing Quantum Advantage... ");
    results.quantum_advantage = rt.block_on(test_quantum_advantage());
    println!("{:.1}x speedup {}", results.quantum_advantage.0,
        if results.quantum_advantage.1 { "✅" } else { "❌" });
    
    // Test 4: Reward Calculation (<1μs)
    print!("Testing Reward Calculation... ");
    results.reward_calculation = rt.block_on(test_reward_speed());
    println!("{:.2}μs {}", results.reward_calculation.0,
        if results.reward_calculation.1 { "✅" } else { "❌" });
    
    // Test 5: Coordination Overhead (<10μs)
    print!("Testing Coordination Overhead... ");
    results.coordination_overhead = rt.block_on(test_coordination());
    println!("{:.2}μs {}", results.coordination_overhead.0,
        if results.coordination_overhead.1 { "✅" } else { "❌" });
    
    // Test 6: Throughput (>1M ops/sec)
    print!("Testing System Throughput... ");
    results.throughput = rt.block_on(test_throughput());
    println!("{:.0} ops/sec {}", results.throughput.0,
        if results.throughput.1 { "✅" } else { "❌" });
    
    // Test 7: Memory Efficiency (<1MB/agent)
    print!("Testing Memory Efficiency... ");
    results.memory_per_agent = rt.block_on(test_memory_efficiency());
    println!("{:.2} KB/agent {}", results.memory_per_agent.0 / 1024.0,
        if results.memory_per_agent.1 { "✅" } else { "❌" });
    
    // Test 8: Scaling Efficiency (>80%)
    print!("Testing Scaling Efficiency... ");
    results.scaling_efficiency = rt.block_on(test_scaling());
    println!("{:.1}% {}", results.scaling_efficiency.0 * 100.0,
        if results.scaling_efficiency.1 { "✅" } else { "❌" });
    
    // Summary
    println!("\n📊 Performance Summary");
    println!("=====================");
    
    let all_passed = results.core_latency.1 && 
                    results.neural_inference.1 && 
                    results.quantum_advantage.1 && 
                    results.reward_calculation.1 && 
                    results.coordination_overhead.1 && 
                    results.throughput.1 && 
                    results.memory_per_agent.1 && 
                    results.scaling_efficiency.1;
    
    if all_passed {
        println!("✅ All performance targets achieved!");
        println!("\n🏆 Q* System is production ready!");
    } else {
        println!("❌ Some performance targets not met");
        println!("\nFailed targets:");
        if !results.core_latency.1 { 
            println!("  - Core latency: {:.2}μs (target: <10μs)", results.core_latency.0); 
        }
        if !results.neural_inference.1 { 
            println!("  - Neural inference: {:.2}μs (target: <1μs)", results.neural_inference.0); 
        }
        if !results.quantum_advantage.1 { 
            println!("  - Quantum advantage: {:.1}x (target: >10x)", results.quantum_advantage.0); 
        }
        if !results.reward_calculation.1 { 
            println!("  - Reward calculation: {:.2}μs (target: <1μs)", results.reward_calculation.0); 
        }
        if !results.coordination_overhead.1 { 
            println!("  - Coordination: {:.2}μs (target: <10μs)", results.coordination_overhead.0); 
        }
        if !results.throughput.1 { 
            println!("  - Throughput: {:.0} ops/sec (target: >1M)", results.throughput.0); 
        }
        if !results.memory_per_agent.1 { 
            println!("  - Memory/agent: {:.2} KB (target: <1MB)", results.memory_per_agent.0 / 1024.0); 
        }
        if !results.scaling_efficiency.1 { 
            println!("  - Scaling: {:.1}% (target: >80%)", results.scaling_efficiency.0 * 100.0); 
        }
    }
}

async fn test_core_latency() -> (f64, bool) {
    let engine = QStarEngine::new(QStarConfig::default()).await.unwrap();
    let state = create_market_states()[0].clone();
    
    // Warm up
    for _ in 0..100 {
        let _ = engine.decide(&state).await;
    }
    
    let (avg, _) = measure_latency(1000, || engine.decide(&state)).await;
    (avg as f64, avg < 10)
}

async fn test_neural_inference() -> (f64, bool) {
    let network = QStarPolicyNetwork::new(QStarNeuralConfig::default());
    let state = create_market_states()[0].clone();
    
    // Warm up
    for _ in 0..100 {
        let _ = network.forward(&state).await;
    }
    
    let (avg, _) = measure_latency(10000, || network.forward(&state)).await;
    (avg as f64, avg < 1)
}

async fn test_quantum_advantage() -> (f64, bool) {
    let quantum = QuantumQStarAgent::new(QuantumConfig {
        num_qubits: 10,
        circuit_depth: 5,
        error_correction: true,
        measurement_shots: 1000,
    });
    
    let classical = ExplorerAgent::default();
    let complex_state = create_market_states()[2].clone(); // Volatile state
    
    // Measure classical
    let start = Instant::now();
    for _ in 0..10 {
        let _ = classical.decide(&complex_state).await;
    }
    let classical_time = start.elapsed();
    
    // Measure quantum
    let start = Instant::now();
    for _ in 0..10 {
        let _ = quantum.decide(&complex_state).await;
    }
    let quantum_time = start.elapsed();
    
    let speedup = classical_time.as_micros() as f64 / quantum_time.as_micros() as f64;
    (speedup, speedup > 10.0)
}

async fn test_reward_speed() -> (f64, bool) {
    let calculator = TradingRewardCalculator::new(RewardConfig::default());
    let action = generate_test_action();
    let impact = generate_market_impact();
    
    // Warm up
    for _ in 0..100 {
        let _ = calculator.calculate(&action, &impact).await;
    }
    
    let (avg, _) = measure_latency(10000, || calculator.calculate(&action, &impact)).await;
    (avg as f64, avg < 1)
}

async fn test_coordination() -> (f64, bool) {
    let orchestrator = create_benchmark_orchestrator().await;
    
    // Spawn agents
    for _ in 0..10 {
        orchestrator.spawn_agent(AgentType::Explorer).await.unwrap();
    }
    
    let task = generate_test_task();
    
    // Warm up
    for _ in 0..10 {
        let id = orchestrator.submit_task(task.clone()).await.unwrap();
        let _ = orchestrator.await_result(&id).await;
    }
    
    // Measure
    let mut overheads = Vec::new();
    for _ in 0..100 {
        let id = orchestrator.submit_task(task.clone()).await.unwrap();
        let result = orchestrator.await_result(&id).await.unwrap();
        overheads.push(result.coordination_overhead_us);
    }
    
    let avg = overheads.iter().sum::<u64>() as f64 / overheads.len() as f64;
    (avg, avg < 10.0)
}

async fn test_throughput() -> (f64, bool) {
    let engine = QStarEngine::new(QStarConfig::default()).await.unwrap();
    let states: Vec<MarketState> = (0..1000)
        .map(|_| create_market_states()[0].clone())
        .collect();
    
    let start = Instant::now();
    let mut count = 0;
    let deadline = start + Duration::from_secs(1);
    
    while Instant::now() < deadline {
        for state in &states {
            let _ = engine.decide(state).await;
            count += 1;
        }
    }
    
    let throughput = count as f64 / start.elapsed().as_secs_f64();
    (throughput, throughput > 1_000_000.0)
}

async fn test_memory_efficiency() -> (f64, bool) {
    let orchestrator = create_default_orchestrator().await;
    
    // Get baseline
    let baseline = get_current_memory();
    
    // Spawn 100 agents
    for _ in 0..100 {
        orchestrator.spawn_agent(AgentType::Explorer).await.unwrap();
    }
    
    // Measure after spawning
    let after = get_current_memory();
    let per_agent = (after - baseline) as f64 / 100.0;
    
    (per_agent, per_agent < 1_048_576.0) // <1MB
}

async fn test_scaling() -> (f64, bool) {
    let orchestrator = create_benchmark_orchestrator().await;
    let task = generate_test_task();
    
    // Test with 10 agents
    for _ in 0..10 {
        orchestrator.spawn_agent(AgentType::Explorer).await.unwrap();
    }
    
    let start = Instant::now();
    for _ in 0..100 {
        let id = orchestrator.submit_task(task.clone()).await.unwrap();
        let _ = orchestrator.await_result(&id).await;
    }
    let time_10 = start.elapsed();
    
    // Test with 100 agents
    for _ in 0..90 {
        orchestrator.spawn_agent(AgentType::Explorer).await.unwrap();
    }
    
    let start = Instant::now();
    for _ in 0..100 {
        let id = orchestrator.submit_task(task.clone()).await.unwrap();
        let _ = orchestrator.await_result(&id).await;
    }
    let time_100 = start.elapsed();
    
    let scaling = (time_10.as_secs_f64() / time_100.as_secs_f64()) / 0.1;
    (scaling, scaling > 0.8)
}

fn get_current_memory() -> usize {
    #[cfg(target_os = "linux")]
    {
        use std::fs;
        let status = fs::read_to_string("/proc/self/status").unwrap();
        for line in status.lines() {
            if line.starts_with("VmRSS:") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                return parts[1].parse::<usize>().unwrap() * 1024;
            }
        }
    }
    0
}