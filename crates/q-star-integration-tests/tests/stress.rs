//! Q* Stress Tests
//! 
//! Pushes the system to its limits to ensure stability under extreme conditions

use futures::stream::{FuturesUnordered, StreamExt};
use q_star_core::*;
use q_star_orchestrator::*;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};
use tokio::time::interval;

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn test_extreme_load() {
    let orchestrator = create_high_performance_orchestrator().await;
    
    // Spawn maximum agents
    for _ in 0..100 {
        orchestrator.spawn_agent(AgentType::Explorer).await.unwrap();
        orchestrator.spawn_agent(AgentType::Exploiter).await.unwrap();
    }
    
    // Submit tasks concurrently from multiple threads
    let submitted = Arc::new(AtomicU64::new(0));
    let completed = Arc::new(AtomicU64::new(0));
    let errors = Arc::new(AtomicU64::new(0));
    
    let mut handles = vec![];
    
    for thread_id in 0..8 {
        let orchestrator = orchestrator.clone();
        let submitted = submitted.clone();
        let completed = completed.clone();
        let errors = errors.clone();
        
        let handle = tokio::spawn(async move {
            for i in 0..10000 {
                let task = create_stress_task(thread_id, i);
                match orchestrator.submit_task(task).await {
                    Ok(id) => {
                        submitted.fetch_add(1, Ordering::Relaxed);
                        match orchestrator.await_result(&id).await {
                            Ok(_) => completed.fetch_add(1, Ordering::Relaxed),
                            Err(_) => errors.fetch_add(1, Ordering::Relaxed),
                        }
                    }
                    Err(_) => errors.fetch_add(1, Ordering::Relaxed),
                }
            }
        });
        
        handles.push(handle);
    }
    
    // Wait for all threads
    futures::future::join_all(handles).await;
    
    let total_submitted = submitted.load(Ordering::Relaxed);
    let total_completed = completed.load(Ordering::Relaxed);
    let total_errors = errors.load(Ordering::Relaxed);
    
    println!("Stress test results:");
    println!("  Submitted: {}", total_submitted);
    println!("  Completed: {}", total_completed);
    println!("  Errors: {}", total_errors);
    println!("  Success rate: {:.2}%", (total_completed as f64 / total_submitted as f64) * 100.0);
    
    // Should handle at least 95% successfully under extreme load
    assert!(total_completed as f64 / total_submitted as f64 > 0.95);
}

#[tokio::test]
async fn test_memory_pressure() {
    let orchestrator = create_test_orchestrator().await;
    
    // Create large state objects
    let large_state = MarketState {
        timestamp: chrono::Utc::now(),
        prices: vec![100.0; 10000],
        volumes: vec![1000.0; 10000],
        technical_indicators: vec![0.5; 10000],
        market_regime: MarketRegime::Volatile,
        volatility: 0.05,
        liquidity: 0.7,
    };
    
    // Submit many tasks with large states
    let mut task_ids = FuturesUnordered::new();
    
    for i in 0..1000 {
        let task = QStarTask {
            id: format!("memory_pressure_{}", i),
            state: large_state.clone(),
            constraints: TaskConstraints::default(),
            priority: TaskPriority::Low,
        };
        
        if let Ok(id) = orchestrator.submit_task(task).await {
            task_ids.push(orchestrator.await_result(&id));
        }
    }
    
    // Process results
    let mut completed = 0;
    while let Some(result) = task_ids.next().await {
        if result.is_ok() {
            completed += 1;
        }
    }
    
    println!("Completed {} tasks under memory pressure", completed);
    assert!(completed > 900); // Should complete most tasks
}

#[tokio::test]
async fn test_rapid_agent_churn() {
    let orchestrator = create_test_orchestrator().await;
    let churn_duration = Duration::from_secs(10);
    let start = Instant::now();
    
    let spawned = Arc::new(AtomicU64::new(0));
    let terminated = Arc::new(AtomicU64::new(0));
    
    // Spawn thread
    let spawn_handle = {
        let orchestrator = orchestrator.clone();
        let spawned = spawned.clone();
        tokio::spawn(async move {
            let mut interval = interval(Duration::from_millis(10));
            while start.elapsed() < churn_duration {
                interval.tick().await;
                if orchestrator.spawn_agent(AgentType::Explorer).await.is_ok() {
                    spawned.fetch_add(1, Ordering::Relaxed);
                }
            }
        })
    };
    
    // Terminate thread
    let terminate_handle = {
        let orchestrator = orchestrator.clone();
        let terminated = terminated.clone();
        tokio::spawn(async move {
            let mut interval = interval(Duration::from_millis(15));
            while start.elapsed() < churn_duration {
                interval.tick().await;
                if let Ok(agents) = orchestrator.list_agents().await {
                    if let Some(agent_id) = agents.first() {
                        if orchestrator.terminate_agent(agent_id).await.is_ok() {
                            terminated.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                }
            }
        })
    };
    
    // Submit tasks during churn
    let task_handle = {
        let orchestrator = orchestrator.clone();
        tokio::spawn(async move {
            let mut completed = 0;
            let mut interval = interval(Duration::from_millis(5));
            
            while start.elapsed() < churn_duration {
                interval.tick().await;
                let task = create_test_task();
                if let Ok(id) = orchestrator.submit_task(task).await {
                    if orchestrator.await_result(&id).await.is_ok() {
                        completed += 1;
                    }
                }
            }
            completed
        })
    };
    
    // Wait for all operations
    spawn_handle.await.unwrap();
    terminate_handle.await.unwrap();
    let tasks_completed = task_handle.await.unwrap();
    
    println!("Agent churn test:");
    println!("  Agents spawned: {}", spawned.load(Ordering::Relaxed));
    println!("  Agents terminated: {}", terminated.load(Ordering::Relaxed));
    println!("  Tasks completed: {}", tasks_completed);
    
    // System should remain stable and complete tasks
    assert!(tasks_completed > 100);
}

#[tokio::test]
async fn test_cascade_failures() {
    let orchestrator = create_test_orchestrator().await;
    
    // Spawn interdependent agents
    let coordinator_id = orchestrator.spawn_agent(AgentType::Coordinator).await.unwrap();
    let mut worker_ids = vec![];
    for _ in 0..10 {
        worker_ids.push(orchestrator.spawn_agent(AgentType::Explorer).await.unwrap());
    }
    
    // Submit tasks
    let mut task_ids = vec![];
    for i in 0..20 {
        let task = create_test_task();
        task_ids.push(orchestrator.submit_task(task).await.unwrap());
    }
    
    // Simulate coordinator failure
    orchestrator.terminate_agent(&coordinator_id).await.unwrap();
    
    // System should recover and complete tasks
    let mut completed = 0;
    for id in task_ids {
        if orchestrator.await_result(&id).await.is_ok() {
            completed += 1;
        }
    }
    
    println!("Completed {} tasks after coordinator failure", completed);
    assert!(completed > 15); // Should complete most tasks despite failure
}

#[tokio::test]
async fn test_byzantine_agents() {
    let mut config = OrchestratorConfig::default();
    config.consensus_mechanism = ConsensusMechanism::Byzantine;
    let orchestrator = Arc::new(QStarOrchestrator::new(config).await.unwrap());
    
    // Spawn mix of good and potentially byzantine agents
    for _ in 0..7 {
        orchestrator.spawn_agent(AgentType::Explorer).await.unwrap();
    }
    for _ in 0..3 {
        // These could be byzantine agents
        orchestrator.spawn_agent(AgentType::Exploiter).await.unwrap();
    }
    
    // Submit critical tasks
    let mut results = vec![];
    for i in 0..100 {
        let task = QStarTask {
            id: format!("byzantine_test_{}", i),
            state: create_test_state(),
            constraints: TaskConstraints {
                max_latency_us: 100,
                required_confidence: 0.9,
                risk_limit: 0.01,
            },
            priority: TaskPriority::High,
        };
        
        if let Ok(id) = orchestrator.submit_task(task).await {
            if let Ok(result) = orchestrator.await_result(&id).await {
                results.push(result);
            }
        }
    }
    
    // Verify byzantine fault tolerance
    let consensus_achieved = results.iter().filter(|r| r.consensus_achieved).count();
    println!("Byzantine consensus achieved: {}/{}", consensus_achieved, results.len());
    
    // Should achieve consensus in majority of cases
    assert!(consensus_achieved as f64 / results.len() as f64 > 0.9);
}

// Helper functions
async fn create_high_performance_orchestrator() -> Arc<QStarOrchestrator> {
    let config = OrchestratorConfig {
        topology: SwarmTopology::Mesh,
        max_agents: 200,
        min_agents: 50,
        spawn_strategy: SpawnStrategy::Aggressive,
        coordination_strategy: CoordinationStrategy::Parallel,
        consensus_mechanism: ConsensusMechanism::Optimistic,
        health_check_interval: Duration::from_secs(5),
        auto_scale: true,
        fault_tolerance: true,
        performance_targets: PerformanceTargets {
            max_latency_us: 100,
            min_throughput: 1_000_000,
            max_memory_mb: 1000,
            target_accuracy: 0.9,
        },
    };
    
    Arc::new(QStarOrchestrator::new(config).await.unwrap())
}

async fn create_test_orchestrator() -> Arc<QStarOrchestrator> {
    Arc::new(QStarOrchestrator::new(OrchestratorConfig::default()).await.unwrap())
}

fn create_stress_task(thread_id: usize, task_id: usize) -> QStarTask {
    QStarTask {
        id: format!("stress_{}_{}", thread_id, task_id),
        state: create_test_state(),
        constraints: TaskConstraints {
            max_latency_us: 50,
            required_confidence: 0.8,
            risk_limit: 0.02,
        },
        priority: match task_id % 3 {
            0 => TaskPriority::High,
            1 => TaskPriority::Medium,
            _ => TaskPriority::Low,
        },
    }
}

fn create_test_task() -> QStarTask {
    QStarTask {
        id: uuid::Uuid::new_v4().to_string(),
        state: create_test_state(),
        constraints: TaskConstraints::default(),
        priority: TaskPriority::Medium,
    }
}

fn create_test_state() -> MarketState {
    MarketState {
        timestamp: chrono::Utc::now(),
        prices: vec![100.0, 101.0, 99.5, 100.5, 101.5],
        volumes: vec![1000.0, 1100.0, 900.0, 1050.0, 1200.0],
        technical_indicators: vec![0.5, 0.6, 0.55, 0.58, 0.62],
        market_regime: MarketRegime::Trending,
        volatility: 0.02,
        liquidity: 0.85,
    }
}