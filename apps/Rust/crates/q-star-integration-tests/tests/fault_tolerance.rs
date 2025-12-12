//! Q* Fault Tolerance Tests
//! 
//! Validates system resilience and recovery mechanisms

use q_star_core::*;
use q_star_orchestrator::*;
use std::sync::Arc;
use std::time::Duration;
use tokio::time::{sleep, timeout};

#[tokio::test]
async fn test_agent_recovery() {
    let orchestrator = create_fault_tolerant_orchestrator().await;
    
    // Spawn initial agents
    let agent_id = orchestrator.spawn_agent(AgentType::Explorer).await.unwrap();
    
    // Submit task
    let task = create_test_task();
    let task_id = orchestrator.submit_task(task.clone()).await.unwrap();
    
    // Simulate agent failure mid-task
    orchestrator.simulate_agent_failure(&agent_id).await.unwrap();
    
    // System should recover and complete task
    let result = timeout(
        Duration::from_secs(5),
        orchestrator.await_result(&task_id)
    ).await;
    
    assert!(result.is_ok());
    let result = result.unwrap().unwrap();
    assert!(result.recovered_from_failure);
    assert!(result.consensus_achieved);
}

#[tokio::test]
async fn test_network_partition() {
    let orchestrator = create_fault_tolerant_orchestrator().await;
    
    // Create two groups of agents
    let mut group_a = vec![];
    let mut group_b = vec![];
    
    for _ in 0..5 {
        group_a.push(orchestrator.spawn_agent(AgentType::Explorer).await.unwrap());
        group_b.push(orchestrator.spawn_agent(AgentType::Exploiter).await.unwrap());
    }
    
    // Simulate network partition
    orchestrator.simulate_network_partition(&group_a, &group_b).await.unwrap();
    
    // Submit tasks during partition
    let mut task_ids = vec![];
    for i in 0..10 {
        let task = create_test_task();
        task_ids.push(orchestrator.submit_task(task).await.unwrap());
    }
    
    // Heal partition after 2 seconds
    sleep(Duration::from_secs(2)).await;
    orchestrator.heal_network_partition().await.unwrap();
    
    // All tasks should eventually complete
    let mut completed = 0;
    for id in task_ids {
        if orchestrator.await_result(&id).await.is_ok() {
            completed += 1;
        }
    }
    
    println!("Completed {}/10 tasks despite network partition", completed);
    assert!(completed >= 8); // Should complete most tasks
}

#[tokio::test]
async fn test_coordinator_election() {
    let orchestrator = create_fault_tolerant_orchestrator().await;
    
    // Spawn multiple potential coordinators
    let mut coordinator_ids = vec![];
    for _ in 0..3 {
        coordinator_ids.push(
            orchestrator.spawn_agent(AgentType::Coordinator).await.unwrap()
        );
    }
    
    // Spawn workers
    for _ in 0..10 {
        orchestrator.spawn_agent(AgentType::Explorer).await.unwrap();
    }
    
    // Get current leader
    let initial_leader = orchestrator.get_coordinator_leader().await.unwrap();
    println!("Initial coordinator leader: {}", initial_leader);
    
    // Kill current leader
    orchestrator.terminate_agent(&initial_leader).await.unwrap();
    
    // Wait for election
    sleep(Duration::from_millis(500)).await;
    
    // New leader should be elected
    let new_leader = orchestrator.get_coordinator_leader().await.unwrap();
    println!("New coordinator leader: {}", new_leader);
    
    assert_ne!(initial_leader, new_leader);
    
    // System should continue functioning
    let task = create_test_task();
    let task_id = orchestrator.submit_task(task).await.unwrap();
    let result = orchestrator.await_result(&task_id).await.unwrap();
    assert!(result.consensus_achieved);
}

#[tokio::test]
async fn test_cascading_recovery() {
    let orchestrator = create_fault_tolerant_orchestrator().await;
    
    // Create dependency chain: Coordinator -> Analyzer -> Workers
    let coordinator = orchestrator.spawn_agent(AgentType::Coordinator).await.unwrap();
    let analyzer = orchestrator.spawn_agent(AgentType::Analyst).await.unwrap();
    let mut workers = vec![];
    for _ in 0..5 {
        workers.push(orchestrator.spawn_agent(AgentType::Explorer).await.unwrap());
    }
    
    // Set dependencies
    orchestrator.set_agent_dependency(&analyzer, &coordinator).await.unwrap();
    for worker in &workers {
        orchestrator.set_agent_dependency(worker, &analyzer).await.unwrap();
    }
    
    // Submit tasks
    let mut task_ids = vec![];
    for _ in 0..10 {
        task_ids.push(orchestrator.submit_task(create_test_task()).await.unwrap());
    }
    
    // Fail the analyzer (middle of chain)
    orchestrator.simulate_agent_failure(&analyzer).await.unwrap();
    
    // System should detect and recover
    sleep(Duration::from_millis(200)).await;
    
    // Verify recovery
    let health = orchestrator.health_check().await.unwrap();
    assert!(health.all_agents_healthy);
    assert!(health.dependencies_satisfied);
    
    // Tasks should complete
    let mut completed = 0;
    for id in task_ids {
        if orchestrator.await_result(&id).await.is_ok() {
            completed += 1;
        }
    }
    assert!(completed >= 8);
}

#[tokio::test]
async fn test_memory_corruption_recovery() {
    let orchestrator = create_fault_tolerant_orchestrator().await;
    
    // Spawn agents with memory
    for _ in 0..5 {
        orchestrator.spawn_agent(AgentType::Explorer).await.unwrap();
    }
    
    // Store critical state
    orchestrator.store_checkpoint("test_state", &create_test_state()).await.unwrap();
    
    // Simulate memory corruption
    orchestrator.simulate_memory_corruption().await.unwrap();
    
    // System should detect and recover from checkpoint
    sleep(Duration::from_millis(500)).await;
    
    // Verify recovery
    let recovered_state = orchestrator.load_checkpoint::<MarketState>("test_state").await;
    assert!(recovered_state.is_ok());
    
    // System should be functional
    let task = create_test_task();
    let task_id = orchestrator.submit_task(task).await.unwrap();
    let result = orchestrator.await_result(&task_id).await.unwrap();
    assert!(result.consensus_achieved);
}

#[tokio::test]
async fn test_resource_exhaustion() {
    let orchestrator = create_fault_tolerant_orchestrator().await;
    
    // Spawn many agents to consume resources
    let mut agent_ids = vec![];
    for _ in 0..50 {
        if let Ok(id) = orchestrator.spawn_agent(AgentType::Explorer).await {
            agent_ids.push(id);
        }
    }
    
    // Set resource limits
    orchestrator.set_resource_limits(ResourceLimits {
        max_memory_mb: 100,
        max_cpu_percent: 80.0,
        max_file_handles: 1000,
    }).await.unwrap();
    
    // Submit resource-intensive tasks
    let mut task_ids = vec![];
    for i in 0..100 {
        let task = QStarTask {
            id: format!("resource_test_{}", i),
            state: create_large_state(),
            constraints: TaskConstraints::default(),
            priority: TaskPriority::Low,
        };
        
        match orchestrator.submit_task(task).await {
            Ok(id) => task_ids.push(id),
            Err(_) => break, // Resource limit reached
        }
    }
    
    // System should gracefully handle resource limits
    assert!(!task_ids.is_empty());
    println!("Submitted {} tasks before resource limit", task_ids.len());
    
    // Should still complete submitted tasks
    let mut completed = 0;
    for id in &task_ids[..task_ids.len().min(20)] {
        if orchestrator.await_result(id).await.is_ok() {
            completed += 1;
        }
    }
    assert!(completed > 0);
}

#[tokio::test]
async fn test_deadlock_detection() {
    let orchestrator = create_fault_tolerant_orchestrator().await;
    
    // Create potential deadlock scenario
    let agent_a = orchestrator.spawn_agent(AgentType::Explorer).await.unwrap();
    let agent_b = orchestrator.spawn_agent(AgentType::Exploiter).await.unwrap();
    
    // Set circular dependencies
    orchestrator.set_agent_dependency(&agent_a, &agent_b).await.unwrap();
    orchestrator.set_agent_dependency(&agent_b, &agent_a).await.unwrap();
    
    // Submit task that requires both agents
    let task = QStarTask {
        id: "deadlock_test".to_string(),
        state: create_test_state(),
        constraints: TaskConstraints {
            max_latency_us: 1000,
            required_confidence: 0.9,
            risk_limit: 0.01,
        },
        priority: TaskPriority::High,
    };
    
    let task_id = orchestrator.submit_task(task).await.unwrap();
    
    // System should detect and resolve deadlock
    let result = timeout(
        Duration::from_secs(5),
        orchestrator.await_result(&task_id)
    ).await;
    
    assert!(result.is_ok());
    let result = result.unwrap().unwrap();
    assert!(result.deadlock_resolved);
}

#[tokio::test]
async fn test_version_mismatch_handling() {
    let orchestrator = create_fault_tolerant_orchestrator().await;
    
    // Spawn agents with different versions
    orchestrator.spawn_agent_with_version(AgentType::Explorer, "1.0.0").await.unwrap();
    orchestrator.spawn_agent_with_version(AgentType::Explorer, "1.0.1").await.unwrap();
    orchestrator.spawn_agent_with_version(AgentType::Explorer, "2.0.0").await.unwrap();
    
    // Submit task requiring version compatibility
    let task = create_test_task();
    let task_id = orchestrator.submit_task(task).await.unwrap();
    
    // System should handle version differences gracefully
    let result = orchestrator.await_result(&task_id).await.unwrap();
    assert!(result.consensus_achieved);
    assert!(result.version_compatibility_handled);
}

// Helper functions
async fn create_fault_tolerant_orchestrator() -> Arc<QStarOrchestrator> {
    let config = OrchestratorConfig {
        topology: SwarmTopology::Hierarchical,
        max_agents: 100,
        min_agents: 5,
        spawn_strategy: SpawnStrategy::Conservative,
        coordination_strategy: CoordinationStrategy::Consensus,
        consensus_mechanism: ConsensusMechanism::Byzantine,
        health_check_interval: Duration::from_millis(100),
        auto_scale: true,
        fault_tolerance: true,
        performance_targets: PerformanceTargets::default(),
    };
    
    Arc::new(QStarOrchestrator::new(config).await.unwrap())
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

fn create_large_state() -> MarketState {
    MarketState {
        timestamp: chrono::Utc::now(),
        prices: vec![100.0; 1000],
        volumes: vec![1000.0; 1000],
        technical_indicators: vec![0.5; 1000],
        market_regime: MarketRegime::Volatile,
        volatility: 0.05,
        liquidity: 0.7,
    }
}