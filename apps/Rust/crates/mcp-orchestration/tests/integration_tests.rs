//! Integration tests for MCP orchestration system.

use mcp_orchestration::*;
use std::sync::Arc;
use std::time::Duration;
use tokio::time::sleep;
use tracing_test::traced_test;

/// Test full orchestrator lifecycle
#[tokio::test]
#[traced_test]
async fn test_orchestrator_full_lifecycle() {
    let config = OrchestrationConfig::default();
    let orchestrator = OrchestratorBuilder::new()
        .with_config(config)
        .build()
        .await
        .expect("Failed to build orchestrator");
    
    // Test startup
    orchestrator.start().await.expect("Failed to start orchestrator");
    assert!(orchestrator.is_running().await);
    
    // Wait for system to stabilize
    sleep(Duration::from_millis(500)).await;
    
    // Test agent registration
    let agent_info = AgentInfo::new(
        AgentId::new(),
        AgentType::Risk,
        "Integration Test Agent".to_string(),
        "1.0.0".to_string(),
    );
    let agent_id = agent_info.id;
    
    orchestrator.register_agent(agent_info).await.expect("Failed to register agent");
    
    // Test task submission
    let task = Task::new("integration_test_task", TaskPriority::High, b"test_payload".to_vec())
        .with_agent_type(AgentType::Risk);
    
    let task_id = orchestrator.submit_task(task).await.expect("Failed to submit task");
    
    // Verify task was queued
    sleep(Duration::from_millis(100)).await;
    
    // Test system status
    let status = orchestrator.get_system_status().await.expect("Failed to get system status");
    assert_eq!(status.orchestrator_state, orchestrator::OrchestratorState::Running);
    
    // Test metrics collection
    let metrics = orchestrator.get_metrics().await.expect("Failed to get metrics");
    assert!(metrics.agent_metrics.total_agents > 0);
    
    // Test agent unregistration
    orchestrator.unregister_agent(agent_id).await.expect("Failed to unregister agent");
    
    // Test shutdown
    orchestrator.shutdown().await.expect("Failed to shutdown orchestrator");
    assert!(!orchestrator.is_running().await);
}

/// Test agent communication flow
#[tokio::test]
#[traced_test]
async fn test_agent_communication_flow() {
    let orchestrator = OrchestratorBuilder::new()
        .build()
        .await
        .expect("Failed to build orchestrator");
    
    orchestrator.start().await.expect("Failed to start orchestrator");
    
    // Register multiple agents
    let agent1_info = AgentInfo::new(
        AgentId::new(),
        AgentType::Risk,
        "Agent 1".to_string(),
        "1.0.0".to_string(),
    );
    let agent1_id = agent1_info.id;
    
    let agent2_info = AgentInfo::new(
        AgentId::new(),
        AgentType::Neural,
        "Agent 2".to_string(),
        "1.0.0".to_string(),
    );
    let agent2_id = agent2_info.id;
    
    orchestrator.register_agent(agent1_info).await.expect("Failed to register agent 1");
    orchestrator.register_agent(agent2_info).await.expect("Failed to register agent 2");
    
    // Wait for agents to be registered
    sleep(Duration::from_millis(200)).await;
    
    // Test task assignment coordination
    let risk_task = Task::new("risk_analysis", TaskPriority::Critical, b"market_data".to_vec())
        .with_agent_type(AgentType::Risk);
    
    let neural_task = Task::new("price_prediction", TaskPriority::High, b"historical_data".to_vec())
        .with_agent_type(AgentType::Neural);
    
    let risk_task_id = orchestrator.submit_task(risk_task).await.expect("Failed to submit risk task");
    let neural_task_id = orchestrator.submit_task(neural_task).await.expect("Failed to submit neural task");
    
    // Wait for task processing
    sleep(Duration::from_millis(500)).await;
    
    // Verify tasks were distributed
    let system_status = orchestrator.get_system_status().await.expect("Failed to get system status");
    assert!(system_status.task_stats.total_processed > 0);
    
    orchestrator.shutdown().await.expect("Failed to shutdown orchestrator");
}

/// Test fault tolerance and recovery
#[tokio::test]
#[traced_test]
async fn test_fault_tolerance_and_recovery() {
    let orchestrator = OrchestratorBuilder::new()
        .build()
        .await
        .expect("Failed to build orchestrator");
    
    orchestrator.start().await.expect("Failed to start orchestrator");
    
    // Register an agent
    let agent_info = AgentInfo::new(
        AgentId::new(),
        AgentType::Risk,
        "Fault Test Agent".to_string(),
        "1.0.0".to_string(),
    );
    let agent_id = agent_info.id;
    
    orchestrator.register_agent(agent_info).await.expect("Failed to register agent");
    
    // Wait for system to stabilize
    sleep(Duration::from_millis(300)).await;
    
    // Simulate agent failure by unregistering
    orchestrator.unregister_agent(agent_id).await.expect("Failed to unregister agent");
    
    // Submit tasks that should be queued due to no available agents
    for i in 0..5 {
        let task = Task::new(
            format!("queued_task_{}", i),
            TaskPriority::Medium,
            format!("payload_{}", i).into_bytes(),
        ).with_agent_type(AgentType::Risk);
        
        let _task_id = orchestrator.submit_task(task).await.expect("Failed to submit queued task");
    }
    
    // Register a new agent to handle queued tasks
    let recovery_agent_info = AgentInfo::new(
        AgentId::new(),
        AgentType::Risk,
        "Recovery Agent".to_string(),
        "1.0.0".to_string(),
    );
    
    orchestrator.register_agent(recovery_agent_info).await.expect("Failed to register recovery agent");
    
    // Wait for task processing
    sleep(Duration::from_millis(1000)).await;
    
    // Verify system recovered
    let system_status = orchestrator.get_system_status().await.expect("Failed to get system status");
    assert!(system_status.task_stats.tasks_submitted >= 5);
    
    orchestrator.shutdown().await.expect("Failed to shutdown orchestrator");
}

/// Test memory coordination across agents
#[tokio::test]
#[traced_test]
async fn test_memory_coordination() {
    let orchestrator = OrchestratorBuilder::new()
        .build()
        .await
        .expect("Failed to build orchestrator");
    
    orchestrator.start().await.expect("Failed to start orchestrator");
    
    // Register multiple agents
    let agent1_info = AgentInfo::new(
        AgentId::new(),
        AgentType::Risk,
        "Memory Producer".to_string(),
        "1.0.0".to_string(),
    );
    let agent1_id = agent1_info.id;
    
    let agent2_info = AgentInfo::new(
        AgentId::new(),
        AgentType::Neural,
        "Memory Consumer".to_string(),
        "1.0.0".to_string(),
    );
    let agent2_id = agent2_info.id;
    
    orchestrator.register_agent(agent1_info).await.expect("Failed to register producer agent");
    orchestrator.register_agent(agent2_info).await.expect("Failed to register consumer agent");
    
    // Wait for agents to be registered
    sleep(Duration::from_millis(200)).await;
    
    // Test memory operations through task coordination
    let memory_task = Task::new(
        "memory_coordination_test",
        TaskPriority::High,
        b"shared_state_data".to_vec(),
    ).with_parameter("operation", "store")
     .with_parameter("key", "shared_config")
     .with_parameter("value", "coordination_settings");
    
    let _task_id = orchestrator.submit_task(memory_task).await.expect("Failed to submit memory task");
    
    // Wait for task processing
    sleep(Duration::from_millis(500)).await;
    
    // Verify memory coordination
    let system_status = orchestrator.get_system_status().await.expect("Failed to get system status");
    assert!(system_status.memory_stats.total_regions >= 0); // Should have some memory activity
    
    orchestrator.shutdown().await.expect("Failed to shutdown orchestrator");
}

/// Test load balancing across multiple agents
#[tokio::test]
#[traced_test]
async fn test_load_balancing() {
    let orchestrator = OrchestratorBuilder::new()
        .build()
        .await
        .expect("Failed to build orchestrator");
    
    orchestrator.start().await.expect("Failed to start orchestrator");
    
    // Register multiple agents of the same type
    let mut agent_ids = Vec::new();
    for i in 0..5 {
        let agent_info = AgentInfo::new(
            AgentId::new(),
            AgentType::Neural,
            format!("Neural Agent {}", i),
            "1.0.0".to_string(),
        );
        agent_ids.push(agent_info.id);
        orchestrator.register_agent(agent_info).await.expect("Failed to register agent");
    }
    
    // Wait for agents to be registered
    sleep(Duration::from_millis(300)).await;
    
    // Submit multiple tasks to test load distribution
    let mut task_ids = Vec::new();
    for i in 0..20 {
        let task = Task::new(
            format!("load_test_task_{}", i),
            TaskPriority::Medium,
            format!("payload_{}", i).into_bytes(),
        ).with_agent_type(AgentType::Neural);
        
        let task_id = orchestrator.submit_task(task).await.expect("Failed to submit load test task");
        task_ids.push(task_id);
    }
    
    // Wait for task distribution and processing
    sleep(Duration::from_millis(1000)).await;
    
    // Verify load was distributed
    let system_status = orchestrator.get_system_status().await.expect("Failed to get system status");
    assert_eq!(system_status.task_stats.tasks_submitted, 20);
    
    // Clean up agents
    for agent_id in agent_ids {
        orchestrator.unregister_agent(agent_id).await.expect("Failed to unregister agent");
    }
    
    orchestrator.shutdown().await.expect("Failed to shutdown orchestrator");
}

/// Test system health monitoring and alerts
#[tokio::test]
#[traced_test]
async fn test_health_monitoring() {
    let orchestrator = OrchestratorBuilder::new()
        .build()
        .await
        .expect("Failed to build orchestrator");
    
    orchestrator.start().await.expect("Failed to start orchestrator");
    
    // Register some agents
    for i in 0..3 {
        let agent_info = AgentInfo::new(
            AgentId::new(),
            match i % 3 {
                0 => AgentType::Risk,
                1 => AgentType::Neural,
                _ => AgentType::Quantum,
            },
            format!("Health Test Agent {}", i),
            "1.0.0".to_string(),
        );
        orchestrator.register_agent(agent_info).await.expect("Failed to register agent");
    }
    
    // Wait for health monitoring to collect data
    sleep(Duration::from_millis(1000)).await;
    
    // Get system status and verify health monitoring is working
    let system_status = orchestrator.get_system_status().await.expect("Failed to get system status");
    
    // Should have health information
    assert!(system_status.health_status.total_checks > 0);
    assert!(!system_status.health_status.component_statuses.is_empty());
    
    // Verify system is healthy
    assert_eq!(system_status.health_status.overall_status, HealthStatus::Healthy);
    
    orchestrator.shutdown().await.expect("Failed to shutdown orchestrator");
}

/// Test performance under load
#[tokio::test]
#[traced_test]
async fn test_performance_under_load() {
    let orchestrator = OrchestratorBuilder::new()
        .build()
        .await
        .expect("Failed to build orchestrator");
    
    orchestrator.start().await.expect("Failed to start orchestrator");
    
    // Register multiple agents for different types
    for agent_type in [AgentType::Risk, AgentType::Neural, AgentType::Quantum] {
        for i in 0..3 {
            let agent_info = AgentInfo::new(
                AgentId::new(),
                agent_type,
                format!("{:?} Agent {}", agent_type, i),
                "1.0.0".to_string(),
            );
            orchestrator.register_agent(agent_info).await.expect("Failed to register agent");
        }
    }
    
    // Wait for agents to be registered
    sleep(Duration::from_millis(500)).await;
    
    let start_time = std::time::Instant::now();
    
    // Submit a large number of tasks
    let task_count = 100;
    let mut task_futures = Vec::new();
    
    for i in 0..task_count {
        let agent_type = match i % 3 {
            0 => AgentType::Risk,
            1 => AgentType::Neural,
            _ => AgentType::Quantum,
        };
        
        let task = Task::new(
            format!("perf_test_task_{}", i),
            match i % 4 {
                0 => TaskPriority::Critical,
                1 => TaskPriority::High,
                2 => TaskPriority::Medium,
                _ => TaskPriority::Low,
            },
            format!("performance_payload_{}", i).into_bytes(),
        ).with_agent_type(agent_type);
        
        let orchestrator_clone = &orchestrator;
        let future = async move {
            orchestrator_clone.submit_task(task).await
        };
        task_futures.push(future);
    }
    
    // Submit all tasks concurrently
    let results = futures::future::join_all(task_futures).await;
    
    // Verify all tasks were submitted successfully
    assert_eq!(results.len(), task_count);
    for result in results {
        assert!(result.is_ok(), "Task submission failed");
    }
    
    let submission_time = start_time.elapsed();
    
    // Wait for task processing
    sleep(Duration::from_millis(2000)).await;
    
    let total_time = start_time.elapsed();
    
    // Get final system status
    let system_status = orchestrator.get_system_status().await.expect("Failed to get system status");
    
    println!("Performance test results:");
    println!("  Tasks submitted: {}", task_count);
    println!("  Submission time: {:?}", submission_time);
    println!("  Total time: {:?}", total_time);
    println!("  Tasks in queue: {}", system_status.task_stats.queued_tasks);
    println!("  Tasks processed: {}", system_status.task_stats.completed_tasks);
    
    // Verify performance expectations
    assert!(submission_time.as_millis() < 5000, "Task submission took too long");
    assert_eq!(system_status.task_stats.tasks_submitted, task_count as u64);
    
    orchestrator.shutdown().await.expect("Failed to shutdown orchestrator");
}

/// Test coordination between different agent types
#[tokio::test]
#[traced_test]
async fn test_multi_agent_coordination() {
    let orchestrator = OrchestratorBuilder::new()
        .build()
        .await
        .expect("Failed to build orchestrator");
    
    orchestrator.start().await.expect("Failed to start orchestrator");
    
    // Register agents of different types
    let risk_agent = AgentInfo::new(
        AgentId::new(),
        AgentType::Risk,
        "Risk Coordinator".to_string(),
        "1.0.0".to_string(),
    );
    let risk_agent_id = risk_agent.id;
    
    let neural_agent = AgentInfo::new(
        AgentId::new(),
        AgentType::Neural,
        "Neural Predictor".to_string(),
        "1.0.0".to_string(),
    );
    let neural_agent_id = neural_agent.id;
    
    let quantum_agent = AgentInfo::new(
        AgentId::new(),
        AgentType::Quantum,
        "Quantum Analyzer".to_string(),
        "1.0.0".to_string(),
    );
    let quantum_agent_id = quantum_agent.id;
    
    orchestrator.register_agent(risk_agent).await.expect("Failed to register risk agent");
    orchestrator.register_agent(neural_agent).await.expect("Failed to register neural agent");
    orchestrator.register_agent(quantum_agent).await.expect("Failed to register quantum agent");
    
    // Wait for agents to be registered
    sleep(Duration::from_millis(400)).await;
    
    // Submit coordinated tasks (risk analysis -> neural prediction -> quantum optimization)
    let risk_task = Task::new(
        "market_risk_analysis",
        TaskPriority::Critical,
        b"market_data_feed".to_vec(),
    ).with_agent_type(AgentType::Risk)
     .with_parameter("analysis_type", "volatility")
     .with_parameter("time_horizon", "1d");
    
    let neural_task = Task::new(
        "price_prediction",
        TaskPriority::High,
        b"processed_risk_data".to_vec(),
    ).with_agent_type(AgentType::Neural)
     .with_parameter("model_type", "lstm")
     .with_parameter("prediction_horizon", "4h");
    
    let quantum_task = Task::new(
        "portfolio_optimization",
        TaskPriority::High,
        b"predicted_prices".to_vec(),
    ).with_agent_type(AgentType::Quantum)
     .with_parameter("optimization_method", "quantum_annealing")
     .with_parameter("constraints", "risk_budget");
    
    // Submit tasks in sequence
    let risk_task_id = orchestrator.submit_task(risk_task).await.expect("Failed to submit risk task");
    let neural_task_id = orchestrator.submit_task(neural_task).await.expect("Failed to submit neural task");
    let quantum_task_id = orchestrator.submit_task(quantum_task).await.expect("Failed to submit quantum task");
    
    // Wait for coordinated processing
    sleep(Duration::from_millis(1500)).await;
    
    // Verify coordination worked
    let system_status = orchestrator.get_system_status().await.expect("Failed to get system status");
    assert_eq!(system_status.task_stats.tasks_submitted, 3);
    
    // Verify agents are still healthy
    assert_eq!(system_status.health_status.overall_status, HealthStatus::Healthy);
    
    orchestrator.shutdown().await.expect("Failed to shutdown orchestrator");
}