//! Integration Tests for Agent Deployment System
//!
//! Tests complete flows:
//! 1. Agent deployment → E2B sandbox → OpenRouter integration
//! 2. Swarm coordination and communication
//! 3. Status monitoring and log streaming
//! 4. Multi-agent workflows

use serde_json::json;
use tokio::time::{sleep, Duration};

mod fixtures;
use fixtures::*;

// ============================================================================
// Integration Tests - Complete Deployment Flow
// ============================================================================

#[tokio::test]
async fn test_complete_agent_deployment_flow() {
    let system = MockAgentSystem::new().await.unwrap();

    // 1. Deploy agent
    let config = AgentConfig {
        agent_type: AgentType::Researcher,
        name: "integration-test-agent".to_string(),
        capabilities: vec!["research".to_string(), "analysis".to_string()],
        llm_model: "anthropic/claude-3-sonnet".to_string(),
        ..Default::default()
    };

    let deployment = system.deploy_agent(config).await.unwrap();
    assert_eq!(deployment.status, DeploymentStatus::Running);

    // 2. Send task to agent
    let task = AgentTask {
        task_id: "task-1".to_string(),
        task_type: "research".to_string(),
        prompt: "Research the benefits of Rust programming language".to_string(),
        max_tokens: 1000,
        ..Default::default()
    };

    let result = system.execute_task(&deployment.agent_id, task).await;
    assert!(result.is_ok());

    let execution = result.unwrap();
    assert!(execution.success);
    assert!(!execution.output.is_empty());
    assert!(execution.tokens_used > 0);

    // 3. Check agent status
    let status = system.get_agent_status(&deployment.agent_id).await.unwrap();
    assert_eq!(status.tasks_completed, 1);
    assert!(status.total_tokens_used > 0);

    // 4. Terminate agent
    system.terminate_agent(&deployment.agent_id).await.unwrap();

    let final_status = system.get_agent_status(&deployment.agent_id).await.unwrap();
    assert_eq!(final_status.status, DeploymentStatus::Terminated);
}

#[tokio::test]
async fn test_agent_e2b_openrouter_integration() {
    let system = MockAgentSystem::new().await.unwrap();

    // Deploy coder agent
    let config = AgentConfig {
        agent_type: AgentType::Coder,
        name: "coder-integration".to_string(),
        sandbox_template: "nodejs".to_string(),
        capabilities: vec!["code_generation".to_string(), "testing".to_string()],
        ..Default::default()
    };

    let deployment = system.deploy_agent(config).await.unwrap();

    // Execute coding task that requires both E2B and OpenRouter
    let task = AgentTask {
        task_id: "code-task-1".to_string(),
        task_type: "code_generation".to_string(),
        prompt: "Write a Node.js function to calculate fibonacci numbers".to_string(),
        execute_code: true, // Should execute in sandbox
        ..Default::default()
    };

    let execution = system.execute_task(&deployment.agent_id, task).await.unwrap();

    // Verify both LLM and code execution happened
    assert!(execution.success);
    assert!(execution.llm_response.is_some());
    assert!(execution.code_output.is_some());

    let code_output = execution.code_output.unwrap();
    assert_eq!(code_output.exit_code, 0);
}

#[tokio::test]
async fn test_agent_file_operations_integration() {
    let system = MockAgentSystem::new().await.unwrap();

    let deployment = system.deploy_agent(AgentConfig::default()).await.unwrap();

    // Upload code file
    let code = r#"
        function greet(name) {
            return `Hello, ${name}!`;
        }
        console.log(greet('World'));
    "#;

    system.upload_file(&deployment.agent_id, "/app/greet.js", code).await.unwrap();

    // Execute the file
    let task = AgentTask {
        task_id: "file-exec-1".to_string(),
        task_type: "execute".to_string(),
        prompt: "node /app/greet.js".to_string(),
        execute_code: true,
        ..Default::default()
    };

    let execution = system.execute_task(&deployment.agent_id, task).await.unwrap();
    assert!(execution.code_output.is_some());

    let output = execution.code_output.unwrap();
    assert!(output.stdout.contains("Hello, World!"));
}

#[tokio::test]
async fn test_agent_streaming_responses() {
    let system = MockAgentSystem::new().await.unwrap();

    let deployment = system.deploy_agent(AgentConfig::default()).await.unwrap();

    let task = AgentTask {
        task_id: "stream-1".to_string(),
        task_type: "generation".to_string(),
        prompt: "Write a detailed explanation of async/await in JavaScript".to_string(),
        stream_response: true,
        ..Default::default()
    };

    let mut stream = system.execute_task_streaming(&deployment.agent_id, task).await.unwrap();

    let mut chunks = Vec::new();
    while let Some(chunk) = stream.next().await {
        chunks.push(chunk);
    }

    assert!(!chunks.is_empty());

    // Verify streaming delivered content incrementally
    assert!(chunks.len() > 1);
}

// ============================================================================
// Integration Tests - Swarm Coordination
// ============================================================================

#[tokio::test]
async fn test_deploy_agent_swarm() {
    let system = MockAgentSystem::new().await.unwrap();

    let swarm_config = SwarmConfig {
        topology: SwarmTopology::Mesh,
        agent_count: 5,
        agent_types: vec![
            (AgentType::Coordinator, 1),
            (AgentType::Researcher, 2),
            (AgentType::Coder, 2),
        ],
        ..Default::default()
    };

    let swarm = system.deploy_swarm(swarm_config).await.unwrap();

    assert_eq!(swarm.agents.len(), 5);
    assert_eq!(swarm.topology, SwarmTopology::Mesh);
    assert_eq!(swarm.status, SwarmStatus::Active);

    // Verify agent types distribution
    let coordinators: Vec<_> = swarm.agents.iter()
        .filter(|a| a.agent_type == AgentType::Coordinator)
        .collect();
    assert_eq!(coordinators.len(), 1);
}

#[tokio::test]
async fn test_swarm_task_orchestration() {
    let system = MockAgentSystem::new().await.unwrap();

    let swarm_config = SwarmConfig {
        topology: SwarmTopology::Hierarchical,
        agent_count: 3,
        agent_types: vec![
            (AgentType::Coordinator, 1),
            (AgentType::Researcher, 1),
            (AgentType::Analyst, 1),
        ],
        ..Default::default()
    };

    let swarm = system.deploy_swarm(swarm_config).await.unwrap();

    // Orchestrate complex task across swarm
    let complex_task = SwarmTask {
        task_id: "swarm-task-1".to_string(),
        description: "Research and analyze Rust ecosystem trends".to_string(),
        subtasks: vec![
            SubTask {
                agent_type: AgentType::Researcher,
                prompt: "Research latest Rust ecosystem trends".to_string(),
            },
            SubTask {
                agent_type: AgentType::Analyst,
                prompt: "Analyze the research findings".to_string(),
            },
        ],
        coordination_strategy: CoordinationStrategy::Sequential,
    };

    let result = system.execute_swarm_task(&swarm.swarm_id, complex_task).await.unwrap();

    assert!(result.success);
    assert_eq!(result.subtask_results.len(), 2);

    // Verify sequential execution
    let timestamps: Vec<_> = result.subtask_results.iter()
        .map(|r| r.completed_at)
        .collect();
    assert!(timestamps.windows(2).all(|w| w[0] < w[1]));
}

#[tokio::test]
async fn test_swarm_parallel_execution() {
    let system = MockAgentSystem::new().await.unwrap();

    let swarm_config = SwarmConfig {
        topology: SwarmTopology::Star,
        agent_count: 4,
        agent_types: vec![
            (AgentType::Coordinator, 1),
            (AgentType::Coder, 3),
        ],
        ..Default::default()
    };

    let swarm = system.deploy_swarm(swarm_config).await.unwrap();

    // Parallel code generation task
    let parallel_task = SwarmTask {
        task_id: "parallel-code-1".to_string(),
        description: "Generate multiple code modules in parallel".to_string(),
        subtasks: vec![
            SubTask {
                agent_type: AgentType::Coder,
                prompt: "Write authentication module".to_string(),
            },
            SubTask {
                agent_type: AgentType::Coder,
                prompt: "Write database module".to_string(),
            },
            SubTask {
                agent_type: AgentType::Coder,
                prompt: "Write API module".to_string(),
            },
        ],
        coordination_strategy: CoordinationStrategy::Parallel,
    };

    let start = std::time::Instant::now();
    let result = system.execute_swarm_task(&swarm.swarm_id, parallel_task).await.unwrap();
    let duration = start.elapsed();

    assert!(result.success);
    assert_eq!(result.subtask_results.len(), 3);

    // Parallel should be faster than sequential
    // (Mock will simulate this with appropriate delays)
    assert!(duration.as_secs() < 5);
}

#[tokio::test]
async fn test_swarm_agent_communication() {
    let system = MockAgentSystem::new().await.unwrap();

    let swarm_config = SwarmConfig {
        topology: SwarmTopology::Mesh,
        agent_count: 3,
        enable_agent_communication: true,
        ..Default::default()
    };

    let swarm = system.deploy_swarm(swarm_config).await.unwrap();

    // Send message from one agent to another
    let agent_1 = &swarm.agents[0].agent_id;
    let agent_2 = &swarm.agents[1].agent_id;

    let message = AgentMessage {
        from_agent: agent_1.clone(),
        to_agent: agent_2.clone(),
        message_type: "data_share".to_string(),
        content: json!({
            "data": "Research findings on Rust async runtime",
            "priority": "high"
        }),
    };

    let result = system.send_agent_message(&swarm.swarm_id, message).await;
    assert!(result.is_ok());

    // Verify message was received
    let messages = system.get_agent_messages(agent_2, 10).await.unwrap();
    assert!(!messages.is_empty());
    assert_eq!(messages[0].from_agent, *agent_1);
}

#[tokio::test]
async fn test_swarm_scaling() {
    let system = MockAgentSystem::new().await.unwrap();

    let swarm_config = SwarmConfig {
        topology: SwarmTopology::Hierarchical,
        agent_count: 3,
        ..Default::default()
    };

    let mut swarm = system.deploy_swarm(swarm_config).await.unwrap();
    assert_eq!(swarm.agents.len(), 3);

    // Scale up
    let scale_result = system.scale_swarm(&swarm.swarm_id, 5).await;
    assert!(scale_result.is_ok());

    swarm = system.get_swarm(&swarm.swarm_id).await.unwrap();
    assert_eq!(swarm.agents.len(), 5);

    // Scale down
    let scale_down = system.scale_swarm(&swarm.swarm_id, 2).await;
    assert!(scale_down.is_ok());

    swarm = system.get_swarm(&swarm.swarm_id).await.unwrap();
    assert_eq!(swarm.agents.len(), 2);
}

// ============================================================================
// Integration Tests - Status Monitoring
// ============================================================================

#[tokio::test]
async fn test_real_time_status_monitoring() {
    let system = MockAgentSystem::new().await.unwrap();

    let deployment = system.deploy_agent(AgentConfig::default()).await.unwrap();

    // Monitor status over time
    let mut status_snapshots = Vec::new();

    for _ in 0..5 {
        let status = system.get_agent_status(&deployment.agent_id).await.unwrap();
        status_snapshots.push(status);
        sleep(Duration::from_millis(200)).await;
    }

    // Verify uptime is increasing
    for i in 1..status_snapshots.len() {
        assert!(status_snapshots[i].uptime_seconds >= status_snapshots[i-1].uptime_seconds);
    }
}

#[tokio::test]
async fn test_agent_health_checks() {
    let system = MockAgentSystem::new().await.unwrap();

    let deployment = system.deploy_agent(AgentConfig::default()).await.unwrap();

    // Perform health check
    let health = system.check_agent_health(&deployment.agent_id).await.unwrap();

    assert_eq!(health.status, HealthStatus::Healthy);
    assert!(health.sandbox_responsive);
    assert!(health.llm_accessible);
    assert!(health.cpu_usage < 100.0);
    assert!(health.memory_mb > 0);
}

#[tokio::test]
async fn test_agent_metrics_collection() {
    let system = MockAgentSystem::new().await.unwrap();

    let deployment = system.deploy_agent(AgentConfig::default()).await.unwrap();

    // Execute some tasks to generate metrics
    for i in 0..10 {
        let task = AgentTask {
            task_id: format!("task-{}", i),
            task_type: "test".to_string(),
            prompt: format!("Task {}", i),
            ..Default::default()
        };

        system.execute_task(&deployment.agent_id, task).await.unwrap();
    }

    // Get metrics
    let metrics = system.get_agent_metrics(&deployment.agent_id, 100).await.unwrap();

    assert!(!metrics.is_empty());
    assert!(metrics.iter().any(|m| m.requests_processed > 0));
    assert!(metrics.iter().any(|m| m.tokens_used > 0));
}

// ============================================================================
// Integration Tests - Log Streaming
// ============================================================================

#[tokio::test]
async fn test_agent_log_streaming() {
    let system = MockAgentSystem::new().await.unwrap();

    let deployment = system.deploy_agent(AgentConfig::default()).await.unwrap();

    // Start log stream
    let mut log_stream = system.stream_agent_logs(&deployment.agent_id).await.unwrap();

    // Execute task to generate logs
    let task = AgentTask {
        task_id: "log-test-1".to_string(),
        task_type: "test".to_string(),
        prompt: "Generate some logs".to_string(),
        execute_code: true,
        ..Default::default()
    };

    system.execute_task(&deployment.agent_id, task).await.unwrap();

    // Collect log entries
    let mut logs = Vec::new();
    for _ in 0..10 {
        if let Some(log) = log_stream.next().await {
            logs.push(log);
        } else {
            break;
        }
    }

    assert!(!logs.is_empty());
    assert!(logs.iter().any(|l| l.level == LogLevel::Info));
}

#[tokio::test]
async fn test_swarm_log_aggregation() {
    let system = MockAgentSystem::new().await.unwrap();

    let swarm_config = SwarmConfig {
        agent_count: 3,
        ..Default::default()
    };

    let swarm = system.deploy_swarm(swarm_config).await.unwrap();

    // Execute swarm task
    let task = SwarmTask {
        task_id: "swarm-log-test".to_string(),
        description: "Test logging".to_string(),
        subtasks: vec![
            SubTask {
                agent_type: AgentType::Researcher,
                prompt: "Task 1".to_string(),
            },
            SubTask {
                agent_type: AgentType::Analyst,
                prompt: "Task 2".to_string(),
            },
        ],
        coordination_strategy: CoordinationStrategy::Parallel,
    };

    system.execute_swarm_task(&swarm.swarm_id, task).await.unwrap();

    // Get aggregated logs
    let logs = system.get_swarm_logs(&swarm.swarm_id, 100).await.unwrap();

    assert!(!logs.is_empty());

    // Verify logs from different agents
    let agent_ids: std::collections::HashSet<_> = logs.iter()
        .map(|l| &l.agent_id)
        .collect();
    assert!(agent_ids.len() > 1);
}

// ============================================================================
// Integration Tests - Error Recovery
// ============================================================================

#[tokio::test]
async fn test_agent_auto_restart_on_failure() {
    let system = MockAgentSystem::new().await.unwrap();

    let config = AgentConfig {
        auto_restart: true,
        max_restart_attempts: 3,
        ..Default::default()
    };

    let deployment = system.deploy_agent(config).await.unwrap();

    // Simulate agent failure
    system.simulate_agent_crash(&deployment.agent_id).await.unwrap();

    // Wait for auto-restart
    sleep(Duration::from_secs(2)).await;

    let status = system.get_agent_status(&deployment.agent_id).await.unwrap();
    assert_eq!(status.status, DeploymentStatus::Running);
    assert_eq!(status.restart_count, 1);
}

#[tokio::test]
async fn test_swarm_fault_tolerance() {
    let system = MockAgentSystem::new().await.unwrap();

    let swarm_config = SwarmConfig {
        agent_count: 5,
        fault_tolerance_enabled: true,
        ..Default::default()
    };

    let swarm = system.deploy_swarm(swarm_config).await.unwrap();

    // Kill one agent
    let agent_to_kill = &swarm.agents[2].agent_id;
    system.terminate_agent(agent_to_kill).await.unwrap();

    // Swarm should still be operational
    let swarm_status = system.get_swarm_status(&swarm.swarm_id).await.unwrap();
    assert_eq!(swarm_status.status, SwarmStatus::Degraded);
    assert_eq!(swarm_status.active_agents, 4);

    // Execute task - should work with remaining agents
    let task = SwarmTask {
        task_id: "fault-test-1".to_string(),
        description: "Test with degraded swarm".to_string(),
        subtasks: vec![
            SubTask {
                agent_type: AgentType::Researcher,
                prompt: "Continue working".to_string(),
            },
        ],
        coordination_strategy: CoordinationStrategy::Sequential,
    };

    let result = system.execute_swarm_task(&swarm.swarm_id, task).await;
    assert!(result.is_ok());
}
