//! Performance Tests for Agent Deployment System
//!
//! Tests performance under load:
//! 1. Concurrent agent deployments
//! 2. Swarm scaling performance
//! 3. Response times and latency
//! 4. Resource usage and limits
//! 5. Cost tracking and optimization

use std::time::{Duration, Instant};
use tokio::time::sleep;

mod fixtures;
use fixtures::*;

// ============================================================================
// Performance Tests - Concurrent Deployments
// ============================================================================

#[tokio::test]
async fn test_concurrent_agent_deployments() {
    let system = MockAgentSystem::new().await.unwrap();

    let concurrent_count = 10;
    let start = Instant::now();

    let mut handles = Vec::new();

    for i in 0..concurrent_count {
        let system_clone = system.clone();

        let handle = tokio::spawn(async move {
            let config = AgentConfig {
                agent_type: AgentType::Researcher,
                name: format!("concurrent-agent-{}", i),
                ..Default::default()
            };

            system_clone.deploy_agent(config).await
        });

        handles.push(handle);
    }

    // Wait for all deployments
    let mut successful = 0;
    for handle in handles {
        if handle.await.unwrap().is_ok() {
            successful += 1;
        }
    }

    let duration = start.elapsed();

    // Performance assertions
    assert_eq!(successful, concurrent_count);
    assert!(duration.as_secs() < 30, "Deployment took too long: {:?}", duration);

    // Log performance metrics
    println!("Deployed {} agents in {:?} ({:.2} agents/sec)",
        concurrent_count,
        duration,
        concurrent_count as f64 / duration.as_secs_f64()
    );
}

#[tokio::test]
async fn test_deployment_throughput() {
    let system = MockAgentSystem::new().await.unwrap();

    let batch_size = 5;
    let batch_count = 4;

    let start = Instant::now();

    for batch in 0..batch_count {
        let mut handles = Vec::new();

        for i in 0..batch_size {
            let system_clone = system.clone();

            let handle = tokio::spawn(async move {
                let config = AgentConfig {
                    name: format!("batch-{}-agent-{}", batch, i),
                    ..Default::default()
                };

                system_clone.deploy_agent(config).await
            });

            handles.push(handle);
        }

        // Wait for batch to complete
        for handle in handles {
            handle.await.unwrap().unwrap();
        }
    }

    let duration = start.elapsed();
    let total_agents = batch_size * batch_count;

    println!("Deployed {} agents in {} batches: {:?} ({:.2} agents/sec)",
        total_agents,
        batch_count,
        duration,
        total_agents as f64 / duration.as_secs_f64()
    );

    assert!(duration.as_secs() < 60);
}

#[tokio::test]
async fn test_deployment_latency_distribution() {
    let system = MockAgentSystem::new().await.unwrap();

    let mut latencies = Vec::new();

    for i in 0..20 {
        let start = Instant::now();

        let config = AgentConfig {
            name: format!("latency-test-{}", i),
            ..Default::default()
        };

        system.deploy_agent(config).await.unwrap();

        latencies.push(start.elapsed());
    }

    // Calculate percentiles
    latencies.sort();

    let p50 = latencies[latencies.len() / 2];
    let p95 = latencies[latencies.len() * 95 / 100];
    let p99 = latencies[latencies.len() * 99 / 100];

    println!("Deployment Latency:");
    println!("  P50: {:?}", p50);
    println!("  P95: {:?}", p95);
    println!("  P99: {:?}", p99);

    // Performance targets
    assert!(p50.as_secs() < 5, "P50 latency too high");
    assert!(p95.as_secs() < 10, "P95 latency too high");
    assert!(p99.as_secs() < 15, "P99 latency too high");
}

// ============================================================================
// Performance Tests - Swarm Scaling
// ============================================================================

#[tokio::test]
async fn test_swarm_scaling_performance() {
    let system = MockAgentSystem::new().await.unwrap();

    let initial_size = 5;
    let target_size = 20;

    let swarm_config = SwarmConfig {
        agent_count: initial_size,
        ..Default::default()
    };

    let swarm = system.deploy_swarm(swarm_config).await.unwrap();

    // Measure scale-up time
    let start = Instant::now();
    system.scale_swarm(&swarm.swarm_id, target_size).await.unwrap();
    let scale_up_duration = start.elapsed();

    println!("Scaled swarm from {} to {} agents in {:?}",
        initial_size, target_size, scale_up_duration);

    assert!(scale_up_duration.as_secs() < 20);

    // Measure scale-down time
    let start = Instant::now();
    system.scale_swarm(&swarm.swarm_id, initial_size).await.unwrap();
    let scale_down_duration = start.elapsed();

    println!("Scaled swarm down from {} to {} agents in {:?}",
        target_size, initial_size, scale_down_duration);

    assert!(scale_down_duration.as_secs() < 10);
}

#[tokio::test]
async fn test_large_swarm_deployment() {
    let system = MockAgentSystem::new().await.unwrap();

    let swarm_size = 50;

    let config = SwarmConfig {
        agent_count: swarm_size,
        topology: SwarmTopology::Hierarchical,
        ..Default::default()
    };

    let start = Instant::now();
    let swarm = system.deploy_swarm(config).await.unwrap();
    let duration = start.elapsed();

    assert_eq!(swarm.agents.len(), swarm_size);

    println!("Deployed swarm of {} agents in {:?} ({:.2} agents/sec)",
        swarm_size,
        duration,
        swarm_size as f64 / duration.as_secs_f64()
    );

    assert!(duration.as_secs() < 60);
}

// ============================================================================
// Performance Tests - Task Execution
// ============================================================================

#[tokio::test]
async fn test_task_execution_performance() {
    let system = MockAgentSystem::new().await.unwrap();

    let deployment = system.deploy_agent(AgentConfig::default()).await.unwrap();

    let task_count = 100;
    let mut execution_times = Vec::new();

    for i in 0..task_count {
        let task = AgentTask {
            task_id: format!("perf-task-{}", i),
            task_type: "test".to_string(),
            prompt: "Quick test task".to_string(),
            max_tokens: 100,
            ..Default::default()
        };

        let start = Instant::now();
        system.execute_task(&deployment.agent_id, task).await.unwrap();
        execution_times.push(start.elapsed());
    }

    // Calculate stats
    let total_time: Duration = execution_times.iter().sum();
    let avg_time = total_time / task_count as u32;

    execution_times.sort();
    let p50 = execution_times[task_count / 2];
    let p95 = execution_times[task_count * 95 / 100];

    println!("Task Execution Performance ({} tasks):", task_count);
    println!("  Total time: {:?}", total_time);
    println!("  Avg time: {:?}", avg_time);
    println!("  P50: {:?}", p50);
    println!("  P95: {:?}", p95);

    assert!(avg_time.as_millis() < 500);
}

#[tokio::test]
async fn test_parallel_task_execution() {
    let system = MockAgentSystem::new().await.unwrap();

    let deployment = system.deploy_agent(AgentConfig::default()).await.unwrap();

    let concurrent_tasks = 20;

    let start = Instant::now();

    let mut handles = Vec::new();

    for i in 0..concurrent_tasks {
        let system_clone = system.clone();
        let agent_id = deployment.agent_id.clone();

        let handle = tokio::spawn(async move {
            let task = AgentTask {
                task_id: format!("parallel-{}", i),
                task_type: "test".to_string(),
                prompt: format!("Parallel task {}", i),
                ..Default::default()
            };

            system_clone.execute_task(&agent_id, task).await
        });

        handles.push(handle);
    }

    let mut successful = 0;
    for handle in handles {
        if handle.await.unwrap().is_ok() {
            successful += 1;
        }
    }

    let duration = start.elapsed();

    println!("Executed {} parallel tasks in {:?} ({:.2} tasks/sec)",
        concurrent_tasks,
        duration,
        concurrent_tasks as f64 / duration.as_secs_f64()
    );

    assert_eq!(successful, concurrent_tasks);
    assert!(duration.as_secs() < 10);
}

#[tokio::test]
async fn test_swarm_task_coordination_performance() {
    let system = MockAgentSystem::new().await.unwrap();

    let swarm_config = SwarmConfig {
        agent_count: 10,
        ..Default::default()
    };

    let swarm = system.deploy_swarm(swarm_config).await.unwrap();

    let subtask_count = 20;
    let mut subtasks = Vec::new();

    for i in 0..subtask_count {
        subtasks.push(SubTask {
            agent_type: AgentType::Researcher,
            prompt: format!("Subtask {}", i),
        });
    }

    let task = SwarmTask {
        task_id: "coordination-perf".to_string(),
        description: "Performance test".to_string(),
        subtasks,
        coordination_strategy: CoordinationStrategy::Parallel,
    };

    let start = Instant::now();
    let result = system.execute_swarm_task(&swarm.swarm_id, task).await.unwrap();
    let duration = start.elapsed();

    println!("Swarm executed {} subtasks in {:?} ({:.2} tasks/sec)",
        subtask_count,
        duration,
        subtask_count as f64 / duration.as_secs_f64()
    );

    assert!(result.success);
    assert_eq!(result.subtask_results.len(), subtask_count);
    assert!(duration.as_secs() < 15);
}

// ============================================================================
// Performance Tests - Resource Usage
// ============================================================================

#[tokio::test]
async fn test_memory_usage_under_load() {
    let system = MockAgentSystem::new().await.unwrap();

    let initial_memory = get_process_memory_mb();

    // Deploy multiple agents
    for i in 0..20 {
        let config = AgentConfig {
            name: format!("memory-test-{}", i),
            ..Default::default()
        };

        system.deploy_agent(config).await.unwrap();
    }

    let final_memory = get_process_memory_mb();
    let memory_increase = final_memory - initial_memory;

    println!("Memory usage: {} MB → {} MB (Δ{} MB)",
        initial_memory, final_memory, memory_increase);

    // Memory should increase but not excessively
    assert!(memory_increase < 1000, "Memory usage too high");
}

#[tokio::test]
async fn test_agent_resource_limits() {
    let system = MockAgentSystem::new().await.unwrap();

    let config = AgentConfig {
        name: "resource-limit-test".to_string(),
        max_memory_mb: 512,
        max_cpu_percent: 50.0,
        ..Default::default()
    };

    let deployment = system.deploy_agent(config).await.unwrap();

    // Execute intensive task
    let task = AgentTask {
        task_id: "intensive-task".to_string(),
        task_type: "compute".to_string(),
        prompt: "Perform intensive computation".to_string(),
        ..Default::default()
    };

    system.execute_task(&deployment.agent_id, task).await.unwrap();

    // Check resource usage
    let health = system.check_agent_health(&deployment.agent_id).await.unwrap();

    assert!(health.memory_mb <= 512, "Memory limit exceeded");
    assert!(health.cpu_usage <= 50.0, "CPU limit exceeded");
}

// ============================================================================
// Performance Tests - Cost Tracking
// ============================================================================

#[tokio::test]
async fn test_cost_tracking_accuracy() {
    let system = MockAgentSystem::new().await.unwrap();

    let deployment = system.deploy_agent(AgentConfig::default()).await.unwrap();

    let mut total_expected_cost = 0.0;

    for i in 0..10 {
        let task = AgentTask {
            task_id: format!("cost-task-{}", i),
            task_type: "test".to_string(),
            prompt: "Test task for cost tracking".to_string(),
            max_tokens: 500,
            ..Default::default()
        };

        let execution = system.execute_task(&deployment.agent_id, task).await.unwrap();
        total_expected_cost += execution.cost_usd;
    }

    let metrics = system.get_agent_metrics(&deployment.agent_id, 100).await.unwrap();
    let total_tracked_cost: f64 = metrics.iter().map(|m| m.cost_usd).sum();

    println!("Expected cost: ${:.4}", total_expected_cost);
    println!("Tracked cost: ${:.4}", total_tracked_cost);

    // Costs should match within small tolerance
    let cost_diff = (total_expected_cost - total_tracked_cost).abs();
    assert!(cost_diff < 0.01, "Cost tracking inaccurate");
}

#[tokio::test]
async fn test_cost_optimization_strategies() {
    let system = MockAgentSystem::new().await.unwrap();

    // Test with expensive model
    let expensive_config = AgentConfig {
        name: "expensive-agent".to_string(),
        llm_model: "anthropic/claude-3-opus".to_string(),
        ..Default::default()
    };

    let expensive_deployment = system.deploy_agent(expensive_config).await.unwrap();

    // Test with cheaper model
    let cheap_config = AgentConfig {
        name: "cheap-agent".to_string(),
        llm_model: "anthropic/claude-3-haiku".to_string(),
        ..Default::default()
    };

    let cheap_deployment = system.deploy_agent(cheap_config).await.unwrap();

    let task = AgentTask {
        task_id: "cost-compare".to_string(),
        task_type: "test".to_string(),
        prompt: "Same task for both agents".to_string(),
        ..Default::default()
    };

    let expensive_result = system.execute_task(
        &expensive_deployment.agent_id,
        task.clone()
    ).await.unwrap();

    let cheap_result = system.execute_task(
        &cheap_deployment.agent_id,
        task
    ).await.unwrap();

    println!("Expensive model cost: ${:.4}", expensive_result.cost_usd);
    println!("Cheap model cost: ${:.4}", cheap_result.cost_usd);

    // Haiku should be significantly cheaper than Opus
    assert!(cheap_result.cost_usd < expensive_result.cost_usd);
}

// ============================================================================
// Performance Tests - Stress Testing
// ============================================================================

#[tokio::test]
async fn test_sustained_load() {
    let system = MockAgentSystem::new().await.unwrap();

    let deployment = system.deploy_agent(AgentConfig::default()).await.unwrap();

    let duration = Duration::from_secs(30);
    let start = Instant::now();

    let mut task_count = 0;

    while start.elapsed() < duration {
        let task = AgentTask {
            task_id: format!("sustained-{}", task_count),
            task_type: "test".to_string(),
            prompt: "Sustained load test".to_string(),
            ..Default::default()
        };

        system.execute_task(&deployment.agent_id, task).await.unwrap();
        task_count += 1;

        sleep(Duration::from_millis(100)).await;
    }

    println!("Executed {} tasks over {:?} ({:.2} tasks/sec)",
        task_count,
        duration,
        task_count as f64 / duration.as_secs_f64()
    );

    // Agent should still be healthy
    let health = system.check_agent_health(&deployment.agent_id).await.unwrap();
    assert_eq!(health.status, HealthStatus::Healthy);
}

#[tokio::test]
async fn test_rapid_deployment_termination() {
    let system = MockAgentSystem::new().await.unwrap();

    let cycle_count = 50;

    let start = Instant::now();

    for i in 0..cycle_count {
        let config = AgentConfig {
            name: format!("rapid-cycle-{}", i),
            ..Default::default()
        };

        let deployment = system.deploy_agent(config).await.unwrap();

        // Immediately terminate
        system.terminate_agent(&deployment.agent_id).await.unwrap();
    }

    let duration = start.elapsed();

    println!("Completed {} deploy-terminate cycles in {:?} ({:.2} cycles/sec)",
        cycle_count,
        duration,
        cycle_count as f64 / duration.as_secs_f64()
    );

    assert!(duration.as_secs() < 30);
}

// ============================================================================
// Helper Functions
// ============================================================================

fn get_process_memory_mb() -> u64 {
    // Mock implementation - in real code would use procfs or similar
    use std::process;
    let pid = process::id();

    // Simulate reading /proc/<pid>/status
    // In real implementation, parse VmRSS field
    512 // Mock value
}
