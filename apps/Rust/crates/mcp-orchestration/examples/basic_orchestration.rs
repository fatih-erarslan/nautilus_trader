//! Basic example of MCP orchestration system usage.

use mcp_orchestration::*;
use std::time::Duration;
use tokio::time::sleep;
use tracing::{info, Level};
use tracing_subscriber;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_max_level(Level::INFO)
        .init();

    info!("Starting MCP Orchestration example...");

    // Create orchestrator with default configuration
    let orchestrator = OrchestratorBuilder::new()
        .build()
        .await?;

    // Start the orchestrator
    orchestrator.start().await?;
    info!("Orchestrator started successfully");

    // Register agents for different types
    let risk_agent = AgentInfo::new(
        AgentId::new(),
        AgentType::Risk,
        "Risk Analysis Agent".to_string(),
        "1.0.0".to_string(),
    );
    let risk_agent_id = risk_agent.id;

    let neural_agent = AgentInfo::new(
        AgentId::new(),
        AgentType::Neural,
        "Neural Prediction Agent".to_string(),
        "1.0.0".to_string(),
    );
    let neural_agent_id = neural_agent.id;

    let quantum_agent = AgentInfo::new(
        AgentId::new(),
        AgentType::Quantum,
        "Quantum Optimization Agent".to_string(),
        "1.0.0".to_string(),
    );
    let quantum_agent_id = quantum_agent.id;

    orchestrator.register_agent(risk_agent).await?;
    orchestrator.register_agent(neural_agent).await?;
    orchestrator.register_agent(quantum_agent).await?;

    info!("Registered 3 agents: Risk, Neural, and Quantum");

    // Wait for agents to be fully registered
    sleep(Duration::from_millis(500)).await;

    // Submit various tasks
    let risk_task = Task::new(
        "market_risk_assessment",
        TaskPriority::Critical,
        b"current_market_data".to_vec(),
    ).with_agent_type(AgentType::Risk)
     .with_parameter("risk_model", "var")
     .with_parameter("confidence_level", "0.95")
     .with_deadline(Timestamp::from_millis(Timestamp::now().as_millis() + 30000)); // 30 second deadline

    let neural_task = Task::new(
        "price_prediction",
        TaskPriority::High,
        b"historical_price_data".to_vec(),
    ).with_agent_type(AgentType::Neural)
     .with_parameter("model_type", "lstm")
     .with_parameter("prediction_horizon", "24h")
     .with_timeout(10000); // 10 second timeout

    let quantum_task = Task::new(
        "portfolio_optimization",
        TaskPriority::High,
        b"risk_and_prediction_data".to_vec(),
    ).with_agent_type(AgentType::Quantum)
     .with_parameter("optimization_method", "qaoa")
     .with_parameter("objective", "maximize_sharpe");

    let risk_task_id = orchestrator.submit_task(risk_task).await?;
    let neural_task_id = orchestrator.submit_task(neural_task).await?;
    let quantum_task_id = orchestrator.submit_task(quantum_task).await?;

    info!("Submitted 3 tasks: Risk Assessment, Price Prediction, and Portfolio Optimization");

    // Submit additional tasks to demonstrate load balancing
    for i in 0..10 {
        let task = Task::new(
            format!("batch_analysis_{}", i),
            TaskPriority::Medium,
            format!("batch_data_{}", i).into_bytes(),
        ).with_agent_type(match i % 3 {
            0 => AgentType::Risk,
            1 => AgentType::Neural,
            _ => AgentType::Quantum,
        });

        let _task_id = orchestrator.submit_task(task).await?;
    }

    info!("Submitted 10 additional batch analysis tasks");

    // Monitor system for a while
    for i in 0..6 {
        sleep(Duration::from_secs(5)).await;
        
        // Get system status
        let status = orchestrator.get_system_status().await?;
        
        info!("=== System Status Update {} ===", i + 1);
        info!("Orchestrator State: {:?}", status.orchestrator_state);
        info!("System Uptime: {}", format::format_duration(status.uptime));
        info!("Overall Health: {:?}", status.health_status.overall_status);
        info!("Active Agents: {}", status.health_status.component_statuses.len());
        info!("Tasks Submitted: {}", status.task_stats.tasks_submitted);
        info!("Tasks Completed: {}", status.task_stats.completed_tasks);
        info!("Tasks Failed: {}", status.task_stats.failed_tasks);
        info!("Queue Depth: {}", status.task_stats.queue_depth);
        info!("Memory Regions: {}", status.memory_stats.total_regions);
        info!("Memory Usage: {}", format::format_bytes(status.memory_stats.memory_usage));
        info!("Coordination Mode: {:?}", status.coordination_state.mode);
        info!("Active Sessions: {}", status.coordination_state.active_sessions.len());

        // Check individual task statuses
        if i == 2 {
            info!("=== Task Status Check ===");
            if let Ok(risk_status) = orchestrator.task_queue.get_task_status(risk_task_id).await {
                info!("Risk Assessment Task: {:?}", risk_status);
            }
            if let Ok(neural_status) = orchestrator.task_queue.get_task_status(neural_task_id).await {
                info!("Neural Prediction Task: {:?}", neural_status);
            }
            if let Ok(quantum_status) = orchestrator.task_queue.get_task_status(quantum_task_id).await {
                info!("Quantum Optimization Task: {:?}", quantum_status);
            }
        }
    }

    // Get comprehensive metrics
    let metrics = orchestrator.get_metrics().await?;
    
    info!("=== Final System Metrics ===");
    info!("Total Agents: {}", metrics.agent_metrics.total_agents);
    info!("Messages Sent: {}", metrics.communication_metrics.messages_sent);
    info!("Messages Received: {}", metrics.communication_metrics.messages_received);
    info!("Tasks Submitted: {}", metrics.task_metrics.tasks_submitted);
    info!("Task Success Rate: {:.2}%", metrics.task_metrics.success_rate * 100.0);
    info!("Memory Cache Hit Ratio: {:.2}%", metrics.memory_metrics.cache_hit_ratio * 100.0);
    info!("System Health Score: {:.1}/100", metrics.health_metrics.system_health_score);
    info!("Recovery Success Rate: {:.2}%", metrics.recovery_metrics.recovery_success_rate * 100.0);

    // Demonstrate memory coordination
    info!("=== Memory Coordination Demo ===");
    
    // Create a shared memory region
    let shared_config_id = orchestrator.shared_memory.create_region(
        "trading_config".to_string(),
        "Shared trading configuration".to_string(),
        risk_agent_id,
        serde_json::to_vec(&serde_json::json!({
            "max_position_size": 1000000,
            "risk_limit": 0.02,
            "trading_hours": "09:30-16:00",
            "instruments": ["BTC", "ETH", "SPY"]
        }))?,
    ).await?;

    info!("Created shared memory region: {}", shared_config_id);

    // Grant access to other agents
    orchestrator.shared_memory.grant_permission(
        shared_config_id,
        risk_agent_id,
        neural_agent_id,
        memory::MemoryPermission::Read,
    ).await?;

    orchestrator.shared_memory.grant_permission(
        shared_config_id,
        risk_agent_id,
        quantum_agent_id,
        memory::MemoryPermission::ReadWrite,
    ).await?;

    info!("Granted memory permissions to Neural (Read) and Quantum (ReadWrite) agents");

    // Read the shared configuration
    let config_region = orchestrator.shared_memory.get_region(shared_config_id, neural_agent_id).await?;
    let config_data: serde_json::Value = serde_json::from_slice(&config_region.data)?;
    info!("Neural agent read shared config: {}", config_data);

    // Update configuration
    let updated_config = serde_json::json!({
        "max_position_size": 1500000,
        "risk_limit": 0.015,
        "trading_hours": "09:30-16:00",
        "instruments": ["BTC", "ETH", "SPY", "QQQ"],
        "last_updated": "quantum_agent",
        "timestamp": Timestamp::now().as_millis()
    });

    orchestrator.shared_memory.update_region(
        shared_config_id,
        quantum_agent_id,
        serde_json::to_vec(&updated_config)?,
        None,
    ).await?;

    info!("Quantum agent updated shared config");

    // List all memory regions accessible by risk agent
    let accessible_regions = orchestrator.shared_memory.list_regions(risk_agent_id).await?;
    info!("Risk agent can access {} memory regions", accessible_regions.len());

    // Demonstrate agent coordination session
    info!("=== Agent Coordination Demo ===");
    
    let session_id = orchestrator.coordination_engine.start_session(
        coordination::CoordinationSessionType::TaskAssignment,
        vec![risk_agent_id, neural_agent_id, quantum_agent_id],
    ).await?;

    info!("Started coordination session: {}", session_id);

    // Submit a coordinated trading task
    let coordinated_task = Task::new(
        "coordinated_trading_strategy",
        TaskPriority::Critical,
        b"live_market_feed".to_vec(),
    ).with_parameter("strategy", "multi_agent_momentum")
     .with_parameter("coordination_session", &session_id)
     .with_parameter("agents", "risk,neural,quantum");

    let coordinated_task_id = orchestrator.submit_task(coordinated_task).await?;
    info!("Submitted coordinated trading task: {}", coordinated_task_id);

    // Wait for coordination
    sleep(Duration::from_secs(5)).await;

    // End coordination session
    orchestrator.coordination_engine.end_session(session_id).await?;
    info!("Ended coordination session");

    // Unregister agents
    orchestrator.unregister_agent(risk_agent_id).await?;
    orchestrator.unregister_agent(neural_agent_id).await?;
    orchestrator.unregister_agent(quantum_agent_id).await?;

    info!("Unregistered all agents");

    // Final system status
    let final_status = orchestrator.get_system_status().await?;
    info!("=== Final System Status ===");
    info!("Total Runtime: {}", format::format_duration(final_status.uptime));
    info!("Total Tasks Processed: {}", final_status.task_stats.completed_tasks + final_status.task_stats.failed_tasks);
    info!("System Health: {:?}", final_status.health_status.overall_status);

    // Shutdown orchestrator
    orchestrator.shutdown().await?;
    info!("Orchestrator shut down successfully");

    info!("MCP Orchestration example completed!");

    Ok(())
}

/// Format utilities module
mod format {
    use std::time::Duration;

    /// Format duration in human-readable form
    pub fn format_duration(duration: Duration) -> String {
        let total_seconds = duration.as_secs();
        let minutes = total_seconds / 60;
        let seconds = total_seconds % 60;
        let millis = duration.subsec_millis();

        if minutes > 0 {
            format!("{}m {}s", minutes, seconds)
        } else if seconds > 0 {
            format!("{}.{:03}s", seconds, millis)
        } else {
            format!("{}ms", millis)
        }
    }

    /// Format bytes in human-readable form
    pub fn format_bytes(bytes: u64) -> String {
        const UNITS: &[&str] = &["B", "KB", "MB", "GB"];
        const THRESHOLD: f64 = 1024.0;

        if bytes == 0 {
            return "0 B".to_string();
        }

        let mut size = bytes as f64;
        let mut unit_index = 0;

        while size >= THRESHOLD && unit_index < UNITS.len() - 1 {
            size /= THRESHOLD;
            unit_index += 1;
        }

        if unit_index == 0 {
            format!("{} {}", bytes, UNITS[unit_index])
        } else {
            format!("{:.2} {}", size, UNITS[unit_index])
        }
    }
}