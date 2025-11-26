//! Swarm Coordination NAPI Bindings
//!
//! Exposes Rust swarm coordination to Node.js for multi-agent orchestration

use napi::bindgen_prelude::*;
use napi_derive::napi;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;

// =============================================================================
// Type Definitions
// =============================================================================

#[napi(object)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmConfig {
    pub topology: String, // "mesh", "hierarchical", "ring", "star"
    pub max_agents: u32,
    pub strategy: String, // "balanced", "specialized", "adaptive"
    pub enable_reasoning: bool,
}

#[napi(object)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmStatus {
    pub swarm_id: String,
    pub topology: String,
    pub agent_count: u32,
    pub status: String,
    pub uptime_seconds: f64,
}

#[napi(object)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    pub agent_type: String, // "trader", "analyzer", "risk-manager", "monitor"
    pub name: String,
    pub capabilities: Vec<String>,
}

#[napi(object)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentInfo {
    pub agent_id: String,
    pub agent_type: String,
    pub name: String,
    pub status: String,
    pub tasks_completed: u32,
}

#[napi(object)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskRequest {
    pub task_type: String,
    pub description: String,
    pub priority: String, // "low", "medium", "high", "critical"
    pub strategy: String, // "parallel", "sequential", "adaptive"
}

#[napi(object)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskResult {
    pub task_id: String,
    pub status: String,
    pub result: Option<String>,
    pub agents_used: Vec<String>,
    pub execution_time_ms: f64,
}

// Global swarm state
lazy_static::lazy_static! {
    static ref SWARM_STATE: Arc<RwLock<Option<SwarmHandle>>> = Arc::new(RwLock::new(None));
}

#[derive(Debug)]
struct SwarmHandle {
    swarm_id: String,
    config: SwarmConfig,
    start_time: std::time::Instant,
    agents: Vec<AgentInfo>,
}

// =============================================================================
// Swarm Coordination Functions
// =============================================================================

/// Initialize a new swarm
#[napi]
pub async fn swarm_init(config: SwarmConfig) -> Result<String> {
    let mut state = SWARM_STATE.write().await;

    if state.is_some() {
        return Err(Error::from_reason("Swarm already initialized"));
    }

    let swarm_id = uuid::Uuid::new_v4().to_string();

    let handle = SwarmHandle {
        swarm_id: swarm_id.clone(),
        config: config.clone(),
        start_time: std::time::Instant::now(),
        agents: Vec::new(),
    };

    *state = Some(handle);

    // In real implementation, this would initialize the actual swarm
    // from crates/swarm with QUIC coordination

    Ok(swarm_id)
}

/// Spawn a new agent in the swarm
#[napi]
pub async fn swarm_spawn_agent(agent_config: AgentConfig) -> Result<String> {
    let mut state = SWARM_STATE.write().await;

    let handle = state.as_mut()
        .ok_or_else(|| Error::from_reason("Swarm not initialized"))?;

    let agent_id = uuid::Uuid::new_v4().to_string();

    let agent_info = AgentInfo {
        agent_id: agent_id.clone(),
        agent_type: agent_config.agent_type,
        name: agent_config.name,
        status: "idle".to_string(),
        tasks_completed: 0,
    };

    handle.agents.push(agent_info);

    Ok(agent_id)
}

/// Get swarm status
#[napi]
pub async fn swarm_get_status() -> Result<SwarmStatus> {
    let state = SWARM_STATE.read().await;

    match state.as_ref() {
        Some(handle) => Ok(SwarmStatus {
            swarm_id: handle.swarm_id.clone(),
            topology: handle.config.topology.clone(),
            agent_count: handle.agents.len() as u32,
            status: "running".to_string(),
            uptime_seconds: handle.start_time.elapsed().as_secs_f64(),
        }),
        None => Err(Error::from_reason("Swarm not initialized")),
    }
}

/// List all agents in the swarm
#[napi]
pub async fn swarm_list_agents() -> Result<Vec<AgentInfo>> {
    let state = SWARM_STATE.read().await;

    match state.as_ref() {
        Some(handle) => Ok(handle.agents.clone()),
        None => Err(Error::from_reason("Swarm not initialized")),
    }
}

/// Orchestrate a task across the swarm
#[napi]
pub async fn swarm_orchestrate_task(task: TaskRequest) -> Result<TaskResult> {
    let start = std::time::Instant::now();
    let state = SWARM_STATE.read().await;

    let handle = state.as_ref()
        .ok_or_else(|| Error::from_reason("Swarm not initialized"))?;

    // In real implementation, this would distribute the task
    // across agents using the swarm coordination system

    let task_id = uuid::Uuid::new_v4().to_string();
    let agents_used: Vec<String> = handle.agents.iter()
        .take(3)
        .map(|a| a.agent_id.clone())
        .collect();

    Ok(TaskResult {
        task_id,
        status: "completed".to_string(),
        result: Some(format!("Task '{}' completed using {} strategy", task.description, task.strategy)),
        agents_used,
        execution_time_ms: start.elapsed().as_millis() as f64,
    })
}

/// Stop a specific agent
#[napi]
pub async fn swarm_stop_agent(agent_id: String) -> Result<bool> {
    let mut state = SWARM_STATE.write().await;

    let handle = state.as_mut()
        .ok_or_else(|| Error::from_reason("Swarm not initialized"))?;

    handle.agents.retain(|a| a.agent_id != agent_id);

    Ok(true)
}

/// Destroy the swarm
#[napi]
pub async fn swarm_destroy() -> Result<bool> {
    let mut state = SWARM_STATE.write().await;

    if state.is_none() {
        return Err(Error::from_reason("No swarm to destroy"));
    }

    *state = None;

    Ok(true)
}

/// Scale swarm up or down
#[napi]
pub async fn swarm_scale(target_agents: u32) -> Result<u32> {
    let state = SWARM_STATE.read().await;

    let handle = state.as_ref()
        .ok_or_else(|| Error::from_reason("Swarm not initialized"))?;

    let current = handle.agents.len() as u32;

    // In real implementation, would spawn/stop agents to reach target

    Ok(current)
}

/// Monitor swarm health
#[napi]
pub async fn swarm_health_check() -> Result<String> {
    let state = SWARM_STATE.read().await;

    match state.as_ref() {
        Some(handle) => {
            let health = if handle.agents.is_empty() {
                "warning: no agents"
            } else {
                "healthy"
            };
            Ok(health.to_string())
        }
        None => Ok("not_initialized".to_string()),
    }
}
