use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::db::Database;
use crate::e2b_client::{E2BClient, SandboxConfig, ExecutionRequest};
use crate::openrouter_client::OpenRouterClient;

// ============================================================================
// Request/Response Types
// ============================================================================

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct DeployAgentRequest {
    pub name: String,
    pub agent_type: String,
    pub task_description: String,
    #[serde(default)]
    pub template: String,
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub capabilities: Vec<String>,
    #[serde(default)]
    pub environment: HashMap<String, String>,
    #[serde(default)]
    pub config: serde_json::Value,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct DeploySwarmRequest {
    pub name: String,
    pub topology: String,
    pub strategy: String,
    pub max_agents: i32,
    pub agents: Vec<SwarmAgent>,
    pub task_description: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SwarmAgent {
    pub name: String,
    pub agent_type: String,
    pub task_description: String,
    #[serde(default)]
    pub capabilities: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ExecuteCommandRequest {
    pub command: String,
    #[serde(default)]
    pub env_vars: HashMap<String, String>,
    #[serde(default)]
    pub timeout: Option<u64>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ScaleSwarmRequest {
    pub target_agents: i32,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct AgentStatus {
    pub id: String,
    pub name: String,
    pub agent_type: String,
    pub status: String,
    pub sandbox_id: Option<String>,
    pub task_description: String,
    pub result: Option<serde_json::Value>,
    pub error_message: Option<String>,
    pub tokens_used: i32,
    pub cost_usd: f64,
    pub execution_time_ms: i32,
    pub created_at: String,
    pub completed_at: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SwarmStatus {
    pub id: String,
    pub name: String,
    pub topology: String,
    pub status: String,
    pub total_agents: i32,
    pub active_agents: i32,
    pub agents: Vec<AgentStatus>,
    pub total_cost_usd: f64,
    pub created_at: String,
    pub completed_at: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct LogStreamQuery {
    #[serde(default = "default_lines")]
    pub lines: usize,
    #[serde(default)]
    pub level: Option<String>,
}

fn default_lines() -> usize {
    100
}

// ============================================================================
// Application State
// ============================================================================

pub struct AgentService {
    pub db: Database,
    pub e2b_client: E2BClient,
    pub openrouter_client: OpenRouterClient,
    pub active_agents: Arc<RwLock<HashMap<String, String>>>, // agent_id -> sandbox_id
}

impl AgentService {
    pub fn new(
        db: Database,
        e2b_api_key: String,
        openrouter_api_key: String,
    ) -> Self {
        Self {
            db,
            e2b_client: E2BClient::new(e2b_api_key),
            openrouter_client: OpenRouterClient::new(openrouter_api_key),
            active_agents: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    // Make agent_type field publicly accessible
    fn get_agent_type_field(&self) -> &str {
        "agent_type"
    }
}

// ============================================================================
// Agent Deployment Handlers
// ============================================================================

pub async fn deploy_agent(
    State(agent_service): State<Arc<AgentService>>,
    Json(req): Json<DeployAgentRequest>,
) -> Result<Json<AgentStatus>, AppError> {
    let agent_id = Uuid::new_v4().to_string();
    let template = if req.template.is_empty() { "base".to_string() } else { req.template.clone() };
    let model = if req.model.is_empty() { "anthropic/claude-3.5-sonnet".to_string() } else { req.model.clone() };
    let now = chrono::Utc::now().to_rfc3339();

    let capabilities_json = serde_json::to_string(&req.capabilities).unwrap_or_default();
    let environment_json = serde_json::to_string(&req.environment).unwrap_or_default();
    let config_json = serde_json::to_string(&req.config).unwrap_or_default();

    // Clone values we need after the move
    let agent_id_for_db = agent_id.clone();
    let name_for_db = req.name.clone();
    let agent_type_for_db = req.agent_type.clone();
    let task_desc_for_db = req.task_description.clone();
    let now_for_db = now.clone();

    // Create agent record in database using rusqlite
    let db_path = agent_service.db.db_path.clone();
    tokio::task::spawn_blocking(move || {
        let conn = rusqlite::Connection::open(&db_path)?;
        conn.execute(
            r#"
            INSERT INTO agent_deployments (
                id, name, agent_type, template, model, capabilities,
                environment, config, task_description, status, user_id,
                tokens_used, cost_usd, execution_time_ms, created_at
            )
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, 'pending', '00000000-0000-0000-0000-000000000000',
                    0, 0.0, 0, ?10)
            "#,
            rusqlite::params![
                agent_id_for_db, name_for_db, agent_type_for_db, template, model,
                capabilities_json, environment_json, config_json, task_desc_for_db, now_for_db
            ],
        )?;
        Ok::<_, anyhow::Error>(())
    }).await
    .map_err(|e| AppError::DatabaseError(e.to_string()))?
    .map_err(|e| AppError::DatabaseError(e.to_string()))?;

    // Deploy sandbox and execute task asynchronously
    let agent_service_clone = Arc::clone(&agent_service);
    let req_clone = req.clone();
    let agent_id_clone = agent_id.clone();

    tokio::spawn(async move {
        if let Err(e) = execute_agent_deployment(agent_service_clone, agent_id_clone, req_clone).await {
            eprintln!("Agent deployment failed: {}", e);
        }
    });

    Ok(Json(AgentStatus {
        id: agent_id,
        name: req.name,
        agent_type: req.agent_type,
        status: "pending".to_string(),
        sandbox_id: None,
        task_description: req.task_description,
        result: None,
        error_message: None,
        tokens_used: 0,
        cost_usd: 0.0,
        execution_time_ms: 0,
        created_at: now,
        completed_at: None,
    }))
}

async fn execute_agent_deployment(
    service: Arc<AgentService>,
    agent_id: String,
    req: DeployAgentRequest,
) -> anyhow::Result<()> {
    let start_time = std::time::Instant::now();

    // Update status to running
    let db_path = service.db.db_path.clone();
    let agent_id_clone = agent_id.clone();
    let now = chrono::Utc::now().to_rfc3339();

    tokio::task::spawn_blocking(move || {
        let conn = rusqlite::Connection::open(&db_path)?;
        conn.execute(
            "UPDATE agent_deployments SET status = 'running', started_at = ?1 WHERE id = ?2",
            rusqlite::params![now, agent_id_clone],
        )?;
        Ok::<_, anyhow::Error>(())
    }).await??;

    // Create E2B sandbox
    let sandbox_config = SandboxConfig {
        template: req.template.clone(),
        timeout: Some(3600),
        env_vars: Some(req.environment.clone()),
        metadata: Some(serde_json::json!({
            "agent_id": agent_id,
            "agent_type": req.agent_type,
        })),
    };

    let sandbox = service.e2b_client.create_sandbox(sandbox_config).await?;

    // Store sandbox ID
    let db_path = service.db.db_path.clone();
    let agent_id_clone = agent_id.clone();
    let sandbox_id_clone = sandbox.sandbox_id.clone();
    let sandbox_url_clone = sandbox.url.clone();

    tokio::task::spawn_blocking(move || {
        let conn = rusqlite::Connection::open(&db_path)?;
        conn.execute(
            "UPDATE agent_deployments SET sandbox_id = ?1, sandbox_url = ?2 WHERE id = ?3",
            rusqlite::params![sandbox_id_clone, sandbox_url_clone, agent_id_clone],
        )?;
        Ok::<_, anyhow::Error>(())
    }).await??;

    service.active_agents.write().await.insert(agent_id.clone(), sandbox.sandbox_id.clone());

    // Execute task with OpenRouter
    let (response, usage) = service.openrouter_client
        .execute_agent_task(
            &req.agent_type,
            &req.task_description,
            &req.model,
            None,
        )
        .await?;

    let cost = service.openrouter_client.calculate_cost(&usage, &req.model);
    let execution_time_ms = start_time.elapsed().as_millis() as i32;

    // Store results
    let result = serde_json::json!({
        "response": response,
        "usage": usage,
        "sandbox_id": sandbox.sandbox_id,
    });
    let result_json = serde_json::to_string(&result)?;

    let db_path = service.db.db_path.clone();
    let agent_id_clone = agent_id.clone();
    let now = chrono::Utc::now().to_rfc3339();

    tokio::task::spawn_blocking(move || {
        let conn = rusqlite::Connection::open(&db_path)?;
        conn.execute(
            r#"
            UPDATE agent_deployments
            SET status = 'completed',
                result = ?1,
                tokens_used = ?2,
                cost_usd = ?3,
                execution_time_ms = ?4,
                completed_at = ?5
            WHERE id = ?6
            "#,
            rusqlite::params![result_json, usage.total_tokens as i32, cost, execution_time_ms, now, agent_id_clone],
        )?;
        Ok::<_, anyhow::Error>(())
    }).await??;

    // Cleanup
    service.active_agents.write().await.remove(&agent_id);

    Ok(())
}

// ============================================================================
// Swarm Deployment Handlers
// ============================================================================

pub async fn deploy_swarm(
    State(agent_service): State<Arc<AgentService>>,
    Json(req): Json<DeploySwarmRequest>,
) -> Result<Json<SwarmStatus>, AppError> {
    let swarm_id = Uuid::new_v4().to_string();
    let now = chrono::Utc::now().to_rfc3339();

    // Create swarm configuration using rusqlite
    let db_path = agent_service.db.db_path.clone();
    let swarm_id_clone = swarm_id.clone();
    let name = req.name.clone();
    let topology = req.topology.clone();
    let strategy = req.strategy.clone();
    let max_agents = req.max_agents;
    let now_for_db = now.clone();

    tokio::task::spawn_blocking(move || {
        let conn = rusqlite::Connection::open(&db_path)?;
        conn.execute(
            r#"
            INSERT INTO swarm_configurations (
                id, name, topology, strategy, max_agents, status, user_id,
                total_agents, active_agents, total_cost_usd, created_at
            )
            VALUES (?1, ?2, ?3, ?4, ?5, 'active', '00000000-0000-0000-0000-000000000000',
                    0, 0, 0.0, ?6)
            "#,
            rusqlite::params![swarm_id_clone, name, topology, strategy, max_agents, now_for_db],
        )?;
        Ok::<_, anyhow::Error>(())
    }).await
    .map_err(|e| AppError::DatabaseError(e.to_string()))?
    .map_err(|e| AppError::DatabaseError(e.to_string()))?;

    // Deploy agents asynchronously
    let agent_service_clone = Arc::clone(&agent_service);
    let req_clone = req.clone();
    let swarm_id_clone = swarm_id.clone();

    tokio::spawn(async move {
        if let Err(e) = execute_swarm_deployment(agent_service_clone, swarm_id_clone, req_clone).await {
            eprintln!("Swarm deployment failed: {}", e);
        }
    });

    Ok(Json(SwarmStatus {
        id: swarm_id,
        name: req.name,
        topology: req.topology,
        status: "active".to_string(),
        total_agents: req.agents.len() as i32,
        active_agents: 0,
        agents: vec![],
        total_cost_usd: 0.0,
        created_at: now,
        completed_at: None,
    }))
}

async fn execute_swarm_deployment(
    service: Arc<AgentService>,
    swarm_id: String,
    req: DeploySwarmRequest,
) -> anyhow::Result<()> {
    let mut agent_handles = vec![];

    for agent_def in req.agents {
        let agent_req = DeployAgentRequest {
            name: agent_def.name.clone(),
            agent_type: agent_def.agent_type.clone(),
            task_description: agent_def.task_description.clone(),
            template: "base".to_string(),
            model: "anthropic/claude-3.5-sonnet".to_string(),
            capabilities: agent_def.capabilities.clone(),
            environment: HashMap::new(),
            config: serde_json::json!({"swarm_id": swarm_id}),
        };

        let service_clone = Arc::clone(&service);

        let handle = tokio::spawn(async move {
            let agent_id = Uuid::new_v4().to_string();
            execute_agent_deployment(service_clone, agent_id, agent_req).await
        });

        agent_handles.push(handle);
    }

    // Wait for all agents to complete
    for handle in agent_handles {
        let _ = handle.await;
    }

    // Update swarm status
    let db_path = service.db.db_path.clone();
    let now = chrono::Utc::now().to_rfc3339();

    tokio::task::spawn_blocking(move || {
        let conn = rusqlite::Connection::open(&db_path)?;
        conn.execute(
            "UPDATE swarm_configurations SET status = 'completed', completed_at = ?1 WHERE id = ?2",
            rusqlite::params![now, swarm_id],
        )?;
        Ok::<_, anyhow::Error>(())
    }).await??;

    Ok(())
}

// ============================================================================
// Agent Status Handlers
// ============================================================================

/// List all agents
pub async fn list_agents(
    State(agent_service): State<Arc<AgentService>>,
) -> impl IntoResponse {
    match agent_service.db.list_all_agents().await {
        Ok(agents) => {
            Json(serde_json::json!({
                "agents": agents,
                "count": agents.len()
            })).into_response()
        }
        Err(e) => {
            eprintln!("Failed to list agents: {}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "error": format!("Failed to list agents: {}", e)
                }))
            ).into_response()
        }
    }
}

pub async fn get_agent_status(
    State(agent_service): State<Arc<AgentService>>,
    Path(agent_id): Path<String>,
) -> Result<Json<AgentStatus>, AppError> {
    let db_path = agent_service.db.db_path.clone();

    let agent_status = tokio::task::spawn_blocking(move || {
        let conn = rusqlite::Connection::open(&db_path)?;

        let status = conn.query_row(
            r#"
            SELECT id, name, agent_type, status, sandbox_id, task_description,
                   result, error_message, tokens_used, cost_usd, execution_time_ms,
                   created_at, completed_at
            FROM agent_deployments
            WHERE id = ?1
            "#,
            [&agent_id],
            |row| {
                let result_str: Option<String> = row.get(6)?;
                let result = result_str.and_then(|s| serde_json::from_str(&s).ok());

                Ok(AgentStatus {
                    id: row.get(0)?,
                    name: row.get(1)?,
                    agent_type: row.get(2)?,
                    status: row.get(3)?,
                    sandbox_id: row.get(4)?,
                    task_description: row.get(5)?,
                    result,
                    error_message: row.get(7)?,
                    tokens_used: row.get(8)?,
                    cost_usd: row.get(9)?,
                    execution_time_ms: row.get(10)?,
                    created_at: row.get(11)?,
                    completed_at: row.get(12)?,
                })
            }
        )
        .map_err(|_| anyhow::anyhow!("Agent not found"))?;

        Ok::<_, anyhow::Error>(status)
    }).await
    .map_err(|e| AppError::InternalError(e.to_string()))?
    .map_err(|e| AppError::NotFound(e.to_string()))?;

    Ok(Json(agent_status))
}

pub async fn get_swarm_status(
    State(agent_service): State<Arc<AgentService>>,
    Path(swarm_id): Path<String>,
) -> Result<Json<SwarmStatus>, AppError> {
    let db_path = agent_service.db.db_path.clone();
    let swarm_id_clone = swarm_id.clone();

    let swarm_status = tokio::task::spawn_blocking(move || {
        let conn = rusqlite::Connection::open(&db_path)?;

        let (id, name, topology, status, total_agents, active_agents, total_cost_usd, created_at, completed_at) = conn.query_row(
            r#"
            SELECT id, name, topology, status, total_agents, active_agents,
                   total_cost_usd, created_at, completed_at
            FROM swarm_configurations
            WHERE id = ?1
            "#,
            [&swarm_id_clone],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, i32>(4)?,
                    row.get::<_, i32>(5)?,
                    row.get::<_, f64>(6)?,
                    row.get::<_, String>(7)?,
                    row.get::<_, Option<String>>(8)?,
                ))
            }
        ).map_err(|_| anyhow::anyhow!("Swarm not found"))?;

        // Get agents in swarm - need to handle config field as JSON
        let mut stmt = conn.prepare(
            r#"
            SELECT id, name, agent_type, status, sandbox_id, task_description,
                   result, error_message, tokens_used, cost_usd, execution_time_ms,
                   created_at, completed_at, config
            FROM agent_deployments
            ORDER BY created_at
            "#
        )?;

        let agents: Vec<AgentStatus> = stmt.query_map([], |row| {
            let result_str: Option<String> = row.get(6)?;
            let result = result_str.and_then(|s| serde_json::from_str(&s).ok());

            // Check if this agent belongs to this swarm by checking config JSON
            let config_str: Option<String> = row.get(13)?;
            let belongs_to_swarm = config_str
                .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
                .and_then(|v| v.get("swarm_id").and_then(|s| s.as_str()).map(|s| s == swarm_id_clone))
                .unwrap_or(false);

            if !belongs_to_swarm {
                return Err(rusqlite::Error::QueryReturnedNoRows);
            }

            Ok(AgentStatus {
                id: row.get(0)?,
                name: row.get(1)?,
                agent_type: row.get(2)?,
                status: row.get(3)?,
                sandbox_id: row.get(4)?,
                task_description: row.get(5)?,
                result,
                error_message: row.get(7)?,
                tokens_used: row.get(8)?,
                cost_usd: row.get(9)?,
                execution_time_ms: row.get(10)?,
                created_at: row.get(11)?,
                completed_at: row.get(12)?,
            })
        })?
        .filter_map(|r| r.ok())
        .collect();

        Ok::<_, anyhow::Error>(SwarmStatus {
            id,
            name,
            topology,
            status,
            total_agents,
            active_agents,
            agents,
            total_cost_usd,
            created_at,
            completed_at,
        })
    }).await
    .map_err(|e| AppError::InternalError(e.to_string()))?
    .map_err(|e| AppError::NotFound(e.to_string()))?;

    Ok(Json(swarm_status))
}

// ============================================================================
// Agent Control Handlers
// ============================================================================

pub async fn terminate_agent(
    State(agent_service): State<Arc<AgentService>>,
    Path(agent_id): Path<String>,
) -> Result<StatusCode, AppError> {
    // Get sandbox ID
    let db_path = agent_service.db.db_path.clone();
    let agent_id_clone = agent_id.clone();

    let sandbox_id: Option<String> = tokio::task::spawn_blocking(move || -> anyhow::Result<Option<String>> {
        let conn = rusqlite::Connection::open(&db_path)?;
        let result = conn.query_row(
            "SELECT sandbox_id FROM agent_deployments WHERE id = ?1",
            [&agent_id_clone],
            |row| row.get(0)
        ).ok();
        Ok(result)
    }).await
    .map_err(|e| AppError::InternalError(e.to_string()))?
    .unwrap_or(None);

    // Terminate sandbox if exists
    if let Some(sandbox_id) = sandbox_id {
        let _ = agent_service.e2b_client.terminate_sandbox(&sandbox_id).await;
        agent_service.active_agents.write().await.remove(&agent_id);
    }

    // Update database
    let db_path = agent_service.db.db_path.clone();
    let now = chrono::Utc::now().to_rfc3339();

    tokio::task::spawn_blocking(move || {
        let conn = rusqlite::Connection::open(&db_path)?;
        conn.execute(
            r#"
            UPDATE agent_deployments
            SET status = 'terminated', terminated_at = ?1
            WHERE id = ?2
            "#,
            rusqlite::params![now, agent_id],
        )?;
        Ok::<_, anyhow::Error>(())
    }).await
    .map_err(|e| AppError::InternalError(e.to_string()))?
    .map_err(|e| AppError::DatabaseError(e.to_string()))?;

    Ok(StatusCode::NO_CONTENT)
}

pub async fn execute_command(
    State(agent_service): State<Arc<AgentService>>,
    Path(agent_id): Path<String>,
    Json(req): Json<ExecuteCommandRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    // Get sandbox ID and status
    let db_path = agent_service.db.db_path.clone();
    let agent_id_clone = agent_id.clone();

    let (sandbox_id, status): (Option<String>, String) = tokio::task::spawn_blocking(move || {
        let conn = rusqlite::Connection::open(&db_path)?;
        conn.query_row(
            "SELECT sandbox_id, status FROM agent_deployments WHERE id = ?1",
            [&agent_id_clone],
            |row| Ok((row.get(0)?, row.get(1)?))
        )
    }).await
    .map_err(|e| AppError::InternalError(e.to_string()))?
    .map_err(|_| AppError::NotFound("Agent not found".to_string()))?;

    let sandbox_id = sandbox_id
        .ok_or_else(|| AppError::BadRequest("No sandbox available".to_string()))?;

    if status != "running" && status != "completed" {
        return Err(AppError::BadRequest("Agent is not running".to_string()));
    }

    // Execute command
    let exec_req = ExecutionRequest {
        code: req.command,
        language: Some("bash".to_string()),
        timeout: req.timeout,
        env_vars: Some(req.env_vars),
    };

    let result = agent_service.e2b_client.execute_code(&sandbox_id, exec_req).await
        .map_err(|e| AppError::InternalError(e.to_string()))?;

    Ok(Json(serde_json::to_value(&result).unwrap()))
}

pub async fn stream_agent_logs(
    State(agent_service): State<Arc<AgentService>>,
    Path(agent_id): Path<String>,
    Query(query): Query<LogStreamQuery>,
) -> Result<Json<Vec<serde_json::Value>>, AppError> {
    let db_path = agent_service.db.db_path.clone();
    let level_filter = query.level.clone();
    let lines = query.lines;

    let logs = tokio::task::spawn_blocking(move || {
        let conn = rusqlite::Connection::open(&db_path)?;

        let mut stmt = if let Some(level) = level_filter {
            let mut stmt = conn.prepare(
                r#"
                SELECT log_level, message, metadata, timestamp
                FROM execution_logs
                WHERE agent_id = ?1 AND log_level = ?2
                ORDER BY timestamp DESC
                LIMIT ?3
                "#
            )?;
            stmt
        } else {
            conn.prepare(
                r#"
                SELECT log_level, message, metadata, timestamp
                FROM execution_logs
                WHERE agent_id = ?1
                ORDER BY timestamp DESC
                LIMIT ?2
                "#
            )?
        };

        let logs = if let Some(level) = query.level {
            stmt.query_map(rusqlite::params![agent_id, level, lines as i64], |row| {
                let metadata_str: Option<String> = row.get(2)?;
                let metadata: Option<serde_json::Value> = metadata_str.and_then(|s| serde_json::from_str(&s).ok());

                Ok(serde_json::json!({
                    "level": row.get::<_, String>(0)?,
                    "message": row.get::<_, String>(1)?,
                    "metadata": metadata,
                    "timestamp": row.get::<_, String>(3)?,
                }))
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?
        } else {
            stmt.query_map(rusqlite::params![agent_id, lines as i64], |row| {
                let metadata_str: Option<String> = row.get(2)?;
                let metadata: Option<serde_json::Value> = metadata_str.and_then(|s| serde_json::from_str(&s).ok());

                Ok(serde_json::json!({
                    "level": row.get::<_, String>(0)?,
                    "message": row.get::<_, String>(1)?,
                    "metadata": metadata,
                    "timestamp": row.get::<_, String>(3)?,
                }))
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?
        };

        Ok::<_, anyhow::Error>(logs)
    }).await
    .map_err(|e| AppError::InternalError(e.to_string()))?
    .map_err(|e| AppError::DatabaseError(e.to_string()))?;

    Ok(Json(logs))
}

// ============================================================================
// Swarm Management Handlers
// ============================================================================

pub async fn list_swarms(
    State(agent_service): State<Arc<AgentService>>,
) -> Result<Json<Vec<SwarmStatus>>, AppError> {
    let db_path = agent_service.db.db_path.clone();

    let swarms = tokio::task::spawn_blocking(move || {
        let conn = rusqlite::Connection::open(&db_path)?;

        let mut stmt = conn.prepare(
            r#"
            SELECT id, name, topology, status, total_agents, active_agents,
                   total_cost_usd, created_at, completed_at
            FROM swarm_configurations
            ORDER BY created_at DESC
            LIMIT 100
            "#
        )?;

        let swarms = stmt.query_map([], |row| {
            Ok(SwarmStatus {
                id: row.get(0)?,
                name: row.get(1)?,
                topology: row.get(2)?,
                status: row.get(3)?,
                total_agents: row.get(4)?,
                active_agents: row.get(5)?,
                agents: vec![],
                total_cost_usd: row.get(6)?,
                created_at: row.get(7)?,
                completed_at: row.get(8)?,
            })
        })?
        .collect::<std::result::Result<Vec<_>, _>>()?;

        Ok::<_, anyhow::Error>(swarms)
    }).await
    .map_err(|e| AppError::InternalError(e.to_string()))?
    .map_err(|e| AppError::DatabaseError(e.to_string()))?;

    Ok(Json(swarms))
}

pub async fn scale_swarm(
    State(agent_service): State<Arc<AgentService>>,
    Path(swarm_id): Path<String>,
    Json(req): Json<ScaleSwarmRequest>,
) -> Result<Json<SwarmStatus>, AppError> {
    // Update swarm configuration
    let db_path = agent_service.db.db_path.clone();
    let swarm_id_clone = swarm_id.clone();
    let target_agents = req.target_agents;

    tokio::task::spawn_blocking(move || {
        let conn = rusqlite::Connection::open(&db_path)?;
        conn.execute(
            "UPDATE swarm_configurations SET max_agents = ?1 WHERE id = ?2",
            rusqlite::params![target_agents, swarm_id_clone],
        )?;
        Ok::<_, anyhow::Error>(())
    }).await
    .map_err(|e| AppError::InternalError(e.to_string()))?
    .map_err(|e| AppError::DatabaseError(e.to_string()))?;

    // Return updated status
    get_swarm_status(State(agent_service), Path(swarm_id)).await
}

// ============================================================================
// Error Handling
// ============================================================================

#[derive(Debug)]
pub enum AppError {
    DatabaseError(String),
    NotFound(String),
    BadRequest(String),
    InternalError(String),
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, error_message) = match self {
            AppError::DatabaseError(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg),
            AppError::NotFound(msg) => (StatusCode::NOT_FOUND, msg),
            AppError::BadRequest(msg) => (StatusCode::BAD_REQUEST, msg),
            AppError::InternalError(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg),
        };

        let body = Json(serde_json::json!({
            "error": error_message
        }));

        (status, body).into_response()
    }
}
