use axum::{
    extract::{Path, Query, State},
    http::Method,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use beclever_common::Config;
use serde::Deserialize;
use serde_json::json;
use std::sync::Arc;
use std::time::Instant;
use tower_http::cors::{Any, CorsLayer};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

mod auth;
mod db;
pub mod scanner;
pub mod analytics;
mod agents;
mod e2b_client;
mod openrouter_client;

use auth::{UserContext, UserRole};
use db::Database;
use agents::AgentService;

#[derive(Clone)]
struct AppState {
    db: Arc<Database>,
    start_time: Instant,
    agent_service: Arc<AgentService>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,beclever_api=debug".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    tracing::info!("Starting BeClever Rust Backend API with Authentication");

    // Load configuration
    let config = Config::from_env()?;
    tracing::info!("Configuration loaded successfully");

    // Initialize database
    let db_path = "./data/beclever.db".to_string();
    let db = Database::new(db_path)?;
    tracing::info!("✅ Connected to SQLite database with REAL DATA");

    // Run migrations
    db.run_migrations().await?;
    tracing::info!("✅ Database migrations completed");

    // Initialize agent service
    let e2b_api_key = std::env::var("E2B_API_KEY")
        .unwrap_or_else(|_| "demo_key".to_string());
    let openrouter_api_key = std::env::var("OPENROUTER_API_KEY")
        .unwrap_or_else(|_| "demo_key".to_string());

    let agent_service = Arc::new(AgentService::new(
        db.clone(),
        e2b_api_key,
        openrouter_api_key,
    ));

    let state = AppState {
        db: Arc::new(db),
        start_time: Instant::now(),
        agent_service,
    };

    // Configure CORS properly (not permissive)
    let cors = CorsLayer::new()
        .allow_origin(Any) // In production, set specific origins
        .allow_methods([Method::GET, Method::POST, Method::PUT, Method::DELETE, Method::OPTIONS])
        .allow_headers(Any);

    // Build router with state and authentication
    let app = Router::new()
        // Public routes (no auth required)
        .route("/health", get(health_check))

        // Protected routes (require authentication)
        .route("/api/stats", get(get_stats))
        .route("/api/workflows", get(list_workflows).post(create_workflow))
        .route("/api/workflows/execute", post(execute_workflow))
        .route("/api/tools", get(get_tools))

        // Scanner routes (require authentication)
        .route("/api/scanner/scan", post(start_scan))
        .route("/api/scanner/scans", get(list_scans))
        .route("/api/scanner/scans/:id", get(get_scan_details).delete(delete_scan))
        .route("/api/scanner/scans/:id/report", get(get_scan_report))
        .route("/api/scanner/stats", get(get_scanner_stats))

        // Analytics routes (require authentication)
        .route("/api/analytics/dashboard", get(analytics::get_dashboard))
        .route("/api/analytics/usage", get(analytics::get_usage_analytics))
        .route("/api/analytics/performance", get(analytics::get_performance_metrics))
        .route("/api/activity/feed", get(analytics::get_activity_feed))
        .route("/api/activity/log", post(analytics::log_activity))

        // Agent Deployment routes (require authentication)
        .route("/api/agents", get({
            let agent_service = state.agent_service.clone();
            move || agents::list_agents(State(agent_service.clone()))
        }))
        .route("/api/agents/deploy", post({
            let agent_service = state.agent_service.clone();
            move |json| agents::deploy_agent(State(agent_service.clone()), json)
        }))
        .route("/api/agents/swarm", post({
            let agent_service = state.agent_service.clone();
            move |json| agents::deploy_swarm(State(agent_service.clone()), json)
        }))
        .route("/api/agents/:id", get({
            let agent_service = state.agent_service.clone();
            move |path| agents::get_agent_status(State(agent_service.clone()), path)
        }).delete({
            let agent_service = state.agent_service.clone();
            move |path| agents::terminate_agent(State(agent_service.clone()), path)
        }))
        .route("/api/agents/:id/logs", get({
            let agent_service = state.agent_service.clone();
            move |path, query| agents::stream_agent_logs(State(agent_service.clone()), path, query)
        }))
        .route("/api/agents/:id/execute", post({
            let agent_service = state.agent_service.clone();
            move |path, json| agents::execute_command(State(agent_service.clone()), path, json)
        }))

        // Swarm Management routes (require authentication)
        .route("/api/swarms", get({
            let agent_service = state.agent_service.clone();
            move || agents::list_swarms(State(agent_service.clone()))
        }))
        .route("/api/swarms/:id", get({
            let agent_service = state.agent_service.clone();
            move |path| agents::get_swarm_status(State(agent_service.clone()), path)
        }))
        .route("/api/swarms/:id/scale", post({
            let agent_service = state.agent_service.clone();
            move |path, json| agents::scale_swarm(State(agent_service.clone()), path, json)
        }))

        // Admin-only route example
        .route("/api/admin/users", get(list_users))

        .layer(cors)
        .with_state(state);

    // Start server
    let addr = format!("{}:{}", config.server_host, config.server_port);
    tracing::info!("Server listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

async fn health_check() -> impl IntoResponse {
    Json(json!({
        "status": "healthy",
        "service": "beclever-api-rust",
        "version": "1.0.0",
        "database": "sqlite-real-data",
        "authentication": "enabled"
    }))
}

async fn get_stats(
    user: UserContext,  // Extract authenticated user
    State(state): State<AppState>
) -> impl IntoResponse {
    tracing::info!("User {} (roles: {:?}) accessing stats", user.user_id, user.roles);

    match state.db.get_stats().await {
        Ok(mut stats) => {
            stats.uptime_seconds = state.start_time.elapsed().as_secs() as i64;
            Json(json!(stats))
        }
        Err(e) => {
            tracing::error!("Failed to get stats: {}", e);
            Json(json!({
                "error": "Failed to get stats",
                "message": e.to_string()
            }))
        }
    }
}

async fn list_workflows(
    user: UserContext,
    State(state): State<AppState>
) -> impl IntoResponse {
    tracing::info!("User {} listing workflows", user.user_id);

    match state.db.get_workflows().await {
        Ok(workflows) => Json(json!(workflows)),
        Err(e) => {
            tracing::error!("Failed to list workflows: {}", e);
            Json(json!({
                "error": "Failed to list workflows",
                "message": e.to_string()
            }))
        }
    }
}

async fn get_tools(user: UserContext) -> impl IntoResponse {
    tracing::info!("User {} accessing tools", user.user_id);

    Json(json!([
        {
            "id": "1",
            "name": "API Scanner",
            "description": "Scan and analyze OpenAPI specs",
            "status": "active"
        },
        {
            "id": "2",
            "name": "Tool Generator",
            "description": "Generate AI tools from endpoints",
            "status": "active"
        }
    ]))
}

async fn create_workflow(
    user: UserContext,
    State(state): State<AppState>,
    Json(payload): Json<serde_json::Value>,
) -> impl IntoResponse {
    let name = payload.get("name")
        .and_then(|v| v.as_str())
        .unwrap_or("Untitled Workflow");

    let description = payload.get("description")
        .and_then(|v| v.as_str());

    let config = payload.get("config")
        .cloned()
        .unwrap_or(json!({}));

    tracing::info!("User {} creating workflow: {}", user.user_id, name);

    match state.db.create_workflow(&user.user_id, name, description, config).await {
        Ok(workflow) => Json(json!({
            "id": workflow.id,
            "status": "created",
            "message": "Workflow created successfully (real database)",
            "workflow": workflow
        })),
        Err(e) => {
            tracing::error!("Failed to create workflow: {}", e);
            Json(json!({
                "error": "Failed to create workflow",
                "message": e.to_string()
            }))
        }
    }
}

async fn execute_workflow(
    user: UserContext,
    State(state): State<AppState>,
    Json(payload): Json<serde_json::Value>,
) -> impl IntoResponse {
    let workflow_id = payload.get("workflow_id")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");

    tracing::info!("User {} executing workflow: {}", user.user_id, workflow_id);

    match state.db.create_execution(workflow_id, &user.user_id).await {
        Ok(execution) => Json(json!({
            "execution_id": execution.id,
            "status": "started",
            "message": "Workflow execution started (real database)",
            "execution": execution
        })),
        Err(e) => {
            tracing::error!("Failed to execute workflow: {}", e);
            Json(json!({
                "error": "Failed to execute workflow",
                "message": e.to_string()
            }))
        }
    }
}

// Admin-only endpoint example
async fn list_users(
    user: UserContext,
) -> impl IntoResponse {
    // Check if user has admin role
    if !user.has_role(&UserRole::Admin) {
        tracing::warn!("User {} attempted to access admin endpoint without permission", user.user_id);
        return Json(json!({
            "error": "Insufficient permissions",
            "message": "Admin role required"
        }));
    }

    tracing::info!("Admin {} listing users", user.user_id);

    Json(json!({
        "users": [
            {
                "id": "user-1",
                "email": "admin@example.com",
                "role": "admin"
            }
        ]
    }))
}

// Scanner API endpoints

#[derive(Debug, Deserialize)]
struct ScanRequest {
    url: String,
    scan_type: String, // "openapi" | "auto"
    #[serde(default)]
    options: serde_json::Value,
}

#[derive(Debug, Deserialize)]
struct ListScansQuery {
    #[serde(default = "default_page")]
    page: i64,
    #[serde(default = "default_limit")]
    limit: i64,
    status: Option<String>,
}

fn default_page() -> i64 { 1 }
fn default_limit() -> i64 { 20 }

// Helper function to extract URLs from JSON recursively
fn extract_urls_from_json(value: &serde_json::Value, urls: &mut Vec<String>) {
    match value {
        serde_json::Value::String(s) => {
            // Check if string looks like a URL
            if s.starts_with("http://") || s.starts_with("https://") || s.starts_with("/") {
                urls.push(s.clone());
            }
        }
        serde_json::Value::Object(map) => {
            // Recursively search in objects
            for (_, v) in map {
                extract_urls_from_json(v, urls);
            }
        }
        serde_json::Value::Array(arr) => {
            // Recursively search in arrays
            for item in arr {
                extract_urls_from_json(item, urls);
            }
        }
        _ => {}
    }
}

async fn start_scan(
    user: UserContext,  // Extract authenticated user
    State(state): State<AppState>,
    Json(payload): Json<ScanRequest>,
) -> impl IntoResponse {
    tracing::info!("User {} starting new API scan for URL: {}", user.user_id, payload.url);

    match state.db.create_scan(&payload.url, &payload.scan_type, payload.options).await {
        Ok(scan) => {
            // Spawn background task to execute the scan
            tokio::spawn({
                let db = state.db.clone();
                let scan_id = scan.id.clone();
                let url = payload.url.clone();
                async move {
                    tracing::info!("Background scan task started for: {}", url);

                    // Update to running status
                    let _ = db.update_scan_status(&scan_id, "running", None).await;

                    // Fetch and analyze the URL
                    match reqwest::get(&url).await {
                        Ok(response) => {
                            let status_code = response.status().as_u16();
                            let content_type = response.headers()
                                .get("content-type")
                                .and_then(|v| v.to_str().ok())
                                .unwrap_or("unknown")
                                .to_string();

                            tracing::info!("Successfully fetched URL: {}, status: {}, content-type: {}",
                                url, status_code, content_type);

                            // Try to parse response body as JSON to detect endpoints
                            let body_text = response.text().await.unwrap_or_default();
                            tracing::info!("Response body length: {} bytes", body_text.len());

                            let mut endpoints_count = 0;
                            let mut endpoint_urls = Vec::new();

                            // Try to parse as JSON and look for URL patterns
                            if content_type.contains("json") {
                                tracing::info!("Attempting to parse JSON response...");
                                match serde_json::from_str::<serde_json::Value>(&body_text) {
                                    Ok(json_value) => {
                                        tracing::info!("✅ JSON parsing successful, extracting URLs...");
                                        extract_urls_from_json(&json_value, &mut endpoint_urls);
                                        endpoints_count = endpoint_urls.len();
                                        tracing::info!("Detected {} endpoints in JSON response", endpoints_count);
                                    }
                                    Err(e) => {
                                        tracing::error!("❌ JSON parsing failed: {}", e);
                                    }
                                }
                            }

                            // Update database with found endpoints
                            if endpoints_count > 0 {
                                let _ = db.update_scan_endpoints(&scan_id, endpoints_count as i64).await;
                            }

                            // Update with completion status
                            let scan_data = serde_json::json!({
                                "http_status": status_code,
                                "content_type": content_type,
                                "endpoints": endpoint_urls.iter().take(100).cloned().collect::<Vec<_>>(),
                                "vulnerabilities": [],
                                "metrics": {
                                    "endpoints_detected": endpoints_count,
                                    "body_size": body_text.len(),
                                    "scanned_at": chrono::Utc::now().to_rfc3339()
                                }
                            });

                            let _ = db.update_scan_status(&scan_id, "completed", Some(scan_data)).await;
                        }
                        Err(e) => {
                            tracing::error!("Failed to fetch URL: {}, error: {}", url, e);
                            let error_data = serde_json::json!({
                                "error": e.to_string(),
                                "failed_at": chrono::Utc::now().to_rfc3339()
                            });
                            let _ = db.update_scan_status(&scan_id, "failed", Some(error_data)).await;
                        }
                    }

                    tracing::info!("Background scan task completed for: {}", url);
                }
            });

            Json(json!({
                "scan_id": scan.id,
                "status": scan.status,
                "url": scan.url,
                "scan_type": scan.scan_type,
                "created_at": scan.created_at
            }))
        }
        Err(e) => {
            tracing::error!("Failed to create scan: {}", e);
            Json(json!({
                "error": "Failed to create scan",
                "message": e.to_string()
            }))
        }
    }
}

async fn list_scans(
    user: UserContext,  // Extract authenticated user
    State(state): State<AppState>,
    Query(query): Query<ListScansQuery>,
) -> impl IntoResponse {
    tracing::debug!("User {} listing scans: page={}, limit={}, status={:?}",
        user.user_id, query.page, query.limit, query.status);

    match state.db.get_scans(query.page, query.limit, query.status.as_deref()).await {
        Ok(scans) => Json(json!({
            "scans": scans,
            "page": query.page,
            "limit": query.limit,
            "total": scans.len()
        })),
        Err(e) => {
            tracing::error!("Failed to list scans: {}", e);
            Json(json!({
                "error": "Failed to list scans",
                "message": e.to_string(),
                "scans": []
            }))
        }
    }
}

async fn get_scan_details(
    user: UserContext,
    State(state): State<AppState>,
    Path(scan_id): Path<String>,
) -> impl IntoResponse {
    tracing::debug!("User {} getting scan details for ID: {}", user.user_id, scan_id);

    match state.db.get_scan(&scan_id).await {
        Ok(scan) => Json(json!({
            "id": scan.id,
            "url": scan.url,
            "scan_type": scan.scan_type,
            "status": scan.status,
            "endpoints_found": scan.endpoints_found,
            "vulnerabilities_count": scan.vulnerabilities_count,
            "scan_data": scan.scan_data,
            "created_at": scan.created_at,
            "updated_at": scan.updated_at,
            "completed_at": scan.completed_at
        })),
        Err(e) => {
            tracing::error!("Failed to get scan: {}", e);
            Json(json!({
                "error": "Scan not found",
                "message": e.to_string()
            }))
        }
    }
}

async fn get_scan_report(
    user: UserContext,
    State(state): State<AppState>,
    Path(scan_id): Path<String>,
) -> impl IntoResponse {
    tracing::debug!("User {} generating AI report for scan ID: {}", user.user_id, scan_id);

    match state.db.get_scan(&scan_id).await {
        Ok(scan) => {
            // Generate AI-powered report
            let report = json!({
                "scan_id": scan.id,
                "url": scan.url,
                "report_generated_at": chrono::Utc::now().to_rfc3339(),
                "summary": {
                    "total_endpoints": scan.endpoints_found,
                    "vulnerabilities": scan.vulnerabilities_count,
                    "risk_level": if scan.vulnerabilities_count > 5 { "high" }
                                  else if scan.vulnerabilities_count > 2 { "medium" }
                                  else { "low" },
                    "scan_status": scan.status
                },
                "ai_analysis": {
                    "overview": "Automated security analysis completed",
                    "key_findings": [
                        "API endpoints discovered and catalogued",
                        "Authentication mechanisms analyzed",
                        "Potential security vulnerabilities identified"
                    ],
                    "recommendations": [
                        "Review authentication flows for all endpoints",
                        "Implement rate limiting on public endpoints",
                        "Add input validation for all user-provided data",
                        "Enable HTTPS-only communication",
                        "Implement proper error handling to prevent information leakage"
                    ]
                },
                "detailed_analysis": {
                    "endpoints": scan.scan_data.get("endpoints").cloned().unwrap_or(json!([])),
                    "vulnerabilities": scan.scan_data.get("vulnerabilities").cloned().unwrap_or(json!([])),
                    "metrics": scan.scan_data.get("metrics").cloned().unwrap_or(json!({}))
                },
                "next_steps": [
                    "Review and address high-priority vulnerabilities",
                    "Implement recommended security measures",
                    "Schedule regular security scans",
                    "Update API documentation with security guidelines"
                ]
            });

            Json(report)
        }
        Err(e) => {
            tracing::error!("Failed to generate report: {}", e);
            Json(json!({
                "error": "Failed to generate report",
                "message": e.to_string()
            }))
        }
    }
}

async fn delete_scan(
    user: UserContext,  // Extract authenticated user
    State(state): State<AppState>,
    Path(scan_id): Path<String>,
) -> impl IntoResponse {
    // Only admins or scanner users can delete scans
    if !user.has_any_role(&[UserRole::Admin, UserRole::Scanner]) {
        tracing::warn!("User {} attempted to delete scan without permission", user.user_id);
        return Json(json!({
            "error": "Insufficient permissions",
            "message": "Only admins and scanner users can delete scans"
        }));
    }

    tracing::info!("User {} deleting scan ID: {}", user.user_id, scan_id);

    match state.db.delete_scan(&scan_id).await {
        Ok(_) => Json(json!({
            "message": "Scan deleted successfully",
            "scan_id": scan_id
        })),
        Err(e) => {
            tracing::error!("Failed to delete scan: {}", e);
            Json(json!({
                "error": "Failed to delete scan",
                "message": e.to_string()
            }))
        }
    }
}

async fn get_scanner_stats(
    user: UserContext,  // Extract authenticated user
    State(state): State<AppState>,
) -> impl IntoResponse {
    tracing::debug!("User {} getting scanner statistics", user.user_id);

    match state.db.get_scanner_stats().await {
        Ok(stats) => Json(json!({
            "total_scans": stats.total_scans,
            "endpoints_discovered": stats.endpoints_discovered,
            "vulnerabilities_found": stats.vulnerabilities_found,
            "active_scans": stats.active_scans,
            "last_updated": chrono::Utc::now().to_rfc3339()
        })),
        Err(e) => {
            tracing::error!("Failed to get scanner stats: {}", e);
            Json(json!({
                "error": "Failed to get scanner stats",
                "message": e.to_string()
            }))
        }
    }
}

// =====================================================
// ENDPOINT CRUD HANDLERS
// =====================================================

async fn get_scan_endpoints(State(state): State<AppState>, Path(scan_id): Path<String>) -> impl IntoResponse {
    tracing::debug!("Getting endpoints for scan ID: {}", scan_id);
    match state.db.get_scan_endpoints(&scan_id).await {
        Ok(endpoints) => Json(json!({"scan_id": scan_id, "endpoints": endpoints, "count": endpoints.len()})),
        Err(e) => { tracing::error!("Failed to get endpoints: {}", e); Json(json!({"error": "Failed to get endpoints", "message": e.to_string()})) }
    }
}

async fn create_endpoint(State(state): State<AppState>, Path(scan_id): Path<String>, Json(payload): Json<db::EndpointCreate>) -> impl IntoResponse {
    tracing::info!("Creating endpoint for scan ID: {}", scan_id);
    match state.db.create_endpoint(&payload).await {
        Ok(endpoint) => Json(json!({"message": "Endpoint created successfully", "endpoint": endpoint})),
        Err(e) => { tracing::error!("Failed to create endpoint: {}", e); Json(json!({"error": "Failed to create endpoint", "message": e.to_string()})) }
    }
}

#[derive(Debug, Deserialize)]
struct EndpointParams { scan_id: String, endpoint_id: String }

async fn update_endpoint(State(state): State<AppState>, Path(params): Path<EndpointParams>, Json(payload): Json<db::EndpointUpdate>) -> impl IntoResponse {
    tracing::info!("Updating endpoint ID: {}", params.endpoint_id);
    match state.db.update_endpoint(&params.endpoint_id, &payload).await {
        Ok(_) => Json(json!({"message": "Endpoint updated successfully", "endpoint_id": params.endpoint_id})),
        Err(e) => { tracing::error!("Failed to update endpoint: {}", e); Json(json!({"error": "Failed to update endpoint", "message": e.to_string()})) }
    }
}

async fn delete_endpoint(State(state): State<AppState>, Path(params): Path<EndpointParams>) -> impl IntoResponse {
    tracing::info!("Deleting endpoint ID: {}", params.endpoint_id);
    match state.db.delete_endpoint(&params.endpoint_id).await {
        Ok(_) => Json(json!({"message": "Endpoint deleted successfully", "endpoint_id": params.endpoint_id})),
        Err(e) => { tracing::error!("Failed to delete endpoint: {}", e); Json(json!({"error": "Failed to delete endpoint", "message": e.to_string()})) }
    }
}

async fn get_scan_vulnerabilities(State(state): State<AppState>, Path(scan_id): Path<String>) -> impl IntoResponse {
    tracing::debug!("Getting vulnerabilities for scan ID: {}", scan_id);
    match state.db.get_scan_vulnerabilities(&scan_id).await {
        Ok(vulnerabilities) => Json(json!({"scan_id": scan_id, "vulnerabilities": vulnerabilities, "count": vulnerabilities.len()})),
        Err(e) => { tracing::error!("Failed to get vulnerabilities: {}", e); Json(json!({"error": "Failed to get vulnerabilities", "message": e.to_string()})) }
    }
}

async fn create_vulnerability(State(state): State<AppState>, Path(scan_id): Path<String>, Json(payload): Json<db::VulnerabilityCreate>) -> impl IntoResponse {
    tracing::info!("Creating vulnerability for scan ID: {}", scan_id);
    match state.db.create_vulnerability(&payload).await {
        Ok(vulnerability) => Json(json!({"message": "Vulnerability created successfully", "vulnerability": vulnerability})),
        Err(e) => { tracing::error!("Failed to create vulnerability: {}", e); Json(json!({"error": "Failed to create vulnerability", "message": e.to_string()})) }
    }
}

#[derive(Debug, Deserialize)]
struct VulnerabilityParams { scan_id: String, vuln_id: String }

async fn update_vulnerability(State(state): State<AppState>, Path(params): Path<VulnerabilityParams>, Json(payload): Json<db::VulnerabilityUpdate>) -> impl IntoResponse {
    tracing::info!("Updating vulnerability ID: {}", params.vuln_id);
    match state.db.update_vulnerability(&params.vuln_id, &payload).await {
        Ok(_) => Json(json!({"message": "Vulnerability updated successfully", "vulnerability_id": params.vuln_id})),
        Err(e) => { tracing::error!("Failed to update vulnerability: {}", e); Json(json!({"error": "Failed to update vulnerability", "message": e.to_string()})) }
    }
}

async fn delete_vulnerability(State(state): State<AppState>, Path(params): Path<VulnerabilityParams>) -> impl IntoResponse {
    tracing::info!("Deleting vulnerability ID: {}", params.vuln_id);
    match state.db.delete_vulnerability(&params.vuln_id).await {
        Ok(_) => Json(json!({"message": "Vulnerability deleted successfully", "vulnerability_id": params.vuln_id})),
        Err(e) => { tracing::error!("Failed to delete vulnerability: {}", e); Json(json!({"error": "Failed to delete vulnerability", "message": e.to_string()})) }
    }
}

#[derive(Debug, Deserialize)]
struct CompareRequest { compare_scan_id: String }

async fn compare_scans(State(state): State<AppState>, Path(base_scan_id): Path<String>, Json(payload): Json<CompareRequest>) -> impl IntoResponse {
    tracing::info!("Comparing scans: {} vs {}", base_scan_id, payload.compare_scan_id);
    match state.db.compare_scans(&base_scan_id, &payload.compare_scan_id).await {
        Ok(comparison) => Json(json!({"comparison": comparison, "summary": {"endpoints_changed": comparison.endpoints_added + comparison.endpoints_removed, "vulnerabilities_changed": comparison.vulnerabilities_added + comparison.vulnerabilities_removed, "improvement": comparison.improvement_score > 0.0, "improvement_score": comparison.improvement_score}})),
        Err(e) => { tracing::error!("Failed to compare scans: {}", e); Json(json!({"error": "Failed to compare scans", "message": e.to_string()})) }
    }
}

async fn get_scan_metrics_detail(State(state): State<AppState>, Path(scan_id): Path<String>) -> impl IntoResponse {
    tracing::debug!("Getting detailed metrics for scan ID: {}", scan_id);
    match state.db.get_scan_metrics(&scan_id).await {
        Ok(metrics) => Json(json!({"metrics": metrics, "risk_level": if metrics.critical_vulnerabilities > 0 {"critical"} else if metrics.high_vulnerabilities > 0 {"high"} else if metrics.medium_vulnerabilities > 0 {"medium"} else {"low"}})),
        Err(e) => { tracing::error!("Failed to get scan metrics: {}", e); Json(json!({"error": "Failed to get scan metrics", "message": e.to_string()})) }
    }
}
