use axum::{
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use beclever_common::Config;
use serde_json::json;
use tower_http::cors::CorsLayer;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

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

    tracing::info!("Starting BeClever Rust Backend API");

    // Load configuration
    let config = Config::from_env()?;
    tracing::info!("Configuration loaded successfully");

    // Running in MOCK MODE - database bypassed for quick development
    tracing::warn!("⚠️  Running in MOCK MODE - serving mock data");
    tracing::warn!("⚠️  Perfect for testing the API without database!");

    // Build router - no state needed for mock endpoints
    let app = Router::new()
        .route("/health", get(health_check))
        .route("/api/stats", get(get_stats))
        .route("/api/workflows", post(create_workflow))
        .route("/api/workflows/execute", post(execute_workflow))
        .route("/api/tools", get(get_tools))
        .layer(CorsLayer::permissive());

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
        "version": "1.0.0"
    }))
}

async fn get_stats() -> impl IntoResponse {
    Json(json!({
        "total_workflows": 42,
        "active_executions": 3,
        "uptime_seconds": 12345,
        "mock_mode": true
    }))
}

async fn get_tools() -> impl IntoResponse {
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
    Json(_payload): Json<serde_json::Value>,
) -> impl IntoResponse {
    Json(json!({
        "id": uuid::Uuid::new_v4(),
        "status": "created",
        "message": "Workflow created successfully (mock mode)"
    }))
}

async fn execute_workflow(
    Json(_payload): Json<serde_json::Value>,
) -> impl IntoResponse {
    Json(json!({
        "execution_id": uuid::Uuid::new_v4(),
        "status": "started",
        "message": "Workflow execution started (mock mode)"
    }))
}
