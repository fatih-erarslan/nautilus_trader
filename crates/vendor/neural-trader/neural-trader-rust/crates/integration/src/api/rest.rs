//! REST API server.

use crate::{NeuralTrader, Result};
use std::sync::Arc;
use axum::{
    routing::{get, post},
    Router, Json, extract::State,
};
use tower_http::{
    trace::TraceLayer,
    cors::{CorsLayer, Any},
    compression::CompressionLayer,
};
use tracing::info;

/// REST API server.
pub struct RestApi {
    trader: Arc<NeuralTrader>,
}

impl RestApi {
    /// Creates a new REST API server.
    pub fn new(trader: Arc<NeuralTrader>) -> Self {
        Self { trader }
    }

    /// Builds the Axum router.
    pub fn router(&self) -> Router {
        let state = self.trader.clone();

        Router::new()
            .route("/health", get(health_check))
            .route("/portfolio", get(get_portfolio))
            .route("/risk", get(get_risk_analysis))
            .route("/strategies", get(list_strategies))
            .route("/strategies/:name/execute", post(execute_strategy))
            .route("/models/train", post(train_model))
            .route("/report", get(generate_report))
            .with_state(state)
            .layer(TraceLayer::new_for_http())
            .layer(CompressionLayer::new())
            .layer(
                CorsLayer::new()
                    .allow_origin(Any)
                    .allow_methods(Any)
                    .allow_headers(Any)
            )
    }

    /// Starts the REST API server.
    pub async fn serve(&self, host: &str, port: u16) -> Result<()> {
        let addr = format!("{}:{}", host, port);
        info!("Starting REST API server on {}", addr);

        let listener = tokio::net::TcpListener::bind(&addr).await?;
        let router = self.router();

        axum::serve(listener, router).await?;

        Ok(())
    }
}

// API handlers

async fn health_check(
    State(trader): State<Arc<NeuralTrader>>,
) -> Json<serde_json::Value> {
    match trader.health_check().await {
        Ok(health) => Json(serde_json::to_value(health).unwrap()),
        Err(e) => Json(serde_json::json!({
            "status": "error",
            "message": e.to_string()
        })),
    }
}

async fn get_portfolio(
    State(trader): State<Arc<NeuralTrader>>,
) -> Json<serde_json::Value> {
    match trader.get_portfolio().await {
        Ok(portfolio) => Json(serde_json::to_value(portfolio).unwrap()),
        Err(e) => Json(serde_json::json!({
            "error": e.to_string()
        })),
    }
}

async fn get_risk_analysis(
    State(trader): State<Arc<NeuralTrader>>,
) -> Json<serde_json::Value> {
    match trader.analyze_risk().await {
        Ok(report) => Json(serde_json::to_value(report).unwrap()),
        Err(e) => Json(serde_json::json!({
            "error": e.to_string()
        })),
    }
}

async fn list_strategies(
    State(_trader): State<Arc<NeuralTrader>>,
) -> Json<serde_json::Value> {
    // TODO: Implement
    Json(serde_json::json!({
        "strategies": ["momentum", "mean_reversion", "pairs_trading"]
    }))
}

async fn execute_strategy(
    State(_trader): State<Arc<NeuralTrader>>,
) -> Json<serde_json::Value> {
    // TODO: Implement
    Json(serde_json::json!({
        "status": "executed"
    }))
}

async fn train_model(
    State(_trader): State<Arc<NeuralTrader>>,
) -> Json<serde_json::Value> {
    // TODO: Implement
    Json(serde_json::json!({
        "status": "training"
    }))
}

async fn generate_report(
    State(_trader): State<Arc<NeuralTrader>>,
) -> Json<serde_json::Value> {
    // TODO: Implement
    Json(serde_json::json!({
        "report": "placeholder"
    }))
}
