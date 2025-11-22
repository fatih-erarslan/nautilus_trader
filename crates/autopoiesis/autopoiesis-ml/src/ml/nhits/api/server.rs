use anyhow::Result;
use axum::{
    extract::{Query, State},
    http::{header, Method, StatusCode},
    response::Json,
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    net::SocketAddr,
    sync::Arc,
    time::Duration,
};
use tokio::{
    net::TcpListener,
    signal,
    sync::RwLock,
    time::timeout,
};
use tower::{ServiceBuilder, ServiceExt};
use tower_http::{
    cors::{Any, CorsLayer},
    timeout::TimeoutLayer,
    trace::TraceLayer,
};
use tracing::{error, info, warn};
use uuid::Uuid;

use crate::ml::nhits::api::{
    handlers::*,
    middleware::{auth_middleware, rate_limit_middleware},
    models::*,
    monitoring::MetricsRegistry,
    websocket::websocket_handler,
};

/// Server configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    /// Server listening address
    pub host: String,
    /// Server listening port
    pub port: u16,
    /// Maximum request timeout in seconds
    pub request_timeout: u64,
    /// Enable CORS
    pub enable_cors: bool,
    /// Enable authentication
    pub enable_auth: bool,
    /// Enable rate limiting
    pub enable_rate_limit: bool,
    /// Maximum concurrent connections
    pub max_connections: usize,
    /// Enable metrics collection
    pub enable_metrics: bool,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".to_string(),
            port: 8080,
            request_timeout: 30,
            enable_cors: true,
            enable_auth: true,
            enable_rate_limit: true,
            max_connections: 10000,
            enable_metrics: true,
        }
    }
}

/// Application state shared across handlers
#[derive(Clone)]
pub struct AppState {
    /// NHITS model instances
    pub models: Arc<RwLock<HashMap<String, Box<dyn crate::ml::nhits::NHITSModelTrait + Send + Sync>>>>,
    /// Active forecast jobs
    pub forecast_jobs: Arc<RwLock<HashMap<Uuid, ForecastJob>>>,
    /// Server configuration
    pub config: ServerConfig,
    /// Metrics registry
    pub metrics: Arc<MetricsRegistry>,
    /// WebSocket connections
    pub websocket_connections: Arc<RwLock<HashMap<Uuid, tokio::sync::mpsc::UnboundedSender<String>>>>,
}

impl AppState {
    pub fn new(config: ServerConfig) -> Self {
        Self {
            models: Arc::new(RwLock::new(HashMap::new())),
            forecast_jobs: Arc::new(RwLock::new(HashMap::new())),
            config,
            metrics: Arc::new(MetricsRegistry::new()),
            websocket_connections: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

/// NHITS API Server
pub struct NHITSServer {
    config: ServerConfig,
    app_state: AppState,
}

impl NHITSServer {
    /// Create a new NHITS API server
    pub fn new(config: ServerConfig) -> Self {
        let app_state = AppState::new(config.clone());
        Self { config, app_state }
    }

    /// Build the application router
    fn build_router(&self) -> Router {
        // API routes
        let api_routes = Router::new()
            .route("/health", get(health_check))
            .route("/models", get(list_models).post(create_model))
            .route("/models/:model_id", get(get_model).put(update_model).delete(delete_model))
            .route("/models/:model_id/train", post(train_model))
            .route("/models/:model_id/forecast", post(create_forecast))
            .route("/forecasts", get(list_forecasts))
            .route("/forecasts/:job_id", get(get_forecast))
            .route("/forecasts/:job_id/cancel", post(cancel_forecast))
            .route("/ws", get(websocket_handler))
            .with_state(self.app_state.clone());

        // Metrics route (if enabled)
        let api_routes = if self.config.enable_metrics {
            api_routes.route("/metrics", get(get_metrics))
        } else {
            api_routes
        };

        // Build middleware stack
        let middleware_stack = ServiceBuilder::new()
            .layer(TraceLayer::new_for_http())
            .layer(TimeoutLayer::new(Duration::from_secs(self.config.request_timeout)));

        // Add CORS if enabled
        let middleware_stack = if self.config.enable_cors {
            middleware_stack.layer(
                CorsLayer::new()
                    .allow_origin(Any)
                    .allow_methods([Method::GET, Method::POST, Method::PUT, Method::DELETE])
                    .allow_headers([header::CONTENT_TYPE, header::AUTHORIZATION])
            )
        } else {
            middleware_stack
        };

        // Add rate limiting if enabled
        let api_routes = if self.config.enable_rate_limit {
            api_routes.layer(axum::middleware::from_fn_with_state(
                self.app_state.clone(),
                rate_limit_middleware,
            ))
        } else {
            api_routes
        };

        // Add authentication if enabled
        let api_routes = if self.config.enable_auth {
            api_routes.layer(axum::middleware::from_fn_with_state(
                self.app_state.clone(),
                auth_middleware,
            ))
        } else {
            api_routes
        };

        Router::new()
            .nest("/api/v1", api_routes)
            .layer(middleware_stack)
    }

    /// Start the server
    pub async fn start(&self) -> Result<()> {
        let app = self.build_router();
        let addr = SocketAddr::from(([0, 0, 0, 0], self.config.port));
        
        info!("Starting NHITS API server on {}", addr);
        
        // Initialize metrics if enabled
        if self.config.enable_metrics {
            self.app_state.metrics.init().await?;
            info!("Metrics collection enabled");
        }

        let listener = TcpListener::bind(addr).await?;
        info!("Server listening on {}", addr);

        // Start background tasks
        let cleanup_task = tokio::spawn(Self::cleanup_task(self.app_state.clone()));
        let metrics_task = if self.config.enable_metrics {
            Some(tokio::spawn(Self::metrics_collection_task(self.app_state.clone())))
        } else {
            None
        };

        // Serve with graceful shutdown
        axum::serve(listener, app)
            .with_graceful_shutdown(Self::shutdown_signal())
            .await?;

        // Wait for background tasks to complete
        cleanup_task.abort();
        if let Some(task) = metrics_task {
            task.abort();
        }

        info!("Server shutdown complete");
        Ok(())
    }

    /// Background task for cleaning up expired jobs and connections
    async fn cleanup_task(state: AppState) {
        let mut interval = tokio::time::interval(Duration::from_secs(60));
        
        loop {
            interval.tick().await;
            
            // Clean up expired forecast jobs
            {
                let mut jobs = state.forecast_jobs.write().await;
                let expired_jobs: Vec<Uuid> = jobs
                    .iter()
                    .filter_map(|(id, job)| {
                        if job.is_expired() {
                            Some(*id)
                        } else {
                            None
                        }
                    })
                    .collect();
                
                for job_id in expired_jobs {
                    jobs.remove(&job_id);
                    info!("Cleaned up expired forecast job: {}", job_id);
                }
            }
            
            // Clean up inactive WebSocket connections
            {
                let mut connections = state.websocket_connections.write().await;
                let inactive_connections: Vec<Uuid> = connections
                    .iter()
                    .filter_map(|(id, sender)| {
                        if sender.is_closed() {
                            Some(*id)
                        } else {
                            None
                        }
                    })
                    .collect();
                
                for conn_id in inactive_connections {
                    connections.remove(&conn_id);
                    info!("Cleaned up inactive WebSocket connection: {}", conn_id);
                }
            }
        }
    }

    /// Background task for collecting system metrics
    async fn metrics_collection_task(state: AppState) {
        let mut interval = tokio::time::interval(Duration::from_secs(10));
        
        loop {
            interval.tick().await;
            
            // Update metrics
            let models_count = state.models.read().await.len();
            let jobs_count = state.forecast_jobs.read().await.len();
            let connections_count = state.websocket_connections.read().await.len();
            
            state.metrics.update_system_metrics(
                models_count,
                jobs_count,
                connections_count,
            ).await;
        }
    }

    /// Graceful shutdown signal handler
    async fn shutdown_signal() {
        let ctrl_c = async {
            signal::ctrl_c()
                .await
                .expect("failed to install Ctrl+C handler");
        };

        #[cfg(unix)]
        let terminate = async {
            signal::unix::signal(signal::unix::SignalKind::terminate())
                .expect("failed to install signal handler")
                .recv()
                .await;
        };

        #[cfg(not(unix))]
        let terminate = std::future::pending::<()>();

        tokio::select! {
            _ = ctrl_c => {
                info!("Received Ctrl+C, starting graceful shutdown");
            },
            _ = terminate => {
                info!("Received SIGTERM, starting graceful shutdown");
            },
        }
    }
}

/// Health check endpoint
async fn health_check() -> Result<Json<HealthResponse>, StatusCode> {
    Ok(Json(HealthResponse {
        status: "healthy".to_string(),
        timestamp: chrono::Utc::now(),
        version: env!("CARGO_PKG_VERSION").to_string(),
    }))
}

/// Get metrics endpoint (Prometheus format)
async fn get_metrics(State(state): State<AppState>) -> Result<String, StatusCode> {
    state.metrics.export_prometheus()
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use tower::ServiceExt;

    #[tokio::test]
    async fn test_health_check() {
        let config = ServerConfig::default();
        let server = NHITSServer::new(config);
        let app = server.build_router();

        let response = app
            .oneshot(Request::builder().uri("/api/v1/health").body(Body::empty()).unwrap())
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_server_creation() {
        let config = ServerConfig {
            host: "127.0.0.1".to_string(),
            port: 8081,
            ..Default::default()
        };
        
        let server = NHITSServer::new(config.clone());
        assert_eq!(server.config.host, "127.0.0.1");
        assert_eq!(server.config.port, 8081);
    }
}