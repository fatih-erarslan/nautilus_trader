//! CQGS Real-time Monitoring Dashboard
//!
//! Provides comprehensive real-time visualization and monitoring of all 49 sentinels,
//! hyperbolic coordination topology, consensus decisions, and quality governance metrics.

use axum::{
    extract::{ws::WebSocket, Query, State, WebSocketUpgrade},
    http::StatusCode,
    response::{Html, IntoResponse},
    routing::{get, post},
    Json, Router,
};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{broadcast, Mutex, RwLock};
use tokio::time::{interval, Duration};
use tower::ServiceBuilder;
use tower_http::cors::CorsLayer;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use crate::cqgs::consensus::{ConsensusMetrics, ConsensusResult, ConsensusSession};
use crate::cqgs::coordination::TopologyMetrics;
use crate::cqgs::sentinels::{SentinelId, SentinelType};
use crate::cqgs::SentinelStatus;
use crate::cqgs::{
    CqgsEvent, HyperbolicCoordinates, QualityGateDecision, QualityViolation, SentinelMetrics,
    SystemStatus, ViolationSeverity,
};

/// Dashboard configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardConfig {
    pub port: u16,
    pub host: String,
    pub update_interval_ms: u64,
    pub max_history_items: usize,
    pub theme: DashboardTheme,
    pub enable_real_time: bool,
    pub enable_notifications: bool,
}

impl Default for DashboardConfig {
    fn default() -> Self {
        Self {
            port: 8080,
            host: "0.0.0.0".to_string(),
            update_interval_ms: 1000, // 1 second updates
            max_history_items: 1000,
            theme: DashboardTheme::HyperbolicDark,
            enable_real_time: true,
            enable_notifications: true,
        }
    }
}

/// Dashboard visual themes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DashboardTheme {
    HyperbolicDark,
    NeuralBlue,
    HiveGold,
    CqgsCorporate,
}

/// Real-time dashboard data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardData {
    pub timestamp: std::time::SystemTime,
    pub system_status: SystemStatus,
    pub sentinel_metrics: HashMap<SentinelId, SentinelMetrics>,
    pub topology_metrics: TopologyMetrics,
    pub consensus_metrics: ConsensusMetrics,
    pub recent_violations: Vec<QualityViolation>,
    pub recent_events: Vec<CqgsEvent>,
    pub hyperbolic_visualization: HyperbolicVisualization,
    pub performance_stats: PerformanceStats,
}

/// Hyperbolic topology visualization data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HyperbolicVisualization {
    pub sentinel_positions: HashMap<SentinelId, HyperbolicCoordinates>,
    pub communication_edges: Vec<CommunicationEdge>,
    pub centroid: HyperbolicCoordinates,
    pub stability_score: f64,
    pub coverage_area: f64,
}

/// Communication edge between sentinels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunicationEdge {
    pub from: SentinelId,
    pub to: SentinelId,
    pub efficiency: f64,
    pub distance: f64,
    pub message_count: u64,
}

/// Performance statistics for dashboard
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceStats {
    pub total_violations_detected: u64,
    pub total_violations_resolved: u64,
    pub average_resolution_time_ms: f64,
    pub system_uptime: Duration,
    pub memory_usage_mb: f64,
    pub cpu_usage_percent: f64,
    pub network_throughput: f64,
}

/// WebSocket message types for real-time updates
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum WebSocketMessage {
    DashboardUpdate {
        data: DashboardData,
    },
    ViolationAlert {
        violation: QualityViolation,
    },
    ConsensusUpdate {
        session_id: Uuid,
        result: Option<ConsensusResult>,
    },
    SystemAlert {
        level: String,
        message: String,
    },
    TopologyChange {
        new_positions: HashMap<SentinelId, HyperbolicCoordinates>,
    },
    PerformanceMetrics {
        stats: PerformanceStats,
    },
}

/// Dashboard state shared across handlers
pub struct DashboardState {
    pub config: DashboardConfig,
    pub current_data: Arc<RwLock<DashboardData>>,
    pub event_history: Arc<Mutex<Vec<CqgsEvent>>>,
    pub violation_history: Arc<Mutex<Vec<QualityViolation>>>,
    pub websocket_clients: Arc<DashMap<Uuid, broadcast::Sender<WebSocketMessage>>>,
    pub event_receiver: Arc<Mutex<broadcast::Receiver<CqgsEvent>>>,
}

/// CQGS Dashboard Server
pub struct CqgsDashboard {
    state: Arc<DashboardState>,
    router: Router,
}

impl CqgsDashboard {
    /// Create new dashboard instance
    pub fn new(config: DashboardConfig, event_receiver: broadcast::Receiver<CqgsEvent>) -> Self {
        let initial_data = DashboardData {
            timestamp: std::time::SystemTime::now(),
            system_status: SystemStatus {
                active_sentinels: 0,
                total_violations: 0,
                resolved_violations: 0,
                system_health: 1.0,
                uptime: Duration::from_secs(0),
                last_consensus: std::time::SystemTime::now(),
                hyperbolic_stability: 1.0,
            },
            sentinel_metrics: HashMap::new(),
            topology_metrics: TopologyMetrics {
                sentinel_count: 0,
                stability: 1.0,
                communication_energy: 0.0,
                coverage_area: 0.0,
                centroid: HyperbolicCoordinates {
                    x: 0.0,
                    y: 0.0,
                    radius: 0.0,
                },
                curvature: -1.5,
            },
            consensus_metrics: ConsensusMetrics::default(),
            recent_violations: Vec::new(),
            recent_events: Vec::new(),
            hyperbolic_visualization: HyperbolicVisualization {
                sentinel_positions: HashMap::new(),
                communication_edges: Vec::new(),
                centroid: HyperbolicCoordinates {
                    x: 0.0,
                    y: 0.0,
                    radius: 0.0,
                },
                stability_score: 1.0,
                coverage_area: 0.0,
            },
            performance_stats: PerformanceStats {
                total_violations_detected: 0,
                total_violations_resolved: 0,
                average_resolution_time_ms: 0.0,
                system_uptime: Duration::from_secs(0),
                memory_usage_mb: 0.0,
                cpu_usage_percent: 0.0,
                network_throughput: 0.0,
            },
        };

        let state = Arc::new(DashboardState {
            config,
            current_data: Arc::new(RwLock::new(initial_data)),
            event_history: Arc::new(Mutex::new(Vec::new())),
            violation_history: Arc::new(Mutex::new(Vec::new())),
            websocket_clients: Arc::new(DashMap::new()),
            event_receiver: Arc::new(Mutex::new(event_receiver)),
        });

        let router = create_dashboard_router(Arc::clone(&state));

        Self { state, router }
    }

    /// Start the dashboard server
    pub async fn start(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let addr = format!("{}:{}", self.state.config.host, self.state.config.port);
        let listener = tokio::net::TcpListener::bind(&addr).await?;

        info!("CQGS Dashboard starting on http://{}", addr);

        // Start background tasks
        let state_clone = Arc::clone(&self.state);
        tokio::spawn(async move {
            dashboard_update_loop(state_clone).await;
        });

        let state_clone = Arc::clone(&self.state);
        tokio::spawn(async move {
            event_processor_loop(state_clone).await;
        });

        // Start server
        axum::serve(listener, self.router.clone()).await?;

        Ok(())
    }

    /// Update dashboard data
    pub async fn update_data(&self, new_data: DashboardData) {
        let mut current_data = self.state.current_data.write().await;
        *current_data = new_data.clone();

        // Broadcast to websocket clients
        self.broadcast_to_clients(WebSocketMessage::DashboardUpdate { data: new_data })
            .await;
    }

    /// Broadcast message to all websocket clients
    async fn broadcast_to_clients(&self, message: WebSocketMessage) {
        for client in self.state.websocket_clients.iter() {
            if let Err(e) = client.value().send(message.clone()) {
                warn!("Failed to send message to websocket client: {}", e);
                // Client disconnected, remove it
                self.state.websocket_clients.remove(client.key());
            }
        }
    }
}

/// Create the dashboard router with all endpoints
fn create_dashboard_router(state: Arc<DashboardState>) -> Router {
    Router::new()
        .route("/", get(dashboard_home))
        .route("/api/status", get(get_system_status))
        .route("/api/sentinels", get(get_sentinel_metrics))
        .route("/api/violations", get(get_recent_violations))
        .route("/api/consensus", get(get_consensus_metrics))
        .route("/api/topology", get(get_topology_metrics))
        .route("/api/performance", get(get_performance_stats))
        .route("/ws", get(websocket_handler))
        .route("/config", get(get_config).post(update_config))
        .with_state(state)
        .layer(
            ServiceBuilder::new()
                .layer(CorsLayer::permissive())
                .into_inner(),
        )
}

/// Dashboard home page handler
async fn dashboard_home() -> impl IntoResponse {
    Html(include_str!("dashboard.html"))
}

/// System status API endpoint
async fn get_system_status(State(state): State<Arc<DashboardState>>) -> impl IntoResponse {
    let data = state.current_data.read().await;
    Json(data.system_status.clone())
}

/// Sentinel metrics API endpoint
async fn get_sentinel_metrics(State(state): State<Arc<DashboardState>>) -> impl IntoResponse {
    let data = state.current_data.read().await;
    Json(data.sentinel_metrics.clone())
}

/// Recent violations API endpoint
async fn get_recent_violations(
    State(state): State<Arc<DashboardState>>,
    Query(params): Query<HashMap<String, String>>,
) -> impl IntoResponse {
    let limit = params
        .get("limit")
        .and_then(|l| l.parse::<usize>().ok())
        .unwrap_or(100);

    let violations = state.violation_history.lock().await;
    let recent: Vec<_> = violations.iter().rev().take(limit).cloned().collect();

    Json(recent)
}

/// Consensus metrics API endpoint
async fn get_consensus_metrics(State(state): State<Arc<DashboardState>>) -> impl IntoResponse {
    let data = state.current_data.read().await;
    Json(data.consensus_metrics.clone())
}

/// Topology metrics API endpoint
async fn get_topology_metrics(State(state): State<Arc<DashboardState>>) -> impl IntoResponse {
    let data = state.current_data.read().await;
    Json(data.topology_metrics.clone())
}

/// Performance statistics API endpoint
async fn get_performance_stats(State(state): State<Arc<DashboardState>>) -> impl IntoResponse {
    let data = state.current_data.read().await;
    Json(data.performance_stats.clone())
}

/// WebSocket handler for real-time updates
async fn websocket_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<DashboardState>>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| websocket_connection(socket, state))
}

/// Handle individual WebSocket connection
async fn websocket_connection(mut socket: WebSocket, state: Arc<DashboardState>) {
    use futures_util::{SinkExt, StreamExt};

    let client_id = Uuid::new_v4();
    let (tx, mut rx) = broadcast::channel(1000);

    // Register client
    state.websocket_clients.insert(client_id, tx.clone());

    info!("WebSocket client {} connected", client_id);

    // Send initial data
    let initial_data = state.current_data.read().await.clone();
    let initial_message = WebSocketMessage::DashboardUpdate { data: initial_data };

    if let Ok(msg_json) = serde_json::to_string(&initial_message) {
        if socket
            .send(axum::extract::ws::Message::Text(msg_json))
            .await
            .is_err()
        {
            state.websocket_clients.remove(&client_id);
            return;
        }
    }

    // Handle incoming and outgoing messages
    loop {
        tokio::select! {
            // Handle incoming WebSocket messages
            msg = socket.recv() => {
                match msg {
                    Some(Ok(msg)) => {
                        // Handle incoming messages if needed
                        debug!("Received WebSocket message: {:?}", msg);
                    },
                    Some(Err(e)) => {
                        error!("WebSocket error: {}", e);
                        break;
                    },
                    None => break,
                }
            },
            // Handle outgoing messages from broadcast
            Ok(message) = rx.recv() => {
                if let Ok(msg_json) = serde_json::to_string(&message) {
                    if socket.send(axum::extract::ws::Message::Text(msg_json)).await.is_err() {
                        break;
                    }
                }
            },
            else => break,
        }
    }

    // Clean up
    state.websocket_clients.remove(&client_id);
    info!("WebSocket client {} disconnected", client_id);
}

/// Get dashboard configuration
async fn get_config(State(state): State<Arc<DashboardState>>) -> impl IntoResponse {
    Json(state.config.clone())
}

/// Update dashboard configuration
async fn update_config(
    State(state): State<Arc<DashboardState>>,
    Json(new_config): Json<DashboardConfig>,
) -> impl IntoResponse {
    // In a real implementation, this would update the config
    // For now, just return success
    info!("Dashboard configuration update requested");
    StatusCode::OK
}

/// Background loop for dashboard data updates
async fn dashboard_update_loop(state: Arc<DashboardState>) {
    let mut interval = interval(Duration::from_millis(state.config.update_interval_ms));

    loop {
        interval.tick().await;

        // In a real implementation, this would collect data from the CQGS system
        // For now, we'll simulate updates
        update_dashboard_data(&state).await;
    }
}

/// Update dashboard data from CQGS system
async fn update_dashboard_data(state: &Arc<DashboardState>) {
    // This would integrate with the actual CQGS system to collect real data
    // For now, we'll create simulated updates

    let mut data = state.current_data.write().await;
    data.timestamp = std::time::SystemTime::now();

    // Update performance stats (simulated)
    data.performance_stats.system_uptime += Duration::from_millis(state.config.update_interval_ms);
    data.performance_stats.memory_usage_mb = 256.0 + (rand::random::<f64>() * 50.0);
    data.performance_stats.cpu_usage_percent = 15.0 + (rand::random::<f64>() * 25.0);

    // Update system health score
    data.system_status.system_health = 0.95 + (rand::random::<f64>() * 0.05);
}

/// Background loop for processing CQGS events
async fn event_processor_loop(state: Arc<DashboardState>) {
    loop {
        let event_opt = {
            let mut receiver = state.event_receiver.lock().await;
            receiver.recv().await.ok()
        };

        if let Some(event) = event_opt {
            // Process the event
            match &event {
                CqgsEvent::ViolationDetected { violation } => {
                    // Add to violation history
                    let mut violations = state.violation_history.lock().await;
                    violations.push(violation.clone());

                    // Keep only recent violations
                    if violations.len() > state.config.max_history_items {
                        violations.remove(0);
                    }

                    // Broadcast violation alert
                    let message = WebSocketMessage::ViolationAlert {
                        violation: violation.clone(),
                    };
                    broadcast_message(&state, message).await;
                }
                CqgsEvent::ConsensusReached {
                    decision,
                    vote_count,
                } => {
                    let message = WebSocketMessage::SystemAlert {
                        level: "info".to_string(),
                        message: format!(
                            "Consensus reached: {:?} ({} votes)",
                            decision, vote_count
                        ),
                    };
                    broadcast_message(&state, message).await;
                }
                _ => {}
            }

            // Add to event history
            let mut events = state.event_history.lock().await;
            events.push(event);

            // Keep only recent events
            if events.len() > state.config.max_history_items {
                events.remove(0);
            }
        }
    }
}

/// Broadcast message to all WebSocket clients
async fn broadcast_message(state: &Arc<DashboardState>, message: WebSocketMessage) {
    for client in state.websocket_clients.iter() {
        if let Err(_) = client.value().send(message.clone()) {
            // Client disconnected, will be cleaned up later
        }
    }
}

/// Dashboard HTML template
const DASHBOARD_HTML: &str = include_str!("dashboard.html");

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::broadcast;

    #[tokio::test]
    async fn test_dashboard_creation() {
        let (_, rx) = broadcast::channel(100);
        let config = DashboardConfig::default();
        let dashboard = CqgsDashboard::new(config, rx);

        assert_eq!(dashboard.state.config.port, 8080);
    }

    #[tokio::test]
    async fn test_dashboard_data_update() {
        let (_, rx) = broadcast::channel(100);
        let config = DashboardConfig::default();
        let dashboard = CqgsDashboard::new(config, rx);

        let test_data = DashboardData {
            timestamp: std::time::SystemTime::now(),
            system_status: SystemStatus {
                active_sentinels: 49,
                total_violations: 10,
                resolved_violations: 8,
                system_health: 0.95,
                uptime: Duration::from_secs(3600),
                last_consensus: std::time::SystemTime::now(),
                hyperbolic_stability: 0.92,
            },
            sentinel_metrics: HashMap::new(),
            topology_metrics: TopologyMetrics {
                sentinel_count: 49,
                stability: 0.92,
                communication_energy: 15.5,
                coverage_area: 23.7,
                centroid: HyperbolicCoordinates {
                    x: 0.1,
                    y: 0.2,
                    radius: 0.15,
                },
                curvature: -1.5,
            },
            consensus_metrics: ConsensusMetrics::default(),
            recent_violations: Vec::new(),
            recent_events: Vec::new(),
            hyperbolic_visualization: HyperbolicVisualization {
                sentinel_positions: HashMap::new(),
                communication_edges: Vec::new(),
                centroid: HyperbolicCoordinates {
                    x: 0.1,
                    y: 0.2,
                    radius: 0.15,
                },
                stability_score: 0.92,
                coverage_area: 23.7,
            },
            performance_stats: PerformanceStats {
                total_violations_detected: 50,
                total_violations_resolved: 45,
                average_resolution_time_ms: 150.0,
                system_uptime: Duration::from_secs(3600),
                memory_usage_mb: 256.0,
                cpu_usage_percent: 25.0,
                network_throughput: 1024.0,
            },
        };

        dashboard.update_data(test_data.clone()).await;

        let current_data = dashboard.state.current_data.read().await;
        assert_eq!(current_data.system_status.active_sentinels, 49);
        assert_eq!(current_data.topology_metrics.sentinel_count, 49);
    }
}
