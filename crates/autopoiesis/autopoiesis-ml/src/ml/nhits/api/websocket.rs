use anyhow::Result;
use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        Query, State,
    },
    response::Response,
};
use futures_util::{sink::SinkExt, stream::StreamExt};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::{
    sync::{mpsc, RwLock},
    time::interval,
};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use crate::ml::nhits::api::{
    models::*,
    server::AppState,
};

/// WebSocket connection parameters
#[derive(Debug, Deserialize)]
pub struct WsParams {
    /// Authentication token
    pub token: Option<String>,
    /// Client identifier
    pub client_id: Option<String>,
    /// Subscribe to specific model updates
    pub model_id: Option<String>,
    /// Subscribe to forecast job updates
    pub job_id: Option<String>,
}

/// WebSocket message types
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(tag = "type")]
pub enum WsMessage {
    /// Connection established
    Connected {
        connection_id: Uuid,
        timestamp: chrono::DateTime<chrono::Utc>,
    },
    /// Heartbeat/ping message
    Ping {
        timestamp: chrono::DateTime<chrono::Utc>,
    },
    /// Pong response
    Pong {
        timestamp: chrono::DateTime<chrono::Utc>,
    },
    /// Real-time forecast update
    ForecastUpdate {
        job_id: Uuid,
        model_id: String,
        progress: f32,
        partial_results: Option<Vec<f64>>,
        timestamp: chrono::DateTime<chrono::Utc>,
    },
    /// Model training progress
    TrainingUpdate {
        model_id: String,
        epoch: u32,
        loss: f64,
        metrics: HashMap<String, f64>,
        timestamp: chrono::DateTime<chrono::Utc>,
    },
    /// System status update
    SystemUpdate {
        active_models: usize,
        active_jobs: usize,
        memory_usage: f64,
        cpu_usage: f64,
        timestamp: chrono::DateTime<chrono::Utc>,
    },
    /// Error message
    Error {
        code: String,
        message: String,
        timestamp: chrono::DateTime<chrono::Utc>,
    },
    /// Subscription confirmation
    Subscribed {
        subscription: String,
        timestamp: chrono::DateTime<chrono::Utc>,
    },
    /// Unsubscription confirmation
    Unsubscribed {
        subscription: String,
        timestamp: chrono::DateTime<chrono::Utc>,
    },
}

/// WebSocket connection state
#[derive(Debug)]
pub struct WsConnection {
    pub id: Uuid,
    pub client_id: Option<String>,
    pub connected_at: Instant,
    pub last_ping: Instant,
    pub subscriptions: Vec<String>,
    pub sender: mpsc::UnboundedSender<String>,
}

impl WsConnection {
    pub fn new(sender: mpsc::UnboundedSender<String>) -> Self {
        let now = Instant::now();
        Self {
            id: Uuid::new_v4(),
            client_id: None,
            connected_at: now,
            last_ping: now,
            subscriptions: Vec::new(),
            sender,
        }
    }

    pub fn is_alive(&self) -> bool {
        self.last_ping.elapsed() < Duration::from_secs(60)
    }

    pub fn update_ping(&mut self) {
        self.last_ping = Instant::now();
    }

    pub fn add_subscription(&mut self, subscription: String) {
        if !self.subscriptions.contains(&subscription) {
            self.subscriptions.push(subscription);
        }
    }

    pub fn remove_subscription(&mut self, subscription: &str) {
        self.subscriptions.retain(|s| s != subscription);
    }
}

/// WebSocket handler for real-time updates
pub async fn websocket_handler(
    ws: WebSocketUpgrade,
    Query(params): Query<WsParams>,
    State(state): State<AppState>,
) -> Response {
    ws.on_upgrade(move |socket| handle_websocket(socket, params, state))
}

/// Handle WebSocket connection
async fn handle_websocket(socket: WebSocket, params: WsParams, state: AppState) {
    let (mut sender, mut receiver) = socket.split();
    let (tx, mut rx) = mpsc::unbounded_channel::<String>();
    
    let mut connection = WsConnection::new(tx);
    connection.client_id = params.client_id.clone();
    
    let connection_id = connection.id;
    info!("WebSocket connection established: {}", connection_id);

    // Add connection to state
    {
        let mut connections = state.websocket_connections.write().await;
        connections.insert(connection_id, connection.sender.clone());
    }

    // Send connection confirmation
    let connect_msg = WsMessage::Connected {
        connection_id,
        timestamp: chrono::Utc::now(),
    };
    
    if let Err(e) = connection.sender.send(serde_json::to_string(&connect_msg).unwrap()) {
        error!("Failed to send connection message: {}", e);
        return;
    }

    // Set up initial subscriptions based on params
    let mut subscriptions = Vec::new();
    if let Some(model_id) = params.model_id {
        subscriptions.push(format!("model:{}", model_id));
    }
    if let Some(job_id) = params.job_id {
        subscriptions.push(format!("job:{}", job_id));
    }

    // Start heartbeat task
    let heartbeat_sender = connection.sender.clone();
    let heartbeat_task = tokio::spawn(async move {
        let mut interval = interval(Duration::from_secs(30));
        loop {
            interval.tick().await;
            let ping_msg = WsMessage::Ping {
                timestamp: chrono::Utc::now(),
            };
            if heartbeat_sender.send(serde_json::to_string(&ping_msg).unwrap()).is_err() {
                break;
            }
        }
    });

    // Handle incoming messages and outgoing messages concurrently
    let receive_task = tokio::spawn(async move {
        while let Some(msg) = receiver.next().await {
            match msg {
                Ok(Message::Text(text)) => {
                    if let Err(e) = handle_ws_message(text, &mut connection, &state).await {
                        error!("Error handling WebSocket message: {}", e);
                    }
                }
                Ok(Message::Binary(_)) => {
                    warn!("Received binary message, ignoring");
                }
                Ok(Message::Ping(payload)) => {
                    connection.update_ping();
                    if let Err(e) = connection.sender.send(
                        serde_json::to_string(&WsMessage::Pong {
                            timestamp: chrono::Utc::now(),
                        }).unwrap()
                    ) {
                        error!("Failed to send pong: {}", e);
                        break;
                    }
                }
                Ok(Message::Pong(_)) => {
                    connection.update_ping();
                }
                Ok(Message::Close(_)) => {
                    info!("WebSocket connection closed by client: {}", connection_id);
                    break;
                }
                Err(e) => {
                    error!("WebSocket error: {}", e);
                    break;
                }
            }
        }
        connection_id
    });

    let send_task = tokio::spawn(async move {
        while let Some(msg) = rx.recv().await {
            if sender.send(Message::Text(msg)).await.is_err() {
                break;
            }
        }
    });

    // Wait for either task to complete
    tokio::select! {
        connection_id = receive_task => {
            info!("WebSocket receive task completed: {:?}", connection_id);
        },
        _ = send_task => {
            info!("WebSocket send task completed");
        },
    }

    // Cleanup
    heartbeat_task.abort();
    {
        let mut connections = state.websocket_connections.write().await;
        connections.remove(&connection_id);
    }
    info!("WebSocket connection cleaned up: {}", connection_id);
}

/// Handle incoming WebSocket messages
async fn handle_ws_message(
    text: String,
    connection: &mut WsConnection,
    state: &AppState,
) -> Result<()> {
    debug!("Received WebSocket message: {}", text);
    
    let message: WsMessage = serde_json::from_str(&text)?;
    
    match message {
        WsMessage::Ping { .. } => {
            connection.update_ping();
            let pong_msg = WsMessage::Pong {
                timestamp: chrono::Utc::now(),
            };
            connection.sender.send(serde_json::to_string(&pong_msg)?)?;
        }
        WsMessage::Pong { .. } => {
            connection.update_ping();
        }
        _ => {
            // Handle subscription requests, etc.
            warn!("Unhandled WebSocket message type");
        }
    }
    
    Ok(())
}

/// Broadcast message to all connected WebSocket clients
pub async fn broadcast_message(
    connections: &Arc<RwLock<HashMap<Uuid, mpsc::UnboundedSender<String>>>>,
    message: &WsMessage,
) -> Result<()> {
    let message_str = serde_json::to_string(message)?;
    let connections = connections.read().await;
    
    for (connection_id, sender) in connections.iter() {
        if let Err(e) = sender.send(message_str.clone()) {
            error!("Failed to send broadcast to connection {}: {}", connection_id, e);
        }
    }
    
    Ok(())
}

/// Broadcast message to specific subscription
pub async fn broadcast_to_subscription(
    connections: &Arc<RwLock<HashMap<Uuid, mpsc::UnboundedSender<String>>>>,
    subscription: &str,
    message: &WsMessage,
) -> Result<()> {
    let message_str = serde_json::to_string(message)?;
    let connections = connections.read().await;
    
    // Note: In a real implementation, you'd maintain subscription mappings
    // For now, we broadcast to all connections
    for (connection_id, sender) in connections.iter() {
        if let Err(e) = sender.send(message_str.clone()) {
            error!("Failed to send subscription message to connection {}: {}", connection_id, e);
        }
    }
    
    Ok(())
}

/// Send forecast update to WebSocket clients
pub async fn send_forecast_update(
    connections: &Arc<RwLock<HashMap<Uuid, mpsc::UnboundedSender<String>>>>,
    job_id: Uuid,
    model_id: String,
    progress: f32,
    partial_results: Option<Vec<f64>>,
) -> Result<()> {
    let message = WsMessage::ForecastUpdate {
        job_id,
        model_id,
        progress,
        partial_results,
        timestamp: chrono::Utc::now(),
    };
    
    broadcast_to_subscription(connections, &format!("job:{}", job_id), &message).await
}

/// Send training update to WebSocket clients
pub async fn send_training_update(
    connections: &Arc<RwLock<HashMap<Uuid, mpsc::UnboundedSender<String>>>>,
    model_id: String,
    epoch: u32,
    loss: f64,
    metrics: HashMap<String, f64>,
) -> Result<()> {
    let message = WsMessage::TrainingUpdate {
        model_id: model_id.clone(),
        epoch,
        loss,
        metrics,
        timestamp: chrono::Utc::now(),
    };
    
    broadcast_to_subscription(connections, &format!("model:{}", model_id), &message).await
}

/// Send system update to WebSocket clients
pub async fn send_system_update(
    connections: &Arc<RwLock<HashMap<Uuid, mpsc::UnboundedSender<String>>>>,
    active_models: usize,
    active_jobs: usize,
    memory_usage: f64,
    cpu_usage: f64,
) -> Result<()> {
    let message = WsMessage::SystemUpdate {
        active_models,
        active_jobs,
        memory_usage,
        cpu_usage,
        timestamp: chrono::Utc::now(),
    };
    
    broadcast_message(connections, &message).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::mpsc;

    #[test]
    fn test_ws_connection_creation() {
        let (tx, _rx) = mpsc::unbounded_channel();
        let connection = WsConnection::new(tx);
        
        assert!(connection.is_alive());
        assert!(connection.subscriptions.is_empty());
    }

    #[test]
    fn test_subscription_management() {
        let (tx, _rx) = mpsc::unbounded_channel();
        let mut connection = WsConnection::new(tx);
        
        connection.add_subscription("model:test".to_string());
        assert_eq!(connection.subscriptions.len(), 1);
        
        connection.add_subscription("model:test".to_string()); // Duplicate
        assert_eq!(connection.subscriptions.len(), 1);
        
        connection.remove_subscription("model:test");
        assert!(connection.subscriptions.is_empty());
    }

    #[tokio::test]
    async fn test_message_serialization() {
        let message = WsMessage::Connected {
            connection_id: Uuid::new_v4(),
            timestamp: chrono::Utc::now(),
        };
        
        let serialized = serde_json::to_string(&message).unwrap();
        let deserialized: WsMessage = serde_json::from_str(&serialized).unwrap();
        
        match deserialized {
            WsMessage::Connected { .. } => assert!(true),
            _ => assert!(false, "Message type mismatch"),
        }
    }
}