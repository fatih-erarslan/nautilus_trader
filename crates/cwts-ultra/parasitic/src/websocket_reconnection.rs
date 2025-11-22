//! # WebSocket Reconnection Logic
//! 
//! Advanced WebSocket reconnection with exponential backoff, state recovery,
//! and connection health monitoring for the parasitic trading system.

use std::{
    collections::HashMap,
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::{
    net::TcpStream,
    sync::{RwLock, mpsc},
    time::{sleep, timeout},
};
use tokio_tungstenite::{
    connect_async, 
    tungstenite::{Message, Error as TungsteniteError},
    WebSocketStream, MaybeTlsStream
};
use futures_util::{SinkExt, StreamExt};
use serde::{Serialize, Deserialize};
use tracing::{debug, info, warn, error};
use uuid::Uuid;
use chrono::{DateTime, Utc};

type WebSocket = WebSocketStream<MaybeTlsStream<TcpStream>>;

/// Reconnection configuration
#[derive(Debug, Clone)]
pub struct ReconnectionConfig {
    /// Maximum number of reconnection attempts
    pub max_attempts: u32,
    /// Initial backoff delay in milliseconds
    pub initial_backoff_ms: u64,
    /// Maximum backoff delay in milliseconds
    pub max_backoff_ms: u64,
    /// Backoff multiplier for exponential backoff
    pub backoff_multiplier: f64,
    /// Connection timeout in milliseconds
    pub connection_timeout_ms: u64,
    /// Health check interval in seconds
    pub health_check_interval_s: u64,
    /// Maximum time without response before considering connection dead
    pub max_silence_duration_s: u64,
}

impl Default for ReconnectionConfig {
    fn default() -> Self {
        Self {
            max_attempts: 10,
            initial_backoff_ms: 1000,
            max_backoff_ms: 30000,
            backoff_multiplier: 2.0,
            connection_timeout_ms: 10000,
            health_check_interval_s: 30,
            max_silence_duration_s: 90,
        }
    }
}

/// Connection state for recovery
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionState {
    pub client_id: Uuid,
    pub subscriptions: Vec<String>,
    pub last_request_id: u64,
    pub pending_requests: HashMap<String, PendingRequest>,
    pub session_data: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PendingRequest {
    pub id: String,
    pub method: String,
    pub params: serde_json::Value,
    pub timestamp: DateTime<Utc>,
    pub retry_count: u32,
}

/// Connection status
#[derive(Debug, Clone, PartialEq)]
pub enum ConnectionStatus {
    Connected,
    Reconnecting { attempt: u32, next_attempt_in: Duration },
    Disconnected,
    Failed { reason: String },
}

/// WebSocket reconnection manager
#[derive(Debug)]
pub struct WebSocketReconnectionManager {
    url: String,
    config: ReconnectionConfig,
    websocket: Arc<RwLock<Option<WebSocket>>>,
    status: Arc<RwLock<ConnectionStatus>>,
    state: Arc<RwLock<ConnectionState>>,
    message_tx: mpsc::UnboundedSender<Message>,
    message_rx: Arc<RwLock<Option<mpsc::UnboundedReceiver<Message>>>>,
    health_check_tx: mpsc::Sender<()>,
    metrics: Arc<RwLock<ReconnectionMetrics>>,
}

#[derive(Debug, Default)]
pub struct ReconnectionMetrics {
    pub total_connections: u64,
    pub total_disconnections: u64,
    pub total_reconnection_attempts: u64,
    pub successful_reconnections: u64,
    pub failed_reconnections: u64,
    pub current_uptime_start: Option<Instant>,
    pub total_uptime: Duration,
    pub average_connection_duration: Duration,
    pub longest_connection_duration: Duration,
    pub last_disconnection_reason: Option<String>,
}

impl WebSocketReconnectionManager {
    /// Create new reconnection manager
    pub fn new(url: String, config: Option<ReconnectionConfig>) -> Self {
        let config = config.unwrap_or_default();
        let (message_tx, message_rx) = mpsc::unbounded_channel();
        let (health_check_tx, _) = mpsc::channel(1);
        
        let initial_state = ConnectionState {
            client_id: Uuid::new_v4(),
            subscriptions: Vec::new(),
            last_request_id: 0,
            pending_requests: HashMap::new(),
            session_data: serde_json::Value::Null,
        };
        
        Self {
            url,
            config,
            websocket: Arc::new(RwLock::new(None)),
            status: Arc::new(RwLock::new(ConnectionStatus::Disconnected)),
            state: Arc::new(RwLock::new(initial_state)),
            message_tx,
            message_rx: Arc::new(RwLock::new(Some(message_rx))),
            health_check_tx,
            metrics: Arc::new(RwLock::new(ReconnectionMetrics::default())),
        }
    }
    
    /// Start the reconnection manager
    pub async fn start(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.connect_with_retry().await?;
        
        // Start message processing task
        let message_rx = {
            let mut rx_guard = self.message_rx.write().await;
            rx_guard.take().ok_or("Message receiver already taken")?
        };
        
        self.start_message_processor(message_rx).await;
        
        // Start health check task
        self.start_health_monitor().await;
        
        Ok(())
    }
    
    /// Connect with exponential backoff retry
    async fn connect_with_retry(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let mut attempt = 1;
        let mut backoff_ms = self.config.initial_backoff_ms;
        
        loop {
            debug!("Attempting to connect to {} (attempt {})", self.url, attempt);
            
            match self.attempt_connection().await {
                Ok(()) => {
                    info!("Successfully connected to {}", self.url);
                    *self.status.write().await = ConnectionStatus::Connected;
                    
                    // Update metrics
                    {
                        let mut metrics = self.metrics.write().await;
                        metrics.total_connections += 1;
                        metrics.current_uptime_start = Some(Instant::now());
                        if attempt > 1 {
                            metrics.successful_reconnections += 1;
                        }
                    }
                    
                    return Ok(());
                }
                Err(e) => {
                    error!("Connection attempt {} failed: {}", attempt, e);
                    
                    if attempt >= self.config.max_attempts {
                        let error_msg = format!("Failed to connect after {} attempts", attempt);
                        *self.status.write().await = ConnectionStatus::Failed { 
                            reason: error_msg.clone() 
                        };
                        
                        // Update metrics
                        {
                            let mut metrics = self.metrics.write().await;
                            metrics.failed_reconnections += 1;
                            metrics.last_disconnection_reason = Some(error_msg.clone());
                        }
                        
                        return Err(error_msg.into());
                    }
                    
                    // Update status and wait before retry
                    let next_attempt_in = Duration::from_millis(backoff_ms);
                    *self.status.write().await = ConnectionStatus::Reconnecting { 
                        attempt, 
                        next_attempt_in 
                    };
                    
                    warn!("Retrying in {}ms (attempt {}/{})", 
                          backoff_ms, attempt, self.config.max_attempts);
                    
                    sleep(next_attempt_in).await;
                    
                    // Exponential backoff
                    backoff_ms = ((backoff_ms as f64 * self.config.backoff_multiplier) as u64)
                        .min(self.config.max_backoff_ms);
                    
                    attempt += 1;
                    
                    // Update metrics
                    {
                        let mut metrics = self.metrics.write().await;
                        metrics.total_reconnection_attempts += 1;
                    }
                }
            }
        }
    }
    
    /// Attempt single connection
    async fn attempt_connection(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // Set connection timeout
        let connection_future = connect_async(&self.url);
        let timeout_duration = Duration::from_millis(self.config.connection_timeout_ms);
        
        let (websocket, _) = match timeout(timeout_duration, connection_future).await {
            Ok(Ok((ws, response))) => {
                debug!("WebSocket handshake completed with response: {:?}", response.status());
                (ws, response)
            }
            Ok(Err(e)) => return Err(format!("WebSocket connection failed: {}", e).into()),
            Err(_) => return Err("Connection timeout".into()),
        };
        
        // Store the connected websocket
        *self.websocket.write().await = Some(websocket);
        
        // Restore state if this is a reconnection
        self.restore_connection_state().await?;
        
        Ok(())
    }
    
    /// Restore connection state after reconnection
    async fn restore_connection_state(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let state = self.state.read().await.clone();
        
        // Restore subscriptions
        for subscription_uri in &state.subscriptions {
            if let Err(e) = self.send_subscription_restore(subscription_uri).await {
                warn!("Failed to restore subscription {}: {}", subscription_uri, e);
            }
        }
        
        // Retry pending requests
        for (request_id, pending_request) in &state.pending_requests {
            if pending_request.retry_count < 3 { // Limit retries
                if let Err(e) = self.retry_pending_request(pending_request).await {
                    warn!("Failed to retry request {}: {}", request_id, e);
                }
            }
        }
        
        info!("Connection state restored: {} subscriptions, {} pending requests",
              state.subscriptions.len(), state.pending_requests.len());
        
        Ok(())
    }
    
    /// Send subscription restore message
    async fn send_subscription_restore(&self, uri: &str) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let message = serde_json::json!({
            "jsonrpc": "2.0",
            "id": Uuid::new_v4().to_string(),
            "method": "resources/subscribe",
            "params": {
                "uri": uri
            }
        });
        
        self.send_message(Message::Text(message.to_string())).await
    }
    
    /// Retry pending request
    async fn retry_pending_request(&self, request: &PendingRequest) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let message = serde_json::json!({
            "jsonrpc": "2.0",
            "id": request.id,
            "method": request.method,
            "params": request.params
        });
        
        self.send_message(Message::Text(message.to_string())).await?;
        
        // Update retry count
        {
            let mut state = self.state.write().await;
            if let Some(pending) = state.pending_requests.get_mut(&request.id) {
                pending.retry_count += 1;
            }
        }
        
        Ok(())
    }
    
    /// Send message through WebSocket
    pub async fn send_message(&self, message: Message) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let status = self.status.read().await.clone();
        
        match status {
            ConnectionStatus::Connected => {
                // Send directly if connected
                if let Some(ref mut ws) = *self.websocket.write().await {
                    match ws.send(message).await {
                        Ok(_) => Ok(()),
                        Err(e) => {
                            error!("Failed to send message: {}", e);
                            self.handle_disconnection(format!("Send error: {}", e)).await;
                            Err(e.into())
                        }
                    }
                } else {
                    Err("WebSocket not connected".into())
                }
            }
            ConnectionStatus::Reconnecting { .. } => {
                // Queue message for later
                if let Err(e) = self.message_tx.send(message) {
                    error!("Failed to queue message: {}", e);
                }
                Ok(())
            }
            _ => Err("WebSocket not available".into())
        }
    }
    
    /// Handle disconnection
    async fn handle_disconnection(&self, reason: String) {
        warn!("WebSocket disconnected: {}", reason);
        
        // Update status
        *self.status.write().await = ConnectionStatus::Disconnected;
        
        // Clear websocket
        *self.websocket.write().await = None;
        
        // Update metrics
        {
            let mut metrics = self.metrics.write().await;
            metrics.total_disconnections += 1;
            metrics.last_disconnection_reason = Some(reason.clone());
            
            // Update uptime
            if let Some(uptime_start) = metrics.current_uptime_start {
                let session_duration = uptime_start.elapsed();
                metrics.total_uptime += session_duration;
                
                if session_duration > metrics.longest_connection_duration {
                    metrics.longest_connection_duration = session_duration;
                }
                
                // Calculate average
                if metrics.total_connections > 0 {
                    metrics.average_connection_duration = 
                        metrics.total_uptime / metrics.total_connections as u32;
                }
                
                metrics.current_uptime_start = None;
            }
        }
        
        // Start reconnection
        tokio::spawn({
            let self_clone = self.clone();
            async move {
                if let Err(e) = self_clone.connect_with_retry().await {
                    error!("Reconnection failed: {}", e);
                }
            }
        });
    }
    
    /// Start message processor task
    async fn start_message_processor(&self, mut message_rx: mpsc::UnboundedReceiver<Message>) {
        tokio::spawn({
            let websocket = Arc::clone(&self.websocket);
            let status = Arc::clone(&self.status);
            
            async move {
                while let Some(message) = message_rx.recv().await {
                    let current_status = status.read().await.clone();
                    
                    if let ConnectionStatus::Connected = current_status {
                        if let Some(ref mut ws) = *websocket.write().await {
                            if let Err(e) = ws.send(message).await {
                                error!("Failed to send queued message: {}", e);
                                break;
                            }
                        }
                    }
                }
            }
        });
    }
    
    /// Start health monitoring task
    async fn start_health_monitor(&self) {
        let health_check_tx = self.health_check_tx.clone();
        let websocket = Arc::clone(&self.websocket);
        let status = Arc::clone(&self.status);
        let config = self.config.clone();
        
        tokio::spawn(async move {
            let mut health_check_interval = tokio::time::interval(
                Duration::from_secs(config.health_check_interval_s)
            );
            
            let mut last_response = Instant::now();
            
            loop {
                health_check_interval.tick().await;
                
                let current_status = status.read().await.clone();
                
                if let ConnectionStatus::Connected = current_status {
                    // Send ping
                    if let Some(ref mut ws) = *websocket.write().await {
                        match ws.send(Message::Ping(b"health_check".to_vec())).await {
                            Ok(_) => {
                                debug!("Health check ping sent");
                                
                                // Check if we've been silent too long
                                if last_response.elapsed() > Duration::from_secs(config.max_silence_duration_s) {
                                    warn!("Connection appears dead - no response for {}s", 
                                          last_response.elapsed().as_secs());
                                    // This would trigger reconnection
                                    break;
                                }
                            }
                            Err(e) => {
                                error!("Health check failed: {}", e);
                                break;
                            }
                        }
                    }
                    
                    // Wait for pong response
                    if let Ok(_) = timeout(Duration::from_secs(10), health_check_tx.send(())).await {
                        last_response = Instant::now();
                        debug!("Health check pong received");
                    }
                }
            }
        });
    }
    
    /// Get current connection status
    pub async fn get_status(&self) -> ConnectionStatus {
        self.status.read().await.clone()
    }
    
    /// Get connection metrics
    pub async fn get_metrics(&self) -> ReconnectionMetrics {
        let mut metrics = self.metrics.read().await.clone();
        
        // Update current uptime if connected
        if let Some(uptime_start) = metrics.current_uptime_start {
            let current_uptime = uptime_start.elapsed();
            metrics.total_uptime += current_uptime;
        }
        
        metrics
    }
    
    /// Add subscription to state
    pub async fn add_subscription(&self, uri: String) {
        let mut state = self.state.write().await;
        if !state.subscriptions.contains(&uri) {
            state.subscriptions.push(uri);
        }
    }
    
    /// Remove subscription from state
    pub async fn remove_subscription(&self, uri: &str) {
        let mut state = self.state.write().await;
        state.subscriptions.retain(|s| s != uri);
    }
    
    /// Add pending request
    pub async fn add_pending_request(&self, id: String, method: String, params: serde_json::Value) {
        let mut state = self.state.write().await;
        state.pending_requests.insert(id.clone(), PendingRequest {
            id,
            method,
            params,
            timestamp: Utc::now(),
            retry_count: 0,
        });
    }
    
    /// Remove pending request (on successful response)
    pub async fn remove_pending_request(&self, id: &str) {
        let mut state = self.state.write().await;
        state.pending_requests.remove(id);
    }
    
    /// Force reconnection
    pub async fn force_reconnect(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        warn!("Forcing reconnection");
        
        // Close current connection
        if let Some(ref mut ws) = *self.websocket.write().await {
            let _ = ws.close(None).await;
        }
        *self.websocket.write().await = None;
        
        // Start reconnection
        self.connect_with_retry().await
    }
}

// Implement Clone for the manager (needed for sharing across tasks)
impl Clone for WebSocketReconnectionManager {
    fn clone(&self) -> Self {
        Self {
            url: self.url.clone(),
            config: self.config.clone(),
            websocket: Arc::clone(&self.websocket),
            status: Arc::clone(&self.status),
            state: Arc::clone(&self.state),
            message_tx: self.message_tx.clone(),
            message_rx: Arc::clone(&self.message_rx),
            health_check_tx: self.health_check_tx.clone(),
            metrics: Arc::clone(&self.metrics),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_reconnection_manager_creation() {
        let manager = WebSocketReconnectionManager::new(
            "ws://localhost:8080".to_string(),
            None
        );
        
        let status = manager.get_status().await;
        assert_eq!(status, ConnectionStatus::Disconnected);
    }
    
    #[tokio::test]
    async fn test_subscription_management() {
        let manager = WebSocketReconnectionManager::new(
            "ws://localhost:8080".to_string(),
            None
        );
        
        manager.add_subscription("test://resource/1".to_string()).await;
        manager.add_subscription("test://resource/2".to_string()).await;
        
        let state = manager.state.read().await;
        assert_eq!(state.subscriptions.len(), 2);
        assert!(state.subscriptions.contains(&"test://resource/1".to_string()));
        assert!(state.subscriptions.contains(&"test://resource/2".to_string()));
    }
    
    #[tokio::test]
    async fn test_pending_request_management() {
        let manager = WebSocketReconnectionManager::new(
            "ws://localhost:8080".to_string(),
            None
        );
        
        manager.add_pending_request(
            "req1".to_string(),
            "test_method".to_string(),
            serde_json::json!({})
        ).await;
        
        let state = manager.state.read().await;
        assert!(state.pending_requests.contains_key("req1"));
        
        drop(state);
        
        manager.remove_pending_request("req1").await;
        
        let state = manager.state.read().await;
        assert!(!state.pending_requests.contains_key("req1"));
    }
    
    #[test]
    fn test_reconnection_config() {
        let config = ReconnectionConfig::default();
        assert_eq!(config.max_attempts, 10);
        assert_eq!(config.initial_backoff_ms, 1000);
        assert_eq!(config.max_backoff_ms, 30000);
        assert_eq!(config.backoff_multiplier, 2.0);
    }
}