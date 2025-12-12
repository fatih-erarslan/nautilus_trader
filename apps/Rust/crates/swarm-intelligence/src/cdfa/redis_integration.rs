//! Redis Integration for Distributed CDFA Communication
//! 
//! This module implements Redis-based distributed communication for CDFA,
//! enabling multi-node coordination, shared memory, and real-time signal
//! distribution across trading infrastructure.

use std::collections::HashMap;
use std::sync::Arc;
use async_trait::async_trait;
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use redis::{Client, Connection, RedisError, Commands, AsyncCommands};
use redis::aio::MultiplexedConnection;

use crate::errors::SwarmError;
use super::ml_integration::{SignalFeatures, ProcessedSignal, MLExperience};

/// Redis integration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RedisConfig {
    /// Redis connection URL
    pub redis_url: String,
    
    /// Connection pool size
    pub pool_size: usize,
    
    /// Key prefix for CDFA data
    pub key_prefix: String,
    
    /// TTL for cached data (seconds)
    pub default_ttl: u64,
    
    /// Signal distribution channels
    pub signal_channels: Vec<String>,
    
    /// Performance metrics channel
    pub metrics_channel: String,
    
    /// Coordination channel
    pub coordination_channel: String,
    
    /// Maximum message size (bytes)
    pub max_message_size: usize,
    
    /// Retry configuration
    pub max_retries: usize,
    pub retry_delay_ms: u64,
    
    /// Batch processing configuration
    pub batch_size: usize,
    pub batch_timeout_ms: u64,
}

impl Default for RedisConfig {
    fn default() -> Self {
        Self {
            redis_url: "redis://127.0.0.1:6379".to_string(),
            pool_size: 10,
            key_prefix: "cdfa".to_string(),
            default_ttl: 3600, // 1 hour
            signal_channels: vec![
                "cdfa:signals:raw".to_string(),
                "cdfa:signals:processed".to_string(),
                "cdfa:signals:fused".to_string(),
            ],
            metrics_channel: "cdfa:metrics".to_string(),
            coordination_channel: "cdfa:coordination".to_string(),
            max_message_size: 1_048_576, // 1MB
            max_retries: 3,
            retry_delay_ms: 100,
            batch_size: 100,
            batch_timeout_ms: 1000,
        }
    }
}

/// Distributed signal message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedSignal {
    /// Signal identifier
    pub signal_id: String,
    
    /// Source node identifier
    pub source_node: String,
    
    /// Signal features or processed data
    pub signal_data: SignalData,
    
    /// Processing metadata
    pub metadata: HashMap<String, serde_json::Value>,
    
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    
    /// Message priority
    pub priority: MessagePriority,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SignalData {
    RawFeatures(SignalFeatures),
    ProcessedSignal(ProcessedSignal),
    FusedResult(HashMap<String, f64>),
    MLExperience(MLExperience),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MessagePriority {
    Low,
    Medium,
    High,
    Critical,
}

/// Coordination message for distributed CDFA
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordinationMessage {
    /// Message type
    pub message_type: CoordinationType,
    
    /// Source node
    pub source_node: String,
    
    /// Target nodes (empty for broadcast)
    pub target_nodes: Vec<String>,
    
    /// Message payload
    pub payload: serde_json::Value,
    
    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CoordinationType {
    /// Request algorithm selection
    AlgorithmSelection,
    
    /// Share performance metrics
    PerformanceUpdate,
    
    /// Coordinate weight optimization
    WeightOptimization,
    
    /// Synchronize model states
    ModelSynchronization,
    
    /// Health check
    HealthCheck,
    
    /// Configuration update
    ConfigUpdate,
    
    /// Emergency stop
    EmergencyStop,
}

/// Redis client wrapper with connection pooling
pub struct RedisClient {
    /// Redis client
    client: Client,
    
    /// Multiplexed connections
    connections: Arc<RwLock<Vec<MultiplexedConnection>>>,
    
    /// Configuration
    config: RedisConfig,
    
    /// Node identifier
    node_id: String,
    
    /// Connection metrics
    metrics: Arc<RwLock<RedisMetrics>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RedisMetrics {
    pub total_messages_sent: u64,
    pub total_messages_received: u64,
    pub total_bytes_sent: u64,
    pub total_bytes_received: u64,
    pub connection_errors: u64,
    pub retry_attempts: u64,
    pub average_latency_ms: f64,
    pub last_update: DateTime<Utc>,
}

impl Default for RedisMetrics {
    fn default() -> Self {
        Self {
            total_messages_sent: 0,
            total_messages_received: 0,
            total_bytes_sent: 0,
            total_bytes_received: 0,
            connection_errors: 0,
            retry_attempts: 0,
            average_latency_ms: 0.0,
            last_update: Utc::now(),
        }
    }
}

impl RedisClient {
    /// Create new Redis client with configuration
    pub async fn new(config: RedisConfig, node_id: String) -> Result<Self, SwarmError> {
        let client = Client::open(config.redis_url.as_str())
            .map_err(|e| SwarmError::ConnectionError(format!("Redis connection failed: {}", e)))?;
        
        // Test connection
        let mut con = client.get_async_connection().await
            .map_err(|e| SwarmError::ConnectionError(format!("Redis async connection failed: {}", e)))?;
        
        // Initialize connection pool
        let mut connections = Vec::new();
        for _ in 0..config.pool_size {
            let conn = client.get_multiplexed_async_connection().await
                .map_err(|e| SwarmError::ConnectionError(format!("Redis multiplexed connection failed: {}", e)))?;
            connections.push(conn);
        }
        
        Ok(Self {
            client,
            connections: Arc::new(RwLock::new(connections)),
            config,
            node_id,
            metrics: Arc::new(RwLock::new(RedisMetrics::default())),
        })
    }
    
    /// Get available connection from pool
    async fn get_connection(&self) -> Result<MultiplexedConnection, SwarmError> {
        let mut connections = self.connections.write().await;
        
        if let Some(conn) = connections.pop() {
            Ok(conn)
        } else {
            // Create new connection if pool is empty
            self.client.get_multiplexed_async_connection().await
                .map_err(|e| SwarmError::ConnectionError(format!("Failed to create Redis connection: {}", e)))
        }
    }
    
    /// Return connection to pool
    async fn return_connection(&self, conn: MultiplexedConnection) {
        let mut connections = self.connections.write().await;
        if connections.len() < self.config.pool_size {
            connections.push(conn);
        }
    }
    
    /// Execute Redis operation with retry logic
    async fn execute_with_retry<F, T>(&self, operation: F) -> Result<T, SwarmError>
    where
        F: Fn() -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<T, RedisError>> + Send>>,
    {
        let mut retries = 0;
        
        loop {
            match operation().await {
                Ok(result) => return Ok(result),
                Err(e) => {
                    retries += 1;
                    
                    // Update error metrics
                    {
                        let mut metrics = self.metrics.write().await;
                        metrics.connection_errors += 1;
                        metrics.retry_attempts += 1;
                    }
                    
                    if retries >= self.config.max_retries {
                        return Err(SwarmError::ConnectionError(format!("Redis operation failed after {} retries: {}", retries, e)));
                    }
                    
                    // Wait before retry
                    tokio::time::sleep(tokio::time::Duration::from_millis(self.config.retry_delay_ms)).await;
                }
            }
        }
    }
    
    /// Publish signal to Redis channel
    pub async fn publish_signal(&self, signal: DistributedSignal) -> Result<(), SwarmError> {
        let start_time = std::time::Instant::now();
        
        // Serialize signal
        let message = serde_json::to_string(&signal)
            .map_err(|e| SwarmError::SerializationError(format!("Signal serialization failed: {}", e)))?;
        
        // Check message size
        if message.len() > self.config.max_message_size {
            return Err(SwarmError::ParameterError(
                format!("Message size {} exceeds limit {}", message.len(), self.config.max_message_size)
            ));
        }
        
        // Determine channel based on signal type
        let channel = match &signal.signal_data {
            SignalData::RawFeatures(_) => &self.config.signal_channels[0],
            SignalData::ProcessedSignal(_) => &self.config.signal_channels[1],
            SignalData::FusedResult(_) => &self.config.signal_channels[2],
            SignalData::MLExperience(_) => &self.config.signal_channels[1],
        };
        
        // Publish with retry logic
        let mut conn = self.get_connection().await?;
        
        let result = self.execute_with_retry(|| {
            let message_clone = message.clone();
            let channel_clone = channel.clone();
            Box::pin(async move {
                conn.publish::<&str, &str, i32>(&channel_clone, &message_clone).await
            })
        }).await;
        
        self.return_connection(conn).await;
        
        // Update metrics
        {
            let mut metrics = self.metrics.write().await;
            metrics.total_messages_sent += 1;
            metrics.total_bytes_sent += message.len() as u64;
            
            let latency = start_time.elapsed().as_millis() as f64;
            let alpha = 0.1;
            metrics.average_latency_ms = alpha * latency + (1.0 - alpha) * metrics.average_latency_ms;
            metrics.last_update = Utc::now();
        }
        
        result.map(|_| ())
    }
    
    /// Subscribe to signal channels
    pub async fn subscribe_to_signals<F>(&self, mut callback: F) -> Result<(), SwarmError>
    where
        F: FnMut(DistributedSignal) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<(), SwarmError>> + Send>> + Send + 'static,
    {
        let mut conn = self.get_connection().await?;
        let mut pubsub = conn.as_pubsub();
        
        // Subscribe to all signal channels
        for channel in &self.config.signal_channels {
            pubsub.subscribe(channel).await
                .map_err(|e| SwarmError::ConnectionError(format!("Failed to subscribe to channel {}: {}", channel, e)))?;
        }
        
        // Listen for messages
        loop {
            let msg = pubsub.on_message().next().await;
            
            if let Some(msg) = msg {
                let payload: String = msg.get_payload()
                    .map_err(|e| SwarmError::DeserializationError(format!("Failed to get message payload: {}", e)))?;
                
                match serde_json::from_str::<DistributedSignal>(&payload) {
                    Ok(signal) => {
                        // Update receive metrics
                        {
                            let mut metrics = self.metrics.write().await;
                            metrics.total_messages_received += 1;
                            metrics.total_bytes_received += payload.len() as u64;
                        }
                        
                        // Call the callback
                        if let Err(e) = callback(signal).await {
                            eprintln!("Signal processing callback failed: {}", e);
                        }
                    },
                    Err(e) => {
                        eprintln!("Failed to deserialize signal: {}", e);
                    }
                }
            }
        }
    }
    
    /// Publish coordination message
    pub async fn publish_coordination(&self, message: CoordinationMessage) -> Result<(), SwarmError> {
        let serialized = serde_json::to_string(&message)
            .map_err(|e| SwarmError::SerializationError(format!("Coordination message serialization failed: {}", e)))?;
        
        let mut conn = self.get_connection().await?;
        
        let result = self.execute_with_retry(|| {
            let msg_clone = serialized.clone();
            let channel = self.config.coordination_channel.clone();
            Box::pin(async move {
                conn.publish::<&str, &str, i32>(&channel, &msg_clone).await
            })
        }).await;
        
        self.return_connection(conn).await;
        result.map(|_| ())
    }
    
    /// Subscribe to coordination messages
    pub async fn subscribe_to_coordination<F>(&self, mut callback: F) -> Result<(), SwarmError>
    where
        F: FnMut(CoordinationMessage) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<(), SwarmError>> + Send>> + Send + 'static,
    {
        let mut conn = self.get_connection().await?;
        let mut pubsub = conn.as_pubsub();
        
        pubsub.subscribe(&self.config.coordination_channel).await
            .map_err(|e| SwarmError::ConnectionError(format!("Failed to subscribe to coordination channel: {}", e)))?;
        
        loop {
            let msg = pubsub.on_message().next().await;
            
            if let Some(msg) = msg {
                let payload: String = msg.get_payload()
                    .map_err(|e| SwarmError::DeserializationError(format!("Failed to get coordination payload: {}", e)))?;
                
                match serde_json::from_str::<CoordinationMessage>(&payload) {
                    Ok(coord_msg) => {
                        // Check if message is for this node
                        if coord_msg.target_nodes.is_empty() || coord_msg.target_nodes.contains(&self.node_id) {
                            if let Err(e) = callback(coord_msg).await {
                                eprintln!("Coordination message callback failed: {}", e);
                            }
                        }
                    },
                    Err(e) => {
                        eprintln!("Failed to deserialize coordination message: {}", e);
                    }
                }
            }
        }
    }
    
    /// Store data in Redis with TTL
    pub async fn store_data(&self, key: &str, data: &serde_json::Value, ttl: Option<u64>) -> Result<(), SwarmError> {
        let full_key = format!("{}:{}", self.config.key_prefix, key);
        let serialized = serde_json::to_string(data)
            .map_err(|e| SwarmError::SerializationError(format!("Data serialization failed: {}", e)))?;
        
        let mut conn = self.get_connection().await?;
        let ttl_seconds = ttl.unwrap_or(self.config.default_ttl);
        
        let result = self.execute_with_retry(|| {
            let key_clone = full_key.clone();
            let data_clone = serialized.clone();
            Box::pin(async move {
                conn.set_ex::<&str, &str, ()>(&key_clone, &data_clone, ttl_seconds).await
            })
        }).await;
        
        self.return_connection(conn).await;
        result
    }
    
    /// Retrieve data from Redis
    pub async fn retrieve_data(&self, key: &str) -> Result<Option<serde_json::Value>, SwarmError> {
        let full_key = format!("{}:{}", self.config.key_prefix, key);
        let mut conn = self.get_connection().await?;
        
        let result: Option<String> = self.execute_with_retry(|| {
            let key_clone = full_key.clone();
            Box::pin(async move {
                conn.get::<&str, Option<String>>(&key_clone).await
            })
        }).await?;
        
        self.return_connection(conn).await;
        
        match result {
            Some(data) => {
                let value = serde_json::from_str(&data)
                    .map_err(|e| SwarmError::DeserializationError(format!("Data deserialization failed: {}", e)))?;
                Ok(Some(value))
            },
            None => Ok(None),
        }
    }
    
    /// Delete data from Redis
    pub async fn delete_data(&self, key: &str) -> Result<bool, SwarmError> {
        let full_key = format!("{}:{}", self.config.key_prefix, key);
        let mut conn = self.get_connection().await?;
        
        let result: i32 = self.execute_with_retry(|| {
            let key_clone = full_key.clone();
            Box::pin(async move {
                conn.del::<&str, i32>(&key_clone).await
            })
        }).await?;
        
        self.return_connection(conn).await;
        Ok(result > 0)
    }
    
    /// Get Redis metrics
    pub async fn get_metrics(&self) -> RedisMetrics {
        self.metrics.read().await.clone()
    }
    
    /// Health check for Redis connection
    pub async fn health_check(&self) -> Result<bool, SwarmError> {
        let mut conn = self.get_connection().await?;
        
        let result = self.execute_with_retry(|| {
            Box::pin(async move {
                redis::cmd("PING").query_async::<_, String>(&mut conn).await
            })
        }).await;
        
        self.return_connection(conn).await;
        
        match result {
            Ok(response) => Ok(response == "PONG"),
            Err(_) => Ok(false),
        }
    }
    
    /// Batch operations for improved performance
    pub async fn batch_store(&self, operations: Vec<(String, serde_json::Value, Option<u64>)>) -> Result<(), SwarmError> {
        if operations.is_empty() {
            return Ok(());
        }
        
        let mut conn = self.get_connection().await?;
        
        // Use pipeline for batch operations
        let mut pipe = redis::pipe();
        
        for (key, data, ttl) in operations {
            let full_key = format!("{}:{}", self.config.key_prefix, key);
            let serialized = serde_json::to_string(&data)
                .map_err(|e| SwarmError::SerializationError(format!("Batch data serialization failed: {}", e)))?;
            
            let ttl_seconds = ttl.unwrap_or(self.config.default_ttl);
            pipe.set_ex(&full_key, &serialized, ttl_seconds);
        }
        
        let result = self.execute_with_retry(|| {
            Box::pin(async move {
                pipe.query_async::<_, ()>(&mut conn).await
            })
        }).await;
        
        self.return_connection(conn).await;
        result
    }
}

/// Distributed CDFA coordinator using Redis
pub struct DistributedCDFACoordinator {
    /// Redis client
    redis_client: Arc<RedisClient>,
    
    /// Node identifier
    node_id: String,
    
    /// Active signals cache
    signal_cache: Arc<RwLock<HashMap<String, DistributedSignal>>>,
    
    /// Coordination state
    coordination_state: Arc<RwLock<HashMap<String, serde_json::Value>>>,
    
    /// Message handlers
    signal_handlers: Arc<RwLock<Vec<Box<dyn Fn(DistributedSignal) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<(), SwarmError>> + Send>> + Send + Sync>>>>,
}

impl DistributedCDFACoordinator {
    /// Create new distributed CDFA coordinator
    pub async fn new(config: RedisConfig, node_id: String) -> Result<Self, SwarmError> {
        let redis_client = Arc::new(RedisClient::new(config, node_id.clone()).await?);
        
        Ok(Self {
            redis_client,
            node_id,
            signal_cache: Arc::new(RwLock::new(HashMap::new())),
            coordination_state: Arc::new(RwLock::new(HashMap::new())),
            signal_handlers: Arc::new(RwLock::new(Vec::new())),
        })
    }
    
    /// Start distributed coordination
    pub async fn start_coordination(&self) -> Result<(), SwarmError> {
        // Start signal subscription
        let redis_clone = self.redis_client.clone();
        let cache_clone = self.signal_cache.clone();
        let handlers_clone = self.signal_handlers.clone();
        
        tokio::spawn(async move {
            let _ = redis_clone.subscribe_to_signals(move |signal| {
                let cache = cache_clone.clone();
                let handlers = handlers_clone.clone();
                Box::pin(async move {
                    // Update cache
                    {
                        let mut cache_guard = cache.write().await;
                        cache_guard.insert(signal.signal_id.clone(), signal.clone());
                    }
                    
                    // Call handlers
                    let handlers_guard = handlers.read().await;
                    for handler in handlers_guard.iter() {
                        if let Err(e) = handler(signal.clone()).await {
                            eprintln!("Signal handler failed: {}", e);
                        }
                    }
                    
                    Ok(())
                })
            }).await;
        });
        
        // Start coordination subscription
        let redis_clone2 = self.redis_client.clone();
        let state_clone = self.coordination_state.clone();
        let node_id_clone = self.node_id.clone();
        
        tokio::spawn(async move {
            let _ = redis_clone2.subscribe_to_coordination(move |message| {
                let state = state_clone.clone();
                let node_id = node_id_clone.clone();
                Box::pin(async move {
                    // Process coordination message
                    match message.message_type {
                        CoordinationType::HealthCheck => {
                            // Respond to health check
                            println!("Health check received by node: {}", node_id);
                        },
                        CoordinationType::ConfigUpdate => {
                            // Update configuration
                            let mut state_guard = state.write().await;
                            state_guard.insert("config_update".to_string(), message.payload);
                        },
                        _ => {
                            // Handle other coordination types
                            println!("Coordination message received: {:?}", message.message_type);
                        }
                    }
                    
                    Ok(())
                })
            }).await;
        });
        
        Ok(())
    }
    
    /// Add signal handler
    pub async fn add_signal_handler<F>(&self, handler: F)
    where
        F: Fn(DistributedSignal) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<(), SwarmError>> + Send>> + Send + Sync + 'static,
    {
        let mut handlers = self.signal_handlers.write().await;
        handlers.push(Box::new(handler));
    }
    
    /// Broadcast signal to all nodes
    pub async fn broadcast_signal(&self, signal_data: SignalData, priority: MessagePriority) -> Result<(), SwarmError> {
        let signal = DistributedSignal {
            signal_id: format!("{}_{}", self.node_id, Utc::now().timestamp_millis()),
            source_node: self.node_id.clone(),
            signal_data,
            metadata: HashMap::new(),
            timestamp: Utc::now(),
            priority,
        };
        
        self.redis_client.publish_signal(signal).await
    }
    
    /// Send coordination message
    pub async fn send_coordination(&self, message_type: CoordinationType, payload: serde_json::Value, targets: Vec<String>) -> Result<(), SwarmError> {
        let message = CoordinationMessage {
            message_type,
            source_node: self.node_id.clone(),
            target_nodes: targets,
            payload,
            timestamp: Utc::now(),
        };
        
        self.redis_client.publish_coordination(message).await
    }
    
    /// Get cached signals
    pub async fn get_cached_signals(&self) -> HashMap<String, DistributedSignal> {
        self.signal_cache.read().await.clone()
    }
    
    /// Store coordination state
    pub async fn store_coordination_state(&self, key: &str, value: serde_json::Value) -> Result<(), SwarmError> {
        self.redis_client.store_data(key, &value, None).await
    }
    
    /// Retrieve coordination state
    pub async fn retrieve_coordination_state(&self, key: &str) -> Result<Option<serde_json::Value>, SwarmError> {
        self.redis_client.retrieve_data(key).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_redis_config_default() {
        let config = RedisConfig::default();
        assert!(!config.redis_url.is_empty());
        assert!(config.pool_size > 0);
        assert!(!config.signal_channels.is_empty());
    }
    
    #[test]
    fn test_distributed_signal_serialization() {
        let signal = DistributedSignal {
            signal_id: "test_signal".to_string(),
            source_node: "node1".to_string(),
            signal_data: SignalData::FusedResult(HashMap::from([
                ("signal1".to_string(), 0.5),
                ("signal2".to_string(), 0.3),
            ])),
            metadata: HashMap::new(),
            timestamp: Utc::now(),
            priority: MessagePriority::Medium,
        };
        
        let serialized = serde_json::to_string(&signal).unwrap();
        let deserialized: DistributedSignal = serde_json::from_str(&serialized).unwrap();
        
        assert_eq!(signal.signal_id, deserialized.signal_id);
        assert_eq!(signal.source_node, deserialized.source_node);
    }
    
    #[test]
    fn test_coordination_message_serialization() {
        let message = CoordinationMessage {
            message_type: CoordinationType::HealthCheck,
            source_node: "node1".to_string(),
            target_nodes: vec!["node2".to_string(), "node3".to_string()],
            payload: serde_json::json!({"status": "active"}),
            timestamp: Utc::now(),
        };
        
        let serialized = serde_json::to_string(&message).unwrap();
        let deserialized: CoordinationMessage = serde_json::from_str(&serialized).unwrap();
        
        assert_eq!(message.source_node, deserialized.source_node);
        assert_eq!(message.target_nodes, deserialized.target_nodes);
    }
    
    #[test]
    fn test_redis_metrics_default() {
        let metrics = RedisMetrics::default();
        assert_eq!(metrics.total_messages_sent, 0);
        assert_eq!(metrics.total_messages_received, 0);
        assert_eq!(metrics.connection_errors, 0);
    }
}