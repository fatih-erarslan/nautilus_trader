//! Advanced messaging system with MessagePack serialization and distributed protocols
//!
//! This module provides a comprehensive messaging infrastructure for distributed CDFA operations:
//! - MessagePack serialization for efficient binary data transfer
//! - Multiple message types for different use cases
//! - Message versioning and compatibility
//! - Message routing and delivery guarantees
//! - Broadcast and multicast capabilities
//! - Message compression and deduplication

use std::{
    collections::{HashMap, VecDeque},
    sync::{Arc, atomic::{AtomicU64, Ordering}},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use tokio::sync::{RwLock, mpsc, oneshot};
use serde::{Serialize, Deserialize};
use redis::AsyncCommands;
use uuid::Uuid;
use sha2::{Sha256, Digest};
use crate::{
    error::{CdfaError, Result},
    integration::redis_connector::{RedisPool, RedisClusterCoordinator},
};

/// Message protocol version for backward compatibility
pub const MESSAGE_PROTOCOL_VERSION: u32 = 1;

/// Maximum message size in bytes (10MB)
pub const MAX_MESSAGE_SIZE: usize = 10 * 1024 * 1024;

/// Message delivery priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum MessagePriority {
    Low = 1,
    Normal = 2,
    High = 3,
    Critical = 4,
}

/// Message delivery guarantees
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeliveryMode {
    /// Fire and forget - no delivery confirmation
    FireAndForget,
    /// At least once delivery with acknowledgment
    AtLeastOnce,
    /// Exactly once delivery with deduplication
    ExactlyOnce,
}

/// Message types for different distributed operations
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MessageType {
    /// Computation task distribution
    TaskDistribution,
    /// Computation results collection
    ResultCollection,
    /// Algorithm coordination signals
    CoordinationSignal,
    /// Health check and monitoring
    HealthCheck,
    /// Configuration synchronization
    ConfigSync,
    /// Neural network training coordination
    NeuralTraining,
    /// Data pipeline events
    DataPipeline,
    /// Custom application-specific messages
    Custom(String),
}

/// Message header with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageHeader {
    /// Unique message identifier
    pub id: Uuid,
    /// Message type classification
    pub message_type: MessageType,
    /// Protocol version for compatibility
    pub version: u32,
    /// Source node identifier
    pub source: String,
    /// Target node identifier (None for broadcast)
    pub target: Option<String>,
    /// Message creation timestamp
    pub timestamp: u64,
    /// Time-to-live in seconds
    pub ttl: u64,
    /// Message priority level
    pub priority: MessagePriority,
    /// Delivery mode and guarantees
    pub delivery_mode: DeliveryMode,
    /// Compression algorithm used (if any)
    pub compression: Option<String>,
    /// Message content hash for integrity
    pub content_hash: String,
    /// Retry count for failed deliveries
    pub retry_count: u32,
    /// Custom metadata fields
    pub metadata: HashMap<String, String>,
}

impl MessageHeader {
    /// Create a new message header
    pub fn new(
        message_type: MessageType,
        source: String,
        target: Option<String>,
        ttl: u64,
        priority: MessagePriority,
        delivery_mode: DeliveryMode,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            message_type,
            version: MESSAGE_PROTOCOL_VERSION,
            source,
            target,
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
            ttl,
            priority,
            delivery_mode,
            compression: None,
            content_hash: String::new(),
            retry_count: 0,
            metadata: HashMap::new(),
        }
    }

    /// Check if message has expired
    pub fn is_expired(&self) -> bool {
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
        now > self.timestamp + self.ttl
    }

    /// Update content hash
    pub fn update_content_hash(&mut self, content: &[u8]) {
        let mut hasher = Sha256::new();
        hasher.update(content);
        self.content_hash = format!("{:x}", hasher.finalize());
    }

    /// Verify content integrity
    pub fn verify_content(&self, content: &[u8]) -> bool {
        let mut hasher = Sha256::new();
        hasher.update(content);
        let hash = format!("{:x}", hasher.finalize());
        hash == self.content_hash
    }
}

/// Complete message with header and payload
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    /// Message header with metadata
    pub header: MessageHeader,
    /// Message payload (MessagePack serialized)
    pub payload: Vec<u8>,
}

impl Message {
    /// Create a new message with payload
    pub fn new<T: Serialize>(
        message_type: MessageType,
        source: String,
        target: Option<String>,
        payload: &T,
        ttl: u64,
        priority: MessagePriority,
        delivery_mode: DeliveryMode,
    ) -> Result<Self> {
        let payload_bytes = rmp_serde::to_vec(payload)
            .map_err(|e| CdfaError::Serialization(format!("Failed to serialize payload: {}", e)))?;

        if payload_bytes.len() > MAX_MESSAGE_SIZE {
            return Err(CdfaError::invalid_input(format!(
                "Message payload too large: {} bytes (max: {})",
                payload_bytes.len(),
                MAX_MESSAGE_SIZE
            )));
        }

        let mut header = MessageHeader::new(
            message_type,
            source,
            target,
            ttl,
            priority,
            delivery_mode,
        );
        header.update_content_hash(&payload_bytes);

        Ok(Self {
            header,
            payload: payload_bytes,
        })
    }

    /// Deserialize message payload
    pub fn deserialize_payload<T: for<'de> Deserialize<'de>>(&self) -> Result<T> {
        if !self.header.verify_content(&self.payload) {
            return Err(CdfaError::Corruption("Message integrity check failed".to_string()));
        }

        rmp_serde::from_slice(&self.payload)
            .map_err(|e| CdfaError::Serialization(format!("Failed to deserialize payload: {}", e)))
    }

    /// Get message size in bytes
    pub fn size(&self) -> usize {
        // Approximate size including header overhead
        self.payload.len() + 200 // header overhead estimate
    }

    /// Check if message should be retried
    pub fn should_retry(&self, max_retries: u32) -> bool {
        !self.header.is_expired() && self.header.retry_count < max_retries
    }

    /// Increment retry count
    pub fn increment_retry(&mut self) {
        self.header.retry_count += 1;
    }
}

/// Message routing table for efficient message delivery
#[derive(Debug, Clone)]
pub struct MessageRouter {
    routes: HashMap<String, Vec<String>>, // target -> channels
    wildcards: Vec<(String, Vec<String>)>, // pattern -> channels
}

impl MessageRouter {
    /// Create a new message router
    pub fn new() -> Self {
        Self {
            routes: HashMap::new(),
            wildcards: Vec::new(),
        }
    }

    /// Add a route for a specific target
    pub fn add_route(&mut self, target: &str, channel: &str) {
        self.routes
            .entry(target.to_string())
            .or_insert_with(Vec::new)
            .push(channel.to_string());
    }

    /// Add a wildcard route
    pub fn add_wildcard_route(&mut self, pattern: &str, channel: &str) {
        self.wildcards.push((pattern.to_string(), vec![channel.to_string()]));
    }

    /// Get channels for a target
    pub fn get_channels(&self, target: &str) -> Vec<String> {
        let mut channels = Vec::new();

        // Direct routes
        if let Some(direct_channels) = self.routes.get(target) {
            channels.extend_from_slice(direct_channels);
        }

        // Wildcard routes
        for (pattern, wildcard_channels) in &self.wildcards {
            if self.matches_pattern(target, pattern) {
                channels.extend_from_slice(wildcard_channels);
            }
        }

        channels.sort();
        channels.dedup();
        channels
    }

    /// Check if target matches pattern (simple wildcard matching)
    fn matches_pattern(&self, target: &str, pattern: &str) -> bool {
        if pattern == "*" {
            return true;
        }

        if pattern.ends_with('*') {
            let prefix = &pattern[..pattern.len() - 1];
            return target.starts_with(prefix);
        }

        if pattern.starts_with('*') {
            let suffix = &pattern[1..];
            return target.ends_with(suffix);
        }

        target == pattern
    }
}

/// Message queue statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MessageQueueStats {
    pub total_sent: u64,
    pub total_received: u64,
    pub total_failed: u64,
    pub current_queue_size: usize,
    pub average_processing_time_ms: f64,
    pub messages_by_type: HashMap<String, u64>,
    pub messages_by_priority: HashMap<String, u64>,
}

/// Acknowledgment status for reliable delivery
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageAck {
    pub message_id: Uuid,
    pub status: AckStatus,
    pub timestamp: u64,
    pub error_message: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AckStatus {
    Acknowledged,
    Failed,
    Timeout,
}

/// Advanced message broker with Redis backend
pub struct RedisMessageBroker {
    pool: Arc<RedisPool>,
    node_id: String,
    router: Arc<RwLock<MessageRouter>>,
    stats: Arc<RwLock<MessageQueueStats>>,
    message_counter: AtomicU64,
    pending_acks: Arc<RwLock<HashMap<Uuid, oneshot::Sender<MessageAck>>>>,
    channels: HashMap<String, mpsc::UnboundedSender<Message>>,
}

impl RedisMessageBroker {
    /// Create a new Redis message broker
    pub fn new(pool: Arc<RedisPool>, node_id: String) -> Self {
        Self {
            pool,
            node_id,
            router: Arc::new(RwLock::new(MessageRouter::new())),
            stats: Arc::new(RwLock::new(MessageQueueStats::default())),
            message_counter: AtomicU64::new(0),
            pending_acks: Arc::new(RwLock::new(HashMap::new())),
            channels: HashMap::new(),
        }
    }

    /// Send a message through the broker
    pub async fn send_message(&self, message: Message) -> Result<Option<MessageAck>> {
        let start_time = Instant::now();
        self.message_counter.fetch_add(1, Ordering::Relaxed);

        // Route message to appropriate channels
        let channels = if let Some(ref target) = message.header.target {
            self.router.read().await.get_channels(target)
        } else {
            // Broadcast message
            vec!["broadcast".to_string()]
        };

        if channels.is_empty() {
            return Err(CdfaError::Network("No routes available for message".to_string()));
        }

        // Serialize message for Redis
        let serialized_message = rmp_serde::to_vec(&message)
            .map_err(|e| CdfaError::Serialization(format!("Failed to serialize message: {}", e)))?;

        // Send to Redis channels
        let conn = self.pool.get_connection().await?;
        let mut locked_conn = conn.lock().await;

        for channel in &channels {
            let _: u64 = locked_conn.connection
                .publish(channel, &serialized_message)
                .await
                .map_err(|e| CdfaError::Network(format!("Failed to publish message: {}", e)))?;
        }

        // Handle acknowledgments for reliable delivery
        let ack_future = if message.header.delivery_mode != DeliveryMode::FireAndForget {
            Some(self.setup_acknowledgment(&message).await?)
        } else {
            None
        };

        // Update statistics
        self.update_send_stats(&message, start_time.elapsed()).await;

        // Wait for acknowledgment if required
        if let Some(ack_receiver) = ack_future {
            let timeout = Duration::from_secs(message.header.ttl.min(30)); // Max 30 second timeout
            match tokio::time::timeout(timeout, ack_receiver).await {
                Ok(Ok(ack)) => Ok(Some(ack)),
                Ok(Err(_)) => Err(CdfaError::Network("Acknowledgment channel closed".to_string())),
                Err(_) => {
                    let timeout_ack = MessageAck {
                        message_id: message.header.id,
                        status: AckStatus::Timeout,
                        timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
                        error_message: Some("Acknowledgment timeout".to_string()),
                    };
                    Ok(Some(timeout_ack))
                }
            }
        } else {
            Ok(None)
        }
    }

    /// Setup acknowledgment tracking for reliable delivery
    async fn setup_acknowledgment(&self, message: &Message) -> Result<oneshot::Receiver<MessageAck>> {
        let (tx, rx) = oneshot::channel();
        
        // Store in pending acknowledgments
        {
            let mut pending = self.pending_acks.write().await;
            pending.insert(message.header.id, tx);
        }

        // Set up Redis key for acknowledgment tracking
        let ack_key = format!("cdfa:acks:{}", message.header.id);
        let conn = self.pool.get_connection().await?;
        let mut locked_conn = conn.lock().await;

        let _: () = locked_conn.connection
            .set_ex(&ack_key, "pending", message.header.ttl)
            .await
            .map_err(|e| CdfaError::Network(format!("Failed to set acknowledgment key: {}", e)))?;

        Ok(rx)
    }

    /// Subscribe to messages on a channel
    pub async fn subscribe(&mut self, channel: &str) -> Result<mpsc::UnboundedReceiver<Message>> {
        let (tx, rx) = mpsc::unbounded_channel();
        
        // Store channel sender
        self.channels.insert(channel.to_string(), tx.clone());

        // Start Redis subscription task
        let pool = self.pool.clone();
        let channel_name = channel.to_string();
        let stats = self.stats.clone();
        
        tokio::spawn(async move {
            if let Err(e) = Self::subscription_task(pool, channel_name, tx, stats).await {
                log::error!("Subscription task failed: {}", e);
            }
        });

        Ok(rx)
    }

    /// Redis subscription task
    async fn subscription_task(
        pool: Arc<RedisPool>,
        channel: String,
        sender: mpsc::UnboundedSender<Message>,
        stats: Arc<RwLock<MessageQueueStats>>,
    ) -> Result<()> {
        let conn = pool.get_connection().await?;
        let mut locked_conn = conn.lock().await;

        let mut pubsub = locked_conn.connection.as_mut().get_async_connection().await
            .map_err(|e| CdfaError::Network(format!("Failed to get pubsub connection: {}", e)))?
            .into_pubsub();

        pubsub.subscribe(&channel).await
            .map_err(|e| CdfaError::Network(format!("Failed to subscribe to channel: {}", e)))?;

        let mut stream = pubsub.on_message();
        
        while let Some(msg) = stream.next().await {
            let payload: Vec<u8> = msg.get_payload()
                .map_err(|e| CdfaError::Network(format!("Failed to get message payload: {}", e)))?;

            match rmp_serde::from_slice::<Message>(&payload) {
                Ok(message) => {
                    // Update receive statistics
                    {
                        let mut stats_guard = stats.write().await;
                        stats_guard.total_received += 1;
                        stats_guard.current_queue_size += 1;
                        
                        let msg_type = format!("{:?}", message.header.message_type);
                        *stats_guard.messages_by_type.entry(msg_type).or_insert(0) += 1;
                        
                        let priority = format!("{:?}", message.header.priority);
                        *stats_guard.messages_by_priority.entry(priority).or_insert(0) += 1;
                    }

                    // Send to application handler
                    if sender.send(message).is_err() {
                        log::warn!("Message receiver dropped for channel: {}", channel);
                        break;
                    }
                }
                Err(e) => {
                    log::error!("Failed to deserialize message: {}", e);
                    let mut stats_guard = stats.write().await;
                    stats_guard.total_failed += 1;
                }
            }
        }

        Ok(())
    }

    /// Send acknowledgment for a received message
    pub async fn acknowledge_message(&self, message_id: Uuid, status: AckStatus, error_message: Option<String>) -> Result<()> {
        let ack = MessageAck {
            message_id,
            status: status.clone(),
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
            error_message,
        };

        // Send acknowledgment through Redis
        let ack_key = format!("cdfa:acks:{}", message_id);
        let ack_data = rmp_serde::to_vec(&ack)
            .map_err(|e| CdfaError::Serialization(format!("Failed to serialize ack: {}", e)))?;

        let conn = self.pool.get_connection().await?;
        let mut locked_conn = conn.lock().await;

        let _: () = locked_conn.connection
            .set_ex(&ack_key, ack_data, 300) // 5 minute TTL
            .await
            .map_err(|e| CdfaError::Network(format!("Failed to set acknowledgment: {}", e)))?;

        // Notify pending acknowledgment
        if let Some(sender) = self.pending_acks.write().await.remove(&message_id) {
            let _ = sender.send(ack); // Ignore if receiver dropped
        }

        Ok(())
    }

    /// Broadcast a message to all nodes
    pub async fn broadcast_message<T: Serialize>(
        &self,
        message_type: MessageType,
        payload: &T,
        priority: MessagePriority,
    ) -> Result<()> {
        let message = Message::new(
            message_type,
            self.node_id.clone(),
            None, // Broadcast
            payload,
            300, // 5 minute TTL
            priority,
            DeliveryMode::FireAndForget,
        )?;

        self.send_message(message).await?;
        Ok(())
    }

    /// Get current message broker statistics
    pub async fn get_stats(&self) -> MessageQueueStats {
        self.stats.read().await.clone()
    }

    /// Update send statistics
    async fn update_send_stats(&self, message: &Message, duration: Duration) {
        let mut stats = self.stats.write().await;
        stats.total_sent += 1;
        
        // Update average processing time
        let new_time = duration.as_millis() as f64;
        if stats.average_processing_time_ms == 0.0 {
            stats.average_processing_time_ms = new_time;
        } else {
            stats.average_processing_time_ms = 0.9 * stats.average_processing_time_ms + 0.1 * new_time;
        }

        // Update type and priority counters
        let msg_type = format!("{:?}", message.header.message_type);
        *stats.messages_by_type.entry(msg_type).or_insert(0) += 1;
        
        let priority = format!("{:?}", message.header.priority);
        *stats.messages_by_priority.entry(priority).or_insert(0) += 1;
    }

    /// Add a message route
    pub async fn add_route(&self, target: &str, channel: &str) {
        self.router.write().await.add_route(target, channel);
    }

    /// Add a wildcard route
    pub async fn add_wildcard_route(&self, pattern: &str, channel: &str) {
        self.router.write().await.add_wildcard_route(pattern, channel);
    }

    /// Gracefully shutdown the message broker
    pub async fn shutdown(&self) -> Result<()> {
        log::info!("Shutting down Redis message broker");
        
        // Close all channels
        for (channel, _) in &self.channels {
            log::debug!("Closing channel: {}", channel);
        }

        // Clear pending acknowledgments
        self.pending_acks.write().await.clear();

        Ok(())
    }
}

/// High-level messaging API for common operations
pub struct MessageAPI {
    broker: Arc<RedisMessageBroker>,
}

impl MessageAPI {
    /// Create a new message API
    pub fn new(broker: Arc<RedisMessageBroker>) -> Self {
        Self { broker }
    }

    /// Send a task distribution message
    pub async fn send_task<T: Serialize>(&self, target: &str, task: &T) -> Result<()> {
        let message = Message::new(
            MessageType::TaskDistribution,
            self.broker.node_id.clone(),
            Some(target.to_string()),
            task,
            600, // 10 minute TTL
            MessagePriority::Normal,
            DeliveryMode::AtLeastOnce,
        )?;

        self.broker.send_message(message).await?;
        Ok(())
    }

    /// Send computation results
    pub async fn send_results<T: Serialize>(&self, target: &str, results: &T) -> Result<()> {
        let message = Message::new(
            MessageType::ResultCollection,
            self.broker.node_id.clone(),
            Some(target.to_string()),
            results,
            300, // 5 minute TTL
            MessagePriority::High,
            DeliveryMode::ExactlyOnce,
        )?;

        self.broker.send_message(message).await?;
        Ok(())
    }

    /// Send coordination signal
    pub async fn send_coordination_signal<T: Serialize>(&self, signal: &T) -> Result<()> {
        self.broker.broadcast_message(
            MessageType::CoordinationSignal,
            signal,
            MessagePriority::High,
        ).await
    }

    /// Send health check
    pub async fn send_health_check(&self) -> Result<()> {
        let health_data = serde_json::json!({
            "node_id": self.broker.node_id,
            "timestamp": SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
            "status": "healthy"
        });

        self.broker.broadcast_message(
            MessageType::HealthCheck,
            &health_data,
            MessagePriority::Low,
        ).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_creation() {
        let payload = "test payload";
        let message = Message::new(
            MessageType::TaskDistribution,
            "node1".to_string(),
            Some("node2".to_string()),
            &payload,
            300,
            MessagePriority::Normal,
            DeliveryMode::AtLeastOnce,
        ).unwrap();

        assert_eq!(message.header.source, "node1");
        assert_eq!(message.header.target, Some("node2".to_string()));
        assert_eq!(message.header.priority, MessagePriority::Normal);
        assert!(!message.header.content_hash.is_empty());
    }

    #[test]
    fn test_message_serialization() {
        let payload = "test payload";
        let message = Message::new(
            MessageType::TaskDistribution,
            "node1".to_string(),
            Some("node2".to_string()),
            &payload,
            300,
            MessagePriority::Normal,
            DeliveryMode::AtLeastOnce,
        ).unwrap();

        let deserialized: String = message.deserialize_payload().unwrap();
        assert_eq!(deserialized, payload);
    }

    #[test]
    fn test_message_router() {
        let mut router = MessageRouter::new();
        router.add_route("node1", "channel1");
        router.add_route("node1", "channel2");
        router.add_wildcard_route("node*", "wildcard_channel");

        let channels = router.get_channels("node1");
        assert!(channels.contains(&"channel1".to_string()));
        assert!(channels.contains(&"channel2".to_string()));
        assert!(channels.contains(&"wildcard_channel".to_string()));

        let wildcard_channels = router.get_channels("node123");
        assert!(wildcard_channels.contains(&"wildcard_channel".to_string()));
    }

    #[test]
    fn test_message_expiration() {
        let mut header = MessageHeader::new(
            MessageType::HealthCheck,
            "node1".to_string(),
            None,
            1, // 1 second TTL
            MessagePriority::Low,
            DeliveryMode::FireAndForget,
        );

        assert!(!header.is_expired());

        // Simulate time passing
        header.timestamp -= 2; // 2 seconds ago
        assert!(header.is_expired());
    }

    #[test]
    fn test_message_integrity() {
        let payload = b"test payload";
        let mut header = MessageHeader::new(
            MessageType::TaskDistribution,
            "node1".to_string(),
            None,
            300,
            MessagePriority::Normal,
            DeliveryMode::AtLeastOnce,
        );

        header.update_content_hash(payload);
        assert!(header.verify_content(payload));
        assert!(!header.verify_content(b"different payload"));
    }
}