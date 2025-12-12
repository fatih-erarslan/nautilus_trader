//! High-performance communication layer for inter-agent message passing.

use crate::error::{OrchestrationError, Result};
use crate::types::{AgentId, MessageId, MessageType, Timestamp};
use async_trait::async_trait;
use bytes::Bytes;
use crossbeam_channel::{bounded, unbounded, Receiver, Sender};
use dashmap::DashMap;
use flume::{Receiver as FlumeReceiver, Sender as FlumeSender};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{mpsc, oneshot, RwLock as TokioRwLock};
use tokio::time::{sleep, timeout};
use tracing::{debug, error, info, warn};

/// Maximum message size in bytes (16MB)
const MAX_MESSAGE_SIZE: usize = 16 * 1024 * 1024;

/// Default message timeout in milliseconds
const DEFAULT_MESSAGE_TIMEOUT: u64 = 5000;

/// Default retry attempts for failed messages
const DEFAULT_RETRY_ATTEMPTS: u32 = 3;

/// Message envelope containing metadata and payload
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    /// Unique message identifier
    pub id: MessageId,
    /// Message type
    pub message_type: MessageType,
    /// Sender agent ID
    pub sender: AgentId,
    /// Recipient agent ID
    pub recipient: AgentId,
    /// Message timestamp
    pub timestamp: Timestamp,
    /// Message priority
    pub priority: u8,
    /// Time-to-live in milliseconds
    pub ttl: u64,
    /// Retry count
    pub retry_count: u32,
    /// Correlation ID for request-response patterns
    pub correlation_id: Option<MessageId>,
    /// Message payload
    pub payload: Bytes,
    /// Message metadata
    pub metadata: HashMap<String, String>,
}

impl Message {
    /// Create a new message
    pub fn new(
        message_type: MessageType,
        sender: AgentId,
        recipient: AgentId,
        payload: Bytes,
    ) -> Self {
        Self {
            id: MessageId::new(),
            message_type,
            sender,
            recipient,
            timestamp: Timestamp::now(),
            priority: 0,
            ttl: DEFAULT_MESSAGE_TIMEOUT,
            retry_count: 0,
            correlation_id: None,
            payload,
            metadata: HashMap::new(),
        }
    }
    
    /// Create a request message
    pub fn request(sender: AgentId, recipient: AgentId, payload: Bytes) -> Self {
        Self::new(MessageType::Request, sender, recipient, payload)
    }
    
    /// Create a response message
    pub fn response(sender: AgentId, recipient: AgentId, payload: Bytes, correlation_id: MessageId) -> Self {
        let mut msg = Self::new(MessageType::Response, sender, recipient, payload);
        msg.correlation_id = Some(correlation_id);
        msg
    }
    
    /// Create an event message
    pub fn event(sender: AgentId, recipient: AgentId, payload: Bytes) -> Self {
        Self::new(MessageType::Event, sender, recipient, payload)
    }
    
    /// Create a heartbeat message
    pub fn heartbeat(sender: AgentId, recipient: AgentId) -> Self {
        Self::new(MessageType::Heartbeat, sender, recipient, Bytes::new())
    }
    
    /// Create a control message
    pub fn control(sender: AgentId, recipient: AgentId, payload: Bytes) -> Self {
        Self::new(MessageType::Control, sender, recipient, payload)
    }
    
    /// Create an error message
    pub fn error(sender: AgentId, recipient: AgentId, payload: Bytes) -> Self {
        Self::new(MessageType::Error, sender, recipient, payload)
    }
    
    /// Set message priority (0 = highest, 255 = lowest)
    pub fn with_priority(mut self, priority: u8) -> Self {
        self.priority = priority;
        self
    }
    
    /// Set time-to-live in milliseconds
    pub fn with_ttl(mut self, ttl: u64) -> Self {
        self.ttl = ttl;
        self
    }
    
    /// Add metadata to the message
    pub fn with_metadata<K, V>(mut self, key: K, value: V) -> Self
    where
        K: Into<String>,
        V: Into<String>,
    {
        self.metadata.insert(key.into(), value.into());
        self
    }
    
    /// Check if the message has expired
    pub fn is_expired(&self) -> bool {
        self.timestamp.elapsed().as_millis() > self.ttl as u128
    }
    
    /// Get the message size in bytes
    pub fn size(&self) -> usize {
        // Rough approximation of message size
        self.payload.len() + 
        self.metadata.iter().map(|(k, v)| k.len() + v.len()).sum::<usize>() +
        200 // Approximate overhead for other fields
    }
    
    /// Increment retry count
    pub fn increment_retry(&mut self) {
        self.retry_count += 1;
    }
    
    /// Check if message can be retried
    pub fn can_retry(&self) -> bool {
        self.retry_count < DEFAULT_RETRY_ATTEMPTS
    }
}

/// Message delivery status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeliveryStatus {
    /// Message is queued for delivery
    Queued,
    /// Message is being delivered
    InTransit,
    /// Message was delivered successfully
    Delivered,
    /// Message delivery failed
    Failed,
    /// Message expired before delivery
    Expired,
    /// Message was dropped due to queue overflow
    Dropped,
}

/// Message acknowledgment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageAck {
    /// Message ID being acknowledged
    pub message_id: MessageId,
    /// Acknowledgment status
    pub status: DeliveryStatus,
    /// Timestamp of acknowledgment
    pub timestamp: Timestamp,
    /// Error message if delivery failed
    pub error: Option<String>,
}

/// Communication layer trait for message passing
#[async_trait]
pub trait CommunicationLayer: Send + Sync {
    /// Send a message to another agent
    async fn send_message(&self, message: Message) -> Result<MessageId>;
    
    /// Receive messages for an agent
    async fn receive_messages(&self, agent_id: AgentId) -> Result<Vec<Message>>;
    
    /// Send a request and wait for response
    async fn send_request(&self, message: Message) -> Result<Message>;
    
    /// Register an agent for message delivery
    async fn register_agent(&self, agent_id: AgentId) -> Result<()>;
    
    /// Unregister an agent
    async fn unregister_agent(&self, agent_id: AgentId) -> Result<()>;
    
    /// Broadcast a message to all agents
    async fn broadcast(&self, message: Message) -> Result<Vec<MessageId>>;
    
    /// Get message delivery status
    async fn get_delivery_status(&self, message_id: MessageId) -> Result<DeliveryStatus>;
    
    /// Get communication statistics
    async fn get_stats(&self) -> Result<CommunicationStats>;
}

/// Communication statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunicationStats {
    /// Total messages sent
    pub messages_sent: u64,
    /// Total messages received
    pub messages_received: u64,
    /// Total messages failed
    pub messages_failed: u64,
    /// Total messages expired
    pub messages_expired: u64,
    /// Total messages dropped
    pub messages_dropped: u64,
    /// Average message processing time in microseconds
    pub avg_processing_time: f64,
    /// Current queue sizes per agent
    pub queue_sizes: HashMap<AgentId, usize>,
    /// Total bytes sent
    pub bytes_sent: u64,
    /// Total bytes received
    pub bytes_received: u64,
}

/// High-performance message router implementation
#[derive(Debug)]
pub struct MessageRouter {
    /// Agent message queues
    queues: Arc<DashMap<AgentId, mpsc::UnboundedSender<Message>>>,
    /// Pending response handlers
    pending_responses: Arc<DashMap<MessageId, oneshot::Sender<Message>>>,
    /// Message delivery status tracking
    delivery_status: Arc<DashMap<MessageId, DeliveryStatus>>,
    /// Communication statistics
    stats: Arc<RwLock<CommunicationStats>>,
    /// Message counters
    message_counter: Arc<AtomicU64>,
    /// Retry queue for failed messages
    retry_queue: Arc<TokioRwLock<Vec<Message>>>,
}

impl MessageRouter {
    /// Create a new message router
    pub fn new() -> Self {
        Self {
            queues: Arc::new(DashMap::new()),
            pending_responses: Arc::new(DashMap::new()),
            delivery_status: Arc::new(DashMap::new()),
            stats: Arc::new(RwLock::new(CommunicationStats {
                messages_sent: 0,
                messages_received: 0,
                messages_failed: 0,
                messages_expired: 0,
                messages_dropped: 0,
                avg_processing_time: 0.0,
                queue_sizes: HashMap::new(),
                bytes_sent: 0,
                bytes_received: 0,
            })),
            message_counter: Arc::new(AtomicU64::new(0)),
            retry_queue: Arc::new(TokioRwLock::new(Vec::new())),
        }
    }
    
    /// Start background tasks for message processing
    pub async fn start(&self) -> Result<()> {
        // Start retry handler
        let retry_queue = Arc::clone(&self.retry_queue);
        let router = self.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_millis(1000));
            loop {
                interval.tick().await;
                if let Err(e) = router.process_retry_queue().await {
                    error!("Error processing retry queue: {}", e);
                }
            }
        });
        
        // Start cleanup task for expired messages
        let delivery_status = Arc::clone(&self.delivery_status);
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(30));
            loop {
                interval.tick().await;
                // Clean up expired message status entries
                delivery_status.retain(|_, status| {
                    !matches!(status, DeliveryStatus::Expired | DeliveryStatus::Delivered)
                });
            }
        });
        
        info!("Message router started successfully");
        Ok(())
    }
    
    /// Process the retry queue
    async fn process_retry_queue(&self) -> Result<()> {
        let mut retry_queue = self.retry_queue.write().await;
        let mut messages_to_retry = Vec::new();
        
        // Extract messages that can be retried
        retry_queue.retain(|msg| {
            if msg.is_expired() {
                self.delivery_status.insert(msg.id, DeliveryStatus::Expired);
                false
            } else if msg.can_retry() {
                messages_to_retry.push(msg.clone());
                false
            } else {
                self.delivery_status.insert(msg.id, DeliveryStatus::Failed);
                false
            }
        });
        
        drop(retry_queue);
        
        // Retry messages
        for mut msg in messages_to_retry {
            msg.increment_retry();
            if let Err(e) = self.deliver_message(msg.clone()).await {
                warn!("Failed to retry message {}: {}", msg.id, e);
                self.retry_queue.write().await.push(msg);
            }
        }
        
        Ok(())
    }
    
    /// Deliver a message to its recipient
    async fn deliver_message(&self, message: Message) -> Result<()> {
        // Check message size
        if message.size() > MAX_MESSAGE_SIZE {
            return Err(OrchestrationError::communication(
                format!("Message size {} exceeds maximum {}", message.size(), MAX_MESSAGE_SIZE)
            ));
        }
        
        // Check if message has expired
        if message.is_expired() {
            self.delivery_status.insert(message.id, DeliveryStatus::Expired);
            return Err(OrchestrationError::communication("Message expired"));
        }
        
        // Get recipient queue
        let queue = self.queues.get(&message.recipient)
            .ok_or_else(|| OrchestrationError::not_found(format!("Agent {}", message.recipient)))?;
        
        // Set delivery status
        self.delivery_status.insert(message.id, DeliveryStatus::InTransit);
        
        // Send message
        match queue.send(message.clone()) {
            Ok(_) => {
                self.delivery_status.insert(message.id, DeliveryStatus::Delivered);
                
                // Update statistics
                let mut stats = self.stats.write();
                stats.messages_sent += 1;
                stats.bytes_sent += message.size() as u64;
                
                debug!("Message {} delivered to agent {}", message.id, message.recipient);
                Ok(())
            }
            Err(_) => {
                self.delivery_status.insert(message.id, DeliveryStatus::Failed);
                
                // Add to retry queue if retryable
                if message.can_retry() {
                    self.retry_queue.write().await.push(message);
                }
                
                Err(OrchestrationError::communication("Failed to deliver message"))
            }
        }
    }
    
    /// Handle request-response pattern
    async fn handle_response(&self, message: Message) -> Result<()> {
        if let Some(correlation_id) = message.correlation_id {
            if let Some((_, sender)) = self.pending_responses.remove(&correlation_id) {
                let _ = sender.send(message);
            }
        }
        Ok(())
    }
}

impl Clone for MessageRouter {
    fn clone(&self) -> Self {
        Self {
            queues: Arc::clone(&self.queues),
            pending_responses: Arc::clone(&self.pending_responses),
            delivery_status: Arc::clone(&self.delivery_status),
            stats: Arc::clone(&self.stats),
            message_counter: Arc::clone(&self.message_counter),
            retry_queue: Arc::clone(&self.retry_queue),
        }
    }
}

#[async_trait]
impl CommunicationLayer for MessageRouter {
    async fn send_message(&self, message: Message) -> Result<MessageId> {
        let message_id = message.id;
        
        // Set initial delivery status
        self.delivery_status.insert(message_id, DeliveryStatus::Queued);
        
        // Deliver message
        self.deliver_message(message).await?;
        
        Ok(message_id)
    }
    
    async fn receive_messages(&self, agent_id: AgentId) -> Result<Vec<Message>> {
        let mut messages = Vec::new();
        
        if let Some(queue) = self.queues.get(&agent_id) {
            let mut receiver = queue.clone();
            
            // Receive all available messages
            while let Ok(message) = receiver.try_recv() {
                messages.push(message);
            }
            
            // Update statistics
            let mut stats = self.stats.write();
            stats.messages_received += messages.len() as u64;
            stats.bytes_received += messages.iter().map(|m| m.size() as u64).sum::<u64>();
        }
        
        Ok(messages)
    }
    
    async fn send_request(&self, message: Message) -> Result<Message> {
        let message_id = message.id;
        
        // Create response channel
        let (tx, rx) = oneshot::channel();
        self.pending_responses.insert(message_id, tx);
        
        // Send request
        self.send_message(message).await?;
        
        // Wait for response with timeout
        match timeout(Duration::from_millis(DEFAULT_MESSAGE_TIMEOUT), rx).await {
            Ok(Ok(response)) => {
                // Handle response
                self.handle_response(response.clone()).await?;
                Ok(response)
            }
            Ok(Err(_)) => Err(OrchestrationError::communication("Response channel closed")),
            Err(_) => {
                self.pending_responses.remove(&message_id);
                Err(OrchestrationError::timeout(DEFAULT_MESSAGE_TIMEOUT))
            }
        }
    }
    
    async fn register_agent(&self, agent_id: AgentId) -> Result<()> {
        let (tx, mut rx) = mpsc::unbounded_channel();
        self.queues.insert(agent_id, tx);
        
        // Start message processing task for this agent
        let router = self.clone();
        tokio::spawn(async move {
            while let Some(message) = rx.recv().await {
                // Handle response messages
                if message.message_type == MessageType::Response {
                    if let Err(e) = router.handle_response(message).await {
                        error!("Failed to handle response: {}", e);
                    }
                }
            }
        });
        
        info!("Agent {} registered for message delivery", agent_id);
        Ok(())
    }
    
    async fn unregister_agent(&self, agent_id: AgentId) -> Result<()> {
        self.queues.remove(&agent_id);
        info!("Agent {} unregistered from message delivery", agent_id);
        Ok(())
    }
    
    async fn broadcast(&self, message: Message) -> Result<Vec<MessageId>> {
        let mut message_ids = Vec::new();
        
        for entry in self.queues.iter() {
            let agent_id = *entry.key();
            let mut broadcast_message = message.clone();
            broadcast_message.id = MessageId::new();
            broadcast_message.recipient = agent_id;
            
            match self.send_message(broadcast_message).await {
                Ok(msg_id) => message_ids.push(msg_id),
                Err(e) => warn!("Failed to broadcast to agent {}: {}", agent_id, e),
            }
        }
        
        Ok(message_ids)
    }
    
    async fn get_delivery_status(&self, message_id: MessageId) -> Result<DeliveryStatus> {
        self.delivery_status
            .get(&message_id)
            .map(|status| status.clone())
            .ok_or_else(|| OrchestrationError::not_found(format!("Message {}", message_id)))
    }
    
    async fn get_stats(&self) -> Result<CommunicationStats> {
        let mut stats = self.stats.read().clone();
        
        // Update queue sizes
        for entry in self.queues.iter() {
            let agent_id = *entry.key();
            // Note: We can't get exact queue size from UnboundedSender
            // This would require a different implementation with custom queues
            stats.queue_sizes.insert(agent_id, 0);
        }
        
        Ok(stats)
    }
}

/// High-performance lockfree message queue
#[derive(Debug)]
pub struct LockfreeMessageQueue {
    /// Flume channel for message passing
    sender: FlumeSender<Message>,
    receiver: FlumeReceiver<Message>,
    /// Queue statistics
    stats: Arc<AtomicU64>,
}

impl LockfreeMessageQueue {
    /// Create a new lockfree message queue
    pub fn new() -> Self {
        let (sender, receiver) = flume::unbounded();
        
        Self {
            sender,
            receiver,
            stats: Arc::new(AtomicU64::new(0)),
        }
    }
    
    /// Create a bounded lockfree message queue
    pub fn bounded(capacity: usize) -> Self {
        let (sender, receiver) = flume::bounded(capacity);
        
        Self {
            sender,
            receiver,
            stats: Arc::new(AtomicU64::new(0)),
        }
    }
    
    /// Send a message to the queue
    pub async fn send(&self, message: Message) -> Result<()> {
        self.sender.send_async(message).await
            .map_err(|_| OrchestrationError::communication("Queue send failed"))?;
        
        self.stats.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
    
    /// Receive a message from the queue
    pub async fn receive(&self) -> Result<Message> {
        self.receiver.recv_async().await
            .map_err(|_| OrchestrationError::communication("Queue receive failed"))
    }
    
    /// Try to receive a message without blocking
    pub fn try_receive(&self) -> Result<Message> {
        self.receiver.try_recv()
            .map_err(|_| OrchestrationError::communication("Queue try_recv failed"))
    }
    
    /// Get the number of messages processed
    pub fn messages_processed(&self) -> u64 {
        self.stats.load(Ordering::Relaxed)
    }
    
    /// Check if the queue is empty
    pub fn is_empty(&self) -> bool {
        self.receiver.is_empty()
    }
    
    /// Get the approximate queue length
    pub fn len(&self) -> usize {
        self.receiver.len()
    }
}

impl Clone for LockfreeMessageQueue {
    fn clone(&self) -> Self {
        Self {
            sender: self.sender.clone(),
            receiver: self.receiver.clone(),
            stats: Arc::clone(&self.stats),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::{sleep, Duration};
    
    #[tokio::test]
    async fn test_message_creation() {
        let sender = AgentId::new();
        let recipient = AgentId::new();
        let payload = Bytes::from("test payload");
        
        let message = Message::request(sender, recipient, payload.clone());
        
        assert_eq!(message.message_type, MessageType::Request);
        assert_eq!(message.sender, sender);
        assert_eq!(message.recipient, recipient);
        assert_eq!(message.payload, payload);
    }
    
    #[tokio::test]
    async fn test_message_router() {
        let router = MessageRouter::new();
        router.start().await.unwrap();
        
        let agent1 = AgentId::new();
        let agent2 = AgentId::new();
        
        // Register agents
        router.register_agent(agent1).await.unwrap();
        router.register_agent(agent2).await.unwrap();
        
        // Send message
        let message = Message::request(agent1, agent2, Bytes::from("test"));
        let message_id = router.send_message(message).await.unwrap();
        
        // Check delivery status
        let status = router.get_delivery_status(message_id).await.unwrap();
        assert_eq!(status, DeliveryStatus::Delivered);
        
        // Receive messages
        let messages = router.receive_messages(agent2).await.unwrap();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].payload, Bytes::from("test"));
    }
    
    #[tokio::test]
    async fn test_request_response() {
        let router = MessageRouter::new();
        router.start().await.unwrap();
        
        let agent1 = AgentId::new();
        let agent2 = AgentId::new();
        
        router.register_agent(agent1).await.unwrap();
        router.register_agent(agent2).await.unwrap();
        
        // Simulate agent2 responding to requests
        let router_clone = router.clone();
        tokio::spawn(async move {
            loop {
                let messages = router_clone.receive_messages(agent2).await.unwrap();
                for msg in messages {
                    if msg.message_type == MessageType::Request {
                        let response = Message::response(
                            agent2,
                            msg.sender,
                            Bytes::from("response"),
                            msg.id,
                        );
                        let _ = router_clone.send_message(response).await;
                    }
                }
                sleep(Duration::from_millis(10)).await;
            }
        });
        
        // Send request and wait for response
        let request = Message::request(agent1, agent2, Bytes::from("request"));
        let response = router.send_request(request).await.unwrap();
        
        assert_eq!(response.message_type, MessageType::Response);
        assert_eq!(response.payload, Bytes::from("response"));
    }
    
    #[tokio::test]
    async fn test_lockfree_queue() {
        let queue = LockfreeMessageQueue::new();
        
        let agent1 = AgentId::new();
        let agent2 = AgentId::new();
        let message = Message::request(agent1, agent2, Bytes::from("test"));
        
        queue.send(message.clone()).await.unwrap();
        
        let received = queue.receive().await.unwrap();
        assert_eq!(received.id, message.id);
        assert_eq!(received.payload, message.payload);
        
        assert_eq!(queue.messages_processed(), 1);
    }
    
    #[tokio::test]
    async fn test_message_expiration() {
        let agent1 = AgentId::new();
        let agent2 = AgentId::new();
        let mut message = Message::request(agent1, agent2, Bytes::from("test"));
        message.ttl = 1; // 1ms TTL
        
        // Wait for expiration
        sleep(Duration::from_millis(10)).await;
        
        assert!(message.is_expired());
    }
    
    #[tokio::test]
    async fn test_broadcast() {
        let router = MessageRouter::new();
        router.start().await.unwrap();
        
        let agent1 = AgentId::new();
        let agent2 = AgentId::new();
        let agent3 = AgentId::new();
        
        router.register_agent(agent1).await.unwrap();
        router.register_agent(agent2).await.unwrap();
        router.register_agent(agent3).await.unwrap();
        
        let message = Message::event(agent1, agent1, Bytes::from("broadcast"));
        let message_ids = router.broadcast(message).await.unwrap();
        
        assert_eq!(message_ids.len(), 3);
        
        // Check that all agents received the message
        let messages1 = router.receive_messages(agent1).await.unwrap();
        let messages2 = router.receive_messages(agent2).await.unwrap();
        let messages3 = router.receive_messages(agent3).await.unwrap();
        
        assert_eq!(messages1.len(), 1);
        assert_eq!(messages2.len(), 1);
        assert_eq!(messages3.len(), 1);
    }
}