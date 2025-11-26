// Message bus for inter-agent communication

use super::AgentId;
use crate::{DistributedError, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tokio::sync::{broadcast, Mutex, RwLock};
use uuid::Uuid;

/// Message type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MessageType {
    /// Task assignment
    TaskAssignment,

    /// Task result
    TaskResult,

    /// Heartbeat/ping
    Heartbeat,

    /// Agent registration
    Registration,

    /// Agent deregistration
    Deregistration,

    /// Consensus request
    ConsensusRequest,

    /// Consensus vote
    ConsensusVote,

    /// Data synchronization
    DataSync,

    /// Error notification
    Error,

    /// Custom message
    Custom,
}

/// Message between agents
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    /// Unique message ID
    pub id: Uuid,

    /// Message type
    pub message_type: MessageType,

    /// Sender agent ID
    pub from: AgentId,

    /// Recipient agent ID (None for broadcast)
    pub to: Option<AgentId>,

    /// Message payload
    pub payload: serde_json::Value,

    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,

    /// Priority (higher = more important)
    pub priority: u32,

    /// TTL (time-to-live) in seconds
    pub ttl: u64,
}

impl Message {
    /// Create new message
    pub fn new(
        message_type: MessageType,
        from: AgentId,
        to: Option<AgentId>,
        payload: serde_json::Value,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            message_type,
            from,
            to,
            payload,
            timestamp: chrono::Utc::now(),
            priority: 5,
            ttl: 300,
        }
    }

    /// Check if message is expired
    pub fn is_expired(&self) -> bool {
        let now = chrono::Utc::now();
        let age = now.signed_duration_since(self.timestamp);
        age.num_seconds() as u64 > self.ttl
    }

    /// Check if message is broadcast
    pub fn is_broadcast(&self) -> bool {
        self.to.is_none()
    }
}

/// Message bus for routing messages between agents
pub struct MessageBus {
    /// Message queues per agent
    queues: Arc<RwLock<HashMap<AgentId, Arc<Mutex<VecDeque<Message>>>>>>,

    /// Broadcast channel
    broadcast_tx: broadcast::Sender<Message>,

    /// Message history (for debugging/audit)
    history: Arc<RwLock<VecDeque<Message>>>,

    /// Maximum history size
    max_history: usize,

    /// Message handlers
    handlers: Arc<RwLock<HashMap<MessageType, MessageHandler>>>,
}

/// Message handler function type
type MessageHandler = Arc<dyn Fn(&Message) -> Result<()> + Send + Sync>;

impl MessageBus {
    /// Create new message bus
    pub fn new() -> Self {
        let (tx, _) = broadcast::channel(1000);

        Self {
            queues: Arc::new(RwLock::new(HashMap::new())),
            broadcast_tx: tx,
            history: Arc::new(RwLock::new(VecDeque::new())),
            max_history: 10000,
            handlers: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Register an agent with the bus
    pub async fn register_agent(&self, agent_id: AgentId) -> Result<()> {
        let mut queues = self.queues.write().await;
        queues.insert(agent_id.clone(), Arc::new(Mutex::new(VecDeque::new())));

        tracing::info!("Registered agent on message bus: {}", agent_id);
        Ok(())
    }

    /// Unregister an agent
    pub async fn unregister_agent(&self, agent_id: &AgentId) -> Result<()> {
        let mut queues = self.queues.write().await;
        queues.remove(agent_id);

        tracing::info!("Unregistered agent from message bus: {}", agent_id);
        Ok(())
    }

    /// Send message
    pub async fn send(&self, message: Message) -> Result<()> {
        // Check expiration
        if message.is_expired() {
            return Err(DistributedError::FederationError(
                "Message expired".to_string(),
            ));
        }

        // Add to history
        {
            let mut history = self.history.write().await;
            history.push_back(message.clone());
            if history.len() > self.max_history {
                history.pop_front();
            }
        }

        // Handle message
        self.handle_message(&message).await?;

        // Route message
        if message.is_broadcast() {
            // Broadcast to all agents
            let _ = self.broadcast_tx.send(message);
        } else {
            // Send to specific agent
            let queues = self.queues.read().await;
            if let Some(recipient) = &message.to {
                if let Some(queue) = queues.get(recipient) {
                    queue.lock().await.push_back(message);
                } else {
                    return Err(DistributedError::AgentNotFound(recipient.clone()));
                }
            }
        }

        Ok(())
    }

    /// Receive messages for an agent
    pub async fn receive(&self, agent_id: &AgentId) -> Result<Vec<Message>> {
        let queues = self.queues.read().await;
        let queue = queues
            .get(agent_id)
            .ok_or_else(|| DistributedError::AgentNotFound(agent_id.clone()))?;

        let mut messages = Vec::new();
        let mut queue_lock = queue.lock().await;

        while let Some(msg) = queue_lock.pop_front() {
            if !msg.is_expired() {
                messages.push(msg);
            }
        }

        Ok(messages)
    }

    /// Subscribe to broadcasts
    pub fn subscribe(&self) -> broadcast::Receiver<Message> {
        self.broadcast_tx.subscribe()
    }

    /// Register message handler
    pub async fn register_handler<F>(&self, message_type: MessageType, handler: F)
    where
        F: Fn(&Message) -> Result<()> + Send + Sync + 'static,
    {
        let mut handlers = self.handlers.write().await;
        handlers.insert(message_type, Arc::new(handler));
    }

    /// Handle message with registered handlers
    async fn handle_message(&self, message: &Message) -> Result<()> {
        let handlers = self.handlers.read().await;

        if let Some(handler) = handlers.get(&message.message_type) {
            handler(message)?;
        }

        Ok(())
    }

    /// Get queue size for an agent
    pub async fn queue_size(&self, agent_id: &AgentId) -> Result<usize> {
        let queues = self.queues.read().await;
        let queue = queues
            .get(agent_id)
            .ok_or_else(|| DistributedError::AgentNotFound(agent_id.clone()))?;

        let size = queue.lock().await.len();
        Ok(size)
    }

    /// Get message statistics
    pub async fn stats(&self) -> MessageBusStats {
        let queues = self.queues.read().await;
        let total_queued: usize = {
            let mut total = 0;
            for queue in queues.values() {
                total += queue.lock().await.len();
            }
            total
        };

        let history_size = self.history.read().await.len();

        MessageBusStats {
            registered_agents: queues.len(),
            total_queued_messages: total_queued,
            history_size,
            broadcast_capacity: self.broadcast_tx.len(),
        }
    }

    /// Clear expired messages from all queues
    pub async fn cleanup_expired(&self) -> Result<usize> {
        let queues = self.queues.read().await;
        let mut removed = 0;

        for queue in queues.values() {
            let mut queue_lock = queue.lock().await;
            let original_len = queue_lock.len();

            queue_lock.retain(|msg| !msg.is_expired());

            removed += original_len - queue_lock.len();
        }

        Ok(removed)
    }
}

impl Default for MessageBus {
    fn default() -> Self {
        Self::new()
    }
}

/// Message bus statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageBusStats {
    /// Number of registered agents
    pub registered_agents: usize,

    /// Total queued messages
    pub total_queued_messages: usize,

    /// History size
    pub history_size: usize,

    /// Broadcast channel capacity
    pub broadcast_capacity: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_creation() {
        let msg = Message::new(
            MessageType::TaskAssignment,
            "agent-1".to_string(),
            Some("agent-2".to_string()),
            serde_json::json!({"task": "test"}),
        );

        assert_eq!(msg.message_type, MessageType::TaskAssignment);
        assert_eq!(msg.from, "agent-1");
        assert!(!msg.is_expired());
        assert!(!msg.is_broadcast());
    }

    #[test]
    fn test_broadcast_message() {
        let msg = Message::new(
            MessageType::Heartbeat,
            "agent-1".to_string(),
            None,
            serde_json::json!({}),
        );

        assert!(msg.is_broadcast());
    }

    #[tokio::test]
    async fn test_message_bus_registration() {
        let bus = MessageBus::new();

        bus.register_agent("agent-1".to_string()).await.unwrap();
        bus.register_agent("agent-2".to_string()).await.unwrap();

        let stats = bus.stats().await;
        assert_eq!(stats.registered_agents, 2);
    }

    #[tokio::test]
    async fn test_message_sending() {
        let bus = MessageBus::new();

        bus.register_agent("agent-1".to_string()).await.unwrap();
        bus.register_agent("agent-2".to_string()).await.unwrap();

        let msg = Message::new(
            MessageType::TaskAssignment,
            "agent-1".to_string(),
            Some("agent-2".to_string()),
            serde_json::json!({}),
        );

        bus.send(msg).await.unwrap();

        let messages = bus.receive(&"agent-2".to_string()).await.unwrap();
        assert_eq!(messages.len(), 1);
    }
}
