//! Message types and routing

use crate::agent::AgentId;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Message ID
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MessageId(Uuid);

impl MessageId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for MessageId {
    fn default() -> Self {
        Self::new()
    }
}

/// Message priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Priority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}

impl Default for Priority {
    fn default() -> Self {
        Self::Normal
    }
}

/// Message types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MessageType {
    /// Request-response pattern
    Request {
        method: String,
        params: serde_json::Value,
    },
    /// Response to a request
    Response {
        request_id: MessageId,
        result: Result<serde_json::Value, String>,
    },
    /// One-way notification
    Notification {
        event: String,
        data: serde_json::Value,
    },
    /// Broadcast to multiple agents
    Broadcast {
        topic: String,
        data: serde_json::Value,
    },
    /// Consensus request
    ConsensusRequest {
        proposal: serde_json::Value,
        timeout_ms: u64,
    },
    /// Vote in consensus
    ConsensusVote {
        request_id: MessageId,
        vote: bool,
        reason: Option<String>,
    },
    /// Heartbeat
    Heartbeat,
    /// Control message
    Control {
        command: String,
        args: Vec<String>,
    },
}

/// Message envelope
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub id: MessageId,
    pub from: AgentId,
    pub to: Option<AgentId>,
    pub message_type: MessageType,
    pub priority: Priority,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub headers: HashMap<String, String>,
    pub correlation_id: Option<MessageId>,
    pub ttl_ms: Option<u64>,
}

impl Message {
    /// Create a new message
    pub fn new(from: AgentId, message_type: MessageType) -> Self {
        Self {
            id: MessageId::new(),
            from,
            to: None,
            message_type,
            priority: Priority::default(),
            timestamp: chrono::Utc::now(),
            headers: HashMap::new(),
            correlation_id: None,
            ttl_ms: None,
        }
    }

    /// Set recipient
    pub fn to(mut self, to: AgentId) -> Self {
        self.to = Some(to);
        self
    }

    /// Set priority
    pub fn with_priority(mut self, priority: Priority) -> Self {
        self.priority = priority;
        self
    }

    /// Add header
    pub fn with_header(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.insert(key.into(), value.into());
        self
    }

    /// Set correlation ID
    pub fn with_correlation_id(mut self, id: MessageId) -> Self {
        self.correlation_id = Some(id);
        self
    }

    /// Set TTL
    pub fn with_ttl_ms(mut self, ttl_ms: u64) -> Self {
        self.ttl_ms = Some(ttl_ms);
        self
    }

    /// Check if message has expired
    pub fn is_expired(&self) -> bool {
        if let Some(ttl_ms) = self.ttl_ms {
            let now = chrono::Utc::now();
            let elapsed = now - self.timestamp;
            elapsed.num_milliseconds() as u64 > ttl_ms
        } else {
            false
        }
    }

    /// Create a request message
    pub fn request(
        from: AgentId,
        method: impl Into<String>,
        params: serde_json::Value,
    ) -> Self {
        Self::new(
            from,
            MessageType::Request {
                method: method.into(),
                params,
            },
        )
    }

    /// Create a response message
    pub fn response(
        from: AgentId,
        request_id: MessageId,
        result: Result<serde_json::Value, String>,
    ) -> Self {
        Self::new(from, MessageType::Response { request_id, result })
            .with_correlation_id(request_id)
    }

    /// Create a notification message
    pub fn notification(
        from: AgentId,
        event: impl Into<String>,
        data: serde_json::Value,
    ) -> Self {
        Self::new(
            from,
            MessageType::Notification {
                event: event.into(),
                data,
            },
        )
    }

    /// Create a broadcast message
    pub fn broadcast(
        from: AgentId,
        topic: impl Into<String>,
        data: serde_json::Value,
    ) -> Self {
        Self::new(
            from,
            MessageType::Broadcast {
                topic: topic.into(),
                data,
            },
        )
    }

    /// Create a heartbeat message
    pub fn heartbeat(from: AgentId) -> Self {
        Self::new(from, MessageType::Heartbeat)
    }
}

/// Message router for handling message dispatch
#[async_trait::async_trait]
pub trait MessageRouter: Send + Sync {
    /// Route a message to its destination
    async fn route(&self, message: Message) -> anyhow::Result<()>;

    /// Subscribe to a topic
    async fn subscribe(&self, agent_id: AgentId, topic: String) -> anyhow::Result<()>;

    /// Unsubscribe from a topic
    async fn unsubscribe(&self, agent_id: AgentId, topic: String) -> anyhow::Result<()>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_creation() {
        let from = AgentId::new();
        let to = AgentId::new();
        
        let msg = Message::request(from, "get_price", serde_json::json!({"symbol": "BTC/USD"}))
            .to(to)
            .with_priority(Priority::High)
            .with_header("trace-id", "12345");
        
        assert_eq!(msg.from, from);
        assert_eq!(msg.to, Some(to));
        assert_eq!(msg.priority, Priority::High);
        assert_eq!(msg.headers.get("trace-id"), Some(&"12345".to_string()));
    }

    #[test]
    fn test_message_expiration() {
        let from = AgentId::new();
        
        // Message with no TTL should not expire
        let msg1 = Message::heartbeat(from);
        assert!(!msg1.is_expired());
        
        // Message with future TTL should not expire
        let msg2 = Message::heartbeat(from).with_ttl_ms(60000);
        assert!(!msg2.is_expired());
        
        // Message with past TTL should expire
        let mut msg3 = Message::heartbeat(from).with_ttl_ms(0);
        msg3.timestamp = chrono::Utc::now() - chrono::Duration::seconds(1);
        assert!(msg3.is_expired());
    }
}