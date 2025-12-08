//! # Coordination Bus
//!
//! Inter-agent communication and coordination system for quantum agents.

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{RwLock, broadcast};
use serde::{Deserialize, Serialize};
use crate::error::{PadsError, PadsResult};
use crate::types::*;

/// Coordination bus for inter-agent communication
pub struct CoordinationBus {
    message_sender: broadcast::Sender<CoordinationMessage>,
    message_receiver: Arc<RwLock<broadcast::Receiver<CoordinationMessage>>>,
    agent_registry: Arc<RwLock<HashMap<String, AgentInfo>>>,
    coordination_metrics: Arc<RwLock<CoordinationMetrics>>,
}

impl CoordinationBus {
    /// Create new coordination bus
    pub async fn new() -> PadsResult<Self> {
        let (sender, receiver) = broadcast::channel(1000);
        
        Ok(Self {
            message_sender: sender,
            message_receiver: Arc::new(RwLock::new(receiver)),
            agent_registry: Arc::new(RwLock::new(HashMap::new())),
            coordination_metrics: Arc::new(RwLock::new(CoordinationMetrics::default())),
        })
    }
    
    /// Register an agent with the coordination bus
    pub async fn register_agent(&self, agent_id: String, agent_info: AgentInfo) -> PadsResult<()> {
        let mut registry = self.agent_registry.write().await;
        registry.insert(agent_id, agent_info);
        
        // Update metrics
        if let Ok(mut metrics) = self.coordination_metrics.write().await {
            metrics.total_agents = registry.len() as u64;
        }
        
        Ok(())
    }
    
    /// Send coordination message
    pub async fn send_message(&self, message: CoordinationMessage) -> PadsResult<()> {
        self.message_sender.send(message)
            .map_err(|e| PadsError::CoordinationFailed(format!("Failed to send message: {}", e)))?;
        
        // Update metrics
        if let Ok(mut metrics) = self.coordination_metrics.write().await {
            metrics.messages_sent += 1;
        }
        
        Ok(())
    }
    
    /// Get coordination metrics
    pub async fn get_metrics(&self) -> CoordinationMetrics {
        self.coordination_metrics.read().await.clone()
    }
    
    /// Get registered agents
    pub async fn get_registered_agents(&self) -> Vec<String> {
        self.agent_registry.read().await.keys().cloned().collect()
    }
}

/// Coordination message between agents
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordinationMessage {
    pub message_id: String,
    pub sender_id: String,
    pub recipient_id: Option<String>, // None for broadcast
    pub message_type: MessageType,
    pub payload: MessagePayload,
    pub timestamp: u64,
    pub priority: MessagePriority,
}

/// Message type enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MessageType {
    /// Signal sharing between agents
    SignalShare,
    /// Status update
    StatusUpdate,
    /// Coordination request
    CoordinationRequest,
    /// Performance feedback
    PerformanceFeedback,
    /// Market data update
    MarketDataUpdate,
    /// System command
    SystemCommand,
}

/// Message payload variants
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MessagePayload {
    /// Agent signal data
    Signal(AgentSignalData),
    /// Status information
    Status(AgentStatus),
    /// Coordination data
    Coordination(CoordinationData),
    /// Performance metrics
    Performance(PerformanceData),
    /// Market data
    MarketData(MarketDataPayload),
    /// System command
    Command(SystemCommand),
}

/// Message priority levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MessagePriority {
    Low,
    Normal,
    High,
    Critical,
}

/// Agent signal data for coordination
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentSignalData {
    pub signal_strength: SignalStrength,
    pub confidence: f64,
    pub trading_action: TradingAction,
    pub quantum_features: QuantumFeatures,
    pub reasoning: Vec<String>,
}

/// Agent status information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentStatus {
    pub status: AgentState,
    pub health_score: f64,
    pub processing_load: f64,
    pub last_update: u64,
}

/// Agent state enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AgentState {
    Initializing,
    Active,
    Processing,
    Idle,
    Error,
    Shutdown,
}

/// Coordination data for agent collaboration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordinationData {
    pub coordination_type: CoordinationType,
    pub participants: Vec<String>,
    pub objective: String,
    pub parameters: HashMap<String, f64>,
}

/// Coordination type enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CoordinationType {
    /// Consensus building
    Consensus,
    /// Load balancing
    LoadBalancing,
    /// Resource sharing
    ResourceSharing,
    /// Joint decision making
    JointDecision,
    /// Error recovery
    ErrorRecovery,
}

/// Performance data for feedback
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceData {
    pub accuracy: f64,
    pub processing_time: u64,
    pub resource_usage: f64,
    pub error_rate: f64,
    pub improvement_suggestions: Vec<String>,
}

/// Market data payload
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketDataPayload {
    pub symbol: String,
    pub price: f64,
    pub volume: u64,
    pub timestamp: u64,
    pub indicators: HashMap<String, f64>,
}

/// System command payload
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemCommand {
    pub command: String,
    pub parameters: HashMap<String, String>,
    pub requires_response: bool,
}

/// Agent information for registry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentInfo {
    pub agent_type: String,
    pub capabilities: Vec<String>,
    pub status: AgentState,
    pub registered_at: u64,
    pub last_heartbeat: u64,
}

/// Coordination metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordinationMetrics {
    pub total_agents: u64,
    pub active_agents: u64,
    pub messages_sent: u64,
    pub messages_received: u64,
    pub coordination_sessions: u64,
    pub average_response_time: std::time::Duration,
}

impl Default for CoordinationMetrics {
    fn default() -> Self {
        Self {
            total_agents: 0,
            active_agents: 0,
            messages_sent: 0,
            messages_received: 0,
            coordination_sessions: 0,
            average_response_time: std::time::Duration::from_millis(0),
        }
    }
}

/// Quantum features for coordination
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumFeatures {
    pub coherence: f64,
    pub entanglement: f64,
    pub superposition_states: Vec<String>,
    pub phase_information: f64,
    pub quantum_advantage: Option<f64>,
}

impl Default for QuantumFeatures {
    fn default() -> Self {
        Self {
            coherence: 0.5,
            entanglement: 0.3,
            superposition_states: vec!["state1".to_string(), "state2".to_string()],
            phase_information: 0.0,
            quantum_advantage: None,
        }
    }
}

/// Helper functions for coordination
impl CoordinationMessage {
    /// Create new coordination message
    pub fn new(
        sender_id: String,
        message_type: MessageType,
        payload: MessagePayload,
    ) -> Self {
        Self {
            message_id: uuid::Uuid::new_v4().to_string(),
            sender_id,
            recipient_id: None,
            message_type,
            payload,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            priority: MessagePriority::Normal,
        }
    }
    
    /// Create targeted message
    pub fn new_targeted(
        sender_id: String,
        recipient_id: String,
        message_type: MessageType,
        payload: MessagePayload,
    ) -> Self {
        let mut message = Self::new(sender_id, message_type, payload);
        message.recipient_id = Some(recipient_id);
        message
    }
    
    /// Set message priority
    pub fn with_priority(mut self, priority: MessagePriority) -> Self {
        self.priority = priority;
        self
    }
}

/// Agent coordination helper
pub struct AgentCoordination {
    coordination_bus: Arc<CoordinationBus>,
    agent_id: String,
}

impl AgentCoordination {
    /// Create new agent coordination
    pub fn new(coordination_bus: Arc<CoordinationBus>, agent_id: String) -> Self {
        Self {
            coordination_bus,
            agent_id,
        }
    }
    
    /// Send signal to other agents
    pub async fn share_signal(&self, signal_data: AgentSignalData) -> PadsResult<()> {
        let message = CoordinationMessage::new(
            self.agent_id.clone(),
            MessageType::SignalShare,
            MessagePayload::Signal(signal_data),
        );
        
        self.coordination_bus.send_message(message).await
    }
    
    /// Update agent status
    pub async fn update_status(&self, status: AgentStatus) -> PadsResult<()> {
        let message = CoordinationMessage::new(
            self.agent_id.clone(),
            MessageType::StatusUpdate,
            MessagePayload::Status(status),
        );
        
        self.coordination_bus.send_message(message).await
    }
    
    /// Request coordination with other agents
    pub async fn request_coordination(&self, coordination_data: CoordinationData) -> PadsResult<()> {
        let message = CoordinationMessage::new(
            self.agent_id.clone(),
            MessageType::CoordinationRequest,
            MessagePayload::Coordination(coordination_data),
        );
        
        self.coordination_bus.send_message(message).await
    }
    
    /// Send performance feedback
    pub async fn send_performance_feedback(&self, performance_data: PerformanceData) -> PadsResult<()> {
        let message = CoordinationMessage::new(
            self.agent_id.clone(),
            MessageType::PerformanceFeedback,
            MessagePayload::Performance(performance_data),
        );
        
        self.coordination_bus.send_message(message).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_coordination_bus_creation() {
        let bus = CoordinationBus::new().await;
        assert!(bus.is_ok());
        
        let bus = bus.unwrap();
        let metrics = bus.get_metrics().await;
        assert_eq!(metrics.total_agents, 0);
        assert_eq!(metrics.messages_sent, 0);
    }
    
    #[tokio::test]
    async fn test_agent_registration() {
        let bus = CoordinationBus::new().await.unwrap();
        
        let agent_info = AgentInfo {
            agent_type: "test_agent".to_string(),
            capabilities: vec!["testing".to_string()],
            status: AgentState::Active,
            registered_at: 0,
            last_heartbeat: 0,
        };
        
        let result = bus.register_agent("agent_1".to_string(), agent_info).await;
        assert!(result.is_ok());
        
        let agents = bus.get_registered_agents().await;
        assert_eq!(agents.len(), 1);
        assert!(agents.contains(&"agent_1".to_string()));
    }
    
    #[tokio::test]
    async fn test_message_sending() {
        let bus = CoordinationBus::new().await.unwrap();
        
        let signal_data = AgentSignalData {
            signal_strength: SignalStrength::Medium,
            confidence: 0.8,
            trading_action: TradingAction::Buy,
            quantum_features: QuantumFeatures::default(),
            reasoning: vec!["Test reasoning".to_string()],
        };
        
        let message = CoordinationMessage::new(
            "test_agent".to_string(),
            MessageType::SignalShare,
            MessagePayload::Signal(signal_data),
        );
        
        let result = bus.send_message(message).await;
        assert!(result.is_ok());
        
        let metrics = bus.get_metrics().await;
        assert_eq!(metrics.messages_sent, 1);
    }
    
    #[test]
    fn test_coordination_message_creation() {
        let signal_data = AgentSignalData {
            signal_strength: SignalStrength::Strong,
            confidence: 0.9,
            trading_action: TradingAction::Sell,
            quantum_features: QuantumFeatures::default(),
            reasoning: vec!["Strong sell signal".to_string()],
        };
        
        let message = CoordinationMessage::new(
            "agent_1".to_string(),
            MessageType::SignalShare,
            MessagePayload::Signal(signal_data),
        );
        
        assert_eq!(message.sender_id, "agent_1");
        assert!(message.recipient_id.is_none());
        assert!(!message.message_id.is_empty());
        assert!(matches!(message.priority, MessagePriority::Normal));
    }
    
    #[test]
    fn test_targeted_message_creation() {
        let status = AgentStatus {
            status: AgentState::Active,
            health_score: 0.95,
            processing_load: 0.3,
            last_update: 12345,
        };
        
        let message = CoordinationMessage::new_targeted(
            "agent_1".to_string(),
            "agent_2".to_string(),
            MessageType::StatusUpdate,
            MessagePayload::Status(status),
        );
        
        assert_eq!(message.sender_id, "agent_1");
        assert_eq!(message.recipient_id, Some("agent_2".to_string()));
    }
    
    #[test]
    fn test_message_priority() {
        let command = SystemCommand {
            command: "shutdown".to_string(),
            parameters: HashMap::new(),
            requires_response: true,
        };
        
        let message = CoordinationMessage::new(
            "system".to_string(),
            MessageType::SystemCommand,
            MessagePayload::Command(command),
        ).with_priority(MessagePriority::Critical);
        
        assert!(matches!(message.priority, MessagePriority::Critical));
    }
}