//! Quantum Key Distribution Manager
//!
//! Orchestrates quantum key distribution operations across the ATS-CP trading system,
//! managing sessions, key generation, and distribution between agents.

use crate::error::QuantumSecurityError;
use crate::key_distribution::*;
use crate::types::*;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{RwLock, Mutex};
use uuid::Uuid;
use tracing::{info, warn, error, debug};

/// Quantum Key Distribution Manager
pub struct QuantumKeyDistributionManager {
    config: QKDConfig,
    nodes: Arc<RwLock<HashMap<String, QKDNode>>>,
    active_sessions: Arc<RwLock<HashMap<Uuid, QKDSession>>>,
    bb84_engine: Arc<BB84Engine>,
    e91_engine: Arc<E91Engine>,
    event_handlers: Arc<RwLock<Vec<Box<dyn QKDEventHandler + Send + Sync>>>>,
    metrics: Arc<RwLock<QKDManagerMetrics>>,
    session_counter: Arc<Mutex<u64>>,
}

/// QKD Manager Metrics
#[derive(Debug, Clone, Default)]
pub struct QKDManagerMetrics {
    pub total_sessions: u64,
    pub successful_sessions: u64,
    pub failed_sessions: u64,
    pub total_key_bits_generated: u64,
    pub average_session_duration_seconds: f64,
    pub average_key_generation_rate_bps: f64,
    pub network_efficiency: f64,
    pub security_violations: u64,
    pub node_failures: u64,
    pub eavesdropping_attempts: u64,
}

/// QKD Event Handler trait
pub trait QKDEventHandler {
    fn handle_event(&self, event: QKDEvent) -> Result<(), QuantumSecurityError>;
}

/// QKD Session Request
#[derive(Debug, Clone)]
pub struct QKDSessionRequest {
    pub alice_agent_id: String,
    pub bob_agent_id: String,
    pub protocol: QKDProtocol,
    pub key_length_bits: u32,
    pub priority: SessionPriority,
    pub security_requirements: QKDSecurityParameters,
    pub channel_preferences: Vec<QuantumChannelType>,
}

/// Session Priority
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum SessionPriority {
    Low,
    Normal,
    High,
    Critical,
}

impl QuantumKeyDistributionManager {
    /// Create a new QKD manager
    pub async fn new(config: QKDConfig) -> Result<Self, QuantumSecurityError> {
        let bb84_engine = Arc::new(BB84Engine::new(config.bb84_config.clone()));
        let e91_engine = Arc::new(E91Engine::new(config.e91_config.clone()));
        
        Ok(Self {
            config,
            nodes: Arc::new(RwLock::new(HashMap::new())),
            active_sessions: Arc::new(RwLock::new(HashMap::new())),
            bb84_engine,
            e91_engine,
            event_handlers: Arc::new(RwLock::new(Vec::new())),
            metrics: Arc::new(RwLock::new(QKDManagerMetrics::default())),
            session_counter: Arc::new(Mutex::new(0)),
        })
    }
    
    /// Register a QKD node
    pub async fn register_node(&self, node: QKDNode) -> Result<(), QuantumSecurityError> {
        let node_id = node.node_id.clone();
        let mut nodes = self.nodes.write().await;
        
        if nodes.contains_key(&node_id) {
            return Err(QuantumSecurityError::NodeAlreadyExists(node_id));
        }
        
        nodes.insert(node_id.clone(), node);
        
        self.emit_event(QKDEvent::NodeStatusChanged {
            node_id,
            status: QKDNodeStatus::Online,
        }).await?;
        
        Ok(())
    }
    
    /// Unregister a QKD node
    pub async fn unregister_node(&self, node_id: &str) -> Result<(), QuantumSecurityError> {
        let mut nodes = self.nodes.write().await;
        
        if let Some(mut node) = nodes.remove(node_id) {
            node.set_status(QKDNodeStatus::Offline);
            
            // Abort active sessions for this node
            let active_sessions = node.active_sessions.clone();
            drop(nodes);
            
            for session_id in active_sessions {
                self.abort_session(session_id).await?;
            }
            
            self.emit_event(QKDEvent::NodeStatusChanged {
                node_id: node_id.to_string(),
                status: QKDNodeStatus::Offline,
            }).await?;
        }
        
        Ok(())
    }
    
    /// Establish quantum keys between two agents
    pub async fn establish_keys(
        &self,
        alice_agent_id: &str,
        bob_agent_id: &str,
    ) -> Result<QKDKeyMaterial, QuantumSecurityError> {
        let request = QKDSessionRequest {
            alice_agent_id: alice_agent_id.to_string(),
            bob_agent_id: bob_agent_id.to_string(),
            protocol: self.config.default_protocol.clone(),
            key_length_bits: 2048, // Default key length
            priority: SessionPriority::Normal,
            security_requirements: self.config.security_parameters.clone(),
            channel_preferences: vec![
                QuantumChannelType::Simulated {
                    error_rate: 0.05,
                    eavesdropping_probability: 0.0,
                }
            ],
        };
        
        self.start_qkd_session(request).await
    }
    
    /// Start a QKD session
    pub async fn start_qkd_session(
        &self,
        request: QKDSessionRequest,
    ) -> Result<QKDKeyMaterial, QuantumSecurityError> {
        // Validate nodes exist and are online
        let (alice_node, bob_node) = self.validate_session_nodes(&request).await?;
        
        // Select optimal channel
        let channel = self.select_optimal_channel(&request, &alice_node, &bob_node).await?;
        
        // Create session
        let mut session = QKDSession::new(
            request.alice_agent_id.clone(),
            request.bob_agent_id.clone(),
            request.protocol.clone(),
            channel,
            request.key_length_bits,
        );
        
        session.security_parameters = request.security_requirements;
        
        // Start session
        session.start().map_err(|e| QuantumSecurityError::QKDError(e))?;
        
        let session_id = session.session_id;
        
        // Store session
        {
            let mut sessions = self.active_sessions.write().await;
            sessions.insert(session_id, session.clone());
        }
        
        // Update node sessions
        {
            let mut nodes = self.nodes.write().await;
            if let Some(alice) = nodes.get_mut(&alice_node.node_id) {
                alice.add_session(session_id);
            }
            if let Some(bob) = nodes.get_mut(&bob_node.node_id) {
                bob.add_session(session_id);
            }
        }
        
        self.emit_event(QKDEvent::SessionStarted {
            session_id,
            protocol: request.protocol.clone(),
        }).await?;
        
        // Execute QKD protocol
        let key_material = self.execute_qkd_protocol(&mut session, &alice_node, &bob_node).await?;
        
        // Complete session
        {
            let mut sessions = self.active_sessions.write().await;
            if let Some(session) = sessions.get_mut(&session_id) {
                session.complete(key_material.final_key.expose().len() as u32 * 8)
                    .map_err(|e| QuantumSecurityError::QKDError(e))?;
            }
        }
        
        // Clean up
        self.cleanup_session(session_id).await?;
        
        self.emit_event(QKDEvent::SessionCompleted {
            session_id,
            key_bits_generated: key_material.final_key.expose().len() as u32 * 8,
        }).await?;
        
        // Update metrics
        self.update_session_metrics(&key_material).await;
        
        Ok(key_material)
    }
    
    /// Execute QKD protocol based on session configuration
    async fn execute_qkd_protocol(
        &self,
        session: &mut QKDSession,
        alice_node: &QKDNode,
        bob_node: &QKDNode,
    ) -> Result<QKDKeyMaterial, QuantumSecurityError> {
        match session.protocol {
            QKDProtocol::BB84 => {
                self.execute_bb84_protocol(session, alice_node, bob_node).await
            }
            QKDProtocol::E91 => {
                self.execute_e91_protocol(session, alice_node, bob_node).await
            }
            _ => Err(QuantumSecurityError::UnsupportedProtocol(
                format!("Protocol {:?} not implemented", session.protocol)
            ))
        }
    }
    
    /// Execute BB84 protocol
    async fn execute_bb84_protocol(
        &self,
        session: &mut QKDSession,
        alice_node: &QKDNode,
        bob_node: &QKDNode,
    ) -> Result<QKDKeyMaterial, QuantumSecurityError> {
        session.status = QKDSessionStatus::QuantumTransmission;
        
        // Phase 1: Quantum transmission
        let (alice_bits, alice_bases, bob_measurements) = self.bb84_engine
            .quantum_transmission_phase(session.key_length_bits * 4).await?;
        
        session.status = QKDSessionStatus::SiftingKeys;
        
        // Phase 2: Key sifting
        let (sifted_key_alice, sifted_key_bob) = self.bb84_engine
            .key_sifting_phase(&alice_bits, &alice_bases, &bob_measurements).await?;
        
        session.status = QKDSessionStatus::ErrorCorrection;
        
        // Phase 3: Error estimation and correction
        let error_rate = self.bb84_engine
            .estimate_error_rate(&sifted_key_alice, &sifted_key_bob).await?;
        
        session.error_rate = error_rate;
        
        if error_rate > session.security_parameters.max_error_rate {
            let error = QKDError::HighErrorRate {
                measured: error_rate,
                threshold: session.security_parameters.max_error_rate,
            };
            
            session.fail(error.clone());
            
            self.emit_event(QKDEvent::HighErrorRateDetected {
                session_id: session.session_id,
                qber: error_rate,
            }).await?;
            
            return Err(QuantumSecurityError::QKDError(error));
        }
        
        let corrected_key = self.bb84_engine
            .error_correction(&sifted_key_alice, &sifted_key_bob, error_rate).await?;
        
        session.status = QKDSessionStatus::PrivacyAmplification;
        
        // Phase 4: Privacy amplification
        let final_key = self.bb84_engine
            .privacy_amplification(&corrected_key, error_rate).await?;
        
        // Create key material
        let key_material = QKDKeyMaterial::new(
            session.session_id,
            final_key,
            format!("bb84_key_{}_{}", alice_node.agent_id, bob_node.agent_id),
            alice_node.agent_id.clone(),
            bob_node.agent_id.clone(),
            QKDProtocol::BB84,
            1.0 - error_rate,
        );
        
        Ok(key_material)
    }
    
    /// Execute E91 protocol
    async fn execute_e91_protocol(
        &self,
        session: &mut QKDSession,
        alice_node: &QKDNode,
        bob_node: &QKDNode,
    ) -> Result<QKDKeyMaterial, QuantumSecurityError> {
        session.status = QKDSessionStatus::QuantumTransmission;
        
        // Phase 1: Entanglement distribution and measurement
        let (alice_measurements, bob_measurements, bell_test_data) = self.e91_engine
            .entanglement_distribution_phase(session.key_length_bits * 2).await?;
        
        session.status = QKDSessionStatus::SiftingKeys;
        
        // Phase 2: Bell inequality test for security
        let bell_violation = self.e91_engine
            .bell_inequality_test(&bell_test_data).await?;
        
        if bell_violation < 2.0 {
            let error = QKDError::SecurityBreach(
                format!("Bell inequality violation too low: {}", bell_violation)
            );
            session.fail(error.clone());
            return Err(QuantumSecurityError::QKDError(error));
        }
        
        // Phase 3: Key sifting
        let (sifted_key_alice, sifted_key_bob) = self.e91_engine
            .key_sifting_phase(&alice_measurements, &bob_measurements).await?;
        
        session.status = QKDSessionStatus::ErrorCorrection;
        
        // Phase 4: Error estimation and correction
        let error_rate = self.e91_engine
            .estimate_error_rate(&sifted_key_alice, &sifted_key_bob).await?;
        
        session.error_rate = error_rate;
        
        if error_rate > session.security_parameters.max_error_rate {
            let error = QKDError::HighErrorRate {
                measured: error_rate,
                threshold: session.security_parameters.max_error_rate,
            };
            session.fail(error.clone());
            return Err(QuantumSecurityError::QKDError(error));
        }
        
        let corrected_key = self.e91_engine
            .error_correction(&sifted_key_alice, &sifted_key_bob, error_rate).await?;
        
        session.status = QKDSessionStatus::PrivacyAmplification;
        
        // Phase 5: Privacy amplification
        let final_key = self.e91_engine
            .privacy_amplification(&corrected_key, error_rate).await?;
        
        // Create key material
        let key_material = QKDKeyMaterial::new(
            session.session_id,
            final_key,
            format!("e91_key_{}_{}", alice_node.agent_id, bob_node.agent_id),
            alice_node.agent_id.clone(),
            bob_node.agent_id.clone(),
            QKDProtocol::E91,
            1.0 - error_rate,
        );
        
        Ok(key_material)
    }
    
    /// Validate session nodes
    async fn validate_session_nodes(
        &self,
        request: &QKDSessionRequest,
    ) -> Result<(QKDNode, QKDNode), QuantumSecurityError> {
        let nodes = self.nodes.read().await;
        
        let alice_node = nodes.values()
            .find(|node| node.agent_id == request.alice_agent_id && node.status == QKDNodeStatus::Online)
            .ok_or_else(|| QuantumSecurityError::NodeNotFound(request.alice_agent_id.clone()))?
            .clone();
        
        let bob_node = nodes.values()
            .find(|node| node.agent_id == request.bob_agent_id && node.status == QKDNodeStatus::Online)
            .ok_or_else(|| QuantumSecurityError::NodeNotFound(request.bob_agent_id.clone()))?
            .clone();
        
        // Check protocol support
        if !alice_node.supports_protocol(&request.protocol) {
            return Err(QuantumSecurityError::UnsupportedProtocol(
                format!("Alice node doesn't support {:?}", request.protocol)
            ));
        }
        
        if !bob_node.supports_protocol(&request.protocol) {
            return Err(QuantumSecurityError::UnsupportedProtocol(
                format!("Bob node doesn't support {:?}", request.protocol)
            ));
        }
        
        Ok((alice_node, bob_node))
    }
    
    /// Select optimal quantum channel
    async fn select_optimal_channel(
        &self,
        request: &QKDSessionRequest,
        alice_node: &QKDNode,
        bob_node: &QKDNode,
    ) -> Result<QuantumChannelType, QuantumSecurityError> {
        // For now, use the first preference or simulated channel
        if let Some(channel) = request.channel_preferences.first() {
            Ok(channel.clone())
        } else {
            Ok(QuantumChannelType::Simulated {
                error_rate: 0.05,
                eavesdropping_probability: 0.0,
            })
        }
    }
    
    /// Abort a QKD session
    pub async fn abort_session(&self, session_id: Uuid) -> Result<(), QuantumSecurityError> {
        let mut sessions = self.active_sessions.write().await;
        
        if let Some(mut session) = sessions.remove(&session_id) {
            session.status = QKDSessionStatus::Aborted;
            
            // Remove from node sessions
            let mut nodes = self.nodes.write().await;
            for node in nodes.values_mut() {
                node.remove_session(session_id);
            }
            
            info!("QKD session {} aborted", session_id);
        }
        
        Ok(())
    }
    
    /// Clean up completed session
    async fn cleanup_session(&self, session_id: Uuid) -> Result<(), QuantumSecurityError> {
        let mut sessions = self.active_sessions.write().await;
        sessions.remove(&session_id);
        
        // Remove from node sessions
        let mut nodes = self.nodes.write().await;
        for node in nodes.values_mut() {
            node.remove_session(session_id);
        }
        
        Ok(())
    }
    
    /// Add event handler
    pub async fn add_event_handler(&self, handler: Box<dyn QKDEventHandler + Send + Sync>) {
        let mut handlers = self.event_handlers.write().await;
        handlers.push(handler);
    }
    
    /// Emit QKD event
    async fn emit_event(&self, event: QKDEvent) -> Result<(), QuantumSecurityError> {
        let handlers = self.event_handlers.read().await;
        
        for handler in handlers.iter() {
            if let Err(e) = handler.handle_event(event.clone()) {
                warn!("Event handler error: {:?}", e);
            }
        }
        
        Ok(())
    }
    
    /// Update session metrics
    async fn update_session_metrics(&self, key_material: &QKDKeyMaterial) {
        let mut metrics = self.metrics.write().await;
        let mut session_counter = self.session_counter.lock().await;
        
        *session_counter += 1;
        metrics.total_sessions += 1;
        metrics.successful_sessions += 1;
        metrics.total_key_bits_generated += key_material.final_key.expose().len() as u64 * 8;
        
        // Calculate averages
        metrics.average_key_generation_rate_bps = 
            metrics.total_key_bits_generated as f64 / metrics.total_sessions as f64;
    }
    
    /// Get session status
    pub async fn get_session_status(&self, session_id: Uuid) -> Option<QKDSession> {
        let sessions = self.active_sessions.read().await;
        sessions.get(&session_id).cloned()
    }
    
    /// Get active sessions
    pub async fn get_active_sessions(&self) -> Vec<QKDSession> {
        let sessions = self.active_sessions.read().await;
        sessions.values().cloned().collect()
    }
    
    /// Get node status
    pub async fn get_node_status(&self, node_id: &str) -> Option<QKDNode> {
        let nodes = self.nodes.read().await;
        nodes.get(node_id).cloned()
    }
    
    /// Get all nodes
    pub async fn get_all_nodes(&self) -> Vec<QKDNode> {
        let nodes = self.nodes.read().await;
        nodes.values().cloned().collect()
    }
    
    /// Get manager metrics
    pub async fn get_metrics(&self) -> QKDManagerMetrics {
        self.metrics.read().await.clone()
    }
    
    /// Health check
    pub async fn health_check(&self) -> Result<QKDManagerHealth, QuantumSecurityError> {
        let nodes = self.nodes.read().await;
        let sessions = self.active_sessions.read().await;
        let metrics = self.metrics.read().await;
        
        let online_nodes = nodes.values().filter(|n| n.status == QKDNodeStatus::Online).count();
        let total_nodes = nodes.len();
        let active_sessions = sessions.len();
        
        let health = QKDManagerHealth {
            healthy: online_nodes > 0 && metrics.security_violations == 0,
            total_nodes,
            online_nodes,
            active_sessions,
            total_sessions: metrics.total_sessions,
            success_rate: if metrics.total_sessions > 0 {
                metrics.successful_sessions as f64 / metrics.total_sessions as f64
            } else {
                0.0
            },
            average_key_rate_bps: metrics.average_key_generation_rate_bps,
            security_violations: metrics.security_violations,
        };
        
        Ok(health)
    }
    
    /// Periodic maintenance
    pub async fn periodic_maintenance(&self) -> Result<(), QuantumSecurityError> {
        // Clean up expired keys
        self.cleanup_expired_keys().await?;
        
        // Check node health
        self.check_node_health().await?;
        
        // Update network efficiency metrics
        self.update_network_metrics().await?;
        
        Ok(())
    }
    
    /// Clean up expired keys
    async fn cleanup_expired_keys(&self) -> Result<(), QuantumSecurityError> {
        let mut nodes = self.nodes.write().await;
        let mut total_cleaned = 0;
        
        for node in nodes.values_mut() {
            for buffer in node.key_buffer.values_mut() {
                let initial_count = buffer.available_keys.len();
                buffer.available_keys.retain(|key| !key.is_expired());
                let cleaned_count = initial_count - buffer.available_keys.len();
                total_cleaned += cleaned_count;
                
                // Update buffer size
                buffer.buffer_size_bytes = buffer.available_keys.iter()
                    .map(|key| key.final_key.expose().len())
                    .sum();
            }
        }
        
        if total_cleaned > 0 {
            info!("Cleaned up {} expired QKD keys", total_cleaned);
        }
        
        Ok(())
    }
    
    /// Check node health
    async fn check_node_health(&self) -> Result<(), QuantumSecurityError> {
        // Implementation would check node connectivity, performance, etc.
        Ok(())
    }
    
    /// Update network metrics
    async fn update_network_metrics(&self) -> Result<(), QuantumSecurityError> {
        // Implementation would calculate network-wide efficiency metrics
        Ok(())
    }
}

/// QKD Manager Health Status
#[derive(Debug, Clone)]
pub struct QKDManagerHealth {
    pub healthy: bool,
    pub total_nodes: usize,
    pub online_nodes: usize,
    pub active_sessions: usize,
    pub total_sessions: u64,
    pub success_rate: f64,
    pub average_key_rate_bps: f64,
    pub security_violations: u64,
}

/// Simple event handler for logging
pub struct LoggingEventHandler;

impl QKDEventHandler for LoggingEventHandler {
    fn handle_event(&self, event: QKDEvent) -> Result<(), QuantumSecurityError> {
        match event {
            QKDEvent::SessionStarted { session_id, protocol } => {
                info!("QKD session started: {} using {:?}", session_id, protocol);
            }
            QKDEvent::SessionCompleted { session_id, key_bits_generated } => {
                info!("QKD session completed: {} generated {} bits", session_id, key_bits_generated);
            }
            QKDEvent::SessionFailed { session_id, error } => {
                error!("QKD session failed: {} - {:?}", session_id, error);
            }
            QKDEvent::HighErrorRateDetected { session_id, qber } => {
                warn!("High error rate detected in session {}: QBER = {:.3}", session_id, qber);
            }
            QKDEvent::EavesdroppingAttemptDetected { session_id, evidence } => {
                error!("Eavesdropping attempt detected in session {}: {}", session_id, evidence);
            }
            QKDEvent::KeyBufferLow { node_id, remaining_keys } => {
                warn!("Key buffer low for node {}: {} keys remaining", node_id, remaining_keys);
            }
            QKDEvent::NodeStatusChanged { node_id, status } => {
                info!("Node {} status changed to {:?}", node_id, status);
            }
            QKDEvent::QuantumChannelCalibrated { node_id, efficiency } => {
                info!("Quantum channel calibrated for node {}: efficiency = {:.3}", node_id, efficiency);
            }
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio_test;
    
    #[tokio::test]
    async fn test_qkd_manager_creation() {
        let config = QKDConfig::default();
        let manager = QuantumKeyDistributionManager::new(config).await;
        assert!(manager.is_ok());
    }
    
    #[tokio::test]
    async fn test_node_registration() {
        let config = QKDConfig::default();
        let manager = QuantumKeyDistributionManager::new(config).await.unwrap();
        
        let capabilities = QuantumCapabilities {
            supported_protocols: vec![QKDProtocol::BB84],
            supported_channels: vec![QuantumChannelType::Simulated { error_rate: 0.05, eavesdropping_probability: 0.0 }],
            max_key_generation_rate_bps: 1000.0,
            min_detection_efficiency: 0.8,
            quantum_memory_coherence_time_ms: 1.0,
            entanglement_generation_rate_hz: 100.0,
        };
        
        let node = QKDNode::new(
            "node1".to_string(),
            "agent1".to_string(),
            capabilities,
            "127.0.0.1:8080".to_string(),
        );
        
        assert!(manager.register_node(node).await.is_ok());
        
        let nodes = manager.get_all_nodes().await;
        assert_eq!(nodes.len(), 1);
        assert_eq!(nodes[0].node_id, "node1");
    }
    
    #[tokio::test]
    async fn test_health_check() {
        let config = QKDConfig::default();
        let manager = QuantumKeyDistributionManager::new(config).await.unwrap();
        
        let health = manager.health_check().await.unwrap();
        assert_eq!(health.total_nodes, 0);
        assert_eq!(health.online_nodes, 0);
        assert_eq!(health.active_sessions, 0);
    }
}