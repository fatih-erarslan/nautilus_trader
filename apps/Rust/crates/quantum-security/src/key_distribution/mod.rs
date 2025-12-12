//! Quantum Key Distribution (QKD) System
//!
//! This module implements quantum key distribution protocols for secure
//! communication between agents in the ATS-CP trading system.

pub mod bb84;
pub mod e91;
pub mod manager;
pub mod protocols;

pub use bb84::*;
pub use e91::*;
pub use manager::*;
pub use protocols::*;

use crate::error::QuantumSecurityError;
use crate::types::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

/// Quantum Key Distribution Protocol Types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum QKDProtocol {
    /// BB84 Protocol (Bennett-Brassard 1984)
    BB84,
    /// E91 Protocol (Ekert 1991) - Entanglement-based
    E91,
    /// SARG04 Protocol (Scarani-Acin-Ribordy-Gisin 2004)
    SARG04,
    /// BBM92 Protocol (Bennett-Brassard-Mermin 1992)
    BBM92,
    /// Continuous Variable QKD
    CVQKD,
}

/// Quantum Channel Types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuantumChannelType {
    /// Fiber optic channel
    FiberOptic {
        wavelength: f64,
        attenuation_db_per_km: f64,
        distance_km: f64,
    },
    /// Free space optical channel
    FreeSpace {
        wavelength: f64,
        atmospheric_loss_db: f64,
        beam_divergence_mrad: f64,
    },
    /// Simulated quantum channel (for testing)
    Simulated {
        error_rate: f64,
        eavesdropping_probability: f64,
    },
}

/// Quantum Key Distribution Session
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QKDSession {
    pub session_id: Uuid,
    pub alice_agent_id: String,
    pub bob_agent_id: String,
    pub protocol: QKDProtocol,
    pub channel: QuantumChannelType,
    pub status: QKDSessionStatus,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub started_at: Option<chrono::DateTime<chrono::Utc>>,
    pub completed_at: Option<chrono::DateTime<chrono::Utc>>,
    pub key_length_bits: u32,
    pub generated_key_bits: u32,
    pub error_rate: f64,
    pub security_parameters: QKDSecurityParameters,
    pub performance_metrics: QKDPerformanceMetrics,
}

/// QKD Session Status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum QKDSessionStatus {
    Initializing,
    Negotiating,
    QuantumTransmission,
    SiftingKeys,
    ErrorCorrection,
    PrivacyAmplification,
    Completed,
    Failed(String),
    Aborted,
}

/// QKD Security Parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QKDSecurityParameters {
    pub max_error_rate: f64,
    pub min_key_rate_bps: f64,
    pub security_parameter_epsilon: f64,
    pub hash_function: String,
    pub error_correction_code: String,
    pub privacy_amplification_method: String,
}

impl Default for QKDSecurityParameters {
    fn default() -> Self {
        Self {
            max_error_rate: 0.11, // 11% QBER threshold
            min_key_rate_bps: 1000.0,
            security_parameter_epsilon: 1e-9,
            hash_function: "SHA3-256".to_string(),
            error_correction_code: "LDPC".to_string(),
            privacy_amplification_method: "Universal_Hash".to_string(),
        }
    }
}

/// QKD Performance Metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QKDPerformanceMetrics {
    pub raw_key_generation_rate_bps: f64,
    pub sifted_key_generation_rate_bps: f64,
    pub final_key_generation_rate_bps: f64,
    pub quantum_bit_error_rate: f64,
    pub channel_efficiency: f64,
    pub total_duration_seconds: f64,
    pub quantum_transmission_time_seconds: f64,
    pub classical_processing_time_seconds: f64,
    pub photon_count: u64,
    pub detection_count: u64,
    pub coincidence_count: u64,
}

/// Quantum State
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuantumState {
    /// Polarization states for photonic QKD
    HorizontalPolarization,
    VerticalPolarization,
    DiagonalPolarization,
    AntiDiagonalPolarization,
    /// Circular polarization states
    LeftCircular,
    RightCircular,
    /// Phase states
    Phase0,
    Phase90,
    Phase180,
    Phase270,
    /// Entangled states
    BellState(BellState),
}

/// Bell States for entanglement-based QKD
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BellState {
    PhiPlus,   // |Φ+⟩ = (|00⟩ + |11⟩)/√2
    PhiMinus,  // |Φ-⟩ = (|00⟩ - |11⟩)/√2
    PsiPlus,   // |Ψ+⟩ = (|01⟩ + |10⟩)/√2
    PsiMinus,  // |Ψ-⟩ = (|01⟩ - |10⟩)/√2
}

/// Measurement Basis
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum MeasurementBasis {
    Rectilinear, // H/V basis
    Diagonal,    // +/- basis
    Circular,    // L/R basis
    Bell,        // Bell state measurement
}

/// Quantum Measurement Result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumMeasurement {
    pub basis: MeasurementBasis,
    pub result: bool, // 0 or 1
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub detection_efficiency: f64,
    pub noise_level: f64,
}

/// QKD Key Material
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QKDKeyMaterial {
    pub session_id: Uuid,
    pub raw_key: SecureBytes,
    pub sifted_key: SecureBytes,
    pub final_key: SecureBytes,
    pub key_id: String,
    pub alice_agent_id: String,
    pub bob_agent_id: String,
    pub protocol: QKDProtocol,
    pub generation_time: chrono::DateTime<chrono::Utc>,
    pub expiry_time: chrono::DateTime<chrono::Utc>,
    pub security_level: f64,
}

/// QKD Network Node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QKDNode {
    pub node_id: String,
    pub agent_id: String,
    pub quantum_capabilities: QuantumCapabilities,
    pub network_address: String,
    pub status: QKDNodeStatus,
    pub active_sessions: Vec<Uuid>,
    pub key_buffer: HashMap<String, QKDKeyBuffer>,
}

/// Quantum Capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumCapabilities {
    pub supported_protocols: Vec<QKDProtocol>,
    pub supported_channels: Vec<QuantumChannelType>,
    pub max_key_generation_rate_bps: f64,
    pub min_detection_efficiency: f64,
    pub quantum_memory_coherence_time_ms: f64,
    pub entanglement_generation_rate_hz: f64,
}

/// QKD Node Status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum QKDNodeStatus {
    Online,
    Offline,
    Calibrating,
    Maintenance,
    Error(String),
}

/// QKD Key Buffer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QKDKeyBuffer {
    pub target_agent_id: String,
    pub available_keys: Vec<QKDKeyMaterial>,
    pub buffer_size_bytes: usize,
    pub max_buffer_size_bytes: usize,
    pub last_key_generation: chrono::DateTime<chrono::Utc>,
    pub key_consumption_rate_bps: f64,
}

/// QKD Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QKDConfig {
    pub enabled_protocols: Vec<QKDProtocol>,
    pub default_protocol: QKDProtocol,
    pub security_parameters: QKDSecurityParameters,
    pub network_topology: QKDNetworkTopology,
    pub key_refresh_interval: chrono::Duration,
    pub max_key_buffer_size_mb: usize,
    pub performance_monitoring: bool,
    pub classical_channel_encryption: bool,
}

impl Default for QKDConfig {
    fn default() -> Self {
        Self {
            enabled_protocols: vec![QKDProtocol::BB84, QKDProtocol::E91],
            default_protocol: QKDProtocol::BB84,
            security_parameters: QKDSecurityParameters::default(),
            network_topology: QKDNetworkTopology::Mesh,
            key_refresh_interval: chrono::Duration::hours(1),
            max_key_buffer_size_mb: 100,
            performance_monitoring: true,
            classical_channel_encryption: true,
        }
    }
}

/// QKD Network Topology
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QKDNetworkTopology {
    PointToPoint,
    Star,
    Ring,
    Mesh,
    Hybrid,
}

/// QKD Error Types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QKDError {
    HighErrorRate { measured: f64, threshold: f64 },
    ChannelFailure(String),
    ProtocolViolation(String),
    AuthenticationFailure,
    KeyExhaustion,
    QuantumChannelNoise { snr_db: f64 },
    ClassicalChannelError(String),
    NodeUnavailable(String),
    SecurityBreach(String),
}

/// QKD Event Types for monitoring and logging
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QKDEvent {
    SessionStarted { session_id: Uuid, protocol: QKDProtocol },
    SessionCompleted { session_id: Uuid, key_bits_generated: u32 },
    SessionFailed { session_id: Uuid, error: QKDError },
    HighErrorRateDetected { session_id: Uuid, qber: f64 },
    EavesdroppingAttemptDetected { session_id: Uuid, evidence: String },
    KeyBufferLow { node_id: String, remaining_keys: usize },
    NodeStatusChanged { node_id: String, status: QKDNodeStatus },
    QuantumChannelCalibrated { node_id: String, efficiency: f64 },
}

impl QKDSession {
    /// Create a new QKD session
    pub fn new(
        alice_agent_id: String,
        bob_agent_id: String,
        protocol: QKDProtocol,
        channel: QuantumChannelType,
        key_length_bits: u32,
    ) -> Self {
        Self {
            session_id: Uuid::new_v4(),
            alice_agent_id,
            bob_agent_id,
            protocol,
            channel,
            status: QKDSessionStatus::Initializing,
            created_at: chrono::Utc::now(),
            started_at: None,
            completed_at: None,
            key_length_bits,
            generated_key_bits: 0,
            error_rate: 0.0,
            security_parameters: QKDSecurityParameters::default(),
            performance_metrics: QKDPerformanceMetrics::default(),
        }
    }
    
    /// Start the session
    pub fn start(&mut self) -> Result<(), QKDError> {
        if self.status != QKDSessionStatus::Initializing {
            return Err(QKDError::ProtocolViolation(
                format!("Cannot start session in state {:?}", self.status)
            ));
        }
        
        self.status = QKDSessionStatus::Negotiating;
        self.started_at = Some(chrono::Utc::now());
        Ok(())
    }
    
    /// Complete the session
    pub fn complete(&mut self, generated_bits: u32) -> Result<(), QKDError> {
        self.status = QKDSessionStatus::Completed;
        self.completed_at = Some(chrono::Utc::now());
        self.generated_key_bits = generated_bits;
        
        if let Some(started_at) = self.started_at {
            self.performance_metrics.total_duration_seconds = 
                chrono::Utc::now().signed_duration_since(started_at).num_milliseconds() as f64 / 1000.0;
        }
        
        Ok(())
    }
    
    /// Fail the session
    pub fn fail(&mut self, error: QKDError) {
        self.status = QKDSessionStatus::Failed(format!("{:?}", error));
        self.completed_at = Some(chrono::Utc::now());
    }
    
    /// Check if session is active
    pub fn is_active(&self) -> bool {
        matches!(self.status, 
            QKDSessionStatus::Negotiating |
            QKDSessionStatus::QuantumTransmission |
            QKDSessionStatus::SiftingKeys |
            QKDSessionStatus::ErrorCorrection |
            QKDSessionStatus::PrivacyAmplification
        )
    }
    
    /// Get session duration
    pub fn duration(&self) -> Option<chrono::Duration> {
        if let (Some(started), Some(completed)) = (self.started_at, self.completed_at) {
            Some(completed.signed_duration_since(started))
        } else if let Some(started) = self.started_at {
            Some(chrono::Utc::now().signed_duration_since(started))
        } else {
            None
        }
    }
}

impl QKDKeyMaterial {
    /// Create new QKD key material
    pub fn new(
        session_id: Uuid,
        final_key: Vec<u8>,
        key_id: String,
        alice_agent_id: String,
        bob_agent_id: String,
        protocol: QKDProtocol,
        security_level: f64,
    ) -> Self {
        let now = chrono::Utc::now();
        
        Self {
            session_id,
            raw_key: SecureBytes::new(vec![]),
            sifted_key: SecureBytes::new(vec![]),
            final_key: SecureBytes::new(final_key),
            key_id,
            alice_agent_id,
            bob_agent_id,
            protocol,
            generation_time: now,
            expiry_time: now + chrono::Duration::hours(24), // Default 24h expiry
            security_level,
        }
    }
    
    /// Check if key is expired
    pub fn is_expired(&self) -> bool {
        chrono::Utc::now() > self.expiry_time
    }
    
    /// Get key age in seconds
    pub fn age_seconds(&self) -> i64 {
        chrono::Utc::now().signed_duration_since(self.generation_time).num_seconds()
    }
    
    /// Get remaining validity time
    pub fn remaining_validity(&self) -> chrono::Duration {
        self.expiry_time.signed_duration_since(chrono::Utc::now())
    }
    
    /// Extract key bytes for use
    pub fn extract_key_bytes(&self, length: usize) -> Option<Vec<u8>> {
        let key_data = self.final_key.expose();
        if key_data.len() >= length {
            Some(key_data[..length].to_vec())
        } else {
            None
        }
    }
}

impl QKDNode {
    /// Create new QKD node
    pub fn new(
        node_id: String,
        agent_id: String,
        quantum_capabilities: QuantumCapabilities,
        network_address: String,
    ) -> Self {
        Self {
            node_id,
            agent_id,
            quantum_capabilities,
            network_address,
            status: QKDNodeStatus::Offline,
            active_sessions: Vec::new(),
            key_buffer: HashMap::new(),
        }
    }
    
    /// Set node status
    pub fn set_status(&mut self, status: QKDNodeStatus) {
        self.status = status;
    }
    
    /// Add active session
    pub fn add_session(&mut self, session_id: Uuid) {
        if !self.active_sessions.contains(&session_id) {
            self.active_sessions.push(session_id);
        }
    }
    
    /// Remove session
    pub fn remove_session(&mut self, session_id: Uuid) {
        self.active_sessions.retain(|&id| id != session_id);
    }
    
    /// Check if node supports protocol
    pub fn supports_protocol(&self, protocol: &QKDProtocol) -> bool {
        self.quantum_capabilities.supported_protocols.contains(protocol)
    }
    
    /// Get key buffer for target agent
    pub fn get_key_buffer(&self, target_agent_id: &str) -> Option<&QKDKeyBuffer> {
        self.key_buffer.get(target_agent_id)
    }
    
    /// Add key to buffer
    pub fn add_key_to_buffer(&mut self, target_agent_id: String, key: QKDKeyMaterial) {
        let buffer = self.key_buffer.entry(target_agent_id.clone()).or_insert_with(|| {
            QKDKeyBuffer {
                target_agent_id,
                available_keys: Vec::new(),
                buffer_size_bytes: 0,
                max_buffer_size_bytes: 1024 * 1024, // 1MB default
                last_key_generation: chrono::Utc::now(),
                key_consumption_rate_bps: 0.0,
            }
        });
        
        buffer.buffer_size_bytes += key.final_key.expose().len();
        buffer.available_keys.push(key);
        buffer.last_key_generation = chrono::Utc::now();
    }
    
    /// Consume key from buffer
    pub fn consume_key(&mut self, target_agent_id: &str, key_length: usize) -> Option<Vec<u8>> {
        if let Some(buffer) = self.key_buffer.get_mut(target_agent_id) {
            // Remove expired keys first
            buffer.available_keys.retain(|key| !key.is_expired());
            
            // Find a key with sufficient length
            for (i, key) in buffer.available_keys.iter().enumerate() {
                if key.final_key.expose().len() >= key_length {
                    let consumed_key = buffer.available_keys.remove(i);
                    buffer.buffer_size_bytes -= consumed_key.final_key.expose().len();
                    return consumed_key.extract_key_bytes(key_length);
                }
            }
        }
        
        None
    }
    
    /// Get buffer status
    pub fn get_buffer_status(&self, target_agent_id: &str) -> Option<(usize, usize, usize)> {
        if let Some(buffer) = self.key_buffer.get(target_agent_id) {
            let available_keys = buffer.available_keys.len();
            let total_bytes = buffer.buffer_size_bytes;
            let max_bytes = buffer.max_buffer_size_bytes;
            Some((available_keys, total_bytes, max_bytes))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_qkd_session_creation() {
        let session = QKDSession::new(
            "alice".to_string(),
            "bob".to_string(),
            QKDProtocol::BB84,
            QuantumChannelType::Simulated { error_rate: 0.05, eavesdropping_probability: 0.0 },
            1024,
        );
        
        assert_eq!(session.alice_agent_id, "alice");
        assert_eq!(session.bob_agent_id, "bob");
        assert_eq!(session.protocol, QKDProtocol::BB84);
        assert_eq!(session.key_length_bits, 1024);
        assert_eq!(session.status, QKDSessionStatus::Initializing);
    }
    
    #[test]
    fn test_qkd_session_lifecycle() {
        let mut session = QKDSession::new(
            "alice".to_string(),
            "bob".to_string(),
            QKDProtocol::BB84,
            QuantumChannelType::Simulated { error_rate: 0.05, eavesdropping_probability: 0.0 },
            1024,
        );
        
        assert!(session.start().is_ok());
        assert_eq!(session.status, QKDSessionStatus::Negotiating);
        assert!(session.is_active());
        
        assert!(session.complete(1024).is_ok());
        assert_eq!(session.status, QKDSessionStatus::Completed);
        assert!(!session.is_active());
        assert_eq!(session.generated_key_bits, 1024);
    }
    
    #[test]
    fn test_qkd_key_material() {
        let key_material = QKDKeyMaterial::new(
            Uuid::new_v4(),
            vec![1, 2, 3, 4, 5, 6, 7, 8],
            "test_key".to_string(),
            "alice".to_string(),
            "bob".to_string(),
            QKDProtocol::BB84,
            0.95,
        );
        
        assert!(!key_material.is_expired());
        assert_eq!(key_material.key_id, "test_key");
        assert_eq!(key_material.extract_key_bytes(4), Some(vec![1, 2, 3, 4]));
        assert_eq!(key_material.extract_key_bytes(10), None);
    }
    
    #[test]
    fn test_qkd_node() {
        let capabilities = QuantumCapabilities {
            supported_protocols: vec![QKDProtocol::BB84, QKDProtocol::E91],
            supported_channels: vec![QuantumChannelType::Simulated { error_rate: 0.05, eavesdropping_probability: 0.0 }],
            max_key_generation_rate_bps: 1000.0,
            min_detection_efficiency: 0.8,
            quantum_memory_coherence_time_ms: 1.0,
            entanglement_generation_rate_hz: 100.0,
        };
        
        let mut node = QKDNode::new(
            "node1".to_string(),
            "agent1".to_string(),
            capabilities,
            "127.0.0.1:8080".to_string(),
        );
        
        assert_eq!(node.node_id, "node1");
        assert_eq!(node.agent_id, "agent1");
        assert!(node.supports_protocol(&QKDProtocol::BB84));
        assert!(!node.supports_protocol(&QKDProtocol::SARG04));
        
        let session_id = Uuid::new_v4();
        node.add_session(session_id);
        assert!(node.active_sessions.contains(&session_id));
        
        node.remove_session(session_id);
        assert!(!node.active_sessions.contains(&session_id));
    }
}