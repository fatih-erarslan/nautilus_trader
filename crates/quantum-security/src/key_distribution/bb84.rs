//! BB84 Quantum Key Distribution Protocol

use crate::error::QuantumSecurityError;
use crate::types::*;
use crate::key_distribution::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// BB84 Protocol Implementation
#[derive(Debug, Clone)]
pub struct BB84Protocol {
    pub enabled: bool,
}

impl BB84Protocol {
    pub fn new() -> Self {
        Self { enabled: true }
    }
}

impl Default for BB84Protocol {
    fn default() -> Self {
        Self::new()
    }
}

/// BB84 Engine for quantum key distribution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BB84Engine {
    /// Engine configuration
    pub config: BB84Config,
    /// Active sessions
    pub active_sessions: HashMap<Uuid, QKDSession>,
    /// Protocol statistics
    pub statistics: BB84Statistics,
    /// Engine status
    pub status: QKDEngineStatus,
}

/// BB84 Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BB84Config {
    /// Number of qubits to prepare
    pub qubit_count: u32,
    /// Basis choice probability for diagonal basis
    pub diagonal_basis_probability: f64,
    /// Error correction threshold
    pub error_correction_threshold: f64,
    /// Privacy amplification factor
    pub privacy_amplification_factor: f64,
    /// Detection efficiency
    pub detection_efficiency: f64,
}

impl Default for BB84Config {
    fn default() -> Self {
        Self {
            qubit_count: 10000,
            diagonal_basis_probability: 0.5,
            error_correction_threshold: 0.11,
            privacy_amplification_factor: 0.5,
            detection_efficiency: 0.8,
        }
    }
}

/// BB84 Statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BB84Statistics {
    pub total_sessions: u64,
    pub successful_sessions: u64,
    pub failed_sessions: u64,
    pub total_qubits_prepared: u64,
    pub total_qubits_measured: u64,
    pub total_key_bits_generated: u64,
    pub average_error_rate: f64,
    pub average_key_generation_rate_bps: f64,
}

/// QKD Engine Status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum QKDEngineStatus {
    Initializing,
    Ready,
    Running,
    Error(String),
    Maintenance,
}

impl BB84Engine {
    /// Create new BB84 engine
    pub fn new(config: BB84Config) -> Self {
        Self {
            config,
            active_sessions: HashMap::new(),
            statistics: BB84Statistics::default(),
            status: QKDEngineStatus::Initializing,
        }
    }
    
    /// Start the engine
    pub fn start(&mut self) -> Result<(), QuantumSecurityError> {
        self.status = QKDEngineStatus::Ready;
        Ok(())
    }
    
    /// Stop the engine
    pub fn stop(&mut self) -> Result<(), QuantumSecurityError> {
        self.active_sessions.clear();
        self.status = QKDEngineStatus::Ready;
        Ok(())
    }
    
    /// Create new BB84 session
    pub fn create_session(
        &mut self,
        alice_agent_id: String,
        bob_agent_id: String,
        key_length_bits: u32,
    ) -> Result<Uuid, QuantumSecurityError> {
        let session = QKDSession::new(
            alice_agent_id,
            bob_agent_id,
            QKDProtocol::BB84,
            QuantumChannelType::Simulated { 
                error_rate: 0.05, 
                eavesdropping_probability: 0.0 
            },
            key_length_bits,
        );
        
        let session_id = session.session_id;
        self.active_sessions.insert(session_id, session);
        self.statistics.total_sessions += 1;
        
        Ok(session_id)
    }
    
    /// Execute BB84 key distribution
    pub async fn execute_key_distribution(
        &mut self,
        session_id: Uuid,
    ) -> Result<QKDKeyMaterial, QuantumSecurityError> {
        let session = self.active_sessions.get_mut(&session_id)
            .ok_or_else(|| QuantumSecurityError::SessionNotFound(session_id))?;
        
        // Start the session
        session.start().map_err(|e| QuantumSecurityError::QKDError(e))?;
        
        // Simulate BB84 protocol steps
        let raw_key = self.quantum_transmission_phase(session).await?;
        let sifted_key = self.sifting_phase(session, raw_key).await?;
        let corrected_key = self.error_correction_phase(session, sifted_key).await?;
        let final_key = self.privacy_amplification_phase(session, corrected_key).await?;
        
        // Create key material
        let key_material = QKDKeyMaterial::new(
            session_id,
            final_key.expose().to_vec(),
            format!("bb84_key_{}", session_id),
            session.alice_agent_id.clone(),
            session.bob_agent_id.clone(),
            QKDProtocol::BB84,
            0.95, // Security level
        );
        
        // Mark session as completed
        session.complete(key_material.final_key.expose().len() as u32 * 8)
            .map_err(|e| QuantumSecurityError::QKDError(e))?;
        
        self.statistics.successful_sessions += 1;
        self.statistics.total_key_bits_generated += key_material.final_key.expose().len() as u64 * 8;
        
        Ok(key_material)
    }
    
    async fn quantum_transmission_phase(
        &self,
        session: &mut QKDSession,
    ) -> Result<SecureBytes, QuantumSecurityError> {
        session.status = QKDSessionStatus::QuantumTransmission;
        
        // Simulate quantum transmission
        let mut raw_key_bits = Vec::new();
        for _ in 0..self.config.qubit_count {
            // Alice prepares random bit
            let bit = rand::random::<bool>();
            // Alice chooses random basis
            let alice_basis = if rand::random::<f64>() < self.config.diagonal_basis_probability {
                MeasurementBasis::Diagonal
            } else {
                MeasurementBasis::Rectilinear
            };
            
            // Bob chooses random basis
            let bob_basis = if rand::random::<f64>() < self.config.diagonal_basis_probability {
                MeasurementBasis::Diagonal
            } else {
                MeasurementBasis::Rectilinear
            };
            
            // Simulate measurement (only keep if bases match)
            if alice_basis == bob_basis && rand::random::<f64>() < self.config.detection_efficiency {
                raw_key_bits.push(bit as u8);
            }
        }
        
        Ok(SecureBytes::new(raw_key_bits))
    }
    
    async fn sifting_phase(
        &self,
        session: &mut QKDSession,
        raw_key: SecureBytes,
    ) -> Result<SecureBytes, QuantumSecurityError> {
        session.status = QKDSessionStatus::SiftingKeys;
        
        // In real BB84, Alice and Bob compare bases and keep only matching ones
        // For simulation, we already filtered in transmission phase
        Ok(raw_key)
    }
    
    async fn error_correction_phase(
        &self,
        session: &mut QKDSession,
        sifted_key: SecureBytes,
    ) -> Result<SecureBytes, QuantumSecurityError> {
        session.status = QKDSessionStatus::ErrorCorrection;
        
        // Simulate error correction (in practice this would use LDPC codes or similar)
        let error_rate = 0.05; // 5% error rate
        session.error_rate = error_rate;
        
        if error_rate > self.config.error_correction_threshold {
            return Err(QuantumSecurityError::QKDError(
                crate::key_distribution::QKDError::HighErrorRate {
                    measured: error_rate,
                    threshold: self.config.error_correction_threshold,
                }
            ));
        }
        
        Ok(sifted_key)
    }
    
    async fn privacy_amplification_phase(
        &self,
        session: &mut QKDSession,
        corrected_key: SecureBytes,
    ) -> Result<SecureBytes, QuantumSecurityError> {
        session.status = QKDSessionStatus::PrivacyAmplification;
        
        // Simulate privacy amplification (reduce key length to ensure security)
        let input_length = corrected_key.expose().len();
        let output_length = (input_length as f64 * self.config.privacy_amplification_factor) as usize;
        
        if output_length == 0 {
            return Err(QuantumSecurityError::KeyGenerationError(
                "Privacy amplification resulted in zero-length key".to_string()
            ));
        }
        
        // Simple simulation: take first n bytes
        let final_key = corrected_key.take(output_length);
        
        Ok(final_key)
    }
    
    /// Get session status
    pub fn get_session_status(&self, session_id: Uuid) -> Option<&QKDSessionStatus> {
        self.active_sessions.get(&session_id).map(|s| &s.status)
    }
    
    /// Remove completed session
    pub fn remove_session(&mut self, session_id: Uuid) -> Option<QKDSession> {
        self.active_sessions.remove(&session_id)
    }
    
    /// Get engine statistics
    pub fn get_statistics(&self) -> &BB84Statistics {
        &self.statistics
    }
    
    /// Key sifting phase
    pub async fn key_sifting_phase(
        &self,
        _alice_bits: &[u8],
        _alice_bases: &[u8], 
        _bob_measurements: &[u8]
    ) -> Result<(Vec<u8>, Vec<u8>), QuantumSecurityError> {
        // Mock implementation for compilation
        Ok((vec![0u8; 100], vec![0u8; 100]))
    }
    
    /// Estimate error rate
    pub async fn estimate_error_rate(
        &self,
        _sifted_key_alice: &[u8],
        _sifted_key_bob: &[u8]
    ) -> Result<f64, QuantumSecurityError> {
        // Mock implementation for compilation
        Ok(0.05) // 5% error rate
    }
}

impl Default for BB84Engine {
    fn default() -> Self {
        Self::new(BB84Config::default())
    }
}