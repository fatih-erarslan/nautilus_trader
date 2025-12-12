//! E91 Quantum Key Distribution Protocol

use crate::error::QuantumSecurityError;
use crate::types::*;
use crate::key_distribution::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// E91 Protocol Implementation
#[derive(Debug, Clone)]
pub struct E91Protocol {
    pub enabled: bool,
}

impl E91Protocol {
    pub fn new() -> Self {
        Self { enabled: true }
    }
}

impl Default for E91Protocol {
    fn default() -> Self {
        Self::new()
    }
}

/// E91 Engine for entanglement-based quantum key distribution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct E91Engine {
    /// Engine configuration
    pub config: E91Config,
    /// Active sessions
    pub active_sessions: HashMap<Uuid, QKDSession>,
    /// Protocol statistics
    pub statistics: E91Statistics,
    /// Engine status
    pub status: QKDEngineStatus,
}

/// E91 Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct E91Config {
    /// Number of entangled pairs to generate
    pub entangled_pair_count: u32,
    /// Bell inequality violation threshold
    pub bell_violation_threshold: f64,
    /// CHSH inequality test probability
    pub chsh_test_probability: f64,
    /// Error correction threshold
    pub error_correction_threshold: f64,
    /// Privacy amplification factor
    pub privacy_amplification_factor: f64,
    /// Entanglement generation efficiency
    pub entanglement_efficiency: f64,
}

impl Default for E91Config {
    fn default() -> Self {
        Self {
            entangled_pair_count: 10000,
            bell_violation_threshold: 2.0, // CHSH > 2 indicates entanglement
            chsh_test_probability: 0.1, // Test 10% of pairs for Bell violation
            error_correction_threshold: 0.11,
            privacy_amplification_factor: 0.5,
            entanglement_efficiency: 0.7,
        }
    }
}

/// E91 Statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct E91Statistics {
    pub total_sessions: u64,
    pub successful_sessions: u64,
    pub failed_sessions: u64,
    pub total_entangled_pairs_generated: u64,
    pub total_bell_tests_performed: u64,
    pub bell_violation_count: u64,
    pub total_key_bits_generated: u64,
    pub average_chsh_value: f64,
    pub average_key_generation_rate_bps: f64,
}

impl E91Engine {
    /// Create new E91 engine
    pub fn new(config: E91Config) -> Self {
        Self {
            config,
            active_sessions: HashMap::new(),
            statistics: E91Statistics::default(),
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
    
    /// Create new E91 session
    pub fn create_session(
        &mut self,
        alice_agent_id: String,
        bob_agent_id: String,
        key_length_bits: u32,
    ) -> Result<Uuid, QuantumSecurityError> {
        let session = QKDSession::new(
            alice_agent_id,
            bob_agent_id,
            QKDProtocol::E91,
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
    
    /// Execute E91 key distribution
    pub async fn execute_key_distribution(
        &mut self,
        session_id: Uuid,
    ) -> Result<QKDKeyMaterial, QuantumSecurityError> {
        let session = self.active_sessions.get_mut(&session_id)
            .ok_or_else(|| QuantumSecurityError::SessionNotFound(session_id))?;
        
        // Start the session
        session.start().map_err(|e| QuantumSecurityError::QKDError(e))?;
        
        // Simulate E91 protocol steps
        let entangled_pairs = self.entanglement_generation_phase(session).await?;
        let bell_test_result = self.bell_inequality_test_phase(session, &entangled_pairs).await?;
        let raw_key = self.measurement_phase(session, entangled_pairs).await?;
        let corrected_key = self.error_correction_phase(session, raw_key).await?;
        let final_key = self.privacy_amplification_phase(session, corrected_key).await?;
        
        // Verify Bell inequality violation for security
        if bell_test_result < self.config.bell_violation_threshold {
            return Err(QuantumSecurityError::QKDError(
                crate::key_distribution::QKDError::SecurityBreach(
                    format!("Bell inequality not violated: CHSH = {}", bell_test_result)
                )
            ));
        }
        
        // Create key material
        let key_material = QKDKeyMaterial::new(
            session_id,
            final_key.expose().to_vec(),
            format!("e91_key_{}", session_id),
            session.alice_agent_id.clone(),
            session.bob_agent_id.clone(),
            QKDProtocol::E91,
            0.98, // Higher security level due to entanglement
        );
        
        // Mark session as completed
        session.complete(key_material.final_key.expose().len() as u32 * 8)
            .map_err(|e| QuantumSecurityError::QKDError(e))?;
        
        self.statistics.successful_sessions += 1;
        self.statistics.total_key_bits_generated += key_material.final_key.expose().len() as u64 * 8;
        self.statistics.average_chsh_value = bell_test_result;
        
        Ok(key_material)
    }
    
    async fn entanglement_generation_phase(
        &self,
        session: &mut QKDSession,
    ) -> Result<Vec<EntangledPair>, QuantumSecurityError> {
        session.status = QKDSessionStatus::QuantumTransmission;
        
        let mut entangled_pairs = Vec::new();
        
        for _ in 0..self.config.entangled_pair_count {
            if rand::random::<f64>() < self.config.entanglement_efficiency {
                // Generate entangled pair in random Bell state
                let bell_state = match rand::random::<u8>() % 4 {
                    0 => BellState::PhiPlus,
                    1 => BellState::PhiMinus,
                    2 => BellState::PsiPlus,
                    _ => BellState::PsiMinus,
                };
                
                entangled_pairs.push(EntangledPair {
                    bell_state,
                    creation_time: chrono::Utc::now(),
                    measured: false,
                });
            }
        }
        
        self.statistics.total_entangled_pairs_generated += entangled_pairs.len() as u64;
        
        Ok(entangled_pairs)
    }
    
    async fn bell_inequality_test_phase(
        &self,
        _session: &mut QKDSession,
        entangled_pairs: &[EntangledPair],
    ) -> Result<f64, QuantumSecurityError> {
        // Perform CHSH test on subset of pairs
        let test_count = (entangled_pairs.len() as f64 * self.config.chsh_test_probability) as usize;
        let mut chsh_sum = 0.0;
        
        for _ in 0..test_count {
            // Simulate CHSH measurement
            // In real E91, this would involve specific angle measurements
            let chsh_value = 2.0 + rand::random::<f64>() * 0.8; // Simulate violation
            chsh_sum += chsh_value;
        }
        
        self.statistics.total_bell_tests_performed += test_count as u64;
        
        let average_chsh = if test_count > 0 {
            chsh_sum / test_count as f64
        } else {
            0.0
        };
        
        if average_chsh > self.config.bell_violation_threshold {
            self.statistics.bell_violation_count += 1;
        }
        
        Ok(average_chsh)
    }
    
    async fn measurement_phase(
        &self,
        session: &mut QKDSession,
        mut entangled_pairs: Vec<EntangledPair>,
    ) -> Result<SecureBytes, QuantumSecurityError> {
        session.status = QKDSessionStatus::SiftingKeys;
        
        let mut key_bits = Vec::new();
        
        for pair in &mut entangled_pairs {
            if !pair.measured {
                // Alice and Bob perform measurements
                let alice_result = rand::random::<bool>();
                let bob_result = match pair.bell_state {
                    BellState::PhiPlus | BellState::PhiMinus => alice_result, // Correlated
                    BellState::PsiPlus | BellState::PsiMinus => !alice_result, // Anti-correlated
                };
                
                // Use Alice's result as key bit (Bob has correlated/anti-correlated result)
                key_bits.push(alice_result as u8);
                pair.measured = true;
            }
        }
        
        Ok(SecureBytes::new(key_bits))
    }
    
    async fn error_correction_phase(
        &self,
        session: &mut QKDSession,
        raw_key: SecureBytes,
    ) -> Result<SecureBytes, QuantumSecurityError> {
        session.status = QKDSessionStatus::ErrorCorrection;
        
        // Simulate error correction
        let error_rate = 0.03; // Lower error rate due to quantum correlations
        session.error_rate = error_rate;
        
        if error_rate > self.config.error_correction_threshold {
            return Err(QuantumSecurityError::QKDError(
                crate::key_distribution::QKDError::HighErrorRate {
                    measured: error_rate,
                    threshold: self.config.error_correction_threshold,
                }
            ));
        }
        
        Ok(raw_key)
    }
    
    async fn privacy_amplification_phase(
        &self,
        session: &mut QKDSession,
        corrected_key: SecureBytes,
    ) -> Result<SecureBytes, QuantumSecurityError> {
        session.status = QKDSessionStatus::PrivacyAmplification;
        
        // Simulate privacy amplification
        let input_length = corrected_key.expose().len();
        let output_length = (input_length as f64 * self.config.privacy_amplification_factor) as usize;
        
        if output_length == 0 {
            return Err(QuantumSecurityError::KeyGenerationError(
                "Privacy amplification resulted in zero-length key".to_string()
            ));
        }
        
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
    pub fn get_statistics(&self) -> &E91Statistics {
        &self.statistics
    }
}

/// Entangled Pair for E91 protocol
#[derive(Debug, Clone)]
pub struct EntangledPair {
    pub bell_state: BellState,
    pub creation_time: chrono::DateTime<chrono::Utc>,
    pub measured: bool,
}

impl Default for E91Engine {
    fn default() -> Self {
        Self::new(E91Config::default())
    }
}