//! QKD Protocols Module

use crate::error::QuantumSecurityError;
use crate::types::*;
use super::*;

/// QKD Protocols Manager
#[derive(Debug, Clone)]
pub struct QKDProtocolsManager {
    pub enabled: bool,
}

impl QKDProtocolsManager {
    pub fn new() -> Self {
        Self { enabled: true }
    }

    /// Establish quantum keys between two agents
    pub async fn establish_keys(
        &self,
        local_agent_id: &str,
        remote_agent_id: &str,
    ) -> Result<QKDKeyMaterial, QuantumSecurityError> {
        if !self.enabled {
            return Err(QuantumSecurityError::QKDSessionFailed(
                "QKD manager is disabled".to_string()
            ));
        }

        // Mock QKD key establishment - in a real implementation this would
        // use actual quantum key distribution protocols
        let key_data = vec![0u8; 32]; // 256-bit key
        let key_material = QKDKeyMaterial::new(
            uuid::Uuid::new_v4(),
            key_data,
            format!("qkd_key_{}_{}", local_agent_id, remote_agent_id),
            local_agent_id.to_string(),
            remote_agent_id.to_string(),
            QKDProtocol::BB84,
            0.95, // Security level
        );

        Ok(key_material)
    }
}

impl Default for QKDProtocolsManager {
    fn default() -> Self {
        Self::new()
    }
}