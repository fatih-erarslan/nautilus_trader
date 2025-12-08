//! Threat Detector Module

use crate::error::QuantumSecurityError;
use crate::types::*;
use super::{ThreatLevel, ThreatOperation};

/// Threat Detector
#[derive(Debug, Clone)]
pub struct ThreatDetector {
    pub enabled: bool,
}

impl ThreatDetector {
    pub fn new() -> Self {
        Self { enabled: true }
    }

    /// Validate security operation against threat level
    pub async fn validate_operation(
        &self,
        threat_level: &ThreatLevel,
        operation: ThreatOperation,
    ) -> Result<(), QuantumSecurityError> {
        if !self.enabled {
            return Ok(());
        }

        // Simple validation logic - in a real implementation this would be more sophisticated
        match (threat_level, operation) {
            (ThreatLevel::Critical, _) => {
                // Block all operations at critical threat level
                Err(QuantumSecurityError::SecurityPolicyViolation(
                    "Operation blocked due to critical threat level".to_string()
                ))
            }
            (ThreatLevel::High, ThreatOperation::KeyGeneration) => {
                // Block key generation at high threat level
                Err(QuantumSecurityError::SecurityPolicyViolation(
                    "Key generation blocked due to high threat level".to_string()
                ))
            }
            _ => Ok(())
        }
    }
}

impl Default for ThreatDetector {
    fn default() -> Self {
        Self::new()
    }
}