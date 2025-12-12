//! Quantum Security Protocols Module

use crate::error::QuantumSecurityError;
use crate::types::*;

/// Security Protocols Manager
#[derive(Debug, Clone)]
pub struct SecurityProtocolsManager {
    pub enabled: bool,
}

impl SecurityProtocolsManager {
    pub fn new() -> Self {
        Self { enabled: true }
    }
}

impl Default for SecurityProtocolsManager {
    fn default() -> Self {
        Self::new()
    }
}