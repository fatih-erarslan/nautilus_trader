//! Threat Response Module

use crate::error::QuantumSecurityError;
use crate::types::*;

/// Threat Response Manager
#[derive(Debug, Clone)]
pub struct ThreatResponseManager {
    pub enabled: bool,
}

impl ThreatResponseManager {
    pub fn new() -> Self {
        Self { enabled: true }
    }
}

impl Default for ThreatResponseManager {
    fn default() -> Self {
        Self::new()
    }
}