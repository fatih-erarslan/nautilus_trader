//! Threat Patterns Module

use crate::error::QuantumSecurityError;
use crate::types::*;

/// Threat Patterns Manager
#[derive(Debug, Clone)]
pub struct ThreatPatternsManager {
    pub enabled: bool,
}

impl ThreatPatternsManager {
    pub fn new() -> Self {
        Self { enabled: true }
    }
}

impl Default for ThreatPatternsManager {
    fn default() -> Self {
        Self::new()
    }
}