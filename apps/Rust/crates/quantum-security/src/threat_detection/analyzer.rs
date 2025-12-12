//! Threat Analyzer Module

use crate::error::QuantumSecurityError;
use crate::types::*;

/// Threat Analyzer
#[derive(Debug, Clone)]
pub struct ThreatAnalyzer {
    pub enabled: bool,
}

impl ThreatAnalyzer {
    pub fn new() -> Self {
        Self { enabled: true }
    }
}

impl Default for ThreatAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}