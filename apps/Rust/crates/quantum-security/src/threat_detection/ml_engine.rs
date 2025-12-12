//! ML Engine Module

use crate::error::QuantumSecurityError;
use crate::types::*;

/// ML Engine
#[derive(Debug, Clone)]
pub struct MLEngine {
    pub enabled: bool,
}

impl MLEngine {
    pub fn new() -> Self {
        Self { enabled: true }
    }
}

impl Default for MLEngine {
    fn default() -> Self {
        Self::new()
    }
}