//! Utility Functions for Quantum Security

use crate::error::QuantumSecurityError;

/// Utility functions
pub struct SecurityUtils;

impl SecurityUtils {
    /// Generate secure random bytes
    pub fn generate_random_bytes(length: usize) -> Vec<u8> {
        (0..length).map(|_| rand::random::<u8>()).collect()
    }
}