//! Nonce management for replay attack protection

use dashmap::DashMap;
use std::time::{Duration, Instant};
use crate::{McpAuthError, McpAuthResult};

/// Thread-safe nonce manager for replay attack protection
pub struct NonceManager {
    /// Used nonces with their expiration times
    used_nonces: DashMap<String, Instant>,

    /// Time-to-live for nonces
    ttl: Duration,

    /// Maximum cache size
    #[allow(dead_code)]
    max_size: usize,
}

impl NonceManager {
    /// Create new nonce manager
    pub fn new(ttl_secs: u64, max_size: usize) -> Self {
        Self {
            used_nonces: DashMap::with_capacity(max_size.min(1000)),
            ttl: Duration::from_secs(ttl_secs),
            max_size,
        }
    }

    /// Check and consume a nonce (returns error if already used)
    pub fn consume(&self, nonce: &str) -> McpAuthResult<()> {
        let now = Instant::now();
        let expiration = now + self.ttl;

        // Check if nonce exists and is still valid
        if let Some(existing) = self.used_nonces.get(nonce) {
            if *existing > now {
                // Nonce still valid, reject as replay
                return Err(McpAuthError::NonceReused {
                    nonce: nonce.to_string(),
                });
            }
        }

        // Insert or update the nonce
        self.used_nonces.insert(nonce.to_string(), expiration);
        Ok(())
    }

    /// Check if a nonce is valid (without consuming it)
    pub fn is_valid(&self, nonce: &str) -> bool {
        let now = Instant::now();

        match self.used_nonces.get(nonce) {
            Some(expiration) => *expiration < now, // Valid if expired (can be reused)
            None => true, // Valid if not used
        }
    }

    /// Get current number of tracked nonces
    pub fn len(&self) -> usize {
        self.used_nonces.len()
    }

    /// Check if manager is empty
    pub fn is_empty(&self) -> bool {
        self.used_nonces.is_empty()
    }

    /// Clear all nonces (for testing/reset)
    pub fn clear(&self) {
        self.used_nonces.clear();
    }
}

impl Default for NonceManager {
    fn default() -> Self {
        Self::new(300, 100_000) // 5 minutes TTL, 100k max entries
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nonce_consumption() {
        let manager = NonceManager::new(60, 1000);

        // First use should succeed
        assert!(manager.consume("nonce-1").is_ok());

        // Second use should fail
        assert!(matches!(
            manager.consume("nonce-1"),
            Err(McpAuthError::NonceReused { .. })
        ));

        // Different nonce should succeed
        assert!(manager.consume("nonce-2").is_ok());
    }

    #[test]
    fn test_nonce_validity() {
        let manager = NonceManager::new(60, 1000);

        assert!(manager.is_valid("unused-nonce"));

        manager.consume("used-nonce").unwrap();
        assert!(!manager.is_valid("used-nonce"));
    }

    #[test]
    fn test_clear() {
        let manager = NonceManager::new(60, 1000);

        manager.consume("nonce-1").unwrap();
        manager.consume("nonce-2").unwrap();

        assert_eq!(manager.len(), 2);

        manager.clear();
        assert!(manager.is_empty());

        // Can reuse nonces after clear
        assert!(manager.consume("nonce-1").is_ok());
    }
}
