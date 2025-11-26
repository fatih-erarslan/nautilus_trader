/// Resilience patterns for fault-tolerant operations
///
/// This module provides circuit breakers, retry policies, and other
/// resilience patterns for building robust distributed systems.

pub mod circuit_breaker;
pub mod integration;

pub use circuit_breaker::{CircuitBreaker, CircuitBreakerConfig, CircuitBreakerMetrics};
pub use integration::{
    ApiCircuitBreaker, E2BSandboxCircuitBreaker, NeuralCircuitBreaker,
    DatabaseCircuitBreaker, TradingSystemCircuitBreakers,
};

use std::sync::Arc;
use tokio::sync::RwLock;
use std::collections::HashMap;

/// Registry for managing multiple circuit breakers
pub struct CircuitBreakerRegistry {
    breakers: Arc<RwLock<HashMap<String, CircuitBreaker>>>,
}

impl CircuitBreakerRegistry {
    /// Create a new circuit breaker registry
    pub fn new() -> Self {
        Self {
            breakers: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Register a new circuit breaker
    pub async fn register(&self, name: String, config: CircuitBreakerConfig) -> CircuitBreaker {
        let breaker = CircuitBreaker::new(name.clone(), config);
        let mut breakers = self.breakers.write().await;
        breakers.insert(name, breaker.clone());
        breaker
    }

    /// Get an existing circuit breaker by name
    pub async fn get(&self, name: &str) -> Option<CircuitBreaker> {
        let breakers = self.breakers.read().await;
        breakers.get(name).cloned()
    }

    /// Get or create a circuit breaker
    pub async fn get_or_create(
        &self,
        name: String,
        config: CircuitBreakerConfig,
    ) -> CircuitBreaker {
        let breakers = self.breakers.read().await;
        if let Some(breaker) = breakers.get(&name) {
            return breaker.clone();
        }
        drop(breakers);

        self.register(name, config).await
    }

    /// Get all circuit breaker names
    pub async fn list_names(&self) -> Vec<String> {
        let breakers = self.breakers.read().await;
        breakers.keys().cloned().collect()
    }

    /// Get metrics for all circuit breakers
    pub async fn get_all_metrics(&self) -> HashMap<String, CircuitBreakerMetrics> {
        let breakers = self.breakers.read().await;
        let mut metrics = HashMap::new();

        for (name, breaker) in breakers.iter() {
            metrics.insert(name.clone(), breaker.get_metrics().await);
        }

        metrics
    }

    /// Remove a circuit breaker
    pub async fn remove(&self, name: &str) -> bool {
        let mut breakers = self.breakers.write().await;
        breakers.remove(name).is_some()
    }

    /// Clear all circuit breakers
    pub async fn clear(&self) {
        let mut breakers = self.breakers.write().await;
        breakers.clear();
    }
}

impl Default for CircuitBreakerRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_registry_register_and_get() {
        let registry = CircuitBreakerRegistry::new();
        let config = CircuitBreakerConfig::default();

        let cb = registry.register("test".to_string(), config.clone()).await;
        assert!(registry.get("test").await.is_some());
    }

    #[tokio::test]
    async fn test_registry_get_or_create() {
        let registry = CircuitBreakerRegistry::new();
        let config = CircuitBreakerConfig::default();

        let cb1 = registry.get_or_create("test".to_string(), config.clone()).await;
        let cb2 = registry.get_or_create("test".to_string(), config.clone()).await;

        // Should be the same instance
        assert_eq!(cb1.get_state().await, cb2.get_state().await);
    }

    #[tokio::test]
    async fn test_registry_list_names() {
        let registry = CircuitBreakerRegistry::new();
        let config = CircuitBreakerConfig::default();

        registry.register("cb1".to_string(), config.clone()).await;
        registry.register("cb2".to_string(), config.clone()).await;

        let names = registry.list_names().await;
        assert_eq!(names.len(), 2);
        assert!(names.contains(&"cb1".to_string()));
        assert!(names.contains(&"cb2".to_string()));
    }

    #[tokio::test]
    async fn test_registry_remove() {
        let registry = CircuitBreakerRegistry::new();
        let config = CircuitBreakerConfig::default();

        registry.register("test".to_string(), config).await;
        assert!(registry.get("test").await.is_some());

        assert!(registry.remove("test").await);
        assert!(registry.get("test").await.is_none());
    }

    #[tokio::test]
    async fn test_registry_clear() {
        let registry = CircuitBreakerRegistry::new();
        let config = CircuitBreakerConfig::default();

        registry.register("cb1".to_string(), config.clone()).await;
        registry.register("cb2".to_string(), config.clone()).await;

        assert_eq!(registry.list_names().await.len(), 2);

        registry.clear().await;
        assert_eq!(registry.list_names().await.len(), 0);
    }
}
