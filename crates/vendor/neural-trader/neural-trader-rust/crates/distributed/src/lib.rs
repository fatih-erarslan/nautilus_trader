// neural-trader-distributed: Distributed systems integration
//
// This crate provides:
// - E2B sandbox integration for isolated execution
// - Agentic-flow federations for distributed agent coordination
// - Agentic-payments for credit-based resource management
// - Auto-scaling and load balancing capabilities

#![warn(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]

pub mod e2b;
pub mod federation;
pub mod payments;
pub mod scaling;

pub use e2b::{SandboxManager, SandboxConfig, SandboxResult};
pub use federation::{FederationTopology, AgentCoordinator, MessageBus};
pub use payments::{CreditSystem, BillingGateway, UsageTracker};
pub use scaling::{AutoScaler, LoadBalancer, HealthChecker};

use thiserror::Error;

/// Result type for distributed operations
pub type Result<T> = std::result::Result<T, DistributedError>;

/// Errors that can occur in distributed systems
#[derive(Debug, Error)]
pub enum DistributedError {
    /// E2B sandbox operation failed
    #[error("E2B sandbox error: {0}")]
    E2bError(String),

    /// Federation coordination failed
    #[error("Federation error: {0}")]
    FederationError(String),

    /// Payment processing failed
    #[error("Payment error: {0}")]
    PaymentError(String),

    /// Scaling operation failed
    #[error("Scaling error: {0}")]
    ScalingError(String),

    /// Network communication failed
    #[error("Network error: {0}")]
    NetworkError(#[from] reqwest::Error),

    /// Serialization/deserialization failed
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),

    /// Database operation failed
    #[error("Database error: {0}")]
    DatabaseError(String),

    /// Configuration error
    #[error("Configuration error: {0}")]
    ConfigError(String),

    /// Timeout occurred
    #[error("Operation timed out")]
    Timeout,

    /// Resource limit exceeded
    #[error("Resource limit exceeded: {0}")]
    ResourceLimitExceeded(String),

    /// Insufficient credits
    #[error("Insufficient credits: need {needed}, have {available}")]
    InsufficientCredits { needed: u64, available: u64 },

    /// Agent not found
    #[error("Agent not found: {0}")]
    AgentNotFound(String),

    /// Sandbox not found
    #[error("Sandbox not found: {0}")]
    SandboxNotFound(String),
}

/// Configuration for the distributed system
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DistributedConfig {
    /// E2B API key
    pub e2b_api_key: Option<String>,

    /// Federation topology type
    pub topology: String,

    /// Maximum number of concurrent sandboxes
    pub max_sandboxes: usize,

    /// Maximum number of agents in federation
    pub max_agents: usize,

    /// Enable auto-scaling
    pub auto_scale: bool,

    /// Payment gateway configuration
    pub payment_gateway_url: Option<String>,

    /// Default credit allocation per user
    pub default_credits: u64,

    /// Database URL for state persistence
    pub database_url: String,
}

impl Default for DistributedConfig {
    fn default() -> Self {
        Self {
            e2b_api_key: None,
            topology: "mesh".to_string(),
            max_sandboxes: 10,
            max_agents: 50,
            auto_scale: true,
            payment_gateway_url: None,
            default_credits: 1000,
            database_url: "sqlite::memory:".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = DistributedConfig::default();
        assert_eq!(config.topology, "mesh");
        assert_eq!(config.max_sandboxes, 10);
        assert_eq!(config.max_agents, 50);
        assert!(config.auto_scale);
    }

    #[test]
    fn test_error_display() {
        let error = DistributedError::InsufficientCredits {
            needed: 100,
            available: 50,
        };
        assert_eq!(
            error.to_string(),
            "Insufficient credits: need 100, have 50"
        );
    }
}
