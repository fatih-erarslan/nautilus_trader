//! # Hive Mind Rust Backend
//! 
//! A comprehensive collective intelligence system for the Ximera trading platform.
//! This crate provides distributed consensus, shared memory, neural pattern recognition,
//! and agent coordination capabilities.

pub mod core;
pub mod consensus;
pub mod memory;
pub mod neural;
pub mod network;
pub mod agents;
pub mod config;
pub mod error;
pub mod metrics;
pub mod utils;

#[cfg(feature = "compliance")]
pub mod compliance;
pub mod performance;
pub mod security;
pub mod financial_security;
pub mod https_server;
pub mod zero_trust;
pub mod secure_main;

// Re-exports for convenience
pub use core::{HiveMind, HiveMindBuilder};
pub use consensus::{ConsensusEngine, ConsensusMessage, ConsensusResult};
pub use memory::{CollectiveMemory, MemoryManager, KnowledgeGraph};
pub use neural::{PatternRecognition, NeuralCoordinator, CollectiveLearning};
pub use network::{P2PNetwork, AgentCommunication, MessageProtocol};
pub use agents::{Agent, AgentManager, AgentCoordinator};
pub use config::{HiveMindConfig, NetworkConfig, ConsensusConfig};
pub use error::{HiveMindError, Result};
pub use metrics::{MetricsCollector, PerformanceMonitor};

#[cfg(feature = "compliance")]
pub use compliance::{
    ComplianceCoordinator, ComplianceEngine, ComplianceResult, ComplianceConfig,
    AuditTrail, AuditEvent, AuditEventType,
    DataProtection, EncryptedData, DataClassification,
    AccessControl, AuthenticationResult,
    RiskManager, PreTradeCheckResult, RiskMetrics,
    RegulatoryReporter, TransactionReportData,
    TradeSurveillance, SuspiciousActivityAlert, KYCStatus,
};

/// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Main hive mind result type
pub type HiveMindResult<T> = Result<T>;

/// Initialization function for the hive mind system
pub async fn init_hive_mind(config: HiveMindConfig) -> HiveMindResult<HiveMind> {
    HiveMindBuilder::new(config).build().await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_hive_mind_initialization() {
        let config = HiveMindConfig::default();
        let result = init_hive_mind(config).await;
        assert!(result.is_ok());
    }
}