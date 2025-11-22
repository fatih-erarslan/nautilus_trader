//! Integration Module
//!
//! Connects Byzantine consensus with existing CWTS quantum trading components

pub mod byzantine_consensus_integration;
pub mod e2b_integration;

pub use byzantine_consensus_integration::{
    ConsensusIntegratedTrader, ConsensusPerformanceReport, IntegrationHealth,
    QuantumArbitrageRequest,
};

pub use e2b_integration::{
    E2BIntegration, E2BTrainingClient, HealthCheckReport, IntegrationStatus,
    PerformanceMetrics, ResourceUtilization, SandboxStatusReport, SandboxStatus, TrainingResult,
};
