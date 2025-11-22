pub mod algorithms;
pub mod common_types;
pub mod execution;
pub mod mcp;

// Real-time market data integration
pub mod data;

// Circuit breakers and fault tolerance
pub mod circuit;

// Data validation and integrity
pub mod validation;

// Audit logging and compliance
pub mod audit;

// Caching systems
pub mod cache;

// Connection pooling
pub mod pool;

// Byzantine Fault Tolerant Consensus System
pub mod consensus;

// Integration with existing trading systems
pub mod integration;

// Deployment and production monitoring
pub mod deployment;

// Scientifically rigorous integration layer
pub mod scientifically_rigorous_integration;

// SEC Rule 15c3-5 Compliance Modules
pub mod compliance;
pub mod emergency;
pub mod risk;

// Optional modules
pub mod memory;
// pub mod analyzers;
// pub mod neural;
pub mod adaptation; // Re-enabled: e2b_integration now available
pub mod error;
pub mod evolution; // Re-enabled: e2b_integration now available
pub mod learning; // Re-enabled: e2b_integration now available
pub mod neural_models;
pub mod result;
pub mod attention;

/// Emergent Architecture Genesis - Complete Bayesian VaR System with E2B Integration
pub mod architecture; // Re-enabled: e2b_integration now available
pub mod neural_integration;
// pub mod exchange;
// pub mod simd;
pub mod gpu;
pub mod quantum {
    pub mod pbit_engine;
    pub mod pbit_orderbook_integration;
    pub mod quantum_correlation_engine;
}
// pub mod nhits;
// pub mod forecasting;
// pub mod autopoietic;
// pub mod cqgs;

// Async-safe wrappers for production environments
#[cfg(feature = "async-wrappers")]
pub mod async_wrappers;

use std::sync::atomic::{AtomicBool, AtomicU64};

// Re-export SEC Rule 15c3-5 compliance components
pub use compliance::sec_rule_15c3_5::{
    KillSwitchEvent, KillSwitchTrigger, Order, OrderSide, OrderType, PreTradeRiskEngine,
    RiskLimits, RiskValidationResult,
};

pub use risk::market_access_controls::{
    CircuitBreakerLevel, DailyRiskLimits, MarketAccessDecision, MarketAccessEngine,
    SystematicRiskMetrics,
};

// pub use audit::regulatory_audit::{
//     AnomalyRecord, AuditRecord, ComplianceStatus, RegulatoryAuditEngine, RegulatoryReport,
// }; // TEMPORARILY DISABLED: Module not exported

pub use emergency::kill_switch::{
    AutoTriggerCondition, EmergencyKillSwitchEngine, KillSwitchConfiguration, KillSwitchLevel,
    KillSwitchStatus,
};

// pub use validation::{
//     ArithmeticError, AutopoiesisValidationResult, AutopoieticSystem, FinancialCalculator,
//     IEEE754ValidationResult, MathematicalValidationFramework, SystemValidationReport,
// }; // TEMPORARILY DISABLED: Module exports need fixing

pub use consensus::{
    AtomicCommit, AtomicTransaction, ByzantineConsensus, ByzantineConsensusSystem,
    ByzantineMessage, ConsensusError, ExchangeId, MessageType, QuantumVerification, ValidatorId,
    ValidatorNetwork,
};

/// Current version for compliance tracking
pub const COMPLIANCE_VERSION: &str = env!("CARGO_PKG_VERSION");

#[repr(align(64))]
pub struct CWTSUltra {
    // Lock-free components
    #[allow(dead_code)]
    running: AtomicBool,
    #[allow(dead_code)]
    capital: AtomicU64,

    // SEC Rule 15c3-5 Compliance Engine
    #[allow(dead_code)]
    compliance_active: AtomicBool,
}

impl Default for CWTSUltra {
    fn default() -> Self {
        Self::new()
    }
}

impl CWTSUltra {
    pub fn new() -> Self {
        Self {
            running: AtomicBool::new(false),
            capital: AtomicU64::new(50_000_000), // Store as microcents
            compliance_active: AtomicBool::new(true), // Compliance always active
        }
    }

    /// Initialize SEC Rule 15c3-5 compliance system
    pub async fn initialize_compliance() -> Result<
        (
            compliance::sec_rule_15c3_5::PreTradeRiskEngine,
            tokio::sync::mpsc::UnboundedReceiver<compliance::sec_rule_15c3_5::AuditEvent>,
            tokio::sync::mpsc::UnboundedReceiver<compliance::sec_rule_15c3_5::EmergencyAlert>,
        ),
        Box<dyn std::error::Error>,
    > {
        let (engine, audit_rx, emergency_rx) = PreTradeRiskEngine::new();
        Ok((engine, audit_rx, emergency_rx))
    }
}
