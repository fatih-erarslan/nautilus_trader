//! # Financial Regulatory Compliance Framework
//! 
//! This module implements comprehensive financial regulatory compliance including:
//! - SOX (Sarbanes-Oxley) Section 404 compliance
//! - PCI DSS Level 1 security standards
//! - GDPR data protection requirements
//! - Basel III operational risk management
//! - MiFID II transaction reporting
//! - AML/KYC compliance framework

pub mod audit_trail;
pub mod data_protection;
pub mod access_control;
pub mod risk_management;
pub mod regulatory_reporting;
pub mod trade_surveillance;
pub mod compliance_engine;

// Re-exports for convenience
pub use audit_trail::{AuditTrail, AuditEvent, AuditEventType, ImmutableLog};
pub use data_protection::{DataProtection, EncryptionManager, PIIHandler, GDPRCompliance};
pub use access_control::{AccessControl, RoleBasedAccess, MultiFactorAuth, PrivilegedAccess};
pub use risk_management::{RiskManager, PositionLimits, RealTimeMonitoring, StressTesting};
pub use regulatory_reporting::{RegulatoryReporter, SOXReporting, MiFIDReporting, BaselReporting};
pub use trade_surveillance::{TradeSurveillance, SuspiciousActivityDetector, MarketAbuseDetection};
pub use compliance_engine::{ComplianceEngine, ComplianceRule, ComplianceResult, ViolationAlert};

use crate::error::Result;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Main compliance coordinator for the hive mind system
#[derive(Debug)]
pub struct ComplianceCoordinator {
    /// Audit trail system for immutable logging
    audit_trail: Arc<AuditTrail>,
    
    /// Data protection and encryption manager
    data_protection: Arc<DataProtection>,
    
    /// Access control and authentication system
    access_control: Arc<RwLock<AccessControl>>,
    
    /// Risk management system
    risk_manager: Arc<RiskManager>,
    
    /// Regulatory reporting engine
    regulatory_reporter: Arc<RegulatoryReporter>,
    
    /// Trade surveillance system
    trade_surveillance: Arc<TradeSurveillance>,
    
    /// Main compliance engine
    compliance_engine: Arc<ComplianceEngine>,
}

impl ComplianceCoordinator {
    /// Initialize the compliance coordinator with all regulatory requirements
    pub async fn new() -> Result<Self> {
        let audit_trail = Arc::new(AuditTrail::new().await?);
        let data_protection = Arc::new(DataProtection::new().await?);
        let access_control = Arc::new(RwLock::new(AccessControl::new().await?));
        let risk_manager = Arc::new(RiskManager::new().await?);
        let regulatory_reporter = Arc::new(RegulatoryReporter::new().await?);
        let trade_surveillance = Arc::new(TradeSurveillance::new().await?);
        let compliance_engine = Arc::new(ComplianceEngine::new().await?);
        
        Ok(Self {
            audit_trail,
            data_protection,
            access_control,
            risk_manager,
            regulatory_reporter,
            trade_surveillance,
            compliance_engine,
        })
    }
    
    /// Start all compliance systems
    pub async fn start(&self) -> Result<()> {
        self.audit_trail.start().await?;
        self.data_protection.start().await?;
        self.access_control.write().await.start().await?;
        self.risk_manager.start().await?;
        self.regulatory_reporter.start().await?;
        self.trade_surveillance.start().await?;
        self.compliance_engine.start().await?;
        
        tracing::info!("All compliance systems started successfully");
        Ok(())
    }
    
    /// Perform comprehensive compliance check
    pub async fn compliance_check(&self) -> Result<ComplianceResult> {
        self.compliance_engine.comprehensive_check().await
    }
    
    /// Get audit trail reference for logging compliance events
    pub fn audit_trail(&self) -> &Arc<AuditTrail> {
        &self.audit_trail
    }
    
    /// Get data protection reference for encryption operations
    pub fn data_protection(&self) -> &Arc<DataProtection> {
        &self.data_protection
    }
    
    /// Get access control reference for authentication operations
    pub fn access_control(&self) -> &Arc<RwLock<AccessControl>> {
        &self.access_control
    }
}

/// Compliance configuration for the entire system
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ComplianceConfig {
    /// SOX compliance configuration
    pub sox: SOXConfig,
    
    /// PCI DSS compliance configuration
    pub pci_dss: PCIDSSConfig,
    
    /// GDPR compliance configuration
    pub gdpr: GDPRConfig,
    
    /// Basel III compliance configuration
    pub basel: BaselConfig,
    
    /// MiFID II compliance configuration
    pub mifid: MiFIDConfig,
    
    /// AML/KYC compliance configuration
    pub aml_kyc: AMLKYCConfig,
}

/// SOX (Sarbanes-Oxley) compliance configuration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SOXConfig {
    /// Enable Section 404 internal controls
    pub enable_internal_controls: bool,
    
    /// Audit trail retention period in days
    pub audit_retention_days: u32,
    
    /// Enable segregation of duties
    pub enable_segregation_duties: bool,
    
    /// Enable change management controls
    pub enable_change_controls: bool,
}

/// PCI DSS compliance configuration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PCIDSSConfig {
    /// PCI DSS compliance level (1-4)
    pub compliance_level: u8,
    
    /// Enable network segmentation
    pub enable_network_segmentation: bool,
    
    /// Enable encryption at rest
    pub enable_encryption_at_rest: bool,
    
    /// Enable vulnerability scanning
    pub enable_vulnerability_scanning: bool,
}

/// GDPR compliance configuration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GDPRConfig {
    /// Enable data subject rights
    pub enable_data_subject_rights: bool,
    
    /// Data retention period in days
    pub data_retention_days: u32,
    
    /// Enable privacy by design
    pub enable_privacy_by_design: bool,
    
    /// Enable data breach notifications
    pub enable_breach_notifications: bool,
}

/// Basel III compliance configuration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BaselConfig {
    /// Enable operational risk management
    pub enable_operational_risk: bool,
    
    /// Capital adequacy ratio threshold
    pub capital_adequacy_threshold: f64,
    
    /// Enable stress testing
    pub enable_stress_testing: bool,
    
    /// Risk reporting frequency in hours
    pub risk_reporting_frequency_hours: u32,
}

/// MiFID II compliance configuration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MiFIDConfig {
    /// Enable transaction reporting
    pub enable_transaction_reporting: bool,
    
    /// Enable best execution reporting
    pub enable_best_execution: bool,
    
    /// Enable market making reporting
    pub enable_market_making: bool,
    
    /// Transaction reporting latency in milliseconds
    pub reporting_latency_ms: u64,
}

/// AML/KYC compliance configuration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AMLKYCConfig {
    /// Enable customer due diligence
    pub enable_cdd: bool,
    
    /// Enable enhanced due diligence
    pub enable_edd: bool,
    
    /// Enable suspicious activity reporting
    pub enable_sar: bool,
    
    /// Transaction monitoring threshold
    pub monitoring_threshold: f64,
}

impl Default for ComplianceConfig {
    fn default() -> Self {
        Self {
            sox: SOXConfig {
                enable_internal_controls: true,
                audit_retention_days: 2555, // 7 years
                enable_segregation_duties: true,
                enable_change_controls: true,
            },
            pci_dss: PCIDSSConfig {
                compliance_level: 1,
                enable_network_segmentation: true,
                enable_encryption_at_rest: true,
                enable_vulnerability_scanning: true,
            },
            gdpr: GDPRConfig {
                enable_data_subject_rights: true,
                data_retention_days: 2555, // 7 years for financial records
                enable_privacy_by_design: true,
                enable_breach_notifications: true,
            },
            basel: BaselConfig {
                enable_operational_risk: true,
                capital_adequacy_threshold: 0.08, // 8%
                enable_stress_testing: true,
                risk_reporting_frequency_hours: 24,
            },
            mifid: MiFIDConfig {
                enable_transaction_reporting: true,
                enable_best_execution: true,
                enable_market_making: true,
                reporting_latency_ms: 1000, // 1 second
            },
            aml_kyc: AMLKYCConfig {
                enable_cdd: true,
                enable_edd: true,
                enable_sar: true,
                monitoring_threshold: 10000.0, // $10,000 threshold
            },
        }
    }
}