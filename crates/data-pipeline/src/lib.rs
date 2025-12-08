//! # Enterprise Data Pipeline Integrity Framework
//!
//! Comprehensive data validation and integrity system for enterprise trading operations.
//! Supports real-time validation, audit compliance, and blockchain-based verification.

mod config;
mod error;
mod types;
mod validation;
mod features;
mod pipeline;
mod fusion;
mod indicators;
mod streaming;
mod sentiment;
mod monitoring;
mod utils;
mod agents;

// Enterprise data integrity modules
mod integrity;
mod lineage;
mod reconciliation;
mod audit;
mod encryption;
mod recovery;
mod blockchain_verification;
mod quality_scoring;

// Real-time validation modules
mod realtime_validation;
mod transformation_validation;
mod consistency_failover;

// Re-exports
pub use config::*;
pub use error::*;
pub use types::*;
pub use validation::*;
pub use features::*;
pub use pipeline::*;
pub use fusion::*;
pub use indicators::*;
pub use streaming::*;
pub use sentiment::*;
pub use monitoring::*;
pub use utils::*;
pub use agents::*;

// Enterprise integrity exports
pub use integrity::*;
pub use lineage::*;
pub use reconciliation::*;
pub use audit::*;
pub use encryption::*;
pub use recovery::*;
pub use blockchain_verification::*;
pub use quality_scoring::*;

// Real-time validation exports
pub use realtime_validation::*;
pub use transformation_validation::*;
pub use consistency_failover::*;

use std::sync::Arc;
use anyhow::Result;
use tracing::{info, error};

/// Enterprise Data Pipeline with integrity features
pub struct EnterpriseDataPipeline {
    integrity_manager: Arc<IntegrityManager>,
    lineage_tracker: Arc<LineageTracker>,
    reconciliation_engine: Arc<ReconciliationEngine>,
    audit_logger: Arc<AuditLogger>,
    encryption_manager: Arc<EncryptionManager>,
    recovery_manager: Arc<RecoveryManager>,
    blockchain_verifier: Arc<BlockchainVerifier>,
    quality_scorer: Arc<QualityScorer>,
}

impl EnterpriseDataPipeline {
    /// Initialize enterprise data pipeline with full integrity suite
    pub async fn new(config: PipelineConfig) -> Result<Self> {
        info!("Initializing Enterprise Data Pipeline with integrity framework");
        
        let integrity_manager = Arc::new(IntegrityManager::new(config.integrity.clone()).await?);
        let lineage_tracker = Arc::new(LineageTracker::new(config.lineage.clone()).await?);
        let reconciliation_engine = Arc::new(ReconciliationEngine::new(config.reconciliation.clone()).await?);
        let audit_logger = Arc::new(AuditLogger::new(config.audit.clone()).await?);
        let encryption_manager = Arc::new(EncryptionManager::new(config.encryption.clone()).await?);
        let recovery_manager = Arc::new(RecoveryManager::new(config.recovery.clone()).await?);
        let blockchain_verifier = Arc::new(BlockchainVerifier::new(config.blockchain.clone()).await?);
        let quality_scorer = Arc::new(QualityScorer::new(config.quality.clone()).await?);
        
        Ok(Self {
            integrity_manager,
            lineage_tracker,
            reconciliation_engine,
            audit_logger,
            encryption_manager,
            recovery_manager,
            blockchain_verifier,
            quality_scorer,
        })
    }
    
    /// Process data through complete integrity pipeline
    pub async fn process_with_integrity(&self, data: RawDataItem) -> Result<ValidatedDataItem> {
        // 1. Track data lineage
        let lineage_id = self.lineage_tracker.track_ingestion(&data).await?;
        
        // 2. Encrypt sensitive data
        let encrypted_data = self.encryption_manager.encrypt_data(data).await?;
        
        // 3. Validate data integrity
        let validation_result = self.integrity_manager.validate_comprehensive(&encrypted_data).await?;
        
        // 4. Score data quality
        let quality_score = self.quality_scorer.score_data(&encrypted_data).await?;
        
        // 5. Perform blockchain verification
        let blockchain_hash = self.blockchain_verifier.verify_and_hash(&encrypted_data).await?;
        
        // 6. Log audit trail
        self.audit_logger.log_processing(&lineage_id, &validation_result, quality_score).await?;
        
        // 7. Create validated item
        let validated_item = ValidatedDataItem {
            data: encrypted_data,
            lineage_id,
            quality_score,
            blockchain_hash,
            validation_timestamp: chrono::Utc::now(),
            audit_trail: validation_result.audit_info,
        };
        
        Ok(validated_item)
    }
    
    /// Reconcile data from multiple sources
    pub async fn reconcile_multi_source(&self, sources: Vec<DataSource>) -> Result<ReconciledDataSet> {
        self.reconciliation_engine.reconcile_sources(sources).await
    }
    
    /// Recover data from backup systems
    pub async fn recover_data(&self, recovery_request: RecoveryRequest) -> Result<RecoveredData> {
        self.recovery_manager.recover_data(recovery_request).await
    }
    
    /// Get comprehensive health status
    pub async fn health_status(&self) -> Result<PipelineHealthStatus> {
        let integrity_health = self.integrity_manager.health_check().await?;
        let lineage_health = self.lineage_tracker.health_check().await?;
        let reconciliation_health = self.reconciliation_engine.health_check().await?;
        let audit_health = self.audit_logger.health_check().await?;
        let encryption_health = self.encryption_manager.health_check().await?;
        let recovery_health = self.recovery_manager.health_check().await?;
        let blockchain_health = self.blockchain_verifier.health_check().await?;
        let quality_health = self.quality_scorer.health_check().await?;
        
        Ok(PipelineHealthStatus {
            overall_status: HealthStatus::Healthy, // Would aggregate component statuses
            components: vec![
                integrity_health,
                lineage_health,
                reconciliation_health,
                audit_health,
                encryption_health,
                recovery_health,
                blockchain_health,
                quality_health,
            ],
            last_check: chrono::Utc::now(),
        })
    }
}

/// Pipeline configuration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PipelineConfig {
    pub integrity: IntegrityConfig,
    pub lineage: LineageConfig,
    pub reconciliation: ReconciliationConfig,
    pub audit: AuditConfig,
    pub encryption: EncryptionConfig,
    pub recovery: RecoveryConfig,
    pub blockchain: BlockchainConfig,
    pub quality: QualityConfig,
}

/// Health status for pipeline components
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum HealthStatus {
    Healthy,
    Warning,
    Critical,
    Offline,
}

/// Overall pipeline health
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PipelineHealthStatus {
    pub overall_status: HealthStatus,
    pub components: Vec<ComponentHealth>,
    pub last_check: chrono::DateTime<chrono::Utc>,
}

/// Component health information
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ComponentHealth {
    pub component_name: String,
    pub status: HealthStatus,
    pub metrics: ComponentMetrics,
    pub issues: Vec<String>,
}

/// Component performance metrics
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ComponentMetrics {
    pub latency_ms: f64,
    pub throughput_per_sec: f64,
    pub error_rate: f64,
    pub memory_usage_mb: f64,
    pub cpu_usage_percent: f64,
}
