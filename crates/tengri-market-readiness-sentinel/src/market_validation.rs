//! Market data validation framework for production trading systems
//!
//! This module provides comprehensive market data validation including:
//! - Real-time data feed validation
//! - Data quality checks
//! - Latency monitoring
//! - Failover mechanisms

use std::sync::Arc;
use std::collections::HashMap;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::{info, warn, error};
use chrono::{DateTime, Utc};
use uuid::Uuid;

use crate::config::MarketReadinessConfig;
use crate::types::ValidationResult;
use crate::{LiquidityStatus, VolumeProfile, ValidationStatus};
use crate::error::MarketReadinessError;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketDataHealth {
    pub feed_status: FeedStatus,
    pub latency_ms: u64,
    pub message_rate: f64,
    pub gap_count: u64,
    pub last_update: DateTime<Utc>,
    pub quality_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FeedStatus {
    Active,
    Stale,
    Disconnected,
    Recovering,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataQualityMetrics {
    pub completeness: f64,
    pub consistency: f64,
    pub freshness: f64,
    pub accuracy: f64,
    pub availability: f64,
}

#[derive(Debug)]
pub struct MarketValidator {
    config: Arc<MarketReadinessConfig>,
    data_sources: Arc<RwLock<HashMap<String, MarketDataHealth>>>,
    quality_metrics: Arc<RwLock<DataQualityMetrics>>,
    failover_manager: Arc<FailoverManager>,
}

impl MarketValidator {
    pub async fn new(config: Arc<MarketReadinessConfig>) -> Result<Self> {
        let failover_manager = Arc::new(FailoverManager::new(config.clone()).await?);
        
        Ok(Self {
            config,
            data_sources: Arc::new(RwLock::new(HashMap::new())),
            quality_metrics: Arc::new(RwLock::new(DataQualityMetrics {
                completeness: 100.0,
                consistency: 100.0,
                freshness: 100.0,
                accuracy: 100.0,
                availability: 100.0,
            })),
            failover_manager,
        })
    }

    pub async fn initialize(&self) -> Result<()> {
        info!("Initializing market data validator...");\n        
        // Initialize data sources monitoring
        self.initialize_data_sources().await?;
        
        // Start quality monitoring
        self.start_quality_monitoring().await?;
        
        // Initialize failover manager
        self.failover_manager.initialize().await?;
        
        info!("Market data validator initialized successfully");
        Ok(())
    }

    pub async fn validate_market_data(&self) -> Result<ValidationResult> {
        let start_time = std::time::Instant::now();
        let mut issues = Vec::new();
        
        // Validate data feeds
        let feed_validation = self.validate_data_feeds().await?;
        if feed_validation.status == ValidationStatus::Failed {
            issues.push(format!("Data feed validation failed: {}", feed_validation.message));
        }
        
        // Validate data quality
        let quality_validation = self.validate_data_quality().await?;
        if quality_validation.status == ValidationStatus::Failed {
            issues.push(format!("Data quality validation failed: {}", quality_validation.message));
        }
        
        // Validate latency
        let latency_validation = self.validate_latency().await?;
        if latency_validation.status == ValidationStatus::Failed {
            issues.push(format!("Latency validation failed: {}", latency_validation.message));
        }
        
        // Validate failover readiness
        let failover_validation = self.validate_failover_readiness().await?;
        if failover_validation.status == ValidationStatus::Failed {
            issues.push(format!("Failover validation failed: {}", failover_validation.message));
        }
        
        let duration = start_time.elapsed().as_millis() as u64;
        
        let result = if issues.is_empty() {
            ValidationResult {
                status: ValidationStatus::Passed,
                message: "All market data validations passed".to_string(),
                details: Some(serde_json::json!({
                    "feeds_validated": self.get_feed_count().await,
                    "quality_score": self.get_overall_quality_score().await?,
                    "average_latency": self.get_average_latency().await?,
                })),
                timestamp: Utc::now(),
                duration_ms: duration,
                confidence: 0.95,
            }
        } else {
            ValidationResult {
                status: ValidationStatus::Failed,
                message: format!("Market data validation failed: {}", issues.join(", ")),
                details: Some(serde_json::json!({
                    "issues": issues,
                    "failed_feeds": self.get_failed_feeds().await,
                })),
                timestamp: Utc::now(),
                duration_ms: duration,
                confidence: 0.8,
            }
        };
        
        Ok(result)
    }

    async fn validate_data_feeds(&self) -> Result<ValidationResult> {
        let data_sources = self.data_sources.read().await;
        let mut failed_feeds = Vec::new();
        let mut stale_feeds = Vec::new();
        
        for (feed_name, health) in data_sources.iter() {
            match health.feed_status {
                FeedStatus::Disconnected => {
                    failed_feeds.push(feed_name.clone());
                },
                FeedStatus::Stale => {
                    stale_feeds.push(feed_name.clone());
                },
                _ => {}
            }
        }
        
        if !failed_feeds.is_empty() {
            return Ok(ValidationResult {
                status: ValidationStatus::Failed,
                message: format!("Data feeds disconnected: {}", failed_feeds.join(", ")),
                details: Some(serde_json::json!({
                    "failed_feeds": failed_feeds,
                    "stale_feeds": stale_feeds,
                })),
                timestamp: Utc::now(),
                duration_ms: 0,
                confidence: 1.0,
            });
        }
        
        if !stale_feeds.is_empty() {
            return Ok(ValidationResult {
                status: ValidationStatus::Warning,
                message: format!("Data feeds stale: {}", stale_feeds.join(", ")),
                details: Some(serde_json::json!({
                    "stale_feeds": stale_feeds,
                })),
                timestamp: Utc::now(),
                duration_ms: 0,
                confidence: 0.9,
            });
        }
        
        Ok(ValidationResult {
            status: ValidationStatus::Passed,
            message: "All data feeds are active".to_string(),
            details: None,
            timestamp: Utc::now(),
            duration_ms: 0,
            confidence: 1.0,
        })
    }

    async fn validate_data_quality(&self) -> Result<ValidationResult> {
        let quality_metrics = self.quality_metrics.read().await;
        let min_threshold = 95.0;
        
        let mut issues = Vec::new();
        
        if quality_metrics.completeness < min_threshold {
            issues.push(format!("Data completeness below threshold: {:.1}%", quality_metrics.completeness));
        }
        
        if quality_metrics.consistency < min_threshold {
            issues.push(format!("Data consistency below threshold: {:.1}%", quality_metrics.consistency));
        }
        
        if quality_metrics.freshness < min_threshold {
            issues.push(format!("Data freshness below threshold: {:.1}%", quality_metrics.freshness));
        }
        
        if quality_metrics.accuracy < min_threshold {
            issues.push(format!("Data accuracy below threshold: {:.1}%", quality_metrics.accuracy));
        }
        
        if quality_metrics.availability < min_threshold {
            issues.push(format!("Data availability below threshold: {:.1}%", quality_metrics.availability));
        }
        
        if !issues.is_empty() {
            return Ok(ValidationResult {
                status: ValidationStatus::Failed,
                message: format!("Data quality issues: {}", issues.join(", ")),
                details: Some(serde_json::json!({
                    "quality_metrics": *quality_metrics,
                    "issues": issues,
                })),
                timestamp: Utc::now(),
                duration_ms: 0,
                confidence: 0.95,
            });
        }
        
        Ok(ValidationResult {
            status: ValidationStatus::Passed,
            message: "Data quality meets all thresholds".to_string(),
            details: Some(serde_json::json!({
                "quality_metrics": *quality_metrics,
            })),
            timestamp: Utc::now(),
            duration_ms: 0,
            confidence: 1.0,
        })
    }

    async fn validate_latency(&self) -> Result<ValidationResult> {
        let data_sources = self.data_sources.read().await;
        let max_latency_ms = self.config.max_latency_ms;
        let mut high_latency_feeds = Vec::new();
        let mut total_latency = 0u64;
        let mut feed_count = 0;
        
        for (feed_name, health) in data_sources.iter() {
            total_latency += health.latency_ms;
            feed_count += 1;
            
            if health.latency_ms > max_latency_ms {
                high_latency_feeds.push((feed_name.clone(), health.latency_ms));
            }
        }
        
        let average_latency = if feed_count > 0 {
            total_latency / feed_count as u64
        } else {
            0
        };
        
        if !high_latency_feeds.is_empty() {
            return Ok(ValidationResult {
                status: ValidationStatus::Warning,
                message: format!("High latency detected in {} feeds", high_latency_feeds.len()),
                details: Some(serde_json::json!({
                    "high_latency_feeds": high_latency_feeds,
                    "average_latency": average_latency,
                    "threshold": max_latency_ms,
                })),
                timestamp: Utc::now(),
                duration_ms: 0,
                confidence: 0.9,
            });
        }
        
        Ok(ValidationResult {
            status: ValidationStatus::Passed,
            message: "All feed latencies within acceptable limits".to_string(),
            details: Some(serde_json::json!({
                "average_latency": average_latency,
                "threshold": max_latency_ms,
            })),
            timestamp: Utc::now(),
            duration_ms: 0,
            confidence: 1.0,
        })
    }

    async fn validate_failover_readiness(&self) -> Result<ValidationResult> {
        self.failover_manager.validate_failover_readiness().await
    }

    async fn initialize_data_sources(&self) -> Result<()> {
        // Initialize monitoring for configured data sources
        let mut data_sources = self.data_sources.write().await;
        
        for source in &self.config.data_sources {
            data_sources.insert(source.clone(), MarketDataHealth {
                feed_status: FeedStatus::Active,
                latency_ms: 0,
                message_rate: 0.0,
                gap_count: 0,
                last_update: Utc::now(),
                quality_score: 100.0,
            });
        }
        
        Ok(())
    }

    async fn start_quality_monitoring(&self) -> Result<()> {
        // Start background task for quality monitoring
        let quality_metrics = self.quality_metrics.clone();
        let data_sources = self.data_sources.clone();
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(60));
            
            loop {
                interval.tick().await;
                
                // Update quality metrics based on data sources
                let sources = data_sources.read().await;
                let mut quality = quality_metrics.write().await;
                
                let active_feeds = sources.values().filter(|h| matches!(h.feed_status, FeedStatus::Active)).count();
                let total_feeds = sources.len();
                
                quality.availability = if total_feeds > 0 {
                    (active_feeds as f64 / total_feeds as f64) * 100.0
                } else {
                    0.0
                };
                
                // Update other quality metrics based on actual data analysis
                quality.completeness = calculate_completeness(&sources);
                quality.consistency = calculate_consistency(&sources);
                quality.freshness = calculate_freshness(&sources);
                quality.accuracy = calculate_accuracy(&sources);
            }
        });
        
        Ok(())
    }

    pub async fn get_liquidity_status(&self) -> Result<LiquidityStatus> {
        let data_sources = self.data_sources.read().await;
        
        // Simplified liquidity assessment based on message rates
        let total_message_rate: f64 = data_sources.values().map(|h| h.message_rate).sum();
        let feed_count = data_sources.len() as f64;
        
        let avg_message_rate = if feed_count > 0.0 {
            total_message_rate / feed_count
        } else {
            0.0
        };
        
        let status = if avg_message_rate > 1000.0 {
            LiquidityStatus::Abundant
        } else if avg_message_rate > 500.0 {
            LiquidityStatus::Normal
        } else if avg_message_rate > 100.0 {
            LiquidityStatus::Scarce
        } else {
            LiquidityStatus::Dry
        };
        
        Ok(status)
    }

    pub async fn get_current_spread(&self) -> Result<f64> {
        // Simplified spread calculation
        Ok(0.01) // 1 basis point default
    }

    pub async fn get_volume_profile(&self) -> Result<VolumeProfile> {
        // Simplified volume profile calculation
        Ok(VolumeProfile {
            average_volume: 1000000.0,
            current_volume: 1200000.0,
            volume_ratio: 1.2,
            participation_rate: 0.15,
        })
    }

    async fn get_feed_count(&self) -> usize {
        self.data_sources.read().await.len()
    }

    async fn get_overall_quality_score(&self) -> Result<f64> {
        let quality_metrics = self.quality_metrics.read().await;
        let overall_score = (quality_metrics.completeness + quality_metrics.consistency + 
                           quality_metrics.freshness + quality_metrics.accuracy + 
                           quality_metrics.availability) / 5.0;
        Ok(overall_score)
    }

    async fn get_average_latency(&self) -> Result<f64> {
        let data_sources = self.data_sources.read().await;
        let total_latency: u64 = data_sources.values().map(|h| h.latency_ms).sum();
        let feed_count = data_sources.len();
        
        if feed_count > 0 {
            Ok(total_latency as f64 / feed_count as f64)
        } else {
            Ok(0.0)
        }
    }

    async fn get_failed_feeds(&self) -> Vec<String> {
        let data_sources = self.data_sources.read().await;
        data_sources.iter()
            .filter_map(|(name, health)| {
                if matches!(health.feed_status, FeedStatus::Disconnected) {
                    Some(name.clone())
                } else {
                    None
                }
            })
            .collect()
    }
}

#[derive(Debug)]
struct FailoverManager {
    config: Arc<MarketReadinessConfig>,
    backup_sources: Arc<RwLock<HashMap<String, BackupSource>>>,
}

#[derive(Debug, Clone)]
struct BackupSource {
    name: String,
    priority: u8,
    status: BackupStatus,
    last_test: DateTime<Utc>,
}

#[derive(Debug, Clone)]
enum BackupStatus {
    Ready,
    Active,
    Failed,
    Testing,
}

impl FailoverManager {
    async fn new(config: Arc<MarketReadinessConfig>) -> Result<Self> {
        Ok(Self {
            config,
            backup_sources: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    async fn initialize(&self) -> Result<()> {
        // Initialize backup sources monitoring
        let mut backup_sources = self.backup_sources.write().await;
        
        for (i, source) in self.config.backup_sources.iter().enumerate() {
            backup_sources.insert(source.clone(), BackupSource {
                name: source.clone(),
                priority: i as u8,
                status: BackupStatus::Ready,
                last_test: Utc::now(),
            });
        }
        
        Ok(())
    }

    async fn validate_failover_readiness(&self) -> Result<ValidationResult> {
        let backup_sources = self.backup_sources.read().await;
        let ready_backups = backup_sources.values()
            .filter(|source| matches!(source.status, BackupStatus::Ready))
            .count();
        
        let total_backups = backup_sources.len();
        
        if ready_backups == 0 {
            return Ok(ValidationResult {
                status: ValidationStatus::Failed,
                message: "No backup sources available for failover".to_string(),
                details: Some(serde_json::json!({
                    "ready_backups": ready_backups,
                    "total_backups": total_backups,
                })),
                timestamp: Utc::now(),
                duration_ms: 0,
                confidence: 1.0,
            });
        }
        
        if ready_backups < total_backups / 2 {
            return Ok(ValidationResult {
                status: ValidationStatus::Warning,
                message: format!("Only {} of {} backup sources ready", ready_backups, total_backups),
                details: Some(serde_json::json!({
                    "ready_backups": ready_backups,
                    "total_backups": total_backups,
                })),
                timestamp: Utc::now(),
                duration_ms: 0,
                confidence: 0.8,
            });
        }
        
        Ok(ValidationResult {
            status: ValidationStatus::Passed,
            message: "Failover system ready".to_string(),
            details: Some(serde_json::json!({
                "ready_backups": ready_backups,
                "total_backups": total_backups,
            })),
            timestamp: Utc::now(),
            duration_ms: 0,
            confidence: 1.0,
        })
    }
}

// Helper functions for quality calculations
fn calculate_completeness(sources: &HashMap<String, MarketDataHealth>) -> f64 {
    // Calculate data completeness based on gap counts
    let total_gaps: u64 = sources.values().map(|h| h.gap_count).sum();
    let feed_count = sources.len() as u64;
    
    if feed_count > 0 {
        let avg_gaps_per_feed = total_gaps as f64 / feed_count as f64;
        (100.0 - avg_gaps_per_feed).max(0.0)
    } else {
        0.0
    }
}

fn calculate_consistency(sources: &HashMap<String, MarketDataHealth>) -> f64 {
    // Calculate data consistency based on quality scores
    let total_quality: f64 = sources.values().map(|h| h.quality_score).sum();
    let feed_count = sources.len() as f64;
    
    if feed_count > 0 {
        total_quality / feed_count
    } else {
        0.0
    }
}

fn calculate_freshness(sources: &HashMap<String, MarketDataHealth>) -> f64 {
    let now = Utc::now();
    let mut fresh_count = 0;
    let total_count = sources.len();
    
    for health in sources.values() {
        let age = now.signed_duration_since(health.last_update);
        if age.num_seconds() < 60 { // Consider fresh if updated within last minute
            fresh_count += 1;
        }
    }
    
    if total_count > 0 {
        (fresh_count as f64 / total_count as f64) * 100.0
    } else {
        0.0
    }
}

fn calculate_accuracy(sources: &HashMap<String, MarketDataHealth>) -> f64 {
    // Calculate accuracy based on message rates and expected rates
    let mut accurate_count = 0;
    let total_count = sources.len();
    
    for health in sources.values() {
        // Simplified accuracy check - message rate within expected range
        if health.message_rate > 10.0 && health.message_rate < 10000.0 {
            accurate_count += 1;
        }
    }
    
    if total_count > 0 {
        (accurate_count as f64 / total_count as f64) * 100.0
    } else {
        0.0
    }
}