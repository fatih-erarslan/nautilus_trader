//! Data Integrity Watchdog Implementation
//! 
//! Combines Compliance Sentinel functionality with enhanced data integrity enforcement

use crate::{TENGRIError, TradingOperation, TENGRIOversightResult, ViolationType};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use chrono::{DateTime, Utc};
use ring::digest::{Context, SHA256};
use serde::{Deserialize, Serialize};

/// Data integrity validation result
#[derive(Debug, Clone)]
pub enum DataIntegrityResult {
    Valid,
    IntegrityBreach { 
        breach_type: IntegrityBreachType, 
        severity: IntegritySeverity,
        evidence: Vec<u8>,
    },
    SuspiciousData { 
        confidence: f64, 
        anomalies: Vec<String> 
    },
}

#[derive(Debug, Clone)]
pub enum IntegrityBreachType {
    HashMismatch,
    TimestampAnomaly,
    SourceCorruption,
    UnauthorizedModification,
    MissingProvenanceData,
}

#[derive(Debug, Clone)]
pub enum IntegritySeverity {
    Critical,  // Immediate shutdown required
    High,      // Quarantine required
    Medium,    // Alert and monitor
    Low,       // Log for investigation
}

/// Data integrity record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrityRecord {
    pub data_id: String,
    pub hash: String,
    pub timestamp: DateTime<Utc>,
    pub source: String,
    pub chain_position: u64,
    pub previous_hash: Option<String>,
    pub verification_status: VerificationStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VerificationStatus {
    Verified,
    Pending,
    Failed { reason: String },
    Suspicious { confidence: f64 },
}

/// Data integrity watchdog
pub struct DataIntegrityWatchdog {
    integrity_chain: Arc<RwLock<HashMap<String, IntegrityRecord>>>,
    hash_context: Arc<RwLock<Context>>,
    breach_counter: Arc<RwLock<u64>>,
    verification_cache: Arc<RwLock<HashMap<String, DataIntegrityResult>>>,
}

impl DataIntegrityWatchdog {
    /// Create new data integrity watchdog
    pub async fn new() -> Result<Self, TENGRIError> {
        let integrity_chain = Arc::new(RwLock::new(HashMap::new()));
        let hash_context = Arc::new(RwLock::new(Context::new(&SHA256)));
        let breach_counter = Arc::new(RwLock::new(0));
        let verification_cache = Arc::new(RwLock::new(HashMap::new()));

        Ok(Self {
            integrity_chain,
            hash_context,
            breach_counter,
            verification_cache,
        })
    }

    /// Validate data integrity for trading operation
    pub async fn validate(&self, operation: &TradingOperation) -> Result<TENGRIOversightResult, TENGRIError> {
        // Check cache first
        if let Some(cached_result) = self.check_cache(&operation.data_source).await {
            return self.convert_integrity_result(cached_result);
        }

        // Comprehensive integrity validation
        let hash_validation = self.validate_data_hash(&operation.data_source).await?;
        let timestamp_validation = self.validate_timestamp(&operation.timestamp).await?;
        let source_validation = self.validate_source_integrity(&operation.data_source).await?;
        let chain_validation = self.validate_integrity_chain(&operation.data_source).await?;

        // Aggregate validation results
        let final_result = self.aggregate_integrity_results(vec![
            hash_validation,
            timestamp_validation,
            source_validation,
            chain_validation,
        ]).await?;

        // Cache result for future use
        self.cache_result(&operation.data_source, final_result.clone()).await;

        self.convert_integrity_result(final_result)
    }

    /// Validate data hash integrity
    async fn validate_data_hash(&self, data: &str) -> Result<DataIntegrityResult, TENGRIError> {
        let mut context = self.hash_context.write().await;
        context.update(data.as_bytes());
        let computed_hash = context.clone().finish();

        // Check against known good hashes (simplified implementation)
        let hash_hex = hex::encode(computed_hash.as_ref());
        
        // Retrieve stored hash if it exists
        let chain = self.integrity_chain.read().await;
        if let Some(record) = chain.get(data) {
            if record.hash != hash_hex {
                return Ok(DataIntegrityResult::IntegrityBreach {
                    breach_type: IntegrityBreachType::HashMismatch,
                    severity: IntegritySeverity::Critical,
                    evidence: computed_hash.as_ref().to_vec(),
                });
            }
        } else {
            // First time seeing this data - create new integrity record
            let new_record = IntegrityRecord {
                data_id: hash_hex.clone(),
                hash: hash_hex,
                timestamp: Utc::now(),
                source: data.to_string(),
                chain_position: chain.len() as u64,
                previous_hash: None,
                verification_status: VerificationStatus::Verified,
            };
            drop(chain);
            
            let mut chain_write = self.integrity_chain.write().await;
            chain_write.insert(data.to_string(), new_record);
        }

        Ok(DataIntegrityResult::Valid)
    }

    /// Validate timestamp consistency
    async fn validate_timestamp(&self, timestamp: &DateTime<Utc>) -> Result<DataIntegrityResult, TENGRIError> {
        let now = Utc::now();
        let time_diff = now.signed_duration_since(*timestamp);

        // Check for future timestamps (suspicious)
        if time_diff.num_seconds() < -60 { // Allow 1 minute clock skew
            return Ok(DataIntegrityResult::IntegrityBreach {
                breach_type: IntegrityBreachType::TimestampAnomaly,
                severity: IntegritySeverity::High,
                evidence: timestamp.to_rfc3339().into_bytes(),
            });
        }

        // Check for extremely old timestamps (might indicate replay attacks)
        if time_diff.num_hours() > 24 {
            return Ok(DataIntegrityResult::SuspiciousData {
                confidence: 0.7,
                anomalies: vec!["Timestamp older than 24 hours".to_string()],
            });
        }

        Ok(DataIntegrityResult::Valid)
    }

    /// Validate source integrity
    async fn validate_source_integrity(&self, source: &str) -> Result<DataIntegrityResult, TENGRIError> {
        // Check for known compromised sources
        let suspicious_patterns = vec![
            "localhost",
            "127.0.0.1",
            "test",
            "mock",
            "fake",
            "dummy",
        ];

        for pattern in suspicious_patterns {
            if source.to_lowercase().contains(pattern) {
                return Ok(DataIntegrityResult::SuspiciousData {
                    confidence: 0.8,
                    anomalies: vec![format!("Source contains suspicious pattern: {}", pattern)],
                });
            }
        }

        // Additional source validation logic would go here
        // (DNS verification, certificate validation, etc.)

        Ok(DataIntegrityResult::Valid)
    }

    /// Validate integrity chain consistency
    async fn validate_integrity_chain(&self, data_id: &str) -> Result<DataIntegrityResult, TENGRIError> {
        let chain = self.integrity_chain.read().await;
        
        if let Some(record) = chain.get(data_id) {
            // Verify chain linkage
            if let Some(prev_hash) = &record.previous_hash {
                let has_previous = chain.values().any(|r| r.hash == *prev_hash);
                if !has_previous {
                    return Ok(DataIntegrityResult::IntegrityBreach {
                        breach_type: IntegrityBreachType::MissingProvenanceData,
                        severity: IntegritySeverity::High,
                        evidence: prev_hash.as_bytes().to_vec(),
                    });
                }
            }
        }

        Ok(DataIntegrityResult::Valid)
    }

    /// Aggregate multiple integrity validation results
    async fn aggregate_integrity_results(
        &self,
        results: Vec<DataIntegrityResult>,
    ) -> Result<DataIntegrityResult, TENGRIError> {
        // Critical breaches take precedence
        for result in &results {
            if let DataIntegrityResult::IntegrityBreach { severity: IntegritySeverity::Critical, .. } = result {
                let mut counter = self.breach_counter.write().await;
                *counter += 1;
                return Ok(result.clone());
            }
        }

        // High severity breaches
        for result in &results {
            if let DataIntegrityResult::IntegrityBreach { severity: IntegritySeverity::High, .. } = result {
                return Ok(result.clone());
            }
        }

        // Aggregate suspicious data
        let suspicious_results: Vec<_> = results.iter().filter_map(|r| match r {
            DataIntegrityResult::SuspiciousData { confidence, anomalies } => Some((confidence, anomalies)),
            _ => None,
        }).collect();

        if !suspicious_results.is_empty() {
            let avg_confidence = suspicious_results.iter().map(|(c, _)| *c).sum::<f64>() / suspicious_results.len() as f64;
            let all_anomalies: Vec<String> = suspicious_results
                .into_iter()
                .flat_map(|(_, anomalies)| anomalies.clone())
                .collect();

            if avg_confidence > 0.7 || all_anomalies.len() > 2 {
                return Ok(DataIntegrityResult::SuspiciousData {
                    confidence: avg_confidence,
                    anomalies: all_anomalies,
                });
            }
        }

        Ok(DataIntegrityResult::Valid)
    }

    /// Convert integrity result to TENGRI oversight result
    fn convert_integrity_result(&self, result: DataIntegrityResult) -> Result<TENGRIOversightResult, TENGRIError> {
        match result {
            DataIntegrityResult::Valid => Ok(TENGRIOversightResult::Approved),
            
            DataIntegrityResult::IntegrityBreach { breach_type, severity, evidence } => {
                match severity {
                    IntegritySeverity::Critical => Ok(TENGRIOversightResult::CriticalViolation {
                        violation_type: ViolationType::IntegrityBreach,
                        immediate_shutdown: true,
                        forensic_data: evidence,
                    }),
                    IntegritySeverity::High => Ok(TENGRIOversightResult::Rejected {
                        reason: format!("High severity integrity breach: {:?}", breach_type),
                        emergency_action: crate::EmergencyAction::QuarantineAgent {
                            agent_id: "data_source_agent".to_string(),
                        },
                    }),
                    IntegritySeverity::Medium | IntegritySeverity::Low => Ok(TENGRIOversightResult::Warning {
                        reason: format!("Integrity concern: {:?}", breach_type),
                        corrective_action: "Investigate data source and verify integrity".to_string(),
                    }),
                }
            },
            
            DataIntegrityResult::SuspiciousData { confidence, anomalies } => {
                if confidence > 0.8 {
                    Ok(TENGRIOversightResult::Rejected {
                        reason: format!("Suspicious data detected: {}", anomalies.join(", ")),
                        emergency_action: crate::EmergencyAction::AlertOperators,
                    })
                } else {
                    Ok(TENGRIOversightResult::Warning {
                        reason: format!("Data anomalies detected: {}", anomalies.join(", ")),
                        corrective_action: "Monitor data source closely".to_string(),
                    })
                }
            },
        }
    }

    /// Check cache for previous validation result
    async fn check_cache(&self, data_id: &str) -> Option<DataIntegrityResult> {
        let cache = self.verification_cache.read().await;
        cache.get(data_id).cloned()
    }

    /// Cache validation result
    async fn cache_result(&self, data_id: &str, result: DataIntegrityResult) {
        let mut cache = self.verification_cache.write().await;
        cache.insert(data_id.to_string(), result);

        // Limit cache size
        if cache.len() > 10000 {
            let oldest_key = cache.keys().next().unwrap().clone();
            cache.remove(&oldest_key);
        }
    }

    /// Get integrity statistics
    pub async fn get_integrity_statistics(&self) -> IntegrityStatistics {
        let chain = self.integrity_chain.read().await;
        let breach_counter = self.breach_counter.read().await;
        let cache = self.verification_cache.read().await;

        IntegrityStatistics {
            total_records: chain.len(),
            total_breaches: *breach_counter,
            cache_size: cache.len(),
            verified_records: chain.values().filter(|r| matches!(r.verification_status, VerificationStatus::Verified)).count(),
            failed_records: chain.values().filter(|r| matches!(r.verification_status, VerificationStatus::Failed { .. })).count(),
        }
    }
}

/// Integrity statistics
#[derive(Debug, Clone)]
pub struct IntegrityStatistics {
    pub total_records: usize,
    pub total_breaches: u64,
    pub cache_size: usize,
    pub verified_records: usize,
    pub failed_records: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio_test;

    #[tokio::test]
    async fn test_data_integrity_validation() {
        let watchdog = DataIntegrityWatchdog::new().await.unwrap();
        
        let operation = TradingOperation {
            id: uuid::Uuid::new_v4(),
            timestamp: Utc::now(),
            operation_type: crate::OperationType::PlaceOrder,
            data_source: "authentic_market_data".to_string(),
            mathematical_model: "test_model".to_string(),
            risk_parameters: crate::RiskParameters {
                max_position_size: 1000.0,
                stop_loss: Some(0.02),
                take_profit: Some(0.05),
                confidence_threshold: 0.95,
            },
            agent_id: "test_agent".to_string(),
        };
        
        let result = watchdog.validate(&operation).await.unwrap();
        assert!(matches!(result, TENGRIOversightResult::Approved));
    }

    #[tokio::test]
    async fn test_suspicious_source_detection() {
        let watchdog = DataIntegrityWatchdog::new().await.unwrap();
        
        let operation = TradingOperation {
            id: uuid::Uuid::new_v4(),
            timestamp: Utc::now(),
            operation_type: crate::OperationType::PlaceOrder,
            data_source: "localhost:8080/mock_data".to_string(),
            mathematical_model: "test_model".to_string(),
            risk_parameters: crate::RiskParameters {
                max_position_size: 1000.0,
                stop_loss: Some(0.02),
                take_profit: Some(0.05),
                confidence_threshold: 0.95,
            },
            agent_id: "test_agent".to_string(),
        };
        
        let result = watchdog.validate(&operation).await.unwrap();
        assert!(matches!(result, TENGRIOversightResult::Warning { .. }));
    }
}