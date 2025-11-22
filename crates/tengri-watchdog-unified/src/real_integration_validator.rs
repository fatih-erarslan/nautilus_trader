//! Real Integration Validator Agent
//! 
//! Validates that all tests use authentic services and real data sources
//! Ensures genuine integration testing with live systems and authentic data flows

use crate::{TENGRIError, TradingOperation, TENGRIOversightResult, ViolationType};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::RwLock;
use chrono::{DateTime, Utc};
use uuid::Uuid;
use serde::{Deserialize, Serialize};
use std::time::Instant;
use regex::Regex;

/// Real Integration Validation Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealIntegrationConfig {
    pub require_live_endpoints: bool,
    pub require_authentic_data: bool,
    pub require_real_databases: bool,
    pub require_actual_services: bool,
    pub block_localhost_connections: bool,
    pub block_test_environments: bool,
    pub minimum_authenticity_threshold: f64,
    pub allowed_test_patterns: HashSet<String>,
    pub required_production_indicators: HashSet<String>,
    pub validation_timeout_ms: u64,
}

impl Default for RealIntegrationConfig {
    fn default() -> Self {
        let mut production_indicators = HashSet::new();
        production_indicators.insert("prod.".to_string());
        production_indicators.insert("production.".to_string());
        production_indicators.insert("live.".to_string());
        production_indicators.insert("real.".to_string());
        production_indicators.insert("authentic.".to_string());
        production_indicators.insert("market.".to_string());
        production_indicators.insert("exchange.".to_string());
        
        let mut allowed_patterns = HashSet::new();
        allowed_patterns.insert("integration_test".to_string());
        allowed_patterns.insert("system_test".to_string());
        allowed_patterns.insert("e2e_test".to_string());
        allowed_patterns.insert("real_data_test".to_string());
        
        Self {
            require_live_endpoints: true,
            require_authentic_data: true,
            require_real_databases: true,
            require_actual_services: true,
            block_localhost_connections: true,
            block_test_environments: true,
            minimum_authenticity_threshold: 0.95,
            allowed_test_patterns: allowed_patterns,
            required_production_indicators: production_indicators,
            validation_timeout_ms: 500,
        }
    }
}

/// Service Authenticity Validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceValidationResult {
    pub service_name: String,
    pub endpoint_url: String,
    pub is_authentic: bool,
    pub authenticity_score: f64,
    pub connection_type: ConnectionType,
    pub validation_evidence: Vec<String>,
    pub warning_flags: Vec<String>,
    pub validation_timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConnectionType {
    ProductionEndpoint,
    LiveMarketData,
    RealDatabase,
    AuthenticExchange,
    TestEnvironment,
    LocalHost,
    MockEndpoint,
    SyntheticService,
}

/// Data Source Authenticity Validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataSourceValidationResult {
    pub source_name: String,
    pub source_type: DataSourceType,
    pub is_authentic: bool,
    pub data_quality_score: f64,
    pub freshness_score: f64,
    pub volume_score: f64,
    pub validation_evidence: Vec<String>,
    pub suspicious_patterns: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataSourceType {
    LiveMarketFeed,
    RealTimeExchange,
    ProductionDatabase,
    AuthenticAPI,
    TestDataSet,
    SyntheticData,
    MockData,
    StaticFile,
}

/// Integration Test Validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationTestValidation {
    pub test_name: String,
    pub uses_real_services: bool,
    pub uses_authentic_data: bool,
    pub has_live_connections: bool,
    pub authenticity_score: f64,
    pub validation_details: Vec<ValidationDetail>,
    pub compliance_status: ComplianceStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationDetail {
    pub component: String,
    pub validation_type: String,
    pub result: bool,
    pub confidence: f64,
    pub evidence: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComplianceStatus {
    FullyCompliant,
    MinorViolations,
    MajorViolations,
    CriticalViolations,
}

/// Real Integration Validator Agent
pub struct RealIntegrationValidator {
    config: RealIntegrationConfig,
    service_registry: Arc<RwLock<HashMap<String, ServiceValidationResult>>>,
    data_source_registry: Arc<RwLock<HashMap<String, DataSourceValidationResult>>>,
    validation_history: Arc<RwLock<Vec<(DateTime<Utc>, IntegrationTestValidation)>>>,
    authenticity_patterns: Arc<RwLock<HashMap<String, Regex>>>,
    violation_counts: Arc<RwLock<HashMap<String, u64>>>,
}

impl RealIntegrationValidator {
    /// Initialize Real Integration Validator
    pub async fn new(config: RealIntegrationConfig) -> Result<Self, TENGRIError> {
        let service_registry = Arc::new(RwLock::new(HashMap::new()));
        let data_source_registry = Arc::new(RwLock::new(HashMap::new()));
        let validation_history = Arc::new(RwLock::new(Vec::new()));
        let authenticity_patterns = Arc::new(RwLock::new(Self::build_authenticity_patterns()?));
        let violation_counts = Arc::new(RwLock::new(HashMap::new()));

        Ok(Self {
            config,
            service_registry,
            data_source_registry,
            validation_history,
            authenticity_patterns,
            violation_counts,
        })
    }

    /// Validate real integration for trading operation
    pub async fn validate_real_integration(&self, operation: &TradingOperation) -> Result<TENGRIOversightResult, TENGRIError> {
        let validation_start = Instant::now();
        
        // Validate service authenticity
        let service_validation = self.validate_service_authenticity(operation).await?;
        
        // Validate data source authenticity
        let data_validation = self.validate_data_source_authenticity(operation).await?;
        
        // Validate integration test authenticity
        let integration_validation = self.validate_integration_test_authenticity(operation).await?;
        
        // Aggregate validation results
        let final_validation = self.aggregate_validation_results(
            service_validation,
            data_validation,
            integration_validation,
        ).await?;

        // Record validation history
        self.record_validation_history(&final_validation).await;

        // Convert to oversight result
        let oversight_result = self.convert_to_oversight_result(&final_validation).await?;

        let validation_duration = validation_start.elapsed();
        if validation_duration.as_millis() > self.config.validation_timeout_ms {
            tracing::warn!("Real integration validation exceeded timeout: {:?}", validation_duration);
        }

        Ok(oversight_result)
    }

    /// Validate service authenticity
    async fn validate_service_authenticity(&self, operation: &TradingOperation) -> Result<ServiceValidationResult, TENGRIError> {
        let mut authenticity_score = 0.0;
        let mut is_authentic = false;
        let mut evidence = Vec::new();
        let mut warning_flags = Vec::new();
        let mut connection_type = ConnectionType::MockEndpoint;

        // Check for production indicators
        for indicator in &self.config.required_production_indicators {
            if operation.data_source.contains(indicator) {
                authenticity_score += 0.2;
                is_authentic = true;
                evidence.push(format!("Production indicator found: {}", indicator));
                connection_type = ConnectionType::ProductionEndpoint;
            }
        }

        // Check for localhost/test environment patterns
        if self.config.block_localhost_connections {
            if operation.data_source.contains("localhost") || operation.data_source.contains("127.0.0.1") {
                authenticity_score -= 0.5;
                warning_flags.push("Localhost connection detected".to_string());
                connection_type = ConnectionType::LocalHost;
            }
        }

        // Check for test environment patterns
        if self.config.block_test_environments {
            let test_patterns = vec!["test.", "dev.", "staging.", "mock.", "fake."];
            for pattern in test_patterns {
                if operation.data_source.contains(pattern) {
                    authenticity_score -= 0.3;
                    warning_flags.push(format!("Test environment pattern detected: {}", pattern));
                    connection_type = ConnectionType::TestEnvironment;
                }
            }
        }

        // Validate against authenticity patterns
        let patterns = self.authenticity_patterns.read().await;
        for (pattern_name, regex) in patterns.iter() {
            if regex.is_match(&operation.data_source) {
                authenticity_score += 0.1;
                evidence.push(format!("Authenticity pattern matched: {}", pattern_name));
            }
        }

        // Determine final authenticity
        is_authentic = authenticity_score >= self.config.minimum_authenticity_threshold;

        let result = ServiceValidationResult {
            service_name: operation.agent_id.clone(),
            endpoint_url: operation.data_source.clone(),
            is_authentic,
            authenticity_score: authenticity_score.max(0.0).min(1.0),
            connection_type,
            validation_evidence: evidence,
            warning_flags,
            validation_timestamp: Utc::now(),
        };

        // Update service registry
        let mut registry = self.service_registry.write().await;
        registry.insert(operation.agent_id.clone(), result.clone());

        Ok(result)
    }

    /// Validate data source authenticity
    async fn validate_data_source_authenticity(&self, operation: &TradingOperation) -> Result<DataSourceValidationResult, TENGRIError> {
        let mut is_authentic = false;
        let mut data_quality_score = 0.0;
        let mut freshness_score = 0.0;
        let mut volume_score = 0.0;
        let mut evidence = Vec::new();
        let mut suspicious_patterns = Vec::new();
        let mut source_type = DataSourceType::MockData;

        // Check for live market data indicators
        let live_indicators = vec!["market", "exchange", "ticker", "quote", "orderbook", "trade"];
        for indicator in live_indicators {
            if operation.data_source.contains(indicator) {
                data_quality_score += 0.15;
                is_authentic = true;
                evidence.push(format!("Live market data indicator: {}", indicator));
                source_type = DataSourceType::LiveMarketFeed;
            }
        }

        // Check for real-time characteristics
        if operation.data_source.contains("real_time") || operation.data_source.contains("live") {
            freshness_score += 0.3;
            evidence.push("Real-time data source detected".to_string());
        }

        // Check for suspicious synthetic patterns
        let synthetic_patterns = vec!["generate", "random", "mock", "fake", "synthetic", "dummy"];
        for pattern in synthetic_patterns {
            if operation.data_source.contains(pattern) {
                data_quality_score -= 0.2;
                suspicious_patterns.push(format!("Synthetic pattern detected: {}", pattern));
                source_type = DataSourceType::SyntheticData;
            }
        }

        // Validate data volume characteristics
        if operation.data_source.len() > 100 {
            volume_score += 0.2;
            evidence.push("Substantial data source configuration".to_string());
        }

        // Determine final authenticity
        let overall_score = (data_quality_score + freshness_score + volume_score) / 3.0;
        is_authentic = overall_score >= self.config.minimum_authenticity_threshold;

        let result = DataSourceValidationResult {
            source_name: operation.data_source.clone(),
            source_type,
            is_authentic,
            data_quality_score: data_quality_score.max(0.0).min(1.0),
            freshness_score: freshness_score.max(0.0).min(1.0),
            volume_score: volume_score.max(0.0).min(1.0),
            validation_evidence: evidence,
            suspicious_patterns,
        };

        // Update data source registry
        let mut registry = self.data_source_registry.write().await;
        registry.insert(operation.data_source.clone(), result.clone());

        Ok(result)
    }

    /// Validate integration test authenticity
    async fn validate_integration_test_authenticity(&self, operation: &TradingOperation) -> Result<IntegrationTestValidation, TENGRIError> {
        let mut uses_real_services = false;
        let mut uses_authentic_data = false;
        let mut has_live_connections = false;
        let mut authenticity_score = 0.0;
        let mut validation_details = Vec::new();

        // Check for real service usage
        if self.config.require_actual_services {
            uses_real_services = !operation.data_source.contains("mock") && !operation.data_source.contains("stub");
            if uses_real_services {
                authenticity_score += 0.3;
                validation_details.push(ValidationDetail {
                    component: "Service Layer".to_string(),
                    validation_type: "Real Service Usage".to_string(),
                    result: true,
                    confidence: 0.9,
                    evidence: "No mock/stub patterns detected".to_string(),
                });
            }
        }

        // Check for authentic data usage
        if self.config.require_authentic_data {
            uses_authentic_data = operation.data_source.contains("market") || operation.data_source.contains("exchange");
            if uses_authentic_data {
                authenticity_score += 0.3;
                validation_details.push(ValidationDetail {
                    component: "Data Layer".to_string(),
                    validation_type: "Authentic Data Usage".to_string(),
                    result: true,
                    confidence: 0.8,
                    evidence: "Market/exchange data source detected".to_string(),
                });
            }
        }

        // Check for live connections
        if self.config.require_live_endpoints {
            has_live_connections = !operation.data_source.contains("localhost") && !operation.data_source.contains("127.0.0.1");
            if has_live_connections {
                authenticity_score += 0.4;
                validation_details.push(ValidationDetail {
                    component: "Network Layer".to_string(),
                    validation_type: "Live Connection".to_string(),
                    result: true,
                    confidence: 0.9,
                    evidence: "Non-localhost endpoint detected".to_string(),
                });
            }
        }

        // Determine compliance status
        let compliance_status = match authenticity_score {
            score if score >= 0.9 => ComplianceStatus::FullyCompliant,
            score if score >= 0.7 => ComplianceStatus::MinorViolations,
            score if score >= 0.5 => ComplianceStatus::MajorViolations,
            _ => ComplianceStatus::CriticalViolations,
        };

        Ok(IntegrationTestValidation {
            test_name: format!("operation_{}", operation.id),
            uses_real_services,
            uses_authentic_data,
            has_live_connections,
            authenticity_score: authenticity_score.max(0.0).min(1.0),
            validation_details,
            compliance_status,
        })
    }

    /// Aggregate validation results
    async fn aggregate_validation_results(
        &self,
        service_validation: ServiceValidationResult,
        data_validation: DataSourceValidationResult,
        integration_validation: IntegrationTestValidation,
    ) -> Result<IntegrationTestValidation, TENGRIError> {
        let mut final_validation = integration_validation;
        
        // Factor in service authenticity
        if !service_validation.is_authentic {
            final_validation.authenticity_score *= 0.8;
            final_validation.validation_details.push(ValidationDetail {
                component: "Service Authenticity".to_string(),
                validation_type: "Service Validation".to_string(),
                result: false,
                confidence: service_validation.authenticity_score,
                evidence: format!("Service not authentic: {}", service_validation.service_name),
            });
        }

        // Factor in data source authenticity
        if !data_validation.is_authentic {
            final_validation.authenticity_score *= 0.8;
            final_validation.validation_details.push(ValidationDetail {
                component: "Data Source Authenticity".to_string(),
                validation_type: "Data Source Validation".to_string(),
                result: false,
                confidence: data_validation.data_quality_score,
                evidence: format!("Data source not authentic: {}", data_validation.source_name),
            });
        }

        // Update compliance status based on aggregated score
        final_validation.compliance_status = match final_validation.authenticity_score {
            score if score >= 0.9 => ComplianceStatus::FullyCompliant,
            score if score >= 0.7 => ComplianceStatus::MinorViolations,
            score if score >= 0.5 => ComplianceStatus::MajorViolations,
            _ => ComplianceStatus::CriticalViolations,
        };

        Ok(final_validation)
    }

    /// Record validation history
    async fn record_validation_history(&self, validation: &IntegrationTestValidation) {
        let mut history = self.validation_history.write().await;
        history.push((Utc::now(), validation.clone()));

        // Keep only last 5,000 entries
        if history.len() > 5000 {
            history.drain(0..500);
        }

        // Update violation counts
        if !matches!(validation.compliance_status, ComplianceStatus::FullyCompliant) {
            let mut counts = self.violation_counts.write().await;
            *counts.entry("integration_violations".to_string()).or_insert(0) += 1;
        }
    }

    /// Convert to oversight result
    async fn convert_to_oversight_result(&self, validation: &IntegrationTestValidation) -> Result<TENGRIOversightResult, TENGRIError> {
        match validation.compliance_status {
            ComplianceStatus::FullyCompliant => Ok(TENGRIOversightResult::Approved),
            ComplianceStatus::MinorViolations => Ok(TENGRIOversightResult::Warning {
                reason: "Minor real integration violations detected".to_string(),
                corrective_action: "Review and strengthen integration authenticity".to_string(),
            }),
            ComplianceStatus::MajorViolations => Ok(TENGRIOversightResult::Rejected {
                reason: "Major real integration violations detected".to_string(),
                emergency_action: crate::EmergencyAction::QuarantineAgent {
                    agent_id: "integration_violation".to_string(),
                },
            }),
            ComplianceStatus::CriticalViolations => Ok(TENGRIOversightResult::CriticalViolation {
                violation_type: ViolationType::IntegrityBreach,
                immediate_shutdown: true,
                forensic_data: serde_json::to_vec(validation).unwrap_or_default(),
            }),
        }
    }

    /// Get validation statistics
    pub async fn get_validation_stats(&self) -> Result<RealIntegrationStats, TENGRIError> {
        let history = self.validation_history.read().await;
        let service_registry = self.service_registry.read().await;
        let data_registry = self.data_source_registry.read().await;
        let violation_counts = self.violation_counts.read().await;

        let total_validations = history.len();
        let compliant_validations = history.iter().filter(|(_, v)| matches!(v.compliance_status, ComplianceStatus::FullyCompliant)).count();
        let compliance_rate = if total_validations > 0 { compliant_validations as f64 / total_validations as f64 } else { 0.0 };

        Ok(RealIntegrationStats {
            total_validations,
            compliant_validations,
            compliance_rate,
            authenticated_services: service_registry.values().filter(|s| s.is_authentic).count(),
            authenticated_data_sources: data_registry.values().filter(|d| d.is_authentic).count(),
            violation_counts: violation_counts.clone(),
            last_validation: history.last().map(|(timestamp, _)| *timestamp),
        })
    }

    /// Build authenticity patterns
    fn build_authenticity_patterns() -> Result<HashMap<String, Regex>, TENGRIError> {
        let mut patterns = HashMap::new();
        
        patterns.insert("market_data".to_string(), Regex::new(r"(market|exchange|ticker|quote).*data")
            .map_err(|e| TENGRIError::DataIntegrityViolation { 
                reason: format!("Failed to compile market data pattern: {}", e) 
            })?);
            
        patterns.insert("production_endpoint".to_string(), Regex::new(r"(prod|production|live)\.")
            .map_err(|e| TENGRIError::DataIntegrityViolation { 
                reason: format!("Failed to compile production endpoint pattern: {}", e) 
            })?);
            
        patterns.insert("real_time_feed".to_string(), Regex::new(r"(real_time|live|streaming).*feed")
            .map_err(|e| TENGRIError::DataIntegrityViolation { 
                reason: format!("Failed to compile real-time feed pattern: {}", e) 
            })?);

        Ok(patterns)
    }
}

/// Validation Statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealIntegrationStats {
    pub total_validations: usize,
    pub compliant_validations: usize,
    pub compliance_rate: f64,
    pub authenticated_services: usize,
    pub authenticated_data_sources: usize,
    pub violation_counts: HashMap<String, u64>,
    pub last_validation: Option<DateTime<Utc>>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{OperationType, RiskParameters};

    #[tokio::test]
    async fn test_real_integration_validator() {
        let config = RealIntegrationConfig::default();
        let validator = RealIntegrationValidator::new(config).await.unwrap();
        
        let operation = TradingOperation {
            id: Uuid::new_v4(),
            timestamp: Utc::now(),
            operation_type: OperationType::PlaceOrder,
            data_source: "prod.exchange.market_data_feed".to_string(),
            mathematical_model: "real_model".to_string(),
            risk_parameters: RiskParameters {
                max_position_size: 1000.0,
                stop_loss: Some(0.02),
                take_profit: Some(0.05),
                confidence_threshold: 0.95,
            },
            agent_id: "test_agent".to_string(),
        };

        let result = validator.validate_real_integration(&operation).await.unwrap();
        assert!(matches!(result, TENGRIOversightResult::Approved));
    }

    #[tokio::test]
    async fn test_mock_service_rejection() {
        let config = RealIntegrationConfig::default();
        let validator = RealIntegrationValidator::new(config).await.unwrap();
        
        let operation = TradingOperation {
            id: Uuid::new_v4(),
            timestamp: Utc::now(),
            operation_type: OperationType::PlaceOrder,
            data_source: "localhost:8080/mock_service".to_string(),
            mathematical_model: "test_model".to_string(),
            risk_parameters: RiskParameters {
                max_position_size: 1000.0,
                stop_loss: Some(0.02),
                take_profit: Some(0.05),
                confidence_threshold: 0.95,
            },
            agent_id: "test_agent".to_string(),
        };

        let result = validator.validate_real_integration(&operation).await.unwrap();
        assert!(matches!(result, TENGRIOversightResult::Rejected { .. } | TENGRIOversightResult::CriticalViolation { .. }));
    }

    #[tokio::test]
    async fn test_validation_stats() {
        let config = RealIntegrationConfig::default();
        let validator = RealIntegrationValidator::new(config).await.unwrap();
        
        let stats = validator.get_validation_stats().await.unwrap();
        assert_eq!(stats.total_validations, 0);
        assert_eq!(stats.compliance_rate, 0.0);
    }
}