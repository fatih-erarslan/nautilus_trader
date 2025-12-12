//! # Data Validation Agent
//!
//! TENGRI-integrated data quality and integrity validation agent.
//! Provides real-time data validation with comprehensive quality checks.

use std::sync::Arc;
use std::time::{Duration, Instant};
use std::collections::HashMap;
use tokio::sync::{RwLock, mpsc, Mutex};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use tracing::{info, warn, error, debug};
use async_trait::async_trait;
use validator::{Validate, ValidationError, ValidationErrors};
use regex::Regex;

use crate::agents::base::{
    DataAgent, DataAgentId, DataAgentType, DataAgentState, DataAgentInfo,
    DataMessage, DataMessageType, MessageMetadata, MessagePriority,
    CoordinationMessage, HealthStatus, HealthLevel, HealthMetrics,
    AgentMetrics, BaseDataAgent, MetricsUpdate
};

/// Data validation agent configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataValidationConfig {
    /// Target validation latency in microseconds
    pub target_latency_us: u64,
    /// Enable TENGRI integration
    pub tengri_enabled: bool,
    /// Validation rules configuration
    pub validation_rules: ValidationRulesConfig,
    /// TENGRI integration settings
    pub tengri_config: TengriIntegrationConfig,
    /// Quality thresholds
    pub quality_thresholds: QualityThresholds,
    /// Anomaly detection settings
    pub anomaly_detection: AnomalyDetectionConfig,
}

impl Default for DataValidationConfig {
    fn default() -> Self {
        Self {
            target_latency_us: 100,
            tengri_enabled: true,
            validation_rules: ValidationRulesConfig::default(),
            tengri_config: TengriIntegrationConfig::default(),
            quality_thresholds: QualityThresholds::default(),
            anomaly_detection: AnomalyDetectionConfig::default(),
        }
    }
}

/// Validation rules configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationRulesConfig {
    /// Enable schema validation
    pub schema_validation: bool,
    /// Enable range validation
    pub range_validation: bool,
    /// Enable consistency validation
    pub consistency_validation: bool,
    /// Enable format validation
    pub format_validation: bool,
    /// Enable business logic validation
    pub business_logic_validation: bool,
    /// Custom validation rules
    pub custom_rules: Vec<CustomValidationRule>,
}

impl Default for ValidationRulesConfig {
    fn default() -> Self {
        Self {
            schema_validation: true,
            range_validation: true,
            consistency_validation: true,
            format_validation: true,
            business_logic_validation: true,
            custom_rules: Vec::new(),
        }
    }
}

/// Custom validation rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomValidationRule {
    pub name: String,
    pub field: String,
    pub rule_type: ValidationRuleType,
    pub parameters: HashMap<String, serde_json::Value>,
    pub error_message: String,
}

/// Validation rule types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationRuleType {
    Range,
    Regex,
    Custom,
    Statistical,
    BusinessLogic,
}

/// TENGRI integration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TengriIntegrationConfig {
    /// TENGRI endpoint URL
    pub endpoint: String,
    /// Authentication token
    pub auth_token: String,
    /// Validation mode
    pub validation_mode: TengriValidationMode,
    /// Timeout for TENGRI calls
    pub timeout_ms: u64,
    /// Retry configuration
    pub retry_config: RetryConfig,
    /// Scientific rigor level
    pub scientific_rigor_level: ScientificRigorLevel,
}

impl Default for TengriIntegrationConfig {
    fn default() -> Self {
        Self {
            endpoint: "http://localhost:8080/tengri/validate".to_string(),
            auth_token: "default_token".to_string(),
            validation_mode: TengriValidationMode::Strict,
            timeout_ms: 5000,
            retry_config: RetryConfig::default(),
            scientific_rigor_level: ScientificRigorLevel::High,
        }
    }
}

/// TENGRI validation modes
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum TengriValidationMode {
    Fast,
    Standard,
    Strict,
    Paranoid,
}

/// Retry configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    pub max_retries: u32,
    pub initial_delay_ms: u64,
    pub max_delay_ms: u64,
    pub backoff_multiplier: f64,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_delay_ms: 100,
            max_delay_ms: 5000,
            backoff_multiplier: 2.0,
        }
    }
}

/// Scientific rigor levels
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ScientificRigorLevel {
    Low,
    Medium,
    High,
    Maximum,
}

/// Quality thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityThresholds {
    /// Minimum data quality score (0.0 - 1.0)
    pub min_quality_score: f64,
    /// Maximum error rate (0.0 - 1.0)
    pub max_error_rate: f64,
    /// Maximum missing data percentage (0.0 - 1.0)
    pub max_missing_data_rate: f64,
    /// Minimum consistency score (0.0 - 1.0)
    pub min_consistency_score: f64,
    /// Maximum outlier percentage (0.0 - 1.0)
    pub max_outlier_rate: f64,
}

impl Default for QualityThresholds {
    fn default() -> Self {
        Self {
            min_quality_score: 0.95,
            max_error_rate: 0.01,
            max_missing_data_rate: 0.05,
            min_consistency_score: 0.90,
            max_outlier_rate: 0.02,
        }
    }
}

/// Anomaly detection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyDetectionConfig {
    /// Enable anomaly detection
    pub enabled: bool,
    /// Detection algorithms
    pub algorithms: Vec<AnomalyDetectionAlgorithm>,
    /// Sensitivity level
    pub sensitivity: AnomalySensitivity,
    /// Window size for analysis
    pub window_size: usize,
    /// Threshold for anomaly score
    pub anomaly_threshold: f64,
}

impl Default for AnomalyDetectionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            algorithms: vec![
                AnomalyDetectionAlgorithm::IsolationForest,
                AnomalyDetectionAlgorithm::LocalOutlierFactor,
                AnomalyDetectionAlgorithm::ZScore,
            ],
            sensitivity: AnomalySensitivity::Medium,
            window_size: 100,
            anomaly_threshold: 0.8,
        }
    }
}

/// Anomaly detection algorithms
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum AnomalyDetectionAlgorithm {
    IsolationForest,
    LocalOutlierFactor,
    ZScore,
    InterquartileRange,
    EllipticEnvelope,
}

/// Anomaly sensitivity levels
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum AnomalySensitivity {
    Low,
    Medium,
    High,
}

/// Validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    pub is_valid: bool,
    pub quality_score: f64,
    pub validation_errors: Vec<DataValidationError>,
    pub warnings: Vec<String>,
    pub metadata: ValidationMetadata,
    pub tengri_result: Option<TengriValidationResult>,
    pub anomaly_result: Option<AnomalyDetectionResult>,
}

/// Data validation error
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataValidationError {
    pub field: String,
    pub error_type: ValidationErrorType,
    pub message: String,
    pub severity: ErrorSeverity,
    pub suggested_fix: Option<String>,
}

/// Validation error types
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ValidationErrorType {
    Schema,
    Range,
    Format,
    Consistency,
    BusinessLogic,
    Anomaly,
    MissingData,
    Duplicate,
}

/// Error severity levels
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ErrorSeverity {
    Warning,
    Error,
    Critical,
}

/// Validation metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationMetadata {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub validation_time_us: u64,
    pub rules_applied: usize,
    pub data_size: usize,
    pub validator_version: String,
}

/// TENGRI validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TengriValidationResult {
    pub integrity_score: f64,
    pub mathematical_consistency: f64,
    pub scientific_rigor_score: f64,
    pub synthetic_detection_score: f64,
    pub anomalies_detected: Vec<String>,
    pub recommendations: Vec<String>,
}

/// Anomaly detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyDetectionResult {
    pub anomalies_detected: Vec<AnomalyDetection>,
    pub anomaly_score: f64,
    pub confidence: f64,
    pub algorithm_results: HashMap<String, f64>,
}

/// Anomaly detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyDetection {
    pub field: String,
    pub value: serde_json::Value,
    pub anomaly_type: AnomalyType,
    pub score: f64,
    pub explanation: String,
}

/// Anomaly types
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum AnomalyType {
    Outlier,
    Inconsistent,
    Unexpected,
    Suspicious,
}

/// TENGRI validator
pub struct TengriValidator {
    config: Arc<TengriIntegrationConfig>,
    client: reqwest::Client,
}

impl TengriValidator {
    /// Create a new TENGRI validator
    pub fn new(config: Arc<TengriIntegrationConfig>) -> Self {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_millis(config.timeout_ms))
            .build()
            .unwrap();
        
        Self {
            config,
            client,
        }
    }
    
    /// Validate data with TENGRI
    pub async fn validate_with_tengri(&self, data: &serde_json::Value) -> Result<TengriValidationResult> {
        let request_payload = serde_json::json!({
            "data": data,
            "validation_mode": self.config.validation_mode,
            "scientific_rigor_level": self.config.scientific_rigor_level
        });
        
        let response = self.client
            .post(&self.config.endpoint)
            .header("Authorization", format!("Bearer {}", self.config.auth_token))
            .json(&request_payload)
            .send()
            .await?;
        
        if response.status().is_success() {
            let result: TengriValidationResult = response.json().await?;
            Ok(result)
        } else {
            Err(anyhow::anyhow!("TENGRI validation failed: {}", response.status()))
        }
    }
}

/// Data validation agent
pub struct DataValidationAgent {
    base: BaseDataAgent,
    config: Arc<DataValidationConfig>,
    tengri_validator: Arc<TengriValidator>,
    validation_cache: Arc<RwLock<HashMap<String, ValidationResult>>>,
    validation_metrics: Arc<RwLock<ValidationMetrics>>,
    state: Arc<RwLock<ValidationState>>,
    regex_cache: Arc<RwLock<HashMap<String, Regex>>>,
}

/// Validation metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationMetrics {
    pub validations_performed: u64,
    pub validations_passed: u64,
    pub validations_failed: u64,
    pub average_validation_time_us: f64,
    pub max_validation_time_us: f64,
    pub tengri_calls: u64,
    pub tengri_failures: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub last_update: chrono::DateTime<chrono::Utc>,
}

impl Default for ValidationMetrics {
    fn default() -> Self {
        Self {
            validations_performed: 0,
            validations_passed: 0,
            validations_failed: 0,
            average_validation_time_us: 0.0,
            max_validation_time_us: 0.0,
            tengri_calls: 0,
            tengri_failures: 0,
            cache_hits: 0,
            cache_misses: 0,
            last_update: chrono::Utc::now(),
        }
    }
}

/// Validation state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationState {
    pub active_validations: usize,
    pub tengri_connection_healthy: bool,
    pub cache_usage: f64,
    pub last_tengri_call: Option<chrono::DateTime<chrono::Utc>>,
    pub is_healthy: bool,
    pub last_health_check: chrono::DateTime<chrono::Utc>,
}

impl Default for ValidationState {
    fn default() -> Self {
        Self {
            active_validations: 0,
            tengri_connection_healthy: true,
            cache_usage: 0.0,
            last_tengri_call: None,
            is_healthy: true,
            last_health_check: chrono::Utc::now(),
        }
    }
}

impl DataValidationAgent {
    /// Create a new data validation agent
    pub async fn new(config: DataValidationConfig) -> Result<Self> {
        let base = BaseDataAgent::new(DataAgentType::DataValidation);
        let config = Arc::new(config);
        let tengri_validator = Arc::new(TengriValidator::new(config.tengri_config.clone().into()));
        let validation_cache = Arc::new(RwLock::new(HashMap::new()));
        let validation_metrics = Arc::new(RwLock::new(ValidationMetrics::default()));
        let state = Arc::new(RwLock::new(ValidationState::default()));
        let regex_cache = Arc::new(RwLock::new(HashMap::new()));
        
        Ok(Self {
            base,
            config,
            tengri_validator,
            validation_cache,
            validation_metrics,
            state,
            regex_cache,
        })
    }
    
    /// Validate data
    pub async fn validate_data(&self, data: &serde_json::Value) -> Result<ValidationResult> {
        let start_time = Instant::now();
        let mut validation_errors = Vec::new();
        let mut warnings = Vec::new();
        let mut quality_score = 1.0;
        
        // Generate cache key
        let cache_key = self.generate_cache_key(data);
        
        // Check cache
        if let Some(cached_result) = self.validation_cache.read().await.get(&cache_key) {
            let mut metrics = self.validation_metrics.write().await;
            metrics.cache_hits += 1;
            return Ok(cached_result.clone());
        }
        
        // Schema validation
        if self.config.validation_rules.schema_validation {
            if let Err(errors) = self.validate_schema(data).await {
                validation_errors.extend(errors);
                quality_score -= 0.1;
            }
        }
        
        // Range validation
        if self.config.validation_rules.range_validation {
            if let Err(errors) = self.validate_ranges(data).await {
                validation_errors.extend(errors);
                quality_score -= 0.1;
            }
        }
        
        // Format validation
        if self.config.validation_rules.format_validation {
            if let Err(errors) = self.validate_formats(data).await {
                validation_errors.extend(errors);
                quality_score -= 0.1;
            }
        }
        
        // Consistency validation
        if self.config.validation_rules.consistency_validation {
            if let Err(errors) = self.validate_consistency(data).await {
                validation_errors.extend(errors);
                quality_score -= 0.1;
            }
        }
        
        // Business logic validation
        if self.config.validation_rules.business_logic_validation {
            if let Err(errors) = self.validate_business_logic(data).await {
                validation_errors.extend(errors);
                quality_score -= 0.1;
            }
        }
        
        // Custom rules validation
        for rule in &self.config.validation_rules.custom_rules {
            if let Err(error) = self.validate_custom_rule(data, rule).await {
                validation_errors.push(error);
                quality_score -= 0.05;
            }
        }
        
        // TENGRI validation
        let tengri_result = if self.config.tengri_enabled {
            match self.tengri_validator.validate_with_tengri(data).await {
                Ok(result) => {
                    let mut metrics = self.validation_metrics.write().await;
                    metrics.tengri_calls += 1;
                    
                    // Update state
                    {
                        let mut state = self.state.write().await;
                        state.last_tengri_call = Some(chrono::Utc::now());
                        state.tengri_connection_healthy = true;
                    }
                    
                    // Adjust quality score based on TENGRI results
                    quality_score = (quality_score + result.integrity_score) / 2.0;
                    
                    Some(result)
                }
                Err(e) => {
                    warn!("TENGRI validation failed: {}", e);
                    let mut metrics = self.validation_metrics.write().await;
                    metrics.tengri_failures += 1;
                    
                    // Update state
                    {
                        let mut state = self.state.write().await;
                        state.tengri_connection_healthy = false;
                    }
                    
                    None
                }
            }
        } else {
            None
        };
        
        // Anomaly detection
        let anomaly_result = if self.config.anomaly_detection.enabled {
            self.detect_anomalies(data).await?
        } else {
            None
        };
        
        // Determine if validation passed
        let is_valid = validation_errors.is_empty() && 
                      quality_score >= self.config.quality_thresholds.min_quality_score;
        
        let validation_time = start_time.elapsed().as_micros() as u64;
        
        let result = ValidationResult {
            is_valid,
            quality_score,
            validation_errors,
            warnings,
            metadata: ValidationMetadata {
                timestamp: chrono::Utc::now(),
                validation_time_us: validation_time,
                rules_applied: self.count_applied_rules(),
                data_size: data.to_string().len(),
                validator_version: env!("CARGO_PKG_VERSION").to_string(),
            },
            tengri_result,
            anomaly_result,
        };
        
        // Cache the result
        self.validation_cache.write().await.insert(cache_key, result.clone());
        
        // Update metrics
        {
            let mut metrics = self.validation_metrics.write().await;
            metrics.validations_performed += 1;
            if is_valid {
                metrics.validations_passed += 1;
            } else {
                metrics.validations_failed += 1;
            }
            metrics.average_validation_time_us = 
                (metrics.average_validation_time_us + validation_time as f64) / 2.0;
            if validation_time as f64 > metrics.max_validation_time_us {
                metrics.max_validation_time_us = validation_time as f64;
            }
            metrics.cache_misses += 1;
            metrics.last_update = chrono::Utc::now();
        }
        
        Ok(result)
    }
    
    /// Generate cache key
    fn generate_cache_key(&self, data: &serde_json::Value) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        data.to_string().hash(&mut hasher);
        
        format!("validation_{}", hasher.finish())
    }
    
    /// Validate schema
    async fn validate_schema(&self, data: &serde_json::Value) -> Result<Vec<DataValidationError>, Vec<DataValidationError>> {
        let mut errors = Vec::new();
        
        // Basic schema validation
        if !data.is_object() {
            errors.push(DataValidationError {
                field: "root".to_string(),
                error_type: ValidationErrorType::Schema,
                message: "Data must be an object".to_string(),
                severity: ErrorSeverity::Error,
                suggested_fix: Some("Ensure data is formatted as a JSON object".to_string()),
            });
        }
        
        if errors.is_empty() {
            Ok(errors)
        } else {
            Err(errors)
        }
    }
    
    /// Validate ranges
    async fn validate_ranges(&self, data: &serde_json::Value) -> Result<Vec<DataValidationError>, Vec<DataValidationError>> {
        let mut errors = Vec::new();
        
        // Range validation logic
        if let Some(obj) = data.as_object() {
            for (key, value) in obj {
                if let Some(num) = value.as_f64() {
                    if num.is_infinite() || num.is_nan() {
                        errors.push(DataValidationError {
                            field: key.clone(),
                            error_type: ValidationErrorType::Range,
                            message: format!("Invalid numeric value: {}", num),
                            severity: ErrorSeverity::Error,
                            suggested_fix: Some("Ensure numeric values are finite".to_string()),
                        });
                    }
                }
            }
        }
        
        if errors.is_empty() {
            Ok(errors)
        } else {
            Err(errors)
        }
    }
    
    /// Validate formats
    async fn validate_formats(&self, data: &serde_json::Value) -> Result<Vec<DataValidationError>, Vec<DataValidationError>> {
        let mut errors = Vec::new();
        
        // Format validation logic
        if let Some(obj) = data.as_object() {
            for (key, value) in obj {
                if key.contains("timestamp") || key.contains("time") {
                    if let Some(time_str) = value.as_str() {
                        if chrono::DateTime::parse_from_rfc3339(time_str).is_err() {
                            errors.push(DataValidationError {
                                field: key.clone(),
                                error_type: ValidationErrorType::Format,
                                message: format!("Invalid timestamp format: {}", time_str),
                                severity: ErrorSeverity::Warning,
                                suggested_fix: Some("Use RFC3339 format for timestamps".to_string()),
                            });
                        }
                    }
                }
            }
        }
        
        if errors.is_empty() {
            Ok(errors)
        } else {
            Err(errors)
        }
    }
    
    /// Validate consistency
    async fn validate_consistency(&self, data: &serde_json::Value) -> Result<Vec<DataValidationError>, Vec<DataValidationError>> {
        let mut errors = Vec::new();
        
        // Consistency validation logic
        if let Some(obj) = data.as_object() {
            // Check for logical consistency
            if let (Some(price), Some(volume)) = (obj.get("price"), obj.get("volume")) {
                if let (Some(p), Some(v)) = (price.as_f64(), volume.as_f64()) {
                    if p <= 0.0 || v < 0.0 {
                        errors.push(DataValidationError {
                            field: "price_volume".to_string(),
                            error_type: ValidationErrorType::Consistency,
                            message: "Price must be positive and volume non-negative".to_string(),
                            severity: ErrorSeverity::Error,
                            suggested_fix: Some("Ensure price > 0 and volume >= 0".to_string()),
                        });
                    }
                }
            }
        }
        
        if errors.is_empty() {
            Ok(errors)
        } else {
            Err(errors)
        }
    }
    
    /// Validate business logic
    async fn validate_business_logic(&self, data: &serde_json::Value) -> Result<Vec<DataValidationError>, Vec<DataValidationError>> {
        let mut errors = Vec::new();
        
        // Business logic validation
        if let Some(obj) = data.as_object() {
            // Market hours validation
            if let Some(timestamp) = obj.get("timestamp") {
                if let Some(time_str) = timestamp.as_str() {
                    if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(time_str) {
                        let hour = dt.hour();
                        if hour < 9 || hour > 16 {
                            errors.push(DataValidationError {
                                field: "timestamp".to_string(),
                                error_type: ValidationErrorType::BusinessLogic,
                                message: "Trade timestamp outside market hours".to_string(),
                                severity: ErrorSeverity::Warning,
                                suggested_fix: Some("Verify timestamp is within market hours".to_string()),
                            });
                        }
                    }
                }
            }
        }
        
        if errors.is_empty() {
            Ok(errors)
        } else {
            Err(errors)
        }
    }
    
    /// Validate custom rule
    async fn validate_custom_rule(&self, data: &serde_json::Value, rule: &CustomValidationRule) -> Result<(), DataValidationError> {
        match rule.rule_type {
            ValidationRuleType::Range => {
                if let Some(obj) = data.as_object() {
                    if let Some(value) = obj.get(&rule.field) {
                        if let Some(num) = value.as_f64() {
                            if let (Some(min), Some(max)) = (
                                rule.parameters.get("min").and_then(|v| v.as_f64()),
                                rule.parameters.get("max").and_then(|v| v.as_f64())
                            ) {
                                if num < min || num > max {
                                    return Err(DataValidationError {
                                        field: rule.field.clone(),
                                        error_type: ValidationErrorType::Range,
                                        message: rule.error_message.clone(),
                                        severity: ErrorSeverity::Error,
                                        suggested_fix: Some(format!("Value must be between {} and {}", min, max)),
                                    });
                                }
                            }
                        }
                    }
                }
            }
            ValidationRuleType::Regex => {
                if let Some(obj) = data.as_object() {
                    if let Some(value) = obj.get(&rule.field) {
                        if let Some(text) = value.as_str() {
                            if let Some(pattern) = rule.parameters.get("pattern").and_then(|v| v.as_str()) {
                                let regex = {
                                    let mut cache = self.regex_cache.write().await;
                                    cache.entry(pattern.to_string())
                                        .or_insert_with(|| Regex::new(pattern).unwrap())
                                        .clone()
                                };
                                
                                if !regex.is_match(text) {
                                    return Err(DataValidationError {
                                        field: rule.field.clone(),
                                        error_type: ValidationErrorType::Format,
                                        message: rule.error_message.clone(),
                                        severity: ErrorSeverity::Error,
                                        suggested_fix: Some(format!("Value must match pattern: {}", pattern)),
                                    });
                                }
                            }
                        }
                    }
                }
            }
            _ => {
                // Handle other rule types
            }
        }
        
        Ok(())
    }
    
    /// Detect anomalies
    async fn detect_anomalies(&self, data: &serde_json::Value) -> Result<Option<AnomalyDetectionResult>> {
        let mut anomalies = Vec::new();
        let mut algorithm_results = HashMap::new();
        
        if let Some(obj) = data.as_object() {
            for (key, value) in obj {
                if let Some(num) = value.as_f64() {
                    // Z-score anomaly detection
                    let z_score = self.calculate_z_score(num, key).await;
                    algorithm_results.insert(format!("z_score_{}", key), z_score);
                    
                    if z_score.abs() > 3.0 {
                        anomalies.push(AnomalyDetection {
                            field: key.clone(),
                            value: value.clone(),
                            anomaly_type: AnomalyType::Outlier,
                            score: z_score.abs(),
                            explanation: format!("Z-score of {} exceeds threshold", z_score),
                        });
                    }
                }
            }
        }
        
        if anomalies.is_empty() {
            Ok(None)
        } else {
            Ok(Some(AnomalyDetectionResult {
                anomalies_detected: anomalies,
                anomaly_score: algorithm_results.values().map(|v| v.abs()).sum::<f64>() / algorithm_results.len() as f64,
                confidence: 0.85,
                algorithm_results,
            }))
        }
    }
    
    /// Calculate Z-score
    async fn calculate_z_score(&self, value: f64, field: &str) -> f64 {
        // Simplified Z-score calculation
        // In a real implementation, this would use historical data
        let mean = 100.0; // Would be calculated from historical data
        let std_dev = 15.0; // Would be calculated from historical data
        
        (value - mean) / std_dev
    }
    
    /// Count applied rules
    fn count_applied_rules(&self) -> usize {
        let mut count = 0;
        
        if self.config.validation_rules.schema_validation { count += 1; }
        if self.config.validation_rules.range_validation { count += 1; }
        if self.config.validation_rules.format_validation { count += 1; }
        if self.config.validation_rules.consistency_validation { count += 1; }
        if self.config.validation_rules.business_logic_validation { count += 1; }
        
        count += self.config.validation_rules.custom_rules.len();
        
        count
    }
    
    /// Get validation metrics
    pub async fn get_validation_metrics(&self) -> ValidationMetrics {
        self.validation_metrics.read().await.clone()
    }
    
    /// Get validation state
    pub async fn get_validation_state(&self) -> ValidationState {
        self.state.read().await.clone()
    }
}

impl From<TengriIntegrationConfig> for Arc<TengriIntegrationConfig> {
    fn from(config: TengriIntegrationConfig) -> Self {
        Arc::new(config)
    }
}

#[async_trait]
impl DataAgent for DataValidationAgent {
    fn get_id(&self) -> DataAgentId {
        self.base.id
    }
    
    fn get_type(&self) -> DataAgentType {
        DataAgentType::DataValidation
    }
    
    async fn get_state(&self) -> DataAgentState {
        self.base.state.read().await.clone()
    }
    
    async fn get_info(&self) -> DataAgentInfo {
        self.base.info.read().await.clone()
    }
    
    async fn start(&self) -> Result<()> {
        info!("Starting data validation agent");
        
        self.base.update_state(DataAgentState::Running).await?;
        
        info!("Data validation agent started successfully");
        Ok(())
    }
    
    async fn stop(&self) -> Result<()> {
        info!("Stopping data validation agent");
        
        self.base.update_state(DataAgentState::Stopping).await?;
        
        // Clear cache
        self.validation_cache.write().await.clear();
        
        self.base.update_state(DataAgentState::Stopped).await?;
        
        info!("Data validation agent stopped successfully");
        Ok(())
    }
    
    async fn process(&self, message: DataMessage) -> Result<DataMessage> {
        let start_time = Instant::now();
        
        // Validate the message data
        let validation_result = self.validate_data(&message.payload).await?;
        
        // Update metrics
        let latency = start_time.elapsed().as_micros() as f64;
        self.base.update_metrics(MetricsUpdate::MessageProcessed(latency)).await?;
        
        // Create response message
        let response = DataMessage {
            id: uuid::Uuid::new_v4(),
            timestamp: chrono::Utc::now(),
            source: self.get_id(),
            destination: message.destination,
            message_type: DataMessageType::ValidationResult,
            payload: serde_json::to_value(validation_result)?,
            metadata: MessageMetadata {
                priority: MessagePriority::High,
                expires_at: Some(chrono::Utc::now() + chrono::Duration::seconds(30)),
                retry_count: 0,
                trace_id: format!("data_validation_{}", uuid::Uuid::new_v4()),
                span_id: format!("span_{}", uuid::Uuid::new_v4()),
            },
        };
        
        Ok(response)
    }
    
    async fn health_check(&self) -> Result<HealthStatus> {
        let state = self.get_validation_state().await;
        let metrics = self.get_validation_metrics().await;
        
        let health_level = if state.is_healthy && state.tengri_connection_healthy {
            HealthLevel::Healthy
        } else if state.is_healthy {
            HealthLevel::Warning
        } else {
            HealthLevel::Critical
        };
        
        Ok(HealthStatus {
            status: health_level,
            last_check: chrono::Utc::now(),
            uptime: self.base.start_time.elapsed(),
            issues: Vec::new(),
            metrics: HealthMetrics {
                cpu_usage_percent: 0.0, // Would be measured
                memory_usage_mb: 0.0,   // Would be measured
                network_usage_mbps: 0.0, // Would be measured
                disk_usage_mb: 0.0,     // Would be measured
                error_rate: metrics.validations_failed as f64 / metrics.validations_performed.max(1) as f64,
                response_time_ms: metrics.average_validation_time_us / 1000.0,
            },
        })
    }
    
    async fn get_metrics(&self) -> Result<AgentMetrics> {
        Ok(self.base.metrics.read().await.clone())
    }
    
    async fn reset(&self) -> Result<()> {
        info!("Resetting data validation agent");
        
        self.validation_cache.write().await.clear();
        
        // Reset metrics
        {
            let mut metrics = self.validation_metrics.write().await;
            *metrics = ValidationMetrics::default();
        }
        
        // Reset state
        {
            let mut state = self.state.write().await;
            *state = ValidationState::default();
        }
        
        info!("Data validation agent reset successfully");
        Ok(())
    }
    
    async fn handle_coordination(&self, message: CoordinationMessage) -> Result<()> {
        debug!("Handling coordination message: {:?}", message.coordination_type);
        
        match message.coordination_type {
            crate::agents::base::CoordinationType::LoadBalancing => {
                info!("Received load balancing coordination");
            }
            crate::agents::base::CoordinationType::StateSync => {
                info!("Received state sync coordination");
            }
            _ => {
                debug!("Unhandled coordination type: {:?}", message.coordination_type);
            }
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::test;
    
    #[test]
    async fn test_data_validation_agent_creation() {
        let config = DataValidationConfig::default();
        let agent = DataValidationAgent::new(config).await;
        assert!(agent.is_ok());
    }
    
    #[test]
    async fn test_data_validation() {
        let config = DataValidationConfig::default();
        let agent = DataValidationAgent::new(config).await.unwrap();
        
        let data = serde_json::json!({
            "price": 100.0,
            "volume": 1000.0,
            "timestamp": "2023-01-01T12:00:00Z"
        });
        
        let result = agent.validate_data(&data).await;
        assert!(result.is_ok());
        
        let validation_result = result.unwrap();
        assert!(validation_result.is_valid);
    }
    
    #[test]
    async fn test_validation_rules() {
        let config = DataValidationConfig::default();
        let agent = DataValidationAgent::new(config).await.unwrap();
        
        // Test invalid data
        let invalid_data = serde_json::json!({
            "price": -100.0,
            "volume": -1000.0
        });
        
        let result = agent.validate_data(&invalid_data).await;
        assert!(result.is_ok());
        
        let validation_result = result.unwrap();
        assert!(!validation_result.is_valid);
        assert!(!validation_result.validation_errors.is_empty());
    }
}