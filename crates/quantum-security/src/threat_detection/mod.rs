//! Quantum-Enhanced Threat Detection System
//!
//! This module implements real-time threat detection using quantum-enhanced
//! machine learning algorithms for the ATS-CP trading system.

pub mod detector;
pub mod analyzer;
pub mod ml_engine;
pub mod patterns;
pub mod response;

pub use detector::*;
pub use analyzer::*;
pub use ml_engine::*;
pub use patterns::*;
pub use response::*;

use crate::error::QuantumSecurityError;
use crate::types::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Threat Level
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ThreatLevel {
    /// No threat detected
    None,
    /// Low threat level
    Low,
    /// Medium threat level
    Medium,
    /// High threat level
    High,
    /// Critical threat level
    Critical,
}

/// Threat Categories
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ThreatCategory {
    /// Network-based threats
    NetworkIntrusion,
    /// Application-level threats
    ApplicationAttack,
    /// Data exfiltration attempts
    DataExfiltration,
    /// Identity and access threats
    IdentityTheft,
    /// Market manipulation attempts
    MarketManipulation,
    /// Insider threats
    InsiderThreat,
    /// Advanced persistent threats
    APT,
    /// Quantum computing attacks
    QuantumAttack,
    /// AI/ML model attacks
    ModelAttack,
    /// Physical security threats
    PhysicalThreat,
    /// Supply chain attacks
    SupplyChain,
    /// Denial of service attacks
    DenialOfService,
    /// Social engineering
    SocialEngineering,
    /// Cryptographic attacks
    CryptographicAttack,
}

/// Threat Operations for validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ThreatOperation {
    Encrypt,
    Decrypt,
    Sign,
    Verify,
    KeyGeneration,
    KeyDistribution,
    Authentication,
    DataAccess,
    Trading,
    Communication,
}

/// Threat Detection Result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreatDetectionResult {
    pub detection_id: Uuid,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub threat_level: ThreatLevel,
    pub threat_category: ThreatCategory,
    pub confidence_score: f64,
    pub threat_indicators: Vec<ThreatIndicator>,
    pub affected_entities: Vec<String>,
    pub recommended_actions: Vec<ThreatResponse>,
    pub quantum_analysis: Option<QuantumThreatAnalysis>,
    pub ml_prediction: Option<MLThreatPrediction>,
    pub risk_score: f64,
    pub severity_score: f64,
}

/// Threat Indicator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreatIndicator {
    pub indicator_id: Uuid,
    pub indicator_type: ThreatIndicatorType,
    pub value: String,
    pub confidence: f64,
    pub source: String,
    pub first_seen: chrono::DateTime<chrono::Utc>,
    pub last_seen: chrono::DateTime<chrono::Utc>,
    pub metadata: HashMap<String, String>,
}

/// Threat Indicator Types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ThreatIndicatorType {
    /// IP address indicators
    IPAddress,
    /// Domain name indicators
    Domain,
    /// File hash indicators
    FileHash,
    /// URL indicators
    URL,
    /// Email address indicators
    Email,
    /// Registry key indicators
    RegistryKey,
    /// Network pattern indicators
    NetworkPattern,
    /// Behavioral pattern indicators
    BehavioralPattern,
    /// Cryptographic anomaly indicators
    CryptographicAnomaly,
    /// Quantum signature indicators
    QuantumSignature,
    /// Trading pattern indicators
    TradingPattern,
    /// Authentication anomaly indicators
    AuthenticationAnomaly,
}

/// Quantum Threat Analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumThreatAnalysis {
    pub analysis_id: Uuid,
    pub quantum_algorithm_used: Option<String>,
    pub quantum_resistance_score: f64,
    pub cryptographic_vulnerability: Option<CryptographicVulnerability>,
    pub quantum_supremacy_timeline: Option<chrono::Duration>,
    pub mitigation_recommendations: Vec<String>,
}

/// Cryptographic Vulnerability
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CryptographicVulnerability {
    pub algorithm: String,
    pub key_size: u32,
    pub vulnerability_type: String,
    pub exploitation_complexity: String,
    pub quantum_break_timeline: Option<chrono::Duration>,
}

/// ML Threat Prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLThreatPrediction {
    pub model_id: String,
    pub model_version: String,
    pub prediction_confidence: f64,
    pub feature_importance: HashMap<String, f64>,
    pub anomaly_score: f64,
    pub prediction_timeline: Option<chrono::Duration>,
    pub similar_threats: Vec<String>,
}

/// Threat Response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreatResponse {
    pub response_id: Uuid,
    pub response_type: ThreatResponseType,
    pub priority: ResponsePriority,
    pub description: String,
    pub automated: bool,
    pub estimated_duration: chrono::Duration,
    pub prerequisites: Vec<String>,
    pub impact_assessment: ResponseImpact,
}

/// Threat Response Types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ThreatResponseType {
    /// Block network traffic
    BlockNetwork,
    /// Isolate affected systems
    IsolateSystem,
    /// Revoke access credentials
    RevokeCredentials,
    /// Rotate cryptographic keys
    RotateKeys,
    /// Update security policies
    UpdatePolicies,
    /// Notify security team
    NotifyTeam,
    /// Initiate incident response
    InitiateIncidentResponse,
    /// Apply security patches
    ApplyPatches,
    /// Backup critical data
    BackupData,
    /// Activate failover systems
    ActivateFailover,
    /// Increase monitoring
    IncreaseMonitoring,
    /// Quantum key redistribution
    QuantumKeyRedistribution,
}

/// Response Priority
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum ResponsePriority {
    Low,
    Medium,
    High,
    Critical,
    Emergency,
}

/// Response Impact Assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseImpact {
    pub business_impact: BusinessImpact,
    pub technical_impact: TechnicalImpact,
    pub user_impact: UserImpact,
    pub estimated_cost: f64,
    pub recovery_time: chrono::Duration,
}

/// Business Impact
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BusinessImpact {
    None,
    Low,
    Medium,
    High,
    Critical,
}

/// Technical Impact
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TechnicalImpact {
    None,
    Performance,
    Availability,
    Functionality,
    Security,
    DataIntegrity,
}

/// User Impact
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UserImpact {
    None,
    Minor,
    Moderate,
    Significant,
    Severe,
}

/// Threat Detection Event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreatDetectionEvent {
    pub event_id: Uuid,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub event_type: ThreatEventType,
    pub source: String,
    pub target: Option<String>,
    pub threat_level: ThreatLevel,
    pub threat_category: ThreatCategory,
    pub raw_data: Vec<u8>,
    pub processed_data: HashMap<String, String>,
    pub context: EventContext,
}

/// Threat Event Types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ThreatEventType {
    /// Anomaly detected in system behavior
    AnomalyDetected,
    /// Suspicious network activity
    SuspiciousNetworkActivity,
    /// Authentication failure
    AuthenticationFailure,
    /// Unauthorized access attempt
    UnauthorizedAccess,
    /// Data exfiltration detected
    DataExfiltration,
    /// Malware detected
    MalwareDetected,
    /// Intrusion attempt
    IntrusionAttempt,
    /// Policy violation
    PolicyViolation,
    /// Cryptographic anomaly
    CryptographicAnomaly,
    /// Quantum threat detected
    QuantumThreatDetected,
    /// Trading anomaly
    TradingAnomaly,
    /// System compromise
    SystemCompromise,
}

/// Event Context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventContext {
    pub session_id: Option<Uuid>,
    pub agent_id: Option<String>,
    pub operation_type: Option<String>,
    pub source_ip: Option<String>,
    pub user_agent: Option<String>,
    pub geo_location: Option<GeoLocation>,
    pub device_fingerprint: Option<String>,
    pub risk_factors: Vec<String>,
}

/// Geo Location
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeoLocation {
    pub country: String,
    pub region: String,
    pub city: String,
    pub latitude: f64,
    pub longitude: f64,
    pub accuracy_km: f64,
}

/// Threat Detection Metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ThreatDetectionMetrics {
    pub total_events_processed: u64,
    pub threats_detected: u64,
    pub false_positives: u64,
    pub false_negatives: u64,
    pub true_positives: u64,
    pub true_negatives: u64,
    pub detection_accuracy: f64,
    pub detection_precision: f64,
    pub detection_recall: f64,
    pub f1_score: f64,
    pub average_detection_time_ms: f64,
    pub threat_distribution: HashMap<ThreatCategory, u64>,
    pub severity_distribution: HashMap<ThreatLevel, u64>,
    pub quantum_threats_detected: u64,
    pub ml_model_performance: HashMap<String, f64>,
}

/// Threat Detection Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreatDetectionConfig {
    pub enabled_categories: Vec<ThreatCategory>,
    pub detection_thresholds: HashMap<ThreatCategory, f64>,
    pub quantum_analysis_enabled: bool,
    pub ml_models_enabled: bool,
    pub real_time_analysis: bool,
    pub automated_response_enabled: bool,
    pub notification_settings: NotificationSettings,
    pub retention_period: chrono::Duration,
    pub sampling_rate: f64,
}

/// Notification Settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationSettings {
    pub email_notifications: bool,
    pub sms_notifications: bool,
    pub webhook_notifications: bool,
    pub severity_threshold: ThreatLevel,
    pub notification_channels: Vec<NotificationChannel>,
}

/// Notification Channel
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationChannel {
    pub channel_id: String,
    pub channel_type: String,
    pub endpoint: String,
    pub enabled: bool,
    pub threat_categories: Vec<ThreatCategory>,
    pub severity_levels: Vec<ThreatLevel>,
}

impl Default for ThreatDetectionConfig {
    fn default() -> Self {
        let mut detection_thresholds = HashMap::new();
        detection_thresholds.insert(ThreatCategory::NetworkIntrusion, 0.7);
        detection_thresholds.insert(ThreatCategory::QuantumAttack, 0.5);
        detection_thresholds.insert(ThreatCategory::MarketManipulation, 0.8);
        
        Self {
            enabled_categories: vec![
                ThreatCategory::NetworkIntrusion,
                ThreatCategory::ApplicationAttack,
                ThreatCategory::QuantumAttack,
                ThreatCategory::MarketManipulation,
                ThreatCategory::IdentityTheft,
            ],
            detection_thresholds,
            quantum_analysis_enabled: true,
            ml_models_enabled: true,
            real_time_analysis: true,
            automated_response_enabled: false,
            notification_settings: NotificationSettings {
                email_notifications: true,
                sms_notifications: false,
                webhook_notifications: true,
                severity_threshold: ThreatLevel::Medium,
                notification_channels: Vec::new(),
            },
            retention_period: chrono::Duration::days(90),
            sampling_rate: 1.0,
        }
    }
}

impl ThreatDetectionResult {
    /// Create new threat detection result
    pub fn new(
        threat_level: ThreatLevel,
        threat_category: ThreatCategory,
        confidence_score: f64,
    ) -> Self {
        Self {
            detection_id: Uuid::new_v4(),
            timestamp: chrono::Utc::now(),
            threat_level,
            threat_category,
            confidence_score,
            threat_indicators: Vec::new(),
            affected_entities: Vec::new(),
            recommended_actions: Vec::new(),
            quantum_analysis: None,
            ml_prediction: None,
            risk_score: 0.0,
            severity_score: 0.0,
        }
    }
    
    /// Add threat indicator
    pub fn add_indicator(&mut self, indicator: ThreatIndicator) {
        self.threat_indicators.push(indicator);
    }
    
    /// Add affected entity
    pub fn add_affected_entity(&mut self, entity: String) {
        if !self.affected_entities.contains(&entity) {
            self.affected_entities.push(entity);
        }
    }
    
    /// Add recommended action
    pub fn add_recommended_action(&mut self, action: ThreatResponse) {
        self.recommended_actions.push(action);
    }
    
    /// Set quantum analysis results
    pub fn set_quantum_analysis(&mut self, analysis: QuantumThreatAnalysis) {
        self.quantum_analysis = Some(analysis);
    }
    
    /// Set ML prediction results
    pub fn set_ml_prediction(&mut self, prediction: MLThreatPrediction) {
        self.ml_prediction = Some(prediction);
    }
    
    /// Calculate composite risk score
    pub fn calculate_risk_score(&mut self) {
        let level_weight = match self.threat_level {
            ThreatLevel::None => 0.0,
            ThreatLevel::Low => 0.2,
            ThreatLevel::Medium => 0.5,
            ThreatLevel::High => 0.8,
            ThreatLevel::Critical => 1.0,
        };
        
        let confidence_weight = self.confidence_score;
        let quantum_weight = self.quantum_analysis.as_ref()
            .map(|qa| 1.0 - qa.quantum_resistance_score)
            .unwrap_or(0.0);
        let ml_weight = self.ml_prediction.as_ref()
            .map(|ml| ml.anomaly_score)
            .unwrap_or(0.0);
        
        self.risk_score = (level_weight * 0.4 + confidence_weight * 0.3 + quantum_weight * 0.2 + ml_weight * 0.1)
            .clamp(0.0, 1.0);
    }
    
    /// Calculate severity score
    pub fn calculate_severity_score(&mut self) {
        let base_severity = match self.threat_level {
            ThreatLevel::None => 0.0,
            ThreatLevel::Low => 0.25,
            ThreatLevel::Medium => 0.5,
            ThreatLevel::High => 0.75,
            ThreatLevel::Critical => 1.0,
        };
        
        let entity_multiplier = (self.affected_entities.len() as f64).ln().max(1.0);
        let confidence_multiplier = self.confidence_score;
        
        self.severity_score = (base_severity * entity_multiplier * confidence_multiplier)
            .clamp(0.0, 1.0);
    }
    
    /// Check if threat requires immediate response
    pub fn requires_immediate_response(&self) -> bool {
        matches!(self.threat_level, ThreatLevel::Critical | ThreatLevel::High) &&
        self.confidence_score > 0.8
    }
    
    /// Get priority for threat response
    pub fn get_response_priority(&self) -> ResponsePriority {
        match (self.threat_level.clone(), self.confidence_score) {
            (ThreatLevel::Critical, score) if score > 0.9 => ResponsePriority::Emergency,
            (ThreatLevel::Critical, _) => ResponsePriority::Critical,
            (ThreatLevel::High, score) if score > 0.8 => ResponsePriority::Critical,
            (ThreatLevel::High, _) => ResponsePriority::High,
            (ThreatLevel::Medium, _) => ResponsePriority::Medium,
            _ => ResponsePriority::Low,
        }
    }
}

impl ThreatIndicator {
    /// Create new threat indicator
    pub fn new(
        indicator_type: ThreatIndicatorType,
        value: String,
        confidence: f64,
        source: String,
    ) -> Self {
        let now = chrono::Utc::now();
        
        Self {
            indicator_id: Uuid::new_v4(),
            indicator_type,
            value,
            confidence,
            source,
            first_seen: now,
            last_seen: now,
            metadata: HashMap::new(),
        }
    }
    
    /// Update last seen timestamp
    pub fn update_last_seen(&mut self) {
        self.last_seen = chrono::Utc::now();
    }
    
    /// Add metadata
    pub fn add_metadata(&mut self, key: String, value: String) {
        self.metadata.insert(key, value);
    }
    
    /// Get age of indicator
    pub fn age(&self) -> chrono::Duration {
        chrono::Utc::now().signed_duration_since(self.first_seen)
    }
    
    /// Check if indicator is stale
    pub fn is_stale(&self, max_age: chrono::Duration) -> bool {
        self.age() > max_age
    }
}

impl ThreatDetectionMetrics {
    /// Update metrics with detection result
    pub fn update_with_result(&mut self, result: &ThreatDetectionResult, processing_time_ms: f64) {
        self.total_events_processed += 1;
        
        if result.threat_level != ThreatLevel::None {
            self.threats_detected += 1;
            *self.threat_distribution.entry(result.threat_category.clone()).or_insert(0) += 1;
            *self.severity_distribution.entry(result.threat_level.clone()).or_insert(0) += 1;
            
            if matches!(result.threat_category, ThreatCategory::QuantumAttack) {
                self.quantum_threats_detected += 1;
            }
        }
        
        // Update average detection time
        self.average_detection_time_ms = 
            (self.average_detection_time_ms * (self.total_events_processed - 1) as f64 + processing_time_ms) / 
            self.total_events_processed as f64;
    }
    
    /// Calculate detection accuracy
    pub fn calculate_accuracy(&mut self) {
        let total = self.true_positives + self.true_negatives + self.false_positives + self.false_negatives;
        if total > 0 {
            self.detection_accuracy = (self.true_positives + self.true_negatives) as f64 / total as f64;
            self.detection_precision = if self.true_positives + self.false_positives > 0 {
                self.true_positives as f64 / (self.true_positives + self.false_positives) as f64
            } else {
                0.0
            };
            self.detection_recall = if self.true_positives + self.false_negatives > 0 {
                self.true_positives as f64 / (self.true_positives + self.false_negatives) as f64
            } else {
                0.0
            };
            
            if self.detection_precision + self.detection_recall > 0.0 {
                self.f1_score = 2.0 * (self.detection_precision * self.detection_recall) / 
                    (self.detection_precision + self.detection_recall);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_threat_detection_result_creation() {
        let mut result = ThreatDetectionResult::new(
            ThreatLevel::High,
            ThreatCategory::NetworkIntrusion,
            0.85,
        );
        
        assert_eq!(result.threat_level, ThreatLevel::High);
        assert_eq!(result.threat_category, ThreatCategory::NetworkIntrusion);
        assert_eq!(result.confidence_score, 0.85);
        
        result.calculate_risk_score();
        result.calculate_severity_score();
        
        assert!(result.risk_score > 0.0);
        assert!(result.severity_score > 0.0);
        assert!(result.requires_immediate_response());
        assert_eq!(result.get_response_priority(), ResponsePriority::Critical);
    }
    
    #[test]
    fn test_threat_indicator() {
        let mut indicator = ThreatIndicator::new(
            ThreatIndicatorType::IPAddress,
            "192.168.1.100".to_string(),
            0.9,
            "network_monitor".to_string(),
        );
        
        indicator.add_metadata("country".to_string(), "unknown".to_string());
        indicator.update_last_seen();
        
        assert_eq!(indicator.value, "192.168.1.100");
        assert_eq!(indicator.confidence, 0.9);
        assert!(indicator.metadata.contains_key("country"));
        assert!(indicator.age().num_seconds() >= 0);
    }
    
    #[test]
    fn test_threat_detection_config() {
        let config = ThreatDetectionConfig::default();
        
        assert!(config.enabled_categories.contains(&ThreatCategory::NetworkIntrusion));
        assert!(config.enabled_categories.contains(&ThreatCategory::QuantumAttack));
        assert!(config.quantum_analysis_enabled);
        assert!(config.ml_models_enabled);
        assert!(config.real_time_analysis);
    }
    
    #[test]
    fn test_threat_metrics() {
        let mut metrics = ThreatDetectionMetrics::default();
        
        let result = ThreatDetectionResult::new(
            ThreatLevel::Medium,
            ThreatCategory::ApplicationAttack,
            0.7,
        );
        
        metrics.update_with_result(&result, 150.0);
        
        assert_eq!(metrics.total_events_processed, 1);
        assert_eq!(metrics.threats_detected, 1);
        assert_eq!(metrics.average_detection_time_ms, 150.0);
        assert_eq!(metrics.threat_distribution.get(&ThreatCategory::ApplicationAttack), Some(&1));
    }
}