//! Advanced intrusion detection system with machine learning capabilities
//! 
//! This module implements enterprise-grade intrusion detection including:
//! - Real-time threat pattern recognition
//! - Behavioral anomaly detection
//! - Machine learning-based threat classification
//! - Automated incident response
//! - Threat intelligence integration

use anyhow::{anyhow, Result};
use chrono::{DateTime, Duration, Utc};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::net::IpAddr;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

/// Intrusion detection system
pub struct IntrusionDetectionSystem {
    /// Threat signatures database
    signatures: Vec<ThreatSignature>,
    
    /// Behavioral baselines for users
    behavioral_baselines: Arc<RwLock<HashMap<String, BehavioralBaseline>>>,
    
    /// Recent events for pattern analysis
    event_history: Arc<RwLock<VecDeque<SecurityEvent>>>,
    
    /// IP reputation database
    ip_reputation: Arc<RwLock<HashMap<IpAddr, IpReputation>>>,
    
    /// Active threats being tracked
    active_threats: Arc<RwLock<HashMap<String, ActiveThreat>>>,
    
    /// Configuration
    config: IdsConfig,
    
    /// Machine learning models (simplified)
    ml_models: MlModels,
}

/// Threat signature for pattern matching
#[derive(Debug, Clone)]
pub struct ThreatSignature {
    pub id: String,
    pub name: String,
    pub description: String,
    pub pattern: Regex,
    pub threat_type: ThreatType,
    pub severity: ThreatSeverity,
    pub confidence_threshold: f64,
    pub mitigation: MitigationAction,
    pub active: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ThreatType {
    SqlInjection,
    XssAttack,
    PathTraversal,
    CommandInjection,
    AuthenticationBypass,
    BruteForce,
    DdosAttack,
    DataExfiltration,
    PrivilegeEscalation,
    MalwareDetection,
    AnomalousBehavior,
    SuspiciousLogin,
    UnauthorizedAccess,
    Custom(String),
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum ThreatSeverity {
    Low = 1,
    Medium = 2,
    High = 3,
    Critical = 4,
    Emergency = 5,
}

#[derive(Debug, Clone)]
pub enum MitigationAction {
    Log,
    Alert,
    Block,
    Quarantine,
    RateLimitIncrease,
    RequireMfa,
    InvalidateSession,
    BanIp,
    Emergency,
}

/// Behavioral baseline for anomaly detection
#[derive(Debug, Clone)]
pub struct BehavioralBaseline {
    pub user_id: String,
    pub typical_login_hours: Vec<u8>,
    pub typical_locations: Vec<IpAddr>,
    pub average_session_duration: Duration,
    pub typical_request_patterns: HashMap<String, f64>,
    pub last_updated: DateTime<Utc>,
    pub confidence_score: f64,
}

/// Security event for analysis
#[derive(Debug, Clone)]
pub struct SecurityEvent {
    pub timestamp: DateTime<Utc>,
    pub source_ip: IpAddr,
    pub user_id: Option<String>,
    pub event_type: String,
    pub details: HashMap<String, String>,
    pub risk_score: f64,
    pub processed: bool,
}

/// IP reputation information
#[derive(Debug, Clone)]
pub struct IpReputation {
    pub ip: IpAddr,
    pub reputation_score: f64, // 0.0 = bad, 1.0 = good
    pub threat_types: Vec<ThreatType>,
    pub last_seen_malicious: Option<DateTime<Utc>>,
    pub country: Option<String>,
    pub asn: Option<u32>,
    pub is_vpn: bool,
    pub is_tor: bool,
    pub last_updated: DateTime<Utc>,
}

/// Active threat being tracked
#[derive(Debug, Clone)]
pub struct ActiveThreat {
    pub threat_id: String,
    pub threat_type: ThreatType,
    pub severity: ThreatSeverity,
    pub source_ip: IpAddr,
    pub target: String,
    pub first_detected: DateTime<Utc>,
    pub last_activity: DateTime<Utc>,
    pub event_count: u32,
    pub mitigation_applied: Vec<MitigationAction>,
    pub status: ThreatStatus,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ThreatStatus {
    Detected,
    Investigating,
    Mitigated,
    Resolved,
    FalsePositive,
}

/// IDS Configuration
#[derive(Debug, Clone)]
pub struct IdsConfig {
    pub enabled: bool,
    pub sensitivity: f64,
    pub auto_response: bool,
    pub behavioral_analysis: bool,
    pub ml_detection: bool,
    pub max_events_history: usize,
    pub baseline_learning_period_days: u32,
    pub threat_correlation_window_minutes: u32,
    pub ip_reputation_update_hours: u32,
}

impl Default for IdsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            sensitivity: 0.8,
            auto_response: true,
            behavioral_analysis: true,
            ml_detection: true,
            max_events_history: 10000,
            baseline_learning_period_days: 30,
            threat_correlation_window_minutes: 60,
            ip_reputation_update_hours: 24,
        }
    }
}

/// Machine learning models (simplified)
#[derive(Debug)]
pub struct MlModels {
    pub anomaly_detector: AnomalyDetector,
    pub threat_classifier: ThreatClassifier,
    pub behavioral_analyzer: BehavioralAnalyzer,
}

#[derive(Debug)]
pub struct AnomalyDetector {
    // Simplified ML model for demonstration
    threshold: f64,
}

#[derive(Debug)]
pub struct ThreatClassifier {
    // Simplified threat classification
    patterns: HashMap<String, f64>,
}

#[derive(Debug)]
pub struct BehavioralAnalyzer {
    // Simplified behavioral analysis
    baseline_variance_threshold: f64,
}

impl IntrusionDetectionSystem {
    pub fn new(config: IdsConfig) -> Self {
        let signatures = Self::create_default_signatures();
        let ml_models = MlModels {
            anomaly_detector: AnomalyDetector { threshold: 0.7 },
            threat_classifier: ThreatClassifier {
                patterns: Self::create_threat_patterns(),
            },
            behavioral_analyzer: BehavioralAnalyzer {
                baseline_variance_threshold: 0.3,
            },
        };
        
        Self {
            signatures,
            behavioral_baselines: Arc::new(RwLock::new(HashMap::new())),
            event_history: Arc::new(RwLock::new(VecDeque::new())),
            ip_reputation: Arc::new(RwLock::new(HashMap::new())),
            active_threats: Arc::new(RwLock::new(HashMap::new())),
            config,
            ml_models,
        }
    }
    
    /// Create default threat signatures
    fn create_default_signatures() -> Vec<ThreatSignature> {
        vec![
            ThreatSignature {
                id: "SQL-001".to_string(),
                name: "SQL Injection Attempt".to_string(),
                description: "Detects SQL injection patterns".to_string(),
                pattern: Regex::new(r"(?i)(union|select|insert|delete|drop|exec|execute)\s*(select|from|where)").unwrap(),
                threat_type: ThreatType::SqlInjection,
                severity: ThreatSeverity::High,
                confidence_threshold: 0.8,
                mitigation: MitigationAction::Block,
                active: true,
            },
            ThreatSignature {
                id: "XSS-001".to_string(),
                name: "Cross-Site Scripting".to_string(),
                description: "Detects XSS attack patterns".to_string(),
                pattern: Regex::new(r"(?i)<script[^>]*>|javascript:|onload\s*=|onerror\s*=").unwrap(),
                threat_type: ThreatType::XssAttack,
                severity: ThreatSeverity::Medium,
                confidence_threshold: 0.7,
                mitigation: MitigationAction::Block,
                active: true,
            },
            ThreatSignature {
                id: "TRAV-001".to_string(),
                name: "Path Traversal".to_string(),
                description: "Detects directory traversal attempts".to_string(),
                pattern: Regex::new(r"(\.\./|\.\.\\|%2e%2e%2f|%2e%2e%5c)").unwrap(),
                threat_type: ThreatType::PathTraversal,
                severity: ThreatSeverity::High,
                confidence_threshold: 0.9,
                mitigation: MitigationAction::Block,
                active: true,
            },
            ThreatSignature {
                id: "CMD-001".to_string(),
                name: "Command Injection".to_string(),
                description: "Detects command injection attempts".to_string(),
                pattern: Regex::new(r"(?i)(;|\||&|`|\$\(|eval\s*\(|exec\s*\(|system\s*\()").unwrap(),
                threat_type: ThreatType::CommandInjection,
                severity: ThreatSeverity::Critical,
                confidence_threshold: 0.8,
                mitigation: MitigationAction::Emergency,
                active: true,
            },
            ThreatSignature {
                id: "BRUTE-001".to_string(),
                name: "Brute Force Attack".to_string(),
                description: "Detects brute force login attempts".to_string(),
                pattern: Regex::new(r".*").unwrap(), // This is handled by behavioral analysis
                threat_type: ThreatType::BruteForce,
                severity: ThreatSeverity::High,
                confidence_threshold: 0.9,
                mitigation: MitigationAction::BanIp,
                active: true,
            },
        ]
    }
    
    /// Create threat classification patterns
    fn create_threat_patterns() -> HashMap<String, f64> {
        let mut patterns = HashMap::new();
        
        // SQL Injection patterns
        patterns.insert("union select".to_string(), 0.95);
        patterns.insert("or 1=1".to_string(), 0.9);
        patterns.insert("drop table".to_string(), 0.98);
        
        // XSS patterns
        patterns.insert("script>alert".to_string(), 0.9);
        patterns.insert("javascript:".to_string(), 0.8);
        patterns.insert("onload=".to_string(), 0.7);
        
        // Command injection patterns
        patterns.insert("eval(".to_string(), 0.9);
        patterns.insert("system(".to_string(), 0.95);
        patterns.insert("exec(".to_string(), 0.9);
        
        patterns
    }
    
    /// Analyze a security event for threats
    pub async fn analyze_event(&self, event: SecurityEvent) -> Result<ThreatAnalysisResult> {
        if !self.config.enabled {
            return Ok(ThreatAnalysisResult::benign());
        }
        
        let mut threats = Vec::new();
        let mut max_severity = ThreatSeverity::Low;
        let mut confidence = 0.0;
        
        // Store event in history
        {
            let mut history = self.event_history.write().await;
            history.push_back(event.clone());
            
            // Limit history size
            while history.len() > self.config.max_events_history {
                history.pop_front();
            }
        }
        
        // 1. Signature-based detection
        for signature in &self.signatures {
            if !signature.active {
                continue;
            }
            
            let match_found = self.check_signature_match(&event, signature).await?;
            if match_found.confidence >= signature.confidence_threshold {
                threats.push(DetectedThreat {
                    threat_type: signature.threat_type.clone(),
                    severity: signature.severity.clone(),
                    confidence: match_found.confidence,
                    signature_id: Some(signature.id.clone()),
                    description: signature.description.clone(),
                    evidence: match_found.evidence,
                });
                
                if signature.severity > max_severity {
                    max_severity = signature.severity.clone();
                }
                
                confidence = confidence.max(match_found.confidence);
            }
        }
        
        // 2. Behavioral analysis
        if self.config.behavioral_analysis && event.user_id.is_some() {
            if let Some(behavioral_threat) = self.analyze_behavioral_anomaly(&event).await? {
                threats.push(behavioral_threat);
            }
        }
        
        // 3. Machine learning detection
        if self.config.ml_detection {
            if let Some(ml_threat) = self.analyze_with_ml(&event).await? {
                threats.push(ml_threat);
            }
        }
        
        // 4. IP reputation check
        if let Some(reputation_threat) = self.check_ip_reputation(&event).await? {
            threats.push(reputation_threat);
        }
        
        // 5. Correlation analysis
        let correlated_threats = self.correlate_with_recent_events(&event).await?;
        threats.extend(correlated_threats);
        
        // Calculate overall risk score
        let risk_score = self.calculate_risk_score(&threats, &event).await;
        
        // Apply automatic mitigation if configured
        if self.config.auto_response && !threats.is_empty() {
            self.apply_automatic_mitigation(&event, &threats).await?;
        }
        
        // Track active threats
        if !threats.is_empty() {
            self.track_active_threat(&event, &threats).await?;
        }
        
        Ok(ThreatAnalysisResult {
            threats,
            risk_score,
            requires_human_review: max_severity >= ThreatSeverity::Critical,
            recommended_actions: self.recommend_actions(&threats).await,
        })
    }
    
    /// Check if event matches a threat signature
    async fn check_signature_match(&self, event: &SecurityEvent, signature: &ThreatSignature) -> Result<SignatureMatch> {
        let mut evidence = Vec::new();
        let mut confidence = 0.0;
        
        // Check event details for pattern matches
        for (key, value) in &event.details {
            if signature.pattern.is_match(value) {
                evidence.push(format!("Pattern match in {}: {}", key, value));
                confidence = 0.9; // High confidence for regex matches
            }
        }
        
        // Special handling for brute force detection
        if signature.threat_type == ThreatType::BruteForce {
            confidence = self.detect_brute_force(event).await?;
            if confidence > 0.0 {
                evidence.push("Multiple failed login attempts detected".to_string());
            }
        }
        
        Ok(SignatureMatch {
            matched: confidence > 0.0,
            confidence,
            evidence,
        })
    }
    
    /// Detect brute force attacks
    async fn detect_brute_force(&self, event: &SecurityEvent) -> Result<f64> {
        if event.event_type != "authentication_failed" {
            return Ok(0.0);
        }
        
        let history = self.event_history.read().await;
        let cutoff_time = Utc::now() - Duration::minutes(15);
        
        let failed_attempts = history.iter()
            .filter(|e| {
                e.timestamp > cutoff_time
                    && e.source_ip == event.source_ip
                    && e.event_type == "authentication_failed"
            })
            .count();
        
        // More than 10 failed attempts in 15 minutes = high confidence brute force
        Ok((failed_attempts as f64 / 10.0).min(1.0))
    }
    
    /// Analyze behavioral anomalies
    async fn analyze_behavioral_anomaly(&self, event: &SecurityEvent) -> Result<Option<DetectedThreat>> {
        let user_id = match &event.user_id {
            Some(id) => id,
            None => return Ok(None),
        };
        
        let baselines = self.behavioral_baselines.read().await;
        let baseline = match baselines.get(user_id) {
            Some(b) => b,
            None => return Ok(None), // No baseline yet
        };
        
        let mut anomaly_score = 0.0;
        let mut evidence = Vec::new();
        
        // Check login time anomaly
        let current_hour = Utc::now().hour() as u8;
        if !baseline.typical_login_hours.contains(&current_hour) {
            anomaly_score += 0.3;
            evidence.push(format!("Unusual login time: {}", current_hour));
        }
        
        // Check location anomaly
        if !baseline.typical_locations.contains(&event.source_ip) {
            anomaly_score += 0.4;
            evidence.push(format!("Unusual location: {}", event.source_ip));
        }
        
        // Check if anomaly score is significant
        if anomaly_score > self.ml_models.behavioral_analyzer.baseline_variance_threshold {
            Ok(Some(DetectedThreat {
                threat_type: ThreatType::AnomalousBehavior,
                severity: if anomaly_score > 0.7 { ThreatSeverity::High } else { ThreatSeverity::Medium },
                confidence: anomaly_score,
                signature_id: None,
                description: "Behavioral anomaly detected".to_string(),
                evidence,
            }))
        } else {
            Ok(None)
        }
    }
    
    /// Analyze with machine learning models
    async fn analyze_with_ml(&self, event: &SecurityEvent) -> Result<Option<DetectedThreat>> {
        // Simplified ML analysis
        let mut ml_score = 0.0;
        let mut evidence = Vec::new();
        
        // Check against threat classification patterns
        for (key, value) in &event.details {
            let value_lower = value.to_lowercase();
            for (pattern, score) in &self.ml_models.threat_classifier.patterns {
                if value_lower.contains(pattern) {
                    ml_score = ml_score.max(*score);
                    evidence.push(format!("ML pattern match: {} in {}", pattern, key));
                }
            }
        }
        
        // Apply anomaly detection threshold
        if ml_score > self.ml_models.anomaly_detector.threshold {
            Ok(Some(DetectedThreat {
                threat_type: ThreatType::Custom("ML_Detection".to_string()),
                severity: if ml_score > 0.9 { ThreatSeverity::High } else { ThreatSeverity::Medium },
                confidence: ml_score,
                signature_id: None,
                description: "Machine learning threat detection".to_string(),
                evidence,
            }))
        } else {
            Ok(None)
        }
    }
    
    /// Check IP reputation
    async fn check_ip_reputation(&self, event: &SecurityEvent) -> Result<Option<DetectedThreat>> {
        let reputation_db = self.ip_reputation.read().await;
        
        if let Some(reputation) = reputation_db.get(&event.source_ip) {
            if reputation.reputation_score < 0.3 {
                return Ok(Some(DetectedThreat {
                    threat_type: ThreatType::Custom("Bad_IP_Reputation".to_string()),
                    severity: ThreatSeverity::Medium,
                    confidence: 1.0 - reputation.reputation_score,
                    signature_id: None,
                    description: format!("Low reputation IP: {}", event.source_ip),
                    evidence: vec![
                        format!("Reputation score: {}", reputation.reputation_score),
                        format!("Threat types: {:?}", reputation.threat_types),
                    ],
                }));
            }
        }
        
        Ok(None)
    }
    
    /// Correlate with recent events
    async fn correlate_with_recent_events(&self, _event: &SecurityEvent) -> Result<Vec<DetectedThreat>> {
        // Simplified correlation - in production, implement sophisticated correlation logic
        Ok(Vec::new())
    }
    
    /// Calculate overall risk score
    async fn calculate_risk_score(&self, threats: &[DetectedThreat], _event: &SecurityEvent) -> f64 {
        if threats.is_empty() {
            return 0.0;
        }
        
        let max_confidence = threats.iter()
            .map(|t| t.confidence)
            .fold(0.0, f64::max);
        
        let severity_weight = threats.iter()
            .map(|t| match t.severity {
                ThreatSeverity::Low => 0.2,
                ThreatSeverity::Medium => 0.4,
                ThreatSeverity::High => 0.8,
                ThreatSeverity::Critical => 0.95,
                ThreatSeverity::Emergency => 1.0,
            })
            .fold(0.0, f64::max);
        
        (max_confidence * 0.6 + severity_weight * 0.4).min(1.0)
    }
    
    /// Apply automatic mitigation
    async fn apply_automatic_mitigation(&self, event: &SecurityEvent, threats: &[DetectedThreat]) -> Result<()> {
        for threat in threats {
            match threat.severity {
                ThreatSeverity::Critical | ThreatSeverity::Emergency => {
                    warn!(
                        "Applying emergency mitigation for {} from IP {}",
                        threat.description, event.source_ip
                    );
                    // In production: Block IP, invalidate sessions, alert SOC
                },
                ThreatSeverity::High => {
                    warn!(
                        "Applying high-priority mitigation for {} from IP {}",
                        threat.description, event.source_ip
                    );
                    // In production: Rate limit, require MFA, alert
                },
                _ => {
                    info!(
                        "Logging threat: {} from IP {}",
                        threat.description, event.source_ip
                    );
                }
            }
        }
        Ok(())
    }
    
    /// Track active threats
    async fn track_active_threat(&self, event: &SecurityEvent, threats: &[DetectedThreat]) -> Result<()> {
        let mut active_threats = self.active_threats.write().await;
        
        for threat in threats {
            let threat_key = format!("{}_{}", event.source_ip, threat.threat_type.to_string());
            
            match active_threats.get_mut(&threat_key) {
                Some(active) => {
                    active.last_activity = event.timestamp;
                    active.event_count += 1;
                },
                None => {
                    let new_threat = ActiveThreat {
                        threat_id: uuid::Uuid::new_v4().to_string(),
                        threat_type: threat.threat_type.clone(),
                        severity: threat.severity.clone(),
                        source_ip: event.source_ip,
                        target: event.details.get("target").unwrap_or(&"unknown".to_string()).clone(),
                        first_detected: event.timestamp,
                        last_activity: event.timestamp,
                        event_count: 1,
                        mitigation_applied: Vec::new(),
                        status: ThreatStatus::Detected,
                    };
                    active_threats.insert(threat_key, new_threat);
                }
            }
        }
        
        Ok(())
    }
    
    /// Recommend actions based on threats
    async fn recommend_actions(&self, threats: &[DetectedThreat]) -> Vec<String> {
        let mut actions = Vec::new();
        
        for threat in threats {
            match threat.threat_type {
                ThreatType::SqlInjection => {
                    actions.push("Review and strengthen input validation".to_string());
                    actions.push("Implement parameterized queries".to_string());
                },
                ThreatType::XssAttack => {
                    actions.push("Implement content security policy".to_string());
                    actions.push("Add output encoding".to_string());
                },
                ThreatType::BruteForce => {
                    actions.push("Implement account lockout policies".to_string());
                    actions.push("Require CAPTCHA after failed attempts".to_string());
                },
                ThreatType::AnomalousBehavior => {
                    actions.push("Require additional authentication".to_string());
                    actions.push("Monitor user activity closely".to_string());
                },
                _ => {
                    actions.push("Investigate further".to_string());
                    actions.push("Apply additional monitoring".to_string());
                }
            }
        }
        
        actions.dedup();
        actions
    }
}

impl std::fmt::Display for ThreatType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ThreatType::SqlInjection => write!(f, "SQL_Injection"),
            ThreatType::XssAttack => write!(f, "XSS_Attack"),
            ThreatType::PathTraversal => write!(f, "Path_Traversal"),
            ThreatType::CommandInjection => write!(f, "Command_Injection"),
            ThreatType::AuthenticationBypass => write!(f, "Authentication_Bypass"),
            ThreatType::BruteForce => write!(f, "Brute_Force"),
            ThreatType::DdosAttack => write!(f, "DDoS_Attack"),
            ThreatType::DataExfiltration => write!(f, "Data_Exfiltration"),
            ThreatType::PrivilegeEscalation => write!(f, "Privilege_Escalation"),
            ThreatType::MalwareDetection => write!(f, "Malware_Detection"),
            ThreatType::AnomalousBehavior => write!(f, "Anomalous_Behavior"),
            ThreatType::SuspiciousLogin => write!(f, "Suspicious_Login"),
            ThreatType::UnauthorizedAccess => write!(f, "Unauthorized_Access"),
            ThreatType::Custom(name) => write!(f, "{}", name),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ThreatAnalysisResult {
    pub threats: Vec<DetectedThreat>,
    pub risk_score: f64,
    pub requires_human_review: bool,
    pub recommended_actions: Vec<String>,
}

impl ThreatAnalysisResult {
    pub fn benign() -> Self {
        Self {
            threats: Vec::new(),
            risk_score: 0.0,
            requires_human_review: false,
            recommended_actions: Vec::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct DetectedThreat {
    pub threat_type: ThreatType,
    pub severity: ThreatSeverity,
    pub confidence: f64,
    pub signature_id: Option<String>,
    pub description: String,
    pub evidence: Vec<String>,
}

#[derive(Debug)]
struct SignatureMatch {
    pub matched: bool,
    pub confidence: f64,
    pub evidence: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_ids_creation() {
        let config = IdsConfig::default();
        let ids = IntrusionDetectionSystem::new(config);
        assert!(!ids.signatures.is_empty());
    }
    
    #[tokio::test]
    async fn test_sql_injection_detection() {
        let config = IdsConfig::default();
        let ids = IntrusionDetectionSystem::new(config);
        
        let mut event = SecurityEvent {
            timestamp: Utc::now(),
            source_ip: "192.168.1.100".parse().unwrap(),
            user_id: Some("test_user".to_string()),
            event_type: "web_request".to_string(),
            details: HashMap::new(),
            risk_score: 0.0,
            processed: false,
        };
        
        event.details.insert("url".to_string(), "/api/users?id=1 UNION SELECT * FROM passwords".to_string());
        
        let result = ids.analyze_event(event).await.unwrap();
        assert!(!result.threats.is_empty());
        assert!(result.threats.iter().any(|t| matches!(t.threat_type, ThreatType::SqlInjection)));
    }
    
    #[tokio::test]
    async fn test_xss_detection() {
        let config = IdsConfig::default();
        let ids = IntrusionDetectionSystem::new(config);
        
        let mut event = SecurityEvent {
            timestamp: Utc::now(),
            source_ip: "192.168.1.100".parse().unwrap(),
            user_id: Some("test_user".to_string()),
            event_type: "web_request".to_string(),
            details: HashMap::new(),
            risk_score: 0.0,
            processed: false,
        };
        
        event.details.insert("input".to_string(), "<script>alert('xss')</script>".to_string());
        
        let result = ids.analyze_event(event).await.unwrap();
        assert!(!result.threats.is_empty());
        assert!(result.threats.iter().any(|t| matches!(t.threat_type, ThreatType::XssAttack)));
    }
    
    #[tokio::test]
    async fn test_benign_event() {
        let config = IdsConfig::default();
        let ids = IntrusionDetectionSystem::new(config);
        
        let mut event = SecurityEvent {
            timestamp: Utc::now(),
            source_ip: "192.168.1.100".parse().unwrap(),
            user_id: Some("test_user".to_string()),
            event_type: "normal_request".to_string(),
            details: HashMap::new(),
            risk_score: 0.0,
            processed: false,
        };
        
        event.details.insert("url".to_string(), "/api/user/profile".to_string());
        
        let result = ids.analyze_event(event).await.unwrap();
        assert!(result.threats.is_empty());
        assert_eq!(result.risk_score, 0.0);
    }
}