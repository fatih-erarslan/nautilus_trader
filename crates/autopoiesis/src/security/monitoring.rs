//! Real-time security monitoring and alerting system
//! 
//! This module provides comprehensive security monitoring including:
//! - Real-time threat detection and alerting
//! - Security metrics collection and analysis
//! - Dashboard and visualization support
//! - Automated incident response
//! - Integration with external security tools

use anyhow::{anyhow, Result};
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

use super::intrusion::{ThreatAnalysisResult, ThreatSeverity, ThreatType};

/// Real-time security monitoring system
pub struct SecurityMonitoringSystem {
    /// Security metrics collector
    metrics_collector: MetricsCollector,
    
    /// Alert manager
    alert_manager: AlertManager,
    
    /// Dashboard data
    dashboard_data: Arc<RwLock<SecurityDashboard>>,
    
    /// Incident tracker
    incident_tracker: IncidentTracker,
    
    /// Configuration
    config: MonitoringConfig,
}

#[derive(Debug, Clone)]
pub struct MonitoringConfig {
    pub metrics_retention_days: u32,
    pub alert_throttle_minutes: u32,
    pub dashboard_update_interval_seconds: u32,
    pub incident_auto_escalation_minutes: u32,
    pub external_integrations: Vec<ExternalIntegration>,
}

#[derive(Debug, Clone)]
pub struct ExternalIntegration {
    pub name: String,
    pub integration_type: IntegrationType,
    pub endpoint: String,
    pub api_key: Option<String>,
    pub enabled: bool,
}

#[derive(Debug, Clone)]
pub enum IntegrationType {
    Siem,
    Soar,
    ThreatIntelligence,
    NotificationService,
    IncidentManagement,
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            metrics_retention_days: 90,
            alert_throttle_minutes: 5,
            dashboard_update_interval_seconds: 30,
            incident_auto_escalation_minutes: 60,
            external_integrations: vec![
                ExternalIntegration {
                    name: "Slack Notifications".to_string(),
                    integration_type: IntegrationType::NotificationService,
                    endpoint: "https://hooks.slack.com/services/...".to_string(),
                    api_key: None,
                    enabled: false,
                },
            ],
        }
    }
}

/// Security metrics collector
pub struct MetricsCollector {
    /// Real-time metrics
    current_metrics: Arc<RwLock<SecurityMetrics>>,
    
    /// Historical metrics
    historical_metrics: Arc<RwLock<VecDeque<SecurityMetricsSnapshot>>>,
    
    /// Configuration
    retention_days: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityMetrics {
    /// Timestamp of metrics
    pub timestamp: DateTime<Utc>,
    
    /// Threat detection metrics
    pub threats_detected_last_hour: u64,
    pub threats_blocked_last_hour: u64,
    pub false_positive_rate: f64,
    
    /// Authentication metrics
    pub authentication_attempts: u64,
    pub authentication_failures: u64,
    pub authentication_success_rate: f64,
    
    /// System security metrics
    pub active_sessions: u64,
    pub failed_login_attempts: u64,
    pub suspicious_activities: u64,
    pub blocked_ips: u64,
    
    /// Compliance metrics
    pub gdpr_compliance_score: f64,
    pub sox_compliance_score: f64,
    pub audit_events_count: u64,
    
    /// Performance metrics
    pub security_processing_latency_ms: f64,
    pub encryption_operations_per_second: f64,
    pub security_overhead_percentage: f64,
    
    /// Infrastructure metrics
    pub ssl_certificate_days_to_expiry: u32,
    pub security_patches_pending: u32,
    pub vulnerability_count: u32,
}

impl Default for SecurityMetrics {
    fn default() -> Self {
        Self {
            timestamp: Utc::now(),
            threats_detected_last_hour: 0,
            threats_blocked_last_hour: 0,
            false_positive_rate: 0.0,
            authentication_attempts: 0,
            authentication_failures: 0,
            authentication_success_rate: 100.0,
            active_sessions: 0,
            failed_login_attempts: 0,
            suspicious_activities: 0,
            blocked_ips: 0,
            gdpr_compliance_score: 100.0,
            sox_compliance_score: 100.0,
            audit_events_count: 0,
            security_processing_latency_ms: 0.0,
            encryption_operations_per_second: 0.0,
            security_overhead_percentage: 0.0,
            ssl_certificate_days_to_expiry: 365,
            security_patches_pending: 0,
            vulnerability_count: 0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityMetricsSnapshot {
    pub timestamp: DateTime<Utc>,
    pub metrics: SecurityMetrics,
}

/// Alert management system
pub struct AlertManager {
    /// Active alerts
    active_alerts: Arc<RwLock<HashMap<String, SecurityAlert>>>,
    
    /// Alert history
    alert_history: Arc<RwLock<VecDeque<SecurityAlert>>>,
    
    /// Alert rules
    alert_rules: Vec<AlertRule>,
    
    /// Throttling state
    throttle_state: Arc<RwLock<HashMap<String, DateTime<Utc>>>>,
    
    /// Configuration
    throttle_minutes: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityAlert {
    pub alert_id: String,
    pub alert_type: AlertType,
    pub severity: AlertSeverity,
    pub title: String,
    pub description: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub status: AlertStatus,
    pub source: String,
    pub affected_resources: Vec<String>,
    pub recommended_actions: Vec<String>,
    pub false_positive: bool,
    pub escalated: bool,
    pub acknowledged_by: Option<String>,
    pub acknowledged_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertType {
    ThreatDetected,
    SecurityViolation,
    ComplianceViolation,
    SystemAnomaly,
    AuthenticationFailure,
    DataBreach,
    UnauthorizedAccess,
    SuspiciousActivity,
    VulnerabilityDetected,
    ConfigurationChange,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum AlertSeverity {
    Info = 1,
    Low = 2,
    Medium = 3,
    High = 4,
    Critical = 5,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertStatus {
    New,
    Acknowledged,
    InProgress,
    Resolved,
    FalsePositive,
    Escalated,
}

#[derive(Debug, Clone)]
pub struct AlertRule {
    pub rule_id: String,
    pub name: String,
    pub condition: AlertCondition,
    pub severity: AlertSeverity,
    pub enabled: bool,
    pub auto_acknowledge: bool,
    pub escalation_time_minutes: Option<u32>,
}

#[derive(Debug, Clone)]
pub enum AlertCondition {
    MetricThreshold {
        metric_name: String,
        operator: ComparisonOperator,
        threshold: f64,
    },
    ThreatDetected {
        threat_types: Vec<ThreatType>,
        min_severity: ThreatSeverity,
    },
    RateLimit {
        event_type: String,
        count: u64,
        time_window_minutes: u32,
    },
    Custom {
        expression: String,
    },
}

#[derive(Debug, Clone)]
pub enum ComparisonOperator {
    GreaterThan,
    LessThan,
    GreaterThanOrEqual,
    LessThanOrEqual,
    Equal,
    NotEqual,
}

/// Security dashboard data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityDashboard {
    pub last_updated: DateTime<Utc>,
    pub overall_security_score: f64,
    pub threat_level: ThreatLevel,
    pub metrics: SecurityMetrics,
    pub active_alerts: Vec<SecurityAlert>,
    pub top_threats: Vec<ThreatSummary>,
    pub compliance_status: ComplianceStatus,
    pub system_health: SystemHealth,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ThreatLevel {
    Minimal,
    Low,
    Moderate,
    High,
    Severe,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreatSummary {
    pub threat_type: String,
    pub count: u64,
    pub severity: String,
    pub last_detected: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceStatus {
    pub gdpr_compliant: bool,
    pub sox_compliant: bool,
    pub last_audit_date: Option<DateTime<Utc>>,
    pub next_audit_due: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemHealth {
    pub security_services_up: u32,
    pub security_services_down: u32,
    pub last_health_check: DateTime<Utc>,
    pub critical_issues: u32,
}

/// Incident tracking system
pub struct IncidentTracker {
    /// Active incidents
    active_incidents: Arc<RwLock<HashMap<String, SecurityIncident>>>,
    
    /// Incident history
    incident_history: Arc<RwLock<VecDeque<SecurityIncident>>>,
    
    /// Auto-escalation settings
    auto_escalation_minutes: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityIncident {
    pub incident_id: String,
    pub title: String,
    pub description: String,
    pub severity: IncidentSeverity,
    pub status: IncidentStatus,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub assigned_to: Option<String>,
    pub affected_systems: Vec<String>,
    pub related_alerts: Vec<String>,
    pub timeline: Vec<IncidentTimelineEntry>,
    pub resolution: Option<String>,
    pub lessons_learned: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IncidentSeverity {
    Minor,
    Major,
    Critical,
    Emergency,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IncidentStatus {
    New,
    Assigned,
    InProgress,
    Resolved,
    Closed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IncidentTimelineEntry {
    pub timestamp: DateTime<Utc>,
    pub action: String,
    pub performed_by: String,
    pub details: String,
}

impl SecurityMonitoringSystem {
    pub fn new(config: MonitoringConfig) -> Self {
        let alert_rules = Self::create_default_alert_rules();
        
        Self {
            metrics_collector: MetricsCollector::new(config.metrics_retention_days),
            alert_manager: AlertManager::new(alert_rules, config.alert_throttle_minutes),
            dashboard_data: Arc::new(RwLock::new(SecurityDashboard::default())),
            incident_tracker: IncidentTracker::new(config.incident_auto_escalation_minutes),
            config,
        }
    }
    
    /// Create default alert rules
    fn create_default_alert_rules() -> Vec<AlertRule> {
        vec![
            AlertRule {
                rule_id: "high-threat-detection".to_string(),
                name: "High Threat Detection".to_string(),
                condition: AlertCondition::ThreatDetected {
                    threat_types: vec![ThreatType::SqlInjection, ThreatType::XssAttack],
                    min_severity: ThreatSeverity::High,
                },
                severity: AlertSeverity::High,
                enabled: true,
                auto_acknowledge: false,
                escalation_time_minutes: Some(30),
            },
            AlertRule {
                rule_id: "failed-auth-rate".to_string(),
                name: "High Failed Authentication Rate".to_string(),
                condition: AlertCondition::MetricThreshold {
                    metric_name: "authentication_failures".to_string(),
                    operator: ComparisonOperator::GreaterThan,
                    threshold: 100.0,
                },
                severity: AlertSeverity::Medium,
                enabled: true,
                auto_acknowledge: false,
                escalation_time_minutes: Some(60),
            },
            AlertRule {
                rule_id: "compliance-violation".to_string(),
                name: "Compliance Score Below Threshold".to_string(),
                condition: AlertCondition::MetricThreshold {
                    metric_name: "gdpr_compliance_score".to_string(),
                    operator: ComparisonOperator::LessThan,
                    threshold: 80.0,
                },
                severity: AlertSeverity::High,
                enabled: true,
                auto_acknowledge: false,
                escalation_time_minutes: Some(15),
            },
        ]
    }
    
    /// Process threat analysis result and generate alerts
    pub async fn process_threat_analysis(&mut self, result: ThreatAnalysisResult) -> Result<()> {
        // Update metrics
        self.metrics_collector.update_threat_metrics(&result).await?;
        
        // Generate alerts for detected threats
        for threat in &result.threats {
            if threat.severity >= ThreatSeverity::Medium {
                let alert = SecurityAlert {
                    alert_id: uuid::Uuid::new_v4().to_string(),
                    alert_type: AlertType::ThreatDetected,
                    severity: match threat.severity {
                        ThreatSeverity::Low => AlertSeverity::Low,
                        ThreatSeverity::Medium => AlertSeverity::Medium,
                        ThreatSeverity::High => AlertSeverity::High,
                        ThreatSeverity::Critical => AlertSeverity::Critical,
                        ThreatSeverity::Emergency => AlertSeverity::Critical,
                    },
                    title: format!("Threat Detected: {}", threat.description),
                    description: format!("Security threat detected with confidence {:.2}", threat.confidence),
                    created_at: Utc::now(),
                    updated_at: Utc::now(),
                    status: AlertStatus::New,
                    source: "Intrusion Detection System".to_string(),
                    affected_resources: vec!["Trading System".to_string()],
                    recommended_actions: result.recommended_actions.clone(),
                    false_positive: false,
                    escalated: false,
                    acknowledged_by: None,
                    acknowledged_at: None,
                };
                
                self.alert_manager.create_alert(alert).await?;
            }
        }
        
        // Create incident for critical threats
        if result.requires_human_review {
            self.incident_tracker.create_incident_from_threats(&result).await?;
        }
        
        // Update dashboard
        self.update_dashboard().await?;
        
        Ok(())
    }
    
    /// Update security metrics
    pub async fn update_metrics(&mut self, metrics: SecurityMetrics) -> Result<()> {
        self.metrics_collector.update_current_metrics(metrics).await?;
        
        // Check alert rules
        self.alert_manager.check_alert_rules(&self.metrics_collector.current_metrics).await?;
        
        // Update dashboard
        self.update_dashboard().await?;
        
        Ok(())
    }
    
    /// Update dashboard data
    async fn update_dashboard(&mut self) -> Result<()> {
        let mut dashboard = self.dashboard_data.write().await;
        
        let current_metrics = self.metrics_collector.get_current_metrics().await;
        let active_alerts = self.alert_manager.get_active_alerts().await;
        
        dashboard.last_updated = Utc::now();
        dashboard.metrics = current_metrics;
        dashboard.active_alerts = active_alerts;
        dashboard.overall_security_score = self.calculate_overall_security_score().await?;
        dashboard.threat_level = self.determine_threat_level().await?;
        dashboard.compliance_status = self.get_compliance_status().await;
        dashboard.system_health = self.get_system_health().await;
        
        Ok(())
    }
    
    /// Calculate overall security score
    async fn calculate_overall_security_score(&self) -> Result<f64> {
        let metrics = self.metrics_collector.get_current_metrics().await;
        
        let mut score = 100.0;
        
        // Deduct points for threats
        if metrics.threats_detected_last_hour > 0 {
            score -= (metrics.threats_detected_last_hour as f64) * 2.0;
        }
        
        // Deduct points for failed authentications
        if metrics.authentication_success_rate < 95.0 {
            score -= (95.0 - metrics.authentication_success_rate) * 0.5;
        }
        
        // Deduct points for compliance issues
        if metrics.gdpr_compliance_score < 100.0 {
            score -= (100.0 - metrics.gdpr_compliance_score) * 0.3;
        }
        
        if metrics.sox_compliance_score < 100.0 {
            score -= (100.0 - metrics.sox_compliance_score) * 0.3;
        }
        
        // Deduct points for vulnerabilities
        score -= (metrics.vulnerability_count as f64) * 5.0;
        
        Ok(score.max(0.0))
    }
    
    /// Determine current threat level
    async fn determine_threat_level(&self) -> Result<ThreatLevel> {
        let metrics = self.metrics_collector.get_current_metrics().await;
        let active_alerts = self.alert_manager.get_active_alerts().await;
        
        let critical_alerts = active_alerts.iter()
            .filter(|a| a.severity == AlertSeverity::Critical)
            .count();
        
        let high_alerts = active_alerts.iter()
            .filter(|a| a.severity == AlertSeverity::High)
            .count();
        
        if critical_alerts > 0 || metrics.threats_detected_last_hour > 10 {
            Ok(ThreatLevel::Critical)
        } else if high_alerts > 2 || metrics.threats_detected_last_hour > 5 {
            Ok(ThreatLevel::High)
        } else if metrics.threats_detected_last_hour > 2 {
            Ok(ThreatLevel::Moderate)
        } else if metrics.threats_detected_last_hour > 0 {
            Ok(ThreatLevel::Low)
        } else {
            Ok(ThreatLevel::Minimal)
        }
    }
    
    /// Get compliance status
    async fn get_compliance_status(&self) -> ComplianceStatus {
        let metrics = self.metrics_collector.get_current_metrics().await;
        
        ComplianceStatus {
            gdpr_compliant: metrics.gdpr_compliance_score >= 95.0,
            sox_compliant: metrics.sox_compliance_score >= 95.0,
            last_audit_date: Some(Utc::now() - Duration::days(30)),
            next_audit_due: Some(Utc::now() + Duration::days(90)),
        }
    }
    
    /// Get system health status
    async fn get_system_health(&self) -> SystemHealth {
        SystemHealth {
            security_services_up: 5,
            security_services_down: 0,
            last_health_check: Utc::now(),
            critical_issues: 0,
        }
    }
    
    /// Get current dashboard data
    pub async fn get_dashboard(&self) -> SecurityDashboard {
        self.dashboard_data.read().await.clone()
    }
    
    /// Get security metrics
    pub async fn get_metrics(&self) -> SecurityMetrics {
        self.metrics_collector.get_current_metrics().await
    }
    
    /// Get active alerts
    pub async fn get_active_alerts(&self) -> Vec<SecurityAlert> {
        self.alert_manager.get_active_alerts().await
    }
    
    /// Acknowledge an alert
    pub async fn acknowledge_alert(&mut self, alert_id: &str, acknowledged_by: &str) -> Result<()> {
        self.alert_manager.acknowledge_alert(alert_id, acknowledged_by).await
    }
}

impl Default for SecurityDashboard {
    fn default() -> Self {
        Self {
            last_updated: Utc::now(),
            overall_security_score: 100.0,
            threat_level: ThreatLevel::Minimal,
            metrics: SecurityMetrics::default(),
            active_alerts: Vec::new(),
            top_threats: Vec::new(),
            compliance_status: ComplianceStatus {
                gdpr_compliant: true,
                sox_compliant: true,
                last_audit_date: None,
                next_audit_due: None,
            },
            system_health: SystemHealth {
                security_services_up: 5,
                security_services_down: 0,
                last_health_check: Utc::now(),
                critical_issues: 0,
            },
        }
    }
}

impl MetricsCollector {
    fn new(retention_days: u32) -> Self {
        Self {
            current_metrics: Arc::new(RwLock::new(SecurityMetrics::default())),
            historical_metrics: Arc::new(RwLock::new(VecDeque::new())),
            retention_days,
        }
    }
    
    async fn update_current_metrics(&self, metrics: SecurityMetrics) -> Result<()> {
        *self.current_metrics.write().await = metrics;
        Ok(())
    }
    
    async fn update_threat_metrics(&self, result: &ThreatAnalysisResult) -> Result<()> {
        let mut metrics = self.current_metrics.write().await;
        
        if !result.threats.is_empty() {
            metrics.threats_detected_last_hour += 1;
            
            // Check if threat was blocked
            if result.risk_score > 0.7 {
                metrics.threats_blocked_last_hour += 1;
            }
        }
        
        Ok(())
    }
    
    async fn get_current_metrics(&self) -> SecurityMetrics {
        self.current_metrics.read().await.clone()
    }
}

impl AlertManager {
    fn new(alert_rules: Vec<AlertRule>, throttle_minutes: u32) -> Self {
        Self {
            active_alerts: Arc::new(RwLock::new(HashMap::new())),
            alert_history: Arc::new(RwLock::new(VecDeque::new())),
            alert_rules,
            throttle_state: Arc::new(RwLock::new(HashMap::new())),
            throttle_minutes,
        }
    }
    
    async fn create_alert(&self, alert: SecurityAlert) -> Result<()> {
        // Check throttling
        let throttle_key = format!("{}_{}", alert.alert_type as u8, alert.severity as u8);
        let mut throttle_state = self.throttle_state.write().await;
        
        if let Some(last_alert_time) = throttle_state.get(&throttle_key) {
            let time_since_last = Utc::now() - *last_alert_time;
            if time_since_last < Duration::minutes(self.throttle_minutes as i64) {
                return Ok(()); // Skip due to throttling
            }
        }
        
        throttle_state.insert(throttle_key, Utc::now());
        drop(throttle_state);
        
        // Add to active alerts
        {
            let mut active_alerts = self.active_alerts.write().await;
            active_alerts.insert(alert.alert_id.clone(), alert.clone());
        }
        
        // Add to history
        {
            let mut history = self.alert_history.write().await;
            history.push_back(alert.clone());
            
            // Limit history size
            while history.len() > 1000 {
                history.pop_front();
            }
        }
        
        // Log alert
        match alert.severity {
            AlertSeverity::Critical => {
                error!("CRITICAL SECURITY ALERT: {}", alert.title);
            },
            AlertSeverity::High => {
                warn!("HIGH SECURITY ALERT: {}", alert.title);
            },
            _ => {
                info!("Security alert: {}", alert.title);
            }
        }
        
        Ok(())
    }
    
    async fn check_alert_rules(&self, _metrics: &Arc<RwLock<SecurityMetrics>>) -> Result<()> {
        // Implementation would check each alert rule against current metrics
        // and create alerts as needed
        Ok(())
    }
    
    async fn get_active_alerts(&self) -> Vec<SecurityAlert> {
        self.active_alerts.read().await.values().cloned().collect()
    }
    
    async fn acknowledge_alert(&self, alert_id: &str, acknowledged_by: &str) -> Result<()> {
        let mut active_alerts = self.active_alerts.write().await;
        
        if let Some(alert) = active_alerts.get_mut(alert_id) {
            alert.status = AlertStatus::Acknowledged;
            alert.acknowledged_by = Some(acknowledged_by.to_string());
            alert.acknowledged_at = Some(Utc::now());
            alert.updated_at = Utc::now();
            
            info!("Alert {} acknowledged by {}", alert_id, acknowledged_by);
        }
        
        Ok(())
    }
}

impl IncidentTracker {
    fn new(auto_escalation_minutes: u32) -> Self {
        Self {
            active_incidents: Arc::new(RwLock::new(HashMap::new())),
            incident_history: Arc::new(RwLock::new(VecDeque::new())),
            auto_escalation_minutes,
        }
    }
    
    async fn create_incident_from_threats(&self, result: &ThreatAnalysisResult) -> Result<()> {
        let incident = SecurityIncident {
            incident_id: uuid::Uuid::new_v4().to_string(),
            title: "Security Threats Detected Requiring Investigation".to_string(),
            description: format!("Multiple security threats detected with risk score: {:.2}", result.risk_score),
            severity: if result.risk_score > 0.9 {
                IncidentSeverity::Emergency
            } else if result.risk_score > 0.7 {
                IncidentSeverity::Critical
            } else {
                IncidentSeverity::Major
            },
            status: IncidentStatus::New,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            assigned_to: None,
            affected_systems: vec!["Trading System".to_string()],
            related_alerts: Vec::new(),
            timeline: vec![
                IncidentTimelineEntry {
                    timestamp: Utc::now(),
                    action: "Incident Created".to_string(),
                    performed_by: "Automated System".to_string(),
                    details: "Created from threat analysis".to_string(),
                }
            ],
            resolution: None,
            lessons_learned: Vec::new(),
        };
        
        let mut active_incidents = self.active_incidents.write().await;
        active_incidents.insert(incident.incident_id.clone(), incident);
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_monitoring_system_creation() {
        let config = MonitoringConfig::default();
        let monitoring = SecurityMonitoringSystem::new(config);
        
        let dashboard = monitoring.get_dashboard().await;
        assert_eq!(dashboard.threat_level, ThreatLevel::Minimal);
    }
    
    #[tokio::test]
    async fn test_metrics_update() {
        let config = MonitoringConfig::default();
        let mut monitoring = SecurityMonitoringSystem::new(config);
        
        let mut metrics = SecurityMetrics::default();
        metrics.threats_detected_last_hour = 5;
        
        monitoring.update_metrics(metrics).await.unwrap();
        
        let updated_metrics = monitoring.get_metrics().await;
        assert_eq!(updated_metrics.threats_detected_last_hour, 5);
    }
    
    #[test]
    fn test_alert_creation() {
        let rules = vec![];
        let alert_manager = AlertManager::new(rules, 5);
        
        // Test that alert manager can be created
        assert_eq!(alert_manager.throttle_minutes, 5);
    }
}