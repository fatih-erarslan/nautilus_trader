//! Compliance Enforcement Agent
//! 
//! Enforces zero-mock policies across all development workflows and integrations
//! Ensures complete compliance with authentic testing requirements

use crate::{TENGRIError, TradingOperation, TENGRIOversightResult, ViolationType, EmergencyAction};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::RwLock;
use chrono::{DateTime, Utc};
use uuid::Uuid;
use serde::{Deserialize, Serialize};
use std::time::Instant;
use regex::Regex;

/// Compliance Policy Framework
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompliancePolicy {
    pub policy_id: String,
    pub policy_name: String,
    pub enforcement_level: EnforcementLevel,
    pub scope: PolicyScope,
    pub rules: Vec<ComplianceRule>,
    pub violations: Vec<ViolationType>,
    pub remediation_actions: Vec<RemediationAction>,
    pub escalation_thresholds: EscalationThresholds,
    pub effective_date: DateTime<Utc>,
    pub review_date: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EnforcementLevel {
    Advisory,      // Warning only
    Mandatory,     // Block operations
    Critical,      // Immediate shutdown
    Emergency,     // Emergency protocols
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PolicyScope {
    Global,        // All operations
    Agent,         // Specific agent
    Operation,     // Specific operation type
    Development,   // Development phase
    Testing,       // Testing phase
    Production,    // Production phase
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceRule {
    pub rule_id: String,
    pub rule_name: String,
    pub description: String,
    pub rule_type: RuleType,
    pub pattern: String,
    pub severity: RuleSeverity,
    pub action: RuleAction,
    pub confidence_threshold: f64,
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RuleType {
    Pattern,       // Regex pattern matching
    Behavior,      // Behavioral analysis
    Statistical,   // Statistical analysis
    Temporal,      // Time-based analysis
    Structural,    // Code structure analysis
    Network,       // Network connection analysis
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RuleSeverity {
    Info,
    Warning,
    Error,
    Critical,
    Emergency,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RuleAction {
    Log,
    Warn,
    Block,
    Quarantine,
    Shutdown,
    Escalate,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemediationAction {
    pub action_id: String,
    pub action_type: String,
    pub description: String,
    pub automated: bool,
    pub estimated_duration: u64,
    pub prerequisites: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EscalationThresholds {
    pub warning_threshold: u64,
    pub error_threshold: u64,
    pub critical_threshold: u64,
    pub emergency_threshold: u64,
    pub time_window_minutes: u64,
}

/// Compliance Violation Record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceViolation {
    pub violation_id: Uuid,
    pub operation_id: Uuid,
    pub agent_id: String,
    pub policy_id: String,
    pub rule_id: String,
    pub violation_type: ViolationType,
    pub severity: RuleSeverity,
    pub description: String,
    pub evidence: Vec<String>,
    pub confidence: f64,
    pub timestamp: DateTime<Utc>,
    pub status: ViolationStatus,
    pub remediation_actions: Vec<String>,
    pub escalation_level: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ViolationStatus {
    Detected,
    Investigating,
    Confirmed,
    Remediated,
    Escalated,
    Closed,
}

/// Compliance Enforcement Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceConfig {
    pub strict_mode: bool,
    pub auto_remediation: bool,
    pub escalation_enabled: bool,
    pub notification_enabled: bool,
    pub audit_logging: bool,
    pub real_time_monitoring: bool,
    pub batch_processing: bool,
    pub enforcement_timeout_ms: u64,
    pub violation_retention_days: u32,
    pub policy_update_frequency_hours: u32,
}

impl Default for ComplianceConfig {
    fn default() -> Self {
        Self {
            strict_mode: true,
            auto_remediation: true,
            escalation_enabled: true,
            notification_enabled: true,
            audit_logging: true,
            real_time_monitoring: true,
            batch_processing: false,
            enforcement_timeout_ms: 100,
            violation_retention_days: 365,
            policy_update_frequency_hours: 24,
        }
    }
}

/// Compliance Enforcement Agent
pub struct ComplianceEnforcementAgent {
    config: ComplianceConfig,
    policies: Arc<RwLock<HashMap<String, CompliancePolicy>>>,
    violations: Arc<RwLock<HashMap<Uuid, ComplianceViolation>>>,
    enforcement_rules: Arc<RwLock<HashMap<String, Regex>>>,
    escalation_counters: Arc<RwLock<HashMap<String, u64>>>,
    remediation_queue: Arc<RwLock<Vec<RemediationTask>>>,
    audit_trail: Arc<RwLock<Vec<AuditEvent>>>,
    policy_engine: Arc<RwLock<PolicyEngine>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemediationTask {
    pub task_id: Uuid,
    pub violation_id: Uuid,
    pub action: RemediationAction,
    pub status: TaskStatus,
    pub created_at: DateTime<Utc>,
    pub scheduled_at: DateTime<Utc>,
    pub completed_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TaskStatus {
    Pending,
    InProgress,
    Completed,
    Failed,
    Cancelled,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEvent {
    pub event_id: Uuid,
    pub event_type: String,
    pub description: String,
    pub agent_id: String,
    pub operation_id: Option<Uuid>,
    pub timestamp: DateTime<Utc>,
    pub metadata: HashMap<String, String>,
}

/// Policy Engine for rule evaluation
#[derive(Debug, Clone)]
pub struct PolicyEngine {
    pub active_policies: HashSet<String>,
    pub rule_cache: HashMap<String, Vec<ComplianceRule>>,
    pub pattern_cache: HashMap<String, Regex>,
}

impl PolicyEngine {
    pub fn new() -> Self {
        Self {
            active_policies: HashSet::new(),
            rule_cache: HashMap::new(),
            pattern_cache: HashMap::new(),
        }
    }

    pub fn load_policy(&mut self, policy: &CompliancePolicy) -> Result<(), TENGRIError> {
        // Compile regex patterns
        for rule in &policy.rules {
            if rule.enabled && matches!(rule.rule_type, RuleType::Pattern) {
                let regex = Regex::new(&rule.pattern)
                    .map_err(|e| TENGRIError::InvalidConfiguration { 
                        reason: format!("Invalid regex pattern in rule {}: {}", rule.rule_id, e) 
                    })?;
                self.pattern_cache.insert(rule.rule_id.clone(), regex);
            }
        }

        self.active_policies.insert(policy.policy_id.clone());
        self.rule_cache.insert(policy.policy_id.clone(), policy.rules.clone());
        Ok(())
    }

    pub fn evaluate_rules(&self, operation: &TradingOperation, policy_id: &str) -> Vec<ComplianceViolation> {
        let mut violations = Vec::new();
        
        if let Some(rules) = self.rule_cache.get(policy_id) {
            for rule in rules {
                if !rule.enabled {
                    continue;
                }

                let violation = match rule.rule_type {
                    RuleType::Pattern => self.evaluate_pattern_rule(operation, rule),
                    RuleType::Behavior => self.evaluate_behavior_rule(operation, rule),
                    RuleType::Statistical => self.evaluate_statistical_rule(operation, rule),
                    RuleType::Temporal => self.evaluate_temporal_rule(operation, rule),
                    RuleType::Structural => self.evaluate_structural_rule(operation, rule),
                    RuleType::Network => self.evaluate_network_rule(operation, rule),
                };

                if let Some(mut violation) = violation {
                    violation.policy_id = policy_id.to_string();
                    violations.push(violation);
                }
            }
        }

        violations
    }

    fn evaluate_pattern_rule(&self, operation: &TradingOperation, rule: &ComplianceRule) -> Option<ComplianceViolation> {
        if let Some(regex) = self.pattern_cache.get(&rule.rule_id) {
            if regex.is_match(&operation.data_source) {
                return Some(ComplianceViolation {
                    violation_id: Uuid::new_v4(),
                    operation_id: operation.id,
                    agent_id: operation.agent_id.clone(),
                    policy_id: String::new(), // Will be filled by caller
                    rule_id: rule.rule_id.clone(),
                    violation_type: ViolationType::SyntheticData,
                    severity: rule.severity.clone(),
                    description: format!("Pattern rule violation: {}", rule.description),
                    evidence: vec![format!("Pattern matched: {}", rule.pattern)],
                    confidence: 0.95,
                    timestamp: Utc::now(),
                    status: ViolationStatus::Detected,
                    remediation_actions: Vec::new(),
                    escalation_level: 0,
                });
            }
        }
        None
    }

    fn evaluate_behavior_rule(&self, operation: &TradingOperation, rule: &ComplianceRule) -> Option<ComplianceViolation> {
        // Behavioral analysis would be implemented here
        // For now, we return None
        None
    }

    fn evaluate_statistical_rule(&self, operation: &TradingOperation, rule: &ComplianceRule) -> Option<ComplianceViolation> {
        // Statistical analysis would be implemented here
        // For now, we return None
        None
    }

    fn evaluate_temporal_rule(&self, operation: &TradingOperation, rule: &ComplianceRule) -> Option<ComplianceViolation> {
        // Temporal analysis would be implemented here
        // For now, we return None
        None
    }

    fn evaluate_structural_rule(&self, operation: &TradingOperation, rule: &ComplianceRule) -> Option<ComplianceViolation> {
        // Structural analysis would be implemented here
        // For now, we return None
        None
    }

    fn evaluate_network_rule(&self, operation: &TradingOperation, rule: &ComplianceRule) -> Option<ComplianceViolation> {
        // Network analysis would be implemented here
        // For now, we return None
        None
    }
}

impl ComplianceEnforcementAgent {
    /// Initialize Compliance Enforcement Agent
    pub async fn new(config: ComplianceConfig) -> Result<Self, TENGRIError> {
        let policies = Arc::new(RwLock::new(HashMap::new()));
        let violations = Arc::new(RwLock::new(HashMap::new()));
        let enforcement_rules = Arc::new(RwLock::new(HashMap::new()));
        let escalation_counters = Arc::new(RwLock::new(HashMap::new()));
        let remediation_queue = Arc::new(RwLock::new(Vec::new()));
        let audit_trail = Arc::new(RwLock::new(Vec::new()));
        let policy_engine = Arc::new(RwLock::new(PolicyEngine::new()));

        let agent = Self {
            config,
            policies,
            violations,
            enforcement_rules,
            escalation_counters,
            remediation_queue,
            audit_trail,
            policy_engine,
        };

        // Load default zero-mock policies
        agent.load_default_policies().await?;

        Ok(agent)
    }

    /// Enforce compliance for trading operation
    pub async fn enforce_compliance(&self, operation: &TradingOperation) -> Result<TENGRIOversightResult, TENGRIError> {
        let enforcement_start = Instant::now();
        
        // Evaluate all active policies
        let violations = self.evaluate_all_policies(operation).await?;
        
        // Process violations
        let enforcement_result = self.process_violations(violations, operation).await?;
        
        // Update escalation counters
        self.update_escalation_counters(&enforcement_result, operation).await?;
        
        // Log audit event
        self.log_audit_event("compliance_enforcement", operation, &enforcement_result).await;
        
        // Check enforcement timeout
        let enforcement_duration = enforcement_start.elapsed();
        if enforcement_duration.as_millis() > self.config.enforcement_timeout_ms {
            tracing::warn!("Compliance enforcement exceeded timeout: {:?}", enforcement_duration);
        }

        Ok(enforcement_result)
    }

    /// Evaluate all active policies
    async fn evaluate_all_policies(&self, operation: &TradingOperation) -> Result<Vec<ComplianceViolation>, TENGRIError> {
        let mut all_violations = Vec::new();
        let policies = self.policies.read().await;
        let policy_engine = self.policy_engine.read().await;

        for policy_id in &policy_engine.active_policies {
            let violations = policy_engine.evaluate_rules(operation, policy_id);
            all_violations.extend(violations);
        }

        // Store violations
        let mut violations_store = self.violations.write().await;
        for violation in &all_violations {
            violations_store.insert(violation.violation_id, violation.clone());
        }

        Ok(all_violations)
    }

    /// Process violations and determine enforcement action
    async fn process_violations(&self, violations: Vec<ComplianceViolation>, operation: &TradingOperation) -> Result<TENGRIOversightResult, TENGRIError> {
        if violations.is_empty() {
            return Ok(TENGRIOversightResult::Approved);
        }

        // Find highest severity violation
        let max_severity = violations.iter()
            .map(|v| match v.severity {
                RuleSeverity::Emergency => 5,
                RuleSeverity::Critical => 4,
                RuleSeverity::Error => 3,
                RuleSeverity::Warning => 2,
                RuleSeverity::Info => 1,
            })
            .max()
            .unwrap_or(1);

        // Determine enforcement action based on severity
        match max_severity {
            5 => {
                // Emergency - immediate shutdown
                self.trigger_emergency_actions(&violations, operation).await?;
                Ok(TENGRIOversightResult::CriticalViolation {
                    violation_type: ViolationType::SecurityViolation,
                    immediate_shutdown: true,
                    forensic_data: serde_json::to_vec(&violations).unwrap_or_default(),
                })
            }
            4 => {
                // Critical - block operation
                self.trigger_critical_actions(&violations, operation).await?;
                Ok(TENGRIOversightResult::CriticalViolation {
                    violation_type: ViolationType::IntegrityBreach,
                    immediate_shutdown: false,
                    forensic_data: serde_json::to_vec(&violations).unwrap_or_default(),
                })
            }
            3 => {
                // Error - reject operation
                self.trigger_error_actions(&violations, operation).await?;
                Ok(TENGRIOversightResult::Rejected {
                    reason: format!("Compliance violations detected: {}", violations.len()),
                    emergency_action: EmergencyAction::QuarantineAgent {
                        agent_id: operation.agent_id.clone(),
                    },
                })
            }
            2 => {
                // Warning - allow with corrective action
                self.trigger_warning_actions(&violations, operation).await?;
                Ok(TENGRIOversightResult::Warning {
                    reason: format!("Compliance warnings detected: {}", violations.len()),
                    corrective_action: "Review and address compliance issues".to_string(),
                })
            }
            _ => {
                // Info - log only
                self.log_info_violations(&violations, operation).await;
                Ok(TENGRIOversightResult::Approved)
            }
        }
    }

    /// Trigger emergency actions
    async fn trigger_emergency_actions(&self, violations: &[ComplianceViolation], operation: &TradingOperation) -> Result<(), TENGRIError> {
        tracing::error!("EMERGENCY: Compliance violations detected - Operation: {} - Violations: {}", operation.id, violations.len());
        
        // Schedule immediate remediation
        for violation in violations {
            self.schedule_remediation(violation, true).await?;
        }

        // Escalate to highest level
        self.escalate_violations(violations, 9).await?;

        Ok(())
    }

    /// Trigger critical actions
    async fn trigger_critical_actions(&self, violations: &[ComplianceViolation], operation: &TradingOperation) -> Result<(), TENGRIError> {
        tracing::error!("CRITICAL: Compliance violations detected - Operation: {} - Violations: {}", operation.id, violations.len());
        
        // Schedule urgent remediation
        for violation in violations {
            self.schedule_remediation(violation, false).await?;
        }

        // Escalate to high level
        self.escalate_violations(violations, 7).await?;

        Ok(())
    }

    /// Trigger error actions
    async fn trigger_error_actions(&self, violations: &[ComplianceViolation], operation: &TradingOperation) -> Result<(), TENGRIError> {
        tracing::warn!("ERROR: Compliance violations detected - Operation: {} - Violations: {}", operation.id, violations.len());
        
        // Schedule standard remediation
        for violation in violations {
            self.schedule_remediation(violation, false).await?;
        }

        // Escalate to medium level
        self.escalate_violations(violations, 5).await?;

        Ok(())
    }

    /// Trigger warning actions
    async fn trigger_warning_actions(&self, violations: &[ComplianceViolation], operation: &TradingOperation) -> Result<(), TENGRIError> {
        tracing::warn!("WARNING: Compliance violations detected - Operation: {} - Violations: {}", operation.id, violations.len());
        
        // Schedule low-priority remediation
        for violation in violations {
            self.schedule_remediation(violation, false).await?;
        }

        Ok(())
    }

    /// Log info violations
    async fn log_info_violations(&self, violations: &[ComplianceViolation], operation: &TradingOperation) {
        tracing::info!("INFO: Compliance violations detected - Operation: {} - Violations: {}", operation.id, violations.len());
    }

    /// Schedule remediation task
    async fn schedule_remediation(&self, violation: &ComplianceViolation, immediate: bool) -> Result<(), TENGRIError> {
        let remediation_action = RemediationAction {
            action_id: Uuid::new_v4().to_string(),
            action_type: "violation_remediation".to_string(),
            description: format!("Remediate violation: {}", violation.description),
            automated: self.config.auto_remediation,
            estimated_duration: if immediate { 0 } else { 300 }, // 5 minutes
            prerequisites: Vec::new(),
        };

        let task = RemediationTask {
            task_id: Uuid::new_v4(),
            violation_id: violation.violation_id,
            action: remediation_action,
            status: TaskStatus::Pending,
            created_at: Utc::now(),
            scheduled_at: if immediate { Utc::now() } else { Utc::now() + chrono::Duration::minutes(5) },
            completed_at: None,
        };

        let mut queue = self.remediation_queue.write().await;
        queue.push(task);

        Ok(())
    }

    /// Escalate violations
    async fn escalate_violations(&self, violations: &[ComplianceViolation], level: u32) -> Result<(), TENGRIError> {
        if !self.config.escalation_enabled {
            return Ok(());
        }

        tracing::warn!("Escalating {} violations to level {}", violations.len(), level);
        
        // Update escalation counters
        let mut counters = self.escalation_counters.write().await;
        for violation in violations {
            *counters.entry(violation.agent_id.clone()).or_insert(0) += 1;
        }

        Ok(())
    }

    /// Update escalation counters
    async fn update_escalation_counters(&self, result: &TENGRIOversightResult, operation: &TradingOperation) -> Result<(), TENGRIError> {
        let mut counters = self.escalation_counters.write().await;
        
        match result {
            TENGRIOversightResult::CriticalViolation { .. } => {
                *counters.entry(operation.agent_id.clone()).or_insert(0) += 5;
            }
            TENGRIOversightResult::Rejected { .. } => {
                *counters.entry(operation.agent_id.clone()).or_insert(0) += 3;
            }
            TENGRIOversightResult::Warning { .. } => {
                *counters.entry(operation.agent_id.clone()).or_insert(0) += 1;
            }
            TENGRIOversightResult::Approved => {
                // No escalation needed
            }
        }

        Ok(())
    }

    /// Log audit event
    async fn log_audit_event(&self, event_type: &str, operation: &TradingOperation, result: &TENGRIOversightResult) {
        let event = AuditEvent {
            event_id: Uuid::new_v4(),
            event_type: event_type.to_string(),
            description: format!("Compliance enforcement for operation {}", operation.id),
            agent_id: operation.agent_id.clone(),
            operation_id: Some(operation.id),
            timestamp: Utc::now(),
            metadata: HashMap::new(),
        };

        let mut audit_trail = self.audit_trail.write().await;
        audit_trail.push(event);

        // Keep only last 10,000 events
        if audit_trail.len() > 10000 {
            audit_trail.drain(0..1000);
        }
    }

    /// Load default zero-mock policies
    async fn load_default_policies(&self) -> Result<(), TENGRIError> {
        let zero_mock_policy = self.create_zero_mock_policy().await?;
        self.register_policy(zero_mock_policy).await?;
        Ok(())
    }

    /// Create zero-mock policy
    async fn create_zero_mock_policy(&self) -> Result<CompliancePolicy, TENGRIError> {
        let mock_detection_rule = ComplianceRule {
            rule_id: "mock_detection".to_string(),
            rule_name: "Mock Framework Detection".to_string(),
            description: "Detect usage of mock frameworks".to_string(),
            rule_type: RuleType::Pattern,
            pattern: r"(mock|fake|stub|dummy|synthetic|generated|test_data)".to_string(),
            severity: RuleSeverity::Critical,
            action: RuleAction::Block,
            confidence_threshold: 0.95,
            enabled: true,
        };

        let localhost_detection_rule = ComplianceRule {
            rule_id: "localhost_detection".to_string(),
            rule_name: "Localhost Connection Detection".to_string(),
            description: "Detect localhost connections in non-development environments".to_string(),
            rule_type: RuleType::Pattern,
            pattern: r"(localhost|127\.0\.0\.1|0\.0\.0\.0)".to_string(),
            severity: RuleSeverity::Error,
            action: RuleAction::Block,
            confidence_threshold: 0.90,
            enabled: true,
        };

        let policy = CompliancePolicy {
            policy_id: "zero_mock_policy".to_string(),
            policy_name: "Zero Mock Testing Policy".to_string(),
            enforcement_level: EnforcementLevel::Critical,
            scope: PolicyScope::Global,
            rules: vec![mock_detection_rule, localhost_detection_rule],
            violations: vec![ViolationType::SyntheticData, ViolationType::IntegrityBreach],
            remediation_actions: vec![
                RemediationAction {
                    action_id: "remove_mocks".to_string(),
                    action_type: "code_remediation".to_string(),
                    description: "Remove mock frameworks and replace with real integrations".to_string(),
                    automated: false,
                    estimated_duration: 1800, // 30 minutes
                    prerequisites: vec!["code_review".to_string()],
                }
            ],
            escalation_thresholds: EscalationThresholds {
                warning_threshold: 3,
                error_threshold: 5,
                critical_threshold: 10,
                emergency_threshold: 15,
                time_window_minutes: 60,
            },
            effective_date: Utc::now(),
            review_date: Utc::now() + chrono::Duration::days(30),
        };

        Ok(policy)
    }

    /// Register policy
    pub async fn register_policy(&self, policy: CompliancePolicy) -> Result<(), TENGRIError> {
        let policy_id = policy.policy_id.clone();
        
        // Register with policy engine
        let mut policy_engine = self.policy_engine.write().await;
        policy_engine.load_policy(&policy)?;
        
        // Store policy
        let mut policies = self.policies.write().await;
        policies.insert(policy_id, policy);

        Ok(())
    }

    /// Get compliance statistics
    pub async fn get_compliance_stats(&self) -> Result<ComplianceStats, TENGRIError> {
        let violations = self.violations.read().await;
        let policies = self.policies.read().await;
        let escalation_counters = self.escalation_counters.read().await;
        let audit_trail = self.audit_trail.read().await;

        let total_violations = violations.len();
        let active_violations = violations.values().filter(|v| !matches!(v.status, ViolationStatus::Closed)).count();
        let critical_violations = violations.values().filter(|v| matches!(v.severity, RuleSeverity::Critical | RuleSeverity::Emergency)).count();

        Ok(ComplianceStats {
            total_policies: policies.len(),
            total_violations,
            active_violations,
            critical_violations,
            escalation_count: escalation_counters.values().sum(),
            audit_events: audit_trail.len(),
            compliance_rate: if total_violations > 0 { 
                (total_violations - active_violations) as f64 / total_violations as f64 
            } else { 
                1.0 
            },
        })
    }
}

/// Compliance Statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceStats {
    pub total_policies: usize,
    pub total_violations: usize,
    pub active_violations: usize,
    pub critical_violations: usize,
    pub escalation_count: u64,
    pub audit_events: usize,
    pub compliance_rate: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{OperationType, RiskParameters};

    #[tokio::test]
    async fn test_compliance_enforcement_agent() {
        let config = ComplianceConfig::default();
        let agent = ComplianceEnforcementAgent::new(config).await.unwrap();
        
        let operation = TradingOperation {
            id: Uuid::new_v4(),
            timestamp: Utc::now(),
            operation_type: OperationType::PlaceOrder,
            data_source: "authentic_market_data".to_string(),
            mathematical_model: "real_model".to_string(),
            risk_parameters: RiskParameters {
                max_position_size: 1000.0,
                stop_loss: Some(0.02),
                take_profit: Some(0.05),
                confidence_threshold: 0.95,
            },
            agent_id: "test_agent".to_string(),
        };

        let result = agent.enforce_compliance(&operation).await.unwrap();
        assert!(matches!(result, TENGRIOversightResult::Approved));
    }

    #[tokio::test]
    async fn test_mock_detection_violation() {
        let config = ComplianceConfig::default();
        let agent = ComplianceEnforcementAgent::new(config).await.unwrap();
        
        let operation = TradingOperation {
            id: Uuid::new_v4(),
            timestamp: Utc::now(),
            operation_type: OperationType::PlaceOrder,
            data_source: "mock_service_endpoint".to_string(),
            mathematical_model: "test_model".to_string(),
            risk_parameters: RiskParameters {
                max_position_size: 1000.0,
                stop_loss: Some(0.02),
                take_profit: Some(0.05),
                confidence_threshold: 0.95,
            },
            agent_id: "test_agent".to_string(),
        };

        let result = agent.enforce_compliance(&operation).await.unwrap();
        assert!(matches!(result, TENGRIOversightResult::CriticalViolation { .. }));
    }

    #[tokio::test]
    async fn test_policy_registration() {
        let config = ComplianceConfig::default();
        let agent = ComplianceEnforcementAgent::new(config).await.unwrap();
        
        let custom_policy = CompliancePolicy {
            policy_id: "custom_policy".to_string(),
            policy_name: "Custom Test Policy".to_string(),
            enforcement_level: EnforcementLevel::Mandatory,
            scope: PolicyScope::Global,
            rules: vec![],
            violations: vec![],
            remediation_actions: vec![],
            escalation_thresholds: EscalationThresholds {
                warning_threshold: 1,
                error_threshold: 2,
                critical_threshold: 3,
                emergency_threshold: 4,
                time_window_minutes: 30,
            },
            effective_date: Utc::now(),
            review_date: Utc::now() + chrono::Duration::days(30),
        };

        let result = agent.register_policy(custom_policy).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_compliance_stats() {
        let config = ComplianceConfig::default();
        let agent = ComplianceEnforcementAgent::new(config).await.unwrap();
        
        let stats = agent.get_compliance_stats().await.unwrap();
        assert_eq!(stats.total_policies, 1); // Default zero-mock policy
        assert_eq!(stats.compliance_rate, 1.0); // No violations yet
    }
}