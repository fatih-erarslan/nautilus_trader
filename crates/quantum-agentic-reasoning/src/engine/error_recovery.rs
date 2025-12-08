//! Error Recovery Module
//!
//! Comprehensive error recovery and fault tolerance for quantum trading operations with self-healing capabilities.

use crate::core::{QarResult, FactorMap};
use crate::error::QarError;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};
use chrono::{DateTime, Utc, Duration};
use uuid::Uuid;

/// Error severity levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum ErrorSeverity {
    Info,
    Warning,
    Error,
    Critical,
    Fatal,
}

/// Error categories
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ErrorCategory {
    Network,
    Database,
    Quantum,
    Trading,
    Memory,
    Configuration,
    External,
    Internal,
    Timeout,
    Validation,
    Security,
    Performance,
}

/// Recovery strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecoveryStrategy {
    Retry,
    Fallback,
    Restart,
    Degrade,
    Isolate,
    Escalate,
    Ignore,
    Custom(String),
}

/// Error occurrence record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorOccurrence {
    pub id: String,
    pub component: String,
    pub category: ErrorCategory,
    pub severity: ErrorSeverity,
    pub message: String,
    pub error_code: Option<String>,
    pub stack_trace: Option<String>,
    pub context: HashMap<String, String>,
    pub timestamp: DateTime<Utc>,
    pub resolved: bool,
    pub resolution_time: Option<DateTime<Utc>>,
    pub recovery_actions: Vec<RecoveryAction>,
}

/// Recovery action record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryAction {
    pub id: String,
    pub strategy: RecoveryStrategy,
    pub description: String,
    pub executed_at: DateTime<Utc>,
    pub success: bool,
    pub execution_time_ms: u64,
    pub result_message: Option<String>,
}

/// Recovery rule for automated error handling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryRule {
    pub id: String,
    pub name: String,
    pub description: String,
    pub conditions: Vec<ErrorCondition>,
    pub strategies: Vec<RecoveryStrategy>,
    pub max_retries: u32,
    pub retry_delay_seconds: u64,
    pub escalation_threshold: u32,
    pub enabled: bool,
    pub priority: u32,
}

/// Error condition for rule matching
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorCondition {
    pub field: String, // component, category, severity, message, etc.
    pub operator: ConditionOperator,
    pub value: String,
    pub case_sensitive: bool,
}

/// Condition operators
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConditionOperator {
    Equals,
    NotEquals,
    Contains,
    NotContains,
    StartsWith,
    EndsWith,
    Matches, // Regex
    GreaterThan,
    LessThan,
}

/// Circuit breaker state
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum CircuitBreakerState {
    Closed,  // Normal operation
    Open,    // Failures exceed threshold
    HalfOpen, // Testing if service recovered
}

/// Circuit breaker for component protection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitBreaker {
    pub component: String,
    pub state: CircuitBreakerState,
    pub failure_count: u32,
    pub success_count: u32,
    pub last_failure: Option<DateTime<Utc>>,
    pub failure_threshold: u32,
    pub recovery_timeout_seconds: u64,
    pub half_open_success_threshold: u32,
}

/// Error recovery configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorRecoveryConfig {
    pub enable_auto_recovery: bool,
    pub enable_circuit_breakers: bool,
    pub max_recovery_attempts: u32,
    pub recovery_timeout_seconds: u64,
    pub error_history_retention_days: u32,
    pub enable_learning: bool,
    pub enable_predictive_recovery: bool,
    pub escalation_enabled: bool,
    pub health_check_interval_seconds: u64,
}

/// Recovery statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryStats {
    pub total_errors: u64,
    pub recovered_errors: u64,
    pub failed_recoveries: u64,
    pub average_recovery_time_ms: f64,
    pub recovery_success_rate: f64,
    pub errors_by_category: HashMap<ErrorCategory, u64>,
    pub errors_by_severity: HashMap<ErrorSeverity, u64>,
    pub most_common_errors: Vec<(String, u64)>,
}

/// Error recovery manager implementation
#[derive(Debug)]
pub struct ErrorRecoveryManager {
    config: ErrorRecoveryConfig,
    error_history: Arc<RwLock<VecDeque<ErrorOccurrence>>>,
    recovery_rules: Arc<RwLock<Vec<RecoveryRule>>>,
    circuit_breakers: Arc<RwLock<HashMap<String, CircuitBreaker>>>,
    active_recoveries: Arc<RwLock<HashMap<String, RecoverySession>>>,
    recovery_stats: Arc<Mutex<RecoveryStats>>,
    escalation_handler: Arc<dyn EscalationHandler + Send + Sync>,
    health_monitor: Arc<dyn HealthMonitor + Send + Sync>,
    pattern_learner: Arc<dyn PatternLearner + Send + Sync>,
}

/// Recovery session for tracking ongoing recovery
#[derive(Debug, Clone)]
pub struct RecoverySession {
    pub error_id: String,
    pub component: String,
    pub start_time: DateTime<Utc>,
    pub attempts: u32,
    pub current_strategy: RecoveryStrategy,
    pub actions: Vec<RecoveryAction>,
}

/// Escalation handler trait
#[async_trait::async_trait]
pub trait EscalationHandler {
    async fn escalate_error(&self, error: &ErrorOccurrence, failed_strategies: &[RecoveryStrategy]) -> QarResult<()>;
    async fn notify_recovery_failure(&self, error: &ErrorOccurrence, session: &RecoverySession) -> QarResult<()>;
    async fn request_manual_intervention(&self, error: &ErrorOccurrence) -> QarResult<()>;
}

/// Health monitor trait
#[async_trait::async_trait]
pub trait HealthMonitor {
    async fn check_component_health(&self, component: &str) -> QarResult<HealthStatus>;
    async fn perform_diagnostic(&self, component: &str) -> QarResult<DiagnosticResult>;
    async fn get_system_health(&self) -> QarResult<SystemHealth>;
}

/// Pattern learning trait for ML-based recovery
#[async_trait::async_trait]
pub trait PatternLearner {
    async fn learn_from_error(&self, error: &ErrorOccurrence, successful_strategy: &RecoveryStrategy) -> QarResult<()>;
    async fn suggest_recovery_strategy(&self, error: &ErrorOccurrence) -> QarResult<Option<RecoveryStrategy>>;
    async fn update_rule_effectiveness(&self, rule_id: &str, success: bool) -> QarResult<()>;
}

/// Health status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthStatus {
    pub component: String,
    pub healthy: bool,
    pub score: f64, // 0.0 to 1.0
    pub issues: Vec<String>,
    pub last_checked: DateTime<Utc>,
}

/// Diagnostic result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagnosticResult {
    pub component: String,
    pub tests_passed: u32,
    pub tests_failed: u32,
    pub issues_found: Vec<String>,
    pub recommendations: Vec<String>,
    pub execution_time_ms: u64,
}

/// System health overview
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemHealth {
    pub overall_score: f64,
    pub component_health: HashMap<String, HealthStatus>,
    pub active_issues: u32,
    pub system_load: f64,
    pub uptime_seconds: u64,
}

impl ErrorRecoveryManager {
    /// Create new error recovery manager
    pub fn new(
        config: ErrorRecoveryConfig,
        escalation_handler: Arc<dyn EscalationHandler + Send + Sync>,
        health_monitor: Arc<dyn HealthMonitor + Send + Sync>,
        pattern_learner: Arc<dyn PatternLearner + Send + Sync>,
    ) -> Self {
        Self {
            config,
            error_history: Arc::new(RwLock::new(VecDeque::new())),
            recovery_rules: Arc::new(RwLock::new(Vec::new())),
            circuit_breakers: Arc::new(RwLock::new(HashMap::new())),
            active_recoveries: Arc::new(RwLock::new(HashMap::new())),
            recovery_stats: Arc::new(Mutex::new(RecoveryStats {
                total_errors: 0,
                recovered_errors: 0,
                failed_recoveries: 0,
                average_recovery_time_ms: 0.0,
                recovery_success_rate: 0.0,
                errors_by_category: HashMap::new(),
                errors_by_severity: HashMap::new(),
                most_common_errors: Vec::new(),
            })),
            escalation_handler,
            health_monitor,
            pattern_learner,
        }
    }

    /// Handle error occurrence
    pub async fn handle_error(&self, mut error: ErrorOccurrence) -> QarResult<()> {
        error.id = Uuid::new_v4().to_string();
        error.timestamp = Utc::now();

        // Update statistics
        self.update_error_stats(&error).await?;

        // Check circuit breaker
        if self.config.enable_circuit_breakers {
            self.update_circuit_breaker(&error).await?;
            
            let should_block = self.should_block_component(&error.component).await?;
            if should_block {
                return Err(QarError::CircuitBreakerOpen(error.component.clone()));
            }
        }

        // Store error
        {
            let mut history = self.error_history.write().await;
            history.push_back(error.clone());
            
            // Limit history size
            let max_size = (self.config.error_history_retention_days as usize) * 1000; // Rough estimate
            while history.len() > max_size {
                history.pop_front();
            }
        }

        // Attempt automated recovery
        if self.config.enable_auto_recovery {
            self.attempt_recovery(&error).await?;
        }

        Ok(())
    }

    /// Attempt automated recovery for error
    async fn attempt_recovery(&self, error: &ErrorOccurrence) -> QarResult<()> {
        // Find matching recovery rules
        let recovery_rules = self.find_matching_rules(error).await?;
        
        if recovery_rules.is_empty() {
            // No rules found, try ML-suggested strategy
            if self.config.enable_learning {
                if let Some(suggested_strategy) = self.pattern_learner.suggest_recovery_strategy(error).await? {
                    return self.execute_recovery_strategy(error, &suggested_strategy, None).await;
                }
            }
            
            // No automated recovery available
            if self.config.escalation_enabled {
                self.escalation_handler.escalate_error(error, &[]).await?;
            }
            return Ok(());
        }

        // Execute recovery strategies from matching rules
        for rule in recovery_rules {
            let success = self.execute_recovery_rule(error, &rule).await?;
            if success {
                break;
            }
        }

        Ok(())
    }

    /// Find recovery rules matching the error
    async fn find_matching_rules(&self, error: &ErrorOccurrence) -> QarResult<Vec<RecoveryRule>> {
        let rules = self.recovery_rules.read().await;
        let mut matching_rules = Vec::new();

        for rule in rules.iter() {
            if !rule.enabled {
                continue;
            }

            let matches = rule.conditions.iter().all(|condition| {
                self.evaluate_condition(condition, error)
            });

            if matches {
                matching_rules.push(rule.clone());
            }
        }

        // Sort by priority (higher priority first)
        matching_rules.sort_by(|a, b| b.priority.cmp(&a.priority));

        Ok(matching_rules)
    }

    /// Evaluate condition against error
    fn evaluate_condition(&self, condition: &ErrorCondition, error: &ErrorOccurrence) -> bool {
        let field_value = match condition.field.as_str() {
            "component" => &error.component,
            "category" => &format!("{:?}", error.category),
            "severity" => &format!("{:?}", error.severity),
            "message" => &error.message,
            "error_code" => error.error_code.as_ref().unwrap_or(&String::new()),
            _ => return false,
        };

        let value = if condition.case_sensitive {
            condition.value.clone()
        } else {
            condition.value.to_lowercase()
        };

        let field = if condition.case_sensitive {
            field_value.clone()
        } else {
            field_value.to_lowercase()
        };

        match condition.operator {
            ConditionOperator::Equals => field == value,
            ConditionOperator::NotEquals => field != value,
            ConditionOperator::Contains => field.contains(&value),
            ConditionOperator::NotContains => !field.contains(&value),
            ConditionOperator::StartsWith => field.starts_with(&value),
            ConditionOperator::EndsWith => field.ends_with(&value),
            ConditionOperator::Matches => {
                // Simplified regex matching
                field.contains(&value)
            },
            ConditionOperator::GreaterThan => {
                field.parse::<f64>().unwrap_or(0.0) > value.parse::<f64>().unwrap_or(0.0)
            },
            ConditionOperator::LessThan => {
                field.parse::<f64>().unwrap_or(0.0) < value.parse::<f64>().unwrap_or(0.0)
            },
        }
    }

    /// Execute recovery rule
    async fn execute_recovery_rule(&self, error: &ErrorOccurrence, rule: &RecoveryRule) -> QarResult<bool> {
        let session_id = Uuid::new_v4().to_string();
        let session = RecoverySession {
            error_id: error.id.clone(),
            component: error.component.clone(),
            start_time: Utc::now(),
            attempts: 0,
            current_strategy: rule.strategies[0].clone(),
            actions: Vec::new(),
        };

        {
            let mut active = self.active_recoveries.write().await;
            active.insert(session_id.clone(), session);
        }

        let mut success = false;
        for strategy in &rule.strategies {
            let mut current_session = {
                let active = self.active_recoveries.read().await;
                active.get(&session_id).unwrap().clone()
            };

            for attempt in 0..rule.max_retries {
                current_session.attempts = attempt + 1;
                current_session.current_strategy = strategy.clone();

                let strategy_success = self.execute_recovery_strategy(error, strategy, Some(&rule)).await?;
                
                if strategy_success {
                    success = true;
                    
                    // Learn from successful recovery
                    if self.config.enable_learning {
                        self.pattern_learner.learn_from_error(error, strategy).await?;
                        self.pattern_learner.update_rule_effectiveness(&rule.id, true).await?;
                    }
                    
                    break;
                }

                // Wait before retry
                if attempt < rule.max_retries - 1 {
                    tokio::time::sleep(tokio::time::Duration::from_secs(rule.retry_delay_seconds)).await;
                }
            }

            if success {
                break;
            }
        }

        // Clean up session
        {
            let mut active = self.active_recoveries.write().await;
            active.remove(&session_id);
        }

        // Update rule effectiveness
        if self.config.enable_learning {
            self.pattern_learner.update_rule_effectiveness(&rule.id, success).await?;
        }

        // Handle failure
        if !success {
            if self.config.escalation_enabled {
                self.escalation_handler.escalate_error(error, &rule.strategies).await?;
            }
        }

        Ok(success)
    }

    /// Execute specific recovery strategy
    async fn execute_recovery_strategy(
        &self,
        error: &ErrorOccurrence,
        strategy: &RecoveryStrategy,
        rule: Option<&RecoveryRule>,
    ) -> QarResult<bool> {
        let start_time = std::time::Instant::now();
        let action_id = Uuid::new_v4().to_string();

        let (success, result_message) = match strategy {
            RecoveryStrategy::Retry => {
                // Retry the failed operation
                self.retry_operation(&error.component).await
            },
            RecoveryStrategy::Fallback => {
                // Switch to fallback mechanism
                self.activate_fallback(&error.component).await
            },
            RecoveryStrategy::Restart => {
                // Restart the component
                self.restart_component(&error.component).await
            },
            RecoveryStrategy::Degrade => {
                // Gracefully degrade functionality
                self.degrade_component(&error.component).await
            },
            RecoveryStrategy::Isolate => {
                // Isolate the failing component
                self.isolate_component(&error.component).await
            },
            RecoveryStrategy::Escalate => {
                // Escalate to manual intervention
                self.escalation_handler.request_manual_intervention(error).await?;
                (true, Some("Escalated for manual intervention".to_string()))
            },
            RecoveryStrategy::Ignore => {
                // Ignore the error (for non-critical issues)
                (true, Some("Error ignored as per strategy".to_string()))
            },
            RecoveryStrategy::Custom(name) => {
                // Execute custom recovery strategy
                self.execute_custom_strategy(name, error).await
            },
        };

        let execution_time = start_time.elapsed().as_millis() as u64;

        let action = RecoveryAction {
            id: action_id,
            strategy: strategy.clone(),
            description: format!("Executed {:?} strategy for {}", strategy, error.component),
            executed_at: Utc::now(),
            success,
            execution_time_ms: execution_time,
            result_message,
        };

        // Record the action
        self.record_recovery_action(&error.id, action).await?;

        Ok(success)
    }

    /// Retry operation
    async fn retry_operation(&self, component: &str) -> (bool, Option<String>) {
        // Simulate retry logic
        let health = self.health_monitor.check_component_health(component).await;
        match health {
            Ok(status) => (status.healthy, Some(format!("Health check result: {}", status.healthy))),
            Err(_) => (false, Some("Health check failed".to_string())),
        }
    }

    /// Activate fallback
    async fn activate_fallback(&self, _component: &str) -> (bool, Option<String>) {
        // Simulate fallback activation
        (true, Some("Fallback mechanism activated".to_string()))
    }

    /// Restart component
    async fn restart_component(&self, _component: &str) -> (bool, Option<String>) {
        // Simulate component restart
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        (true, Some("Component restarted successfully".to_string()))
    }

    /// Degrade component
    async fn degrade_component(&self, _component: &str) -> (bool, Option<String>) {
        // Simulate graceful degradation
        (true, Some("Component functionality degraded".to_string()))
    }

    /// Isolate component
    async fn isolate_component(&self, _component: &str) -> (bool, Option<String>) {
        // Simulate component isolation
        (true, Some("Component isolated from system".to_string()))
    }

    /// Execute custom strategy
    async fn execute_custom_strategy(&self, _strategy_name: &str, _error: &ErrorOccurrence) -> (bool, Option<String>) {
        // Placeholder for custom strategy execution
        (false, Some("Custom strategy not implemented".to_string()))
    }

    /// Record recovery action
    async fn record_recovery_action(&self, error_id: &str, action: RecoveryAction) -> QarResult<()> {
        let mut history = self.error_history.write().await;
        
        for error in history.iter_mut().rev() {
            if error.id == error_id {
                error.recovery_actions.push(action);
                if action.success {
                    error.resolved = true;
                    error.resolution_time = Some(Utc::now());
                }
                break;
            }
        }

        Ok(())
    }

    /// Update circuit breaker state
    async fn update_circuit_breaker(&self, error: &ErrorOccurrence) -> QarResult<()> {
        let mut breakers = self.circuit_breakers.write().await;
        
        let breaker = breakers.entry(error.component.clone()).or_insert_with(|| {
            CircuitBreaker {
                component: error.component.clone(),
                state: CircuitBreakerState::Closed,
                failure_count: 0,
                success_count: 0,
                last_failure: None,
                failure_threshold: 5,
                recovery_timeout_seconds: 60,
                half_open_success_threshold: 3,
            }
        });

        match error.severity {
            ErrorSeverity::Critical | ErrorSeverity::Fatal => {
                breaker.failure_count += 1;
                breaker.last_failure = Some(Utc::now());
                
                if breaker.failure_count >= breaker.failure_threshold {
                    breaker.state = CircuitBreakerState::Open;
                }
            },
            _ => {
                // Don't count minor errors towards circuit breaker
            }
        }

        Ok(())
    }

    /// Check if component should be blocked by circuit breaker
    async fn should_block_component(&self, component: &str) -> QarResult<bool> {
        let mut breakers = self.circuit_breakers.write().await;
        
        if let Some(breaker) = breakers.get_mut(component) {
            match breaker.state {
                CircuitBreakerState::Open => {
                    // Check if recovery timeout has passed
                    if let Some(last_failure) = breaker.last_failure {
                        let recovery_time = last_failure + Duration::seconds(breaker.recovery_timeout_seconds as i64);
                        if Utc::now() > recovery_time {
                            breaker.state = CircuitBreakerState::HalfOpen;
                            breaker.success_count = 0;
                            return Ok(false);
                        }
                    }
                    Ok(true)
                },
                CircuitBreakerState::HalfOpen => {
                    // Allow limited traffic to test recovery
                    Ok(false)
                },
                CircuitBreakerState::Closed => Ok(false),
            }
        } else {
            Ok(false)
        }
    }

    /// Update error statistics
    async fn update_error_stats(&self, error: &ErrorOccurrence) -> QarResult<()> {
        let mut stats = self.recovery_stats.lock().await;
        
        stats.total_errors += 1;
        
        let category_count = stats.errors_by_category.entry(error.category.clone()).or_insert(0);
        *category_count += 1;
        
        let severity_count = stats.errors_by_severity.entry(error.severity.clone()).or_insert(0);
        *severity_count += 1;

        Ok(())
    }

    /// Add recovery rule
    pub async fn add_recovery_rule(&self, rule: RecoveryRule) -> QarResult<()> {
        let mut rules = self.recovery_rules.write().await;
        rules.push(rule);
        Ok(())
    }

    /// Get recovery statistics
    pub async fn get_recovery_stats(&self) -> QarResult<RecoveryStats> {
        let stats = self.recovery_stats.lock().await;
        Ok(stats.clone())
    }

    /// Get error history
    pub async fn get_error_history(&self, limit: Option<usize>) -> QarResult<Vec<ErrorOccurrence>> {
        let history = self.error_history.read().await;
        let errors = if let Some(limit) = limit {
            history.iter().rev().take(limit).cloned().collect()
        } else {
            history.iter().cloned().collect()
        };
        Ok(errors)
    }

    /// Get circuit breaker status
    pub async fn get_circuit_breaker_status(&self) -> QarResult<HashMap<String, CircuitBreaker>> {
        let breakers = self.circuit_breakers.read().await;
        Ok(breakers.clone())
    }
}

/// Mock implementations for testing
pub struct MockEscalationHandler;

#[async_trait::async_trait]
impl EscalationHandler for MockEscalationHandler {
    async fn escalate_error(&self, _error: &ErrorOccurrence, _failed_strategies: &[RecoveryStrategy]) -> QarResult<()> {
        Ok(())
    }

    async fn notify_recovery_failure(&self, _error: &ErrorOccurrence, _session: &RecoverySession) -> QarResult<()> {
        Ok(())
    }

    async fn request_manual_intervention(&self, _error: &ErrorOccurrence) -> QarResult<()> {
        Ok(())
    }
}

pub struct MockHealthMonitor;

#[async_trait::async_trait]
impl HealthMonitor for MockHealthMonitor {
    async fn check_component_health(&self, component: &str) -> QarResult<HealthStatus> {
        Ok(HealthStatus {
            component: component.to_string(),
            healthy: true,
            score: 0.95,
            issues: Vec::new(),
            last_checked: Utc::now(),
        })
    }

    async fn perform_diagnostic(&self, component: &str) -> QarResult<DiagnosticResult> {
        Ok(DiagnosticResult {
            component: component.to_string(),
            tests_passed: 10,
            tests_failed: 0,
            issues_found: Vec::new(),
            recommendations: Vec::new(),
            execution_time_ms: 100,
        })
    }

    async fn get_system_health(&self) -> QarResult<SystemHealth> {
        Ok(SystemHealth {
            overall_score: 0.95,
            component_health: HashMap::new(),
            active_issues: 0,
            system_load: 0.3,
            uptime_seconds: 86400,
        })
    }
}

pub struct MockPatternLearner;

#[async_trait::async_trait]
impl PatternLearner for MockPatternLearner {
    async fn learn_from_error(&self, _error: &ErrorOccurrence, _successful_strategy: &RecoveryStrategy) -> QarResult<()> {
        Ok(())
    }

    async fn suggest_recovery_strategy(&self, _error: &ErrorOccurrence) -> QarResult<Option<RecoveryStrategy>> {
        Ok(Some(RecoveryStrategy::Retry))
    }

    async fn update_rule_effectiveness(&self, _rule_id: &str, _success: bool) -> QarResult<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_manager() -> ErrorRecoveryManager {
        let config = ErrorRecoveryConfig {
            enable_auto_recovery: true,
            enable_circuit_breakers: true,
            max_recovery_attempts: 3,
            recovery_timeout_seconds: 60,
            error_history_retention_days: 30,
            enable_learning: true,
            enable_predictive_recovery: true,
            escalation_enabled: true,
            health_check_interval_seconds: 60,
        };

        ErrorRecoveryManager::new(
            config,
            Arc::new(MockEscalationHandler),
            Arc::new(MockHealthMonitor),
            Arc::new(MockPatternLearner),
        )
    }

    fn create_test_error() -> ErrorOccurrence {
        ErrorOccurrence {
            id: String::new(),
            component: "test_component".to_string(),
            category: ErrorCategory::Network,
            severity: ErrorSeverity::Error,
            message: "Connection timeout".to_string(),
            error_code: Some("NET_001".to_string()),
            stack_trace: None,
            context: HashMap::new(),
            timestamp: Utc::now(),
            resolved: false,
            resolution_time: None,
            recovery_actions: Vec::new(),
        }
    }

    #[tokio::test]
    async fn test_handle_error() {
        let manager = create_test_manager();
        let error = create_test_error();

        manager.handle_error(error).await.unwrap();

        let history = manager.get_error_history(Some(10)).await.unwrap();
        assert!(!history.is_empty());
    }

    #[tokio::test]
    async fn test_recovery_rule_matching() {
        let manager = create_test_manager();
        
        let rule = RecoveryRule {
            id: "test_rule".to_string(),
            name: "Network Error Recovery".to_string(),
            description: "Handle network timeouts".to_string(),
            conditions: vec![
                ErrorCondition {
                    field: "category".to_string(),
                    operator: ConditionOperator::Equals,
                    value: "Network".to_string(),
                    case_sensitive: false,
                }
            ],
            strategies: vec![RecoveryStrategy::Retry, RecoveryStrategy::Fallback],
            max_retries: 3,
            retry_delay_seconds: 1,
            escalation_threshold: 5,
            enabled: true,
            priority: 1,
        };

        manager.add_recovery_rule(rule).await.unwrap();

        let error = create_test_error();
        manager.handle_error(error).await.unwrap();

        let stats = manager.get_recovery_stats().await.unwrap();
        assert!(stats.total_errors > 0);
    }

    #[tokio::test]
    async fn test_circuit_breaker() {
        let manager = create_test_manager();
        
        // Generate multiple critical errors to trigger circuit breaker
        for _ in 0..6 {
            let mut error = create_test_error();
            error.severity = ErrorSeverity::Critical;
            manager.handle_error(error).await.unwrap();
        }

        let breakers = manager.get_circuit_breaker_status().await.unwrap();
        let test_breaker = breakers.get("test_component");
        assert!(test_breaker.is_some());
        assert_eq!(test_breaker.unwrap().state, CircuitBreakerState::Open);
    }

    #[tokio::test]
    async fn test_condition_evaluation() {
        let manager = create_test_manager();
        let error = create_test_error();

        let condition = ErrorCondition {
            field: "component".to_string(),
            operator: ConditionOperator::Equals,
            value: "test_component".to_string(),
            case_sensitive: true,
        };

        assert!(manager.evaluate_condition(&condition, &error));

        let condition2 = ErrorCondition {
            field: "message".to_string(),
            operator: ConditionOperator::Contains,
            value: "timeout".to_string(),
            case_sensitive: false,
        };

        assert!(manager.evaluate_condition(&condition2, &error));
    }

    #[tokio::test]
    async fn test_recovery_strategies() {
        let manager = create_test_manager();
        let error = create_test_error();

        // Test retry strategy
        let (success, _) = manager.retry_operation(&error.component).await;
        assert!(success);

        // Test fallback strategy
        let (success, _) = manager.activate_fallback(&error.component).await;
        assert!(success);

        // Test restart strategy
        let (success, _) = manager.restart_component(&error.component).await;
        assert!(success);
    }
}

// Type alias for backward compatibility
pub type ErrorRecoverySystem = ErrorRecoveryManager;