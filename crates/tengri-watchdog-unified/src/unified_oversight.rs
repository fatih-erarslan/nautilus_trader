//! Unified Oversight Implementation
//! 
//! Central coordination and state management for all TENGRI watchdog components
//! Provides unified decision-making and emergency response coordination

use crate::{TENGRIError, TradingOperation, TENGRIOversightResult, ViolationType, EmergencyAction};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Unified oversight state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OversightState {
    pub session_id: Uuid,
    pub start_time: DateTime<Utc>,
    pub active_operations: HashMap<Uuid, ActiveOperation>,
    pub watchdog_states: WatchdogStates,
    pub system_health: SystemHealth,
    pub emergency_status: EmergencyStatus,
    pub oversight_metrics: OversightMetrics,
    pub coordination_log: Vec<CoordinationEvent>,
}

/// Active operation tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActiveOperation {
    pub operation_id: Uuid,
    pub operation: TradingOperation,
    pub start_time: DateTime<Utc>,
    pub watchdog_results: HashMap<String, WatchdogResult>,
    pub overall_status: OperationStatus,
    pub risk_score: f64,
    pub validation_progress: ValidationProgress,
}

/// Watchdog result wrapper
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WatchdogResult {
    pub watchdog_name: String,
    pub result: String, // Serialized TENGRIOversightResult
    pub timestamp: DateTime<Utc>,
    pub processing_time_ms: u64,
    pub confidence_score: f64,
}

/// Operation status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OperationStatus {
    Pending,
    InProgress,
    Approved,
    Rejected { reason: String },
    Emergency { action: String },
    Completed,
}

/// Validation progress tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationProgress {
    pub total_watchdogs: usize,
    pub completed_watchdogs: usize,
    pub failed_watchdogs: usize,
    pub progress_percentage: f64,
    pub estimated_completion: Option<DateTime<Utc>>,
}

/// States of all watchdog components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WatchdogStates {
    pub data_integrity: WatchdogState,
    pub scientific_rigor: WatchdogState,
    pub production_readiness: WatchdogState,
    pub synthetic_detection: WatchdogState,
    pub mathematical_validation: WatchdogState,
    pub emergency_protocols: WatchdogState,
}

/// Individual watchdog state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WatchdogState {
    pub status: WatchdogStatus,
    pub last_update: DateTime<Utc>,
    pub operations_processed: u64,
    pub average_processing_time_ms: f64,
    pub success_rate: f64,
    pub error_count: u64,
    pub performance_metrics: WatchdogPerformanceMetrics,
}

/// Watchdog status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WatchdogStatus {
    Active,
    Degraded { reason: String },
    Failed { error: String },
    Maintenance,
    Disabled,
}

/// Performance metrics for individual watchdogs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WatchdogPerformanceMetrics {
    pub response_time_p50: f64,
    pub response_time_p95: f64,
    pub response_time_p99: f64,
    pub throughput_ops_per_second: f64,
    pub memory_usage_mb: f64,
    pub cpu_usage_percent: f64,
}

/// Overall system health
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemHealth {
    pub overall_status: SystemStatus,
    pub health_score: f64,
    pub active_alerts: Vec<HealthAlert>,
    pub resource_utilization: ResourceUtilization,
    pub performance_indicators: PerformanceIndicators,
}

/// System status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SystemStatus {
    Healthy,
    Degraded,
    Critical,
    Emergency,
}

/// Health alert
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthAlert {
    pub alert_id: Uuid,
    pub severity: AlertSeverity,
    pub component: String,
    pub message: String,
    pub timestamp: DateTime<Utc>,
    pub acknowledged: bool,
}

/// Alert severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
    Emergency,
}

/// Resource utilization tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUtilization {
    pub cpu_percent: f64,
    pub memory_percent: f64,
    pub disk_percent: f64,
    pub network_mbps: f64,
    pub open_file_descriptors: u64,
    pub thread_count: u64,
}

/// Performance indicators
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceIndicators {
    pub total_operations_per_second: f64,
    pub average_response_time_ms: f64,
    pub error_rate_percent: f64,
    pub uptime_seconds: u64,
    pub emergency_response_time_ns: u64,
}

/// Emergency status tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmergencyStatus {
    pub is_active: bool,
    pub emergency_level: EmergencyLevel,
    pub triggered_by: Option<String>,
    pub trigger_time: Option<DateTime<Utc>>,
    pub active_protocols: Vec<String>,
    pub recovery_status: RecoveryStatus,
}

/// Emergency severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EmergencyLevel {
    None,
    Low,
    Medium,
    High,
    Critical,
}

/// Recovery status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecoveryStatus {
    NotNeeded,
    InProgress,
    Completed,
    Failed { reason: String },
}

/// Oversight metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OversightMetrics {
    pub total_operations: u64,
    pub approved_operations: u64,
    pub rejected_operations: u64,
    pub emergency_operations: u64,
    pub average_validation_time_ms: f64,
    pub consensus_accuracy: f64,
    pub false_positive_rate: f64,
    pub false_negative_rate: f64,
}

/// Coordination event logging
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordinationEvent {
    pub event_id: Uuid,
    pub timestamp: DateTime<Utc>,
    pub event_type: CoordinationEventType,
    pub description: String,
    pub participants: Vec<String>,
    pub outcome: EventOutcome,
}

/// Coordination event types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CoordinationEventType {
    OperationValidation,
    EmergencyResponse,
    WatchdogCoordination,
    SystemHealthCheck,
    ConfigurationUpdate,
    RecoveryAction,
}

/// Event outcome
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EventOutcome {
    Success,
    Failure { reason: String },
    Partial { details: String },
}

/// Decision matrix for consensus building
#[derive(Debug, Clone)]
pub struct DecisionMatrix {
    pub operation_id: Uuid,
    pub watchdog_votes: HashMap<String, DecisionVote>,
    pub weighted_score: f64,
    pub confidence_level: f64,
    pub consensus_reached: bool,
    pub final_decision: Option<TENGRIOversightResult>,
}

/// Individual watchdog vote
#[derive(Debug, Clone)]
pub struct DecisionVote {
    pub watchdog_name: String,
    pub vote: VoteType,
    pub confidence: f64,
    pub weight: f64,
    pub justification: String,
    pub processing_time_ms: u64,
}

/// Vote types
#[derive(Debug, Clone)]
pub enum VoteType {
    Approve,
    Warn,
    Reject,
    Emergency,
}

impl OversightState {
    /// Create new oversight state
    pub fn new() -> Self {
        Self {
            session_id: Uuid::new_v4(),
            start_time: Utc::now(),
            active_operations: HashMap::new(),
            watchdog_states: WatchdogStates::new(),
            system_health: SystemHealth::new(),
            emergency_status: EmergencyStatus::new(),
            oversight_metrics: OversightMetrics::new(),
            coordination_log: Vec::new(),
        }
    }

    /// Add active operation
    pub fn add_active_operation(&mut self, operation: TradingOperation) {
        let active_op = ActiveOperation {
            operation_id: operation.id,
            operation: operation.clone(),
            start_time: Utc::now(),
            watchdog_results: HashMap::new(),
            overall_status: OperationStatus::Pending,
            risk_score: 0.0,
            validation_progress: ValidationProgress {
                total_watchdogs: 6, // Total number of watchdogs
                completed_watchdogs: 0,
                failed_watchdogs: 0,
                progress_percentage: 0.0,
                estimated_completion: None,
            },
        };
        self.active_operations.insert(operation.id, active_op);
    }

    /// Update watchdog result
    pub fn update_watchdog_result(&mut self, operation_id: Uuid, watchdog_name: String, result: WatchdogResult) {
        if let Some(active_op) = self.active_operations.get_mut(&operation_id) {
            active_op.watchdog_results.insert(watchdog_name, result);
            active_op.validation_progress.completed_watchdogs += 1;
            active_op.validation_progress.progress_percentage = 
                (active_op.validation_progress.completed_watchdogs as f64 / active_op.validation_progress.total_watchdogs as f64) * 100.0;
        }
    }

    /// Check if operation is complete
    pub fn is_operation_complete(&self, operation_id: &Uuid) -> bool {
        if let Some(active_op) = self.active_operations.get(operation_id) {
            active_op.validation_progress.completed_watchdogs >= active_op.validation_progress.total_watchdogs
        } else {
            false
        }
    }

    /// Update system health
    pub fn update_system_health(&mut self, health: SystemHealth) {
        self.system_health = health;
    }

    /// Trigger emergency
    pub fn trigger_emergency(&mut self, level: EmergencyLevel, trigger: String) {
        self.emergency_status.is_active = true;
        self.emergency_status.emergency_level = level;
        self.emergency_status.triggered_by = Some(trigger);
        self.emergency_status.trigger_time = Some(Utc::now());
        self.emergency_status.recovery_status = RecoveryStatus::NotNeeded;
    }

    /// Log coordination event
    pub fn log_coordination_event(&mut self, event: CoordinationEvent) {
        self.coordination_log.push(event);
        
        // Keep only last 1000 events
        if self.coordination_log.len() > 1000 {
            self.coordination_log.remove(0);
        }
    }
}

impl WatchdogStates {
    pub fn new() -> Self {
        Self {
            data_integrity: WatchdogState::new(),
            scientific_rigor: WatchdogState::new(),
            production_readiness: WatchdogState::new(),
            synthetic_detection: WatchdogState::new(),
            mathematical_validation: WatchdogState::new(),
            emergency_protocols: WatchdogState::new(),
        }
    }
}

impl WatchdogState {
    pub fn new() -> Self {
        Self {
            status: WatchdogStatus::Active,
            last_update: Utc::now(),
            operations_processed: 0,
            average_processing_time_ms: 0.0,
            success_rate: 100.0,
            error_count: 0,
            performance_metrics: WatchdogPerformanceMetrics::new(),
        }
    }
}

impl WatchdogPerformanceMetrics {
    pub fn new() -> Self {
        Self {
            response_time_p50: 10.0,
            response_time_p95: 50.0,
            response_time_p99: 100.0,
            throughput_ops_per_second: 100.0,
            memory_usage_mb: 256.0,
            cpu_usage_percent: 25.0,
        }
    }
}

impl SystemHealth {
    pub fn new() -> Self {
        Self {
            overall_status: SystemStatus::Healthy,
            health_score: 100.0,
            active_alerts: Vec::new(),
            resource_utilization: ResourceUtilization::new(),
            performance_indicators: PerformanceIndicators::new(),
        }
    }
}

impl ResourceUtilization {
    pub fn new() -> Self {
        Self {
            cpu_percent: 25.0,
            memory_percent: 50.0,
            disk_percent: 30.0,
            network_mbps: 10.0,
            open_file_descriptors: 1024,
            thread_count: 32,
        }
    }
}

impl PerformanceIndicators {
    pub fn new() -> Self {
        Self {
            total_operations_per_second: 100.0,
            average_response_time_ms: 50.0,
            error_rate_percent: 0.1,
            uptime_seconds: 0,
            emergency_response_time_ns: 50,
        }
    }
}

impl EmergencyStatus {
    pub fn new() -> Self {
        Self {
            is_active: false,
            emergency_level: EmergencyLevel::None,
            triggered_by: None,
            trigger_time: None,
            active_protocols: Vec::new(),
            recovery_status: RecoveryStatus::NotNeeded,
        }
    }
}

impl OversightMetrics {
    pub fn new() -> Self {
        Self {
            total_operations: 0,
            approved_operations: 0,
            rejected_operations: 0,
            emergency_operations: 0,
            average_validation_time_ms: 0.0,
            consensus_accuracy: 100.0,
            false_positive_rate: 0.0,
            false_negative_rate: 0.0,
        }
    }
}

/// Unified oversight coordinator
pub struct UnifiedOversightCoordinator {
    state: Arc<RwLock<OversightState>>,
    decision_matrices: Arc<RwLock<HashMap<Uuid, DecisionMatrix>>>,
    coordination_rules: CoordinationRules,
    emergency_thresholds: EmergencyThresholds,
}

/// Coordination rules for decision making
#[derive(Debug, Clone)]
pub struct CoordinationRules {
    pub consensus_threshold: f64,
    pub minimum_votes: usize,
    pub emergency_override_enabled: bool,
    pub weighted_voting: bool,
    pub watchdog_weights: HashMap<String, f64>,
}

/// Emergency thresholds
#[derive(Debug, Clone)]
pub struct EmergencyThresholds {
    pub health_score_critical: f64,
    pub error_rate_critical: f64,
    pub response_time_critical: f64,
    pub memory_usage_critical: f64,
    pub cpu_usage_critical: f64,
}

impl UnifiedOversightCoordinator {
    /// Create new unified oversight coordinator
    pub fn new() -> Self {
        let coordination_rules = CoordinationRules {
            consensus_threshold: 0.8,
            minimum_votes: 3,
            emergency_override_enabled: true,
            weighted_voting: true,
            watchdog_weights: {
                let mut weights = HashMap::new();
                weights.insert("data_integrity".to_string(), 1.0);
                weights.insert("scientific_rigor".to_string(), 1.5);
                weights.insert("production_readiness".to_string(), 1.2);
                weights.insert("synthetic_detection".to_string(), 2.0);
                weights.insert("mathematical_validation".to_string(), 1.5);
                weights.insert("emergency_protocols".to_string(), 2.0);
                weights
            },
        };

        let emergency_thresholds = EmergencyThresholds {
            health_score_critical: 30.0,
            error_rate_critical: 10.0,
            response_time_critical: 1000.0,
            memory_usage_critical: 90.0,
            cpu_usage_critical: 95.0,
        };

        Self {
            state: Arc::new(RwLock::new(OversightState::new())),
            decision_matrices: Arc::new(RwLock::new(HashMap::new())),
            coordination_rules,
            emergency_thresholds,
        }
    }

    /// Coordinate watchdog decisions
    pub async fn coordinate_decision(&self, operation_id: Uuid, watchdog_name: String, result: TENGRIOversightResult, processing_time_ms: u64) -> Result<Option<TENGRIOversightResult>, TENGRIError> {
        let mut matrices = self.decision_matrices.write().await;
        
        // Get or create decision matrix
        let matrix = matrices.entry(operation_id).or_insert_with(|| DecisionMatrix {
            operation_id,
            watchdog_votes: HashMap::new(),
            weighted_score: 0.0,
            confidence_level: 0.0,
            consensus_reached: false,
            final_decision: None,
        });

        // Convert result to vote
        let vote = self.convert_result_to_vote(&result);
        let confidence = self.calculate_confidence(&result);
        let weight = self.coordination_rules.watchdog_weights.get(&watchdog_name).unwrap_or(&1.0).clone();

        // Add vote to matrix
        matrix.watchdog_votes.insert(watchdog_name.clone(), DecisionVote {
            watchdog_name: watchdog_name.clone(),
            vote,
            confidence,
            weight,
            justification: format!("{:?}", result),
            processing_time_ms,
        });

        // Check if consensus reached
        if matrix.watchdog_votes.len() >= self.coordination_rules.minimum_votes {
            if let Some(final_decision) = self.calculate_consensus(matrix).await? {
                matrix.final_decision = Some(final_decision.clone());
                matrix.consensus_reached = true;
                
                // Log coordination event
                self.log_coordination_event(CoordinationEvent {
                    event_id: Uuid::new_v4(),
                    timestamp: Utc::now(),
                    event_type: CoordinationEventType::OperationValidation,
                    description: format!("Consensus reached for operation {}", operation_id),
                    participants: matrix.watchdog_votes.keys().cloned().collect(),
                    outcome: EventOutcome::Success,
                }).await;

                return Ok(Some(final_decision));
            }
        }

        // Update state
        let watchdog_result = WatchdogResult {
            watchdog_name: watchdog_name.clone(),
            result: format!("{:?}", result),
            timestamp: Utc::now(),
            processing_time_ms,
            confidence_score: confidence,
        };

        let mut state = self.state.write().await;
        state.update_watchdog_result(operation_id, watchdog_name, watchdog_result);

        Ok(None)
    }

    /// Convert oversight result to vote
    fn convert_result_to_vote(&self, result: &TENGRIOversightResult) -> VoteType {
        match result {
            TENGRIOversightResult::Approved => VoteType::Approve,
            TENGRIOversightResult::Warning { .. } => VoteType::Warn,
            TENGRIOversightResult::Rejected { .. } => VoteType::Reject,
            TENGRIOversightResult::CriticalViolation { .. } => VoteType::Emergency,
        }
    }

    /// Calculate confidence from result
    fn calculate_confidence(&self, result: &TENGRIOversightResult) -> f64 {
        match result {
            TENGRIOversightResult::Approved => 0.95,
            TENGRIOversightResult::Warning { .. } => 0.7,
            TENGRIOversightResult::Rejected { .. } => 0.85,
            TENGRIOversightResult::CriticalViolation { .. } => 0.99,
        }
    }

    /// Calculate consensus from decision matrix
    async fn calculate_consensus(&self, matrix: &DecisionMatrix) -> Result<Option<TENGRIOversightResult>, TENGRIError> {
        // Check for emergency votes - these override everything
        let emergency_votes: Vec<_> = matrix.watchdog_votes.values().filter(|v| matches!(v.vote, VoteType::Emergency)).collect();
        if !emergency_votes.is_empty() {
            return Ok(Some(TENGRIOversightResult::CriticalViolation {
                violation_type: ViolationType::SecurityViolation,
                immediate_shutdown: true,
                forensic_data: format!("Emergency consensus: {} votes", emergency_votes.len()).into_bytes(),
            }));
        }

        // Weighted voting
        let mut weighted_scores = HashMap::new();
        let mut total_weight = 0.0;

        for vote in matrix.watchdog_votes.values() {
            let score = match vote.vote {
                VoteType::Approve => 1.0,
                VoteType::Warn => 0.5,
                VoteType::Reject => 0.0,
                VoteType::Emergency => -1.0,
            };
            
            let weighted_score = score * vote.weight * vote.confidence;
            weighted_scores.insert(vote.watchdog_name.clone(), weighted_score);
            total_weight += vote.weight;
        }

        let average_score = weighted_scores.values().sum::<f64>() / total_weight;

        // Determine consensus
        if average_score >= self.coordination_rules.consensus_threshold {
            Ok(Some(TENGRIOversightResult::Approved))
        } else if average_score >= 0.3 {
            // Aggregate warnings
            let warnings: Vec<String> = matrix.watchdog_votes.values()
                .filter(|v| matches!(v.vote, VoteType::Warn))
                .map(|v| v.justification.clone())
                .collect();
            
            Ok(Some(TENGRIOversightResult::Warning {
                reason: format!("Consensus warning: {}", warnings.join("; ")),
                corrective_action: "Review flagged concerns".to_string(),
            }))
        } else {
            // Aggregate rejections
            let rejections: Vec<String> = matrix.watchdog_votes.values()
                .filter(|v| matches!(v.vote, VoteType::Reject))
                .map(|v| v.justification.clone())
                .collect();
            
            Ok(Some(TENGRIOversightResult::Rejected {
                reason: format!("Consensus rejection: {}", rejections.join("; ")),
                emergency_action: EmergencyAction::RollbackToSafeState,
            }))
        }
    }

    /// Monitor system health
    pub async fn monitor_system_health(&self) -> Result<SystemHealth, TENGRIError> {
        let mut state = self.state.write().await;
        
        // Collect health metrics from all watchdogs
        let mut health_score = 100.0;
        let mut active_alerts = Vec::new();
        
        // Check watchdog states
        for (name, watchdog_state) in vec![
            ("data_integrity", &state.watchdog_states.data_integrity),
            ("scientific_rigor", &state.watchdog_states.scientific_rigor),
            ("production_readiness", &state.watchdog_states.production_readiness),
            ("synthetic_detection", &state.watchdog_states.synthetic_detection),
            ("mathematical_validation", &state.watchdog_states.mathematical_validation),
            ("emergency_protocols", &state.watchdog_states.emergency_protocols),
        ] {
            match watchdog_state.status {
                WatchdogStatus::Active => {
                    if watchdog_state.success_rate < 95.0 {
                        health_score -= 5.0;
                        active_alerts.push(HealthAlert {
                            alert_id: Uuid::new_v4(),
                            severity: AlertSeverity::Warning,
                            component: name.to_string(),
                            message: format!("Success rate below 95%: {:.1}%", watchdog_state.success_rate),
                            timestamp: Utc::now(),
                            acknowledged: false,
                        });
                    }
                },
                WatchdogStatus::Degraded { ref reason } => {
                    health_score -= 15.0;
                    active_alerts.push(HealthAlert {
                        alert_id: Uuid::new_v4(),
                        severity: AlertSeverity::Warning,
                        component: name.to_string(),
                        message: format!("Degraded: {}", reason),
                        timestamp: Utc::now(),
                        acknowledged: false,
                    });
                },
                WatchdogStatus::Failed { ref error } => {
                    health_score -= 30.0;
                    active_alerts.push(HealthAlert {
                        alert_id: Uuid::new_v4(),
                        severity: AlertSeverity::Critical,
                        component: name.to_string(),
                        message: format!("Failed: {}", error),
                        timestamp: Utc::now(),
                        acknowledged: false,
                    });
                },
                _ => {}
            }
        }

        // Check emergency thresholds
        if health_score <= self.emergency_thresholds.health_score_critical {
            state.trigger_emergency(EmergencyLevel::Critical, "System health critical".to_string());
        }

        // Update system health
        let system_health = SystemHealth {
            overall_status: if health_score > 80.0 {
                SystemStatus::Healthy
            } else if health_score > 50.0 {
                SystemStatus::Degraded
            } else {
                SystemStatus::Critical
            },
            health_score,
            active_alerts,
            resource_utilization: ResourceUtilization::new(),
            performance_indicators: PerformanceIndicators::new(),
        };

        state.update_system_health(system_health.clone());
        
        Ok(system_health)
    }

    /// Emergency response coordination
    pub async fn trigger_emergency_response(&self, reason: &str) -> Result<(), TENGRIError> {
        let emergency_start = Instant::now();
        
        let mut state = self.state.write().await;
        state.trigger_emergency(EmergencyLevel::Critical, reason.to_string());
        
        // Log emergency event
        let event = CoordinationEvent {
            event_id: Uuid::new_v4(),
            timestamp: Utc::now(),
            event_type: CoordinationEventType::EmergencyResponse,
            description: format!("Emergency response triggered: {}", reason),
            participants: vec!["unified_oversight".to_string()],
            outcome: EventOutcome::Success,
        };
        
        state.log_coordination_event(event);
        
        // Verify emergency response time
        let elapsed = emergency_start.elapsed();
        if elapsed.as_nanos() > 100 {
            return Err(TENGRIError::EmergencyProtocolTriggered {
                reason: format!("Emergency response exceeded 100ns: {:?}", elapsed),
            });
        }

        Ok(())
    }

    /// Log coordination event
    async fn log_coordination_event(&self, event: CoordinationEvent) {
        let mut state = self.state.write().await;
        state.log_coordination_event(event);
    }

    /// Get current oversight state
    pub async fn get_oversight_state(&self) -> OversightState {
        let state = self.state.read().await;
        state.clone()
    }

    /// Get decision matrix for operation
    pub async fn get_decision_matrix(&self, operation_id: &Uuid) -> Option<DecisionMatrix> {
        let matrices = self.decision_matrices.read().await;
        matrices.get(operation_id).cloned()
    }

    /// Clean up completed operations
    pub async fn cleanup_completed_operations(&self, max_age: Duration) -> Result<usize, TENGRIError> {
        let mut state = self.state.write().await;
        let mut matrices = self.decision_matrices.write().await;
        
        let cutoff_time = Utc::now() - chrono::Duration::from_std(max_age).unwrap();
        let mut cleaned_count = 0;
        
        // Remove old active operations
        state.active_operations.retain(|_id, operation| {
            if operation.start_time < cutoff_time && matches!(operation.overall_status, OperationStatus::Completed) {
                cleaned_count += 1;
                false
            } else {
                true
            }
        });
        
        // Remove old decision matrices
        matrices.retain(|id, matrix| {
            if !state.active_operations.contains_key(id) {
                false
            } else {
                true
            }
        });
        
        Ok(cleaned_count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio_test;

    #[tokio::test]
    async fn test_oversight_state_management() {
        let mut state = OversightState::new();
        
        let operation = TradingOperation {
            id: Uuid::new_v4(),
            timestamp: Utc::now(),
            operation_type: crate::OperationType::PlaceOrder,
            data_source: "test_source".to_string(),
            mathematical_model: "test_model".to_string(),
            risk_parameters: crate::RiskParameters {
                max_position_size: 1000.0,
                stop_loss: Some(0.02),
                take_profit: Some(0.05),
                confidence_threshold: 0.95,
            },
            agent_id: "test_agent".to_string(),
        };
        
        state.add_active_operation(operation.clone());
        assert!(state.active_operations.contains_key(&operation.id));
        
        let result = WatchdogResult {
            watchdog_name: "test_watchdog".to_string(),
            result: "Approved".to_string(),
            timestamp: Utc::now(),
            processing_time_ms: 10,
            confidence_score: 0.95,
        };
        
        state.update_watchdog_result(operation.id, "test_watchdog".to_string(), result);
        assert_eq!(state.active_operations[&operation.id].validation_progress.completed_watchdogs, 1);
    }

    #[tokio::test]
    async fn test_unified_oversight_coordinator() {
        let coordinator = UnifiedOversightCoordinator::new();
        
        let operation_id = Uuid::new_v4();
        let result = TENGRIOversightResult::Approved;
        
        let decision = coordinator.coordinate_decision(
            operation_id,
            "test_watchdog".to_string(),
            result,
            10
        ).await.unwrap();
        
        // Should not have consensus yet (need minimum votes)
        assert!(decision.is_none());
        
        // Add more votes
        for i in 0..3 {
            let watchdog_name = format!("watchdog_{}", i);
            coordinator.coordinate_decision(
                operation_id,
                watchdog_name,
                TENGRIOversightResult::Approved,
                10
            ).await.unwrap();
        }
        
        // Should have consensus now
        let final_decision = coordinator.coordinate_decision(
            operation_id,
            "final_watchdog".to_string(),
            TENGRIOversightResult::Approved,
            10
        ).await.unwrap();
        
        assert!(final_decision.is_some());
        assert!(matches!(final_decision.unwrap(), TENGRIOversightResult::Approved));
    }

    #[tokio::test]
    async fn test_emergency_response() {
        let coordinator = UnifiedOversightCoordinator::new();
        
        let result = coordinator.trigger_emergency_response("Test emergency").await;
        assert!(result.is_ok());
        
        let state = coordinator.get_oversight_state().await;
        assert!(state.emergency_status.is_active);
        assert!(matches!(state.emergency_status.emergency_level, EmergencyLevel::Critical));
    }

    #[tokio::test]
    async fn test_system_health_monitoring() {
        let coordinator = UnifiedOversightCoordinator::new();
        
        let health = coordinator.monitor_system_health().await.unwrap();
        assert!(matches!(health.overall_status, SystemStatus::Healthy));
        assert!(health.health_score > 0.0);
    }

    #[tokio::test]
    async fn test_coordination_rules() {
        let rules = CoordinationRules {
            consensus_threshold: 0.8,
            minimum_votes: 3,
            emergency_override_enabled: true,
            weighted_voting: true,
            watchdog_weights: HashMap::new(),
        };
        
        assert_eq!(rules.consensus_threshold, 0.8);
        assert_eq!(rules.minimum_votes, 3);
        assert!(rules.emergency_override_enabled);
        assert!(rules.weighted_voting);
    }
}