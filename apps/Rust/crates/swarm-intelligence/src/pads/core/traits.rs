//! # PADS Core Traits
//!
//! Core trait definitions for the Panarchy Adaptive Decision System.

use std::collections::HashMap;
use std::future::Future;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::core::{
    PadsResult, DecisionLayer, AdaptiveCyclePhase, DecisionContext,
    SystemHealth
};
use super::types::{
    DecisionAlternative, DecisionCriteria, PerformanceMetrics,
    SystemState, SystemEvent
};

/// Core system capabilities trait
#[async_trait]
pub trait PadsCore: Send + Sync {
    /// Initialize the system component
    async fn initialize(&mut self) -> PadsResult<()>;
    
    /// Start the system component
    async fn start(&mut self) -> PadsResult<()>;
    
    /// Stop the system component gracefully
    async fn stop(&mut self) -> PadsResult<()>;
    
    /// Check if the component is healthy
    async fn is_healthy(&self) -> bool;
    
    /// Get component metrics
    async fn get_metrics(&self) -> PerformanceMetrics;
    
    /// Handle system events
    async fn handle_event(&mut self, event: SystemEvent) -> PadsResult<()>;
}

/// Adaptive behavior trait for learning and evolution
#[async_trait]
pub trait Adaptive: Send + Sync {
    /// Learn from a decision outcome
    async fn learn_from_outcome(
        &mut self,
        context: &DecisionContext,
        outcome: &DecisionOutcome,
    ) -> PadsResult<()>;
    
    /// Adapt behavior based on performance metrics
    async fn adapt_behavior(&mut self, metrics: &PerformanceMetrics) -> PadsResult<()>;
    
    /// Get current adaptation state
    async fn get_adaptation_state(&self) -> AdaptationState;
    
    /// Reset adaptation to initial state
    async fn reset_adaptation(&mut self) -> PadsResult<()>;
}

/// Decision outcome for learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionOutcome {
    /// Decision context
    pub context: DecisionContext,
    
    /// Chosen action
    pub action: String,
    
    /// Actual performance achieved
    pub performance: PerformanceMetrics,
    
    /// Success score (0.0 to 1.0)
    pub success_score: f64,
    
    /// Lessons learned
    pub lessons: Vec<String>,
    
    /// Improvement suggestions
    pub improvements: Vec<String>,
}

/// Adaptation state information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptationState {
    /// Learning rate
    pub learning_rate: f64,
    
    /// Exploration rate
    pub exploration_rate: f64,
    
    /// Adaptation count
    pub adaptation_count: u64,
    
    /// Performance trend
    pub performance_trend: f64,
    
    /// Confidence level
    pub confidence: f64,
}

/// Decision making capability trait
#[async_trait]
pub trait DecisionMaker: Send + Sync {
    /// Generate decision alternatives
    async fn generate_alternatives(
        &self,
        context: &DecisionContext,
    ) -> PadsResult<Vec<DecisionAlternative>>;
    
    /// Evaluate alternatives against criteria
    async fn evaluate_alternatives(
        &self,
        alternatives: &[DecisionAlternative],
        criteria: &[DecisionCriteria],
    ) -> PadsResult<HashMap<String, f64>>;
    
    /// Select the best alternative
    async fn select_alternative(
        &self,
        evaluations: &HashMap<String, f64>,
        context: &DecisionContext,
    ) -> PadsResult<String>;
    
    /// Validate a decision
    async fn validate_decision(
        &self,
        alternative: &DecisionAlternative,
        context: &DecisionContext,
    ) -> PadsResult<bool>;
}

/// Panarchy modeling trait
#[async_trait]
pub trait PanarchyModel: Send + Sync {
    /// Get current adaptive cycle phase
    async fn get_current_phase(&self) -> AdaptiveCyclePhase;
    
    /// Check if phase transition is needed
    async fn should_transition(&self) -> PadsResult<bool>;
    
    /// Transition to the next phase
    async fn transition_phase(&mut self) -> PadsResult<AdaptiveCyclePhase>;
    
    /// Calculate phase characteristics
    async fn calculate_phase_characteristics(&self) -> PhaseMetrics;
    
    /// Assess system resilience
    async fn assess_resilience(&self) -> PadsResult<f64>;
    
    /// Detect emergent behaviors
    async fn detect_emergence(&self) -> PadsResult<Vec<EmergentPattern>>;
}

/// Phase metrics for panarchy analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseMetrics {
    /// Potential for change
    pub potential: f64,
    
    /// System connectedness
    pub connectedness: f64,
    
    /// System resilience
    pub resilience: f64,
    
    /// Innovation capacity
    pub innovation: f64,
    
    /// Efficiency level
    pub efficiency: f64,
    
    /// Phase stability
    pub stability: f64,
}

/// Emergent pattern detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmergentPattern {
    /// Pattern identifier
    pub id: String,
    
    /// Pattern type
    pub pattern_type: String,
    
    /// Emergence strength (0.0 to 1.0)
    pub strength: f64,
    
    /// Pattern description
    pub description: String,
    
    /// Contributing factors
    pub factors: Vec<String>,
    
    /// Predicted impact
    pub impact: f64,
}

/// Coordination capability trait
#[async_trait]
pub trait Coordinator: Send + Sync {
    /// Coordinate between system components
    async fn coordinate(&mut self, request: CoordinationRequest) -> PadsResult<CoordinationResponse>;
    
    /// Synchronize component states
    async fn synchronize(&mut self, components: &[String]) -> PadsResult<()>;
    
    /// Resolve conflicts between components
    async fn resolve_conflict(&mut self, conflict: ConflictDescription) -> PadsResult<Resolution>;
    
    /// Optimize resource allocation
    async fn optimize_resources(&mut self, resources: &ResourceConstraints) -> PadsResult<ResourceAllocation>;
}

/// Coordination request structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordinationRequest {
    /// Request identifier
    pub id: String,
    
    /// Requesting component
    pub requestor: String,
    
    /// Target components
    pub targets: Vec<String>,
    
    /// Coordination type
    pub coordination_type: CoordinationType,
    
    /// Request data
    pub data: HashMap<String, String>,
    
    /// Priority level
    pub priority: u8,
}

/// Types of coordination
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CoordinationType {
    /// Information sharing
    InformationSharing,
    
    /// Resource allocation
    ResourceAllocation,
    
    /// Task assignment
    TaskAssignment,
    
    /// Conflict resolution
    ConflictResolution,
    
    /// Performance optimization
    PerformanceOptimization,
}

/// Coordination response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordinationResponse {
    /// Response identifier
    pub id: String,
    
    /// Request identifier this responds to
    pub request_id: String,
    
    /// Success indicator
    pub success: bool,
    
    /// Response data
    pub data: HashMap<String, String>,
    
    /// Additional actions required
    pub actions: Vec<String>,
}

/// Conflict description
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConflictDescription {
    /// Conflict identifier
    pub id: String,
    
    /// Conflicting components
    pub components: Vec<String>,
    
    /// Conflict type
    pub conflict_type: String,
    
    /// Description of the conflict
    pub description: String,
    
    /// Impact assessment
    pub impact: f64,
}

/// Conflict resolution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Resolution {
    /// Resolution identifier
    pub id: String,
    
    /// Resolution strategy
    pub strategy: String,
    
    /// Actions to take
    pub actions: Vec<String>,
    
    /// Expected outcome
    pub expected_outcome: String,
    
    /// Success probability
    pub success_probability: f64,
}

/// Resource constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceConstraints {
    /// Available CPU capacity
    pub cpu_capacity: f64,
    
    /// Available memory
    pub memory_capacity: f64,
    
    /// Network bandwidth
    pub network_bandwidth: f64,
    
    /// Storage capacity
    pub storage_capacity: f64,
    
    /// Custom resource constraints
    pub custom_constraints: HashMap<String, f64>,
}

/// Resource allocation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceAllocation {
    /// Component allocations
    pub allocations: HashMap<String, HashMap<String, f64>>,
    
    /// Utilization efficiency
    pub efficiency: f64,
    
    /// Allocation confidence
    pub confidence: f64,
    
    /// Reallocation recommendations
    pub recommendations: Vec<String>,
}

/// Monitoring capability trait
#[async_trait]
pub trait Monitor: Send + Sync {
    /// Collect system metrics
    async fn collect_metrics(&self) -> PerformanceMetrics;
    
    /// Assess system health
    async fn assess_health(&self) -> SystemHealth;
    
    /// Detect anomalies
    async fn detect_anomalies(&self) -> PadsResult<Vec<Anomaly>>;
    
    /// Generate performance report
    async fn generate_report(&self, timeframe: std::time::Duration) -> PerformanceReport;
    
    /// Set up alerting rules
    async fn setup_alerts(&mut self, rules: Vec<AlertRule>) -> PadsResult<()>;
}

/// Anomaly detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Anomaly {
    /// Anomaly identifier
    pub id: String,
    
    /// Anomaly type
    pub anomaly_type: String,
    
    /// Severity level (0.0 to 1.0)
    pub severity: f64,
    
    /// Description
    pub description: String,
    
    /// Affected metrics
    pub metrics: Vec<String>,
    
    /// Detection confidence
    pub confidence: f64,
    
    /// Recommended actions
    pub recommendations: Vec<String>,
}

/// Performance report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceReport {
    /// Report identifier
    pub id: String,
    
    /// Reporting timeframe
    pub timeframe: std::time::Duration,
    
    /// Key performance indicators
    pub kpis: HashMap<String, f64>,
    
    /// Trend analysis
    pub trends: HashMap<String, TrendAnalysis>,
    
    /// Recommendations
    pub recommendations: Vec<String>,
    
    /// Overall health score
    pub health_score: f64,
}

/// Trend analysis data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendAnalysis {
    /// Trend direction
    pub direction: TrendDirection,
    
    /// Trend strength (0.0 to 1.0)
    pub strength: f64,
    
    /// Confidence in trend (0.0 to 1.0)
    pub confidence: f64,
    
    /// Predicted future values
    pub predictions: Vec<f64>,
}

/// Trend direction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrendDirection {
    /// Increasing trend
    Increasing,
    
    /// Decreasing trend
    Decreasing,
    
    /// Stable trend
    Stable,
    
    /// Volatile/unpredictable trend
    Volatile,
}

/// Alert rule configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertRule {
    /// Rule identifier
    pub id: String,
    
    /// Metric to monitor
    pub metric: String,
    
    /// Threshold value
    pub threshold: f64,
    
    /// Comparison operator
    pub operator: ComparisonOperator,
    
    /// Alert severity
    pub severity: AlertSeverity,
    
    /// Alert message template
    pub message: String,
}

/// Comparison operators for alerts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComparisonOperator {
    /// Greater than
    GreaterThan,
    
    /// Less than
    LessThan,
    
    /// Equal to
    EqualTo,
    
    /// Greater than or equal
    GreaterThanOrEqual,
    
    /// Less than or equal
    LessThanOrEqual,
}

/// Alert severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum AlertSeverity {
    /// Low priority alert
    Low,
    
    /// Medium priority alert
    Medium,
    
    /// High priority alert
    High,
    
    /// Critical priority alert
    Critical,
}

/// Governance capability trait
#[async_trait]
pub trait Governance: Send + Sync {
    /// Validate an action against policies
    async fn validate_action(&self, action: &ActionRequest) -> PadsResult<ValidationResult>;
    
    /// Apply governance policies
    async fn apply_policies(&mut self, context: &GovernanceContext) -> PadsResult<()>;
    
    /// Audit system actions
    async fn audit_action(&mut self, action: &CompletedAction) -> PadsResult<()>;
    
    /// Generate compliance report
    async fn generate_compliance_report(&self) -> ComplianceReport;
    
    /// Update policies
    async fn update_policies(&mut self, policies: Vec<Policy>) -> PadsResult<()>;
}

/// Action request for validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionRequest {
    /// Action identifier
    pub id: String,
    
    /// Action type
    pub action_type: String,
    
    /// Requesting component
    pub requestor: String,
    
    /// Action parameters
    pub parameters: HashMap<String, String>,
    
    /// Risk level
    pub risk_level: f64,
    
    /// Impact assessment
    pub impact: f64,
}

/// Validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    /// Validation passed
    pub approved: bool,
    
    /// Validation reasons
    pub reasons: Vec<String>,
    
    /// Required modifications
    pub modifications: Vec<String>,
    
    /// Risk assessment
    pub risk_assessment: f64,
    
    /// Compliance score
    pub compliance_score: f64,
}

/// Governance context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GovernanceContext {
    /// Current system state
    pub system_state: SystemState,
    
    /// Active policies
    pub active_policies: Vec<String>,
    
    /// Compliance requirements
    pub compliance_requirements: Vec<String>,
    
    /// Risk tolerance
    pub risk_tolerance: f64,
}

/// Completed action for auditing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletedAction {
    /// Action identifier
    pub id: String,
    
    /// Original request
    pub request: ActionRequest,
    
    /// Execution result
    pub result: ActionResult,
    
    /// Execution time
    pub execution_time: std::time::Duration,
    
    /// Resource usage
    pub resource_usage: HashMap<String, f64>,
}

/// Action execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionResult {
    /// Success indicator
    pub success: bool,
    
    /// Result data
    pub data: HashMap<String, String>,
    
    /// Error messages (if any)
    pub errors: Vec<String>,
    
    /// Performance metrics
    pub metrics: PerformanceMetrics,
}

/// Compliance report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceReport {
    /// Report identifier
    pub id: String,
    
    /// Reporting period
    pub period: std::time::Duration,
    
    /// Overall compliance score
    pub compliance_score: f64,
    
    /// Policy adherence
    pub policy_adherence: HashMap<String, f64>,
    
    /// Violations found
    pub violations: Vec<ComplianceViolation>,
    
    /// Recommendations
    pub recommendations: Vec<String>,
}

/// Compliance violation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceViolation {
    /// Violation identifier
    pub id: String,
    
    /// Violation type
    pub violation_type: String,
    
    /// Severity level
    pub severity: AlertSeverity,
    
    /// Description
    pub description: String,
    
    /// Policy violated
    pub policy: String,
    
    /// Remediation actions
    pub remediation: Vec<String>,
}

/// Policy definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Policy {
    /// Policy identifier
    pub id: String,
    
    /// Policy name
    pub name: String,
    
    /// Policy description
    pub description: String,
    
    /// Policy rules
    pub rules: Vec<PolicyRule>,
    
    /// Enforcement level
    pub enforcement: EnforcementLevel,
    
    /// Policy category
    pub category: String,
}

/// Policy rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyRule {
    /// Rule identifier
    pub id: String,
    
    /// Rule condition
    pub condition: String,
    
    /// Rule action
    pub action: String,
    
    /// Rule priority
    pub priority: u8,
}

/// Policy enforcement levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EnforcementLevel {
    /// Advisory only
    Advisory,
    
    /// Warning on violation
    Warning,
    
    /// Block action on violation
    Blocking,
    
    /// Mandatory compliance
    Mandatory,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_adaptation_state() {
        let state = AdaptationState {
            learning_rate: 0.1,
            exploration_rate: 0.2,
            adaptation_count: 100,
            performance_trend: 0.05,
            confidence: 0.8,
        };
        
        assert_eq!(state.learning_rate, 0.1);
        assert_eq!(state.adaptation_count, 100);
    }
    
    #[test]
    fn test_phase_metrics() {
        let metrics = PhaseMetrics {
            potential: 0.8,
            connectedness: 0.6,
            resilience: 0.7,
            innovation: 0.9,
            efficiency: 0.5,
            stability: 0.6,
        };
        
        assert!(metrics.potential > 0.5);
        assert!(metrics.innovation > metrics.efficiency);
    }
    
    #[test]
    fn test_coordination_request() {
        let request = CoordinationRequest {
            id: "coord-001".to_string(),
            requestor: "component-a".to_string(),
            targets: vec!["component-b".to_string()],
            coordination_type: CoordinationType::ResourceAllocation,
            data: HashMap::new(),
            priority: 3,
        };
        
        assert_eq!(request.priority, 3);
        assert!(matches!(request.coordination_type, CoordinationType::ResourceAllocation));
    }
}