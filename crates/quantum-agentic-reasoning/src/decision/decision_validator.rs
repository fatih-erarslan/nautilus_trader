//! Decision Validation Module
//!
//! Comprehensive validation and verification of trading decisions with quantum consistency checks.

use crate::core::{QarResult, TradingDecision, DecisionType, FactorMap};
use crate::analysis::AnalysisResult;
use crate::error::QarError;
use super::{RiskAssessment, ExecutionPlan, QuantumInsights, EnhancedTradingDecision};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Decision validator for comprehensive validation
pub struct DecisionValidator {
    config: ValidationConfig,
    validation_rules: ValidationRuleSet,
    consistency_checker: ConsistencyChecker,
    compliance_monitor: ComplianceMonitor,
    validation_history: Vec<ValidationRecord>,
    performance_tracker: ValidationPerformanceTracker,
}

/// Validation configuration
#[derive(Debug, Clone)]
pub struct ValidationConfig {
    /// Enable strict validation mode
    pub strict_mode: bool,
    /// Maximum acceptable risk level
    pub max_risk_threshold: f64,
    /// Minimum confidence threshold
    pub min_confidence: f64,
    /// Enable quantum consistency checks
    pub quantum_consistency: bool,
    /// Enable regulatory compliance checks
    pub compliance_checks: bool,
    /// Validation timeout
    pub validation_timeout: std::time::Duration,
    /// Enable performance validation
    pub performance_validation: bool,
}

/// Comprehensive validation rule set
#[derive(Debug)]
pub struct ValidationRuleSet {
    /// Basic validation rules
    pub basic_rules: Vec<BasicValidationRule>,
    /// Risk validation rules
    pub risk_rules: Vec<RiskValidationRule>,
    /// Quantum validation rules
    pub quantum_rules: Vec<QuantumValidationRule>,
    /// Market condition rules
    pub market_rules: Vec<MarketValidationRule>,
    /// Compliance rules
    pub compliance_rules: Vec<ComplianceRule>,
    /// Custom validation rules
    pub custom_rules: Vec<CustomValidationRule>,
}

/// Basic validation rule
#[derive(Debug)]
pub struct BasicValidationRule {
    /// Rule identifier
    pub id: String,
    /// Rule name
    pub name: String,
    /// Rule description
    pub description: String,
    /// Validation function
    pub validator: ValidationFunction,
    /// Rule severity
    pub severity: ValidationSeverity,
    /// Rule enabled
    pub enabled: bool,
}

/// Risk validation rule
#[derive(Debug)]
pub struct RiskValidationRule {
    /// Rule identifier
    pub id: String,
    /// Risk metric to validate
    pub risk_metric: RiskMetric,
    /// Threshold value
    pub threshold: f64,
    /// Comparison type
    pub comparison: ComparisonType,
    /// Rule severity
    pub severity: ValidationSeverity,
}

/// Quantum validation rule
#[derive(Debug)]
pub struct QuantumValidationRule {
    /// Rule identifier
    pub id: String,
    /// Quantum property to validate
    pub quantum_property: QuantumProperty,
    /// Expected range
    pub expected_range: (f64, f64),
    /// Tolerance level
    pub tolerance: f64,
    /// Rule severity
    pub severity: ValidationSeverity,
}

/// Market validation rule
#[derive(Debug)]
pub struct MarketValidationRule {
    /// Rule identifier
    pub id: String,
    /// Market condition to check
    pub condition: MarketCondition,
    /// Decision compatibility
    pub compatible_decisions: Vec<DecisionType>,
    /// Rule severity
    pub severity: ValidationSeverity,
}

/// Compliance rule
#[derive(Debug)]
pub struct ComplianceRule {
    /// Rule identifier
    pub id: String,
    /// Regulatory framework
    pub framework: RegulatoryFramework,
    /// Rule description
    pub description: String,
    /// Compliance check function
    pub checker: ComplianceChecker,
    /// Mandatory flag
    pub mandatory: bool,
}

/// Custom validation rule
#[derive(Debug)]
pub struct CustomValidationRule {
    /// Rule identifier
    pub id: String,
    /// Rule name
    pub name: String,
    /// Custom validation logic
    pub logic: Box<dyn Fn(&TradingDecision, &FactorMap, &AnalysisResult) -> QarResult<ValidationResult> + Send + Sync>,
    /// Rule priority
    pub priority: ValidationPriority,
}

/// Validation function type
pub type ValidationFunction = fn(&TradingDecision, &FactorMap, &AnalysisResult) -> QarResult<ValidationResult>;

/// Compliance checker function type
pub type ComplianceChecker = fn(&TradingDecision, &RiskAssessment) -> QarResult<ComplianceResult>;

/// Validation severity levels
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValidationSeverity {
    Info,
    Warning,
    Error,
    Critical,
    Blocking,
}

/// Risk metrics for validation
#[derive(Debug, Clone)]
pub enum RiskMetric {
    VaR95,
    VaR99,
    ExpectedShortfall,
    MaxDrawdown,
    Volatility,
    LiquidityRisk,
    ConcentrationRisk,
    QuantumDecoherence,
}

/// Comparison types for validation
#[derive(Debug, Clone)]
pub enum ComparisonType {
    LessThan,
    LessEqual,
    GreaterThan,
    GreaterEqual,
    Equal,
    NotEqual,
    InRange,
    OutOfRange,
}

/// Quantum properties for validation
#[derive(Debug, Clone)]
pub enum QuantumProperty {
    Coherence,
    Entanglement,
    Superposition,
    Fidelity,
    Purity,
    Entropy,
    Uncertainty,
}

/// Market conditions for validation
#[derive(Debug, Clone)]
pub enum MarketCondition {
    HighVolatility,
    LowLiquidity,
    TrendingMarket,
    RangingMarket,
    CrisisConditions,
    NormalConditions,
}

/// Regulatory frameworks
#[derive(Debug, Clone)]
pub enum RegulatoryFramework {
    MiFID2,
    FINRA,
    SEC,
    FCA,
    ESMA,
    Custom(String),
}

/// Validation priority levels
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum ValidationPriority {
    Low,
    Medium,
    High,
    Critical,
}

/// Consistency checker for quantum and classical alignment
#[derive(Debug)]
pub struct ConsistencyChecker {
    /// Quantum-classical consistency threshold
    pub consistency_threshold: f64,
    /// Temporal consistency window
    pub temporal_window: std::time::Duration,
    /// Historical decisions for consistency check
    pub decision_history: Vec<TradingDecision>,
    /// Consistency metrics
    pub consistency_metrics: ConsistencyMetrics,
}

/// Consistency metrics
#[derive(Debug)]
pub struct ConsistencyMetrics {
    /// Quantum-classical alignment score
    pub quantum_classical_alignment: f64,
    /// Temporal consistency score
    pub temporal_consistency: f64,
    /// Risk consistency score
    pub risk_consistency: f64,
    /// Strategy consistency score
    pub strategy_consistency: f64,
}

/// Compliance monitor
#[derive(Debug)]
pub struct ComplianceMonitor {
    /// Active regulatory frameworks
    pub active_frameworks: Vec<RegulatoryFramework>,
    /// Compliance status
    pub compliance_status: HashMap<RegulatoryFramework, ComplianceStatus>,
    /// Violation history
    pub violation_history: Vec<ComplianceViolation>,
    /// Compliance metrics
    pub compliance_metrics: ComplianceMetrics,
}

/// Compliance status
#[derive(Debug, Clone)]
pub enum ComplianceStatus {
    Compliant,
    Warning,
    Violation,
    Exempted,
}

/// Compliance violation
#[derive(Debug, Clone)]
pub struct ComplianceViolation {
    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Framework violated
    pub framework: RegulatoryFramework,
    /// Violation description
    pub description: String,
    /// Severity level
    pub severity: ValidationSeverity,
    /// Resolution status
    pub resolved: bool,
}

/// Compliance metrics
#[derive(Debug)]
pub struct ComplianceMetrics {
    /// Overall compliance score
    pub overall_score: f64,
    /// Violations per framework
    pub violations_per_framework: HashMap<RegulatoryFramework, usize>,
    /// Time to resolution
    pub avg_resolution_time: std::time::Duration,
    /// Compliance trend
    pub compliance_trend: ComplianceTrend,
}

/// Compliance trend
#[derive(Debug)]
pub enum ComplianceTrend {
    Improving,
    Stable,
    Deteriorating,
}

/// Comprehensive validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    /// Overall validation status
    pub status: ValidationStatus,
    /// Individual rule results
    pub rule_results: Vec<RuleValidationResult>,
    /// Consistency check results
    pub consistency_results: ConsistencyResults,
    /// Compliance check results
    pub compliance_results: ComplianceResults,
    /// Validation summary
    pub summary: ValidationSummary,
    /// Recommended actions
    pub recommendations: Vec<ValidationRecommendation>,
}

/// Validation status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ValidationStatus {
    Passed,
    PassedWithWarnings,
    Failed,
    Blocked,
    RequiresReview,
}

/// Individual rule validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuleValidationResult {
    /// Rule identifier
    pub rule_id: String,
    /// Rule name
    pub rule_name: String,
    /// Validation outcome
    pub outcome: ValidationOutcome,
    /// Result message
    pub message: String,
    /// Severity level
    pub severity: ValidationSeverity,
    /// Validation score
    pub score: f64,
}

/// Validation outcome
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationOutcome {
    Pass,
    Warning,
    Fail,
    Skip,
    Error,
}

/// Consistency check results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsistencyResults {
    /// Quantum consistency check
    pub quantum_consistency: ConsistencyResult,
    /// Temporal consistency check
    pub temporal_consistency: ConsistencyResult,
    /// Risk consistency check
    pub risk_consistency: ConsistencyResult,
    /// Overall consistency score
    pub overall_consistency: f64,
}

/// Individual consistency result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsistencyResult {
    /// Consistency score
    pub score: f64,
    /// Is consistent
    pub is_consistent: bool,
    /// Inconsistency details
    pub inconsistencies: Vec<String>,
    /// Confidence in consistency check
    pub confidence: f64,
}

/// Compliance check results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceResults {
    /// Framework compliance results
    pub framework_results: HashMap<String, ComplianceResult>,
    /// Overall compliance status
    pub overall_status: ComplianceStatus,
    /// Compliance score
    pub compliance_score: f64,
}

/// Individual compliance result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceResult {
    /// Framework name
    pub framework: String,
    /// Compliance status
    pub status: ComplianceStatus,
    /// Violations found
    pub violations: Vec<String>,
    /// Compliance score
    pub score: f64,
}

/// Validation summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationSummary {
    /// Total rules checked
    pub total_rules: usize,
    /// Rules passed
    pub rules_passed: usize,
    /// Rules failed
    pub rules_failed: usize,
    /// Warnings generated
    pub warnings: usize,
    /// Critical issues
    pub critical_issues: usize,
    /// Validation duration
    pub validation_duration: std::time::Duration,
    /// Overall score
    pub overall_score: f64,
}

/// Validation recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationRecommendation {
    /// Recommendation type
    pub recommendation_type: RecommendationType,
    /// Priority level
    pub priority: ValidationPriority,
    /// Description
    pub description: String,
    /// Expected impact
    pub expected_impact: String,
    /// Implementation effort
    pub implementation_effort: ImplementationEffort,
}

/// Recommendation types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecommendationType {
    AdjustRisk,
    ModifyStrategy,
    ChangePosition,
    AddConstraints,
    RequireApproval,
    RejectDecision,
    RequestReview,
    UpdateParameters,
}

/// Implementation effort levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImplementationEffort {
    Trivial,
    Low,
    Medium,
    High,
    Significant,
}

/// Validation record for history tracking
#[derive(Debug, Clone)]
pub struct ValidationRecord {
    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Decision validated
    pub decision: TradingDecision,
    /// Validation result
    pub result: ValidationResult,
    /// Validation duration
    pub duration: std::time::Duration,
    /// Validator version
    pub validator_version: String,
}

/// Validation performance tracker
#[derive(Debug)]
pub struct ValidationPerformanceTracker {
    /// Validation success rate
    pub success_rate: f64,
    /// Average validation time
    pub avg_validation_time: std::time::Duration,
    /// Rule performance metrics
    pub rule_performance: HashMap<String, RulePerformance>,
    /// False positive rate
    pub false_positive_rate: f64,
    /// False negative rate
    pub false_negative_rate: f64,
}

/// Rule performance metrics
#[derive(Debug)]
pub struct RulePerformance {
    /// Rule execution count
    pub execution_count: usize,
    /// Success rate
    pub success_rate: f64,
    /// Average execution time
    pub avg_execution_time: std::time::Duration,
    /// False positive rate
    pub false_positive_rate: f64,
    /// Rule effectiveness score
    pub effectiveness_score: f64,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            strict_mode: false,
            max_risk_threshold: 0.1,
            min_confidence: 0.6,
            quantum_consistency: true,
            compliance_checks: true,
            validation_timeout: std::time::Duration::from_secs(30),
            performance_validation: true,
        }
    }
}

impl DecisionValidator {
    /// Create a new decision validator
    pub fn new(config: ValidationConfig) -> QarResult<Self> {
        let validation_rules = Self::initialize_validation_rules();
        
        let consistency_checker = ConsistencyChecker {
            consistency_threshold: 0.8,
            temporal_window: std::time::Duration::from_secs(3600),
            decision_history: Vec::new(),
            consistency_metrics: ConsistencyMetrics {
                quantum_classical_alignment: 1.0,
                temporal_consistency: 1.0,
                risk_consistency: 1.0,
                strategy_consistency: 1.0,
            },
        };

        let compliance_monitor = ComplianceMonitor {
            active_frameworks: vec![
                RegulatoryFramework::MiFID2,
                RegulatoryFramework::FINRA,
            ],
            compliance_status: HashMap::new(),
            violation_history: Vec::new(),
            compliance_metrics: ComplianceMetrics {
                overall_score: 1.0,
                violations_per_framework: HashMap::new(),
                avg_resolution_time: std::time::Duration::from_secs(3600),
                compliance_trend: ComplianceTrend::Stable,
            },
        };

        let performance_tracker = ValidationPerformanceTracker {
            success_rate: 0.95,
            avg_validation_time: std::time::Duration::from_millis(100),
            rule_performance: HashMap::new(),
            false_positive_rate: 0.02,
            false_negative_rate: 0.01,
        };

        Ok(Self {
            config,
            validation_rules,
            consistency_checker,
            compliance_monitor,
            validation_history: Vec::new(),
            performance_tracker,
        })
    }

    /// Validate a trading decision comprehensively
    pub async fn validate_decision(
        &mut self,
        decision: &TradingDecision,
        factors: &FactorMap,
        analysis: &AnalysisResult,
        risk_assessment: &RiskAssessment,
        quantum_insights: Option<&QuantumInsights>,
    ) -> QarResult<ValidationResult> {
        let start_time = std::time::Instant::now();

        // Execute basic validation rules
        let basic_results = self.validate_basic_rules(decision, factors, analysis).await?;

        // Execute risk validation rules
        let risk_results = self.validate_risk_rules(decision, risk_assessment).await?;

        // Execute quantum validation rules if enabled
        let quantum_results = if self.config.quantum_consistency && quantum_insights.is_some() {
            self.validate_quantum_rules(decision, quantum_insights.unwrap()).await?
        } else {
            Vec::new()
        };

        // Execute market condition validation
        let market_results = self.validate_market_conditions(decision, factors, analysis).await?;

        // Execute compliance validation
        let compliance_results = if self.config.compliance_checks {
            self.validate_compliance(decision, risk_assessment).await?
        } else {
            ComplianceResults {
                framework_results: HashMap::new(),
                overall_status: ComplianceStatus::Exempted,
                compliance_score: 1.0,
            }
        };

        // Execute consistency checks
        let consistency_results = self.check_consistency(decision, factors, analysis).await?;

        // Combine all rule results
        let mut all_rule_results = basic_results;
        all_rule_results.extend(risk_results);
        all_rule_results.extend(quantum_results);
        all_rule_results.extend(market_results);

        // Determine overall validation status
        let status = self.determine_validation_status(&all_rule_results, &consistency_results, &compliance_results);

        // Generate validation summary
        let validation_duration = start_time.elapsed();
        let summary = self.generate_validation_summary(&all_rule_results, validation_duration);

        // Generate recommendations
        let recommendations = self.generate_recommendations(&all_rule_results, &consistency_results, &compliance_results)?;

        let result = ValidationResult {
            status,
            rule_results: all_rule_results,
            consistency_results,
            compliance_results,
            summary,
            recommendations,
        };

        // Record validation for history and performance tracking
        self.record_validation(decision.clone(), result.clone(), validation_duration);

        Ok(result)
    }

    /// Initialize default validation rules
    fn initialize_validation_rules() -> ValidationRuleSet {
        ValidationRuleSet {
            basic_rules: vec![
                BasicValidationRule {
                    id: "confidence_check".to_string(),
                    name: "Confidence Level Check".to_string(),
                    description: "Ensures decision confidence meets minimum threshold".to_string(),
                    validator: Self::validate_confidence,
                    severity: ValidationSeverity::Error,
                    enabled: true,
                },
                BasicValidationRule {
                    id: "expected_return_check".to_string(),
                    name: "Expected Return Validation".to_string(),
                    description: "Validates expected return is reasonable".to_string(),
                    validator: Self::validate_expected_return,
                    severity: ValidationSeverity::Warning,
                    enabled: true,
                },
            ],
            risk_rules: vec![
                RiskValidationRule {
                    id: "var_limit_check".to_string(),
                    risk_metric: RiskMetric::VaR95,
                    threshold: 0.05,
                    comparison: ComparisonType::LessEqual,
                    severity: ValidationSeverity::Critical,
                },
                RiskValidationRule {
                    id: "liquidity_risk_check".to_string(),
                    risk_metric: RiskMetric::LiquidityRisk,
                    threshold: 0.7,
                    comparison: ComparisonType::LessEqual,
                    severity: ValidationSeverity::Warning,
                },
            ],
            quantum_rules: vec![
                QuantumValidationRule {
                    id: "coherence_check".to_string(),
                    quantum_property: QuantumProperty::Coherence,
                    expected_range: (0.7, 1.0),
                    tolerance: 0.1,
                    severity: ValidationSeverity::Warning,
                },
                QuantumValidationRule {
                    id: "entanglement_check".to_string(),
                    quantum_property: QuantumProperty::Entanglement,
                    expected_range: (0.0, 1.0),
                    tolerance: 0.05,
                    severity: ValidationSeverity::Info,
                },
            ],
            market_rules: vec![
                MarketValidationRule {
                    id: "high_volatility_check".to_string(),
                    condition: MarketCondition::HighVolatility,
                    compatible_decisions: vec![DecisionType::Hold],
                    severity: ValidationSeverity::Warning,
                },
                MarketValidationRule {
                    id: "low_liquidity_check".to_string(),
                    condition: MarketCondition::LowLiquidity,
                    compatible_decisions: vec![DecisionType::Hold, DecisionType::Sell],
                    severity: ValidationSeverity::Error,
                },
            ],
            compliance_rules: vec![
                ComplianceRule {
                    id: "mifid2_position_limit".to_string(),
                    framework: RegulatoryFramework::MiFID2,
                    description: "Position size limits under MiFID2".to_string(),
                    checker: Self::check_mifid2_compliance,
                    mandatory: true,
                },
            ],
            custom_rules: Vec::new(),
        }
    }

    /// Validate basic decision rules
    async fn validate_basic_rules(
        &self,
        decision: &TradingDecision,
        factors: &FactorMap,
        analysis: &AnalysisResult,
    ) -> QarResult<Vec<RuleValidationResult>> {
        let mut results = Vec::new();

        for rule in &self.validation_rules.basic_rules {
            if !rule.enabled {
                continue;
            }

            let validation_result = (rule.validator)(decision, factors, analysis)?;
            
            results.push(RuleValidationResult {
                rule_id: rule.id.clone(),
                rule_name: rule.name.clone(),
                outcome: if validation_result.status == ValidationStatus::Passed {
                    ValidationOutcome::Pass
                } else {
                    ValidationOutcome::Fail
                },
                message: validation_result.summary.overall_score.to_string(),
                severity: rule.severity.clone(),
                score: validation_result.summary.overall_score,
            });
        }

        Ok(results)
    }

    /// Validate risk rules
    async fn validate_risk_rules(
        &self,
        decision: &TradingDecision,
        risk_assessment: &RiskAssessment,
    ) -> QarResult<Vec<RuleValidationResult>> {
        let mut results = Vec::new();

        for rule in &self.validation_rules.risk_rules {
            let risk_value = self.extract_risk_metric_value(&rule.risk_metric, risk_assessment);
            let passes = self.compare_values(risk_value, rule.threshold, &rule.comparison);

            results.push(RuleValidationResult {
                rule_id: rule.id.clone(),
                rule_name: format!("{:?} Validation", rule.risk_metric),
                outcome: if passes { ValidationOutcome::Pass } else { ValidationOutcome::Fail },
                message: format!("{:?}: {:.4} vs threshold {:.4}", rule.risk_metric, risk_value, rule.threshold),
                severity: rule.severity.clone(),
                score: if passes { 1.0 } else { 0.0 },
            });
        }

        Ok(results)
    }

    /// Validate quantum rules
    async fn validate_quantum_rules(
        &self,
        decision: &TradingDecision,
        quantum_insights: &QuantumInsights,
    ) -> QarResult<Vec<RuleValidationResult>> {
        let mut results = Vec::new();

        for rule in &self.validation_rules.quantum_rules {
            let quantum_value = self.extract_quantum_property_value(&rule.quantum_property, quantum_insights);
            let in_range = quantum_value >= rule.expected_range.0 && quantum_value <= rule.expected_range.1;
            let passes = in_range || (quantum_value - rule.expected_range.0).abs() <= rule.tolerance ||
                        (quantum_value - rule.expected_range.1).abs() <= rule.tolerance;

            results.push(RuleValidationResult {
                rule_id: rule.id.clone(),
                rule_name: format!("{:?} Validation", rule.quantum_property),
                outcome: if passes { ValidationOutcome::Pass } else { ValidationOutcome::Warning },
                message: format!("{:?}: {:.4} (expected: {:.4}-{:.4})", 
                    rule.quantum_property, quantum_value, rule.expected_range.0, rule.expected_range.1),
                severity: rule.severity.clone(),
                score: if passes { 1.0 } else { 0.5 },
            });
        }

        Ok(results)
    }

    /// Validate market condition rules
    async fn validate_market_conditions(
        &self,
        decision: &TradingDecision,
        factors: &FactorMap,
        analysis: &AnalysisResult,
    ) -> QarResult<Vec<RuleValidationResult>> {
        let mut results = Vec::new();

        for rule in &self.validation_rules.market_rules {
            let condition_met = self.check_market_condition(&rule.condition, factors, analysis)?;
            let decision_compatible = rule.compatible_decisions.contains(&decision.decision_type);
            let passes = !condition_met || decision_compatible;

            results.push(RuleValidationResult {
                rule_id: rule.id.clone(),
                rule_name: format!("{:?} Compatibility", rule.condition),
                outcome: if passes { ValidationOutcome::Pass } else { ValidationOutcome::Warning },
                message: if condition_met {
                    format!("{:?} detected, decision {:?} compatibility: {}", 
                        rule.condition, decision.decision_type, decision_compatible)
                } else {
                    format!("{:?} not detected", rule.condition)
                },
                severity: rule.severity.clone(),
                score: if passes { 1.0 } else { 0.3 },
            });
        }

        Ok(results)
    }

    /// Validate compliance requirements
    async fn validate_compliance(
        &self,
        decision: &TradingDecision,
        risk_assessment: &RiskAssessment,
    ) -> QarResult<ComplianceResults> {
        let mut framework_results = HashMap::new();

        for rule in &self.validation_rules.compliance_rules {
            let compliance_result = (rule.checker)(decision, risk_assessment)?;
            
            framework_results.insert(
                format!("{:?}", rule.framework),
                ComplianceResult {
                    framework: format!("{:?}", rule.framework),
                    status: if compliance_result.compliance_score > 0.8 {
                        ComplianceStatus::Compliant
                    } else if compliance_result.compliance_score > 0.6 {
                        ComplianceStatus::Warning
                    } else {
                        ComplianceStatus::Violation
                    },
                    violations: compliance_result.framework_results.keys().cloned().collect(),
                    score: compliance_result.compliance_score,
                }
            );
        }

        let overall_score = framework_results.values()
            .map(|r| r.score)
            .sum::<f64>() / framework_results.len().max(1) as f64;

        let overall_status = if overall_score > 0.8 {
            ComplianceStatus::Compliant
        } else if overall_score > 0.6 {
            ComplianceStatus::Warning
        } else {
            ComplianceStatus::Violation
        };

        Ok(ComplianceResults {
            framework_results,
            overall_status,
            compliance_score: overall_score,
        })
    }

    /// Check decision consistency
    async fn check_consistency(
        &mut self,
        decision: &TradingDecision,
        factors: &FactorMap,
        analysis: &AnalysisResult,
    ) -> QarResult<ConsistencyResults> {
        // Quantum consistency (placeholder - would involve quantum state comparison)
        let quantum_consistency = ConsistencyResult {
            score: 0.9,
            is_consistent: true,
            inconsistencies: Vec::new(),
            confidence: 0.8,
        };

        // Temporal consistency
        let temporal_consistency = self.check_temporal_consistency(decision)?;

        // Risk consistency
        let risk_consistency = self.check_risk_consistency(decision)?;

        let overall_consistency = (quantum_consistency.score + 
                                 temporal_consistency.score + 
                                 risk_consistency.score) / 3.0;

        // Update consistency metrics
        self.consistency_checker.consistency_metrics = ConsistencyMetrics {
            quantum_classical_alignment: quantum_consistency.score,
            temporal_consistency: temporal_consistency.score,
            risk_consistency: risk_consistency.score,
            strategy_consistency: overall_consistency,
        };

        Ok(ConsistencyResults {
            quantum_consistency,
            temporal_consistency,
            risk_consistency,
            overall_consistency,
        })
    }

    /// Check temporal consistency
    fn check_temporal_consistency(&self, decision: &TradingDecision) -> QarResult<ConsistencyResult> {
        if self.consistency_checker.decision_history.is_empty() {
            return Ok(ConsistencyResult {
                score: 1.0,
                is_consistent: true,
                inconsistencies: Vec::new(),
                confidence: 1.0,
            });
        }

        let recent_decisions: Vec<_> = self.consistency_checker.decision_history
            .iter()
            .filter(|d| {
                let time_diff = decision.timestamp - d.timestamp;
                time_diff.num_seconds() <= self.consistency_checker.temporal_window.as_secs() as i64
            })
            .collect();

        if recent_decisions.is_empty() {
            return Ok(ConsistencyResult {
                score: 1.0,
                is_consistent: true,
                inconsistencies: Vec::new(),
                confidence: 1.0,
            });
        }

        // Check for rapid decision type changes
        let mut inconsistencies = Vec::new();
        let mut consistent_decisions = 0;

        for recent_decision in &recent_decisions {
            if recent_decision.decision_type == decision.decision_type ||
               (matches!(recent_decision.decision_type, DecisionType::Hold) && 
                matches!(decision.decision_type, DecisionType::Buy | DecisionType::Sell)) {
                consistent_decisions += 1;
            } else {
                inconsistencies.push(format!(
                    "Decision type changed from {:?} to {:?}",
                    recent_decision.decision_type, decision.decision_type
                ));
            }
        }

        let consistency_score = consistent_decisions as f64 / recent_decisions.len() as f64;
        let is_consistent = consistency_score >= self.consistency_checker.consistency_threshold;

        Ok(ConsistencyResult {
            score: consistency_score,
            is_consistent,
            inconsistencies,
            confidence: 0.8,
        })
    }

    /// Check risk consistency
    fn check_risk_consistency(&self, decision: &TradingDecision) -> QarResult<ConsistencyResult> {
        // Simple risk consistency check
        let risk_score = decision.risk_assessment.unwrap_or(0.5);
        let confidence = decision.confidence;

        // Risk and confidence should be inversely related
        let expected_relationship = (1.0 - risk_score - confidence).abs() < 0.3;
        
        let inconsistencies = if !expected_relationship {
            vec![format!("Risk-confidence relationship inconsistent: risk={:.2}, confidence={:.2}", 
                        risk_score, confidence)]
        } else {
            Vec::new()
        };

        Ok(ConsistencyResult {
            score: if expected_relationship { 1.0 } else { 0.5 },
            is_consistent: expected_relationship,
            inconsistencies,
            confidence: 0.7,
        })
    }

    /// Extract risk metric value
    fn extract_risk_metric_value(&self, metric: &RiskMetric, risk_assessment: &RiskAssessment) -> f64 {
        match metric {
            RiskMetric::VaR95 => risk_assessment.var_95,
            RiskMetric::VaR99 => risk_assessment.var_95 * 1.3, // Approximate VaR99
            RiskMetric::ExpectedShortfall => risk_assessment.expected_shortfall,
            RiskMetric::MaxDrawdown => risk_assessment.max_drawdown_risk,
            RiskMetric::LiquidityRisk => risk_assessment.liquidity_risk,
            _ => risk_assessment.risk_score,
        }
    }

    /// Extract quantum property value
    fn extract_quantum_property_value(&self, property: &QuantumProperty, insights: &QuantumInsights) -> f64 {
        match property {
            QuantumProperty::Coherence => {
                insights.superposition_analysis.get(0).cloned().unwrap_or(0.5)
            },
            QuantumProperty::Entanglement => {
                insights.entanglement_correlations.values().next().cloned().unwrap_or(0.0)
            },
            QuantumProperty::Superposition => {
                insights.superposition_analysis.iter().sum::<f64>() / insights.superposition_analysis.len() as f64
            },
            QuantumProperty::Uncertainty => insights.measurement_uncertainty,
            _ => 0.5,
        }
    }

    /// Compare values based on comparison type
    fn compare_values(&self, value: f64, threshold: f64, comparison: &ComparisonType) -> bool {
        match comparison {
            ComparisonType::LessThan => value < threshold,
            ComparisonType::LessEqual => value <= threshold,
            ComparisonType::GreaterThan => value > threshold,
            ComparisonType::GreaterEqual => value >= threshold,
            ComparisonType::Equal => (value - threshold).abs() < 1e-10,
            ComparisonType::NotEqual => (value - threshold).abs() >= 1e-10,
            ComparisonType::InRange => value >= threshold && value <= threshold * 2.0, // Simplified
            ComparisonType::OutOfRange => value < threshold || value > threshold * 2.0,
        }
    }

    /// Check market condition
    fn check_market_condition(
        &self,
        condition: &MarketCondition,
        factors: &FactorMap,
        analysis: &AnalysisResult,
    ) -> QarResult<bool> {
        match condition {
            MarketCondition::HighVolatility => {
                let volatility = factors.get_factor(&crate::core::StandardFactors::Volatility)?;
                Ok(volatility > 0.6)
            },
            MarketCondition::LowLiquidity => {
                let liquidity = factors.get_factor(&crate::core::StandardFactors::Liquidity)?;
                Ok(liquidity < 0.3)
            },
            MarketCondition::TrendingMarket => {
                Ok(analysis.trend_strength > 0.7)
            },
            MarketCondition::RangingMarket => {
                Ok(analysis.trend_strength < 0.3)
            },
            MarketCondition::CrisisConditions => {
                let volatility = factors.get_factor(&crate::core::StandardFactors::Volatility)?;
                let risk = factors.get_factor(&crate::core::StandardFactors::Risk)?;
                Ok(volatility > 0.8 && risk > 0.8)
            },
            MarketCondition::NormalConditions => {
                let volatility = factors.get_factor(&crate::core::StandardFactors::Volatility)?;
                let risk = factors.get_factor(&crate::core::StandardFactors::Risk)?;
                Ok(volatility < 0.5 && risk < 0.5)
            },
        }
    }

    /// Determine overall validation status
    fn determine_validation_status(
        &self,
        rule_results: &[RuleValidationResult],
        consistency_results: &ConsistencyResults,
        compliance_results: &ComplianceResults,
    ) -> ValidationStatus {
        let critical_failures = rule_results.iter()
            .filter(|r| matches!(r.severity, ValidationSeverity::Critical | ValidationSeverity::Blocking) &&
                       matches!(r.outcome, ValidationOutcome::Fail))
            .count();

        let regular_failures = rule_results.iter()
            .filter(|r| matches!(r.severity, ValidationSeverity::Error) &&
                       matches!(r.outcome, ValidationOutcome::Fail))
            .count();

        let warnings = rule_results.iter()
            .filter(|r| matches!(r.outcome, ValidationOutcome::Warning))
            .count();

        // Check compliance
        let compliance_violations = matches!(compliance_results.overall_status, ComplianceStatus::Violation);

        // Check consistency
        let consistency_issues = consistency_results.overall_consistency < 0.7;

        if critical_failures > 0 || compliance_violations {
            ValidationStatus::Blocked
        } else if regular_failures > 0 || consistency_issues {
            ValidationStatus::Failed
        } else if warnings > 0 {
            ValidationStatus::PassedWithWarnings
        } else {
            ValidationStatus::Passed
        }
    }

    /// Generate validation summary
    fn generate_validation_summary(
        &self,
        rule_results: &[RuleValidationResult],
        validation_duration: std::time::Duration,
    ) -> ValidationSummary {
        let total_rules = rule_results.len();
        let rules_passed = rule_results.iter()
            .filter(|r| matches!(r.outcome, ValidationOutcome::Pass))
            .count();
        let rules_failed = rule_results.iter()
            .filter(|r| matches!(r.outcome, ValidationOutcome::Fail))
            .count();
        let warnings = rule_results.iter()
            .filter(|r| matches!(r.outcome, ValidationOutcome::Warning))
            .count();
        let critical_issues = rule_results.iter()
            .filter(|r| matches!(r.severity, ValidationSeverity::Critical | ValidationSeverity::Blocking))
            .count();

        let overall_score = if total_rules > 0 {
            rule_results.iter().map(|r| r.score).sum::<f64>() / total_rules as f64
        } else {
            1.0
        };

        ValidationSummary {
            total_rules,
            rules_passed,
            rules_failed,
            warnings,
            critical_issues,
            validation_duration,
            overall_score,
        }
    }

    /// Generate validation recommendations
    fn generate_recommendations(
        &self,
        rule_results: &[RuleValidationResult],
        consistency_results: &ConsistencyResults,
        compliance_results: &ComplianceResults,
    ) -> QarResult<Vec<ValidationRecommendation>> {
        let mut recommendations = Vec::new();

        // Generate recommendations based on failed rules
        for result in rule_results {
            if matches!(result.outcome, ValidationOutcome::Fail) {
                match result.severity {
                    ValidationSeverity::Critical | ValidationSeverity::Blocking => {
                        recommendations.push(ValidationRecommendation {
                            recommendation_type: RecommendationType::RejectDecision,
                            priority: ValidationPriority::Critical,
                            description: format!("Critical rule failure: {}", result.rule_name),
                            expected_impact: "Prevents potentially harmful decision execution".to_string(),
                            implementation_effort: ImplementationEffort::Trivial,
                        });
                    },
                    ValidationSeverity::Error => {
                        recommendations.push(ValidationRecommendation {
                            recommendation_type: RecommendationType::AdjustRisk,
                            priority: ValidationPriority::High,
                            description: format!("Address rule failure: {}", result.rule_name),
                            expected_impact: "Improves decision safety and compliance".to_string(),
                            implementation_effort: ImplementationEffort::Low,
                        });
                    },
                    _ => {},
                }
            }
        }

        // Generate consistency recommendations
        if consistency_results.overall_consistency < 0.7 {
            recommendations.push(ValidationRecommendation {
                recommendation_type: RecommendationType::RequestReview,
                priority: ValidationPriority::Medium,
                description: "Decision consistency issues detected".to_string(),
                expected_impact: "Ensures decision alignment with strategy".to_string(),
                implementation_effort: ImplementationEffort::Medium,
            });
        }

        // Generate compliance recommendations
        if matches!(compliance_results.overall_status, ComplianceStatus::Violation) {
            recommendations.push(ValidationRecommendation {
                recommendation_type: RecommendationType::RequireApproval,
                priority: ValidationPriority::Critical,
                description: "Compliance violations detected".to_string(),
                expected_impact: "Ensures regulatory compliance".to_string(),
                implementation_effort: ImplementationEffort::High,
            });
        }

        Ok(recommendations)
    }

    /// Record validation for history and performance tracking
    fn record_validation(
        &mut self,
        decision: TradingDecision,
        result: ValidationResult,
        duration: std::time::Duration,
    ) {
        let record = ValidationRecord {
            timestamp: chrono::Utc::now(),
            decision: decision.clone(),
            result,
            duration,
            validator_version: "1.0.0".to_string(),
        };

        self.validation_history.push(record);

        // Update decision history for consistency checking
        self.consistency_checker.decision_history.push(decision);

        // Maintain history size
        if self.validation_history.len() > 1000 {
            self.validation_history.remove(0);
        }
        if self.consistency_checker.decision_history.len() > 100 {
            self.consistency_checker.decision_history.remove(0);
        }

        // Update performance metrics
        self.update_performance_metrics(&duration);
    }

    /// Update performance metrics
    fn update_performance_metrics(&mut self, validation_duration: &std::time::Duration) {
        let total_validations = self.validation_history.len();
        let successful_validations = self.validation_history.iter()
            .filter(|r| matches!(r.result.status, ValidationStatus::Passed | ValidationStatus::PassedWithWarnings))
            .count();

        self.performance_tracker.success_rate = if total_validations > 0 {
            successful_validations as f64 / total_validations as f64
        } else {
            1.0
        };

        // Update average validation time
        let total_time: std::time::Duration = self.validation_history.iter()
            .map(|r| r.duration)
            .sum();
        
        self.performance_tracker.avg_validation_time = if total_validations > 0 {
            total_time / total_validations as u32
        } else {
            *validation_duration
        };
    }

    /// Basic validation functions
    fn validate_confidence(
        decision: &TradingDecision,
        _factors: &FactorMap,
        _analysis: &AnalysisResult,
    ) -> QarResult<ValidationResult> {
        let passes = decision.confidence >= 0.6;
        Ok(ValidationResult {
            status: if passes { ValidationStatus::Passed } else { ValidationStatus::Failed },
            rule_results: Vec::new(),
            consistency_results: ConsistencyResults {
                quantum_consistency: ConsistencyResult {
                    score: 1.0,
                    is_consistent: true,
                    inconsistencies: Vec::new(),
                    confidence: 1.0,
                },
                temporal_consistency: ConsistencyResult {
                    score: 1.0,
                    is_consistent: true,
                    inconsistencies: Vec::new(),
                    confidence: 1.0,
                },
                risk_consistency: ConsistencyResult {
                    score: 1.0,
                    is_consistent: true,
                    inconsistencies: Vec::new(),
                    confidence: 1.0,
                },
                overall_consistency: 1.0,
            },
            compliance_results: ComplianceResults {
                framework_results: HashMap::new(),
                overall_status: ComplianceStatus::Compliant,
                compliance_score: 1.0,
            },
            summary: ValidationSummary {
                total_rules: 1,
                rules_passed: if passes { 1 } else { 0 },
                rules_failed: if passes { 0 } else { 1 },
                warnings: 0,
                critical_issues: 0,
                validation_duration: std::time::Duration::from_millis(1),
                overall_score: if passes { 1.0 } else { 0.0 },
            },
            recommendations: Vec::new(),
        })
    }

    fn validate_expected_return(
        decision: &TradingDecision,
        _factors: &FactorMap,
        _analysis: &AnalysisResult,
    ) -> QarResult<ValidationResult> {
        let expected_return = decision.expected_return.unwrap_or(0.0);
        let reasonable = expected_return >= -0.5 && expected_return <= 0.5; // ±50% is reasonable

        Ok(ValidationResult {
            status: if reasonable { ValidationStatus::Passed } else { ValidationStatus::PassedWithWarnings },
            rule_results: Vec::new(),
            consistency_results: ConsistencyResults {
                quantum_consistency: ConsistencyResult {
                    score: 1.0,
                    is_consistent: true,
                    inconsistencies: Vec::new(),
                    confidence: 1.0,
                },
                temporal_consistency: ConsistencyResult {
                    score: 1.0,
                    is_consistent: true,
                    inconsistencies: Vec::new(),
                    confidence: 1.0,
                },
                risk_consistency: ConsistencyResult {
                    score: 1.0,
                    is_consistent: true,
                    inconsistencies: Vec::new(),
                    confidence: 1.0,
                },
                overall_consistency: 1.0,
            },
            compliance_results: ComplianceResults {
                framework_results: HashMap::new(),
                overall_status: ComplianceStatus::Compliant,
                compliance_score: 1.0,
            },
            summary: ValidationSummary {
                total_rules: 1,
                rules_passed: if reasonable { 1 } else { 0 },
                rules_failed: 0,
                warnings: if reasonable { 0 } else { 1 },
                critical_issues: 0,
                validation_duration: std::time::Duration::from_millis(1),
                overall_score: if reasonable { 1.0 } else { 0.7 },
            },
            recommendations: Vec::new(),
        })
    }

    fn check_mifid2_compliance(
        _decision: &TradingDecision,
        risk_assessment: &RiskAssessment,
    ) -> QarResult<ComplianceResults> {
        // Simplified MiFID2 compliance check
        let compliant = risk_assessment.var_95 <= 0.05; // Simple VaR limit

        Ok(ComplianceResults {
            framework_results: HashMap::from([
                ("MiFID2".to_string(), ComplianceResult {
                    framework: "MiFID2".to_string(),
                    status: if compliant { ComplianceStatus::Compliant } else { ComplianceStatus::Violation },
                    violations: if compliant { Vec::new() } else { vec!["VaR limit exceeded".to_string()] },
                    score: if compliant { 1.0 } else { 0.5 },
                })
            ]),
            overall_status: if compliant { ComplianceStatus::Compliant } else { ComplianceStatus::Violation },
            compliance_score: if compliant { 1.0 } else { 0.5 },
        })
    }

    /// Get validation history
    pub fn get_validation_history(&self) -> &[ValidationRecord] {
        &self.validation_history
    }

    /// Get performance metrics
    pub fn get_performance_metrics(&self) -> &ValidationPerformanceTracker {
        &self.performance_tracker
    }

    /// Get consistency metrics
    pub fn get_consistency_metrics(&self) -> &ConsistencyMetrics {
        &self.consistency_checker.consistency_metrics
    }

    /// Get compliance metrics
    pub fn get_compliance_metrics(&self) -> &ComplianceMetrics {
        &self.compliance_monitor.compliance_metrics
    }

    /// Update validation configuration
    pub fn update_config(&mut self, config: ValidationConfig) {
        self.config = config;
    }

    /// Add custom validation rule
    pub fn add_custom_rule(&mut self, rule: CustomValidationRule) {
        self.validation_rules.custom_rules.push(rule);
    }

    /// Enable/disable validation rule
    pub fn set_rule_enabled(&mut self, rule_id: &str, enabled: bool) {
        for rule in &mut self.validation_rules.basic_rules {
            if rule.id == rule_id {
                rule.enabled = enabled;
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{StandardFactors, DecisionType};
    use crate::analysis::{TrendDirection, VolatilityLevel, MarketRegime};

    fn create_test_decision() -> TradingDecision {
        TradingDecision {
            decision_type: DecisionType::Buy,
            confidence: 0.8,
            expected_return: Some(0.05),
            risk_assessment: Some(0.3),
            urgency_score: Some(0.6),
            reasoning: "Test decision".to_string(),
            timestamp: chrono::Utc::now(),
        }
    }

    fn create_test_factors() -> FactorMap {
        let mut factors = std::collections::HashMap::new();
        factors.insert(StandardFactors::Volatility.to_string(), 0.2);
        factors.insert(StandardFactors::Liquidity.to_string(), 0.7);
        factors.insert(StandardFactors::Risk.to_string(), 0.4);
        factors.insert(StandardFactors::Trend.to_string(), 0.7);
        factors.insert(StandardFactors::Momentum.to_string(), 0.6);
        factors.insert(StandardFactors::Volume.to_string(), 0.8);
        factors.insert(StandardFactors::Sentiment.to_string(), 0.7);
        factors.insert(StandardFactors::Efficiency.to_string(), 0.6);
        FactorMap::new(factors).unwrap()
    }

    fn create_test_analysis() -> AnalysisResult {
        AnalysisResult {
            timestamp: chrono::Utc::now(),
            trend: TrendDirection::Bullish,
            trend_strength: 0.8,
            volatility: VolatilityLevel::Medium,
            regime: MarketRegime::Bull,
            confidence: 0.9,
            metrics: HashMap::new(),
        }
    }

    fn create_test_risk_assessment() -> RiskAssessment {
        RiskAssessment {
            risk_score: 0.4,
            var_95: 0.03,
            expected_shortfall: 0.045,
            max_drawdown_risk: 0.06,
            liquidity_risk: 0.2,
            risk_adjusted_return: 0.12,
        }
    }

    fn create_test_quantum_insights() -> QuantumInsights {
        QuantumInsights {
            superposition_analysis: vec![0.8, 0.2],
            entanglement_correlations: HashMap::from([
                ("factor1_factor2".to_string(), 0.7),
            ]),
            interference_patterns: vec![0.6, 0.4, 0.2],
            measurement_uncertainty: 0.1,
        }
    }

    #[tokio::test]
    async fn test_decision_validator_creation() {
        let config = ValidationConfig::default();
        let validator = DecisionValidator::new(config);
        assert!(validator.is_ok());
    }

    #[tokio::test]
    async fn test_comprehensive_validation() {
        let config = ValidationConfig::default();
        let mut validator = DecisionValidator::new(config).unwrap();
        
        let decision = create_test_decision();
        let factors = create_test_factors();
        let analysis = create_test_analysis();
        let risk_assessment = create_test_risk_assessment();
        let quantum_insights = create_test_quantum_insights();

        let result = validator.validate_decision(
            &decision, 
            &factors, 
            &analysis, 
            &risk_assessment, 
            Some(&quantum_insights)
        ).await;

        assert!(result.is_ok());
        let validation_result = result.unwrap();
        
        assert!(!validation_result.rule_results.is_empty());
        assert!(validation_result.summary.total_rules > 0);
        assert!(validation_result.summary.overall_score >= 0.0 && validation_result.summary.overall_score <= 1.0);
    }

    #[tokio::test]
    async fn test_basic_validation_rules() {
        let config = ValidationConfig::default();
        let validator = DecisionValidator::new(config).unwrap();
        
        let decision = create_test_decision();
        let factors = create_test_factors();
        let analysis = create_test_analysis();

        let results = validator.validate_basic_rules(&decision, &factors, &analysis).await;
        assert!(results.is_ok());
        
        let rule_results = results.unwrap();
        assert!(!rule_results.is_empty());
        
        for result in &rule_results {
            assert!(!result.rule_id.is_empty());
            assert!(result.score >= 0.0 && result.score <= 1.0);
        }
    }

    #[tokio::test]
    async fn test_risk_validation_rules() {
        let config = ValidationConfig::default();
        let validator = DecisionValidator::new(config).unwrap();
        
        let decision = create_test_decision();
        let risk_assessment = create_test_risk_assessment();

        let results = validator.validate_risk_rules(&decision, &risk_assessment).await;
        assert!(results.is_ok());
        
        let rule_results = results.unwrap();
        assert!(!rule_results.is_empty());
    }

    #[tokio::test]
    async fn test_quantum_validation_rules() {
        let config = ValidationConfig::default();
        let validator = DecisionValidator::new(config).unwrap();
        
        let decision = create_test_decision();
        let quantum_insights = create_test_quantum_insights();

        let results = validator.validate_quantum_rules(&decision, &quantum_insights).await;
        assert!(results.is_ok());
        
        let rule_results = results.unwrap();
        assert!(!rule_results.is_empty());
    }

    #[tokio::test]
    async fn test_market_condition_validation() {
        let config = ValidationConfig::default();
        let validator = DecisionValidator::new(config).unwrap();
        
        let decision = create_test_decision();
        let factors = create_test_factors();
        let analysis = create_test_analysis();

        let results = validator.validate_market_conditions(&decision, &factors, &analysis).await;
        assert!(results.is_ok());
    }

    #[tokio::test]
    async fn test_compliance_validation() {
        let config = ValidationConfig::default();
        let validator = DecisionValidator::new(config).unwrap();
        
        let decision = create_test_decision();
        let risk_assessment = create_test_risk_assessment();

        let results = validator.validate_compliance(&decision, &risk_assessment).await;
        assert!(results.is_ok());
        
        let compliance_results = results.unwrap();
        assert!(compliance_results.compliance_score >= 0.0 && compliance_results.compliance_score <= 1.0);
    }

    #[tokio::test]
    async fn test_consistency_checking() {
        let config = ValidationConfig::default();
        let mut validator = DecisionValidator::new(config).unwrap();
        
        let decision = create_test_decision();
        let factors = create_test_factors();
        let analysis = create_test_analysis();

        let results = validator.check_consistency(&decision, &factors, &analysis).await;
        assert!(results.is_ok());
        
        let consistency_results = results.unwrap();
        assert!(consistency_results.overall_consistency >= 0.0 && consistency_results.overall_consistency <= 1.0);
    }

    #[test]
    fn test_risk_metric_extraction() {
        let config = ValidationConfig::default();
        let validator = DecisionValidator::new(config).unwrap();
        let risk_assessment = create_test_risk_assessment();

        let var_95 = validator.extract_risk_metric_value(&RiskMetric::VaR95, &risk_assessment);
        assert_eq!(var_95, risk_assessment.var_95);

        let liquidity_risk = validator.extract_risk_metric_value(&RiskMetric::LiquidityRisk, &risk_assessment);
        assert_eq!(liquidity_risk, risk_assessment.liquidity_risk);
    }

    #[test]
    fn test_quantum_property_extraction() {
        let config = ValidationConfig::default();
        let validator = DecisionValidator::new(config).unwrap();
        let quantum_insights = create_test_quantum_insights();

        let coherence = validator.extract_quantum_property_value(&QuantumProperty::Coherence, &quantum_insights);
        assert!(coherence >= 0.0 && coherence <= 1.0);

        let uncertainty = validator.extract_quantum_property_value(&QuantumProperty::Uncertainty, &quantum_insights);
        assert_eq!(uncertainty, quantum_insights.measurement_uncertainty);
    }

    #[test]
    fn test_value_comparison() {
        let config = ValidationConfig::default();
        let validator = DecisionValidator::new(config).unwrap();

        assert!(validator.compare_values(0.5, 1.0, &ComparisonType::LessThan));
        assert!(!validator.compare_values(1.5, 1.0, &ComparisonType::LessThan));
        
        assert!(validator.compare_values(1.0, 1.0, &ComparisonType::LessEqual));
        assert!(validator.compare_values(0.5, 1.0, &ComparisonType::LessEqual));
        
        assert!(validator.compare_values(1.5, 1.0, &ComparisonType::GreaterThan));
        assert!(!validator.compare_values(0.5, 1.0, &ComparisonType::GreaterThan));
    }

    #[tokio::test]
    async fn test_market_condition_detection() {
        let config = ValidationConfig::default();
        let validator = DecisionValidator::new(config).unwrap();
        let factors = create_test_factors();
        let analysis = create_test_analysis();

        let high_vol = validator.check_market_condition(&MarketCondition::HighVolatility, &factors, &analysis);
        assert!(high_vol.is_ok());

        let trending = validator.check_market_condition(&MarketCondition::TrendingMarket, &factors, &analysis);
        assert!(trending.is_ok());
        assert!(trending.unwrap()); // Should be true with trend_strength = 0.8
    }

    #[test]
    fn test_validation_status_determination() {
        let config = ValidationConfig::default();
        let validator = DecisionValidator::new(config).unwrap();

        // Test with critical failure
        let critical_rule_result = vec![RuleValidationResult {
            rule_id: "test".to_string(),
            rule_name: "Test Rule".to_string(),
            outcome: ValidationOutcome::Fail,
            message: "Test failure".to_string(),
            severity: ValidationSeverity::Critical,
            score: 0.0,
        }];

        let consistency_results = ConsistencyResults {
            quantum_consistency: ConsistencyResult {
                score: 1.0,
                is_consistent: true,
                inconsistencies: Vec::new(),
                confidence: 1.0,
            },
            temporal_consistency: ConsistencyResult {
                score: 1.0,
                is_consistent: true,
                inconsistencies: Vec::new(),
                confidence: 1.0,
            },
            risk_consistency: ConsistencyResult {
                score: 1.0,
                is_consistent: true,
                inconsistencies: Vec::new(),
                confidence: 1.0,
            },
            overall_consistency: 1.0,
        };

        let compliance_results = ComplianceResults {
            framework_results: HashMap::new(),
            overall_status: ComplianceStatus::Compliant,
            compliance_score: 1.0,
        };

        let status = validator.determine_validation_status(&critical_rule_result, &consistency_results, &compliance_results);
        assert_eq!(status, ValidationStatus::Blocked);
    }

    #[tokio::test]
    async fn test_recommendation_generation() {
        let config = ValidationConfig::default();
        let validator = DecisionValidator::new(config).unwrap();

        let failed_rule_result = vec![RuleValidationResult {
            rule_id: "test".to_string(),
            rule_name: "Test Rule".to_string(),
            outcome: ValidationOutcome::Fail,
            message: "Test failure".to_string(),
            severity: ValidationSeverity::Critical,
            score: 0.0,
        }];

        let consistency_results = ConsistencyResults {
            quantum_consistency: ConsistencyResult {
                score: 0.5,
                is_consistent: false,
                inconsistencies: vec!["Test inconsistency".to_string()],
                confidence: 0.8,
            },
            temporal_consistency: ConsistencyResult {
                score: 0.5,
                is_consistent: false,
                inconsistencies: Vec::new(),
                confidence: 0.8,
            },
            risk_consistency: ConsistencyResult {
                score: 0.5,
                is_consistent: false,
                inconsistencies: Vec::new(),
                confidence: 0.8,
            },
            overall_consistency: 0.5,
        };

        let compliance_results = ComplianceResults {
            framework_results: HashMap::new(),
            overall_status: ComplianceStatus::Violation,
            compliance_score: 0.3,
        };

        let recommendations = validator.generate_recommendations(&failed_rule_result, &consistency_results, &compliance_results);
        assert!(recommendations.is_ok());
        
        let recommendations = recommendations.unwrap();
        assert!(!recommendations.is_empty());
        
        // Should have recommendations for critical rule failure, consistency issues, and compliance violations
        assert!(recommendations.iter().any(|r| matches!(r.recommendation_type, RecommendationType::RejectDecision)));
        assert!(recommendations.iter().any(|r| matches!(r.recommendation_type, RecommendationType::RequestReview)));
        assert!(recommendations.iter().any(|r| matches!(r.recommendation_type, RecommendationType::RequireApproval)));
    }

    #[tokio::test]
    async fn test_validation_performance_tracking() {
        let config = ValidationConfig::default();
        let mut validator = DecisionValidator::new(config).unwrap();
        
        let decision = create_test_decision();
        let factors = create_test_factors();
        let analysis = create_test_analysis();
        let risk_assessment = create_test_risk_assessment();

        // Perform multiple validations
        for _ in 0..5 {
            let _ = validator.validate_decision(&decision, &factors, &analysis, &risk_assessment, None).await;
        }

        assert_eq!(validator.validation_history.len(), 5);
        assert!(validator.performance_tracker.success_rate >= 0.0 && validator.performance_tracker.success_rate <= 1.0);
        assert!(validator.performance_tracker.avg_validation_time.as_millis() > 0);
    }
}