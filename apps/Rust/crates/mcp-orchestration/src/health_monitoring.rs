//! Health Monitoring System
//!
//! Implements comprehensive health monitoring with predictive analytics,
//! automatic failover, and real-time alerting for the entire swarm ecosystem.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use tokio::sync::{RwLock, Mutex, mpsc, broadcast};
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use dashmap::DashMap;
use priority_queue::PriorityQueue;
use smallvec::SmallVec;
use tracing::{debug, info, warn, error, instrument, span, Level};
use metrics::{counter, gauge, histogram};
use chrono::{DateTime, Utc};

use crate::agents::{AgentHealth, AgentMetrics, MCPMessage};
use crate::{MCPOrchestrationError, SwarmType, HierarchyLevel, AgentStatus};

/// Health monitor configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthMonitorConfig {
    /// Health check interval in milliseconds
    pub check_interval_ms: u64,
    /// Health check timeout in milliseconds
    pub timeout_ms: u64,
    /// Number of consecutive failures before marking unhealthy
    pub failure_threshold: u32,
    /// Number of consecutive successes before marking healthy
    pub recovery_threshold: u32,
    /// Enable predictive health monitoring
    pub predictive_monitoring: bool,
    /// Enable auto-recovery
    pub auto_recovery: bool,
    /// Health score calculation weights
    pub score_weights: HealthScoreWeights,
    /// Alert configuration
    pub alerting: AlertConfig,
}

impl Default for HealthMonitorConfig {
    fn default() -> Self {
        Self {
            check_interval_ms: 5000,
            timeout_ms: 2000,
            failure_threshold: 3,
            recovery_threshold: 2,
            predictive_monitoring: true,
            auto_recovery: true,
            score_weights: HealthScoreWeights::default(),
            alerting: AlertConfig::default(),
        }
    }
}

/// Health score calculation weights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthScoreWeights {
    pub response_time_weight: f64,
    pub cpu_usage_weight: f64,
    pub memory_usage_weight: f64,
    pub error_rate_weight: f64,
    pub uptime_weight: f64,
    pub connectivity_weight: f64,
}

impl Default for HealthScoreWeights {
    fn default() -> Self {
        Self {
            response_time_weight: 0.25,
            cpu_usage_weight: 0.20,
            memory_usage_weight: 0.20,
            error_rate_weight: 0.15,
            uptime_weight: 0.10,
            connectivity_weight: 0.10,
        }
    }
}

/// Alert configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertConfig {
    /// Enable alerting
    pub enabled: bool,
    /// Alert channels
    pub channels: Vec<AlertChannel>,
    /// Alert thresholds
    pub thresholds: AlertThresholds,
    /// Alert rate limiting
    pub rate_limiting: AlertRateLimiting,
}

impl Default for AlertConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            channels: vec![AlertChannel::Log, AlertChannel::Metrics],
            thresholds: AlertThresholds::default(),
            rate_limiting: AlertRateLimiting::default(),
        }
    }
}

/// Alert channels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertChannel {
    Log,
    Metrics,
    Webhook,
    Email,
    Slack,
    PagerDuty,
}

/// Alert thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertThresholds {
    pub critical_health_score: f64,
    pub warning_health_score: f64,
    pub critical_response_time_ms: u64,
    pub warning_response_time_ms: u64,
    pub critical_error_rate: f64,
    pub warning_error_rate: f64,
    pub critical_cpu_usage: f64,
    pub warning_cpu_usage: f64,
    pub critical_memory_usage: f64,
    pub warning_memory_usage: f64,
}

impl Default for AlertThresholds {
    fn default() -> Self {
        Self {
            critical_health_score: 0.3,
            warning_health_score: 0.6,
            critical_response_time_ms: 10000,
            warning_response_time_ms: 5000,
            critical_error_rate: 0.1,
            warning_error_rate: 0.05,
            critical_cpu_usage: 0.95,
            warning_cpu_usage: 0.8,
            critical_memory_usage: 0.95,
            warning_memory_usage: 0.85,
        }
    }
}

/// Alert rate limiting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertRateLimiting {
    /// Maximum alerts per minute per agent
    pub max_alerts_per_minute: u32,
    /// Minimum time between duplicate alerts
    pub duplicate_suppression_minutes: u32,
    /// Alert escalation time
    pub escalation_time_minutes: u32,
}

impl Default for AlertRateLimiting {
    fn default() -> Self {
        Self {
            max_alerts_per_minute: 5,
            duplicate_suppression_minutes: 10,
            escalation_time_minutes: 30,
        }
    }
}

/// Comprehensive health record for an agent
#[derive(Debug, Clone)]
pub struct ComprehensiveHealthRecord {
    pub agent_id: String,
    pub swarm_type: SwarmType,
    pub hierarchy_level: HierarchyLevel,
    pub current_health: AgentHealth,
    pub health_history: SmallVec<[TimestampedHealth; 50]>,
    pub health_score: f64,
    pub health_trend: HealthTrend,
    pub predictive_analysis: PredictiveAnalysis,
    pub consecutive_failures: u32,
    pub consecutive_successes: u32,
    pub last_check_time: Instant,
    pub last_successful_check: Option<Instant>,
    pub check_statistics: HealthCheckStatistics,
    pub anomaly_detection: AnomalyDetectionResult,
}

/// Timestamped health information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimestampedHealth {
    pub timestamp: DateTime<Utc>,
    pub health: AgentHealth,
    pub check_duration_ms: u64,
}

/// Health trend analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthTrend {
    pub overall_direction: TrendDirection,
    pub response_time_trend: MetricTrend,
    pub cpu_usage_trend: MetricTrend,
    pub memory_usage_trend: MetricTrend,
    pub error_rate_trend: MetricTrend,
    pub reliability_trend: MetricTrend,
    pub trend_confidence: f64,
    pub trend_duration: Duration,
}

/// Trend direction for individual metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricTrend {
    pub direction: TrendDirection,
    pub slope: f64,
    pub confidence: f64,
    pub volatility: f64,
}

/// Trend direction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrendDirection {
    Improving,
    Stable,
    Degrading,
    Critical,
    Unknown,
}

/// Predictive analysis for health forecasting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictiveAnalysis {
    pub predicted_failure_probability: f64,
    pub time_to_failure_estimate: Option<Duration>,
    pub confidence_score: f64,
    pub contributing_factors: Vec<String>,
    pub recommended_actions: Vec<RecommendedAction>,
    pub prediction_model: PredictionModel,
    pub last_prediction_time: DateTime<Utc>,
}

/// Prediction models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PredictionModel {
    LinearRegression,
    ExponentialSmoothing,
    AnomalyDetection,
    MachineLearning,
    HybridModel,
}

/// Recommended actions for health improvement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecommendedAction {
    ReduceLoad,
    RestartAgent,
    ScaleUp,
    ScaleDown,
    CheckConnectivity,
    InvestigateMemoryLeak,
    InvestigateCpuSpike,
    UpdateConfiguration,
    MaintenanceRequired,
    MonitorClosely,
}

/// Health check statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckStatistics {
    pub total_checks: u64,
    pub successful_checks: u64,
    pub failed_checks: u64,
    pub timeout_checks: u64,
    pub average_check_duration_ms: f64,
    pub min_check_duration_ms: u64,
    pub max_check_duration_ms: u64,
    pub success_rate: f64,
    pub availability: f64,
}

/// Anomaly detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyDetectionResult {
    pub anomalies_detected: Vec<DetectedAnomaly>,
    pub anomaly_score: f64,
    pub baseline_deviation: f64,
    pub detection_confidence: f64,
    pub last_analysis_time: DateTime<Utc>,
}

/// Individual detected anomaly
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedAnomaly {
    pub anomaly_type: AnomalyType,
    pub metric_name: String,
    pub current_value: f64,
    pub expected_value: f64,
    pub deviation_score: f64,
    pub severity: AnomalySeverity,
    pub first_detected: DateTime<Utc>,
    pub description: String,
}

/// Types of anomalies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnomalyType {
    SpikeTrend,
    DropTrend,
    Oscillation,
    Flatline,
    Outlier,
    SeasonalDeviation,
    CorrelationBreak,
}

/// Anomaly severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnomalySeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Health monitoring system
pub struct HealthMonitor {
    config: HealthMonitorConfig,
    health_records: Arc<DashMap<String, ComprehensiveHealthRecord>>,
    swarm_health: Arc<DashMap<SwarmType, SwarmHealth>>,
    global_health: Arc<RwLock<GlobalHealth>>,
    health_checker: Arc<HealthChecker>,
    trend_analyzer: Arc<TrendAnalyzer>,
    predictive_engine: Arc<PredictiveEngine>,
    anomaly_detector: Arc<AnomalyDetector>,
    alert_manager: Arc<AlertManager>,
    recovery_coordinator: Arc<RecoveryCoordinator>,
    metrics_collector: Arc<HealthMetricsCollector>,
    event_bus: Arc<HealthEventBus>,
}

/// Swarm-level health information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmHealth {
    pub swarm_type: SwarmType,
    pub total_agents: usize,
    pub healthy_agents: usize,
    pub degraded_agents: usize,
    pub failed_agents: usize,
    pub average_health_score: f64,
    pub swarm_health_score: f64,
    pub availability: f64,
    pub response_time_p50: u64,
    pub response_time_p95: u64,
    pub response_time_p99: u64,
    pub error_rate: f64,
    pub last_updated: DateTime<Utc>,
}

/// Global system health
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalHealth {
    pub system_health_score: f64,
    pub total_agents: usize,
    pub healthy_agents: usize,
    pub degraded_agents: usize,
    pub failed_agents: usize,
    pub system_availability: f64,
    pub critical_alerts: u32,
    pub warning_alerts: u32,
    pub last_updated: DateTime<Utc>,
    pub uptime_seconds: u64,
    pub start_time: DateTime<Utc>,
}

/// Health checker for executing health checks
pub struct HealthChecker {
    check_strategies: HashMap<String, Box<dyn HealthCheckStrategy>>,
    check_queue: Arc<PriorityQueue<String, i64>>,
    active_checks: Arc<DashMap<String, ActiveHealthCheck>>,
    check_history: Arc<RwLock<Vec<HealthCheckEvent>>>,
}

/// Health check strategy trait
#[async_trait::async_trait]
pub trait HealthCheckStrategy: Send + Sync {
    async fn check_health(&self, agent_id: &str) -> HealthCheckResult;
    fn strategy_name(&self) -> &str;
    fn check_timeout(&self) -> Duration;
}

/// Health check result
#[derive(Debug, Clone)]
pub struct HealthCheckResult {
    pub agent_id: String,
    pub success: bool,
    pub health: Option<AgentHealth>,
    pub error: Option<String>,
    pub check_duration: Duration,
    pub timestamp: DateTime<Utc>,
}

/// Active health check
#[derive(Debug, Clone)]
pub struct ActiveHealthCheck {
    pub check_id: String,
    pub agent_id: String,
    pub start_time: Instant,
    pub timeout: Duration,
    pub strategy: String,
}

/// Health check event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckEvent {
    pub timestamp: DateTime<Utc>,
    pub agent_id: String,
    pub success: bool,
    pub duration_ms: u64,
    pub error: Option<String>,
    pub health_score: f64,
}

/// Trend analyzer for health trend analysis
pub struct TrendAnalyzer {
    analysis_algorithms: Vec<Box<dyn TrendAnalysisAlgorithm>>,
    trend_cache: Arc<DashMap<String, CachedTrendAnalysis>>,
    analysis_history: Arc<RwLock<Vec<TrendAnalysisEvent>>>,
}

/// Trend analysis algorithm trait
pub trait TrendAnalysisAlgorithm: Send + Sync {
    fn analyze_trend(&self, history: &[TimestampedHealth]) -> HealthTrend;
    fn algorithm_name(&self) -> &str;
    fn required_data_points(&self) -> usize;
}

/// Cached trend analysis
#[derive(Debug, Clone)]
pub struct CachedTrendAnalysis {
    pub trend: HealthTrend,
    pub analysis_time: Instant,
    pub validity_duration: Duration,
}

/// Trend analysis event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendAnalysisEvent {
    pub timestamp: DateTime<Utc>,
    pub agent_id: String,
    pub trend_direction: TrendDirection,
    pub confidence: f64,
    pub contributing_metrics: Vec<String>,
}

/// Predictive engine for failure prediction
pub struct PredictiveEngine {
    prediction_models: HashMap<PredictionModel, Box<dyn PredictionAlgorithm>>,
    prediction_cache: Arc<DashMap<String, CachedPrediction>>,
    model_performance: Arc<DashMap<PredictionModel, ModelPerformance>>,
    prediction_history: Arc<RwLock<Vec<PredictionEvent>>>,
}

/// Prediction algorithm trait
#[async_trait::async_trait]
pub trait PredictionAlgorithm: Send + Sync {
    async fn predict(&self, health_record: &ComprehensiveHealthRecord) -> PredictiveAnalysis;
    fn model_name(&self) -> &str;
    fn update_model(&self, feedback: &PredictionFeedback);
}

/// Cached prediction
#[derive(Debug, Clone)]
pub struct CachedPrediction {
    pub prediction: PredictiveAnalysis,
    pub prediction_time: Instant,
    pub validity_duration: Duration,
}

/// Model performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPerformance {
    pub accuracy: f64,
    pub precision: f64,
    pub recall: f64,
    pub f1_score: f64,
    pub false_positive_rate: f64,
    pub false_negative_rate: f64,
    pub total_predictions: u64,
    pub correct_predictions: u64,
}

/// Prediction event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionEvent {
    pub timestamp: DateTime<Utc>,
    pub agent_id: String,
    pub model: PredictionModel,
    pub failure_probability: f64,
    pub time_to_failure: Option<Duration>,
    pub actual_outcome: Option<bool>,
}

/// Prediction feedback for model improvement
#[derive(Debug, Clone)]
pub struct PredictionFeedback {
    pub agent_id: String,
    pub predicted_failure: bool,
    pub actual_failure: bool,
    pub prediction_time: DateTime<Utc>,
    pub failure_time: Option<DateTime<Utc>>,
}

/// Anomaly detector for identifying unusual patterns
pub struct AnomalyDetector {
    detection_algorithms: Vec<Box<dyn AnomalyDetectionAlgorithm>>,
    baseline_models: Arc<DashMap<String, BaselineModel>>,
    anomaly_history: Arc<RwLock<Vec<AnomalyEvent>>>,
    detection_thresholds: Arc<RwLock<AnomalyThresholds>>,
}

/// Anomaly detection algorithm trait
pub trait AnomalyDetectionAlgorithm: Send + Sync {
    fn detect_anomalies(&self, health_record: &ComprehensiveHealthRecord, baseline: &BaselineModel) -> AnomalyDetectionResult;
    fn algorithm_name(&self) -> &str;
    fn update_baseline(&self, baseline: &mut BaselineModel, new_data: &TimestampedHealth);
}

/// Baseline model for anomaly detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineModel {
    pub agent_id: String,
    pub response_time_baseline: StatisticalBaseline,
    pub cpu_usage_baseline: StatisticalBaseline,
    pub memory_usage_baseline: StatisticalBaseline,
    pub error_rate_baseline: StatisticalBaseline,
    pub model_updated: DateTime<Utc>,
    pub data_points_count: u64,
}

/// Statistical baseline for a metric
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalBaseline {
    pub mean: f64,
    pub std_dev: f64,
    pub min: f64,
    pub max: f64,
    pub percentile_95: f64,
    pub percentile_99: f64,
    pub trend_slope: f64,
    pub seasonality_pattern: Vec<f64>,
}

/// Anomaly detection thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyThresholds {
    pub mild_threshold: f64,
    pub moderate_threshold: f64,
    pub severe_threshold: f64,
    pub critical_threshold: f64,
}

/// Anomaly event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyEvent {
    pub timestamp: DateTime<Utc>,
    pub agent_id: String,
    pub anomaly_type: AnomalyType,
    pub severity: AnomalySeverity,
    pub metric_name: String,
    pub deviation_score: f64,
    pub resolved: bool,
    pub resolution_time: Option<DateTime<Utc>>,
}

/// Alert manager for health alerts
pub struct AlertManager {
    config: AlertConfig,
    active_alerts: Arc<DashMap<String, ActiveAlert>>,
    alert_history: Arc<RwLock<Vec<AlertEvent>>>,
    alert_channels: Vec<Box<dyn AlertChannel>>,
    rate_limiter: Arc<AlertRateLimiter>,
    escalation_manager: Arc<EscalationManager>,
}

/// Active alert
#[derive(Debug, Clone)]
pub struct ActiveAlert {
    pub alert_id: String,
    pub agent_id: String,
    pub alert_type: AlertType,
    pub severity: AlertSeverity,
    pub message: String,
    pub created_time: Instant,
    pub last_notification: Option<Instant>,
    pub notification_count: u32,
    pub acknowledged: bool,
    pub escalated: bool,
}

/// Alert types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertType {
    HealthDegradation,
    HighResponseTime,
    HighErrorRate,
    ResourceExhaustion,
    AnomalyDetected,
    PredictedFailure,
    AgentUnreachable,
    SystemWideDegradation,
}

/// Alert severity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
    Emergency,
}

/// Alert event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertEvent {
    pub timestamp: DateTime<Utc>,
    pub alert_id: String,
    pub agent_id: String,
    pub alert_type: AlertType,
    pub severity: AlertSeverity,
    pub message: String,
    pub acknowledged: bool,
    pub resolved: bool,
    pub resolution_time: Option<DateTime<Utc>>,
}

/// Alert channel trait
#[async_trait::async_trait]
pub trait AlertChannel: Send + Sync {
    async fn send_alert(&self, alert: &ActiveAlert) -> Result<(), AlertError>;
    fn channel_name(&self) -> &str;
    fn supports_severity(&self, severity: &AlertSeverity) -> bool;
}

/// Alert error
#[derive(Debug, thiserror::Error)]
pub enum AlertError {
    #[error("Network error: {0}")]
    NetworkError(String),
    #[error("Configuration error: {0}")]
    ConfigurationError(String),
    #[error("Rate limited")]
    RateLimited,
    #[error("Channel unavailable")]
    ChannelUnavailable,
}

/// Alert rate limiter
pub struct AlertRateLimiter {
    rate_limits: Arc<DashMap<String, RateLimit>>,
    config: AlertRateLimiting,
}

/// Rate limit tracking
#[derive(Debug, Clone)]
pub struct RateLimit {
    pub alerts_sent: u32,
    pub window_start: Instant,
    pub last_alert_time: Option<Instant>,
}

/// Escalation manager
pub struct EscalationManager {
    escalation_rules: Vec<EscalationRule>,
    active_escalations: Arc<DashMap<String, EscalationState>>,
}

/// Escalation rule
#[derive(Debug, Clone)]
pub struct EscalationRule {
    pub rule_id: String,
    pub trigger_conditions: Vec<EscalationCondition>,
    pub escalation_delay: Duration,
    pub target_channels: Vec<String>,
    pub repeat_interval: Option<Duration>,
}

/// Escalation condition
#[derive(Debug, Clone)]
pub enum EscalationCondition {
    UnacknowledgedAlert(Duration),
    RepeatedAlerts(u32, Duration),
    SeverityLevel(AlertSeverity),
    AgentType(SwarmType),
}

/// Escalation state
#[derive(Debug, Clone)]
pub struct EscalationState {
    pub alert_id: String,
    pub escalation_level: u32,
    pub last_escalation: Instant,
    pub escalation_count: u32,
}

/// Recovery coordinator
pub struct RecoveryCoordinator {
    recovery_strategies: HashMap<String, Box<dyn RecoveryStrategy>>,
    active_recoveries: Arc<DashMap<String, ActiveRecovery>>,
    recovery_history: Arc<RwLock<Vec<RecoveryEvent>>>,
    dependency_resolver: Arc<DependencyResolver>,
}

/// Recovery strategy trait
#[async_trait::async_trait]
pub trait RecoveryStrategy: Send + Sync {
    async fn execute_recovery(&self, agent_id: &str, health_issue: &HealthIssue) -> RecoveryResult;
    fn strategy_name(&self) -> &str;
    fn can_handle(&self, health_issue: &HealthIssue) -> bool;
    fn estimated_duration(&self) -> Duration;
}

/// Health issue identification
#[derive(Debug, Clone)]
pub struct HealthIssue {
    pub issue_type: HealthIssueType,
    pub severity: IssueSeverity,
    pub affected_metrics: Vec<String>,
    pub probable_causes: Vec<String>,
    pub detection_confidence: f64,
}

/// Health issue types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HealthIssueType {
    HighLatency,
    HighErrorRate,
    MemoryLeak,
    CpuSpike,
    NetworkIssue,
    StorageIssue,
    ConfigurationProblem,
    DependencyFailure,
}

/// Issue severity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IssueSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Recovery result
#[derive(Debug, Clone)]
pub struct RecoveryResult {
    pub success: bool,
    pub recovery_duration: Duration,
    pub actions_taken: Vec<String>,
    pub health_improvement: Option<f64>,
    pub error_message: Option<String>,
}

/// Active recovery
#[derive(Debug, Clone)]
pub struct ActiveRecovery {
    pub recovery_id: String,
    pub agent_id: String,
    pub strategy_name: String,
    pub start_time: Instant,
    pub estimated_completion: Instant,
    pub current_step: String,
    pub progress_percentage: f64,
}

/// Recovery event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryEvent {
    pub timestamp: DateTime<Utc>,
    pub agent_id: String,
    pub strategy_name: String,
    pub issue_type: HealthIssueType,
    pub success: bool,
    pub duration: Duration,
    pub health_improvement: Option<f64>,
}

/// Dependency resolver for recovery coordination
pub struct DependencyResolver {
    agent_dependencies: Arc<DashMap<String, Vec<String>>>,
    dependency_graph: Arc<RwLock<HashMap<String, Vec<String>>>>,
}

/// Health metrics collector
pub struct HealthMetricsCollector {
    metrics: Arc<DashMap<String, HealthMetricValue>>,
    collection_interval: Duration,
    aggregation_windows: Vec<Duration>,
}

/// Health metric value
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthMetricValue {
    pub metric_name: String,
    pub value: f64,
    pub timestamp: DateTime<Utc>,
    pub tags: HashMap<String, String>,
}

/// Health event bus for system-wide health events
pub struct HealthEventBus {
    event_sender: broadcast::Sender<HealthEvent>,
    subscribers: Arc<DashMap<String, HealthEventSubscriber>>,
}

/// Health event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HealthEvent {
    AgentHealthChanged {
        agent_id: String,
        old_status: AgentStatus,
        new_status: AgentStatus,
        health_score: f64,
    },
    SwarmHealthChanged {
        swarm_type: SwarmType,
        health_score: f64,
        availability: f64,
    },
    AlertGenerated {
        alert_id: String,
        agent_id: String,
        severity: AlertSeverity,
    },
    RecoveryInitiated {
        agent_id: String,
        strategy: String,
    },
    RecoveryCompleted {
        agent_id: String,
        success: bool,
    },
    AnomalyDetected {
        agent_id: String,
        anomaly_type: AnomalyType,
        severity: AnomalySeverity,
    },
}

/// Health event subscriber
pub struct HealthEventSubscriber {
    pub subscriber_id: String,
    pub event_filter: HealthEventFilter,
    pub event_handler: Box<dyn HealthEventHandler>,
}

/// Health event filter
#[derive(Debug, Clone)]
pub struct HealthEventFilter {
    pub agent_ids: Option<Vec<String>>,
    pub swarm_types: Option<Vec<SwarmType>>,
    pub event_types: Option<Vec<String>>,
    pub severity_threshold: Option<AlertSeverity>,
}

/// Health event handler trait
#[async_trait::async_trait]
pub trait HealthEventHandler: Send + Sync {
    async fn handle_event(&self, event: &HealthEvent) -> Result<(), Box<dyn std::error::Error>>;
    fn handler_name(&self) -> &str;
}

impl HealthMonitor {
    /// Create new health monitor
    pub async fn new() -> Result<Self, MCPOrchestrationError> {
        let config = HealthMonitorConfig::default();
        
        let health_records = Arc::new(DashMap::new());
        let swarm_health = Arc::new(DashMap::new());
        let global_health = Arc::new(RwLock::new(GlobalHealth {
            system_health_score: 1.0,
            total_agents: 0,
            healthy_agents: 0,
            degraded_agents: 0,
            failed_agents: 0,
            system_availability: 1.0,
            critical_alerts: 0,
            warning_alerts: 0,
            last_updated: Utc::now(),
            uptime_seconds: 0,
            start_time: Utc::now(),
        }));
        
        let health_checker = Arc::new(HealthChecker {
            check_strategies: HashMap::new(), // Initialize with actual strategies
            check_queue: Arc::new(PriorityQueue::new()),
            active_checks: Arc::new(DashMap::new()),
            check_history: Arc::new(RwLock::new(Vec::new())),
        });
        
        let trend_analyzer = Arc::new(TrendAnalyzer {
            analysis_algorithms: vec![], // Initialize with actual algorithms
            trend_cache: Arc::new(DashMap::new()),
            analysis_history: Arc::new(RwLock::new(Vec::new())),
        });
        
        let predictive_engine = Arc::new(PredictiveEngine {
            prediction_models: HashMap::new(), // Initialize with actual models
            prediction_cache: Arc::new(DashMap::new()),
            model_performance: Arc::new(DashMap::new()),
            prediction_history: Arc::new(RwLock::new(Vec::new())),
        });
        
        let anomaly_detector = Arc::new(AnomalyDetector {
            detection_algorithms: vec![], // Initialize with actual algorithms
            baseline_models: Arc::new(DashMap::new()),
            anomaly_history: Arc::new(RwLock::new(Vec::new())),
            detection_thresholds: Arc::new(RwLock::new(AnomalyThresholds {
                mild_threshold: 2.0,
                moderate_threshold: 3.0,
                severe_threshold: 4.0,
                critical_threshold: 5.0,
            })),
        });
        
        let alert_manager = Arc::new(AlertManager {
            config: config.alerting.clone(),
            active_alerts: Arc::new(DashMap::new()),
            alert_history: Arc::new(RwLock::new(Vec::new())),
            alert_channels: vec![], // Initialize with actual channels
            rate_limiter: Arc::new(AlertRateLimiter {
                rate_limits: Arc::new(DashMap::new()),
                config: config.alerting.rate_limiting.clone(),
            }),
            escalation_manager: Arc::new(EscalationManager {
                escalation_rules: vec![], // Initialize with actual rules
                active_escalations: Arc::new(DashMap::new()),
            }),
        });
        
        let recovery_coordinator = Arc::new(RecoveryCoordinator {
            recovery_strategies: HashMap::new(), // Initialize with actual strategies
            active_recoveries: Arc::new(DashMap::new()),
            recovery_history: Arc::new(RwLock::new(Vec::new())),
            dependency_resolver: Arc::new(DependencyResolver {
                agent_dependencies: Arc::new(DashMap::new()),
                dependency_graph: Arc::new(RwLock::new(HashMap::new())),
            }),
        });
        
        let metrics_collector = Arc::new(HealthMetricsCollector {
            metrics: Arc::new(DashMap::new()),
            collection_interval: Duration::from_millis(1000),
            aggregation_windows: vec![
                Duration::from_secs(60),
                Duration::from_secs(300),
                Duration::from_secs(3600),
            ],
        });
        
        let (event_sender, _) = broadcast::channel(1000);
        let event_bus = Arc::new(HealthEventBus {
            event_sender,
            subscribers: Arc::new(DashMap::new()),
        });
        
        Ok(Self {
            config,
            health_records,
            swarm_health,
            global_health,
            health_checker,
            trend_analyzer,
            predictive_engine,
            anomaly_detector,
            alert_manager,
            recovery_coordinator,
            metrics_collector,
            event_bus,
        })
    }
    
    /// Start health monitoring
    #[instrument(skip(self))]
    pub async fn start_monitoring(&self) -> Result<(), MCPOrchestrationError> {
        info!("Starting comprehensive health monitoring system");
        
        // Start health checking
        self.start_health_checking().await?;
        
        // Start trend analysis
        self.start_trend_analysis().await?;
        
        // Start predictive monitoring
        if self.config.predictive_monitoring {
            self.start_predictive_monitoring().await?;
        }
        
        // Start anomaly detection
        self.start_anomaly_detection().await?;
        
        // Start metrics collection
        self.start_metrics_collection().await?;
        
        // Start alert processing
        self.start_alert_processing().await?;
        
        // Start recovery coordination
        if self.config.auto_recovery {
            self.start_recovery_coordination().await?;
        }
        
        info!("Health monitoring system started successfully");
        Ok(())
    }
    
    /// Start health checking background task
    async fn start_health_checking(&self) -> Result<(), MCPOrchestrationError> {
        let health_checker = Arc::clone(&self.health_checker);
        let health_records = Arc::clone(&self.health_records);
        let config = self.config.clone();
        let event_bus = Arc::clone(&self.event_bus);
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_millis(config.check_interval_ms));
            
            loop {
                interval.tick().await;
                
                if let Err(e) = Self::perform_health_checks(
                    &health_checker,
                    &health_records,
                    &config,
                    &event_bus,
                ).await {
                    error!("Health check error: {}", e);
                }
            }
        });
        
        Ok(())
    }
    
    /// Perform health checks on all registered agents
    async fn perform_health_checks(
        health_checker: &HealthChecker,
        health_records: &DashMap<String, ComprehensiveHealthRecord>,
        config: &HealthMonitorConfig,
        event_bus: &HealthEventBus,
    ) -> Result<(), MCPOrchestrationError> {
        debug!("Performing comprehensive health checks");
        
        // Collect agents to check
        let agents_to_check: Vec<String> = health_records.iter()
            .map(|entry| entry.key().clone())
            .collect();
        
        // Process health checks in parallel
        let check_futures: Vec<_> = agents_to_check.into_iter()
            .map(|agent_id| {
                let health_checker = Arc::clone(health_checker);
                let health_records = health_records.clone();
                let config = config.clone();
                let event_bus = Arc::clone(event_bus);
                
                tokio::spawn(async move {
                    Self::check_agent_health(agent_id, health_checker, health_records, config, event_bus).await
                })
            })
            .collect();
        
        // Wait for all checks to complete
        futures::future::join_all(check_futures).await;
        
        Ok(())
    }
    
    /// Check health for a specific agent
    async fn check_agent_health(
        agent_id: String,
        health_checker: Arc<HealthChecker>,
        health_records: DashMap<String, ComprehensiveHealthRecord>,
        config: HealthMonitorConfig,
        event_bus: Arc<HealthEventBus>,
    ) -> Result<(), MCPOrchestrationError> {
        let start_time = Instant::now();
        
        // Execute health check
        let check_result = Self::execute_health_check(&agent_id, &health_checker).await;
        let check_duration = start_time.elapsed();
        
        // Update health record
        if let Some(mut health_record) = health_records.get_mut(&agent_id) {
            let old_status = health_record.current_health.status.clone();
            
            match check_result {
                Ok(health) => {
                    // Update current health
                    health_record.current_health = health.clone();
                    health_record.last_check_time = Instant::now();
                    health_record.last_successful_check = Some(Instant::now());
                    health_record.consecutive_failures = 0;
                    health_record.consecutive_successes += 1;
                    
                    // Add to history
                    let timestamped_health = TimestampedHealth {
                        timestamp: Utc::now(),
                        health: health.clone(),
                        check_duration_ms: check_duration.as_millis() as u64,
                    };
                    
                    health_record.health_history.push(timestamped_health);
                    if health_record.health_history.len() > 50 {
                        health_record.health_history.remove(0);
                    }
                    
                    // Update statistics
                    health_record.check_statistics.total_checks += 1;
                    health_record.check_statistics.successful_checks += 1;
                    Self::update_check_statistics(&mut health_record.check_statistics, check_duration, true);
                    
                    // Calculate health score
                    health_record.health_score = Self::calculate_health_score(&health, &config.score_weights);
                    
                    // Emit health change event if status changed
                    if old_status != health.status {
                        let _ = event_bus.event_sender.send(HealthEvent::AgentHealthChanged {
                            agent_id: agent_id.clone(),
                            old_status,
                            new_status: health.status,
                            health_score: health_record.health_score,
                        });
                    }
                }
                Err(error) => {
                    // Handle check failure
                    health_record.consecutive_failures += 1;
                    health_record.consecutive_successes = 0;
                    health_record.last_check_time = Instant::now();
                    
                    // Update statistics
                    health_record.check_statistics.total_checks += 1;
                    health_record.check_statistics.failed_checks += 1;
                    Self::update_check_statistics(&mut health_record.check_statistics, check_duration, false);
                    
                    // Mark as unhealthy if threshold exceeded
                    if health_record.consecutive_failures >= config.failure_threshold {
                        let old_status = health_record.current_health.status.clone();
                        health_record.current_health.status = AgentStatus::Failed;
                        health_record.health_score = 0.0;
                        
                        // Emit health change event
                        let _ = event_bus.event_sender.send(HealthEvent::AgentHealthChanged {
                            agent_id: agent_id.clone(),
                            old_status,
                            new_status: AgentStatus::Failed,
                            health_score: 0.0,
                        });
                    }
                    
                    warn!("Health check failed for agent {}: {}", agent_id, error);
                }
            }
        }
        
        Ok(())
    }
    
    /// Execute health check for an agent
    async fn execute_health_check(
        agent_id: &str,
        health_checker: &HealthChecker,
    ) -> Result<AgentHealth, String> {
        // In a real implementation, this would use the appropriate strategy
        // For now, simulate a health check
        
        tokio::time::sleep(Duration::from_millis(10)).await;
        
        // Simulate occasional failures
        if rand::random::<f64>() < 0.05 {
            return Err("Simulated health check failure".to_string());
        }
        
        Ok(AgentHealth {
            status: AgentStatus::Running,
            cpu_usage: rand::random::<f64>() * 0.8,
            memory_usage: rand::random::<f64>() * 0.7,
            uptime_seconds: 3600,
            last_error: None,
            response_time_us: (rand::random::<u64>() % 2000) + 100,
        })
    }
    
    /// Update health check statistics
    fn update_check_statistics(
        stats: &mut HealthCheckStatistics,
        duration: Duration,
        success: bool,
    ) {
        let duration_ms = duration.as_millis() as u64;
        
        // Update min/max
        if stats.total_checks == 0 {
            stats.min_check_duration_ms = duration_ms;
            stats.max_check_duration_ms = duration_ms;
        } else {
            stats.min_check_duration_ms = stats.min_check_duration_ms.min(duration_ms);
            stats.max_check_duration_ms = stats.max_check_duration_ms.max(duration_ms);
        }
        
        // Update average
        stats.average_check_duration_ms = 
            (stats.average_check_duration_ms * (stats.total_checks - 1) as f64 + duration_ms as f64) 
            / stats.total_checks as f64;
        
        // Update success rate
        stats.success_rate = stats.successful_checks as f64 / stats.total_checks as f64;
        
        // Update availability (simplified calculation)
        stats.availability = stats.success_rate;
    }
    
    /// Calculate health score based on metrics
    fn calculate_health_score(health: &AgentHealth, weights: &HealthScoreWeights) -> f64 {
        let response_time_score = 1.0 - (health.response_time_us as f64 / 10000.0).min(1.0);
        let cpu_score = 1.0 - health.cpu_usage;
        let memory_score = 1.0 - health.memory_usage;
        let uptime_score = (health.uptime_seconds as f64 / 86400.0).min(1.0); // Max 1 day
        let error_score = if health.last_error.is_some() { 0.0 } else { 1.0 };
        let connectivity_score = if matches!(health.status, AgentStatus::Running) { 1.0 } else { 0.0 };
        
        (response_time_score * weights.response_time_weight +
         cpu_score * weights.cpu_usage_weight +
         memory_score * weights.memory_usage_weight +
         error_score * weights.error_rate_weight +
         uptime_score * weights.uptime_weight +
         connectivity_score * weights.connectivity_weight).max(0.0).min(1.0)
    }
    
    /// Start trend analysis
    async fn start_trend_analysis(&self) -> Result<(), MCPOrchestrationError> {
        let trend_analyzer = Arc::clone(&self.trend_analyzer);
        let health_records = Arc::clone(&self.health_records);
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_millis(30000)); // 30 seconds
            
            loop {
                interval.tick().await;
                
                if let Err(e) = Self::analyze_health_trends(&trend_analyzer, &health_records).await {
                    error!("Trend analysis error: {}", e);
                }
            }
        });
        
        Ok(())
    }
    
    /// Analyze health trends for all agents
    async fn analyze_health_trends(
        trend_analyzer: &TrendAnalyzer,
        health_records: &DashMap<String, ComprehensiveHealthRecord>,
    ) -> Result<(), MCPOrchestrationError> {
        debug!("Analyzing health trends");
        
        for mut health_record in health_records.iter_mut() {
            if health_record.health_history.len() >= 5 {
                // Analyze trend using the first available algorithm
                let trend = if let Some(algorithm) = trend_analyzer.analysis_algorithms.first() {
                    algorithm.analyze_trend(&health_record.health_history)
                } else {
                    // Default trend analysis
                    Self::default_trend_analysis(&health_record.health_history)
                };
                
                health_record.health_trend = trend;
            }
        }
        
        Ok(())
    }
    
    /// Default trend analysis implementation
    fn default_trend_analysis(history: &[TimestampedHealth]) -> HealthTrend {
        if history.len() < 2 {
            return HealthTrend {
                overall_direction: TrendDirection::Unknown,
                response_time_trend: MetricTrend {
                    direction: TrendDirection::Unknown,
                    slope: 0.0,
                    confidence: 0.0,
                    volatility: 0.0,
                },
                cpu_usage_trend: MetricTrend {
                    direction: TrendDirection::Unknown,
                    slope: 0.0,
                    confidence: 0.0,
                    volatility: 0.0,
                },
                memory_usage_trend: MetricTrend {
                    direction: TrendDirection::Unknown,
                    slope: 0.0,
                    confidence: 0.0,
                    volatility: 0.0,
                },
                error_rate_trend: MetricTrend {
                    direction: TrendDirection::Unknown,
                    slope: 0.0,
                    confidence: 0.0,
                    volatility: 0.0,
                },
                reliability_trend: MetricTrend {
                    direction: TrendDirection::Unknown,
                    slope: 0.0,
                    confidence: 0.0,
                    volatility: 0.0,
                },
                trend_confidence: 0.0,
                trend_duration: Duration::from_secs(0),
            };
        }
        
        // Simple linear trend analysis
        let first = &history[0];
        let last = &history[history.len() - 1];
        
        let response_time_change = last.health.response_time_us as f64 - first.health.response_time_us as f64;
        let cpu_change = last.health.cpu_usage - first.health.cpu_usage;
        let memory_change = last.health.memory_usage - first.health.memory_usage;
        
        let response_time_direction = if response_time_change > 100.0 {
            TrendDirection::Degrading
        } else if response_time_change < -100.0 {
            TrendDirection::Improving
        } else {
            TrendDirection::Stable
        };
        
        let cpu_direction = if cpu_change > 0.1 {
            TrendDirection::Degrading
        } else if cpu_change < -0.1 {
            TrendDirection::Improving
        } else {
            TrendDirection::Stable
        };
        
        let memory_direction = if memory_change > 0.1 {
            TrendDirection::Degrading
        } else if memory_change < -0.1 {
            TrendDirection::Improving
        } else {
            TrendDirection::Stable
        };
        
        // Determine overall direction
        let degrading_count = [&response_time_direction, &cpu_direction, &memory_direction]
            .iter()
            .filter(|&&ref d| matches!(d, TrendDirection::Degrading))
            .count();
        
        let overall_direction = if degrading_count >= 2 {
            TrendDirection::Degrading
        } else if degrading_count == 0 {
            TrendDirection::Improving
        } else {
            TrendDirection::Stable
        };
        
        HealthTrend {
            overall_direction,
            response_time_trend: MetricTrend {
                direction: response_time_direction,
                slope: response_time_change,
                confidence: 0.7,
                volatility: 0.2,
            },
            cpu_usage_trend: MetricTrend {
                direction: cpu_direction,
                slope: cpu_change,
                confidence: 0.7,
                volatility: 0.2,
            },
            memory_usage_trend: MetricTrend {
                direction: memory_direction,
                slope: memory_change,
                confidence: 0.7,
                volatility: 0.2,
            },
            error_rate_trend: MetricTrend {
                direction: TrendDirection::Stable,
                slope: 0.0,
                confidence: 0.5,
                volatility: 0.1,
            },
            reliability_trend: MetricTrend {
                direction: TrendDirection::Stable,
                slope: 0.0,
                confidence: 0.5,
                volatility: 0.1,
            },
            trend_confidence: 0.7,
            trend_duration: Duration::from_secs(300), // 5 minutes
        }
    }
    
    /// Start predictive monitoring
    async fn start_predictive_monitoring(&self) -> Result<(), MCPOrchestrationError> {
        let predictive_engine = Arc::clone(&self.predictive_engine);
        let health_records = Arc::clone(&self.health_records);
        let event_bus = Arc::clone(&self.event_bus);
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_millis(60000)); // 1 minute
            
            loop {
                interval.tick().await;
                
                if let Err(e) = Self::generate_predictions(&predictive_engine, &health_records, &event_bus).await {
                    error!("Predictive monitoring error: {}", e);
                }
            }
        });
        
        Ok(())
    }
    
    /// Generate predictions for all agents
    async fn generate_predictions(
        predictive_engine: &PredictiveEngine,
        health_records: &DashMap<String, ComprehensiveHealthRecord>,
        event_bus: &HealthEventBus,
    ) -> Result<(), MCPOrchestrationError> {
        debug!("Generating health predictions");
        
        for mut health_record in health_records.iter_mut() {
            if health_record.health_history.len() >= 10 {
                // Generate prediction using available models
                let prediction = Self::default_prediction(&health_record);
                
                // Check for high failure probability
                if prediction.predicted_failure_probability > 0.7 {
                    // Emit prediction event
                    let _ = event_bus.event_sender.send(HealthEvent::AnomalyDetected {
                        agent_id: health_record.agent_id.clone(),
                        anomaly_type: AnomalyType::SpikeTrend,
                        severity: AnomalySeverity::High,
                    });
                }
                
                health_record.predictive_analysis = prediction;
            }
        }
        
        Ok(())
    }
    
    /// Default prediction implementation
    fn default_prediction(health_record: &ComprehensiveHealthRecord) -> PredictiveAnalysis {
        // Simple prediction based on trends
        let failure_probability = match health_record.health_trend.overall_direction {
            TrendDirection::Critical => 0.9,
            TrendDirection::Degrading => 0.6,
            TrendDirection::Stable => 0.1,
            TrendDirection::Improving => 0.05,
            TrendDirection::Unknown => 0.2,
        };
        
        let time_to_failure = if failure_probability > 0.5 {
            Some(Duration::from_hours(2))
        } else {
            None
        };
        
        let recommended_actions = if failure_probability > 0.7 {
            vec![RecommendedAction::ReduceLoad, RecommendedAction::MonitorClosely]
        } else if failure_probability > 0.4 {
            vec![RecommendedAction::MonitorClosely]
        } else {
            vec![]
        };
        
        PredictiveAnalysis {
            predicted_failure_probability: failure_probability,
            time_to_failure_estimate: time_to_failure,
            confidence_score: 0.7,
            contributing_factors: vec!["Historical trend analysis".to_string()],
            recommended_actions,
            prediction_model: PredictionModel::LinearRegression,
            last_prediction_time: Utc::now(),
        }
    }
    
    /// Start anomaly detection
    async fn start_anomaly_detection(&self) -> Result<(), MCPOrchestrationError> {
        let anomaly_detector = Arc::clone(&self.anomaly_detector);
        let health_records = Arc::clone(&self.health_records);
        let event_bus = Arc::clone(&self.event_bus);
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_millis(15000)); // 15 seconds
            
            loop {
                interval.tick().await;
                
                if let Err(e) = Self::detect_anomalies(&anomaly_detector, &health_records, &event_bus).await {
                    error!("Anomaly detection error: {}", e);
                }
            }
        });
        
        Ok(())
    }
    
    /// Detect anomalies in agent health
    async fn detect_anomalies(
        anomaly_detector: &AnomalyDetector,
        health_records: &DashMap<String, ComprehensiveHealthRecord>,
        event_bus: &HealthEventBus,
    ) -> Result<(), MCPOrchestrationError> {
        debug!("Detecting health anomalies");
        
        for mut health_record in health_records.iter_mut() {
            // Get or create baseline model
            let baseline = anomaly_detector.baseline_models
                .entry(health_record.agent_id.clone())
                .or_insert_with(|| Self::create_baseline_model(&health_record.agent_id));
            
            // Detect anomalies
            let anomaly_result = Self::detect_agent_anomalies(&health_record, &baseline);
            
            // Emit events for detected anomalies
            for anomaly in &anomaly_result.anomalies_detected {
                if matches!(anomaly.severity, AnomalySeverity::High | AnomalySeverity::Critical) {
                    let _ = event_bus.event_sender.send(HealthEvent::AnomalyDetected {
                        agent_id: health_record.agent_id.clone(),
                        anomaly_type: anomaly.anomaly_type.clone(),
                        severity: anomaly.severity.clone(),
                    });
                }
            }
            
            health_record.anomaly_detection = anomaly_result;
        }
        
        Ok(())
    }
    
    /// Create baseline model for an agent
    fn create_baseline_model(agent_id: &str) -> BaselineModel {
        BaselineModel {
            agent_id: agent_id.to_string(),
            response_time_baseline: StatisticalBaseline {
                mean: 1000.0,
                std_dev: 200.0,
                min: 100.0,
                max: 5000.0,
                percentile_95: 2000.0,
                percentile_99: 3000.0,
                trend_slope: 0.0,
                seasonality_pattern: vec![],
            },
            cpu_usage_baseline: StatisticalBaseline {
                mean: 0.5,
                std_dev: 0.2,
                min: 0.1,
                max: 0.9,
                percentile_95: 0.8,
                percentile_99: 0.9,
                trend_slope: 0.0,
                seasonality_pattern: vec![],
            },
            memory_usage_baseline: StatisticalBaseline {
                mean: 0.4,
                std_dev: 0.15,
                min: 0.2,
                max: 0.8,
                percentile_95: 0.7,
                percentile_99: 0.8,
                trend_slope: 0.0,
                seasonality_pattern: vec![],
            },
            error_rate_baseline: StatisticalBaseline {
                mean: 0.01,
                std_dev: 0.005,
                min: 0.0,
                max: 0.05,
                percentile_95: 0.02,
                percentile_99: 0.03,
                trend_slope: 0.0,
                seasonality_pattern: vec![],
            },
            model_updated: Utc::now(),
            data_points_count: 0,
        }
    }
    
    /// Detect anomalies for a specific agent
    fn detect_agent_anomalies(
        health_record: &ComprehensiveHealthRecord,
        baseline: &BaselineModel,
    ) -> AnomalyDetectionResult {
        let mut anomalies = Vec::new();
        
        let current_health = &health_record.current_health;
        
        // Check response time anomalies
        let response_time_ms = current_health.response_time_us as f64 / 1000.0;
        if response_time_ms > baseline.response_time_baseline.mean + 3.0 * baseline.response_time_baseline.std_dev {
            anomalies.push(DetectedAnomaly {
                anomaly_type: AnomalyType::SpikeTrend,
                metric_name: "response_time".to_string(),
                current_value: response_time_ms,
                expected_value: baseline.response_time_baseline.mean,
                deviation_score: (response_time_ms - baseline.response_time_baseline.mean) / baseline.response_time_baseline.std_dev,
                severity: if response_time_ms > baseline.response_time_baseline.percentile_99 {
                    AnomalySeverity::Critical
                } else {
                    AnomalySeverity::High
                },
                first_detected: Utc::now(),
                description: format!("Response time spike detected: {:.1}ms vs baseline {:.1}ms", 
                                   response_time_ms, baseline.response_time_baseline.mean),
            });
        }
        
        // Check CPU usage anomalies
        if current_health.cpu_usage > baseline.cpu_usage_baseline.mean + 3.0 * baseline.cpu_usage_baseline.std_dev {
            anomalies.push(DetectedAnomaly {
                anomaly_type: AnomalyType::SpikeTrend,
                metric_name: "cpu_usage".to_string(),
                current_value: current_health.cpu_usage,
                expected_value: baseline.cpu_usage_baseline.mean,
                deviation_score: (current_health.cpu_usage - baseline.cpu_usage_baseline.mean) / baseline.cpu_usage_baseline.std_dev,
                severity: if current_health.cpu_usage > 0.95 {
                    AnomalySeverity::Critical
                } else if current_health.cpu_usage > 0.85 {
                    AnomalySeverity::High
                } else {
                    AnomalySeverity::Medium
                },
                first_detected: Utc::now(),
                description: format!("CPU usage spike detected: {:.1}% vs baseline {:.1}%", 
                                   current_health.cpu_usage * 100.0, baseline.cpu_usage_baseline.mean * 100.0),
            });
        }
        
        // Check memory usage anomalies
        if current_health.memory_usage > baseline.memory_usage_baseline.mean + 3.0 * baseline.memory_usage_baseline.std_dev {
            anomalies.push(DetectedAnomaly {
                anomaly_type: AnomalyType::SpikeTrend,
                metric_name: "memory_usage".to_string(),
                current_value: current_health.memory_usage,
                expected_value: baseline.memory_usage_baseline.mean,
                deviation_score: (current_health.memory_usage - baseline.memory_usage_baseline.mean) / baseline.memory_usage_baseline.std_dev,
                severity: if current_health.memory_usage > 0.95 {
                    AnomalySeverity::Critical
                } else if current_health.memory_usage > 0.85 {
                    AnomalySeverity::High
                } else {
                    AnomalySeverity::Medium
                },
                first_detected: Utc::now(),
                description: format!("Memory usage spike detected: {:.1}% vs baseline {:.1}%", 
                                   current_health.memory_usage * 100.0, baseline.memory_usage_baseline.mean * 100.0),
            });
        }
        
        let anomaly_score = anomalies.iter()
            .map(|a| a.deviation_score)
            .fold(0.0, |acc, score| acc + score.abs()) / anomalies.len().max(1) as f64;
        
        AnomalyDetectionResult {
            anomalies_detected: anomalies,
            anomaly_score,
            baseline_deviation: anomaly_score,
            detection_confidence: 0.8,
            last_analysis_time: Utc::now(),
        }
    }
    
    /// Start metrics collection
    async fn start_metrics_collection(&self) -> Result<(), MCPOrchestrationError> {
        let metrics_collector = Arc::clone(&self.metrics_collector);
        let health_records = Arc::clone(&self.health_records);
        let swarm_health = Arc::clone(&self.swarm_health);
        let global_health = Arc::clone(&self.global_health);
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(metrics_collector.collection_interval);
            
            loop {
                interval.tick().await;
                
                if let Err(e) = Self::collect_health_metrics(
                    &metrics_collector,
                    &health_records,
                    &swarm_health,
                    &global_health,
                ).await {
                    error!("Health metrics collection error: {}", e);
                }
            }
        });
        
        Ok(())
    }
    
    /// Collect health metrics
    async fn collect_health_metrics(
        metrics_collector: &HealthMetricsCollector,
        health_records: &DashMap<String, ComprehensiveHealthRecord>,
        swarm_health: &DashMap<SwarmType, SwarmHealth>,
        global_health: &RwLock<GlobalHealth>,
    ) -> Result<(), MCPOrchestrationError> {
        debug!("Collecting health metrics");
        
        // Collect individual agent metrics
        for health_record in health_records.iter() {
            let agent_id = health_record.key();
            let record = health_record.value();
            
            // Response time metric
            metrics_collector.metrics.insert(
                format!("{}_response_time", agent_id),
                HealthMetricValue {
                    metric_name: "response_time_us".to_string(),
                    value: record.current_health.response_time_us as f64,
                    timestamp: Utc::now(),
                    tags: HashMap::from([
                        ("agent_id".to_string(), agent_id.clone()),
                        ("swarm_type".to_string(), format!("{:?}", record.swarm_type)),
                    ]),
                },
            );
            
            // Health score metric
            metrics_collector.metrics.insert(
                format!("{}_health_score", agent_id),
                HealthMetricValue {
                    metric_name: "health_score".to_string(),
                    value: record.health_score,
                    timestamp: Utc::now(),
                    tags: HashMap::from([
                        ("agent_id".to_string(), agent_id.clone()),
                        ("swarm_type".to_string(), format!("{:?}", record.swarm_type)),
                    ]),
                },
            );
            
            // Export to Prometheus-style metrics
            gauge!("agent_health_score", record.health_score, "agent_id" => agent_id.clone());
            gauge!("agent_response_time_us", record.current_health.response_time_us as f64, "agent_id" => agent_id.clone());
            gauge!("agent_cpu_usage", record.current_health.cpu_usage, "agent_id" => agent_id.clone());
            gauge!("agent_memory_usage", record.current_health.memory_usage, "agent_id" => agent_id.clone());
        }
        
        // Update global health metrics
        let mut global = global_health.write().await;
        global.total_agents = health_records.len();
        global.healthy_agents = health_records.iter()
            .filter(|r| r.health_score > 0.8)
            .count();
        global.degraded_agents = health_records.iter()
            .filter(|r| r.health_score > 0.3 && r.health_score <= 0.8)
            .count();
        global.failed_agents = health_records.iter()
            .filter(|r| r.health_score <= 0.3)
            .count();
        
        global.system_health_score = if global.total_agents > 0 {
            health_records.iter()
                .map(|r| r.health_score)
                .sum::<f64>() / global.total_agents as f64
        } else {
            1.0
        };
        
        global.system_availability = global.healthy_agents as f64 / global.total_agents.max(1) as f64;
        global.last_updated = Utc::now();
        
        // Export global metrics
        gauge!("system_health_score", global.system_health_score);
        gauge!("system_availability", global.system_availability);
        gauge!("total_agents", global.total_agents as f64);
        gauge!("healthy_agents", global.healthy_agents as f64);
        gauge!("failed_agents", global.failed_agents as f64);
        
        Ok(())
    }
    
    /// Start alert processing
    async fn start_alert_processing(&self) -> Result<(), MCPOrchestrationError> {
        let alert_manager = Arc::clone(&self.alert_manager);
        let health_records = Arc::clone(&self.health_records);
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_millis(5000)); // 5 seconds
            
            loop {
                interval.tick().await;
                
                if let Err(e) = Self::process_health_alerts(&alert_manager, &health_records).await {
                    error!("Alert processing error: {}", e);
                }
            }
        });
        
        Ok(())
    }
    
    /// Process health alerts
    async fn process_health_alerts(
        alert_manager: &AlertManager,
        health_records: &DashMap<String, ComprehensiveHealthRecord>,
    ) -> Result<(), MCPOrchestrationError> {
        debug!("Processing health alerts");
        
        for health_record in health_records.iter() {
            let agent_id = health_record.key();
            let record = health_record.value();
            
            // Check for alert conditions
            let mut alert_triggered = false;
            let mut alert_type = AlertType::HealthDegradation;
            let mut severity = AlertSeverity::Info;
            let mut message = String::new();
            
            if record.health_score <= alert_manager.config.thresholds.critical_health_score {
                alert_triggered = true;
                alert_type = AlertType::HealthDegradation;
                severity = AlertSeverity::Critical;
                message = format!("Critical health degradation: score {:.2}", record.health_score);
            } else if record.health_score <= alert_manager.config.thresholds.warning_health_score {
                alert_triggered = true;
                alert_type = AlertType::HealthDegradation;
                severity = AlertSeverity::Warning;
                message = format!("Health degradation warning: score {:.2}", record.health_score);
            }
            
            // Check response time
            if record.current_health.response_time_us >= alert_manager.config.thresholds.critical_response_time_ms * 1000 {
                alert_triggered = true;
                alert_type = AlertType::HighResponseTime;
                severity = AlertSeverity::Critical;
                message = format!("Critical response time: {}μs", record.current_health.response_time_us);
            } else if record.current_health.response_time_us >= alert_manager.config.thresholds.warning_response_time_ms * 1000 {
                alert_triggered = true;
                alert_type = AlertType::HighResponseTime;
                severity = AlertSeverity::Warning;
                message = format!("High response time: {}μs", record.current_health.response_time_us);
            }
            
            // Create alert if triggered and not rate limited
            if alert_triggered {
                let alert_key = format!("{}_{:?}", agent_id, alert_type);
                
                if Self::should_send_alert(&alert_manager.rate_limiter, &alert_key) {
                    let alert = ActiveAlert {
                        alert_id: Uuid::new_v4().to_string(),
                        agent_id: agent_id.clone(),
                        alert_type,
                        severity,
                        message,
                        created_time: Instant::now(),
                        last_notification: None,
                        notification_count: 0,
                        acknowledged: false,
                        escalated: false,
                    };
                    
                    alert_manager.active_alerts.insert(alert.alert_id.clone(), alert);
                }
            }
        }
        
        Ok(())
    }
    
    /// Check if alert should be sent based on rate limiting
    fn should_send_alert(rate_limiter: &AlertRateLimiter, alert_key: &str) -> bool {
        let now = Instant::now();
        
        let mut rate_limit = rate_limiter.rate_limits
            .entry(alert_key.to_string())
            .or_insert_with(|| RateLimit {
                alerts_sent: 0,
                window_start: now,
                last_alert_time: None,
            });
        
        // Reset window if expired
        if now.duration_since(rate_limit.window_start) > Duration::from_secs(60) {
            rate_limit.alerts_sent = 0;
            rate_limit.window_start = now;
        }
        
        // Check rate limit
        if rate_limit.alerts_sent >= rate_limiter.config.max_alerts_per_minute {
            return false;
        }
        
        // Check duplicate suppression
        if let Some(last_alert) = rate_limit.last_alert_time {
            if now.duration_since(last_alert) < Duration::from_secs(rate_limiter.config.duplicate_suppression_minutes as u64 * 60) {
                return false;
            }
        }
        
        // Update rate limit
        rate_limit.alerts_sent += 1;
        rate_limit.last_alert_time = Some(now);
        
        true
    }
    
    /// Start recovery coordination
    async fn start_recovery_coordination(&self) -> Result<(), MCPOrchestrationError> {
        let recovery_coordinator = Arc::clone(&self.recovery_coordinator);
        let health_records = Arc::clone(&self.health_records);
        let event_bus = Arc::clone(&self.event_bus);
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_millis(10000)); // 10 seconds
            
            loop {
                interval.tick().await;
                
                if let Err(e) = Self::coordinate_recovery(&recovery_coordinator, &health_records, &event_bus).await {
                    error!("Recovery coordination error: {}", e);
                }
            }
        });
        
        Ok(())
    }
    
    /// Coordinate recovery actions
    async fn coordinate_recovery(
        recovery_coordinator: &RecoveryCoordinator,
        health_records: &DashMap<String, ComprehensiveHealthRecord>,
        event_bus: &HealthEventBus,
    ) -> Result<(), MCPOrchestrationError> {
        debug!("Coordinating recovery actions");
        
        for health_record in health_records.iter() {
            let agent_id = health_record.key();
            let record = health_record.value();
            
            // Check if recovery is needed
            if record.health_score < 0.3 && record.consecutive_failures >= 3 {
                // Check if recovery is already in progress
                if !recovery_coordinator.active_recoveries.contains_key(agent_id) {
                    // Identify health issue
                    let health_issue = Self::identify_health_issue(&record);
                    
                    // Find appropriate recovery strategy
                    if let Some((strategy_name, strategy)) = recovery_coordinator.recovery_strategies
                        .iter()
                        .find(|(_, strategy)| strategy.can_handle(&health_issue)) {
                        
                        // Initiate recovery
                        info!("Initiating recovery for agent {} using strategy {}", agent_id, strategy_name);
                        
                        let recovery = ActiveRecovery {
                            recovery_id: Uuid::new_v4().to_string(),
                            agent_id: agent_id.clone(),
                            strategy_name: strategy_name.clone(),
                            start_time: Instant::now(),
                            estimated_completion: Instant::now() + strategy.estimated_duration(),
                            current_step: "Starting recovery".to_string(),
                            progress_percentage: 0.0,
                        };
                        
                        recovery_coordinator.active_recoveries.insert(agent_id.clone(), recovery);
                        
                        // Emit recovery event
                        let _ = event_bus.event_sender.send(HealthEvent::RecoveryInitiated {
                            agent_id: agent_id.clone(),
                            strategy: strategy_name.clone(),
                        });
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Identify health issue from health record
    fn identify_health_issue(health_record: &ComprehensiveHealthRecord) -> HealthIssue {
        let mut affected_metrics = Vec::new();
        let mut probable_causes = Vec::new();
        let mut issue_type = HealthIssueType::ConfigurationProblem;
        let mut severity = IssueSeverity::Medium;
        
        // Analyze health metrics to identify issues
        if health_record.current_health.response_time_us > 5000 {
            affected_metrics.push("response_time".to_string());
            probable_causes.push("Network latency or processing delays".to_string());
            issue_type = HealthIssueType::HighLatency;
        }
        
        if health_record.current_health.cpu_usage > 0.9 {
            affected_metrics.push("cpu_usage".to_string());
            probable_causes.push("High computational load or inefficient processing".to_string());
            issue_type = HealthIssueType::CpuSpike;
            severity = IssueSeverity::High;
        }
        
        if health_record.current_health.memory_usage > 0.9 {
            affected_metrics.push("memory_usage".to_string());
            probable_causes.push("Memory leak or excessive memory consumption".to_string());
            issue_type = HealthIssueType::MemoryLeak;
            severity = IssueSeverity::High;
        }
        
        if health_record.consecutive_failures >= 5 {
            severity = IssueSeverity::Critical;
        }
        
        HealthIssue {
            issue_type,
            severity,
            affected_metrics,
            probable_causes,
            detection_confidence: 0.8,
        }
    }
    
    /// Register an agent for health monitoring
    pub async fn register_agent(&self, agent_id: String, swarm_type: SwarmType, hierarchy_level: HierarchyLevel) {
        let health_record = ComprehensiveHealthRecord {
            agent_id: agent_id.clone(),
            swarm_type,
            hierarchy_level,
            current_health: AgentHealth {
                status: AgentStatus::Starting,
                cpu_usage: 0.0,
                memory_usage: 0.0,
                uptime_seconds: 0,
                last_error: None,
                response_time_us: 0,
            },
            health_history: SmallVec::new(),
            health_score: 1.0,
            health_trend: HealthTrend {
                overall_direction: TrendDirection::Unknown,
                response_time_trend: MetricTrend {
                    direction: TrendDirection::Unknown,
                    slope: 0.0,
                    confidence: 0.0,
                    volatility: 0.0,
                },
                cpu_usage_trend: MetricTrend {
                    direction: TrendDirection::Unknown,
                    slope: 0.0,
                    confidence: 0.0,
                    volatility: 0.0,
                },
                memory_usage_trend: MetricTrend {
                    direction: TrendDirection::Unknown,
                    slope: 0.0,
                    confidence: 0.0,
                    volatility: 0.0,
                },
                error_rate_trend: MetricTrend {
                    direction: TrendDirection::Unknown,
                    slope: 0.0,
                    confidence: 0.0,
                    volatility: 0.0,
                },
                reliability_trend: MetricTrend {
                    direction: TrendDirection::Unknown,
                    slope: 0.0,
                    confidence: 0.0,
                    volatility: 0.0,
                },
                trend_confidence: 0.0,
                trend_duration: Duration::from_secs(0),
            },
            predictive_analysis: PredictiveAnalysis {
                predicted_failure_probability: 0.0,
                time_to_failure_estimate: None,
                confidence_score: 0.0,
                contributing_factors: vec![],
                recommended_actions: vec![],
                prediction_model: PredictionModel::LinearRegression,
                last_prediction_time: Utc::now(),
            },
            consecutive_failures: 0,
            consecutive_successes: 0,
            last_check_time: Instant::now(),
            last_successful_check: None,
            check_statistics: HealthCheckStatistics {
                total_checks: 0,
                successful_checks: 0,
                failed_checks: 0,
                timeout_checks: 0,
                average_check_duration_ms: 0.0,
                min_check_duration_ms: 0,
                max_check_duration_ms: 0,
                success_rate: 1.0,
                availability: 1.0,
            },
            anomaly_detection: AnomalyDetectionResult {
                anomalies_detected: vec![],
                anomaly_score: 0.0,
                baseline_deviation: 0.0,
                detection_confidence: 0.0,
                last_analysis_time: Utc::now(),
            },
        };
        
        self.health_records.insert(agent_id, health_record);
    }
    
    /// Get health statistics for the monitoring system
    pub async fn get_health_statistics(&self) -> HealthStatistics {
        let global_health = self.global_health.read().await;
        
        HealthStatistics {
            total_agents: global_health.total_agents,
            healthy_agents: global_health.healthy_agents,
            degraded_agents: global_health.degraded_agents,
            failed_agents: global_health.failed_agents,
            system_health_score: global_health.system_health_score,
            system_availability: global_health.system_availability,
            critical_alerts: global_health.critical_alerts,
            warning_alerts: global_health.warning_alerts,
            uptime_seconds: global_health.uptime_seconds,
        }
    }
}

/// Health monitoring statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthStatistics {
    pub total_agents: usize,
    pub healthy_agents: usize,
    pub degraded_agents: usize,
    pub failed_agents: usize,
    pub system_health_score: f64,
    pub system_availability: f64,
    pub critical_alerts: u32,
    pub warning_alerts: u32,
    pub uptime_seconds: u64,
}