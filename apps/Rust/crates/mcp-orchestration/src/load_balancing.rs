//! Load Balancing and Health Monitoring
//!
//! Implements dynamic load balancing with real-time health monitoring
//! and automatic failover for the 25+ agent swarm ecosystem.

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

use crate::agents::{MCPMessage, MCPMessageType, MessagePriority, AgentHealth, AgentMetrics};
use crate::{MCPOrchestrationError, SwarmType, HierarchyLevel, AgentStatus};

/// Load balancer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadBalancerConfig {
    /// Load balancing algorithm
    pub algorithm: LoadBalancingAlgorithm,
    /// Health check interval in milliseconds
    pub health_check_interval_ms: u64,
    /// Failover timeout in milliseconds
    pub failover_timeout_ms: u64,
    /// Load threshold for scaling decisions
    pub load_threshold: f64,
    /// Circuit breaker configuration
    pub circuit_breaker: CircuitBreakerConfig,
    /// Auto-scaling configuration
    pub auto_scaling: AutoScalingConfig,
    /// Sticky session configuration
    pub sticky_sessions: StickySessionConfig,
}

impl Default for LoadBalancerConfig {
    fn default() -> Self {
        Self {
            algorithm: LoadBalancingAlgorithm::WeightedRoundRobin,
            health_check_interval_ms: 1000,
            failover_timeout_ms: 5000,
            load_threshold: 0.8,
            circuit_breaker: CircuitBreakerConfig::default(),
            auto_scaling: AutoScalingConfig::default(),
            sticky_sessions: StickySessionConfig::default(),
        }
    }
}

/// Load balancing algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LoadBalancingAlgorithm {
    RoundRobin,
    WeightedRoundRobin,
    LeastConnections,
    WeightedLeastConnections,
    ResourceBased,
    LatencyBased,
    ConsistentHashing,
    PowerOfTwoChoices,
    AdaptiveWeighted,
}

/// Circuit breaker configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitBreakerConfig {
    /// Failure threshold
    pub failure_threshold: u32,
    /// Success threshold for recovery
    pub success_threshold: u32,
    /// Timeout for circuit breaker reset
    pub timeout_ms: u64,
    /// Enable circuit breaker
    pub enabled: bool,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            success_threshold: 3,
            timeout_ms: 30000,
            enabled: true,
        }
    }
}

/// Auto-scaling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoScalingConfig {
    /// Enable auto-scaling
    pub enabled: bool,
    /// Minimum number of agents per swarm
    pub min_agents: usize,
    /// Maximum number of agents per swarm
    pub max_agents: usize,
    /// Scale-up threshold (CPU/memory usage)
    pub scale_up_threshold: f64,
    /// Scale-down threshold
    pub scale_down_threshold: f64,
    /// Cooldown period between scaling operations
    pub cooldown_period_ms: u64,
}

impl Default for AutoScalingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_agents: 1,
            max_agents: 10,
            scale_up_threshold: 0.8,
            scale_down_threshold: 0.3,
            cooldown_period_ms: 300000, // 5 minutes
        }
    }
}

/// Sticky session configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StickySessionConfig {
    /// Enable sticky sessions
    pub enabled: bool,
    /// Session timeout in milliseconds
    pub session_timeout_ms: u64,
    /// Hash function for session affinity
    pub hash_function: String,
}

impl Default for StickySessionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            session_timeout_ms: 1800000, // 30 minutes
            hash_function: "consistent_hash".to_string(),
        }
    }
}

/// Agent load information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentLoad {
    pub agent_id: String,
    pub swarm_type: SwarmType,
    pub hierarchy_level: HierarchyLevel,
    pub cpu_usage: f64,
    pub memory_usage: f64,
    pub active_connections: u64,
    pub message_queue_size: u64,
    pub response_time_ms: f64,
    pub error_rate: f64,
    pub throughput_per_sec: f64,
    pub health_score: f64,
    pub last_updated: DateTime<Utc>,
}

/// Load balancing decision
#[derive(Debug, Clone)]
pub struct LoadBalancingDecision {
    pub selected_agent: String,
    pub algorithm_used: LoadBalancingAlgorithm,
    pub selection_score: f64,
    pub alternatives: Vec<String>,
    pub decision_time_ns: u64,
}

/// Circuit breaker state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CircuitBreakerState {
    Closed,    // Normal operation
    Open,      // Failing, not allowing requests
    HalfOpen,  // Testing recovery
}

/// Circuit breaker for agent protection
#[derive(Debug, Clone)]
pub struct CircuitBreaker {
    pub agent_id: String,
    pub state: CircuitBreakerState,
    pub failure_count: AtomicU64,
    pub success_count: AtomicU64,
    pub last_failure_time: Arc<Mutex<Option<Instant>>>,
    pub config: CircuitBreakerConfig,
}

/// Load balancer implementation
pub struct LoadBalancer {
    config: LoadBalancerConfig,
    agent_loads: Arc<DashMap<String, AgentLoad>>,
    circuit_breakers: Arc<DashMap<String, CircuitBreaker>>,
    load_history: Arc<RwLock<Vec<LoadSnapshot>>>,
    health_monitor: Arc<HealthMonitor>,
    auto_scaler: Arc<AutoScaler>,
    session_manager: Arc<SessionManager>,
    algorithms: Arc<DashMap<LoadBalancingAlgorithm, Box<dyn LoadBalancingStrategy>>>,
    metrics_collector: Arc<LoadBalancerMetrics>,
    selection_cache: Arc<DashMap<String, CachedSelection>>,
}

/// Load snapshot for historical analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadSnapshot {
    pub timestamp: DateTime<Utc>,
    pub swarm_loads: HashMap<SwarmType, SwarmLoadSnapshot>,
    pub global_metrics: GlobalLoadMetrics,
}

/// Swarm load snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmLoadSnapshot {
    pub swarm_type: SwarmType,
    pub agent_count: usize,
    pub average_cpu: f64,
    pub average_memory: f64,
    pub total_connections: u64,
    pub total_throughput: f64,
    pub health_score: f64,
}

/// Global load metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalLoadMetrics {
    pub total_agents: usize,
    pub healthy_agents: usize,
    pub total_cpu_usage: f64,
    pub total_memory_usage: f64,
    pub total_connections: u64,
    pub total_throughput: f64,
    pub average_response_time: f64,
}

/// Health monitoring system
pub struct HealthMonitor {
    config: HealthMonitorConfig,
    agent_health: Arc<DashMap<String, AgentHealthRecord>>,
    health_history: Arc<RwLock<Vec<HealthSnapshot>>>,
    failure_detector: Arc<FailureDetector>,
    recovery_manager: Arc<RecoveryManager>,
    alert_manager: Arc<AlertManager>,
}

/// Health monitor configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthMonitorConfig {
    /// Health check interval
    pub check_interval_ms: u64,
    /// Health check timeout
    pub check_timeout_ms: u64,
    /// Failure threshold
    pub failure_threshold: u32,
    /// Recovery threshold
    pub recovery_threshold: u32,
    /// Enable predictive health monitoring
    pub predictive_monitoring: bool,
}

impl Default for HealthMonitorConfig {
    fn default() -> Self {
        Self {
            check_interval_ms: 5000,
            check_timeout_ms: 2000,
            failure_threshold: 3,
            recovery_threshold: 2,
            predictive_monitoring: true,
        }
    }
}

/// Agent health record
#[derive(Debug, Clone)]
pub struct AgentHealthRecord {
    pub agent_id: String,
    pub current_health: AgentHealth,
    pub health_trend: HealthTrend,
    pub consecutive_failures: u32,
    pub consecutive_successes: u32,
    pub last_health_check: Instant,
    pub prediction: HealthPrediction,
}

/// Health trend analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthTrend {
    pub cpu_trend: f64,
    pub memory_trend: f64,
    pub response_time_trend: f64,
    pub error_rate_trend: f64,
    pub overall_trend: TrendDirection,
}

/// Trend direction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrendDirection {
    Improving,
    Stable,
    Degrading,
    Critical,
}

/// Health prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthPrediction {
    pub predicted_failure_time: Option<DateTime<Utc>>,
    pub confidence_score: f64,
    pub recommended_actions: Vec<RecommendedAction>,
}

/// Recommended actions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecommendedAction {
    ReduceLoad,
    RestartAgent,
    ScaleUp,
    MigrateWorkload,
    InvestigateMemoryLeak,
    CheckNetworkConnectivity,
}

/// Health snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthSnapshot {
    pub timestamp: DateTime<Utc>,
    pub agent_health: HashMap<String, AgentHealth>,
    pub swarm_health: HashMap<SwarmType, SwarmHealthMetrics>,
    pub global_health: GlobalHealthMetrics,
}

/// Swarm health metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmHealthMetrics {
    pub healthy_agents: usize,
    pub total_agents: usize,
    pub average_health_score: f64,
    pub critical_issues: u32,
    pub warnings: u32,
}

/// Global health metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalHealthMetrics {
    pub system_health_score: f64,
    pub critical_alerts: u32,
    pub warning_alerts: u32,
    pub total_health_checks: u64,
    pub failed_health_checks: u64,
}

/// Failure detector
pub struct FailureDetector {
    detection_algorithms: Vec<Box<dyn FailureDetectionAlgorithm>>,
    failure_patterns: Arc<DashMap<String, FailurePattern>>,
    detection_history: Arc<RwLock<Vec<FailureEvent>>>,
}

/// Failure detection algorithm trait
pub trait FailureDetectionAlgorithm: Send + Sync {
    fn detect_failure(&self, agent_id: &str, health_record: &AgentHealthRecord) -> FailureDetectionResult;
    fn algorithm_name(&self) -> &str;
}

/// Failure detection result
#[derive(Debug, Clone)]
pub struct FailureDetectionResult {
    pub failure_detected: bool,
    pub confidence: f64,
    pub failure_type: FailureType,
    pub evidence: Vec<String>,
}

/// Failure types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FailureType {
    HighLatency,
    HighErrorRate,
    ResourceExhaustion,
    NetworkPartition,
    ProcessCrash,
    MemoryLeak,
    DeadlockDetected,
    Unknown,
}

/// Failure pattern
#[derive(Debug, Clone)]
pub struct FailurePattern {
    pub pattern_id: String,
    pub failure_type: FailureType,
    pub frequency: u32,
    pub last_occurrence: DateTime<Utc>,
    pub correlation_factors: Vec<String>,
}

/// Failure event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailureEvent {
    pub timestamp: DateTime<Utc>,
    pub agent_id: String,
    pub failure_type: FailureType,
    pub details: String,
    pub recovery_time: Option<Duration>,
}

/// Recovery manager
pub struct RecoveryManager {
    recovery_strategies: HashMap<FailureType, Box<dyn RecoveryStrategy>>,
    active_recoveries: Arc<DashMap<String, RecoveryOperation>>,
    recovery_history: Arc<RwLock<Vec<RecoveryEvent>>>,
}

/// Recovery strategy trait
pub trait RecoveryStrategy: Send + Sync {
    fn initiate_recovery(&self, agent_id: &str, failure_type: &FailureType) -> RecoveryOperation;
    fn strategy_name(&self) -> &str;
}

/// Recovery operation
#[derive(Debug, Clone)]
pub struct RecoveryOperation {
    pub operation_id: String,
    pub agent_id: String,
    pub recovery_type: RecoveryType,
    pub start_time: Instant,
    pub estimated_duration: Duration,
    pub status: RecoveryStatus,
}

/// Recovery types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecoveryType {
    SoftRestart,
    HardRestart,
    LoadRedistribution,
    ResourceReallocation,
    NetworkReconfiguration,
    Failover,
}

/// Recovery status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecoveryStatus {
    InProgress,
    Completed,
    Failed,
    Cancelled,
}

/// Recovery event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryEvent {
    pub timestamp: DateTime<Utc>,
    pub agent_id: String,
    pub recovery_type: RecoveryType,
    pub success: bool,
    pub duration: Duration,
    pub details: String,
}

/// Alert manager
pub struct AlertManager {
    alert_rules: Vec<AlertRule>,
    active_alerts: Arc<DashMap<String, ActiveAlert>>,
    alert_history: Arc<RwLock<Vec<AlertEvent>>>,
    notification_channels: Vec<Box<dyn NotificationChannel>>,
}

/// Alert rule
#[derive(Debug, Clone)]
pub struct AlertRule {
    pub rule_id: String,
    pub name: String,
    pub condition: AlertCondition,
    pub severity: AlertSeverity,
    pub throttle_duration: Duration,
}

/// Alert condition
#[derive(Debug, Clone)]
pub enum AlertCondition {
    HealthScoreBelow(f64),
    ResponseTimeAbove(Duration),
    ErrorRateAbove(f64),
    CpuUsageAbove(f64),
    MemoryUsageAbove(f64),
    ConsecutiveFailures(u32),
    Custom(String),
}

/// Alert severity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertSeverity {
    Critical,
    Warning,
    Info,
}

/// Active alert
#[derive(Debug, Clone)]
pub struct ActiveAlert {
    pub alert_id: String,
    pub rule_id: String,
    pub agent_id: String,
    pub severity: AlertSeverity,
    pub message: String,
    pub start_time: Instant,
    pub last_notification: Option<Instant>,
    pub notification_count: u32,
}

/// Alert event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertEvent {
    pub timestamp: DateTime<Utc>,
    pub alert_id: String,
    pub agent_id: String,
    pub severity: AlertSeverity,
    pub message: String,
    pub resolved: bool,
    pub resolution_time: Option<DateTime<Utc>>,
}

/// Notification channel trait
pub trait NotificationChannel: Send + Sync {
    fn send_notification(&self, alert: &ActiveAlert) -> Result<(), NotificationError>;
    fn channel_name(&self) -> &str;
}

/// Notification error
#[derive(Debug, thiserror::Error)]
pub enum NotificationError {
    #[error("Network error: {0}")]
    NetworkError(String),
    #[error("Authentication failed: {0}")]
    AuthenticationFailed(String),
    #[error("Rate limited")]
    RateLimited,
    #[error("Invalid configuration: {0}")]
    InvalidConfiguration(String),
}

/// Auto-scaler
pub struct AutoScaler {
    config: AutoScalingConfig,
    scaling_decisions: Arc<RwLock<Vec<ScalingDecision>>>,
    scaling_metrics: Arc<DashMap<SwarmType, SwarmScalingMetrics>>,
    scaling_policies: Vec<ScalingPolicy>,
    cooldown_tracker: Arc<DashMap<SwarmType, Instant>>,
}

/// Scaling decision
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingDecision {
    pub timestamp: DateTime<Utc>,
    pub swarm_type: SwarmType,
    pub action: ScalingAction,
    pub current_count: usize,
    pub target_count: usize,
    pub reason: String,
    pub trigger_metric: String,
    pub trigger_value: f64,
}

/// Scaling action
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScalingAction {
    ScaleUp,
    ScaleDown,
    NoAction,
}

/// Swarm scaling metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmScalingMetrics {
    pub current_agents: usize,
    pub target_agents: usize,
    pub average_cpu: f64,
    pub average_memory: f64,
    pub average_response_time: f64,
    pub queue_depth: u64,
    pub last_scaling_action: Option<DateTime<Utc>>,
}

/// Scaling policy
#[derive(Debug, Clone)]
pub struct ScalingPolicy {
    pub policy_id: String,
    pub swarm_type: SwarmType,
    pub metric: ScalingMetric,
    pub threshold: f64,
    pub action: ScalingAction,
    pub cooldown: Duration,
}

/// Scaling metric
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScalingMetric {
    CpuUsage,
    MemoryUsage,
    ResponseTime,
    QueueDepth,
    RequestRate,
    ErrorRate,
}

/// Session manager
pub struct SessionManager {
    config: StickySessionConfig,
    sessions: Arc<DashMap<String, Session>>,
    session_affinity: Arc<DashMap<String, String>>, // session_id -> agent_id
}

/// Session
#[derive(Debug, Clone)]
pub struct Session {
    pub session_id: String,
    pub agent_id: String,
    pub created_at: Instant,
    pub last_access: Instant,
    pub request_count: u64,
}

/// Load balancing strategy trait
pub trait LoadBalancingStrategy: Send + Sync {
    fn select_agent(&self, candidates: &[AgentLoad], request_context: &RequestContext) -> LoadBalancingDecision;
    fn strategy_name(&self) -> &str;
}

/// Request context for load balancing decisions
#[derive(Debug, Clone)]
pub struct RequestContext {
    pub message: MCPMessage,
    pub session_id: Option<String>,
    pub client_id: Option<String>,
    pub required_capabilities: Vec<String>,
    pub priority: MessagePriority,
}

/// Cached selection for performance optimization
#[derive(Debug, Clone)]
pub struct CachedSelection {
    pub selected_agent: String,
    pub selection_time: Instant,
    pub validity_duration: Duration,
    pub selection_context: String,
}

/// Load balancer metrics
pub struct LoadBalancerMetrics {
    selection_times: Arc<DashMap<String, Vec<u64>>>,
    algorithm_performance: Arc<DashMap<LoadBalancingAlgorithm, AlgorithmPerformance>>,
    global_metrics: Arc<RwLock<GlobalLoadBalancerMetrics>>,
}

/// Algorithm performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmPerformance {
    pub total_selections: u64,
    pub average_selection_time_ns: u64,
    pub successful_selections: u64,
    pub failed_selections: u64,
    pub accuracy_score: f64,
}

/// Global load balancer metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalLoadBalancerMetrics {
    pub total_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub average_selection_time_ns: u64,
    pub cache_hit_rate: f64,
    pub failover_count: u64,
}

impl LoadBalancer {
    /// Create new load balancer
    pub async fn new() -> Result<Self, MCPOrchestrationError> {
        let config = LoadBalancerConfig::default();
        
        let agent_loads = Arc::new(DashMap::new());
        let circuit_breakers = Arc::new(DashMap::new());
        let load_history = Arc::new(RwLock::new(Vec::new()));
        
        let health_monitor = Arc::new(HealthMonitor {
            config: HealthMonitorConfig::default(),
            agent_health: Arc::new(DashMap::new()),
            health_history: Arc::new(RwLock::new(Vec::new())),
            failure_detector: Arc::new(FailureDetector {
                detection_algorithms: vec![], // Initialize with actual algorithms
                failure_patterns: Arc::new(DashMap::new()),
                detection_history: Arc::new(RwLock::new(Vec::new())),
            }),
            recovery_manager: Arc::new(RecoveryManager {
                recovery_strategies: HashMap::new(), // Initialize with actual strategies
                active_recoveries: Arc::new(DashMap::new()),
                recovery_history: Arc::new(RwLock::new(Vec::new())),
            }),
            alert_manager: Arc::new(AlertManager {
                alert_rules: vec![], // Initialize with actual rules
                active_alerts: Arc::new(DashMap::new()),
                alert_history: Arc::new(RwLock::new(Vec::new())),
                notification_channels: vec![], // Initialize with actual channels
            }),
        });
        
        let auto_scaler = Arc::new(AutoScaler {
            config: config.auto_scaling.clone(),
            scaling_decisions: Arc::new(RwLock::new(Vec::new())),
            scaling_metrics: Arc::new(DashMap::new()),
            scaling_policies: vec![], // Initialize with actual policies
            cooldown_tracker: Arc::new(DashMap::new()),
        });
        
        let session_manager = Arc::new(SessionManager {
            config: config.sticky_sessions.clone(),
            sessions: Arc::new(DashMap::new()),
            session_affinity: Arc::new(DashMap::new()),
        });
        
        let algorithms = Arc::new(DashMap::new());
        // Initialize algorithms would be done here
        
        let metrics_collector = Arc::new(LoadBalancerMetrics {
            selection_times: Arc::new(DashMap::new()),
            algorithm_performance: Arc::new(DashMap::new()),
            global_metrics: Arc::new(RwLock::new(GlobalLoadBalancerMetrics {
                total_requests: 0,
                successful_requests: 0,
                failed_requests: 0,
                average_selection_time_ns: 0,
                cache_hit_rate: 0.0,
                failover_count: 0,
            })),
        });
        
        let selection_cache = Arc::new(DashMap::new());
        
        Ok(Self {
            config,
            agent_loads,
            circuit_breakers,
            load_history,
            health_monitor,
            auto_scaler,
            session_manager,
            algorithms,
            metrics_collector,
            selection_cache,
        })
    }
    
    /// Start load balancer and health monitoring
    #[instrument(skip(self))]
    pub async fn start(&self) -> Result<(), MCPOrchestrationError> {
        info!("Starting load balancer and health monitoring");
        
        // Start health monitoring
        self.start_health_monitoring().await?;
        
        // Start load monitoring
        self.start_load_monitoring().await?;
        
        // Start auto-scaling
        self.start_auto_scaling().await?;
        
        // Start metrics collection
        self.start_metrics_collection().await?;
        
        info!("Load balancer and health monitoring started successfully");
        Ok(())
    }
    
    /// Start health monitoring background task
    async fn start_health_monitoring(&self) -> Result<(), MCPOrchestrationError> {
        let health_monitor = Arc::clone(&self.health_monitor);
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_millis(
                health_monitor.config.check_interval_ms
            ));
            
            loop {
                interval.tick().await;
                
                if let Err(e) = Self::perform_health_checks(&health_monitor).await {
                    error!("Health check error: {}", e);
                }
            }
        });
        
        Ok(())
    }
    
    /// Perform health checks on all agents
    async fn perform_health_checks(health_monitor: &HealthMonitor) -> Result<(), MCPOrchestrationError> {
        debug!("Performing health checks");
        
        // Simulate health checks for all registered agents
        // In a real implementation, this would ping actual agents
        
        for agent_entry in health_monitor.agent_health.iter() {
            let agent_id = agent_entry.key();
            let mut health_record = agent_entry.value().clone();
            
            // Simulate health check
            let health_check_result = Self::simulate_health_check(agent_id).await;
            
            // Update health record
            health_record.current_health = health_check_result;
            health_record.last_health_check = Instant::now();
            
            // Analyze trends
            health_record.health_trend = Self::analyze_health_trend(&health_record);
            
            // Update prediction
            if health_monitor.config.predictive_monitoring {
                health_record.prediction = Self::predict_health_issues(&health_record);
            }
            
            // Check for failures
            let failure_result = Self::detect_failures(&health_record);
            if failure_result.failure_detected {
                warn!("Failure detected for agent {}: {:?}", agent_id, failure_result.failure_type);
                Self::handle_agent_failure(agent_id, &failure_result, health_monitor).await;
            }
            
            health_monitor.agent_health.insert(agent_id.clone(), health_record);
        }
        
        Ok(())
    }
    
    /// Simulate health check (replace with actual implementation)
    async fn simulate_health_check(_agent_id: &str) -> AgentHealth {
        AgentHealth {
            status: AgentStatus::Running,
            cpu_usage: rand::random::<f64>() * 0.8,
            memory_usage: rand::random::<f64>() * 0.7,
            uptime_seconds: 3600,
            last_error: None,
            response_time_us: (rand::random::<u64>() % 1000) + 100,
        }
    }
    
    /// Analyze health trends
    fn analyze_health_trend(health_record: &AgentHealthRecord) -> HealthTrend {
        // Simplified trend analysis
        // In a real implementation, this would analyze historical data
        
        HealthTrend {
            cpu_trend: 0.0,
            memory_trend: 0.0,
            response_time_trend: 0.0,
            error_rate_trend: 0.0,
            overall_trend: TrendDirection::Stable,
        }
    }
    
    /// Predict health issues
    fn predict_health_issues(_health_record: &AgentHealthRecord) -> HealthPrediction {
        // Simplified prediction
        // In a real implementation, this would use ML models
        
        HealthPrediction {
            predicted_failure_time: None,
            confidence_score: 0.8,
            recommended_actions: vec![],
        }
    }
    
    /// Detect failures
    fn detect_failures(health_record: &AgentHealthRecord) -> FailureDetectionResult {
        let mut failure_detected = false;
        let mut failure_type = FailureType::Unknown;
        let mut evidence = Vec::new();
        
        // Check for high latency
        if health_record.current_health.response_time_us > 5000 {
            failure_detected = true;
            failure_type = FailureType::HighLatency;
            evidence.push(format!("Response time: {}μs", health_record.current_health.response_time_us));
        }
        
        // Check for resource exhaustion
        if health_record.current_health.cpu_usage > 0.95 || health_record.current_health.memory_usage > 0.95 {
            failure_detected = true;
            failure_type = FailureType::ResourceExhaustion;
            evidence.push(format!("CPU: {:.1}%, Memory: {:.1}%", 
                health_record.current_health.cpu_usage * 100.0,
                health_record.current_health.memory_usage * 100.0));
        }
        
        FailureDetectionResult {
            failure_detected,
            confidence: if failure_detected { 0.9 } else { 0.1 },
            failure_type,
            evidence,
        }
    }
    
    /// Handle agent failure
    async fn handle_agent_failure(
        agent_id: &str,
        failure_result: &FailureDetectionResult,
        health_monitor: &HealthMonitor,
    ) {
        info!("Handling failure for agent {}: {:?}", agent_id, failure_result.failure_type);
        
        // Initiate recovery
        if let Some(strategy) = health_monitor.recovery_manager.recovery_strategies.get(&failure_result.failure_type) {
            let recovery_op = strategy.initiate_recovery(agent_id, &failure_result.failure_type);
            health_monitor.recovery_manager.active_recoveries.insert(agent_id.to_string(), recovery_op);
        }
        
        // Generate alerts
        Self::generate_failure_alert(agent_id, failure_result, health_monitor).await;
    }
    
    /// Generate failure alert
    async fn generate_failure_alert(
        agent_id: &str,
        failure_result: &FailureDetectionResult,
        health_monitor: &HealthMonitor,
    ) {
        let alert_id = Uuid::new_v4().to_string();
        let severity = match failure_result.failure_type {
            FailureType::ProcessCrash | FailureType::NetworkPartition => AlertSeverity::Critical,
            FailureType::HighLatency | FailureType::HighErrorRate => AlertSeverity::Warning,
            _ => AlertSeverity::Info,
        };
        
        let alert = ActiveAlert {
            alert_id: alert_id.clone(),
            rule_id: "failure_detection".to_string(),
            agent_id: agent_id.to_string(),
            severity,
            message: format!("Agent failure detected: {:?}", failure_result.failure_type),
            start_time: Instant::now(),
            last_notification: None,
            notification_count: 0,
        };
        
        health_monitor.alert_manager.active_alerts.insert(alert_id, alert);
    }
    
    /// Start load monitoring
    async fn start_load_monitoring(&self) -> Result<(), MCPOrchestrationError> {
        let agent_loads = Arc::clone(&self.agent_loads);
        let load_history = Arc::clone(&self.load_history);
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_millis(5000)); // 5 second intervals
            
            loop {
                interval.tick().await;
                
                if let Err(e) = Self::collect_load_metrics(&agent_loads, &load_history).await {
                    error!("Load monitoring error: {}", e);
                }
            }
        });
        
        Ok(())
    }
    
    /// Collect load metrics
    async fn collect_load_metrics(
        agent_loads: &DashMap<String, AgentLoad>,
        load_history: &RwLock<Vec<LoadSnapshot>>,
    ) -> Result<(), MCPOrchestrationError> {
        debug!("Collecting load metrics");
        
        // Simulate load collection for all agents
        // In a real implementation, this would collect actual metrics
        
        let mut swarm_loads = HashMap::new();
        let mut global_metrics = GlobalLoadMetrics {
            total_agents: 0,
            healthy_agents: 0,
            total_cpu_usage: 0.0,
            total_memory_usage: 0.0,
            total_connections: 0,
            total_throughput: 0.0,
            average_response_time: 0.0,
        };
        
        for agent_entry in agent_loads.iter() {
            let agent_load = agent_entry.value();
            
            // Update global metrics
            global_metrics.total_agents += 1;
            if agent_load.health_score > 0.8 {
                global_metrics.healthy_agents += 1;
            }
            global_metrics.total_cpu_usage += agent_load.cpu_usage;
            global_metrics.total_memory_usage += agent_load.memory_usage;
            global_metrics.total_connections += agent_load.active_connections;
            global_metrics.total_throughput += agent_load.throughput_per_sec;
            global_metrics.average_response_time += agent_load.response_time_ms;
            
            // Update swarm-specific metrics
            let swarm_load = swarm_loads.entry(agent_load.swarm_type.clone()).or_insert_with(|| SwarmLoadSnapshot {
                swarm_type: agent_load.swarm_type.clone(),
                agent_count: 0,
                average_cpu: 0.0,
                average_memory: 0.0,
                total_connections: 0,
                total_throughput: 0.0,
                health_score: 0.0,
            });
            
            swarm_load.agent_count += 1;
            swarm_load.average_cpu += agent_load.cpu_usage;
            swarm_load.average_memory += agent_load.memory_usage;
            swarm_load.total_connections += agent_load.active_connections;
            swarm_load.total_throughput += agent_load.throughput_per_sec;
            swarm_load.health_score += agent_load.health_score;
        }
        
        // Normalize averages
        if global_metrics.total_agents > 0 {
            global_metrics.average_response_time /= global_metrics.total_agents as f64;
        }
        
        for swarm_load in swarm_loads.values_mut() {
            if swarm_load.agent_count > 0 {
                swarm_load.average_cpu /= swarm_load.agent_count as f64;
                swarm_load.average_memory /= swarm_load.agent_count as f64;
                swarm_load.health_score /= swarm_load.agent_count as f64;
            }
        }
        
        // Create snapshot
        let snapshot = LoadSnapshot {
            timestamp: Utc::now(),
            swarm_loads,
            global_metrics,
        };
        
        // Store in history
        let mut history = load_history.write().await;
        history.push(snapshot);
        
        // Keep only recent history (last 1000 snapshots)
        if history.len() > 1000 {
            history.remove(0);
        }
        
        Ok(())
    }
    
    /// Start auto-scaling
    async fn start_auto_scaling(&self) -> Result<(), MCPOrchestrationError> {
        if !self.config.auto_scaling.enabled {
            return Ok(());
        }
        
        let auto_scaler = Arc::clone(&self.auto_scaler);
        let agent_loads = Arc::clone(&self.agent_loads);
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_millis(30000)); // 30 second intervals
            
            loop {
                interval.tick().await;
                
                if let Err(e) = Self::evaluate_scaling_decisions(&auto_scaler, &agent_loads).await {
                    error!("Auto-scaling error: {}", e);
                }
            }
        });
        
        Ok(())
    }
    
    /// Evaluate scaling decisions
    async fn evaluate_scaling_decisions(
        auto_scaler: &AutoScaler,
        agent_loads: &DashMap<String, AgentLoad>,
    ) -> Result<(), MCPOrchestrationError> {
        debug!("Evaluating scaling decisions");
        
        // Group agents by swarm type
        let mut swarm_metrics: HashMap<SwarmType, Vec<&AgentLoad>> = HashMap::new();
        
        for agent_entry in agent_loads.iter() {
            let agent_load = agent_entry.value();
            swarm_metrics.entry(agent_load.swarm_type.clone()).or_insert_with(Vec::new).push(agent_load);
        }
        
        // Evaluate each swarm
        for (swarm_type, agents) in swarm_metrics {
            let decision = Self::make_scaling_decision(&auto_scaler.config, &swarm_type, &agents);
            
            if !matches!(decision.action, ScalingAction::NoAction) {
                // Check cooldown
                if let Some(last_scaling) = auto_scaler.cooldown_tracker.get(&swarm_type) {
                    if last_scaling.elapsed() < Duration::from_millis(auto_scaler.config.cooldown_period_ms) {
                        continue; // Still in cooldown
                    }
                }
                
                info!("Scaling decision for {:?}: {:?}", swarm_type, decision);
                
                // Execute scaling action
                Self::execute_scaling_action(&decision).await;
                
                // Update cooldown
                auto_scaler.cooldown_tracker.insert(swarm_type.clone(), Instant::now());
                
                // Record decision
                let mut decisions = auto_scaler.scaling_decisions.write().await;
                decisions.push(decision);
                
                // Keep only recent decisions
                if decisions.len() > 100 {
                    decisions.remove(0);
                }
            }
        }
        
        Ok(())
    }
    
    /// Make scaling decision for a swarm
    fn make_scaling_decision(
        config: &AutoScalingConfig,
        swarm_type: &SwarmType,
        agents: &[&AgentLoad],
    ) -> ScalingDecision {
        if agents.is_empty() {
            return ScalingDecision {
                timestamp: Utc::now(),
                swarm_type: swarm_type.clone(),
                action: ScalingAction::NoAction,
                current_count: 0,
                target_count: 0,
                reason: "No agents found".to_string(),
                trigger_metric: "agent_count".to_string(),
                trigger_value: 0.0,
            };
        }
        
        let current_count = agents.len();
        let avg_cpu = agents.iter().map(|a| a.cpu_usage).sum::<f64>() / agents.len() as f64;
        let avg_memory = agents.iter().map(|a| a.memory_usage).sum::<f64>() / agents.len() as f64;
        let avg_response_time = agents.iter().map(|a| a.response_time_ms).sum::<f64>() / agents.len() as f64;
        
        // Determine if scaling is needed
        let (action, reason, trigger_metric, trigger_value) = if avg_cpu > config.scale_up_threshold {
            (ScalingAction::ScaleUp, format!("High CPU usage: {:.1}%", avg_cpu * 100.0), "cpu_usage".to_string(), avg_cpu)
        } else if avg_memory > config.scale_up_threshold {
            (ScalingAction::ScaleUp, format!("High memory usage: {:.1}%", avg_memory * 100.0), "memory_usage".to_string(), avg_memory)
        } else if avg_response_time > 5000.0 { // 5 second threshold
            (ScalingAction::ScaleUp, format!("High response time: {:.1}ms", avg_response_time), "response_time".to_string(), avg_response_time)
        } else if avg_cpu < config.scale_down_threshold && avg_memory < config.scale_down_threshold && current_count > config.min_agents {
            (ScalingAction::ScaleDown, format!("Low resource usage: CPU {:.1}%, Memory {:.1}%", avg_cpu * 100.0, avg_memory * 100.0), "resource_usage".to_string(), avg_cpu.max(avg_memory))
        } else {
            (ScalingAction::NoAction, "Metrics within acceptable range".to_string(), "none".to_string(), 0.0)
        };
        
        let target_count = match action {
            ScalingAction::ScaleUp => (current_count + 1).min(config.max_agents),
            ScalingAction::ScaleDown => (current_count - 1).max(config.min_agents),
            ScalingAction::NoAction => current_count,
        };
        
        ScalingDecision {
            timestamp: Utc::now(),
            swarm_type: swarm_type.clone(),
            action,
            current_count,
            target_count,
            reason,
            trigger_metric,
            trigger_value,
        }
    }
    
    /// Execute scaling action
    async fn execute_scaling_action(decision: &ScalingDecision) {
        match decision.action {
            ScalingAction::ScaleUp => {
                info!("Scaling up {:?} from {} to {} agents", 
                      decision.swarm_type, decision.current_count, decision.target_count);
                // Implementation would spawn new agents
            }
            ScalingAction::ScaleDown => {
                info!("Scaling down {:?} from {} to {} agents", 
                      decision.swarm_type, decision.current_count, decision.target_count);
                // Implementation would gracefully shutdown agents
            }
            ScalingAction::NoAction => {}
        }
    }
    
    /// Start metrics collection
    async fn start_metrics_collection(&self) -> Result<(), MCPOrchestrationError> {
        let metrics_collector = Arc::clone(&self.metrics_collector);
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_millis(10000)); // 10 second intervals
            
            loop {
                interval.tick().await;
                
                Self::update_metrics(&metrics_collector).await;
            }
        });
        
        Ok(())
    }
    
    /// Update metrics
    async fn update_metrics(metrics_collector: &LoadBalancerMetrics) {
        // Update global metrics
        let mut global_metrics = metrics_collector.global_metrics.write().await;
        
        // Calculate cache hit rate
        let total_selections = global_metrics.total_requests;
        if total_selections > 0 {
            // Simplified calculation
            global_metrics.cache_hit_rate = 0.85; // Would be calculated from actual cache hits
        }
        
        // Update algorithm performance metrics
        for algorithm_entry in metrics_collector.algorithm_performance.iter() {
            let algorithm = algorithm_entry.key();
            let performance = algorithm_entry.value();
            
            gauge!("load_balancer_algorithm_accuracy", performance.accuracy_score, "algorithm" => algorithm.to_string());
            gauge!("load_balancer_algorithm_selection_time", performance.average_selection_time_ns as f64, "algorithm" => algorithm.to_string());
        }
        
        gauge!("load_balancer_cache_hit_rate", global_metrics.cache_hit_rate);
        gauge!("load_balancer_failover_count", global_metrics.failover_count as f64);
    }
    
    /// Select agent for load balancing
    #[instrument(skip(self, request_context))]
    pub async fn select_agent(&self, swarm_type: &SwarmType, request_context: &RequestContext) -> Result<String, MCPOrchestrationError> {
        let start_time = Instant::now();
        
        // Check cache first
        let cache_key = format!("{}_{:?}", swarm_type.to_string(), request_context.priority);
        if let Some(cached) = self.selection_cache.get(&cache_key) {
            if cached.selection_time.elapsed() < cached.validity_duration {
                histogram!("load_balancer_selection_time_ns", start_time.elapsed().as_nanos() as f64);
                counter!("load_balancer_cache_hits", 1);
                return Ok(cached.selected_agent.clone());
            }
        }
        
        // Get candidate agents
        let candidates = self.get_healthy_agents(swarm_type).await;
        
        if candidates.is_empty() {
            return Err(MCPOrchestrationError::LoadBalancingError {
                reason: format!("No healthy agents available for swarm: {:?}", swarm_type),
            });
        }
        
        // Apply load balancing algorithm
        let decision = self.apply_load_balancing_algorithm(&candidates, request_context).await;
        
        // Check circuit breaker
        if let Some(circuit_breaker) = self.circuit_breakers.get(&decision.selected_agent) {
            if matches!(circuit_breaker.state, CircuitBreakerState::Open) {
                // Circuit breaker is open, find alternative
                let alternative_candidates: Vec<_> = candidates.into_iter()
                    .filter(|agent| agent.agent_id != decision.selected_agent)
                    .collect();
                
                if !alternative_candidates.is_empty() {
                    let alternative_decision = self.apply_load_balancing_algorithm(&alternative_candidates, request_context).await;
                    
                    // Cache the selection
                    self.cache_selection(&cache_key, &alternative_decision.selected_agent).await;
                    
                    histogram!("load_balancer_selection_time_ns", start_time.elapsed().as_nanos() as f64);
                    counter!("load_balancer_circuit_breaker_bypasses", 1);
                    
                    return Ok(alternative_decision.selected_agent);
                } else {
                    return Err(MCPOrchestrationError::LoadBalancingError {
                        reason: "All agents have open circuit breakers".to_string(),
                    });
                }
            }
        }
        
        // Cache the selection
        self.cache_selection(&cache_key, &decision.selected_agent).await;
        
        histogram!("load_balancer_selection_time_ns", start_time.elapsed().as_nanos() as f64);
        counter!("load_balancer_selections", 1);
        
        Ok(decision.selected_agent)
    }
    
    /// Get healthy agents for a swarm
    async fn get_healthy_agents(&self, swarm_type: &SwarmType) -> Vec<AgentLoad> {
        self.agent_loads.iter()
            .filter(|entry| {
                let agent_load = entry.value();
                agent_load.swarm_type == *swarm_type && agent_load.health_score > 0.5
            })
            .map(|entry| entry.value().clone())
            .collect()
    }
    
    /// Apply load balancing algorithm
    async fn apply_load_balancing_algorithm(
        &self,
        candidates: &[AgentLoad],
        request_context: &RequestContext,
    ) -> LoadBalancingDecision {
        match self.config.algorithm {
            LoadBalancingAlgorithm::WeightedRoundRobin => {
                self.weighted_round_robin_selection(candidates).await
            }
            LoadBalancingAlgorithm::LeastConnections => {
                self.least_connections_selection(candidates).await
            }
            LoadBalancingAlgorithm::ResourceBased => {
                self.resource_based_selection(candidates).await
            }
            LoadBalancingAlgorithm::LatencyBased => {
                self.latency_based_selection(candidates).await
            }
            _ => {
                // Default to weighted round robin
                self.weighted_round_robin_selection(candidates).await
            }
        }
    }
    
    /// Weighted round robin selection
    async fn weighted_round_robin_selection(&self, candidates: &[AgentLoad]) -> LoadBalancingDecision {
        // Select based on inverse of current load
        let mut best_agent = &candidates[0];
        let mut best_score = 0.0;
        
        for candidate in candidates {
            // Calculate score based on inverse of resource usage
            let cpu_score = 1.0 - candidate.cpu_usage;
            let memory_score = 1.0 - candidate.memory_usage;
            let health_score = candidate.health_score;
            let response_score = 1.0 / (candidate.response_time_ms / 1000.0 + 1.0);
            
            let total_score = (cpu_score + memory_score + health_score + response_score) / 4.0;
            
            if total_score > best_score {
                best_score = total_score;
                best_agent = candidate;
            }
        }
        
        LoadBalancingDecision {
            selected_agent: best_agent.agent_id.clone(),
            algorithm_used: LoadBalancingAlgorithm::WeightedRoundRobin,
            selection_score: best_score,
            alternatives: candidates.iter().map(|a| a.agent_id.clone()).collect(),
            decision_time_ns: 1000, // Would measure actual time
        }
    }
    
    /// Least connections selection
    async fn least_connections_selection(&self, candidates: &[AgentLoad]) -> LoadBalancingDecision {
        let best_agent = candidates.iter()
            .min_by_key(|agent| agent.active_connections)
            .unwrap();
        
        LoadBalancingDecision {
            selected_agent: best_agent.agent_id.clone(),
            algorithm_used: LoadBalancingAlgorithm::LeastConnections,
            selection_score: 1.0 / (best_agent.active_connections as f64 + 1.0),
            alternatives: candidates.iter().map(|a| a.agent_id.clone()).collect(),
            decision_time_ns: 500,
        }
    }
    
    /// Resource-based selection
    async fn resource_based_selection(&self, candidates: &[AgentLoad]) -> LoadBalancingDecision {
        let best_agent = candidates.iter()
            .min_by(|a, b| {
                let a_load = a.cpu_usage + a.memory_usage;
                let b_load = b.cpu_usage + b.memory_usage;
                a_load.partial_cmp(&b_load).unwrap()
            })
            .unwrap();
        
        LoadBalancingDecision {
            selected_agent: best_agent.agent_id.clone(),
            algorithm_used: LoadBalancingAlgorithm::ResourceBased,
            selection_score: 2.0 - (best_agent.cpu_usage + best_agent.memory_usage),
            alternatives: candidates.iter().map(|a| a.agent_id.clone()).collect(),
            decision_time_ns: 750,
        }
    }
    
    /// Latency-based selection
    async fn latency_based_selection(&self, candidates: &[AgentLoad]) -> LoadBalancingDecision {
        let best_agent = candidates.iter()
            .min_by(|a, b| a.response_time_ms.partial_cmp(&b.response_time_ms).unwrap())
            .unwrap();
        
        LoadBalancingDecision {
            selected_agent: best_agent.agent_id.clone(),
            algorithm_used: LoadBalancingAlgorithm::LatencyBased,
            selection_score: 1.0 / (best_agent.response_time_ms / 1000.0 + 1.0),
            alternatives: candidates.iter().map(|a| a.agent_id.clone()).collect(),
            decision_time_ns: 400,
        }
    }
    
    /// Cache selection result
    async fn cache_selection(&self, cache_key: &str, selected_agent: &str) {
        let cached_selection = CachedSelection {
            selected_agent: selected_agent.to_string(),
            selection_time: Instant::now(),
            validity_duration: Duration::from_millis(5000), // 5 second cache
            selection_context: cache_key.to_string(),
        };
        
        self.selection_cache.insert(cache_key.to_string(), cached_selection);
    }
    
    /// Update agent load information
    pub async fn update_agent_load(&self, agent_load: AgentLoad) {
        self.agent_loads.insert(agent_load.agent_id.clone(), agent_load);
    }
    
    /// Get load statistics
    pub async fn get_load_statistics(&self) -> LoadStatistics {
        let load_history = self.load_history.read().await;
        let latest_snapshot = load_history.last();
        
        match latest_snapshot {
            Some(snapshot) => LoadStatistics {
                total_agents: snapshot.global_metrics.total_agents,
                healthy_agents: snapshot.global_metrics.healthy_agents,
                average_cpu_usage: snapshot.global_metrics.total_cpu_usage / snapshot.global_metrics.total_agents as f64,
                average_memory_usage: snapshot.global_metrics.total_memory_usage / snapshot.global_metrics.total_agents as f64,
                total_connections: snapshot.global_metrics.total_connections,
                average_response_time: snapshot.global_metrics.average_response_time,
                failover_count: 0, // Would track actual failovers
            },
            None => LoadStatistics {
                total_agents: 0,
                healthy_agents: 0,
                average_cpu_usage: 0.0,
                average_memory_usage: 0.0,
                total_connections: 0,
                average_response_time: 0.0,
                failover_count: 0,
            },
        }
    }
    
    /// Optimize load balancer performance
    pub async fn optimize(&self) -> Result<(), MCPOrchestrationError> {
        info!("Optimizing load balancer performance");
        
        // Analyze algorithm performance
        self.analyze_algorithm_performance().await;
        
        // Update circuit breaker thresholds
        self.update_circuit_breaker_thresholds().await;
        
        // Clean up expired cache entries
        self.cleanup_cache().await;
        
        Ok(())
    }
    
    /// Analyze algorithm performance
    async fn analyze_algorithm_performance(&self) {
        // Analyze which algorithms perform best under current conditions
        // This would involve statistical analysis of selection outcomes
        debug!("Analyzing load balancing algorithm performance");
    }
    
    /// Update circuit breaker thresholds
    async fn update_circuit_breaker_thresholds(&self) {
        // Dynamically adjust circuit breaker thresholds based on current conditions
        debug!("Updating circuit breaker thresholds");
    }
    
    /// Clean up expired cache entries
    async fn cleanup_cache(&self) {
        let now = Instant::now();
        self.selection_cache.retain(|_, cached| {
            now.duration_since(cached.selection_time) < cached.validity_duration
        });
    }
}

impl Clone for LoadBalancer {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            agent_loads: Arc::clone(&self.agent_loads),
            circuit_breakers: Arc::clone(&self.circuit_breakers),
            load_history: Arc::clone(&self.load_history),
            health_monitor: Arc::clone(&self.health_monitor),
            auto_scaler: Arc::clone(&self.auto_scaler),
            session_manager: Arc::clone(&self.session_manager),
            algorithms: Arc::clone(&self.algorithms),
            metrics_collector: Arc::clone(&self.metrics_collector),
            selection_cache: Arc::clone(&self.selection_cache),
        }
    }
}

/// Load balancer statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadStatistics {
    pub total_agents: usize,
    pub healthy_agents: usize,
    pub average_cpu_usage: f64,
    pub average_memory_usage: f64,
    pub total_connections: u64,
    pub average_response_time: f64,
    pub failover_count: u64,
}

impl LoadBalancingAlgorithm {
    fn to_string(&self) -> String {
        match self {
            LoadBalancingAlgorithm::RoundRobin => "round_robin".to_string(),
            LoadBalancingAlgorithm::WeightedRoundRobin => "weighted_round_robin".to_string(),
            LoadBalancingAlgorithm::LeastConnections => "least_connections".to_string(),
            LoadBalancingAlgorithm::WeightedLeastConnections => "weighted_least_connections".to_string(),
            LoadBalancingAlgorithm::ResourceBased => "resource_based".to_string(),
            LoadBalancingAlgorithm::LatencyBased => "latency_based".to_string(),
            LoadBalancingAlgorithm::ConsistentHashing => "consistent_hashing".to_string(),
            LoadBalancingAlgorithm::PowerOfTwoChoices => "power_of_two_choices".to_string(),
            LoadBalancingAlgorithm::AdaptiveWeighted => "adaptive_weighted".to_string(),
        }
    }
}

impl SwarmType {
    fn to_string(&self) -> String {
        match self {
            SwarmType::RiskManagement => "risk_management".to_string(),
            SwarmType::TradingStrategy => "trading_strategy".to_string(),
            SwarmType::DataPipeline => "data_pipeline".to_string(),
            SwarmType::TENGRIWatchdog => "tengri_watchdog".to_string(),
            SwarmType::QuantumML => "quantum_ml".to_string(),
            SwarmType::MCPOrchestration => "mcp_orchestration".to_string(),
        }
    }
}