//! Crisis Management System
//!
//! Advanced crisis management system with automatic responses, adaptive decision-making,
//! and intelligent recovery mechanisms for trading operations.
//!
//! Features:
//! - Automatic crisis detection and classification
//! - Adaptive response strategies based on crisis type
//! - Intelligent recovery and normalization protocols
//! - Real-time crisis monitoring and alerting
//! - Machine learning-based crisis prediction
//! - Integration with emergency override system

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicF64, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::{RwLock, Mutex, broadcast, mpsc};
use tokio::time::{interval, sleep};
use serde::{Deserialize, Serialize};
use tracing::{info, warn, error, debug, instrument};

use crate::emergency_override::{EmergencyEvent, EmergencyLevel, EmergencyAction, MarketConditions};

/// Crisis types that can be detected and managed
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CrisisType {
    /// Financial market crash
    MarketCrash,
    /// Liquidity crisis
    LiquidityCrisis,
    /// Flash crash event
    FlashCrash,
    /// Systemic risk event
    SystemicRisk,
    /// Correlation breakdown
    CorrelationBreakdown,
    /// Extreme volatility
    VolatilityStorm,
    /// Whale manipulation
    WhaleManipulation,
    /// Technical failure
    TechnicalFailure,
    /// Regulatory event
    RegulatoryEvent,
    /// Geopolitical crisis
    GeopoliticalCrisis,
}

/// Crisis severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum CrisisSeverity {
    /// Minor crisis - localized impact
    Minor = 1,
    /// Moderate crisis - significant impact
    Moderate = 2,
    /// Major crisis - widespread impact
    Major = 3,
    /// Severe crisis - systemic impact
    Severe = 4,
    /// Critical crisis - market-wide impact
    Critical = 5,
}

/// Crisis lifecycle phases
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CrisisPhase {
    /// Pre-crisis indicators detected
    PreCrisis,
    /// Crisis onset
    Onset,
    /// Crisis escalation
    Escalation,
    /// Crisis peak
    Peak,
    /// Crisis de-escalation
    DeEscalation,
    /// Recovery phase
    Recovery,
    /// Post-crisis normalization
    Normalization,
}

/// Crisis response strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CrisisResponse {
    /// Immediate defensive actions
    Defensive {
        reduce_exposure: f64,
        increase_cash: f64,
        activate_hedging: bool,
    },
    /// Opportunistic actions during crisis
    Opportunistic {
        target_assets: Vec<String>,
        buy_levels: Vec<f64>,
        max_exposure: f64,
    },
    /// Wait and monitor approach
    WaitAndMonitor {
        monitoring_interval: Duration,
        trigger_conditions: Vec<String>,
    },
    /// Aggressive recovery actions
    Recovery {
        rebalance_portfolio: bool,
        increase_positions: f64,
        target_allocation: HashMap<String, f64>,
    },
}

/// Crisis detection and management system
pub struct CrisisManagementSystem {
    /// System configuration
    config: Arc<RwLock<CrisisConfig>>,
    /// Current crisis state
    crisis_state: Arc<RwLock<CrisisState>>,
    /// Crisis history
    crisis_history: Arc<RwLock<VecDeque<CrisisRecord>>>,
    /// Crisis detection engine
    detection_engine: Arc<CrisisDetectionEngine>,
    /// Response engine
    response_engine: Arc<CrisisResponseEngine>,
    /// Recovery engine
    recovery_engine: Arc<CrisisRecoveryEngine>,
    /// Crisis broadcast channel
    crisis_broadcast: broadcast::Sender<CrisisEvent>,
    /// System running state
    running: Arc<AtomicBool>,
    /// Performance metrics
    metrics: Arc<CrisisMetrics>,
}

/// Crisis management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrisisConfig {
    /// Detection sensitivity (0.0 to 1.0)
    pub detection_sensitivity: f64,
    /// Response time target (nanoseconds)
    pub response_time_target_ns: u64,
    /// Maximum crisis duration before escalation
    pub max_crisis_duration: Duration,
    /// Auto-recovery enabled
    pub auto_recovery: bool,
    /// Recovery timeout
    pub recovery_timeout: Duration,
    /// Crisis history retention period
    pub history_retention: Duration,
    /// Enable machine learning predictions
    pub enable_ml_prediction: bool,
    /// Crisis response strategies
    pub response_strategies: HashMap<CrisisType, CrisisResponse>,
}

impl Default for CrisisConfig {
    fn default() -> Self {
        let mut response_strategies = HashMap::new();
        
        response_strategies.insert(CrisisType::MarketCrash, CrisisResponse::Defensive {
            reduce_exposure: 0.8,
            increase_cash: 0.3,
            activate_hedging: true,
        });
        
        response_strategies.insert(CrisisType::FlashCrash, CrisisResponse::Opportunistic {
            target_assets: vec!["BTC".to_string(), "ETH".to_string()],
            buy_levels: vec![0.9, 0.8, 0.7],
            max_exposure: 0.1,
        });
        
        response_strategies.insert(CrisisType::LiquidityCrisis, CrisisResponse::WaitAndMonitor {
            monitoring_interval: Duration::from_secs(30),
            trigger_conditions: vec!["liquidity_improvement".to_string()],
        });
        
        Self {
            detection_sensitivity: 0.7,
            response_time_target_ns: 100_000, // 100 microseconds
            max_crisis_duration: Duration::from_hours(4),
            auto_recovery: true,
            recovery_timeout: Duration::from_hours(2),
            history_retention: Duration::from_days(30),
            enable_ml_prediction: true,
            response_strategies,
        }
    }
}

/// Current crisis state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrisisState {
    /// Current crisis type (if any)
    pub current_crisis: Option<CrisisType>,
    /// Crisis severity level
    pub severity: CrisisSeverity,
    /// Current phase of crisis
    pub phase: CrisisPhase,
    /// Crisis start time
    pub start_time: Option<Instant>,
    /// Crisis duration
    pub duration: Duration,
    /// Active response strategies
    pub active_responses: Vec<CrisisResponse>,
    /// Recovery status
    pub recovery_active: bool,
    /// Predicted crisis end time
    pub predicted_end: Option<Instant>,
    /// Crisis impact metrics
    pub impact_metrics: CrisisImpactMetrics,
}

/// Crisis impact metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrisisImpactMetrics {
    /// Portfolio impact percentage
    pub portfolio_impact: f64,
    /// Maximum drawdown during crisis
    pub max_drawdown: f64,
    /// Liquidity impact
    pub liquidity_impact: f64,
    /// Volatility increase
    pub volatility_increase: f64,
    /// Recovery time estimate
    pub recovery_time_estimate: Duration,
}

/// Crisis event notification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrisisEvent {
    /// Event type
    pub event_type: CrisisEventType,
    /// Crisis type
    pub crisis_type: CrisisType,
    /// Event severity
    pub severity: CrisisSeverity,
    /// Event timestamp
    pub timestamp: Instant,
    /// Event description
    pub description: String,
    /// Additional event data
    pub data: HashMap<String, String>,
}

/// Crisis event types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CrisisEventType {
    /// Crisis detected
    CrisisDetected,
    /// Crisis escalated
    CrisisEscalated,
    /// Crisis de-escalated
    CrisisDeEscalated,
    /// Crisis ended
    CrisisEnded,
    /// Recovery started
    RecoveryStarted,
    /// Recovery completed
    RecoveryCompleted,
    /// Response activated
    ResponseActivated,
    /// Response modified
    ResponseModified,
}

/// Crisis historical record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrisisRecord {
    /// Crisis type
    pub crisis_type: CrisisType,
    /// Crisis severity
    pub severity: CrisisSeverity,
    /// Start time
    pub start_time: SystemTime,
    /// End time
    pub end_time: Option<SystemTime>,
    /// Duration
    pub duration: Duration,
    /// Responses used
    pub responses: Vec<CrisisResponse>,
    /// Final outcome
    pub outcome: CrisisOutcome,
    /// Lessons learned
    pub lessons: Vec<String>,
}

/// Crisis outcome classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CrisisOutcome {
    /// Successfully managed
    Successful,
    /// Partially successful
    Partial,
    /// Failed to manage effectively
    Failed,
    /// Ongoing crisis
    Ongoing,
}

/// Crisis detection engine
pub struct CrisisDetectionEngine {
    /// Detection algorithms
    detectors: HashMap<CrisisType, Box<dyn CrisisDetector + Send + Sync>>,
    /// Machine learning models
    ml_models: Arc<RwLock<HashMap<CrisisType, MLModel>>>,
    /// Detection history
    detection_history: Arc<RwLock<Vec<DetectionEvent>>>,
    /// Detection metrics
    metrics: Arc<DetectionMetrics>,
}

/// Crisis detector trait
pub trait CrisisDetector {
    /// Detect crisis based on market conditions
    fn detect(&self, conditions: &MarketConditions) -> CrisisDetectionResult;
    /// Get detector name
    fn name(&self) -> &str;
}

/// Crisis detection result
#[derive(Debug, Clone)]
pub struct CrisisDetectionResult {
    /// Detection confidence (0.0 to 1.0)
    pub confidence: f64,
    /// Crisis severity if detected
    pub severity: CrisisSeverity,
    /// Detection latency
    pub latency_ns: u64,
    /// Supporting indicators
    pub indicators: HashMap<String, f64>,
}

/// Machine learning model for crisis prediction
pub struct MLModel {
    /// Model type
    model_type: String,
    /// Model parameters
    parameters: HashMap<String, f64>,
    /// Training data
    training_data: Vec<TrainingDataPoint>,
    /// Model performance metrics
    performance: ModelPerformance,
}

/// Training data point for ML models
#[derive(Debug, Clone)]
pub struct TrainingDataPoint {
    /// Input features
    pub features: Vec<f64>,
    /// Target crisis type
    pub target_crisis: Option<CrisisType>,
    /// Target severity
    pub target_severity: CrisisSeverity,
    /// Timestamp
    pub timestamp: SystemTime,
}

/// ML model performance metrics
#[derive(Debug, Clone)]
pub struct ModelPerformance {
    /// Accuracy score
    pub accuracy: f64,
    /// Precision score
    pub precision: f64,
    /// Recall score
    pub recall: f64,
    /// F1 score
    pub f1_score: f64,
    /// False positive rate
    pub false_positive_rate: f64,
}

/// Detection event
#[derive(Debug, Clone)]
pub struct DetectionEvent {
    /// Crisis type detected
    pub crisis_type: CrisisType,
    /// Detection confidence
    pub confidence: f64,
    /// Detection timestamp
    pub timestamp: Instant,
    /// Detector name
    pub detector: String,
    /// True positive (if known)
    pub true_positive: Option<bool>,
}

/// Detection metrics
pub struct DetectionMetrics {
    /// Total detections
    pub total_detections: AtomicU64,
    /// True positives
    pub true_positives: AtomicU64,
    /// False positives
    pub false_positives: AtomicU64,
    /// Average detection time
    pub avg_detection_time_ns: AtomicU64,
    /// Detection accuracy
    pub accuracy: AtomicF64,
}

/// Crisis response engine
pub struct CrisisResponseEngine {
    /// Response strategies
    strategies: HashMap<CrisisType, Box<dyn ResponseStrategy + Send + Sync>>,
    /// Response history
    response_history: Arc<RwLock<Vec<ResponseEvent>>>,
    /// Response metrics
    metrics: Arc<ResponseMetrics>,
}

/// Response strategy trait
pub trait ResponseStrategy {
    /// Execute response strategy
    fn execute(&self, crisis_type: CrisisType, severity: CrisisSeverity) -> ResponseResult;
    /// Get strategy name
    fn name(&self) -> &str;
}

/// Response execution result
#[derive(Debug, Clone)]
pub struct ResponseResult {
    /// Actions taken
    pub actions: Vec<EmergencyAction>,
    /// Execution time
    pub execution_time_ns: u64,
    /// Success status
    pub success: bool,
    /// Error message (if any)
    pub error_message: Option<String>,
}

/// Response event
#[derive(Debug, Clone)]
pub struct ResponseEvent {
    /// Crisis type
    pub crisis_type: CrisisType,
    /// Response strategy used
    pub strategy: String,
    /// Actions taken
    pub actions: Vec<EmergencyAction>,
    /// Timestamp
    pub timestamp: Instant,
    /// Success status
    pub success: bool,
}

/// Response metrics
pub struct ResponseMetrics {
    /// Total responses
    pub total_responses: AtomicU64,
    /// Successful responses
    pub successful_responses: AtomicU64,
    /// Average response time
    pub avg_response_time_ns: AtomicU64,
    /// Response effectiveness
    pub effectiveness: AtomicF64,
}

/// Crisis recovery engine
pub struct CrisisRecoveryEngine {
    /// Recovery strategies
    strategies: HashMap<CrisisType, Box<dyn RecoveryStrategy + Send + Sync>>,
    /// Recovery history
    recovery_history: Arc<RwLock<Vec<RecoveryEvent>>>,
    /// Recovery metrics
    metrics: Arc<RecoveryMetrics>,
}

/// Recovery strategy trait
pub trait RecoveryStrategy {
    /// Execute recovery strategy
    fn execute(&self, crisis_record: &CrisisRecord) -> RecoveryResult;
    /// Get strategy name
    fn name(&self) -> &str;
}

/// Recovery execution result
#[derive(Debug, Clone)]
pub struct RecoveryResult {
    /// Recovery actions
    pub actions: Vec<EmergencyAction>,
    /// Recovery time estimate
    pub recovery_time_estimate: Duration,
    /// Success probability
    pub success_probability: f64,
}

/// Recovery event
#[derive(Debug, Clone)]
pub struct RecoveryEvent {
    /// Crisis type
    pub crisis_type: CrisisType,
    /// Recovery strategy used
    pub strategy: String,
    /// Recovery actions
    pub actions: Vec<EmergencyAction>,
    /// Timestamp
    pub timestamp: Instant,
    /// Recovery success
    pub success: bool,
}

/// Recovery metrics
pub struct RecoveryMetrics {
    /// Total recovery attempts
    pub total_recoveries: AtomicU64,
    /// Successful recoveries
    pub successful_recoveries: AtomicU64,
    /// Average recovery time
    pub avg_recovery_time_ns: AtomicU64,
    /// Recovery success rate
    pub success_rate: AtomicF64,
}

/// Crisis system metrics
pub struct CrisisMetrics {
    /// Total crises handled
    pub total_crises: AtomicU64,
    /// Successfully resolved crises
    pub resolved_crises: AtomicU64,
    /// Average crisis duration
    pub avg_crisis_duration_ms: AtomicU64,
    /// System uptime
    pub uptime_seconds: AtomicU64,
    /// Performance score
    pub performance_score: AtomicF64,
}

impl CrisisManagementSystem {
    /// Create new crisis management system
    pub async fn new(config: CrisisConfig) -> Result<Self, CrisisError> {
        let config = Arc::new(RwLock::new(config));
        
        let crisis_state = Arc::new(RwLock::new(CrisisState {
            current_crisis: None,
            severity: CrisisSeverity::Minor,
            phase: CrisisPhase::PreCrisis,
            start_time: None,
            duration: Duration::from_secs(0),
            active_responses: vec![],
            recovery_active: false,
            predicted_end: None,
            impact_metrics: CrisisImpactMetrics {
                portfolio_impact: 0.0,
                max_drawdown: 0.0,
                liquidity_impact: 0.0,
                volatility_increase: 0.0,
                recovery_time_estimate: Duration::from_secs(0),
            },
        }));
        
        let crisis_history = Arc::new(RwLock::new(VecDeque::new()));
        
        let detection_engine = Arc::new(CrisisDetectionEngine {
            detectors: HashMap::new(),
            ml_models: Arc::new(RwLock::new(HashMap::new())),
            detection_history: Arc::new(RwLock::new(Vec::new())),
            metrics: Arc::new(DetectionMetrics {
                total_detections: AtomicU64::new(0),
                true_positives: AtomicU64::new(0),
                false_positives: AtomicU64::new(0),
                avg_detection_time_ns: AtomicU64::new(0),
                accuracy: AtomicF64::new(0.0),
            }),
        });
        
        let response_engine = Arc::new(CrisisResponseEngine {
            strategies: HashMap::new(),
            response_history: Arc::new(RwLock::new(Vec::new())),
            metrics: Arc::new(ResponseMetrics {
                total_responses: AtomicU64::new(0),
                successful_responses: AtomicU64::new(0),
                avg_response_time_ns: AtomicU64::new(0),
                effectiveness: AtomicF64::new(0.0),
            }),
        });
        
        let recovery_engine = Arc::new(CrisisRecoveryEngine {
            strategies: HashMap::new(),
            recovery_history: Arc::new(RwLock::new(Vec::new())),
            metrics: Arc::new(RecoveryMetrics {
                total_recoveries: AtomicU64::new(0),
                successful_recoveries: AtomicU64::new(0),
                avg_recovery_time_ns: AtomicU64::new(0),
                success_rate: AtomicF64::new(0.0),
            }),
        });
        
        let (crisis_broadcast, _) = broadcast::channel(1000);
        let running = Arc::new(AtomicBool::new(false));
        
        let metrics = Arc::new(CrisisMetrics {
            total_crises: AtomicU64::new(0),
            resolved_crises: AtomicU64::new(0),
            avg_crisis_duration_ms: AtomicU64::new(0),
            uptime_seconds: AtomicU64::new(0),
            performance_score: AtomicF64::new(1.0),
        });
        
        Ok(Self {
            config,
            crisis_state,
            crisis_history,
            detection_engine,
            response_engine,
            recovery_engine,
            crisis_broadcast,
            running,
            metrics,
        })
    }
    
    /// Start the crisis management system
    #[instrument(skip(self))]
    pub async fn start(&self) -> Result<(), CrisisError> {
        info!("Starting Crisis Management System");
        
        self.running.store(true, Ordering::SeqCst);
        
        // Start monitoring tasks
        self.start_crisis_monitoring().await;
        self.start_response_monitoring().await;
        self.start_recovery_monitoring().await;
        
        info!("Crisis Management System started successfully");
        Ok(())
    }
    
    /// Stop the crisis management system
    #[instrument(skip(self))]
    pub async fn stop(&self) -> Result<(), CrisisError> {
        info!("Stopping Crisis Management System");
        
        self.running.store(false, Ordering::SeqCst);
        
        // Wait for all tasks to complete
        sleep(Duration::from_secs(1)).await;
        
        info!("Crisis Management System stopped");
        Ok(())
    }
    
    /// Start crisis monitoring and detection
    async fn start_crisis_monitoring(&self) {
        let detection_engine = self.detection_engine.clone();
        let crisis_state = self.crisis_state.clone();
        let crisis_broadcast = self.crisis_broadcast.clone();
        let running = self.running.clone();
        let metrics = self.metrics.clone();
        
        tokio::spawn(async move {
            let mut interval = interval(Duration::from_millis(10));
            
            while running.load(Ordering::SeqCst) {
                interval.tick().await;
                
                // Simulate market conditions for crisis detection
                let market_conditions = MarketConditions {
                    volatility: 0.02 + (rand::random::<f64>() * 0.08),
                    liquidity: 50_000.0 + (rand::random::<f64>() * 100_000.0),
                    drawdown: rand::random::<f64>() * 0.15,
                    correlations: HashMap::new(),
                    whale_activity: rand::random::<f64>(),
                    flash_crash_probability: rand::random::<f64>() * 0.2,
                    systemic_risk: rand::random::<f64>() * 0.3,
                    market_regime: "normal".to_string(),
                    timestamp: SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap()
                        .as_secs(),
                };
                
                // Check for crisis conditions
                if let Some(crisis_type) = Self::detect_crisis(&market_conditions).await {
                    let mut state = crisis_state.write().await;
                    
                    if state.current_crisis.is_none() {
                        // New crisis detected
                        state.current_crisis = Some(crisis_type);
                        state.start_time = Some(Instant::now());
                        state.phase = CrisisPhase::Onset;
                        state.severity = Self::calculate_severity(&market_conditions);
                        
                        metrics.total_crises.fetch_add(1, Ordering::SeqCst);
                        
                        // Broadcast crisis event
                        let event = CrisisEvent {
                            event_type: CrisisEventType::CrisisDetected,
                            crisis_type,
                            severity: state.severity,
                            timestamp: Instant::now(),
                            description: format!("Crisis detected: {:?}", crisis_type),
                            data: HashMap::new(),
                        };
                        
                        let _ = crisis_broadcast.send(event);
                        
                        info!("Crisis detected: {:?} - Severity: {:?}", crisis_type, state.severity);
                    }
                }
            }
        });
    }
    
    /// Start response monitoring
    async fn start_response_monitoring(&self) {
        let response_engine = self.response_engine.clone();
        let crisis_state = self.crisis_state.clone();
        let running = self.running.clone();
        
        tokio::spawn(async move {
            let mut interval = interval(Duration::from_millis(100));
            
            while running.load(Ordering::SeqCst) {
                interval.tick().await;
                
                let state = crisis_state.read().await;
                
                if let Some(crisis_type) = state.current_crisis {
                    if state.active_responses.is_empty() {
                        // Execute response strategy
                        let result = Self::execute_response_strategy(crisis_type, state.severity).await;
                        
                        if result.success {
                            response_engine.metrics.successful_responses.fetch_add(1, Ordering::SeqCst);
                            info!("Crisis response executed successfully for {:?}", crisis_type);
                        } else {
                            error!("Crisis response failed for {:?}: {:?}", crisis_type, result.error_message);
                        }
                        
                        response_engine.metrics.total_responses.fetch_add(1, Ordering::SeqCst);
                    }
                }
            }
        });
    }
    
    /// Start recovery monitoring
    async fn start_recovery_monitoring(&self) {
        let recovery_engine = self.recovery_engine.clone();
        let crisis_state = self.crisis_state.clone();
        let running = self.running.clone();
        
        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(30));
            
            while running.load(Ordering::SeqCst) {
                interval.tick().await;
                
                let state = crisis_state.read().await;
                
                if state.current_crisis.is_some() && state.phase == CrisisPhase::DeEscalation {
                    if !state.recovery_active {
                        // Start recovery process
                        info!("Starting crisis recovery process");
                        
                        // In real implementation, would execute recovery strategies
                        recovery_engine.metrics.total_recoveries.fetch_add(1, Ordering::SeqCst);
                    }
                }
            }
        });
    }
    
    /// Detect crisis based on market conditions
    async fn detect_crisis(conditions: &MarketConditions) -> Option<CrisisType> {
        // Simple crisis detection logic
        if conditions.flash_crash_probability > 0.8 {
            Some(CrisisType::FlashCrash)
        } else if conditions.volatility > 0.08 {
            Some(CrisisType::VolatilityStorm)
        } else if conditions.liquidity < 30_000.0 {
            Some(CrisisType::LiquidityCrisis)
        } else if conditions.systemic_risk > 0.7 {
            Some(CrisisType::SystemicRisk)
        } else if conditions.whale_activity > 0.9 {
            Some(CrisisType::WhaleManipulation)
        } else {
            None
        }
    }
    
    /// Calculate crisis severity
    fn calculate_severity(conditions: &MarketConditions) -> CrisisSeverity {
        let mut severity_score = 0.0;
        
        // Volatility contribution
        severity_score += conditions.volatility * 2.0;
        
        // Liquidity contribution
        if conditions.liquidity < 50_000.0 {
            severity_score += (50_000.0 - conditions.liquidity) / 50_000.0;
        }
        
        // Drawdown contribution
        severity_score += conditions.drawdown;
        
        // Systemic risk contribution
        severity_score += conditions.systemic_risk;
        
        // Convert to severity level
        if severity_score > 3.0 {
            CrisisSeverity::Critical
        } else if severity_score > 2.0 {
            CrisisSeverity::Severe
        } else if severity_score > 1.0 {
            CrisisSeverity::Major
        } else if severity_score > 0.5 {
            CrisisSeverity::Moderate
        } else {
            CrisisSeverity::Minor
        }
    }
    
    /// Execute response strategy for crisis
    async fn execute_response_strategy(crisis_type: CrisisType, severity: CrisisSeverity) -> ResponseResult {
        let start_time = Instant::now();
        
        // Simulate response execution
        let actions = match crisis_type {
            CrisisType::FlashCrash => vec![
                EmergencyAction::EnableAntifragileMode,
                EmergencyAction::ReducePositions(0.3),
            ],
            CrisisType::MarketCrash => vec![
                EmergencyAction::ReducePositions(0.7),
                EmergencyAction::ActivateHedging,
                EmergencyAction::IncreaseCashReserves,
            ],
            CrisisType::LiquidityCrisis => vec![
                EmergencyAction::ReducePositions(0.5),
                EmergencyAction::SetEmergencyStops,
            ],
            CrisisType::VolatilityStorm => vec![
                EmergencyAction::ReducePositions(0.4),
                EmergencyAction::ActivateHedging,
            ],
            CrisisType::WhaleManipulation => vec![
                EmergencyAction::EnableWhaleFollowing,
                EmergencyAction::ActivateCrisisArbitrage,
            ],
            _ => vec![EmergencyAction::SetEmergencyStops],
        };
        
        let execution_time = start_time.elapsed().as_nanos() as u64;
        
        ResponseResult {
            actions,
            execution_time_ns: execution_time,
            success: true,
            error_message: None,
        }
    }
    
    /// Get current crisis management status
    pub async fn get_status(&self) -> CrisisManagementStatus {
        let state = self.crisis_state.read().await;
        let history = self.crisis_history.read().await;
        
        CrisisManagementStatus {
            current_crisis: state.current_crisis,
            severity: state.severity,
            phase: state.phase,
            duration: state.duration,
            recovery_active: state.recovery_active,
            recent_crises: history.iter().take(10).cloned().collect(),
            system_metrics: CrisisSystemMetrics {
                total_crises: self.metrics.total_crises.load(Ordering::SeqCst),
                resolved_crises: self.metrics.resolved_crises.load(Ordering::SeqCst),
                avg_crisis_duration_ms: self.metrics.avg_crisis_duration_ms.load(Ordering::SeqCst),
                performance_score: self.metrics.performance_score.load(Ordering::SeqCst),
            },
        }
    }
}

/// Crisis management status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrisisManagementStatus {
    pub current_crisis: Option<CrisisType>,
    pub severity: CrisisSeverity,
    pub phase: CrisisPhase,
    pub duration: Duration,
    pub recovery_active: bool,
    pub recent_crises: Vec<CrisisRecord>,
    pub system_metrics: CrisisSystemMetrics,
}

/// Crisis system metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrisisSystemMetrics {
    pub total_crises: u64,
    pub resolved_crises: u64,
    pub avg_crisis_duration_ms: u64,
    pub performance_score: f64,
}

/// Crisis management errors
#[derive(Debug, thiserror::Error)]
pub enum CrisisError {
    #[error("Configuration error: {0}")]
    Configuration(String),
    #[error("Detection error: {0}")]
    Detection(String),
    #[error("Response error: {0}")]
    Response(String),
    #[error("Recovery error: {0}")]
    Recovery(String),
    #[error("System error: {0}")]
    System(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_crisis_system_initialization() {
        let config = CrisisConfig::default();
        let system = CrisisManagementSystem::new(config).await.unwrap();
        
        let status = system.get_status().await;
        assert!(status.current_crisis.is_none());
        assert_eq!(status.phase, CrisisPhase::PreCrisis);
    }
    
    #[tokio::test]
    async fn test_crisis_detection() {
        let conditions = MarketConditions {
            volatility: 0.12,
            liquidity: 20_000.0,
            drawdown: 0.08,
            correlations: HashMap::new(),
            whale_activity: 0.3,
            flash_crash_probability: 0.9,
            systemic_risk: 0.2,
            market_regime: "volatile".to_string(),
            timestamp: 1234567890,
        };
        
        let crisis_type = CrisisManagementSystem::detect_crisis(&conditions).await;
        assert_eq!(crisis_type, Some(CrisisType::FlashCrash));
    }
    
    #[tokio::test]
    async fn test_severity_calculation() {
        let conditions = MarketConditions {
            volatility: 0.15,
            liquidity: 30_000.0,
            drawdown: 0.12,
            correlations: HashMap::new(),
            whale_activity: 0.5,
            flash_crash_probability: 0.3,
            systemic_risk: 0.8,
            market_regime: "crisis".to_string(),
            timestamp: 1234567890,
        };
        
        let severity = CrisisManagementSystem::calculate_severity(&conditions);
        assert!(severity >= CrisisSeverity::Major);
    }
    
    #[tokio::test]
    async fn test_response_strategy() {
        let result = CrisisManagementSystem::execute_response_strategy(
            CrisisType::FlashCrash,
            CrisisSeverity::Major
        ).await;
        
        assert!(result.success);
        assert!(!result.actions.is_empty());
        assert!(result.execution_time_ns > 0);
    }
}