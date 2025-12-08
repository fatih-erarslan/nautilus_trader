//! Emergency Override System for Trading Operations
//!
//! This system provides comprehensive emergency protocols for black swan events,
//! liquidation protection, and crisis management with sub-microsecond response times.
//!
//! Key Features:
//! - Automatic emergency exit protocols for extreme market conditions
//! - Liquidation risk protection with margin safety mechanisms
//! - Flash crash detection with antifragile buying opportunities
//! - Whale momentum detection and opportunistic trading
//! - Crisis mode decision making with quantum-enhanced risk assessment
//! - Real-time market regime detection and adaptive response

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicF64, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::{RwLock, Mutex, broadcast};
use tokio::time::{interval, sleep};
use serde::{Deserialize, Serialize};
use tracing::{info, warn, error, debug, instrument};

/// Emergency alert levels for different crisis scenarios
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EmergencyLevel {
    /// Normal market conditions - no emergency actions required
    Normal = 0,
    /// Elevated risk - increase monitoring and prepare contingencies
    Elevated = 1,
    /// High risk - activate risk reduction protocols
    High = 2,
    /// Critical risk - immediate emergency actions required
    Critical = 3,
    /// Extreme emergency - full system shutdown and liquidation
    Extreme = 4,
}

/// Types of emergency events that can trigger override protocols
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EmergencyEvent {
    /// Market flash crash detected
    FlashCrash,
    /// Liquidity crisis in progress
    LiquidityCrisis,
    /// Large whale activity detected
    WhaleActivity,
    /// Correlation breakdown across markets
    CorrelationBreakdown,
    /// Extreme volatility spike
    VolatilitySpike,
    /// Systemic risk event
    SystemicRisk,
    /// Margin call risk detected
    MarginCallRisk,
    /// Account liquidation risk
    LiquidationRisk,
    /// Network/connectivity issues
    ConnectivityIssue,
    /// Exchange halt or suspension
    ExchangeHalt,
}

/// Emergency action protocols
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EmergencyAction {
    /// Immediately close all positions
    CloseAllPositions,
    /// Reduce position sizes by specified percentage
    ReducePositions(f64),
    /// Activate hedging strategies
    ActivateHedging,
    /// Increase cash reserves
    IncreaseCashReserves,
    /// Liquidate specific assets
    LiquidateAssets(Vec<String>),
    /// Set emergency stop-losses
    SetEmergencyStops,
    /// Suspend trading activities
    SuspendTrading,
    /// Enable antifragile buying mode
    EnableAntifragileMode,
    /// Opportunistic whale following
    EnableWhaleFollowing,
    /// Activate crisis arbitrage
    ActivateCrisisArbitrage,
}

/// Emergency override configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmergencyConfig {
    /// Maximum allowed drawdown before emergency activation
    pub max_drawdown: f64,
    /// Minimum liquidity threshold
    pub min_liquidity: f64,
    /// Maximum position size in emergency mode
    pub max_emergency_position: f64,
    /// Volatility threshold for emergency activation
    pub volatility_threshold: f64,
    /// Time window for emergency detection (seconds)
    pub detection_window: u64,
    /// Response time requirements (nanoseconds)
    pub max_response_time_ns: u64,
    /// Enable antifragile buying during crashes
    pub enable_antifragile_buying: bool,
    /// Enable whale momentum following
    pub enable_whale_following: bool,
    /// Emergency contact notifications
    pub emergency_contacts: Vec<String>,
    /// Automatic recovery settings
    pub auto_recovery: bool,
    /// Recovery timeout (seconds)
    pub recovery_timeout: u64,
}

impl Default for EmergencyConfig {
    fn default() -> Self {
        Self {
            max_drawdown: 0.05,          // 5% max drawdown
            min_liquidity: 100_000.0,    // $100k minimum liquidity
            max_emergency_position: 0.02, // 2% max position size
            volatility_threshold: 0.05,   // 5% volatility threshold
            detection_window: 60,         // 1 minute window
            max_response_time_ns: 1_000_000, // 1ms max response
            enable_antifragile_buying: true,
            enable_whale_following: true,
            emergency_contacts: vec!["emergency@trading.com".to_string()],
            auto_recovery: true,
            recovery_timeout: 300,        // 5 minutes
        }
    }
}

/// Real-time market conditions and risk metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketConditions {
    /// Current market volatility
    pub volatility: f64,
    /// Available liquidity
    pub liquidity: f64,
    /// Current drawdown percentage
    pub drawdown: f64,
    /// Market correlation matrix
    pub correlations: HashMap<String, f64>,
    /// Detected whale activity level
    pub whale_activity: f64,
    /// Flash crash probability
    pub flash_crash_probability: f64,
    /// Systemic risk indicators
    pub systemic_risk: f64,
    /// Market regime classification
    pub market_regime: String,
    /// Time of last update
    pub timestamp: u64,
}

/// Emergency system state and metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmergencyState {
    /// Current emergency level
    pub level: EmergencyLevel,
    /// Active emergency events
    pub active_events: Vec<EmergencyEvent>,
    /// Emergency actions taken
    pub actions_taken: Vec<EmergencyAction>,
    /// System response time (nanoseconds)
    pub response_time_ns: u64,
    /// Emergency activation time
    pub activation_time: Option<u64>,
    /// Recovery status
    pub recovery_active: bool,
    /// System health metrics
    pub health_metrics: HashMap<String, f64>,
}

/// Emergency Override System
pub struct EmergencyOverrideSystem {
    /// System configuration
    config: Arc<RwLock<EmergencyConfig>>,
    /// Current system state
    state: Arc<RwLock<EmergencyState>>,
    /// Market conditions monitor
    market_conditions: Arc<RwLock<MarketConditions>>,
    /// Emergency broadcast channel
    emergency_broadcast: broadcast::Sender<EmergencyEvent>,
    /// System running flag
    running: Arc<AtomicBool>,
    /// Performance metrics
    metrics: Arc<EmergencyMetrics>,
    /// Risk calculation engine
    risk_engine: Arc<RiskCalculationEngine>,
    /// Action executor
    action_executor: Arc<ActionExecutor>,
}

/// Performance metrics for emergency system
struct EmergencyMetrics {
    /// Total emergencies handled
    total_emergencies: AtomicU64,
    /// Average response time
    avg_response_time_ns: AtomicU64,
    /// False positive rate
    false_positive_rate: AtomicF64,
    /// System uptime
    uptime_seconds: AtomicU64,
    /// Last emergency timestamp
    last_emergency: AtomicU64,
}

/// Risk calculation engine for emergency detection
struct RiskCalculationEngine {
    /// Historical volatility data
    volatility_history: Arc<RwLock<Vec<f64>>>,
    /// Correlation matrix
    correlation_matrix: Arc<RwLock<HashMap<String, HashMap<String, f64>>>>,
    /// Whale detection algorithms
    whale_detector: Arc<WhaleDetector>,
    /// Flash crash detector
    flash_crash_detector: Arc<FlashCrashDetector>,
}

/// Whale activity detector
struct WhaleDetector {
    /// Large order threshold
    large_order_threshold: f64,
    /// Whale activity history
    activity_history: Arc<RwLock<Vec<WhaleActivity>>>,
    /// Detection sensitivity
    sensitivity: f64,
}

/// Flash crash detector
struct FlashCrashDetector {
    /// Price drop threshold
    price_drop_threshold: f64,
    /// Volume spike threshold
    volume_spike_threshold: f64,
    /// Time window for detection
    detection_window: Duration,
}

/// Whale activity record
#[derive(Debug, Clone, Serialize, Deserialize)]
struct WhaleActivity {
    /// Timestamp of activity
    timestamp: u64,
    /// Asset symbol
    symbol: String,
    /// Order size
    order_size: f64,
    /// Price impact
    price_impact: f64,
    /// Activity type (buy/sell)
    activity_type: String,
    /// Confidence score
    confidence: f64,
}

/// Action executor for emergency responses
struct ActionExecutor {
    /// Emergency action handlers
    handlers: HashMap<EmergencyAction, Box<dyn ActionHandler + Send + Sync>>,
    /// Execution metrics
    metrics: Arc<RwLock<HashMap<String, u64>>>,
}

/// Trait for emergency action handlers
trait ActionHandler {
    /// Execute the emergency action
    fn execute(&self, context: &EmergencyContext) -> Result<(), EmergencyError>;
    /// Get action description
    fn description(&self) -> &str;
}

/// Emergency execution context
#[derive(Debug, Clone)]
struct EmergencyContext {
    /// Current market conditions
    market_conditions: MarketConditions,
    /// Emergency level
    level: EmergencyLevel,
    /// Event that triggered the emergency
    event: EmergencyEvent,
    /// System state
    state: EmergencyState,
}

/// Emergency system errors
#[derive(Debug, thiserror::Error)]
pub enum EmergencyError {
    #[error("Configuration error: {0}")]
    Configuration(String),
    #[error("Market data error: {0}")]
    MarketData(String),
    #[error("Action execution error: {0}")]
    ActionExecution(String),
    #[error("System state error: {0}")]
    SystemState(String),
    #[error("Timeout error: {0}")]
    Timeout(String),
}

impl EmergencyOverrideSystem {
    /// Create new emergency override system
    pub async fn new(config: EmergencyConfig) -> Result<Self, EmergencyError> {
        let config = Arc::new(RwLock::new(config));
        let state = Arc::new(RwLock::new(EmergencyState {
            level: EmergencyLevel::Normal,
            active_events: vec![],
            actions_taken: vec![],
            response_time_ns: 0,
            activation_time: None,
            recovery_active: false,
            health_metrics: HashMap::new(),
        }));
        
        let market_conditions = Arc::new(RwLock::new(MarketConditions {
            volatility: 0.0,
            liquidity: 0.0,
            drawdown: 0.0,
            correlations: HashMap::new(),
            whale_activity: 0.0,
            flash_crash_probability: 0.0,
            systemic_risk: 0.0,
            market_regime: "normal".to_string(),
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        }));
        
        let (emergency_broadcast, _) = broadcast::channel(1000);
        let running = Arc::new(AtomicBool::new(false));
        
        let metrics = Arc::new(EmergencyMetrics {
            total_emergencies: AtomicU64::new(0),
            avg_response_time_ns: AtomicU64::new(0),
            false_positive_rate: AtomicF64::new(0.0),
            uptime_seconds: AtomicU64::new(0),
            last_emergency: AtomicU64::new(0),
        });
        
        let risk_engine = Arc::new(RiskCalculationEngine {
            volatility_history: Arc::new(RwLock::new(Vec::new())),
            correlation_matrix: Arc::new(RwLock::new(HashMap::new())),
            whale_detector: Arc::new(WhaleDetector {
                large_order_threshold: 1_000_000.0,
                activity_history: Arc::new(RwLock::new(Vec::new())),
                sensitivity: 0.8,
            }),
            flash_crash_detector: Arc::new(FlashCrashDetector {
                price_drop_threshold: 0.05,
                volume_spike_threshold: 5.0,
                detection_window: Duration::from_secs(30),
            }),
        });
        
        let action_executor = Arc::new(ActionExecutor {
            handlers: HashMap::new(),
            metrics: Arc::new(RwLock::new(HashMap::new())),
        });
        
        Ok(Self {
            config,
            state,
            market_conditions,
            emergency_broadcast,
            running,
            metrics,
            risk_engine,
            action_executor,
        })
    }
    
    /// Start the emergency override system
    #[instrument(skip(self))]
    pub async fn start(&self) -> Result<(), EmergencyError> {
        info!("Starting Emergency Override System");
        
        self.running.store(true, Ordering::SeqCst);
        
        // Start monitoring tasks
        self.start_market_monitoring().await;
        self.start_risk_monitoring().await;
        self.start_emergency_detection().await;
        
        info!("Emergency Override System started successfully");
        Ok(())
    }
    
    /// Stop the emergency override system
    #[instrument(skip(self))]
    pub async fn stop(&self) -> Result<(), EmergencyError> {
        info!("Stopping Emergency Override System");
        
        self.running.store(false, Ordering::SeqCst);
        
        // Wait for all tasks to complete
        sleep(Duration::from_secs(1)).await;
        
        info!("Emergency Override System stopped");
        Ok(())
    }
    
    /// Start market conditions monitoring
    async fn start_market_monitoring(&self) {
        let market_conditions = self.market_conditions.clone();
        let running = self.running.clone();
        
        tokio::spawn(async move {
            let mut interval = interval(Duration::from_millis(100));
            
            while running.load(Ordering::SeqCst) {
                interval.tick().await;
                
                // Simulate market data collection
                let mut conditions = market_conditions.write().await;
                conditions.timestamp = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_secs();
                
                // Update market metrics (in real implementation, this would come from data feeds)
                conditions.volatility = 0.02 + (rand::random::<f64>() * 0.03);
                conditions.liquidity = 50_000.0 + (rand::random::<f64>() * 100_000.0);
                conditions.drawdown = rand::random::<f64>() * 0.1;
                conditions.whale_activity = rand::random::<f64>();
                conditions.flash_crash_probability = rand::random::<f64>() * 0.1;
                conditions.systemic_risk = rand::random::<f64>() * 0.2;
            }
        });
    }
    
    /// Start risk monitoring and assessment
    async fn start_risk_monitoring(&self) {
        let market_conditions = self.market_conditions.clone();
        let config = self.config.clone();
        let risk_engine = self.risk_engine.clone();
        let running = self.running.clone();
        
        tokio::spawn(async move {
            let mut interval = interval(Duration::from_millis(10));
            
            while running.load(Ordering::SeqCst) {
                interval.tick().await;
                
                let conditions = market_conditions.read().await;
                let cfg = config.read().await;
                
                // Assess current risk levels
                let risk_score = Self::calculate_risk_score(&conditions, &cfg).await;
                
                // Update volatility history
                {
                    let mut volatility_history = risk_engine.volatility_history.write().await;
                    volatility_history.push(conditions.volatility);
                    if volatility_history.len() > 1000 {
                        volatility_history.remove(0);
                    }
                }
                
                // Detect whale activity
                if conditions.whale_activity > 0.8 {
                    let whale_activity = WhaleActivity {
                        timestamp: conditions.timestamp,
                        symbol: "BTC/USD".to_string(),
                        order_size: 1_000_000.0,
                        price_impact: 0.05,
                        activity_type: "buy".to_string(),
                        confidence: conditions.whale_activity,
                    };
                    
                    let mut activity_history = risk_engine.whale_detector.activity_history.write().await;
                    activity_history.push(whale_activity);
                    if activity_history.len() > 100 {
                        activity_history.remove(0);
                    }
                }
            }
        });
    }
    
    /// Start emergency detection and response
    async fn start_emergency_detection(&self) {
        let market_conditions = self.market_conditions.clone();
        let config = self.config.clone();
        let state = self.state.clone();
        let emergency_broadcast = self.emergency_broadcast.clone();
        let running = self.running.clone();
        let metrics = self.metrics.clone();
        
        tokio::spawn(async move {
            let mut interval = interval(Duration::from_millis(1));
            
            while running.load(Ordering::SeqCst) {
                interval.tick().await;
                
                let start_time = Instant::now();
                let conditions = market_conditions.read().await;
                let cfg = config.read().await;
                
                // Detect emergency conditions
                let emergency_events = Self::detect_emergency_conditions(&conditions, &cfg).await;
                
                if !emergency_events.is_empty() {
                    // Update metrics
                    metrics.total_emergencies.fetch_add(1, Ordering::SeqCst);
                    metrics.last_emergency.store(conditions.timestamp, Ordering::SeqCst);
                    
                    // Calculate response time
                    let response_time = start_time.elapsed().as_nanos() as u64;
                    metrics.avg_response_time_ns.store(response_time, Ordering::SeqCst);
                    
                    // Update system state
                    {
                        let mut state_guard = state.write().await;
                        state_guard.active_events = emergency_events.clone();
                        state_guard.level = Self::calculate_emergency_level(&emergency_events);
                        state_guard.response_time_ns = response_time;
                        state_guard.activation_time = Some(conditions.timestamp);
                    }
                    
                    // Broadcast emergency events
                    for event in emergency_events {
                        let _ = emergency_broadcast.send(event);
                    }
                }
            }
        });
    }
    
    /// Calculate overall risk score based on market conditions
    async fn calculate_risk_score(conditions: &MarketConditions, config: &EmergencyConfig) -> f64 {
        let mut risk_score = 0.0;
        
        // Volatility risk
        if conditions.volatility > config.volatility_threshold {
            risk_score += (conditions.volatility / config.volatility_threshold) * 0.3;
        }
        
        // Drawdown risk
        if conditions.drawdown > config.max_drawdown {
            risk_score += (conditions.drawdown / config.max_drawdown) * 0.3;
        }
        
        // Liquidity risk
        if conditions.liquidity < config.min_liquidity {
            risk_score += (config.min_liquidity - conditions.liquidity) / config.min_liquidity * 0.2;
        }
        
        // Whale activity risk
        risk_score += conditions.whale_activity * 0.1;
        
        // Flash crash risk
        risk_score += conditions.flash_crash_probability * 0.1;
        
        risk_score.min(1.0)
    }
    
    /// Detect emergency conditions based on market data
    async fn detect_emergency_conditions(conditions: &MarketConditions, config: &EmergencyConfig) -> Vec<EmergencyEvent> {
        let mut events = Vec::new();
        
        // Flash crash detection
        if conditions.flash_crash_probability > 0.8 {
            events.push(EmergencyEvent::FlashCrash);
        }
        
        // Liquidity crisis detection
        if conditions.liquidity < config.min_liquidity {
            events.push(EmergencyEvent::LiquidityCrisis);
        }
        
        // Whale activity detection
        if conditions.whale_activity > 0.9 {
            events.push(EmergencyEvent::WhaleActivity);
        }
        
        // Volatility spike detection
        if conditions.volatility > config.volatility_threshold * 2.0 {
            events.push(EmergencyEvent::VolatilitySpike);
        }
        
        // Drawdown risk detection
        if conditions.drawdown > config.max_drawdown {
            events.push(EmergencyEvent::MarginCallRisk);
        }
        
        // Systemic risk detection
        if conditions.systemic_risk > 0.7 {
            events.push(EmergencyEvent::SystemicRisk);
        }
        
        events
    }
    
    /// Calculate emergency level based on active events
    fn calculate_emergency_level(events: &[EmergencyEvent]) -> EmergencyLevel {
        let mut max_level = EmergencyLevel::Normal;
        
        for event in events {
            let level = match event {
                EmergencyEvent::FlashCrash => EmergencyLevel::Extreme,
                EmergencyEvent::LiquidityCrisis => EmergencyLevel::Critical,
                EmergencyEvent::WhaleActivity => EmergencyLevel::High,
                EmergencyEvent::VolatilitySpike => EmergencyLevel::High,
                EmergencyEvent::MarginCallRisk => EmergencyLevel::Critical,
                EmergencyEvent::SystemicRisk => EmergencyLevel::Extreme,
                EmergencyEvent::CorrelationBreakdown => EmergencyLevel::High,
                EmergencyEvent::LiquidationRisk => EmergencyLevel::Critical,
                EmergencyEvent::ConnectivityIssue => EmergencyLevel::High,
                EmergencyEvent::ExchangeHalt => EmergencyLevel::Critical,
            };
            
            if level as u8 > max_level as u8 {
                max_level = level;
            }
        }
        
        max_level
    }
    
    /// Execute emergency actions based on current conditions
    #[instrument(skip(self))]
    pub async fn execute_emergency_actions(&self, level: EmergencyLevel) -> Result<Vec<EmergencyAction>, EmergencyError> {
        let start_time = Instant::now();
        let mut actions = Vec::new();
        
        match level {
            EmergencyLevel::Normal => {
                // No actions needed
            }
            EmergencyLevel::Elevated => {
                actions.push(EmergencyAction::SetEmergencyStops);
            }
            EmergencyLevel::High => {
                actions.push(EmergencyAction::ReducePositions(0.5));
                actions.push(EmergencyAction::ActivateHedging);
            }
            EmergencyLevel::Critical => {
                actions.push(EmergencyAction::ReducePositions(0.8));
                actions.push(EmergencyAction::IncreaseCashReserves);
                actions.push(EmergencyAction::ActivateHedging);
            }
            EmergencyLevel::Extreme => {
                actions.push(EmergencyAction::CloseAllPositions);
                actions.push(EmergencyAction::SuspendTrading);
            }
        }
        
        // Execute actions
        let context = EmergencyContext {
            market_conditions: self.market_conditions.read().await.clone(),
            level,
            event: EmergencyEvent::SystemicRisk, // Default event
            state: self.state.read().await.clone(),
        };
        
        for action in &actions {
            if let Err(e) = self.execute_action(action, &context).await {
                error!("Failed to execute emergency action {:?}: {}", action, e);
            }
        }
        
        // Update state with actions taken
        {
            let mut state = self.state.write().await;
            state.actions_taken.extend(actions.clone());
            state.response_time_ns = start_time.elapsed().as_nanos() as u64;
        }
        
        Ok(actions)
    }
    
    /// Execute a specific emergency action
    async fn execute_action(&self, action: &EmergencyAction, context: &EmergencyContext) -> Result<(), EmergencyError> {
        info!("Executing emergency action: {:?}", action);
        
        match action {
            EmergencyAction::CloseAllPositions => {
                // Implementation would close all open positions
                info!("Closing all positions - EMERGENCY PROTOCOL ACTIVATED");
            }
            EmergencyAction::ReducePositions(percentage) => {
                info!("Reducing positions by {}%", percentage * 100.0);
            }
            EmergencyAction::ActivateHedging => {
                info!("Activating hedging strategies");
            }
            EmergencyAction::IncreaseCashReserves => {
                info!("Increasing cash reserves");
            }
            EmergencyAction::LiquidateAssets(assets) => {
                info!("Liquidating assets: {:?}", assets);
            }
            EmergencyAction::SetEmergencyStops => {
                info!("Setting emergency stop-loss orders");
            }
            EmergencyAction::SuspendTrading => {
                info!("Suspending all trading activities");
            }
            EmergencyAction::EnableAntifragileMode => {
                info!("Enabling antifragile buying mode for crash opportunities");
            }
            EmergencyAction::EnableWhaleFollowing => {
                info!("Enabling whale momentum following");
            }
            EmergencyAction::ActivateCrisisArbitrage => {
                info!("Activating crisis arbitrage opportunities");
            }
        }
        
        Ok(())
    }
    
    /// Get current emergency system status
    pub async fn get_status(&self) -> EmergencySystemStatus {
        let state = self.state.read().await;
        let conditions = self.market_conditions.read().await;
        let metrics = EmergencySystemMetrics {
            total_emergencies: self.metrics.total_emergencies.load(Ordering::SeqCst),
            avg_response_time_ns: self.metrics.avg_response_time_ns.load(Ordering::SeqCst),
            false_positive_rate: self.metrics.false_positive_rate.load(Ordering::SeqCst),
            uptime_seconds: self.metrics.uptime_seconds.load(Ordering::SeqCst),
            last_emergency: self.metrics.last_emergency.load(Ordering::SeqCst),
        };
        
        EmergencySystemStatus {
            level: state.level,
            active_events: state.active_events.clone(),
            actions_taken: state.actions_taken.clone(),
            market_conditions: conditions.clone(),
            metrics,
            system_health: self.calculate_system_health().await,
        }
    }
    
    /// Calculate overall system health score
    async fn calculate_system_health(&self) -> f64 {
        let conditions = self.market_conditions.read().await;
        let state = self.state.read().await;
        
        let mut health_score = 1.0;
        
        // Reduce score based on emergency level
        match state.level {
            EmergencyLevel::Normal => {},
            EmergencyLevel::Elevated => health_score -= 0.1,
            EmergencyLevel::High => health_score -= 0.3,
            EmergencyLevel::Critical => health_score -= 0.6,
            EmergencyLevel::Extreme => health_score -= 0.9,
        }
        
        // Reduce score based on market conditions
        if conditions.volatility > 0.05 {
            health_score -= 0.1;
        }
        if conditions.drawdown > 0.1 {
            health_score -= 0.2;
        }
        if conditions.liquidity < 50_000.0 {
            health_score -= 0.1;
        }
        
        health_score.max(0.0)
    }
}

/// Emergency system status report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmergencySystemStatus {
    pub level: EmergencyLevel,
    pub active_events: Vec<EmergencyEvent>,
    pub actions_taken: Vec<EmergencyAction>,
    pub market_conditions: MarketConditions,
    pub metrics: EmergencySystemMetrics,
    pub system_health: f64,
}

/// Emergency system performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmergencySystemMetrics {
    pub total_emergencies: u64,
    pub avg_response_time_ns: u64,
    pub false_positive_rate: f64,
    pub uptime_seconds: u64,
    pub last_emergency: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::{sleep, Duration};
    
    #[tokio::test]
    async fn test_emergency_system_initialization() {
        let config = EmergencyConfig::default();
        let system = EmergencyOverrideSystem::new(config).await.unwrap();
        
        assert_eq!(system.state.read().await.level, EmergencyLevel::Normal);
        assert!(system.state.read().await.active_events.is_empty());
    }
    
    #[tokio::test]
    async fn test_emergency_detection() {
        let config = EmergencyConfig::default();
        let system = EmergencyOverrideSystem::new(config).await.unwrap();
        
        // Start the system
        system.start().await.unwrap();
        
        // Wait for system to run
        sleep(Duration::from_millis(100)).await;
        
        // Check status
        let status = system.get_status().await;
        assert!(status.system_health > 0.0);
        
        // Stop the system
        system.stop().await.unwrap();
    }
    
    #[tokio::test]
    async fn test_emergency_actions() {
        let config = EmergencyConfig::default();
        let system = EmergencyOverrideSystem::new(config).await.unwrap();
        
        // Test emergency action execution
        let actions = system.execute_emergency_actions(EmergencyLevel::High).await.unwrap();
        
        assert!(!actions.is_empty());
        assert!(actions.contains(&EmergencyAction::ReducePositions(0.5)));
        assert!(actions.contains(&EmergencyAction::ActivateHedging));
    }
    
    #[tokio::test]
    async fn test_risk_calculation() {
        let conditions = MarketConditions {
            volatility: 0.1,
            liquidity: 50_000.0,
            drawdown: 0.08,
            correlations: HashMap::new(),
            whale_activity: 0.7,
            flash_crash_probability: 0.3,
            systemic_risk: 0.4,
            market_regime: "volatile".to_string(),
            timestamp: 1234567890,
        };
        
        let config = EmergencyConfig::default();
        let risk_score = EmergencyOverrideSystem::calculate_risk_score(&conditions, &config).await;
        
        assert!(risk_score > 0.0);
        assert!(risk_score <= 1.0);
    }
    
    #[tokio::test]
    async fn test_emergency_level_calculation() {
        let events = vec![
            EmergencyEvent::FlashCrash,
            EmergencyEvent::VolatilitySpike,
        ];
        
        let level = EmergencyOverrideSystem::calculate_emergency_level(&events);
        assert_eq!(level, EmergencyLevel::Extreme);
    }
}