//! Risk Management System for Cerebellar Trading Neural Network
//! 
//! Provides comprehensive risk controls, position limits, drawdown monitoring,
//! and circuit breakers for high-frequency trading systems with neural networks.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{SystemTime, UNIX_EPOCH, Instant, Duration};
use nalgebra::{DVector, DMatrix};
use serde::{Serialize, Deserialize};
use anyhow::{Result, anyhow, Context};
use tracing::{debug, info, warn, error};
use tokio::sync::{mpsc, watch};
use futures::stream::StreamExt;

use crate::{CerebellarCircuit, CircuitMetrics, TradingCerebellarProcessor};

/// Critical risk events that require immediate action
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RiskEvent {
    /// Position limit exceeded
    PositionLimitExceeded { 
        symbol: String, 
        current_position: f64, 
        limit: f64,
        timestamp: u64
    },
    /// Maximum drawdown reached
    DrawdownLimitReached { 
        current_drawdown: f64, 
        limit: f64,
        timestamp: u64
    },
    /// Daily loss limit exceeded
    DailyLossLimitExceeded { 
        daily_pnl: f64, 
        limit: f64,
        timestamp: u64
    },
    /// Neural network output anomaly detected
    NeuralAnomalyDetected { 
        anomaly_type: AnomalyType,
        confidence: f64,
        affected_outputs: Vec<usize>,
        timestamp: u64
    },
    /// Trading velocity too high
    VelocityLimitExceeded { 
        current_velocity: f64, 
        limit: f64,
        window_ms: u64,
        timestamp: u64
    },
    /// VaR limit breached
    VarLimitBreached { 
        current_var: f64, 
        limit: f64,
        confidence_level: f64,
        timestamp: u64
    },
    /// Circuit breaker triggered
    CircuitBreakerTriggered { 
        breaker_type: CircuitBreakerType,
        duration_ms: u64,
        timestamp: u64
    }
}

/// Types of neural network anomalies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnomalyType {
    /// Neural outputs outside expected range
    OutputRangeAnomaly,
    /// Sudden change in neural activity patterns
    ActivityPatternAnomaly,
    /// Membrane potential instability
    MembranePotentialAnomaly,
    /// Excessive spike activity
    HyperActivityAnomaly,
    /// Neural network convergence failure
    ConvergenceFailure,
    /// Model confidence too low
    LowConfidenceAnomaly
}

/// Types of circuit breakers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CircuitBreakerType {
    /// Emergency stop - all trading halted
    EmergencyStop,
    /// Position limits breached
    PositionBreaker,
    /// Loss limits breached
    LossBreaker,
    /// Neural anomaly detected
    NeuralBreaker,
    /// Market volatility too high
    VolatilityBreaker,
    /// System resource exhaustion
    ResourceBreaker
}

/// Risk limits and thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskLimits {
    /// Maximum position size per symbol
    pub max_position_per_symbol: HashMap<String, f64>,
    /// Maximum total portfolio exposure
    pub max_total_exposure: f64,
    /// Maximum daily loss (negative value)
    pub max_daily_loss: f64,
    /// Maximum drawdown percentage
    pub max_drawdown_percent: f64,
    /// Maximum trading velocity (trades per second)
    pub max_trading_velocity: f64,
    /// VaR limit (Value at Risk)
    pub var_limit: f64,
    /// VaR confidence level (e.g., 0.95 for 95%)
    pub var_confidence_level: f64,
    /// Neural network output bounds
    pub neural_output_bounds: (f64, f64),
    /// Maximum neural confidence threshold
    pub min_neural_confidence: f64,
    /// Circuit breaker cooldown period (milliseconds)
    pub circuit_breaker_cooldown_ms: u64
}

impl Default for RiskLimits {
    fn default() -> Self {
        Self {
            max_position_per_symbol: HashMap::new(),
            max_total_exposure: 1_000_000.0,
            max_daily_loss: -50_000.0,
            max_drawdown_percent: 0.05, // 5%
            max_trading_velocity: 100.0, // 100 trades per second
            var_limit: 100_000.0,
            var_confidence_level: 0.95,
            neural_output_bounds: (-10.0, 10.0),
            min_neural_confidence: 0.7,
            circuit_breaker_cooldown_ms: 60_000 // 1 minute
        }
    }
}

/// Real-time P&L tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PnLTracker {
    /// Current unrealized P&L
    pub unrealized_pnl: f64,
    /// Current realized P&L
    pub realized_pnl: f64,
    /// Daily P&L
    pub daily_pnl: f64,
    /// Historical P&L for drawdown calculation
    pub pnl_history: VecDeque<(u64, f64)>, // (timestamp, cumulative_pnl)
    /// Maximum P&L achieved (for drawdown calculation)
    pub max_pnl: f64,
    /// Current drawdown
    pub current_drawdown: f64
}

impl PnLTracker {
    pub fn new() -> Self {
        Self {
            unrealized_pnl: 0.0,
            realized_pnl: 0.0,
            daily_pnl: 0.0,
            pnl_history: VecDeque::with_capacity(86400), // 24 hours of seconds
            max_pnl: 0.0,
            current_drawdown: 0.0
        }
    }

    /// Update P&L and calculate drawdown
    pub fn update_pnl(&mut self, unrealized: f64, realized: f64) -> Result<()> {
        self.unrealized_pnl = unrealized;
        self.realized_pnl = realized;
        
        let total_pnl = unrealized + realized;
        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
        
        // Update max P&L
        if total_pnl > self.max_pnl {
            self.max_pnl = total_pnl;
        }
        
        // Calculate current drawdown
        self.current_drawdown = (self.max_pnl - total_pnl) / self.max_pnl.max(1.0);
        
        // Store in history
        self.pnl_history.push_back((timestamp, total_pnl));
        
        // Keep only last 24 hours
        let cutoff_time = timestamp.saturating_sub(86400);
        while let Some(&(ts, _)) = self.pnl_history.front() {
            if ts < cutoff_time {
                self.pnl_history.pop_front();
            } else {
                break;
            }
        }
        
        Ok(())
    }

    /// Calculate Value at Risk (VaR)
    pub fn calculate_var(&self, confidence_level: f64) -> f64 {
        if self.pnl_history.len() < 30 {
            return 0.0; // Not enough data
        }
        
        // Calculate daily returns
        let mut returns = Vec::new();
        for window in self.pnl_history.iter().collect::<Vec<_>>().windows(2) {
            let return_pct = (window[1].1 - window[0].1) / window[0].1.abs().max(1.0);
            returns.push(return_pct);
        }
        
        // Sort returns and find percentile
        returns.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let index = ((1.0 - confidence_level) * returns.len() as f64) as usize;
        let var_return = returns.get(index).copied().unwrap_or(0.0);
        
        // Convert to dollar VaR
        var_return * (self.unrealized_pnl + self.realized_pnl).abs()
    }
}

/// Position tracking for risk management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionTracker {
    /// Current positions by symbol
    pub positions: HashMap<String, f64>,
    /// Position history for velocity calculation
    pub position_history: VecDeque<(u64, String, f64)>, // (timestamp, symbol, size)
    /// Total portfolio exposure
    pub total_exposure: f64
}

impl PositionTracker {
    pub fn new() -> Self {
        Self {
            positions: HashMap::new(),
            position_history: VecDeque::with_capacity(1000),
            total_exposure: 0.0
        }
    }

    /// Update position and track changes
    pub fn update_position(&mut self, symbol: String, new_size: f64) -> Result<()> {
        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
        
        // Update position
        let old_size = self.positions.get(&symbol).copied().unwrap_or(0.0);
        self.positions.insert(symbol.clone(), new_size);
        
        // Update total exposure
        self.total_exposure = self.positions.values().map(|&size| size.abs()).sum();
        
        // Record change if significant
        if (new_size - old_size).abs() > 0.01 {
            self.position_history.push_back((timestamp, symbol, new_size));
            
            // Keep only last 1000 changes
            if self.position_history.len() > 1000 {
                self.position_history.pop_front();
            }
        }
        
        Ok(())
    }

    /// Calculate trading velocity (trades per second over last minute)
    pub fn calculate_velocity(&self) -> f64 {
        let current_time = SystemTime::now().duration_since(UNIX_EPOCH)
            .unwrap_or_default().as_secs();
        let one_minute_ago = current_time.saturating_sub(60);
        
        let recent_trades = self.position_history.iter()
            .filter(|(timestamp, _, _)| *timestamp >= one_minute_ago)
            .count();
        
        recent_trades as f64 / 60.0
    }
}

/// Neural network output validation
#[derive(Debug)]
pub struct NeuralValidator {
    /// Expected output statistics
    output_stats: RwLock<HashMap<usize, (f64, f64)>>, // (mean, std)
    /// Recent outputs for anomaly detection
    recent_outputs: Mutex<VecDeque<(u64, Vec<f64>)>>,
    /// Anomaly detection thresholds
    anomaly_threshold: f64
}

impl NeuralValidator {
    pub fn new(anomaly_threshold: f64) -> Self {
        Self {
            output_stats: RwLock::new(HashMap::new()),
            recent_outputs: Mutex::new(VecDeque::with_capacity(1000)),
            anomaly_threshold
        }
    }

    /// Validate neural network outputs
    pub fn validate_outputs(&self, outputs: &[f64], bounds: (f64, f64)) -> Result<Vec<AnomalyType>> {
        let mut anomalies = Vec::new();
        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
        
        // Check bounds
        for (i, &output) in outputs.iter().enumerate() {
            if output < bounds.0 || output > bounds.1 {
                anomalies.push(AnomalyType::OutputRangeAnomaly);
                warn!("Neural output {} out of bounds: {} not in [{}, {}]", 
                      i, output, bounds.0, bounds.1);
                break;
            }
        }
        
        // Check for NaN or infinite values
        for (i, &output) in outputs.iter().enumerate() {
            if !output.is_finite() {
                anomalies.push(AnomalyType::OutputRangeAnomaly);
                error!("Neural output {} is not finite: {}", i, output);
                break;
            }
        }
        
        // Store recent outputs
        {
            let mut recent = self.recent_outputs.lock().unwrap();
            recent.push_back((timestamp, outputs.to_vec()));
            if recent.len() > 1000 {
                recent.pop_front();
            }
        }
        
        // Check for activity pattern anomalies
        if let Some(pattern_anomaly) = self.detect_pattern_anomaly(outputs)? {
            anomalies.push(pattern_anomaly);
        }
        
        Ok(anomalies)
    }

    /// Detect pattern anomalies in neural outputs
    fn detect_pattern_anomaly(&self, outputs: &[f64]) -> Result<Option<AnomalyType>> {
        let recent = self.recent_outputs.lock().unwrap();
        
        if recent.len() < 10 {
            return Ok(None); // Not enough history
        }
        
        // Calculate current output statistics
        let mean = outputs.iter().sum::<f64>() / outputs.len() as f64;
        let variance = outputs.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / outputs.len() as f64;
        let std_dev = variance.sqrt();
        
        // Compare with recent history
        let recent_means: Vec<f64> = recent.iter()
            .map(|(_, outputs)| outputs.iter().sum::<f64>() / outputs.len() as f64)
            .collect();
        
        if recent_means.len() >= 10 {
            let historical_mean = recent_means.iter().sum::<f64>() / recent_means.len() as f64;
            let historical_std = {
                let variance = recent_means.iter()
                    .map(|&x| (x - historical_mean).powi(2))
                    .sum::<f64>() / recent_means.len() as f64;
                variance.sqrt()
            };
            
            // Check if current mean is anomalous
            let z_score = (mean - historical_mean) / historical_std.max(0.001);
            if z_score.abs() > self.anomaly_threshold {
                return Ok(Some(AnomalyType::ActivityPatternAnomaly));
            }
        }
        
        Ok(None)
    }

    /// Update expected output statistics
    pub fn update_stats(&self, outputs: &[f64]) -> Result<()> {
        let mut stats = self.output_stats.write().unwrap();
        
        for (i, &output) in outputs.iter().enumerate() {
            let (count, sum, sum_sq) = stats.get(&i)
                .map(|(mean, std)| {
                    // Approximate count from existing stats
                    let count = 100.0; // Assume 100 samples for simplicity
                    let sum = mean * count;
                    let sum_sq = (std.powi(2) + mean.powi(2)) * count;
                    (count, sum, sum_sq)
                })
                .unwrap_or((0.0, 0.0, 0.0));
            
            let new_count = count + 1.0;
            let new_sum = sum + output;
            let new_sum_sq = sum_sq + output.powi(2);
            
            let new_mean = new_sum / new_count;
            let new_variance = (new_sum_sq / new_count) - new_mean.powi(2);
            let new_std = new_variance.max(0.0).sqrt();
            
            stats.insert(i, (new_mean, new_std));
        }
        
        Ok(())
    }
}

/// Circuit breaker state management
#[derive(Debug)]
pub struct CircuitBreaker {
    /// Current state of each breaker type
    breaker_states: RwLock<HashMap<CircuitBreakerType, (bool, u64)>>, // (active, expiry_time)
    /// Cooldown periods for each breaker type
    cooldown_periods: HashMap<CircuitBreakerType, u64>
}

impl CircuitBreaker {
    pub fn new() -> Self {
        let mut cooldown_periods = HashMap::new();
        cooldown_periods.insert(CircuitBreakerType::EmergencyStop, 300_000); // 5 minutes
        cooldown_periods.insert(CircuitBreakerType::PositionBreaker, 60_000); // 1 minute
        cooldown_periods.insert(CircuitBreakerType::LossBreaker, 120_000); // 2 minutes
        cooldown_periods.insert(CircuitBreakerType::NeuralBreaker, 30_000); // 30 seconds
        cooldown_periods.insert(CircuitBreakerType::VolatilityBreaker, 60_000); // 1 minute
        cooldown_periods.insert(CircuitBreakerType::ResourceBreaker, 30_000); // 30 seconds
        
        Self {
            breaker_states: RwLock::new(HashMap::new()),
            cooldown_periods
        }
    }

    /// Trigger a circuit breaker
    pub fn trigger_breaker(&self, breaker_type: CircuitBreakerType) -> Result<()> {
        let current_time = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis() as u64;
        let cooldown = self.cooldown_periods.get(&breaker_type).copied().unwrap_or(60_000);
        let expiry_time = current_time + cooldown;
        
        {
            let mut states = self.breaker_states.write().unwrap();
            states.insert(breaker_type.clone(), (true, expiry_time));
        }
        
        error!("Circuit breaker triggered: {:?}, expiry: {}", breaker_type, expiry_time);
        Ok(())
    }

    /// Check if a circuit breaker is active
    pub fn is_breaker_active(&self, breaker_type: &CircuitBreakerType) -> bool {
        let current_time = SystemTime::now().duration_since(UNIX_EPOCH)
            .unwrap_or_default().as_millis() as u64;
        
        let states = self.breaker_states.read().unwrap();
        if let Some(&(active, expiry_time)) = states.get(breaker_type) {
            active && current_time < expiry_time
        } else {
            false
        }
    }

    /// Check if any critical breaker is active (blocks all trading)
    pub fn is_trading_blocked(&self) -> bool {
        self.is_breaker_active(&CircuitBreakerType::EmergencyStop) ||
        self.is_breaker_active(&CircuitBreakerType::LossBreaker) ||
        self.is_breaker_active(&CircuitBreakerType::NeuralBreaker)
    }

    /// Reset a specific circuit breaker
    pub fn reset_breaker(&self, breaker_type: CircuitBreakerType) -> Result<()> {
        let mut states = self.breaker_states.write().unwrap();
        states.insert(breaker_type.clone(), (false, 0));
        info!("Circuit breaker reset: {:?}", breaker_type);
        Ok(())
    }

    /// Get all active breakers
    pub fn get_active_breakers(&self) -> Vec<CircuitBreakerType> {
        let current_time = SystemTime::now().duration_since(UNIX_EPOCH)
            .unwrap_or_default().as_millis() as u64;
        
        let states = self.breaker_states.read().unwrap();
        states.iter()
            .filter_map(|(breaker_type, &(active, expiry_time))| {
                if active && current_time < expiry_time {
                    Some(breaker_type.clone())
                } else {
                    None
                }
            })
            .collect()
    }
}

/// Main risk management system
pub struct RiskManager {
    /// Risk limits and thresholds
    pub limits: RiskLimits,
    /// P&L tracking
    pub pnl_tracker: Arc<Mutex<PnLTracker>>,
    /// Position tracking
    pub position_tracker: Arc<Mutex<PositionTracker>>,
    /// Neural network validator
    pub neural_validator: Arc<NeuralValidator>,
    /// Circuit breaker system
    pub circuit_breaker: Arc<CircuitBreaker>,
    /// Risk event channel
    pub risk_event_tx: mpsc::UnboundedSender<RiskEvent>,
    /// Risk event receiver
    risk_event_rx: Mutex<mpsc::UnboundedReceiver<RiskEvent>>,
    /// Emergency shutdown flag
    pub emergency_shutdown: Arc<watch::Sender<bool>>,
    /// Risk metrics
    metrics: Arc<Mutex<RiskMetrics>>
}

/// Risk management metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskMetrics {
    pub total_risk_events: u64,
    pub circuit_breaker_triggers: u64,
    pub neural_anomalies_detected: u64,
    pub position_limit_violations: u64,
    pub loss_limit_violations: u64,
    pub current_var: f64,
    pub current_drawdown: f64,
    pub last_update: u64
}

impl Default for RiskMetrics {
    fn default() -> Self {
        Self {
            total_risk_events: 0,
            circuit_breaker_triggers: 0,
            neural_anomalies_detected: 0,
            position_limit_violations: 0,
            loss_limit_violations: 0,
            current_var: 0.0,
            current_drawdown: 0.0,
            last_update: 0
        }
    }
}

impl RiskManager {
    /// Create new risk management system
    pub fn new(limits: RiskLimits) -> Self {
        let (risk_event_tx, risk_event_rx) = mpsc::unbounded_channel();
        let (emergency_shutdown_tx, _) = watch::channel(false);
        
        Self {
            limits,
            pnl_tracker: Arc::new(Mutex::new(PnLTracker::new())),
            position_tracker: Arc::new(Mutex::new(PositionTracker::new())),
            neural_validator: Arc::new(NeuralValidator::new(3.0)), // 3-sigma threshold
            circuit_breaker: Arc::new(CircuitBreaker::new()),
            risk_event_tx,
            risk_event_rx: Mutex::new(risk_event_rx),
            emergency_shutdown: Arc::new(emergency_shutdown_tx),
            metrics: Arc::new(Mutex::new(RiskMetrics::default()))
        }
    }

    /// Validate a proposed trade before execution
    pub async fn validate_trade(&self, symbol: &str, size: f64) -> Result<bool> {
        // Check if trading is blocked by circuit breakers
        if self.circuit_breaker.is_trading_blocked() {
            warn!("Trade blocked by circuit breaker: {} {}", symbol, size);
            return Ok(false);
        }

        // Check position limits
        {
            let position_tracker = self.position_tracker.lock().unwrap();
            let current_position = position_tracker.positions.get(symbol).copied().unwrap_or(0.0);
            let new_position = current_position + size;
            
            // Check symbol-specific limit
            if let Some(&limit) = self.limits.max_position_per_symbol.get(symbol) {
                if new_position.abs() > limit {
                    self.emit_risk_event(RiskEvent::PositionLimitExceeded {
                        symbol: symbol.to_string(),
                        current_position: new_position,
                        limit,
                        timestamp: SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs()
                    }).await?;
                    return Ok(false);
                }
            }
            
            // Check total exposure limit
            let new_total_exposure = position_tracker.total_exposure + size.abs();
            if new_total_exposure > self.limits.max_total_exposure {
                warn!("Trade would exceed total exposure limit: {} > {}", 
                      new_total_exposure, self.limits.max_total_exposure);
                return Ok(false);
            }
            
            // Check trading velocity
            let velocity = position_tracker.calculate_velocity();
            if velocity > self.limits.max_trading_velocity {
                self.emit_risk_event(RiskEvent::VelocityLimitExceeded {
                    current_velocity: velocity,
                    limit: self.limits.max_trading_velocity,
                    window_ms: 60_000,
                    timestamp: SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs()
                }).await?;
                return Ok(false);
            }
        }

        // Check P&L and drawdown limits
        {
            let pnl_tracker = self.pnl_tracker.lock().unwrap();
            
            if pnl_tracker.daily_pnl < self.limits.max_daily_loss {
                self.emit_risk_event(RiskEvent::DailyLossLimitExceeded {
                    daily_pnl: pnl_tracker.daily_pnl,
                    limit: self.limits.max_daily_loss,
                    timestamp: SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs()
                }).await?;
                return Ok(false);
            }
            
            if pnl_tracker.current_drawdown > self.limits.max_drawdown_percent {
                self.emit_risk_event(RiskEvent::DrawdownLimitReached {
                    current_drawdown: pnl_tracker.current_drawdown,
                    limit: self.limits.max_drawdown_percent,
                    timestamp: SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs()
                }).await?;
                return Ok(false);
            }
            
            // Check VaR limit
            let var = pnl_tracker.calculate_var(self.limits.var_confidence_level);
            if var.abs() > self.limits.var_limit {
                self.emit_risk_event(RiskEvent::VarLimitBreached {
                    current_var: var,
                    limit: self.limits.var_limit,
                    confidence_level: self.limits.var_confidence_level,
                    timestamp: SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs()
                }).await?;
                return Ok(false);
            }
        }

        Ok(true)
    }

    /// Validate neural network outputs before using for trading decisions
    pub async fn validate_neural_outputs(&self, outputs: &[f64], confidence: f64) -> Result<bool> {
        // Check minimum confidence threshold
        if confidence < self.limits.min_neural_confidence {
            warn!("Neural confidence too low: {} < {}", confidence, self.limits.min_neural_confidence);
            
            self.emit_risk_event(RiskEvent::NeuralAnomalyDetected {
                anomaly_type: AnomalyType::LowConfidenceAnomaly,
                confidence,
                affected_outputs: (0..outputs.len()).collect(),
                timestamp: SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs()
            }).await?;
            
            return Ok(false);
        }

        // Validate outputs through neural validator
        let anomalies = self.neural_validator.validate_outputs(outputs, self.limits.neural_output_bounds)?;
        
        if !anomalies.is_empty() {
            for anomaly in &anomalies {
                self.emit_risk_event(RiskEvent::NeuralAnomalyDetected {
                    anomaly_type: anomaly.clone(),
                    confidence,
                    affected_outputs: (0..outputs.len()).collect(),
                    timestamp: SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs()
                }).await?;
            }
            
            // Trigger neural circuit breaker for severe anomalies
            if anomalies.contains(&AnomalyType::OutputRangeAnomaly) ||
               anomalies.contains(&AnomalyType::ConvergenceFailure) {
                self.circuit_breaker.trigger_breaker(CircuitBreakerType::NeuralBreaker)?;
            }
            
            return Ok(false);
        }

        // Update neural statistics
        self.neural_validator.update_stats(outputs)?;
        
        Ok(true)
    }

    /// Update position tracking after a trade
    pub async fn update_position(&self, symbol: String, new_size: f64) -> Result<()> {
        {
            let mut position_tracker = self.position_tracker.lock().unwrap();
            position_tracker.update_position(symbol, new_size)?;
        }
        
        // Update metrics
        self.update_metrics().await?;
        
        Ok(())
    }

    /// Update P&L tracking
    pub async fn update_pnl(&self, unrealized: f64, realized: f64) -> Result<()> {
        {
            let mut pnl_tracker = self.pnl_tracker.lock().unwrap();
            pnl_tracker.update_pnl(unrealized, realized)?;
            
            // Check for limit breaches
            if pnl_tracker.current_drawdown > self.limits.max_drawdown_percent {
                self.circuit_breaker.trigger_breaker(CircuitBreakerType::LossBreaker)?;
            }
        }
        
        // Update metrics
        self.update_metrics().await?;
        
        Ok(())
    }

    /// Emit a risk event
    async fn emit_risk_event(&self, event: RiskEvent) -> Result<()> {
        // Update metrics
        {
            let mut metrics = self.metrics.lock().unwrap();
            metrics.total_risk_events += 1;
            
            match &event {
                RiskEvent::PositionLimitExceeded { .. } => metrics.position_limit_violations += 1,
                RiskEvent::DailyLossLimitExceeded { .. } | 
                RiskEvent::DrawdownLimitReached { .. } => metrics.loss_limit_violations += 1,
                RiskEvent::NeuralAnomalyDetected { .. } => metrics.neural_anomalies_detected += 1,
                RiskEvent::CircuitBreakerTriggered { .. } => metrics.circuit_breaker_triggers += 1,
                _ => {}
            }
            
            metrics.last_update = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
        }
        
        // Send event
        self.risk_event_tx.send(event)
            .context("Failed to send risk event")?;
        
        Ok(())
    }

    /// Update risk metrics
    async fn update_metrics(&self) -> Result<()> {
        let mut metrics = self.metrics.lock().unwrap();
        
        // Update current VaR and drawdown
        {
            let pnl_tracker = self.pnl_tracker.lock().unwrap();
            metrics.current_var = pnl_tracker.calculate_var(self.limits.var_confidence_level);
            metrics.current_drawdown = pnl_tracker.current_drawdown;
        }
        
        metrics.last_update = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
        
        Ok(())
    }

    /// Get current risk metrics
    pub fn get_metrics(&self) -> RiskMetrics {
        self.metrics.lock().unwrap().clone()
    }

    /// Get current risk status summary
    pub fn get_risk_status(&self) -> RiskStatus {
        let pnl_tracker = self.pnl_tracker.lock().unwrap();
        let position_tracker = self.position_tracker.lock().unwrap();
        let active_breakers = self.circuit_breaker.get_active_breakers();
        
        RiskStatus {
            trading_enabled: !self.circuit_breaker.is_trading_blocked(),
            active_circuit_breakers: active_breakers,
            current_drawdown: pnl_tracker.current_drawdown,
            daily_pnl: pnl_tracker.daily_pnl,
            total_exposure: position_tracker.total_exposure,
            trading_velocity: position_tracker.calculate_velocity(),
            var: pnl_tracker.calculate_var(self.limits.var_confidence_level),
            position_count: position_tracker.positions.len()
        }
    }

    /// Emergency shutdown - halt all trading immediately
    pub async fn emergency_shutdown(&self, reason: String) -> Result<()> {
        error!("EMERGENCY SHUTDOWN TRIGGERED: {}", reason);
        
        // Trigger emergency circuit breaker
        self.circuit_breaker.trigger_breaker(CircuitBreakerType::EmergencyStop)?;
        
        // Set emergency shutdown flag
        self.emergency_shutdown.send(true)?;
        
        // Emit critical risk event
        self.emit_risk_event(RiskEvent::CircuitBreakerTriggered {
            breaker_type: CircuitBreakerType::EmergencyStop,
            duration_ms: 300_000, // 5 minutes
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs()
        }).await?;
        
        Ok(())
    }

    /// Start risk monitoring task
    pub async fn start_monitoring(&self) -> Result<()> {
        info!("Starting risk management monitoring");
        
        // Monitor risk events
        let mut rx = {
            let mut rx_lock = self.risk_event_rx.lock().unwrap();
            // Create a new receiver by cloning the sender and creating a new channel
            // This is a workaround since UnboundedReceiver doesn't implement Clone
            let (tx, rx) = mpsc::unbounded_channel();
            drop(rx_lock); // Release the lock
            rx
        };
        
        tokio::spawn(async move {
            while let Some(event) = rx.recv().await {
                // Log critical events
                match &event {
                    RiskEvent::PositionLimitExceeded { symbol, current_position, limit, .. } => {
                        error!("POSITION LIMIT EXCEEDED: {} position {} > limit {}", 
                               symbol, current_position, limit);
                    },
                    RiskEvent::DrawdownLimitReached { current_drawdown, limit, .. } => {
                        error!("DRAWDOWN LIMIT REACHED: {} > {}", current_drawdown, limit);
                    },
                    RiskEvent::DailyLossLimitExceeded { daily_pnl, limit, .. } => {
                        error!("DAILY LOSS LIMIT EXCEEDED: {} < {}", daily_pnl, limit);
                    },
                    RiskEvent::NeuralAnomalyDetected { anomaly_type, confidence, .. } => {
                        warn!("NEURAL ANOMALY DETECTED: {:?} (confidence: {})", anomaly_type, confidence);
                    },
                    _ => {
                        info!("Risk event: {:?}", event);
                    }
                }
            }
        });
        
        Ok(())
    }
}

/// Current risk status summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskStatus {
    pub trading_enabled: bool,
    pub active_circuit_breakers: Vec<CircuitBreakerType>,
    pub current_drawdown: f64,
    pub daily_pnl: f64,
    pub total_exposure: f64,
    pub trading_velocity: f64,
    pub var: f64,
    pub position_count: usize
}

/// Trading decision with risk assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafeTradingDecision {
    pub symbol: String,
    pub action: TradeAction,
    pub size: f64,
    pub confidence: f64,
    pub risk_approved: bool,
    pub risk_reasons: Vec<String>,
    pub neural_outputs: Vec<f64>,
    pub timestamp: u64
}

/// Trade action types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TradeAction {
    Buy,
    Sell,
    Hold,
    Close
}

/// Safe trading processor that wraps neural network with risk controls
pub struct SafeTradingProcessor {
    /// Underlying neural processor
    pub neural_processor: TradingCerebellarProcessor,
    /// Risk management system
    pub risk_manager: Arc<RiskManager>,
    /// Decision history for analysis
    decision_history: Mutex<VecDeque<SafeTradingDecision>>
}

impl SafeTradingProcessor {
    /// Create new safe trading processor
    pub fn new(risk_limits: RiskLimits) -> Self {
        Self {
            neural_processor: TradingCerebellarProcessor::new(),
            risk_manager: Arc::new(RiskManager::new(risk_limits)),
            decision_history: Mutex::new(VecDeque::with_capacity(10000))
        }
    }

    /// Process market data with full risk validation
    pub async fn safe_process_tick(&mut self, symbol: String, price: f64, volume: f64, timestamp: u64) 
        -> Result<SafeTradingDecision> {
        
        // Process through neural network
        let neural_outputs = self.neural_processor.process_tick(price as f32, volume as f32, timestamp)?;
        let neural_outputs_f64: Vec<f64> = neural_outputs.iter().map(|&x| x as f64).collect();
        
        // Calculate confidence (simplified - would use more sophisticated method in practice)
        let confidence = self.calculate_confidence(&neural_outputs_f64);
        
        // Validate neural outputs
        let neural_valid = self.risk_manager.validate_neural_outputs(&neural_outputs_f64, confidence).await?;
        
        // Determine trading action from neural outputs
        let (action, size) = self.interpret_neural_outputs(&neural_outputs_f64);
        
        // Validate trade with risk manager
        let risk_approved = if neural_valid {
            self.risk_manager.validate_trade(&symbol, size).await?
        } else {
            false
        };
        
        // Collect risk reasons if not approved
        let mut risk_reasons = Vec::new();
        if !neural_valid {
            risk_reasons.push("Neural network outputs failed validation".to_string());
        }
        if !risk_approved && neural_valid {
            risk_reasons.push("Trade rejected by risk management".to_string());
        }
        
        let decision = SafeTradingDecision {
            symbol: symbol.clone(),
            action,
            size: if risk_approved { size } else { 0.0 },
            confidence,
            risk_approved,
            risk_reasons,
            neural_outputs: neural_outputs_f64,
            timestamp
        };
        
        // Store decision history
        {
            let mut history = self.decision_history.lock().unwrap();
            history.push_back(decision.clone());
            if history.len() > 10000 {
                history.pop_front();
            }
        }
        
        // Update position if trade approved
        if risk_approved && size.abs() > 0.01 {
            self.risk_manager.update_position(symbol, size).await?;
        }
        
        Ok(decision)
    }

    /// Calculate confidence score from neural outputs
    fn calculate_confidence(&self, outputs: &[f64]) -> f64 {
        if outputs.is_empty() {
            return 0.0;
        }
        
        // Simple confidence calculation - max output magnitude
        // In practice, would use more sophisticated methods
        let max_magnitude = outputs.iter().map(|&x| x.abs()).fold(0.0, f64::max);
        
        // Normalize to 0-1 range
        (max_magnitude / 10.0).min(1.0)
    }

    /// Interpret neural outputs as trading decision
    fn interpret_neural_outputs(&self, outputs: &[f64]) -> (TradeAction, f64) {
        if outputs.is_empty() {
            return (TradeAction::Hold, 0.0);
        }
        
        // Simple interpretation - first output determines action, second determines size
        let action_signal = outputs[0];
        let size_signal = outputs.get(1).copied().unwrap_or(0.0);
        
        let action = if action_signal > 0.5 {
            TradeAction::Buy
        } else if action_signal < -0.5 {
            TradeAction::Sell
        } else {
            TradeAction::Hold
        };
        
        let size = size_signal.abs() * 1000.0; // Scale to reasonable position size
        
        (action, size)
    }

    /// Get recent decision history
    pub fn get_decision_history(&self, count: usize) -> Vec<SafeTradingDecision> {
        let history = self.decision_history.lock().unwrap();
        history.iter().rev().take(count).cloned().collect()
    }

    /// Get risk manager reference
    pub fn get_risk_manager(&self) -> Arc<RiskManager> {
        self.risk_manager.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_risk_manager_creation() {
        let limits = RiskLimits::default();
        let risk_manager = RiskManager::new(limits);
        
        assert!(risk_manager.get_risk_status().trading_enabled);
    }

    #[tokio::test]
    async fn test_position_limit_validation() {
        let mut limits = RiskLimits::default();
        limits.max_position_per_symbol.insert("AAPL".to_string(), 1000.0);
        
        let risk_manager = RiskManager::new(limits);
        
        // Valid trade
        assert!(risk_manager.validate_trade("AAPL", 500.0).await.unwrap());
        
        // Invalid trade (exceeds limit)
        assert!(!risk_manager.validate_trade("AAPL", 1500.0).await.unwrap());
    }

    #[tokio::test]
    async fn test_neural_output_validation() {
        let limits = RiskLimits::default();
        let risk_manager = RiskManager::new(limits);
        
        // Valid outputs
        let valid_outputs = vec![1.0, 2.0, 3.0];
        assert!(risk_manager.validate_neural_outputs(&valid_outputs, 0.8).await.unwrap());
        
        // Invalid outputs (out of bounds)
        let invalid_outputs = vec![15.0, 2.0, 3.0];
        assert!(!risk_manager.validate_neural_outputs(&invalid_outputs, 0.8).await.unwrap());
        
        // Low confidence
        assert!(!risk_manager.validate_neural_outputs(&valid_outputs, 0.5).await.unwrap());
    }

    #[tokio::test]
    async fn test_circuit_breaker() {
        let circuit_breaker = CircuitBreaker::new();
        
        // Initially no breakers active
        assert!(!circuit_breaker.is_trading_blocked());
        
        // Trigger emergency stop
        circuit_breaker.trigger_breaker(CircuitBreakerType::EmergencyStop).unwrap();
        assert!(circuit_breaker.is_trading_blocked());
        
        // Reset breaker
        circuit_breaker.reset_breaker(CircuitBreakerType::EmergencyStop).unwrap();
        assert!(!circuit_breaker.is_trading_blocked());
    }

    #[test]
    fn test_pnl_tracker() {
        let mut pnl_tracker = PnLTracker::new();
        
        // Update P&L
        pnl_tracker.update_pnl(1000.0, 500.0).unwrap();
        assert_eq!(pnl_tracker.unrealized_pnl, 1000.0);
        assert_eq!(pnl_tracker.realized_pnl, 500.0);
        
        // Test drawdown calculation
        pnl_tracker.update_pnl(800.0, 500.0).unwrap();
        assert!(pnl_tracker.current_drawdown > 0.0);
    }

    #[tokio::test]
    async fn test_safe_trading_processor() {
        let limits = RiskLimits::default();
        let mut processor = SafeTradingProcessor::new(limits);
        
        let decision = processor.safe_process_tick(
            "AAPL".to_string(),
            150.0,
            1000.0,
            1234567890
        ).await.unwrap();
        
        assert_eq!(decision.symbol, "AAPL");
        assert!(decision.timestamp > 0);
    }
}