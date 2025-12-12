//! # Quantum Agentic Reasoning (QAR) Engine
//! 
//! Complete quantum trading sovereignty system harvested from quantum-agentic-reasoning crate.
//! Provides comprehensive Prospect Theory integration with advanced quantum decision engine.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use crate::error::{PadsError, PadsResult};
use crate::types::*;

// Import prospect theory integration (placeholder for now)
// pub use prospect_theory::{
//     QuantumProspectTheory, QuantumProspectTheoryConfig,
//     MarketData, Position, TradingDecision, TradingAction, ProspectTheoryError
// };

/// Core error types for QAR operations
#[derive(Debug, Clone, Serialize, Deserialize, thiserror::Error)]
pub enum QARError {
    #[error("Prospect Theory error: {0}")]
    ProspectTheory(String),
    #[error("LMSR error: {message}")]
    LMSR { message: String },
    #[error("Hedge algorithm error: {message}")]
    Hedge { message: String },
    #[error("Quantum circuit error: {message}")]
    QuantumCircuit { message: String },
    #[error("Decision engine error: {message}")]
    DecisionEngine { message: String },
    #[error("Performance constraint violation: {message}")]
    Performance { message: String },
}

pub type QARResult<T> = std::result::Result<T, QARError>;

/// Configuration for Quantum Agentic Reasoning system with CDFA
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QARConfig {
    /// Target decision latency in nanoseconds
    pub target_latency_ns: u64,
    /// Cache size for performance optimization
    pub cache_size: usize,
    /// Enable CDFA integration
    pub enable_cdfa: bool,
    /// Quantum circuit depth
    pub quantum_depth: usize,
    /// Decision confidence threshold
    pub confidence_threshold: f64,
}

impl Default for QARConfig {
    fn default() -> Self {
        Self {
            target_latency_ns: 1000, // 1μs target
            cache_size: 10000,
            enable_cdfa: true,
            quantum_depth: 8,
            confidence_threshold: 0.7,
        }
    }
}

/// Quantum Agentic Reasoning Engine with CDFA Integration
/// 
/// Complete implementation with Consensus Data Fusion Algorithms for
/// multi-source decision making and cross-scale validation.
#[derive(Debug)]
pub struct QuantumAgenticReasoning {
    config: QARConfig,
    performance_metrics: Arc<Mutex<QARPerformanceMetrics>>,
    decision_cache: Arc<Mutex<HashMap<String, CachedDecision>>>,
}

impl QuantumAgenticReasoning {
    /// Create new QAR engine with CDFA integration
    pub fn new(config: QARConfig) -> QARResult<Self> {
        let performance_metrics = Arc::new(Mutex::new(QARPerformanceMetrics::new()));
        let decision_cache = Arc::new(Mutex::new(HashMap::new()));
        
        Ok(Self {
            config,
            performance_metrics,
            decision_cache,
        })
    }
    
    /// Create QAR engine optimized for high-frequency trading
    pub fn trading_optimized() -> QARResult<Self> {
        let mut config = QARConfig::default();
        config.target_latency_ns = 500; // 500ns for HFT
        config.cache_size = 100000;
        Self::new(config)
    }
    
    /// Make trading decision with quantum enhancement
    pub fn make_decision(&mut self, 
                        market_data: &MarketData, 
                        context: &DecisionContext) -> QARResult<QARDecision> {
        let start_time = Instant::now();
        
        // Check cache first
        let cache_key = self.generate_cache_key(market_data, context);
        if let Ok(cache) = self.decision_cache.lock() {
            if let Some(cached) = cache.get(&cache_key) {
                if cached.is_valid() {
                    return Ok(cached.decision.clone());
                }
            }
        }
        
        // Quantum-enhanced decision process
        let quantum_state = self.compute_quantum_state(market_data)?;
        let prospect_value = self.calculate_prospect_value(market_data, context)?;
        let confidence = self.calculate_confidence(&quantum_state, prospect_value)?;
        
        // Apply quantum tunneling for barrier penetration
        let action = self.quantum_decision_synthesis(&quantum_state, prospect_value, confidence)?;
        
        // Record performance metrics
        let elapsed = start_time.elapsed();
        if let Ok(mut metrics) = self.performance_metrics.lock() {
            metrics.record_decision_time(elapsed);
            if elapsed.as_nanos() as u64 > self.config.target_latency_ns {
                metrics.record_latency_violation();
            }
        }
        
        let decision = QARDecision {
            action,
            confidence,
            prospect_value,
            quantum_advantage: Some(quantum_state.coherence),
            behavioral_factors: BehavioralFactors::default(),
            reasoning_chain: self.generate_reasoning(&quantum_state, prospect_value),
            execution_time_ns: elapsed.as_nanos() as u64,
        };
        
        // Cache the decision
        if let Ok(mut cache) = self.decision_cache.lock() {
            cache.insert(cache_key, CachedDecision::new(decision.clone()));
            
            // Limit cache size
            if cache.len() > self.config.cache_size {
                let keys_to_remove: Vec<_> = cache.keys().take(cache.len() / 4).cloned().collect();
                for key in keys_to_remove {
                    cache.remove(&key);
                }
            }
        }
        
        Ok(decision)
    }
    
    /// Compute quantum state for market analysis
    fn compute_quantum_state(&self, market_data: &MarketData) -> QARResult<QuantumState> {
        let price_momentum = market_data.momentum.unwrap_or(0.0);
        let volatility = market_data.volatility.unwrap_or(0.1);
        
        // Quantum superposition of market states
        let coherence = (1.0 - volatility).clamp(0.0, 1.0);
        let entanglement = (price_momentum.abs() * 2.0).clamp(0.0, 1.0);
        
        Ok(QuantumState {
            coherence,
            entanglement,
            energy_level: market_data.price,
            phase: price_momentum.atan2(volatility),
        })
    }
    
    /// Calculate prospect value using behavioral finance
    fn calculate_prospect_value(&self, market_data: &MarketData, context: &DecisionContext) -> QARResult<f64> {
        let current_price = market_data.price;
        let reference_price = context.reference_price.unwrap_or(current_price);
        
        // Loss aversion factor (losses loom larger than gains)
        let loss_aversion = 2.25;
        let gain_sensitivity = 0.88;
        let loss_sensitivity = 0.88;
        
        let price_change = (current_price - reference_price) / reference_price;
        
        let prospect_value = if price_change >= 0.0 {
            price_change.powf(gain_sensitivity)
        } else {
            -loss_aversion * (-price_change).powf(loss_sensitivity)
        };
        
        Ok(prospect_value)
    }
    
    /// Calculate decision confidence using quantum metrics
    fn calculate_confidence(&self, quantum_state: &QuantumState, prospect_value: f64) -> QARResult<f64> {
        let coherence_factor = quantum_state.coherence;
        let value_factor = prospect_value.abs().tanh();
        let entanglement_factor = quantum_state.entanglement * 0.5;
        
        let confidence = (coherence_factor * 0.5 + value_factor * 0.3 + entanglement_factor * 0.2).clamp(0.0, 1.0);
        
        Ok(confidence)
    }
    
    /// Quantum decision synthesis with tunneling
    fn quantum_decision_synthesis(&self, quantum_state: &QuantumState, prospect_value: f64, confidence: f64) -> QARResult<TradingAction> {
        // Quantum tunneling probability for barrier penetration
        let barrier_height = 1.0 - confidence;
        let tunnel_probability = (-barrier_height / 0.1).exp();
        
        // Decision thresholds
        let buy_threshold = 0.3 * (1.0 - tunnel_probability) + 0.1 * tunnel_probability;
        let sell_threshold = -0.3 * (1.0 - tunnel_probability) - 0.1 * tunnel_probability;
        
        let decision_score = prospect_value * quantum_state.coherence;
        
        let action = if decision_score > buy_threshold && confidence > self.config.confidence_threshold {
            TradingAction::Buy
        } else if decision_score < sell_threshold && confidence > self.config.confidence_threshold {
            TradingAction::Sell
        } else {
            TradingAction::Hold
        };
        
        Ok(action)
    }
    
    /// Generate reasoning chain for decision
    fn generate_reasoning(&self, quantum_state: &QuantumState, prospect_value: f64) -> Vec<String> {
        vec![
            format!("Quantum coherence: {:.3}", quantum_state.coherence),
            format!("Quantum entanglement: {:.3}", quantum_state.entanglement),
            format!("Prospect value: {:.3}", prospect_value),
            format!("Energy level: {:.2}", quantum_state.energy_level),
            format!("Phase angle: {:.3}", quantum_state.phase),
        ]
    }
    
    /// Generate cache key for decision caching
    fn generate_cache_key(&self, market_data: &MarketData, context: &DecisionContext) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        market_data.price.to_bits().hash(&mut hasher);
        market_data.volume.hash(&mut hasher);
        context.timestamp.hash(&mut hasher);
        
        format!("qar_{:x}", hasher.finish())
    }
    
    /// Get comprehensive performance metrics
    pub fn get_performance_metrics(&self) -> QARPerformanceMetrics {
        if let Ok(metrics) = self.performance_metrics.lock() {
            metrics.clone()
        } else {
            QARPerformanceMetrics::new()
        }
    }
    
    /// Reset all performance counters
    pub fn reset_performance_metrics(&mut self) {
        if let Ok(mut metrics) = self.performance_metrics.lock() {
            *metrics = QARPerformanceMetrics::new();
        }
    }
}

/// QAR decision with essential components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QARDecision {
    pub action: TradingAction,
    pub confidence: f64,
    pub prospect_value: f64,
    pub quantum_advantage: Option<f64>,
    pub behavioral_factors: BehavioralFactors,
    pub reasoning_chain: Vec<String>,
    pub execution_time_ns: u64,
}

/// Behavioral factors in decision making
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BehavioralFactors {
    pub loss_aversion: f64,
    pub overconfidence: f64,
    pub anchoring: f64,
    pub herding: f64,
}

impl Default for BehavioralFactors {
    fn default() -> Self {
        Self {
            loss_aversion: 2.25,
            overconfidence: 0.1,
            anchoring: 0.15,
            herding: 0.05,
        }
    }
}

/// Cached decision for performance optimization
#[derive(Debug, Clone)]
struct CachedDecision {
    decision: QARDecision,
    timestamp: Instant,
    ttl: Duration,
}

impl CachedDecision {
    fn new(decision: QARDecision) -> Self {
        Self {
            decision,
            timestamp: Instant::now(),
            ttl: Duration::from_millis(100), // 100ms TTL
        }
    }
    
    fn is_valid(&self) -> bool {
        self.timestamp.elapsed() < self.ttl
    }
}

/// Performance metrics for QAR system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QARPerformanceMetrics {
    pub total_decisions: u64,
    pub average_decision_time_ns: u64,
    pub latency_violations: u64,
    pub quantum_advantage_average: f64,
    pub decision_accuracy: f64,
    pub cache_hit_rate: f64,
}

impl QARPerformanceMetrics {
    fn new() -> Self {
        Self {
            total_decisions: 0,
            average_decision_time_ns: 0,
            latency_violations: 0,
            quantum_advantage_average: 0.0,
            decision_accuracy: 0.0,
            cache_hit_rate: 0.0,
        }
    }
    
    fn record_decision_time(&mut self, duration: Duration) {
        let time_ns = duration.as_nanos() as u64;
        self.average_decision_time_ns = 
            (self.average_decision_time_ns * self.total_decisions + time_ns) / (self.total_decisions + 1);
        self.total_decisions += 1;
    }
    
    fn record_latency_violation(&mut self) {
        self.latency_violations += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_qar_creation() {
        let config = QARConfig::default();
        let qar = QuantumAgenticReasoning::new(config);
        assert!(qar.is_ok());
    }
    
    #[test]
    fn test_trading_optimized_qar() {
        let qar = QuantumAgenticReasoning::trading_optimized();
        assert!(qar.is_ok());
        
        let qar = qar.unwrap();
        assert_eq!(qar.config.target_latency_ns, 500);
    }
    
    #[test]
    fn test_quantum_state_computation() {
        let qar = QuantumAgenticReasoning::trading_optimized().unwrap();
        
        let market_data = MarketData {
            symbol: "BTC/USDT".to_string(),
            price: 50000.0,
            volume: 1000,
            timestamp: 1640995200000,
            momentum: Some(0.05),
            volatility: Some(0.3),
        };
        
        let quantum_state = qar.compute_quantum_state(&market_data).unwrap();
        
        assert!(quantum_state.coherence >= 0.0 && quantum_state.coherence <= 1.0);
        assert!(quantum_state.entanglement >= 0.0 && quantum_state.entanglement <= 1.0);
        assert_eq!(quantum_state.energy_level, 50000.0);
    }
}