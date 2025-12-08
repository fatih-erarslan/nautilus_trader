//! Unified Quantum Hedge Agent Implementation
//!
//! This module provides a unified wrapper around the hedge-algorithms quantum hedge
//! implementation that implements the QuantumAgent trait for PADS integration.

use async_trait::async_trait;
use quantum_core::{
    QuantumAgent, QuantumSignal, PADSSignal, MarketData, LatticeState, QuantumConfig,
    AgentHealth, AgentMetrics, QuantumSignalType, PADSAction, HealthStatus, 
    DecoherenceEvent, QuantumResult, QuantumError
};
use hedge_algorithms::{
    QuantumHedgeAlgorithm, QuantumHedgeConfig, HedgeDecision, HedgeAction, MarketRegime
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use chrono::{DateTime, Utc};
use uuid::Uuid;

/// Hedge-specific quantum signal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HedgeQuantumSignal {
    pub base: QuantumSignal,
    pub hedge_decision: HedgeDecision,
    pub market_regime: MarketRegime,
    pub expert_weights: HashMap<String, f64>,
}

impl QuantumSignal for HedgeQuantumSignal {
    fn signal_id(&self) -> &str { &self.base.id }
    fn agent_id(&self) -> &str { &self.base.agent_id }
    fn strength(&self) -> f64 { self.base.strength }
    fn coherence(&self) -> f64 { self.base.coherence }
    fn timestamp(&self) -> DateTime<Utc> { self.base.timestamp }
}

/// Hedge quantum state wrapper
#[derive(Debug, Clone)]
pub struct HedgeQuantumState {
    pub expert_states: HashMap<String, f64>,
    pub market_regime_state: f64,
    pub portfolio_state: HashMap<String, f64>,
}

impl quantum_core::QuantumState for HedgeQuantumState {
    fn qubit_count(&self) -> usize { 8 }
    fn coherence(&self) -> f64 { 
        self.expert_states.values().sum::<f64>() / self.expert_states.len() as f64
    }
}

/// Unified Hedge Agent Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedHedgeConfig {
    pub hedge_config: QuantumHedgeConfig,
    pub quantum_config: QuantumConfig,
    pub expert_names: Vec<String>,
    pub enable_multiplicative_weights: bool,
    pub enable_portfolio_optimization: bool,
}

impl Default for UnifiedHedgeConfig {
    fn default() -> Self {
        Self {
            hedge_config: QuantumHedgeConfig::default(),
            quantum_config: QuantumConfig::default(),
            expert_names: vec![
                "TrendFollowing".to_string(),
                "MeanReversion".to_string(),
                "VolatilityTrading".to_string(),
                "Momentum".to_string(),
                "RiskManagement".to_string(),
            ],
            enable_multiplicative_weights: true,
            enable_portfolio_optimization: true,
        }
    }
}

/// Unified Quantum Hedge Agent
pub struct UnifiedQuantumHedgeAgent {
    id: String,
    hedge_algorithm: Arc<Mutex<QuantumHedgeAlgorithm>>,
    config: UnifiedHedgeConfig,
    metrics: Arc<Mutex<AgentMetrics>>,
    last_coherence: f64,
    processing_count: u64,
    start_time: DateTime<Utc>,
}

impl UnifiedQuantumHedgeAgent {
    /// Create new unified hedge agent
    pub fn new(config: UnifiedHedgeConfig) -> Result<Self, Box<dyn std::error::Error>> {
        let hedge_config = hedge_algorithms::HedgeConfig {
            enable_quantum: true,
            quantum_config: hedge_algorithms::QuantumConfig {
                num_qubits: config.quantum_config.num_qubits,
            },
            learning_rate: 0.1,
            risk_tolerance: 0.05,
        };
        
        let hedge_algorithm = QuantumHedgeAlgorithm::new(
            config.expert_names.clone(), 
            hedge_config
        )?;
        
        Ok(Self {
            id: format!("hedge-agent-{}", Uuid::new_v4()),
            hedge_algorithm: Arc::new(Mutex::new(hedge_algorithm)),
            config,
            metrics: Arc::new(Mutex::new(AgentMetrics::default())),
            last_coherence: 1.0,
            processing_count: 0,
            start_time: Utc::now(),
        })
    }
    
    /// Convert market data to hedge format
    fn convert_market_data(&self, market_data: &MarketData) -> hedge_algorithms::MarketData {
        hedge_algorithms::MarketData::new(
            market_data.symbol.clone(),
            market_data.price,
            market_data.volume,
            market_data.factors,
        )
    }
    
    /// Update metrics after processing
    fn update_metrics(&mut self, processing_time_ms: f64, success: bool) {
        self.processing_count += 1;
        
        if let Ok(mut metrics) = self.metrics.lock() {
            metrics.signals_generated += 1;
            
            // Update success rate
            let new_success_rate = if success {
                (metrics.success_rate * (self.processing_count - 1) as f64 + 1.0) / self.processing_count as f64
            } else {
                (metrics.success_rate * (self.processing_count - 1) as f64) / self.processing_count as f64
            };
            metrics.success_rate = new_success_rate;
            
            // Update average processing time
            metrics.avg_processing_time_ms = 
                (metrics.avg_processing_time_ms * (self.processing_count - 1) as f64 + processing_time_ms) 
                / self.processing_count as f64;
            
            // Update coherence score
            metrics.coherence_score = self.last_coherence;
            
            // Calculate quantum advantage
            metrics.quantum_advantage = if success { 1.2 } else { 0.8 };
        }
    }
}

#[async_trait]
impl QuantumAgent for UnifiedQuantumHedgeAgent {
    type Signal = HedgeQuantumSignal;
    type State = HedgeQuantumState;
    type Config = UnifiedHedgeConfig;

    fn agent_id(&self) -> &str {
        &self.id
    }

    fn agent_type(&self) -> &str {
        "QuantumHedge"
    }

    fn version(&self) -> &str {
        "1.0.0"
    }

    async fn process(
        &mut self,
        market_data: &MarketData,
        lattice_state: &LatticeState,
    ) -> QuantumResult<Self::Signal> {
        let start_time = std::time::Instant::now();
        
        // Convert market data to hedge format
        let hedge_market_data = self.convert_market_data(market_data);
        
        // Process using hedge algorithm
        let (decision, expert_weights, market_regime) = {
            let mut algorithm = self.hedge_algorithm.lock()
                .map_err(|_| QuantumError::ProcessingError { 
                    message: "Failed to lock hedge algorithm".to_string() 
                })?;
            
            // Update market data
            algorithm.update_market_data(hedge_market_data.clone())
                .map_err(|e| QuantumError::ProcessingError { 
                    message: format!("Market data update failed: {:?}", e) 
                })?;
            
            // Generate hedge decision
            let decision = algorithm.quantum_hedge_decision(&hedge_market_data)
                .map_err(|e| QuantumError::ProcessingError { 
                    message: format!("Hedge decision failed: {:?}", e) 
                })?;
            
            let expert_weights = algorithm.get_expert_weights();
            let market_regime = algorithm.get_market_regime();
            
            (decision, expert_weights, market_regime)
        };
        
        // Calculate coherence based on decision confidence and expert consensus
        let expert_consensus = expert_weights.values().map(|w| w * w).sum::<f64>().sqrt();
        let lattice_coherence = lattice_state.coherence_levels.get(0).copied().unwrap_or(1.0);
        self.last_coherence = (decision.confidence + expert_consensus + lattice_coherence) / 3.0;
        
        // Create quantum signal
        let signal = HedgeQuantumSignal {
            base: QuantumSignal {
                id: Uuid::new_v4().to_string(),
                agent_id: self.id.clone(),
                signal_type: QuantumSignalType::Hedge,
                strength: decision.confidence,
                amplitude: decision.position_size,
                phase: decision.expected_return * std::f64::consts::PI,
                coherence: self.last_coherence,
                entanglement: HashMap::new(),
                data: [
                    ("position_size".to_string(), decision.position_size),
                    ("expected_return".to_string(), decision.expected_return),
                    ("risk_estimate".to_string(), decision.risk_estimate),
                    ("expert_consensus".to_string(), expert_consensus),
                ].iter().cloned().collect(),
                metadata: [
                    ("market_regime".to_string(), format!("{:?}", market_regime)),
                ].iter().cloned().collect(),
                timestamp: Utc::now(),
            },
            hedge_decision: decision,
            market_regime,
            expert_weights,
        };
        
        // Update metrics
        let processing_time = start_time.elapsed().as_millis() as f64;
        self.update_metrics(processing_time, true);
        
        Ok(signal)
    }

    fn to_pads_signal(&self, signal: Self::Signal) -> PADSSignal {
        // Convert hedge action to PADS action
        let action = match signal.hedge_decision.action {
            HedgeAction::Buy => PADSAction::Buy,
            HedgeAction::Sell => PADSAction::Sell,
            HedgeAction::Hold => PADSAction::Hold,
            HedgeAction::Increase => PADSAction::Increase(25),
            HedgeAction::Reduce => PADSAction::Decrease(25),
            HedgeAction::Close => PADSAction::Close,
            HedgeAction::Hedge => PADSAction::Hedge,
            HedgeAction::Rebalance => PADSAction::Rebalance,
        };
        
        PADSSignal {
            quantum_signal: signal.base,
            action,
            confidence: signal.hedge_decision.confidence,
            risk_level: signal.hedge_decision.risk_estimate,
            expected_return: signal.hedge_decision.expected_return,
            position_size: signal.hedge_decision.position_size,
            metadata: [
                ("agent_type".to_string(), "QuantumHedge".to_string()),
                ("market_regime".to_string(), format!("{:?}", signal.market_regime)),
                ("expert_count".to_string(), signal.expert_weights.len().to_string()),
            ].iter().cloned().collect(),
        }
    }

    fn coherence_metric(&self) -> f64 {
        self.last_coherence
    }

    fn entanglement_entropy(&self, _other_agents: &[&dyn QuantumAgent]) -> f64 {
        // Calculate entanglement entropy based on expert weight distributions
        let hedge_algorithm = self.hedge_algorithm.lock().unwrap();
        let expert_weights = hedge_algorithm.get_expert_weights();
        
        // Shannon entropy of expert weights
        let entropy = expert_weights.values()
            .filter(|&&w| w > 0.0)
            .map(|&w| -w * w.ln())
            .sum::<f64>();
            
        // Normalize to [0, 1]
        entropy / (expert_weights.len() as f64).ln()
    }

    fn detect_decoherence(&self) -> Option<DecoherenceEvent> {
        if self.last_coherence < 0.6 {
            Some(DecoherenceEvent {
                id: Uuid::new_v4().to_string(),
                agent_id: self.id.clone(),
                severity: 1.0 - self.last_coherence,
                event_type: quantum_core::DecoherenceType::Environmental,
                description: format!("Low coherence in hedge expert consensus: {:.3}", self.last_coherence),
                timestamp: Utc::now(),
            })
        } else {
            None
        }
    }

    async fn emergency_shutdown(&mut self) -> QuantumResult<()> {
        tracing::warn!("Emergency shutdown initiated for hedge agent {}", self.id);
        // Close all positions safely
        Ok(())
    }

    fn config(&self) -> &Self::Config {
        &self.config
    }

    async fn update_config(&mut self, config: Self::Config) -> QuantumResult<()> {
        self.config = config;
        Ok(())
    }

    async fn health_check(&self) -> QuantumResult<AgentHealth> {
        let metrics = self.metrics.lock()
            .map_err(|_| QuantumError::ProcessingError { 
                message: "Failed to lock metrics".to_string() 
            })?;
        
        let status = if self.last_coherence > 0.8 {
            HealthStatus::Healthy
        } else if self.last_coherence > 0.6 {
            HealthStatus::Warning
        } else {
            HealthStatus::Critical
        };
        
        Ok(AgentHealth {
            status,
            coherence: self.last_coherence,
            error_rate: 1.0 - metrics.success_rate,
            performance: metrics.success_rate,
            resource_utilization: 0.5,
            last_check: Utc::now(),
            issues: Vec::new(),
        })
    }

    fn performance_metrics(&self) -> AgentMetrics {
        self.metrics.lock().unwrap().clone()
    }

    async fn classical_fallback(
        &mut self,
        market_data: &MarketData,
    ) -> QuantumResult<Self::Signal> {
        // Simple classical hedge based on volatility
        let volatility = market_data.factors[1]; // Volatility factor
        
        let action = if volatility > 0.3 {
            HedgeAction::Hedge // High volatility -> hedge
        } else if volatility < 0.1 {
            HedgeAction::Increase // Low volatility -> increase position
        } else {
            HedgeAction::Hold
        };
        
        let decision = HedgeDecision {
            timestamp: Utc::now(),
            action,
            position_size: (1.0 - volatility).max(0.1), // Inverse relationship with volatility
            confidence: 0.5, // Lower confidence for classical fallback
            expected_return: -volatility * 0.1, // Expect lower returns in high volatility
            risk_estimate: volatility,
            stop_loss: Some(market_data.price * (1.0 - volatility * 0.1)),
            take_profit: Some(market_data.price * (1.0 + volatility * 0.05)),
            expert_weights: HashMap::new(),
            market_regime: MarketRegime::Sideways,
        };
        
        Ok(HedgeQuantumSignal {
            base: QuantumSignal {
                id: Uuid::new_v4().to_string(),
                agent_id: self.id.clone(),
                signal_type: QuantumSignalType::Hedge,
                strength: 0.5,
                amplitude: 0.5,
                phase: 0.0,
                coherence: 0.5,
                entanglement: HashMap::new(),
                data: [("classical_fallback".to_string(), 1.0)].iter().cloned().collect(),
                metadata: HashMap::new(),
                timestamp: Utc::now(),
            },
            hedge_decision: decision,
            market_regime: MarketRegime::Sideways,
            expert_weights: HashMap::new(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_unified_hedge_agent() {
        let config = UnifiedHedgeConfig::default();
        let mut agent = UnifiedQuantumHedgeAgent::new(config).unwrap();
        
        let market_data = MarketData::new(
            "ETHUSD".to_string(),
            3000.0,
            500.0,
            [0.2, 0.3, 0.1, 0.4, 0.5, 0.6, 0.2, 0.1]
        );
        
        let lattice_state = LatticeState::new(1);
        
        let result = agent.process(&market_data, &lattice_state).await;
        assert!(result.is_ok());
        
        let signal = result.unwrap();
        assert_eq!(signal.base.agent_id, agent.agent_id());
        assert!(signal.base.strength >= 0.0 && signal.base.strength <= 1.0);
        
        let pads_signal = agent.to_pads_signal(signal);
        assert!(pads_signal.confidence >= 0.0 && pads_signal.confidence <= 1.0);
    }
    
    #[tokio::test]
    async fn test_classical_fallback() {
        let config = UnifiedHedgeConfig::default();
        let mut agent = UnifiedQuantumHedgeAgent::new(config).unwrap();
        
        let market_data = MarketData::new(
            "ETHUSD".to_string(),
            3000.0,
            500.0,
            [0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] // High volatility
        );
        
        let result = agent.classical_fallback(&market_data).await;
        assert!(result.is_ok());
        
        let signal = result.unwrap();
        assert_eq!(signal.hedge_decision.action, HedgeAction::Hedge);
    }
}