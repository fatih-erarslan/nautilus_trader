//! Unified Quantum Agentic Reasoning Agent Implementation
//!
//! This module provides a unified wrapper around the quantum-agentic-reasoning crate
//! that implements the QuantumAgent trait for seamless PADS integration.

use async_trait::async_trait;
use quantum_core::{
    QuantumAgent, QuantumSignal, PADSSignal, MarketData, LatticeState, QuantumConfig,
    AgentHealth, AgentMetrics, QuantumSignalType, PADSAction, HealthStatus, 
    DecoherenceEvent, QuantumResult, QuantumError
};
use quantum_agentic_reasoning::{
    QuantumAgenticReasoning, QARConfig, QARDecision, QuantumProspectTheoryConfig
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use chrono::{DateTime, Utc};
use uuid::Uuid;

/// QAR-specific quantum signal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QARQuantumSignal {
    pub base: QuantumSignal,
    pub decision: QARDecision,
    pub behavioral_factors: HashMap<String, f64>,
}

impl QuantumSignal for QARQuantumSignal {
    fn signal_id(&self) -> &str { &self.base.id }
    fn agent_id(&self) -> &str { &self.base.agent_id }
    fn strength(&self) -> f64 { self.base.strength }
    fn coherence(&self) -> f64 { self.base.coherence }
    fn timestamp(&self) -> DateTime<Utc> { self.base.timestamp }
}

/// QAR quantum state wrapper
#[derive(Debug, Clone)]
pub struct QARQuantumState {
    pub decision_context: HashMap<String, f64>,
    pub prospect_values: Vec<f64>,
    pub behavioral_state: HashMap<String, f64>,
}

impl quantum_core::QuantumState for QARQuantumState {
    fn qubit_count(&self) -> usize { 8 }
    fn coherence(&self) -> f64 { 
        self.behavioral_state.get("coherence").copied().unwrap_or(1.0)
    }
}

/// Unified QAR Agent Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedQARConfig {
    pub qar_config: QARConfig,
    pub quantum_config: QuantumConfig,
    pub enable_behavioral_bias: bool,
    pub prospect_theory_weight: f64,
    pub cdfa_integration: bool,
}

impl Default for UnifiedQARConfig {
    fn default() -> Self {
        Self {
            qar_config: QARConfig::default(),
            quantum_config: QuantumConfig::default(),
            enable_behavioral_bias: true,
            prospect_theory_weight: 0.7,
            cdfa_integration: true,
        }
    }
}

/// Unified Quantum Agentic Reasoning Agent
pub struct UnifiedQARAgent {
    id: String,
    qar_engine: Arc<Mutex<QuantumAgenticReasoning>>,
    config: UnifiedQARConfig,
    metrics: Arc<Mutex<AgentMetrics>>,
    last_coherence: f64,
    processing_count: u64,
    start_time: DateTime<Utc>,
}

impl UnifiedQARAgent {
    /// Create new unified QAR agent
    pub fn new(config: UnifiedQARConfig) -> Result<Self, Box<dyn std::error::Error>> {
        let qar_engine = QuantumAgenticReasoning::new(config.qar_config.clone())?;
        
        Ok(Self {
            id: format!("qar-agent-{}", Uuid::new_v4()),
            qar_engine: Arc::new(Mutex::new(qar_engine)),
            config,
            metrics: Arc::new(Mutex::new(AgentMetrics::default())),
            last_coherence: 1.0,
            processing_count: 0,
            start_time: Utc::now(),
        })
    }
    
    /// Convert market data to QAR format
    fn convert_market_data(&self, market_data: &MarketData) -> quantum_agentic_reasoning::MarketData {
        quantum_agentic_reasoning::MarketData {
            symbol: market_data.symbol.clone(),
            current_price: market_data.price,
            possible_outcomes: vec![
                market_data.price * 1.1,
                market_data.price * 1.05,
                market_data.price * 0.95,
                market_data.price * 0.9,
            ],
            buy_probabilities: vec![0.3, 0.3, 0.2, 0.2],
            sell_probabilities: vec![0.2, 0.2, 0.3, 0.3],
            hold_probabilities: vec![0.25, 0.25, 0.25, 0.25],
            frame: prospect_theory::FramingContext {
                frame_type: prospect_theory::FrameType::Neutral,
                emphasis: 0.5,
            },
            timestamp: market_data.timestamp as u64,
        }
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
            
            // Update uptime
            let uptime = Utc::now().signed_duration_since(self.start_time);
            metrics.uptime_percentage = 100.0; // Simplified for now
        }
    }
}

#[async_trait]
impl QuantumAgent for UnifiedQARAgent {
    type Signal = QARQuantumSignal;
    type State = QARQuantumState;
    type Config = UnifiedQARConfig;

    fn agent_id(&self) -> &str {
        &self.id
    }

    fn agent_type(&self) -> &str {
        "QuantumAgenticReasoning"
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
        
        // Convert market data to QAR format
        let qar_market_data = self.convert_market_data(market_data);
        
        // Process using QAR engine
        let decision = {
            let mut engine = self.qar_engine.lock()
                .map_err(|_| QuantumError::ProcessingError { 
                    message: "Failed to lock QAR engine".to_string() 
                })?;
            
            engine.make_decision(&qar_market_data, None)
                .map_err(|e| QuantumError::ProcessingError { 
                    message: format!("QAR processing failed: {:?}", e) 
                })?
        };
        
        // Calculate coherence based on decision confidence and lattice state
        let agent_index = lattice_state.coherence_levels.len().saturating_sub(1);
        let lattice_coherence = lattice_state.coherence_levels.get(agent_index).copied().unwrap_or(1.0);
        let decision_coherence = decision.confidence;
        self.last_coherence = (lattice_coherence + decision_coherence) / 2.0;
        
        // Create quantum signal
        let signal = QARQuantumSignal {
            base: QuantumSignal {
                id: Uuid::new_v4().to_string(),
                agent_id: self.id.clone(),
                signal_type: QuantumSignalType::Prospect,
                strength: decision.confidence,
                amplitude: (decision.prospect_value + 1.0) / 2.0, // Normalize to [0,1]
                phase: decision.prospect_value * std::f64::consts::PI,
                coherence: self.last_coherence,
                entanglement: HashMap::new(),
                data: [
                    ("prospect_value".to_string(), decision.prospect_value),
                    ("confidence".to_string(), decision.confidence),
                    ("execution_time_ns".to_string(), decision.execution_time_ns as f64),
                ].iter().cloned().collect(),
                metadata: HashMap::new(),
                timestamp: Utc::now(),
            },
            decision: decision.clone(),
            behavioral_factors: [
                ("overconfidence".to_string(), decision.behavioral_factors.overconfidence),
                ("loss_aversion".to_string(), decision.behavioral_factors.loss_aversion),
                ("anchoring".to_string(), decision.behavioral_factors.anchoring),
            ].iter().cloned().collect(),
        };
        
        // Update metrics
        let processing_time = start_time.elapsed().as_millis() as f64;
        self.update_metrics(processing_time, true);
        
        Ok(signal)
    }

    fn to_pads_signal(&self, signal: Self::Signal) -> PADSSignal {
        // Convert QAR decision to PADS action
        let action = match signal.decision.action {
            quantum_agentic_reasoning::TradingAction::Buy => PADSAction::Buy,
            quantum_agentic_reasoning::TradingAction::Sell => PADSAction::Sell,
            quantum_agentic_reasoning::TradingAction::Hold => PADSAction::Hold,
            quantum_agentic_reasoning::TradingAction::StrongBuy => PADSAction::Increase(80),
            quantum_agentic_reasoning::TradingAction::StrongSell => PADSAction::Decrease(80),
        };
        
        PADSSignal {
            quantum_signal: signal.base,
            action,
            confidence: signal.decision.confidence,
            risk_level: 1.0 - signal.decision.confidence,
            expected_return: signal.decision.prospect_value / 100.0, // Convert to percentage
            position_size: signal.decision.confidence * 0.1, // 10% max position
            metadata: [
                ("agent_type".to_string(), "QAR".to_string()),
                ("behavioral_bias".to_string(), "enabled".to_string()),
            ].iter().cloned().collect(),
        }
    }

    fn coherence_metric(&self) -> f64 {
        self.last_coherence
    }

    fn entanglement_entropy(&self, _other_agents: &[&dyn QuantumAgent]) -> f64 {
        // Calculate entanglement entropy based on behavioral factor correlations
        // Simplified implementation
        0.5 // Mid-level entanglement
    }

    fn detect_decoherence(&self) -> Option<DecoherenceEvent> {
        if self.last_coherence < 0.7 {
            Some(DecoherenceEvent {
                id: Uuid::new_v4().to_string(),
                agent_id: self.id.clone(),
                severity: 1.0 - self.last_coherence,
                event_type: quantum_core::DecoherenceType::Environmental,
                description: "Low coherence detected in QAR decision making".to_string(),
                timestamp: Utc::now(),
            })
        } else {
            None
        }
    }

    async fn emergency_shutdown(&mut self) -> QuantumResult<()> {
        tracing::warn!("Emergency shutdown initiated for QAR agent {}", self.id);
        // Graceful shutdown logic here
        Ok(())
    }

    fn config(&self) -> &Self::Config {
        &self.config
    }

    async fn update_config(&mut self, config: Self::Config) -> QuantumResult<()> {
        self.config = config;
        // Update underlying QAR engine configuration
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
            resource_utilization: 0.5, // Simplified
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
        // Simple classical decision based on trend
        let trend_factor = market_data.factors[0]; // Trend is first factor
        
        let action = if trend_factor > 0.1 {
            quantum_agentic_reasoning::TradingAction::Buy
        } else if trend_factor < -0.1 {
            quantum_agentic_reasoning::TradingAction::Sell
        } else {
            quantum_agentic_reasoning::TradingAction::Hold
        };
        
        let decision = QARDecision {
            action,
            confidence: 0.5, // Lower confidence for classical fallback
            prospect_value: trend_factor,
            quantum_advantage: Some(0.0), // No quantum advantage in fallback
            behavioral_factors: quantum_agentic_reasoning::BehavioralFactors::default(),
            reasoning_chain: vec!["Classical trend following".to_string()],
            execution_time_ns: 1000, // Fast classical processing
        };
        
        Ok(QARQuantumSignal {
            base: QuantumSignal {
                id: Uuid::new_v4().to_string(),
                agent_id: self.id.clone(),
                signal_type: QuantumSignalType::Prospect,
                strength: 0.5,
                amplitude: 0.5,
                phase: 0.0,
                coherence: 0.5, // Lower coherence for classical fallback
                entanglement: HashMap::new(),
                data: [("classical_fallback".to_string(), 1.0)].iter().cloned().collect(),
                metadata: HashMap::new(),
                timestamp: Utc::now(),
            },
            decision,
            behavioral_factors: HashMap::new(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_unified_qar_agent() {
        let config = UnifiedQARConfig::default();
        let mut agent = UnifiedQARAgent::new(config).unwrap();
        
        let market_data = MarketData::new(
            "BTCUSD".to_string(),
            50000.0,
            1000.0,
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
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
    async fn test_health_check() {
        let config = UnifiedQARConfig::default();
        let agent = UnifiedQARAgent::new(config).unwrap();
        
        let health = agent.health_check().await.unwrap();
        assert_eq!(health.status, HealthStatus::Healthy);
        assert!(health.coherence > 0.0);
    }
}